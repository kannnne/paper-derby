"""Command line entry point and scheduler.

    python -m hkjc backfill START END   # 回補賽果
    python -m hkjc post [DATE]          # 賽後：抓賽果、結算模擬注、重新訓練、出賽後報告
    python -m hkjc pre [--force]        # 賽前：排位表 + 即時賠率 → 預測 → 模擬落注 → 出投注參考
    python -m hkjc race_bet DATE VENUE RACE  # 單場注：開跑前約 8 分鐘用該場最新賠率決定（tick 自動做）
    python -m hkjc tick                 # 排程器每 10 分鐘呼叫一次，自動判斷做 pre / race_bet / post
    python -m hkjc backtest [START END] # 回測策略網格
    python -m hkjc report               # 模擬投注總結
"""
import argparse
import fcntl
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .db import ROOT, connect

HKT = ZoneInfo("Asia/Hong_Kong")
# $HKJC_LOG_DIR / $HKJC_REPORT_DIR (with $HKJC_DB, $HKJC_MODEL_DIR) allow dry runs on copies
LOG_DIR = Path(os.environ.get("HKJC_LOG_DIR") or ROOT / "data" / "logs")
REPORT_DIR = Path(os.environ.get("HKJC_REPORT_DIR") or ROOT / "reports")
PRE_LEAD = timedelta(hours=2)        # 第一場開跑前 2 小時
PRE_LATEST = timedelta(minutes=15)   # 錯過咗都最遲開跑前 15 分鐘補做（之後只出參考報告、唔落模擬注）
POST_DELAY = timedelta(minutes=45)   # 最後一場開跑後 45 分鐘先抓賽果
LIVE_AFTER = timedelta(minutes=5)     # 每場開跑後約 5 分鐘開始抓嗰場賽果；完整名次表未出就每次 tick 再抓
POST_LOOKBACK_DAYS = 7               # tick 只會自動重試最近 7 日未完成嘅賽馬日
NO_RESULTS_GIVE_UP_DAYS = 2          # 賽馬日過咗 2 日都完全冇賽果頁 -> 當取消 (abandoned)
SCORES_REFRESH = timedelta(minutes=55)    # 賠率開售後、落注前：大約每小時用最新賠率更新評分
SCORES_HORIZON = timedelta(days=3)         # 只為 3 日內嘅賽馬日出預覽評分
NEXT_MEETING_CACHE = timedelta(hours=1)   # 下一個賽馬日資料每小時先上網更新一次
# 單場注（per-race late betting）：每場開跑前約 8 分鐘用該場最新賠率決定；過關／跨場注照舊 T-2h 落。
RACE_BET_LEAD = timedelta(minutes=8)       # 目標：開跑前 8 分鐘（picks decide_at）
RACE_BET_EARLIEST = timedelta(minutes=12)  # tick 最早開跑前 12 分鐘做
RACE_BET_LATEST = timedelta(minutes=3)     # 最遲開跑前 3 分鐘；之後當錯過，唔落注
LOCK_NAME = "hkjc.lock"

log = logging.getLogger("hkjc")


def now_hkt() -> datetime:
    return datetime.now(HKT).replace(tzinfo=None)


def _parse_ts(s):
    return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M")


def _run_exists(conn, kind, date, venue):
    return conn.execute(
        "SELECT 1 FROM runs WHERE kind=? AND date=? AND venue=?", (kind, date, venue)
    ).fetchone() is not None


def _publish(conn, what):
    """Refresh the dashboard (site/) and push to GitHub Pages if enabled; never raises."""
    try:
        from . import publish
        res = publish.publish(conn, message=what)
        log.info("publish %s: %s", what, res.get("status"))
    except Exception:
        log.warning("publish failed", exc_info=True)


def _record_run(conn, run_id, kind, date, venue, notes):
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, kind, date, venue, created_at, notes) VALUES (?,?,?,?,?,?)",
        (run_id, kind, date, venue, now_hkt().isoformat(timespec="seconds"),
         json.dumps(notes, ensure_ascii=False, default=str)),
    )
    conn.commit()


# ---------------------------------------------------------------- pre-race
def cmd_pre(conn, force=False, mtg=None, place_bets=True):
    """Racecard + live odds -> predictions -> paper tickets -> pre-race report.

    ``mtg``: the meeting dict from scrape.next_meeting (looked up when None).
    ``place_bets=False``: report only (scheduler too late), recorded as kind 'pre_report'."""
    from . import odds, pipeline, scrape

    if mtg is None:
        mtg = scrape.next_meeting(conn)
    if not mtg:
        log.info("no upcoming meeting found")
        return None
    date, venue = mtg["date"], mtg["venue"]
    kind = "pre_bet" if place_bets else "pre_report"
    if not force and (_run_exists(conn, "pre_bet", date, venue) or _run_exists(conn, kind, date, venue)):
        log.info("%s already done for %s %s", kind, date, venue)
        return None
    if _run_exists(conn, "race_bet", date, venue):
        # single-race bets already decided race by race: re-deciding the meeting would void them
        log.warning("%s %s already has race-time (race_bet) runs - pre-race run refused", date, venue)
        return None
    st = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?", (date, venue)).fetchone()
    if st is not None and st["status"] in ("results", "abandoned"):
        # e.g. `pre --force` after the results: re-deciding would mix new decisions with the
        # settled ones (the dashboard / report would lose the other robots' decisions)
        log.warning("%s %s already has final results (%s) - pre-race run refused", date, venue, st["status"])
        return None
    scrape.scrape_racecard(conn, date, venue, refresh=True)
    n = odds.snapshot_meeting(conn, date, venue)
    log.info("odds snapshot rows: %s", n)
    if n:
        try:     # FCT / TRI / FF / DBL live odds for 今日精選 (hkjc.picks); optional, never fatal
            nx = odds.snapshot_meeting(conn, date, venue, pools=odds.EXOTIC_POOLS)
            log.info("exotic odds snapshot rows: %s", nx)
        except Exception as e:
            log.warning("exotic odds snapshot failed (%s); 今日精選 uses estimated odds", e)
    run_id = f"{'pre' if place_bets else 'prerep'}-{date}-{venue}-{now_hkt():%H%M}"
    result = pipeline.pre_meeting(conn, run_id, date, venue, place_bets=place_bets)
    path = REPORT_DIR / f"{date}_{venue}_賽前投注參考.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result["report_md"], encoding="utf-8")
    # recorded last: if anything above fails the next tick retries, and the re-run's
    # paper.place_tickets(replace=True) voids the tickets of the failed attempt
    _record_run(conn, run_id, kind, date, venue, {"odds_rows": n, **result.get("notes", {})})
    log.info("pre report written: %s", path)
    _publish(conn, f"賽前 {date} {venue}")
    return path


def cmd_scores(conn, mtg=None, force=False):
    """賠率開售就出評分 + 機械人決定預覽（唔落注）：WIN/PLA/QIN/QPL（+ 二重彩/單T/四連環/孖寶）賠率快照 ->
    預測 -> 評分 -> 預覽機械人決定同今日精選（bot_decisions ticket_id NULL / picks，run kind 'pre_scores'）
    -> 更新網站。
    冇賠率（未開售）就乜都唔做。Returns number of horses scored (0 = nothing done)."""
    from . import odds, pipeline, scrape

    if mtg is None:
        mtg = scrape.next_meeting(conn)
    if not mtg:
        return 0
    date, venue = mtg["date"], mtg["venue"]
    if not force and (_run_exists(conn, "pre_bet", date, venue) or _run_exists(conn, "pre_report", date, venue)):
        return 0
    stamp = now_hkt().strftime("%Y-%m-%dT%H:%M:%S")
    n = odds.snapshot_meeting(conn, date, venue, pools=("WIN", "PLA", "QIN", "QPL"), snapshot_at=stamp)
    if not n:
        log.info("scores preview: odds for %s %s not on sale yet", date, venue)
        return 0
    try:     # FCT / TRI / FF / DBL for the 今日精選 preview (optional)
        n += odds.snapshot_meeting(conn, date, venue, pools=odds.EXOTIC_POOLS, snapshot_at=stamp) or 0
    except Exception as e:
        log.warning("preview exotic odds snapshot failed (%s)", e)
    # keep only the newest preview snapshot of this meeting (no tickets depend on them before the betting run)
    if not (_run_exists(conn, "pre_bet", date, venue) or _run_exists(conn, "pre_report", date, venue)):
        conn.execute("DELETE FROM odds_snapshots WHERE date=? AND venue=? AND snapshot_at<?", (date, venue, stamp))
        conn.commit()
    scrape.scrape_racecard(conn, date, venue, refresh=True)
    run_id = f"score-{date}-{venue}-{now_hkt():%H%M}"
    k = pipeline.preview_scores(conn, run_id, date, venue)
    _record_run(conn, run_id, "pre_scores", date, venue, {"odds_rows": n, "horses": k})
    log.info("scores preview %s %s: %s horses scored with live odds", date, venue, k)
    _publish(conn, f"評分更新 {date} {venue}")
    return k


# ------------------------------------------------------- per-race late betting
def _race_window(conn, date, venue, race_no, now=None):
    """('early' | 'open' | 'late', post datetime) for the per-race decision window
    [post − RACE_BET_EARLIEST, post − RACE_BET_LATEST] (minutes)."""
    row = conn.execute("SELECT post_time FROM races WHERE date=? AND venue=? AND race_no=?",
                       (date, venue, int(race_no))).fetchone()
    if not row or not row[0]:
        return "open", None
    post = _parse_ts(row[0])
    now = now or now_hkt()
    if now < post - RACE_BET_EARLIEST:
        return "early", post
    if now > post - RACE_BET_LATEST:
        return "late", post
    return "open", post


def race_bet_run_id(date, venue, race_no):
    return f"race-{date}-{venue}-R{int(race_no):02d}"


def _pre_bet_run(conn, date, venue):
    """(run_id, notes) of the meeting's latest T-2h betting run, or (None, {})."""
    r = conn.execute("SELECT run_id, notes FROM runs WHERE kind='pre_bet' AND date=? AND venue=? "
                     "ORDER BY created_at DESC, run_id DESC LIMIT 1", (date, venue)).fetchone()
    if not r:
        return None, {}
    try:
        notes = json.loads(r["notes"] or "{}")
    except ValueError:
        notes = {}
    return r["run_id"], notes if isinstance(notes, dict) else {}


def _has_pending(conn, pre_run_id, race_no=None):
    q = "SELECT 1 FROM bot_decisions WHERE run_id=? AND action='pending'"
    prm = [pre_run_id]
    if race_no is not None:
        q += " AND race_no=?"
        prm.append(int(race_no))
    return conn.execute(q + " LIMIT 1", prm).fetchone() is not None


def _per_race_meeting(conn, pre_run_id, notes):
    """Meeting decided with per-race late betting (older meetings had every ticket at T-2h)."""
    return bool(notes.get("per_race")) or _has_pending(conn, pre_run_id)


def _refresh_race_card(conn, date, venue, race_no):
    """One racecard page (this race only) -> latest scratchings / post time.  Never raises."""
    from . import scrape
    try:
        f = scrape.get_fetcher()
        pg = scrape.parse_racecard(f.get(scrape.racecard_url(date, venue, race_no), refresh=True).text)
        if pg["race"].get("race_no") != int(race_no) or pg["race"].get("date") != date or not pg["runners"]:
            log.info("race_bet: racecard page for %s %s R%s not usable - kept stored card", date, venue, race_no)
            return False
        nos = pg.get("meeting_race_nos") or [r[0] for r in conn.execute(
            "SELECT race_no FROM races WHERE date=? AND venue=?", (date, venue))] or [int(race_no)]
        scrape.write_racecard(conn, {"date": date, "venue": venue, "race_nos": nos,
                                     "races": {int(race_no): pg}, "schedule": pg.get("schedule")})
        return True
    except Exception as e:
        log.warning("race_bet: racecard refresh for %s %s R%s failed (%s) - scratchings from odds only",
                    date, venue, race_no, e)
        return False


RACE_BET_LAST_TRY = timedelta(minutes=5)   # no live odds: keep retrying until post − 5 min, then use stored odds


def cmd_race_bet(conn, date, venue, race_no, publish=True):
    """單場注（開跑前約 8 分鐘）：只抓呢場最新賠率（WIN/PLA/QIN/QPL + 二重彩/單T/四連環/孖寶）->
    退出馬 -> 用最新賠率重新預測 -> 每隻機械人決定呢場單場注（落模擬注）+ 更新呢場今日精選。
    Needs the meeting's T-2h run (pre_bet).  Idempotent: recorded as runs kind 'race_bet'
    (run_id race-DATE-VENUE-Rnn); a second call does nothing.  Returns the run notes (None = nothing done)."""
    from . import bots as bots_mod, odds, pipeline

    rn = int(race_no)
    rid = race_bet_run_id(date, venue, rn)
    if conn.execute("SELECT 1 FROM runs WHERE run_id=?", (rid,)).fetchone():
        log.info("race_bet %s already done", rid)
        return None
    pre_id, _notes = _pre_bet_run(conn, date, venue)
    if pre_id is None:
        log.warning("race_bet %s: no pre_bet run for %s %s - nothing to decide", rid, date, venue)
        return None
    st = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?", (date, venue)).fetchone()
    if st is not None and st["status"] in ("results", "abandoned"):
        log.warning("race_bet %s: meeting already %s - refused", rid, st["status"])
        return None
    if conn.execute("SELECT 1 FROM runners u JOIN races r ON r.race_id=u.race_id WHERE r.date=? AND r.venue=? "
                    "AND r.race_no=? AND u.finish_pos IS NOT NULL LIMIT 1", (date, venue, rn)).fetchone():
        log.warning("race_bet %s: race already has results - refused (no betting after the off)", rid)
        return None
    state, _post_dt = _race_window(conn, date, venue, rn)
    if state == "early":
        log.info("race_bet %s: too early (window opens %s before the off)", rid, RACE_BET_EARLIEST)
        return None
    if state == "late":
        log.warning("race_bet %s: window already closed - handled as missed (no bet)", rid)
        return race_bet_missed(conn, date, venue, rn, publish=publish)
    # latest odds for this race; the race only counts as having fresh odds when THIS snapshot
    # brought WIN odds for it (other pools alone don't count)
    t0 = now_hkt().isoformat(timespec="seconds")
    n, nx, err = 0, 0, None
    try:
        n = odds.snapshot_meeting(conn, date, venue, race_nos=[rn]) or 0
    except Exception as e:          # API error: same as "no live odds" (retry, then stored odds)
        err = f"{type(e).__name__}: {e}"
        log.warning("race_bet %s: odds snapshot failed (%s)", rid, err)
    # every main pool the race offers (WIN always; PLA / QIN / QPL when we ever had odds for them,
    # e.g. QPL is not offered in small fields) must have come back fresh in THIS snapshot
    def _pools(since=None):
        q = ("SELECT pool FROM odds_snapshots WHERE date=? AND venue=? AND race_no=? AND odds IS NOT NULL"
             + (" AND snapshot_at>=?" if since else "") + " GROUP BY pool")
        return {r[0] for r in conn.execute(q, (date, venue, rn) + ((since,) if since else ()))}
    fresh_pools = _pools(t0)
    required = ({"WIN"} | (_pools() & set(odds.DEFAULT_POOLS))) & set(odds.DEFAULT_POOLS)
    stale_main = sorted(required - fresh_pools)
    win_rows = conn.execute("SELECT COUNT(*) FROM odds_snapshots WHERE date=? AND venue=? AND race_no=? "
                            "AND pool='WIN' AND odds IS NOT NULL AND snapshot_at>=?",
                            (date, venue, rn, t0)).fetchone()[0]
    fresh = not stale_main
    if fresh:
        try:
            nx = odds.snapshot_meeting(conn, date, venue, pools=odds.EXOTIC_POOLS, race_nos=[rn]) or 0
        except Exception as e:
            log.warning("race_bet %s: exotic odds snapshot failed (%s)", rid, e)
    else:
        post = conn.execute("SELECT post_time FROM races WHERE date=? AND venue=? AND race_no=?",
                            (date, venue, rn)).fetchone()
        if post and post[0] and now_hkt() < _parse_ts(post[0]) - RACE_BET_LAST_TRY:
            log.warning("race_bet %s: no fresh odds for %s (%s rows, error=%s) - will retry next tick",
                        rid, ",".join(stale_main), n, err)
            return None
        log.warning("race_bet %s: still no fresh odds for %s at the last chance - deciding with the latest "
                    "stored odds (flagged stale_odds)", rid, ",".join(stale_main))
    card = _refresh_race_card(conn, date, venue, rn)
    # the racecard refresh may move the post time; network / model time may have used up the window
    state, post_dt = _race_window(conn, date, venue, rn)
    if state != "open":
        log.warning("race_bet %s: window %s after fetching odds - handled as missed (no bet)", rid, state)
        return race_bet_missed(conn, date, venue, rn, publish=publish) if state == "late" else None
    try:
        notes = pipeline.race_bet(conn, pre_id, rid, date, venue, rn,
                                  deadline=(post_dt - RACE_BET_LATEST) if post_dt else None, clock=now_hkt)
    except bots_mod.RaceWindowClosed as e:
        log.warning("race_bet %s: %s - the robots still undecided are handled as missed", rid, e)
        return race_bet_missed(conn, date, venue, rn, publish=publish)
    fresh_all = _pools(t0)
    stale_pools = sorted((_pools() - fresh_all))          # incl. optional exotic pools kept from earlier
    notes.update({"odds_rows": n, "win_rows": win_rows, "exotic_rows": nx, "racecard_refreshed": card,
                  "stale_odds": not fresh, "odds_error": err, "fresh_pools": sorted(fresh_all),
                  "stale_pools": stale_pools})
    _record_run(conn, rid, "race_bet", date, venue, notes)
    log.info("race_bet %s: %s bets $%s (%s decisions, %s picks items)", rid, notes["bets"], notes["stake"],
             notes["decisions"], notes["picks_items"])
    if publish:
        _publish(conn, f"單場注 {date} {venue} 第{rn}場")
    return notes


def race_bet_missed(conn, date, venue, race_no, publish=True):
    """Window missed (later than post_time - RACE_BET_LATEST, no race_bet run): no bet - pending rows
    -> skip 「錯過開跑前落注時間，冇落注」, picks items -> 'missed'; recorded as race_bet {"missed": true}."""
    from . import pipeline

    rid = race_bet_run_id(date, venue, race_no)
    if conn.execute("SELECT 1 FROM runs WHERE run_id=?", (rid,)).fetchone():
        return None
    pre_id, _notes = _pre_bet_run(conn, date, venue)
    if pre_id is None:
        return None
    notes = pipeline.race_missed(conn, pre_id, date, venue, race_no, race_run_id=rid)
    _record_run(conn, rid, "race_bet", date, venue, notes)
    log.warning("race_bet %s missed: window passed, %s pending rows -> skip, %s picks items missed",
                rid, notes["rows"], notes["picks_items"])
    if publish:
        _publish(conn, f"單場注錯過 {date} {venue} 第{int(race_no)}場")
    return notes


def races_for_race_bet(conn, now):
    """[(date, venue, race_no, post_time, 'bet'|'missed')] due for the race-time step now."""
    out = []
    since = (now - timedelta(days=POST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for m in conn.execute("SELECT DISTINCT date, venue FROM runs WHERE kind='pre_bet' AND date>=? AND date<=?",
                          (since, now.strftime("%Y-%m-%d"))).fetchall():
        d, v = m["date"], m["venue"]
        pre_id, notes = _pre_bet_run(conn, d, v)
        if pre_id is None or not _per_race_meeting(conn, pre_id, notes):
            continue
        done = {r[0] for r in conn.execute("SELECT run_id FROM runs WHERE kind='race_bet' AND date=? AND venue=?",
                                           (d, v))}
        for r in conn.execute("SELECT race_no, post_time FROM races WHERE date=? AND venue=? AND post_time IS NOT NULL "
                              "ORDER BY race_no", (d, v)).fetchall():
            if race_bet_run_id(d, v, r["race_no"]) in done:
                continue
            post = _parse_ts(r["post_time"])
            if post - RACE_BET_EARLIEST <= now <= post - RACE_BET_LATEST:
                out.append((d, v, int(r["race_no"]), r["post_time"], "bet"))
            elif now > post - RACE_BET_LATEST:
                out.append((d, v, int(r["race_no"]), r["post_time"], "missed"))
    return out


def _race_bet_step(conn, now):
    """tick: per-race late betting for meetings with a T-2h run - in the window -> cmd_race_bet,
    after it -> race_bet_missed.  Errors are logged (the next tick retries while the window lasts)."""
    for d, v, rn, post, what in races_for_race_bet(conn, now):
        try:
            if what == "bet":
                cmd_race_bet(conn, d, v, rn)
            else:
                race_bet_missed(conn, d, v, rn)
        except Exception:
            log.exception("tick: race_bet for %s %s R%s failed (will retry next tick while in the window)", d, v, rn)


# A race counts as done only when its FULL results table is in (placings for the site);
# the preliminary page (dividends + top 4 only, ~10 min after the start) is not enough.
# (Void races are removed from `races` by the scraper, so they are never counted as due.)
DONE_RACES_SQL = (
    "SELECT COUNT(*) FROM races r WHERE r.date=? AND r.venue=? AND ("
    "EXISTS (SELECT 1 FROM runners u WHERE u.race_id=r.race_id AND u.finish_pos=1) OR "
    "EXISTS (SELECT 1 FROM dividends d WHERE d.race_id=r.race_id AND d.pool='WIN' AND d.combination='REFUND'))")
WIN_RACES_SQL = ("SELECT COUNT(DISTINCT r.race_id) FROM races r JOIN dividends d ON d.race_id=r.race_id "
                 "AND d.pool='WIN' WHERE r.date=? AND r.venue=?")


def _races_due_vs_done(conn, date, venue, now, after=None):
    """(races that started at least ``after`` (default LIVE_AFTER) ago, races with full results)."""
    cut = (now - (after or LIVE_AFTER)).strftime("%Y-%m-%dT%H:%M")
    due = conn.execute("SELECT COUNT(*) FROM races WHERE date=? AND venue=? AND post_time IS NOT NULL "
                       "AND post_time<=?", (date, venue, cut)).fetchone()[0]
    return due, conn.execute(DONE_RACES_SQL, (date, venue)).fetchone()[0]


def cmd_live(conn, date, venue=None):
    """賽馬進行中：抓已跑完場次嘅賽果、結算已完成嘅注項、更新網站（唔使等全日跑完）。"""
    from . import paper, scrape

    row = conn.execute("SELECT venue FROM meetings WHERE date=?" + (" AND venue=?" if venue else ""),
                       (date, venue) if venue else (date,)).fetchone()
    venue = venue or (row["venue"] if row else None)
    before = _races_due_vs_done(conn, date, venue, now_hkt())[1] if venue else 0
    wins_before = conn.execute(WIN_RACES_SQL, (date, venue)).fetchone()[0] if venue else 0
    n = scrape.scrape_meeting_results(conn, date, venue, refresh=True)
    if not n:
        log.info("live results %s: none yet", date)
        return 0
    venue = venue or conn.execute("SELECT venue FROM meetings WHERE date=?", (date,)).fetchone()["venue"]
    after = _races_due_vs_done(conn, date, venue, now_hkt())[1]
    wins_after = conn.execute(WIN_RACES_SQL, (date, venue)).fetchone()[0]
    settled = paper.settle(conn, date)
    log.info("live results %s %s: %s races with full results (was %s), %s with dividends (was %s), %s tickets settled",
             date, venue, after, before, wins_after, wins_before, settled)
    if after != before or wins_after != wins_before or settled:
        _publish(conn, f"即時賽果 {date} {venue} 第{after}場")
    return after


# --------------------------------------------------------------- post-race
def cmd_post(conn, date=None):
    from . import paper, pipeline, scrape

    if date is None:
        row = conn.execute(
            "SELECT date, venue FROM meetings WHERE date<=? AND (status IS NULL OR status NOT IN ('results','abandoned')) "
            "ORDER BY date DESC LIMIT 1", (now_hkt().strftime("%Y-%m-%d"),)
        ).fetchone()
        if not row:
            log.info("no meeting awaiting results")
            return None
        date, venue = row["date"], row["venue"]
    else:
        venue = None
    n = scrape.scrape_meeting_results(conn, date, venue)
    if not n:
        _maybe_give_up(conn, date)
        log.info("results not available yet for %s", date)
        return None
    st = conn.execute("SELECT venue, status FROM meetings WHERE date=?", (date,)).fetchone()
    if st is None or st["status"] not in ("results", "abandoned"):
        log.info("results for %s still preliminary", date)
        return None
    venue = st["venue"]
    settled = paper.settle(conn, date)
    run_id = f"post-{date}-{venue}"
    result = pipeline.post_meeting(conn, run_id, date, venue)
    path = REPORT_DIR / f"{date}_{venue}_賽後分析.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result["report_md"], encoding="utf-8")
    _record_run(conn, run_id, "post_analysis", date, venue, {"settled": settled, **result.get("notes", {})})
    log.info("post report written: %s", path)
    _publish(conn, f"賽後 {date} {venue}")
    return path


def _maybe_give_up(conn, date):
    """The results site has no page at all for a meeting NO_RESULTS_GIVE_UP_DAYS after
    its date (e.g. whole meeting cancelled by typhoon): mark it 'abandoned' so the
    scheduler stops retrying, and settle its paper tickets (every leg refunded)."""
    from . import paper

    if datetime.strptime(date, "%Y-%m-%d") > now_hkt() - timedelta(days=NO_RESULTS_GIVE_UP_DAYS):
        return False
    cur = conn.execute("UPDATE meetings SET status='abandoned', updated_at=? WHERE date=? AND "
                       "(status IS NULL OR status NOT IN ('results','abandoned'))",
                       (now_hkt().isoformat(timespec="seconds"), date))
    conn.commit()
    if cur.rowcount:
        log.warning("%s: still no results %d days later - marked abandoned; settled %d paper tickets",
                    date, NO_RESULTS_GIVE_UP_DAYS, paper.settle(conn, date))
    return bool(cur.rowcount)


# ------------------------------------------------------------------ tick
def _next_meeting(conn, now):
    """Next meeting from the meetings table when its info is < 1 h old (no HTTP);
    otherwise scrape.next_meeting (which refreshes the meetings row)."""
    from . import scrape

    row = conn.execute(
        "SELECT date, venue, first_post_time, n_races, updated_at FROM meetings WHERE first_post_time > ? "
        "AND (status IS NULL OR status IN ('scheduled','card')) ORDER BY first_post_time LIMIT 1",
        (now.strftime("%Y-%m-%dT%H:%M"),)).fetchone()
    if row and row["updated_at"] and row["updated_at"][:19] >= (now - NEXT_MEETING_CACHE).isoformat(timespec="seconds"):
        return {"date": row["date"], "venue": row["venue"], "first_post_time": row["first_post_time"],
                "n_races": row["n_races"]}
    return scrape.next_meeting(conn)


def cmd_tick(conn):
    """Idempotent: safe to call every few minutes.  Errors are logged, never raised,
    so one failing step (e.g. network down) doesn't stop the others; the next tick
    simply retries."""
    now = now_hkt()
    since = (now - timedelta(days=POST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    # 00) per-race late betting (time-critical, so first): each race ~8 min before its start
    try:
        _race_bet_step(conn, now)
    except Exception:
        log.exception("tick: race_bet step failed (will retry next tick)")
    # 0) live: today's meeting in progress -> results race by race (each race ~8 min after its start)
    for lm in conn.execute(
            "SELECT m.date, m.venue, MIN(r.post_time) AS first_post, MAX(r.post_time) AS last_post FROM meetings m "
            "JOIN races r ON r.date=m.date AND r.venue=m.venue WHERE m.date=? AND "
            "(m.status IS NULL OR m.status NOT IN ('results','abandoned')) GROUP BY m.date, m.venue",
            (now.strftime("%Y-%m-%d"),)).fetchall():
        if not lm["first_post"] or not lm["last_post"]:
            continue
        if _parse_ts(lm["first_post"]) + LIVE_AFTER <= now < _parse_ts(lm["last_post"]) + POST_DELAY:
            due, done = _races_due_vs_done(conn, lm["date"], lm["venue"], now)
            if due > done:
                try:
                    cmd_live(conn, lm["date"], lm["venue"])
                except Exception:
                    log.exception("tick: live results for %s %s failed (will retry next tick)", lm["date"], lm["venue"])
    # 1) post: any recent meeting whose last race should be over and not yet 'results'
    pend = conn.execute(
        "SELECT m.date, m.venue, MAX(r.post_time) AS last_post FROM meetings m "
        "LEFT JOIN races r ON r.date=m.date AND r.venue=m.venue "
        "WHERE m.date<=? AND m.date>=? AND (m.status IS NULL OR m.status NOT IN ('results','abandoned')) "
        "GROUP BY m.date, m.venue",
        (now.strftime("%Y-%m-%d"), since),
    ).fetchall()
    for p in pend:
        last = _parse_ts(p["last_post"]) if p["last_post"] else datetime.strptime(p["date"], "%Y-%m-%d") + timedelta(hours=23)
        if now >= last + POST_DELAY:
            try:
                cmd_post(conn, p["date"])
            except Exception:
                log.exception("tick: post step for %s %s failed (will retry next tick)", p["date"], p["venue"])
    # 1b) live results already marked the meeting 'results' -> the post step (settle, report,
    #     retrain) still has to run once for meetings this system tracked (had a pre run)
    for p in conn.execute(
            "SELECT m.date, m.venue, MAX(r.post_time) AS last_post FROM meetings m "
            "JOIN races r ON r.date=m.date AND r.venue=m.venue WHERE m.status='results' AND m.date>=? AND m.date<=? "
            "AND EXISTS (SELECT 1 FROM runs u WHERE u.date=m.date AND u.venue=m.venue AND u.kind LIKE 'pre%') "
            "AND NOT EXISTS (SELECT 1 FROM runs u WHERE u.date=m.date AND u.venue=m.venue AND u.kind='post_analysis') "
            "GROUP BY m.date, m.venue", (since, now.strftime("%Y-%m-%d"))).fetchall():
        if p["last_post"] and now >= _parse_ts(p["last_post"]) + POST_DELAY:
            try:
                cmd_post(conn, p["date"])
            except Exception:
                log.exception("tick: post step for %s %s failed (will retry next tick)", p["date"], p["venue"])
    if now.minute < 10:     # about once an hour: meetings the scheduler no longer retries
        for r in conn.execute("SELECT date, venue, status FROM meetings WHERE date<? AND "
                              "(status IS NULL OR status NOT IN ('results','abandoned'))", (since,)):
            log.warning("meeting %s %s still '%s' - run `python -m hkjc post %s` by hand",
                        r["date"], r["venue"], r["status"], r["date"])

    # 2) pre: next meeting info (cached hourly); run the betting step in the window
    try:
        mtg = _next_meeting(conn, now)
    except Exception:
        log.exception("tick: next meeting lookup failed (will retry next tick)")
        return
    if not mtg or not mtg.get("first_post_time"):
        return
    first = _parse_ts(mtg["first_post_time"])
    d, v = mtg["date"], mtg["venue"]
    done = _run_exists(conn, "pre_bet", d, v) or _run_exists(conn, "pre_report", d, v)
    log.info("tick: next meeting %s %s, race 1 %s, pre-race run %s", d, v, mtg["first_post_time"],
             "done" if done else f"due {first - PRE_LEAD:%Y-%m-%d %H:%M}")
    if done:
        return
    if now < first - PRE_LEAD and first - now <= SCORES_HORIZON:
        last = conn.execute("SELECT MAX(created_at) FROM runs WHERE kind='pre_scores' AND date=? AND venue=?",
                            (d, v)).fetchone()[0]
        if not last or _parse_ts(last) <= now - SCORES_REFRESH:
            try:
                cmd_scores(conn, mtg=mtg)
            except Exception:
                log.exception("tick: scores preview for %s %s failed (will retry next tick)", d, v)
        return
    try:
        if first - PRE_LEAD <= now <= first - PRE_LATEST:
            if now > first - PRE_LEAD + timedelta(minutes=20):
                log.warning("pre-race run is late (scheduler missed the 2h mark): %s before race 1",
                            first - now)
            cmd_pre(conn, mtg=mtg)
        elif first - PRE_LATEST < now < first:
            log.warning("too late for paper bets (%s before race 1): report only, no tickets", first - now)
            cmd_pre(conn, mtg=mtg, place_bets=False)
    except Exception:
        log.exception("tick: pre step for %s %s failed (will retry next tick)", d, v)


def cmd_report(conn):
    from . import paper
    print(paper.summary_md(conn))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hkjc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill"); b.add_argument("start"); b.add_argument("end")
    p = sub.add_parser("post"); p.add_argument("date", nargs="?")
    q = sub.add_parser("pre"); q.add_argument("--force", action="store_true")
    sub.add_parser("scores")
    sub.add_parser("resettle")
    lv = sub.add_parser("live"); lv.add_argument("date", nargs="?")
    rb = sub.add_parser("race_bet"); rb.add_argument("date"); rb.add_argument("venue"); rb.add_argument("race", type=int)
    sub.add_parser("tick")
    t = sub.add_parser("backtest"); t.add_argument("start", nargs="?"); t.add_argument("end", nargs="?")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "hkjc.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    lock = None
    if args.cmd in ("pre", "post", "tick", "race_bet", "resettle", "live", "scores"):
        lock = _acquire_lock()
        if lock is None:
            log.warning("another hkjc pre/post/tick run is in progress - %s skipped", args.cmd)
            return
    conn = connect()
    try:
        if args.cmd == "backfill":
            from . import scrape
            scrape.backfill(conn, args.start, args.end)
        elif args.cmd == "post":
            print(cmd_post(conn, args.date))
        elif args.cmd == "live":
            print(cmd_live(conn, args.date or now_hkt().strftime("%Y-%m-%d")))
        elif args.cmd == "race_bet":
            print(cmd_race_bet(conn, args.date, args.venue.upper(), args.race))
        elif args.cmd == "resettle":
            from . import paper
            res = paper.resettle_wrong_refunds(conn)
            print(res)
            print("（本機跑完要 `python -m hkjc.dbdump dump data/hkjc.db data/db` 再 commit，"
                  "雲端先會用到；雲端用 Actions 指令 resettle 會自動 commit）")
            for d in res["dates"]:
                _publish(conn, f"重新結算 {d}")
        elif args.cmd == "scores":
            print(cmd_scores(conn, force=True))
        elif args.cmd == "pre":
            print(cmd_pre(conn, force=args.force))
        elif args.cmd == "tick":
            cmd_tick(conn)
        elif args.cmd == "backtest":
            from . import backtest
            backtest.main(conn, args.start, args.end)
        elif args.cmd == "report":
            cmd_report(conn)
    except Exception:
        log.exception("command %s failed", args.cmd)
        raise
    finally:
        conn.close()
        if lock is not None:
            lock.close()


def _acquire_lock():
    """Non-blocking exclusive lock (data/logs/hkjc.lock) so two pre/post/tick runs
    (e.g. launchd + a manual run) never work on the same meeting at once.
    Returns the open lock file, or None if another run holds it."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOG_DIR / LOCK_NAME, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


if __name__ == "__main__":
    main()
