"""Pre-meeting and post-meeting workflows called by cli.py (reports in Traditional Chinese)."""
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from . import bots, model, paper, picks, publish, scores, strategy
from .odds import latest_odds

log = logging.getLogger("hkjc.pipeline")

# Per-race late betting: the T-2h run (pre_meeting, kind 'pre_bet') places only the MULTI-RACE
# tickets; single-race decisions / 今日精選 items are written as 'pending' and decided ~8 min
# before each race with that race's latest odds (race_bet, via hkjc.cli.cmd_race_bet).
PER_RACE = True
# race_bet also refreshes the race's predictions + 評分 stored under the T-2h run, so the site /
# post-race report show the numbers the race-time decisions used.
RACE_BET_UPDATE_SCORES = True

VENUE_ZH = {"ST": "沙田", "HV": "跑馬地"}
SURFACE_ZH = {"TURF": "草地", "AWT": "全天候(泥地)"}
DISCLAIMER = ("> ⚠️ 本報告只供模擬投注 (paper trade) 研究用途，並非投資或投注建議。"
              "馬會各彩池抽水約 17.5%–25%，長期回測顯示大部分策略為負回報。博彩要有節制。")


def _pct(x, nd=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{nd}f}%"


def _num(x, nd=1):
    return "-" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(冇資料)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def _ticket_lines(tickets, names) -> str:
    out = []
    for t in tickets:
        legs = []
        for leg in t["legs"]:
            sel = "、".join(f"{h} {names.get((leg['race_no'], int(h)), '')}".strip() for h in leg.get("selections", []))
            if leg.get("bankers"):
                bk = "、".join(f"{h} {names.get((leg['race_no'], int(h)), '')}".strip() for h in leg["bankers"])
                sel = f"膽 {bk}　拖 {sel}"
            if leg.get("positions"):
                sel = " / ".join(",".join(str(h) for h in pos) for pos in leg["positions"])
            pool = leg.get("pool") or t.get("pool") or ""
            legs.append(f"第{leg['race_no']}場 {pool}: {sel}")
        status = ""
        if t.get("payout") is not None:
            status = f" → 派彩 ${t['payout']:,.2f}（盈虧 {t['profit']:+,.2f}）"
        kind = t.get("spec") or (t.get("notes") or {}).get("structure") or t.get("pool") or ""
        # ADAPTIVE tickets are numbered by race number: say so instead of "第N條"
        label = (f"{t['strategy']}（第{t['legs'][0]['race_no']}場）" if t["strategy"] == "ADAPTIVE" and t["legs"]
                 else f"{t['strategy']} 第{t.get('ticket_no', 1)}條")
        out.append(f"- **{label}**（{kind}，"
                   f"{t['n_combos']} 注 × ${t['unit_stake']:.0f} = ${t['total_stake']:,.0f}）{status}\n  "
                   + "；".join(legs))
    return "\n".join(out) + "\n" if out else "_(冇投注)_\n"


def _backtest_hint(conn) -> str:
    rows = conn.execute(
        "SELECT strategy, params, n_bets, roi FROM backtest_results "
        "WHERE strategy IN ('PLA 6x63 [confidence]', '3xQPL 2x3 [confidence]')").fetchall()
    out = []
    for r in rows:
        p = json.loads(r["params"])
        if p.get("period") != "all":
            continue
        m = p.get("metrics", {})
        out.append(f"- {r['strategy']}（來源 {p.get('source')}）：{r['n_bets']} 個賽馬日，ROI {_pct(r['roi'])}，"
                   f"有錢賺嘅賽馬日 {_pct(m.get('profitable_rate'))}")
    if not out:
        return ""
    return "### 回測提示（固定組合過往表現）\n" + "\n".join(sorted(out)) + "\n詳見 `reports/回測報告.md`。\n"


def _adaptive_md(pred, odds, config) -> str:
    """Per-race ADAPTIVE decision table (按場自動決定)."""
    try:
        cfg = ((config or {}).get("ADAPTIVE") or {})
        plan = strategy.adaptive_plan(pred, odds=odds or None, cfg=cfg)
    except Exception as e:  # report must not fail because of the adaptive table
        log.warning("adaptive plan failed: %s", e)
        return ""
    rows = []
    for p in plan:
        pr, ch, best = p["profile"], p["chosen"], p["best"]
        if ch:
            leg = ch["leg"]
            sel = ("膽 " + ",".join(map(str, leg["bankers"])) + " 拖 " if leg.get("bankers") else "") + \
                ",".join(map(str, leg.get("selections", [])))
            dec = f"{ch['name']}（{sel}，${ch['stake']:.0f}）"
            er, hit = ch["exp_roi"], ch["hit"]
        else:
            tau = {**strategy.ADAPTIVE_DEFAULT, **cfg}["tau"]
            why = "命中機率太低" if best and best["exp_roi"] >= tau else "預期回報唔夠"
            dec = "唔落" + (f"（{why}；最好係 {best['name']} 預期 {best['exp_roi'] * 100:+.0f}%、"
                           f"命中 {best['hit'] * 100:.0f}%）" if best else "（冇賠率估唔到）")
            er, hit = None, None
        rows.append({"場次": p["race_no"], "首選": pr["top_horse"], "p1": _pct(pr["p1"]), "p2": _pct(pr["p2"]),
                     "差距": _pct(pr["gap"]), "模型/市場": _num(pr.get("edge1"), 2), "決定": dec,
                     "預期ROI": _pct(er), "命中機率": _pct(hit)})
    if not rows:
        return ""
    return ("## 按場自動決定（ADAPTIVE）\n"
            "每場按模型機率 + 賠率計每個玩法嘅預期回報同命中機率，揀預期 ROI 最高而且過門檻嘅玩法；"
            "冇合格就唔落。詳見 `reports/回測報告.md`「按場自動決定策略」。\n\n" + _table(pd.DataFrame(rows)))


def _bots_md(decisions, names, conn) -> str:
    out = []
    for b in bots.BOTS:
        ds = [d for d in decisions if d["bot_id"] == b["id"]]
        bets = [d for d in ds if d["action"] == "bet" and d.get("stake")]
        stake = sum(d["stake"] or 0 for d in bets)
        out.append(f"### {b['name_zh']}（{b['style_zh']}）— 落 {len(bets)} 注，${stake:,.0f}；"
                   f"資金 ${paper.bankroll(conn, bot_id=b['id']):,.2f}")
        for d in ds:
            if d["action"] == "bet" and d.get("stake"):
                where = f"第{d['race_no']}場" if d["race_no"] else "過關"
                out.append(f"- ✅ {where} {d.get('structure') or ''}：{d['reason']}")
        pend = [d for d in ds if d["action"] == bots.PENDING]
        if pend:
            prev = [f"第{d['race_no']}場 {d.get('structure') or ''} ${d['stake'] or 0:,.0f}"
                    for d in pend if d.get("stake")]
            out.append(f"- ⏳ 待定 {len({d['race_no'] for d in pend})} 場（單場注開跑前約 8 分鐘用最新賠率先決定）"
                       + ("；而家預覽會買：" + "；".join(prev[:12]) if prev else "；而家預覽全部唔買"))
        skipped = [d for d in ds if d["action"] not in ("bet", bots.PENDING)]
        if skipped:
            out.append("- 唔買：" + "；".join(
                (f"第{d['race_no']}場 " if d["race_no"] else "全日 ") + d["reason"] for d in skipped[:12]))
        out.append("")
    return "\n".join(out) + "\n"


def _picks_md(pk) -> str:
    """「## 今日精選」: the analysis-driven picks (hkjc.picks) - same items as the site."""
    return picks.picks_md(pk)


def _post_times(conn, date, venue) -> dict:
    return {int(r[0]): r[1] for r in conn.execute("SELECT race_no, post_time FROM races WHERE date=? AND venue=?",
                                                  (date, venue))}


def _picks_cfg(conn, date, venue):
    nr = conn.execute("SELECT n_races FROM meetings WHERE date=? AND venue=?", (date, venue)).fetchone()
    return {"n_races": nr[0]} if nr and nr[0] else None


def _make_picks(conn, run_id, date, venue, pred, odds, per_race=False):
    """今日精選 for this run (stored in table picks).  Never raises: None on failure.
    ``per_race``: single-race items get status 'pending' + decide_at, the others 'final'."""
    try:
        pinput = pred.copy()
        pt = _post_times(conn, date, venue)
        pinput["post_time"] = [pt.get(int(r)) for r in pinput["race_no"]]
        pk = picks.make_picks(pinput, odds or None, _picks_cfg(conn, date, venue))
        if per_race:
            from .cli import RACE_BET_LEAD
            picks.mark_pending(pk, pt, int(RACE_BET_LEAD.total_seconds() // 60))
        picks.save_picks(conn, run_id, date, venue, pk)
        return pk
    except Exception:
        log.exception("今日精選 failed; robots / report continue without it")
        return None


def _live_frame(conn, date, venue, pred, odds):
    """Model predictions + racecard columns + live WIN / PLA odds, scratched horses removed."""
    card = pd.read_sql_query(
        "SELECT race_no, horse_no, horse_name, jockey_name, trainer_name, draw, weight, rating, scratched "
        "FROM racecard WHERE date=? AND venue=?", conn, params=(date, venue))
    pred = pred.drop(columns=[c for c in ("horse_name",) if c in pred.columns]).merge(
        card, on=["race_no", "horse_no"], how="left")
    pred = pred[pred["scratched"].fillna(0) == 0].copy()

    def _o(rn, pool, h):
        return (odds.get(int(rn), {}).get(pool, {}) or {}).get(str(int(h))) if odds else None
    pred["win_odds_live"] = [_o(r, "WIN", h) for r, h in zip(pred.race_no, pred.horse_no)]
    pred["pla_odds"] = [_o(r, "PLA", h) for r, h in zip(pred.race_no, pred.horse_no)]
    return pred


# ------------------------------------------------------------- preview scores
def preview_scores(conn, run_id, date, venue) -> int:
    """賠率一開售就出評分（唔落注、唔出精選）：racecard + latest odds -> predictions +
    horse_scores under ``run_id`` (kind 'pre_scores').  Older preview runs of the same
    meeting are deleted so the DB / git history doesn't grow every refresh.
    Returns the number of horses scored."""
    odds = latest_odds(conn, date, venue)
    pred = model.predict_card(conn, date, venue, kind="combined", odds=odds or None)
    if pred.empty:
        return 0
    card = pd.read_sql_query("SELECT race_no, horse_no, horse_name, jockey_name, trainer_name, scratched "
                             "FROM racecard WHERE date=? AND venue=?", conn, params=(date, venue))
    pred = pred.drop(columns=[c for c in ("horse_name",) if c in pred.columns]).merge(
        card, on=["race_no", "horse_no"], how="left")
    pred = pred[pred["scratched"].fillna(0) == 0].copy()
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, race_id, horse_no, p_win, p_place, model_version, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(run_id, r.race_id, int(r.horse_no), float(r.p_win), float(r.p_place), r.model_version, now)
         for r in pred.itertuples()])
    sc = scores.score_meeting(conn, date, venue, pred_df=pred, odds=odds or None)
    scores.save_scores(conn, run_id, sc)
    try:
        preview_decisions(conn, run_id, date, venue, pred, odds, sc)
    except Exception:     # the scores preview must never fail because of the decision preview
        log.exception("decision preview failed for %s %s", date, venue)
    old = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE kind='pre_scores' AND date=? AND venue=? "
                                      "AND run_id!=?", (date, venue, run_id))]
    for rid in old:
        for t in PREVIEW_TABLES:
            conn.execute(f"DELETE FROM {t} WHERE run_id=?", (rid,))
    conn.commit()
    return len(sc)


PREVIEW_TABLES = ("horse_scores", "predictions", "bot_decisions", "picks", "runs")


def preview_decisions(conn, run_id, date, venue, pred, odds, sc=None) -> int:
    """預覽（未落注）: every robot's decisions + 今日精選 with the current odds, written under the
    preview ``run_id`` - bot_decisions rows with ticket_id NULL and the picks dict in ``picks``.
    NEVER places tickets or touches paper_tickets / bankroll_ledger (bots.decide only reads the
    bankroll).  Skipped (returns 0) once the meeting has a betting run or any paper ticket, so a
    forced preview can never mix with real decisions.  Returns the number of decision rows."""
    if conn.execute("SELECT 1 FROM runs WHERE date=? AND venue=? AND kind IN ('pre_bet','pre_report') LIMIT 1",
                    (date, venue)).fetchone() or \
            conn.execute("SELECT 1 FROM paper_tickets WHERE date=? AND venue=? LIMIT 1", (date, venue)).fetchone():
        log.info("decision preview skipped for %s %s: a betting run exists", date, venue)
        return 0
    pred = pred.copy()

    def _o(rn, pool, h):
        return (odds.get(int(rn), {}).get(pool, {}) or {}).get(str(int(h))) if odds else None
    pred["win_odds_live"] = [_o(r, "WIN", h) for r, h in zip(pred.race_no, pred.horse_no)]
    pred["pla_odds"] = [_o(r, "PLA", h) for r, h in zip(pred.race_no, pred.horse_no)]
    if sc is not None and len(sc) and "score" not in pred.columns:
        pred = pred.merge(sc[["race_no", "horse_no", "score"]], on=["race_no", "horse_no"], how="left")
    pk = _make_picks(conn, run_id, date, venue, pred, odds)       # stored in table picks under run_id
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for b in bots.BOTS:
        try:
            decs = bots.decide(b, pred, odds or None, paper.bankroll(conn, bot_id=b["id"]), picks=pk)
        except Exception as e:
            log.exception("robot %s preview failed", b["id"])
            decs = bots._error_rows(b["id"], pred, e)
        for d in decs:
            rows.append((run_id, b["id"], date, venue, d["race_no"], d["action"], d["structure"],
                         json.dumps(d["selections"], ensure_ascii=False, default=int), d["stake"], d["reason"],
                         None, now))
    conn.execute("DELETE FROM bot_decisions WHERE run_id=?", (run_id,))
    conn.executemany(
        "INSERT OR REPLACE INTO bot_decisions (run_id, bot_id, date, venue, race_no, action, structure, "
        "selections, stake, reason, ticket_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------- pre-meeting
def pre_meeting(conn, run_id, date, venue, config=None, place_bets=True) -> dict:
    """place_bets=False (scheduler woke up too late, < 15 min before race 1): the
    report and predictions are still produced, but no paper tickets are placed."""
    odds = latest_odds(conn, date, venue)
    pred = model.predict_card(conn, date, venue, kind="combined", odds=odds or None)
    if pred.empty:
        raise RuntimeError(f"no racecard predictions for {date} {venue}")
    pred = _live_frame(conn, date, venue, pred, odds)
    per_race = bool(place_bets and PER_RACE)

    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, race_id, horse_no, p_win, p_place, model_version, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(run_id, r.race_id, int(r.horse_no), float(r.p_win), float(r.p_place), r.model_version, now)
         for r in pred.itertuples()])
    conn.commit()

    # 馬匹評分 + 理由
    try:
        sc = scores.score_meeting(conn, date, venue, pred_df=pred, odds=odds or None)
        scores.save_scores(conn, run_id, sc)
        pred = pred.merge(sc[["race_no", "horse_no", "score", "rank_in_race", "reasons"]],
                          on=["race_no", "horse_no"], how="left")
    except Exception:
        log.exception("scoring failed; continuing without scores")
        pred["score"], pred["reasons"] = float("nan"), None

    # 今日精選（分析引擎；daily_picks 機械人照買）
    pk = _make_picks(conn, run_id, date, venue, pred, odds, per_race=per_race)
    if place_bets:     # the real run supersedes the 預覽: drop the preview decisions / picks of this meeting
        for (rid,) in conn.execute("SELECT run_id FROM runs WHERE kind='pre_scores' AND date=? AND venue=?",
                                   (date, venue)).fetchall():
            for t in ("bot_decisions", "picks"):
                conn.execute(f"DELETE FROM {t} WHERE run_id=?", (rid,))
        conn.commit()

    # 機械人各自決定（每場都有買/唔買原因）
    if place_bets:
        decisions = bots.run_bots(conn, run_id, date, venue, pred, odds or None, picks=pk, defer_single=per_race)
    else:
        decisions = []
        for b in bots.BOTS:
            try:
                decisions += bots.decide(b, pred, odds or None, paper.bankroll(conn, bot_id=b["id"]), picks=pk)
            except Exception:     # report-only run: a robot's bug must not lose the report
                log.exception("robot %s failed (report-only run)", b["id"])
    tickets = [dict(d["ticket"], bot_id=d["bot_id"]) for d in decisions if d.get("ticket")]
    ids = [d.get("ticket_id") for d in decisions if d.get("ticket_id")]
    names = {(int(r.race_no), int(r.horse_no)): r.horse_name for r in pred.itertuples()}

    snap = conn.execute("SELECT MAX(snapshot_at) FROM odds_snapshots WHERE date=? AND venue=?",
                        (date, venue)).fetchone()[0]
    races = conn.execute("SELECT race_no, post_time, distance, surface, course, race_class, rating_band, race_name "
                         "FROM races WHERE date=? AND venue=? ORDER BY race_no", (date, venue)).fetchall()
    md = [f"# {date} {VENUE_ZH.get(venue, venue)} 賽前投注參考\n",
          f"生成時間：{now}　｜　賠率快照：{snap or '未有（賠率未開售，只用模型）'}　｜　模型：{pred['model_version'].iloc[0]}\n",
          DISCLAIMER + "\n",
          _picks_md(pk),
          ("## 模擬投注（機械人已記錄）\n"
           + ("過關／跨場注而家落；**單場注**（標「⏳ 待定」）每場開跑前約 8 分鐘用該場最新賠率先決定。\n"
              if per_race else "") if place_bets else
           "## 參考組合（⚠️ 排程太遲執行，已過落注時限：只供參考，機械人冇記錄模擬注）\n"),
          _bots_md(decisions, names, conn),
          _backtest_hint(conn),
          _adaptive_md(pred, odds, config),
          "## 各場預測\n",
          "欄位：勝率/入三甲機率＝模型估計；合理獨贏＝1/勝率；位置EV＝入三甲機率×位置賠率（>1 代表賠率高過模型估值）。\n"]
    for rc in races:
        g = pred[pred.race_no == rc["race_no"]].sort_values("p_place", ascending=False)
        if g.empty:
            continue
        t = pd.DataFrame({
            "馬號": g.horse_no.astype(int), "馬名": g.horse_name, "騎師": g.jockey_name, "練馬師": g.trainer_name,
            "檔位": g.draw.astype("Int64"), "負磅": g.weight.astype("Int64"), "勝率": g.p_win.map(_pct), "入三甲機率": g.p_place.map(_pct),
            "評分": g["score"].map(lambda x: _num(x, 0)),
            "理由": g["reasons"].map(lambda r: "；".join(r) if isinstance(r, list) else ""),
            "合理獨贏": (1 / g.p_win).map(lambda x: _num(x)), "獨贏賠率": g.win_odds_live.map(_num),
            "位置賠率": g.pla_odds.map(_num),
            "位置EV": (g.p_place * g.pla_odds).map(lambda x: _num(x, 2)),
        })
        md.append(f"### 第{rc['race_no']}場 {rc['post_time'][11:] if rc['post_time'] else ''}　"
                  f"{rc['race_class'] or ''} {rc['distance'] or ''}米 {SURFACE_ZH.get(rc['surface'], rc['surface'] or '')}"
                  f" {('「' + rc['course'] + '」賽道') if rc['course'] else ''}　{rc['race_name'] or ''}\n")
        md.append(_table(t))
    md.append(_long_term_md(conn, venue, date))
    return {"report_md": "\n".join(md), "notes": {"tickets": ids, "n_pred": len(pred), "odds_snapshot": snap,
                                                  "placed": bool(place_bets), "per_race": per_race}}


# ------------------------------------------------------------ per-race late betting
def _score_race(conn, date, venue, pred, odds):
    """評分 of the race(s) in ``pred`` (upcoming-meeting path even after earlier races have results:
    scores.score_meeting switches to the training frame once any race of the day is in)."""
    if not scores._is_past(conn, date, venue):
        return scores.score_meeting(conn, date, venue, pred_df=pred, odds=odds or None)
    from . import features as F
    card = F.build_card_frame(conn, date, venue, odds=odds or None)
    card = card[card["race_id"].isin(set(pred["race_id"]))]
    nm = scores._names(conn, date, venue, False)
    card = card.drop(columns=[c for c in ("jockey_name", "trainer_name") if c in card.columns]).merge(
        nm[["race_id", "horse_no", "jockey_name", "trainer_name"]], on=["race_id", "horse_no"], how="left")
    keys = set(zip(card["race_id"], card["horse_no"].astype(int)))
    pred = pred[[(a, int(b)) in keys for a, b in zip(pred["race_id"], pred["horse_no"])]]
    return scores.score_horses(pred, card)


def race_bet(conn, pre_run_id, race_run_id, date, venue, race_no, deadline=None, clock=None) -> dict:
    """Race-time step (~8 min before race ``race_no``): re-predict with the latest odds (only this
    race's rows are used), re-evaluate the race's pending 今日精選 items and let every robot decide
    this race's single-race bets (bots.race_bots: tickets under ``race_run_id``, pending rows of the
    T-2h run ``pre_run_id`` replaced).  Odds / racecard must already be refreshed by the caller.
    Returns notes for the runs row."""
    rn = int(race_no)
    odds = latest_odds(conn, date, venue)
    pred = model.predict_card(conn, date, venue, kind="combined", odds=odds or None)
    if len(pred):
        pred = _live_frame(conn, date, venue, pred, odds)
        pred = pred[pred["race_no"].astype(int) == rn].copy()
    now = datetime.now().isoformat(timespec="seconds")
    rid = f"{date.replace('-', '')}-{venue}-{rn:02d}"
    if len(pred):
        try:
            sc = _score_race(conn, date, venue, pred, odds)
            if len(sc):
                pred = pred.merge(sc[["race_no", "horse_no", "score", "rank_in_race", "reasons"]],
                                  on=["race_no", "horse_no"], how="left")
            if RACE_BET_UPDATE_SCORES:
                conn.execute("DELETE FROM predictions WHERE run_id=? AND race_id=?", (pre_run_id, rid))
                conn.executemany(
                    "INSERT OR REPLACE INTO predictions (run_id, race_id, horse_no, p_win, p_place, model_version, "
                    "created_at) VALUES (?,?,?,?,?,?,?)",
                    [(pre_run_id, r.race_id, int(r.horse_no), float(r.p_win), float(r.p_place), r.model_version, now)
                     for r in pred.itertuples()])
                conn.commit()
                if len(sc):
                    scores.save_scores(conn, pre_run_id, sc)
        except Exception:
            log.exception("race_bet %s %s R%s: scoring failed; deciding without new scores", date, venue, rn)
        pt = _post_times(conn, date, venue)
        pred["post_time"] = [pt.get(int(r)) for r in pred["race_no"]]
    else:
        log.warning("race_bet %s %s R%s: no predictions for the race (all runners scratched?)", date, venue, rn)
    pk = picks.load_picks(conn, date, venue, run_id=pre_run_id)
    n_items = 0
    if pk:
        try:
            if len(pred):
                upd = picks.reevaluate_race(pk, pred, odds or None, rn, _picks_cfg(conn, date, venue))
            else:
                upd = picks.reevaluate_race(pk, pd.DataFrame(columns=["race_no", "horse_no", "p_win", "p_place"]),
                                            odds or None, rn)
            n_items = len(upd)
            if upd:
                picks.save_picks(conn, pre_run_id, date, venue, pk)
        except Exception:
            log.exception("race_bet %s %s R%s: 今日精選 re-evaluation failed (daily_picks buys nothing new)",
                          date, venue, rn)
            try:     # don't leave the race's items 'pending' forever: show them as not bought
                for it in pk.get("items") or []:
                    if it.get("status") == "pending" and int(it.get("race_no") or 0) == int(rn):
                        it["status"] = "missed"
                        it["reason"] = "系統計算出錯，今場冇落注（原本預覽：" + str(it.get("reason") or "") + "）"
                picks.save_picks(conn, pre_run_id, date, venue, pk)
            except Exception:
                log.exception("race_bet %s %s R%s: could not mark picks items missed", date, venue, rn)
            pk = None
    if pk is None:
        pk = {"items": []}
    decs = bots.race_bots(conn, pre_run_id, race_run_id, date, venue, rn,
                          pred if len(pred) else pd.DataFrame(columns=["race_no", "horse_no", "p_win", "p_place"]),
                          odds or None, picks=pk, deadline=deadline, clock=clock)
    snap = conn.execute("SELECT MAX(snapshot_at) FROM odds_snapshots WHERE date=? AND venue=? AND race_no=?",
                        (date, venue, rn)).fetchone()[0]
    bets_ = [d for d in decs if d["action"] == "bet" and d.get("ticket_id")]
    return {"race_no": rn, "pre_run_id": pre_run_id, "n_pred": int(len(pred)), "odds_snapshot": snap,
            "decisions": len(decs), "bets": len(bets_), "tickets": [d["ticket_id"] for d in bets_],
            "stake": round(sum(float(d["stake"] or 0) for d in bets_), 2), "picks_items": n_items,
            "bots": sorted({d["bot_id"] for d in decs})}


def race_missed(conn, pre_run_id, date, venue, race_no, race_run_id=None) -> dict:
    """The race-time window passed without (a complete) decision: orphan tickets of the race run
    (placed but never linked to a decision row) are voided + refunded, pending rows -> skip, and the
    race's picks items that were never bought (pending, or final without a ticket) -> missed."""
    voided = bots.reconcile_race_orphans(conn, pre_run_id, race_run_id, race_no) if race_run_id else []
    n = bots.mark_race_missed(conn, pre_run_id, race_no)
    m = 0
    pk = picks.load_picks(conn, date, venue, run_id=pre_run_id)
    if pk:
        m = picks.mark_missed(pk, race_no)
        bought = set()
        for (det,) in conn.execute("SELECT detail FROM paper_tickets WHERE date=? AND venue=? AND bot_id='daily_picks' "
                                   "AND status!='void'", (date, venue)):
            try:
                k = (json.loads(det or "{}").get("notes") or {}).get("pick_key")
            except ValueError:
                k = None
            if k:
                bought.add(k)
        for it in pk.get("items") or []:
            if (int(it.get("race_no") or 0) == int(race_no) and it.get("status") == "final"
                    and not it.get("no_bet") and it.get("kind") == "single" and it.get("key") not in bought):
                it["status"] = "missed"
                it["decided_at"] = None
                it["reason"] = "錯過開跑前落注時間，冇落注（原本：" + str(it.get("reason") or "") + "）"
                m += 1
        if m:
            picks.recount(pk)
            picks.save_picks(conn, pre_run_id, date, venue, pk)
    return {"race_no": int(race_no), "pre_run_id": pre_run_id, "missed": True, "rows": n, "picks_items": m,
            "voided_tickets": voided}


# --------------------------------------------------------------------- post-meeting
def _hist(conn, venue, before, days=365):
    since = (datetime.strptime(before, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    return pd.read_sql_query(
        "SELECT r.date, r.venue, r.distance, r.surface, r.course, u.* FROM runners u JOIN races r USING(race_id) "
        "WHERE r.venue=? AND r.date>=? AND r.date<=? AND u.scratched=0", conn, params=(venue, since, before))


def _rates(df, keys, min_n):
    df = df.assign(win=(df.finish_pos == 1).astype(int), plc=(df.finish_pos <= 3).astype(int))
    g = df.groupby(keys).agg(出賽=("win", "size"), 頭馬=("win", "sum"), 入三甲=("plc", "sum")).reset_index()
    g = g[g["出賽"] >= min_n]
    g["勝出率"] = g["頭馬"] / g["出賽"]
    g["入三甲率"] = g["入三甲"] / g["出賽"]
    return g.sort_values(["入三甲率", "出賽"], ascending=False)


def _long_term_md(conn, venue, date) -> str:
    h = _hist(conn, venue, (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"))
    if h.empty:
        return ""
    out = [f"## {VENUE_ZH.get(venue, venue)} 近 365 日長期參考\n"]
    h = h.rename(columns={"jockey_name": "騎師", "trainer_name": "練馬師"})
    j = _rates(h, ["騎師"], 40).head(10)
    t = _rates(h, ["練馬師"], 40).head(10)
    jt = _rates(h, ["騎師", "練馬師"], 12).head(12)
    for title, d in (("騎師（≥40 騎）", j), ("練馬師（≥40 匹次）", t), ("騎練組合（≥12 次）", jt)):
        d = d.copy()
        d["勝出率"] = d["勝出率"].map(_pct); d["入三甲率"] = d["入三甲率"].map(_pct)
        out += [f"### {title}\n", _table(d)]
    h = h.assign(檔位組=pd.cut(h.draw, [0, 3, 6, 9, 14], labels=["1-3", "4-6", "7-9", "10+"]),
                 路程=h.distance.astype("Int64").astype(str) + "米 " + h.surface.map(SURFACE_ZH).fillna(""))
    d = _rates(h.dropna(subset=["檔位組"]), ["路程", "檔位組"], 20)
    p = d.pivot_table(index="路程", columns="檔位組", values="入三甲率", observed=False)
    p = p.apply(lambda c: c.map(_pct)).reset_index()
    out += ["### 檔位入三甲率（按路程/跑道）\n", _table(p)]
    return "\n".join(out)


def post_meeting(conn, run_id, date, venue, retrain=True) -> dict:
    races = pd.read_sql_query("SELECT * FROM races WHERE date=? AND venue=? ORDER BY race_no", conn,
                              params=(date, venue))
    run = pd.read_sql_query(
        "SELECT u.*, r.race_no FROM runners u JOIN races r USING(race_id) WHERE r.date=? AND r.venue=?",
        conn, params=(date, venue))
    pre = conn.execute("SELECT run_id FROM runs WHERE kind='pre_bet' AND date=? AND venue=? "
                       "ORDER BY created_at DESC LIMIT 1", (date, venue)).fetchone()
    pred = pd.DataFrame()
    if pre:
        pred = pd.read_sql_query("SELECT race_id, horse_no, p_win, p_place FROM predictions WHERE run_id=?",
                                 conn, params=(pre["run_id"],))
    md = [f"# {date} {VENUE_ZH.get(venue, venue)} 賽後分析\n", DISCLAIMER + "\n", "## 逐場賽果 vs 模型\n"]
    rows, hit_top1_place, hit_qpl, n_races = [], 0, 0, 0
    for rc in races.itertuples():
        g = run[(run.race_id == rc.race_id) & (run.scratched == 0)].sort_values("finish_pos")
        top3 = g[g.finish_pos <= 3]
        fav = g.sort_values("win_odds").head(1)
        placed = set(top3.horse_no.astype(int))
        pick, qpl3 = "-", "-"
        if not pred.empty:
            pg = pred[pred.race_id == rc.race_id].sort_values("p_place", ascending=False)
            if len(pg):
                n_races += 1
                p1 = int(pg.horse_no.iloc[0])
                ok = p1 in placed
                hit_top1_place += ok
                pick = f"{p1}{'✅' if ok else '❌'}"
                q = set(pg.horse_no.astype(int).head(3))
                nq = len(q & placed)
                hit_qpl += nq >= 2
                qpl3 = f"{sorted(q)} 中{nq}"
        rows.append({
            "場": rc.race_no, "班次/路程": f"{rc.race_class or ''} {rc.distance}米 {SURFACE_ZH.get(rc.surface, '')}",
            "頭馬": " ".join(f"{int(x.horse_no)}{x.horse_name}" for x in top3.head(1).itertuples()),
            "騎師/練馬師": " ".join(f"{x.jockey_name}/{x.trainer_name}" for x in top3.head(1).itertuples()),
            "檔位": ",".join(str(int(x)) for x in top3.draw.head(3) if pd.notna(x)),
            "頭三名": ",".join(str(int(x)) for x in top3.horse_no),
            "獨贏賠率": _num(top3.win_odds.iloc[0]) if len(top3) else "-",
            "大熱": (f"{int(fav.horse_no.iloc[0])}({_num(fav.win_odds.iloc[0])})"
                     f"{'✅' if len(fav) and int(fav.horse_no.iloc[0]) in placed else '❌'}") if len(fav) else "-",
            "模型首選入三甲": pick, "模型前三(位置Q)": qpl3,
        })
    md.append(_table(pd.DataFrame(rows)))
    if n_races:
        md.append(f"模型首選入三甲：{hit_top1_place}/{n_races}（{_pct(hit_top1_place / n_races)}）；"
                  f"前三揀中位置Q：{hit_qpl}/{n_races}\n")

    st = run[run.scratched == 0].assign(win=lambda d: (d.finish_pos == 1).astype(int),
                                         plc=lambda d: (d.finish_pos <= 3).astype(int))
    for title, keys in (("騎師", ["jockey_name"]), ("練馬師", ["trainer_name"]),
                        ("騎練組合（有入三甲）", ["jockey_name", "trainer_name"])):
        g = st.groupby(keys).agg(出賽=("win", "size"), 頭馬=("win", "sum"), 入三甲=("plc", "sum")).reset_index()
        g = g[g["入三甲"] > 0].sort_values(["頭馬", "入三甲"], ascending=False).head(12)
        g = g.rename(columns={"jockey_name": "騎師", "trainer_name": "練馬師"})
        md += [f"## 當日{title}\n", _table(g)]
    wins = st[st.win == 1].merge(races[["race_id", "distance", "surface"]], on="race_id")
    if len(wins):
        md.append("## 當日頭馬檔位 / 路程 / 跑道\n")
        md.append(_table(pd.DataFrame({
            "路程": wins.distance.astype(str) + "米", "跑道": wins.surface.map(SURFACE_ZH),
            "檔位": wins.draw.astype("Int64"), "獨贏賠率": wins.win_odds,
        }).sort_values("路程")))

    tk = conn.execute("SELECT * FROM paper_tickets WHERE date=? AND venue=? AND status!='void' ORDER BY strategy, ticket_no",
                      (date, venue)).fetchall()
    tickets = []
    for t in tk:      # pool / spec / notes (structure) live in the detail JSON
        det = json.loads(t["detail"] or "{}")
        tickets.append(dict(t, legs=json.loads(t["legs"]), pool=det.get("pool"), spec=det.get("spec"),
                            notes=det.get("notes") or {}))
    names = {(int(r.race_no), int(r.horse_no)): r.horse_name for r in run.itertuples()}
    md += ["## 模擬投注結算\n", _ticket_lines(tickets, names)]
    if tickets:
        stake = sum(t["total_stake"] for t in tickets)
        pay = sum(t["payout"] or 0 for t in tickets)
        md.append(f"當日成本 ${stake:,.0f}，派彩 ${pay:,.2f}，盈虧 {pay - stake:+,.2f}\n")
    md.append(paper.summary_md(conn))

    notes = {}
    if retrain:
        nxt = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        for kind in ("combined", "fundamental"):
            try:
                m = model.latest_model(conn, nxt, kind=kind)
                notes[f"model_{kind}"] = m.model_version
            except Exception as e:  # keep the report even if retraining fails
                log.exception("retrain %s failed", kind)
                notes[f"model_{kind}_error"] = str(e)
    return {"report_md": "\n".join(md), "notes": notes}
