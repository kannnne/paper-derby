"""System self-check (自動覆核), run after every scheduler tick in the cloud loop.

    python -m hkjc health            # report only
    python -m hkjc health --fix      # try to repair what can be repaired, then re-check
    python -m hkjc health --fix --notify   # + open / update / close a GitHub issue (cloud)

Checks (all from the DB, no HTTP):
  scores_missing   upcoming meeting within ~20 h has no score / preview run       -> cmd_scores
  scores_stale     preview older than 2 h while waiting for the betting run       -> cmd_scores
  pre_missing      past T-2h+20 min, before T-15 min, no betting run              -> cmd_pre
  pre_missed       racing started and there was no betting run (cannot be fixed)
  race_bet_missed  per-race late betting: a race passed post_time - 3 min with 'pending' rows and no
                   race_bet run (-> cli.race_bet_missed: no bet, rows -> skip), or its race_bet run
                   recorded {"missed": true} (cannot be fixed; reported until 3 h after the race)
  live_lagging     a race started > 25 min ago but its full results table is not in (report;
                   tick keeps re-fetching; alerts only after 75 min)
  post_missing     tracked meeting finished, results in, no post-race analysis   -> cmd_post
  unsettled        open paper tickets on a meeting that has final results       -> paper.settle
  wrong_refunds    tickets refunded as "pool not offered" though the pool paid  -> paper.resettle_wrong_refunds
  site_stale       site/data/index.json older than the latest run               -> publish
"""
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

from . import cli
from .db import ROOT, connect

log = logging.getLogger("hkjc.health")
ISSUE_TITLE = "⚠️ 系統自動覆核：發現問題"
ISSUE_LABEL = "health"


def _ts(s):
    return datetime.strptime(str(s)[:16], "%Y-%m-%dT%H:%M")


def _last_run(conn, date, venue, kinds):
    q = ",".join("?" * len(kinds))
    r = conn.execute(f"SELECT MAX(created_at) FROM runs WHERE date=? AND venue=? AND kind IN ({q})",
                     (date, venue, *kinds)).fetchone()[0]
    return _ts(r) if r else None


def check(conn, now=None) -> list[dict]:
    now = now or cli.now_hkt()
    out = []

    def add(code, level, msg, fix=None, **kw):
        out.append({"code": code, "level": level, "msg": msg, "fix": fix, **kw})

    # upcoming / today's meeting (pre-race side)
    for m in conn.execute("SELECT date, venue, first_post_time FROM meetings WHERE first_post_time IS NOT NULL "
                          "AND (status IS NULL OR status IN ('scheduled','card','provisional')) AND date>=?",
                          ((now - timedelta(days=1)).strftime("%Y-%m-%d"),)).fetchall():
        d, v, first = m["date"], m["venue"], _ts(m["first_post_time"])
        bet = _last_run(conn, d, v, ("pre_bet",))
        rep = _last_run(conn, d, v, ("pre_report",))
        prev = _last_run(conn, d, v, ("pre_scores",))
        if bet is None and first - timedelta(hours=20) <= now < first - cli.PRE_LEAD:
            if prev is None:
                add("scores_missing", "warn", f"{d} {v}：開跑前 20 小時內仍未有評分／預覽（可能賠率未開售）",
                    "scores", date=d, venue=v)
            elif now - prev > timedelta(hours=2):
                add("scores_stale", "warn", f"{d} {v}：預覽已經 {int((now - prev).total_seconds() // 60)} 分鐘冇更新",
                    "scores", date=d, venue=v)
        if bet is None and first - cli.PRE_LEAD + timedelta(minutes=20) <= now <= first - cli.PRE_LATEST:
            add("pre_missing", "error", f"{d} {v}：已過開跑前 2 小時，仲未正式落注", "pre", date=d, venue=v)
        if bet is None and rep is None and first <= now <= first + timedelta(hours=6):
            add("pre_missed", "error", f"{d} {v}：第一場已開跑，但冇做到正式落注（機械人今場冇正式注項）",
                None, date=d, venue=v)

    # per-race late betting (single-race bets ~8 min before each race)
    today = now.strftime("%Y-%m-%d")
    since_rb = (now - timedelta(days=cli.POST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for m in conn.execute("SELECT DISTINCT date, venue FROM runs WHERE kind='pre_bet' AND date>=? AND date<=?",
                          (since_rb, today)).fetchall():
        d, v = m["date"], m["venue"]
        pre_id, notes = cli._pre_bet_run(conn, d, v)
        if pre_id is None or not cli._per_race_meeting(conn, pre_id, notes):
            continue
        runs = {}
        for r in conn.execute("SELECT run_id, notes FROM runs WHERE kind='race_bet' AND date=? AND venue=?", (d, v)):
            try:
                runs[r["run_id"]] = json.loads(r["notes"] or "{}") or {}
            except ValueError:
                runs[r["run_id"]] = {}
        for r in conn.execute("SELECT race_no, post_time FROM races WHERE date=? AND venue=? AND post_time IS NOT NULL",
                              (d, v)).fetchall():
            rn, post = int(r["race_no"]), _ts(r["post_time"])
            rid = cli.race_bet_run_id(d, v, rn)
            if rid not in runs and now > post - cli.RACE_BET_LATEST and cli._has_pending(conn, pre_id, rn):
                add("race_bet_missed", "error", f"{d} {v} 第{rn}場：已過開跑前 3 分鐘，單場注仲係「待定」冇決定",
                    "race_bet_missed", date=d, venue=v, race_no=rn)
            elif runs.get(rid, {}).get("missed") and now <= post + timedelta(hours=3):
                add("race_bet_missed", "error", f"{d} {v} 第{rn}場：錯過開跑前落注時間，呢場單場注冇落",
                    None, date=d, venue=v, race_no=rn)

    # races in progress
    for m in conn.execute("SELECT m.date, m.venue, MAX(r.post_time) last_post FROM meetings m JOIN races r "
                          "ON r.date=m.date AND r.venue=m.venue WHERE m.date=? AND "
                          "(m.status IS NULL OR m.status NOT IN ('results','abandoned')) GROUP BY m.date, m.venue",
                          (today,)).fetchall():
        # report only: tick's live step is already re-fetching every tick inside its window
        late, done = cli._races_due_vs_done(conn, m["date"], m["venue"], now, after=timedelta(minutes=25))
        very_late, _ = cli._races_due_vs_done(conn, m["date"], m["venue"], now, after=timedelta(minutes=75))
        if late > done:
            add("live_lagging", "error" if very_late > done else "warn",
                f"{m['date']} {m['venue']}：開跑超過 25 分鐘嘅有 {late} 場，但只有 {done} 場有完整名次"
                + ("（有場次超過 75 分鐘）" if very_late > done else ""), None, date=m["date"], venue=m["venue"])

    # after racing
    since = (now - timedelta(days=cli.POST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for m in conn.execute(
            "SELECT m.date, m.venue, m.status, MAX(r.post_time) last_post FROM meetings m JOIN races r "
            "ON r.date=m.date AND r.venue=m.venue WHERE m.date>=? AND m.date<=? "
            "AND EXISTS (SELECT 1 FROM runs u WHERE u.date=m.date AND u.venue=m.venue AND u.kind LIKE 'pre%') "
            "GROUP BY m.date, m.venue", (since, today)).fetchall():
        if not m["last_post"] or now < _ts(m["last_post"]) + cli.POST_DELAY + timedelta(minutes=30):
            continue
        if not _last_run(conn, m["date"], m["venue"], ("post_analysis",)):
            add("post_missing", "error", f"{m['date']} {m['venue']}：已經跑完但未做賽後分析／結算",
                "post", date=m["date"], venue=m["venue"])
        if m["status"] in ("results", "abandoned"):
            n = conn.execute("SELECT COUNT(*) FROM paper_tickets WHERE date=? AND venue=? AND status='open'",
                             (m["date"], m["venue"])).fetchone()[0]
            if n:
                add("unsettled", "error", f"{m['date']} {m['venue']}：有 {n} 張模擬注未結算",
                    "settle", date=m["date"], venue=m["venue"])

    # tickets refunded as "pool not offered" although the pool's dividends are in (settled too
    # early while racing) -> re-settle automatically (runs inside the cloud loop, so the cloud DB
    # itself is corrected)
    from . import paper as _paper
    since60 = (now - timedelta(days=60)).strftime("%Y-%m-%d")
    wrong = _paper.wrong_pool_refunds(conn, since=since60)
    if wrong:
        add("wrong_refunds", "warn", f"有 {len(wrong)} 張模擬注之前被錯當退款，要重新結算", "resettle",
            since=since60)
    # already re-settled once and STILL looks wrong: don't loop, tell Kane (error pages once)
    stuck = sorted(set(_paper.wrong_pool_refunds(conn, since=since60, include_resettled=True)) - set(wrong))
    if stuck:
        add("wrong_refunds_stuck", "error",
            f"有 {len(stuck)} 張模擬注重新結算後仍然似退款錯誤（ticket {', '.join(map(str, stuck[:10]))}），要人手查",
            None)

    # website freshness
    idx = ROOT / "site" / "data" / "index.json"
    last = conn.execute("SELECT MAX(created_at) FROM runs WHERE kind!='walkforward'").fetchone()[0]
    if last and idx.exists():
        try:
            upd = json.loads(idx.read_text(encoding="utf-8")).get("updated_at")
        except ValueError:
            upd = None
        if not upd or _ts(upd) < _ts(last) - timedelta(minutes=5):
            add("site_stale", "warn", f"網站數據（{upd}）比最新運行（{last}）舊", "publish")
    return out


def fix(conn, issues) -> list[str]:
    from . import paper
    done = []
    for it in issues:
        f = it.get("fix")
        try:
            if f == "scores":
                cli.cmd_scores(conn, force=True)
            elif f == "pre":
                cli.cmd_pre(conn)
            elif f == "race_bet_missed":
                cli.race_bet_missed(conn, it["date"], it["venue"], it["race_no"])
            elif f == "live":
                cli.cmd_live(conn, it["date"], it["venue"])
            elif f == "post":
                cli.cmd_post(conn, it["date"])
            elif f == "settle":
                paper.settle(conn, it["date"])
                cli._publish(conn, f"補結算 {it['date']}")
            elif f == "resettle":
                res = paper.resettle_wrong_refunds(conn, since=it.get("since"))
                log.warning("health: re-settled %s wrongly refunded tickets (%s)", res["reopened"], res["dates"])
                cli._publish(conn, f"重新結算 {res['reopened']} 張注")
            elif f == "publish":
                cli._publish(conn, "網站補更新")
            else:
                continue
            done.append(it["code"])
        except Exception:
            log.exception("health: fix %s failed", it["code"])
    return done


def _gh(*args):
    if not shutil.which("gh") or not os.environ.get("GH_TOKEN"):
        return None
    return subprocess.run(["gh", *args], text=True, capture_output=True)


def _signature(issues) -> str:
    return "<!-- health-sig: " + ";".join(sorted(f"{i['code']}@{i.get('date', '')}{i.get('venue', '')}"
                                                   for i in issues)) + " -->"


def notify(issues, fixed):
    """Keep one open GitHub issue in sync (GitHub emails the repo owner on new issues).
    Only 'error' issues page; a comment is added only when the set of problems changes,
    so a problem that lasts hours doesn't send an email every tick."""
    issues = [i for i in issues if i["level"] == "error"]
    r = _gh("issue", "list", "--label", ISSUE_LABEL, "--state", "open", "--json", "number", "--jq", ".[0].number")
    if r is None:
        return "no gh"
    num = (r.stdout or "").strip()
    now = cli.now_hkt().strftime("%Y-%m-%d %H:%M")
    if issues:
        sig = _signature(issues)
        body = (f"**{now} HKT** 自動覆核發現以下問題（已嘗試自動補救：{', '.join(fixed) or '冇'}）：\n\n" +
                "\n".join(f"- 🔴 `{i['code']}` {i['msg']}" for i in issues) +
                "\n\n系統會繼續覆核；問題有變先會再留言，解決後呢個 issue 會自動關閉。\n" + sig)
        if num:
            last = _gh("issue", "view", num, "--json", "body,comments", "--jq",
                       "(.comments | last | .body) // .body")
            if last is not None and sig in (last.stdout or ""):
                return f"unchanged #{num}"
            _gh("issue", "comment", num, "--body", body)
            return f"commented #{num}"
        _gh("label", "create", ISSUE_LABEL, "--color", "d73a4a", "--force")
        _gh("issue", "create", "--title", ISSUE_TITLE, "--label", ISSUE_LABEL, "--body", body)
        return "issue created"
    if num:
        _gh("issue", "close", num, "--comment", f"{now} HKT：再覆核已經冇問題，自動關閉。")
        return f"closed #{num}"
    return "healthy"


def run(conn, do_fix=False, do_notify=False) -> list[dict]:
    issues = check(conn)
    fixed = []
    if issues and do_fix:
        fixed = fix(conn, issues)
        issues = check(conn)
    for i in issues:
        log.warning("health %s: %s", i["code"], i["msg"])
    if not issues:
        log.info("health: OK" + (f" (fixed {', '.join(fixed)})" if fixed else ""))
    if do_notify:
        log.info("health notify: %s", notify(issues, fixed))
    return issues


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn = connect()
    try:
        issues = run(conn, "--fix" in argv, "--notify" in argv)
    finally:
        conn.close()
    for i in issues:
        print(f"[{i['level']}] {i['code']}: {i['msg']}")
    print("OK" if not issues else f"{len(issues)} issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
