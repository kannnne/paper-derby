"""System self-check (自動覆核), run after every scheduler tick in the cloud loop.

    python -m hkjc health            # report only
    python -m hkjc health --fix      # try to repair what can be repaired, then re-check
    python -m hkjc health --fix --notify   # + open / update / close a GitHub issue (cloud)

Checks (all from the DB, no HTTP):
  scores_missing   upcoming meeting within ~20 h has no score / preview run       -> cmd_scores
  scores_stale     preview older than 2 h while waiting for the betting run       -> cmd_scores
  pre_missing      past T-2h+20 min, before T-15 min, no betting run              -> cmd_pre
  pre_missed       racing started and there was no betting run (cannot be fixed)
  live_lagging     a race finished > 20 min ago but its result is not in yet     -> cmd_live
  post_missing     tracked meeting finished, results in, no post-race analysis   -> cmd_post
  unsettled        open paper tickets on a meeting that has final results       -> paper.settle
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

    # races in progress
    today = now.strftime("%Y-%m-%d")
    for m in conn.execute("SELECT m.date, m.venue, MAX(r.post_time) last_post FROM meetings m JOIN races r "
                          "ON r.date=m.date AND r.venue=m.venue WHERE m.date=? AND "
                          "(m.status IS NULL OR m.status NOT IN ('results','abandoned')) GROUP BY m.date, m.venue",
                          (today,)).fetchall():
        late = conn.execute("SELECT COUNT(*) FROM races WHERE date=? AND venue=? AND post_time<=?",
                            (m["date"], m["venue"], (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M"))).fetchone()[0]
        done = conn.execute("SELECT COUNT(DISTINCT r.race_id) FROM races r JOIN dividends d ON d.race_id=r.race_id "
                            "AND d.pool='WIN' WHERE r.date=? AND r.venue=?", (m["date"], m["venue"])).fetchone()[0]
        if late > done:
            add("live_lagging", "warn", f"{m['date']} {m['venue']}：已跑完 {late} 場但只有 {done} 場賽果",
                "live", date=m["date"], venue=m["venue"])

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
            elif f == "live":
                cli.cmd_live(conn, it["date"], it["venue"])
            elif f == "post":
                cli.cmd_post(conn, it["date"])
            elif f == "settle":
                paper.settle(conn, it["date"])
                cli._publish(conn, f"補結算 {it['date']}")
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


def notify(issues, fixed):
    """Keep one open GitHub issue in sync (GitHub emails the repo owner on new issues)."""
    r = _gh("issue", "list", "--label", ISSUE_LABEL, "--state", "open", "--json", "number", "--jq", ".[0].number")
    if r is None:
        return "no gh"
    num = (r.stdout or "").strip()
    now = cli.now_hkt().strftime("%Y-%m-%d %H:%M")
    if issues:
        body = (f"**{now} HKT** 自動覆核發現以下問題（已嘗試自動補救：{', '.join(fixed) or '冇'}）：\n\n" +
                "\n".join(f"- {'🔴' if i['level'] == 'error' else '🟡'} `{i['code']}` {i['msg']}" for i in issues) +
                "\n\n系統會每 5 分鐘再覆核；問題解決後呢個 issue 會自動關閉。")
        if num:
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
