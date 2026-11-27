"""Self-check (hkjc.health): each problem is detected, and fixes are dispatched."""
from datetime import datetime

import pytest

from hkjc import cli, health
from hkjc.db import connect

D, V = "2026-10-21", "HV"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("INSERT INTO meetings (date, venue, n_races, first_post_time, status) VALUES (?,?,2,?, 'card')",
              (D, V, f"{D}T18:40"))
    for i, t in enumerate(("18:40", "19:10"), 1):
        c.execute("INSERT INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,?,?)",
                  (f"20261021-HV-0{i}", D, V, i, f"{D}T{t}"))
    c.commit()
    return c


def _at(monkeypatch, s):
    monkeypatch.setattr(cli, "now_hkt", lambda: datetime.strptime(s, "%Y-%m-%d %H:%M"))


def _codes(conn):
    return sorted(i["code"] for i in health.check(conn))


def _run(conn, kind, ts):
    conn.execute("INSERT INTO runs (run_id, kind, date, venue, created_at) VALUES (?,?,?,?,?)",
                 (f"{kind}-{ts}", kind, D, V, ts))
    conn.commit()


def test_pre_race_checks(conn, monkeypatch):
    _at(monkeypatch, "2026-10-20 15:00")                     # 27 h before: nothing expected yet
    assert _codes(conn) == []
    _at(monkeypatch, "2026-10-21 08:00")                     # within 20 h, no preview
    assert _codes(conn) == ["scores_missing"]
    _run(conn, "pre_scores", "2026-10-21T08:00:00")
    _at(monkeypatch, "2026-10-21 10:30")                     # preview 2.5 h old
    assert _codes(conn) == ["scores_stale"]
    _at(monkeypatch, "2026-10-21 17:05")                     # T-2h+25min, no betting run
    assert _codes(conn) == ["pre_missing"]
    _at(monkeypatch, "2026-10-21 18:45")                     # racing started, never bet
    assert "pre_missed" in _codes(conn)
    _run(conn, "pre_bet", "2026-10-21T16:41:00")
    assert "pre_missed" not in _codes(conn)


def test_live_and_post_checks(conn, monkeypatch):
    _run(conn, "pre_bet", "2026-10-21T16:41:00")
    _at(monkeypatch, "2026-10-21 19:06")                     # race 1 started 26 min ago, no result
    assert _codes(conn) == ["live_lagging"]
    conn.execute("INSERT INTO dividends (race_id, pool, combination, dividend) VALUES ('20261021-HV-01','WIN','1',50)")
    conn.commit()
    assert _codes(conn) == ["live_lagging"]                  # dividends only (preliminary) is still lagging
    conn.execute("INSERT INTO runners (race_id, horse_no, finish_pos) VALUES ('20261021-HV-01',1,1)")
    conn.commit()
    assert _codes(conn) == []                                # full table in
    conn.execute("INSERT OR REPLACE INTO dividends (race_id, pool, combination, dividend) VALUES ('20261021-HV-01','WIN','1',50)")
    conn.execute("INSERT INTO dividends (race_id, pool, combination, dividend) VALUES ('20261021-HV-02','WIN','2',50)")
    conn.execute("UPDATE meetings SET status='results'")
    conn.execute("INSERT INTO paper_tickets (date, venue, strategy, legs, total_stake, status) VALUES (?,?,?,?,?, 'open')",
                 (D, V, "x", "[]", 10))
    conn.commit()
    _at(monkeypatch, "2026-10-21 20:40")                     # last race + 90 min: no post, open ticket
    assert _codes(conn) == ["post_missing", "unsettled"]


def test_fix_dispatches_and_reports(conn, monkeypatch):
    _run(conn, "pre_bet", "2026-10-21T16:41:00")
    conn.execute("UPDATE meetings SET status='results'")
    conn.commit()
    _at(monkeypatch, "2026-10-21 21:00")
    called = []
    monkeypatch.setattr(cli, "cmd_post", lambda c, date=None: (called.append(date),
                        cli._record_run(c, "post-x", "post_analysis", D, V, {})))
    issues = health.run(conn, do_fix=True)
    assert called == [D] and issues == []


def test_notify_only_errors_and_only_on_change(monkeypatch):
    """No email storm: warnings never page; an open issue gets a comment only when problems change."""
    calls = []
    state = {"last": ""}

    class R:
        def __init__(self, out=""):
            self.stdout = out

    def fake_gh(*args):
        calls.append(args[:2])
        if args[:2] == ("issue", "list"):
            return R("7")
        if args[:2] == ("issue", "view"):
            return R(state["last"])
        if args[:2] == ("issue", "comment"):
            state["last"] = args[args.index("--body") + 1]
        return R()

    monkeypatch.setattr(health, "_gh", fake_gh)
    _at(monkeypatch, "2026-10-21 20:00")
    warn = [{"code": "live_lagging", "level": "warn", "msg": "x", "date": D, "venue": V}]
    err = [{"code": "post_missing", "level": "error", "msg": "y", "date": D, "venue": V}]
    assert health.notify(warn, []).startswith("closed")           # warnings alone: nothing to page
    assert health.notify(err, []) == "commented #7"
    assert health.notify(err, []) == "unchanged #7"               # same problem next tick: no new email
    assert health.notify(err + [{**err[0], "code": "unsettled"}], []) == "commented #7"


def test_wrong_refunds_are_resettled_by_health(tmp_path, monkeypatch):
    """A ticket refunded as 'pool not offered' although the pool's dividends are in now is
    detected (warn, no paging) and re-settled by health --fix (so the cloud DB is corrected)."""
    from hkjc import paper
    c = connect(tmp_path / "w.db")
    c.execute("INSERT INTO meetings (date, venue, n_races, status) VALUES (?,?,1,'results')", (D, V))
    rid = "20261021-HV-01"
    c.execute("INSERT INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,1,?)", (rid, D, V, f"{D}T18:40"))
    for h, pos in ((3, 1), (5, 2), (7, 3), (9, 4)):
        c.execute("INSERT INTO runners (race_id, horse_no, finish_pos) VALUES (?,?,?)", (rid, h, pos))
    c.execute("INSERT INTO dividends VALUES (?,?,?,?)", (rid, "WIN", "3", 42.0))
    c.commit()
    t = {"strategy": "FF_box", "ticket_no": 1, "pool": "FF", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "FF", "selections": [3, 5, 7, 9]}]}
    paper.place_tickets(c, "r", D, V, [t])
    paper.settle(c, D)                                    # meeting final, FF missing -> refund 10
    c.execute("INSERT INTO dividends VALUES (?,?,?,?)", (rid, "FF", "3,5,7,9", 230.0))
    c.commit()
    _at(monkeypatch, "2026-10-22 10:00")
    assert "wrong_refunds" in _codes(c)
    monkeypatch.setattr(cli, "_publish", lambda *a, **k: None)
    health.run(c, do_fix=True)
    assert "wrong_refunds" not in _codes(c)
    assert c.execute("SELECT payout FROM paper_tickets").fetchone()[0] == pytest.approx(230.0)


def _refund_world(tmp_path, pool_rows):
    from hkjc import paper
    c = connect(tmp_path / "g.db")
    c.execute("INSERT INTO meetings (date, venue, n_races, status) VALUES (?,?,2,'results')", (D, V))
    for rn in (1, 2):
        rid = f"20261021-HV-0{rn}"
        c.execute("INSERT INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,?,?)",
                  (rid, D, V, rn, f"{D}T1{rn}:40"))
        c.execute("INSERT INTO runners (race_id, horse_no, finish_pos) VALUES (?,?,1)", (rid, rn))
        c.execute("INSERT INTO dividends VALUES (?,?,?,?)", (rid, "WIN", str(rn), 30.0))
    for rid, pool, comb, dv in pool_rows:
        c.execute("INSERT INTO dividends VALUES (?,?,?,?)", (rid, pool, comb, dv))
    c.commit()
    return c, paper


@pytest.mark.parametrize("pool,legs,rows", [
    ("FF", [{"race_no": 1, "pool": "FF", "selections": [1, 3, 5, 7]}],
     [("20261021-HV-01", "FF", "REFUND", 10.0)]),
    ("DBL", [{"race_no": 1, "pool": "DBL", "selections": [1]}, {"race_no": 2, "pool": "DBL", "selections": [2]}],
     [("20261021-HV-02", "DBL", "REFUND", 10.0)]),
])
def test_genuine_pool_refund_is_not_a_wrong_refund_and_never_loops(tmp_path, monkeypatch, pool, legs, rows):
    """Review finding: an explicit refund row ('REFUND') is not a payout -> a genuinely refunded
    ticket must not be flagged / re-settled again and again (ledger churn every tick)."""
    c, paper = _refund_world(tmp_path, rows)
    t = {"strategy": f"{pool}_x", "ticket_no": 1, "pool": pool, "spec": None, "unit_stake": 10.0, "legs": legs}
    paper.place_tickets(c, "r", D, V, [t])
    assert paper.settle(c, D) == 1
    assert paper.wrong_pool_refunds(c) == []
    _at(monkeypatch, "2026-10-22 10:00")
    monkeypatch.setattr(cli, "_publish", lambda *a, **k: None)
    n0 = c.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0]
    for _ in range(3):
        health.run(c, do_fix=True)
    assert c.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0] == n0
    assert not {"wrong_refunds", "wrong_refunds_stuck"} & set(_codes(c))


def test_resettle_happens_once_then_escalates_instead_of_looping(tmp_path, monkeypatch):
    """If a ticket still looks wrong after one automatic re-settle, health stops re-settling it and
    raises one 'wrong_refunds_stuck' error instead."""
    c, paper = _refund_world(tmp_path, [])
    t = {"strategy": "FF_x", "ticket_no": 1, "pool": "FF", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "FF", "selections": [1, 3, 5, 7]}]}
    paper.place_tickets(c, "r", D, V, [t])
    paper.settle(c, D)                                            # final, no FF -> refund
    c.execute("INSERT INTO dividends VALUES ('20261021-HV-01','FF','2,4,6,8',230.0)")   # FF paid other horses
    c.commit()
    _at(monkeypatch, "2026-10-22 10:00")
    monkeypatch.setattr(cli, "_publish", lambda *a, **k: None)
    health.run(c, do_fix=True)                                    # re-settled once: now a loss
    assert c.execute("SELECT payout FROM paper_tickets").fetchone()[0] == pytest.approx(0.0)
    n1 = c.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0]
    health.run(c, do_fix=True)
    assert c.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0] == n1
    assert "wrong_refunds" not in _codes(c)
