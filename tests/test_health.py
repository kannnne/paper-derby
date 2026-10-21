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
    _at(monkeypatch, "2026-10-21 19:05")                     # race 1 started 25 min ago, no result
    assert _codes(conn) == ["live_lagging"]
    conn.execute("INSERT INTO dividends (race_id, pool, combination, dividend) VALUES ('20261021-HV-01','WIN','1',50)")
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
