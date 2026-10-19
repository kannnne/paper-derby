"""Scheduler (cli.cmd_tick / cmd_pre / cmd_post) edge cases with a mocked clock.

No network: scrape / odds / pipeline calls are monkeypatched.  Every DB is a fresh
temp file - the real data/hkjc.db is never opened."""
import json
import logging
from datetime import datetime

import pytest

from hkjc import cli, db, paper, pipeline, scrape

D = "2026-10-21"           # HV night meeting, race 1 18:40, 9 races
FIRST = f"{D}T18:40"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    yield c
    c.close()


@pytest.fixture
def clock(monkeypatch):
    t = {"now": datetime(2026, 10, 19, 10, 0)}

    def set_(s):
        t["now"] = datetime.strptime(s, "%Y-%m-%d %H:%M")
    monkeypatch.setattr(cli, "now_hkt", lambda: t["now"])
    monkeypatch.setattr(paper, "_now", lambda: t["now"].isoformat(timespec="seconds"))
    return set_


def _card_meeting(conn, date=D, first=FIRST, n=9, status="card", updated="2026-10-21T16:00:00"):
    conn.execute("INSERT OR REPLACE INTO meetings (date, venue, n_races, first_post_time, status, updated_at) "
                 "VALUES (?,?,?,?,?,?)", (date, "HV", n, first, status, updated))
    h0, m0 = int(first[11:13]), int(first[14:16])
    for rn in range(1, n + 1):
        h, m = divmod(h0 * 60 + m0 + 30 * (rn - 1), 60)
        conn.execute("INSERT OR REPLACE INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,?,?)",
                     (f"{date.replace('-', '')}-HV-{rn:02d}", date, "HV", rn, f"{date}T{h:02d}:{m:02d}"))
    conn.commit()


@pytest.fixture
def calls(monkeypatch):
    """Record cmd_pre / cmd_post / next_meeting calls instead of doing the work."""
    rec = {"pre": [], "post": [], "next": 0}

    def fake_pre(conn, force=False, mtg=None, place_bets=True):
        rec["pre"].append((mtg["date"] if mtg else None, place_bets))
    monkeypatch.setattr(cli, "cmd_pre", fake_pre)
    monkeypatch.setattr(cli, "cmd_post", lambda conn, date=None: rec["post"].append(date))

    def fake_next(conn=None, fetcher=None):
        rec["next"] += 1
        return {"date": D, "venue": "HV", "first_post_time": FIRST, "n_races": 9}
    monkeypatch.setattr(scrape, "next_meeting", fake_next)
    return rec


# ------------------------------------------------------------------ pre window
def test_pre_fires_once_in_window(conn, clock, calls):
    _card_meeting(conn)
    for t in ("2026-10-21 16:00", "2026-10-21 16:30"):         # before T-2h: nothing
        clock(t); cli.cmd_tick(conn)
    assert calls["pre"] == []
    clock("2026-10-21 16:40"); cli.cmd_tick(conn)
    assert calls["pre"] == [(D, True)]
    cli._record_run(conn, "pre-x", "pre_bet", D, "HV", {})
    clock("2026-10-21 16:50"); cli.cmd_tick(conn)                # done already: no second run
    clock("2026-10-21 18:30"); cli.cmd_tick(conn)
    assert calls["pre"] == [(D, True)]


def test_mac_asleep_wakes_at_T_minus_30_runs_late(conn, clock, calls, caplog):
    _card_meeting(conn)
    clock("2026-10-21 18:10")
    with caplog.at_level(logging.WARNING, logger="hkjc"):
        cli.cmd_tick(conn)
    assert calls["pre"] == [(D, True)]
    assert any("late" in r.getMessage() for r in caplog.records)


def test_wakes_after_cutoff_report_only_then_nothing_after_post_time(conn, clock, calls, caplog):
    _card_meeting(conn)
    clock("2026-10-21 18:30")                                   # T-10 min: too late for bets
    with caplog.at_level(logging.WARNING, logger="hkjc"):
        cli.cmd_tick(conn)
    assert calls["pre"] == [(D, False)]
    assert any("report only" in r.getMessage() for r in caplog.records)
    cli._record_run(conn, "prerep-x", "pre_report", D, "HV", {})
    clock("2026-10-21 18:35"); cli.cmd_tick(conn)               # report-only also runs once
    calls["pre"].clear()
    _card_meeting(conn, updated="2026-10-21T18:40:00")
    conn.execute("DELETE FROM runs"); conn.commit()
    clock("2026-10-21 18:45"); cli.cmd_tick(conn)               # race 1 started: nothing at all
    assert calls["pre"] == []


def test_next_meeting_cached_hourly(conn, clock, calls):
    _card_meeting(conn, updated="2026-10-21T09:30:00")
    clock("2026-10-21 10:00"); cli.cmd_tick(conn)               # row 30 min old -> no HTTP
    assert calls["next"] == 0
    clock("2026-10-21 10:40"); cli.cmd_tick(conn)               # 70 min old -> refresh
    assert calls["next"] == 1


def test_network_down_tick_logs_and_recovers(conn, clock, monkeypatch, caplog):
    _card_meeting(conn, updated="2026-10-18T00:00:00")          # stale -> live lookup

    def down(*a, **k):
        raise ConnectionError("network unreachable")
    monkeypatch.setattr(scrape, "next_meeting", down)
    monkeypatch.setattr(scrape, "scrape_meeting_results", down)
    _card_meeting(conn, date="2026-10-19", first="2026-10-19T12:30", n=2, status="card",
                  updated="2026-10-19T09:00:00")
    clock("2026-10-19 23:00")
    with caplog.at_level(logging.ERROR, logger="hkjc"):
        cli.cmd_tick(conn)                                      # must not raise
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "post step" in msgs and "next meeting lookup failed" in msgs
    # network back: the next tick works again
    rec = []
    monkeypatch.setattr(scrape, "next_meeting", lambda conn=None, fetcher=None: rec.append(1) or None)
    monkeypatch.setattr(cli, "cmd_post", lambda conn, date=None: rec.append(date))
    clock("2026-10-19 23:10"); cli.cmd_tick(conn)
    assert rec == ["2026-10-19", 1]


def test_pend_ignores_finished_and_old_meetings(conn, clock, calls):
    # thousands of historical meetings: status 'results', post_time NULL -> never re-scraped
    for i in range(1, 29):
        conn.execute("INSERT INTO meetings (date, venue, status) VALUES (?,?,?)", (f"2024-02-{i:02d}", "ST", "results"))
        conn.execute("INSERT INTO races (race_id, date, venue, race_no) VALUES (?,?,?,1)",
                     (f"202402{i:02d}-ST-01", f"2024-02-{i:02d}", "ST"))
    conn.execute("INSERT INTO meetings (date, venue, status) VALUES ('2023-10-08','ST','abandoned')")
    conn.execute("INSERT INTO meetings (date, venue, status) VALUES ('2026-09-01','ST','provisional')")  # > 7 days old
    conn.commit()
    clock("2026-10-19 23:00"); cli.cmd_tick(conn)
    assert calls["post"] == []


# ------------------------------------------------------------------ post
def _one_race_with_ticket(conn, date="2026-10-19"):
    _card_meeting(conn, date=date, first=f"{date}T12:30", n=1, status="card", updated=f"{date}T09:00:00")
    paper.place_tickets(conn, "pre-t", date, "HV", [{
        "strategy": "WIN_TEST", "ticket_no": 1, "pool": "WIN", "spec": None, "unit_stake": 10.0,
        "legs": [{"race_no": 1, "pool": "WIN", "selections": [3]}]}])


def _write_results(conn, date, status):
    rid = f"{date.replace('-', '')}-HV-01"
    conn.execute("INSERT OR REPLACE INTO runners (race_id, horse_no, finish_pos, scratched) VALUES (?,3,1,0)", (rid,))
    conn.execute("INSERT OR REPLACE INTO dividends (race_id, pool, combination, dividend) VALUES (?, 'WIN', '3', 45.5)", (rid,))
    conn.execute("UPDATE meetings SET status=? WHERE date=?", (status, date))
    conn.commit()
    return 1


def test_provisional_results_retried_without_settlement(conn, clock, monkeypatch):
    _one_race_with_ticket(conn)
    n_scrape = []
    monkeypatch.setattr(scrape, "scrape_meeting_results",
                        lambda c, d, v=None: n_scrape.append(d) or _write_results(c, d, "provisional"))
    monkeypatch.setattr(scrape, "next_meeting", lambda conn=None, fetcher=None: None)
    post_calls = []
    monkeypatch.setattr(pipeline, "post_meeting", lambda *a, **k: post_calls.append(a) or {"report_md": "x", "notes": {}})
    clock("2026-10-19 13:30"); cli.cmd_tick(conn)
    clock("2026-10-19 13:40"); cli.cmd_tick(conn)
    assert n_scrape == ["2026-10-19", "2026-10-19"]               # retried every tick
    assert conn.execute("SELECT status FROM paper_tickets").fetchone()[0] == "open"
    assert post_calls == []


def test_post_idempotent_ledger(conn, clock, monkeypatch, tmp_path):
    _one_race_with_ticket(conn)
    monkeypatch.setattr(cli, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(scrape, "scrape_meeting_results", lambda c, d, v=None: _write_results(c, d, "results"))
    monkeypatch.setattr(pipeline, "post_meeting", lambda *a, **k: {"report_md": "# 賽後", "notes": {}})
    clock("2026-10-19 13:30")
    p = cli.cmd_post(conn, "2026-10-19")
    assert p.exists() and p.read_text(encoding="utf-8") == "# 賽後"
    led = conn.execute("SELECT * FROM bankroll_ledger").fetchall()
    assert paper.bankroll(conn) == pytest.approx(10000 - 10 + 45.5)
    cli.cmd_post(conn, "2026-10-19")
    assert [tuple(r) for r in conn.execute("SELECT * FROM bankroll_ledger")] == [tuple(r) for r in led]
    assert conn.execute("SELECT kind FROM runs").fetchall()[0][0] == "post_analysis"


def test_abandoned_meeting_no_infinite_retries(conn, clock, monkeypatch):
    """Typhoon: the results site never shows the meeting.  Retried for 2 days, then
    marked abandoned, tickets refunded, and the scheduler stops asking."""
    _one_race_with_ticket(conn)
    tries = []
    monkeypatch.setattr(scrape, "scrape_meeting_results", lambda c, d, v=None: tries.append(d) or 0)
    monkeypatch.setattr(scrape, "next_meeting", lambda conn=None, fetcher=None: None)
    clock("2026-10-19 23:00"); cli.cmd_tick(conn)
    clock("2026-10-20 12:00"); cli.cmd_tick(conn)
    assert len(tries) == 2
    assert conn.execute("SELECT status FROM meetings WHERE date='2026-10-19'").fetchone()[0] == "card"
    clock("2026-10-21 00:10"); cli.cmd_tick(conn)
    assert conn.execute("SELECT status FROM meetings WHERE date='2026-10-19'").fetchone()[0] == "abandoned"
    t = conn.execute("SELECT status, payout FROM paper_tickets").fetchone()
    assert tuple(t) == ("settled", 10.0)                          # refunded
    assert paper.bankroll(conn) == pytest.approx(10000.0)
    clock("2026-10-21 00:20"); cli.cmd_tick(conn)
    assert len(tries) == 3                                       # no more retries


# ------------------------------------------------------------------ pre robustness
def test_pre_failure_after_tickets_then_rerun_no_duplicates(conn, clock, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(scrape, "scrape_racecard", lambda *a, **k: 9)
    from hkjc import odds
    monkeypatch.setattr(odds, "snapshot_meeting", lambda *a, **k: 0)
    state = {"fail": True, "n": 0}

    def fake_pre_meeting(conn, run_id, date, venue, config=None, place_bets=True):
        state["n"] += 1
        tk = [{"strategy": "PLA_TEST", "ticket_no": 1, "pool": "PLA", "spec": None, "unit_stake": 10.0,
               "legs": [{"race_no": 1, "pool": "PLA", "selections": [1]}]}]
        if state["n"] == 1:   # first attempt also had an ADAPTIVE ticket that the re-run drops
            tk.append({"strategy": "ADAPTIVE", "ticket_no": 2, "pool": "WIN", "spec": None, "unit_stake": 10.0,
                       "legs": [{"race_no": 2, "pool": "WIN", "selections": [4]}]})
        paper.place_tickets(conn, run_id, date, venue, tk)
        if state["fail"]:
            raise RuntimeError("boom after tickets were placed")
        return {"report_md": "# 賽前", "notes": {}}
    monkeypatch.setattr(pipeline, "pre_meeting", fake_pre_meeting)
    mtg = {"date": D, "venue": "HV", "first_post_time": FIRST, "n_races": 9}
    clock("2026-10-21 16:40")
    with pytest.raises(RuntimeError):
        cli.cmd_pre(conn, mtg=mtg)
    assert not cli._run_exists(conn, "pre_bet", D, "HV")           # failed run not recorded
    state["fail"] = False
    clock("2026-10-21 16:50")
    assert cli.cmd_pre(conn, mtg=mtg).exists()
    rows = conn.execute("SELECT strategy, status FROM paper_tickets ORDER BY ticket_id").fetchall()
    assert [tuple(r) for r in rows] == [("PLA_TEST", "void"), ("ADAPTIVE", "void"), ("PLA_TEST", "open")]
    assert paper.bankroll(conn) == pytest.approx(10000 - 10)
    assert cli.cmd_pre(conn, mtg=mtg) is None                       # recorded: no third run


def test_lock_prevents_overlapping_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LOG_DIR", tmp_path)
    a = cli._acquire_lock()
    assert a is not None
    assert cli._acquire_lock() is None
    a.close()
    b = cli._acquire_lock()
    assert b is not None
    b.close()


def test_env_overrides_db_and_model_dir(tmp_path, monkeypatch):
    from hkjc import model
    monkeypatch.setenv("HKJC_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("HKJC_MODEL_DIR", str(tmp_path / "m"))
    c = db.connect()
    assert db.db_path() == tmp_path / "x.db"
    assert c.execute("PRAGMA database_list").fetchone()[2] == str(tmp_path / "x.db")
    c.close()
    assert model.default_model_dir() == tmp_path / "m"
    monkeypatch.delenv("HKJC_DB"); monkeypatch.delenv("HKJC_MODEL_DIR")
    assert db.db_path() == db.DEFAULT_DB and model.default_model_dir() == model.MODEL_DIR


def test_ticket_lines_labels_from_db_rows():
    tk = [{"strategy": "ADAPTIVE", "ticket_no": 8, "pool": "QIN", "spec": None,
           "notes": {"structure": "QIN 膽1x腳4"}, "n_combos": 4, "unit_stake": 10.0, "total_stake": 40.0,
           "payout": 0.0, "profit": -40.0,
           "legs": [{"race_no": 8, "pool": "QIN", "bankers": [10], "selections": [2, 7, 8, 3]}]}]
    s = pipeline._ticket_lines(tk, {(8, 10): "首飾天空"})
    assert "ADAPTIVE（第8場）" in s and "QIN 膽1x腳4" in s and "（，" not in s
    assert "膽 10 首飾天空" in s and "-40.00" in s
    assert paper._money(-704.08) == "-$704.08" and paper._money(1234.5) == "$1,234.50"


# ------------------------------------------------------------------ review fixes
def test_pre_refused_once_meeting_has_results(conn, clock, monkeypatch):
    """`pre --force` after the results must not re-decide (it would mix new decisions with
    the settled ones and the dashboard would lose the other robots' decisions)."""
    _card_meeting(conn, date="2026-10-19", first="2026-10-19T12:30", n=2, status="results")
    called = []
    monkeypatch.setattr(scrape, "scrape_racecard", lambda *a, **k: called.append("card") or 2)
    monkeypatch.setattr(pipeline, "pre_meeting", lambda *a, **k: called.append("pre"))
    mtg = {"date": "2026-10-19", "venue": "HV", "first_post_time": "2026-10-19T12:30", "n_races": 2}
    clock("2026-10-19 19:40")
    assert cli.cmd_pre(conn, force=True, mtg=mtg) is None
    assert called == [] and conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_pre_and_post_publish_exactly_once_into_isolated_site(conn, clock, monkeypatch):
    from hkjc import db as dbmod, odds, publish
    # conftest: tests never export into the real site/ or read data/publish.json
    assert publish.SITE_DIR != dbmod.ROOT / "site" and publish.PUBLISH_CONFIG != dbmod.ROOT / "data" / "publish.json"
    pubs = []
    monkeypatch.setattr(publish, "publish", lambda c, message=None, **k: pubs.append(message) or {"status": "exported"})
    monkeypatch.setattr(scrape, "scrape_racecard", lambda *a, **k: 1)
    monkeypatch.setattr(odds, "snapshot_meeting", lambda *a, **k: 0)
    monkeypatch.setattr(pipeline, "pre_meeting", lambda *a, **k: {"report_md": "# 賽前", "notes": {}})
    _one_race_with_ticket(conn)
    mtg = {"date": "2026-10-19", "venue": "HV", "first_post_time": "2026-10-19T12:30", "n_races": 1}
    clock("2026-10-19 10:30")
    assert cli.cmd_pre(conn, mtg=mtg) is not None
    assert pubs == ["賽前 2026-10-19 HV"]
    monkeypatch.setattr(scrape, "scrape_meeting_results", lambda c, d, v=None: _write_results(c, d, "results"))
    monkeypatch.setattr(pipeline, "post_meeting", lambda *a, **k: {"report_md": "# 賽後", "notes": {}})
    clock("2026-10-19 13:30")
    assert cli.cmd_post(conn, "2026-10-19") is not None
    assert pubs == ["賽前 2026-10-19 HV", "賽後 2026-10-19 HV"]


def test_publish_failure_never_breaks_post(conn, clock, monkeypatch):
    from hkjc import publish

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(publish, "publish", boom)
    _one_race_with_ticket(conn)
    monkeypatch.setattr(scrape, "scrape_meeting_results", lambda c, d, v=None: _write_results(c, d, "results"))
    monkeypatch.setattr(pipeline, "post_meeting", lambda *a, **k: {"report_md": "# 賽後", "notes": {}})
    clock("2026-10-19 13:30")
    assert cli.cmd_post(conn, "2026-10-19").exists()
    assert cli._run_exists(conn, "post_analysis", "2026-10-19", "HV")
