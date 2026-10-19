"""Tests for hkjc.paper (paper-trading robot).  Every expected number is derived by hand.

The core fixture is a REAL meeting copied from data/hkjc.db: 2024-05-22 Happy
Valley (9 races, 12 runners each, no scratchings).  Dividends are HK$ per $10.
"""
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from hkjc import db, paper

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "hkjc.db"
DATE, VENUE = "2024-05-22", "HV"

# race_no -> {pool: {combination: dividend per $10}}   (copied from the dividends table)
DIVS = {
    1: {"WIN": {"10": 63.0}, "PLA": {"10": 23.0, "3": 51.0, "4": 16.5},
        "QPL": {"3,10": 180.5, "3,4": 151.5, "4,10": 58.0}},
    2: {"WIN": {"10": 27.5}, "PLA": {"10": 11.5, "12": 28.0, "5": 58.0},
        "QPL": {"10,12": 56.0, "5,10": 117.0, "5,12": 288.5}},
    3: {"WIN": {"1": 275.5}, "PLA": {"1": 100.0, "3": 19.0, "5": 12.0},
        "QPL": {"1,3": 322.0, "1,5": 224.0, "3,5": 36.0}},
    4: {"WIN": {"10": 19.5}, "PLA": {"10": 10.1, "3": 60.0, "5": 26.5},
        "QPL": {"3,10": 93.5, "3,5": 289.5, "5,10": 48.5}},
    5: {"WIN": {"6": 60.0}, "PLA": {"12": 44.5, "3": 28.0, "6": 20.5},
        "QPL": {"3,12": 163.0, "3,6": 99.0, "6,12": 166.0}},
    6: {"WIN": {"8": 64.5}, "PLA": {"1": 13.0, "12": 49.5, "8": 20.5},
        "QPL": {"1,12": 108.5, "1,8": 41.0, "8,12": 141.5}},
    7: {"WIN": {"11": 113.5}, "PLA": {"1": 11.5, "11": 38.5, "8": 23.0},
        "QPL": {"1,11": 60.5, "1,8": 33.5, "8,11": 124.5}},
    8: {"WIN": {"6": 60.5}, "PLA": {"1": 17.0, "3": 45.0, "6": 22.5},
        "QPL": {"1,3": 126.5, "1,6": 67.5, "3,6": 167.5}},
    9: {"WIN": {"8": 54.0}, "PLA": {"7": 20.5, "8": 21.5, "9": 23.0},
        "QPL": {"7,8": 58.0, "7,9": 66.0, "8,9": 66.0}},
}

# The tickets the robot builds for this meeting with the market baseline (final
# odds): see tests/test_backtest.py::test_real_meeting_robot_tickets.
PLA_TICKET = {"strategy": "PLA_6x63", "ticket_no": 1, "pool": "PLA", "spec": "6x63",
              "unit_stake": 10.0, "n_combos": 63, "total_stake": 630.0,
              "legs": [{"race_no": 2, "selections": [10]}, {"race_no": 3, "selections": [5]},
                       {"race_no": 4, "selections": [10]}, {"race_no": 6, "selections": [1]},
                       {"race_no": 7, "selections": [1]}, {"race_no": 9, "selections": [7]}]}
QPL_TICKETS = [
    {"strategy": "QPL_2x3", "ticket_no": 1, "pool": "QPL", "spec": "2x3", "unit_stake": 10.0,
     "n_combos": 15, "total_stake": 150.0,
     "legs": [{"race_no": 1, "selections": [4, 5, 10]}, {"race_no": 2, "selections": [10, 6, 9]}]},
    {"strategy": "QPL_2x3", "ticket_no": 2, "pool": "QPL", "spec": "2x3", "unit_stake": 10.0,
     "n_combos": 15, "total_stake": 150.0,
     "legs": [{"race_no": 4, "selections": [10, 4, 7]}, {"race_no": 6, "selections": [1, 2, 8]}]},
    {"strategy": "QPL_2x3", "ticket_no": 3, "pool": "QPL", "spec": "2x3", "unit_stake": 10.0,
     "n_combos": 15, "total_stake": 150.0,
     "legs": [{"race_no": 7, "selections": [1, 8, 7]}, {"race_no": 9, "selections": [7, 4, 8]}]},
]

# ---- hand settlement -------------------------------------------------------------
# PLA 6x63 = every non-empty subset of the 6 legs, so with leg factors f_i = div/10
# (all six bankers placed!):  payout = 10 * (prod(1 + f_i) - 1)
#   f = 11.5/10, 12.0/10, 10.1/10, 13.0/10, 11.5/10, 20.5/10 = 1.15 1.20 1.01 1.30 1.15 2.05
#   prod(1+f) = 2.15 * 2.20 * 2.01 * 2.30 * 2.15 * 3.05
#             = 4.73 -> 9.5073 -> 21.86679 -> 47.0135985 -> 143.391475425
#   payout = 10 * 142.391475425 = 1423.91475425
PLA_PAYOUT = 1423.91475425
# QPL 2x3 = leg A singles (3 pairs) + leg B singles (3 pairs) + A x B doubles (9 units)
#  T1: R1 {4,5,10}: pairs 4-5, 4-10, 5-10 -> 4,10 pays 58.0;  R2 {6,9,10}: pairs
#      6-9, 6-10, 9-10 -> none (winning pairs 10-12, 5-10, 5-12).  doubles need R2 -> 0.
#      payout = 58.0
#  T2: R4 {4,7,10}: none (winning 3-10, 3-5, 5-10);  R6 {1,2,8}: 1,8 pays 41.0 -> 41.0
#  T3: R7 {1,7,8}: 1,8 pays 33.5;  R9 {4,7,8}: 7,8 pays 58.0;
#      double (1,8)/(7,8) = 10 * 3.35 * 5.80 = 194.30
#      payout = 33.5 + 58.0 + 194.3 = 285.8
QPL_PAYOUTS = [58.0, 41.0, 285.8]


def build_meeting_db(path, divs=DIVS, scratched=None, status="results"):
    conn = db.connect(path)
    conn.execute("INSERT INTO meetings (date, venue, n_races, status) VALUES (?,?,?,?)",
                 (DATE, VENUE, len(divs), status))
    for rn, pools in divs.items():
        rid = f"{DATE.replace('-', '')}-{VENUE}-{rn:02d}"
        conn.execute("INSERT INTO races (race_id, date, venue, race_no) VALUES (?,?,?,?)",
                     (rid, DATE, VENUE, rn))
        winner = int(next(iter(pools["WIN"])))
        placed = [int(h) for h in pools["PLA"] if int(h) != winner]
        order = [winner] + placed + [h for h in range(1, 13) if h != winner and h not in placed]
        for pos, h in enumerate(order, start=1):
            scr = int(h in (scratched or {}).get(rn, ()))
            conn.execute("INSERT INTO runners (race_id, horse_no, finish_pos, win_odds, scratched) "
                         "VALUES (?,?,?,?,?)", (rid, h, None if scr else pos, 5.0, scr))
        for pool, m in pools.items():
            for c, d in m.items():
                conn.execute("INSERT INTO dividends VALUES (?,?,?,?)", (rid, pool, c, d))
    conn.commit()
    return conn


@pytest.fixture
def conn(tmp_path):
    c = build_meeting_db(tmp_path / "t.db")
    yield c
    c.close()


def _balance_check(conn):
    rows = conn.execute("SELECT amount, balance FROM bankroll_ledger ORDER BY id").fetchall()
    run = 0.0
    for a, b in rows:
        run = round(run + a, 2)
        assert b == pytest.approx(run, abs=0.005)
    return run


def test_place_and_settle_real_meeting(conn):
    ids = paper.place_tickets(conn, "test-run", DATE, VENUE, [PLA_TICKET] + QPL_TICKETS)
    assert len(ids) == 4
    # seed 10,000 - 630 - 3*150 = 8,920
    assert paper.bankroll(conn) == pytest.approx(8920.0)
    assert conn.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0] == 4

    n = paper.settle(conn, DATE)
    assert n == 4
    got = {r["ticket_no"] if r["strategy"] == "QPL_2x3" else 0: r["payout"]
           for r in conn.execute("SELECT * FROM paper_tickets")}
    assert got[0] == pytest.approx(round(PLA_PAYOUT, 2))
    assert [got[1], got[2], got[3]] == pytest.approx(QPL_PAYOUTS)
    # 8,920 + 1,423.91 + 58 + 41 + 285.8 = 10,728.71
    assert paper.bankroll(conn) == pytest.approx(10728.71, abs=0.005)
    assert _balance_check(conn) == pytest.approx(10728.71, abs=0.005)
    r = conn.execute("SELECT * FROM paper_tickets WHERE strategy='PLA_6x63'").fetchone()
    assert r["status"] == "settled" and r["profit"] == pytest.approx(1423.91 - 630, abs=0.01)
    det = json.loads(r["detail"])
    assert len(det["combinations"]) == 63 and det["n_winning_units"] == 63

    # idempotent
    assert paper.settle(conn, DATE) == 0
    assert paper.bankroll(conn) == pytest.approx(10728.71, abs=0.005)
    assert conn.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0] == 1 + 4 + 4


def test_single_race_ticket(conn):
    # WIN on horse 8 in race 9 (won, $54 per $10) at $20 unit -> payout 108
    t = {"strategy": "WIN_top1", "ticket_no": 1, "pool": "WIN", "spec": None, "unit_stake": 20.0,
         "legs": [{"race_no": 9, "selections": [8]}]}
    paper.place_tickets(conn, "r", DATE, VENUE, [t])
    paper.settle(conn, DATE)
    r = conn.execute("SELECT payout, total_stake FROM paper_tickets").fetchone()
    assert r["total_stake"] == 20.0 and r["payout"] == pytest.approx(108.0)
    assert paper.bankroll(conn) == pytest.approx(10000 - 20 + 108)


def test_scratched_leg_in_allup(tmp_path):
    # Race 2 horse 10 scratched: in the PLA 6x63 that leg's factor becomes 1.0
    # (refund / carried forward).  payout = 10 * (prod over others(1+f) * (1+1) - 1)
    #   others: 2.20 * 2.01 * 2.30 * 2.15 * 3.05 = 66.693709500 ; *2 = 133.387419
    #   payout = 10 * 132.387419 = 1323.87419
    c = build_meeting_db(tmp_path / "s.db", scratched={2: [10]})
    paper.place_tickets(c, "r", DATE, VENUE, [PLA_TICKET])
    paper.settle(c, DATE)
    r = c.execute("SELECT payout, detail FROM paper_tickets").fetchone()
    assert r["payout"] == pytest.approx(1323.87, abs=0.01)
    assert json.loads(r["detail"])["n_refunded_units"] == 1      # the 1-leg unit on race 2


def test_pending_then_abandoned_race(tmp_path):
    divs = {k: v for k, v in DIVS.items() if k != 9}         # race 9 never got results
    c = build_meeting_db(tmp_path / "a.db", divs=divs, status="provisional")
    paper.place_tickets(c, "r", DATE, VENUE, [QPL_TICKETS[2]])
    assert paper.settle(c, DATE) == 0                         # still waiting for race 9
    c.execute("UPDATE meetings SET status='abandoned'")
    c.commit()
    assert paper.settle(c, DATE) == 1
    # race 9 void -> each R9 unit refunded (factor 1): R7 singles 33.5 + R9 singles 3*10
    # + doubles: (1,8)/(any R9 pair) = 3 units * 10 * 3.35 * 1 = 100.5 -> 164.0
    r = c.execute("SELECT payout, detail FROM paper_tickets").fetchone()
    assert r["payout"] == pytest.approx(33.5 + 30.0 + 100.5)
    assert json.loads(r["detail"])["void_legs"] == [9]


def test_replace_voids_previous_open_tickets(conn):
    paper.place_tickets(conn, "run1", DATE, VENUE, [PLA_TICKET])
    paper.place_tickets(conn, "run2", DATE, VENUE, [PLA_TICKET])
    st = [r[0] for r in conn.execute("SELECT status FROM paper_tickets ORDER BY ticket_id")]
    assert st == ["void", "open"]
    assert paper.bankroll(conn) == pytest.approx(10000 - 630)
    assert paper.settle(conn, DATE) == 1
    assert _balance_check(conn) == pytest.approx(10000 - 630 + 1423.91, abs=0.005)


def test_starting_bankroll_and_bad_ticket(conn):
    assert paper.bankroll(conn) == 10000.0                    # unseeded default
    bad = dict(PLA_TICKET, n_combos=62)
    with pytest.raises(ValueError):
        paper.place_tickets(conn, "r", DATE, VENUE, [bad], starting_bankroll=5000)
    # seed happened (5000) but the failed ticket was rolled back
    assert paper.bankroll(conn) == 5000.0
    assert conn.execute("SELECT COUNT(*) FROM paper_tickets").fetchone()[0] == 0


def test_summary_md(conn):
    paper.place_tickets(conn, "r", DATE, VENUE, [PLA_TICKET] + QPL_TICKETS)
    paper.settle(conn, DATE)
    md = paper.summary_md(conn)
    assert "模擬投注總結" in md and "PLA_6x63" in md and "QPL_2x3" in md
    assert "10,728.71" in md and "跑馬地" in md
    # QPL: stake 450, return 384.8 -> ROI -14.5%
    assert "-14.5%" in md


# ------------------------------------------------------------------ end-to-end on a DB copy
@pytest.mark.skipif(not REAL_DB.exists(), reason="no data/hkjc.db")
def test_end_to_end_on_scratch_copy(tmp_path):
    """Copy the real DB (sqlite backup API - safe with the concurrent writer),
    build market-baseline tickets for a past meeting, place, settle, check ledger."""
    from hkjc import backtest as bt, strategy as st
    src = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    dst_path = tmp_path / "copy.db"
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    src.close(); dst.close()
    c = db.connect(dst_path)
    c.execute("DELETE FROM paper_tickets"); c.execute("DELETE FROM bankroll_ledger"); c.commit()
    md = bt.load_data(c, "market", DATE, DATE)
    if not md:
        pytest.skip("meeting not in DB")
    frame = pd.concat([r["df"] for r in md[0]["races"].values()], ignore_index=True)
    tickets = st.build_tickets(frame)
    assert [t["legs"] for t in tickets] == [
        [{**l, "pool": t["pool"]} for l in t["legs"]] for t in [PLA_TICKET] + QPL_TICKETS]
    ids = paper.place_tickets(c, "e2e", DATE, VENUE, tickets)
    assert paper.settle(c, DATE) == len(ids) == 4
    pays = [r[0] for r in c.execute("SELECT payout FROM paper_tickets ORDER BY ticket_id")]
    assert pays == pytest.approx([round(PLA_PAYOUT, 2)] + QPL_PAYOUTS)
    stake = sum(t["total_stake"] for t in tickets)
    assert stake == 1080.0
    assert paper.bankroll(c) == pytest.approx(10000 - stake + sum(pays), abs=0.005)
    assert _balance_check(c) == pytest.approx(paper.bankroll(c), abs=0.005)
    c.close()


@pytest.mark.skipif(not REAL_DB.exists(), reason="no data/hkjc.db")
def test_end_to_end_with_adaptive_tickets(tmp_path):
    """Scratch copy: model (wf-combined-v1) predictions for a past meeting where the
    ADAPTIVE product bets; place the default product set (fixed baseline + adaptive),
    settle, and check every ADAPTIVE payout against an independent settle_single on
    the DB dividends, plus the ledger arithmetic."""
    from hkjc import backtest as bt, bets, strategy as st
    src = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    if not src.execute("SELECT 1 FROM predictions WHERE run_id='wf-combined-v1' LIMIT 1").fetchone():
        pytest.skip("no wf-combined-v1 predictions")
    dst_path = tmp_path / "copy.db"
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    src.close(); dst.close()
    c = db.connect(dst_path)
    c.execute("DELETE FROM paper_tickets"); c.execute("DELETE FROM bankroll_ledger"); c.commit()
    md = bt.load_data(c, "wf-combined-v1", "2025-01-01", "2025-06-30")
    chosen = None
    for m in md:
        frame = pd.concat([r["df"] for r in m["races"].values()], ignore_index=True)
        tk = st.build_tickets(frame)
        if any(t["strategy"] == "ADAPTIVE" for t in tk):
            chosen = (m, tk)
            break
    if chosen is None:
        pytest.skip("no meeting with ADAPTIVE bets in window")
    m, tickets = chosen
    names = [t["strategy"] for t in tickets]
    assert names.count("PLA_6x63") == 1 and names.count("QPL_2x3") == 3
    ids = paper.place_tickets(c, "e2e-ad", m["date"], m["venue"], tickets)
    assert paper.settle(c, m["date"]) == len(ids)
    for t, tid in zip(tickets, ids):
        row = c.execute("SELECT payout, total_stake FROM paper_tickets WHERE ticket_id=?", (tid,)).fetchone()
        assert row["total_stake"] == t["total_stake"]
        if t["strategy"] != "ADAPTIVE":
            continue
        leg = t["legs"][0]
        rid = f"{m['date'].replace('-', '')}-{m['venue']}-{leg['race_no']:02d}"
        rows = c.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id=?", (rid,)).fetchall()
        scr = [r[0] for r in c.execute("SELECT horse_no FROM runners WHERE race_id=? AND scratched=1", (rid,))]
        ref = bets.settle_single(t["pool"], st.leg_combos(leg), [tuple(r) for r in rows], 10.0, scr)
        assert row["payout"] == pytest.approx(round(ref["payout"], 2))
    stake = sum(t["total_stake"] for t in tickets)
    pays = sum(r[0] for r in c.execute("SELECT payout FROM paper_tickets"))
    assert paper.bankroll(c) == pytest.approx(10000 - stake + pays, abs=0.01)
    assert _balance_check(c) == pytest.approx(paper.bankroll(c), abs=0.005)
    assert "ADAPTIVE" in paper.summary_md(c)
    c.close()


def test_replace_voids_all_strategies_of_earlier_run(conn):
    """A re-run (e.g. after a crash) whose ADAPTIVE product now skips every race must
    not leave the first run's ADAPTIVE ticket live; the same run may add tickets."""
    adaptive = {"strategy": "ADAPTIVE", "ticket_no": 1, "pool": "WIN", "spec": None, "unit_stake": 10.0,
                "legs": [{"race_no": 1, "selections": [10]}]}
    paper.place_tickets(conn, "run1", DATE, VENUE, [PLA_TICKET, adaptive])
    paper.place_tickets(conn, "run1", DATE, VENUE, QPL_TICKETS[:1])        # same run: kept
    assert conn.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0] == 3
    paper.place_tickets(conn, "run2", DATE, VENUE, [PLA_TICKET])
    st = [tuple(r) for r in conn.execute("SELECT run_id, strategy, status FROM paper_tickets ORDER BY ticket_id")]
    assert st == [("run1", "PLA_6x63", "void"), ("run1", "ADAPTIVE", "void"), ("run1", "QPL_2x3", "void"),
                  ("run2", "PLA_6x63", "open")]
    assert paper.bankroll(conn) == pytest.approx(10000 - 630)
    assert _balance_check(conn) == pytest.approx(paper.bankroll(conn), abs=0.005)
