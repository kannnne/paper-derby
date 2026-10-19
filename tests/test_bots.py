"""Tests for hkjc.bots (10 paper-trading robots) and the multi-bot parts of hkjc.paper.

Synthetic meetings check each robot's rules and reason texts; DB tests run on a
scratch COPY of data/hkjc.db (sqlite backup API) - the real DB is never written.
"""
import json
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hkjc import bets, bots, db, paper

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "hkjc.db"
DATE, VENUE = "2026-10-19", "ST"


# ----------------------------------------------------------------- synthetic meetings
def race(rn, p_win, win_odds=None, p_place=None, date="20990101", venue="ST", scratched=()):
    n = len(p_win)
    p = np.asarray(p_win, float)
    p = p / p.sum()
    pp = p_place if p_place is not None else np.clip(p * 2.6, 0.02, 0.95)
    return pd.DataFrame({"race_id": f"{date}-{venue}-{rn:02d}", "race_no": rn, "horse_no": range(1, n + 1),
                         "p_win": p, "p_place": pp,
                         "win_odds": win_odds if win_odds is not None else np.round(0.82 / p, 1),
                         "scratched": [int(h in scratched) for h in range(1, n + 1)]})


def meeting(n_races=8, seed=1, field=12, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for rn in range(1, n_races + 1):
        p = rng.dirichlet(np.full(field, 1.2))
        out.append(race(rn, p, **kw))
    return pd.concat(out, ignore_index=True)


def has_number(s):
    return re.search(r"\d", s) is not None


def test_bot_table():
    ids = [b["id"] for b in bots.BOTS]
    assert ids == ["place_only", "fixed_combo", "adaptive", "win_value", "qpl_box", "qin_banker",
                   "tri_banker", "pla_parlay", "favourite", "monkey"]
    for b in bots.BOTS:
        assert b["name_zh"] and b["style_zh"] and "\n" not in b["style_zh"]
        assert len(b["description_zh"]) > 40
        assert b["start_bankroll"] == 10_000 and b["unit_stake"] == 10
    assert bots.BOT_BY_ID["place_only"]["name_zh"] == "位置專家"


@pytest.mark.parametrize("bid", [b["id"] for b in bots.BOTS])
def test_every_race_gets_a_reasoned_decision(bid):
    pred = meeting(8, seed=3)
    decs = bots.decide(bid, pred, None, 10_000)
    races = {d["race_no"] for d in decs if d["race_no"] > 0}
    assert races == set(range(1, 9))
    for d in decs:
        assert d["bot_id"] == bid and d["action"] in ("bet", "skip")
        assert d["reason"] and has_number(d["reason"]), d
        assert (d["ticket"] is not None) == (d["action"] == "bet" and d["stake"] > 0)
        if d["ticket"]:
            t = d["ticket"]
            assert d["stake"] == t["total_stake"] == t["unit_stake"] * t["n_combos"]
            assert t["reason"] == d["reason"] and t["bot_id"] == bid
    # meeting-level robots: race_no 0 rows
    if bid in ("fixed_combo", "pla_parlay"):
        assert any(d["race_no"] == 0 for d in decs)
    else:
        assert not any(d["race_no"] == 0 for d in decs)
    # structure unique per (race, structure) - bot_decisions primary key
    keys = [(d["race_no"], d["structure"]) for d in decs]
    assert len(keys) == len(set(keys))


def test_place_only_reason_numbers_live_odds():
    # horse 6: p_place 0.68, live PLA odds 1.9 -> EV 1.292 >= 0.90 ; others EV < 0.9
    p_win = [0.05] * 5 + [0.30] + [0.05] * 6
    pp = [0.15] * 5 + [0.68] + [0.15] * 6
    pred = race(1, p_win, p_place=pp)
    odds = {1: {"PLA": {str(h): (1.9 if h == 6 else 4.0) for h in range(1, 13)},
                "WIN": {str(h): 3.0 if h == 6 else 15.0 for h in range(1, 13)}}}
    d, = bots.decide("place_only", pred, odds, 10_000)
    assert d["action"] == "bet" and d["selections"] == [6] and d["stake"] == 10
    for s in ("6號", "上名率68%", "位置賠率1.9", "期望值1.29", "門檻0.90", "$10"):
        assert s in d["reason"], (s, d["reason"])
    assert "估計" not in d["reason"]                           # live odds
    t = d["ticket"]
    assert t["pool"] == "PLA" and t["spec"] is None and t["legs"][0]["selections"] == [6]


def test_place_only_skip_reason_and_max_two():
    p_win = [0.1] * 10
    pred = race(1, p_win, p_place=[0.3] * 10)
    odds = {1: {"PLA": {str(h): 2.0 for h in range(1, 11)}}}          # EV 0.6 everywhere
    d, = bots.decide("place_only", pred, odds, 10_000)
    assert d["action"] == "skip"
    assert "最高期望值只得0.60" in d["reason"] and "唔夠門檻0.90" in d["reason"] and "唔買" in d["reason"]
    odds = {1: {"PLA": {str(h): 4.0 for h in range(1, 11)}}}          # EV 1.2 everywhere
    d, = bots.decide("place_only", pred, odds, 10_000)
    assert d["action"] == "bet" and len(d["selections"]) == 2 and d["stake"] == 20
    assert "$20" in d["reason"]


def test_place_only_only_ever_pla_singles():
    for seed in range(6):
        pred = meeting(10, seed=seed)
        rng = np.random.default_rng(seed)
        odds = {rn: {"PLA": {str(h): float(rng.uniform(1.1, 6)) for h in range(1, 13)}} for rn in range(1, 11)}
        for o in (None, odds):
            for d in bots.decide("place_only", pred, o, 10_000):
                assert d["race_no"] > 0
                if d["ticket"]:
                    t = d["ticket"]
                    assert t["pool"] == "PLA" and t["spec"] is None and len(t["legs"]) == 1
                    assert t["legs"][0]["pool"] == "PLA" and 1 <= len(t["legs"][0]["selections"]) <= 2
                    assert not t["legs"][0].get("bankers")


def test_estimated_and_market_free_odds_are_labelled():
    pred = meeting(2, seed=5)
    d = bots.decide("win_value", pred, None, 10_000)[0]
    assert "(估計賠率)" in d["reason"]
    nomkt = pred.assign(win_odds=np.nan)
    for bid in ("win_value", "place_only"):
        for d in bots.decide(bid, nomkt, None, 10_000):
            assert d["action"] == "skip" and "未開售" in d["reason"]
    fav = bots.decide("favourite", nomkt, None, 10_000)
    assert all(d["action"] == "bet" for d in fav) and "未有賠率，用模型首選" in fav[0]["reason"]


def test_controls_bet_every_race_and_monkey_reproducible():
    pred = meeting(9, seed=7)
    fav = bots.decide("favourite", pred, None, 10_000)
    assert sum(1 for d in fav if d["ticket"]) == 18            # WIN + PLA in each of 9 races
    for rn in range(1, 10):
        g = pred[pred.race_no == rn]
        fh = int(g.sort_values(["win_odds", "horse_no"]).horse_no.iloc[0])
        assert [d["selections"] for d in fav if d["race_no"] == rn] == [[fh], [fh]]
        assert {d["ticket"]["pool"] for d in fav if d["race_no"] == rn} == {"WIN", "PLA"}
    m1 = bots.decide("monkey", pred, None, 10_000)
    m2 = bots.decide("monkey", pred.sample(frac=1, random_state=3), None, 10_000)
    assert [d["selections"] for d in m1] == [d["selections"] for d in m2]
    assert all(d["action"] == "bet" and d["ticket"]["pool"] == "PLA" for d in m1) and len(m1) == 9
    # seed comes from race_id: another meeting date -> different picks somewhere
    other = pred.assign(race_id=pred.race_id.str.replace("20990101", "20990102"))
    m3 = bots.decide("monkey", other, None, 10_000)
    assert [d["selections"] for d in m3] != [d["selections"] for d in m1]
    assert "隨機（種子 20990101-ST-01）" in m1[0]["reason"]


def test_scratched_horses_never_selected():
    pred = meeting(8, seed=11)
    # scratch each race's top model pick (racecard flag) ...
    top = pred.sort_values(["race_no", "p_win"], ascending=[True, False]).groupby("race_no").head(1)
    scr = set(zip(top.race_no, top.horse_no))
    pred["scratched"] = [int((r, h) in scr) for r, h in zip(pred.race_no, pred.horse_no)]
    # ... and in race 1 also horse number of the 2nd pick via live odds 'SCR' (None)
    g1 = pred[(pred.race_no == 1) & (pred.scratched == 0)].sort_values("p_win", ascending=False)
    h_scr = int(g1.horse_no.iloc[0])
    odds = {1: {"WIN": {str(h): (None if h == h_scr else 5.0) for h in range(1, 13)}}}
    for b in bots.BOTS:
        for o in (None, odds):
            banned = scr | ({(1, h_scr)} if o else set())
            for d in bots.decide(b, pred, o, 10_000):
                t = d["ticket"]
                if not t:
                    continue
                for leg in t["legs"]:
                    hs = list(leg.get("selections", [])) + list(leg.get("bankers", []))
                    for h in hs:
                        assert (leg["race_no"], h) not in banned, (b["id"], leg)


def test_insufficient_bankroll():
    pred = meeting(3, seed=2)
    decs = bots.decide("favourite", pred, None, 15.0)
    assert [d["action"] for d in decs] == ["bet", "skip", "skip", "skip", "skip", "skip"]
    assert "資金不足" in decs[1]["reason"] and "$5" in decs[1]["reason"]
    fc = bots.decide("fixed_combo", meeting(8, seed=2), None, 500.0)
    top = {d["structure"]: d for d in fc if d["race_no"] == 0}
    assert top["PLA 6x63"]["action"] == "skip" and "資金不足" in top["PLA 6x63"]["reason"]
    assert [top[f"QPL 2x3 #{i}"]["action"] for i in (1, 2, 3)] == ["bet", "bet", "bet"]   # 3 x 150 <= 500
    # leg rows of the unfunded 6x63 that are not QPL legs say so
    legs = [d for d in fc if d["race_no"] > 0 and d["structure"] == "過關腳"]
    assert any("資金不足" in d["reason"] for d in legs)


def test_fixed_combo_and_parlay_structure():
    pred = meeting(10, seed=4)
    fc = bots.decide("fixed_combo", pred, None, 10_000)
    top = [d for d in fc if d["race_no"] == 0]
    assert [d["structure"] for d in top] == ["PLA 6x63", "QPL 2x3 #1", "QPL 2x3 #2", "QPL 2x3 #3"]
    assert [d["stake"] for d in top] == [630, 150, 150, 150]
    assert top[0]["ticket"]["spec"] == "6x63" and len(top[0]["ticket"]["legs"]) == 6
    leg_races = {l["race_no"] for d in top for l in d["ticket"]["legs"]}
    per = {d["race_no"]: d for d in fc if d["race_no"] > 0}
    assert set(per) == set(range(1, 11))
    for rn, d in per.items():
        assert d["stake"] == 0 and d["ticket"] is None
        assert (d["action"] == "bet") == (rn in leg_races)
        assert d["reason"].startswith(f"第{rn}場係過關腳") == (rn in leg_races)
    # pla_parlay: confident races -> one 3x4 ticket on the 3 highest p_place top picks
    p = [0.55, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
    races = [race(rn, p, p_place=[0.70 + rn / 100] + [0.2] * 7, win_odds=[3.0] + [8.0] * 7) for rn in range(1, 6)]
    pp = bots.decide("pla_parlay", pd.concat(races), None, 10_000)
    d0 = pp[0]
    assert d0["race_no"] == 0 and d0["action"] == "bet" and d0["stake"] == 40
    assert [l["race_no"] for l in d0["ticket"]["legs"]] == [3, 4, 5] and d0["ticket"]["spec"] == "3x4"
    assert "上名率75%" in d0["reason"] and "4 注 $40" in d0["reason"]
    assert [d["action"] for d in pp[1:]] == ["skip", "skip", "bet", "bet", "bet"]
    weak = pd.concat([race(rn, p, p_place=[0.5] + [0.2] * 7) for rn in range(1, 6)])
    pp = bots.decide("pla_parlay", weak, None, 10_000)
    assert pp[0]["action"] == "skip" and "只得 0 場首選合格" in pp[0]["reason"]


def test_qin_tri_qpl_rules():
    strong = race(1, [0.45, 0.2, 0.1, 0.08, 0.05, 0.04, 0.03, 0.03, 0.02], win_odds=[2.2, 5, 9, 12, 20, 25, 30, 30, 50])
    flat = race(2, [0.12] * 8 + [0.04], win_odds=[7.0] * 8 + [20])
    pred = pd.concat([strong, flat])
    q = bots.decide("qin_banker", pred, None, 10_000)
    assert q[1]["action"] == "skip" and "唔算明顯首選" in q[1]["reason"]
    assert "首選 1號 勝率45%" in q[0]["reason"]
    t = bots.decide("tri_banker", pred, None, 10_000)
    assert t[1]["action"] == "skip" and "唔算兩隻強馬" in t[1]["reason"]
    if t[0]["action"] == "bet":
        tk = t[0]["ticket"]
        assert tk["pool"] == "TRI" and tk["legs"][0]["bankers"] == [1, 2] and tk["n_combos"] == 4
    small = race(3, [0.3, 0.25, 0.2, 0.15, 0.1, 0.0001][:5])
    d, = bots.decide("qpl_box", small, None, 10_000)
    assert d["action"] == "skip" and "少過 7 匹冇位置Q" in d["reason"]
    d = bots.decide("qpl_box", strong, None, 10_000)[0]
    assert d["action"] == "bet" and d["ticket"]["n_combos"] == 3 and "命中機率" in d["reason"]


def test_adaptive_matches_strategy_decide_race():
    from hkjc import strategy as st
    pred = meeting(8, seed=9)
    decs = bots.decide("adaptive", pred, None, 10_000)
    for d in decs:
        g = pred[pred.race_no == d["race_no"]]
        ref = st.decide_race(g.drop(columns=["race_id"]), None, {})
        if ref:
            assert d["action"] == "bet" and d["ticket"]["legs"] == ref[0]["legs"]
            assert d["stake"] == ref[0]["total_stake"] and "預期回報" in d["reason"]
        else:
            assert d["action"] == "skip"


# ----------------------------------------------------------------- DB copy tests
@pytest.fixture(scope="module")
def master_copy(tmp_path_factory):
    if not REAL_DB.exists():
        pytest.skip("no data/hkjc.db")
    src = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    if not src.execute("SELECT 1 FROM predictions WHERE run_id='wf-combined-v1' AND race_id LIKE ? LIMIT 1",
                       (DATE.replace("-", "") + "-" + VENUE + "%",)).fetchone():
        src.close()
        pytest.skip("meeting predictions not in DB")
    p = tmp_path_factory.mktemp("botdb") / "master.db"
    dst = sqlite3.connect(p)
    src.backup(dst)
    src.close()
    dst.close()
    c = db.connect(p)
    for t in ("paper_tickets", "bankroll_ledger", "bot_decisions"):
        c.execute(f"DELETE FROM {t}")
    c.commit()
    c.close()
    return p


@pytest.fixture
def cdb(master_copy, tmp_path):
    p = tmp_path / "copy.db"
    shutil.copy(master_copy, p)
    c = db.connect(p)
    yield c
    c.close()


def _ledger_ok(conn, bot_id):
    rows = conn.execute("SELECT amount, balance FROM bankroll_ledger WHERE bot_id IS ? ORDER BY id",
                        (bot_id,)).fetchall()
    run = 0.0
    for a, b in rows:
        run = round(run + a, 2)
        assert b == pytest.approx(run, abs=0.005)
    return run


def test_run_bots_settle_and_ledgers_real_meeting(cdb):
    pred = bots.meeting_frame(cdb, DATE, VENUE)
    assert pred["scratched"].sum() >= 1                      # this meeting has scratchings
    decs = bots.run_bots(cdb, "pre-test-1", DATE, VENUE, pred, None)
    # every bot x every predicted race has a decision row in the DB
    races = set(pred["race_no"])
    for b in bots.BOTS:
        got = {r[0] for r in cdb.execute("SELECT race_no FROM bot_decisions WHERE bot_id=? AND race_no>0",
                                         (b["id"],))}
        assert got == races
    nd = cdb.execute("SELECT COUNT(*) FROM bot_decisions").fetchone()[0]
    assert nd == len(decs)
    # scratched horses never on a ticket
    scr = {(int(r), int(h)) for r, h in pred.loc[pred.scratched == 1, ["race_no", "horse_no"]].values}
    for t in cdb.execute("SELECT legs FROM paper_tickets"):
        for leg in json.loads(t[0]):
            for h in leg.get("selections", []) + leg.get("bankers", []):
                assert (leg["race_no"], h) not in scr
    # ticket ids linked + reasons stored
    for d in decs:
        if d["ticket"]:
            row = cdb.execute("SELECT bot_id, reason, total_stake FROM paper_tickets WHERE ticket_id=?",
                              (d["ticket_id"],)).fetchone()
            assert row["bot_id"] == d["bot_id"] and row["reason"] == d["reason"] and row["total_stake"] == d["stake"]
    stakes = {b["id"]: sum(d["stake"] for d in decs if d["bot_id"] == b["id"]) for b in bots.BOTS}
    for bid, s in stakes.items():
        assert paper.bankroll(cdb, bid) == pytest.approx(10_000 - s)
    assert paper.bankroll(cdb) == 10_000                      # legacy ledger untouched / unseeded
    n_open = cdb.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0]
    assert paper.settle(cdb, DATE) == n_open
    for b in bots.BOTS:
        bid = b["id"]
        pays = cdb.execute("SELECT COALESCE(SUM(payout),0), COALESCE(SUM(total_stake),0) FROM paper_tickets "
                           "WHERE bot_id=? AND status='settled'", (bid,)).fetchone()
        assert pays[1] == pytest.approx(stakes[bid])
        assert paper.bankroll(cdb, bid) == pytest.approx(10_000 - stakes[bid] + pays[0], abs=0.01)
        assert _ledger_ok(cdb, bid) == pytest.approx(paper.bankroll(cdb, bid), abs=0.005)
    # same P&L as the backtest's settlement of the same decisions (shared decide path)
    from hkjc import backtest as bt
    md = bt.load_data(cdb, "wf-combined-v1", DATE, DATE)
    frames = bots._meeting_frames(md)
    for b in bots.BOTS:
        rows, kept = bots.simulate(b, md, frames, keep_decisions=True)
        mine = [d for d in decs if d["bot_id"] == b["id"]]
        assert bots._strip(kept[(DATE, VENUE)]) == bots._strip(mine), b["id"]
        pays = cdb.execute("SELECT COALESCE(SUM(payout),0) FROM paper_tickets WHERE bot_id=?",
                           (b["id"],)).fetchone()[0]
        assert rows[0]["returns"] == pytest.approx(pays, abs=0.01 * max(1, rows[0]["n_bets"]))
    # summaries
    summ = {s["bot_id"]: s for s in paper.bot_summary(cdb)}
    assert list(summ) == [b["id"] for b in bots.BOTS]
    fav = summ["favourite"]
    assert fav["stake"] == pytest.approx(stakes["favourite"]) and fav["n_meetings"] == 1
    assert fav["profit"] == pytest.approx(fav["returns"] - fav["stake"], abs=0.01)
    md_txt = paper.summary_md(cdb)
    assert "10 隻機械人" in md_txt and "位置專家" in md_txt and "舊有單一機械人" not in md_txt
    assert paper.settle(cdb, DATE) == 0                       # idempotent
    # after settlement a re-run must not bet again (no double betting after results)
    n_t = cdb.execute("SELECT COUNT(*) FROM paper_tickets").fetchone()[0]
    assert bots.run_bots(cdb, "pre-test-late", DATE, VENUE, pred, None) == []
    assert cdb.execute("SELECT COUNT(*) FROM paper_tickets").fetchone()[0] == n_t


def test_run_bots_idempotent_rerun(cdb):
    pred = bots.meeting_frame(cdb, DATE, VENUE)
    d1 = bots.run_bots(cdb, "pre-a", DATE, VENUE, pred, None)
    bal1 = {b["id"]: paper.bankroll(cdb, b["id"]) for b in bots.BOTS}
    n1 = cdb.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0]
    d2 = bots.run_bots(cdb, "pre-b", DATE, VENUE, pred, None)          # e.g. pre --force
    d3 = bots.run_bots(cdb, "pre-b", DATE, VENUE, pred, None)          # same run twice
    assert bots._strip(d1) == bots._strip(d2) == bots._strip(d3)
    assert cdb.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0] == n1
    assert cdb.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='void'").fetchone()[0] == 2 * n1
    for b in bots.BOTS:
        assert paper.bankroll(cdb, b["id"]) == pytest.approx(bal1[b["id"]])
        _ledger_ok(cdb, b["id"])
    assert cdb.execute("SELECT COUNT(*) FROM bot_decisions").fetchone()[0] == len(d1)
    assert {r[0] for r in cdb.execute("SELECT DISTINCT run_id FROM bot_decisions")} == {"pre-b"}
    # a legacy (bot_id NULL) placement for the same meeting leaves bot tickets alone, and vice versa
    legacy = {"strategy": "ADAPTIVE", "ticket_no": 1, "pool": "WIN", "spec": None, "unit_stake": 10.0,
              "legs": [{"race_no": 1, "selections": [1]}]}
    paper.place_tickets(cdb, "legacy-run", DATE, VENUE, [legacy])
    assert cdb.execute("SELECT COUNT(*) FROM paper_tickets WHERE status='open'").fetchone()[0] == n1 + 1
    bots.run_bots(cdb, "pre-c", DATE, VENUE, pred, None)
    assert cdb.execute("SELECT status FROM paper_tickets WHERE bot_id IS NULL").fetchone()[0] == "open"
    assert paper.bankroll(cdb) == 9_990
    md_txt = paper.summary_md(cdb)
    assert "10 隻機械人" in md_txt and "舊有單一機械人" in md_txt


def test_place_tickets_old_signature_still_works(tmp_path):
    c = db.connect(tmp_path / "x.db")
    t = {"strategy": "WIN_top1", "ticket_no": 1, "pool": "WIN", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "selections": [3]}]}
    paper.place_tickets(c, "r", "2024-05-22", "HV", [t])
    paper.place_tickets(c, "r", "2024-05-22", "HV", [t], bot_id="monkey", reason="測試")
    assert paper.bankroll(c) == 9_990 and paper.bankroll(c, "monkey") == 9_990
    rows = c.execute("SELECT bot_id, reason FROM paper_tickets ORDER BY ticket_id").fetchall()
    assert [tuple(r) for r in rows] == [(None, None), ("monkey", "測試")]
    assert paper.void_open(c, "2024-05-22", "HV", "monkey") == [2]
    assert paper.bankroll(c, "monkey") == 10_000 and paper.bankroll(c) == 9_990
    c.close()


def test_backtest_bots_small_window(cdb, tmp_path):
    """Backtest machinery on a few meetings (tuning off): json shape + curve arithmetic."""
    from hkjc import backtest as bt
    md = bt.load_data(cdb, "wf-combined-v1", "2025-09-01", "2025-09-30") + \
        bt.load_data(cdb, "wf-combined-v1", DATE, DATE)
    if len(md) < 2:
        pytest.skip("not enough meetings")
    out = tmp_path / "bb.json"
    df = bots.backtest_bots(cdb, tune_params=False, path=out, meetings=md,
                            bots=["favourite", "monkey", "pla_parlay"])
    js = json.loads(out.read_text())
    assert list(js) == ["favourite", "monkey", "pla_parlay"]
    for bid, v in js.items():
        o = v["oos"]
        rows = df[(df.bot_id == bid) & (df.phase == "oos")]
        assert o["stake"] == pytest.approx(rows.stake.sum()) and o["n_meetings"] == len(md)
        assert v["curve"][0]["bankroll"] == 10_000
        assert v["curve"][-1]["bankroll"] == pytest.approx(10_000 + rows.returns.sum() - rows.stake.sum(), abs=0.05)
        assert len(o["roi_ci"]) == 2 and set(v) >= {"params", "in_sample", "oos", "curve", "by_season"}


@pytest.mark.parametrize("crash", ["win_value", "fixed_combo"])
def test_run_bots_one_robot_crash_is_isolated(tmp_path, monkeypatch, crash):
    """A bug in one robot: the other nine still decide and bet; the broken one writes a
    'skip' row for every race (and race 0 for meeting-level robots) and places nothing."""
    c = db.connect(tmp_path / "crash.db")
    pred = meeting(n_races=8)

    def boom(*a, **k):
        raise ZeroDivisionError("bug")
    monkeypatch.setitem(bots._KINDS, bots._CFG[crash]["kind"], boom)
    decs = bots.run_bots(c, "pre-crash", "2099-01-01", "ST", pred, None)
    rows = c.execute("SELECT bot_id, race_no, action, reason FROM bot_decisions").fetchall()
    assert {r[0] for r in rows} == {b["id"] for b in bots.BOTS}
    mine = [r for r in rows if r[0] == crash]
    want = ([0] if crash == "fixed_combo" else []) + list(range(1, 9))
    assert sorted(r[1] for r in mine) == want
    assert all(r[2] == "skip" and "系統錯誤" in r[3] and has_number(r[3]) for r in mine)
    assert c.execute("SELECT COUNT(*) FROM paper_tickets WHERE bot_id=?", (crash,)).fetchone()[0] == 0
    assert paper.bankroll(c, crash) == 10_000
    assert c.execute("SELECT COUNT(*) FROM paper_tickets WHERE bot_id='favourite'").fetchone()[0] == 16
    assert len([d for d in decs if d["bot_id"] == crash]) == len(want)
    c.close()


def test_parlay_non_leg_reasons_are_accurate():
    """Per-race rows of a meeting-level robot: a race that is not a leg says it wasn't among
    the chosen races (not 'low confidence'), and on a no-parlay day it points to race 0."""
    pred = meeting(n_races=8, seed=3)
    b = bots.BOT_BY_ID["pla_parlay"]
    yes = bots.decide({**b, "config": {**b["config"], "min_p": 0.0, "min_ev": None}}, pred, None, 10_000)
    assert any(d["race_no"] == 0 and d["action"] == "bet" for d in yes)
    non = [d for d in yes if d["race_no"] > 0 and d["action"] == "skip"]
    assert len(non) == 5 and all("唔係入選" in d["reason"] and "信心唔夠高" not in d["reason"] for d in non)
    no = bots.decide({**b, "config": {**b["config"], "min_p": 0.999}}, pred, None, 10_000)
    assert [d["action"] for d in no if d["race_no"] == 0] == ["skip"]
    assert all("今日冇過關飛" in d["reason"] and has_number(d["reason"]) for d in no if d["race_no"] > 0)
