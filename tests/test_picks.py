"""Tests for hkjc.picks (今日精選 analysis engine) and the multi-race pools it adds.

Hand-checked candidate maths (FF / TCE / QTT / DBL probabilities and unit counts), selection
rules, the no-odds case, settlement of the new pools through hkjc.paper, and live/backtest
parity on a read-only view of data/hkjc.db (never written).
"""
import sqlite3
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hkjc import bets, db, paper, picks, strategy as st

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "hkjc.db"
OPEN_TH = {f: {"tau": -1.0, "h": 0.0} for f in picks.FAMILIES}
OFF_TH = {f: None for f in picks.FAMILIES}


# ----------------------------------------------------------------- synthetic meetings
def race(rn, p_win, odds_margin=0.82, date="20990101"):
    p = np.asarray(p_win, float)
    p = p / p.sum()
    return pd.DataFrame({"race_id": f"{date}-ST-{rn:02d}", "race_no": rn, "horse_no": range(1, len(p) + 1),
                         "p_win": p, "p_place": np.clip(p * 2.6, 0.02, 0.95),
                         "win_odds": np.round(odds_margin / p, 1), "horse_name": [f"馬{h}" for h in range(1, len(p) + 1)]})


def meeting(n=8, field=10, seed=1):
    rng = np.random.default_rng(seed)
    return pd.concat([race(rn, rng.dirichlet(np.full(field, 1.3))) for rn in range(1, n + 1)], ignore_index=True)


def gen_harville(p, lam2, lam3, order):
    """Brute-force P(order) under the discounted Harville model (q_k = p**lam_k)."""
    p = np.asarray(p, float) / np.sum(p)
    lams = [1.0, lam2, lam3, lam3]
    used, prob = [], 1.0
    for k, h in enumerate(order):
        w = p ** lams[k]
        rest = [i for i in range(len(p)) if i not in used]
        prob *= w[h] / sum(w[i] for i in rest)
        used.append(h)
    return prob


P10 = [0.25, 0.18, 0.14, 0.11, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02]


def _cands(pred, odds=None, cfg=None):
    cands, ctx = picks.enumerate_candidates(pred, odds, cfg)
    return {c["key"]: c for c in cands}, ctx


# ----------------------------------------------------------------- candidate maths
def test_ff_two_bankers_probability_and_cost():
    by, ctx = _cands(race(1, P10))
    c = by["1:FF:bank:[2, 4]"]
    # 2 bankers (1, 2) + 4 legs (3..6): C(4,2) = 6 combinations, $60
    assert c["n_combos"] == 6 and c["stake"] == 60.0
    assert c["legs"][0]["bankers"] == [1, 2] and c["legs"][0]["selections"] == [3, 4, 5, 6]
    lam2, lam3 = picks.LAM2, picks.LAM3
    want = 0.0
    for a, b in ((2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)):   # 0-based legs 3..6
        for o in permutations((0, 1, a, b)):
            want += gen_harville(P10, lam2, lam3, o)
    assert c["hit_raw"] == pytest.approx(want, rel=1e-9)
    # FF hit probability adjusted by the 2024-25 family factor
    assert c["hit_prob"] == pytest.approx(want * picks.HIT_ADJ["FF"], rel=1e-9)


def test_tce_positional_and_qtt_box_probabilities():
    by, _ = _cands(race(1, P10))
    t = by["1:TCE:pos:[[1], [2], [3, 4, 5]]"]           # 頭兩名定位拖 3 腳: 3 combos
    assert t["n_combos"] == 3 and t["legs"][0]["positions"] == [[1], [2], [3, 4, 5]]
    want = sum(gen_harville(P10, picks.LAM2, picks.LAM3, (0, 1, k)) for k in (2, 3, 4))
    assert t["hit_raw"] == pytest.approx(want, rel=1e-9)
    q = by["1:QTT:box:4"]                               # 複式 4 隻: 4! = 24 combos, $240
    assert q["n_combos"] == 24 and q["stake"] == 240.0
    want = sum(gen_harville(P10, picks.LAM2, picks.LAM3, o) for o in permutations((0, 1, 2, 3)))
    assert q["hit_raw"] == pytest.approx(want, rel=1e-9)
    # QTT single: exactly the ordered top-4 probability
    s = by["1:QTT:pos:[[1], [2], [3], [4]]"]
    assert s["n_combos"] == 1
    assert s["hit_raw"] == pytest.approx(gen_harville(P10, picks.LAM2, picks.LAM3, (0, 1, 2, 3)), rel=1e-9)
    # expected return = sum P(c) * estimated dividend / 10 (before the family adjustment)
    r = picks.Race(__import__("hkjc.bots", fromlist=["prepare"]).prepare(race(1, P10))[0], picks._cfg())
    er = sum(r.prob("QTT", c) * r.div("QTT", c)[0] / 10 for c in bets.multiple("QTT", [1, 2, 3, 4]))
    assert q["er_raw"] == pytest.approx(er, rel=1e-9)


def test_dbl_probability_cost_and_runner_up_dividend():
    pred = pd.concat([race(1, P10), race(2, P10[::-1])], ignore_index=True)
    by, ctx = _cands(pred)
    c = by["DBL:1-2:(2, 3)"]
    # leg 1: top-2 by win prob (1, 2); last leg: top-3 by P(1st or 2nd) = horses 10, 9, 8
    assert c["legs"][0]["selections"] == [1, 2] and c["legs"][1]["selections"] == [10, 9, 8]
    assert c["n_combos"] == 6 and c["stake"] == 60.0 and c["kind"] == "multi_race" and c["race_no"] == 2
    p2 = np.asarray(P10[::-1]) / sum(P10)
    top2 = 0.0                                          # P(10, 9 or 8 finishes 1st or 2nd in leg 2)
    for a in range(10):
        for b in range(10):
            if a != b and ({a, b} & {9, 8, 7}):
                top2 += gen_harville(p2, picks.LAM2, picks.LAM3, (a, b))
    want = (P10[0] + P10[1]) / sum(P10) * top2
    assert c["hit_raw"] == pytest.approx(want, rel=1e-9)
    # dividends: winner and runner-up priced separately; runner-up much lower (calibration a<0)
    races = ctx["races"]
    d1, _ = picks._mr_div("DBL", [races[1], races[2]], "1/10", picks._cfg(), 1)
    d2, _ = picks._mr_div("DBL", [races[1], races[2]], "1/10", picks._cfg(), 2)
    assert d1 > d2 >= 10.0


def test_schedule_and_field_size_rules():
    s = picks.multi_race_legs(range(1, 12))
    assert s["DBL"][0] == (1, 2) and s["DBL"][-1] == (10, 11) and len(s["DBL"]) == 10
    assert s["TBL"] == [(3, 4, 5), (9, 10, 11)]
    assert picks.multi_race_legs(range(1, 10), n_races=11)["TBL"] == [(3, 4, 5)]   # races 10/11 missing
    assert picks.multi_race_legs([1, 2, 3, 4, 5, 7, 8])["DBL"] == [(1, 2), (2, 3), (3, 4), (4, 5), (7, 8)]
    by, _ = _cands(race(1, [0.4, 0.3, 0.2, 0.06, 0.04]))       # 5 runners: no QPL / FF / QTT
    pools = {c["pool"] for c in by.values()}
    assert {"WIN", "PLA", "QIN", "FCT", "TCE", "TRI"} <= pools and not pools & {"QPL", "FF", "QTT"}
    assert all(c["stake"] <= 300 for c in by.values())


def test_item_ticket_units_match_paper(tmp_path):
    pk = picks.make_picks(meeting(), None, {"thresholds": OPEN_TH, "max_items": 8, "max_total": 5000})
    c = db.connect(tmp_path / "t.db")
    for it in pk["items"]:
        t = it["ticket"]
        assert t["n_combos"] == it["n_combos"] and t["total_stake"] == it["stake"]
        assert t["notes"]["pick_key"] == it["key"]
    ids = paper.place_tickets(c, "r", "2099-01-01", "ST", [it["ticket"] for it in pk["items"]], bot_id="daily_picks")
    assert len(ids) == len(pk["items"])


# ----------------------------------------------------------------- selection rules
def _c(key, fam, pool, kind, rn, roi, hit, stake=10.0, races=None):
    return {"key": key, "family": fam, "pool": pool, "kind": kind, "race_no": rn, "races": races or [rn],
            "exp_roi": roi, "hit_prob": hit, "stake": stake, "n_combos": int(stake / 10), "unit_stake": 10.0}


def test_selection_rules():
    th = {**OFF_TH, "WIN": {"tau": 0.0, "h": 0.1}, "PLA": {"tau": 0.0, "h": 0.1}, "QIN": {"tau": 0.1, "h": 0.1},
          "TRI": {"tau": 0.0, "h": 0.05}, "DBL": {"tau": 0.2, "h": 0.05}}
    cfg = {"thresholds": th, "max_items": 4, "max_total": 100.0}
    cands = [
        _c("w1", "WIN", "WIN", "single", 1, 0.30, 0.2),
        _c("p1", "PLA", "PLA", "single", 1, 0.25, 0.5),       # same group as w1 in race 1 -> dropped
        _c("q1", "QIN", "QIN", "single", 1, 0.15, 0.2),       # margin 0.05 (tau 0.1)
        _c("t1", "TRI", "TRI", "single", 1, 0.20, 0.1),       # 3rd item in race 1 -> over max_per_race
        _c("q2", "QIN", "QIN", "single", 2, 0.05, 0.2),       # below tau 0.1
        _c("f2", "FCT", "FCT", "single", 2, 0.90, 0.3),       # family disabled
        _c("d1", "DBL", "DBL", "multi_race", 2, 0.60, 0.1, races=[1, 2]),
        _c("d2", "DBL", "DBL", "multi_race", 3, 0.50, 0.1, races=[2, 3]),   # shares race 2 with d1
        _c("t3", "TRI", "TRI", "single", 3, 0.10, 0.1, stake=90.0),        # would break the $100 cap
        _c("w4", "WIN", "WIN", "single", 4, 0.01, 0.2),
    ]
    sel, ref = picks.select_items(cands, cfg)
    # margins: d1 .40, w1 .30 (= d2, more likely to hit), p1 .25, t1 .20, t3 .10, q1 .05, w4 .01
    assert [c["key"] for c in sel] == ["d1", "w1", "t1", "w4"]
    keys = [c["key"] for c in sel]
    assert "p1" not in keys and "f2" not in keys and "q2" not in keys and "d2" not in keys and "t3" not in keys
    assert len(keys) <= 4 and sum(c["stake"] for c in sel) <= 100 and ref == []
    # ranking is by the margin over each family's own threshold
    assert keys[0] == "d1"                                # margin 0.40
    sel, ref = picks.select_items(cands, {"thresholds": OFF_TH})
    assert sel == [] and ref == []


def test_select_section_rules():
    cfg = picks._cfg({"section_min_hit": {**picks.SECTION_MIN_HIT, "WIN": 0.2, "TRI": 0.1}})
    win = [_c("1:WIN:top:1", "WIN", "WIN", "single", 1, -0.05, 0.40), _c("1:WIN:top:2", "WIN", "WIN", "single", 1, 0.10, 0.25),
           _c("2:WIN:top:1", "WIN", "WIN", "single", 2, 0.05, 0.30), _c("3:WIN:top:4", "WIN", "WIN", "single", 3, 0.90, 0.05)]
    got = picks.select_section(win, "WIN", cfg)
    # long shot (5% < 20%) excluded; best 2 by exp_roi from DIFFERENT races
    assert [c["key"] for c in got] == ["1:WIN:top:2", "2:WIN:top:1"]
    tri = [_c("1:TRI:box:3", "TRI", "TRI", "single", 1, -0.3, 0.06), _c("2:TRI:box:4", "TRI", "TRI", "single", 2, -0.1, 0.08)]
    got = picks.select_section(tri, "TRI", cfg)        # none reaches 10%: the most likely one, flagged
    assert [c["key"] for c in got] == ["2:TRI:box:4"] and got[0]["below_min_hit"]
    assert picks.select_section([], "QIN", cfg) == []
    # section membership: WIN / PLA sections take single horses only; generic all-ups are no section
    assert picks._section_pool(_c("1:PLA:box:2", "PLA", "PLA", "single", 1, 0, 0.5)) is None
    assert picks._section_pool({**_c("AU:PLA:3x4:1,2,3", "ALLUP", "PLA", "allup", 3, 0, 0.5)}) is None
    assert picks._section_pool(_c("1:FF:bank:[2, 4]", "FF", "FF", "single", 1, 0, 0.1)) == "FF"


def test_make_picks_sections_and_fixed_products():
    from hkjc import bots
    pred = meeting(8, field=10)
    pk = picks.make_picks(pred, None)
    secs = [it["section"] for it in pk["items"]]
    order = [code for code, _ in picks.SECTIONS]
    assert secs == sorted(secs, key=order.index)              # grouped, in SECTIONS order
    count = {s: secs.count(s) for s in set(secs)}
    assert count == {"PLA_6x63": 1, "QPL_2x3": 3, "WIN": 2, "PLA": 2, "QIN": 1, "QPL": 1, "FCT": 1, "TRI": 1,
                     "TCE": 1, "FF": 1, "QTT": 1, "DBL": 1, "TBL": 1}
    assert [s["section"] for s in pk["sections"]] == [c for c in order if c in count]
    six = [it for it in pk["items"] if it["section"] == "PLA_6x63"][0]
    assert six["n_combos"] == 63 and six["stake"] == 630 and six["kind"] == "allup" and six["star"] is False
    q = [it for it in pk["items"] if it["section"] == "QPL_2x3"]
    assert all(x["n_combos"] == 15 and x["stake"] == 150 for x in q)
    # identical to the fixed_combo robot's tickets
    fc = [d["ticket"] for d in bots.decide("fixed_combo", pred, None, 10_000) if d["ticket"]]
    mine = [six] + q
    assert [[(l["race_no"], l["selections"]) for l in t["legs"]] for t in fc] == \
           [[(l["race_no"], l["selections"]) for l in it["legs_raw"]] for it in mine]
    assert pk["n_items"] == len(pk["items"]) and pk["total_stake"] == pytest.approx(sum(i["stake"] for i in pk["items"]))
    for it in pk["items"]:
        assert it["grade"] == "精選" and it["section_zh"] == picks.SECTION_ZH[it["section"]]
        assert it["star"] == picks.passes({**it, "family": it["family"]}, picks._cfg())
        assert ("⭐ 值博" in it["reason"]) == it["star"]
        assert "預計命中" in it["reason"] and "估計回報" in it["reason"] and "上名率" not in it["reason"]
        if it["section"] not in picks.FIXED_SECTIONS:
            assert it["stake"] <= 300
        assert it["ticket"]["notes"]["section"] == it["section"]
    assert "入三甲機率" in six["reason"] and pk["legend"].startswith("⭐")
    assert "入三甲機率" in pk["banker"]["reason"]


def test_no_odds_nothing_priced():
    pred = meeting().drop(columns=["win_odds"])
    pk = picks.make_picks(pred, None)
    assert pk["items"] == [] and pk["n_items"] == 0
    assert "賠率未開售" in pk["odds_note"] and "今日冇可以計算嘅玩法" in pk["method_note"]
    assert pk["banker"] is not None                      # 全日最穩 needs only the model


def test_live_odds_used_and_capped_floor():
    pred = race(1, P10)
    live = {1: {"WIN": {str(h): float(o) for h, o in zip(pred.horse_no, pred.win_odds)},
                "PLA": {str(h): 1.5 for h in pred.horse_no},
                "FF": {"1,2,3,4": 40.0, "1,2,3,5": 80.0}}}
    by, ctx = _cands(pred, live)
    assert by["1:PLA:top:1"]["odds_source"] == "live"
    r = ctx["races"][1]
    assert r.div("PLA", "1") == (15.0, "live")
    assert r.div("FF", "1,2,3,4") == (400.0, "live")
    d, src = r.div("FF", "7,8,9,10")                   # not in the trimmed book: >= the book's top odds
    assert d >= 800.0 and src == "est"
    pk = picks.make_picks(pred, live, {"thresholds": OPEN_TH})
    assert pk["odds_note"].startswith("即時賠率：獨贏、位置、四連環") and "估計賠率" in pk["odds_note"]


# ----------------------------------------------------------------- multi-race pools: bets + paper
def test_settle_multi_race_rules():
    legs = [{"race_no": 1, "selections": [3, 5]}, {"race_no": 2, "selections": [7, 2]}]
    assert bets.multi_race_combos("DBL", legs) == ["3/7", "3/2", "5/7", "5/2"]
    rows = [("DBL", "3/7", 250.0), ("DBL", "3/2", 40.0)]
    r = bets.settle_multi_race("DBL", legs, rows)
    assert r["payout"] == pytest.approx(250.0 + 40.0) and r["n_winning_units"] == 2 and r["stake"] == 40
    # 'F' = any horse in that leg
    r = bets.settle_multi_race("DBL", legs, [("DBL", "5/F", 30.0)])
    assert r["payout"] == pytest.approx(60.0)                  # both 5/7 and 5/2 match
    # no rows (pool not run) -> refund
    r = bets.settle_multi_race("DBL", legs, [])
    assert r["payout"] == 40 and r["n_refunded_units"] == 4
    # withdrawn selection -> substitute = leg favourite (6 here)
    r = bets.settle_multi_race("DBL", legs, [("DBL", "6/7", 100.0)], scratched_by_leg={1: {5}}, substitutes={1: 6})
    assert r["payout"] == pytest.approx(100.0) and r["detail"][2]["effective"] == "6/7"
    t = bets.settle_multi_race("TBL", legs + [{"race_no": 3, "selections": [1]}], [("TBL", "3/7/1", 900.0)])
    assert t["n_units"] == 4 and t["payout"] == pytest.approx(900.0)
    with pytest.raises(ValueError):
        bets.multi_race_combos("TBL", legs)


@pytest.fixture()
def sdb(tmp_path):
    c = db.connect(tmp_path / "s.db")
    date, v = "2099-02-02", "ST"
    c.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)", (date, v, 3, None, "results", "x"))
    for rn in (1, 2, 3):
        rid = f"20990202-ST-{rn:02d}"
        c.execute("INSERT INTO races (race_id, date, venue, race_no) VALUES (?,?,?,?)", (rid, date, v, rn))
        for h in range(1, 9):
            c.execute("INSERT INTO runners (race_id, horse_no, finish_pos, win_odds, scratched) VALUES (?,?,?,?,?)",
                      (rid, h, h, 2.0 + h, 0))
    rows = [("20990202-ST-01", "WIN", "1", 30.0), ("20990202-ST-01", "FCT", "1,2", 90.0),
            ("20990202-ST-01", "TCE", "1,2,3", 400.0), ("20990202-ST-01", "QTT", "1,2,3,4", 2000.0),
            ("20990202-ST-01", "FF", "1,2,3,4", 300.0),
            ("20990202-ST-02", "WIN", "1", 30.0), ("20990202-ST-02", "DBL", "1/1", 80.0),
            ("20990202-ST-02", "DBL", "1/2", 20.0),
            ("20990202-ST-03", "WIN", "1", 30.0), ("20990202-ST-03", "TBL", "1/1/1", 260.0),
            ("20990202-ST-03", "TBL", "1/1/2", 50.0)]
    c.executemany("INSERT INTO dividends VALUES (?,?,?,?)", rows)
    c.commit()
    yield c
    c.close()


def test_paper_settles_new_pools(sdb):
    D = "2099-02-02"
    T = [
        {"strategy": "daily_picks:FCT", "pool": "FCT", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "FCT", "positions": [[1], [2, 3]]}]},                         # 2 units
        {"strategy": "daily_picks:TCE", "pool": "TCE", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "TCE", "bankers": [1], "selections": [2, 3], "mode": "multiple"}]},  # 6
        {"strategy": "daily_picks:QTT", "pool": "QTT", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "QTT", "positions": [[1], [2], [3, 4], [3, 4]]}]},             # 2
        {"strategy": "daily_picks:FF", "pool": "FF", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "FF", "bankers": [1, 2], "selections": [3, 4, 5]}]},           # 3
        {"strategy": "daily_picks:DBL", "pool": "DBL", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "DBL", "selections": [1]},
                  {"race_no": 2, "pool": "DBL", "selections": [1, 2, 3]}]},                             # 3
        {"strategy": "daily_picks:TBL", "pool": "TBL", "spec": None, "unit_stake": 10.0,
         "legs": [{"race_no": 1, "pool": "TBL", "selections": [1, 4]}, {"race_no": 2, "pool": "TBL", "selections": [1]},
                  {"race_no": 3, "pool": "TBL", "selections": [2]}]},                                   # 2
    ]
    ids = paper.place_tickets(sdb, "r1", D, "ST", T, bot_id="daily_picks")
    stakes = [r[0] for r in sdb.execute("SELECT total_stake FROM paper_tickets ORDER BY ticket_id")]
    assert stakes == [20.0, 60.0, 20.0, 30.0, 30.0, 20.0]
    assert paper.settle(sdb, D) == 6
    pay = [r[0] for r in sdb.execute("SELECT payout FROM paper_tickets ORDER BY ticket_id")]
    assert pay == pytest.approx([90.0, 400.0, 2000.0, 300.0, 80.0 + 20.0, 50.0])
    assert paper.bankroll(sdb, "daily_picks") == pytest.approx(10_000 - 180 + sum(pay))
    assert len(ids) == 6 and paper.settle(sdb, D) == 0


def test_settle_item_db_matches_paper(sdb):
    item = {"kind": "multi_race", "pool": "DBL", "unit_stake": 10.0, "legs_raw": [
        {"race_no": 1, "pool": "DBL", "selections": [1]}, {"race_no": 2, "pool": "DBL", "selections": [2]}]}
    r = picks.settle_item(item, picks.DBData(sdb, "2099-02-02", "ST"))
    assert r["payout"] == 20.0 and r["hit"] and r["profit"] == 10.0


# ----------------------------------------------------------------- backtest / live parity (real DB, read-only)
@pytest.fixture(scope="module")
def ro():
    if not REAL_DB.exists():
        pytest.skip("no data/hkjc.db")
    c = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    if not c.execute("SELECT 1 FROM predictions WHERE run_id='wf-combined-v1' LIMIT 1").fetchone():
        pytest.skip("no walk-forward predictions")
    yield c
    c.close()


def test_backtest_uses_live_path(ro, tmp_path):
    data = picks.load_backtest_data(ro, "wf-combined-v1", "2025-07-01", "2025-09-20")
    assert {m["season"] for m in data[0]} == {"2024-25", "2025-26"}
    rep = tmp_path / "r.md"
    rep.write_text("# 報告\n\n## 其他\nx\n", encoding="utf-8")
    js = picks.backtest_picks(ro, data=data, tune=False, path=tmp_path / "pb.json", report_path=rep)
    assert js["parity"] is not None and js["parity"]["n_candidates"] > 100      # asserted inside: identical items
    assert js["thresholds"] == picks.THRESHOLDS
    txt = rep.read_text(encoding="utf-8")
    assert txt.count(picks.REPORT_HEADING) == 1 and "## 其他" in txt
    picks.write_report_section(js, rep)                                          # idempotent replace
    assert rep.read_text(encoding="utf-8").count(picks.REPORT_HEADING) == 1
    o = js["oos"]["overall"]
    assert o["n"] == sum(v["n"] for v in js["oos"]["by_section"].values())
    assert js["section_min_hit"] == picks.SECTION_MIN_HIT
    assert "每種玩法各一個" in txt


# ----------------------------------------------------------------- exotic odds snapshot (mocked API)
def test_exotic_snapshot_cap_and_dbl_last_leg(tmp_path, monkeypatch):
    from hkjc import odds
    assert odds.normalise_comb("DBL", "07/13") == "7/13"
    assert odds.cap_book({"1,2": 5.0, "1,3": 999.0, "2,3": None, "3,4": 7.0}, cap=1) == {"1,2": 5.0}

    def fake(date, venue, rno, pools, session=None):
        po = odds.PoolOdds()
        po["FF"] = {"1,2,3,4": 50.0, "1,2,3,5": 999.0}
        po.meta["FF"] = {"leg_races": [rno]}
        if rno < 3:
            po["DBL"] = {f"{rno}/1": 12.0}
            po.meta["DBL"] = {"leg_races": [rno, rno + 1]}
        return po
    monkeypatch.setattr(odds, "fetch_pools", fake)
    c = db.connect(tmp_path / "o.db")
    n = odds.snapshot_meeting(c, "2099-03-03", "HV", pools=odds.EXOTIC_POOLS, race_nos=[1, 2, 3], snapshot_at="T")
    rows = c.execute("SELECT race_no, pool, combination, odds FROM odds_snapshots ORDER BY pool, race_no").fetchall()
    assert n == 5 and [tuple(r) for r in rows] == [
        (2, "DBL", "1/1", 12.0), (3, "DBL", "2/1", 12.0),
        (1, "FF", "1,2,3,4", 50.0), (2, "FF", "1,2,3,4", 50.0), (3, "FF", "1,2,3,4", 50.0)]
    assert odds.latest_odds(c, "2099-03-03", "HV")[3]["DBL"] == {"2/1": 12.0}


def test_void_settled_reverses_ledger(sdb):
    D = "2099-02-02"
    T = [{"strategy": "daily_picks:FCT", "pool": "FCT", "spec": None, "unit_stake": 10.0,
          "legs": [{"race_no": 1, "pool": "FCT", "positions": [[1], [2, 3]]}]}]
    paper.place_tickets(sdb, "r1", D, "ST", T, bot_id="daily_picks")
    paper.settle(sdb, D)
    assert paper.bankroll(sdb, "daily_picks") == pytest.approx(10_000 - 20 + 90)
    ids = paper.void_settled(sdb, D, "ST", "daily_picks", "重做")
    assert len(ids) == 1 and paper.bankroll(sdb, "daily_picks") == pytest.approx(10_000)
    st_ = sdb.execute("SELECT status, payout, profit FROM paper_tickets WHERE ticket_id=?", (ids[0],)).fetchone()
    assert tuple(st_) == ("void", 20.0, 0)
    assert paper.void_settled(sdb, D, "ST", "daily_picks") == []
