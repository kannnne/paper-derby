"""Tests for hkjc.backtest.  Hand-derived numbers in comments."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hkjc import backtest as bt, bets

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "data" / "hkjc.db"


def test_season_of():
    assert bt.season_of("2024-09-08") == "2024-25"
    assert bt.season_of("2025-07-16") == "2024-25"
    assert bt.season_of("2026-10-19") == "2026-27"
    assert bt.season_of("2024-01-01") == "2023-24"


def test_drawdown_and_streak():
    # cumulative: -10, -20, +30, +20, -40  -> peak 30, trough -40 -> MDD 70
    p = np.array([-10, -10, 50, -10, -60.0])
    assert bt._max_drawdown(p) == 70.0
    assert bt._longest_losing(p) == 2
    # starts from 0: losing from the start counts: -5,-5 -> MDD 10
    assert bt._max_drawdown(np.array([-5.0, -5.0])) == 10.0
    assert bt._longest_losing(np.array([1.0, -1, -1, -1, 2, -1])) == 3


def test_settle_combos_equals_bets_settle_single():
    rows = [("TRI", "7,8,9", 305.0), ("WIN", "8", 54.0), ("QPL", "7,8", 58.0),
            ("QPL", "7,9", 66.0), ("QPL", "8,9", 66.0), ("PLA", "7", 20.5)]
    idx = bets.index_dividends(rows)
    for pool, combos, scr in [("TRI", ["5,7,8", "7,8,9", "1,7,9"], set()),
                              ("QPL", ["7,8", "8,9", "1,2", "4,7"], {4}),
                              ("WIN", ["8"], set())]:
        ref = bets.settle_single(pool, combos, rows, 10.0, scr)
        stake, pay, nwin = bt.settle_combos(pool, combos, idx, scr)
        assert (stake, round(pay, 6), nwin) == (ref["stake"], ref["payout"], ref["n_winning_units"])
    # QPL: 7,8 pays 58 + 8,9 pays 66 + 1,2 loses + 4,7 refunded (4 scratched) 10 = 134
    assert bt.settle_combos("QPL", ["7,8", "8,9", "1,2", "4,7"], idx, {4})[1] == 134.0


def test_summarise_toy():
    S = [{"sid": 0, "name": "A", "family": "單場"}, {"sid": 1, "name": "B", "family": "單場"}]
    mt = pd.DataFrame({"mi": [0, 1, 2], "date": ["2024-09-08", "2024-09-11", "2025-09-07"],
                       "venue": ["ST", "HV", "ST"], "season": ["2024-25", "2024-25", "2025-26"],
                       "n_races": [10, 9, 10]})
    b = pd.DataFrame({"sid": [0, 0, 0, 0], "mi": [0, 0, 1, 2],
                      "date": ["2024-09-08", "2024-09-08", "2024-09-11", "2025-09-07"],
                      "venue": ["ST", "ST", "HV", "ST"], "season": ["2024-25", "2024-25", "2024-25", "2025-26"],
                      "surface": ["TURF"] * 4, "stake": [10.0, 10, 10, 10], "payout": [0.0, 25, 0, 0],
                      "hit": [False, True, False, False]})
    sm = bt.summarise(b, S, mt, n_boot=200).set_index("strategy")
    a = sm.loc["A"]
    # stake 40, return 25 -> ROI -37.5%; hits 1/4; profits -10,+15,-10,-10 -> cum -10,5,-5,-15 -> MDD 20
    assert a["roi"] == pytest.approx(-0.375) and a["hit_rate"] == 0.25 and a["n_bets"] == 4
    assert a["max_dd"] == 20.0 and a["lose_streak"] == 2
    assert a["by_venue"]["ST"]["roi"] == pytest.approx(25 / 30 - 1, abs=1e-4)
    assert a["by_season"]["2025-26"]["n"] == 1
    assert a["ci_lo"] <= a["roi"] <= a["ci_hi"]
    assert sm.loc["B"]["n_bets"] == 0
    oos = bt.summarise(b, S, mt, seasons=("2025-26",), n_boot=50).set_index("strategy")
    assert oos.loc["A"]["roi"] == -1.0


def test_default_grid_size():
    m = bt.default_strategies("market")
    w = bt.default_strategies("wf-combined-v1")
    assert len(m) >= 90 and len(w) > len(m)
    fams = {s["family"] for s in w}
    assert fams == {"固定組合", "過關結構", "單場", "篩選"}


# ------------------------------------------------------------------ real-data checks
def _real_conn():
    if not REAL_DB.exists():
        pytest.skip("no data/hkjc.db")
    return sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)


def test_real_meeting_robot_tickets():
    """2024-05-22 HV, market baseline.  Favourites (final odds) by race:
    R2 #10 2.7, R3 #5 2.8, R4 #10 1.9, R6 #1 3.1, R7 #1 2.2, R9 #7 4.1, R1 #4/#5 4.3,
    R8 #1 4.6, R5 #4/#8 5.1 -> PLA 6x63 'confidence' = the six shortest favourites
    R2 R3 R4 R6 R7 R9; all six placed -> payout 1423.91 (see test_paper.py)."""
    conn = _real_conn()
    md = bt.load_data(conn, "market", "2024-05-22", "2024-05-22")
    if not md:
        pytest.skip("meeting not loaded yet")
    S = [s for s in bt.default_strategies("market") if s["name"] in
         ("PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]", "WIN top1")]
    res = bt.run_grid(conn, "market", strategies=S, meetings=md)
    b = res["bets"]
    pla = b[b["sid"] == S[0]["sid"]].iloc[0]
    assert pla["stake"] == 630.0 and pla["payout"] == pytest.approx(1423.91475425)
    qpl = b[b["sid"] == S[1]["sid"]].iloc[0]
    assert qpl["stake"] == 450.0 and qpl["payout"] == pytest.approx(58.0 + 41.0 + 285.8)
    win = b[b["sid"] == S[2]["sid"]]
    # favourites won R2 (#10 27.5) and R4 (#10 19.5) only:
    # R1 fav #4 (tie with #5 at 4.3, lower number) 2nd; R3 #5 2nd; R5 #4 11th;
    # R6 #1 2nd; R7 #1 2nd; R8 #1 2nd; R9 #7 2nd  -> return 47.0 on 90
    assert len(win) == 9 and win["stake"].sum() == 90.0
    assert win["payout"].sum() == pytest.approx(27.5 + 19.5)


def test_market_sanity_big_sample():
    """Market favourite: win rate ~30-35%, WIN ROI roughly -10%..-25% (takeout 17.5%)."""
    conn = _real_conn()
    md = bt.load_data(conn, "market")
    if len(md) < 40:
        pytest.skip("not enough meetings yet")
    s = bt.sanity(md)
    assert 0.27 <= s["fav_win_rate"] <= 0.38
    assert -0.30 <= s["fav_roi"] <= -0.05
    assert 1.18 <= s["avg_overround"] <= 1.26          # 1/(1-0.175) = 1.212
    assert s["win_div_matches_odds"] > 0.97


def test_main_writes_results_on_copy(tmp_path):
    """Full pipeline (small window) on a scratch copy: rows written + report produced."""
    conn = _real_conn()
    dst = sqlite3.connect(tmp_path / "c.db")
    conn.backup(dst)
    dst.close()
    from hkjc import db
    c = db.connect(tmp_path / "c.db")
    out = bt.main(c, "2024-05-01", "2024-05-31", source="market", report_path=tmp_path / "r.md")
    q = "SELECT COUNT(*) FROM backtest_results WHERE json_extract(params,'$.source')='market'"
    n = c.execute(q).fetchone()[0]
    # one row per strategy (fixed grid + adaptive settings) per period (all / IS / OOS)
    n_strat = len(out["results"]["market"]["res"]["strategies"])
    assert n_strat > len(bt.default_strategies("market"))
    assert n == 3 * n_strat
    # rerun replaces, doesn't duplicate
    bt.main(c, "2024-05-01", "2024-05-31", source="market", report_path=tmp_path / "r.md")
    assert c.execute(q).fetchone()[0] == n
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "回測報告" in md and "結論" in md and "最終獨贏賠率" in md
    assert out["sources"] == ["market"]


def test_adaptive_fast_path_matches_decide_race():
    """The backtest's precomputed candidate path must choose exactly what the
    robot's strategy.decide_race chooses, and settle it identically."""
    from hkjc import strategy as st
    conn = _real_conn()
    if not conn.execute("SELECT 1 FROM predictions WHERE run_id='wf-combined-v1' LIMIT 1").fetchone():
        pytest.skip("no wf-combined-v1 predictions")
    md = bt.load_data(conn, "wf-combined-v1", "2024-10-01", "2024-11-15")
    if len(md) < 5:
        pytest.skip("not enough meetings")
    l2, l3, _ = bt.fit_lambdas(md, seasons=("2024-25",))
    cal, _ = bt.fit_calibration(md, ("2024-25",), l2, l3)
    pre = bt.adaptive_precompute(md, l2, l3, cal)
    recs = bt.adaptive_records(pre, md, tau=0.0, h_min=0.15)
    cfg = {"tau": 0.0, "h_min": 0.15, "lam2": l2, "lam3": l3, "calib": cal, "max_stake": 200.0}
    n_checked = 0
    for mi, m in enumerate(md):
        for rn, r in m["races"].items():
            t = st.decide_race(r["df"], None, cfg)
            got = recs[(recs["mi"] == mi) & (recs["race_no"] == rn)] if len(recs) else recs
            if t and bt.pool_live(r["idx"], t[0]["pool"]):
                assert len(got) == 1
                g = got.iloc[0]
                assert g["structure"] == t[0]["notes"]["structure"]
                stake, pay, _ = bt.settle_combos(t[0]["pool"], st.leg_combos(t[0]["legs"][0]), r["idx"], r["scr"])
                assert (g["stake"], g["payout"]) == pytest.approx((stake, pay))
                n_checked += 1
            elif not t:
                assert len(got) == 0
    assert n_checked > 0


def test_fit_lambdas_market_is_plausible():
    conn = _real_conn()
    md = bt.load_data(conn, "market", "2024-09-01", "2025-07-31")
    if len(md) < 40:
        pytest.skip("not enough meetings")
    l2, l3, n = bt.fit_lambdas(md)
    # literature (Lo & Bacon-Shone; Benter): discount < 1 for lower places, l3 <= l2
    assert 0.6 <= l3 <= l2 <= 1.1 and n > 500
