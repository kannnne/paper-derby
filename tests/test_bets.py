"""Tests for hkjc.bets - every expected number below is derived by hand in the comments."""
import re
from itertools import combinations
from math import comb, perm
from pathlib import Path

import pytest

from hkjc import bets

FIX = Path(__file__).parent / "fixtures"

# Independent transcription of the HKJC "All Up Combinations" table
# https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/all-up-betting/
# row -> (#1-leg, #2-leg, #3-leg, #4-leg, #5-leg, #6-leg) combinations
HKJC_TABLE = {
    "2x1": (0, 1, 0, 0, 0, 0), "2x3": (2, 1, 0, 0, 0, 0),
    "3x1": (0, 0, 1, 0, 0, 0), "3x3": (0, 3, 0, 0, 0, 0), "3x4": (0, 3, 1, 0, 0, 0),
    "3x6": (3, 3, 0, 0, 0, 0), "3x7": (3, 3, 1, 0, 0, 0),
    "4x1": (0, 0, 0, 1, 0, 0), "4x4": (0, 0, 4, 0, 0, 0), "4x5": (0, 0, 4, 1, 0, 0),
    "4x6": (0, 6, 0, 0, 0, 0), "4x10": (4, 6, 0, 0, 0, 0), "4x11": (0, 6, 4, 1, 0, 0),
    "4x14": (4, 6, 4, 0, 0, 0), "4x15": (4, 6, 4, 1, 0, 0),
    "5x1": (0, 0, 0, 0, 1, 0), "5x5": (0, 0, 0, 5, 0, 0), "5x6": (0, 0, 0, 5, 1, 0),
    "5x10": (0, 10, 0, 0, 0, 0), "5x15": (5, 10, 0, 0, 0, 0), "5x16": (0, 0, 10, 5, 1, 0),
    "5x20": (0, 10, 10, 0, 0, 0), "5x25": (5, 10, 10, 0, 0, 0), "5x26": (0, 10, 10, 5, 1, 0),
    "5x30": (5, 10, 10, 5, 0, 0), "5x31": (5, 10, 10, 5, 1, 0),
    "6x1": (0, 0, 0, 0, 0, 1), "6x6": (0, 0, 0, 0, 6, 0), "6x7": (0, 0, 0, 0, 6, 1),
    "6x15": (0, 15, 0, 0, 0, 0), "6x20": (0, 0, 20, 0, 0, 0), "6x21": (6, 15, 0, 0, 0, 0),
    "6x22": (0, 0, 0, 15, 6, 1), "6x35": (0, 15, 20, 0, 0, 0), "6x41": (6, 15, 20, 0, 0, 0),
    "6x42": (0, 0, 20, 15, 6, 1), "6x50": (0, 15, 20, 15, 0, 0), "6x56": (6, 15, 20, 15, 0, 0),
    "6x57": (0, 15, 20, 15, 6, 1), "6x62": (6, 15, 20, 15, 6, 0), "6x63": (6, 15, 20, 15, 6, 1),
}


# ------------------------------------------------------------------ all-up formulas
def test_formula_table_matches_hkjc_exactly():
    assert set(bets.ALLUP_FORMULAS) == set(HKJC_TABLE)
    for spec, cols in HKJC_TABLE.items():
        n = int(spec.split("x")[0])
        subs = bets.allup_subsets(n, spec)
        by_size = tuple(sum(1 for s in subs if len(s) == k) for k in range(1, 7))
        assert by_size == cols, spec
        m = int(spec.split("x")[1])
        assert len(subs) == m == bets.allup_count(spec) == sum(cols), spec
        assert len(set(subs)) == len(subs)                       # no duplicates
        assert all(list(s) == sorted(s) and all(0 <= i < n for i in s) for s in subs)


def test_unofficial_7_8_leg_formulas_counts():
    for spec, sizes in bets.ALLUP_FORMULAS_FOOTBALL_STYLE.items():
        n, m = map(int, spec.split("x"))
        assert sum(comb(n, k) for k in sizes) == m, spec
        assert len(bets.allup_subsets(n, spec)) == m


def test_specific_subsets():
    assert bets.allup_subsets(2, "2x3") == [(0,), (1,), (0, 1)]
    assert bets.allup_subsets(2, "2x1") == [(0, 1)]
    assert bets.allup_combos(3, "3x4") == [(0, 1), (0, 2), (1, 2), (0, 1, 2)]
    # 6x63 = 6+15+20+15+6+1 = 2^6 - 1
    assert len(bets.allup_subsets(6, "6x63")) == 63 == 2 ** 6 - 1
    assert bets.allup_specs(2) == ["2x1", "2x3"]
    assert len(bets.allup_specs(6)) == 15
    with pytest.raises(ValueError):
        bets.allup_subsets(5, "6x63")
    with pytest.raises(ValueError):
        bets.allup_formula("6x64")


# ------------------------------------------------------------------ generators vs HKJC chance table
# https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/chance-table/
def test_canonical_format():
    assert bets.canonical("QPL", "12,1") == "1,12"
    assert bets.canonical("QIN", "01,07") == "1,7"
    assert bets.canonical("TCE", "1,12,11") == "1,12,11"     # ordered pool keeps order
    assert bets.canonical("TRI", (12, 11, 1)) == "1,11,12"
    assert bets.canonical("PLA", 7) == "7"
    with pytest.raises(ValueError):
        bets.canonical("QPL", "1,1")


def test_multiple_counts():
    for n in range(2, 15):
        hs = list(range(1, n + 1))
        assert len(bets.multiple("QIN", hs)) == comb(n, 2)
        assert len(bets.multiple("QPL", hs)) == comb(n, 2)
        assert len(bets.multiple("FCT", hs)) == n * (n - 1)
        assert len(bets.multiple("TRI", hs)) == comb(n, 3)
        assert len(bets.multiple("TCE", hs)) == perm(n, 3)
        assert len(bets.multiple("FF", hs)) == comb(n, 4)
        assert len(bets.multiple("QTT", hs)) == perm(n, 4)
    # chance-table spot values
    assert len(bets.multiple("QIN", range(1, 15))) == 91
    assert len(bets.multiple("TRI", range(1, 15))) == 364
    assert len(bets.multiple("FF", range(1, 9))) == 70
    assert len(bets.multiple("TCE", range(1, 8))) == 210
    assert len(bets.multiple("QTT", range(1, 8))) == 840
    assert bets.multiple("QPL", [12, 1, 11]) == ["1,11", "1,12", "11,12"]


def test_banker_counts_match_chance_table():
    legs = lambda n: list(range(10, 10 + n))  # noqa: E731
    for n in range(2, 8):
        assert len(bets.banker("QPL", [1], legs(n))) == n               # Q/QP 1 banker
        assert len(bets.banker("TRI", [1], legs(n))) == comb(n, 2)      # Trio 1 banker
        assert len(bets.banker("TRI", [1, 2], legs(n))) == n            # Trio 2 bankers
        assert len(bets.banker("FF", [1], legs(n))) == comb(n, 3)
        assert len(bets.banker("FF", [1, 2], legs(n))) == comb(n, 2)
        assert len(bets.banker("FF", [1, 2, 3], legs(n))) == n
        assert len(bets.banker("FCT", [1], legs(n), mode="single")) == n
        assert len(bets.banker("FCT", [1], legs(n), mode="multiple")) == 2 * n
        assert len(bets.banker("TCE", [1], legs(n), mode="single")) == n * (n - 1)
        assert len(bets.banker("TCE", [1], legs(n), mode="multiple")) == 3 * n * (n - 1)
        assert len(bets.banker("TCE", [1, 2], legs(n), mode="single")) == n
        assert len(bets.banker("TCE", [1, 2], legs(n), mode="multiple")) == 6 * n
        assert len(bets.banker("QTT", [1], legs(n), mode="single")) == perm(n, 3)
        assert len(bets.banker("QTT", [1], legs(n), mode="multiple")) == 4 * perm(n, 3)
        assert len(bets.banker("QTT", [1, 2], legs(n), mode="single")) == perm(n, 2)
        assert len(bets.banker("QTT", [1, 2], legs(n), mode="multiple")) == 12 * perm(n, 2)
        assert len(bets.banker("QTT", [1, 2, 3], legs(n), mode="single")) == n
        assert len(bets.banker("QTT", [1, 2, 3], legs(n), mode="multiple")) == 24 * n
    # chance table spot checks: Quartet 2 bankers 4 legs multiple = 144; Tierce 2 bankers 7 legs multiple = 42
    assert len(bets.banker("QTT", [1, 2], [3, 4, 5, 6], mode="multiple")) == 144
    assert len(bets.banker("TCE", [1, 2], range(3, 10), mode="multiple")) == 42
    # every combination contains the bankers
    assert all("1" in c.split(",") for c in bets.banker("TCE", [1], [2, 3, 4]))
    assert bets.banker("QPL", [5], [1, 9]) == ["1,5", "5,9"]
    assert bets.banker("FCT", [5], [1, 9], mode="single") == ["5,1", "5,9"]


def test_positional_tce():
    # 1 first; 2/3 second; 2/3/4 third -> (1,2,3),(1,2,4),(1,3,2),(1,3,4)
    assert bets.positional("TCE", [[1], [2, 3], [2, 3, 4]]) == ["1,2,3", "1,2,4", "1,3,2", "1,3,4"]
    # banker at 2nd position with legs 1..4 elsewhere: 4*3 = 12
    assert len(bets.positional("TCE", [[1, 2, 3, 4], [9], [1, 2, 3, 4]])) == 12
    assert bets.count_units("TCE", positions=[[1], [2, 3], [2, 3, 4]]) == 4
    with pytest.raises(ValueError):
        bets.positional("QPL", [[1], [2]])


# ------------------------------------------------------------------ ticket units / cost
def test_ticket_units_spec_examples():
    pla_legs = [{"race_no": r, "selections": [1]} for r in range(1, 7)]
    assert bets.ticket_units("PLA", pla_legs, "6x63") == 63
    assert bets.ticket_cost("PLA", pla_legs, "6x63", 10) == 630
    qpl_legs = [{"race_no": 1, "selections": [1, 11, 12]}, {"race_no": 2, "selections": [2, 5, 8]}]
    # 2x3 = 2 singles + 1 double: 3 + 3 + 3*3 = 15
    assert bets.ticket_units("QPL", qpl_legs, "2x3") == 15
    assert bets.ticket_cost("QPL", qpl_legs, "2x3", unit_stake=10) == 150
    # bare-list legs work too; PLA with 2 horses per leg, 3x7: 3*2 + 3*4 + 8 = 26
    assert bets.ticket_units("PLA", [[1, 2], [3, 4], [5, 6]], "3x7") == 26
    # QPL banker leg: 1 banker + 3 legs = 3 pairs; 2x1 with other leg 4 horses (6 pairs) = 18
    assert bets.ticket_units("QPL", [{"bankers": [1], "selections": [2, 3, 4]}, [1, 2, 3, 4]], "2x1") == 18
    assert bets.leg_combos("QPL", [1, 11, 12]) == ["1,11", "1,12", "11,12"]


# ------------------------------------------------------------------ single settlement
R1_HV_20261014 = [  # from tests/fixtures/results_20261014_HV_R1.html dividend table
    {"pool": "WIN", "combination": "1", "dividend": 42.0},
    {"pool": "PLA", "combination": "1", "dividend": 17.5},
    {"pool": "PLA", "combination": "12", "dividend": 28.0},
    {"pool": "PLA", "combination": "11", "dividend": 14.0},
    {"pool": "QIN", "combination": "1,12", "dividend": 206.5},
    {"pool": "QPL", "combination": "1,12", "dividend": 73.0},
    {"pool": "QPL", "combination": "1,11", "dividend": 37.0},
    {"pool": "QPL", "combination": "11,12", "dividend": 59.0},
    {"pool": "FCT", "combination": "1,12", "dividend": 394.0},
    {"pool": "TCE", "combination": "1,12,11", "dividend": 1602.0},
    {"pool": "TRI", "combination": "1,11,12", "dividend": 246.0},
    {"pool": "FF", "combination": "1,7,11,12", "dividend": 1739.0},
    {"pool": "QTT", "combination": "1,12,11,7", "dividend": 23259.0},
]


def test_fixture_contains_the_dividends_used():
    html = (FIX / "results_20261014_HV_R1.html").read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", html)
    for s in ["73.00", "37.00", "59.00", "17.50", "206.50", "1,602.00"]:
        assert s in text


def test_settle_single_qpl_box():
    # QPL box 1,11,12 -> all 3 pairs win: 73 + 37 + 59 = 169 per $10 units
    r = bets.settle_single("QPL", bets.multiple("QPL", [1, 11, 12]), R1_HV_20261014, 10)
    assert r["payout"] == pytest.approx(169.0)
    assert r["stake"] == 30 and r["n_winning_units"] == 3 and r["profit"] == pytest.approx(139.0)
    # $20 unit doubles the return
    r = bets.settle_single("QPL", ["12,1"], R1_HV_20261014, 20)
    assert r["payout"] == pytest.approx(146.0)


def test_settle_single_ordered_and_losers():
    r = bets.settle_single("TCE", bets.banker("TCE", [1], [11, 12], mode="multiple"), R1_HV_20261014, 10)
    # 3*2*1 = 6 permutations, only 1,12,11 wins 1602
    assert r["n_units"] == 6 and r["payout"] == pytest.approx(1602.0)
    r = bets.settle_single("QIN", ["1,11"], R1_HV_20261014)
    assert r["payout"] == 0 and r["detail"][0]["status"] == "lose"
    # dividends given as {pool: {comb: div}} mapping also accepted
    r = bets.settle_single("WIN", ["1"], {"WIN": {"1": 42.0}}, 5)
    assert r["payout"] == pytest.approx(21.0)
    with pytest.raises(ValueError):
        bets.settle_single("WIN", ["1"], [])


def test_settle_single_dead_heat():
    # dead-heat for 1st between 3 and 7 -> two WIN rows (25.0 and 41.5).
    # QPL with 3 and 7 first/second and a dead-heat for 3rd between 5 and 9 ->
    # five winning pairs 3,7 3,5 3,9 5,7 7,9 (one row each; 5,9 does NOT win).
    rows = [
        {"pool": "WIN", "combination": "3", "dividend": 25.0},
        {"pool": "WIN", "combination": "7", "dividend": 41.5},
        {"pool": "QPL", "combination": "3,7", "dividend": 30.0},
        {"pool": "QPL", "combination": "3,5", "dividend": 40.0},
        {"pool": "QPL", "combination": "3,9", "dividend": 55.0},
        {"pool": "QPL", "combination": "5,7", "dividend": 60.0},
        {"pool": "QPL", "combination": "7,9", "dividend": 80.0},
    ]
    r = bets.settle_single("WIN", ["3", "7", "1"], rows, 10)
    assert r["payout"] == pytest.approx(66.5) and r["n_winning_units"] == 2
    # QPL box 3,5,9: pairs 3,5 (40) 3,9 (55) 5,9 (lose: both dead-heated 3rd) -> 95
    r = bets.settle_single("QPL", bets.multiple("QPL", [3, 5, 9]), rows, 10)
    assert r["payout"] == pytest.approx(95.0) and r["n_winning_units"] == 2


def test_settle_single_scratched_refund():
    # horse 4 scratched; QPL box 1,4,11 -> 1,4 and 4,11 refunded ($10 each), 1,11 wins 37
    r = bets.settle_single("QPL", bets.multiple("QPL", [1, 4, 11]), R1_HV_20261014, 10, scratched={4})
    assert r["payout"] == pytest.approx(57.0)
    assert r["n_refunded_units"] == 2 and r["n_winning_units"] == 1
    # pool not offered (no QTT rows) -> refund
    rows = [x for x in R1_HV_20261014 if x["pool"] != "QTT"]
    r = bets.settle_single("QTT", ["1,12,11,7"], rows, 10)
    assert r["payout"] == 10 and r["detail"][0]["status"] == "pool_refund"


# ------------------------------------------------------------------ all-up settlement
def _pla_rows(placed):
    return [{"pool": "WIN", "combination": str(placed[0][0]), "dividend": 50.0}] + [
        {"pool": "PLA", "combination": str(h), "dividend": d} for h, d in placed]


def test_place_6x63_four_of_six_legs():
    """6 legs, 1 horse each; legs 1-4 place paying 17.5, 22.5, 13.5, 60.0 per $10;
    legs 5,6 lose.

    Only subsets made entirely of the 4 winning legs pay.  With f_i = d_i/10 =
    1.75, 2.25, 1.35, 6.00 the sum over non-empty subsets S of Π_{i∈S} f_i is
        Π(1+f_i) - 1 = 2.75 × 3.25 × 2.35 × 7.00 - 1
                     = 8.9375 × 2.35 × 7 - 1 = 21.003125 × 7 - 1 = 147.021875 - 1
                     = 146.021875
    × $10 unit  => payout $1,460.21875;  winning units = 2^4 - 1 = 15;
    stake = 63 × $10 = $630.
    Cross-check by explicit sum: singles 10×(1.75+2.25+1.35+6) = 113.5;
    doubles 10×(3.9375+2.3625+10.5+3.0375+13.5+8.1) = 414.375;
    trebles 10×(5.315625+23.625+14.175+18.225) = 613.40625;
    four-fold 10×(1.75×2.25×1.35×6) = 318.9375;  total = 1460.21875 ✓
    """
    sel = [3, 5, 7, 9, 11, 2]
    divs = [17.5, 22.5, 13.5, 60.0]
    by_race = {}
    for i, r in enumerate(range(1, 7)):
        if i < 4:
            by_race[r] = _pla_rows([(sel[i], divs[i]), (sel[i] + 1, 20.0), (sel[i] + 20, 30.0)])
        else:
            by_race[r] = _pla_rows([(13, 15.0), (14, 25.0), (12, 35.0)])
    legs = [{"race_no": r, "selections": [sel[r - 1]]} for r in range(1, 7)]
    res = bets.settle_allup("PLA", legs, "6x63", by_race, unit_stake=10)
    assert res["n_units"] == 63 and res["stake"] == 630
    assert res["payout"] == pytest.approx(1460.21875)
    assert res["n_winning_units"] == 15
    assert res["profit"] == pytest.approx(830.21875)
    # explicit brute-force re-derivation
    f = [d / 10 for d in divs]
    brute = sum(10 * __import__("math").prod(f[i] for i in s)
                for k in range(1, 5) for s in combinations(range(4), k))
    assert brute == pytest.approx(1460.21875)


def test_qpl_2x3_with_fixture_numbers():
    """Leg 1 = 2026-10-14 HV R1, QPL 1,11,12 (all three pairs pay: 73, 37, 59).
    Leg 2 = hypothetical race, QPL 2,5,8 where only 2,5 pays 25.50.

    singles leg1: 10 × (7.3+3.7+5.9) = 169.00
    singles leg2: 10 × 2.55           =  25.50
    doubles: Σ_a Σ_b 10 × f_a × f_b = 10 × 16.9 × 2.55 = 430.95
    total = 625.45 ; stake = 15 units × $10 = 150 ; winning units = 3 + 1 + 3 = 7
    """
    leg2 = [{"pool": "PLA", "combination": "2", "dividend": 15.0},
            {"pool": "QPL", "combination": "2,5", "dividend": 25.5},
            {"pool": "QPL", "combination": "2,6", "dividend": 40.0},
            {"pool": "QPL", "combination": "5,6", "dividend": 50.0}]
    legs = [{"race_no": 1, "selections": [1, 11, 12]}, {"race_no": 2, "selections": [2, 5, 8]}]
    res = bets.settle_allup("QPL", legs, "2x3", {1: R1_HV_20261014, 2: leg2}, 10)
    assert res["n_units"] == 15 and res["stake"] == 150
    assert res["payout"] == pytest.approx(625.45)
    assert res["n_winning_units"] == 7
    singles = [d for d in res["detail"] if len(d["legs"]) == 1]
    assert sum(d["return"] for d in singles) == pytest.approx(194.5)
    # same ticket as 2x1 (double only) = 430.95 on 9 units
    res21 = bets.settle_allup("QPL", legs, "2x1", {1: R1_HV_20261014, 2: leg2}, 10)
    assert res21["n_units"] == 9 and res21["payout"] == pytest.approx(430.95)


def test_allup_dead_heat_leg():
    """PLA 2x1: leg 1 horse 5 dead-heats for 3rd (PLA rows for 5 and 9 both exist,
    5 pays 12.0); leg 2 horse 2 places 18.0.  Return = 10 × 1.2 × 1.8 = 21.6."""
    r1 = [{"pool": "PLA", "combination": h, "dividend": d}
          for h, d in [("1", 11.0), ("3", 15.0), ("5", 12.0), ("9", 14.0)]]
    r2 = [{"pool": "PLA", "combination": h, "dividend": d} for h, d in [("2", 18.0), ("4", 30.0), ("6", 20.0)]]
    res = bets.settle_allup("PLA", [{"race_no": 1, "selections": [5]}, {"race_no": 2, "selections": [2]}],
                            "2x1", {1: r1, 2: r2}, 10)
    assert res["payout"] == pytest.approx(21.6)


def test_allup_scratched_selection():
    """QPL 2x3, leg1 = R1 above with selections 1,4,11 and horse 4 scratched;
    leg2 = 2,5,8 with 2,5 paying 25.50.

    leg1 factors: 1,4 -> 1.0 (refund/carry fwd), 4,11 -> 1.0, 1,11 -> 3.7 ; sum 5.7
    leg2 factors: 2,5 -> 2.55, others 0                                   ; sum 2.55
    singles leg1: 10 × 5.7 = 57.0 (two $10 refunds + $37)
    singles leg2: 25.5
    doubles: 10 × 5.7 × 2.55 = 145.35   (1,4/2,5 carried forward -> 25.50 etc.)
    total 227.85
    """
    leg2 = [{"pool": "QPL", "combination": "2,5", "dividend": 25.5},
            {"pool": "QPL", "combination": "2,6", "dividend": 40.0},
            {"pool": "QPL", "combination": "5,6", "dividend": 50.0}]
    legs = [{"race_no": 1, "selections": [1, 4, 11]}, {"race_no": 2, "selections": [2, 5, 8]}]
    res = bets.settle_allup("QPL", legs, "2x3", {1: R1_HV_20261014, 2: leg2}, 10,
                            scratched_by_race={1: {4}})
    assert res["payout"] == pytest.approx(227.85)
    assert res["n_refunded_units"] == 2          # the two one-race refunds
    d = {x["combination"]: x for x in res["detail"]}
    assert d["1,4/2,5"]["return"] == pytest.approx(25.5)
    assert d["1,4/2,5"]["status"] == "win"
    assert d["1,4/2,8"]["status"] == "lose"      # 2,8 did not place
    assert d["1,4"]["status"] == "refund" and d["1,4"]["return"] == 10
    # PLA all-up where every leg horse is scratched -> full refund of stake
    pla = {1: _pla_rows([(1, 15.0), (2, 16.0), (3, 17.0)]), 2: _pla_rows([(1, 15.0), (2, 16.0), (3, 17.0)])}
    res = bets.settle_allup("PLA", [{"race_no": 1, "selections": [9]}, {"race_no": 2, "selections": [9]}],
                            "2x3", pla, 10, scratched_by_race={1: [9], 2: [9]})
    assert res["payout"] == pytest.approx(30.0) and res["n_refunded_units"] == 3


def test_allup_place_only_two_places_paid():
    """Field of 6: only 1st/2nd get PLA dividends.  Leg 1 horse 3 finished 3rd ->
    no row -> loses; leg 2 horse 1 places 13.0.  PLA 2x3:
    singles: 0 + 13.0; double: 0 -> payout 13.0."""
    r1 = [{"pool": "WIN", "combination": "2", "dividend": 30.0},
          {"pool": "PLA", "combination": "2", "dividend": 14.0},
          {"pool": "PLA", "combination": "5", "dividend": 19.0}]
    r2 = _pla_rows([(1, 13.0), (4, 20.0), (6, 25.0)])
    res = bets.settle_allup("PLA", [{"race_no": 1, "selections": [3]}, {"race_no": 2, "selections": [1]}],
                            "2x3", {1: r1, 2: r2}, 10)
    assert res["payout"] == pytest.approx(13.0)


def test_allup_pool_not_offered_leg_is_carried_forward():
    """QPL not offered in race 1 (e.g. fewer than 7 starters, rule 3.9(b)) - only
    WIN/PLA rows exist.  Leg factor 1.0 (assumed carry-forward).  QPL 2x1 with one
    pair per leg; leg 2 pair pays 25.5 -> 25.5."""
    r1 = [{"pool": "WIN", "combination": "2", "dividend": 30.0},
          {"pool": "PLA", "combination": "2", "dividend": 14.0}]
    r2 = [{"pool": "QPL", "combination": "2,5", "dividend": 25.5}]
    res = bets.settle_allup("QPL", [{"race_no": 1, "selections": [1, 2]}, {"race_no": 2, "selections": [2, 5]}],
                            "2x1", {1: r1, 2: r2}, 10)
    assert res["payout"] == pytest.approx(25.5)
    assert res["legs"][0]["outcomes"] == {"1,2": "pool_refund"}
    with pytest.raises(ValueError):           # results missing entirely
        bets.settle_allup("QPL", [{"race_no": 1, "selections": [1, 2]}, {"race_no": 3, "selections": [2, 5]}],
                          "2x1", {1: r1, 2: r2}, 10)


def test_allup_accepts_sqlite_rows(tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE dividends (race_id TEXT, pool TEXT, combination TEXT, dividend REAL)")
    conn.executemany("INSERT INTO dividends VALUES ('x',?,?,?)",
                     [(r["pool"], r["combination"], r["dividend"]) for r in R1_HV_20261014])
    rows = conn.execute("SELECT * FROM dividends").fetchall()
    r = bets.settle_single("PLA", ["1", "11", "12"], rows, 10)
    assert r["payout"] == pytest.approx(17.5 + 14.0 + 28.0)
