"""Independent adversarial review tests for hkjc.bets and hkjc.odds.

Everything used as an oracle here is written from scratch in this file (no use
of hkjc.bets helpers): brute-force itertools generators, an independent
all-up settler using the identity  sum_S prod_{i in S} F_i , and reference
numbers transcribed from the HKJC pages fetched on 2026-10-19:
  * all-up table  https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/all-up-betting/
  * chance table  https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/chance-table/
  * Horse Race Betting Rules s.3 (rev. 2026-03-30), rules 3.9, 3.10, 3.11.

Tests marked ``xfail(strict=True)`` document CONFIRMED bugs: they fail today and
will start "XPASS -> fail" once the bug is fixed (then drop the marker).
"""
from __future__ import annotations

import itertools
import json
import os
import random
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from hkjc import bets, odds

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures"
DB = ROOT / "data" / "hkjc.db"

# ============================================================== reference data
# Verbatim from the HKJC all-up page table (columns: 1 Leg .. 6 Legs, blank = 0)
AU_PAGE = """
2 x 1|,1,,,,
2 x 3|2,1,,,,
3 x 1|,,1,,,
3 x 3|,3,,,,
3 x 4|,3,1,,,
3 x 6|3,3,,,,
3 x 7|3,3,1,,,
4 x 1|,,,1,,
4 x 4|,,4,,,
4 x 5|,,4,1,,
4 x 6|,6,,,,
4 x 10|4,6,,,,
4 x 11|,6,4,1,,
4 x 14|4,6,4,,,
4 x 15|4,6,4,1,,
5 x 1|,,,,1,
5 x 5|,,,5,,
5 x 6|,,,5,1,
5 x 10|,10,,,,
5 x 15|5,10,,,,
5 x 16|,,10,5,1,
5 x 20|,10,10,,,
5 x 25|5,10,10,,,
5 x 26|,10,10,5,1,
5 x 30|5,10,10,5,,
5 x 31|5,10,10,5,1,
6 x 1|,,,,,1
6 x 6|,,,,6,
6 x 7|,,,,6,1
6 x 15|,15,,,,
6 x 20|,,20,,,
6 x 21|6,15,,,,
6 x 22|,,,15,6,1
6 x 35|,15,20,,,
6 x 41|6,15,20,,,
6 x 42|,,20,15,6,1
6 x 50|,15,20,15,,
6 x 56|6,15,20,15,,
6 x 57|,15,20,15,6,1
6 x 62|6,15,20,15,6,
6 x 63|6,15,20,15,6,1
"""


def _au_table():
    out = {}
    for line in AU_PAGE.strip().splitlines():
        name, cols = line.split("|")
        out[name.replace(" ", "")] = tuple(int(c) if c else 0 for c in cols.split(","))
    return out


AU = _au_table()


def my_formula_sizes(spec):
    """Subset sizes of a formula derived from the page table (full layers only)."""
    cols = AU[spec]
    n = int(spec.split("x")[0])
    sizes = set()
    for k, cnt in enumerate(cols, start=1):
        if cnt:
            assert cnt == comb(n, k), (spec, k)   # HKJC formulas always use whole layers
            sizes.add(k)
    return sizes


# ============================================================== 1. all-up formulas
def test_allup_table_vs_hkjc_page():
    assert set(bets.ALLUP_FORMULAS) == set(AU)
    for spec, cols in AU.items():
        n, m = map(int, spec.split("x"))
        subs = bets.allup_subsets(n, spec)
        assert len(subs) == m == sum(cols) == bets.allup_count(spec), spec
        assert len(set(subs)) == len(subs), spec
        by_k = tuple(sum(1 for s in subs if len(s) == k) for k in range(1, 7))
        assert by_k == cols, spec
        # brute force: every subset of the given sizes appears exactly once
        brute = {s for k in my_formula_sizes(spec) for s in itertools.combinations(range(n), k)}
        assert set(subs) == brute, spec


def test_allup_specific_counts():
    s63 = bets.allup_subsets(6, "6x63")
    assert len(s63) == 63 and len(set(s63)) == 63
    assert set(s63) == {tuple(i for i in range(6) if m >> i & 1) for m in range(1, 64)}
    assert len(bets.allup_subsets(2, "2x3")) == 3
    assert sorted(bets.allup_subsets(2, "2x3")) == [(0,), (0, 1), (1,)]


# ============================================================== 2. combination counts
ORDERED = {"FCT", "TCE", "QTT"}
K = {"WIN": 1, "PLA": 1, "QIN": 2, "QPL": 2, "FCT": 2, "TRI": 3, "TCE": 3, "FF": 4, "QTT": 4}


def brute_multiple(pool, horses):
    k = K[pool]
    out = set()
    for c in itertools.product(horses, repeat=k):
        if len(set(c)) < k:
            continue
        out.add(c if pool in ORDERED else tuple(sorted(c)))
    return out


def brute_banker(pool, bankers, legs, mode="multiple"):
    """Every k-combination drawn from bankers+legs that contains ALL bankers.
    Ordered pools, single mode: bankers occupy positions 1..b in given order."""
    k = K[pool]
    allh = list(bankers) + list(legs)
    out = set()
    for c in itertools.product(allh, repeat=k):
        if len(set(c)) < k or not set(bankers) <= set(c):
            continue
        if pool in ORDERED:
            if mode == "single" and tuple(c[:len(bankers)]) != tuple(bankers):
                continue
            out.add(c)
        else:
            out.add(tuple(sorted(c)))
    return out


def as_tuples(strs):
    return {tuple(int(x) for x in s.split(",")) for s in strs}


def test_multiple_vs_brute_force():
    rng = random.Random(1)
    for pool in K:
        for n in range(1, 10):
            hs = rng.sample(range(1, 15), n)
            got = bets.multiple(pool, hs)
            assert len(got) == len(set(got))
            assert as_tuples(got) == brute_multiple(pool, hs), (pool, hs)


def test_banker_vs_brute_force():
    rng = random.Random(2)
    for pool in ("QIN", "QPL", "TRI", "FF", "FCT", "TCE", "QTT"):
        for b in range(1, K[pool]):
            for n in range(0, 8):
                hs = rng.sample(range(1, 15), b + n)
                bk, lg = hs[:b], hs[b:]
                modes = ("single", "multiple") if pool in ORDERED else ("multiple",)
                for mode in modes:
                    got = bets.banker(pool, bk, lg, mode=mode)
                    assert len(got) == len(set(got))
                    assert as_tuples(got) == brute_banker(pool, bk, lg, mode), (pool, bk, lg, mode)


# Chance table, every non-blank number on the page.  Key: (pool, bankers, mode)
# -> {no. of selections (legs for banker rows): count}
def _row(start, vals):
    return {start + i: v for i, v in enumerate(vals) if v is not None}


CHANCE = {
    ("QIN", 0, "m"): _row(2, [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91]),
    ("QIN", 1, "m"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ("TRI", 0, "m"): _row(3, [1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364]),
    ("TRI", 1, "m"): _row(3, [3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78]),
    ("TRI", 2, "m"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ("FF", 0, "m"): _row(5, [5, 15, 35, 70, 126, 210, 330, 495, 715, 1001]) | {4: 1},
    ("FF", 1, "m"): _row(4, [4, 10, 20, 35, 56, 84, 120, 165, 220, 286]),
    ("FF", 2, "m"): _row(3, [3, 6, 10, 15, 21, 28, 36, 45, 55, 66]),
    ("FF", 3, "m"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("FCT", 0, "m"): _row(2, [2, 6, 12, 20, 30, 42, 56, 72, 90, 110, 132, 156, 182]),
    ("FCT", 1, "s"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    ("FCT", 1, "m"): _row(2, [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]),
    ("TCE", 0, "m"): _row(3, [6, 24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184]),
    ("TCE", 1, "s"): _row(2, [2, 6, 12, 20, 30, 42, 56, 72, 90, 110, 132, 156]),
    ("TCE", 1, "m"): _row(3, [18, 36, 60, 90, 126, 168, 216, 270, 330, 396, 468]),
    ("TCE", 2, "s"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    ("TCE", 2, "m"): _row(2, [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]),
    ("QTT", 0, "m"): _row(4, [24, 120, 360, 840, 1680, 3024, 5040, 7920, 11880, 17160, 24024]),
    ("QTT", 1, "s"): _row(3, [6, 24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716]),
    ("QTT", 1, "m"): _row(4, [96, 240, 480, 840, 1344, 2016, 2880, 3960, 5280, 6864]),
    ("QTT", 2, "s"): _row(2, [2, 6, 12, 20, 30, 42, 56, 72, 90, 110, 132]),
    ("QTT", 2, "m"): _row(3, [72, 144, 240, 360, 504, 672, 864, 1080, 1320, 1584]),
    ("QTT", 3, "s"): _row(2, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("QTT", 3, "m"): _row(2, [48, 72, 96, 120, 144, 168, 192, 216, 240, 264]),
}
CHANCE[("QPL", 0, "m")] = CHANCE[("QIN", 0, "m")]
CHANCE[("QPL", 1, "m")] = CHANCE[("QIN", 1, "m")]


@pytest.mark.parametrize("key", sorted(CHANCE))
def test_chance_table(key):
    pool, nb, mode = key
    mode = "single" if mode == "s" else "multiple"
    for n, expected in CHANCE[key].items():
        bk = list(range(1, nb + 1))
        sel = list(range(20, 20 + n))
        if nb == 0:
            got = len(bets.multiple(pool, sel))
            brute = len(brute_multiple(pool, sel))
        else:
            got = len(bets.banker(pool, bk, sel, mode=mode))
            brute = len(brute_banker(pool, bk, sel, mode))
        assert brute == expected, (key, n, "my brute force vs page")
        assert got == expected, (key, n, "module vs page")


# ============================================================== 3. independent settler
REFUND_MARKERS = {"-", "REFUND", "退款", ""}


def my_key(pool, combo):
    nums = [int(x) for x in re.findall(r"\d+", str(combo))]
    return tuple(nums) if pool in ORDERED else tuple(sorted(nums))


def my_index(rows):
    """{pool: {'refund': bool, 'divs': {tuple: Fraction|None}}}"""
    out = {}
    for r in rows:
        if isinstance(r, dict):
            p, c, d = r["pool"], r["combination"], r["dividend"]
        else:
            p, c, d = r[0], r[1], r[2]
        p = str(p).upper()
        e = out.setdefault(p, {"refund": False, "divs": {}})
        if str(c).strip().upper() in REFUND_MARKERS or not re.search(r"\d", str(c)):
            e["refund"] = True                       # whole pool refunded (rule 3.9(b), void)
            continue
        e["divs"][my_key(p, c)] = None if d is None else Fraction(str(d))
    return out


def my_factor(pool, combo, idx, scratched):
    if set(combo) & set(scratched):
        return Fraction(1)                           # rule 3.10(a) / 3.10(c)
    e = idx.get(pool)
    if e is None or e["refund"]:
        return Fraction(1)                           # pool not offered / refunded
    if combo in e["divs"]:
        d = e["divs"][combo]
        return Fraction(1) if d is None else d / 10
    return Fraction(0)


def my_leg_combos(pool, selections):
    if K[pool] == 1:
        return [(h,) for h in dict.fromkeys(selections)]
    return [tuple(sorted(c)) for c in itertools.combinations(sorted(set(selections)), K[pool])]


def my_settle_single(pool, selections, rows, unit, scratched=()):
    idx = my_index(rows)
    cs = my_leg_combos(pool, selections)
    return Fraction(str(unit)) * sum((my_factor(pool, c, idx, scratched) for c in cs), Fraction(0)), len(cs)


def my_settle_allup(pool, legs, spec, rows_by_race, unit, scr_by_race):
    """payout = unit * sum_{S in formula} prod_{i in S} F_i,  F_i = sum_c factor_i(c).
    (product distributes over the per-leg choices, so no enumeration of units)."""
    sizes = my_formula_sizes(spec)
    F, N = [], []
    for leg in legs:
        idx = my_index(rows_by_race[leg["race_no"]])
        cs = my_leg_combos(pool, leg["selections"])
        scr = scr_by_race.get(leg["race_no"], ())
        F.append(sum((my_factor(pool, c, idx, scr) for c in cs), Fraction(0)))
        N.append(len(cs))
    pay, units = Fraction(0), 0
    for mask in range(1, 1 << len(legs)):
        S = [i for i in range(len(legs)) if mask >> i & 1]
        if len(S) not in sizes:
            continue
        p, u = Fraction(1), 1
        for i in S:
            p *= F[i]
            u *= N[i]
        pay += p
        units += u
    return Fraction(str(unit)) * pay, units


def cents(x):
    return round(float(x) + 1e-9, 2)


# ----------------------------------------------------------- random race generator
def gen_race(rng, refund_style="none"):
    """A random race consistent with rules 3.9 / 3.11.
    refund_style for a closed pool: 'none' (no rows), 'dash' (legacy DB row
    combination '-' dividend NULL), 'refund' (current parse.py row 'REFUND', 10.0)."""
    declared = rng.randint(4, 14)
    horses = list(range(1, declared + 1))
    scratched = set(rng.sample(horses, rng.choice([0, 0, 0, 1, 1, 2])))
    starters = [h for h in horses if h not in scratched]
    rng.shuffle(starters)
    # competition ranking with an optional 2-horse dead heat at 1st/2nd/3rd
    ranks = list(range(1, len(starters) + 1))
    dh = rng.choice([None, None, None, 1, 2, 3])
    if dh is not None and dh < len(starters):
        ranks[dh] = ranks[dh - 1]                    # horses at idx dh-1, dh share rank dh
    rank = dict(zip(starters, ranks))
    places = 2 if declared <= 6 else 3               # rule 3.9(a)
    rows = [{"pool": "WIN", "combination": str(h), "dividend": float(rng.randint(21, 400)) / 2}
            for h in starters if rank[h] == 1]

    def closed(pool):
        if refund_style == "dash":
            rows.append({"pool": pool, "combination": "-", "dividend": None})
        elif refund_style == "refund":
            rows.append({"pool": pool, "combination": "REFUND", "dividend": 10.0})

    pla_closed = len(starters) < (5 if places == 3 else 4)        # rule 3.9(b)
    if pla_closed:
        closed("PLA")
    else:
        for h in starters:
            if rank[h] <= places:
                rows.append({"pool": "PLA", "combination": str(h),
                             "dividend": round(rng.uniform(10.1, 120.0), 1)})
    if declared >= 7:                                 # QPL offered
        if len(starters) < 7:                         # rule 3.9(b): closed, refunded
            closed("QPL")
        else:                                         # rule 3.11(a)(ii): all pairs of top-3 ranks
            top = sorted(h for h in starters if rank[h] <= 3)
            for a, b in itertools.combinations(top, 2):
                c = f"{b},{a}" if rng.random() < 0.5 else f"{a},{b}"   # unsorted on purpose
                rows.append({"pool": "QPL", "combination": c,
                             "dividend": float(rng.randint(21, 4000)) / 2})
    rng.shuffle(rows)
    return {"declared": declared, "horses": horses, "scratched": scratched, "rows": rows}


def _ticket(rng, races, n_legs, n_sel):
    legs = []
    for rno in rng.sample(sorted(races), n_legs):
        legs.append({"race_no": rno, "selections": rng.sample(races[rno]["horses"],
                                                              min(n_sel, len(races[rno]["horses"])))})
    return legs


@pytest.mark.parametrize("seed", range(40))
def test_property_allup_and_single_vs_independent(seed):
    rng = random.Random(1000 + seed)
    races = {r: gen_race(rng) for r in range(1, 12)}
    rows = {r: v["rows"] for r, v in races.items()}
    scr = {r: v["scratched"] for r, v in races.items()}
    for _ in range(4):
        unit = rng.choice([10, 10, 5, 20, 1])
        # PLA 6x63, 1 horse per leg (SPEC strategy) + random other formulas / sizes
        legs = _ticket(rng, races, 6, 1)
        exp, units = my_settle_allup("PLA", legs, "6x63", rows, unit, scr)
        got = bets.settle_allup("PLA", legs, "6x63", rows, unit, scr)
        assert got["n_units"] == units == 63
        assert cents(got["payout"]) == cents(exp), (legs, got["payout"], float(exp))
        # QPL 2x3, 3 horses per leg (SPEC strategy)
        legs = _ticket(rng, races, 2, 3)
        exp, units = my_settle_allup("QPL", legs, "2x3", rows, unit, scr)
        got = bets.settle_allup("QPL", legs, "2x3", rows, unit, scr)
        assert got["n_units"] == units
        assert cents(got["payout"]) == cents(exp), (legs, got["payout"], float(exp))
        # random formula / pool / selection count
        spec = rng.choice(sorted(AU))
        n = int(spec.split("x")[0])
        pool = rng.choice(["PLA", "QPL"])
        legs = _ticket(rng, races, n, rng.randint(1, 4))
        exp, units = my_settle_allup(pool, legs, spec, rows, unit, scr)
        got = bets.settle_allup(pool, legs, spec, rows, unit, scr)
        assert got["n_units"] == units == bets.ticket_units(pool, legs, spec)
        assert cents(got["payout"]) == cents(exp), (pool, spec, legs)
        # single race
        rno = rng.choice(sorted(races))
        for pool in ("PLA", "QPL", "WIN"):
            sel = rng.sample(races[rno]["horses"], min(rng.randint(1, 5), len(races[rno]["horses"])))
            exp, nu = my_settle_single(pool, sel, rows[rno], unit, scr[rno])
            got = bets.settle_single(pool, bets.multiple(pool, sel), rows[rno], unit, scr[rno])
            assert got["n_units"] == nu
            assert cents(got["payout"]) == cents(exp), (pool, sel, rows[rno])


def test_dead_heat_third_real_shape():
    """Rule 3.11(a)(ii)(c): dead heat for 3rd (2 horses) -> QPL winning combinations
    are 1st&2nd, 1st&3rd x2, 2nd&3rd x2 AND 3rd&3rd (2/15 each) = 6 rows;
    PLA pays 4 dividends.  Real example in DB: 20230910-ST-07 (PLA 5,7,8,9;
    QPL all 6 pairs incl. 5,9 = 111.0).  NB tests/test_bets.py::test_settle_single_dead_heat
    states '5,9 does NOT win' - wrong per 3.11, harmless only because the
    module pays whatever rows exist."""
    rows = [("WIN", "8", 18.5), ("PLA", "5", 17.5), ("PLA", "7", 10.1), ("PLA", "8", 12.0),
            ("PLA", "9", 49.5), ("QPL", "5,7", 29.5), ("QPL", "5,8", 25.5), ("QPL", "5,9", 111.0),
            ("QPL", "7,8", 10.5), ("QPL", "7,9", 43.0), ("QPL", "8,9", 87.5)]
    r = bets.settle_single("QPL", bets.multiple("QPL", [5, 9, 1]), rows, 10)
    assert r["payout"] == pytest.approx(111.0)           # 1,5 1,9 lose; 5,9 wins 111
    r = bets.settle_single("PLA", ["5", "9", "1"], rows, 10)
    assert r["payout"] == pytest.approx(17.5 + 49.5)
    # PLA 2x1 through the dead-heated horse
    res = bets.settle_allup("PLA", [{"race_no": 7, "selections": [9]}, {"race_no": 8, "selections": [9]}],
                            "2x1", {7: rows, 8: rows}, 10)
    assert res["payout"] == pytest.approx(10 * 4.95 * 4.95)


def test_two_place_dividends_small_field_and_qpl_absent():
    """Rule 3.9(a): 4-6 declared starters -> 2 place dividends; QPL not offered
    (no rows) -> leg factor 1.0 (module assumption, carried forward/refunded)."""
    rows = [("WIN", "2", 41.0), ("PLA", "1", 12.0), ("PLA", "2", 14.0), ("QIN", "1,2", 16.0)]
    r = bets.settle_single("PLA", ["3"], rows, 10)
    assert r["payout"] == 0
    r = bets.settle_single("QPL", ["1,2"], rows, 10)
    assert r["detail"][0]["status"] == "pool_refund" and r["payout"] == 10


# ------------------------------------------------------- CONFIRMED BUG: refund rows
LEG2 = [{"pool": "QPL", "combination": "1,2", "dividend": 25.5},
        {"pool": "PLA", "combination": "1", "dividend": 11.0}]


def test_bug_refund_row_from_parse_py_is_refund():
    # QPL closed (<7 starters, rule 3.9(b)); parse.py stores combination='REFUND', dividend=10
    r1 = [{"pool": "WIN", "combination": "3", "dividend": 31.0},
          {"pool": "QPL", "combination": "REFUND", "dividend": 10.0}]
    r = bets.settle_single("QPL", ["3,4"], r1, 10)
    assert r["payout"] == 10                  # actual: 0.0 (status 'lose')


def test_bug_dash_row_in_db_is_refund():
    # exactly what data/hkjc.db holds for 20230920-HV-01 (7 declared, 1 WV -> QPL closed)
    r1 = [{"pool": "WIN", "combination": "3", "dividend": 31.0},
          {"pool": "PLA", "combination": "3", "dividend": 12.5},
          {"pool": "QPL", "combination": "-", "dividend": None}]
    legs = [{"race_no": 1, "selections": [3, 4, 2]}, {"race_no": 2, "selections": [1, 2]}]
    res = bets.settle_allup("QPL", legs, "2x3", {1: r1, 2: LEG2}, 10)
    # leg1: 3 pairs x factor 1 = 3 ; leg2: 2.55 ; singles 10*3 + 25.5 ; double 10*3*2.55
    assert res["payout"] == pytest.approx(30 + 25.5 + 76.5)   # actual: 25.5


def test_bug_void_race_all_pools_refunded():
    void = [{"pool": p, "combination": "REFUND", "dividend": 10.0} for p in ("WIN", "PLA", "QIN", "QPL")]
    legs = [{"race_no": 1, "selections": [5]}, {"race_no": 2, "selections": [1]}]
    res = bets.settle_allup("PLA", legs, "2x3", {1: void, 2: LEG2}, 10)
    assert res["payout"] == pytest.approx(10 + 11 + 11)        # rule 3.15 refund; actual 11


def _refund_style_divergences(style):
    """Settle 120 random tickets on races whose closed pools carry DB-style refund
    rows; return (n_divergent, n_divergent_without_refund_leg)."""
    bad, unexplained = 0, 0
    for seed in range(10):
        rng = random.Random(5000 + seed)
        races = {r: gen_race(rng, refund_style=style) for r in range(1, 12)}
        rows = {r: v["rows"] for r, v in races.items()}
        scr = {r: v["scratched"] for r, v in races.items()}
        for _ in range(6):
            for pool, spec, nl, ns in (("PLA", "6x63", 6, 1), ("QPL", "2x3", 2, 3)):
                legs = _ticket(rng, races, nl, ns)
                exp, _ = my_settle_allup(pool, legs, spec, rows, 10, scr)
                got = bets.settle_allup(pool, legs, spec, rows, 10, scr)
                if cents(got["payout"]) != cents(exp):
                    bad += 1
                    if not any(my_index(rows[l["race_no"]]).get(pool, {}).get("refund") for l in legs):
                        unexplained += 1
    return bad, unexplained


@pytest.mark.parametrize("style", ["dash", "refund"])
def test_refund_style_divergence_only_on_closed_pool_legs(style):
    bad, unexplained = _refund_style_divergences(style)
    assert unexplained == 0


@pytest.mark.parametrize("style", ["dash", "refund"])
def test_bug_property_with_db_style_refund_rows(style):
    bad, _ = _refund_style_divergences(style)
    assert bad == 0


# ------------------------------------------------------- real data from data/hkjc.db
def _db():
    if not DB.exists():
        pytest.skip("no data/hkjc.db")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    n = conn.execute("SELECT COUNT(*) FROM meetings WHERE status='results'").fetchone()[0]
    if n < 10:
        pytest.skip("fewer than 10 complete meetings in DB yet")
    return conn


def _meeting_data(conn, date, venue):
    races, scr = {}, {}
    for (rid, rno) in conn.execute("SELECT race_id, race_no FROM races WHERE date=? AND venue=? ORDER BY race_no",
                                   (date, venue)):
        rows = conn.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id=?", (rid,)).fetchall()
        run = conn.execute("SELECT horse_no, scratched FROM runners WHERE race_id=?", (rid,)).fetchall()
        if not rows or not run:
            continue
        races[rno] = {"rows": rows, "horses": sorted(h for h, _ in run)}
        scr[rno] = {h for h, s in run if s}
    return races, scr


def _has_refund_marker(rows, pool):
    return any(p == pool and (str(c).strip().upper() in REFUND_MARKERS) for p, c, _ in rows)


def test_real_meetings_pla_6x63_and_qpl_2x3():
    conn = _db()
    meetings = conn.execute("SELECT date, venue FROM meetings WHERE status='results' ORDER BY date").fetchall()
    rng = random.Random(20261019)
    picked = [m for m in rng.sample(meetings, len(meetings))]
    checked, explained = 0, []
    for date, venue in picked:
        races, scr = _meeting_data(conn, date, venue)
        if len(races) < 6:
            continue
        rows = {r: v["rows"] for r, v in races.items()}
        tickets = [("PLA", "6x63", [{"race_no": r, "selections": [rng.choice(races[r]["horses"])]}
                                    for r in rng.sample(sorted(races), 6)])]
        six = rng.sample(sorted(races), 6)
        for t in range(3):
            tickets.append(("QPL", "2x3", [{"race_no": r, "selections": rng.sample(races[r]["horses"], 3)}
                                           for r in six[2 * t:2 * t + 2]]))
        for pool, spec, legs in tickets:
            exp, units = my_settle_allup(pool, legs, spec, rows, 10, scr)
            got = bets.settle_allup(pool, legs, spec, rows, 10, scr)
            assert got["n_units"] == units == (63 if pool == "PLA" else 15)
            if cents(got["payout"]) != cents(exp):
                # only allowed difference: the confirmed refund-marker bug
                assert any(_has_refund_marker(rows[l["race_no"]], pool) for l in legs), (date, venue, legs)
                explained.append((date, venue, pool, float(got["payout"]), float(exp)))
        checked += 1
        print("checked", date, venue)
        if checked == 10:
            break
    assert checked == 10
    print("divergences explained by refund-marker bug:", explained)


def test_real_qpl_closed_race_20230920_hv_01():
    conn = _db()
    rows = conn.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id='20230920-HV-01'").fetchall()
    if not rows:
        pytest.skip("race not in DB")
    r = bets.settle_single("QPL", ["3,4"], rows, 10)
    assert r["payout"] == 10


# ============================================================== 5. odds.py
def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_query_strings_byte_identical():
    assert odds.ODDS_QUERY.encode("utf-8") == _load("odds_query.json")["query"].encode("utf-8")
    assert odds.MEETING_QUERY.encode("utf-8") == _load("odds_meeting_query.json")["query"].encode("utf-8")
    assert _load("odds_query.json")["operationName"] == "racing"
    assert _load("odds_meeting_query.json")["operationName"] == "raceMeetings"


@pytest.mark.parametrize("pool,raw,exp", [
    ("QPL", "01,07", "1,7"), ("QPL", "07,01", "1,7"), ("QIN", "12,03", "3,12"),
    ("WIN", "01", "1"), ("PLA", "14", "14"), ("FCT", "07,01", "7,1"), ("TCE", "3,1,2", "3,1,2"),
])
def test_normalise_comb(pool, raw, exp):
    assert odds.normalise_comb(pool, raw) == exp


@pytest.mark.parametrize("raw,exp", [("SCR", None), ("---", None), ("", None), (None, None),
                                     ("4.1", 4.1), ("10", 10.0), ("1,234", 1234.0), (" 2.5 ", 2.5)])
def test_parse_odds_value(raw, exp):
    assert odds.parse_odds_value(raw) == exp


def _resp(pools):
    return {"data": {"raceMeetings": [{"pmPools": pools}]}}


def test_parse_pools_scr_dashes_and_empty():
    resp = _resp([
        {"oddsType": "WIN", "status": "START_SELL", "sellStatus": "START_SELL", "lastUpdateTime": "x",
         "oddsNodes": [{"combString": "01", "oddsValue": "3.5"}, {"combString": "02", "oddsValue": "SCR"},
                       {"combString": "03", "oddsValue": "---"}]},
        {"oddsType": "QPL", "status": "START_SELL", "sellStatus": "START_SELL", "lastUpdateTime": "x",
         "oddsNodes": [{"combString": "03,01", "oddsValue": "7.0"}, {"combString": "01,02", "oddsValue": "SCR"}]},
        {"oddsType": "QIN", "status": "DEFINED", "sellStatus": "STOP_SELL", "lastUpdateTime": "", "oddsNodes": []},
    ])
    p = odds.parse_pools(resp, 1)
    assert p["WIN"] == {"1": 3.5, "2": None, "3": None}
    assert p["QPL"] == {"1,3": 7.0, "1,2": None}
    assert p["QIN"] == {} and p.meta["QIN"]["last_update"] is None
    assert p.scratched == {2}                      # '---' is not a scratching
    assert odds.parse_pools({"data": {"raceMeetings": []}}) == {}
    assert odds.parse_pools({"errors": [{"message": "x"}], "data": {"raceMeetings": None}}) == {}


class _Resp:
    def __init__(self, d):
        self.d = d

    def raise_for_status(self):
        pass

    def json(self):
        return self.d


class _Sess:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.n += 1
        return _Resp(self.fn(json))


WL_ERR = {"errors": [{"message": "Internal server error - WHITELIST_ERROR"}], "data": {"raceMeetings": None}}


def test_whitelist_error_raises(monkeypatch):
    monkeypatch.setattr(odds, "MIN_INTERVAL_S", 0.0)
    s = _Sess(lambda b: WL_ERR)
    with pytest.raises(odds.OddsAPIError, match="WHITELIST_ERROR"):
        odds.fetch_pools("2026-10-19", "ST", 1, session=s)
    assert s.n == 1                                # GraphQL errors are not retried (good)
    from hkjc import db
    conn = db.connect(":memory:")
    with pytest.raises(odds.OddsAPIError):
        odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[1], session=_Sess(lambda b: WL_ERR))
    # meeting query failing too: falls back to DB / probing, still raises on the odds call
    with pytest.raises(odds.OddsAPIError):
        odds.snapshot_meeting(conn, "2026-10-19", "ST", session=_Sess(lambda b: WL_ERR))
    assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 0


def test_snapshot_idempotent_and_hk_time(monkeypatch):
    monkeypatch.setattr(odds, "MIN_INTERVAL_S", 0.0)
    from hkjc import db
    conn = db.connect(":memory:")
    r8 = _load("odds_20261019_ST_R8.json")
    s = _Sess(lambda b: r8)
    n1 = odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[8], session=s, snapshot_at="2026-10-19T16:00:00")
    n2 = odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[8], session=s, snapshot_at="2026-10-19T16:00:00")
    assert n1 == n2 == 12 + 12 + 66 + 66
    assert conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == n1     # no duplicates
    before = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None, microsecond=0)
    odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[8], session=s)
    after = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None, microsecond=0)
    ts = {t for (t,) in conn.execute("SELECT DISTINCT snapshot_at FROM odds_snapshots")} - {"2026-10-19T16:00:00"}
    assert len(ts) == 1
    t = datetime.fromisoformat(ts.pop())
    assert before <= t <= after                      # HK local time (UTC+8), not UTC / machine tz
    lo = odds.latest_odds(conn, "2026-10-19", "ST")
    assert lo[8]["WIN"]["4"] is None and lo[8]["QPL"]["2,10"] == 1.9


def test_post_time_hk():
    assert odds._hk_minute("2026-10-21T10:40:00Z") == "2026-10-21T18:40"
    assert odds._hk_minute("2026-10-21T18:40:00+08:00") == "2026-10-21T18:40"


# Final WIN odds 2026-10-19 ST R1, read from the official results page
# racing.hkjc.com/zh-hk/local/information/localresults?racedate=2026/10/19&Racecourse=ST&RaceNo=1
R1_FINAL_WIN = {6: 5.1, 11: 57.0, 13: 11.0, 4: 5.5, 1: 4.1, 8: 4.6, 2: 9.0, 3: 27.0, 9: 87.0,
                12: 25.0, 7: 53.0, 5: 37.0, 10: 31.0}


def test_fixture_r1_matches_results_page():
    p = odds.parse_pools(_load("odds_20261019_ST_R1.json"), 1)
    assert {int(k): v for k, v in p["WIN"].items()} == R1_FINAL_WIN


@pytest.mark.skipif(os.environ.get("HKJC_LIVE") != "1", reason="set HKJC_LIVE=1 for the live API check")
def test_live_r1_matches_results_page():
    p = odds.fetch_pools("2026-10-19", "ST", 1, ("WIN",))
    assert {int(k): v for k, v in p["WIN"].items()} == R1_FINAL_WIN
