"""HKJC betting maths: combination generators, all-up formulas, ticket costing and
settlement.  Pure functions, no I/O.  PAPER TRADING ONLY.

Conventions (SPEC.md)
---------------------
* Pool codes: WIN PLA QIN QPL FCT TCE TRI FF QTT.
* Combination strings: unordered pools (WIN/PLA/QIN/QPL/TRI/FF) -> horse numbers
  ascending, comma separated ("1,12"); ordered pools (FCT/TCE/QTT) -> finishing
  order ("1,12,11").
* Dividends are HK$ per $10 unit (as published by HKJC).  A combination staked
  at ``unit_stake`` returns ``unit_stake * dividend / 10``.

Sources (fetched 2026-10-19)
----------------------------
[AU]  HKJC "All Up Betting" table (racing), 2..6 legs:
      https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/all-up-betting/
      - the full formula table implemented in ``ALLUP_FORMULAS`` (2..6 legs) is a
        verbatim transcription of that page's table (columns = # of 1-leg,
        2-leg, ... 6-leg combinations).
      - "Withdrawal - Bet combinations involving ONE RACE only will be REFUNDED.
        For bet combinations involving MORE THAN ONE RACE, the stake(s) is/are
        automatically staked in subsequent selected race(s) and NO REFUND."
[LP]  HKJC "Pari-Mutuel Local Pools": racing all-up / cross-pool all-up is 2 to 6
      races (WIN/PLA/QIN/QPL), FCT/TRI all-up 2 to 3 races; Place pays 1st-3rd,
      or only 1st-2nd with 4-6 declared starters.
      https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/
[CT]  HKJC "Chance Table - Pari-Mutuel Pools" (unit counts for multiple / banker
      bets, used to verify the generators in tests):
      https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/chance-table/
[R3]  Horse Race Betting Rules, Section 3 (Pari-mutuel), rev. 2026-03-30:
      https://special.hkjc.com/e-win/en-US/betting-info/racing/betting-rules/
      (-/media/Sites/JCBW/Special/betting-rules/Horse_Race_Rule_3_Eng_20260330)
      - 3.9(a) two Place dividends with 4-6 declared starters, three with 7+.
      - 3.9(b) single-race pool closed and ALL bets refunded when starters fall
        below: WIN<2, PLA(3 div)<5, PLA(2 div)<4, QIN<3, QPL<7, FCT<4, TCE<4,
        TRI<5, QTT<6, FF<6.
      - 3.10(a) single-race pool: bets involving a Withdrawal are refunded; for
        multiple bets only the combinations involving the Withdrawal are refunded
        and the rest stand (so a QPL pair containing a scratched horse is refunded).
      - 3.10(c) All Up: (i) a bet on a Withdrawal in a one-race combination is
        refunded; (ii) in a multi-leg combination it is carried forward to the
        next leg.  => mathematically the leg factor is exactly 1.0 either way.
      - 3.11 dead-heats: the Net Pool is subdivided among the winning
        combinations, i.e. HKJC publishes one dividend row per winning
        combination; we pay each staked combination its own row.
      - 3.15 void race: single-race pool bets refunded in full.

Assumptions (not explicitly confirmed in the sources above)
-----------------------------------------------------------
* A leg whose pool was refunded / not offered (e.g. QPL with <7 starters, void
  race) is treated inside an all-up exactly like a withdrawal: factor 1.0
  (carried forward / refunded).  Rule 3.10(c) only literally covers withdrawals.
* No rounding is applied to all-up returns (HKJC carries the full dividend
  entitlement forward).  Results are rounded to cents only for display.
* A dividend row whose value is None (e.g. "REFUND" text) is treated as refund.
* All-up formulas for 7 and 8 legs are not offered for racing (max 6 legs [LP]);
  ``ALLUP_FORMULAS_FOOTBALL_STYLE`` lists the conventional 7/8-leg formulas
  (as used by HKJC football all-up) for completeness/analysis only.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations as _comb, permutations as _perm, product as _product
from math import comb as _ncr, prod as _prod

# ----------------------------------------------------------------- constants
POOL_SIZE = {"WIN": 1, "PLA": 1, "QIN": 2, "QPL": 2, "FCT": 2, "TRI": 3, "TCE": 3,
             "FF": 4, "QTT": 4}
ORDERED_POOLS = frozenset({"FCT", "TCE", "QTT"})
UNORDERED_POOLS = frozenset({"WIN", "PLA", "QIN", "QPL", "TRI", "FF"})
ALLUP_POOLS = frozenset({"WIN", "PLA", "QIN", "QPL", "FCT", "TRI"})   # [LP]
ALLUP_MAX_LEGS = {"WIN": 6, "PLA": 6, "QIN": 6, "QPL": 6, "FCT": 3, "TRI": 3}  # [LP]

# "NxM" -> subset sizes included.  Verbatim from the HKJC table [AU].
ALLUP_FORMULAS: dict[str, tuple[int, ...]] = {
    "2x1": (2,), "2x3": (1, 2),
    "3x1": (3,), "3x3": (2,), "3x4": (2, 3), "3x6": (1, 2), "3x7": (1, 2, 3),
    "4x1": (4,), "4x4": (3,), "4x5": (3, 4), "4x6": (2,), "4x10": (1, 2),
    "4x11": (2, 3, 4), "4x14": (1, 2, 3), "4x15": (1, 2, 3, 4),
    "5x1": (5,), "5x5": (4,), "5x6": (4, 5), "5x10": (2,), "5x15": (1, 2),
    "5x16": (3, 4, 5), "5x20": (2, 3), "5x25": (1, 2, 3), "5x26": (2, 3, 4, 5),
    "5x30": (1, 2, 3, 4), "5x31": (1, 2, 3, 4, 5),
    "6x1": (6,), "6x6": (5,), "6x7": (5, 6), "6x15": (2,), "6x20": (3,),
    "6x21": (1, 2), "6x22": (4, 5, 6), "6x35": (2, 3), "6x41": (1, 2, 3),
    "6x42": (3, 4, 5, 6), "6x50": (2, 3, 4), "6x56": (1, 2, 3, 4),
    "6x57": (2, 3, 4, 5, 6), "6x62": (1, 2, 3, 4, 5), "6x63": (1, 2, 3, 4, 5, 6),
}
# 7/8 legs: NOT available for HKJC racing (max 6 races).  Conventional formulas
# (HKJC football style) - ASSUMED, for analysis only.
ALLUP_FORMULAS_FOOTBALL_STYLE: dict[str, tuple[int, ...]] = {
    "7x1": (7,), "7x7": (6,), "7x8": (6, 7), "7x21": (5,), "7x35": (4,),
    "7x120": (2, 3, 4, 5, 6, 7), "7x127": (1, 2, 3, 4, 5, 6, 7),
    "8x1": (8,), "8x8": (7,), "8x9": (7, 8), "8x28": (6,), "8x56": (5,), "8x70": (4,),
    "8x247": (2, 3, 4, 5, 6, 7, 8), "8x255": (1, 2, 3, 4, 5, 6, 7, 8),
}
_ALL_FORMULAS = {**ALLUP_FORMULAS, **ALLUP_FORMULAS_FOOTBALL_STYLE}


# ----------------------------------------------------------------- helpers
def _norm_pool(pool: str) -> str:
    p = str(pool).upper()
    if p not in POOL_SIZE:
        raise ValueError(f"unsupported pool {pool!r}")
    return p


def parse_combination(combo) -> tuple[int, ...]:
    """'1,12' / '01,12' / (1, 12) / 7 -> tuple of ints (order preserved)."""
    if isinstance(combo, int):
        return (combo,)
    if isinstance(combo, str):
        parts = [p for p in combo.replace(" ", "").replace("-", ",").split(",") if p]
        return tuple(int(p) for p in parts)
    return tuple(int(h) for h in combo)


def canonical(pool: str, combo) -> str:
    """SPEC canonical combination string for ``pool``."""
    pool = _norm_pool(pool)
    horses = parse_combination(combo)
    if len(horses) != POOL_SIZE[pool]:
        raise ValueError(f"{pool} combination needs {POOL_SIZE[pool]} horses: {combo!r}")
    if len(set(horses)) != len(horses):
        raise ValueError(f"duplicate horse in combination {combo!r}")
    if pool in UNORDERED_POOLS:
        horses = tuple(sorted(horses))
    return ",".join(str(h) for h in horses)


def _uniq(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ----------------------------------------------------------------- all-up formulas
def allup_specs(n_legs: int, include_unofficial: bool = False) -> list[str]:
    """All formula names for ``n_legs`` legs, in HKJC table order."""
    src = _ALL_FORMULAS if include_unofficial else ALLUP_FORMULAS
    return [k for k in src if int(k.split("x")[0]) == n_legs]


def allup_formula(spec: str) -> tuple[int, tuple[int, ...]]:
    """'6x63' -> (6, (1,2,3,4,5,6)).  Raises ValueError for unknown formulas."""
    s = str(spec).lower().replace(" ", "").replace("×", "x").replace("*", "x")
    if s not in _ALL_FORMULAS:
        raise ValueError(f"unknown all-up formula {spec!r}")
    return int(s.split("x")[0]), _ALL_FORMULAS[s]


def allup_subsets(n_legs: int, spec: str) -> list[tuple[int, ...]]:
    """All leg-index subsets (0-based, ascending) covered by formula ``spec``.

    e.g. allup_subsets(2, "2x3") -> [(0,), (1,), (0, 1)]."""
    n, sizes = allup_formula(spec)
    if n != n_legs:
        raise ValueError(f"formula {spec} needs {n} legs, got {n_legs}")
    out: list[tuple[int, ...]] = []
    for k in sizes:
        out.extend(_comb(range(n_legs), k))
    return out


def allup_combos(n_legs: int, spec: str) -> list[tuple[int, ...]]:
    """Alias of :func:`allup_subsets` (name used in SPEC.md)."""
    return allup_subsets(n_legs, spec)


def allup_count(spec: str) -> int:
    """Number of leg-subsets in a formula = sum C(n,k) (equals M in 'NxM')."""
    n, sizes = allup_formula(spec)
    return sum(_ncr(n, k) for k in sizes)


# ----------------------------------------------------------------- single-race generators
def multiple(pool: str, horses: Iterable[int]) -> list[str]:
    """複式 (box/multiple): every combination of the selected horses.

    WIN/PLA: one per horse; QIN/QPL C(n,2); TRI C(n,3); FF C(n,4);
    FCT n(n-1); TCE n(n-1)(n-2); QTT n(n-1)(n-2)(n-3)   (see [CT])."""
    pool = _norm_pool(pool)
    hs = _uniq(int(h) for h in horses)
    k = POOL_SIZE[pool]
    gen = _perm(hs, k) if pool in ORDERED_POOLS else _comb(sorted(hs), k)
    return _uniq(canonical(pool, c) for c in gen)


def banker(pool: str, bankers: Iterable[int], legs: Iterable[int],
           mode: str = "multiple") -> list[str]:
    """膽拖 (banker + legs).  Every combination contains all bankers, the remaining
    places filled from ``legs``.

    Unordered pools: bankers + each C(legs, k-b).
    Ordered pools (FCT/TCE/QTT):
      mode="multiple" (膽拖複式): bankers may finish in ANY of the positions
                                (all permutations of bankers ∪ chosen legs);
      mode="single"   (膽拖單式/位置膽): bankers fixed in positions 1..b in the
                                given order, legs fill the rest in any order.
    Counts match the HKJC chance table [CT], e.g. TCE 1 banker + n legs:
    single n(n-1), multiple 3n(n-1); QTT 2 bankers: single P(n,2), multiple 12·P(n,2).
    """
    pool = _norm_pool(pool)
    bs = _uniq(int(b) for b in bankers)
    ls = [h for h in _uniq(int(x) for x in legs) if h not in bs]
    k = POOL_SIZE[pool]
    if not bs:
        return multiple(pool, ls)
    if len(bs) >= k:
        raise ValueError(f"{pool}: {len(bs)} bankers but combination size is {k}")
    need = k - len(bs)
    out = []
    if pool in UNORDERED_POOLS:
        for extra in _comb(sorted(ls), need):
            out.append(canonical(pool, bs + list(extra)))
    elif mode == "multiple":
        for extra in _comb(ls, need):
            for p in _perm(bs + list(extra)):
                out.append(canonical(pool, p))
    elif mode == "single":
        for extra in _perm(ls, need):
            out.append(canonical(pool, bs + list(extra)))
    else:
        raise ValueError("mode must be 'multiple' or 'single'")
    return _uniq(out)


def positional(pool: str, positions: list[Iterable[int]]) -> list[str]:
    """Ordered pools: explicit horse set per finishing position (e.g. TCE
    [[1], [2,3], [2,3,4]] = 1 first, 2/3 second, 2/3/4 third).  Combinations
    with a repeated horse are dropped.  Also covers banker-with-position
    variants (put the banker alone in its position)."""
    pool = _norm_pool(pool)
    if pool not in ORDERED_POOLS:
        raise ValueError("positional() only applies to FCT/TCE/QTT")
    if len(positions) != POOL_SIZE[pool]:
        raise ValueError(f"{pool} needs {POOL_SIZE[pool]} position sets")
    sets = [_uniq(int(h) for h in s) for s in positions]
    return _uniq(canonical(pool, c) for c in _product(*sets) if len(set(c)) == len(c))


def single_combos(pool: str, selections: Iterable[int] = (), bankers: Iterable[int] = (),
                  positions: list | None = None, mode: str = "multiple") -> list[str]:
    """Dispatcher: positions -> :func:`positional`; bankers -> :func:`banker`;
    otherwise :func:`multiple`."""
    if positions is not None:
        return positional(pool, positions)
    bankers = list(bankers)
    if bankers:
        return banker(pool, bankers, selections, mode=mode)
    return multiple(pool, selections)


def count_units(pool: str, selections: Iterable[int] = (), bankers: Iterable[int] = (),
                positions: list | None = None, mode: str = "multiple") -> int:
    """Number of unit bets of a single-race bet."""
    return len(single_combos(pool, selections, bankers, positions, mode))


# ----------------------------------------------------------------- all-up legs / cost
def _leg_spec(leg, default_pool: str):
    """Normalise a leg to (race_no|None, pool, combos list)."""
    if isinstance(leg, Mapping):
        pool = _norm_pool(leg.get("pool") or default_pool)
        if leg.get("combos") is not None:
            combos = _uniq(canonical(pool, c) for c in leg["combos"])
        else:
            combos = single_combos(pool, leg.get("selections", ()), leg.get("bankers", ()),
                                   leg.get("positions"), leg.get("mode", "multiple"))
        return leg.get("race_no"), pool, combos
    pool = _norm_pool(default_pool)
    return None, pool, multiple(pool, leg)


def leg_combos(pool: str, leg) -> list[str]:
    """Combinations covered by one all-up leg.  ``leg`` is a list of horse numbers
    or a dict {race_no, selections, [bankers], [pool], [combos]}.
    PLA leg: one combination per horse; QPL leg with k horses: C(k,2) pairs."""
    return _leg_spec(leg, pool)[2]


def ticket_units(pool: str, legs: list, spec: str) -> int:
    """Number of unit bets of an all-up ticket: Σ over formula subsets of
    Π(#combinations in each leg of the subset).
    e.g. PLA 6x63 (1 horse/leg) = 63; QPL 2x3 (3 horses/leg) = 3+3+3·3 = 15."""
    sizes = [len(leg_combos(pool, leg)) for leg in legs]
    return sum(_prod(sizes[i] for i in sub) for sub in allup_subsets(len(legs), spec))


def ticket_cost(pool: str, legs: list, spec: str, unit_stake: float = 10.0) -> float:
    """Total stake of an all-up ticket (units × unit_stake)."""
    return ticket_units(pool, legs, spec) * float(unit_stake)


# ----------------------------------------------------------------- dividends index
def _row_get(row, key, idx):
    if isinstance(row, Mapping):
        return row.get(key)
    try:
        return row[key]          # sqlite3.Row
    except (IndexError, KeyError, TypeError):
        return row[idx]


REFUND_KEY = "__POOL_REFUND__"
_REFUND_MARKERS = {"-", "REFUND", "退款", ""}


def index_dividends(rows) -> dict[str, dict[str, float | None]]:
    """rows: iterable of dicts / sqlite3.Row with pool, combination, dividend
    (or (pool, combination, dividend) tuples), or an already-built
    {pool: {combination: dividend}} mapping.  Combinations are re-canonicalised."""
    if isinstance(rows, Mapping):
        items = [(p, c, d) for p, m in rows.items() for c, d in m.items()]
    else:
        items = [(_row_get(r, "pool", 0), _row_get(r, "combination", 1),
                  _row_get(r, "dividend", 2)) for r in rows]
    out: dict[str, dict[str, float | None]] = {}
    for pool, combo, div in items:
        p = str(pool).upper()
        if combo is None or str(combo).strip().upper() in _REFUND_MARKERS:
            # whole pool refunded (rules 3.9(b), 3.15): legacy rows ('-', NULL), parser rows ('REFUND', 10.0)
            out.setdefault(p, {})[REFUND_KEY] = None
            continue
        try:
            key = canonical(p, combo) if p in POOL_SIZE else str(combo)
        except ValueError:
            key = str(combo)
        out.setdefault(p, {})[key] = None if div is None else float(div)
    return out


def combo_outcome(pool: str, combo, div_index: Mapping, scratched=frozenset()) -> tuple[str, float, float | None]:
    """Outcome of one combination in one race.

    Returns (status, factor, dividend) where factor is the multiplier applied to
    the stake (return = stake × factor):
      'refund'      factor 1.0 - contains a scratched horse  (rule 3.10(a)/(c))
      'pool_refund' factor 1.0 - pool not offered/refunded for this race (assumed)
      'win'         factor dividend/10
      'lose'        factor 0.0
    """
    pool = _norm_pool(pool)
    key = canonical(pool, combo)
    scr = {int(s) for s in (scratched or ())}
    if scr & set(parse_combination(key)):
        return "refund", 1.0, None
    rows = div_index.get(pool)
    if not rows or REFUND_KEY in rows:
        return "pool_refund", 1.0, None
    if key in rows:
        d = rows[key]
        if d is None:
            return "refund", 1.0, None
        return "win", d / 10.0, d
    return "lose", 0.0, None


# ----------------------------------------------------------------- settlement
def settle_single(pool: str, combos, dividends_rows, unit_stake: float = 10.0,
                  scratched=frozenset()) -> dict:
    """Settle a single-race bet (list of combinations, each one unit).

    dividends_rows: that race's dividend rows (see :func:`index_dividends`).
    Dead heats: each staked combination is paid its own published row.
    Returns {payout, stake, profit, n_units, n_winning_units, n_refunded_units, detail}.
    """
    idx = index_dividends(dividends_rows)
    if not idx:
        raise ValueError("no dividend rows for this race (results missing?)")
    pool = _norm_pool(pool)
    combos = _uniq(canonical(pool, c) for c in combos)
    detail, payout, n_win, n_ref = [], 0.0, 0, 0
    for c in combos:
        status, factor, div = combo_outcome(pool, c, idx, scratched)
        ret = unit_stake * factor
        payout += ret
        n_win += status == "win"
        n_ref += status in ("refund", "pool_refund")
        detail.append({"combination": c, "status": status, "dividend": div,
                       "return": round(ret, 6)})
    stake = unit_stake * len(combos)
    return {"payout": round(payout, 6), "stake": stake, "profit": round(payout - stake, 6),
            "n_units": len(combos), "n_winning_units": n_win,
            "n_refunded_units": n_ref, "detail": detail}


def settle_allup(pool: str, legs: list, spec: str, dividends_by_race: Mapping,
                 unit_stake: float = 10.0, scratched_by_race: Mapping | None = None) -> dict:
    """Settle an all-up ticket.

    legs: list of {race_no, selections, [bankers], [pool]} (leg order = race order).
    dividends_by_race: {race_no: dividend rows of that race}.
    scratched_by_race: {race_no: iterable of scratched horse numbers}.

    For each subset of legs in the formula and each choice of one combination per
    leg, the unit returns unit_stake × Π(leg factor), with leg factor =
    dividend/10 (win), 0 (lose), 1.0 (scratched / pool refunded: refund or carry
    forward, rule 3.10(c)).  Returns {payout, stake, profit, n_units,
    n_winning_units, n_refunded_units, legs, detail}.
    """
    scratched_by_race = scratched_by_race or {}
    leg_info = []
    for i, leg in enumerate(legs):
        race_no, lpool, combos = _leg_spec(leg, pool)
        if race_no is None:
            raise ValueError("each all-up leg needs a race_no")
        rows = dividends_by_race.get(race_no)
        if rows is None:
            rows = dividends_by_race.get(str(race_no))
        idx = index_dividends(rows or [])
        if not idx:
            raise ValueError(f"no dividend rows for race {race_no}")
        scr = scratched_by_race.get(race_no, scratched_by_race.get(str(race_no), ()))
        outs = {c: combo_outcome(lpool, c, idx, scr) for c in combos}
        leg_info.append({"race_no": race_no, "pool": lpool, "combos": combos, "outcomes": outs})

    detail, payout, n_win, n_ref, n_units = [], 0.0, 0, 0, 0
    for sub in allup_subsets(len(legs), spec):
        for choice in _product(*(leg_info[i]["combos"] for i in sub)):
            factor, statuses, divs = 1.0, [], []
            for i, c in zip(sub, choice):
                st, f, d = leg_info[i]["outcomes"][c]
                factor *= f
                statuses.append(st)
                divs.append(d)
            if "lose" in statuses:
                status = "lose"
                factor = 0.0
            elif "win" in statuses:
                status = "win"
            else:
                status = "refund"
            ret = unit_stake * factor
            payout += ret
            n_units += 1
            n_win += status == "win"
            n_ref += status == "refund"
            detail.append({
                "legs": [leg_info[i]["race_no"] for i in sub],
                "combination": "/".join(choice),
                "status": status, "leg_status": statuses, "dividends": divs,
                "return": round(ret, 6)})
    stake = unit_stake * n_units
    return {"payout": round(payout, 6), "stake": stake, "profit": round(payout - stake, 6),
            "n_units": n_units, "n_winning_units": n_win, "n_refunded_units": n_ref,
            "legs": [{"race_no": li["race_no"], "pool": li["pool"],
                      "outcomes": {c: o[0] for c, o in li["outcomes"].items()}}
                     for li in leg_info],
            "detail": detail}


# ----------------------------------------------------------------- multi-race pools (DBL / TBL)
# HKJC 孖寶 (Double) / 三寶 (Treble): pick the WINNER of every leg except the last, and a horse
# that finishes 1st OR 2nd in the last leg - the results pages publish one dividend row for
# "winners / last-leg winner" and one for "winners / last-leg runner-up" (verified on every
# 2023-26 meeting in data/hkjc.db: 2 DBL / TBL rows per pool, e.g. 2026-10-19 R1-2 "6/7"
# $3,025 and "6/2" $101).  Dividends are stored on the LAST leg's race_id, combinations
# '/'-separated in leg order; an 'F' part means "any horse" (e.g. a void leg).
# Withdrawals: HKJC multi-leg pools use a SUBSTITUTE - a withdrawn selection is replaced by
# the leg's favourite (the starter with the largest WIN-pool investment, i.e. the lowest final
# win odds) [R3 multi-race substitute rule; ASSUMED here from the HKJC betting guide, not
# re-verified].  With no substitute known the unit is refunded.
MULTI_RACE_POOLS = frozenset({"DBL", "TBL"})
MULTI_RACE_LEGS = {"DBL": 2, "TBL": 3}


def _leg_horses(leg) -> list[int]:
    if isinstance(leg, Mapping):
        hs = list(leg.get("bankers") or []) + list(leg.get("selections") or [])
    else:
        hs = list(leg)
    return _uniq(int(h) for h in hs)


def multi_race_combos(pool: str, legs: list) -> list[str]:
    """All unit combinations of a DBL / TBL bet: one horse per leg ('a/b' or 'a/b/c')."""
    p = str(pool).upper()
    if p not in MULTI_RACE_POOLS:
        raise ValueError(f"not a multi-race pool: {pool!r}")
    if len(legs) != MULTI_RACE_LEGS[p]:
        raise ValueError(f"{p} needs {MULTI_RACE_LEGS[p]} legs, got {len(legs)}")
    return ["/".join(str(h) for h in c) for c in _product(*(_leg_horses(l) for l in legs))]


def multi_race_units(pool: str, legs: list) -> int:
    return len(multi_race_combos(pool, legs))


def _mr_rows(dividend_rows, pool):
    out = []
    for r in dividend_rows or []:
        if isinstance(r, Mapping):
            p, c, d = r.get("pool"), r.get("combination"), r.get("dividend")
        else:
            p, c, d = _row_get(r, "pool", 0), _row_get(r, "combination", 1), _row_get(r, "dividend", 2)
        if str(p).upper() == pool:
            out.append((str(c), d))
    return out


def settle_multi_race(pool: str, legs: list, dividend_rows, unit_stake: float = 10.0,
                      scratched_by_leg=None, substitutes=None) -> dict:
    """Settle a DBL / TBL bet.

    legs: [{race_no, selections}] in leg order.  dividend_rows: the pool's rows of the last
    leg race ((pool, combination, dividend) tuples / dicts; other pools are ignored).
    scratched_by_leg / substitutes: {race_no: set of withdrawn horses} / {race_no: favourite}.
    A unit wins every published row it matches ('F' matches any horse).  No rows for the pool,
    or a refund row -> every unit refunded (pool not offered / void)."""
    p = str(pool).upper()
    combos = multi_race_combos(p, legs)
    rnos = [l.get("race_no") if isinstance(l, Mapping) else None for l in legs]
    scr = scratched_by_leg or {}
    subs = substitutes or {}
    rows = _mr_rows(dividend_rows, p)
    refund_all = not rows or any(c.strip().upper() in _REFUND_MARKERS for c, _d in rows)
    parsed = []
    for c, d in rows:
        parsed.append(([x.strip() for x in c.split("/")], d))
    detail, payout, n_win, n_ref = [], 0.0, 0, 0
    for c in combos:
        hs = [int(x) for x in c.split("/")]
        eff, refund = [], refund_all
        for rn, h in zip(rnos, hs):
            s = {int(x) for x in (scr.get(rn) or scr.get(str(rn)) or ())}
            if h in s:
                sub = subs.get(rn, subs.get(str(rn)))
                if sub is None:
                    refund = True
                    eff.append(h)
                else:
                    eff.append(int(sub))
            else:
                eff.append(h)
        if refund:
            status, ret, div = "refund", unit_stake, None
        else:
            div = 0.0
            hit = False
            for parts, d in parsed:
                if len(parts) != len(eff):
                    continue
                if all(pp.upper() == "F" or (pp.isdigit() and int(pp) == h) for pp, h in zip(parts, eff)):
                    hit = True
                    div += 10.0 if d is None else float(d)
            status = "win" if hit else "lose"
            ret = unit_stake * div / 10.0 if hit else 0.0
            div = div if hit else None
        payout += ret
        n_win += status == "win"
        n_ref += status == "refund"
        detail.append({"combination": c, "effective": "/".join(map(str, eff)), "status": status,
                       "dividend": div, "return": round(ret, 6)})
    stake = unit_stake * len(combos)
    return {"payout": round(payout, 6), "stake": stake, "profit": round(payout - stake, 6),
            "n_units": len(combos), "n_winning_units": n_win, "n_refunded_units": n_ref,
            "detail": detail}
