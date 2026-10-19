"""Ticket construction (砌飛) from model predictions.  PAPER TRADING ONLY.

Public API
----------
build_tickets(pred_df, odds=None, config=None) -> list[dict]
    Products (config["products"]; default PLA_6x63, QPL_2x3, ADAPTIVE):
      * ADAPTIVE        - per race: skip or the structure with the best expected ROI
                          (decide_race; see the ADAPTIVE section below)
      * ADAPTIVE_ALLUP  - PLA / QPL parlays from qualifying races only, legs/formula
                          chosen by expected value (build_adaptive_allup)
    and the user's fixed baseline products:
      * PLA_6x63  - 1 ticket: 6 races x 1 horse, place all-up 6x63 (63 units)
      * QPL_2x3   - 3 tickets: each 2 races x 3 horses (3 QPL pairs per leg),
                    QPL all-up 2x3 (3 + 3 + 3*3 = 15 units); 6 distinct races.
build_allup(pred_df, pool, n_legs, spec, ...) -> dict | None   generic all-up ticket
build_single(race_df, spec, ...) -> dict | None                generic single-race ticket
single_spec_name(spec) -> str

Ticket dict (SPEC.md):
    {strategy, ticket_no, pool, spec, legs=[{race_no, pool, selections, [bankers],
     [positions], [mode]}], unit_stake, n_combos, total_stake, notes}
``spec`` is the all-up formula ("6x63") or None for a single-race bet.

pred_df columns: race_no, horse_no, p_win, p_place  (optional win_odds, pla_odds).
``odds`` (optional): {race_no: {pool: {combination: decimal odds}}} as returned
by hkjc.odds.latest_odds(); used for value filters (EV = p × odds).

Race-selection rules (``race_rule``):
    'confidence'  races whose leg has the highest hit probability
    'value'       races with the largest edge (EV with live odds when given,
                  otherwise model p / market p from win_odds); ties -> confidence
    'first6'/'first', 'last6'/'last'  first / last n races of the card
    'fixed'       explicit list ``fixed_races``
    'consecutive' (QPL multi-ticket) best set of non-overlapping adjacent race pairs
"""
from __future__ import annotations

import copy
from itertools import combinations

import numpy as np
import pandas as pd

from . import bets

DEFAULT_CONFIG = {
    "unit_stake": 10.0,
    # the user's fixed baseline products + the per-race ADAPTIVE product alongside.
    # "ADAPTIVE_ALLUP" is available (add it to products) but off by default: in the
    # 2024-25 tuning season no setting qualified on >= 20 meetings and the few
    # parlays it built lost money (see reports/回測報告.md, 按場自動決定策略).
    "products": ["PLA_6x63", "QPL_2x3", "ADAPTIVE"],
    "PLA_6x63": {
        "pool": "PLA", "n_legs": 6, "spec": "6x63", "horses_per_leg": 1,
        "race_rule": "confidence", "fixed_races": None,
        "value_filter": False, "min_ev": 1.0,      # only used when place odds known
    },
    "QPL_2x3": {
        "pool": "QPL", "n_tickets": 3, "legs_per_ticket": 2, "spec": "2x3",
        "horses_per_leg": 3, "race_rule": "confidence", "pairing": "chrono",
        "fixed_races": None,
    },
    "ADAPTIVE": {},          # overrides of ADAPTIVE_DEFAULT (defined below)
    "ADAPTIVE_ALLUP": {},    # overrides of ADAPTIVE_ALLUP_DEFAULT
}

RANK_BY = {"WIN": "p_win", "PLA": "p_place", "QIN": "p_win", "QPL": "p_place",
           "FCT": "p_win", "TCE": "p_win", "TRI": "p_win", "FF": "p_win", "QTT": "p_win"}


# ----------------------------------------------------------------- probability helpers
def market_probs(win_odds) -> np.ndarray:
    """Normalised implied win probabilities from decimal win odds (NaN -> 0)."""
    o = np.asarray(win_odds, dtype=float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), 0.0)
    s = inv.sum()
    return inv / s if s > 0 else inv


def harville(p_win, n_places: int | None = None) -> np.ndarray:
    """P(horse finishes in the first ``n_places``) under the Harville model.

    n_places defaults to the HKJC place rule: 3 with 7+ runners, 2 with 4-6 [R3 3.9(a)].
    Exact enumeration of ordered finishing prefixes."""
    p = np.asarray(p_win, dtype=float)
    n = len(p)
    if n == 0:
        return p
    s = p.sum()
    p = p / s if s > 0 else np.full(n, 1.0 / n)
    if n_places is None:
        n_places = 3 if n >= 7 else 2
    n_places = min(n_places, n)
    eps = 1e-12
    out = p.copy()
    if n_places >= 2:
        # second place: sum_a p_a * p_i/(1-p_a)
        d1 = np.maximum(1.0 - p, eps)
        m2 = p[:, None] * p[None, :] / d1[:, None]       # P(a first, b second)
        np.fill_diagonal(m2, 0.0)
        out = out + m2.sum(axis=0)
    if n_places >= 3:
        d2 = np.maximum(1.0 - p[:, None] - p[None, :], eps)
        t = m2[:, :, None] * p[None, None, :] / d2[:, :, None]   # P(a,b,c)
        idx = np.arange(n)
        t[idx, :, idx] = 0.0
        t[:, idx, idx] = 0.0
        t[idx, idx, :] = 0.0
        out = out + t.sum(axis=(0, 1))
    if n_places >= 4:
        raise ValueError("n_places > 3 not supported")
    return np.clip(out, 0.0, 1.0)


def p_at_least_two(ps) -> float:
    """P(at least 2 of the given horses place), independence approximation
    (used only to *rank* QPL legs)."""
    ps = [float(x) for x in ps]
    if len(ps) < 2:
        return 0.0
    n = len(ps)
    # P(exactly k) via convolution
    dist = np.zeros(n + 1)
    dist[0] = 1.0
    for x in ps:
        dist[1:] = dist[1:] * (1 - x) + dist[:-1] * x
        dist[0] *= (1 - x)
    return float(dist[2:].sum())


# ----------------------------------------------------------------- frame helpers
def _prep(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    need = {"race_no", "horse_no", "p_win", "p_place"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"pred_df missing columns {sorted(miss)}")
    df["race_no"] = df["race_no"].astype(int)
    df["horse_no"] = df["horse_no"].astype(int)
    if "scratched" in df.columns:
        df = df[df["scratched"].fillna(0).astype(int) == 0]
    df = df[df["p_win"].notna()]
    if "win_odds" in df.columns:
        df["p_mkt"] = df.groupby("race_no")["win_odds"].transform(lambda s: market_probs(s.values))
        df["p_mkt_place"] = df.groupby("race_no")["p_mkt"].transform(lambda s: harville(s.values))
    return df


def _odds_lookup(odds, race_no, pool, combo):
    if not odds:
        return None
    r = odds.get(race_no) or odds.get(str(race_no)) or {}
    m = r.get(pool) or {}
    v = m.get(combo)
    return float(v) if v else None


def ranked(race_df: pd.DataFrame, by: str = "p_win") -> list[int]:
    """Horse numbers sorted by ``by`` descending (ties -> lower horse number)."""
    d = race_df.sort_values([by, "horse_no"], ascending=[False, True])
    return [int(h) for h in d["horse_no"]]


def _leg_for_race(race_df, pool, k, odds=None, value_filter=False, min_ev=1.0):
    """Pick k horses for an all-up leg; returns (selections, confidence, value) or None."""
    by = RANK_BY[pool]
    order = ranked(race_df, by)
    if len(order) < max(k, bets.POOL_SIZE[pool]):
        return None
    rn = int(race_df["race_no"].iloc[0])
    prob = dict(zip(race_df["horse_no"].astype(int), race_df[by].astype(float)))
    pw = dict(zip(race_df["horse_no"].astype(int), race_df["p_win"].astype(float)))
    sels = order[:k]
    if value_filter and pool == "PLA" and k == 1:
        cands = []
        for h in order:
            o = _odds_lookup(odds, rn, "PLA", str(h))
            if o is None and "pla_odds" in race_df.columns:
                v = race_df.loc[race_df["horse_no"] == h, "pla_odds"].iloc[0]
                o = float(v) if pd.notna(v) else None
            if o is not None and prob[h] * o >= min_ev:
                cands.append(h)
        if not cands:
            return None
        sels = [cands[0]]           # highest p_place among value horses
    # confidence = P(leg hits)
    if pool == "PLA":
        conf = 1 - np.prod([1 - prob[h] for h in sels])
    elif pool == "WIN":
        conf = sum(pw[h] for h in sels)
    elif pool == "QPL":
        conf = p_at_least_two([prob[h] for h in sels])
    elif pool == "QIN":
        conf = sum(pw[a] * pw[b] / max(1 - pw[a], 1e-9) for a in sels for b in sels if a != b)
    else:
        conf = sum(pw[h] for h in sels)
    # value = edge of the leg
    value = None
    if odds:
        if pool in ("PLA", "WIN"):
            evs = [prob[h] * o for h in sels if (o := _odds_lookup(odds, rn, pool, str(h)))]
            value = float(np.mean(evs)) if evs else None
    if value is None and "p_mkt" in race_df.columns:
        col_m = "p_mkt_place" if by == "p_place" else "p_mkt"
        pm = dict(zip(race_df["horse_no"].astype(int), race_df[col_m].astype(float)))
        edges = [prob[h] / pm[h] for h in sels if pm.get(h, 0) > 0]
        value = float(np.mean(edges)) if edges else None
    return sels, float(conf), value


def _race_table(df, pool, k, odds=None, value_filter=False, min_ev=1.0):
    rows = []
    for rn, g in df.groupby("race_no", sort=True):
        leg = _leg_for_race(g, pool, k, odds, value_filter, min_ev)
        if leg is None:
            continue
        rows.append({"race_no": int(rn), "selections": leg[0], "conf": leg[1],
                     "value": leg[2]})
    return rows


def _choose_races(table, n, rule="confidence", fixed_races=None):
    if len(table) < n and rule != "fixed":
        return None
    if rule in ("confidence",):
        pick = sorted(table, key=lambda r: (-r["conf"], r["race_no"]))[:n]
    elif rule == "value":
        def key(r):
            v = r["value"] if r["value"] is not None else 1.0
            return (-round(v, 9), -r["conf"], r["race_no"])
        pick = sorted(table, key=key)[:n]
    elif rule in ("first", "first6"):
        pick = sorted(table, key=lambda r: r["race_no"])[:n]
    elif rule in ("last", "last6"):
        pick = sorted(table, key=lambda r: r["race_no"])[-n:]
    elif rule == "fixed":
        want = list(fixed_races or [])
        byno = {r["race_no"]: r for r in table}
        pick = [byno[x] for x in want if x in byno]
        if len(pick) < n:
            return None
        pick = pick[:n]
    else:
        raise ValueError(f"unknown race_rule {rule!r}")
    return sorted(pick, key=lambda r: r["race_no"])


def _ticket(strategy, ticket_no, pool, spec, legs, unit_stake, notes=None):
    if spec:
        n = bets.ticket_units(pool, legs, spec)
    else:
        leg = legs[0]
        n = bets.count_units(pool, leg.get("selections", ()), leg.get("bankers", ()),
                             leg.get("positions"), leg.get("mode", "multiple"))
    return {"strategy": strategy, "ticket_no": ticket_no, "pool": pool, "spec": spec,
            "legs": legs, "unit_stake": float(unit_stake), "n_combos": int(n),
            "total_stake": round(float(unit_stake) * n, 2), "notes": notes or {}}


# ----------------------------------------------------------------- all-up builders
def build_allup(pred_df, pool="PLA", n_legs=6, spec="6x63", horses_per_leg=1,
                race_rule="confidence", fixed_races=None, unit_stake=10.0,
                odds=None, value_filter=False, min_ev=1.0, strategy=None,
                ticket_no=1, _prepped=False):
    """One all-up ticket: ``n_legs`` races chosen by ``race_rule``, the top
    ``horses_per_leg`` horses in each (by p_place for PLA/QPL, p_win otherwise)."""
    df = pred_df if _prepped else _prep(pred_df)
    table = _race_table(df, pool, horses_per_leg, odds, value_filter, min_ev)
    if value_filter and len(table) < n_legs:      # not enough value races: fill
        have = {r["race_no"] for r in table}
        extra = [r for r in _race_table(df, pool, horses_per_leg, odds) if r["race_no"] not in have]
        table = table + extra
    pick = _choose_races(table, n_legs, race_rule, fixed_races)
    if not pick:
        return None
    legs = [{"race_no": r["race_no"], "pool": pool, "selections": list(r["selections"])}
            for r in pick]
    name = strategy or f"{pool}_{spec}"
    return _ticket(name, ticket_no, pool, spec, legs, unit_stake,
                   {"race_rule": race_rule,
                    "conf": {r["race_no"]: round(r["conf"], 4) for r in pick}})


def build_multi_allup(pred_df, pool="QPL", n_tickets=3, legs_per_ticket=2, spec="2x3",
                      horses_per_leg=3, race_rule="confidence", pairing="chrono",
                      fixed_races=None, unit_stake=10.0, odds=None, strategy=None,
                      _prepped=False):
    """``n_tickets`` all-up tickets on distinct races (e.g. 3 x QPL 2x3 on 6 races).

    pairing: 'chrono' - chosen races sorted by race number, consecutive groups;
             'rank'   - best with 2nd best, 3rd with 4th, ...
    race_rule 'consecutive' picks the best non-overlapping groups of adjacent races."""
    df = pred_df if _prepped else _prep(pred_df)
    table = _race_table(df, pool, horses_per_leg, odds)
    need = n_tickets * legs_per_ticket
    if race_rule == "consecutive":
        byno = {r["race_no"]: r for r in table}
        groups = []
        for rn in sorted(byno):
            g = [rn + i for i in range(legs_per_ticket)]
            if all(x in byno for x in g):
                groups.append((g, float(np.prod([byno[x]["conf"] for x in g]))))
        best, best_score = None, -1.0
        for combo in combinations(groups, n_tickets):
            used = [x for g, _ in combo for x in g]
            if len(set(used)) != len(used):
                continue
            score = sum(s for _, s in combo)
            if score > best_score + 1e-15:
                best, best_score = combo, score
        if best is None:
            return []
        grouped = [[byno[x] for x in g] for g, _ in sorted(best, key=lambda t: t[0][0])]
    else:
        pick = _choose_races(table, need, race_rule, fixed_races)
        if not pick:
            return []
        if pairing == "rank":
            order = sorted(pick, key=lambda r: (-r["conf"], r["race_no"]))
        elif pairing == "chrono":
            order = sorted(pick, key=lambda r: r["race_no"])
        else:
            raise ValueError(f"unknown pairing {pairing!r}")
        grouped = [sorted(order[i * legs_per_ticket:(i + 1) * legs_per_ticket],
                          key=lambda r: r["race_no"]) for i in range(n_tickets)]
        grouped.sort(key=lambda g: g[0]["race_no"])
    name = strategy or f"{pool}_{spec}"
    out = []
    for i, g in enumerate(grouped, start=1):
        legs = [{"race_no": r["race_no"], "pool": pool, "selections": list(r["selections"])}
                for r in g]
        out.append(_ticket(name, i, pool, spec, legs, unit_stake,
                           {"race_rule": race_rule, "pairing": pairing,
                            "conf": {r["race_no"]: round(r["conf"], 4) for r in g}}))
    return out


def build_tickets(pred_df, odds=None, config=None) -> list[dict]:
    """The robot's fixed products for one meeting (see module docstring)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    unit = float(cfg.get("unit_stake", 10.0))
    df = _prep(pred_df)
    if "pla_odds" not in df.columns and odds:
        df["pla_odds"] = [_odds_lookup(odds, r, "PLA", str(h)) for r, h in zip(df["race_no"], df["horse_no"])]
    tickets: list[dict] = []
    for prod in cfg.get("products", []):
        c = cfg.get(prod) or {}
        if prod == "ADAPTIVE":
            ac = {"unit_stake": unit, **c}
            for rn, g in df.groupby("race_no", sort=True):
                tickets += decide_race(g, odds, ac)
            continue
        if prod == "ADAPTIVE_ALLUP":
            tickets += build_adaptive_allup(df, odds, {"unit_stake": unit, **c})
            continue
        if c.get("n_tickets", 1) > 1 or "legs_per_ticket" in c:
            tickets += build_multi_allup(
                df, pool=c["pool"], n_tickets=c.get("n_tickets", 3),
                legs_per_ticket=c.get("legs_per_ticket", 2), spec=c["spec"],
                horses_per_leg=c.get("horses_per_leg", 3), race_rule=c.get("race_rule", "confidence"),
                pairing=c.get("pairing", "chrono"), fixed_races=c.get("fixed_races"),
                unit_stake=c.get("unit_stake", unit), odds=odds, strategy=prod, _prepped=True)
        else:
            t = build_allup(
                df, pool=c["pool"], n_legs=c["n_legs"], spec=c["spec"],
                horses_per_leg=c.get("horses_per_leg", 1), race_rule=c.get("race_rule", "confidence"),
                fixed_races=c.get("fixed_races"), unit_stake=c.get("unit_stake", unit), odds=odds,
                value_filter=c.get("value_filter", False), min_ev=c.get("min_ev", 1.0),
                strategy=prod, _prepped=True)
            if t:
                tickets.append(t)
    return tickets


# ----------------------------------------------------------------- single-race builders
def single_spec_name(spec: dict) -> str:
    """{'pool':'TRI','bankers':1,'legs':4} -> 'TRI 膽1x腳4' etc."""
    p = spec["pool"]
    if "positions" in spec:
        return f"{p} 位置 " + "/".join("".join(str(x) for x in s) for s in spec["positions"])
    if "top" in spec:
        return f"{p} top{spec['top']}"
    if "box" in spec:
        return f"{p} 複式{spec['box']}"
    if "bankers" in spec:
        m = ""
        if p in bets.ORDERED_POOLS:
            m = "(膽定首位)" if spec.get("mode", "multiple") == "single" else "(膽任何位)"
        return f"{p} 膽{spec['bankers']}x腳{spec['legs']}{m}"
    if "value_ev" in spec:
        return f"{p} 值博EV≥{spec['value_ev']}"
    raise ValueError(f"bad spec {spec}")


def race_orders(race_df: pd.DataFrame) -> dict:
    """Pre-computed rankings of one race (used by single_selection; the backtest
    computes these once per race for speed)."""
    out = {"race_no": int(race_df["race_no"].iloc[0]),
           "p_win": ranked(race_df, "p_win"), "p_place": ranked(race_df, "p_place")}
    if "win_odds" in race_df.columns:
        ev = race_df["p_win"].astype(float) * race_df["win_odds"].astype(float)
        out["win_ev"] = {int(h): float(e) for h, e in zip(race_df["horse_no"], ev) if pd.notna(e)}
    return out


def single_selection(race_df, spec: dict, orders: dict | None = None):
    """Leg dict for a single-race spec, or None when the field is too small.

    spec forms (horse ranks are by p_place for PLA/QPL, p_win otherwise):
      {'pool':'WIN','top':1}                 top-n horses, one unit each
      {'pool':'QIN','box':4}                 複式 of the top 4
      {'pool':'TRI','bankers':1,'legs':4}    膽拖: top-1 banker + next 4 as legs
      {'pool':'TCE','bankers':1,'legs':3,'mode':'single'}  banker fixed 1st
      {'pool':'TCE','positions':[[1],[2,3],[2,3,4]]}       ranks per position
      {'pool':'WIN','value_ev':1.2}          horse with max p_win*win_odds if >= threshold
    ``orders`` (from race_orders) may be given instead of / in addition to race_df.
    """
    if orders is None:
        orders = race_orders(race_df)
    pool = spec["pool"]
    by = spec.get("rank_by", RANK_BY[pool])
    order = orders[by]
    rn = orders["race_no"]
    k = bets.POOL_SIZE[pool]
    if "top" in spec:
        n = spec["top"]
        if len(order) < n:
            return None
        return {"race_no": rn, "pool": pool, "selections": order[:n]}
    if "box" in spec:
        n = spec["box"]
        if len(order) < max(n, k) or n < k:
            return None
        return {"race_no": rn, "pool": pool, "selections": order[:n]}
    if "bankers" in spec:
        b, l = spec["bankers"], spec["legs"]
        if len(order) < b + l or b + l < k:
            return None
        leg = {"race_no": rn, "pool": pool, "bankers": order[:b], "selections": order[b:b + l]}
        if pool in bets.ORDERED_POOLS:
            leg["mode"] = spec.get("mode", "multiple")
        return leg
    if "positions" in spec:
        mx = max(max(s) for s in spec["positions"])
        if len(order) < mx:
            return None
        return {"race_no": rn, "pool": pool,
                "positions": [[order[i - 1] for i in s] for s in spec["positions"]]}
    if "value_ev" in spec:
        if pool != "WIN" or not orders.get("win_ev"):
            return None
        h, e = max(orders["win_ev"].items(), key=lambda kv: (kv[1], -kv[0]))
        if e < spec["value_ev"]:
            return None
        return {"race_no": rn, "pool": pool, "selections": [h]}
    raise ValueError(f"bad spec {spec}")


def build_single(race_df, spec: dict, unit_stake=10.0, strategy=None, ticket_no=1):
    """Single-race ticket for one race's prediction rows (see single_selection)."""
    leg = single_selection(race_df, spec)
    if leg is None:
        return None
    return _ticket(strategy or single_spec_name(spec), ticket_no, spec["pool"], None,
                   [leg], unit_stake, {"spec": spec})


def leg_combos(leg: dict) -> list[str]:
    """All unit combinations of a single-race leg dict."""
    return bets.single_combos(leg["pool"], leg.get("selections", ()), leg.get("bankers", ()),
                              leg.get("positions"), leg.get("mode", "multiple"))


# =====================================================================================
# ADAPTIVE: structure decided per race from the model's probabilities + odds
# =====================================================================================
# HKJC deduction per pool = 100% - share paid as dividends, from the HKJC "Local Pools"
# page (fetched 2026-10-19):
#   https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/
#   WIN/PLA/QIN/QPL/DBL 82.5% paid -> 17.5%; FCT 80.5% -> 19.5%; TRI 77% -> 23%;
#   TCE/FF/QTT/Treble 75% -> 25%.
TAKEOUT = {"WIN": 0.175, "PLA": 0.175, "QIN": 0.175, "QPL": 0.175, "FCT": 0.195,
           "TRI": 0.23, "TCE": 0.25, "FF": 0.25, "QTT": 0.25}
MIN_DIVIDEND = 10.0          # HKJC never pays less than the $10 stake back on a winning unit

# Candidate structures (kept modest).  Horse ranks: PLA/QPL by model place prob,
# others by model win prob.
ADAPTIVE_CANDIDATES = (
    [{"pool": "WIN", "top": 1}, {"pool": "PLA", "top": 1}, {"pool": "PLA", "top": 2},
     {"pool": "QPL", "box": 3}, {"pool": "QPL", "box": 4}]
    + [{"pool": "QPL", "bankers": 1, "legs": n} for n in range(2, 6)]
    + [{"pool": "QIN", "box": 2}, {"pool": "QIN", "box": 3}]
    + [{"pool": "QIN", "bankers": 1, "legs": n} for n in range(2, 6)]
    + [{"pool": "TRI", "bankers": 1, "legs": n} for n in range(3, 7)]
    + [{"pool": "TRI", "bankers": 2, "legs": n} for n in range(2, 6)]
    + [{"pool": "TRI", "box": 4}, {"pool": "TRI", "box": 5}]
    + [{"pool": "FF", "bankers": 2, "legs": n} for n in range(3, 6)]
)

# Defaults = the setting tuned on the 2024-25 season ONLY with source wf-combined-v1
# (hkjc.backtest.adaptive_backtest): tau/h_min chosen by in-sample ROI among settings
# with >= 100 bets; lam2/lam3 (Stern/Henery-style discount for 2nd/3rd place) by
# maximum likelihood on 2024-25 results; dividend calibration
# log(div) = a + b*log(10*(1-takeout)/P_market) fitted on 2024-25 winning combinations.
# The all-up thresholds never qualified often enough in 2024-25 (< 20 meetings) for any
# setting, so the middle of the grid is used and the product will rarely bet.
ADAPTIVE_DEFAULT = {
    "tau": 0.0, "h_min": 0.30, "max_stake": 200.0, "unit_stake": 10.0,
    "lam2": 0.85, "lam3": 0.70,
    "calib": {"PLA": (0.044, 0.986), "QIN": (0.207, 0.961), "QPL": (0.064, 0.982),
              "TRI": (0.184, 0.974), "FF": (0.319, 0.961)},
    "candidates": None,
}
ADAPTIVE_ALLUP_DEFAULT = {
    "pla_min_p": 0.55, "qpl_min_hit": 0.40, "min_edge": 1.05, "tau": 0.0,
    "max_legs": 6, "unit_stake": 10.0, "pools": ("PLA", "QPL"), "qpl_horses": 3,
    "lam2": 0.85, "lam3": 0.70, "calib": ADAPTIVE_DEFAULT["calib"],
}


class RaceProbs:
    """Finishing-order probabilities for one race from win probabilities.

    Generalised Harville (Stern/Henery-style): the k-th place is drawn from the
    remaining horses with weights p**lam_k (lam=1 -> plain Harville).
    n_places = 3 with >= 7 runners else 2 (HKJC place rule)."""

    def __init__(self, horses, p_win, lam2=1.0, lam3=1.0):
        self.horses = [int(h) for h in horses]
        self.ix = {h: i for i, h in enumerate(self.horses)}
        p = np.asarray(p_win, float)
        p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
        s = p.sum()
        self.p = p / s if s > 0 else np.full(len(p), 1.0 / max(len(p), 1))
        self.n = len(self.p)
        self.n_places = 3 if self.n >= 7 else 2
        eps = 1e-12
        q2, q3 = self.p ** lam2, self.p ** lam3
        self.q3 = q3
        self.S3 = q3.sum()
        m2 = self.p[:, None] * q2[None, :] / np.maximum(q2.sum() - q2[:, None], eps)
        np.fill_diagonal(m2, 0.0)
        self.m2 = m2                                   # P(a 1st, b 2nd)
        t3 = m2[:, :, None] * q3[None, None, :] / np.maximum(self.S3 - q3[:, None, None] - q3[None, :, None], eps)
        i = np.arange(self.n)
        t3[i, :, i] = 0.0
        t3[:, i, i] = 0.0
        t3[i, i, :] = 0.0
        self.t3 = t3                                   # P(a,b,c in order)
        self._t4 = None
        if self.n_places == 3:
            self.place = t3.sum(axis=(1, 2)) + t3.sum(axis=(0, 2)) + t3.sum(axis=(0, 1))
            m01, m02, m12 = t3.sum(axis=2), t3.sum(axis=1), t3.sum(axis=0)
            self.pair3 = m01 + m01.T + m02 + m02.T + m12 + m12.T   # P(a and b both top 3)
        else:
            self.place = self.p + m2.sum(axis=0)
            self.pair3 = m2 + m2.T

    @property
    def t4(self):
        if self._t4 is None:
            q = self.q3
            den = self.S3 - q[:, None, None, None] - q[None, :, None, None] - q[None, None, :, None]
            t4 = self.t3[:, :, :, None] * q[None, None, None, :] / np.maximum(den, 1e-12)
            i = np.arange(self.n)
            t4[i, :, :, i] = 0.0
            t4[:, i, :, i] = 0.0
            t4[:, :, i, i] = 0.0
            self._t4 = t4
        return self._t4

    def _idx(self, combo):
        return [self.ix[int(h)] for h in bets.parse_combination(combo)]

    def prob(self, pool, combo) -> float:
        """Probability that ``combo`` (canonical string) wins in ``pool``."""
        try:
            x = self._idx(combo)
        except KeyError:
            return 0.0
        if pool == "WIN":
            return float(self.p[x[0]])
        if pool == "PLA":
            return float(self.place[x[0]])
        if pool == "QIN":
            a, b = x
            return float(self.m2[a, b] + self.m2[b, a])
        if pool == "QPL":
            return float(self.pair3[x[0], x[1]]) if self.n_places == 3 else 0.0
        if pool == "FCT":
            return float(self.m2[x[0], x[1]])
        if pool == "TCE":
            return float(self.t3[x[0], x[1], x[2]])
        if pool == "TRI":
            a, b, c = x
            t = self.t3
            return float(t[a, b, c] + t[a, c, b] + t[b, a, c] + t[b, c, a] + t[c, a, b] + t[c, b, a])
        if pool == "FF":
            t4 = self.t4
            from itertools import permutations as _pm
            return float(sum(t4[a, b, c, d] for a, b, c, d in _pm(x)))
        if pool == "QTT":
            return float(self.t4[x[0], x[1], x[2], x[3]])
        raise ValueError(pool)

    def hit_prob(self, pool, combos) -> float:
        """P(at least one of ``combos`` wins).  Exclusive pools -> sum; PLA/QPL via
        the ordered-top-3 distribution."""
        if pool in ("PLA", "QPL"):
            if pool == "QPL" and self.n_places < 3:
                return 0.0
            k = self.n_places
            sets = [set(self._idx(c)) for c in combos]
            if k == 3:
                n = self.n
                A, B, C = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
                mask = np.zeros_like(self.t3, dtype=bool)
                for s in sets:
                    m = np.ones_like(mask)
                    for h in s:
                        m &= (A == h) | (B == h) | (C == h)
                    mask |= m
                return float(self.t3[mask].sum())
            n = self.n
            A, B = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            mask = np.zeros_like(self.m2, dtype=bool)
            for s in sets:
                m = np.ones_like(mask)
                for h in s:
                    m &= (A == h) | (B == h)
                mask |= m
            return float(self.m2[mask].sum())
        return float(min(1.0, sum(self.prob(pool, c) for c in combos)))


def race_profile(race_df: pd.DataFrame) -> dict:
    """Shape of a race from the model: p1>=p2>=p3 (win), gap, top-3 cumulative,
    entropy, field size, top pick's place prob, model/market edge of the top picks."""
    d = race_df.sort_values(["p_win", "horse_no"], ascending=[False, True])
    p = d["p_win"].astype(float).to_numpy()
    p = p / p.sum() if p.sum() > 0 else p
    pad = np.concatenate([p, np.zeros(3)])
    prof = {"race_no": int(d["race_no"].iloc[0]), "field": int(len(p)),
            "p1": float(pad[0]), "p2": float(pad[1]), "p3": float(pad[2]),
            "gap": float(pad[0] - pad[1]), "top3": float(pad[:3].sum()),
            "entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
            "top_horse": int(d["horse_no"].iloc[0]),
            "top_p_place": float(d["p_place"].iloc[0]) if "p_place" in d else None}
    if "p_mkt" in d.columns:
        pm = d["p_mkt"].astype(float).to_numpy()
        for k in range(3):
            prof[f"edge{k + 1}"] = float(p[k] / pm[k]) if k < len(p) and pm[k] > 0 else None
    return prof


def _ensure_market(df: pd.DataFrame, odds=None) -> pd.DataFrame:
    """Add win_odds (from column win_odds_live or live odds dict) and p_mkt."""
    df = df.copy()
    if "win_odds" not in df.columns or df["win_odds"].isna().all():
        if "win_odds_live" in df.columns:
            df["win_odds"] = df["win_odds_live"]
        elif odds:
            df["win_odds"] = [_odds_lookup(odds, r, "WIN", str(h)) for r, h in zip(df["race_no"], df["horse_no"])]
    if "win_odds" in df.columns:
        df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
        df["p_mkt"] = df.groupby("race_no")["win_odds"].transform(lambda s: market_probs(s.values))
    return df


class DividendEstimator:
    """Estimated dividend per $10 for a combination.

    1. live odds (decimal per $1) when given -> odds * 10 (WIN/PLA/QIN/QPL pre-race);
    2. WIN with a ``win_odds`` column -> 10 * win_odds;
    3. otherwise market-implied: 10 * (1 - takeout) / P_market(combo) (Harville on
       1/win odds), optionally calibrated log(div) = a + b*log(raw) per pool
       (fitted on the 2024-25 winning combinations).  Never below $10."""

    def __init__(self, race_df, odds_race=None, calib=None, lam2=1.0, lam3=1.0):
        self.odds = odds_race or {}
        self.calib = calib or {}
        self.win_odds = {}
        self.mkt = None
        if "win_odds" in race_df.columns:
            wo = pd.to_numeric(race_df["win_odds"], errors="coerce")
            self.win_odds = {int(h): float(o) for h, o in zip(race_df["horse_no"], wo) if pd.notna(o) and o > 0}
            if len(self.win_odds) >= 2 and "p_mkt" in race_df.columns:
                # same place-discount as the model so that model/market edges compare like with like
                self.mkt = RaceProbs(race_df["horse_no"], race_df["p_mkt"].fillna(0).to_numpy(), lam2, lam3)

    def raw_market(self, pool, combo):
        if self.mkt is None:
            return None
        pm = self.mkt.prob(pool, combo)
        if pm <= 0:
            return None
        return 10.0 * (1 - TAKEOUT[pool]) / max(pm, 1e-6)

    def __call__(self, pool, combo):
        o = (self.odds.get(pool) or {}).get(combo)
        if o:
            return max(MIN_DIVIDEND, float(o) * 10.0)
        if pool == "WIN" and self.win_odds:
            o = self.win_odds.get(int(combo))
            return max(MIN_DIVIDEND, o * 10.0) if o else None
        raw = self.raw_market(pool, combo)
        if raw is None:
            return None
        a, b = self.calib.get(pool, (0.0, 1.0))
        return max(MIN_DIVIDEND, float(np.exp(a + b * np.log(raw))))


def race_candidates(race_df, odds_race=None, cfg=None, probs=None, est=None) -> list[dict]:
    """Evaluate every candidate structure for one race.

    Returns [{spec, name, pool, leg, combos, n_units, stake, hit, exp_return,
    exp_roi}], exp_return = sum_c P_model(c) * est_div(c) / 10 per $1 of a unit,
    exp_roi = exp_return / n_units - 1.  Candidates whose pool cannot be priced
    (no odds / no market) are dropped."""
    cfg = {**ADAPTIVE_DEFAULT, **(cfg or {})}
    df = race_df[race_df["p_win"].notna()]
    if len(df) < 4:
        return []
    probs = probs or RaceProbs(df["horse_no"], df["p_win"].to_numpy(), cfg["lam2"], cfg["lam3"])
    est = est or DividendEstimator(df, odds_race, cfg.get("calib"), cfg["lam2"], cfg["lam3"])
    rn = int(df["race_no"].iloc[0])
    orders = {"race_no": rn,
              "p_win": [probs.horses[i] for i in np.lexsort((probs.horses, -probs.p))],
              "p_place": [probs.horses[i] for i in np.lexsort((probs.horses, -probs.place))]}
    out = []
    unit = float(cfg["unit_stake"])
    for spec in cfg.get("candidates") or ADAPTIVE_CANDIDATES:
        pool = spec["pool"]
        if pool == "QPL" and probs.n_places < 3:
            continue                                   # QPL not offered with < 7 runners
        leg = single_selection(None, spec, orders=orders)
        if leg is None:
            continue
        combos = leg_combos(leg)
        er = 0.0
        ok = True
        for c in combos:
            d = est(pool, c)
            if d is None:
                ok = False
                break
            er += probs.prob(pool, c) * d / 10.0
        if not ok:
            continue
        n = len(combos)
        out.append({"spec": spec, "name": single_spec_name(spec), "pool": pool, "leg": leg,
                    "combos": combos, "n_units": n, "stake": unit * n,
                    "hit": probs.hit_prob(pool, combos), "exp_return": er, "exp_roi": er / n - 1})
    return out


def choose_candidate(cands, tau=0.1, h_min=0.15, max_stake=None):
    """Best expected-ROI candidate with exp_roi >= tau and hit >= h_min (None = skip)."""
    ok = [c for c in cands if c["exp_roi"] >= tau and c["hit"] >= h_min
          and (max_stake is None or c["stake"] <= max_stake + 1e-9)]
    if not ok:
        return None
    return max(ok, key=lambda c: (round(c["exp_roi"], 12), c["hit"], -c["n_units"]))


def decide_race(race_df, odds=None, cfg=None) -> list[dict]:
    """ADAPTIVE: the ticket(s) for one race, or [] to skip.

    ``odds``: live odds of this race {pool: {combo: odds}} or the meeting dict
    {race_no: {...}}.  Live WIN odds are also used for market probabilities."""
    cfg = {**ADAPTIVE_DEFAULT, **(cfg or {})}
    if race_df.empty:
        return []
    rn = int(race_df["race_no"].iloc[0])
    if odds and rn in odds and isinstance(odds.get(rn), dict):
        odds_race = odds[rn]
    elif odds and str(rn) in odds:
        odds_race = odds[str(rn)]
    else:
        odds_race = odds if odds and any(k in bets.POOL_SIZE for k in odds) else None
    df = _ensure_market(race_df, {rn: odds_race} if odds_race else None)
    if "scratched" in df.columns:
        df = df[df["scratched"].fillna(0).astype(int) == 0]
    cands = race_candidates(df, odds_race, cfg)
    best = choose_candidate(cands, cfg["tau"], cfg["h_min"], cfg.get("max_stake"))
    if best is None:
        return []
    prof = race_profile(df)
    t = _ticket("ADAPTIVE", rn, best["pool"], None, [best["leg"]], cfg["unit_stake"],
                {"structure": best["name"], "exp_roi": round(best["exp_roi"], 4),
                 "hit_prob": round(best["hit"], 4),
                 "profile": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in prof.items()}})
    return [t]


def adaptive_plan(pred_df, odds=None, cfg=None) -> list[dict]:
    """Per race: profile, best candidate overall, and the chosen one (None = skip).
    For reports."""
    cfg = {**ADAPTIVE_DEFAULT, **(cfg or {})}
    df = _ensure_market(_prep(pred_df), odds)
    out = []
    for rn, g in df.groupby("race_no", sort=True):
        odds_race = (odds or {}).get(int(rn)) if odds else None
        cands = race_candidates(g, odds_race, cfg)
        best_any = max(cands, key=lambda c: c["exp_roi"]) if cands else None
        chosen = choose_candidate(cands, cfg["tau"], cfg["h_min"], cfg.get("max_stake"))
        out.append({"race_no": int(rn), "profile": race_profile(g), "best": best_any, "chosen": chosen,
                    "n_candidates": len(cands)})
    return out


# ----------------------------------------------------------------- adaptive all-up
def allup_leg_info(race_df, pool, probs, est, n_horses=3):
    """One candidate all-up leg for a race: selections, hit prob, model/market edge,
    expected factor sum E = sum_c P(c) * div_c / 10 and combos."""
    if pool == "QPL" and probs.n_places < 3:
        return None
    order = [probs.horses[i] for i in np.lexsort((probs.horses, -probs.place))]
    sels = order[:1] if pool == "PLA" else order[:n_horses]
    if len(sels) < (1 if pool == "PLA" else 2):
        return None
    combos = bets.multiple(pool, sels)
    e = 0.0
    for c in combos:
        d = est(pool, c)
        if d is None:
            return None
        e += probs.prob(pool, c) * d / 10.0
    hit = probs.hit_prob(pool, combos)
    edge = None
    if est.mkt is not None:
        mhit = est.mkt.hit_prob(pool, combos)
        edge = hit / mhit if mhit > 0 else None
    return {"race_no": int(race_df["race_no"].iloc[0]), "pool": pool, "selections": sels,
            "combos": combos, "hit": hit, "edge": edge, "E": e}


def best_allup(legs, tau=0.0, max_legs=6):
    """Choose number of legs (2..max_legs, highest expected factor per unit first) and the official
    NxM formula maximising expected ROI.  Legs are independent races, so a unit's
    expected return is the product of its legs' expected factors and
    E[payout]/unit_stake = sum_{subset} prod_{i in subset} E_i,
    units = sum_{subset} prod n_i.  Returns (legs, spec, exp_roi) or None."""
    legs = sorted(legs, key=lambda l: (-l["E"] / len(l["combos"]), l["race_no"]))
    best = None
    for k in range(2, min(max_legs, len(legs)) + 1):
        use = sorted(legs[:k], key=lambda l: l["race_no"])
        for spec in bets.allup_specs(k):
            subs = bets.allup_subsets(k, spec)
            er = sum(float(np.prod([use[i]["E"] for i in s])) for s in subs)
            units = sum(int(np.prod([len(use[i]["combos"]) for i in s])) for s in subs)
            roi = er / units - 1
            if best is None or roi > best[2] + 1e-12:
                best = (use, spec, roi)
    if best is None or best[2] < tau:
        return None
    return best


def build_adaptive_allup(pred_df, odds=None, cfg=None) -> list[dict]:
    """ADAPTIVE_ALLUP: PLA / QPL parlays only from qualifying races
    (PLA: top place prob >= pla_min_p; QPL: top-3 leg hit >= qpl_min_hit; both
    model/market edge >= min_edge).  < 2 qualifying races -> no parlay."""
    cfg = {**ADAPTIVE_ALLUP_DEFAULT, **(cfg or {})}
    df = _ensure_market(_prep(pred_df), odds)
    infos = {p: [] for p in cfg["pools"]}
    for rn, g in df.groupby("race_no", sort=True):
        if len(g) < 4:
            continue
        probs = RaceProbs(g["horse_no"], g["p_win"].to_numpy(), cfg["lam2"], cfg["lam3"])
        est = DividendEstimator(g, (odds or {}).get(int(rn)) if odds else None, cfg.get("calib"),
                                cfg["lam2"], cfg["lam3"])
        for pool in cfg["pools"]:
            li = allup_leg_info(g, pool, probs, est, cfg["qpl_horses"])
            if li:
                infos[pool].append(li)
    return adaptive_allup_from_legs(infos, cfg)


def adaptive_allup_from_legs(infos, cfg) -> list[dict]:
    cfg = {**ADAPTIVE_ALLUP_DEFAULT, **(cfg or {})}
    out = []
    for pool, legs in infos.items():
        thr = cfg["pla_min_p"] if pool == "PLA" else cfg["qpl_min_hit"]
        q = [l for l in legs if l["hit"] >= thr and (l["edge"] or 0) >= cfg["min_edge"]]
        if len(q) < 2:
            continue
        b = best_allup(q, cfg["tau"], cfg["max_legs"])
        if b is None:
            continue
        use, spec, roi = b
        tl = [{"race_no": l["race_no"], "pool": pool, "selections": list(l["selections"])} for l in use]
        out.append(_ticket(f"ADAPTIVE_ALLUP", len(out) + 1, pool, spec, tl, cfg["unit_stake"],
                           {"exp_roi": round(roi, 4), "n_legs": len(use),
                            "legs": {l["race_no"]: {"hit": round(l["hit"], 4),
                                                    "edge": round(l["edge"] or 0, 4)} for l in use}}))
    return out
