"""今日精選 (third version) - analysis-driven daily picks across ALL HKJC pools.  PAPER ONLY.

Nothing is fixed in advance: for every race (and race pair / triple / all-up) the engine prices
every structure of every pool from the model's probabilities, and only items whose estimated
return AND hit probability clear thresholds tuned on the 2024-25 season (walk-forward
predictions ``wf-combined-v1``; 2025-26 + 2026-27 out of sample) are graded 「精選」.  When
nothing qualifies, ``items`` holds at most 3 nearest candidates graded 「參考」 (clearly marked
未過門檻) and method_note says 「今日冇玩法過門檻，冇精選」.

Public API
----------
make_picks(pred_df, odds, cfg=None) -> dict              SPEC_V2 「今日精選（第三版）」 picks dict
enumerate_candidates(pred_df, odds, cfg=None) -> (cands, ctx)   every priced candidate
select_items(cands, cfg=None) -> (selected, reference)  grading + de-duplication + day caps
settle_item(item, data) -> {hit, payout, profit, detail, stake}  real-dividend settlement
item_ticket(item) -> SPEC ticket dict                    (what the daily_picks robot buys)
save_picks(conn, run_id, date, venue, picks) / load_picks(conn, run_id=None, date, venue)
backtest_picks(conn, source='wf-combined-v1') -> dict     see the BACKTEST section

Pools and structures (horse ranks by model p_win; PLA / QPL by model p_place)
------------------------------------------------------------------------------
WIN  單注 top-1..4               PLA  單注 top-1..4, 2 隻
QIN / QPL  複式 2-4, 1 膽拖 2-5 腳 (QPL needs >= 7 starters)
FCT  單注 1-2, 複式 2-4, 首選定頭拖 2-4 腳, 1 膽拖 2-4 腳(膽任何位)
TRI  複式 3-6, 1 膽拖 3-6, 2 膽拖 2-6         TCE  單注 1-2-3, 複式 3-4, 頭兩名定位拖 2-4,
     首選定頭拖 3-5, 頭兩選互換拖 2-4         FF   複式 4-6, 1 膽拖 4-6, 2 膽拖 3-6, 3 膽拖 2-5
QTT  單注 1-2-3-4, 頭三名定位拖 2-4, 1-2 定位 + 3/4 名複式 (2-3 隻), 1-2 互換 + 3-4 互換,
     首選定頭 + 3 隻, 複式 4
DBL  孖寶 (r-1, r) for every race pair: first leg 1-2 horses x last leg 1-3 horses
TBL  三寶 legs (3,4,5) and (n-2,n-1,n): 1-2 x 1-2 x 1-3 horses
All-up WIN / PLA / QIN / QPL 過關: the best k legs (by expected factor per unit) for k = 2..6,
     every official NxM formula, the one with the best expected ROI that fits the stake cap.
Field-size rules [R3 3.9(b)]: QPL >= 7, FCT/TCE >= 4, TRI >= 5, FF/QTT >= 6 starters.
Items above ``max_item_stake`` ($300) are not considered.

Probabilities: discounted Harville (strategy.RaceProbs, lam2 0.85 / lam3 0.70 fitted on
2024-25) from the model's p_win; PLA uses the model's own p_place (入三甲機率).  Legs of
multi-race bets are independent races.  DBL / TBL pay the winners of the earlier legs and the
1st OR 2nd of the last leg (two dividends - see bets.settle_multi_race).
Dividends (per $10): live odds where they exist (WIN/PLA/QIN/QPL; FCT/TRI/FF/DBL when the
pre-race exotic snapshot ran - a combination missing from a capped live book is floored at the
book's highest stored odds), otherwise market-implied from the WIN odds,
10 x (1 - takeout) / P_market(combo), calibrated log(div) = a + b log(raw) per pool on the
2024-25 winning combinations (strategy.DividendEstimator; DBL/TBL with separate calibrations for
the winner and the runner-up dividend).  Takeout: WIN/PLA/QIN/QPL/DBL 17.5%, FCT 19.5%,
TRI 23%, TCE/FF/QTT/TBL 25% (HKJC Local Pools page, see strategy.TAKEOUT).
With no odds at all (pools not open yet) no pool can be priced -> nothing is graded and
odds_note says so.

Grading: a candidate is 「精選」 when its pool family (WIN PLA QIN QPL FCT TRI TCE FF QTT DBL TBL
ALLUP) is enabled and exp_roi >= tau_family and hit_prob >= h_family (THRESHOLDS; a family whose
best 2024-25 setting did not make money is disabled).  Candidates are ranked by their margin over
their own family's tau; per race at most ``max_per_race`` items and one per pool group
({WIN,PLA} {QIN,QPL,FCT} {TRI,TCE,FF,QTT}); multi-race items of the same pool may not share a
race; at most one all-up per base pool; day caps ``max_items`` (8) and ``max_total`` ($1,500).

CAVEAT: the backtest prices with FINAL win odds (known only at the off) and estimates every
other pool from them - live picks use odds ~2 hours before race 1, so the historical results are
optimistic.  Model probabilities are walk-forward (trained only on earlier meetings).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import bets, strategy as st

log = logging.getLogger("hkjc.picks")
HKT = ZoneInfo("Asia/Hong_Kong")

UNIT = 10.0
LAM2 = st.ADAPTIVE_DEFAULT["lam2"]
LAM3 = st.ADAPTIVE_DEFAULT["lam3"]
TAKEOUT = {**st.TAKEOUT, "DBL": 0.175, "TBL": 0.25}
SINGLE_POOLS = ("WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT")
MULTI_POOLS = ("DBL", "TBL")
ALLUP_POOLS = ("WIN", "PLA", "QIN", "QPL")
FAMILIES = SINGLE_POOLS + MULTI_POOLS + ("ALLUP",)
CAPPED_LIVE = {"FCT", "TRI", "FF", "DBL"}          # live books trimmed at the 999 display cap
MIN_FIELD = {"WIN": 4, "PLA": 4, "QIN": 4, "QPL": 7, "FCT": 4, "TCE": 4, "TRI": 5, "FF": 6, "QTT": 6}
POOL_ZH = {"WIN": "獨贏", "PLA": "位置", "QIN": "連贏", "QPL": "位置Q", "FCT": "二重彩", "TRI": "單T",
           "TCE": "三重彩", "FF": "四連環", "QTT": "四重彩", "DBL": "孖寶", "TBL": "三寶"}
ALLUP_ZH = {"WIN": "獨贏過關", "PLA": "位置過關", "QIN": "連贏過關", "QPL": "位置Q過關"}
GROUP = {"WIN": "A", "PLA": "A", "QIN": "B", "QPL": "B", "FCT": "B", "TRI": "C", "TCE": "C", "FF": "C", "QTT": "C"}
SRC_ZH = {"live": "即時賠率", "est": "估計賠率"}
NO_PICKS_NOTE = "今日冇玩法過門檻，冇精選"
TITLE = "今日精選"

# Dividend calibration log(div) = a + b*log(raw market estimate), fitted on the 2024-25 winning
# combinations (fit_calibration below; backtest_picks re-fits and reports the values).
# PLA/QIN/QPL/TRI/FF: the ADAPTIVE values (strategy.ADAPTIVE_DEFAULT); FCT/TCE/QTT and the
# DBL/TBL winner (1) / runner-up (2) dividends: fitted for this module.
# Fitted 2024-25 (n winning combinations): FCT 850, TCE 854, QTT 854, DBL 726 / 741, TBL 166 / 169
# (the refit reproduces the ADAPTIVE PLA/QIN/QPL/TRI/FF values exactly).  The DBL/TBL runner-up
# dividend is only ~17-19% of its "fair" market value (median ratio) - most of the net pool goes
# to the winner combination.
CALIB = {**st.ADAPTIVE_DEFAULT["calib"],
         "FCT": (0.311, 0.952), "TCE": (0.114, 0.986), "QTT": (0.511, 0.955)}
MR_CALIB = {("DBL", 1): (0.392, 0.93), ("DBL", 2): (-1.845, 1.014),
            ("TBL", 1): (0.441, 0.944), ("TBL", 2): (-1.559, 0.985)}

# Per-family bias correction (fit_adjustments, 2024-25 candidates ONLY): realised / estimated
# expected return (the single largest payout removed) and realised / estimated hit rate over
# every priced candidate of the family.  e.g. QTT: the discounted-Harville 4th place and the
# QTT dividend estimate overstate the expected return by ~1/3, so QTT estimates are scaled down.
RET_ADJ = {"WIN": 0.92, "PLA": 0.983, "QIN": 0.944, "QPL": 0.957, "FCT": 0.944, "TRI": 1.015, "TCE": 0.907,
           "FF": 0.936, "QTT": 0.62, "DBL": 0.963, "TBL": 0.873, "ALLUP": 0.73}
HIT_ADJ = {"WIN": 0.952, "PLA": 0.982, "QIN": 0.96, "QPL": 0.957, "FCT": 0.973, "TRI": 0.97, "TCE": 0.992,
           "FF": 0.869, "QTT": 0.679, "DBL": 0.927, "TBL": 1.018, "ALLUP": 0.91}

# Thresholds per pool family, tuned on 2024-25 ONLY (backtest_picks; data/picks_backtest.json).
# None = family disabled (no setting with enough 2024-25 bets made money).
# 2024-25 tuning (robust ROI = largest single payout removed): QIN +5.0% (141 items), FCT +5.4%
# (102), DBL +69.7% (65), TBL +78.6% (115); WIN/PLA/QPL/TRI/TCE/FF/QTT/all-ups lost money in every
# eligible setting.  Out of sample these four did NOT hold up (see reports/回測報告.md).
THRESHOLDS = {
    "WIN": None, "PLA": None, "QIN": {"tau": 0.0, "h": 0.1}, "QPL": None,
    "FCT": {"tau": 0.05, "h": 0.1}, "TRI": None, "TCE": None, "FF": None, "QTT": None,
    "DBL": {"tau": 0.3, "h": 0.05}, "TBL": {"tau": 0.1, "h": 0.05}, "ALLUP": None,
}
TUNE_TAU = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
TUNE_H = {"WIN": (0.1, 0.2, 0.3), "PLA": (0.3, 0.45, 0.6), "QIN": (0.1, 0.2, 0.3),
          "QPL": (0.2, 0.35, 0.5), "FCT": (0.05, 0.1, 0.2), "TRI": (0.05, 0.1, 0.2),
          "TCE": (0.02, 0.05, 0.1), "FF": (0.02, 0.05, 0.1), "QTT": (0.01, 0.02, 0.05),
          "DBL": (0.05, 0.1, 0.2), "TBL": (0.02, 0.05, 0.1), "ALLUP": (0.05, 0.15, 0.3)}
TUNE_MIN_N = {**{p: 60 for p in SINGLE_POOLS}, "DBL": 40, "TBL": 20, "ALLUP": 20}
TUNE_MIN_EXP_HITS = 10.0        # a setting must expect >= 10 winning bets in 2024-25
REF_DEFAULT_H = {f: TUNE_H[f][1] for f in FAMILIES}   # 參考 ranking for disabled families

DEFAULT_CONFIG = {
    "unit_stake": UNIT, "max_items": 8, "max_total": 1500.0, "max_item_stake": 300.0,
    "max_per_race": 2, "max_allups": 2, "n_reference": 3, "allup_max_legs": 6,
    "lam2": LAM2, "lam3": LAM3, "calib": CALIB, "mr_calib": MR_CALIB,
    "thresholds": THRESHOLDS, "families": FAMILIES, "ret_adj": RET_ADJ, "hit_adj": HIT_ADJ,
}


def _cfg(cfg=None) -> dict:
    out = dict(DEFAULT_CONFIG)
    out["thresholds"] = dict(THRESHOLDS)
    for k, v in (cfg or {}).items():
        if k == "thresholds" and v is not None:
            out["thresholds"].update(v)
        else:
            out[k] = v
    return out


# =====================================================================================
# structures
# =====================================================================================
def _r(a, b):
    return list(range(a, b + 1))


def _structures():
    """(pool, kind, arg, structure_zh) - ranks are 1-based within the pool's ranking."""
    S = []
    for i in range(1, 5):
        S.append(("WIN", "top", i, "單注"))
        S.append(("PLA", "top", i, "單注"))
    S.append(("PLA", "box", 2, "2 隻"))
    for p in ("QIN", "QPL"):
        for k in (2, 3, 4):
            S.append((p, "box", k, "單注" if k == 2 else f"複式 {k} 隻"))
        for n in range(2, 6):
            S.append((p, "bank", (1, n), f"1 膽拖 {n} 腳"))
    S.append(("FCT", "pos", [[1], [2]], "單注（首選勝、次選亞）"))
    for k in (2, 3, 4):
        S.append(("FCT", "box", k, f"複式 {k} 隻"))
    for n in (2, 3, 4):
        S.append(("FCT", "pos", [[1], _r(2, n + 1)], f"首選定頭位拖 {n} 腳"))
        S.append(("FCT", "bank", (1, n), f"1 膽拖 {n} 腳（膽任何位）"))
    for k in (3, 4, 5, 6):
        S.append(("TRI", "box", k, f"複式 {k} 隻"))
    for n in (3, 4, 5, 6):
        S.append(("TRI", "bank", (1, n), f"1 膽拖 {n} 腳"))
    for n in (2, 3, 4, 5, 6):
        S.append(("TRI", "bank", (2, n), f"2 膽拖 {n} 腳"))
    S.append(("TCE", "pos", [[1], [2], [3]], "單注（頭三選順序）"))
    for k in (3, 4):
        S.append(("TCE", "box", k, f"複式 {k} 隻"))
    for n in (2, 3, 4):
        S.append(("TCE", "pos", [[1], [2], _r(3, n + 2)], f"頭兩名定位拖 {n} 腳"))
        S.append(("TCE", "pos", [[1, 2], [1, 2], _r(3, n + 2)], f"頭兩選互換拖 {n} 腳"))
    for n in (3, 4, 5):
        S.append(("TCE", "pos", [[1], _r(2, n + 1), _r(2, n + 1)], f"首選定頭位拖 {n} 腳"))
    for k in (4, 5, 6):
        S.append(("FF", "box", k, "單注" if k == 4 else f"複式 {k} 隻"))
    for n in (4, 5, 6):
        S.append(("FF", "bank", (1, n), f"1 膽拖 {n} 腳"))
    for n in (3, 4, 5, 6):
        S.append(("FF", "bank", (2, n), f"2 膽拖 {n} 腳"))
    for n in (2, 3, 4, 5):
        S.append(("FF", "bank", (3, n), f"3 膽拖 {n} 腳"))
    S.append(("QTT", "pos", [[1], [2], [3], [4]], "單注（頭四選順序）"))
    for n in (2, 3, 4):
        S.append(("QTT", "pos", [[1], [2], [3], _r(4, n + 3)], f"頭三名定位拖 {n} 腳"))
    S.append(("QTT", "pos", [[1], [2], [3, 4], [3, 4]], "頭兩名定位、三四名互換"))
    S.append(("QTT", "pos", [[1], [2], [3, 4, 5], [3, 4, 5]], "頭兩名定位、三四名 3 隻複式"))
    S.append(("QTT", "pos", [[1, 2], [1, 2], [3, 4], [3, 4]], "頭兩選互換、三四選互換"))
    S.append(("QTT", "pos", [[1], [2, 3, 4], [2, 3, 4], [2, 3, 4]], "首選定頭位、後三名 3 隻複式"))
    S.append(("QTT", "box", 4, "複式 4 隻"))
    return S


STRUCTURES = _structures()


# =====================================================================================
# per-race pricing
# =====================================================================================
class Race:
    """One race after scratchings (from bots.prepare): probabilities, live odds, pricing."""

    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.rn = ctx.rn
        self.n = ctx.n
        self.probs = ctx.probs
        self.live = ctx.live or {}
        self.pw, self.pp = ctx.pw, ctx.pp
        self.est = st.DividendEstimator(ctx.df, self.live or None, cfg["calib"], cfg["lam2"], cfg["lam3"])
        self.order_win = ctx.order_win
        self.order_place = ctx.order_place
        p, m2 = self.probs.p, self.probs.m2
        self.p_top2 = {h: float(p[i] + m2[:, i].sum()) for h, i in self.probs.ix.items()}
        self.p_second = {h: float(m2[:, i].sum()) for h, i in self.probs.ix.items()}
        self.order_top2 = sorted(self.horses, key=lambda h: (-self.p_top2[h], h))
        self._pm, self._dm = {}, {}
        self.has_market = self.est.mkt is not None or bool(self.live)

    @property
    def horses(self):
        return self.ctx.horses

    def prob(self, pool, combo):
        k = (pool, combo)
        v = self._pm.get(k)
        if v is None:
            v = self.pp.get(int(combo), 0.0) if pool == "PLA" else self.probs.prob(pool, combo)
            self._pm[k] = v
        return v

    def div(self, pool, combo):
        """(estimated dividend per $10, 'live'|'est') or (None, None)."""
        k = (pool, combo)
        if k in self._dm:
            return self._dm[k]
        book = self.live.get(pool) or {}
        o = book.get(combo)
        if o:
            out = (max(st.MIN_DIVIDEND, float(o) * 10.0), "live")
        else:
            d = self.est(pool, combo)
            if book and pool in CAPPED_LIVE:
                floor = max(book.values()) * 10.0
                d = floor if d is None else max(d, floor)
            out = (d, "est") if d is not None else (None, None)
        self._dm[k] = out
        return out

    def mkt_win(self, h):
        """Market win probability (normalised 1/win odds) of horse h, or None."""
        if self.est.mkt is None:
            return None
        i = self.est.mkt.ix.get(int(h))
        return None if i is None else float(self.est.mkt.p[i])

    def mkt_second(self, h):
        if self.est.mkt is None:
            return None
        i = self.est.mkt.ix.get(int(h))
        return None if i is None else float(self.est.mkt.m2[:, i].sum())


def _leg_from_structure(race, pool, kind, arg):
    order = race.order_place if pool in ("PLA", "QPL") else race.order_win
    leg = {"race_no": race.rn, "pool": pool}
    if kind == "top":
        if len(order) < arg:
            return None
        leg["selections"] = [order[arg - 1]]
    elif kind == "box":
        if len(order) < max(arg, bets.POOL_SIZE[pool]):
            return None
        leg["selections"] = order[:arg]
    elif kind == "bank":
        b, n = arg
        if len(order) < b + n:
            return None
        leg["bankers"] = order[:b]
        leg["selections"] = order[b:b + n]
        if pool in bets.ORDERED_POOLS:
            leg["mode"] = "multiple"
    elif kind == "pos":
        mx = max(max(s) for s in arg)
        if len(order) < mx:
            return None
        leg["positions"] = [[order[i - 1] for i in s] for s in arg]
    return leg


def _single_candidates(race, cfg):
    out = []
    unit = float(cfg["unit_stake"])
    fams = set(cfg["families"])
    for pool, kind, arg, name in STRUCTURES:
        if pool not in fams or race.n < MIN_FIELD[pool]:
            continue
        if pool == "QPL" and race.probs.n_places < 3:
            continue
        leg = _leg_from_structure(race, pool, kind, arg)
        if leg is None:
            continue
        combos = st.leg_combos(leg)
        n = len(combos)
        if not n or n * unit > cfg["max_item_stake"] + 1e-9:
            continue
        er, srcs, ok = 0.0, set(), True
        for c in combos:
            d, s = race.div(pool, c)
            if d is None:
                ok = False
                break
            srcs.add(s)
            er += race.prob(pool, c) * d / 10.0
        if not ok:
            continue
        if pool == "PLA" and n == 1:
            hit = race.pp.get(leg["selections"][0], 0.0)
        elif pool in ("PLA", "QPL"):
            hit = race.probs.hit_prob(pool, combos)
        else:                                   # exclusive pools: one winning combination
            hit = min(1.0, sum(race.prob(pool, c) for c in combos))
        if hit <= 0:
            continue
        out.append(_cand("single", pool, pool, [leg], n, unit, hit, er, srcs, name,
                         key=f"{race.rn}:{pool}:{kind}:{json.dumps(arg)}"))
    return out


def _cand(kind, family, pool, legs, n, unit, hit, er, srcs, name, spec=None, key=None, extra=None):
    exp_ret = er                                        # sum over units of E[payout per $1 staked]
    c = {"er_raw": float(er), "hit_raw": float(hit), "kind": kind, "family": family, "pool": pool, "spec": spec, "legs": legs,
         "races": sorted({int(l["race_no"]) for l in legs}),
         "race_no": max(int(l["race_no"]) for l in legs),
         "n_combos": int(n), "unit_stake": unit, "stake": round(unit * n, 2),
         "hit_prob": float(min(1.0, hit)), "exp_roi": float(exp_ret / n - 1.0),
         "est_return_if_hit": float(unit * exp_ret / hit) if hit > 0 else None,
         "odds_source": "live" if srcs == {"live"} else "est", "structure_zh": name, "key": key}
    if extra:
        c.update(extra)
    return c


# ---- multi-race (DBL / TBL) ------------------------------------------------------------
def multi_race_legs(race_nos, n_races=None) -> dict:
    """{'DBL': [(r-1, r), ...], 'TBL': [(3,4,5), (n-2,n-1,n)]} from the races of a card
    (HKJC schedule: a Double on every consecutive pair, Trebles on races 3-5 and the last 3 -
    observed on every 2023-26 meeting in data/hkjc.db)."""
    rs = sorted({int(r) for r in race_nos})
    have = set(rs)
    dbl = [(r - 1, r) for r in rs if r - 1 in have]
    tbl = []
    if rs:
        n = int(n_races) if n_races else max(rs)
        for t in ((3, 4, 5), (n - 2, n - 1, n)):
            if all(x in have for x in t) and t not in tbl and t[0] >= 1:
                tbl.append(t)
    return {"DBL": dbl, "TBL": tbl}


def _mr_div(pool, legs_races, combo, cfg, typ):
    """Estimated DBL/TBL dividend per $10 for winners (typ 1) / last-leg runner-up (typ 2)."""
    hs = [int(x) for x in combo.split("/")]
    last = legs_races[-1]
    if typ == 1:
        o = (last.live.get(pool) or {}).get(combo)
        if o:
            return max(st.MIN_DIVIDEND, float(o) * 10.0), "live"
    pm = 1.0
    for race, h in zip(legs_races[:-1], hs[:-1]):
        x = race.mkt_win(h)
        if x is None:
            return None, None
        pm *= x
    x = last.mkt_win(hs[-1]) if typ == 1 else last.mkt_second(hs[-1])
    if x is None:
        return None, None
    pm *= x
    raw = 10.0 * (1 - TAKEOUT[pool]) / max(pm, 1e-7)
    a, b = cfg["mr_calib"].get((pool, typ), (0.0, 1.0))
    d = max(st.MIN_DIVIDEND, float(np.exp(a + b * np.log(raw))))
    book = last.live.get(pool) or {}
    if typ == 1 and book and pool in CAPPED_LIVE:
        d = max(d, max(book.values()) * 10.0)
    return d, "est"


def _multi_candidates(races_by_rn, cfg):
    out = []
    unit = float(cfg["unit_stake"])
    sched = multi_race_legs(races_by_rn, cfg.get("n_races"))
    for pool in MULTI_POOLS:
        if pool not in cfg["families"]:
            continue
        for rset in sched[pool]:
            legs_r = [races_by_rn[r] for r in rset]
            ks = [(1, 2)] * (len(rset) - 1) + [(1, 2, 3)]
            for choice in _product_ks(ks):
                sels = [r.order_win[:k] for r, k in zip(legs_r[:-1], choice[:-1])]
                sels.append(legs_r[-1].order_top2[:choice[-1]])
                if any(len(s) < k for s, k in zip(sels, choice)):
                    continue
                legs = [{"race_no": r.rn, "pool": pool, "selections": list(s)} for r, s in zip(legs_r, sels)]
                combos = bets.multi_race_combos(pool, legs)
                n = len(combos)
                if n * unit > cfg["max_item_stake"] + 1e-9:
                    continue
                er, srcs, ok = 0.0, set(), True
                for c in combos:
                    hs = [int(x) for x in c.split("/")]
                    pfirst = float(np.prod([r.pw[h] for r, h in zip(legs_r[:-1], hs[:-1])]))
                    d1, s1 = _mr_div(pool, legs_r, c, cfg, 1)
                    d2, s2 = _mr_div(pool, legs_r, c, cfg, 2)
                    if d1 is None or d2 is None:
                        ok = False
                        break
                    srcs |= {s1, s2}
                    er += pfirst * (legs_r[-1].pw[hs[-1]] * d1 + legs_r[-1].p_second[hs[-1]] * d2) / 10.0
                if not ok:
                    continue
                hit = 1.0
                for r, s in zip(legs_r[:-1], sels[:-1]):
                    hit *= sum(r.pw[h] for h in s)
                last = legs_r[-1]
                idx = [last.probs.ix[h] for h in sels[-1]]
                mask = np.ones(last.n, bool)
                mask[idx] = False
                miss = float(last.probs.m2[np.ix_(mask, mask)].sum())
                hit *= max(0.0, 1.0 - miss)
                if hit <= 0:
                    continue
                name = " × ".join(str(len(s)) for s in sels) + " 隻（尾關跑頭兩名都有派彩）"
                out.append(_cand("multi_race", pool, pool, legs, n, unit, hit, er, srcs, name,
                                 key=f"{pool}:{'-'.join(map(str, rset))}:{choice}"))
    return out


def _product_ks(ks):
    from itertools import product
    return list(product(*ks))


# ---- all-ups -----------------------------------------------------------------------------
def _allup_leg(race, pool, cfg):
    if race.n < MIN_FIELD[pool] or (pool == "QPL" and race.probs.n_places < 3):
        return None
    if pool == "WIN":
        sels = race.order_win[:1]
    elif pool == "PLA":
        sels = race.order_place[:1]
    elif pool == "QIN":
        sels = race.order_win[:2]
    else:
        sels = race.order_place[:3]
    combos = bets.multiple(pool, sels)
    e, srcs = 0.0, set()
    for c in combos:
        d, s = race.div(pool, c)
        if d is None:
            return None
        srcs.add(s)
        e += race.prob(pool, c) * d / 10.0
    hit = race.pp.get(sels[0], 0.0) if pool == "PLA" else race.probs.hit_prob(pool, combos)
    return {"race_no": race.rn, "pool": pool, "selections": list(sels), "combos": combos,
            "E": e, "hit": float(hit), "srcs": srcs}


def _p_at_least(hits, k):
    dist = np.zeros(len(hits) + 1)
    dist[0] = 1.0
    for h in hits:
        dist[1:] = dist[1:] * (1 - h) + dist[:-1] * h
        dist[0] *= (1 - h)
    return float(dist[k:].sum())


def _allup_candidates(races_by_rn, cfg):
    if "ALLUP" not in cfg["families"]:
        return []
    out = []
    unit = float(cfg["unit_stake"])
    for pool in ALLUP_POOLS:
        legs = [l for l in (_allup_leg(r, pool, cfg) for r in races_by_rn.values()) if l]
        legs.sort(key=lambda l: (-l["E"] / len(l["combos"]), l["race_no"]))
        for k in range(2, min(cfg["allup_max_legs"], len(legs)) + 1):
            use = sorted(legs[:k], key=lambda l: l["race_no"])
            for spec in bets.allup_specs(k):
                subs = bets.allup_subsets(k, spec)
                units = sum(int(np.prod([len(use[i]["combos"]) for i in s])) for s in subs)
                if units * unit > cfg["max_item_stake"] + 1e-9:
                    continue
                er = sum(float(np.prod([use[i]["E"] for i in s])) for s in subs)
                kmin = min(len(s) for s in subs)
                hit = _p_at_least([l["hit"] for l in use], kmin)
                if hit <= 0:
                    continue
                srcs = set().union(*(l["srcs"] for l in use))
                tl = [{"race_no": l["race_no"], "pool": pool, "selections": l["selections"]} for l in use]
                name = f"{ALLUP_ZH[pool]} {spec}"
                out.append(_cand("allup", "ALLUP", pool, tl, units, unit, hit, er, srcs, name, spec=spec,
                                 key=f"AU:{pool}:{spec}:" + ",".join(str(l['race_no']) for l in use),
                                 extra={"kmin": kmin}))
    return out


# =====================================================================================
# enumerate / select
# =====================================================================================
def _horse_info(pred_df):
    info, post = {}, {}
    for r in pred_df.itertuples(index=False):
        d = r._asdict()
        k = (int(d["race_no"]), int(d["horse_no"]))
        sc = d.get("score")
        info[k] = {"horse_name": str(d.get("horse_name") or "") if pd.notna(d.get("horse_name")) else "",
                   "score": None if sc is None or pd.isna(sc) else round(float(sc), 1),
                   "jockey": (str(d.get("jockey_name") or d.get("jockey") or "")
                              if pd.notna(d.get("jockey_name", d.get("jockey"))) else ""),
                   "pla_odds": d.get("pla_odds"), "p_place": d.get("p_place"),
                   "scratched": int(d.get("scratched") or 0) if pd.notna(d.get("scratched")) else 0}
        pt = d.get("post_time")
        if pt is not None and isinstance(pt, str):
            post.setdefault(k[0], pt)
    return info, post


def enumerate_candidates(pred_df, odds=None, cfg=None):
    """Every priced candidate of a meeting (no thresholds applied).

    Returns (cands, ctx) with ctx = {races: {race_no: Race}, has_odds, live_pools}."""
    from . import bots as _bots            # bots imports this module lazily (daily_picks robot)
    cfg = _cfg(cfg)
    ctxs = _bots.prepare(pred_df, odds)
    races = {c.rn: Race(c, cfg) for c in ctxs if c.n >= 4}
    cands = []
    for r in races.values():
        cands += _single_candidates(r, cfg)
    cands += _multi_candidates(races, cfg)
    cands += _allup_candidates(races, cfg)
    for c in cands:
        _adjust(c, cfg)
    live_pools = sorted({p for r in races.values() for p, b in r.live.items() if b})
    has_market = any(r.est.mkt is not None for r in races.values())
    return cands, {"races": races, "has_odds": bool(live_pools) or has_market, "live_pools": live_pools,
                   "has_live": bool(live_pools)}


def _adjust(c, cfg):
    """Per-family bias correction of the model estimates (fitted on 2024-25 candidates):
    expected return x RET_ADJ[family], hit probability x HIT_ADJ[family]."""
    kr = (cfg.get("ret_adj") or {}).get(c["family"], 1.0)
    kh = (cfg.get("hit_adj") or {}).get(c["family"], 1.0)
    er, hit, n = c["er_raw"] * kr, min(1.0, c["hit_raw"] * kh), c["n_combos"]
    c["exp_roi"] = float(er / n - 1.0)
    c["hit_prob"] = float(hit)
    c["est_return_if_hit"] = float(c["unit_stake"] * er / hit) if hit > 0 else None


def passes(c, cfg) -> bool:
    th = cfg["thresholds"].get(c["family"])
    if not th:
        return False
    return (c["exp_roi"] >= th["tau"] - 1e-12 and c["hit_prob"] >= th["h"] - 1e-12
            and c["stake"] <= cfg["max_item_stake"] + 1e-9)


def _margin(c, cfg):
    th = cfg["thresholds"].get(c["family"]) or {"tau": 0.0}
    return c["exp_roi"] - th["tau"]


def _sort_key(c, cfg):
    return (-round(_margin(c, cfg), 10), -round(c["hit_prob"], 10), c["stake"], c["key"] or "")


def _conflicts(c, chosen, cfg):
    if c["kind"] == "single":
        same = [x for x in chosen if x["kind"] == "single" and x["race_no"] == c["race_no"]]
        if len(same) >= cfg["max_per_race"]:
            return True
        return any(GROUP[x["pool"]] == GROUP[c["pool"]] for x in same)
    if c["kind"] == "multi_race":
        return any(x["kind"] == "multi_race" and x["pool"] == c["pool"] and set(x["races"]) & set(c["races"])
                   for x in chosen)
    allups = [x for x in chosen if x["kind"] == "allup"]
    return len(allups) >= cfg["max_allups"] or any(x["pool"] == c["pool"] for x in allups)


def select_items(cands, cfg=None):
    """(精選 list, 參考 list).  参考 only when nothing passes: the nearest candidates."""
    cfg = _cfg(cfg)
    ok = sorted([c for c in cands if passes(c, cfg)], key=lambda c: _sort_key(c, cfg))
    chosen, total = [], 0.0
    for c in ok:
        if len(chosen) >= cfg["max_items"]:
            break
        if total + c["stake"] > cfg["max_total"] + 1e-9:
            continue
        if _conflicts(c, chosen, cfg):
            continue
        chosen.append(c)
        total += c["stake"]
    ref = []
    if not chosen:
        ref = nearest(cands, cfg, cfg["n_reference"])
    return chosen, ref


def _ref_gap(c, cfg):
    th = cfg["thresholds"].get(c["family"])
    tau = th["tau"] if th else 0.0
    h = th["h"] if th else REF_DEFAULT_H[c["family"]]
    return max(0.0, tau - c["exp_roi"]) + max(0.0, h - c["hit_prob"])


def nearest(cands, cfg, n=3):
    """The n candidates closest to the thresholds (distinct races / pools)."""
    pool = sorted(cands, key=lambda c: (round(_ref_gap(c, cfg), 10), -c["exp_roi"], c["key"] or ""))
    out, seen = [], set()
    for c in pool:
        k = (c["kind"], tuple(c["races"])) if c["kind"] != "allup" else ("allup",)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= n:
            break
    return out


# =====================================================================================
# formatting
# =====================================================================================
def _pc(x):
    return f"{x * 100:.0f}%"


def _hs(hs):
    return "、".join(str(int(h)) for h in hs)


def _roi(x):
    return f"{x * 100:+.0f}%"


def _money(x):
    return f"${x:,.0f}"


def _horse(info, rn, h):
    i = info.get((rn, int(h)), {})
    return {"horse_no": int(h), "horse_name": i.get("horse_name") or "", "score": i.get("score")}


def _sel_txt(c, races):
    """Chinese description of the selections with the model numbers."""
    parts = []
    for leg in c["legs"]:
        r = races[int(leg["race_no"])]
        pool = leg["pool"]
        pre = f"第{r.rn}場 " if c["kind"] != "single" else ""
        if leg.get("positions"):
            names = ["頭位", "第二", "第三", "第四"]
            seg = []
            for nm, s in zip(names, leg["positions"]):
                if len(s) == 1:
                    seg.append(f"{nm} {s[0]}（勝率 {_pc(r.pw[s[0]])}）")
                else:
                    seg.append(f"{nm} {_hs(s)}")
            parts.append(pre + " → ".join(seg))
        elif leg.get("bankers"):
            b = leg["bankers"]
            parts.append(pre + f"膽 {_hs(b)}（勝率 {'/'.join(_pc(r.pw[h]) for h in b)}），腳 {_hs(leg['selections'])}")
        else:
            s = leg["selections"]
            if pool in ("PLA", "QPL"):
                prob = "入三甲機率 " + "/".join(_pc(r.pp[h]) for h in s)
            elif c["kind"] == "multi_race" and leg is c["legs"][-1]:
                prob = "頭兩名機率 " + "/".join(_pc(r.p_top2[h]) for h in s)
            else:
                prob = "勝率 " + "/".join(_pc(r.pw[h]) for h in s)
            odd = ""
            if c["kind"] == "single" and len(s) == 1 and pool in ("WIN", "PLA"):
                d, src = r.div(pool, str(s[0]))
                if d:
                    odd = f"，{POOL_ZH[pool]}賠率 {d / 10:.1f}"
            parts.append(pre + f"{_hs(s)}{'號' if len(s) == 1 else ''}（{prob}{odd}）")
    return "；".join(parts)


def _where(c):
    if c["kind"] == "single":
        return f"第{c['race_no']}場 {POOL_ZH[c['pool']]} {c['structure_zh']}"
    if c["kind"] == "multi_race":
        return f"第{c['races'][0]}–{c['races'][-1]}場 {POOL_ZH[c['pool']]} {c['structure_zh']}"
    return f"{c['structure_zh']}（{len(c['legs'])} 關，中 {c['kmin']} 關或以上有派彩）"


def _reason(c, races, grade, cfg):
    txt = (f"{_where(c)}：{_sel_txt(c, races)}，共 {c['n_combos']} 注 {_money(c['stake'])}，"
           f"預計命中 {_pc(c['hit_prob'])}，估計回報 {_roi(c['exp_roi'])}（{SRC_ZH[c['odds_source']]}）")
    if c.get("est_return_if_hit"):
        txt += f"，中咗平均派約 {_money(c['est_return_if_hit'])}"
    if grade == "參考":
        th = cfg["thresholds"].get(c["family"])
        if not th:
            why = "呢類玩法用 2024-25 季歷史數據驗證未見正回報，唔會評做精選"
        else:
            bits = []
            if c["exp_roi"] < th["tau"]:
                bits.append(f"估計回報 {_roi(c['exp_roi'])} 低過門檻 {_roi(th['tau'])}")
            if c["hit_prob"] < th["h"]:
                bits.append(f"命中 {_pc(c['hit_prob'])} 低過門檻 {_pc(th['h'])}")
            why = "；".join(bits) or "已有更好嘅項目或超出每日上限"
        txt = f"未過門檻（只供參考，冇落注）：{txt}。{why}"
    return txt


def _item(c, rank, grade, races, info, post, cfg):
    legs = []
    for leg in c["legs"]:
        rn = int(leg["race_no"])
        if leg.get("positions"):
            uniq = []
            for s in leg["positions"]:
                for h in s:
                    if h not in uniq:
                        uniq.append(h)
            sels, bk, pos = uniq, [], [list(map(int, s)) for s in leg["positions"]]
        else:
            sels, bk, pos = leg.get("selections") or [], leg.get("bankers") or [], None
        d = {"race_no": rn, "bankers": [_horse(info, rn, h) for h in bk],
             "selections": [_horse(info, rn, h) for h in sels], "positions": pos}
        if c["kind"] != "single":
            d["pool_zh"] = POOL_ZH[leg["pool"]]
        legs.append(d)
    pool_zh = ALLUP_ZH[c["pool"]] if c["kind"] == "allup" else POOL_ZH[c["pool"]]
    it = {"rank": rank, "kind": c["kind"], "race_no": c["race_no"], "races": c["races"],
          "post_time": post.get(c["races"][0]), "pool": c["pool"], "pool_zh": pool_zh,
          "structure_zh": c["structure_zh"], "legs": legs, "n_combos": c["n_combos"],
          "unit_stake": c["unit_stake"], "stake": c["stake"], "hit_prob": round(c["hit_prob"], 4),
          "exp_roi": round(c["exp_roi"], 4),
          "est_return_if_hit": None if c["est_return_if_hit"] is None else round(c["est_return_if_hit"], 2),
          "odds_source": SRC_ZH[c["odds_source"]], "grade": grade, "reason": _reason(c, races, grade, cfg),
          "result": None, "family": c["family"], "spec": c.get("spec"), "key": c["key"]}
    it["ticket"] = item_ticket({**it, "legs_raw": c["legs"]})
    it["legs_raw"] = c["legs"]
    return it


def item_ticket(item) -> dict:
    """SPEC ticket dict for a picks item (the daily_picks robot buys exactly this)."""
    legs = [dict(l) for l in (item.get("legs_raw") or [])]
    pool, unit = item["pool"], float(item["unit_stake"])
    if item["kind"] == "allup":
        t = st._ticket(f"daily_picks:{pool}_{item['spec']}", item["rank"], pool, item["spec"], legs, unit)
    elif item["kind"] == "multi_race":
        n = bets.multi_race_units(pool, legs)
        t = {"strategy": f"daily_picks:{pool}", "ticket_no": item["rank"], "pool": pool, "spec": None,
             "legs": legs, "unit_stake": unit, "n_combos": n, "total_stake": round(unit * n, 2), "notes": {}}
    else:
        t = st._ticket(f"daily_picks:{pool}", item["rank"], pool, None, legs, unit)
    t["notes"] = {"pick_rank": item["rank"], "pick_key": item.get("key"), "structure": item["structure_zh"],
                  "exp_roi": item["exp_roi"], "hit_prob": item["hit_prob"]}
    if int(t["n_combos"]) != int(item["n_combos"]):
        raise ValueError(f"picks item {item.get('key')}: ticket units {t['n_combos']} != {item['n_combos']}")
    return t


def _banker(ctx_races, info, post):
    best = None
    for rn, r in ctx_races.items():
        for h in r.horses:
            p = r.pp[h]
            k = (p, -rn, -h)
            if best is None or k > best[0]:
                best = (k, rn, h)
    if best is None:
        return None
    _, rn, h = best
    i = info.get((rn, h), {})
    pp = ctx_races[rn].pp[h]
    why = f"全日入三甲機率最高（{pp * 100:.1f}%）"
    if i.get("score") is not None:
        rank = ctx_races[rn].order_win.index(h) + 1
        why += f"；評分 {i['score']:.0f}，同場排第 {rank}"
    po = i.get("pla_odds")
    if po is not None and pd.notna(po):
        why += f"；位置賠率 {float(po):g}"
        po = float(po)
    else:
        po = None
    return {"race_no": rn, "post_time": post.get(rn), "horse_no": int(h), "horse_name": i.get("horse_name") or "",
            "jockey": i.get("jockey") or "", "score": i.get("score"), "p_place": round(pp, 4),
            "pla_odds": po, "reason": why, "result": None}


def _odds_note(ctx):
    if not ctx["has_odds"]:
        return ("賠率未開售：所有玩法都要市場賠率先計得到預期回報，暫時冇任何玩法可以評級；"
                "開售後會重新計算")
    if not ctx["has_live"]:
        return ("冇即時賠率：全部玩法派彩由獨贏賠率推算（估計賠率，用 2024-25 季派彩校準）")
    order = SINGLE_POOLS + MULTI_POOLS
    lp = [p for p in order if p in ctx["live_pools"]]
    est = [p for p in order if p not in lp]
    txt = "即時賠率：" + "、".join(POOL_ZH[p] for p in lp)
    if est:
        txt += "；估計賠率（冇即時賠率，由獨贏賠率推算並用 2024-25 季派彩校準）：" + "、".join(POOL_ZH[p] for p in est)
    if "DBL" in lp:
        txt += "；孖寶尾關亞軍派彩冇即時賠率，一律估計"
    return txt


def _race_notes(cands, chosen, cfg):
    """Per race: best single candidate and whether it was picked (daily_picks skip reasons)."""
    out = {}
    picked = {c["key"] for c in chosen}
    for c in cands:
        if c["kind"] != "single":
            continue
        cur = out.get(c["race_no"])
        if cur is None or _ref_gap(c, cfg) < _ref_gap(cur, cfg) - 1e-12:
            out[c["race_no"]] = c
    notes = {}
    for rn, c in out.items():
        th = cfg["thresholds"].get(c["family"])
        if c["key"] in picked:
            continue
        if th and passes(c, cfg):
            why = "過門檻但已有更好嘅項目或超出每日上限"
        elif not th:
            why = "呢類玩法用 2024-25 季歷史數據驗證未見正回報"
        else:
            why = f"門檻：估計回報 ≥ {_roi(th['tau'])}、命中 ≥ {_pc(th['h'])}"
        notes[rn] = (f"最接近門檻嘅係 {POOL_ZH[c['pool']]} {c['structure_zh']}：估計回報 {_roi(c['exp_roi'])}"
                     f"（{SRC_ZH[c['odds_source']]}）、預計命中 {_pc(c['hit_prob'])}（{why}）")
    return notes


def make_picks(pred_df, odds=None, cfg=None) -> dict:
    """「今日精選」 dict (SPEC_V2 第三版) for one meeting.

    pred_df: race_no, horse_no, p_win, p_place [+ race_id, scratched, win_odds, win_odds_live,
    pla_odds, horse_name, jockey_name, score, post_time].  odds: {race_no: {pool: {combo: odds}}}
    (odds.latest_odds) or None.  Items carry an internal ``ticket`` (stripped on publish)."""
    cfg = _cfg(cfg)
    info, post = _horse_info(pred_df)
    cands, ctx = enumerate_candidates(pred_df, odds, cfg)
    chosen, ref = select_items(cands, cfg)
    races = ctx["races"]
    items = [_item(c, i, "精選", races, info, post, cfg) for i, c in enumerate(chosen, 1)]
    items += [_item(c, len(items) + i, "參考", races, info, post, cfg) for i, c in enumerate(ref, 1)]
    enabled = [f for f in FAMILIES if cfg["thresholds"].get(f)]
    method = ("由模型機率逐場比較馬會所有玩法（獨贏、位置、連贏、位置Q、二重彩、單T、三重彩、四連環、四重彩、"
              "孖寶、三寶、過關）同各種結構（單注、複式、膽拖、定位），計預計命中同估計回報；"
              "只揀估計回報同命中機率都過門檻嘅項目"
              f"（每日最多 {cfg['max_items']} 項、{_money(cfg['max_total'])}，每場最多 {cfg['max_per_race']} 項）。"
              "門檻只用 2024-25 季歷史數據調校，2025-26 起樣本外驗證；模擬研究，非投注建議。")
    if enabled:
        method += "現時有門檻嘅玩法：" + "、".join("過關" if f == "ALLUP" else POOL_ZH[f] for f in enabled) + "。"
    else:
        method += "用 2024-25 季歷史數據驗證，冇一類玩法過門檻後有正回報，所以暫時唔會出精選。"
    if not chosen:
        method = NO_PICKS_NOTE + "。" + method
    sel = [it for it in items if it["grade"] == "精選"]
    return {"title": TITLE, "generated_at": datetime.now(HKT).replace(tzinfo=None).isoformat(timespec="seconds"),
            "method_note": method, "odds_note": _odds_note(ctx),
            "total_stake": round(sum(it["stake"] for it in sel), 2), "n_items": len(sel),
            "total_payout": None, "total_profit": None, "items": items,
            "banker": _banker(races, info, post),
            "race_notes": {str(k): v for k, v in _race_notes(cands, chosen, cfg).items()},
            "n_candidates": len(cands)}


# =====================================================================================
# settlement (backtest + publish of 參考 items)
# =====================================================================================
class DBData:
    """Dividend access for one meeting from the database (publish / tests)."""

    def __init__(self, conn, date, venue):
        self.conn, self.date, self.venue = conn, date, venue

    def rid(self, rn):
        return f"{self.date.replace('-', '')}-{self.venue}-{int(rn):02d}"

    def idx(self, rn):
        rows = self.conn.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id=?",
                                 (self.rid(rn),)).fetchall()
        good = [(p, c, d) for p, c, d in rows if p in bets.POOL_SIZE]
        return bets.index_dividends(good) if good else None

    def scr(self, rn):
        return {int(r[0]) for r in self.conn.execute(
            "SELECT horse_no FROM runners WHERE race_id=? AND scratched=1", (self.rid(rn),))}

    def mr_rows(self, pool, rn):
        return [tuple(r) for r in self.conn.execute(
            "SELECT pool, combination, dividend FROM dividends WHERE race_id=? AND pool=?", (self.rid(rn), pool))]

    def fav(self, rn):
        r = self.conn.execute("SELECT horse_no FROM runners WHERE race_id=? AND COALESCE(scratched,0)=0 "
                              "AND win_odds > 0 ORDER BY win_odds, horse_no LIMIT 1", (self.rid(rn),)).fetchone()
        return None if r is None else int(r[0])


class BTData:
    """Dividend access for a backtest meeting (hkjc.backtest.load_data) + DBL/TBL rows."""

    def __init__(self, m, mr):
        self.m, self.mrd = m, mr

    def idx(self, rn):
        r = self.m["races"].get(int(rn))
        return None if r is None else r["idx"]

    def scr(self, rn):
        r = self.m["races"].get(int(rn))
        return set() if r is None else r["scr"]

    def mr_rows(self, pool, rn):
        r = self.m["races"].get(int(rn))
        return [] if r is None else self.mrd.get((r["race_id"], pool), [])

    def fav(self, rn):
        r = self.m["races"].get(int(rn))
        if r is None:
            return None
        d = r["df"].sort_values(["win_odds", "horse_no"])
        return int(d["horse_no"].iloc[0]) if len(d) else None


def settle_item(item, data) -> dict:
    """Settle a picks item / candidate with real dividends.  A leg race without results counts
    as refunded (void).  Returns {stake, payout, profit, hit, n_win, detail}."""
    legs = item.get("legs_raw") or item["legs"]
    pool, unit = item["pool"], float(item["unit_stake"])
    if item["kind"] == "multi_race":
        void = any(data.idx(l["race_no"]) is None for l in legs)
        rows = [] if void else data.mr_rows(pool, legs[-1]["race_no"])
        res = bets.settle_multi_race(pool, legs, rows, unit,
                                     {l["race_no"]: data.scr(l["race_no"]) for l in legs},
                                     {l["race_no"]: data.fav(l["race_no"]) for l in legs})
    elif item["kind"] == "allup":
        dbr = {l["race_no"]: (data.idx(l["race_no"]) or {"VOID": {bets.REFUND_KEY: None}}) for l in legs}
        scr = {l["race_no"]: data.scr(l["race_no"]) for l in legs}
        res = bets.settle_allup(pool, legs, item["spec"], dbr, unit, scr)
        res["leg_hits"] = [any(o == "win" for o in l["outcomes"].values()) for l in res["legs"]]
    else:
        leg = legs[0]
        idx = data.idx(leg["race_no"])
        combos = st.leg_combos({**leg, "pool": pool})
        if idx is None:
            res = {"stake": unit * len(combos), "payout": unit * len(combos), "n_winning_units": 0}
        else:
            pay, nw = 0.0, 0
            for c in combos:
                status, factor, _ = bets.combo_outcome(pool, c, idx, data.scr(leg["race_no"]))
                pay += unit * factor
                nw += status == "win"
            res = {"stake": unit * len(combos), "payout": pay, "n_winning_units": nw}
    stake, pay, nw = float(res["stake"]), float(res["payout"]), int(res["n_winning_units"])
    detail = (f"中 {nw} 注，派彩 ${pay:,.1f}" if nw else
              ("退款" if abs(pay - stake) < 1e-6 and pay > 0 else "冇中"))
    return {"stake": round(stake, 2), "payout": round(pay, 2), "profit": round(pay - stake, 2),
            "hit": nw > 0, "n_win": nw, "detail": detail, "leg_hits": res.get("leg_hits")}


# =====================================================================================
# storage
# =====================================================================================
def save_picks(conn, run_id, date, venue, picks) -> None:
    conn.execute("INSERT OR REPLACE INTO picks (run_id, date, venue, payload, created_at) VALUES (?,?,?,?,?)",
                 (run_id, date, venue, json.dumps(picks, ensure_ascii=False, default=_json_default),
                  datetime.now(HKT).replace(tzinfo=None).isoformat(timespec="seconds")))
    conn.commit()


def load_picks(conn, date, venue, run_id=None):
    """Stored picks of a meeting: of ``run_id`` when given, else the latest stored run."""
    if run_id is not None:
        r = conn.execute("SELECT payload FROM picks WHERE run_id=? AND date=? AND venue=?",
                         (run_id, date, venue)).fetchone()
    else:
        r = conn.execute("SELECT payload FROM picks WHERE date=? AND venue=? ORDER BY created_at DESC, run_id DESC "
                         "LIMIT 1", (date, venue)).fetchone()
    return None if r is None else json.loads(r[0])


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(type(x))


def public_picks(picks):
    """Copy of a stored picks dict without the internal fields (tickets, raw legs, keys)."""
    if not picks:
        return picks
    out = {k: v for k, v in picks.items() if k not in ("race_notes", "n_candidates")}
    out["items"] = [{k: v for k, v in it.items() if k not in ("ticket", "legs_raw", "key", "family", "spec")}
                    for it in picks.get("items") or []]
    return out


def picks_md(picks) -> str:
    """「## 今日精選」 section of the pre-race report (same items as the site)."""
    if not picks:
        return "## 今日精選\n_(未有精選：模型未有預測)_\n"
    out = ["## 今日精選", "", f"_{picks.get('method_note') or ''}_", "", f"{picks.get('odds_note') or ''}", ""]
    sel = [i for i in picks.get("items") or [] if i["grade"] == "精選"]
    ref = [i for i in picks.get("items") or [] if i["grade"] != "精選"]
    if not sel:
        out.append("**今日冇精選**")
        out.append("")
    if sel or ref:
        out += ["| # | 評級 | 場次 | 玩法 | 結構 | 選擇 | 注數 | 注碼 | 預計命中 | 估計回報 | 賠率 | 結果 |",
                "|---:|---|---|---|---|---|---:|---:|---:|---:|---|---|"]
        for it in sel + ref:
            legs = []
            for l in it["legs"]:
                if l.get("positions"):
                    s = " / ".join(",".join(map(str, p)) for p in l["positions"])
                else:
                    s = ",".join(str(h["horse_no"]) for h in l["selections"])
                    if l["bankers"]:
                        s = "膽 " + ",".join(str(h["horse_no"]) for h in l["bankers"]) + " 拖 " + s
                legs.append((f"R{l['race_no']} " if it["kind"] != "single" else "") + s)
            races = (f"第{it['race_no']}場" if it["kind"] == "single" else
                     "第" + "、".join(map(str, it["races"])) + "場")
            res = it.get("result")
            rtxt = "-" if not res else (f"{'✅' if res['hit'] else '❌'} 派 ${res['payout']:,.0f}（{res['profit']:+,.0f}）"
                                        + ("（假設）" if res.get("hypothetical") else ""))
            out.append(f"| {it['rank']} | {it['grade']} | {races} | {it['pool_zh']} | {it['structure_zh']} | "
                       f"{'；'.join(legs)} | {it['n_combos']} | ${it['stake']:,.0f} | {it['hit_prob'] * 100:.0f}% | "
                       f"{it['exp_roi'] * 100:+.0f}% | {it['odds_source']} | {rtxt} |")
        out.append("")
        for it in sel + ref:
            out.append(f"- **#{it['rank']} {it['grade']}**：{it['reason']}")
        out.append("")
    b = picks.get("banker")
    if b:
        out.append(f"**全日最穩**：第{b['race_no']}場 {b['horse_no']}號 {b['horse_name']} — {b['reason']}")
    tot = f"**精選合共 {picks.get('n_items', 0)} 項 ${picks.get('total_stake') or 0:,.0f}（模擬）**"
    if picks.get("total_payout") is not None:
        tot += f"　派彩 ${picks['total_payout']:,.2f}，盈虧 {picks['total_profit']:+,.2f}"
    out += ["", tot, ""]
    return "\n".join(out) + "\n"


# =====================================================================================
# BACKTEST: walk-forward over 2024-09 -> latest with the real dividends
# =====================================================================================
BACKTEST_PATH = Path(__file__).resolve().parent.parent / "data" / "picks_backtest.json"
IS_SEASONS = ("2024-25",)
OOS_SEASONS = ("2025-26", "2026-27")
REPORT_HEADING = "## 每日精選（分析決定玩法）"
FAMILY_ZH = {**POOL_ZH, "ALLUP": "過關"}


def load_backtest_data(conn, source="wf-combined-v1", start=None, end=None):
    """(meetings from hkjc.backtest.load_data, {(race_id, pool): DBL/TBL dividend rows})."""
    from . import backtest as bt
    meetings = bt.load_data(conn, source, start, end)
    nr = {(r[0], r[1]): r[2] for r in conn.execute("SELECT date, venue, n_races FROM meetings")}
    for m in meetings:
        m["n_races"] = nr.get((m["date"], m["venue"]))
    rows = conn.execute("SELECT race_id, pool, combination, dividend FROM dividends WHERE pool IN ('DBL','TBL')")
    mr = {}
    for rid, pool, comb, div in rows:
        mr.setdefault((rid, pool), []).append((pool, comb, div))
    return meetings, mr


def meeting_pred(m) -> pd.DataFrame:
    """A backtest meeting as a live-style pred_df (final win odds as the market)."""
    dfs = []
    for rn in sorted(m["races"]):
        r = m["races"][rn]
        d = r["df"][["race_no", "horse_no", "p_win", "p_place", "win_odds"]].copy()
        d.insert(0, "race_id", r["race_id"])
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def fit_calibration(meetings, mr, seasons=IS_SEASONS, lam2=LAM2, lam3=LAM3):
    """log(div) = a + b log(raw market estimate) on the winning combinations of ``seasons``:
    single-race FCT/TCE/QTT (+ PLA/QIN/QPL/TRI/FF for comparison with ADAPTIVE) and the
    DBL/TBL winner (1) / last-leg runner-up (2) dividends."""
    xs = {}
    for m in meetings:
        if m["season"] not in seasons:
            continue
        mk = {}
        for rn, r in m["races"].items():
            est = st.DividendEstimator(r["df"], None, None, lam2, lam3)
            mk[rn] = est.mkt
            for pool in ("PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT"):
                rows = r["idx"].get(pool) or {}
                if bets.REFUND_KEY in rows:
                    continue
                for combo, d in rows.items():
                    if not d:
                        continue
                    raw = est.raw_market(pool, combo)
                    if raw:
                        xs.setdefault(pool, []).append((np.log(raw), np.log(d)))
        for pool in MULTI_POOLS:
            sched = multi_race_legs(m["races"], m.get("n_races"))[pool]
            for rset in sched:
                last = m["races"][rset[-1]]
                fin = last["finish"]
                for _p, comb, d in mr.get((last["race_id"], pool), []):
                    if not d or "F" in comb.upper():
                        continue
                    try:
                        hs = [int(x) for x in comb.split("/")]
                    except ValueError:
                        continue
                    if len(hs) != len(rset) or any(mk.get(r) is None for r in rset):
                        continue
                    typ = {1: 1, 2: 2}.get(fin.get(hs[-1]))
                    if typ is None:
                        continue
                    pm = 1.0
                    ok = True
                    for r, h in zip(rset[:-1], hs[:-1]):
                        i = mk[r].ix.get(h)
                        if i is None:
                            ok = False
                            break
                        pm *= mk[r].p[i]
                    i = mk[rset[-1]].ix.get(hs[-1])
                    if not ok or i is None:
                        continue
                    pm *= mk[rset[-1]].p[i] if typ == 1 else mk[rset[-1]].m2[:, i].sum()
                    if pm <= 0:
                        continue
                    raw = 10 * (1 - TAKEOUT[pool]) / pm
                    xs.setdefault((pool, typ), []).append((np.log(raw), np.log(d)))
    out = {}
    for k, v in xs.items():
        if len(v) < 30:
            continue
        a = np.array(v)
        b, c = np.polyfit(a[:, 0], a[:, 1], 1)
        out[k] = {"a": round(float(c), 3), "b": round(float(b), 3), "n": len(v),
                  "median_ratio": round(float(np.median(np.exp(a[:, 1] - a[:, 0]))), 3)}
    return out


def _slot(c, mi):
    if c["kind"] == "single":
        return (mi, c["race_no"])
    if c["kind"] == "multi_race":
        return (mi, tuple(c["races"]))
    return (mi, 0)


def _agg(rows):
    stake = float(sum(r["stake"] for r in rows))
    pay = float(sum(r["payout"] for r in rows))
    n = len(rows)
    return {"n": n, "stake": round(stake, 2), "payout": round(pay, 2), "profit": round(pay - stake, 2),
            "roi": round(pay / stake - 1, 4) if stake else None,
            "hit_rate": round(sum(1 for r in rows if r["hit"]) / n, 4) if n else None,
            "n_meetings": len({r["mi"] for r in rows})}


def _boot_ci(rows, mis, n_boot=2000, seed=0):
    """95% bootstrap CI of ROI, resampling MEETINGS (all meetings of the period, incl. empty)."""
    if not rows or not mis:
        return [None, None]
    idx = {m: i for i, m in enumerate(mis)}
    s = np.zeros(len(mis))
    p = np.zeros(len(mis))
    for r in rows:
        s[idx[r["mi"]]] += r["stake"]
        p[idx[r["mi"]]] += r["payout"]
    if s.sum() <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(s), np.full(len(s), 1.0 / len(s)), size=n_boot).astype(float)
    sb, pb = W @ s, W @ p
    ok = sb > 0
    roi = pb[ok] / sb[ok] - 1
    return [round(float(np.percentile(roi, 2.5)), 4), round(float(np.percentile(roi, 97.5)), 4)]


def fit_adjustments(meetings, mr, seasons=IS_SEASONS):
    """Per family over every priced 2024-25 candidate (estimates WITHOUT adjustment):
    RET_ADJ = realised return (largest single payout removed) / estimated return,
    HIT_ADJ = realised hit rate / estimated hit probability."""
    cfg = _cfg({"ret_adj": {}, "hit_adj": {}})
    acc = {f: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for f in FAMILIES}   # stake, est_ret, pay, top, hits, est_hits
    for m in meetings:
        if m["season"] not in seasons:
            continue
        cands, _ = enumerate_candidates(meeting_pred(m), None, {**cfg, "n_races": m.get("n_races")})
        dat = BTData(m, mr)
        for c in cands:
            r = settle_item(c, dat)
            a = acc[c["family"]]
            a[0] += r["stake"]
            a[1] += (1 + c["exp_roi"]) * r["stake"]
            a[2] += r["payout"]
            a[3] = max(a[3], r["payout"])
            a[4] += r["hit"]
            a[5] += c["hit_prob"]
    out = {}
    for f, (stake, est, pay, top, hits, eh) in acc.items():
        if stake <= 0:
            continue
        out[f] = {"ret_adj": round((pay - top) / est, 3), "hit_adj": round(hits / eh, 3) if eh else 1.0,
                  "est_return": round(est / stake, 4), "real_return": round(pay / stake, 4),
                  "real_return_ex_top": round((pay - top) / stake, 4), "n": None}
    return out


def _run(per_meeting, mis, cfg):
    """Selected items of meetings ``mis`` under cfg -> rows (mi, family, stake, payout, hit, hit_prob)."""
    rows = []
    for mi in mis:
        chosen, _ = select_items(per_meeting[mi], cfg)
        for c in chosen:
            r = c["res"]
            rows.append({"mi": mi, "family": c["family"], "stake": r["stake"], "payout": r["payout"],
                         "hit": r["hit"], "hit_prob": c["hit_prob"]})
    return rows


def _score(rows):
    stake = sum(r["stake"] for r in rows)
    pay = sum(r["payout"] for r in rows)
    top = max((r["payout"] for r in rows), default=0.0)
    return {"n": len(rows), "stake": round(stake, 2), "roi": round(pay / stake - 1, 4) if stake else None,
            "robust_roi": round((pay - top) / stake - 1, 4) if stake else None,
            "exp_hits": round(sum(r["hit_prob"] for r in rows), 1), "n_hits": sum(1 for r in rows if r["hit"]),
            "hit_rate": round(sum(1 for r in rows if r["hit"]) / len(rows), 4) if rows else None,
            "n_meetings": len({r["mi"] for r in rows})}


def tune_thresholds(per_meeting, is_mis):
    """Thresholds from the 2024-25 meetings only.

    1. Per family, every (tau, h) of the grid is run through select_items with ONLY that family
       enabled (same caps as live).  Eligible settings: >= TUNE_MIN_N items and >= TUNE_MIN_EXP_HITS
       expected hits.  Best = highest ROBUST ROI (ROI after removing the single largest payout, so
       one jackpot cannot decide - QTT/TCE/TBL dividends are extremely heavy-tailed).  The family is
       enabled only if that robust ROI > 0.
    2. Combined check: all enabled families together; while the combined robust ROI <= 0, drop the
       family whose removal helps most (backward elimination).
    Returns (thresholds, grid rows, elimination log)."""
    off = {f: None for f in FAMILIES}
    thr, grid = {}, []
    for f in FAMILIES:
        fm = [[c for c in per_meeting[mi] if c["family"] == f] for mi in range(len(per_meeting))]
        best = None
        for tau in TUNE_TAU:
            for h in TUNE_H[f]:
                cfg = _cfg({"thresholds": {**off, f: {"tau": tau, "h": h}}})
                sc = _score(_run(fm, is_mis, cfg))
                row = {"family": f, "tau": tau, "h": h, **sc}
                grid.append(row)
                if (sc["n"] >= TUNE_MIN_N[f] and sc["exp_hits"] >= TUNE_MIN_EXP_HITS and sc["robust_roi"] is not None
                        and (best is None or sc["robust_roi"] > best["robust_roi"] + 1e-12)):
                    best = row
        thr[f] = {"tau": best["tau"], "h": best["h"]} if best is not None and best["robust_roi"] > 0 else None
        if best is not None:
            log.info("picks tune %s: tau=%s h=%s n=%s roi=%+.3f robust=%+.3f -> %s", f, best["tau"], best["h"],
                     best["n"], best["roi"], best["robust_roi"], "enabled" if thr[f] else "disabled")
    elim = []
    while True:
        on = [f for f in FAMILIES if thr.get(f)]
        if not on:
            break
        cur = _score(_run(per_meeting, is_mis, _cfg({"thresholds": thr})))
        elim.append({"families": on, **cur})
        if cur["robust_roi"] is not None and cur["robust_roi"] > 0:
            break
        trial = []
        for f in on:
            t2 = {**thr, f: None}
            sc = _score(_run(per_meeting, is_mis, _cfg({"thresholds": t2})))
            trial.append((sc["robust_roi"] if sc["robust_roi"] is not None else -9, f))
        _, drop = max(trial)
        log.info("picks tune: combined robust ROI %s <= 0 -> drop %s", cur["robust_roi"], drop)
        thr[drop] = None
    return thr, grid, elim


def backtest_picks(conn, source="wf-combined-v1", write=True, path=None, report_path=None,
                   tune=True, data=None) -> dict:
    """Walk-forward history of make_picks (same code path as live) with the real dividends.

    2024-25 = tuning season (calibration + thresholds), 2025-26 + 2026-27 = out of sample.
    Writes data/picks_backtest.json (internal research - NOT published) and the
    「## 每日精選（分析決定玩法）」 section of reports/回測報告.md."""
    from . import bots as _bots
    meetings, mr = data if data is not None else load_backtest_data(conn, source)
    calib_fit = fit_calibration(meetings, mr)
    adj_fit = fit_adjustments(meetings, mr)
    enum_cfg = _cfg()
    n_cands = 0
    per_meeting = []
    for mi, m in enumerate(meetings):
        fr = meeting_pred(m)
        cands, _ctx = enumerate_candidates(fr, None, {**enum_cfg, "n_races": m.get("n_races")})
        dat = BTData(m, mr)
        for c in cands:
            c["res"] = settle_item(c, dat)
            c["mi"] = mi
        per_meeting.append(cands)
        n_cands += len(cands)
    is_mis0 = [i for i, m in enumerate(meetings) if m["season"] in IS_SEASONS]
    if tune:
        thr, grid, elim = tune_thresholds(per_meeting, is_mis0)
    else:
        thr, grid, elim = dict(THRESHOLDS), [], []
    cfg = _cfg({"thresholds": thr})
    sel_rows, meet_rows = [], []
    for mi, (m, cands) in enumerate(zip(meetings, per_meeting)):
        chosen, _ref = select_items(cands, cfg)
        st_, py = 0.0, 0.0
        for c in chosen:
            r = c["res"]
            sel_rows.append({"mi": mi, "season": m["season"], "family": c["family"], "kind": c["kind"],
                             "pool": c["pool"], "structure": c["structure_zh"], "stake": r["stake"],
                             "payout": r["payout"], "hit": r["hit"], "exp_roi": c["exp_roi"],
                             "hit_prob": c["hit_prob"], "date": m["date"], "venue": m["venue"]})
            st_ += r["stake"]
            py += r["payout"]
        meet_rows.append({"mi": mi, "date": m["date"], "venue": m["venue"], "season": m["season"],
                          "n_items": len(chosen), "stake": round(st_, 2), "payout": round(py, 2)})
    # live/backtest parity: the last OOS meeting re-decided through make_picks from an independent load
    parity = None
    oos_mis = [i for i, m in enumerate(meetings) if m["season"] in OOS_SEASONS]
    is_mis = [i for i, m in enumerate(meetings) if m["season"] in IS_SEASONS]
    if oos_mis and conn is not None:
        mi = oos_mis[-1]
        m = meetings[mi]
        live_like = _bots.meeting_frame(conn, m["date"], m["venue"], source)
        live_like = live_like[live_like["race_id"].isin({r["race_id"] for r in m["races"].values()})]
        live_cfg = {**cfg, "n_races": m.get("n_races")}
        live = make_picks(live_like, None, live_cfg)
        want = [(c["key"], round(c["exp_roi"], 4)) for c in select_items(per_meeting[mi], cfg)[0]]
        got = [(it["key"], it["exp_roi"]) for it in live["items"] if it["grade"] == "精選"]
        assert got == want, f"live/backtest picks mismatch on {m['date']} {m['venue']}: {got} vs {want}"
        cands_live, _ = enumerate_candidates(live_like, None, live_cfg)
        assert len(cands_live) == len(per_meeting[mi]), "candidate count mismatch"
        parity = {"date": m["date"], "venue": m["venue"], "n_items": len(got), "keys": [k for k, _ in got],
                  "n_candidates": len(cands_live)}

    def block(mis_set, seasons):
        rr = [r for r in sel_rows if r["season"] in seasons]
        mis = [i for i in range(len(meetings)) if meetings[i]["season"] in seasons]
        fam = {}
        for f in FAMILIES:
            x = [r for r in rr if r["family"] == f]
            if x:
                fam[f] = {**_agg(x), "roi_ci": _boot_ci(x, mis)}
        mrs = [m for m in meet_rows if m["season"] in seasons]
        return {"overall": {**_agg(rr), "roi_ci": _boot_ci(rr, mis)}, "by_family": fam,
                "n_meetings": len(mis), "zero_pick_days": sum(1 for m in mrs if not m["n_items"]),
                "zero_pick_share": round(sum(1 for m in mrs if not m["n_items"]) / len(mrs), 4) if mrs else None,
                "avg_items_per_day": round(float(np.mean([m["n_items"] for m in mrs])), 2) if mrs else None,
                "avg_stake_per_day": round(float(np.mean([m["stake"] for m in mrs])), 2) if mrs else None}

    by_season = {s: block(None, (s,)) for s in sorted({m["season"] for m in meetings})}
    js = {"source": source, "generated_at": datetime.now(HKT).replace(tzinfo=None).isoformat(timespec="seconds"),
          "period": [meetings[0]["date"], meetings[-1]["date"]] if meetings else None,
          "n_meetings": len(meetings), "n_candidates": int(n_cands),
          "thresholds": thr, "module_thresholds": THRESHOLDS,
          "matches_module": {f: (thr.get(f) == THRESHOLDS.get(f)) for f in FAMILIES},
          "calibration_fit": {(f"{k[0]}_{k[1]}" if isinstance(k, tuple) else k): v for k, v in calib_fit.items()},
          "adjustment_fit": adj_fit, "adjustment_used": {"ret_adj": RET_ADJ, "hit_adj": HIT_ADJ},
          "elimination": elim,
          "calibration_used": {**{k: list(v) for k, v in CALIB.items()},
                               **{f"{k[0]}_{k[1]}": list(v) for k, v in MR_CALIB.items()}},
          "config": {k: v for k, v in cfg.items() if k in ("unit_stake", "max_items", "max_total", "max_item_stake",
                                                         "max_per_race", "max_allups", "allup_max_legs")},
          "in_sample": block(None, IS_SEASONS), "oos": block(None, OOS_SEASONS), "by_season": by_season,
          "grid": grid, "parity": parity, "meetings": meet_rows,
          "selected": sel_rows[-400:],
          "caveat": ("回測用最終獨贏賠率（開跑前先知道）推算所有彩池派彩；真實賽前約 2 小時嘅賠率會唔同，"
                     "結果偏樂觀。模型預測係 walk-forward（每次只用之前嘅賽果訓練）。門檻只用 2024-25 季調。")}
    if write:
        p = Path(path) if path else BACKTEST_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(js, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
        write_report_section(js, report_path)
    return js


def _p(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def _ci(c):
    return "-" if not c or c[0] is None else f"[{c[0] * 100:+.0f}%, {c[1] * 100:+.0f}%]"


def report_section(js) -> str:
    L = [REPORT_HEADING, "",
         f"來源 `{js['source']}`（walk-forward 模型預測），期間 {js['period'][0]} 至 {js['period'][1]}，"
         f"{js['n_meetings']} 個賽馬日，共評估 {js['n_candidates']:,} 個候選玩法（每場比較獨贏、位置、連贏、位置Q、"
         "二重彩、單T、三重彩、四連環、四重彩、孖寶、三寶同過關嘅各種結構）。"
         "每個賽馬日都係行同實盤一樣嘅 `picks.make_picks` 計法（已喺最後一個樣本外賽馬日核對結果完全一樣），"
         "用真實派彩結算。", "",
         "> ⚠️ " + js["caveat"], "",
         "### 門檻（只用 2024-25 季調校）", "",
         "每類玩法喺 2024-25 試晒「估計回報 ≥ τ、預計命中 ≥ h」嘅組合（同實盤一樣行每日上限），"
         f"要至少 {min(TUNE_MIN_N.values())}–60 項而且預計命中 ≥ {TUNE_MIN_EXP_HITS:.0f} 次先算數；"
         "揀「剔除最大一注派彩之後」回報最好嘅設定（避免一次中四重彩/三寶頭獎就決定晒），"
         "最好嘅設定都蝕錢嘅玩法就**停用**（唔會評做精選）。估計回報先按每類玩法 2024-25 嘅"
         "「實際 ÷ 估計」比例校正（例如四重彩估計偏高約三分之一，乘 0.62）。", "",
         "| 玩法 | τ（估計回報） | h（預計命中） | 2024-25 項目 | 2024-25 回報 | 2024-25 回報（剔除最大一注派彩） |",
         "|---|---:|---:|---:|---:|---:|"]
    grid = pd.DataFrame(js["grid"]) if js["grid"] else pd.DataFrame()
    for f in FAMILIES:
        th = js["thresholds"].get(f)
        if th and not len(grid):
            L.append(f"| {FAMILY_ZH[f]} | {th['tau'] * 100:+.0f}% | {th['h'] * 100:.0f}% | - | - | - |")
        elif th:
            g = grid[(grid["family"] == f) & (grid["tau"] == th["tau"]) & (grid["h"] == th["h"])].iloc[0]
            L.append(f"| {FAMILY_ZH[f]} | {th['tau'] * 100:+.0f}% | {th['h'] * 100:.0f}% | {int(g['n'])} | {_p(g['roi'])} | "
                     f"{_p(g['robust_roi'])} |")
        else:
            if len(grid):
                g = grid[(grid["family"] == f) & (grid["n"] >= TUNE_MIN_N[f]) & (grid["exp_hits"] >= TUNE_MIN_EXP_HITS)]
                bt = g.sort_values("robust_roi", ascending=False).iloc[0] if len(g) else None
            else:
                bt = None
            if bt is None:
                L.append(f"| {FAMILY_ZH[f]} | 停用 | - | 注數不足 | - | - |")
            else:
                L.append(f"| {FAMILY_ZH[f]} | 停用（最好 τ {bt['tau'] * 100:+.0f}%） | {bt['h'] * 100:.0f}% | {int(bt['n'])} | "
                         f"{_p(bt['roi'])} | {_p(bt['robust_roi'])} |")
    L += ["", "### 結果（每個賽馬日實際入選嘅精選）", "",
          "| 期間 | 賽馬日 | 冇精選日子 | 平均每日項目 | 平均每日注碼 | 項目數 | 命中率 | 注碼 | 派彩 | 回報率 | 95% 區間（按賽馬日 bootstrap） |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for lab, b in (("2024-25（樣本內）", js["in_sample"]), ("2025-26 + 2026-27（樣本外）", js["oos"])):
        o = b["overall"]
        hit = "-" if o["hit_rate"] is None else f"{o['hit_rate'] * 100:.1f}%"
        L.append(f"| {lab} | {b['n_meetings']} | {b['zero_pick_days']}（{(b['zero_pick_share'] or 0) * 100:.0f}%） | "
                 f"{b['avg_items_per_day']} | ${b['avg_stake_per_day'] or 0:,.0f} | {o['n']} | {hit} | "
                 f"${o['stake']:,.0f} | ${o['payout']:,.0f} | {_p(o['roi'])} | {_ci(o['roi_ci'])} |")
    L += ["", "#### 按玩法", "",
          "| 玩法 | 樣本內 項目 / 命中率 / 回報 | 樣本外 項目 | 樣本外 有入選嘅賽馬日 | 樣本外 命中率 | 樣本外 回報 | 樣本外 95% 區間 |",
          "|---|---|---:|---:|---:|---:|---|"]
    for f in FAMILIES:
        a, b = js["in_sample"]["by_family"].get(f), js["oos"]["by_family"].get(f)
        if not a and not b:
            continue
        ia = "-" if not a else f"{a['n']} / {a['hit_rate'] * 100:.0f}% / {_p(a['roi'])}"
        if b:
            L.append(f"| {FAMILY_ZH[f]} | {ia} | {b['n']} | {b['n_meetings']} | {b['hit_rate'] * 100:.1f}% | "
                     f"{_p(b['roi'])} | {_ci(b['roi_ci'])} |")
        else:
            L.append(f"| {FAMILY_ZH[f]} | {ia} | 0 | 0 | - | - | - |")
    oo = js["oos"]["overall"]
    if oo["roi"] is not None:
        if oo["roi"] < 0:
            L += ["", f"**結論**：樣本內 {_p(js['in_sample']['overall']['roi'])} 係用同一季數據揀門檻嘅結果（偏樂觀）；"
                  f"樣本外 {_p(oo['roi'])}、區間 {_ci(oo['roi_ci'])}，2024-25 嘅門檻效果冇延續落去，"
                  "證據唔支持精選有長期優勢。"]
        else:
            L += ["", f"**結論**：樣本外 {_p(oo['roi'])}、區間 {_ci(oo['roi_ci'])}；區間跨過 0 嘅話仍然唔可以當係有優勢。"]
    L += ["", "**點樣睇**：區間闊而且跨過 0 就代表冇證據有長期優勢（測試咗好多種玩法同門檻，好睇嘅數字好可能係運氣）。"
          "冇精選嘅日子系統唔會落注（「每日精選」機械人當日唔買）。精選只係模型分析，**並非投注建議**。", ""]
    if js.get("parity"):
        p = js["parity"]
        L.append(f"實盤/回測一致性：{p['date']} {p['venue']} 用實盤路徑重新計，{p['n_candidates']} 個候選、"
                 f"{p['n_items']} 項精選完全一樣。")
        L.append("")
    return "\n".join(L) + "\n"


def write_report_section(js, report_path=None):
    """Replace (or append) the picks section of reports/回測報告.md."""
    from . import backtest as bt
    p = Path(report_path) if report_path else bt.REPORT_PATH
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    sec = report_section(js)
    if REPORT_HEADING in txt:
        head, rest = txt.split(REPORT_HEADING, 1)
        nxt = rest.find("\n## ")
        tail = rest[nxt + 1:] if nxt >= 0 else ""
        txt = head + sec + ("\n" + tail if tail else "")
    else:
        txt = txt.rstrip("\n") + "\n\n" + sec
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")
    return p
