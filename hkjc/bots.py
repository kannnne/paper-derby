"""10 paper-trading robots (10 隻機械人).  SIMULATION ONLY - never places real bets.

Public API (SPEC_V2.md)
-----------------------
BOTS: list[dict(id, name_zh, style_zh, description_zh, config, start_bankroll, unit_stake)]
decide(bot, pred_df, odds, bankroll, cache=None) -> list[Decision]
    Decision = dict(bot_id, race_no, action 'bet'|'skip', structure, selections, stake,
                    reason, ticket (None or a SPEC ticket dict)).
    EVERY race of ``pred_df`` gets at least one row per robot (one row per ticket
    when it bets, one 'skip' row otherwise), each with a specific Chinese reason
    quoting the numbers used.  Meeting-level robots (fixed_combo, pla_parlay) also
    write race_no=0 rows (one per parlay ticket, or one skip row); their per-race rows
    say whether that race is a leg (stake 0 - the stake sits on the race_no=0 row).
run_bots(conn, run_id, date, venue, pred_df, odds, bots=None) -> list[Decision]
    per robot: void + refund its earlier open tickets of the meeting, read its own
    bankroll, decide, paper.place_tickets(bot_id=..., reason=...), replace its
    bot_decisions rows (ticket_id linked).  Re-running a meeting is idempotent.
backtest_bots(conn, source='wf-combined-v1', tune=True, write=True, path=...) -> DataFrame
    walk-forward history of every robot through the SAME ``decide`` code, thresholds
    tuned on 2024-25 only, out-of-sample 2025-26 + 2026-27; writes data/bot_backtest.json.
meeting_frame(conn, date, venue, source='wf-combined-v1') -> DataFrame
    a past meeting as a live-style pred_df (predictions + final win odds) for dry runs.

Inputs
------
pred_df: race_no, horse_no, p_win, p_place (model.predict_card / pipeline frame);
    optional race_id, scratched, win_odds, win_odds_live, pla_odds, horse_name.
odds: {race_no: {pool: {combination: decimal odds per $1}}} (odds.latest_odds) or None.

Odds used in a decision: live odds when given (WIN/PLA/QIN/QPL); otherwise the
backtest's dividend estimator (strategy.DividendEstimator: market-implied from win
odds, Harville with the 2024-25 lambda discount, calibrated on 2024-25 dividends) and
the reason says 「(估計賠率)」; with no market at all (odds not open yet) the fair
model odds less the pool takeout are used (「(估計賠率：未開售，用模型推算)」) - value
robots then never find value and skip.  Scratched horses (pred_df.scratched, or
'SCR' = None in the live WIN/PLA odds) are dropped before any decision.

Caveat: the backtest prices with FINAL win odds (known only at the off) and estimates
every other pool from them, so historical results are optimistic versus betting 2 hours
before the first race.
"""
from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import bets, paper, strategy as st

log = logging.getLogger("hkjc.bots")

UNIT = 10.0
START_BANKROLL = 10_000.0
LAM2 = st.ADAPTIVE_DEFAULT["lam2"]
LAM3 = st.ADAPTIVE_DEFAULT["lam3"]
CALIB = st.ADAPTIVE_DEFAULT["calib"]
BACKTEST_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_backtest.json"
IS_SEASONS = ("2024-25",)
OOS_SEASONS = ("2025-26", "2026-27")
CAVEAT = ("回測用最終獨贏賠率（開跑前先知道），其他彩池派彩由獨贏賠率估計；"
          "真實賽前 2 小時落注嘅結果會差啲。門檻只用 2024-25 季調校，2025-26 + 2026-27 係樣本外。")
SRC_TAG = {"live": "", "est": "(估計賠率)", "model": "(估計賠率：未開售，用模型推算)"}
_SRC_RANK = {"live": 0, "est": 1, "model": 2}
POOL_ZH = {"WIN": "獨贏", "PLA": "位置", "QIN": "連贏", "QPL": "位置Q", "TRI": "單T", "FF": "四連環",
           "TCE": "三重彩", "FCT": "二重彩"}

# =====================================================================================
# robot definitions.  Threshold values = tuned on 2024-25 with wf-combined-v1
# (backtest_bots(tune=True); see data/bot_backtest.json for the grids' winners).
# =====================================================================================
_CFG = {
    "place_only": {"kind": "place_only", "min_ev": 0.9, "min_p": 0.3, "max_per_race": 2},
    "fixed_combo": {"kind": "fixed_combo"},
    "adaptive": {"kind": "adaptive", "tau": 0.0, "h_min": 0.30, "max_stake": 200.0},
    "win_value": {"kind": "win_value", "min_ev": 0.95, "min_p": 0.15},
    "qpl_box": {"kind": "qpl_box", "min_hit": 0.3, "tau": None},
    "qin_banker": {"kind": "qin_banker", "min_p1": 0.3, "min_ratio": 1.5, "n_legs": 2, "tau": 0.0},
    "tri_banker": {"kind": "tri_banker", "min_top2": 0.4, "min_p2": 0.15, "n_legs": 4, "tau": None},
    "pla_parlay": {"kind": "pla_parlay", "n_legs": 3, "spec": "3x4", "min_p": 0.65, "min_ev": 0.85},
    "favourite": {"kind": "favourite"},
    "monkey": {"kind": "monkey"},
}

# tuning grids (2024-25 only).  Robots without a grid have nothing to tune.
GRIDS = {
    # NB the estimated place / win odds (from final win odds, calibrated on 2024-25) put
    # almost every horse's EV between 0.8 and 1.0 - the model is close to the market -
    # so the value thresholds are searched below 1 as well and chosen by 2024-25 ROI.
    "place_only": {"min_ev": [0.85, 0.88, 0.9, 0.92, 0.95, 1.0, 1.05], "min_p": [0.0, 0.3, 0.5]},
    "adaptive": {"tau": [0.0, 0.1, 0.2, 0.4], "h_min": [0.05, 0.15, 0.30]},
    "win_value": {"min_ev": [0.9, 0.95, 1.0, 1.05, 1.1], "min_p": [0.0, 0.08, 0.15]},
    "qpl_box": {"min_hit": [0.3, 0.4, 0.5, 0.6], "tau": [None, 0.0, 0.1]},
    "qin_banker": {"min_p1": [0.25, 0.3, 0.35, 0.4], "n_legs": [2, 3, 4], "tau": [None, 0.0]},
    "tri_banker": {"min_top2": [0.4, 0.5, 0.6], "n_legs": [3, 4, 5], "tau": [None, 0.0]},
    "pla_parlay": {"min_p": [0.0, 0.5, 0.55, 0.6, 0.65], "min_ev": [None, 0.85, 0.9]},
}
MIN_IS_BETS = {"pla_parlay": 20}      # parlay: >= 20 meetings with a ticket; others >= 100 tickets
DEFAULT_MIN_IS_BETS = 100


def _ev_txt(x):
    return "唔睇期望值" if x is None else f"≥{x:.2f}"


_LOW_EV_NOTE = "（門檻低過 1，因為賽前估計賠率通常偏保守；數值係用 2024-25 季歷史數據調出嚟）"


def _describe(bid, c):
    if bid == "place_only":
        return ("每場計每隻馬嘅「上名率 × 位置賠率」（期望值），"
                f"期望值{_ev_txt(c['min_ev'])}" + (f"而且上名率≥{c['min_p']:.0%}" if c["min_p"] else "")
                + f"先買位置，每場最多 {c['max_per_race']} 隻，每隻 $10"
                + (_LOW_EV_NOTE if c["min_ev"] < 1 else "") + "。"
                "唔夠門檻就成場唔買，所以有啲日子可能一注都冇。只買單場位置，唔玩過關、唔玩其他彩池。")
    if bid == "fixed_combo":
        return ("每個賽馬日照買用戶原本嘅固定組合：1 條位置 6x63 過關（模型最有信心嘅 6 場、每場 1 隻，"
                "63 注 $630），加 3 條位置Q 2x3 過關（另揀 6 場、每場頭 3 選，每條 15 注 $150）。"
                "唔理賠率，純粹跟模型信心，係比較用嘅基準。")
    if bid == "adaptive":
        return ("每場將獨贏、位置、連贏、位置Q、單T、四連環等三十幾種玩法逐個計預期回報同命中機率，"
                f"揀預期回報最高、而且預期回報≥{c['tau']:+.0%}、命中機率≥{c['h_min']:.0%} 嘅一種"
                f"（每場最多 ${c['max_stake']:.0f}）；冇一種合格就嗰場唔買。")
    if bid == "win_value":
        return ("每場計每隻馬嘅「勝率 × 獨贏賠率」，揀最高嗰隻；"
                f"期望值{_ev_txt(c['min_ev'])}" + (f"而且勝率≥{c['min_p']:.0%}" if c["min_p"] else "")
                + " 先買獨贏 $10" + (_LOW_EV_NOTE if c["min_ev"] < 1 else "")
                + "。即係買模型覺得被低估嘅馬，中嘅次數少但一中就多。")
    if bid == "qpl_box":
        return ("睇模型頭三選（按上名率）夠唔夠集中：三隻之中至少兩隻跑入前三嘅機率"
                f"≥{c['min_hit']:.0%}" + ("" if c["tau"] is None else f"、預期回報≥{c['tau']:+.0%}")
                + " 就買呢三隻嘅位置Q 複式（3 注 $30）；唔夠集中就唔買。少過 7 匹出賽冇位置Q。")
    if bid == "qin_banker":
        return (f"模型首選勝率≥{c['min_p1']:.0%}、而且最少係第二選嘅 {c['min_ratio']:.1f} 倍（明顯首選），"
                f"就用佢做膽，拖其後 {c['n_legs']} 隻買連贏（{c['n_legs']} 注）"
                + ("" if c["tau"] is None else f"，預期回報要≥{c['tau']:+.0%}") + "。冇明顯首選就唔買。")
    if bid == "tri_banker":
        return (f"模型頭兩選勝率加埋≥{c['min_top2']:.0%}（而且第二選≥{c['min_p2']:.0%}）就當佢哋係兩隻強馬做雙膽，"
                f"拖之後 {c['n_legs']} 隻買單T（{c['n_legs']} 注）"
                + ("" if c["tau"] is None else f"，預期回報要≥{c['tau']:+.0%}")
                + "。要頭三名全中先有派彩，機會細但派彩大。")
    if bid == "pla_parlay":
        return (f"全日揀模型上名率最高嘅 {c['n_legs']} 場首選，每場首選上名率都要≥{c['min_p']:.0%}"
                + ("" if c["min_ev"] is None else f"、位置期望值≥{c['min_ev']:.2f}")
                + f"，就買位置 {c['spec']} 過關（3 個二關 + 1 個三關 = 4 注 $40）。唔夠 {c['n_legs']} 場合格就成日唔買。")
    if bid == "favourite":
        return ("每場揀獨贏賠率最低嗰隻（大熱），買獨贏 $10 + 位置 $10。完全唔用模型，"
                "係對照組：睇吓其他機械人有冇贏到「跟大熱」。未有賠率時用模型首選代替。")
    if bid == "monkey":
        return ("每場隨機揀一隻馬買位置 $10，隨機種子用場次編號（race_id），所以結果可以重現。"
                "代表「完全唔識睇馬」，長遠應該大約蝕抽水嗰個數，係最基本嘅對照組。")
    raise KeyError(bid)


_META = [
    ("place_only", "位置專家", "只買單場位置，揀上名機會高又抵食嘅馬"),
    ("fixed_combo", "固定組合", "用戶原本嘅固定過關組合，每個賽馬日照買"),
    ("adaptive", "按場自動", "每場自己揀最抵嘅玩法，唔抵就唔買"),
    ("win_value", "獨贏價值", "搵賠率被低估嘅獨贏"),
    ("qpl_box", "位置Q 複式", "頭三選夠集中先出手買位置Q"),
    ("qin_banker", "連贏膽拖", "有明顯首選先買，一膽拖幾腳"),
    ("tri_banker", "單T 膽拖", "兩隻強馬做雙膽搏單T"),
    ("pla_parlay", "位置過關", "揀全日最穩陣嘅 3 場做 3x4 過關"),
    ("favourite", "跟大熱（對照組）", "每場買大熱獨贏 + 位置"),
    ("monkey", "隨機猴子（對照組）", "每場隨機揀一隻買位置"),
]


def _make_bot(bid, name, style, cfg):
    return {"id": bid, "name_zh": name, "style_zh": style, "description_zh": _describe(bid, cfg),
            "config": dict(cfg), "start_bankroll": START_BANKROLL, "unit_stake": UNIT}


BOTS = [_make_bot(b, n, s, _CFG[b]) for b, n, s in _META]
BOT_BY_ID = {b["id"]: b for b in BOTS}


def get_bot(bot):
    if isinstance(bot, str):
        return BOT_BY_ID[bot]
    return bot


# =====================================================================================
# formatting helpers
# =====================================================================================
def _pc(x):
    return f"{x * 100:.0f}%"


def _od(x):
    if x is None:
        return "-"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def _money(x):
    return f"${x:,.0f}"


def _hs(hs):
    return "、".join(str(int(h)) for h in hs)


def _worst(srcs):
    srcs = list(srcs) or ["live"]
    return max(srcs, key=lambda s: _SRC_RANK[s])


# =====================================================================================
# per-race context (probabilities + odds), shared by every robot
# =====================================================================================
class RaceCtx:
    """One race after scratchings: model probabilities, live odds, estimator."""

    def __init__(self, rn, race_id, g, live):
        self.rn = int(rn)
        self.race_id = race_id
        g = g.sort_values("horse_no").reset_index(drop=True)
        self.df = g
        self.horses = [int(h) for h in g["horse_no"]]
        self.n = len(self.horses)
        self.pw = dict(zip(self.horses, g["p_win"].astype(float)))
        self.pp = dict(zip(self.horses, g["p_place"].astype(float)))
        wo = pd.to_numeric(g["win_odds"], errors="coerce")
        self.win_odds = {h: float(o) for h, o in zip(self.horses, wo) if pd.notna(o) and o > 0}
        self.live = live                                   # {pool: {combo: odds}} (non-None only)
        self.order_win = sorted(self.horses, key=lambda h: (-self.pw[h], h))
        self.order_place = sorted(self.horses, key=lambda h: (-self.pp[h], h))
        self.probs = st.RaceProbs(self.horses, g["p_win"].to_numpy(float), LAM2, LAM3)
        self.est = st.DividendEstimator(g, live or None, CALIB, LAM2, LAM3)
        self._cands = None

    @property
    def n_places(self):
        return self.probs.n_places

    def model_p(self, pool, combo):
        if pool == "PLA":
            return self.pp.get(int(combo), 0.0)
        return self.probs.prob(pool, combo)

    def odds(self, pool, combo):
        """(decimal odds per $1, source) with source live / est / model."""
        v = (self.live.get(pool) or {}).get(combo)
        if v:
            return max(1.0, float(v)), "live"
        d = self.est(pool, combo)
        if d is not None:
            return d / 10.0, "est"
        p = self.probs.prob(pool, combo) if pool != "PLA" else self.probs.place[self.probs.ix[int(combo)]]
        if p <= 0:
            return None, "model"
        return max(1.0, (1 - st.TAKEOUT[pool]) / p), "model"

    def combos_value(self, pool, combos):
        """(expected return per unit staked, source): sum P(c) * odds(c) / n."""
        er, srcs = 0.0, []
        for c in combos:
            o, s = self.odds(pool, c)
            srcs.append(s)
            if o is None:
                continue
            er += self.probs.prob(pool, c) * o
        return er / max(len(combos), 1), _worst(srcs)

    def adaptive_cands(self):
        if self._cands is None:
            cfg = {**st.ADAPTIVE_DEFAULT, "unit_stake": UNIT}
            self._cands = st.race_candidates(self.df, self.live or None, cfg, probs=self.probs, est=self.est)
        return self._cands


def _race_odds(odds, rn):
    if not odds:
        return {}
    r = odds.get(int(rn))
    if r is None:
        r = odds.get(str(int(rn)))
    return r or {}


def prepare(pred_df: pd.DataFrame, odds=None, cache=None) -> list[RaceCtx]:
    """pred_df + live odds -> [RaceCtx] (race order), scratched horses removed.

    ``cache`` (dict, optional): memo of RaceCtx by (race_id, horses) - only valid when
    the same inputs are passed again (the backtest grid)."""
    need = {"race_no", "horse_no", "p_win", "p_place"}
    miss = need - set(pred_df.columns)
    if miss:
        raise ValueError(f"pred_df missing columns {sorted(miss)}")
    df = pred_df.copy()
    df["race_no"] = df["race_no"].astype(int)
    df["horse_no"] = df["horse_no"].astype(int)
    if "scratched" in df.columns:
        df = df[pd.to_numeric(df["scratched"], errors="coerce").fillna(0).astype(int) == 0]
    df = df[df["p_win"].notna() & df["p_place"].notna()]
    out = []
    for rn, g in df.groupby("race_no", sort=True):
        orc = _race_odds(odds, rn)
        live = {}
        scr = set()
        for pool, m in orc.items():
            if not m:
                continue
            good = {str(k): float(v) for k, v in m.items() if v}
            if good:
                live[pool] = good
                if pool in ("WIN", "PLA"):          # 'SCR' arrives as None next to priced horses
                    scr |= {int(k) for k, v in m.items() if not v and str(k).isdigit()}
        g = g[~g["horse_no"].isin(scr)].copy()
        if g.empty:
            continue
        for col, pool in (("win_odds_live", "WIN"), ("pla_odds", "PLA")):
            if pool not in live and col in g.columns:
                vals = {str(int(h)): float(v) for h, v in zip(g["horse_no"], pd.to_numeric(g[col], errors="coerce"))
                        if pd.notna(v) and v > 0}
                if vals:
                    live[pool] = vals
        if "WIN" in live:
            g["win_odds"] = [live["WIN"].get(str(int(h))) for h in g["horse_no"]]
        elif "win_odds" not in g.columns:
            g["win_odds"] = np.nan
        g["win_odds"] = pd.to_numeric(g["win_odds"], errors="coerce")
        g["p_mkt"] = st.market_probs(g["win_odds"].to_numpy(float))
        rid = None
        if "race_id" in g.columns and g["race_id"].notna().any():
            rid = str(g["race_id"].dropna().iloc[0])
        key = (rid, int(rn), tuple(int(h) for h in g["horse_no"]))
        if cache is not None and rid is not None and key in cache:
            out.append(cache[key])
            continue
        ctx = RaceCtx(rn, rid, g[["race_no", "horse_no", "p_win", "p_place", "win_odds", "p_mkt"]], live)
        if cache is not None and rid is not None:
            cache[key] = ctx
        out.append(ctx)
    return out


# =====================================================================================
# decisions
# =====================================================================================
class _Bank:
    def __init__(self, amount):
        self.left = float(amount)

    def can(self, stake):
        return self.left + 1e-9 >= stake

    def take(self, stake):
        self.left = round(self.left - stake, 2)


def _dec(bot_id, rn, action, structure, selections, stake, reason, ticket=None):
    return {"bot_id": bot_id, "race_no": int(rn), "action": action, "structure": structure,
            "selections": selections, "stake": float(stake), "reason": reason, "ticket": ticket}


def _single_ticket(bot_id, rn, pool, leg, code):
    t = st._ticket(f"{bot_id}:{code}", rn, pool, None, [{"race_no": rn, "pool": pool, **leg}], UNIT)
    return t


def _bet_or_broke(bank, bot_id, rn, structure, selections, ticket, reason):
    """'bet' decision if the bankroll covers the ticket, else a 資金不足 skip."""
    stake = ticket["total_stake"]
    if not bank.can(stake):
        return _dec(bot_id, rn, "skip", structure, selections, 0.0,
                    f"資金不足：餘額 {_money(bank.left)} 唔夠注碼 {_money(stake)}（原本：{reason}）→ 唔買")
    bank.take(stake)
    ticket = {**ticket, "reason": reason, "bot_id": bot_id}
    return _dec(bot_id, rn, "bet", structure, selections, stake, reason, ticket)


def _too_small(bot_id, ctx, structure):
    return _dec(bot_id, ctx.rn, "skip", structure, [], 0.0, f"得 {ctx.n} 匹出賽，太少 → 唔買")


# ---- place_only ---------------------------------------------------------------------
def _place_only(bid, c, ctxs, bank):
    out = []
    S = "PLA 位置"
    for x in ctxs:
        if x.n < 4:
            out.append(_too_small(bid, x, S))
            continue
        rows = []
        for h in x.horses:
            o, s = x.odds("PLA", str(h))
            if o is None:
                continue
            rows.append((x.pp[h] * o, h, o, s))
        rows.sort(key=lambda r: (-r[0], r[1]))
        if not rows:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0, "冇位置賠率亦估唔到 → 唔買"))
            continue
        ok = [r for r in rows if r[0] >= c["min_ev"] - 1e-12 and x.pp[r[1]] >= c["min_p"] - 1e-12
              and r[3] != "model"]
        ok = ok[:c["max_per_race"]]
        if not ok:
            ev, h, o, s = rows[0]
            why = (f"最高期望值只得{ev:.2f}（{h}號 上名率{_pc(x.pp[h])} × 位置賠率{_od(o)}{SRC_TAG[s]}），"
                   f"唔夠門檻{c['min_ev']:.2f} → 唔買")
            if s == "model":
                why = (f"未開售冇市場賠率：{h}號 上名率{_pc(x.pp[h])}，模型推算位置賠率{_od(o)}{SRC_TAG[s]}，"
                       f"期望值{ev:.2f} 唔可以當真（門檻{c['min_ev']:.2f}要真賠率）→ 唔買")
            elif ev >= c["min_ev"]:
                why = (f"{h}號 期望值{ev:.2f}{SRC_TAG[s]}夠高，但上名率只得{_pc(x.pp[h])}"
                       f"（要≥{_pc(c['min_p'])}）→ 唔買")
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0, why))
            continue
        sels = [r[1] for r in ok]
        parts = [f"{h}號 上名率{_pc(x.pp[h])}，位置賠率{_od(o)}{SRC_TAG[s]} → 期望值{ev:.2f} ≥ 門檻{c['min_ev']:.2f}"
                 for ev, h, o, s in ok]
        t = _single_ticket(bid, x.rn, "PLA", {"selections": sels}, "PLA")
        reason = "；".join(parts) + f"，買位置 {_money(t['total_stake'])}"
        out.append(_bet_or_broke(bank, bid, x.rn, S, sels, t, reason))
    return out


# ---- win_value -----------------------------------------------------------------------
def _win_value(bid, c, ctxs, bank):
    out = []
    S = "WIN 獨贏"
    for x in ctxs:
        if x.n < 4:
            out.append(_too_small(bid, x, S))
            continue
        rows = []
        for h in x.horses:
            o, s = x.odds("WIN", str(h))
            if o is not None:
                rows.append((x.pw[h] * o, h, o, s))
        rows.sort(key=lambda r: (-r[0], r[1]))
        if not rows:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0, "冇獨贏賠率 → 唔買"))
            continue
        ok = [r for r in rows if r[0] >= c["min_ev"] - 1e-12 and x.pw[r[1]] >= c["min_p"] - 1e-12
              and r[3] != "model"]
        if not ok:
            ev, h, o, s = rows[0]
            why = (f"最高期望值只得{ev:.2f}（{h}號 勝率{_pc(x.pw[h])} × 獨贏賠率{_od(o)}{SRC_TAG[s]}），"
                   f"唔夠門檻{c['min_ev']:.2f} → 唔買")
            if s == "model":
                why = (f"未開售冇市場賠率：{h}號 勝率{_pc(x.pw[h])}，模型推算獨贏賠率{_od(o)}{SRC_TAG[s]}，"
                       f"期望值{ev:.2f} 唔可以當真（門檻{c['min_ev']:.2f}要真賠率）→ 唔買")
            elif ev >= c["min_ev"]:
                why = (f"{h}號 期望值{ev:.2f}{SRC_TAG[s]}，但勝率只得{_pc(x.pw[h])}"
                       f"（要≥{_pc(c['min_p'])}）→ 唔買")
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0, why))
            continue
        ev, h, o, s = ok[0]
        t = _single_ticket(bid, x.rn, "WIN", {"selections": [h]}, "WIN")
        reason = (f"{h}號 勝率{_pc(x.pw[h])}，獨贏賠率{_od(o)}{SRC_TAG[s]} → 期望值{ev:.2f} ≥ 門檻{c['min_ev']:.2f}，"
                  f"買獨贏 {_money(t['total_stake'])}")
        out.append(_bet_or_broke(bank, bid, x.rn, S, [h], t, reason))
    return out


# ---- adaptive ------------------------------------------------------------------------
def _leg_txt(leg):
    if leg.get("bankers"):
        return f"膽 {_hs(leg['bankers'])} 拖 {_hs(leg['selections'])}"
    return _hs(leg.get("selections", []))


def _adaptive(bid, c, ctxs, bank):
    out = []
    for x in ctxs:
        cands = x.adaptive_cands() if x.n >= 4 else []
        if not cands:
            why = (f"得 {x.n} 匹出賽，太少 → 唔買" if x.n < 4 else
                   f"未開售冇賠率（{x.n} 匹出賽），計唔到任何玩法嘅預期回報 → 唔買")
            out.append(_dec(bid, x.rn, "skip", "自動", [], 0.0, why))
            continue
        best = st.choose_candidate(cands, c["tau"], c["h_min"], c.get("max_stake"))
        if best is None:
            top = max(cands, key=lambda k: (round(k["exp_roi"], 12), k["hit"]))
            src = "live" if top["pool"] in x.live else "est"
            if top["exp_roi"] < c["tau"]:
                why = (f"最好嘅玩法係 {top['name']}（{_leg_txt(top['leg'])}），預期回報只得{top['exp_roi']:+.0%}"
                       f"{SRC_TAG[src]}，唔夠門檻{c['tau']:+.0%} → 唔買")
            else:
                good = [k for k in cands if k["exp_roi"] >= c["tau"]]
                k = max(good, key=lambda k: k["hit"])
                why = (f"預期回報夠嘅玩法（例如 {k['name']}，{k['exp_roi']:+.0%}{SRC_TAG[src]}）命中機率只得"
                       f"{_pc(k['hit'])}，唔夠門檻{_pc(c['h_min'])} → 唔買")
            out.append(_dec(bid, x.rn, "skip", "自動", [], 0.0, why))
            continue
        leg = {k: v for k, v in best["leg"].items() if k not in ("race_no", "pool")}
        t = st._ticket(f"{bid}:{best['pool']}", x.rn, best["pool"], None, [best["leg"]], UNIT,
                       {"structure": best["name"], "exp_roi": round(best["exp_roi"], 4),
                        "hit_prob": round(best["hit"], 4)})
        src = "live" if best["pool"] in x.live else "est"
        reason = (f"揀 {best['name']}（{_leg_txt(best['leg'])}）：預期回報{best['exp_roi']:+.0%}{SRC_TAG[src]} "
                  f"≥ 門檻{c['tau']:+.0%}，命中機率{_pc(best['hit'])} ≥ {_pc(c['h_min'])}，"
                  f"買 {best['n_units']} 注 {_money(t['total_stake'])}")
        sel = leg.get("selections", []) if not leg.get("bankers") else leg
        out.append(_bet_or_broke(bank, bid, x.rn, best["name"], sel, t, reason))
    return out


# ---- qpl_box -------------------------------------------------------------------------
def _qpl_box(bid, c, ctxs, bank):
    out = []
    S = "QPL 複式3"
    for x in ctxs:
        if x.n < 7:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0, f"得 {x.n} 匹出賽（少過 7 匹冇位置Q）→ 唔買"))
            continue
        top = x.order_place[:3]
        combos = bets.multiple("QPL", top)
        hit = x.probs.hit_prob("QPL", combos)
        er, src = x.combos_value("QPL", combos)
        roi = er - 1
        head = f"頭三選 {_hs(top)}（上名率 {'/'.join(_pc(x.pp[h]) for h in top)}），位置Q 命中機率{_pc(hit)}"
        evt = f"預期回報{roi:+.0%}{SRC_TAG[src]}"
        if hit < c["min_hit"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head} < 門檻{_pc(c['min_hit'])}（唔夠集中），{evt} → 唔買"))
            continue
        if c["tau"] is not None and roi < c["tau"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head} 夠集中，但{evt} < 門檻{c['tau']:+.0%} → 唔買"))
            continue
        t = _single_ticket(bid, x.rn, "QPL", {"selections": top}, "QPL")
        reason = (f"{head} ≥ 門檻{_pc(c['min_hit'])}，{evt} → 買位置Q 複式 3 注 {_money(t['total_stake'])}")
        out.append(_bet_or_broke(bank, bid, x.rn, S, top, t, reason))
    return out


# ---- qin_banker ----------------------------------------------------------------------
def _qin_banker(bid, c, ctxs, bank):
    out = []
    n = int(c["n_legs"])
    S = f"QIN 膽1x腳{n}"
    for x in ctxs:
        if x.n < 1 + n:
            out.append(_too_small(bid, x, S))
            continue
        b, legs = x.order_win[0], x.order_win[1:1 + n]
        p1, p2 = x.pw[b], x.pw[x.order_win[1]]
        ratio = p1 / p2 if p2 > 0 else float("inf")
        head = f"首選 {b}號 勝率{_pc(p1)}，第二選 {x.order_win[1]}號 {_pc(p2)}（{ratio:.1f}倍）"
        if p1 < c["min_p1"] - 1e-12 or ratio < c["min_ratio"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head}，唔算明顯首選（要勝率≥{_pc(c['min_p1'])}而且≥{c['min_ratio']:.1f}倍）→ 唔買"))
            continue
        combos = bets.banker("QIN", [b], legs)
        hit = x.probs.hit_prob("QIN", combos)
        er, src = x.combos_value("QIN", combos)
        roi = er - 1
        if c["tau"] is not None and roi < c["tau"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head} 係明顯首選，但連贏膽拖預期回報只得{roi:+.0%}{SRC_TAG[src]}，"
                            f"唔夠門檻{c['tau']:+.0%} → 唔買"))
            continue
        t = _single_ticket(bid, x.rn, "QIN", {"bankers": [b], "selections": legs}, "QIN")
        reason = (f"{head} → 明顯首選，連贏 {b} 膽拖 {_hs(legs)} 共 {len(combos)} 注 {_money(t['total_stake'])}，"
                  f"命中機率{_pc(hit)}，預期回報{roi:+.0%}{SRC_TAG[src]}")
        out.append(_bet_or_broke(bank, bid, x.rn, S, {"bankers": [b], "legs": legs}, t, reason))
    return out


# ---- tri_banker ----------------------------------------------------------------------
def _tri_banker(bid, c, ctxs, bank):
    out = []
    n = int(c["n_legs"])
    S = f"TRI 膽2x腳{n}"
    for x in ctxs:
        if x.n < 2 + n:
            out.append(_too_small(bid, x, S))
            continue
        b = x.order_win[:2]
        legs = x.order_win[2:2 + n]
        p1, p2 = x.pw[b[0]], x.pw[b[1]]
        head = f"頭兩選 {b[0]}號({_pc(p1)}) + {b[1]}號({_pc(p2)}) 合共{_pc(p1 + p2)}"
        if p1 + p2 < c["min_top2"] - 1e-12 or p2 < c["min_p2"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head}，唔算兩隻強馬（要合共≥{_pc(c['min_top2'])}、第二選≥{_pc(c['min_p2'])}）→ 唔買"))
            continue
        combos = bets.banker("TRI", b, legs)
        hit = x.probs.hit_prob("TRI", combos)
        er, src = x.combos_value("TRI", combos)
        roi = er - 1
        if c["tau"] is not None and roi < c["tau"] - 1e-12:
            out.append(_dec(bid, x.rn, "skip", S, [], 0.0,
                            f"{head}，但單T 預期回報只得{roi:+.0%}{SRC_TAG[src]}，唔夠門檻{c['tau']:+.0%} → 唔買"))
            continue
        t = _single_ticket(bid, x.rn, "TRI", {"bankers": b, "selections": legs}, "TRI")
        reason = (f"{head} ≥ 門檻{_pc(c['min_top2'])} → 單T {_hs(b)} 膽拖 {_hs(legs)} 共 {len(combos)} 注 "
                  f"{_money(t['total_stake'])}，命中機率{_pc(hit)}，預期回報{roi:+.0%}{SRC_TAG[src]}")
        out.append(_bet_or_broke(bank, bid, x.rn, S, {"bankers": b, "legs": legs}, t, reason))
    return out


# ---- favourite / monkey (controls) ------------------------------------------------------
def _favourite(bid, c, ctxs, bank):
    out = []
    for x in ctxs:
        priced = [(o, h, s) for h in x.horses for o, s in [x.odds("WIN", str(h))] if s != "model" and o]
        if priced:
            o, h, s = min(priced, key=lambda r: (r[0], r[1]))
            head = f"大熱 {h}號 獨贏賠率{_od(o)}{SRC_TAG[s]}（全場最低）"
        else:
            h = x.order_win[0]
            head = f"未有賠率，用模型首選 {h}號（勝率{_pc(x.pw[h])}）當大熱"
        for pool, code in (("WIN", "WIN 大熱"), ("PLA", "PLA 大熱")):
            t = _single_ticket(bid, x.rn, pool, {"selections": [h]}, pool)
            reason = f"{head} → {POOL_ZH[pool]} {_money(t['total_stake'])}（每場獨贏 + 位置各 $10）"
            out.append(_bet_or_broke(bank, bid, x.rn, code, [h], t, reason))
    return out


def monkey_pick(ctx: RaceCtx) -> tuple[int, int]:
    """(horse, seed): deterministic 'random' horse of a race, seeded by its race_id."""
    key = ctx.race_id or f"R{ctx.rn}:" + ",".join(map(str, ctx.horses))
    seed = zlib.crc32(key.encode("utf-8"))
    rng = np.random.default_rng(seed)
    return int(ctx.horses[int(rng.integers(len(ctx.horses)))]), seed


def _monkey(bid, c, ctxs, bank):
    out = []
    for x in ctxs:
        h, seed = monkey_pick(x)
        t = _single_ticket(bid, x.rn, "PLA", {"selections": [h]}, "PLA")
        reason = (f"隨機（種子 {x.race_id or x.rn}）喺 {x.n} 匹入面揀中 {h}號"
                  f"（模型上名率{_pc(x.pp[h])}）→ 位置 {_money(t['total_stake'])}")
        out.append(_bet_or_broke(bank, bid, x.rn, "PLA 隨機", [h], t, reason))
    return out


# ---- meeting-level: fixed_combo / pla_parlay ------------------------------------------
def _leg_rows(bid, ctxs, tickets_dec, label):
    """Per-race rows of a meeting-level robot: is this race a leg of a placed parlay?"""
    out = []
    for x in ctxs:
        parts, placed, any_leg, sels = [], False, False, {}
        for d, t in tickets_dec:
            for leg in t["legs"]:
                if int(leg["race_no"]) == x.rn:
                    any_leg = True
                    name = d["structure"]
                    sel = leg["selections"]
                    sels[name] = list(sel)
                    conf = "/".join(_pc(x.pp[h]) for h in sel)
                    parts.append(f"{name} 揀 {_hs(sel)}（上名率{conf}）"
                                 + ("" if d["action"] == "bet" else "，但呢條過關冇買（" +
                                    ("資金不足" if "資金不足" in d["reason"] else "唔合格") + "）"))
                    placed |= d["action"] == "bet"
        h = x.order_place[0] if x.horses else None
        if any_leg:
            out.append(_dec(bid, x.rn, "bet" if placed else "skip", label,
                            sels, 0.0,
                            f"第{x.rn}場係過關腳：" + "；".join(parts) + "（注碼計喺過關飛，見第0場）"))
        else:
            top = f"首選 {h}號 上名率{_pc(x.pp[h])}" if h else f"{x.n} 匹出賽"
            why = (f"第{x.rn}場唔係過關腳（{top}，唔係入選嘅最有信心場次）" if tickets_dec else
                   f"第{x.rn}場：今日冇過關飛（原因見第0場跨場決定；{top}）")
            out.append(_dec(bid, x.rn, "skip", label, [], 0.0, why))
    return out


def _fixed_combo(bid, c, ctxs, bank):
    cols = ["race_no", "horse_no", "p_win", "p_place"]
    frame = pd.concat([x.df[cols] for x in ctxs], ignore_index=True) if ctxs else None
    decs_t = []
    head = []
    if frame is not None and len(frame):
        t = st.build_allup(frame, pool="PLA", n_legs=6, spec="6x63", horses_per_leg=1, race_rule="confidence",
                           unit_stake=UNIT, strategy=f"{bid}:PLA_6x63", _prepped=True)
        qframe = (pd.concat([x.df[cols] for x in ctxs if x.n >= 7], ignore_index=True)
                  if any(x.n >= 7 for x in ctxs) else None)
        qts = (st.build_multi_allup(qframe, pool="QPL", n_tickets=3, legs_per_ticket=2, spec="2x3",
                                    horses_per_leg=3, race_rule="confidence", pairing="chrono",
                                    unit_stake=UNIT, strategy=f"{bid}:QPL_2x3", _prepped=True)
               if qframe is not None else [])
    else:
        t, qts = None, []
    byrn = {x.rn: x for x in ctxs}
    out = []
    if t is None:
        out.append(_dec(bid, 0, "skip", "PLA 6x63", [], 0.0, f"只得 {len(ctxs)} 場有預測，唔夠 6 場做位置 6x63 → 唔買"))
    else:
        legs = "、".join(f"第{l['race_no']}場 {l['selections'][0]}號({_pc(byrn[l['race_no']].pp[l['selections'][0]])})"
                        for l in t["legs"])
        reason = f"固定玩法：模型最有信心 6 場 {legs} → 位置 6x63 過關 63 注 {_money(t['total_stake'])}"
        sel = [{"race_no": l["race_no"], "selections": l["selections"]} for l in t["legs"]]
        d = _bet_or_broke(bank, bid, 0, "PLA 6x63", sel, t, reason)
        out.append(d)
        decs_t.append((d, t))
    if not qts:
        out.append(_dec(bid, 0, "skip", "QPL 2x3", [], 0.0, "7 匹或以上出賽嘅場次唔夠 6 場，做唔到 3 條位置Q 2x3 → 唔買"))
    for q in qts:
        legs = "；".join(f"第{l['race_no']}場 {_hs(l['selections'])}" for l in q["legs"])
        name = f"QPL 2x3 #{q['ticket_no']}"
        reason = f"固定玩法：位置Q 2x3 第{q['ticket_no']}條（{legs}，每場頭 3 選）15 注 {_money(q['total_stake'])}"
        sel = [{"race_no": l["race_no"], "selections": l["selections"]} for l in q["legs"]]
        d = _bet_or_broke(bank, bid, 0, name, sel, q, reason)
        out.append(d)
        decs_t.append((d, q))
    return out + _leg_rows(bid, ctxs, decs_t, "過關腳")


def _pla_parlay(bid, c, ctxs, bank):
    n, spec = int(c["n_legs"]), c["spec"]
    S = f"PLA {spec}"
    rows = []
    for x in ctxs:
        if x.n < 4:
            continue
        h = x.order_place[0]
        o, s = x.odds("PLA", str(h))
        ev = x.pp[h] * o if o else None
        rows.append({"x": x, "h": h, "p": x.pp[h], "o": o, "s": s, "ev": ev})
    ranked = sorted(rows, key=lambda r: (-r["p"], r["x"].rn))
    ok = [r for r in ranked if r["p"] >= c["min_p"] - 1e-12
          and (c["min_ev"] is None or (r["ev"] is not None and r["s"] != "model"
                                       and r["ev"] >= c["min_ev"] - 1e-12))]
    out = []
    decs_t = []
    rule = f"上名率≥{_pc(c['min_p'])}" + ("" if c["min_ev"] is None else f"、期望值≥{c['min_ev']:.2f}")
    if len(ok) < n:
        best = "、".join(f"第{r['x'].rn}場 {r['h']}號({_pc(r['p'])})" for r in ranked[:n])
        out.append(_dec(bid, 0, "skip", S, [], 0.0,
                        f"只得 {len(ok)} 場首選合格（{rule}，要 {n} 場）；最有信心嘅係 {best} → 今日唔買"))
    else:
        pick = sorted(ok[:n], key=lambda r: r["x"].rn)
        legs = [{"race_no": r["x"].rn, "pool": "PLA", "selections": [r["h"]]} for r in pick]
        t = st._ticket(f"{bid}:PLA_{spec}", 1, "PLA", spec, legs, UNIT)
        subs = bets.allup_subsets(n, spec)
        evs = [r["ev"] if r["ev"] is not None else r["p"] for r in pick]
        e = sum(float(np.prod([evs[i] for i in s])) for s in subs) / t["n_combos"]
        src = _worst(r["s"] for r in pick)
        desc = "、".join(f"第{r['x'].rn}場 {r['h']}號(上名率{_pc(r['p'])}，位置賠率{_od(r['o'])})" for r in pick)
        reason = (f"最有信心 {n} 場：{desc}，全部 {rule} → 位置 {spec} 過關 {t['n_combos']} 注 "
                  f"{_money(t['total_stake'])}，每注期望值{e:.2f}{SRC_TAG[src]}")
        sel = [{"race_no": l["race_no"], "selections": l["selections"]} for l in legs]
        d = _bet_or_broke(bank, bid, 0, S, sel, t, reason)
        out.append(d)
        decs_t.append((d, t))
    return out + _leg_rows(bid, ctxs, decs_t, "過關腳")


_KINDS = {"place_only": _place_only, "fixed_combo": _fixed_combo, "adaptive": _adaptive,
          "win_value": _win_value, "qpl_box": _qpl_box, "qin_banker": _qin_banker,
          "tri_banker": _tri_banker, "pla_parlay": _pla_parlay, "favourite": _favourite,
          "monkey": _monkey}


def decide(bot, pred_df, odds, bankroll, cache=None) -> list[dict]:
    """Decisions of one robot for one meeting (see module docstring).

    ``bot``: a BOTS entry or its id.  ``bankroll``: the robot's balance before the
    meeting; tickets are taken in order and any ticket the remaining balance cannot
    cover becomes a 「資金不足」 skip.  ``cache``: optional RaceCtx memo (backtest)."""
    bot = get_bot(bot)
    cfg = {**_CFG.get(bot["id"], {}), **(bot.get("config") or {})}
    ctxs = prepare(pred_df, odds, cache)
    bank = _Bank(bankroll)
    fn = _KINDS[cfg.get("kind", bot["id"])]
    return fn(bot["id"], cfg, ctxs, bank)


def _error_rows(bid, pred_df, exc) -> list[dict]:
    """'skip' rows (every race + race 0 for meeting-level robots) when a robot crashed."""
    try:
        rns = sorted({int(x) for x in pred_df["race_no"].dropna()})
    except Exception:
        rns = []
    kind = _CFG.get(bid, {}).get("kind", bid)
    if kind in ("fixed_combo", "pla_parlay"):
        rns = [0] + rns
    why = f"系統錯誤（{type(exc).__name__}），今日呢隻機械人唔落注；共 {len([r for r in rns if r])} 場都唔買"
    return [_dec(bid, rn, "skip", "-", [], 0.0, why) for rn in rns]


def tickets_of(decisions) -> list[dict]:
    return [d["ticket"] for d in decisions if d.get("ticket")]


# =====================================================================================
# live: run every robot for a meeting
# =====================================================================================
def run_bots(conn, run_id, date, venue, pred_df, odds, bots=None) -> list[dict]:
    """Every robot decides for meeting (date, venue) and its tickets are paper-placed.

    Per robot: void + refund its still-open tickets of this meeting (any run - so a
    re-run is idempotent), read its own bankroll, decide, place the tickets with
    bot_id + reason, and replace its bot_decisions rows of the meeting.  A robot that
    already has SETTLED tickets for the meeting is left alone (no double betting after
    the results).  Returns all decisions (each with ticket_id or None)."""
    allout = []
    now = paper._now()
    for bot in (bots or BOTS):
        bot = get_bot(bot)
        bid = bot["id"]
        if conn.execute("SELECT 1 FROM paper_tickets WHERE bot_id=? AND date=? AND venue=? AND status='settled' "
                        "LIMIT 1", (bid, date, venue)).fetchone():
            log.warning("run_bots: %s already has settled tickets for %s %s - not re-deciding", bid, date, venue)
            continue
        paper.seed_bankroll(conn, bot.get("start_bankroll", START_BANKROLL), bid)
        paper.void_open(conn, date, venue, bid, None, f"作廢退回 (被 {run_id} 重新決定)")
        bal = paper.bankroll(conn, bid)
        try:
            decs = decide(bot, pred_df, odds, bal)
            tk = [d for d in decs if d["ticket"]]
            ids = paper.place_tickets(conn, run_id, date, venue, [d["ticket"] for d in tk],
                                      starting_bankroll=bot.get("start_bankroll", START_BANKROLL),
                                      replace=False, bot_id=bid) if tk else []
        except Exception as e:
            # one robot's bug must not stop the other robots or lose the pre-race report:
            # this robot sits the meeting out (its tickets, if any got placed, are refunded)
            log.exception("run_bots: robot %s failed for %s %s - skipping it this meeting", bid, date, venue)
            paper.void_open(conn, date, venue, bid, None, f"作廢退回 ({run_id} 系統錯誤)")
            decs = _error_rows(bid, pred_df, e)
            tk, ids = [], []
        for d, tid in zip(tk, ids):
            d["ticket_id"] = tid
        # per-race leg rows of meeting-level robots -> the first parlay ticket containing the race
        for d in decs:
            if d.get("ticket_id") is None and d["race_no"] > 0 and d["structure"] == "過關腳" and d["action"] == "bet":
                for p in tk:
                    if any(int(l["race_no"]) == d["race_no"] for l in p["ticket"]["legs"]):
                        d["ticket_id"] = p["ticket_id"]
                        break
            d.setdefault("ticket_id", None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM bot_decisions WHERE bot_id=? AND date=? AND venue=?", (bid, date, venue))
            conn.executemany(
                "INSERT OR REPLACE INTO bot_decisions (run_id, bot_id, date, venue, race_no, action, structure, "
                "selections, stake, reason, ticket_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, bid, date, venue, d["race_no"], d["action"], d["structure"],
                  json.dumps(d["selections"], ensure_ascii=False, default=int), d["stake"], d["reason"],
                  d["ticket_id"], now) for d in decs])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        allout += decs
    return allout


# =====================================================================================
# backtest
# =====================================================================================
def _meeting_frames(meetings):
    frames = []
    for m in meetings:
        dfs = []
        for rn in sorted(m["races"]):
            r = m["races"][rn]
            d = r["df"][["race_no", "horse_no", "p_win", "p_place", "win_odds"]].copy()
            d.insert(0, "race_id", r["race_id"])
            dfs.append(d)
        frames.append(pd.concat(dfs, ignore_index=True))
    return frames


def meeting_frame(conn, date, venue, source="wf-combined-v1") -> pd.DataFrame:
    """A past meeting as a live-style pred_df: model predictions of ``source`` +
    final win odds, scratched runners included with scratched=1 (decide drops them).
    Races without predictions are absent."""
    q = ("SELECT r.race_id, r.race_no, u.horse_no, u.horse_name, p.p_win, p.p_place, u.win_odds, "
         "COALESCE(u.scratched, 0) AS scratched FROM races r JOIN runners u ON u.race_id=r.race_id "
         "LEFT JOIN predictions p ON p.race_id=u.race_id AND p.horse_no=u.horse_no AND p.run_id=? "
         "WHERE r.date=? AND r.venue=? ORDER BY r.race_no, u.horse_no")
    df = pd.read_sql_query(q, conn, params=(source, date, venue))
    keep = df.groupby("race_no")["p_win"].transform(lambda s: s.notna().any())
    return df[keep].reset_index(drop=True)


def _settle_bt(t, m, cache):
    from .backtest import settle_combos
    key = (m["date"], m["venue"], t["pool"], t["spec"], t["unit_stake"],
           json.dumps(t["legs"], sort_keys=True, default=int))
    if key in cache:
        return cache[key]
    if t["spec"]:
        dbr = {l["race_no"]: m["races"][l["race_no"]]["idx"] for l in t["legs"]}
        scr = {l["race_no"]: m["races"][l["race_no"]]["scr"] for l in t["legs"]}
        res = bets.settle_allup(t["pool"], t["legs"], t["spec"], dbr, t["unit_stake"], scr)
        out = (res["stake"], res["payout"], res["n_winning_units"])
    else:
        leg = t["legs"][0]
        r = m["races"][leg["race_no"]]
        out = settle_combos(t["pool"], st.leg_combos({**leg, "pool": t["pool"]}), r["idx"], r["scr"],
                            t["unit_stake"])
    assert abs(out[0] - t["total_stake"]) < 1e-6, (out, t)
    cache[key] = out
    return out


def simulate(bot, meetings, frames, cfg=None, start=START_BANKROLL, ctx_cache=None, settle_cache=None,
             keep_decisions=False):
    """Chronological run of one robot: decide (same code as live) -> settle with the
    real dividends -> bankroll.  Returns (per-meeting rows, decisions by meeting)."""
    bot = get_bot(bot)
    if cfg is not None:
        bot = {**bot, "config": {**bot["config"], **cfg}}
    ctx_cache = {} if ctx_cache is None else ctx_cache
    settle_cache = {} if settle_cache is None else settle_cache
    bal = float(start)
    rows, kept = [], {}
    for m, fr in zip(meetings, frames):
        decs = decide(bot, fr, None, bal, cache=ctx_cache)
        stake = pay = 0.0
        n = hits = 0
        for d in decs:
            t = d["ticket"]
            if not t:
                continue
            s, p, nw = _settle_bt(t, m, settle_cache)
            stake += s
            pay += p
            n += 1
            hits += nw > 0
        bal = round(bal + pay - stake, 2)
        rows.append({"bot_id": bot["id"], "date": m["date"], "venue": m["venue"], "season": m["season"],
                     "stake": round(stake, 2), "returns": round(pay, 2), "profit": round(pay - stake, 2),
                     "n_bets": n, "n_hits": hits, "bankroll": bal,
                     "n_skip_funds": sum(1 for d in decs if "資金不足" in d["reason"])})
        if keep_decisions:
            kept[(m["date"], m["venue"])] = decs
    return rows, kept


def _agg(rows):
    stake = sum(r["stake"] for r in rows)
    ret = sum(r["returns"] for r in rows)
    n = sum(r["n_bets"] for r in rows)
    return {"stake": round(stake, 2), "returns": round(ret, 2), "profit": round(ret - stake, 2),
            "roi": round(ret / stake - 1, 4) if stake else None, "n_bets": n,
            "n_meetings": len(rows), "n_meetings_bet": sum(1 for r in rows if r["n_bets"]),
            "hit_rate": round(sum(r["n_hits"] for r in rows) / n, 4) if n else None}


def _boot_ci(rows, n_boot=2000, seed=0):
    s = np.array([r["stake"] for r in rows], float)
    p = np.array([r["returns"] for r in rows], float)
    if not len(s) or s.sum() <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(s), np.full(len(s), 1.0 / len(s)), size=n_boot).astype(float)
    sb, pb = W @ s, W @ p
    ok = sb > 0
    roi = pb[ok] / sb[ok] - 1
    return [round(float(np.percentile(roi, 2.5)), 4), round(float(np.percentile(roi, 97.5)), 4)]


def _grid_points(grid):
    keys = list(grid)
    pts = [{}]
    for k in keys:
        pts = [{**p, k: v} for p in pts for v in grid[k]]
    return pts


def tune(bot, meetings, frames, ctx_cache, settle_cache):
    """Grid search on the in-sample meetings: best ROI among settings with enough bets
    (fallback: the setting with most bets).  Returns (best cfg overrides, grid table)."""
    bot = get_bot(bot)
    grid = GRIDS.get(bot["id"])
    if not grid:
        return {}, []
    min_n = MIN_IS_BETS.get(bot["id"], DEFAULT_MIN_IS_BETS)
    table = []
    for pt in _grid_points(grid):
        rows, _ = simulate(bot, meetings, frames, pt, ctx_cache=ctx_cache, settle_cache=settle_cache)
        a = _agg(rows)
        n = a["n_meetings_bet"] if bot["id"] == "pla_parlay" else a["n_bets"]
        table.append({"params": pt, **a, "n_eligible": n})
    elig = [g for g in table if g["n_eligible"] >= min_n and g["roi"] is not None]
    if elig:
        best = max(elig, key=lambda g: (g["roi"], g["n_bets"]))
    else:
        best = max(table, key=lambda g: (g["n_eligible"], g["roi"] or -9))
    return dict(best["params"]), table


def backtest_bots(conn, source="wf-combined-v1", tune_params=True, write=True, path=None,
                  meetings=None, bots=None) -> pd.DataFrame:
    """Walk-forward history of every robot (see module docstring).  Returns one row per
    robot x meeting (phase 'is' = 2024-25 with a fresh HK$10,000, 'oos' = 2025-26 +
    2026-27 with a fresh HK$10,000) and writes ``data/bot_backtest.json``."""
    from . import backtest as bt
    meetings = meetings if meetings is not None else bt.load_data(conn, source)
    frames = _meeting_frames(meetings)
    is_idx = [i for i, m in enumerate(meetings) if m["season"] in IS_SEASONS]
    oos_idx = [i for i, m in enumerate(meetings) if m["season"] in OOS_SEASONS]
    M_is = [meetings[i] for i in is_idx]
    F_is = [frames[i] for i in is_idx]
    M_oos = [meetings[i] for i in oos_idx]
    F_oos = [frames[i] for i in oos_idx]
    ctx_cache, settle_cache = {}, {}
    out_rows, js = [], {}
    for bot in (bots or BOTS):
        bot = get_bot(bot)
        params, table = (tune(bot, M_is, F_is, ctx_cache, settle_cache) if tune_params else ({}, []))
        cfg = {**bot["config"], **params}
        is_rows, _ = simulate(bot, M_is, F_is, params, ctx_cache=ctx_cache, settle_cache=settle_cache)
        oos_rows, kept = simulate(bot, M_oos, F_oos, params, ctx_cache=ctx_cache, settle_cache=settle_cache,
                                  keep_decisions=True)
        # live/backtest parity: the last OOS meeting re-decided from an independent DB load
        if M_oos and conn is not None:
            m = M_oos[-1]
            bal_before = (oos_rows[-2]["bankroll"] if len(oos_rows) > 1 else START_BANKROLL)
            live_like = meeting_frame(conn, m["date"], m["venue"], source)
            live_like = live_like[live_like["race_id"].isin({r["race_id"] for r in m["races"].values()})]
            again = decide({**bot, "config": cfg}, live_like, None, bal_before)
            assert _strip(again) == _strip(kept[(m["date"], m["venue"])]), f"live/backtest mismatch {bot['id']}"
        for ph, rows in (("is", is_rows), ("oos", oos_rows)):
            for r in rows:
                out_rows.append({**r, "phase": ph})
        a_is, a_oos = _agg(is_rows), _agg(oos_rows)
        by_season = {}
        for s in sorted({r["season"] for r in is_rows + oos_rows}):
            by_season[s] = _agg([r for r in is_rows + oos_rows if r["season"] == s])
        curve = ([{"date": "2025-09-01", "bankroll": START_BANKROLL}] +
                 [{"date": r["date"], "venue": r["venue"], "bankroll": r["bankroll"]} for r in oos_rows])
        js[bot["id"]] = {
            "name": bot["name_zh"], "source": source,
            "params": {k: v for k, v in cfg.items() if k != "kind"},
            "tuned": bool(GRIDS.get(bot["id"])) and tune_params,
            "matches_live_config": all(bot["config"].get(k) == v for k, v in params.items()),
            "live_config": {k: v for k, v in bot["config"].items() if k != "kind"},
            "in_sample": {"period": "2024-25", **a_is},
            "oos": {"period": "2025-26 + 2026-27", **a_oos, "roi_ci": _boot_ci(oos_rows),
                    "final_bankroll": oos_rows[-1]["bankroll"] if oos_rows else START_BANKROLL,
                    "max_drawdown": _max_dd([START_BANKROLL] + [r["bankroll"] for r in oos_rows]),
                    "n_skip_funds": sum(r["n_skip_funds"] for r in oos_rows)},
            "curve": curve, "by_season": by_season,
            "grid": [{"params": g["params"], "roi": g["roi"], "n_bets": g["n_bets"],
                      "n_meetings_bet": g["n_meetings_bet"]} for g in table],
            "caveat": CAVEAT,
        }
    if write:
        p = Path(path) if path else BACKTEST_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(js, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    df = pd.DataFrame(out_rows)
    df.attrs["json"] = js
    return df


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


def _max_dd(curve):
    peak, mdd = -np.inf, 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return round(float(mdd), 2)


def _strip(decs):
    """Comparable view of decisions (drops nothing semantic; ticket dicts compared whole)."""
    return [json.dumps({k: v for k, v in d.items() if k != "ticket_id"}, sort_keys=True, default=_json_default,
                       ensure_ascii=False) for d in decs]
