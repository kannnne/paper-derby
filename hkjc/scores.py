"""Per-horse scores (馬匹評分) with sub-scores and short Traditional-Chinese reasons.

Public API
----------
score_horses(pred_df, card_df) -> DataFrame[race_id, race_no, horse_no, score, rank_in_race,
                                            sub_scores, reasons, p_win, p_place]
    ``pred_df``: model.predict_card output (or walk-forward predictions) with at least
    race_id, horse_no, p_win (p_place optional). ``card_df``: features.build_card_frame /
    build_training_frame rows (same race_id / horse_no keys). Optional ``jockey_name`` /
    ``trainer_name`` columns in card_df are used in the reasons (score_meeting adds them).
score_meeting(conn, date, venue, run_id=None, pred_df=None, odds=None, model_dir=None)
    Past meeting (results in DB): predictions of ``run_id`` (default chain: run_id ->
    'wf-combined-v1' -> latest 'pre_bet' run of that meeting -> 'wf-fundamental-v1') +
    build_training_frame rows of that date (features are leak-free).
    Upcoming meeting: ``pred_df`` if given, else stored predictions of ``run_id`` if any,
    else the latest 'combined' model (exactly what model.predict_card does, reusing the
    card frame); features from build_card_frame (+ live odds -> market sub-score).
    ~1.5-3.5 s per meeting (feature building dominates; model import ~2 s cold).
save_scores(conn, run_id, df) -> int   writes ``horse_scores`` (JSON, ensure_ascii=False).

Composite score (評分, 0-100)
-----------------------------
    x     = ln(p_win * N)            N = runners in the race (p_win sums to 1 per race),
                                     so x = 0 is the "average horse" (p_win = 1/N)
    score = 100 / (1 + exp(-(x - C) / S))      C = SCORE_C = -0.57, S = SCORE_S = 0.77

The logistic is strictly increasing, so within a race the score order is exactly the
p_win order (``rank_in_race`` 1 = highest p_win, ties broken by horse number) and the
score is bounded in (0, 100) without clipping. C and S were fitted (least squares) to
the empirical CDF of x over the walk-forward predictions 'wf-combined-v1'
(2024-09 -> 2026-10, 22,571 runners), so a score is approximately the historical
percentile of that runner (max CDF error 4 points) and is comparable across races
and field sizes. Reference points: p_win = 1/N -> 67.7; 2/N -> 83.8; 3/N -> 89.7;
0.5/N -> 46.0; 0.25/N -> 25.7; 0.1/N -> 9.5. Upcoming meetings without odds use the
fundamental fallback model whose probabilities are flatter, so scores are less extreme.
Scores are NOT rounded (exact monotonicity); round only for display.

Sub-scores (0-100, relative within the race)
--------------------------------------------
For each group in SUB_SCORE_FEATURES: every feature is turned into a within-race
percentile rank oriented so that higher = better (sign of the weight), missing values
count as neutral (0.5); the group composite is the |weight|-weighted mean, and the
sub-score is the within-race percentile of the composite: (rank - 1) / (N - 1) * 100
(average ranks for ties, 50 for a single runner). Weights are the within-race rank
correlations of each feature with ``place`` on 2024-09 -> 2026-10 (sign = direction).
``market`` is None when no runner of the race has WIN odds (past meetings use the
final WIN odds of the results page -- not known 2 hours before the race).

Reasons
-------
Each group produces at most one phrase with concrete numbers whose polarity follows the
sign of (sub_score - 50); salience = |sub_score - 50| / 50 * REASON_WEIGHTS[group]
(+ fixed salience for debut / long rest / winless record). We keep the strongest
positive and the strongest negative factor, then add the next most salient up to 3
(4 when very salient). Debutants always get 「初出馬，無往績」. Never NaN / None text.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger("hkjc.scores")

# ------------------------------------------------------------------ calibration
SCORE_C = -0.57
SCORE_S = 0.77
CALIBRATION_SOURCE = "wf-combined-v1, 2024-09-08 → 2026-10-19, 22,571 匹次"


def score_from_pwin(p_win, field_size):
    """Composite score 0-100 from p_win and the number of runners (vectorised)."""
    p = np.clip(np.asarray(p_win, dtype=float), 1e-9, 1.0)
    n = np.asarray(field_size, dtype=float)
    x = np.log(p * n)
    return 100.0 / (1.0 + np.exp(-(x - SCORE_C) / SCORE_S))


def _ref(mult):
    return float(score_from_pwin(mult / 10.0, 10))


SCORE_EXPLANATION_ZH = (
    "評分（0–100）由模型勝出機率計出：先計「勝出機率 ÷ 平均馬勝出機率（1 ÷ 出賽匹數）」嘅倍數，"
    "取對數後用固定 S 形曲線映射到 0–100："
    f"評分 = 100 ÷ (1 + e^(−(ln(勝出機率 × 出賽匹數) − ({SCORE_C})) ÷ {SCORE_S}))。"
    f"參數由 2024-09 至 2026-10 walk-forward 回測約 22,600 匹次嘅分佈校準，評分大約等於該馬喺歷史所有出賽馬匹中嘅百分位，"
    "所以唔同場次、唔同出賽匹數之間可以直接比較。"
    f"參考：勝出機率等於平均馬 ≈ {_ref(1):.0f} 分，兩倍 ≈ {_ref(2):.0f} 分，三倍 ≈ {_ref(3):.0f} 分，"
    f"一半 ≈ {_ref(0.5):.0f} 分，十分一 ≈ {_ref(0.1):.0f} 分。同場內評分高低次序同模型勝出機率完全一致。"
    "子評分（往績、騎師、練馬師、騎練組合、檔位、路程場地、班次負磅、市場）係同場馬匹之間嘅相對百分位（0 = 同場最差，100 = 同場最好），"
    "只作解釋用途，綜合評分以模型機率為準；未有賠率時「市場」子評分留空。"
    "評分只供研究參考，屬模擬研究，並非投注建議；博彩要有節制。"
)

# ------------------------------------------------------------------ sub-scores
# group -> [(feature, weight)]; weight = within-race rank correlation with `place`
# (2024-09 -> 2026-10); negative weight = lower value is better.
SUB_SCORE_FEATURES: dict[str, list[tuple[str, float]]] = {
    "form": [("h_pct_avg3", -0.267), ("h_pct_last1", -0.245), ("h_pct_avg6", -0.258),
             ("h_lbw_avg3", -0.247), ("h_spd_avg3", 0.206), ("h_place_rate", 0.186),
             ("h_perf_avg3", 0.119)],
    "jockey": [("j_win365", 0.144), ("j_pl365", 0.148), ("j_pl60", 0.135),
               ("j_venue_pl", 0.143), ("j_surf_pl", 0.148), ("j_dist_pl", 0.137)],
    "trainer": [("t_win365", 0.060), ("t_pl365", 0.068), ("t_pl60", 0.073),
                ("t_venue_pl", 0.076), ("t_surf_pl", 0.070), ("t_dist_pl", 0.074)],
    "combo": [("jt_pl", 0.155), ("jt_win", 0.136)],
    "draw": [("draw", -0.088), ("draw_pl_resid", 0.072), ("draw_win_resid", 0.062),
             ("draw_c_pl_resid", 0.057)],
    "fit": [("h_surf_place", 0.191), ("h_venue_place", 0.184), ("h_dist_place", 0.165),
            ("h_vc_place", 0.095)],
    "class": [("h_dw_act", 0.088), ("wt_rel", 0.071), ("h_band_change", 0.047),
              ("h_class_change", -0.034)],
    "market": [("mkt_p", 1.0)],
}
SUB_SCORE_NAMES_ZH = {
    "form": "往績", "jockey": "騎師", "trainer": "練馬師", "combo": "騎練組合",
    "draw": "檔位", "fit": "路程場地", "class": "班次負磅", "market": "市場",
}
SUB_SCORE_KEYS = list(SUB_SCORE_FEATURES)

# salience weight of each group in the reasons = correlation of the sub-score with
# `place` on 2024-09 -> 2026-10 relative to form (form .274, jockey .149, trainer .082,
# combo .152, draw .077, fit .195, class .111, market .390); market damped from 1.4 to
# 0.5 so that the fundamental factors are explained first.
REASON_WEIGHTS = {"form": 1.0, "jockey": 0.55, "trainer": 0.3, "combo": 0.55, "draw": 0.3,
                  "fit": 0.7, "class": 0.4, "market": 0.5}
MIN_SALIENCE = 0.08
STRONG_SALIENCE = 0.4

VENUE_ZH = {"HV": "跑馬地", "ST": "沙田"}
CLASS_ZH = {-2.0: "一級賽", -1.0: "二級賽", 0.0: "三級賽", 1.0: "第一班", 2.0: "第二班",
            3.0: "第三班", 4.0: "第四班", 5.0: "第五班", 5.5: "新馬賽"}

# form statistics that may be quoted in the reasons: (feature, lower_is_better)
FORM_DISPLAY = [("h_pos_avg3", True), ("h_lbw_avg3", True), ("h_pos_last1", True),
                ("h_spd_avg3", False)]

OUT_COLS = ["race_id", "race_no", "horse_no", "score", "rank_in_race", "sub_scores", "reasons",
            "p_win", "p_place"]


# ------------------------------------------------------------------ helpers
def _num(v):
    """float or None (NaN / None / non-numeric -> None)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _raw_rate(shrunk, n, prior, k):
    """Undo features._shrink: (w + prior*k)/(n + k) -> w/n (None when n == 0)."""
    s, n = _num(shrunk), _num(n)
    if s is None or n is None or n <= 0:
        return None
    return float(min(max((s * (n + k) - prior * k) / n, 0.0), 1.0))


def _raw_count(shrunk, n, prior, k):
    s, n = _num(shrunk), _num(n)
    if s is None or n is None or n <= 0:
        return None
    return int(round(min(max(s * (n + k) - prior * k, 0.0), n)))


def _pct(r):
    return f"{100 * r:.0f}%"


def _odds_txt(o):
    return f"{o:.1f}" if o < 10 else f"{o:.0f}"


def _name(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _race_no(race_id):
    try:
        return int(str(race_id).rsplit("-", 1)[-1])
    except ValueError:
        return None


def _within_race_pct(values: pd.Series, race: pd.Series) -> pd.Series:
    """(rank-1)/(n-1)*100 within race (average ranks; single runner -> 50)."""
    r = values.groupby(race).rank(method="average")
    n = values.groupby(race).transform("count")
    out = (r - 1) / (n - 1).where(n > 1) * 100.0
    return out.where(n > 1, 50.0)


def compute_sub_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-score columns ``sub_<group>`` (0-100 floats, market NaN when no odds)."""
    race = df["race_id"]
    out = pd.DataFrame(index=df.index)
    for grp, feats in SUB_SCORE_FEATURES.items():
        num = pd.Series(0.0, index=df.index)
        den = 0.0
        any_feat = False
        for f, w in feats:
            if f not in df.columns:
                continue
            any_feat = True
            v = pd.to_numeric(df[f], errors="coerce").astype(float)
            if grp == "market":
                # missing odds inside a priced race -> outsider (like model._mkt_logp)
                v = v.fillna(v.groupby(race).transform("min") * 0.999)
            r = v.groupby(race).rank(pct=True, method="average", ascending=w > 0)
            n = v.groupby(race).transform("count")
            # pct rank in (0, 1] -> rescale to [0, 1]; missing / single value -> neutral
            r = ((r * n - 1) / (n - 1).where(n > 1)).fillna(0.5)
            num = num + abs(w) * r
            den += abs(w)
        if not any_feat or den == 0:
            out["sub_" + grp] = np.nan
            continue
        comp = num / den
        sub = _within_race_pct(comp, race)
        if grp == "market":
            src = feats[0][0]
            has = (pd.to_numeric(df[src], errors="coerce").notna().groupby(race).transform("any")
                   if src in df.columns else pd.Series(False, index=df.index))
            sub = sub.where(has)
        out["sub_" + grp] = sub
    return out


# ------------------------------------------------------------------ reasons
def _cand(group, text, sub, extra=None):
    """Candidate phrase; strength from the sub-score unless ``extra`` (fixed) given."""
    if extra is not None:
        s = extra
    else:
        s = 0.0 if sub is None else (sub - 50.0) / 50.0 * REASON_WEIGHTS.get(group, 0.3)
    return {"group": group, "text": text, "strength": float(s)}


def _form_reasons(r, sub):
    out = []
    h_n = _num(r.get("h_n")) or 0.0
    first = (_num(r.get("h_first_start")) == 1.0) or h_n == 0
    if first:
        out.append(dict(_cand("form", "初出馬，無往績", None, extra=-0.3), forced=True))
        return out
    n3 = int(min(h_n, 3))
    avg3, lbw3 = _num(r.get("h_pos_avg3")), _num(r.get("h_lbw_avg3"))
    last, lbw = _num(r.get("h_pos_last1")), _num(r.get("h_lbw_last1"))
    good = sub is None or sub >= 50
    opts = []        # (within-race goodness 0..1, text)
    g = _num(r.get("_fg_h_pos_avg3"))
    if n3 >= 2 and avg3 is not None and g is not None:
        opts.append((g, f"近{n3}仗平均第{avg3:.1f}名"))
    g = _num(r.get("_fg_h_lbw_avg3"))
    if n3 >= 2 and lbw3 is not None and g is not None and lbw3 >= 0.1:
        opts.append((g, f"近{n3}仗平均{'只' if good else ''}輸{lbw3:.1f}個馬位"))
    g = _num(r.get("_fg_h_pos_last1"))
    if last is not None and g is not None:
        if last == 1:
            t = "上仗勝出"
        else:
            t = f"上仗第{last:.0f}名"
            if lbw is not None and lbw >= 0.1:
                t += f"，輸{lbw:.1f}個馬位"
        opts.append((g, t))
    g, rk = _num(r.get("_fg_h_spd_avg3")), _num(r.get("_frk_h_spd_avg3"))
    if good and g is not None and rk is not None and rk <= 3:
        opts.append((g, f"近{n3}仗速度指數同場第{rk:.0f}" if n3 >= 2 else f"上仗速度指數同場第{rk:.0f}"))
    t = None
    if opts:
        t = (max(opts, key=lambda o: o[0]) if good else min(opts, key=lambda o: o[0]))[1]
    if t:
        if sub is not None and sub >= 99.9:
            t += "，同場往績最佳"
        elif sub is not None and sub <= 15:
            t += "，同場偏弱"
        out.append(_cand("form", t, sub))
    wins = _num(r.get("h_wins"))
    if wins is not None:
        if h_n >= 10 and wins == 0:
            out.append(_cand("record", f"近年出賽{h_n:.0f}次仍未勝出", None, extra=-0.2))
        elif wins >= 3:
            out.append(_cand("record", f"近年{h_n:.0f}戰{wins:.0f}勝", None,
                             extra=0.12 + 0.03 * min(wins, 6)))
    days = _num(r.get("h_days_since"))
    if days is not None and days >= 60:
        if days >= 90:
            out.append(_cand("rest", f"休息{days:.0f}日復出", None, extra=-0.25))
        else:
            out.append(_cand("rest", f"相隔{days:.0f}日再出", None, extra=-0.1))
    return out


def _person_reason(group, who, name, n, shrunk_win, sub):
    if sub is None:
        return []
    n = _num(n)
    if n is None:
        return []
    label = f"{who}{name}" if name else who
    if n < 10:
        return [_cand(group, f"{label}近一年只有{n:.0f}次出賽", sub)]
    w = _raw_rate(shrunk_win, n, 0.08, 30.0)
    if w is None:
        return []
    if sub >= 50:
        t = f"{label}近一年勝出率{_pct(w)}（{n:.0f}次）"
        if sub >= 99.9:
            t += "，同場最高"
    else:
        t = f"{label}近一年勝出率{_pct(w)}，同場偏低"
    return [_cand(group, t, sub)]


def _combo_reason(r, sub):
    if sub is None:
        return []
    n = _num(r.get("jt_n"))
    if n is None:
        return []
    j, t = _name(r.get("jockey_name")), _name(r.get("trainer_name"))
    pair = f"{j}/{t}組合" if j and t else "騎練組合"
    if n == 0:
        return [_cand("combo", f"{pair}首度合作", sub)]
    pl = _raw_rate(r.get("jt_pl"), n, 0.25, 10.0)
    if n < 5 or pl is None:
        c = _cand("combo", f"{pair}只合作過{n:.0f}次", sub)
        c["strength"] *= 0.5
        return [c]
    txt = f"{pair}近年上名率{_pct(pl)}（{n:.0f}次）"
    if sub < 50:
        txt += "，偏低"
    return [_cand("combo", txt, sub)]


def _draw_reason(r, sub):
    d = _num(r.get("draw"))
    if d is None or sub is None:
        return []
    rel = _num(r.get("draw_rel"))
    if d <= 3 or (rel is not None and rel <= 0.3):
        pos = "內檔"
    elif rel is not None and rel >= 0.7:
        pos = "偏外"
    else:
        pos = "中間檔"
    res = _num(r.get("draw_pl_resid"))
    t = f"檔位{d:.0f}{pos}"
    if res is not None and abs(res) >= 0.01 and (res > 0) == (sub >= 50):
        side = "較高" if res > 0 else "較低"
        t += f"，此場地路程同區檔位上名率{side}（{'+' if res > 0 else '−'}{abs(res) * 100:.0f}個百分點）"
    elif sub >= 80:
        t += "，同場檔位較有利"
    elif sub <= 20:
        t += "，同場檔位較蝕"
    return [_cand("draw", t, sub)]


def _fit_reason(r, sub):
    if sub is None:
        return []
    h_n = _num(r.get("h_n")) or 0.0
    if h_n == 0:
        return []
    venue = VENUE_ZH.get(str(r.get("venue")), "此場地")
    awt = _num(r.get("surface_awt")) == 1.0
    opts = []
    for nk, pk, label, first in (("h_dist_n", "h_dist_place", "此路程", "首次跑此路程"),
                                 ("h_venue_n", "h_venue_place", venue, f"首次在{venue}出賽"),
                                 ("h_surf_n", "h_surf_place", "全天候跑道" if awt else "草地",
                                  "首次跑全天候跑道" if awt else None)):
        n = _num(r.get(nk))
        if n is None:
            continue
        if n == 0:
            if first:
                opts.append((-0.5, first))
            continue
        m = _raw_count(r.get(pk), n, 0.25, 3.0)
        if m is None:
            continue
        rate = m / n
        txt = f"{label}{n:.0f}戰{m}上名" if m > 0 else f"{label}{n:.0f}戰未上名"
        # evidence-weighted deviation from the ~25% base place rate
        opts.append(((rate - 0.25) * min(n, 8) / 8.0, txt))
    if not opts:
        return []
    if sub >= 50:
        best = max(opts, key=lambda o: o[0])
        if best[0] <= 0:
            return []
    else:
        best = min(opts, key=lambda o: o[0])
        if best[0] >= 0:
            return []
    return [_cand("fit", best[1], sub)]


def _class_reason(r, sub):
    if sub is None:
        return []
    opts = []          # (own sign, magnitude, text)
    ch = _num(r.get("h_class_change"))
    now = _num(r.get("class_num"))
    if ch is not None and now is not None and ch != 0:
        prev = now - ch
        a, b = CLASS_ZH.get(prev), CLASS_ZH.get(now)
        if a and b:
            if ch < 0:
                opts.append((1, abs(ch), f"由{a}升上{b}作賽"))
            else:
                opts.append((-1, abs(ch), f"由{a}降落{b}作賽"))
    dw = _num(r.get("h_dw_act"))
    if dw is not None and abs(dw) >= 4:
        opts.append((1 if dw > 0 else -1, abs(dw) / 6.0,
                     f"負磅比上仗{'加' if dw > 0 else '減'}{abs(dw):.0f}磅"))
    w, rel = _num(r.get("act_weight")), _num(r.get("wt_rel"))
    if w is not None and rel is not None:
        if rel >= 4:
            opts.append((1, rel / 8.0, f"負磅{w:.0f}磅屬同場高磅（評分較高）"))
        elif rel <= -6:
            opts.append((-1, -rel / 8.0, f"負磅{w:.0f}磅，同場評分較低"))
    want = 1 if sub >= 50 else -1
    opts = [o for o in opts if o[0] == want]
    if not opts:
        return []
    return [_cand("class", max(opts, key=lambda o: o[1])[2], sub)]


def _market_reason(r, sub, n_runners):
    if sub is None:
        return []
    o, rk = _num(r.get("win_odds")), _num(r.get("mkt_rank"))
    if o is None or rk is None:
        return []
    if rk <= 3:
        t = f"獨贏賠率{_odds_txt(o)}，市場第{rk:.0f}熱門"
    elif rk > n_runners / 2:
        t = f"獨贏賠率{_odds_txt(o)}，市場冷門"
    else:
        t = f"獨贏賠率{_odds_txt(o)}，市場第{rk:.0f}熱門"
    return [_cand("market", t, sub)]


def _select(cands, fallback):
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: -abs(c["strength"])):
        if c["group"] in seen or not c["text"]:
            continue
        seen.add(c["group"])
        uniq.append(c)
    chosen = [c for c in uniq if c.get("forced")]
    rest = [c for c in uniq if not c.get("forced")]
    pos = [c for c in rest if c["strength"] >= MIN_SALIENCE]
    neg = [c for c in rest if c["strength"] <= -MIN_SALIENCE]
    if pos and not any(c["strength"] > 0 for c in chosen):
        chosen.append(pos[0])
    if neg and not any(c["strength"] < 0 for c in chosen):
        chosen.append(neg[0])
    for c in rest:
        if c in chosen:
            continue
        s = abs(c["strength"])
        if (len(chosen) < 3 and s >= MIN_SALIENCE) or (len(chosen) < 4 and s >= STRONG_SALIENCE):
            chosen.append(c)
    if len(chosen) < 2:
        chosen.append({"group": "model", "text": fallback, "strength": 0.0})
    for c in rest:
        if len(chosen) >= 2:
            break
        if c not in chosen:
            chosen.append(c)
    chosen = chosen[:4]
    # strengths first (strongest first), then weaknesses
    chosen.sort(key=lambda c: (c["strength"] < 0, -abs(c["strength"])))
    return [c["text"] for c in chosen]


def build_reasons(r: dict, subs: dict, p_win: float, rank: int, n_runners: int) -> list[str]:
    cands = []
    cands += _form_reasons(r, subs.get("form"))
    cands += _person_reason("jockey", "騎師", _name(r.get("jockey_name")), r.get("j_n365"),
                            r.get("j_win365"), subs.get("jockey"))
    cands += _person_reason("trainer", "練馬師", _name(r.get("trainer_name")), r.get("t_n365"),
                            r.get("t_win365"), subs.get("trainer"))
    cands += _combo_reason(r, subs.get("combo"))
    cands += _draw_reason(r, subs.get("draw"))
    cands += _fit_reason(r, subs.get("fit"))
    cands += _class_reason(r, subs.get("class"))
    cands += _market_reason(r, subs.get("market"), n_runners)
    pw = _num(p_win) or 0.0
    ptxt = "不足1%" if pw < 0.01 else _pct(pw)
    fallback = f"模型勝出機率{ptxt}，同場排第{rank}"
    out = _select(cands, fallback)
    return [t for t in out if t and "nan" not in t.lower() and "none" not in t.lower()]


# ------------------------------------------------------------------ main API
def score_horses(pred_df: pd.DataFrame, card_df: pd.DataFrame) -> pd.DataFrame:
    """Scores, within-race sub-scores and reasons for every row of ``pred_df``."""
    if pred_df is None or len(pred_df) == 0:
        return pd.DataFrame(columns=OUT_COLS)
    p = pred_df.copy()
    p["horse_no"] = pd.to_numeric(p["horse_no"]).astype(int)
    keep_p = [c for c in ("race_id", "horse_no", "p_win", "p_place") if c in p.columns]
    p = p[keep_p]
    c = card_df.copy() if card_df is not None else pd.DataFrame(columns=["race_id", "horse_no"])
    if len(c):
        c["horse_no"] = pd.to_numeric(c["horse_no"]).astype(int)
        c = c.drop(columns=[x for x in ("p_win", "p_place") if x in c.columns])
        c = c.drop_duplicates(["race_id", "horse_no"])
    d = p.merge(c, on=["race_id", "horse_no"], how="left").reset_index(drop=True)
    if "p_place" not in d.columns:
        d["p_place"] = np.nan
    d["p_win"] = pd.to_numeric(d["p_win"], errors="coerce").astype(float)
    n_run = d.groupby("race_id")["horse_no"].transform("size").astype(float)
    d["score"] = score_from_pwin(d["p_win"].fillna(1e-9), n_run)
    d["rank_in_race"] = (d.sort_values(["race_id", "horse_no"])
                         .groupby("race_id")["p_win"].rank(method="first", ascending=False)
                         .reindex(d.index).astype(int))
    rn = d["race_no"] if "race_no" in d.columns else pd.Series(np.nan, index=d.index)
    d["race_no"] = [int(x) if _num(x) is not None else _race_no(rid)
                    for x, rid in zip(rn, d["race_id"])]
    subs = compute_sub_scores(d)
    for f, asc in FORM_DISPLAY:
        if f in d.columns:
            v = pd.to_numeric(d[f], errors="coerce").astype(float)
            g = v.groupby(d["race_id"])
            n = g.transform("count")
            rk = g.rank(method="min", ascending=asc)           # 1 = best
            d["_frk_" + f] = rk
            d["_fg_" + f] = (1 - (rk - 1) / (n - 1).where(n > 1)).where(n > 1, 0.5)
    sub_dicts, reasons = [], []
    recs = d.to_dict("records")
    sub_recs = subs.to_dict("records")
    for r, s, nr in zip(recs, sub_recs, n_run):
        sd = {k: (None if _num(s.get("sub_" + k)) is None else int(round(s["sub_" + k])))
              for k in SUB_SCORE_KEYS}
        sub_dicts.append(sd)
        reasons.append(build_reasons(r, sd, r["p_win"], int(r["rank_in_race"]), int(nr)))
    d["sub_scores"] = sub_dicts
    d["reasons"] = reasons
    out = d[OUT_COLS].sort_values(["race_id", "rank_in_race"]).reset_index(drop=True)
    return out


def _is_past(conn, date, venue) -> bool:
    r = conn.execute("""SELECT COUNT(*) FROM runners u JOIN races r ON r.race_id = u.race_id
                        WHERE r.date=? AND r.venue=? AND u.finish_pos = 1""", (date, venue)).fetchone()
    return bool(r and r[0])


def _stored_predictions(conn, run_id, date, venue) -> pd.DataFrame:
    prefix = f"{date.replace('-', '')}-{venue}-%"
    return pd.read_sql_query("""SELECT race_id, horse_no, p_win, p_place FROM predictions
                                WHERE run_id=? AND race_id LIKE ? ORDER BY race_id, horse_no""",
                             conn, params=(run_id, prefix))


def _names(conn, date, venue, past) -> pd.DataFrame:
    if past:
        q = """SELECT u.race_id, u.horse_no, u.horse_name, u.jockey_name, u.trainer_name
               FROM runners u JOIN races r ON r.race_id = u.race_id WHERE r.date=? AND r.venue=?"""
        df = pd.read_sql_query(q, conn, params=(date, venue))
    else:
        q = """SELECT race_no, horse_no, horse_name, jockey_name, trainer_name
               FROM racecard WHERE date=? AND venue=?"""
        df = pd.read_sql_query(q, conn, params=(date, venue))
        df["race_id"] = [f"{date.replace('-', '')}-{venue}-{int(n):02d}" for n in df["race_no"]]
        df = df.drop(columns=["race_no"])
    df["horse_no"] = df["horse_no"].astype(int)
    return df.drop_duplicates(["race_id", "horse_no"])


def score_meeting(conn, date: str, venue: str, run_id: str | None = None,
                  pred_df: pd.DataFrame | None = None, odds=None, model_dir=None) -> pd.DataFrame:
    """Scores for one meeting (past: walk-forward predictions + leak-free training-frame
    features of that date; upcoming: predict_card + build_card_frame). ``run_id`` names
    the predictions run to use (stored in ``predictions``)."""
    from . import features as F
    past = _is_past(conn, date, venue)
    if past:
        frame = F.build_training_frame(conn, start=date, end=date)
        frame = frame[(frame["date"] == date) & (frame["venue"] == venue)]
        if pred_df is None:
            chain = [r for r in (run_id, "wf-combined-v1") if r]
            pre = conn.execute("SELECT run_id FROM runs WHERE kind='pre_bet' AND date=? AND venue=? "
                               "ORDER BY created_at DESC LIMIT 1", (date, venue)).fetchone()
            if pre:
                chain.append(pre[0])
            chain.append("wf-fundamental-v1")
            pred_df = pd.DataFrame()
            for rid in chain:
                pred_df = _stored_predictions(conn, rid, date, venue)
                if len(pred_df):
                    break
            if pred_df.empty:
                log.warning("score_meeting %s %s: no stored predictions", date, venue)
                return pd.DataFrame(columns=OUT_COLS)
        # only runners that actually started (training frame excludes scratched)
        keys = set(zip(frame["race_id"], frame["horse_no"].astype(int)))
        pred_df = pred_df[[(a, int(b)) in keys for a, b in zip(pred_df["race_id"], pred_df["horse_no"])]]
        card = frame
    else:
        card = F.build_card_frame(conn, date, venue, odds=odds)
        if pred_df is None and run_id:
            pred_df = _stored_predictions(conn, run_id, date, venue)
            if pred_df.empty:
                pred_df = None
        if pred_df is None:
            from . import model as M
            if card.empty:
                return pd.DataFrame(columns=OUT_COLS)
            # same as model.predict_card, reusing the card frame built above
            m = M.latest_model(conn, date, kind="combined", model_dir=model_dir)
            pred_df = M.predict(m, card)
        if len(card):
            keys = set(zip(card["race_id"], card["horse_no"].astype(int)))
            pred_df = pred_df[[(a, int(b)) in keys
                               for a, b in zip(pred_df["race_id"], pred_df["horse_no"])]]
    if card is not None and len(card):
        nm = _names(conn, date, venue, past)
        card = card.drop(columns=[c for c in ("jockey_name", "trainer_name") if c in card.columns])
        card = card.merge(nm[["race_id", "horse_no", "jockey_name", "trainer_name"]],
                          on=["race_id", "horse_no"], how="left")
    return score_horses(pred_df, card)


def save_scores(conn, run_id: str, df: pd.DataFrame) -> int:
    """Write ``horse_scores`` rows (replaces this run's rows for the races in ``df``)."""
    if df is None or len(df) == 0:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    races = sorted(set(df["race_id"]))
    conn.executemany("DELETE FROM horse_scores WHERE run_id=? AND race_id=?",
                     [(run_id, r) for r in races])
    rows = []
    for r in df.itertuples(index=False):
        subs = {k: (None if _num(v) is None else int(round(float(v))))
                for k, v in dict(r.sub_scores or {}).items()}
        rows.append((run_id, r.race_id, int(r.horse_no), float(r.score), int(r.rank_in_race),
                     json.dumps(subs, ensure_ascii=False),
                     json.dumps(list(r.reasons or []), ensure_ascii=False), now))
    conn.executemany("INSERT OR REPLACE INTO horse_scores (run_id, race_id, horse_no, score, rank_in_race,"
                     " sub_scores, reasons, created_at) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def load_scores(conn, run_id: str, race_ids=None) -> pd.DataFrame:
    """Read back ``horse_scores`` (sub_scores / reasons decoded)."""
    q = "SELECT * FROM horse_scores WHERE run_id=?"
    df = pd.read_sql_query(q, conn, params=(run_id,))
    if race_ids is not None:
        df = df[df["race_id"].isin(set(race_ids))]
    df["sub_scores"] = df["sub_scores"].map(lambda s: json.loads(s) if s else {})
    df["reasons"] = df["reasons"].map(lambda s: json.loads(s) if s else [])
    return df.reset_index(drop=True)
