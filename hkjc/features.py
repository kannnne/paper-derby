"""Leak-free feature engineering for HKJC runners.

Public API
----------
build_training_frame(conn, start=None, end=None) -> DataFrame
    One row per starter (scratched / TNP excluded) of every *completed* race whose
    date is in [start, end], with targets ``win``, ``place``, ``finish_pos`` and
    features computed only from races on dates strictly before the race date
    (the whole race day is excluded: at the decision time, 2 hours before the
    first race, no result of that card is known).
build_card_frame(conn, date, venue, odds=None) -> DataFrame
    Same columns for an upcoming meeting, from the ``racecard`` table plus live
    WIN odds (``hkjc.odds.latest_odds``) for the ``mkt_`` columns.

Both paths go through the same :func:`compute_features` on a "base" frame of
historical starters + target rows, so column semantics are identical.

Feature groups (see FUNDAMENTAL_FEATURES / MARKET_FEATURES)
    h_*    horse form / record (prior runs only)
    j_*    jockey, t_* trainer, jt_* jockey-trainer combo (prior days only)
    draw*  draw and historical draw bias for the venue/course/distance
    race context: field_size, distance, surface, venue, going, class, weights
    *_rk   within-race percentile ranks of a few key features
    mkt_*  market (win odds) -- kept separate so models can run with/without.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
WIN_PRIOR = 0.08
PLACE_PRIOR = 0.25

GOING_MAP = {
    "快地": -2.0, "好地至快地": -1.0, "好地": 0.0, "好地至黏地": 0.5, "黏地": 1.0,
    "黏地至軟地": 1.5, "軟地": 2.0, "軟地至大爛地": 2.5, "大爛地": 3.0, "爛地": 3.0,
    # all-weather track descriptions
    "濕快地": -0.5, "濕慢地": 1.0, "泥地": 1.0, "封地": 0.5,
}

CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}

FUNDAMENTAL_FEATURES = [
    # horse
    "h_n", "h_win_rate", "h_place_rate", "h_pos_last1", "h_pos_avg3", "h_pos_avg6",
    "h_pct_last1", "h_pct_avg3", "h_pct_avg6", "h_lbw_last1", "h_lbw_avg3", "h_days_since",
    "h_dw_act", "h_dw_body", "h_first_start", "h_first_up",
    "h_dist_n", "h_dist_win", "h_dist_place", "h_venue_n", "h_venue_win", "h_venue_place",
    "h_surf_n", "h_surf_win", "h_surf_place", "h_vc_n", "h_vc_win", "h_vc_place",
    "h_class_change", "h_band_change", "h_spd_last1", "h_spd_avg3", "h_spd_best3",
    "h_early_avg3", "h_n_365", "h_perf_last1", "h_perf_avg3", "h_perf_avg6", "h_perf_best3",
    "h_perf_vs_band", "h_wins",
    # jockey
    "j_n365", "j_win365", "j_pl365", "j_n60", "j_win60", "j_pl60",
    "j_venue_win", "j_venue_pl", "j_surf_win", "j_surf_pl", "j_dist_win", "j_dist_pl",
    # trainer
    "t_n365", "t_win365", "t_pl365", "t_n60", "t_win60", "t_pl60",
    "t_venue_win", "t_venue_pl", "t_surf_win", "t_surf_pl", "t_dist_win", "t_dist_pl",
    # combo
    "jt_n", "jt_win", "jt_pl",
    # draw
    "draw", "draw_rel", "draw_pl_resid", "draw_win_resid", "draw_c_pl_resid",
    # race context
    "field_size", "distance", "surface_awt", "venue_hv", "going_num", "class_num", "band_hi",
    "act_weight", "wt_rel", "body_weight",
    # within-race ranks
    "h_spd_avg3_rk", "h_pct_avg3_rk", "h_place_rate_rk", "j_win365_rk", "t_win365_rk",
    "h_lbw_avg3_rk", "h_perf_avg3_rk", "h_perf_vs_band_rk",
]
MARKET_FEATURES = ["mkt_log_odds", "mkt_p", "mkt_rank"]
ALL_FEATURES = FUNDAMENTAL_FEATURES + MARKET_FEATURES
ID_COLS = ["race_id", "date", "venue", "race_no", "horse_no", "horse_id", "horse_name",
           "jockey_id", "trainer_id"]
TARGET_COLS = ["win", "place", "finish_pos", "n_places"]

_BASE_COLS = [
    "race_id", "date", "venue", "race_no", "race_class", "rating_band", "distance", "surface",
    "course", "going", "winning_time_s", "horse_no", "horse_id", "horse_name", "jockey_id",
    "jockey_name", "trainer_id", "trainer_name", "act_weight", "body_weight", "draw",
    "finish_pos", "finish_status", "lbw_num", "running_pos", "finish_time_s", "win_odds",
]


# ------------------------------------------------------------------ parsing helpers
def class_num(race_class) -> float:
    """Numeric class: G1=-2, G2=-1, G3=0, 第一班=1 ... 第五班=5, 新馬賽 (griffin)=5.5,
    四歲 (4yo classic series)=1. Lower = stronger. NaN when unknown."""
    if race_class is None or (isinstance(race_class, float) and np.isnan(race_class)):
        return np.nan
    s = str(race_class)
    m = re.search(r"([一二三])級賽", s)
    if m:
        return {1: -2.0, 2: -1.0, 3: 0.0}[CN_NUM[m.group(1)]]
    m = re.search(r"第([一二三四五])班", s)
    if m:
        return float(CN_NUM[m.group(1)])
    if "新馬" in s:
        return 5.5
    if "四歲" in s:
        return 1.0
    if "國際" in s or "錦標" in s:
        return 0.0
    return np.nan


def band_hi(rating_band) -> float:
    """Upper bound of the rating band '60-40' -> 60 ; '85+' -> 85."""
    if rating_band is None or (isinstance(rating_band, float) and np.isnan(rating_band)):
        return np.nan
    m = re.match(r"\s*(\d+)", str(rating_band))
    return float(m.group(1)) if m else np.nan


CLASS_LEVEL = {-2.0: 120.0, -1.0: 112.0, 0.0: 105.0, 1.0: 100.0, 2.0: 90.0, 3.0: 70.0,
               4.0: 50.0, 5.0: 28.0, 5.5: 30.0}
LBW_POINTS = 1.5     # rating points per length beaten (rough HK handicapping scale)


def band_mid(rating_band, cls_num=np.nan) -> float:
    """Approximate rating level of a race: middle of the rating band ('60-40' -> 50,
    '40-0' -> 27.5 since class 5 horses are rarely below ~15), else a class default."""
    if isinstance(rating_band, str):
        m = re.match(r"\s*(\d+)\s*-\s*(\d+)", rating_band)
        if m:
            hi, lo = float(m.group(1)), float(m.group(2))
            lo = max(lo, hi - 25.0)
            return (hi + lo) / 2.0
        m = re.match(r"\s*(\d+)\s*\+", rating_band)
        if m:
            return float(m.group(1)) + 10.0
    return CLASS_LEVEL.get(cls_num, np.nan)


def going_num(going) -> float:
    if going is None or (isinstance(going, float) and np.isnan(going)):
        return np.nan
    s = str(going).strip()
    return GOING_MAP.get(s, np.nan)


def _first_running_pos(s) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    tok = str(s).split()
    try:
        return float(tok[0]) if tok else np.nan
    except ValueError:
        return np.nan


def dist_bucket(d):
    """Distances within +-100m share a bucket (1600 & 1650 together)."""
    return (np.round(np.asarray(d, dtype=float) / 200.0) * 200.0)


def _shrink(w, n, prior, k):
    return (w + prior * k) / (n + k)


# ------------------------------------------------------------------ loading
def _ro_query(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)


def load_history(conn, before: str | None = None, start: str | None = None) -> pd.DataFrame:
    """Starters of completed races (a race counts as completed when it has a winner)."""
    where, params = [], []
    if before:
        where.append("r.date < ?")
        params.append(before)
    if start:
        where.append("r.date >= ?")
        params.append(start)
    w = (" AND " + " AND ".join(where)) if where else ""
    sql = f"""
        SELECT r.race_id, r.date, r.venue, r.race_no, r.race_class, r.rating_band, r.distance,
               r.surface, r.course, r.going, r.winning_time_s,
               u.horse_no, u.horse_id, u.horse_name, u.jockey_id, u.jockey_name,
               u.trainer_id, u.trainer_name, u.act_weight, u.body_weight, u.draw,
               u.finish_pos, u.finish_status, u.lbw_num, u.running_pos, u.finish_time_s,
               u.win_odds
        FROM runners u JOIN races r ON r.race_id = u.race_id
        WHERE COALESCE(u.scratched, 0) = 0
          AND COALESCE(u.finish_status, '') NOT IN ('TNP', 'WV', 'WV-A', 'WX', 'WX-A', 'WXNR')
          AND r.race_id IN (SELECT race_id FROM runners WHERE finish_pos = 1)
          {w}
        ORDER BY r.date, r.race_no, u.horse_no"""
    df = _ro_query(conn, sql, params)
    df["_known"] = 1.0
    return df


def _table_cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _jockey_allowance_estimates(conn, before: str) -> dict:
    """Apprentice allowance per jockey estimated from past racecard-vs-result pairs
    (racecard weight - actual weight carried), most recent meeting first."""
    try:
        df = _ro_query(conn, """
            SELECT c.date, c.jockey_id, c.jockey_name, c.weight, u.act_weight
            FROM racecard c JOIN runners u
              ON u.race_id = printf('%s-%s-%02d', replace(c.date,'-',''), c.venue, c.race_no)
             AND u.horse_no = c.horse_no
            WHERE c.date < ? AND COALESCE(c.scratched,0)=0 AND u.act_weight IS NOT NULL
              AND c.weight IS NOT NULL""", (before,))
    except Exception:
        return {}
    if df.empty:
        return {}
    df["allow"] = (df["weight"] - df["act_weight"]).clip(lower=0, upper=10)
    df["key"] = df["jockey_id"].fillna(df["jockey_name"])
    last = df.sort_values("date").groupby("key").tail(20).groupby("key")["allow"].median()
    return last.to_dict()


def card_rows(conn, date: str, venue: str, odds=None) -> pd.DataFrame:
    """Racecard rows (non-scratched) shaped like :func:`load_history` output, with
    unknown outcomes. ``odds`` = {race_no: {'WIN': {'1': 3.4, ...}}} (as returned by
    ``hkjc.odds.latest_odds``); when None it is looked up in odds_snapshots."""
    cols = _table_cols(conn, "racecard")
    extra = ", c.jockey_allowance" if "jockey_allowance" in cols else ""
    df = _ro_query(conn, f"""
        SELECT c.date, c.venue, c.race_no, c.horse_no, c.horse_id, c.horse_name,
               c.jockey_id, c.jockey_name, c.trainer_id, c.trainer_name, c.weight,
               c.body_weight, c.draw {extra}
        FROM racecard c WHERE c.date=? AND c.venue=? AND COALESCE(c.scratched,0)=0
        ORDER BY c.race_no, c.horse_no""", (date, venue))
    if df.empty:
        return df
    df["race_id"] = [f"{d.replace('-', '')}-{v}-{int(n):02d}" for d, v, n in
                     zip(df["date"], df["venue"], df["race_no"])]
    races = _ro_query(conn, """SELECT race_id, race_class, rating_band, distance, surface, course,
                                      going FROM races WHERE date=? AND venue=?""", (date, venue))
    df = df.merge(races, on="race_id", how="left")
    if "jockey_allowance" in df.columns:
        allow = df["jockey_allowance"].fillna(0)
    else:
        est = _jockey_allowance_estimates(conn, date)
        allow = df["jockey_id"].fillna(df["jockey_name"]).map(est).fillna(0)
    df["act_weight"] = df["weight"] - allow
    for c in ("winning_time_s", "finish_pos", "lbw_num", "running_pos", "finish_time_s"):
        df[c] = np.nan
    df["finish_status"] = None
    if odds is None:
        try:
            from .odds import latest_odds
            odds = latest_odds(conn, date, venue)
        except Exception:
            odds = {}
    df["win_odds"] = [_lookup_odds(odds, rn, hn) for rn, hn in zip(df["race_no"], df["horse_no"])]
    # a horse scratched after the racecard was scraped shows 'SCR' (stored as NULL) in the
    # live WIN pool: drop it so it is neither predicted nor bet on
    scr = [_odds_scratched(odds, rn, hn) for rn, hn in zip(df["race_no"], df["horse_no"])]
    df = df[~np.asarray(scr, dtype=bool)].reset_index(drop=True)
    df["_known"] = 0.0
    return df[_BASE_COLS + ["_known"]]


def _odds_scratched(odds, race_no, horse_no) -> bool:
    """True when the live WIN pool lists the horse with no odds (SCR) while other
    horses of the race are priced."""
    if not odds:
        return False
    book = odds.get(int(race_no)) or odds.get(str(race_no)) or {}
    win = book.get("WIN") or {}
    key = str(int(horse_no))
    if key not in win or win[key] is not None:
        return False
    return any(v is not None for v in win.values())


def _lookup_odds(odds, race_no, horse_no):
    if not odds:
        return np.nan
    book = odds.get(int(race_no)) or odds.get(str(race_no)) or {}
    win = book.get("WIN") or {}
    v = win.get(str(int(horse_no)))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if v >= 1.0 else np.nan


# ------------------------------------------------------------------ aggregation helpers
def _prior_cum(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """Per-row sums of ``cols`` over rows with the same ``keys`` on dates strictly
    before the row's date."""
    k = keys + ["date"]
    daily = df.groupby(k, sort=True)[cols].sum()
    cum = daily.groupby(level=list(range(len(keys)))).cumsum() - daily
    out = df[k].join(cum, on=k)
    return out[cols].fillna(0.0)


def _prior_window(df: pd.DataFrame, keys: list[str], cols: list[str], days: int) -> pd.DataFrame:
    """Per-row sums of ``cols`` over the same ``keys`` on dates in [date-days, date)."""
    k = keys + ["date"]
    daily = df.groupby(k, sort=True)[cols].sum().reset_index()
    inc = daily.groupby(keys)[cols].cumsum()
    before = inc - daily[cols]                       # strictly before this day
    daily_c = daily[k].copy()
    for c in cols:
        daily_c["b_" + c] = before[c].values
        daily_c["i_" + c] = inc[c].values
    # cumulative (inclusive) at the last day strictly before (date - days)
    left = daily_c[k].copy()
    left["cut"] = left["date"] - pd.Timedelta(days=days)
    left["_ord"] = np.arange(len(left))
    right = daily_c[k + ["i_" + c for c in cols]].rename(columns={"date": "cut"})
    left = left.sort_values("cut")
    right = right.sort_values("cut")
    m = pd.merge_asof(left, right, on="cut", by=keys, allow_exact_matches=False,
                      direction="backward").sort_values("_ord")
    res = daily_c[k].copy()
    for c in cols:
        res[c] = daily_c["b_" + c].values - m["i_" + c].fillna(0.0).values
    out = df[k].merge(res, on=k, how="left")
    out.index = df.index
    return out[cols].fillna(0.0)


def _rate_feats(df, keys, prefix, window=None, k_win=20.0, k_pl=20.0, with_n=True):
    cols = ["_start", "_win", "_place"]
    agg = _prior_cum(df, keys, cols) if window is None else _prior_window(df, keys, cols, window)
    out = {}
    if with_n:
        out[prefix + "_n"] = agg["_start"]
    out[prefix + "_win"] = _shrink(agg["_win"], agg["_start"], WIN_PRIOR, k_win)
    out[prefix + "_pl"] = _shrink(agg["_place"], agg["_start"], PLACE_PRIOR, k_pl)
    return out


# ------------------------------------------------------------------ main computation
def _identity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Stable keys for jockey / trainer / horse: id when known, else id learned from
    the name elsewhere in the frame, else the name itself."""
    for who in ("jockey", "trainer"):
        ids, names = df[f"{who}_id"], df[f"{who}_name"]
        known = pd.DataFrame({"n": names, "i": ids}).dropna()
        n2i = known.drop_duplicates("n", keep="last").set_index("n")["i"]
        k = ids.copy()
        miss = k.isna()
        k[miss] = names[miss].map(n2i)
        miss = k.isna()
        k[miss] = "N:" + names[miss].fillna("?")
        df[f"_{who[0]}key"] = k.astype(str)
    hk = df["horse_id"].copy()
    miss = hk.isna()
    hk[miss] = "N:" + df.loc[miss, "horse_name"].fillna("?")
    df["_hkey"] = hk.astype(str)
    return df


def compute_features(base: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of ``base`` (historical starters with ``_known``=1 and
    possibly target rows with unknown outcomes, ``_known``=0)."""
    df = base.copy()
    df = df.reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("distance", "act_weight", "body_weight", "draw", "finish_pos", "lbw_num",
              "finish_time_s", "winning_time_s", "win_odds", "horse_no", "race_no"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _identity_keys(df)
    known = df["_known"].fillna(0).astype(float)

    # ---------------- race context
    df["field_size"] = df.groupby("race_id")["horse_no"].transform("size").astype(float)
    df["n_places"] = np.where(df["field_size"] >= 7, 3, 2)
    df["surface_awt"] = (df["surface"] == "AWT").astype(float)
    df["venue_hv"] = (df["venue"] == "HV").astype(float)
    df["going_num"] = df["going"].map(going_num).astype(float)
    df["class_num"] = df["race_class"].map(class_num).astype(float)
    df["band_hi"] = df["rating_band"].map(band_hi).astype(float)
    df["_band_mid"] = [band_mid(b, c) for b, c in zip(df["rating_band"], df["class_num"])]
    df["wt_rel"] = df["act_weight"] - df.groupby("race_id")["act_weight"].transform("mean")
    df["dist_b"] = dist_bucket(df["distance"])
    df["course_key"] = np.where(df["surface"] == "AWT", "AWT", df["course"].fillna("?"))
    df["draw_rel"] = df["draw"] / df["field_size"]
    df["draw_b"] = np.ceil(df["draw"] / 2.0)

    # ---------------- targets (NaN for unknown rows)
    fp = df["finish_pos"]
    df["win"] = np.where(known > 0, (fp == 1).astype(float), np.nan)
    df["place"] = np.where(known > 0, (fp <= df["n_places"]).astype(float), np.nan)
    df["_start"] = known
    df["_win"] = df["win"].fillna(0.0)
    df["_place"] = df["place"].fillna(0.0)
    exp_pl = df["n_places"] / df["field_size"]
    exp_w = 1.0 / df["field_size"]
    df["_pl_resid"] = np.where(known > 0, df["_place"] - exp_pl, 0.0)
    df["_w_resid"] = np.where(known > 0, df["_win"] - exp_w, 0.0)

    # finish pos with non-finishers as last; pct 0 = winner, 1 = last
    fpf = np.where(known > 0, fp.fillna(df["field_size"]), np.nan)
    df["_fp"] = fpf
    df["_pct"] = (df["_fp"] - 1) / (df["field_size"] - 1).clip(lower=1)
    df["_lbw"] = np.where(known > 0, df["lbw_num"].clip(upper=30), np.nan)
    df.loc[(known > 0) & (fp == 1), "_lbw"] = 0.0
    df["_early"] = df["running_pos"].map(_first_running_pos) / df["field_size"]
    # performance rating: race level minus points per length beaten
    df["_perf"] = df["_band_mid"] - LBW_POINTS * df["_lbw"].clip(upper=15)

    # ---------------- speed figure (par from strictly earlier days)
    par_keys = ["venue", "surface", "distance"]
    df["_wt_sum"] = np.where((known > 0) & (fp == 1) & df["winning_time_s"].notna(),
                             df["winning_time_s"], 0.0)
    df["_wt_cnt"] = np.where((known > 0) & (fp == 1) & df["winning_time_s"].notna(), 1.0, 0.0)
    df["distance_k"] = df["distance"].fillna(-1)
    df["surface_k"] = df["surface"].fillna("?")
    pk = ["venue", "surface_k", "distance_k"]
    par = _prior_cum(df, pk, ["_wt_sum", "_wt_cnt"])
    par_t = np.where(par["_wt_cnt"] >= 3, par["_wt_sum"] / par["_wt_cnt"].replace(0, np.nan), np.nan)
    df["_spd"] = (par_t - df["finish_time_s"]) / (df["distance"] / 1000.0)
    df["_spd"] = df["_spd"].clip(-15, 15)
    del par_keys

    # ---------------- horse sequence features
    df = df.sort_values(["_hkey", "date", "race_no"], kind="mergesort")
    g = df.groupby("_hkey", sort=False)
    prev = lambda c, n=1: g[c].shift(n)
    df["h_pos_last1"] = prev("_fp")
    df["h_pct_last1"] = prev("_pct")
    df["h_lbw_last1"] = prev("_lbw")
    df["h_spd_last1"] = prev("_spd")
    for c, name, w in [("_fp", "h_pos_avg3", 3), ("_fp", "h_pos_avg6", 6), ("_pct", "h_pct_avg3", 3),
                       ("_pct", "h_pct_avg6", 6), ("_lbw", "h_lbw_avg3", 3), ("_spd", "h_spd_avg3", 3),
                       ("_early", "h_early_avg3", 3), ("_perf", "h_perf_avg3", 3),
                       ("_perf", "h_perf_avg6", 6)]:
        s = g[c].shift(1)
        df[name] = s.groupby(df["_hkey"]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    s = g["_spd"].shift(1)
    df["h_spd_best3"] = s.groupby(df["_hkey"]).rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    s = g["_perf"].shift(1)
    df["h_perf_best3"] = s.groupby(df["_hkey"]).rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    df["h_perf_last1"] = prev("_perf")
    prev_date = prev("date")
    df["h_days_since"] = (df["date"] - prev_date).dt.days.astype(float)
    df["h_dw_act"] = df["act_weight"] - prev("act_weight")
    df["h_dw_body"] = df["body_weight"] - prev("body_weight")
    df["h_class_change"] = df["class_num"] - prev("class_num")
    df["h_band_change"] = df["band_hi"] - prev("band_hi")
    df = df.sort_index()

    df = df.copy()

    # ---------------- horse records (prior days)
    hc = _prior_cum(df, ["_hkey"], ["_start", "_win", "_place"])
    df["h_n"] = hc["_start"]
    df["h_win_rate"] = _shrink(hc["_win"], hc["_start"], WIN_PRIOR, 3.0)
    df["h_place_rate"] = _shrink(hc["_place"], hc["_start"], PLACE_PRIOR, 3.0)
    df["h_wins"] = hc["_win"]
    df["h_perf_vs_band"] = df["h_perf_avg3"] - df["_band_mid"]
    df["h_first_start"] = (df["h_n"] == 0).astype(float)
    df["h_first_up"] = ((df["h_n"] == 0) | (df["h_days_since"] >= 60)).astype(float)
    df["h_n_365"] = _prior_window(df, ["_hkey"], ["_start"], 365)["_start"]
    for cond, name in [(["dist_b"], "dist"), (["venue"], "venue"), (["surface_k"], "surf"),
                       (["venue", "course_key"], "vc")]:
        a = _prior_cum(df, ["_hkey"] + cond, ["_start", "_win", "_place"])
        df[f"h_{name}_n"] = a["_start"]
        df[f"h_{name}_win"] = _shrink(a["_win"], a["_start"], WIN_PRIOR, 3.0)
        df[f"h_{name}_place"] = _shrink(a["_place"], a["_start"], PLACE_PRIOR, 3.0)

    df = df.copy()

    # ---------------- jockey / trainer
    for who, key in (("j", "_jkey"), ("t", "_tkey")):
        a = _rate_feats(df, [key], f"{who}_", window=365, k_win=30, k_pl=30)
        df[f"{who}_n365"], df[f"{who}_win365"], df[f"{who}_pl365"] = a[f"{who}__n"], a[f"{who}__win"], a[f"{who}__pl"]
        a = _rate_feats(df, [key], f"{who}_", window=60, k_win=20, k_pl=20)
        df[f"{who}_n60"], df[f"{who}_win60"], df[f"{who}_pl60"] = a[f"{who}__n"], a[f"{who}__win"], a[f"{who}__pl"]
        for cond, name in [(["venue"], "venue"), (["surface_k"], "surf"), (["dist_b"], "dist")]:
            a = _rate_feats(df, [key] + cond, f"{who}_{name}", window=365, k_win=30, k_pl=30,
                            with_n=False)
            df[f"{who}_{name}_win"] = a[f"{who}_{name}_win"]
            df[f"{who}_{name}_pl"] = a[f"{who}_{name}_pl"]
    a = _rate_feats(df, ["_jkey", "_tkey"], "jt", window=None, k_win=10, k_pl=10)
    df["jt_n"], df["jt_win"], df["jt_pl"] = a["jt_n"], a["jt_win"], a["jt_pl"]

    df = df.copy()

    # ---------------- draw bias (prior days), residual vs. expectation
    df["draw_bk"] = df["draw_b"].fillna(-1)
    a = _prior_cum(df, ["venue", "surface_k", "distance_k", "draw_bk"], ["_start", "_pl_resid", "_w_resid"])
    df["draw_pl_resid"] = a["_pl_resid"] / (a["_start"] + 50.0)
    df["draw_win_resid"] = a["_w_resid"] / (a["_start"] + 50.0)
    a = _prior_cum(df, ["venue", "course_key", "distance_k", "draw_bk"], ["_start", "_pl_resid"])
    df["draw_c_pl_resid"] = a["_pl_resid"] / (a["_start"] + 50.0)

    # ---------------- within-race percentile ranks (pre-race info only)
    for c, asc in [("h_spd_avg3", True), ("h_pct_avg3", False), ("h_place_rate", True),
                   ("j_win365", True), ("t_win365", True), ("h_lbw_avg3", False),
                   ("h_perf_avg3", True), ("h_perf_vs_band", True)]:
        df[c + "_rk"] = df.groupby("race_id")[c].rank(pct=True, ascending=asc)

    df = df.copy()

    # ---------------- market
    odds = df["win_odds"].where(df["win_odds"] >= 1.0)   # HKJC minimum odds is 1.0
    inv = 1.0 / odds
    df["mkt_log_odds"] = np.log(odds)
    df["mkt_p"] = inv / inv.groupby(df["race_id"]).transform("sum")
    df["mkt_rank"] = odds.groupby(df["race_id"]).rank(method="min")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["finish_pos"] = df["finish_pos"]
    keep = ID_COLS + ["win_odds", "_known"] + TARGET_COLS + ALL_FEATURES
    out = df[keep].copy()
    out = out.sort_values(["date", "race_no", "horse_no"]).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ public API
def build_training_frame(conn, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Rows for completed races with start <= date <= end (all history before is used)."""
    before = None
    if end is not None:
        before = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base = load_history(conn, before=before)
    if base.empty:
        return pd.DataFrame(columns=ID_COLS + ["win_odds"] + TARGET_COLS + ALL_FEATURES)
    out = compute_features(base)
    if start is not None:
        out = out[out["date"] >= start]
    out = out.drop(columns=["_known"]).reset_index(drop=True)
    return out


def build_card_frame(conn, date: str, venue: str, odds=None) -> pd.DataFrame:
    """Feature rows for an upcoming meeting (racecard table), history = races before ``date``."""
    card = card_rows(conn, date, venue, odds=odds)
    if card.empty:
        return pd.DataFrame(columns=ID_COLS + ["win_odds"] + TARGET_COLS + ALL_FEATURES)
    hist = load_history(conn, before=date)
    base = pd.concat([hist, card], ignore_index=True)
    out = compute_features(base)
    out = out[(out["_known"] == 0) & (out["date"] == date) & (out["venue"] == venue)]
    out = out.drop(columns=["_known"]).reset_index(drop=True)
    for c in ("win", "place", "finish_pos"):
        out[c] = np.nan
    return out
