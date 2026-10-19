"""Win / place probability models + walk-forward backtest predictions.

Design (two stage model)
------------------------
Stage 1  Four base learners: HistGradientBoosting and L2 logistic regression, each
         on targets ``win`` and ``place``. Inputs: fundamental features (approach
         'hgb' -- not the default -- also feeds the ``mkt_`` features here).
Stage 2  Conditional logit (per-race softmax) fitted on *out-of-fold* stage-1
         scores (date-blocked folds):  u = sum_k b_k * logit(s_k) [+ b_m * log(mkt_p)
         for kind='combined', approach 'stack' = default];
         p_win = exp(u) / sum_race exp(u)  -> sums to 1 per race.
Place    Discounted Harville (Henery/Stern style exponents lambda2, lambda3 fitted on
         OOF) blended with the stage-1 place scores by a small logistic model, then
         normalised per race so that sum(p_place) = places paid (3, or 2 when the
         field has <= 6 starters).

kind='fundamental'  -> no market information at all.
kind='combined'     -> adds market (final WIN odds in training/backtest; live odds
                       at decision time). Races without any odds fall back to the
                       fundamental model (``used_market`` = False).

Public API
----------
train(df, kind='combined', approach=None) -> RaceModel
predict(model, df) -> DataFrame[race_id, horse_no, p_win, p_place, used_market]
save(model, path) / load(path)
walkforward(conn, start, end, kind='combined', retrain='monthly', min_train_days=365) -> run_id
latest_model(conn, date, kind) / predict_card(conn, date, venue, kind='combined') -> DataFrame
run_validation(conn, start='2024-09-01') -> dict   (writes data/models/validation.md)
    CLI: python -m hkjc.model --start 2024-09-01
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features as F
from .db import ROOT

log = logging.getLogger("hkjc.model")

MODEL_DIR = ROOT / "data" / "models"


def default_model_dir() -> Path:
    """Directory of saved models: $HKJC_MODEL_DIR if set (dry runs), else data/models."""
    env = os.environ.get("HKJC_MODEL_DIR")
    return Path(env).expanduser() if env else MODEL_DIR
MODEL_VERSION = "v1"
EPS = 1e-6

HGB_PARAMS = dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=7, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
N_OOF_FOLDS = 4
# chosen by walk-forward validation (see data/models/validation.md): for 'combined'
# the Benter stack ('stack': fundamental learners + log(mkt_p) in the stage-2
# conditional logit) beat 'hgb' (mkt_ features inside the stage-1 learners) on
# race log loss and place log loss over 2024-09 .. 2026-10.
DEFAULT_APPROACH = {"fundamental": "stack", "combined": "stack"}
BURN_IN_DAYS = 60          # first days of data have no history: excluded from training


# ======================================================================= helpers
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _codes(race_ids):
    codes, uniq = pd.factorize(np.asarray(race_ids), sort=False)
    return codes, len(uniq)


def race_softmax(u, codes, R):
    u = np.asarray(u, dtype=float)
    mx = np.full(R, -np.inf)
    np.maximum.at(mx, codes, u)
    e = np.exp(u - mx[codes])
    s = np.bincount(codes, weights=e, minlength=R)
    return e / s[codes]


def normalise_to_k(p, codes, R, k, iters=60):
    """Shift each race's logits so that sum(p) == k (k per row, constant within race)."""
    z = _logit(p)
    kk = np.bincount(codes, weights=np.asarray(k, dtype=float), minlength=R) / np.bincount(codes, minlength=R)
    c = np.zeros(R)
    for _ in range(iters):
        q = _sigmoid(z + c[codes])
        s = np.bincount(codes, weights=q, minlength=R)
        d = np.bincount(codes, weights=q * (1 - q), minlength=R)
        step = (kk - s) / np.maximum(d, 1e-9)
        c += np.clip(step, -5, 5)
        if np.max(np.abs(kk - s)) < 1e-10:
            break
    return np.clip(_sigmoid(z + c[codes]), EPS, 1 - EPS)


def _pad(values, codes, R):
    """Rows -> (R, N) matrix (zeros as padding) + positions."""
    pos = pd.Series(codes).groupby(codes).cumcount().values
    N = int(pos.max()) + 1 if len(pos) else 1
    M = np.zeros((R, N))
    M[codes, pos] = values
    return M, pos


def harville_place(p_win, codes, R, n_places, lam2=1.0, lam3=1.0):
    """P(top n_places) under discounted Harville: 2nd place drawn from p^lam2,
    3rd from p^lam3 (each renormalised over the remaining runners)."""
    P, pos = _pad(p_win, codes, R)
    np_r = np.zeros(R)
    np_r[codes] = n_places
    q = np.where(P > 0, P ** lam2, 0.0)
    q = q / q.sum(1, keepdims=True)
    r = np.where(P > 0, P ** lam3, 0.0)
    r = r / r.sum(1, keepdims=True)
    one_q = np.clip(1 - q, 1e-9, None)
    # second place: sum_j p_j q_i / (1 - q_j), j != i
    s = (P / one_q).sum(1, keepdims=True)
    p2 = q * (s - P / one_q)
    # third: sum_{j!=k, j,k != i} p_j q_k/(1-q_j) * r_i / (1 - r_j - r_k)
    A = P[:, :, None] * q[:, None, :] / one_q[:, :, None]       # (R, j, k)
    N = P.shape[1]
    eye = np.eye(N, dtype=bool)[None]
    A = np.where(eye, 0.0, A)
    den = np.clip(1 - r[:, :, None] - r[:, None, :], 1e-9, None)
    B = A / den
    tot = B.sum((1, 2))[:, None]
    p3 = r * (tot - B.sum(2) - B.sum(1))
    out = P + p2 + np.where(np_r[:, None] >= 3, p3, 0.0)
    return np.clip(out[codes, pos], EPS, 1 - EPS)


# ------------------------------------------------------------- conditional logit
def fit_clogit(Z, codes, R, y, l2=1e-4):
    """Maximise sum_races log softmax(Z beta)[winner]; y = winner indicator
    (dead heats share weight)."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    ys = np.bincount(codes, weights=y, minlength=R)
    ok = ys[codes] > 0
    Z, y, c0 = Z[ok], y[ok], codes[ok]
    c, R2 = _codes(c0)
    w = y / np.bincount(c, weights=y, minlength=R2)[c]

    def f(beta):
        u = Z @ beta
        p = race_softmax(u, c, R2)
        ll = np.sum(w * np.log(np.clip(p, 1e-300, None)))
        g = Z.T @ (w - p * np.bincount(c, weights=w, minlength=R2)[c])
        return -ll / R2 + l2 * beta @ beta, -g / R2 + 2 * l2 * beta

    beta0 = np.ones(Z.shape[1]) * 0.5
    res = minimize(f, beta0, jac=True, method="L-BFGS-B")
    return res.x


def fit_logistic(X, y, l2=1e-4):
    X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, dtype=float)

    def f(b):
        z = X @ b
        p = np.clip(_sigmoid(z), 1e-12, 1 - 1e-12)
        ll = np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        g = X.T @ (y - p) / len(y)
        return -ll + l2 * b @ b, -g + 2 * l2 * b

    res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return res.x


# ======================================================================= model
@dataclass
class RaceModel:
    kind: str
    approach: str
    features: list
    learners: dict = None             # stage-1 base learners (name -> fitted model)
    beta: np.ndarray = None           # stage-2 conditional logit coefficients
    beta_fund: np.ndarray = None      # fallback (no market) coefficients
    lam: tuple = (1.0, 1.0)
    place_coef: np.ndarray = None     # logistic blend [1, logit(harville), logit(place scores)...]
    place_method: str = "blend"
    train_start: str = None
    train_end: str = None             # last race date used for fitting
    data_end: str = None              # last date of the frame passed to train()
    n_train: int = 0
    version: str = MODEL_VERSION
    created_at: str = None
    info: dict = field(default_factory=dict)
    fallback: object = None           # fundamental RaceModel for races without odds (approach 'hgb')

    @property
    def model_version(self):
        return f"{self.kind}-{self.approach}-{self.version}-to{self.train_end}"


def _stage1_features(kind, approach):
    if kind == "combined" and approach == "hgb":
        return F.FUNDAMENTAL_FEATURES + F.MARKET_FEATURES
    return list(F.FUNDAMENTAL_FEATURES)


class _LR:
    """Imputer + scaler + L2 logistic regression (robust linear base learner)."""

    def __init__(self, C=0.1):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                keep_empty_features=True),
                                  StandardScaler(), LogisticRegression(C=C, max_iter=3000))

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)


# base learners of stage 1: (name, learner, target)
BASE_LEARNERS = [("hgb_win", "hgb", "win"), ("hgb_place", "hgb", "place"),
                 ("lr_win", "lr", "win"), ("lr_place", "lr", "place")]


def _fit_base(kind, X, y):
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    else:
        clf = _LR()
    clf.fit(X, y)
    return clf


def _fit_hgb(X, y):
    return _fit_base("hgb", X, y)


_STAGE1_CACHE: dict = {}


def _stage1(df, feats):
    """Full-data fits of every base learner + out-of-fold scores (date-blocked folds).
    Returns (dict name->fitted learner, (n, n_learners) OOF probability matrix)."""
    key = (tuple(feats), df["date"].min(), df["date"].max(), len(df),
           float(df["win"].sum()), float(df["place"].sum()))
    if key in _STAGE1_CACHE:
        return _STAGE1_CACHE[key]
    X = df[feats].to_numpy(dtype=float)
    ys = {"win": df["win"].to_numpy(dtype=float), "place": df["place"].to_numpy(dtype=float)}
    dates = np.sort(df["date"].unique())
    blocks = np.array_split(dates, N_OOF_FOLDS)
    oof = np.zeros((len(df), len(BASE_LEARNERS)))
    dcol = df["date"].to_numpy()
    for b in blocks:
        te = np.isin(dcol, b)
        for j, (_, kind, tgt) in enumerate(BASE_LEARNERS):
            clf = _fit_base(kind, X[~te], ys[tgt][~te])
            oof[te, j] = clf.predict_proba(X[te])[:, 1]
    learners = {name: _fit_base(kind, X, ys[tgt]) for name, kind, tgt in BASE_LEARNERS}
    out = (learners, oof)
    if len(_STAGE1_CACHE) >= 64:
        _STAGE1_CACHE.clear()
    _STAGE1_CACHE[key] = out
    return out


def _stage1_scores(model, X):
    return np.column_stack([model.learners[name].predict_proba(X)[:, 1]
                            for name, _, _ in BASE_LEARNERS])


def _mkt_logp(df):
    """log(normalised market prob); missing odds filled with the race minimum,
    races without any odds -> NaN."""
    mp = df["mkt_p"].astype(float)
    fill = mp.groupby(df["race_id"]).transform("min")
    mp = mp.fillna(fill)
    return np.log(np.clip(mp.to_numpy(), 1e-4, 1.0))


PLACE_IDX = [j for j, (_, _, t) in enumerate(BASE_LEARNERS) if t == "place"]


def _Z(kind, approach, S, df):
    """Stage-2 design matrix: logits of every base learner (+ log market prob)."""
    cols = [_logit(S[:, j]) for j in range(S.shape[1])]
    if kind == "combined" and approach == "stack":
        cols.append(_mkt_logp(df))
    return np.column_stack(cols)


def _place_from_win(p_win, S, codes, R, n_places, lam, coef):
    h = harville_place(p_win, codes, R, n_places, *lam)
    if coef is None:
        p = h
    else:
        X = np.column_stack([np.ones(len(h)), _logit(h)] + [_logit(S[:, j]) for j in PLACE_IDX])
        p = _sigmoid(X @ coef)
    return normalise_to_k(p, codes, R, n_places)


def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(df: pd.DataFrame, kind: str = "combined", approach: str | None = None,
          burn_in_days: int = BURN_IN_DAYS) -> RaceModel:
    """Train on a training frame (rows with known targets).

    approach (only matters for kind='combined'; default DEFAULT_APPROACH[kind]):
      'stack': market enters only at stage 2 as log(mkt_p) (Benter style);
      'hgb'  : mkt_ features are stage-1 inputs of every base learner; a fundamental
               model is attached as ``fallback`` for races without odds."""
    if kind not in ("fundamental", "combined"):
        raise ValueError(kind)
    approach = approach or DEFAULT_APPROACH[kind]
    d = df[df["win"].notna()].copy()
    data_end = d["date"].max() if len(d) else None
    if burn_in_days and len(d):
        first = pd.Timestamp(d["date"].min())
        d = d[pd.to_datetime(d["date"]) >= first + pd.Timedelta(days=burn_in_days)]
    d = d.sort_values(["date", "race_no", "horse_no"]).reset_index(drop=True)
    feats = _stage1_features(kind, approach)
    learners, oof = _stage1(d, feats)
    codes, R = _codes(d["race_id"])
    yw = d["win"].to_numpy(float)
    yp = d["place"].to_numpy(float)
    Z = _Z(kind, approach, oof, d)
    nb = oof.shape[1]
    if Z.shape[1] > nb:
        # market coefficient is fitted only on races where every runner has odds;
        # the stage-1 learners (fundamental features) use every race.
        has = d["mkt_p"].notna().groupby(d["race_id"]).transform("all").to_numpy()
        c2, R2 = _codes(d["race_id"][has])
        beta = fit_clogit(Z[has], c2, R2, yw[has])
        beta_fund = fit_clogit(Z[:, :nb], codes, R, yw)
        u = np.where(has, Z @ beta, Z[:, :nb] @ beta_fund)
    else:
        beta = beta_fund = fit_clogit(Z, codes, R, yw)
        u = Z @ beta
    p_win = race_softmax(u, codes, R)
    npl = d["n_places"].to_numpy(float)
    # place: fit Harville discount exponents on OOF p_win
    best = None
    for l2 in (0.7, 0.8, 0.9, 1.0):
        for l3 in (0.6, 0.7, 0.8, 0.9, 1.0):
            if l3 > l2:
                continue
            h = harville_place(p_win, codes, R, npl, l2, l3)
            ll = _logloss(yp, normalise_to_k(h, codes, R, npl))
            if best is None or ll < best[0]:
                best = (ll, (l2, l3))
    ll_h, lam = best
    h = harville_place(p_win, codes, R, npl, *lam)
    coef = fit_logistic(np.column_stack([_logit(h)] + [_logit(oof[:, j]) for j in PLACE_IDX]), yp)
    ll_b = _logloss(yp, _place_from_win(p_win, oof, codes, R, npl, lam, coef))
    method = "blend" if ll_b < ll_h else "harville"
    m = RaceModel(kind=kind, approach=approach, features=feats, learners=learners,
                  beta=beta, beta_fund=beta_fund, lam=lam,
                  place_coef=coef if method == "blend" else None, place_method=method,
                  train_start=d["date"].min(), train_end=d["date"].max(), data_end=data_end,
                  n_train=len(d),
                  created_at=datetime.now().isoformat(timespec="seconds"))
    m.info = {"oof_place_ll_harville": ll_h, "oof_place_ll_blend": ll_b,
              "oof_win_race_ll": float(-np.mean(np.log(p_win[yw > 0]))),
              "beta": [float(x) for x in beta], "beta_names": [n for n, _, _ in BASE_LEARNERS]
              + (["log_mkt_p"] if Z.shape[1] > nb else []), "lam": list(lam),
              "place_coef": [float(x) for x in coef], "n_races": int(R)}
    if kind == "combined" and approach == "hgb":
        m.fallback = train(df, kind="fundamental", burn_in_days=burn_in_days)
    return m


def predict(model: RaceModel, df: pd.DataFrame) -> pd.DataFrame:
    """Per-race normalised p_win (sums to 1) and p_place (sums to places paid).
    Extra column ``used_market`` tells whether market odds entered the prediction."""
    if df.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", "p_win", "p_place", "used_market"])
    d = df.reset_index(drop=True)
    X = d[model.features].to_numpy(dtype=float)
    S = _stage1_scores(model, X)
    codes, R = _codes(d["race_id"])
    Z = _Z(model.kind, model.approach, S, d)
    nb = S.shape[1]
    u = Z @ model.beta
    used_mkt = np.full(len(d), model.kind == "combined")
    if Z.shape[1] > nb:
        no_mkt = d["mkt_p"].isna().groupby(d["race_id"]).transform("all").to_numpy()
        if no_mkt.any():
            u = np.where(no_mkt, Z[:, :nb] @ model.beta_fund, u)
            used_mkt = ~no_mkt
    elif model.kind == "combined":
        used_mkt = d["mkt_p"].notna().to_numpy()
    p_win = race_softmax(u, codes, R)
    fb_rows = None
    if model.kind == "combined" and model.approach == "hgb" and model.fallback is not None:
        no_mkt = d["mkt_p"].isna().groupby(d["race_id"]).transform("all").to_numpy()
        if no_mkt.any():
            fb_rows = no_mkt
        used_mkt = ~no_mkt
    if "n_places" in d and d["n_places"].notna().all():
        npl = d["n_places"].to_numpy(float)
    else:
        npl = np.where(d.groupby("race_id")["horse_no"].transform("size") >= 7, 3.0, 2.0)
    coef = model.place_coef if model.place_method == "blend" else None
    p_place = _place_from_win(p_win, S, codes, R, npl, model.lam, coef)
    if fb_rows is not None:
        fb = predict(model.fallback, d[fb_rows])
        p_win[fb_rows] = fb["p_win"].to_numpy()
        p_place[fb_rows] = fb["p_place"].to_numpy()
    p_win = np.clip(p_win, EPS, 1 - EPS)
    p_win = p_win / np.bincount(codes, weights=p_win, minlength=R)[codes]
    return pd.DataFrame({"race_id": d["race_id"].to_numpy(), "horse_no": d["horse_no"].astype(int).to_numpy(),
                         "p_win": p_win, "p_place": p_place, "used_market": used_mkt})


def save(model: RaceModel, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    meta = {k: getattr(model, k, None) for k in ("kind", "approach", "train_start", "train_end", "data_end",
                                            "n_train", "version", "created_at", "place_method")}
    meta["info"] = model.info
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return path


class _Unpickler(pickle.Unpickler):
    """Maps classes pickled from ``python -m hkjc.model`` (module '__main__')."""

    def find_class(self, module, name):
        if module == "__main__" and name in ("RaceModel", "_LR"):
            module = __name__ if __name__ != "__main__" else "hkjc.model"
        return super().find_class(module, name)


def load(path) -> RaceModel:
    with open(path, "rb") as fh:
        return _Unpickler(fh).load()


# ======================================================================= walk-forward
def _period_starts(start, end, retrain):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if retrain == "monthly":
        first = s.to_period("M").to_timestamp()
        starts = list(pd.date_range(first, e, freq="MS"))
    elif retrain == "quarterly":
        first = s.to_period("Q").to_timestamp()
        starts = list(pd.date_range(first, e, freq="QS"))
    elif isinstance(retrain, int):
        starts = list(pd.date_range(s, e, freq=f"{retrain}D"))
    else:
        raise ValueError(retrain)
    starts = [max(x, s) for x in starts]
    return [(a, b) for a, b in zip(starts, starts[1:] + [e + pd.Timedelta(days=1)])]


def walkforward(conn, start: str, end: str, kind: str = "combined", retrain="monthly",
                min_train_days: int = 365, run_id: str | None = None, approach: str | None = None,
                df: pd.DataFrame | None = None, write: bool = True) -> str:
    """Chronological retrain/predict loop; writes ``predictions`` + a ``runs`` row."""
    t0 = time.time()
    run_id = run_id or f"wf-{kind}-{MODEL_VERSION}"
    if df is None:
        df = F.build_training_frame(conn, end=end)
    first_date = pd.Timestamp(df["date"].min())
    preds, periods = [], []
    for ps, pe in _period_starts(start, end, retrain):
        ps_s, pe_s = ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d")
        test = df[(df["date"] >= ps_s) & (df["date"] < pe_s) & (df["date"] <= end)]
        if test.empty:
            continue
        if (ps - first_date).days < min_train_days:
            log.info("skip %s: only %d days of history", ps_s, (ps - first_date).days)
            continue
        tr = df[df["date"] < ps_s]
        m = train(tr, kind=kind, approach=approach)
        p = predict(m, test)
        p["model_version"] = m.model_version
        preds.append(p)
        periods.append({"start": ps_s, "end": pe_s, "train_end": m.train_end, "n_train": m.n_train,
                        "n_pred": len(p), "beta": m.info["beta"], "lam": m.info["lam"],
                        "place_method": m.place_method,
                        "oof_place_ll_harville": m.info["oof_place_ll_harville"],
                        "oof_place_ll_blend": m.info["oof_place_ll_blend"]})
        log.info("%s %s: trained to %s (%d rows), predicted %d", run_id, ps_s, m.train_end,
                 m.n_train, len(p))
    allp = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    if write and len(allp):
        now = datetime.now().isoformat(timespec="seconds")
        rows = [(run_id, r.race_id, int(r.horse_no), float(r.p_win), float(r.p_place),
                 r.model_version, now) for r in allp.itertuples()]
        conn.execute("DELETE FROM predictions WHERE run_id=?", (run_id,))
        conn.executemany("INSERT OR REPLACE INTO predictions (run_id, race_id, horse_no, p_win, p_place,"
                         " model_version, created_at) VALUES (?,?,?,?,?,?,?)", rows)
        notes = {"kind": kind, "approach": approach or DEFAULT_APPROACH[kind], "start": start, "end": end, "retrain": retrain,
                 "min_train_days": min_train_days, "n_predictions": len(rows),
                 "runtime_s": round(time.time() - t0, 1), "periods": periods,
                 "market_note": "combined 模型回測用最終獨贏賠率（賽前 2 小時不可完全得知）"}
        conn.execute("INSERT OR REPLACE INTO runs (run_id, kind, date, venue, created_at, notes)"
                     " VALUES (?,?,?,?,?,?)", (run_id, "walkforward", end, None, now,
                                               json.dumps(notes, ensure_ascii=False, default=str)))
        conn.commit()
    walkforward.last_predictions = allp
    return run_id


# ======================================================================= live card
def _last_completed_before(conn, date):
    r = conn.execute("""SELECT MAX(r.date) FROM races r WHERE r.date < ? AND r.race_id IN
                        (SELECT race_id FROM runners WHERE finish_pos = 1)""", (date,)).fetchone()
    return r[0] if r else None


def latest_model(conn, date: str, kind: str = "combined", model_dir=None,
                 retrain: bool = True) -> RaceModel:
    """The saved model for ``kind`` trained on all completed races before ``date``
    (re-trained and saved when the saved one is stale).

    ``model_dir`` defaults to :func:`default_model_dir` ($HKJC_MODEL_DIR or data/models).
    Guard: a model re-trained for a PAST date (e.g. a re-run / dry run of an old
    meeting) is saved only as ``<kind>_to<train_end>.pkl``; ``<kind>_latest.pkl``
    is never replaced by a model built on less data than the saved one."""
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    path = model_dir / f"{kind}_latest.pkl"
    want_end = _last_completed_before(conn, date)
    saved_end = None
    if path.exists():
        m = load(path)
        end = getattr(m, "data_end", None) or m.train_end
        saved_end = end
        current = (m.version == MODEL_VERSION and m.approach == DEFAULT_APPROACH[kind]
                   and list(m.features) == _stage1_features(kind, m.approach))
        if end is not None and end < date and ((end == want_end and current) or not retrain):
            return m
    if not retrain and path.exists():
        return load(path)
    prev = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = F.build_training_frame(conn, end=prev)
    m = train(df, kind=kind)
    new_end = getattr(m, "data_end", None) or m.train_end
    if saved_end is None or new_end is None or str(new_end) >= str(saved_end):
        save(m, path)
    else:
        log.warning("latest_model(%s, %s): model for a past date (data to %s) - keeping the newer "
                    "%s (data to %s)", kind, date, new_end, path.name, saved_end)
    save(m, model_dir / f"{kind}_to{m.train_end}.pkl")
    return m


def predict_card(conn, date: str, venue: str, kind: str = "combined", odds=None,
                 model_dir=None) -> pd.DataFrame:
    """Predictions for an upcoming meeting from the racecard (+ live odds when present).
    For kind='combined', races without odds fall back to the fundamental model
    (column ``used_market`` says which)."""
    card = F.build_card_frame(conn, date, venue, odds=odds)
    if card.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", "p_win", "p_place", "used_market"])
    m = latest_model(conn, date, kind=kind, model_dir=model_dir)
    out = predict(m, card)
    out = out.merge(card[["race_id", "horse_no", "race_no", "horse_name", "win_odds"]],
                    on=["race_id", "horse_no"], how="left")
    out["model_version"] = m.model_version
    return out


# ======================================================================= validation
def _load_eval(conn, run_id):
    return pd.read_sql_query("""
        SELECT p.race_id, p.horse_no, p.p_win, p.p_place, r.date, r.venue, r.surface, r.race_no
        FROM predictions p JOIN races r ON r.race_id = p.race_id
        WHERE p.run_id = ? ORDER BY p.race_id, p.horse_no""", conn, params=(run_id,))


def _dividends(conn, pool):
    d = pd.read_sql_query("SELECT race_id, combination, dividend FROM dividends WHERE pool=?",
                          conn, params=(pool,))
    d = d[d["combination"].str.fullmatch(r"\d+")]
    d["horse_no"] = d["combination"].astype(int)
    return d.groupby(["race_id", "horse_no"])["dividend"].sum().reset_index()


def evaluate(pred: pd.DataFrame, frame: pd.DataFrame, win_div=None, pla_div=None) -> dict:
    """Metrics of a prediction frame (race_id, horse_no, p_win, p_place) against the
    training frame targets + market baseline."""
    d = pred.merge(frame[["race_id", "horse_no", "win", "place", "n_places", "mkt_p", "venue",
                          "date", "field_size"] + (["surface_awt"] if "surface_awt" in frame else [])],
                   on=["race_id", "horse_no"], how="inner")
    d = d[d["win"].notna()].copy()
    # market baseline (normalised 1/odds; missing -> tiny)
    d["mkt_p"] = d["mkt_p"].fillna(1e-3)
    d["mkt_p"] = d["mkt_p"] / d.groupby("race_id")["mkt_p"].transform("sum")
    codes, R = _codes(d["race_id"])
    d["mkt_place"] = normalise_to_k(harville_place(d["mkt_p"].to_numpy(), codes, R,
                                                   d["n_places"].to_numpy(float)),
                                    codes, R, d["n_places"].to_numpy(float))
    res = {}
    y, yp = d["win"].to_numpy(float), d["place"].to_numpy(float)

    def block(pw, pp, tag):
        r = {}
        r["win_logloss"] = _logloss(y, pw)
        r["win_brier"] = float(np.mean((pw - y) ** 2))
        wsum = np.bincount(codes, weights=y, minlength=R)
        lw = np.bincount(codes, weights=y * np.log(np.clip(pw, EPS, 1)), minlength=R) / np.maximum(wsum, 1)
        r["race_logloss"] = float(-np.mean(lw))
        r["place_logloss"] = _logloss(yp, pp)
        r["place_brier"] = float(np.mean((pp - yp) ** 2))
        # McFadden pseudo R2 vs uniform 1/N
        lu = np.log(1.0 / d["field_size"].to_numpy(float))
        l0 = np.bincount(codes, weights=y * lu, minlength=R) / np.maximum(wsum, 1)
        r["pseudo_r2"] = float(1 - np.sum(lw) / np.sum(l0))
        res[tag] = r
        res.setdefault("_race_ll", {})[tag] = -lw

    block(d["p_win"].to_numpy(), d["p_place"].to_numpy(), "model")
    block(d["mkt_p"].to_numpy(), d["mkt_place"].to_numpy(), "market")
    # paired per-race log-likelihood gain of the model over the market (+ standard error)
    g = res["_race_ll"]["market"] - res["_race_ll"]["model"]
    res["gain_vs_market"] = float(np.mean(g))
    res["gain_vs_market_se"] = float(np.std(g, ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan
    # top pick
    top = d.loc[d.groupby("race_id")["p_win"].idxmax()]
    fav = d.loc[d.groupby("race_id")["mkt_p"].idxmax()]
    res["n_races"] = int(R)
    res["n_runners"] = int(len(d))
    for tag, t in (("model_top", top), ("market_fav", fav)):
        r = {"win_rate": float(t["win"].mean()), "place_rate": float(t["place"].mean())}
        if win_div is not None:
            w = t.merge(win_div, on=["race_id", "horse_no"], how="left")["dividend"].fillna(0)
            r["win_roi"] = float(w.sum() / (10 * len(t)) - 1)
        if pla_div is not None:
            has_pla = t["race_id"].isin(pla_div["race_id"].unique())
            tt = t[has_pla]
            pl = tt.merge(pla_div, on=["race_id", "horse_no"], how="left")["dividend"].fillna(0)
            r["pla_roi"] = float(pl.sum() / (10 * len(tt)) - 1) if len(tt) else np.nan
            r["n_pla_bets"] = int(len(tt))
        res[tag] = r
    # calibration by decile of p_win
    d["dec"] = pd.qcut(d["p_win"].rank(method="first"), 10, labels=False)
    cal = d.groupby("dec").agg(n=("win", "size"), p_mean=("p_win", "mean"), win_rate=("win", "mean"),
                               pp_mean=("p_place", "mean"), place_rate=("place", "mean"))
    res["calibration"] = cal.reset_index()
    res["_d"] = d
    return res


def _fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def permutation_importance_table(df, kind="fundamental", cutoff=None, n_repeats=3, top=25):
    """Permutation importance of the stage-1 *place* HGB (more positives than win ->
    less noisy), trained before ``cutoff`` and scored (log loss) on rows from ``cutoff`` on."""
    from sklearn.inspection import permutation_importance
    dates = np.sort(df["date"].unique())
    cutoff = cutoff or dates[int(len(dates) * 0.75)]
    tr = df[(df["date"] < cutoff) & df["win"].notna()]
    te = df[(df["date"] >= cutoff) & df["win"].notna()]
    if kind == "combined":
        tr = tr[tr["mkt_p"].notna()]
        te = te[te["mkt_p"].notna()]
    feats = _stage1_features(kind, "hgb" if kind == "combined" else "stack")
    tr = tr[pd.to_datetime(tr["date"]) >= pd.Timestamp(tr["date"].min()) + pd.Timedelta(days=BURN_IN_DAYS)]
    clf = _fit_base("hgb", tr[feats].to_numpy(float), tr["place"].to_numpy(float))
    r = permutation_importance(clf, te[feats].to_numpy(float), te["place"].to_numpy(float),
                               scoring="neg_log_loss", n_repeats=n_repeats, random_state=0, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "importance": r.importances_mean, "std": r.importances_std})
    return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True), cutoff


def validation_report(conn, run_ids: dict, out_path=MODEL_DIR / "validation.md", frame=None,
                      importances=None, extra_md: str = "") -> str:
    """run_ids = {'fundamental': 'wf-fundamental-v1', 'combined': 'wf-combined-v1'}."""
    frame = frame if frame is not None else F.build_training_frame(conn)
    win_div, pla_div = _dividends(conn, "WIN"), _dividends(conn, "PLA")
    L = ["# 模型驗證報告（Walk-forward 回測）", ""]
    L.append(f"產生時間：{datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append("> **注意**：`combined`（綜合）模型同市場基準都用**最終獨贏賠率**，"
             "實戰喺第一場開跑前 2 小時只有即時賠率，所以 combined 嘅回測結果會比實戰樂觀；"
             "`fundamental`（基本面）模型完全唔用賠率。")
    L.append("")
    evals = {}
    for kind, rid in run_ids.items():
        pred = _load_eval(conn, rid)
        if pred.empty:
            continue
        ev = evaluate(pred[["race_id", "horse_no", "p_win", "p_place"]], frame, win_div, pla_div)
        evals[kind] = ev
        d = ev["_d"]
        L.append(f"## {kind}（run_id `{rid}`）")
        L.append("")
        L.append(f"回測期：{d['date'].min()} → {d['date'].max()}，{ev['n_races']} 場，{ev['n_runners']} 匹次。")
        L.append("")
        L.append("| 指標 | 模型 | 市場（最終賠率） |")
        L.append("|---|---|---|")
        names = [("race_logloss", "每場勝出者 log loss（越低越好）"), ("pseudo_r2", "McFadden 偽 R²（越高越好）"),
                 ("win_logloss", "獨贏 log loss（逐匹）"), ("win_brier", "獨贏 Brier"),
                 ("place_logloss", "位置 log loss"), ("place_brier", "位置 Brier")]
        for k, nm in names:
            L.append(f"| {nm} | {_fmt(ev['model'][k])} | {_fmt(ev['market'][k])} |")
        L.append("")
        L.append("| 揀法（每場 1 隻，平注 $10） | 勝出率 | 入三甲率 | 獨贏 ROI | 位置 ROI |")
        L.append("|---|---|---|---|---|")
        for tag, nm in (("model_top", "模型首選"), ("market_fav", "市場大熱")):
            r = ev[tag]
            L.append(f"| {nm} | {_pct(r['win_rate'])} | {_pct(r['place_rate'])} | {_pct(r.get('win_roi'))} | {_pct(r.get('pla_roi'))} |")
        L.append("")
        L.append("**校準（按 p_win 十分位）**")
        L.append("")
        L.append("| 十分位 | 匹次 | 平均 p_win | 實際勝出率 | 平均 p_place | 實際入三甲率 |")
        L.append("|---|---|---|---|---|---|")
        for r in ev["calibration"].itertuples():
            L.append(f"| {int(r.dec) + 1} | {r.n} | {_pct(r.p_mean)} | {_pct(r.win_rate)} | {_pct(r.pp_mean)} | {_pct(r.place_rate)} |")
        L.append("")
        # breakdowns
        L.append("**分場地 / 跑道**")
        L.append("")
        L.append("| 分組 | 場數 | 模型 race log loss | 市場 race log loss | 模型首選勝出率 | 首選獨贏 ROI | 首選位置 ROI |")
        L.append("|---|---|---|---|---|---|---|")
        surf = frame[["race_id", "surface_awt"]].drop_duplicates("race_id")
        p2 = pred.merge(surf, on="race_id", how="left")
        groups = [("沙田 ST", p2["venue"] == "ST"), ("跑馬地 HV", p2["venue"] == "HV"),
                  ("草地 TURF", p2["surface_awt"] == 0), ("全天候 AWT", p2["surface_awt"] == 1)]
        for nm, mask in groups:
            sub = p2[mask]
            if sub.empty:
                continue
            e2 = evaluate(sub[["race_id", "horse_no", "p_win", "p_place"]], frame, win_div, pla_div)
            L.append(f"| {nm} | {e2['n_races']} | {_fmt(e2['model']['race_logloss'])} | {_fmt(e2['market']['race_logloss'])} | "
                     f"{_pct(e2['model_top']['win_rate'])} | {_pct(e2['model_top']['win_roi'])} | {_pct(e2['model_top']['pla_roi'])} |")
        L.append("")
        # by season
        L.append("**分季**")
        L.append("")
        L.append("| 馬季 | 場數 | 模型 race log loss | 市場 race log loss | 首選勝出率 | 首選獨贏 ROI | 首選位置 ROI |")
        L.append("|---|---|---|---|---|---|---|")
        dts = pd.to_datetime(p2["date"])
        season = np.where(dts.dt.month >= 9, dts.dt.year, dts.dt.year - 1)
        for s in sorted(set(season)):
            sub = p2[season == s]
            e2 = evaluate(sub[["race_id", "horse_no", "p_win", "p_place"]], frame, win_div, pla_div)
            L.append(f"| {s}/{(s + 1) % 100:02d} | {e2['n_races']} | {_fmt(e2['model']['race_logloss'])} | {_fmt(e2['market']['race_logloss'])} | "
                     f"{_pct(e2['model_top']['win_rate'])} | {_pct(e2['model_top']['win_roi'])} | {_pct(e2['model_top']['pla_roi'])} |")
        L.append("")
    if evals:
        L.append("## 結論")
        L.append("")
        for kind, ev in evals.items():
            mo, mk = ev["model"], ev["market"]
            diff = mk["race_logloss"] - mo["race_logloss"]
            verdict = "勝過" if diff > 0 else "輸畀"
            se = ev["gain_vs_market_se"]
            t = ev["gain_vs_market"] / se if se and se > 0 else np.nan
            sig = "統計上顯著" if abs(t) >= 2 else "統計上唔顯著"
            L.append(f"- **{kind}**：每場 log loss {mo['race_logloss']:.4f} vs 市場 {mk['race_logloss']:.4f}"
                     f"（{verdict}市場 {abs(diff):.4f}，標準誤 {se:.4f}，t={t:.2f}，{sig}）；"
                     f"偽 R² {mo['pseudo_r2']:.3f} vs {mk['pseudo_r2']:.3f}；"
                     f"位置 log loss {mo['place_logloss']:.4f} vs {mk['place_logloss']:.4f}；"
                     f"首選獨贏 ROI {_pct(ev['model_top'].get('win_roi'))}、位置 ROI {_pct(ev['model_top'].get('pla_roi'))}。")
        L.append("- 基本面模型有獨立資訊（明顯好過隨機），但準確度遠低於市場；綜合模型對市場嘅改善好細、"
                 "未達統計顯著（主要係修正大熱 / 冷門偏差），而且回測用咗最終賠率，實戰優勢會更細甚至冇。"
                 "平注首選全部都係負回報 —— 單靠「買模型首選」唔會贏錢。")
        L.append("")
    if importances:
        for kind, (imp, cutoff) in importances.items():
            L.append(f"## 特徵重要性（permutation importance，{kind} 第一層位置 HGB，"
                     f"用 {cutoff} 之前訓練、之後評分，log loss 增幅）")
            L.append("")
            L.append("| # | 特徵 | 重要性 | 標準差 |")
            L.append("|---|---|---|---|")
            for i, r in enumerate(imp.itertuples(), 1):
                L.append(f"| {i} | `{r.feature}` | {r.importance:.5f} | {r.std:.5f} |")
            L.append("")
    if extra_md:
        L.append(extra_md)
    md = "\n".join(L)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    validation_report.last_evals = evals
    return md


# ======================================================================= driver
def run_validation(conn, start: str = "2024-09-01", end: str | None = None,
                   out_path=MODEL_DIR / "validation.md", compare: bool = True,
                   save_final: bool = True, min_train_days: int = 330) -> dict:
    """Walk-forward both kinds (written to ``predictions``), optional comparison runs
    (not written), permutation importances, final models, Chinese report."""
    t0 = time.time()
    if end is None:
        end = conn.execute("SELECT MAX(date) FROM races WHERE race_id IN "
                           "(SELECT race_id FROM runners WHERE finish_pos=1)").fetchone()[0]
    frame = F.build_training_frame(conn, end=end)
    t_feat = time.time() - t0
    timings = {"features_s": round(t_feat, 1)}
    run_ids = {}
    for kind in ("fundamental", "combined"):
        t = time.time()
        run_ids[kind] = walkforward(conn, start, end, kind=kind, df=frame,
                                    min_train_days=min_train_days)
        timings[f"wf_{kind}_s"] = round(time.time() - t, 1)
    extra = []
    win_div, pla_div = _dividends(conn, "WIN"), _dividends(conn, "PLA")
    if compare:
        rows = []
        alt = "hgb" if DEFAULT_APPROACH["combined"] == "stack" else "stack"
        label = {"stack": "combined（stack，市場只入第二層 log(mkt_p)）",
                 "hgb": "combined（HGB，市場特徵入第一層）"}
        variants = [("fundamental（無市場資料）", None, run_ids["fundamental"]),
                    (label[DEFAULT_APPROACH["combined"]] + " ← 採用", None, run_ids["combined"]),
                    (label[alt], ("combined", alt), None)]
        for name, spec, rid in variants:
            if rid:
                pr = _load_eval(conn, rid)[["race_id", "horse_no", "p_win", "p_place"]]
            else:
                t = time.time()
                walkforward(conn, start, end, kind=spec[0], approach=spec[1], df=frame, write=False,
                            min_train_days=min_train_days)
                timings[f"wf_combined_{alt}_s"] = round(time.time() - t, 1)
                pr = walkforward.last_predictions
            ev = evaluate(pr, frame, win_div, pla_div)
            rows.append((name, ev))
        L = ["## 模型方案比較（同一 walk-forward 回測期）", "",
             "| 方案 | race log loss | 偽 R² | 位置 log loss | 首選勝出率 | 首選獨贏 ROI | 首選位置 ROI |",
             "|---|---|---|---|---|---|---|"]
        for name, ev in rows:
            r, t = ev["model"], ev["model_top"]
            L.append(f"| {name} | {r['race_logloss']:.4f} | {r['pseudo_r2']:.4f} | {r['place_logloss']:.4f} | "
                     f"{_pct(t['win_rate'])} | {_pct(t['win_roi'])} | {_pct(t['pla_roi'])} |")
        r, t = rows[0][1]["market"], rows[0][1]["market_fav"]
        L.append(f"| 市場（最終賠率正規化；位置用 Harville） | {r['race_logloss']:.4f} | {r['pseudo_r2']:.4f} | "
                 f"{r['place_logloss']:.4f} | {_pct(t['win_rate'])} | {_pct(t['win_roi'])} | {_pct(t['pla_roi'])} |")
        L.append("")
        # place method evidence from the walk-forward periods
        notes = json.loads(conn.execute("SELECT notes FROM runs WHERE run_id=?",
                                        (run_ids["combined"],)).fetchone()[0])
        llh = np.mean([p["oof_place_ll_harville"] for p in notes["periods"]])
        llb = np.mean([p["oof_place_ll_blend"] for p in notes["periods"]])
        lam = notes["periods"][-1]["lam"]
        L.append(f"位置機率方法（combined，各期訓練集 OOF 平均位置 log loss）：折扣 Harville {llh:.4f} vs "
                 f"Harville＋第一層位置分數混合 {llb:.4f}；最近一期 Harville 折扣指數 λ2={lam[0]}, λ3={lam[1]}。")
        L.append("")
        L.append(SELECTION_NOTE)
        L.append("")
        extra.append("\n".join(L))
        validation_report.comparison = rows
    imps = {}
    cutoff = (pd.Timestamp(end) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    t = time.time()
    imps["fundamental"] = permutation_importance_table(frame, "fundamental", cutoff=cutoff)
    imps["combined（HGB 版，含 mkt_）"] = permutation_importance_table(frame, "combined", cutoff=cutoff)
    timings["importance_s"] = round(time.time() - t, 1)
    if save_final:
        nxt = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        t = time.time()
        for kind in ("fundamental", "combined"):
            latest_model(conn, nxt, kind=kind)
        timings["final_models_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t0, 1)
    extra.append("## 運行時間\n\n" + "、".join(f"{k}={v}s" for k, v in timings.items()) + "\n")
    extra.append(_caveats_md())
    md = validation_report(conn, run_ids, out_path=out_path, frame=frame, importances=imps,
                           extra_md="\n".join(extra))
    return {"run_ids": run_ids, "timings": timings, "report": md,
            "evals": validation_report.last_evals}


SELECTION_NOTE = ("揀選理由：以每場 log loss（同位置 log loss）為主要準則 —— 呢個係機率預測嘅正確評分；"
                  "ROI 樣本方差太大（首選獨贏 ROI 標準誤約 ±3–4%），唔用嚟揀模型。stack 版 log loss 最低，"
                  "而且冇賠率時可以乾淨咁退返純基本面係數。注意：方案係喺同一回測期揀，有少少選擇偏差。")


def _caveats_md() -> str:
    return """## 注意事項
- 回測入面嘅「市場」同 combined 模型都用**最終獨贏賠率**（開跑一刻先知），實戰只可以用開跑前 2 小時嘅即時賠率，賠率之後仲會郁，所以 combined 實戰表現大概會差過回測。
- 冇馬匹評分（rating）歷史：賽果頁冇評分，只用班次 / 評分範圍 / 負磅 / 頭馬距離去估馬匹實力；排位表有評分但歷史排位表冇儲存。
- 排位表「負磅」未扣見習騎師減磅；`racecard` 表冇儲存 `jockey_allowance`，所以 `build_card_frame` 用過往排位表對賽果估計減磅（冇資料時當 0）。建議喺 racecard 表加 `jockey_allowance` 欄。
- 首季（2023-09 起）係暖身期：特徵需要歷史，頭 60 日唔用嚟訓練；walk-forward 每月重新訓練，只用該月之前嘅賽事。
- ROI 用 dividends 表實際派彩（每 $10），平頭馬按派彩表計；位置 ROI 只計有位置派彩嘅場次。
- 樣本約兩季，ROI 誤差好大（首選獨贏每場派彩方差高），唔好單靠 ROI 判斷模型。
"""


if __name__ == "__main__":
    import argparse

    # run through the importable module so pickled models reference hkjc.model.*
    from hkjc import model as _model
    from hkjc.db import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = _model.run_validation(connect(), a.start, a.end, compare=not a.no_compare)
    print(res["report"])
