"""Model tests: probability sanity, place model, save/load, walk-forward, card prediction."""
import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from hkjc import features as F
from hkjc import model as M
from tests.test_features import _synth_racecard, make_subset_db


@pytest.fixture(scope="module")
def subset_db(tmp_path_factory):
    p = tmp_path_factory.mktemp("model") / "subset.db"
    return make_subset_db(p)


@pytest.fixture(scope="module")
def frame(subset_db):
    conn = sqlite3.connect(str(subset_db))
    return F.build_training_frame(conn)


@pytest.fixture(scope="module")
def split(frame):
    dates = np.sort(frame["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    return frame[frame["date"] < cut], frame[frame["date"] >= cut]


@pytest.fixture(scope="module")
def models(split):
    tr, _ = split
    return {k: M.train(tr, kind=k, burn_in_days=20) for k in ("fundamental", "combined")}


def _check_probs(p, te):
    assert len(p) == len(te)
    s = p.groupby("race_id")["p_win"].sum()
    assert np.allclose(s, 1.0, atol=1e-6)
    assert ((p["p_win"] > 0) & (p["p_win"] < 1)).all()
    assert ((p["p_place"] > 0) & (p["p_place"] < 1)).all()
    k = te.groupby("race_id")["n_places"].first()
    sp = p.groupby("race_id")["p_place"].sum()
    assert np.allclose(sp.loc[k.index], k, atol=1e-4)
    # place prob never below win prob (to rounding)
    assert (p["p_place"] >= p["p_win"] - 1e-6).all()


def test_probabilities(models, split):
    _, te = split
    for kind, m in models.items():
        p = M.predict(m, te)
        _check_probs(p, te)
        assert set(p.columns) >= {"race_id", "horse_no", "p_win", "p_place"}
        assert m.train_end < te["date"].min()
    # the fundamental model must not depend on market columns
    te2 = te.copy()
    te2[F.MARKET_FEATURES] = np.nan
    p1 = M.predict(models["fundamental"], te)
    p2 = M.predict(models["fundamental"], te2)
    assert np.allclose(p1["p_win"], p2["p_win"]) and np.allclose(p1["p_place"], p2["p_place"])
    # the combined model falls back gracefully when odds are missing
    p3 = M.predict(models["combined"], te2)
    _check_probs(p3, te2)
    assert not p3["used_market"].any()


def test_model_beats_uniform(models, split):
    _, te = split
    ev = M.evaluate(M.predict(models["fundamental"], te), te)
    assert ev["model"]["pseudo_r2"] > 0.0


def test_save_load_roundtrip(models, split, tmp_path):
    _, te = split
    m = models["combined"]
    path = M.save(m, tmp_path / "m.pkl")
    m2 = M.load(path)
    a, b = M.predict(m, te), M.predict(m2, te)
    assert np.allclose(a["p_win"], b["p_win"]) and np.allclose(a["p_place"], b["p_place"])
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["kind"] == "combined" and meta["train_end"] == m.train_end


def test_harville_uniform_and_ordering():
    codes = np.repeat(np.arange(2), [10, 6])
    p = np.r_[np.full(10, 0.1), np.full(6, 1 / 6)]
    k = np.r_[np.full(10, 3.0), np.full(6, 2.0)]
    h = M.harville_place(p, codes, 2, k)
    assert np.allclose(h[:10], 0.3) and np.allclose(h[10:], 2 / 6)
    p = np.array([0.5, 0.3, 0.15, 0.05, 0.0 + 1e-9])
    p = p / p.sum()
    codes = np.zeros(5, dtype=int)
    h = M.harville_place(p, codes, 1, np.full(5, 3.0))
    assert np.isclose(h.sum(), 3.0, atol=1e-6)
    assert np.all(np.diff(h) <= 1e-12)
    hl = M.harville_place(p, codes, 1, np.full(5, 3.0), 0.8, 0.7)
    assert hl[0] < h[0]          # discounting lowers the favourite's place chance


def test_clogit_recovers_coefficients():
    rng = np.random.default_rng(1)
    R, N = 3000, 10
    codes = np.repeat(np.arange(R), N)
    Z = rng.normal(size=(R * N, 2))
    beta = np.array([1.2, -0.5])
    p = M.race_softmax(Z @ beta, codes, R).reshape(R, N)
    win = np.array([rng.choice(N, p=row) for row in p])
    y = np.zeros(R * N)
    y[np.arange(R) * N + win] = 1
    b = M.fit_clogit(Z, codes, R, y, l2=0.0)
    assert np.allclose(b, beta, atol=0.12)


def test_normalise_to_k():
    codes = np.repeat(np.arange(3), [8, 12, 5])
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.9, size=len(codes))
    k = np.r_[np.full(8, 3.0), np.full(12, 3.0), np.full(5, 2.0)]
    q = M.normalise_to_k(p, codes, 3, k)
    assert np.allclose(np.bincount(codes, weights=q), [3, 3, 2])
    # order preserved within race
    for r in range(3):
        m = codes == r
        assert (np.argsort(p[m]) == np.argsort(q[m])).all()


def test_walkforward_writes_predictions(subset_db, tmp_path):
    p2 = tmp_path / "wf.db"
    shutil.copy(subset_db, p2)
    conn = sqlite3.connect(str(p2))
    dates = [r[0] for r in conn.execute("SELECT DISTINCT date FROM races ORDER BY date")]
    start = dates[int(len(dates) * 0.7)]
    end = dates[-1]
    rid = M.walkforward(conn, start, end, kind="fundamental", retrain="monthly", min_train_days=30,
                        run_id="wf-test")
    assert rid == "wf-test"
    pred = pd.read_sql_query("SELECT p.*, r.date FROM predictions p JOIN races r ON r.race_id=p.race_id "
                             "WHERE run_id='wf-test'", conn)
    assert len(pred) > 0
    assert pred["date"].min() >= start and pred["date"].max() <= end
    assert np.allclose(pred.groupby("race_id")["p_win"].sum(), 1.0, atol=1e-6)
    run = conn.execute("SELECT kind, notes FROM runs WHERE run_id='wf-test'").fetchone()
    assert run[0] == "walkforward"
    notes = json.loads(run[1])
    for per in notes["periods"]:
        assert per["train_end"] < per["start"]          # trained strictly before the period


def test_predict_card(subset_db, tmp_path):
    p2 = tmp_path / "card.db"
    shutil.copy(subset_db, p2)
    conn = sqlite3.connect(str(p2))
    dates = [r[0] for r in conn.execute("SELECT DISTINCT date FROM races ORDER BY date")]
    D = dates[-2]
    venue, odds = _synth_racecard(conn, D)
    mdir = tmp_path / "models"
    out = M.predict_card(conn, D, venue, kind="combined", odds=odds, model_dir=mdir)
    assert len(out) > 0 and out["used_market"].all()
    assert np.allclose(out.groupby("race_id")["p_win"].sum(), 1.0, atol=1e-6)
    m = M.load(mdir / "combined_latest.pkl")
    assert m.train_end < D
    # no odds -> combined model falls back to its no-market coefficients
    out2 = M.predict_card(conn, D, venue, kind="combined", odds={}, model_dir=mdir)
    assert not out2["used_market"].any()
    assert np.allclose(out2.groupby("race_id")["p_win"].sum(), 1.0, atol=1e-6)


def test_latest_model_for_past_date_never_clobbers_latest(subset_db, tmp_path, monkeypatch):
    """Re-running an old meeting retrains a model on less data: it is saved as
    <kind>_to<date>.pkl but <kind>_latest.pkl (newer data) is kept."""
    conn = sqlite3.connect(str(subset_db))
    dates = [r[0] for r in conn.execute("SELECT DISTINCT date FROM races ORDER BY date")]
    mdir = tmp_path / "models"
    newest = M.latest_model(conn, "2099-01-01", kind="fundamental", model_dir=mdir)
    path = mdir / "fundamental_latest.pkl"
    before = path.read_bytes()
    past = dates[-3]
    m = M.latest_model(conn, past, kind="fundamental", model_dir=mdir)
    assert m.data_end < past <= newest.data_end
    assert path.read_bytes() == before                      # latest untouched
    assert (mdir / f"fundamental_to{m.train_end}.pkl").exists()
    # the newer model is still served for a current date (no retrain needed)
    again = M.latest_model(conn, "2099-01-01", kind="fundamental", model_dir=mdir)
    assert again.data_end == newest.data_end
    # $HKJC_MODEL_DIR is honoured when model_dir is not given
    monkeypatch.setenv("HKJC_MODEL_DIR", str(mdir))
    assert M.latest_model(conn, "2099-01-01", kind="fundamental").data_end == newest.data_end
