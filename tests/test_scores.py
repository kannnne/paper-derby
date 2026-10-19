"""Horse scores (hkjc.scores): monotonicity vs p_win, range, reasons, DB round-trip,
score_meeting on a copy of the real DB (upcoming 2026-10-21 HV, past 2026-10-14 HV)."""
import json
import re
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

from hkjc import db
from hkjc import scores as SC
from hkjc.db import ROOT

REAL_DB = ROOT / "data" / "hkjc.db"
REAL_MODELS = ROOT / "data" / "models"
CJK = re.compile(r"[一-鿿]")


# ------------------------------------------------------------------ helpers
def _check_scores(df, pred=None):
    assert len(df)
    assert list(df.columns[:7]) == ["race_id", "race_no", "horse_no", "score", "rank_in_race",
                                     "sub_scores", "reasons"]
    assert df["score"].between(0, 100).all()
    assert df["score"].notna().all()
    for rid, g in df.groupby("race_id"):
        g = g.sort_values("p_win", ascending=False, kind="mergesort")
        # rank_in_race is exactly the p_win order
        assert sorted(g["rank_in_race"]) == list(range(1, len(g) + 1))
        gr = g.sort_values("rank_in_race")
        assert np.all(np.diff(gr["p_win"].to_numpy()) <= 1e-15)
        # strictly monotone: higher p_win -> strictly higher score
        sp = g[["p_win", "score"]].to_numpy()
        for i in range(len(sp) - 1):
            if sp[i, 0] > sp[i + 1, 0] + 1e-12:
                assert sp[i, 1] > sp[i + 1, 1]
        assert (g["race_no"] == int(rid[-2:])).all()
    for subs, reasons in zip(df["sub_scores"], df["reasons"]):
        assert set(subs) == set(SC.SUB_SCORE_KEYS)
        for k, v in subs.items():
            assert v is None or (isinstance(v, int) and 0 <= v <= 100), (k, v)
        assert 2 <= len(reasons) <= 4
        for t in reasons:
            assert isinstance(t, str) and t.strip()
            assert CJK.search(t)
            assert not re.search(r"nan|none|null|inf", t, re.I), t


def _synthetic(n_races=3, seed=0):
    rng = np.random.default_rng(seed)
    rows, preds = [], []
    for r in range(1, n_races + 1):
        n = [12, 7, 1][r - 1] if n_races == 3 else 10
        rid = f"20250101-ST-{r:02d}"
        p = rng.dirichlet(np.ones(n))
        for h in range(1, n + 1):
            preds.append({"race_id": rid, "horse_no": h, "p_win": p[h - 1],
                          "p_place": min(1, 3 * p[h - 1])})
            first = h == 1
            rows.append({
                "race_id": rid, "race_no": r, "horse_no": h, "venue": "ST", "date": "2025-01-01",
                "jockey_name": f"騎師{h}", "trainer_name": f"練{h}",
                "h_n": 0 if first else rng.integers(1, 30), "h_first_start": 1.0 if first else 0.0,
                "h_pos_avg3": np.nan if first else rng.uniform(1, 12),
                "h_pos_last1": np.nan if first else float(rng.integers(1, 13)),
                "h_lbw_last1": np.nan if first else rng.uniform(0, 10),
                "h_lbw_avg3": np.nan if first else rng.uniform(0, 10),
                "h_pct_avg3": np.nan if first else rng.uniform(0, 1),
                "h_pct_last1": np.nan if first else rng.uniform(0, 1),
                "h_spd_avg3": np.nan if first else rng.normal(),
                "h_days_since": np.nan if first else float(rng.integers(14, 200)),
                "h_wins": 0.0, "j_n365": float(rng.integers(5, 600)),
                "j_win365": rng.uniform(0.04, 0.2), "t_n365": 300.0, "t_win365": rng.uniform(0.04, 0.15),
                "jt_n": float(rng.integers(0, 40)), "jt_pl": rng.uniform(0.1, 0.5),
                "draw": float(h), "draw_rel": h / n, "draw_pl_resid": rng.normal(0, 0.03),
                "h_dist_n": float(rng.integers(0, 5)), "h_dist_place": 0.3,
                "act_weight": float(rng.integers(113, 136)), "wt_rel": rng.normal(0, 5),
                "class_num": 4.0, "h_class_change": float(rng.choice([-1, 0, 1])),
                "win_odds": np.nan, "mkt_p": np.nan, "mkt_rank": np.nan,
            })
    return pd.DataFrame(preds), pd.DataFrame(rows)


# ------------------------------------------------------------------ unit tests
def test_score_formula_monotone_and_reference_points():
    p = np.linspace(0.001, 0.9, 400)
    s = SC.score_from_pwin(p, 12)
    assert np.all(np.diff(s) > 0)
    assert s.min() > 0 and s.max() < 100
    # comparable across field sizes: same multiple of 1/N -> same score
    assert SC.score_from_pwin(1 / 8, 8) == pytest.approx(SC.score_from_pwin(1 / 14, 14))
    assert SC.score_from_pwin(1 / 10, 10) == pytest.approx(67.7, abs=0.1)
    assert "評分" in SC.SCORE_EXPLANATION_ZH and str(SC.SCORE_S) in SC.SCORE_EXPLANATION_ZH
    assert set(SC.SUB_SCORE_NAMES_ZH) == set(SC.SUB_SCORE_FEATURES)


def test_score_horses_synthetic():
    pred, card = _synthetic()
    out = SC.score_horses(pred, card)
    _check_scores(out)
    assert len(out) == len(pred)
    # single-runner race: neutral sub-scores, rank 1
    one = out[out["race_id"] == "20250101-ST-03"].iloc[0]
    assert one["rank_in_race"] == 1 and one["sub_scores"]["form"] == 50
    # no odds -> market None
    assert all(s["market"] is None for s in out["sub_scores"])
    # debutant always flagged
    deb = out[(out["race_id"] == "20250101-ST-01") & (out["horse_no"] == 1)].iloc[0]
    assert "初出馬，無往績" in deb["reasons"]


def test_score_horses_without_features_and_with_odds():
    pred, card = _synthetic()
    # no card rows at all -> still valid scores / reasons
    out = SC.score_horses(pred, card.iloc[0:0])
    _check_scores(out)
    # odds present -> market sub-score follows mkt_p
    card = card.copy()
    card["win_odds"] = 1.0 / pred["p_win"].clip(0.01) * 0.82
    inv = 1 / card["win_odds"]
    card["mkt_p"] = inv / inv.groupby(card["race_id"]).transform("sum")
    card["mkt_rank"] = card.groupby("race_id")["win_odds"].rank(method="min")
    out = SC.score_horses(pred, card)
    _check_scores(out)
    r1 = out[out["race_id"] == "20250101-ST-01"]
    m = r1["sub_scores"].map(lambda s: s["market"])
    assert m.notna().all()
    assert m.iloc[0] == 100          # top p_win is the market favourite here


def test_save_scores_round_trip(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    pred, card = _synthetic()
    out = SC.score_horses(pred, card)
    n = SC.save_scores(conn, "test-run", out)
    assert n == len(out)
    raw = conn.execute("SELECT sub_scores, reasons FROM horse_scores LIMIT 1").fetchone()
    assert "\\u" not in raw[1] and CJK.search(raw[1])      # ensure_ascii=False
    back = SC.load_scores(conn, "test-run")
    assert len(back) == len(out)
    m = out.merge(back, on=["race_id", "horse_no"], suffixes=("", "_db"))
    assert np.allclose(m["score"], m["score_db"])
    assert (m["rank_in_race"] == m["rank_in_race_db"]).all()
    assert all(a == b for a, b in zip(m["sub_scores"], m["sub_scores_db"]))
    assert all(a == b for a, b in zip(m["reasons"], m["reasons_db"]))
    # re-saving replaces, never duplicates
    SC.save_scores(conn, "test-run", out)
    assert conn.execute("SELECT COUNT(*) FROM horse_scores").fetchone()[0] == len(out)


# ------------------------------------------------------------------ real data (DB copy)
@pytest.fixture(scope="module")
def real_copy(tmp_path_factory):
    if not REAL_DB.exists():
        pytest.skip("no data/hkjc.db")
    d = tmp_path_factory.mktemp("scores")
    src = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True)
    dst = sqlite3.connect(str(d / "copy.db"))
    src.backup(dst)
    src.close()
    dst.close()
    mdir = d / "models"
    mdir.mkdir()
    for f in REAL_MODELS.glob("combined_latest.*"):
        shutil.copy(f, mdir / f.name)
    mp = pytest.MonkeyPatch()
    mp.setenv("HKJC_DB", str(d / "copy.db"))
    mp.setenv("HKJC_MODEL_DIR", str(mdir))
    conn = db.connect(d / "copy.db")
    yield conn, mdir
    conn.close()
    mp.undo()


def test_score_meeting_upcoming_2026_10_21(real_copy):
    conn, mdir = real_copy
    n_card = conn.execute("SELECT COUNT(*) FROM racecard WHERE date='2026-10-21' AND venue='HV' "
                          "AND COALESCE(scratched,0)=0").fetchone()[0]
    if n_card == 0:
        pytest.skip("no racecard for 2026-10-21 HV")
    t = time.time()
    out = SC.score_meeting(conn, "2026-10-21", "HV", model_dir=mdir)
    assert time.time() - t < 30          # < 10 s warm; generous for a loaded CI box
    _check_scores(out)
    assert set(out["race_no"]) == set(range(1, 10))
    assert len(out) <= n_card
    # per race p_win sums to 1
    assert np.allclose(out.groupby("race_id")["p_win"].sum(), 1.0)
    # jockey names are used in the reasons (racecard names)
    assert any("潘頓" in t for r in out["reasons"] for t in r)
    n = SC.save_scores(conn, "test-pre-20261021", out)
    back = SC.load_scores(conn, "test-pre-20261021")
    assert n == len(back) == len(out)


def test_score_meeting_past_2026_10_14(real_copy):
    conn, mdir = real_copy
    have = conn.execute("SELECT COUNT(*) FROM predictions WHERE run_id='wf-combined-v1' "
                        "AND race_id LIKE '20261014-HV-%'").fetchone()[0]
    if have == 0:
        pytest.skip("no walk-forward predictions for 2026-10-14")
    out = SC.score_meeting(conn, "2026-10-14", "HV")
    _check_scores(out)
    assert set(out["race_no"]) == set(range(1, 10))
    # uses the stored walk-forward probabilities
    wf = pd.read_sql_query("SELECT race_id, horse_no, p_win FROM predictions WHERE run_id='wf-combined-v1'"
                           " AND race_id LIKE '20261014-HV-%'", conn)
    m = out.merge(wf, on=["race_id", "horse_no"], suffixes=("", "_wf"))
    assert len(m) == len(out) and np.allclose(m["p_win"], m["p_win_wf"])
    # final odds known for a past meeting -> market sub-score present
    assert all(s["market"] is not None for s in out["sub_scores"])
    # the model's top pick (rank 1) is the max score of each race
    top = out.loc[out.groupby("race_id")["score"].idxmax()]
    assert (top["rank_in_race"] == 1).all()
