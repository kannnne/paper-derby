"""Feature-engineering tests on a small real-data subset copied from data/hkjc.db."""
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hkjc import db as dbmod
from hkjc import features as F

REAL_DB = Path(__file__).resolve().parent.parent / "data" / "hkjc.db"
SUBSET_END = "2024-01-31"


def make_subset_db(path: Path, end: str = SUBSET_END) -> Path:
    """Copy races/runners/dividends/meetings with date <= end into a fresh DB."""
    if not REAL_DB.exists():
        pytest.skip("data/hkjc.db not available")
    conn = sqlite3.connect(str(path))
    dbmod.init_db(conn)
    conn.execute("ATTACH DATABASE ? AS src", (str(REAL_DB),))   # read only usage
    conn.execute("INSERT INTO meetings SELECT * FROM src.meetings WHERE date <= ?", (end,))
    conn.execute("INSERT INTO races SELECT * FROM src.races WHERE date <= ?", (end,))
    conn.execute("INSERT INTO runners SELECT u.* FROM src.runners u JOIN src.races r "
                 "ON r.race_id = u.race_id WHERE r.date <= ?", (end,))
    conn.execute("INSERT INTO dividends SELECT d.* FROM src.dividends d JOIN src.races r "
                 "ON r.race_id = d.race_id WHERE r.date <= ?", (end,))
    conn.commit()
    conn.execute("DETACH DATABASE src")
    n = conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0]
    conn.close()
    if n < 2000:
        pytest.skip("not enough backfilled data for tests")
    return path


@pytest.fixture(scope="module")
def subset_db(tmp_path_factory):
    p = tmp_path_factory.mktemp("feat") / "subset.db"
    return make_subset_db(p)


def _conn(path):
    c = sqlite3.connect(str(path), timeout=30)
    return c


def _pick_date(conn, frac=0.7):
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM races WHERE race_id IN (SELECT race_id FROM runners WHERE finish_pos=1) "
        "ORDER BY date").fetchall()]
    return dates[int(len(dates) * frac)]


def _assert_frames_equal(a, b, cols):
    a = a.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    b = b.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    assert list(a["race_id"]) == list(b["race_id"])
    assert list(a["horse_no"]) == list(b["horse_no"])
    for c in cols:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        same = np.isclose(x, y, rtol=1e-9, atol=1e-12) | (np.isnan(x) & np.isnan(y))
        assert same.all(), f"column {c} differs: {x[~same][:5]} vs {y[~same][:5]}"


# --------------------------------------------------------------------------- unit
def test_parsers():
    assert F.class_num("第五班") == 5
    assert F.class_num("第四班（條件限制）") == 4
    assert F.class_num("一級賽") == -2
    assert F.class_num("三級賽") == 0
    assert F.class_num("新馬賽") == 5.5
    assert np.isnan(F.class_num(None))
    assert F.band_hi("60-40") == 60
    assert F.band_hi("85+") == 85
    assert F.band_mid("60-40") == 50
    assert F.band_mid("40-0") == 27.5
    assert F.band_mid(None, 3.0) == 70
    assert F.going_num("好地") == 0
    assert list(F.dist_bucket([1000, 1600, 1650, 1800])) == [1000, 1600, 1600, 1800]


def test_market_features_accept_minimum_odds():
    base = pd.DataFrame({c: [None] * 3 for c in F._BASE_COLS})
    base["race_id"] = "20240101-ST-01"
    base["date"], base["venue"], base["race_no"] = "2024-01-01", "ST", 1
    base["horse_no"] = [1, 2, 3]
    base["horse_id"] = ["A1", "A2", "A3"]
    base["win_odds"] = [1.0, 15.0, 30.0]
    base["finish_pos"] = [1, 2, 3]
    base["_known"] = 1.0
    out = F.compute_features(base)
    inv = np.array([1.0, 1 / 15, 1 / 30])
    assert np.allclose(out["mkt_p"], inv / inv.sum())
    assert out["mkt_rank"].tolist() == [1, 2, 3]


def test_prior_cum_excludes_same_day():
    df = pd.DataFrame({"k": ["a", "a", "a", "a", "b"],
                       "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-05", "2024-03-10",
                                               "2024-01-05"]),
                       "_start": [1.0, 1, 1, 1, 1], "_win": [1.0, 0, 1, 0, 1]})
    c = F._prior_cum(df, ["k"], ["_start", "_win"])
    assert c["_start"].tolist() == [0, 0, 2, 3, 0]
    assert c["_win"].tolist() == [0, 0, 1, 2, 0]
    w = F._prior_window(df, ["k"], ["_start", "_win"], 60)
    # 2024-03-10 window [2024-01-10, 2024-03-10) -> nothing
    assert w["_start"].tolist() == [0, 0, 2, 0, 0]
    w = F._prior_window(df, ["k"], ["_start"], 4)
    # 2024-01-05 window [2024-01-01, 2024-01-05) -> both 01-01 rows
    assert w["_start"].tolist() == [0, 0, 2, 0, 0]


# --------------------------------------------------------------------------- frame
def test_training_frame_basic(subset_db):
    conn = _conn(subset_db)
    df = F.build_training_frame(conn)
    assert len(df) > 2000
    for c in F.ALL_FEATURES + F.TARGET_COLS:
        assert c in df.columns, c
    # no scratched horses
    scr = pd.read_sql_query("SELECT race_id, horse_no FROM runners WHERE scratched=1", conn)
    m = df.merge(scr, on=["race_id", "horse_no"])
    assert m.empty
    # targets
    assert df.groupby("race_id")["win"].sum().min() >= 1
    exp_pl = np.where(df["field_size"] >= 7, 3, 2)
    assert (df["n_places"] == exp_pl).all()
    assert ((df["place"] == 1) == (df["finish_pos"] <= df["n_places"])).all()
    small = df[df["field_size"] <= 6]
    if len(small):
        assert small.groupby("race_id")["place"].sum().max() <= 3   # 2 (+ dead heat)
    # market probabilities normalised
    s = df.groupby("race_id")["mkt_p"].sum()
    assert np.allclose(s, 1.0)
    # every runner with odds (including the HKJC minimum 1.0) gets a market probability
    assert df.loc[df["win_odds"] >= 1.0, "mkt_p"].notna().all()
    # first meeting: no history at all
    first = df[df["date"] == df["date"].min()]
    assert (first["h_n"] == 0).all() and (first["j_n365"] == 0).all()
    # start/end filtering
    d = _pick_date(conn)
    part = F.build_training_frame(conn, start=d, end=d)
    assert set(part["date"]) == {d}
    _assert_frames_equal(part, df[df["date"] == d], F.ALL_FEATURES)


def test_horse_history_consistency(subset_db):
    """h_n equals the number of the horse's earlier race days, h_pos_last1 its last finish."""
    conn = _conn(subset_db)
    df = F.build_training_frame(conn)
    hid = df["horse_id"].value_counts().index[0]
    h = df[df["horse_id"] == hid].sort_values("date")
    assert h["h_n"].tolist() == list(range(len(h)))
    fp = h["finish_pos"].fillna(h["field_size"]).tolist()
    assert np.isnan(h["h_pos_last1"].iloc[0])
    assert h["h_pos_last1"].iloc[1:].tolist() == fp[:-1]


def test_no_leakage_perturb_future(subset_db, tmp_path):
    """Delete everything after race date D and scramble every result on D:
    features of all D races must be unchanged."""
    conn = _conn(subset_db)
    D = _pick_date(conn)
    before = F.build_training_frame(conn)
    before = before[before["date"] == D]
    assert len(before) > 50
    conn.close()

    p2 = tmp_path / "perturbed.db"
    shutil.copy(subset_db, p2)
    c2 = _conn(p2)
    later = "SELECT race_id FROM races WHERE date > ?"
    c2.execute(f"DELETE FROM runners WHERE race_id IN ({later})", (D,))
    c2.execute(f"DELETE FROM dividends WHERE race_id IN ({later})", (D,))
    c2.execute("DELETE FROM races WHERE date > ?", (D,))
    rng = np.random.default_rng(0)
    rows = c2.execute("SELECT u.race_id, u.horse_no FROM runners u JOIN races r ON r.race_id=u.race_id "
                      "WHERE r.date=? AND u.scratched=0 ORDER BY u.race_id, u.horse_no", (D,)).fetchall()
    by_race = {}
    for rid, hn in rows:
        by_race.setdefault(rid, []).append(hn)
    for rid, hns in by_race.items():
        perm = rng.permutation(len(hns)) + 1
        for hn, fp in zip(hns, perm):
            c2.execute("UPDATE runners SET finish_pos=?, finish_status=?, lbw_num=?, finish_time_s=?, "
                       "running_pos=? WHERE race_id=? AND horse_no=?",
                       (int(fp), str(fp), float(fp - 1) * 1.3, 60.0 + fp, f"{fp} {fp} {fp}", rid, hn))
        c2.execute("UPDATE races SET winning_time_s = winning_time_s + 7 WHERE race_id=?", (rid,))
    c2.commit()
    after = F.build_training_frame(c2)
    after = after[after["date"] == D]
    assert after["date"].max() == D
    # perturbation really happened
    m = before.merge(after, on=["race_id", "horse_no"], suffixes=("_a", "_b"))
    assert (m["finish_pos_a"] != m["finish_pos_b"]).any()
    _assert_frames_equal(before, after, F.ALL_FEATURES)


def _synth_racecard(conn, D):
    """Fill the racecard table for date D from that meeting's runners (allowance 0)."""
    venue = conn.execute("SELECT venue FROM races WHERE date=? LIMIT 1", (D,)).fetchone()[0]
    rows = conn.execute("""
        SELECT r.date, r.venue, r.race_no, u.horse_no, u.horse_id, u.horse_name, u.jockey_id,
               u.jockey_name, u.trainer_id, u.trainer_name, u.act_weight, u.draw, u.body_weight,
               u.scratched, u.win_odds, u.finish_status
        FROM runners u JOIN races r ON r.race_id = u.race_id WHERE r.date=? AND r.venue=?""",
                        (D, venue)).fetchall()
    odds = {}
    for (d, v, rno, hn, hid, hname, jid, jn, tid, tn, w, dr, bw, scr, wo, fs) in rows:
        scr = 1 if (scr or fs in ("TNP",)) else 0
        conn.execute("INSERT OR REPLACE INTO racecard (date, venue, race_no, horse_no, horse_id, horse_name,"
                     " jockey_id, jockey_name, trainer_id, trainer_name, weight, draw, body_weight, scratched)"
                     " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (d, v, rno, hn, hid, hname, jid, jn, tid, tn, w, dr, bw, scr))
        if wo is not None:
            odds.setdefault(rno, {}).setdefault("WIN", {})[str(hn)] = wo
    conn.commit()
    return venue, odds


def test_card_frame_matches_training(subset_db, tmp_path):
    p2 = tmp_path / "card.db"
    shutil.copy(subset_db, p2)
    conn = _conn(p2)
    D = _pick_date(conn, 0.8)
    venue, odds = _synth_racecard(conn, D)
    train = F.build_training_frame(conn)
    train = train[train["date"] == D]
    card = F.build_card_frame(conn, D, venue, odds=odds)
    assert len(card) == len(train) > 50
    assert card["win"].isna().all()
    _assert_frames_equal(train, card, F.ALL_FEATURES + ["n_places"])
    # without odds: market columns empty, the rest unchanged
    card2 = F.build_card_frame(conn, D, venue, odds={})
    assert card2["mkt_p"].isna().all()
    _assert_frames_equal(train, card2, F.FUNDAMENTAL_FEATURES)


def test_card_frame_ignores_results_on_and_after_date(subset_db, tmp_path):
    """Card features only use races before the card date even when later results exist."""
    p2 = tmp_path / "card2.db"
    shutil.copy(subset_db, p2)
    conn = _conn(p2)
    D = _pick_date(conn, 0.5)
    venue, odds = _synth_racecard(conn, D)
    card_full = F.build_card_frame(conn, D, venue, odds=odds)
    later = "SELECT race_id FROM races WHERE date >= ?"
    conn.execute(f"DELETE FROM runners WHERE race_id IN ({later})", (D,))
    conn.execute("UPDATE races SET winning_time_s=NULL WHERE date >= ?", (D,))
    conn.commit()
    card_cut = F.build_card_frame(conn, D, venue, odds=odds)
    _assert_frames_equal(card_full, card_cut, F.ALL_FEATURES)


def test_speed_on_full_db():
    if not REAL_DB.exists():
        pytest.skip("no db")
    conn = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True, timeout=60)
    t = time.time()
    df = F.build_training_frame(conn)
    dt = time.time() - t
    assert len(df) > 0
    assert dt < 60, f"feature build took {dt:.1f}s"


def test_card_rows_drop_horse_scratched_in_live_odds(subset_db, tmp_path):
    """A horse scratched after the racecard scrape shows SCR (None) in the live WIN
    pool: it must not be predicted (nor bet on); other races are untouched."""
    p2 = tmp_path / "card3.db"
    shutil.copy(subset_db, p2)
    conn = _conn(p2)
    D = _pick_date(conn, 0.8)
    venue, odds = _synth_racecard(conn, D)
    rn = sorted(odds)[0]
    victim = sorted(odds[rn]["WIN"], key=int)[0]
    base = F.card_rows(conn, D, venue, odds=odds)
    odds[rn]["WIN"][victim] = None                      # 'SCR' in the live pool
    cut = F.card_rows(conn, D, venue, odds=odds)
    assert len(cut) == len(base) - 1
    assert not ((cut["race_no"] == rn) & (cut["horse_no"] == int(victim))).any()
    # a race whose WIN pool isn't open yet (all None) keeps every horse
    rn2 = sorted(odds)[1]
    odds[rn2]["WIN"] = {k: None for k in odds[rn2]["WIN"]}
    cut2 = F.card_rows(conn, D, venue, odds=odds)
    assert (cut2["race_no"] == rn2).sum() == (base["race_no"] == rn2).sum()
