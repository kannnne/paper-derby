"""預覽（未落注）: robot decisions + 今日精選 previewed with the hourly scores run (kind 'pre_scores').

Synthetic DB, model / scoring stubbed: checks that a preview never writes tickets or ledger rows,
keeps only the newest preview, is superseded by the real betting run in the export, and is
flagged in the exported JSON.  Nothing touches data/hkjc.db or site/."""
import json

import numpy as np
import pandas as pd
import pytest

from hkjc import bots, cli, db, paper, pipeline, publish

D, V = "2099-05-05", "HV"


def _rid(rn):
    return f"{D.replace('-', '')}-{V}-{rn:02d}"


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "p.db")
    c.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)", (D, V, 2, f"{D}T19:15", "card", "x"))
    for rn in (1, 2):
        c.execute("INSERT INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,?,?)",
                  (_rid(rn), D, V, rn, f"{D}T{18 + rn}:15"))
        for h in range(1, 9):
            c.execute("INSERT INTO racecard (date, venue, race_no, horse_no, horse_name, jockey_name, trainer_name, "
                      "weight, draw, scratched) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (D, V, rn, h, f"馬{rn}{h}", "騎", "練", 120, h, 0))
    c.commit()
    yield c
    c.close()


def _pred():
    rng = np.random.default_rng(7)
    out = []
    for rn in (1, 2):
        p = rng.dirichlet(np.full(8, 1.5))
        out.append(pd.DataFrame({"race_id": _rid(rn), "race_no": rn, "horse_no": range(1, 9), "p_win": p,
                                 "p_place": np.clip(p * 2.6, 0.02, 0.95), "win_odds": np.round(0.82 / p, 1),
                                 "horse_name": [f"馬{rn}{h}" for h in range(1, 9)], "model_version": "test"}))
    return pd.concat(out, ignore_index=True)


def _odds(pred):
    return {int(rn): {"WIN": {str(h): float(o) for h, o in zip(g.horse_no, g.win_odds)},
                      "PLA": {str(h): round(max(1.05, float(o) / 3.2), 1) for h, o in zip(g.horse_no, g.win_odds)}}
            for rn, g in pred.groupby("race_no")}


@pytest.fixture()
def stubs(monkeypatch, conn):
    pred = _pred()
    odds = _odds(pred)
    monkeypatch.setattr(pipeline.model, "predict_card", lambda *a, **k: pred.drop(columns=["horse_name"]).copy())
    monkeypatch.setattr(pipeline, "latest_odds", lambda *a, **k: odds)

    def fake_scores(c, date, venue, pred_df=None, odds=None):
        d = pred_df[["race_id", "race_no", "horse_no"]].copy()
        d["score"] = 50.0 + 100 * pred_df["p_win"].values
        d["rank_in_race"] = d.groupby("race_no")["score"].rank(ascending=False).astype(int)
        d["sub_scores"] = [{} for _ in range(len(d))]
        d["reasons"] = [["測試"] for _ in range(len(d))]
        return d
    monkeypatch.setattr(pipeline.scores, "score_meeting", fake_scores)
    return pred, odds


def _counts(c):
    return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("paper_tickets", "bankroll_ledger", "bot_decisions", "picks", "horse_scores", "predictions")}


def _preview(c, run_id):
    n = pipeline.preview_scores(c, run_id, D, V)
    cli._record_run(c, run_id, "pre_scores", D, V, {})
    return n


def test_preview_writes_decisions_and_picks_but_no_tickets(conn, stubs):
    assert _preview(conn, "score-a") == 16
    n = _counts(conn)
    assert n["paper_tickets"] == 0 and n["bankroll_ledger"] == 0          # never bets, never seeds a ledger
    rows = conn.execute("SELECT * FROM bot_decisions WHERE run_id='score-a'").fetchall()
    assert {r["bot_id"] for r in rows} == {b["id"] for b in bots.BOTS}
    assert all(r["ticket_id"] is None for r in rows)
    assert any(r["action"] == "bet" and r["stake"] > 0 for r in rows)
    pk = json.loads(conn.execute("SELECT payload FROM picks WHERE run_id='score-a'").fetchone()[0])
    assert pk["items"] and pk["n_items"] == len(pk["items"])
    for b in bots.BOTS:       # every robot x every race has a row
        got = {r["race_no"] for r in rows if r["bot_id"] == b["id"] and r["race_no"] > 0}
        assert got == {1, 2}


def test_preview_cleanup_keeps_only_newest(conn, stubs):
    _preview(conn, "score-a")
    one = _counts(conn)
    _preview(conn, "score-b")
    assert _counts(conn) == one                                          # same size, older run gone
    for t in ("bot_decisions", "picks", "horse_scores", "predictions", "runs"):
        assert conn.execute(f"SELECT COUNT(*) FROM {t} WHERE run_id='score-a'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM bot_decisions WHERE run_id='score-b'").fetchone()[0] > 0


def test_preview_skipped_once_betting_run_exists(conn, stubs):
    cli._record_run(conn, "pre-x", "pre_bet", D, V, {})
    assert pipeline.preview_decisions(conn, "score-z", D, V, _pred(), stubs[1]) == 0
    assert conn.execute("SELECT COUNT(*) FROM bot_decisions").fetchone()[0] == 0


def test_export_flags_preview_and_excludes_it_from_totals(conn, stubs, tmp_path):
    _preview(conn, "score-a")
    out = tmp_path / "site"
    publish.export_site(conn, out)
    m = json.loads((out / "data" / "meetings" / f"{D}_{V}.json").read_text(encoding="utf-8"))
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import test_publish as tp
    tp.validate_meeting(m)
    tp.validate_picks(m["picks"])
    assert m["preview"] is True and m["preview_at"]
    assert m["picks"]["preview"] is True and m["picks"]["items"]
    tks = [t for b in m["bots"] for t in b["tickets"]]
    assert tks and all(t["status"] == "preview" and t["payout"] is None and t["ticket_id"] is None for t in tks)
    assert all(len(t["leg_hits"]) == len(t["legs"]) and t["legs"] for t in tks)
    stake = {b["bot_id"]: b["stake"] for b in m["bots"]}
    assert sum(stake.values()) == pytest.approx(sum(t["total_stake"] for t in tks))
    assert all(b.get("preview") for b in m["bots"])
    decs = [d for r in m["races"] for d in r["decisions"]] + m["meeting_decisions"]
    assert decs and all(d.get("preview") for d in decs)
    idx = json.loads((out / "data" / "index.json").read_text(encoding="utf-8"))
    meta = {x["date"]: x for x in idx["meetings"]}[D]
    assert meta["preview"] is True and meta["has_picks"] is True and meta["has_decisions"] is True
    for b in idx["bots"]:                                   # no preview stake in the leaderboard / bankrolls
        assert b["live"]["stake"] == 0 and b["live"]["open_stake"] == 0 and b["bankroll"] == 10_000
    for p in (out / "data" / "bots").glob("*.json"):        # nor in a robot's ticket history
        assert json.loads(p.read_text(encoding="utf-8"))["n_tickets"] == 0


def test_betting_run_supersedes_preview(conn, stubs, tmp_path, monkeypatch):
    pred, odds = stubs
    _preview(conn, "score-a")
    # the real T-2h run (stubbed model): tickets placed, preview decisions / picks of the meeting dropped
    monkeypatch.setattr(pipeline.model, "predict_card", lambda *a, **k: pred.drop(columns=["horse_name"]).copy())
    pipeline.pre_meeting(conn, "pre-real", D, V)
    cli._record_run(conn, "pre-real", "pre_bet", D, V, {})
    assert conn.execute("SELECT COUNT(*) FROM bot_decisions WHERE run_id='score-a'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM picks WHERE run_id='score-a'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_tickets").fetchone()[0] > 0
    m = publish.export_meeting(conn, D, V)
    assert m["preview"] is False and m["picks"]["preview"] is False
    assert all(t["status"] == "open" for b in m["bots"] for t in b["tickets"])
    assert m["decisions_run_id"] == "pre-real"
    # a later forced preview never mixes with the real decisions
    assert pipeline.preview_decisions(conn, "score-late", D, V, pred, odds) == 0


def test_preview_meetings_not_counted_after_results(conn, stubs):
    _preview(conn, "score-a")
    conn.execute("UPDATE meetings SET status='results' WHERE date=?", (D,))
    conn.commit()
    assert all(s["n_meetings"] == 0 for s in paper.bot_summary(conn))
    live, _ = publish._live_stats(conn, [b["id"] for b in bots.BOTS])
    assert all(v["live"]["n_meetings"] == 0 for v in live.values())


def test_cmd_scores_preview_end_to_end(conn, stubs, monkeypatch):
    from hkjc import odds as om, scrape
    pools_seen = []

    def fake_snap(c, date, venue, pools=("WIN",), snapshot_at=None, **k):
        pools_seen.append(tuple(pools))
        c.executemany("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?)",
                      [(date, venue, 1, p, "1", 2.0, snapshot_at) for p in pools])
        c.commit()
        return len(pools)
    monkeypatch.setattr(om, "snapshot_meeting", fake_snap)
    monkeypatch.setattr(scrape, "scrape_racecard", lambda *a, **k: 0)
    conn.execute("INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?)", (D, V, 1, "WIN", "1", 3.0, "2000-01-01T00:00:00"))
    conn.commit()
    mtg = {"date": D, "venue": V, "first_post_time": f"{D}T19:15", "n_races": 2}
    assert cli.cmd_scores(conn, mtg=mtg) == 16
    assert pools_seen[0] == ("WIN", "PLA", "QIN", "QPL") and pools_seen[1] == tuple(om.EXOTIC_POOLS)
    assert conn.execute("SELECT COUNT(*) FROM odds_snapshots WHERE snapshot_at='2000-01-01T00:00:00'").fetchone()[0] == 0
    assert conn.execute("SELECT kind FROM runs").fetchall()[0][0] == "pre_scores"
    assert _counts(conn)["paper_tickets"] == 0 and _counts(conn)["bankroll_ledger"] == 0
    assert conn.execute("SELECT COUNT(*) FROM bot_decisions WHERE ticket_id IS NULL").fetchone()[0] > 0
