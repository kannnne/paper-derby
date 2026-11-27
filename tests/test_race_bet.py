"""Per-race late betting: single-race bets are decided ~8 min before EACH race with that race's
latest odds (cli.cmd_race_bet), multi-race ones stay with the T-2h run (pipeline.pre_meeting).

Synthetic 6-race meeting, model / odds / scoring stubbed, no network; every DB is a temp file."""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from hkjc import bots, cli, db, health, paper, picks, pipeline, scrape
from hkjc import odds as om

_REAL_RACE_WINDOW = cli._race_window

D, V = "2099-05-05", "HV"
N_RACES = 6
PRE = "pre-real"


def _rid(rn):
    return f"{D.replace('-', '')}-{V}-{rn:02d}"


def _post(rn):
    h, m = divmod(19 * 60 + 15 + 30 * (rn - 1), 60)
    return f"{D}T{h:02d}:{m:02d}"


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "r.db")
    c.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)", (D, V, N_RACES, _post(1), "card", "x"))
    for rn in range(1, N_RACES + 1):
        c.execute("INSERT INTO races (race_id, date, venue, race_no, post_time) VALUES (?,?,?,?,?)",
                  (_rid(rn), D, V, rn, _post(rn)))
        for h in range(1, 9):
            c.execute("INSERT INTO racecard (date, venue, race_no, horse_no, horse_name, jockey_name, trainer_name, "
                      "weight, draw, scratched) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (D, V, rn, h, f"馬{rn}{h}", "騎", "練", 120, h, 0))
    c.commit()
    yield c
    c.close()


def _pred(seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for rn in range(1, N_RACES + 1):
        p = rng.dirichlet(np.full(8, 1.5))
        out.append(pd.DataFrame({"race_id": _rid(rn), "race_no": rn, "horse_no": range(1, 9), "p_win": p,
                                 "p_place": np.clip(p * 2.6, 0.02, 0.95), "win_odds": np.round(0.82 / p, 1),
                                 "model_version": "test"}))
    return pd.concat(out, ignore_index=True)


def _odds(pred):
    return {int(rn): {"WIN": {str(h): float(o) for h, o in zip(g.horse_no, g.win_odds)},
                      "PLA": {str(h): round(max(1.05, float(o) / 3.2), 1) for h, o in zip(g.horse_no, g.win_odds)}}
            for rn, g in pred.groupby("race_no")}


@pytest.fixture()
def world(monkeypatch, conn):
    """Stubbed model / odds / scores; ``world['pred']`` / ``world['odds']`` can be swapped per test."""
    w = {"pred": _pred(), "snaps": [], "cards": []}
    w["odds"] = _odds(w["pred"])
    monkeypatch.setattr(pipeline.model, "predict_card", lambda *a, **k: w["pred"].copy())
    monkeypatch.setattr(pipeline, "latest_odds", lambda *a, **k: w["odds"])

    def fake_scores(c, date, venue, pred_df=None, odds=None):
        d = pred_df[["race_id", "race_no", "horse_no"]].copy()
        d["score"] = 50.0 + 100 * pred_df["p_win"].values
        d["rank_in_race"] = d.groupby("race_no")["score"].rank(ascending=False).astype(int)
        d["sub_scores"] = [{} for _ in range(len(d))]
        d["reasons"] = [["測試"] for _ in range(len(d))]
        return d
    monkeypatch.setattr(pipeline.scores, "score_meeting", fake_scores)

    def fake_snap(c, date, venue, pools=om.DEFAULT_POOLS, race_nos=None, **k):
        w["snaps"].append((tuple(pools), None if race_nos is None else list(race_nos)))
        c.execute("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?)",
                  (date, venue, race_nos[0] if race_nos else 1, pools[0], "1", 2.0,
                   cli.now_hkt().isoformat(timespec="seconds")))
        c.commit()
        return 5
    monkeypatch.setattr(om, "snapshot_meeting", fake_snap)
    monkeypatch.setattr(cli, "_refresh_race_card", lambda c, d, v, rn: w["cards"].append(rn) or False)
    monkeypatch.setattr(cli, "_publish", lambda *a, **k: None)
    # tests without the `clock` fixture call cmd_race_bet "inside the window"; time-window tests
    # use `clock`, which puts the real window check back
    monkeypatch.setattr(cli, "_race_window", lambda *a, **k: ("open", None))
    return w


def _pre(conn):
    res = pipeline.pre_meeting(conn, PRE, D, V)
    cli._record_run(conn, PRE, "pre_bet", D, V, res["notes"])
    return res


def _tickets(conn):
    return [dict(t, legs=json.loads(t["legs"])) for t in conn.execute("SELECT * FROM paper_tickets ORDER BY ticket_id")]


def _payload(conn):
    return picks.load_picks(conn, D, V, run_id=PRE)


# ------------------------------------------------------------------------- T-2h run
def test_t2h_places_only_multi_race_tickets(conn, world):
    res = _pre(conn)
    assert res["notes"]["per_race"] is True
    tks = _tickets(conn)
    assert tks, "the fixed 6x63 / QPL 2x3 all-ups are still bought at T-2h"
    assert all(len({l["race_no"] for l in t["legs"]}) > 1 for t in tks)
    rows = conn.execute("SELECT * FROM bot_decisions WHERE run_id=?", (PRE,)).fetchall()
    for r in rows:
        single = bots.is_single_race(r["bot_id"], dict(r))
        assert (r["action"] == "pending") == single, (r["bot_id"], r["race_no"], r["action"])
        if single:
            assert r["reason"].startswith(bots.PENDING_PREFIX) and r["ticket_id"] is None
    # every single-race robot has a pending row for every race; meeting-level robots none
    for b in bots.BOTS:
        got = {r["race_no"] for r in rows if r["bot_id"] == b["id"] and r["action"] == "pending"}
        if bots.bot_kind(b["id"]) in bots.MEETING_LEVEL_KINDS:
            assert got == set()
        else:
            assert got == set(range(1, N_RACES + 1)), b["id"]
    assert any(r["action"] == "pending" and r["stake"] > 0 for r in rows)        # preview stake kept
    pk = _payload(conn)
    for it in pk["items"]:
        if it["kind"] == "single":
            assert it["status"] == "pending"
            post = datetime.strptime(_post(it["race_no"]), "%Y-%m-%dT%H:%M")
            assert it["decide_at"] == (post - cli.RACE_BET_LEAD).strftime("%Y-%m-%dT%H:%M")
        else:
            assert it["status"] == "final"
    # daily_picks bought exactly the final (multi-race) items
    dp = {t["detail"] and json.loads(t["detail"])["notes"].get("pick_key") for t in tks if t["bot_id"] == "daily_picks"}
    assert dp == {it["key"] for it in pk["items"] if it["status"] == "final"}


# --------------------------------------------------------------------- race-time step
def test_race_bet_decides_one_race_only(conn, world):
    _pre(conn)
    before = {t["ticket_id"]: t["status"] for t in _tickets(conn)}
    created = {r["bot_id"]: r["created_at"] for r in conn.execute("SELECT * FROM bot_decisions WHERE run_id=?", (PRE,))}
    # race 1's odds move: the favourite drifts, another horse is backed
    world["odds"] = {k: {p: dict(b) for p, b in v.items()} for k, v in world["odds"].items()}
    world["odds"][1]["WIN"] = {h: round(o * (2.0 if h == "1" else 0.8), 1) for h, o in world["odds"][1]["WIN"].items()}
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes and notes["race_no"] == 1
    assert ((tuple(om.DEFAULT_POOLS), [1]) in world["snaps"]) and ((tuple(om.EXOTIC_POOLS), [1]) in world["snaps"])
    assert world["cards"] == [1]
    # pending rows of race 1 replaced, the other races still pending
    assert not bots.pending_rows(conn, PRE, 1)
    for rn in range(2, N_RACES + 1):
        assert bots.pending_rows(conn, PRE, rn)
    rows1 = conn.execute("SELECT * FROM bot_decisions WHERE run_id=? AND race_no=1", (PRE,)).fetchall()
    assert {r["action"] for r in rows1} <= {"bet", "skip"}
    assert all(r["created_at"] == created[r["bot_id"]] for r in rows1)       # one run per robot on the site
    # new tickets: race 1 only, under the race run id, linked from their rows; old tickets untouched
    after = _tickets(conn)
    new = [t for t in after if t["ticket_id"] not in before]
    assert new and notes["bets"] == len(new)
    assert all({l["race_no"] for l in t["legs"]} == {1} and t["run_id"] == "race-2099-05-05-HV-R01" for t in new)
    assert all(before[t["ticket_id"]] == t["status"] == "open" for t in after if t["ticket_id"] in before)
    linked = {r["ticket_id"] for r in rows1 if r["action"] == "bet" and r["ticket_id"]
              and bots.is_single_race(r["bot_id"], dict(r))}           # (過關腳 rows keep their all-up ticket)
    assert {t["ticket_id"] for t in new} <= linked | {None} and linked <= {t["ticket_id"] for t in new}
    # picks: race-1 single items final, bought by daily_picks; other races still pending
    pk = _payload(conn)
    r1 = [it for it in pk["items"] if it["kind"] == "single" and it["race_no"] == 1]
    assert all(it["status"] == "final" and it["decided_at"] and it["decide_at"] for it in r1)
    assert all(it["status"] == "pending" for it in pk["items"] if it["kind"] == "single" and it["race_no"] != 1)
    keys = {json.loads(t["detail"])["notes"].get("pick_key") for t in new if t["bot_id"] == "daily_picks"}
    assert keys == {it["key"] for it in r1 if not it.get("no_bet")}
    run = conn.execute("SELECT kind, notes FROM runs WHERE run_id='race-2099-05-05-HV-R01'").fetchone()
    assert run["kind"] == "race_bet" and json.loads(run["notes"])["bets"] == len(new)
    # idempotent: a second call does nothing
    n_led = conn.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0]
    assert cli.cmd_race_bet(conn, D, V, 1) is None
    assert len(_tickets(conn)) == len(after)
    assert conn.execute("SELECT COUNT(*) FROM bankroll_ledger").fetchone()[0] == n_led


def test_race_bet_rerun_after_crash_does_not_double_bet(conn, world):
    _pre(conn)
    n0 = len(_tickets(conn))
    pre_bal = {b["id"]: paper.bankroll(conn, b["id"]) for b in bots.BOTS}
    real = bots._replace_race_rows
    state = {"n": 0}

    def boom(*a, **k):                # tickets of the first robot placed, then the process dies
        state["n"] += 1
        raise RuntimeError("killed")
    bots._replace_race_rows = boom
    try:
        with pytest.raises(RuntimeError):
            cli.cmd_race_bet(conn, D, V, 2)
    finally:
        bots._replace_race_rows = real
    assert not conn.execute("SELECT 1 FROM runs WHERE kind='race_bet'").fetchone()     # not recorded: retried
    cli.cmd_race_bet(conn, D, V, 2)
    live = [t for t in _tickets(conn)[n0:] if t["status"] == "open"]
    per_bot = {}
    for t in live:
        per_bot.setdefault(t["bot_id"], []).append(t)
    for bid, ts in per_bot.items():           # each robot's open race-2 stake debited exactly once
        rows = conn.execute("SELECT SUM(stake) FROM bot_decisions WHERE run_id=? AND bot_id=? AND race_no=2 "
                            "AND action='bet'", (PRE, bid)).fetchone()[0] or 0
        assert sum(t["total_stake"] for t in ts) == pytest.approx(rows)
        assert paper.bankroll(conn, bid) == pytest.approx(pre_bal[bid] - rows)


def test_scratched_horse_never_selected(conn, world):
    _pre(conn)
    pred = world["pred"]
    g = pred[pred.race_no == 3]
    fav = int(g.sort_values("p_win", ascending=False).horse_no.iloc[0])     # model's top pick scratched late
    conn.execute("UPDATE racecard SET scratched=1 WHERE date=? AND race_no=3 AND horse_no=?", (D, fav))
    conn.commit()
    world["odds"] = {k: {p: dict(b) for p, b in v.items()} for k, v in world["odds"].items()}
    for pool in ("WIN", "PLA"):
        world["odds"][3][pool][str(fav)] = None                              # 'SCR' in the live book
    cli.cmd_race_bet(conn, D, V, 3)
    for t in _tickets(conn):
        for l in t["legs"]:
            if l["race_no"] == 3 and t["run_id"].startswith("race-"):
                hs = set(l.get("selections") or []) | set(l.get("bankers") or [])
                hs |= {h for pos in (l.get("positions") or []) for h in pos}
                assert fav not in {int(h) for h in hs}
    for it in _payload(conn)["items"]:
        if it["kind"] == "single" and it["race_no"] == 3:
            for l in it["legs"]:
                assert fav not in {h["horse_no"] for h in l["selections"] + l["bankers"]}


def test_bankroll_check_at_race_time(conn, world):
    _pre(conn)
    for bid in ("monkey", "daily_picks"):          # both broke before race 4
        bal = paper.bankroll(conn, bid)
        paper._ledger(conn, None, -bal, "test: empty", bot_id=bid)
    conn.commit()
    cli.cmd_race_bet(conn, D, V, 4)
    m = conn.execute("SELECT * FROM bot_decisions WHERE run_id=? AND bot_id='monkey' AND race_no=4", (PRE,)).fetchall()
    assert [r["action"] for r in m] == ["skip"] and "資金不足" in m[0]["reason"]
    assert not conn.execute("SELECT 1 FROM paper_tickets WHERE bot_id='monkey' AND run_id LIKE 'race-%'").fetchone()
    # daily_picks is never limited by its bankroll (track record stays complete; ledger may go negative)
    items = [it for it in _payload(conn)["items"] if it["kind"] == "single" and it["race_no"] == 4
             and not it.get("no_bet")]
    dp = conn.execute("SELECT * FROM paper_tickets WHERE bot_id='daily_picks' AND run_id LIKE 'race-%'").fetchall()
    assert len(dp) == len(items)
    if items:
        assert paper.bankroll(conn, "daily_picks") < 0


def test_daily_picks_ignores_bankroll_other_robots_do_not():
    pred = _pred()
    odds = _odds(pred)
    pk = picks.make_picks(pred, odds)
    dp = bots.decide("daily_picks", pred, odds, 0.0, picks=pk)
    bets = [d for d in dp if d["action"] == "bet"]
    assert len(bets) == len([it for it in pk["items"] if it.get("ticket")]) and bets
    assert not any("資金不足" in d["reason"] for d in dp)
    mk = bots.decide("monkey", pred, odds, 0.0)
    assert mk and all(d["action"] == "skip" and "資金不足" in d["reason"] for d in mk)


def test_daily_picks_zero_bankroll_still_places_tickets(tmp_path):
    c = db.connect(tmp_path / "z.db")
    pred = _pred()
    odds = _odds(pred)
    pk = picks.make_picks(pred, odds)
    for bid in ("daily_picks", "monkey"):
        paper.seed_bankroll(c, 0.0, bid)
    bots.run_bots(c, "pre-z", D, V, pred, odds, bots=["daily_picks", "monkey"], picks=pk)
    n = {r[0]: r[1] for r in c.execute("SELECT bot_id, COUNT(*) FROM paper_tickets GROUP BY bot_id")}
    assert n.get("daily_picks") == len([it for it in pk["items"] if it.get("ticket")]) and "monkey" not in n
    assert paper.bankroll(c, "daily_picks") < 0
    c.close()


def test_race_bet_needs_pre_bet(conn, world):
    assert cli.cmd_race_bet(conn, D, V, 1) is None
    assert world["snaps"] == []


def test_forced_pre_refused_after_race_bets(conn, world, monkeypatch):
    _pre(conn)
    cli.cmd_race_bet(conn, D, V, 1)
    monkeypatch.setattr(scrape, "scrape_racecard", lambda *a, **k: 0)
    called = []
    monkeypatch.setattr(pipeline, "pre_meeting", lambda *a, **k: called.append(1))
    mtg = {"date": D, "venue": V, "first_post_time": _post(1), "n_races": N_RACES}
    assert cli.cmd_pre(conn, force=True, mtg=mtg) is None and called == []


# ------------------------------------------------------------------ scheduler window
@pytest.fixture()
def clock(monkeypatch):
    t = {"now": datetime(2099, 5, 5, 12, 0)}

    def set_(s):
        t["now"] = datetime.strptime(f"{D} {s}", "%Y-%m-%d %H:%M")
    monkeypatch.setattr(cli, "now_hkt", lambda: t["now"])
    monkeypatch.setattr(paper, "_now", lambda: t["now"].isoformat(timespec="seconds"))
    monkeypatch.setattr(cli, "_race_window", _REAL_RACE_WINDOW)
    return set_


@pytest.fixture()
def quiet_tick(monkeypatch):
    """Everything else tick does is stubbed (no HTTP)."""
    monkeypatch.setattr(cli, "cmd_live", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "cmd_post", lambda *a, **k: None)
    monkeypatch.setattr(cli, "cmd_scores", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "cmd_pre", lambda *a, **k: None)
    monkeypatch.setattr(scrape, "next_meeting", lambda *a, **k: None)


def test_tick_window(conn, world, clock, quiet_tick, monkeypatch):
    clock("17:15")
    _pre(conn)
    calls = []
    real = cli.cmd_race_bet
    monkeypatch.setattr(cli, "cmd_race_bet", lambda c, d, v, rn, **k: calls.append(rn) or real(c, d, v, rn))
    clock("19:02"); cli.cmd_tick(conn)               # race 1 19:15: T-13 -> before the window
    assert calls == []
    clock("19:04"); cli.cmd_tick(conn)               # T-11 -> inside
    assert calls == [1]
    clock("19:06"); cli.cmd_tick(conn)               # second tick: already done
    assert calls == [1]
    assert len(conn.execute("SELECT * FROM runs WHERE kind='race_bet'").fetchall()) == 1
    # race 2 (19:45): the loop was down until T-2 -> missed: no bet, pending -> skip, picks 'missed'
    n_tk = len(_tickets(conn))
    clock("19:43"); cli.cmd_tick(conn)
    assert calls == [1] and len(_tickets(conn)) == n_tk
    rows = conn.execute("SELECT * FROM bot_decisions WHERE run_id=? AND race_no=2 AND bot_id NOT IN "
                        "('fixed_combo','pla_parlay')", (PRE,)).fetchall()
    assert rows and all(r["action"] == "skip" and r["reason"] == bots.MISSED_REASON and r["stake"] == 0
                        for r in rows)
    run = conn.execute("SELECT notes FROM runs WHERE run_id='race-2099-05-05-HV-R02'").fetchone()
    assert json.loads(run["notes"])["missed"] is True
    assert all(it["status"] == "missed" for it in _payload(conn)["items"]
               if it["kind"] == "single" and it["race_no"] == 2)
    assert "race_bet_missed" in {i["code"] for i in health.check(conn)}       # Kane is told
    clock("19:44"); cli.cmd_tick(conn)               # nothing more for race 2
    assert len(_tickets(conn)) == n_tk and calls == [1]


def test_tick_ignores_meetings_without_per_race_mode(conn, world, clock, quiet_tick, monkeypatch):
    clock("17:15")
    cli._record_run(conn, "pre-old", "pre_bet", D, V, {"placed": True})     # an old-style T-2h run
    calls = []
    monkeypatch.setattr(cli, "cmd_race_bet", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(cli, "race_bet_missed", lambda *a, **k: calls.append(a))
    clock("19:04"); cli.cmd_tick(conn)
    clock("21:00"); cli.cmd_tick(conn)
    assert calls == []


def test_health_race_bet_missed_and_fix(conn, world, clock):
    clock("17:15")
    _pre(conn)
    clock("19:10")                                    # race 1 in 5 min: still in the window
    assert "race_bet_missed" not in {i["code"] for i in health.check(conn)}
    clock("19:13")                                    # T-2, still pending, no race_bet run
    iss = [i for i in health.check(conn) if i["code"] == "race_bet_missed"]
    assert len(iss) == 1 and iss[0]["race_no"] == 1 and iss[0]["fix"] == "race_bet_missed"
    assert health.fix(conn, iss) == ["race_bet_missed"]
    assert not bots.pending_rows(conn, PRE, 1)
    iss = [i for i in health.check(conn) if i["code"] == "race_bet_missed"]
    assert len(iss) == 1 and iss[0]["fix"] is None   # recorded as missed: still reported (not fixable)
    clock("23:00")
    assert all(i.get("race_no") != 1 for i in health.check(conn) if i["code"] == "race_bet_missed")


def test_no_live_odds_retries_then_decides_flagged(conn, world, clock, monkeypatch):
    """No live odds at race time: retry on later ticks while there is time (no run recorded);
    at the last chance (post − 5 min) decide with the stored odds and flag it stale_odds."""
    _pre(conn)
    monkeypatch.setattr(om, "snapshot_meeting", lambda *a, **k: 0)
    clock(_post(1)[11:].replace(":", ":"))                       # placeholder, set precisely below
    h, m = map(int, _post(1)[11:].split(":"))
    clock(f"{h:02d}:{m - 10:02d}" if m >= 10 else f"{h - 1:02d}:{m + 50:02d}")   # post − 10 min
    assert cli.cmd_race_bet(conn, D, V, 1) is None
    assert not cli._run_exists(conn, "race_bet", D, V)
    assert bots.pending_rows(conn, PRE, 1)                        # still pending, will retry
    clock(f"{h:02d}:{m - 4:02d}" if m >= 4 else f"{h - 1:02d}:{m + 56:02d}")     # post − 4 min: last chance
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes and notes["stale_odds"] is True
    assert not bots.pending_rows(conn, PRE, 1)


# --------------------------------------------------- Codex review round 1 (window / crash / odds)
def _at(clock, minutes_before_post, rn=1):
    h, m = map(int, _post(rn)[11:].split(":"))
    t = h * 60 + m - minutes_before_post
    clock(f"{t // 60:02d}:{t % 60:02d}")


def _race_tickets(conn, rn=1):
    return conn.execute("SELECT * FROM paper_tickets WHERE run_id=?", (cli.race_bet_run_id(D, V, rn),)).fetchall()


def test_direct_call_outside_window_never_bets(conn, world, clock):
    _pre(conn)
    _at(clock, 120)                                     # 2 h before: too early
    assert cli.cmd_race_bet(conn, D, V, 1) is None
    assert not _race_tickets(conn) and bots.pending_rows(conn, PRE, 1)
    _at(clock, -1)                                      # after the off: missed, no bet
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes and notes["missed"] is True
    assert not _race_tickets(conn) and not bots.pending_rows(conn, PRE, 1)


def test_window_closing_during_odds_fetch_means_missed(conn, world, clock, monkeypatch):
    _pre(conn)
    _at(clock, 4)                                       # inside the window...
    real = om.snapshot_meeting

    def slow_snap(*a, **k):                             # ...but the fetch takes us past post − 3
        r = real(*a, **k)
        _at(clock, 1)
        return r
    monkeypatch.setattr(om, "snapshot_meeting", slow_snap)
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes and notes["missed"] is True
    assert not [t for t in _race_tickets(conn) if t["status"] == "open"]


def test_crash_after_tickets_then_window_passes_voids_orphans(conn, world, clock, monkeypatch):
    _pre(conn)
    bal0 = {b["id"]: paper.bankroll(conn, b["id"]) for b in bots.BOTS}
    _at(clock, 8)
    real = bots._replace_race_rows

    def boom(c, *a, **k):            # die right after the first robot that placed tickets
        if [t for t in _race_tickets(c) if t["status"] == "open"]:
            raise RuntimeError("crash after tickets were placed")
        return real(c, *a, **k)
    monkeypatch.setattr(bots, "_replace_race_rows", boom)
    with pytest.raises(RuntimeError):
        cli.cmd_race_bet(conn, D, V, 1)
    assert [t for t in _race_tickets(conn) if t["status"] == "open"]          # orphans exist
    monkeypatch.setattr(bots, "_replace_race_rows", real)
    _at(clock, 2)                                                            # window gone
    notes = cli.race_bet_missed(conn, D, V, 1)
    assert notes["voided_tickets"]
    assert not [t for t in _race_tickets(conn) if t["status"] == "open"]
    for b in bots.BOTS:
        assert paper.bankroll(conn, b["id"]) == pytest.approx(bal0[b["id"]])  # every stake refunded
    pk = _payload(conn)
    assert not [i for i in pk["items"] if int(i.get("race_no") or 0) == 1 and i.get("kind") == "single"
                and i.get("status") in ("pending", "final") and not i.get("no_bet")]


def test_odds_api_error_retries_then_stale_fallback(conn, world, clock, monkeypatch):
    _pre(conn)

    def broken(*a, **k):
        raise om.OddsAPIError("boom")
    monkeypatch.setattr(om, "snapshot_meeting", broken)
    _at(clock, 10)
    assert cli.cmd_race_bet(conn, D, V, 1) is None and not cli._run_exists(conn, "race_bet", D, V)
    _at(clock, 4)
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes["stale_odds"] is True and "OddsAPIError" in notes["odds_error"]


def test_other_pools_without_win_are_not_fresh(conn, world, clock, monkeypatch):
    _pre(conn)

    def qpl_only(c, date, venue, pools=om.DEFAULT_POOLS, race_nos=None, **k):
        c.execute("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?)",
                  (date, venue, race_nos[0], "QPL", "1,2", 5.0, cli.now_hkt().isoformat(timespec="seconds")))
        c.commit()
        return 1
    monkeypatch.setattr(om, "snapshot_meeting", qpl_only)
    _at(clock, 10)
    assert cli.cmd_race_bet(conn, D, V, 1) is None                 # no fresh WIN: wait
    _at(clock, 4)
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes["stale_odds"] is True and notes["win_rows"] == 0


def test_fresh_win_with_old_pla_is_not_fresh(conn, world, clock, monkeypatch):
    """Codex round 2: a fresh WIN snapshot must not make the race 'fresh' while PLA (offered,
    stored from T−2h) did not come back - retry until post − 5, then decide flagged stale_odds."""
    _pre(conn)
    conn.execute("INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?,?)",
                 (D, V, 1, "PLA", "1", 1.4, f"{D}T17:15:00"))      # old PLA book from T−2h
    conn.commit()

    def win_only(c, date, venue, pools=om.DEFAULT_POOLS, race_nos=None, **k):
        if "WIN" not in pools:
            return 0
        c.execute("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?)",
                  (date, venue, race_nos[0], "WIN", "1", 2.0, cli.now_hkt().isoformat(timespec="seconds")))
        c.commit()
        return 8
    monkeypatch.setattr(om, "snapshot_meeting", win_only)
    _at(clock, 10)
    assert cli.cmd_race_bet(conn, D, V, 1) is None                 # PLA not fresh: wait
    assert not _race_tickets(conn)
    _at(clock, 4)
    notes = cli.cmd_race_bet(conn, D, V, 1)
    assert notes["stale_odds"] is True and "PLA" in notes["stale_pools"] and "WIN" in notes["fresh_pools"]
