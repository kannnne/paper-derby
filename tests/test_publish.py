"""Tests for hkjc.publish (static dashboard export + git publishing).

A small synthetic database is built from scratch (one finished meeting with results and
one far-future meeting with a racecard only), robots' decisions / paper tickets are
written through the real paper module, then the exported JSON is validated against the
SPEC_V2 shapes.  Nothing touches data/hkjc.db, site/ or data/publish.json.
"""
import json
import re
import sqlite3
from pathlib import Path

import pytest

from hkjc import db, paper, publish

PAST, FUT = "2026-01-01", "2099-01-06"


# ------------------------------------------------------------------ fixture DB
def _rid(date, venue, rn):
    return f"{date.replace('-', '')}-{venue}-{rn:02d}"


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "pub.db")
    c.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)", (PAST, "ST", 2, f"{PAST}T13:00", "results", "x"))
    c.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)", (FUT, "HV", 1, f"{FUT}T19:15", "card", "x"))
    names = {1: "快樂之星", 2: "金色閃電", 3: "風之子", 4: "勇往直前"}
    for rn in (1, 2):
        rid = _rid(PAST, "ST", rn)
        c.execute("INSERT INTO races (race_id, date, venue, race_no, race_class, rating_band, distance, surface, "
                  "course, going, race_name, post_time, field_size) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (rid, PAST, "ST", rn, "第四班", "60-40", 1200, "TURF", "A", "好地", f"測試讓賽{rn}",
                   f"{PAST}T{12 + rn}:00", 4))
        for h in range(1, 5):
            fin = h if rn == 1 else 5 - h
            c.execute("INSERT INTO runners (race_id, horse_no, horse_name, jockey_name, trainer_name, act_weight, "
                      "draw, finish_pos, finish_status, lbw, win_odds, scratched) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, h, names[h], "甲騎師", "乙練馬師", 120 + h, h, fin, str(fin), "-", 2.0 * h + 1, 0))
            c.execute("INSERT INTO racecard (date, venue, race_no, horse_no, horse_name, jockey_name, trainer_name, "
                      "weight, draw, scratched) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (PAST, "ST", rn, h, names[h], "甲騎師", "乙練馬師", 120 + h, h, 0))
        w = 1 if rn == 1 else 4
        o = [1, 2, 3] if rn == 1 else [4, 3, 2]
        divs = [("WIN", str(w), 35.0), ("PLA", str(o[0]), 14.0), ("PLA", str(o[1]), 18.5), ("PLA", str(o[2]), 21.0),
                ("QIN", ",".join(map(str, sorted(o[:2]))), 60.0), ("QPL", ",".join(map(str, sorted(o[:2]))), 25.0),
                ("TRI", ",".join(map(str, sorted(o))), 120.0)]
        c.executemany("INSERT INTO dividends VALUES (?,?,?,?)", [(rid, p, k, d) for p, k, d in divs])
    c.execute("INSERT INTO races (race_id, date, venue, race_no, race_class, distance, surface, course, race_name, "
              "post_time) VALUES (?,?,?,?,?,?,?,?,?,?)",
              (_rid(FUT, "HV", 1), FUT, "HV", 1, "第五班", 1650, "TURF", "C", "未來讓賽", f"{FUT}T19:15"))
    for h in range(1, 4):
        c.execute("INSERT INTO racecard (date, venue, race_no, horse_no, horse_name, jockey_name, trainer_name, "
                  "weight, draw, scratched) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (FUT, "HV", 1, h, names[h], "丙騎師", "丁練馬師", 130 - h, h, 1 if h == 3 else 0))
    c.commit()

    # pre-meeting runs: predictions, scores, decisions, tickets
    for date, venue, run, races in ((PAST, "ST", "pre-past", (1, 2)), (FUT, "HV", "pre-fut", (1,))):
        c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)", (run, "pre_bet", date, venue, f"{date}T10:00:00", "{}"))
        for rn in races:
            rid = _rid(date, venue, rn)
            hs = (1, 2, 3, 4) if venue == "ST" else (1, 2)
            pw = {1: .4, 2: .3, 3: .2, 4: .1} if venue == "ST" else {1: .6, 2: .4}
            for h in hs:
                c.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                          (run, rid, h, pw[h], min(.95, pw[h] * 2.2), "test-model-v1", f"{date}T10:00:00"))
                subs = {k: 50 + 10 * (len(hs) - h) for k in publish.SUB_SCORE_KEYS}
                subs["market"] = None
                c.execute("INSERT INTO horse_scores VALUES (?,?,?,?,?,?,?,?)",
                          (run, rid, h, 90 - 15 * h, h, json.dumps(subs), json.dumps(["近3仗平均第2.3名", "檔位1好檔"],
                                                                                    ensure_ascii=False),
                           f"{date}T10:00:00"))
    c.commit()
    t1 = paper.place_tickets(c, "pre-past", PAST, "ST",
                             [{"strategy": "PLA", "ticket_no": 1, "pool": "PLA", "unit_stake": 10.0,
                               "legs": [{"race_no": 1, "selections": [1, 4], "pool": "PLA"}]}],
                             replace=False, bot_id="place_only", reason="上名率高")
    t2 = paper.place_tickets(c, "pre-past", PAST, "ST",
                             [{"strategy": "PLA_2x3", "ticket_no": 1, "pool": "PLA", "spec": "2x3", "unit_stake": 10.0,
                               "legs": [{"race_no": 1, "selections": [1], "pool": "PLA"},
                                        {"race_no": 2, "selections": [4], "pool": "PLA"}]}],
                             replace=False, bot_id="pla_parlay", reason="兩場最穩")
    t3 = paper.place_tickets(c, "pre-fut", FUT, "HV",
                             [{"strategy": "WIN", "ticket_no": 1, "pool": "WIN", "unit_stake": 10.0,
                               "legs": [{"race_no": 1, "selections": [1], "pool": "WIN"}]}],
                             replace=False, bot_id="win_value", reason="勝率×賠率夠高")
    dec = [
        ("pre-past", "place_only", PAST, "ST", 1, "bet", "PLA 位置", [1, 4], 20.0, "上名率高", t1[0]),
        ("pre-past", "place_only", PAST, "ST", 2, "skip", "PLA 位置", [], 0.0, "冇馬過門檻", None),
        ("pre-past", "pla_parlay", PAST, "ST", 0, "bet", "PLA 2x3", [{"race_no": 1, "selections": [1]},
                                                                     {"race_no": 2, "selections": [4]}],
         30.0, "兩場最穩", t2[0]),
        ("pre-past", "pla_parlay", PAST, "ST", 1, "bet", "過關腳", [1], 0.0, "過關第1關", t2[0]),
        ("pre-past", "pla_parlay", PAST, "ST", 2, "bet", "過關腳", [4], 0.0, "過關第2關", t2[0]),
        ("pre-past", "monkey", PAST, "ST", 1, "skip", "PLA 隨機", [], 0.0, "資金不足", None),
        ("pre-fut", "win_value", FUT, "HV", 1, "bet", "WIN 獨贏", [1], 10.0, "勝率×賠率夠高", t3[0]),
        ("pre-fut", "qin_banker", FUT, "HV", 1, "skip", "QIN 膽1x腳2", [], 0.0, "冇明顯首選", None),
    ]
    c.executemany("INSERT INTO bot_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  [(*d[:7], json.dumps(d[7], ensure_ascii=False), d[8], d[9], d[10], "2026-01-01T10:00:00")
                   for d in dec])
    c.commit()
    assert paper.settle(c, PAST) == 2
    yield c
    c.close()


# ------------------------------------------------------------------ minimal SPEC_V2 validator
NUM = (int, float)
OPT_NUM = (int, float, type(None))
OPT_STR = (str, type(None))
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _req(obj, spec, where):
    assert isinstance(obj, dict), f"{where}: not an object"
    for k, typ in spec.items():
        assert k in obj, f"{where}: missing {k}"
        assert isinstance(obj[k], typ), f"{where}.{k}: {type(obj[k]).__name__} not {typ}"


def validate_index(d):
    _req(d, {"updated_at": str, "disclaimer": str, "next_meeting": (dict, type(None)), "meetings": list,
             "bots": list}, "index")
    assert "1834 633" in d["disclaimer"] and "非投注建議" in d["disclaimer"]
    if d["next_meeting"]:
        _req(d["next_meeting"], {"date": str, "venue": str, "first_post_time": OPT_STR}, "next_meeting")
    for m in d["meetings"]:
        _req(m, {"date": str, "venue": str, "status": str, "file": str}, "meetings[]")
        assert DATE_RE.match(m["date"]) and m["status"] in ("pre", "results", "abandoned")
        assert m["file"] == f"data/meetings/{m['date']}_{m['venue']}.json"
    for b in d["bots"]:
        _req(b, {"id": str, "name": str, "style": str, "description": str, "bankroll": NUM, "start_bankroll": NUM,
                 "live": dict}, "bots[]")
        assert "backtest" not in b, "backtest figures must not be published (live paper results only)"
        _req(b["live"], {"n_meetings": int, "stake": NUM, "returns": NUM, "profit": NUM, "roi": OPT_NUM,
                         "hit_rate": OPT_NUM}, "bots[].live")


def validate_meeting(d):
    _req(d, {"date": str, "venue": str, "venue_zh": str, "status": str, "generated_at": str,
             "odds_snapshot_at": OPT_STR, "model_version": OPT_STR, "races": list, "bots": list}, "meeting")
    for r in d["races"]:
        _req(r, {"race_no": int, "post_time": OPT_STR, "race_class": OPT_STR, "distance": (int, type(None)),
                 "surface": OPT_STR, "course": OPT_STR, "going": OPT_STR, "race_name": OPT_STR, "horses": list,
                 "decisions": list, "result": (dict, type(None))}, "race")
        for h in r["horses"]:
            _req(h, {"horse_no": int, "horse_name": str, "jockey": str, "trainer": str, "draw": (int, type(None)),
                     "weight": (int, type(None)), "score": OPT_NUM, "rank": (int, type(None)), "p_win": OPT_NUM,
                     "p_place": OPT_NUM, "win_odds": OPT_NUM, "pla_odds": OPT_NUM, "sub_scores": dict,
                     "reasons": list, "scratched": bool, "result": (dict, type(None))}, "horse")
            assert set(h["sub_scores"]) == set(publish.SUB_SCORE_KEYS)
            assert all(isinstance(x, str) for x in h["reasons"])
            if h["result"] is not None:
                _req(h["result"], {"finish_pos": (int, type(None)), "final_win_odds": OPT_NUM}, "horse.result")
        for x in r["decisions"]:
            _req(x, {"bot_id": str, "action": str, "structure": str, "selections": (list, dict), "stake": NUM,
                     "reason": str, "payout": OPT_NUM, "profit": OPT_NUM}, "decision")
            assert x["action"] in ("bet", "skip")
        if r["result"] is not None:
            _req(r["result"], {"placings": list, "dividends": dict}, "race.result")
            for pool, rows in r["result"]["dividends"].items():
                assert pool in publish.RESULT_POOLS
                assert all(isinstance(c, str) and isinstance(v, OPT_NUM) for c, v in rows)
    assert "note" not in d, "the 賽後補做 notice is no longer published"
    for b in d["bots"]:
        _req(b, {"bot_id": str, "stake": NUM, "payout": OPT_NUM, "profit": OPT_NUM, "tickets": list,
                 "skips": list}, "meeting.bots")
        for t in b["tickets"]:
            _req(t, {"strategy": str, "legs": list, "total_stake": NUM, "payout": OPT_NUM, "profit": OPT_NUM,
                     "reason": str, "status": str, "race_no": int, "structure": str, "selections": (list, dict),
                     "leg_hits": list, "hit": (bool, type(None))}, "ticket")
            assert len(t["leg_hits"]) == len(t["legs"])
            assert (t["payout"] is None) == (t["status"] != "settled")
        for x in b["skips"]:
            _req(x, {"race_no": int, "structure": str, "reason": str}, "skip")
    if "picks" in d:
        validate_picks(d["picks"])


def validate_picks(p):
    """SPEC_V2 「今日精選（第三版）」 shape - exactly what site/app.js reads."""
    if p is None:
        return
    _req(p, {"title": str, "generated_at": str, "method_note": str, "odds_note": str, "total_stake": NUM,
             "n_items": int, "total_payout": OPT_NUM, "total_profit": OPT_NUM, "items": list,
             "banker": (dict, type(None))}, "picks")
    assert p["title"] == "今日精選"
    assert p["n_items"] == sum(1 for i in p["items"] if i["grade"] == "精選")
    assert p["total_stake"] == pytest.approx(sum(i["stake"] for i in p["items"] if i["grade"] == "精選"))
    if not p["n_items"]:
        assert "今日冇玩法過門檻，冇精選" in p["method_note"]
        assert sum(1 for i in p["items"] if i["grade"] == "參考") <= 3
    for it in p["items"]:
        _req(it, {"rank": int, "kind": str, "race_no": int, "races": list, "post_time": OPT_STR, "pool": str,
                  "pool_zh": str, "structure_zh": str, "legs": list, "n_combos": int, "unit_stake": NUM,
                  "stake": NUM, "hit_prob": NUM, "exp_roi": NUM, "est_return_if_hit": OPT_NUM,
                  "odds_source": str, "grade": str, "reason": str, "result": (dict, type(None))}, "pick item")
        assert it["kind"] in ("single", "multi_race", "allup")
        assert it["pool"] in ("WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT", "DBL", "TBL")
        assert it["odds_source"] in ("即時賠率", "估計賠率") and it["grade"] in ("精選", "參考")
        assert 0 <= it["hit_prob"] <= 1 and it["stake"] == pytest.approx(it["n_combos"] * it["unit_stake"])
        assert it["race_no"] == max(it["races"])
        for k in ("ticket", "legs_raw", "key"):
            assert k not in it, f"internal field {k} published"
        if it["grade"] == "參考":
            assert "未過門檻" in it["reason"]
        assert "上名率" not in it["reason"]
        for l in it["legs"]:
            _req(l, {"race_no": int, "bankers": list, "selections": list, "positions": (list, type(None))}, "pick leg")
            for h in l["bankers"] + l["selections"]:
                _req(h, {"horse_no": int, "horse_name": str, "score": OPT_NUM}, "pick horse")
            if l["positions"] is not None:
                assert all(isinstance(x, list) and all(isinstance(h, int) for h in x) for x in l["positions"])
        if it["result"] is not None:
            _req(it["result"], {"hit": bool, "payout": NUM, "profit": NUM, "detail": str}, "pick result")
    if p["banker"]:
        _req(p["banker"], {"race_no": int, "horse_no": int, "horse_name": str, "p_place": NUM, "reason": str,
                           "result": (dict, type(None))}, "banker")
        if p["banker"]["result"]:
            _req(p["banker"]["result"], {"finish_pos": (int, type(None)), "placed": (bool, type(None))}, "banker.result")


def validate_history(d):
    _req(d, {"updated_at": str, "bots": list}, "history")
    for b in d["bots"]:
        _req(b, {"bot_id": str, "live": list}, "history.bots")
        assert "backtest" not in b, "backtest curves must not be published"
        for p in b["live"]:
            _req(p, {"date": str, "venue": str, "stake": NUM, "payout": NUM, "profit": NUM, "bankroll": NUM}, "live pt")


def _load(p):
    txt = Path(p).read_text(encoding="utf-8")
    assert "NaN" not in txt and "Infinity" not in txt
    return json.loads(txt)


# ------------------------------------------------------------------ tests
def test_export_schema(conn, tmp_path):
    out = tmp_path / "site"
    files = publish.export_site(conn, out)
    names = sorted(p.name for p in files if p.parent.name != "bots")
    assert names == sorted(["index.json", "bot_history.json", f"{PAST}_ST.json", f"{FUT}_HV.json"])
    assert sorted(p.stem for p in files if p.parent.name == "bots") == sorted(b["id"] for b in publish.bot_meta())
    idx = _load(out / "data" / "index.json")
    validate_index(idx)
    assert idx["next_meeting"]["date"] == FUT and idx["next_meeting"]["venue"] == "HV"
    assert [m["status"] for m in idx["meetings"]] == ["pre", "results"]      # newest first
    assert [b["id"] for b in idx["bots"]][:10] == [b["id"] for b in publish.bot_meta()][:10]
    for m in idx["meetings"]:
        validate_meeting(_load(out / m["file"]))
    validate_history(_load(out / "data" / "bot_history.json"))

    bots = {b["id"]: b for b in idx["bots"]}
    po = bots["place_only"]
    # PLA 1 & 4 in race 1: horse 1 placed (14.0), horse 4 did not -> $14 back on $20
    assert po["live"]["stake"] == 20.0 and po["live"]["returns"] == 14.0 and po["live"]["profit"] == -6.0
    assert po["bankroll"] == 9994.0 and po["live"]["n_meetings"] == 1 and po["live"]["hit_rate"] == 1.0
    assert bots["win_value"]["live"]["open_stake"] == 10.0 and bots["win_value"]["live"]["stake"] == 0


def test_export_meeting_content(conn, tmp_path):
    out = tmp_path / "site"
    publish.export_site(conn, out)
    m = _load(out / "data" / "meetings" / f"{PAST}_ST.json")
    assert m["status"] == "results" and m["model_version"] == "test-model-v1"
    r1 = m["races"][0]
    assert [h["horse_no"] for h in r1["horses"]] == [1, 2, 3, 4]
    assert r1["horses"][0]["score"] == 75.0 and r1["horses"][0]["rank"] == 1
    assert r1["horses"][0]["result"]["finish_pos"] == 1 and r1["horses"][0]["sub_scores"]["market"] is None
    assert [p["horse_no"] for p in r1["result"]["placings"][:3]] == [1, 2, 3]
    assert r1["result"]["dividends"]["WIN"] == [["1", 35.0]]
    d = {(x["bot_id"], x["structure"]): x for x in r1["decisions"]}
    assert d[("place_only", "PLA 位置")]["payout"] == 14.0 and d[("place_only", "PLA 位置")]["profit"] == -6.0
    leg = d[("pla_parlay", "過關腳")]
    assert leg["parlay_leg"] is True and leg["payout"] is None          # P&L only on the race-0 row
    md = m["meeting_decisions"]
    assert len(md) == 1 and md[0]["bot_id"] == "pla_parlay" and md[0]["profit"] is not None
    # all-up 2x3 PLA: 1 (14.0) x 4 (14.0) -> doubles 19.6 + singles 14 + 14 = $47.6 on $30
    assert md[0]["payout"] == pytest.approx(47.6)
    bots = {b["bot_id"]: b for b in m["bots"]}
    assert bots["pla_parlay"]["stake"] == 30.0 and bots["pla_parlay"]["profit"] == pytest.approx(17.6)
    assert bots["monkey"]["stake"] == 0 and bots["monkey"]["tickets"] == []

    f = _load(out / "data" / "meetings" / f"{FUT}_HV.json")
    assert f["status"] == "pre" and f["races"][0]["result"] is None
    hs = {h["horse_no"]: h for h in f["races"][0]["horses"]}
    assert hs[3]["scratched"] is True and hs[3]["score"] is None
    assert hs[1]["win_odds"] is None and hs[1]["p_win"] == 0.6
    fb = {b["bot_id"]: b for b in f["bots"]}
    assert fb["win_value"]["payout"] is None and fb["win_value"]["tickets"][0]["status"] == "open"

    hist = _load(out / "data" / "bot_history.json")
    h = {b["bot_id"]: b for b in hist["bots"]}
    assert h["place_only"]["live"] == [{"date": PAST, "venue": "ST", "stake": 20.0, "payout": 14.0,
                                        "profit": -6.0, "bankroll": 9994.0}]
    assert "backtest" not in h["place_only"]


def test_latest_run_only(conn, tmp_path):
    # an older pre run for the past meeting must not be exported
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)", ("pre-old", "pre_bet", PAST, "ST", f"{PAST}T08:00:00", "{}"))
    conn.execute("INSERT INTO horse_scores VALUES (?,?,?,?,?,?,?,?)",
                 ("pre-old", _rid(PAST, "ST", 1), 1, 1.0, 4, "{}", "[]", f"{PAST}T08:00:00"))
    conn.commit()
    out = tmp_path / "s"
    publish.export_site(conn, out)
    m = _load(out / "data" / "meetings" / f"{PAST}_ST.json")
    assert m["scores_run_id"] == "pre-past" and m["races"][0]["horses"][0]["score"] == 75.0


def test_privacy_scan(conn, tmp_path):
    out = tmp_path / "site"
    files = publish.export_site(conn, out)
    assert files
    for p in files:
        txt = p.read_text(encoding="utf-8")
        for bad in ("@", "/Users/", Path.home().name, str(Path.home()), str(tmp_path)):
            assert bad not in txt, f"{p.name} contains {bad!r}"


def test_stale_meeting_files_removed(conn, tmp_path):
    out = tmp_path / "site"
    (out / "data" / "meetings").mkdir(parents=True)
    stale = out / "data" / "meetings" / "2000-01-01_ST.json"
    stale.write_text("{}")
    publish.export_site(conn, out)
    assert not stale.exists()


def test_publish_disabled_commits_without_push(conn, tmp_path):
    cfg = tmp_path / "publish.json"
    cfg.write_text(json.dumps({"enabled": False, "remote": "", "branch": "main"}))
    site = tmp_path / "site"
    res = publish.publish(conn, "測試", site_dir=site, config_path=cfg)
    assert res["status"] == "exported" and res["committed"] and not res["pushed"] and res["error"] is None
    assert (site / ".git").exists() and "data/index.json" in res["files"]
    # nothing changed -> no new commit, still fine
    res2 = publish.publish(conn, site_dir=site, config_path=cfg)
    assert res2["status"] == "exported" and res2["pushed"] is False


def test_publish_missing_config_created_disabled(conn, tmp_path):
    cfg = tmp_path / "sub" / "publish.json"
    res = publish.publish(conn, site_dir=tmp_path / "site", config_path=cfg)
    assert res["status"] == "exported" and not res["pushed"]
    assert json.loads(cfg.read_text()) == {"enabled": False, "remote": "", "branch": "main"}


def test_publish_bogus_remote_fails_gracefully(conn, tmp_path):
    cfg = tmp_path / "publish.json"
    cfg.write_text(json.dumps({"enabled": True, "remote": str(tmp_path / "does-not-exist.git"), "branch": "main"}))
    res = publish.publish(conn, site_dir=tmp_path / "site", config_path=cfg)
    assert res["status"] == "push_failed" and res["committed"] and not res["pushed"] and res["error"]


def test_publish_push_to_local_bare_repo(conn, tmp_path):
    import subprocess
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    cfg = tmp_path / "publish.json"
    cfg.write_text(json.dumps({"enabled": True, "remote": str(bare), "branch": "main"}))
    res = publish.publish(conn, site_dir=tmp_path / "site", config_path=cfg)
    assert res["status"] == "pushed" and res["pushed"]
    log = subprocess.run(["git", "-C", str(bare), "log", "--format=%an <%ae>", "main"], capture_output=True, text=True)
    assert log.stdout.strip().startswith(publish.DEFAULT_AUTHOR[0])


def test_publish_never_raises_on_export_error(tmp_path):
    c = sqlite3.connect(":memory:")
    c.close()
    res = publish.publish(c, site_dir=tmp_path / "site", config_path=tmp_path / "p.json")
    assert res["status"] == "export_failed" and res["error"]


def test_no_backtest_published(conn, tmp_path):
    """The public dashboard shows live paper results only: no exported file carries backtest
    figures or curves (data/bot_backtest.json is internal research output)."""
    assert not hasattr(publish, "load_bot_backtest")
    out = tmp_path / "site"
    files = publish.export_site(conn, out)
    for p in files:
        txt = p.read_text(encoding="utf-8")
        assert '"backtest"' not in txt and "oos_roi" not in txt and "roi_ci" not in txt, p.name
        assert "回測" not in txt, p.name


# ------------------------------------------------------------------ review fixes
def test_decisions_each_bots_latest_run(conn, tmp_path):
    """A robot re-decided by a later run must not hide the other robots' decisions."""
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)", ("pre-later", "pre_bet", PAST, "ST", f"{PAST}T11:00:00", "{}"))
    conn.execute("INSERT INTO bot_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("pre-later", "qin_banker", PAST, "ST", 1, "skip", "QIN 膽1x腳2", "[]", 0.0, "首選勝率只得20%", None,
                  f"{PAST}T11:00:00"))
    conn.commit()
    out = tmp_path / "s"
    publish.export_site(conn, out)
    m = _load(out / "data" / "meetings" / f"{PAST}_ST.json")
    ids = {d["bot_id"] for d in m["races"][0]["decisions"]}
    assert {"place_only", "pla_parlay", "monkey", "qin_banker"} <= ids
    assert len(m["meeting_decisions"]) == 1


def test_unchanged_meeting_file_not_rewritten(conn, tmp_path, monkeypatch):
    out = tmp_path / "s"
    monkeypatch.setattr(publish, "_now", lambda: "2026-01-01T12:00:00")
    publish.export_site(conn, out)
    f = out / "data" / "meetings" / f"{PAST}_ST.json"
    before = f.read_bytes()
    monkeypatch.setattr(publish, "_now", lambda: "2026-01-02T12:00:00")
    publish.export_site(conn, out)
    assert f.read_bytes() == before                       # generated_at alone is not a change
    assert _load(out / "data" / "index.json")["updated_at"] == "2026-01-02T12:00:00"
    conn.execute("UPDATE horse_scores SET score=80 WHERE run_id='pre-past' AND horse_no=1")
    conn.commit()
    publish.export_site(conn, out)
    m = _load(f)
    assert m["generated_at"] == "2026-01-02T12:00:00" and m["races"][0]["horses"][0]["score"] == 80.0


def test_bot_ticket_files(conn, tmp_path):
    out = tmp_path / "s"
    publish.export_site(conn, out)
    po = _load(out / "data" / "bots" / "place_only.json")
    assert po["n_tickets"] == 1 and po["tickets"][0]["date"] == PAST and po["tickets"][0]["venue_zh"] == "沙田"
    assert po["tickets"][0]["payout"] == 14.0 and po["tickets"][0]["reason"] == "入三甲機率高"
    wv = _load(out / "data" / "bots" / "win_value.json")
    assert wv["tickets"][0]["status"] == "open" and wv["tickets"][0]["payout"] is None
    assert _load(out / "data" / "bots" / "monkey.json")["tickets"] == []
    stale = out / "data" / "bots" / "old_bot.json"
    stale.write_text("{}")
    publish.export_site(conn, out)
    assert not stale.exists()


def test_commit_identity_ignores_personal_git_env(conn, tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Real Person")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "real.person@example.com")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "real.person@example.com")
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    cfg = tmp_path / "publish.json"
    cfg.write_text(json.dumps({"enabled": True, "remote": str(bare), "branch": "main"}))
    assert publish.publish(conn, site_dir=tmp_path / "site", config_path=cfg)["status"] == "pushed"
    log = subprocess.run(["git", "-C", str(bare), "log", "--format=%an|%ae|%cn|%ce", "main"],
                         capture_output=True, text=True).stdout
    assert "real.person" not in log and "Real Person" not in log
    assert log.strip() == "|".join([*publish.DEFAULT_AUTHOR, *publish.DEFAULT_AUTHOR])
    files = subprocess.run(["git", "-C", str(bare), "ls-tree", "-r", "--name-only", "main"],
                           capture_output=True, text=True).stdout.split()
    assert files and all(f.startswith("data/") for f in files)


def test_site_static_files_review_fixes():
    """site/ code: robot page reads data/bots/<id>.json, tables in .scroll-x can't widen the
    phone layout, Finder junk is never committed, assets are cache-busted."""
    site = Path(publish.__file__).resolve().parent.parent / "site"
    if not (site / "app.js").exists():
        pytest.skip("no site/")
    js = (site / "app.js").read_text(encoding="utf-8")
    assert '"data/bots/"' in js
    css = (site / "style.css").read_text(encoding="utf-8")
    rule = re.search(r"\.scroll-x \{[^}]*\}", css).group(0)
    assert "position: relative" in rule
    html = (site / "index.html").read_text(encoding="utf-8")
    assert re.search(r'app\.js\?v=\d+', html) and re.search(r'style\.css\?v=\d+', html)
    # live paper results only: no backtest display anywhere on the site
    for txt in (js, html):
        assert "backtest" not in txt and "回測" not in txt and "樣本外" not in txt
    assert ".DS_Store" in (site / ".gitignore").read_text(encoding="utf-8")
    for txt in (js, css, html):
        assert Path.home().name not in txt and "/Users/" not in txt


# ------------------------------------------------------------------ 今日精選 (picks)
OPEN_TH = {f: {"tau": -1.0, "h": 0.0} for f in ("WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT",
                                                 "DBL", "TBL", "ALLUP")}


def _pred(c, date=PAST, venue="ST", run="pre-past"):
    import pandas as pd
    return pd.read_sql_query(
        "SELECT p.race_id, CAST(substr(p.race_id, 13, 2) AS INTEGER) AS race_no, p.horse_no, p.p_win, p.p_place, "
        "u.win_odds, u.horse_name FROM predictions p LEFT JOIN runners u ON u.race_id=p.race_id AND "
        "u.horse_no=p.horse_no WHERE p.run_id=? ORDER BY race_no, p.horse_no", c, params=(run,))


def _run_daily_picks(c, th=None, run="pre-past"):
    from hkjc import bots, picks
    pred = _pred(c)
    pk = picks.make_picks(pred, None, {"thresholds": th or OPEN_TH, "max_items": 3})
    picks.save_picks(c, run, PAST, "ST", pk)
    decs = bots.run_bots(c, run, PAST, "ST", pred, None, bots=["daily_picks"], picks=pk)
    return pk, decs


def test_picks_from_daily_picks_run_and_settle(conn, tmp_path):
    pk, decs = _run_daily_picks(conn)
    sel = [i for i in pk["items"] if i["grade"] == "精選"]
    assert 1 <= len(sel) <= 3
    bets_ = [d for d in decs if d["action"] == "bet"]
    assert len(bets_) == len(sel) and {d["race_no"] for d in decs} >= {0, 1, 2}
    out = tmp_path / "s"
    publish.export_site(conn, out)
    m = _load(out / "data" / "meetings" / f"{PAST}_ST.json")
    p = m["picks"]
    validate_picks(p)
    assert [i["rank"] for i in p["items"]] == [i["rank"] for i in pk["items"]]
    assert p["total_payout"] is None and all(i["result"] is None for i in p["items"])   # tickets still open
    assert p["items"][0]["legs"][0]["selections"][0]["horse_name"]                   # names from the card
    assert p["banker"]["race_no"] in (1, 2) and p["banker"]["result"] is not None
    assert {mm["date"]: mm["has_picks"] for mm in _load(out / "data" / "index.json")["meetings"]}[PAST] is True
    # daily_picks robot's card lists exactly these tickets
    db_ = {b["bot_id"]: b for b in m["bots"]}["daily_picks"]
    assert db_["n_tickets"] == len(sel)
    paper.settle(conn, PAST)
    publish.export_site(conn, out)
    p = _load(out / "data" / "meetings" / f"{PAST}_ST.json")["picks"]
    validate_picks(p)
    rows = conn.execute("SELECT payout, profit FROM paper_tickets WHERE bot_id='daily_picks' ORDER BY ticket_no").fetchall()
    got = [i["result"] for i in p["items"] if i["grade"] == "精選"]
    assert [g["payout"] for g in got] == pytest.approx([r[0] for r in rows])
    assert all(g["hypothetical"] is False for g in got)
    assert p["total_payout"] == pytest.approx(sum(r[0] for r in rows))
    assert p["total_profit"] == pytest.approx(p["total_payout"] - p["total_stake"])


def test_picks_reference_items_hypothetical_results(conn, tmp_path):
    from hkjc import picks
    pk, decs = _run_daily_picks(conn, th={f: None for f in OPEN_TH})
    assert pk["n_items"] == 0 and pk["items"] and all(i["grade"] == "參考" for i in pk["items"])
    assert "今日冇玩法過門檻，冇精選" in pk["method_note"]
    assert not [d for d in decs if d["action"] == "bet"]
    assert all("唔買" in d["reason"] for d in decs)
    out = tmp_path / "s"
    publish.export_site(conn, out)
    p = _load(out / "data" / "meetings" / f"{PAST}_ST.json")["picks"]
    validate_picks(p)
    for it, raw in zip(p["items"], pk["items"]):
        assert it["result"]["hypothetical"] is True and "假設落注" in it["result"]["detail"]
        assert it["result"]["payout"] == pytest.approx(picks.settle_item(raw, picks.DBData(conn, PAST, "ST"))["payout"])
    assert p["total_payout"] == 0.0 and p["total_stake"] == 0


def test_picks_null_without_stored_picks(conn, tmp_path):
    out = tmp_path / "s"
    publish.export_site(conn, out)
    for f in ("meetings/%s_ST.json" % PAST, "meetings/%s_HV.json" % FUT):
        assert _load(out / "data" / f)["picks"] is None
    assert not any(m["has_picks"] for m in _load(out / "data" / "index.json")["meetings"])


def test_picks_latest_daily_picks_run_wins(conn, tmp_path):
    from hkjc import picks
    _run_daily_picks(conn, run="pre-a")
    pk_b, _ = _run_daily_picks(conn, th={f: None for f in OPEN_TH}, run="pre-b")
    conn.execute("UPDATE picks SET created_at='2000-01-01T00:00:00' WHERE run_id='pre-b'")  # older storage time
    conn.commit()
    m = publish.export_meeting(conn, PAST, "ST")
    assert m["picks"]["n_items"] == 0           # the robot's latest decisions are pre-b's
    assert picks.load_picks(conn, PAST, "ST")["n_items"] > 0


def test_pre_report_picks_md(conn):
    from hkjc import pipeline
    pk, _ = _run_daily_picks(conn)
    md = pipeline._picks_md(pk)
    assert md.startswith("## 今日精選") and "精選" in md and "全日最穩" in md and "入三甲機率" in md
    assert "| 1 | 精選 |" in md
    pk0, _ = _run_daily_picks(conn, th={f: None for f in OPEN_TH}, run="pre-z")
    md0 = pipeline._picks_md(pk0)
    assert "今日冇精選" in md0 and "參考" in md0
    assert pipeline._picks_md(None).startswith("## 今日精選")


def test_bots_grouped_tickets_and_skips(conn, tmp_path):
    out = tmp_path / "s"
    publish.export_site(conn, out)
    m = _load(out / "data" / "meetings" / f"{PAST}_ST.json")
    bots = {b["bot_id"]: b for b in m["bots"]}
    po = bots["place_only"]
    assert po["tickets"][0]["race_no"] == 1 and po["tickets"][0]["structure"] == "PLA 位置"
    assert po["tickets"][0]["selections"] == [1, 4] and po["tickets"][0]["leg_hits"] == [True]
    assert po["tickets"][0]["hit"] is True
    assert po["skips"] == [{"race_no": 2, "structure": "PLA 位置", "reason": "冇馬過門檻"}]
    pp = bots["pla_parlay"]["tickets"][0]
    assert pp["race_no"] == 0 and pp["structure"] == "PLA 2x3" and pp["leg_hits"] == [True, True]
    assert bots["monkey"]["skips"][0]["reason"] == "資金不足"


def test_reason_text_renamed():
    assert publish._reason_txt("（注碼計喺過關飛，見第0場）") == "（注碼計喺過關飛，見「過關決定」）"
    assert publish._reason_txt("今日冇過關飛（原因見第0場跨場決定；x）") == "今日冇過關飛（原因見「過關決定」；x）"
