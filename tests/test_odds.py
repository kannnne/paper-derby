"""Tests for hkjc.odds (offline, using real responses saved on 2026-10-19).

Live checks run only with HKJC_LIVE=1 (they hit info.cld.hkjc.com)."""
import json
import os
from pathlib import Path

import pytest

from hkjc import db, odds

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class FakeResp:
    def __init__(self, data):
        self._d = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._d


class FakeSession:
    """Serves fixtures; records request bodies."""

    def __init__(self, by_race, meeting=None):
        self.by_race, self.meeting, self.bodies = by_race, meeting, []

    def post(self, url, json=None, headers=None, timeout=None):
        assert url == odds.ENDPOINT
        assert headers["Origin"] == "https://bet.hkjc.com"
        self.bodies.append(json)
        if json["operationName"] == "raceMeetings":
            return FakeResp(self.meeting or {"data": {"raceMeetings": []}})
        rn = json["variables"]["raceNo"]
        return FakeResp(self.by_race.get(rn, {"data": {"raceMeetings": [{"pmPools": []}]}}))


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    monkeypatch.setattr(odds, "MIN_INTERVAL_S", 0.0)


def test_query_texts_are_exact_whitelisted_copies():
    assert odds.ODDS_QUERY == load("odds_query.json")["query"]
    assert odds.MEETING_QUERY == load("odds_meeting_query.json")["query"]


def test_normalise():
    assert odds.normalise_comb("QPL", "07,01") == "1,7"
    assert odds.normalise_comb("WIN", "01") == "1"
    assert odds.normalise_comb("FCT", "07,01") == "7,1"
    assert odds.parse_odds_value("SCR") is None
    assert odds.parse_odds_value("---") is None
    assert odds.parse_odds_value("10") == 10.0


def test_parse_final_odds_with_scratching_20261019_st_r8():
    p = odds.parse_pools(load("odds_20261019_ST_R8.json"), 8)
    assert set(p) == {"WIN", "PLA", "QIN", "QPL"}
    assert p.scratched == {4}
    assert len(p["WIN"]) == 12 and p["WIN"]["4"] is None
    assert p["WIN"]["2"] == 2.8 and p["WIN"]["10"] == 2.5
    assert p["PLA"]["10"] == 1.1
    assert len(p["QPL"]) == 66                                   # C(12,2)
    assert sum(v is None for v in p["QPL"].values()) == 11       # every pair with #4
    assert p["QPL"]["2,10"] == 1.9 and p["QIN"]["2,10"] == 3.5
    assert all(k == ",".join(sorted(k.split(","), key=int)) for k in p["QPL"])
    assert p.meta["WIN"]["status"] == "PAYOUT"
    # final WIN odds agree with the meeting query's runner winOdds
    m = odds.parse_meeting(load("odds_meeting_20261019_ST.json"), "2026-10-19", "ST")
    r8 = {r["horse_no"]: r["win_odds"] for r in m["races"][7]["runners"]}
    assert {int(k): v for k, v in p["WIN"].items()} == r8


def test_parse_live_and_not_open():
    live = odds.parse_pools(load("odds_20261019_ST_R9_live.json"), 9)
    assert live.meta["WIN"]["sell_status"] == "START_SELL"
    assert len(live["WIN"]) == 12 and live["WIN"]["8"] == 2.9
    closed = odds.parse_pools(load("odds_20261021_HV_R1_notopen.json"), 1)
    assert set(closed) == {"WIN", "PLA", "QIN", "QPL"}
    assert all(book == {} for book in closed.values())
    assert closed.meta["WIN"] == {"status": "DEFINED", "sell_status": "STOP_SELL",
                                  "last_update": None, "n_nodes": 0}
    assert not closed.scratched


def test_parse_meeting_20261021_hv():
    m = odds.parse_meeting(load("odds_meeting_20261021_HV.json"), "2026-10-21", "HV")
    assert m["venue"] == "HV" and m["n_races"] == 9 and m["status"] == "DEFINED"
    assert m["first_post_time"] == "2026-10-21T18:40"
    assert [r["post_time"][-5:] for r in m["races"]][:3] == ["18:40", "19:10", "19:40"]
    r1 = m["races"][0]
    assert r1["distance"] == 1650 and r1["surface"] == "TURF" and r1["course"] == "B"
    starters = [x for x in r1["runners"] if not x["standby"]]
    assert len(starters) == r1["field_size"] == 11
    assert r1["runners"][0]["horse_id"] == "K362" and r1["runners"][0]["jockey_id"] == "OJM"
    assert m["scratched"] == {}
    assert {"date": "2026-10-19", "venue": "ST", "status": "STARTED", "n_races": 11} in m["active_meetings"]


def test_parse_meeting_scratchings_20261019_st():
    m = odds.parse_meeting(load("odds_meeting_20261019_ST.json"), "2026-10-19", "ST")
    assert m["scratched"] == {8: [4]}
    assert m["first_post_time"] == "2026-10-19T12:30" and m["n_races"] == 11
    assert odds.parse_meeting(load("odds_meeting_20261019_ST.json"), "2026-10-20", "ST") is None


def test_fetch_pools_sends_whitelisted_body():
    s = FakeSession({8: load("odds_20261019_ST_R8.json")})
    p = odds.fetch_pools("2026-10-19", "ST", 8, ("WIN", "PLA"), session=s)
    body = s.bodies[0]
    assert body["query"] == load("odds_query.json")["query"]
    assert body["variables"] == {"date": "2026-10-19", "venueCode": "ST", "raceNo": 8,
                                 "oddsTypes": ["WIN", "PLA"]}
    assert p.scratched == {4}


def test_snapshot_and_latest(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    meeting = load("odds_meeting_20261019_ST.json")
    s = FakeSession({8: load("odds_20261019_ST_R8.json"), 9: load("odds_20261019_ST_R9_live.json")}, meeting)
    n = odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[8, 9], session=s,
                              snapshot_at="2026-10-19T16:00:00")
    assert n == 2 * (12 + 12 + 66 + 66)
    # second (newer) snapshot of race 9 only, with a changed price
    later = load("odds_20261019_ST_R9_live.json")
    for p in later["data"]["raceMeetings"][0]["pmPools"]:
        if p["oddsType"] == "WIN":
            for node in p["oddsNodes"]:
                if node["combString"] == "08":
                    node["oddsValue"] = "2.5"
    s2 = FakeSession({9: later})
    odds.snapshot_meeting(conn, "2026-10-19", "ST", race_nos=[9], session=s2,
                          snapshot_at="2026-10-19T16:30:00")
    lo = odds.latest_odds(conn, "2026-10-19", "ST")
    assert set(lo) == {8, 9}
    assert lo[9]["WIN"]["8"] == 2.5          # newest snapshot wins
    assert lo[8]["WIN"]["4"] is None and lo[8]["WIN"]["2"] == 2.8
    assert len(lo[8]["QPL"]) == 66
    # race list from meeting query when race_nos not given; not-open pools -> 0 rows
    s3 = FakeSession({i: load("odds_20261021_HV_R1_notopen.json") for i in range(1, 10)},
                     load("odds_meeting_20261021_HV.json"))
    assert odds.snapshot_meeting(conn, "2026-10-21", "HV", session=s3) == 0
    assert sum(b["operationName"] == "racing" for b in s3.bodies) == 9
    assert odds.latest_odds(conn, "2026-10-21", "HV") == {}


def test_now_hk_format():
    import re
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", odds.now_hk())


@pytest.mark.skipif(os.environ.get("HKJC_LIVE") != "1", reason="set HKJC_LIVE=1 for live API checks")
def test_live_api():
    m = odds.fetch_meeting()
    assert m is not None and m["races"]
    p = odds.fetch_pools(m["date"], m["venue"], 1, ("WIN",))
    assert "WIN" in p
