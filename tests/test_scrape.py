"""Scraper tests: fetcher behaviour (fake HTTP) and DB writes from cached real pages (offline)."""
import gzip
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest
import requests

from hkjc import db as dbmod
from hkjc import scrape
from hkjc.parse import parse_results

FX = Path(__file__).parent / "fixtures"


@pytest.fixture
def conn(tmp_path):
    c = dbmod.connect(tmp_path / "t.db")
    yield c
    c.close()


def offline_fetcher(tmp_path, *dirs):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    for d in dirs:
        for p in (FX / d).glob("*.html.gz"):
            shutil.copy(p, cache / p.name)
    return scrape.Fetcher(cache_dir=cache, offline=True, min_interval=0)


# ------------------------------------------------------------------ fetcher
class FakeResp:
    def __init__(self, status, text="", headers=None):
        self.status_code, self.text, self.headers, self.encoding = status, text, headers or {}, None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_fetcher_cache_redirects_and_retry(tmp_path, monkeypatch):
    calls = []
    script = {
        "https://racing.hkjc.com/a": [FakeResp(503), FakeResp(200, "<html>A</html>")],
        "https://racing.hkjc.com/old": [FakeResp(302, headers={"Location": "/archive/old"})],
        "https://racing.hkjc.com/archive/old": [FakeResp(200, "<html>OLD</html>")],
        "https://racing.hkjc.com/ovs": [FakeResp(302, headers={"Location": "/zh-hk/overseas/results?x=1"})],
    }

    def fake_get(self, url, timeout=None, allow_redirects=True):
        assert allow_redirects is False
        assert "Mozilla" in self.headers["User-Agent"]
        calls.append((url, time.monotonic()))
        return script[url].pop(0)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(scrape.time, "sleep", lambda s: None)   # skip retry backoff
    f = scrape.Fetcher(cache_dir=tmp_path, min_interval=0)
    p = f.get("https://racing.hkjc.com/a")
    assert p.text == "<html>A</html>" and not p.from_cache
    assert len(calls) == 2                                       # 503 retried
    p2 = f.get("https://racing.hkjc.com/a")
    assert p2.from_cache and p2.text == "<html>A</html>" and len(calls) == 2
    old = f.get("https://racing.hkjc.com/old")
    assert old.text == "<html>OLD</html>" and old.final_url.endswith("/archive/old")
    ovs = f.get("https://racing.hkjc.com/ovs")
    assert ovs.overseas
    assert f.read_cache("https://racing.hkjc.com/ovs").overseas      # overseas marker cached too
    # gzip cache on disk keyed by URL
    assert (tmp_path / scrape.Fetcher.cache_name("https://racing.hkjc.com/a")).exists()
    with gzip.open(tmp_path / scrape.Fetcher.cache_name("https://racing.hkjc.com/a"), "rt") as fh:
        assert fh.readline().startswith("<!--CACHE final=")


def test_fetcher_refresh_and_throttle(tmp_path, monkeypatch):
    stamps = []

    def fake_get(self, url, timeout=None, allow_redirects=True):
        stamps.append(time.monotonic())
        return FakeResp(200, f"<html>{len(stamps)}</html>")

    monkeypatch.setattr(requests.Session, "get", fake_get)
    f = scrape.Fetcher(cache_dir=tmp_path)            # default 0.4s interval
    assert f.min_interval >= 0.4
    f.get("https://racing.hkjc.com/x")
    assert f.get("https://racing.hkjc.com/x").text == "<html>1</html>"          # cached
    assert f.get("https://racing.hkjc.com/x", refresh=True).text == "<html>2</html>"
    assert stamps[1] - stamps[0] >= 0.39


def test_urls_and_ids():
    assert scrape.race_id("2026-10-14", "HV", 1) == "20261014-HV-01"
    assert scrape.results_url("2026-10-14", "HV", 3) == \
        "https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=2026/10/14&Racecourse=HV&RaceNo=3"
    assert scrape.racecard_url("2026-10-21", "HV", 1) == \
        "https://racing.hkjc.com/zh-hk/local/information/racecard?racedate=2026/10/21&Racecourse=HV&RaceNo=1"


# ------------------------------------------------------------------ discovery
def test_discover_meeting_dates_offline(tmp_path):
    f = offline_fetcher(tmp_path, "cache_20241113_HV")
    nodata = (FX / "results_nodata_20230911.html").read_text(encoding="utf-8")
    for d in ("2024-11-12", "2024-11-14"):
        u = scrape.results_url(d)
        f.write_cache(scrape.Page(u, u, nodata, False))
    u = scrape.results_url("2024-11-15")
    f.write_cache(scrape.Page(u, "https://racing.hkjc.com/zh-hk/overseas/results?RaceDate=20241115",
                              scrape.OVERSEAS_MARKER, False))
    assert scrape.probe_date("2024-11-15", f)["status"] == "overseas"
    assert scrape.probe_date("2024-11-14", f)["status"] == "none"
    got = scrape.discover_meeting_dates("2024-11-12", "2024-11-15", fetcher=f)
    assert got == [("2024-11-13", "HV")]


# ------------------------------------------------------------------ meeting results -> DB
def test_scrape_meeting_with_cancelled_races(conn, tmp_path):
    """2024-11-13 HV: races 1-6 run, races 7-9 cancelled (此場賽事宣佈取消)."""
    f = offline_fetcher(tmp_path, "cache_20241113_HV")
    n = scrape.scrape_meeting_results(conn, "2024-11-13", fetcher=f)
    assert n == 6
    m = conn.execute("SELECT * FROM meetings WHERE date='2024-11-13'").fetchone()
    assert (m["venue"], m["n_races"], m["status"]) == ("HV", 9, "results")
    races = conn.execute("SELECT race_id FROM races WHERE date='2024-11-13' ORDER BY race_no").fetchall()
    assert [r[0] for r in races] == [f"20241113-HV-0{i}" for i in range(1, 7)]
    # every written race: exactly one winner, WIN/PLA/QPL dividends present
    for (rid,) in races:
        assert conn.execute("SELECT COUNT(*) FROM runners WHERE race_id=? AND finish_pos=1", (rid,)).fetchone()[0] == 1
        pools = {r[0] for r in conn.execute("SELECT pool FROM dividends WHERE race_id=?", (rid,))}
        assert {"WIN", "PLA", "QPL"} <= pools
        assert conn.execute("SELECT COUNT(*) FROM runners WHERE race_id=? AND scratched=0 AND "
                            "(horse_id IS NULL OR jockey_id IS NULL OR win_odds IS NULL)", (rid,)).fetchone()[0] == 0
    # retired jockey shown without a profile link -> synthetic id
    assert conn.execute("SELECT jockey_id FROM runners WHERE race_id='20241113-HV-02' AND horse_no=9"
                        ).fetchone()[0] == "ZH:湯普新"
    # void race: dividends only (refunds + multi-leg pools paying with F)
    assert conn.execute("SELECT COUNT(*) FROM runners WHERE race_id='20241113-HV-07'").fetchone()[0] == 0
    d7 = {(r["pool"], r["combination"]): r["dividend"] for r in
          conn.execute("SELECT * FROM dividends WHERE race_id='20241113-HV-07'")}
    assert d7[("WIN", "REFUND")] == 10.0 and d7[("DBL", "7/F")] == 27.0
    # spot values vs the page (race 1 = probe page)
    p1 = parse_results(f.read_cache(scrape.results_url("2024-11-13")).text)
    w = next(x for x in p1["runners"] if x["finish_pos"] == 1)
    row = conn.execute("SELECT * FROM runners WHERE race_id='20241113-HV-01' AND horse_no=?",
                       (w["horse_no"],)).fetchone()
    assert (row["horse_id"], row["jockey_id"], row["win_odds"], row["finish_time_s"]) == \
        (w["horse_id"], w["jockey_id"], w["win_odds"], w["finish_time_s"])
    # idempotent re-run
    assert scrape.scrape_meeting_results(conn, "2024-11-13", fetcher=f) == 6
    assert conn.execute("SELECT COUNT(*) FROM races WHERE date='2024-11-13'").fetchone()[0] == 6


def test_write_abandoned_meeting(conn):
    p = parse_results((FX / "results_20231008_ST_R1_abandoned.html").read_text(encoding="utf-8"))
    bundle = {"date": "2023-10-08", "venue": "ST", "race_nos": [1], "races": {1: p}}
    scrape.write_meeting_results(conn, bundle)
    m = conn.execute("SELECT status, n_races FROM meetings WHERE date='2023-10-08'").fetchone()
    assert tuple(m) == ("abandoned", 1)
    assert conn.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM dividends WHERE combination='REFUND'").fetchone()[0] == 9


def test_preliminary_page_not_final(conn):
    """Reduced table (名次/馬號/馬名/騎師 only) -> meeting 'provisional', no partial runner rows."""
    full = parse_results((FX / "results_20261019_ST_R8_sameday_withdrawn.html").read_text(encoding="utf-8"))
    pre = parse_results((FX / "results_20261019_ST_R9_prelim.html").read_text(encoding="utf-8"))
    bundle = {"date": "2026-10-19", "venue": "ST", "race_nos": [8, 9], "races": {8: full, 9: pre}}
    scrape.write_meeting_results(conn, bundle, expected_n=11)
    m = conn.execute("SELECT status, n_races FROM meetings WHERE date='2026-10-19'").fetchone()
    assert tuple(m) == ("provisional", 11)
    assert conn.execute("SELECT COUNT(*) FROM runners WHERE race_id='20261019-ST-09'").fetchone()[0] == 0
    assert conn.execute("SELECT dividend FROM dividends WHERE race_id='20261019-ST-09' AND pool='WIN'"
                        ).fetchone()[0] == 93.5
    r8 = conn.execute("SELECT * FROM runners WHERE race_id='20261019-ST-08' AND horse_no=4").fetchone()
    assert (r8["scratched"], r8["finish_pos"], r8["horse_id"]) == (1, None, "K296")


# ------------------------------------------------------------------ racecard -> DB
def test_scrape_racecard_and_merge_with_results(conn, tmp_path):
    f = offline_fetcher(tmp_path, "cache_card_20261021_HV")
    n = scrape.scrape_racecard(conn, "2026-10-21", "HV", fetcher=f)
    assert n == 9
    m = conn.execute("SELECT * FROM meetings WHERE date='2026-10-21'").fetchone()
    assert (m["venue"], m["n_races"], m["status"], m["first_post_time"]) == ("HV", 9, "card", "2026-10-21T18:40")
    r1 = conn.execute("SELECT * FROM races WHERE race_id='20261021-HV-01'").fetchone()
    assert (r1["post_time"], r1["distance"], r1["course"], r1["race_class"], r1["prize"]) == \
        ("2026-10-21T18:40", 1650, "B", "第四班", 1170000)
    assert conn.execute("SELECT post_time FROM races WHERE race_id='20261021-HV-09'").fetchone()[0] == \
        "2026-10-21T22:50"
    c = conn.execute("SELECT * FROM racecard WHERE date='2026-10-21' AND race_no=1 AND horse_no=3").fetchone()
    assert (c["horse_id"], c["jockey_id"], c["weight"], c["draw"], c["rating"], c["age"], c["sex"]) == \
        ("J079", "WPN", 133, 10, 58, 7, "閹")
    assert conn.execute("SELECT COUNT(*) FROM racecard WHERE date='2026-10-21'").fetchone()[0] == 103
    # a later results write must not wipe post_time / first_post_time
    scrape.upsert_merge(conn, "races", [{"race_id": "20261021-HV-01", "date": "2026-10-21", "venue": "HV",
                                         "race_no": 1, "going": "好地", "post_time": None}], ("race_id",))
    scrape.upsert_merge(conn, "meetings", [{"date": "2026-10-21", "venue": "HV", "status": "results",
                                            "first_post_time": None}], ("date", "venue"))
    assert conn.execute("SELECT post_time, going FROM races WHERE race_id='20261021-HV-01'").fetchone()[:] == \
        ("2026-10-21T18:40", "好地")
    assert conn.execute("SELECT first_post_time FROM meetings WHERE date='2026-10-21'").fetchone()[0] == \
        "2026-10-21T18:40"


def test_racecard_scratched_keeps_details(conn, tmp_path):
    f = scrape.Fetcher(cache_dir=tmp_path, offline=True, min_interval=0)
    u = scrape.racecard_url("2026-10-19", "ST", 8)
    f.write_cache(scrape.Page(u, u, (FX / "racecard_20261019_ST_R8.html").read_text(encoding="utf-8"), False))
    dbmod.upsert(conn, "racecard", [{"date": "2026-10-19", "venue": "ST", "race_no": 8, "horse_no": 4,
                                     "horse_id": "K296", "jockey_id": "FEL", "weight": 125, "scratched": 0}])
    from hkjc.parse import parse_racecard
    p = parse_racecard(f.read_cache(u).text)
    scrape.write_racecard(conn, {"date": "2026-10-19", "venue": "ST", "race_nos": p["meeting_race_nos"],
                                 "races": {8: p}, "schedule": []})
    r = conn.execute("SELECT * FROM racecard WHERE date='2026-10-19' AND race_no=8 AND horse_no=4").fetchone()
    assert (r["scratched"], r["jockey_id"], r["weight"]) == (1, "FEL", 125)


def test_next_meeting_offline(conn, tmp_path, monkeypatch):
    f = offline_fetcher(tmp_path, "cache_card_20261021_HV")
    now = datetime(2026, 10, 19, 17, 30, tzinfo=scrape.HKT)
    monkeypatch.setattr(scrape, "hk_now", lambda: now)
    nm = scrape.next_meeting(conn, fetcher=f)
    assert nm == {"date": "2026-10-21", "venue": "HV", "first_post_time": "2026-10-21T18:40", "n_races": 9}
    m = conn.execute("SELECT status, first_post_time FROM meetings WHERE date='2026-10-21'").fetchone()
    assert tuple(m) == ("scheduled", "2026-10-21T18:40")
