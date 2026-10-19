"""HTTP fetching (polite, cached) + DB writes for HKJC results / racecards.

CLI:
    python -m hkjc.scrape backfill 2023-09-01 2026-10-19 [--workers 3]
    python -m hkjc.scrape meeting 2026-10-14 [HV]
    python -m hkjc.scrape card [YYYY-MM-DD [VENUE]]
    python -m hkjc.scrape next
    python -m hkjc.scrape discover 2023-09-01 2024-08-31
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests

from . import db as dbmod
from .parse import parse_racecard, parse_results

log = logging.getLogger("hkjc.scrape")

HKT = ZoneInfo("Asia/Hong_Kong")
BASE = "https://racing.hkjc.com"
RESULTS_PATH = "/zh-hk/local/information/localresults"
RACECARD_PATH = "/zh-hk/local/information/racecard"
RAW_DIR = dbmod.ROOT / "data" / "raw"

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MIN_INTERVAL = 0.4          # seconds between requests, per thread
MAX_CONCURRENT = 3          # global cap on in-flight requests
RETRIES = 4
OVERSEAS_MARKER = "<!--HKJC-OVERSEAS-REDIRECT-->"

# Results pages for a meeting are treated as immutable once this many days old.
IMMUTABLE_AFTER_DAYS = 3


# ================================================================ fetching
@dataclass
class Page:
    url: str
    final_url: str
    text: str
    from_cache: bool

    @property
    def overseas(self) -> bool:
        return "/overseas/" in self.final_url or self.text.startswith(OVERSEAS_MARKER)


class Fetcher:
    """Thread-safe polite fetcher with a gzip HTML cache keyed by URL."""

    def __init__(self, cache_dir: Path = RAW_DIR, min_interval=MIN_INTERVAL,
                 max_concurrent=MAX_CONCURRENT, retries=RETRIES, offline=False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.sem = threading.BoundedSemaphore(max_concurrent)
        self.retries = retries
        self.offline = offline
        self._local = threading.local()
        self.n_requests = 0
        self._lock = threading.Lock()

    # -- cache
    @staticmethod
    def cache_name(url: str) -> str:
        u = url.replace(BASE, "")
        slug = re.sub(r"[^A-Za-z0-9]+", "_", u).strip("_")
        return slug[:200] + ".html.gz"

    def cache_path(self, url: str) -> Path:
        return self.cache_dir / self.cache_name(url)

    def read_cache(self, url: str):
        p = self.cache_path(url)
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            first = f.readline()
            body = f.read()
        m = re.match(r"<!--CACHE final=(\S*) fetched=(\S*)-->", first)
        final = m.group(1) if m else url
        return Page(url, final, body, True)

    def write_cache(self, page: Page):
        p = self.cache_path(page.url)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(f"<!--CACHE final={page.final_url} fetched={datetime.now(HKT).isoformat()}-->\n")
            f.write(page.text)
        tmp.replace(p)

    # -- http
    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
            })
            self._local.session = s
            self._local.last = 0.0
        return s

    def _throttle(self):
        wait = self._local.last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._local.last = time.monotonic()

    def _get_once(self, url):
        s = self._session()
        cur = url
        for _ in range(6):  # follow redirects manually so overseas redirects can be detected
            with self.sem:
                self._throttle()
                r = s.get(cur, timeout=30, allow_redirects=False)
                with self._lock:
                    self.n_requests += 1
            if r.status_code in (301, 302, 303, 307, 308):
                loc = urljoin(cur, r.headers.get("Location", ""))
                if "/overseas/" in loc:
                    return Page(url, loc, OVERSEAS_MARKER, False)
                cur = loc
                continue
            if r.status_code == 200:
                r.encoding = "utf-8"
                return Page(url, cur, r.text, False)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code} for {cur}")
            r.raise_for_status()
            return Page(url, cur, r.text, False)
        raise requests.HTTPError(f"too many redirects for {url}")

    def get(self, url: str, refresh: bool = False) -> Page:
        if not refresh:
            cached = self.read_cache(url)
            if cached is not None:
                return cached
        if self.offline:
            cached = self.read_cache(url)
            if cached is not None:
                return cached
            raise RuntimeError(f"offline and not cached: {url}")
        delay = 1.5
        last_exc = None
        for attempt in range(self.retries):
            try:
                page = self._get_once(url)
                self.write_cache(page)
                return page
            except (requests.RequestException, OSError) as e:
                last_exc = e
                log.warning("fetch failed (%s/%s) %s: %s", attempt + 1, self.retries, url, e)
                time.sleep(delay)
                delay *= 2.5
        raise last_exc


_FETCHER: Fetcher | None = None


def get_fetcher() -> Fetcher:
    global _FETCHER
    if _FETCHER is None:
        _FETCHER = Fetcher()
    return _FETCHER


# ================================================================ helpers
def hk_now() -> datetime:
    return datetime.now(HKT)


def hk_today() -> Date:
    return hk_now().date()


def _d(x) -> Date:
    if isinstance(x, Date):
        return x
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def race_id(d, venue, race_no) -> str:
    return f"{_d(d):%Y%m%d}-{venue}-{int(race_no):02d}"


def results_url(d, venue=None, race_no=None) -> str:
    u = f"{BASE}{RESULTS_PATH}?racedate={_d(d):%Y/%m/%d}"
    if venue:
        u += f"&Racecourse={venue}"
    if race_no:
        u += f"&RaceNo={int(race_no)}"
    return u


def racecard_url(d=None, venue=None, race_no=None) -> str:
    u = f"{BASE}{RACECARD_PATH}"
    if d is None:
        return u
    u += f"?racedate={_d(d):%Y/%m/%d}"
    if venue:
        u += f"&Racecourse={venue}"
    if race_no:
        u += f"&RaceNo={int(race_no)}"
    return u


def _is_immutable(d) -> bool:
    return _d(d) <= hk_today() - timedelta(days=IMMUTABLE_AFTER_DAYS)


def _now_iso() -> str:
    return hk_now().strftime("%Y-%m-%dT%H:%M:%S")


def upsert_merge(conn, table: str, rows, pk: tuple[str, ...]) -> int:
    """Insert, or update existing row keeping old values where the new value is NULL."""
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    upd = ", ".join(f"{c}=COALESCE(excluded.{c}, {table}.{c})" for c in cols if c not in pk)
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
           f"ON CONFLICT({','.join(pk)}) DO UPDATE SET {upd}")
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    conn.commit()
    return len(rows)


NAME_ID_PREFIX = "ZH:"


def normalize_person_ids(conn) -> int:
    """Fill jockey_id / trainer_id for people shown without a profile link.

    Retired / visiting jockeys and trainers (e.g. 湯普新, 容天鵬, overseas trainers on
    international days) are rendered as plain text, so the page has no id.  We use the
    real id if the same name is linked anywhere in runners/racecard, else a stable
    synthetic id 'ZH:<chinese name>'.  Synthetic ids are upgraded once a real id is seen.
    """
    changed = 0
    for role in ("jockey", "trainer"):
        idc, namec = f"{role}_id", f"{role}_name"
        known = {}
        for tbl in ("runners", "racecard"):
            for name, pid in conn.execute(
                    f"SELECT {namec}, {idc} FROM {tbl} WHERE {idc} IS NOT NULL AND {idc} NOT LIKE ? "
                    f"AND {namec} IS NOT NULL GROUP BY 1, 2", (NAME_ID_PREFIX + "%",)):
                known.setdefault(name, pid)
        for tbl in ("runners", "racecard"):
            rows = conn.execute(
                f"SELECT DISTINCT {namec}, {idc} FROM {tbl} WHERE {namec} IS NOT NULL AND "
                f"({idc} IS NULL OR {idc} LIKE ?)", (NAME_ID_PREFIX + "%",)).fetchall()
            for name, pid in rows:
                new = known.get(name, NAME_ID_PREFIX + name)
                if new != pid:
                    cur = conn.execute(
                        f"UPDATE {tbl} SET {idc}=? WHERE {namec}=? AND ({idc} IS NULL OR {idc} LIKE ?)",
                        (new, name, NAME_ID_PREFIX + "%"))
                    changed += cur.rowcount
    conn.commit()
    return changed


# ================================================================ probing / discovery
def probe_date(d, fetcher: Fetcher | None = None, refresh: bool | None = None) -> dict:
    """Fetch the results page for a date (no venue / race no).

    Returns dict(date, status, venue, race_nos, parsed) where status is one of
    'meeting', 'none' (沒有相關資料), 'overseas' (redirected to overseas results),
    'mismatch' (page shows another date, e.g. future date -> latest meeting).
    """
    f = fetcher or get_fetcher()
    d = _d(d)
    if refresh is None:
        refresh = not _is_immutable(d)
    page = f.get(results_url(d), refresh=refresh)
    out = {"date": d.isoformat(), "venue": None, "race_nos": [], "parsed": None}
    if page.overseas:
        out["status"] = "overseas"
        return out
    p = parse_results(page.text)
    if p["no_data"]:
        out["status"] = "none"
        return out
    if p["meeting_date"] != d.isoformat():
        out["status"] = "mismatch"
        out["shown_date"] = p["meeting_date"]
        return out
    out.update(status="meeting", venue=p["meeting_venue"], race_nos=p["meeting_race_nos"], parsed=p)
    return out


def dropdown_dates(fetcher: Fetcher | None = None) -> list[str]:
    """Dates (ISO) in the results page dropdown: 2025-07 onward + the next meeting.

    NB: includes overseas simulcast dates (they redirect to /overseas/ results)."""
    from .parse import parse_dropdown_dates
    f = fetcher or get_fetcher()
    page = f.get(f"{BASE}{RESULTS_PATH}", refresh=True)
    return parse_dropdown_dates(page.text)


def discover_meeting_dates(start, end, workers: int = 3, fetcher: Fetcher | None = None,
                           exhaustive: bool = True) -> list[tuple[str, str]]:
    """Local HK meetings (date, venue) within [start, end].

    Approach: probe the results page for EVERY calendar day in the range (cheap
    ~15KB 'no data' page for non-race days; cached forever once the date is a few
    days old).  The page itself tells us the venue, and overseas simulcast days
    redirect to /overseas/ and are dropped.  With exhaustive=False only Wed/Sat/Sun,
    public-holiday-ish dates and dropdown dates are probed (not recommended).
    """
    f = fetcher or get_fetcher()
    start, end = _d(start), _d(end)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    if not exhaustive:
        dd = set()
        try:
            dd = {_d(x) for x in dropdown_dates(f)}
        except Exception as e:  # pragma: no cover
            log.warning("dropdown fetch failed: %s", e)
        days = [x for x in days if x.weekday() in (2, 5, 6) or x in dd or (x.month, x.day) in
                {(1, 1), (10, 1), (12, 25), (12, 26), (7, 1), (5, 1)}]
    found = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe_date, x, f): x for x in days}
        for fu in as_completed(futs):
            x = futs[fu]
            done += 1
            try:
                r = fu.result()
            except Exception as e:
                log.error("probe %s failed: %s", x, e)
                continue
            if r["status"] == "meeting":
                found.append((r["date"], r["venue"]))
            if done % 50 == 0:
                log.info("discover: probed %d/%d days, %d meetings so far", done, len(days), len(found))
    found.sort()
    return found


# ================================================================ results
def collect_meeting_results(d, venue=None, fetcher: Fetcher | None = None,
                            refresh: bool | None = None) -> dict | None:
    """Fetch + parse every race of a meeting. No DB access (thread-safe)."""
    f = fetcher or get_fetcher()
    d = _d(d)
    immutable = _is_immutable(d)
    if refresh is None:
        refresh = not immutable
    probe = probe_date(d, f, refresh=refresh)
    if probe["status"] != "meeting":
        return None
    v = probe["venue"]
    if venue and venue != v:
        log.warning("%s: requested venue %s but page shows %s; using %s", d, venue, v, v)
    race_nos = probe["race_nos"] or ([probe["parsed"]["race"].get("race_no")]
                                     if probe["parsed"] else [])
    races = {}
    pr = probe["parsed"]
    if pr and pr["race"].get("race_no"):
        races[pr["race"]["race_no"]] = pr
    for n in race_nos:
        if n in races and (races[n]["complete"] or races[n].get("void")):
            continue
        url = results_url(d, v, n)
        page = f.get(url, refresh=refresh)
        p = parse_results(page.text)
        if not p["complete"] and not p.get("void") and page.from_cache:
            page = f.get(url, refresh=True)
            p = parse_results(page.text)
        if p["no_data"] or p["race"].get("race_no") != n or p["meeting_date"] != d.isoformat():
            log.warning("%s %s R%s: unexpected page (race_no=%s date=%s no_data=%s)", d, v, n,
                        p["race"].get("race_no"), p["meeting_date"], p["no_data"])
            continue
        races[n] = p
    return {"date": d.isoformat(), "venue": v, "race_nos": race_nos, "races": races}


def _expected_n_races(conn, d, venue, fetcher) -> int | None:
    """For recent meetings, the racecard tells how many races are scheduled."""
    row = conn.execute("SELECT n_races, status FROM meetings WHERE date=? AND venue=?",
                       (d, venue)).fetchone()
    try:
        page = fetcher.get(racecard_url(d, venue, 1), refresh=True)
        rc = parse_racecard(page.text)
        if rc["race"].get("date") == d and rc["meeting_race_nos"]:
            return max(rc["meeting_race_nos"])
    except Exception as e:  # pragma: no cover
        log.warning("racecard check failed for %s %s: %s", d, venue, e)
    if row and row["n_races"] and row["status"] == "card":
        return row["n_races"]
    return None


def write_meeting_results(conn, bundle: dict, expected_n: int | None = None) -> int:
    d, v = bundle["date"], bundle["venue"]
    races = bundle["races"]
    n_written = 0
    all_complete = bool(races)
    void = []
    for n in sorted(races):
        p = races[n]
        rid = race_id(d, v, n)
        r = p["race"]
        if p.get("void"):
            # cancelled (此場賽事宣佈取消) or abandoned race: no races/runners rows; dividends are
            # kept (single-race pools = 'REFUND' rows; multi-race pools may pay with 'F' legs)
            log.warning("%s void race (%s) - only dividends stored", rid,
                        "cancelled" if p.get("cancelled") else "abandoned")
            void.append(n)
            conn.execute("DELETE FROM runners WHERE race_id=?", (rid,))
            conn.execute("DELETE FROM races WHERE race_id=?", (rid,))
            conn.execute("DELETE FROM dividends WHERE race_id=?", (rid,))
            dbmod.upsert(conn, "dividends", [
                {"race_id": rid, "pool": x["pool"], "combination": x["combination"],
                 "dividend": x["dividend"]} for x in p["dividends"]])
            continue
        complete = p["complete"]
        all_complete &= complete
        if not p["runners"] and not p["dividends"]:
            log.info("%s: no results yet - skipped", rid)
            continue
        race_row = {
            "race_id": rid, "date": d, "venue": v, "race_no": n,
            "season_race_index": r.get("season_race_index"),
            "race_class": r.get("race_class"), "rating_band": r.get("rating_band"),
            "distance": r.get("distance"), "surface": r.get("surface"),
            "course": r.get("course"), "going": r.get("going"),
            "race_name": r.get("race_name"), "prize": r.get("prize"),
            "field_size": r.get("field_size") if p["full_table"] else None,
            "winning_time_s": r.get("winning_time_s"),
            "sectional_times": json.dumps(r["sectional_times"]) if r.get("sectional_times") else None,
        }
        upsert_merge(conn, "races", [race_row], ("race_id",))
        runner_rows = []
        for x in p["runners"]:
            runner_rows.append({
                "race_id": rid, "horse_no": x.get("horse_no"), "horse_id": x.get("horse_id"),
                "horse_name": x.get("horse_name"), "jockey_id": x.get("jockey_id"),
                "jockey_name": x.get("jockey_name"), "trainer_id": x.get("trainer_id"),
                "trainer_name": x.get("trainer_name"), "act_weight": x.get("act_weight"),
                "body_weight": x.get("body_weight"), "draw": x.get("draw"),
                "finish_pos": x.get("finish_pos"), "finish_status": x.get("finish_status") or None,
                "lbw": x.get("lbw"), "lbw_num": x.get("lbw_num"),
                "running_pos": x.get("running_pos"), "finish_time_s": x.get("finish_time_s"),
                "win_odds": x.get("win_odds"), "scratched": x.get("scratched", 0),
            })
        runner_rows = [x for x in runner_rows if x["horse_no"] is not None]
        if not p["full_table"]:
            # preliminary page (only 名次/馬號/馬名/騎師 of the placed horses): keep dividends
            # for early settlement, but don't write partial runner rows
            runner_rows = []
        if runner_rows:
            conn.execute("DELETE FROM runners WHERE race_id=?", (rid,))
            dbmod.upsert(conn, "runners", runner_rows)
        if p["dividends"]:
            conn.execute("DELETE FROM dividends WHERE race_id=?", (rid,))
            dbmod.upsert(conn, "dividends", [
                {"race_id": rid, "pool": x["pool"], "combination": x["combination"],
                 "dividend": x["dividend"]} for x in p["dividends"]])
        if p.get("unknown_pools"):
            log.warning("%s unknown pools: %s", rid, p["unknown_pools"])
        n_written += 1
    n_races = max(bundle["race_nos"]) if bundle["race_nos"] else len(races)
    if expected_n and expected_n > n_races:
        n_races = expected_n
    normalize_person_ids(conn)
    final = all_complete and len(races) == n_races and set(races) == set(range(1, n_races + 1))
    status = "results" if final else "provisional"
    if void and len(void) == len(races):
        status = "abandoned"
    elif void:
        log.warning("%s %s: void races %s (n_races still counts them)", d, v, void)
    upsert_merge(conn, "meetings", [{
        "date": d, "venue": v, "n_races": n_races,
        "status": status,
        "updated_at": _now_iso(),
    }], ("date", "venue"))
    conn.commit()
    return n_written


def scrape_meeting_results(conn, date, venue=None, refresh: bool | None = None,
                           fetcher: Fetcher | None = None) -> int:
    """Fetch all races of a meeting and upsert meetings/races/runners/dividends.

    Returns the number of races written (0 if 沒有相關資料 / not a local meeting).
    The meeting is marked status='results' only when every race has the full
    results table (with 獨贏賠率) and a WIN dividend, and (for recent meetings)
    the number of races matches the racecard; otherwise status='provisional'.
    """
    f = fetcher or get_fetcher()
    bundle = collect_meeting_results(date, venue, f, refresh)
    if bundle is None:
        return 0
    expected = None
    if not _is_immutable(bundle["date"]):
        expected = _expected_n_races(conn, bundle["date"], bundle["venue"], f)
    return write_meeting_results(conn, bundle, expected)


# ================================================================ racecard
def collect_racecard(d=None, venue=None, refresh: bool = True, fetcher: Fetcher | None = None):
    f = fetcher or get_fetcher()
    page = f.get(racecard_url(d, venue, 1 if d and venue else None), refresh=refresh)
    first = parse_racecard(page.text)
    rd = first["race"].get("date")
    v = first["race"].get("venue")
    if d is not None and rd != _d(d).isoformat():
        return None
    if not first["meeting_race_nos"] and not first["runners"]:
        return None
    pages = {}
    if first["race"].get("race_no"):
        pages[first["race"]["race_no"]] = first
    for n in first["meeting_race_nos"]:
        if n in pages:
            continue
        pg = f.get(racecard_url(rd, v, n), refresh=refresh)
        p = parse_racecard(pg.text)
        if p["race"].get("race_no") != n or p["race"].get("date") != rd:
            log.warning("racecard %s %s R%s: unexpected page", rd, v, n)
            continue
        pages[n] = p
    return {"date": rd, "venue": v, "race_nos": first["meeting_race_nos"], "races": pages,
            "schedule": first["schedule"]}


def write_racecard(conn, bundle) -> int:
    d, v = bundle["date"], bundle["venue"]
    now = _now_iso()
    n = 0
    for no in sorted(bundle["races"]):
        p = bundle["races"][no]
        r = p["race"]
        upsert_merge(conn, "races", [{
            "race_id": race_id(d, v, no), "date": d, "venue": v, "race_no": no,
            "race_class": r.get("race_class"), "rating_band": r.get("rating_band"),
            "distance": r.get("distance"), "surface": r.get("surface"), "course": r.get("course"),
            "going": r.get("going"), "race_name": r.get("race_name"), "prize": r.get("prize"),
            "post_time": r.get("post_time"),
        }], ("race_id",))
        rows = []
        for x in p["runners"]:
            rows.append({
                "date": d, "venue": v, "race_no": no, "horse_no": x["horse_no"],
                "horse_id": x.get("horse_id"), "horse_name": x.get("horse_name"),
                "jockey_id": x.get("jockey_id"), "jockey_name": x.get("jockey_name"),
                "trainer_id": x.get("trainer_id"), "trainer_name": x.get("trainer_name"),
                "weight": x.get("weight"), "draw": x.get("draw"), "rating": x.get("rating"),
                "rating_change": x.get("rating_change"), "body_weight": x.get("body_weight"),
                "last6": x.get("last6"), "gear": x.get("gear"), "age": x.get("age"),
                "sex": x.get("sex"), "days_since_last": x.get("days_since_last"),
                "jockey_allowance": x.get("jockey_allowance"),
                "scratched": x.get("scratched", 0), "scraped_at": now,
            })
        if rows:
            # a scratched horse's row has '-' everywhere: keep previously scraped details
            fresh = [x for x in rows if not x["scratched"]]
            scr = [x for x in rows if x["scratched"]]
            dbmod.upsert(conn, "racecard", fresh)
            if scr:
                upsert_merge(conn, "racecard", scr, ("date", "venue", "race_no", "horse_no"))
                conn.executemany(
                    "UPDATE racecard SET scratched=1 WHERE date=? AND venue=? AND race_no=? AND horse_no=?",
                    [(x["date"], x["venue"], x["race_no"], x["horse_no"]) for x in scr])
        n += 1
    normalize_person_ids(conn)
    first = bundle["races"].get(1) or (bundle["races"][min(bundle["races"])] if bundle["races"] else None)
    row = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?", (d, v)).fetchone()
    status = row["status"] if row and row["status"] in ("results", "provisional") else "card"
    upsert_merge(conn, "meetings", [{
        "date": d, "venue": v, "n_races": max(bundle["race_nos"]) if bundle["race_nos"] else n,
        "first_post_time": first["race"].get("post_time") if first and 1 in bundle["races"] else None,
        "status": status, "updated_at": now,
    }], ("date", "venue"))
    conn.commit()
    return n


def scrape_racecard(conn, date=None, venue=None, refresh: bool = True,
                    fetcher: Fetcher | None = None) -> int:
    """Upsert racecard + races (with post_time) + meetings(status='card'). Returns #races."""
    bundle = collect_racecard(date, venue, refresh, fetcher)
    if bundle is None:
        return 0
    return write_racecard(conn, bundle)


def next_meeting(conn=None, fetcher: Fetcher | None = None) -> dict | None:
    """Next upcoming local meeting whose first race has not started yet.

    Uses the racecard page's meeting schedule (賽期表 buttons) and the results
    dropdown; fetches race 1 of the candidate's racecard for the post time.
    """
    f = fetcher or get_fetcher()
    now = hk_now()
    today = now.date()
    cands = []
    try:
        page = f.get(racecard_url(), refresh=True)
        rc = parse_racecard(page.text)
        cands += [(s["date"], s["venue"]) for s in rc["schedule"] if s["date"]]
    except Exception as e:  # pragma: no cover
        log.warning("racecard schedule fetch failed: %s", e)
    try:
        cands += [(x, None) for x in dropdown_dates(f) if _d(x) >= today]
    except Exception as e:  # pragma: no cover
        log.warning("dropdown fetch failed: %s", e)
    seen = set()
    for d, v in sorted(cands, key=lambda t: t[0]):
        if d in seen or _d(d) < today:
            continue
        try:
            page = f.get(racecard_url(d, v, 1) if v else racecard_url(d), refresh=True)
            rc = parse_racecard(page.text)
            race = rc["race"]
            if race.get("date") != d or not race.get("post_time"):
                continue
            if race.get("race_no") != 1:
                page = f.get(racecard_url(d, race["venue"], 1), refresh=True)
                rc = parse_racecard(page.text)
                race = rc["race"]
        except Exception as e:
            log.warning("next_meeting: racecard for %s %s failed: %s", d, v, e)
            continue
        seen.add(d)
        pt = datetime.strptime(race["post_time"], "%Y-%m-%dT%H:%M").replace(tzinfo=HKT)
        if pt <= now:
            continue
        out = {"date": d, "venue": race["venue"], "first_post_time": race["post_time"],
               "n_races": max(rc["meeting_race_nos"]) if rc["meeting_race_nos"] else None}
        if conn is not None:
            row = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?",
                               (d, out["venue"])).fetchone()
            upsert_merge(conn, "meetings", [{
                "date": d, "venue": out["venue"], "n_races": out["n_races"],
                "first_post_time": out["first_post_time"],
                "status": row["status"] if row and row["status"] else "scheduled",
                "updated_at": _now_iso()}], ("date", "venue"))
        return out
    return None


# ================================================================ backfill
def backfill(conn, start, end, workers: int = 3, fetcher: Fetcher | None = None,
             force: bool = False) -> dict:
    """Discover + scrape every local meeting in [start, end]. Resumable."""
    f = fetcher or get_fetcher()
    t0 = time.time()
    log.info("backfill %s -> %s: discovering meetings (probing every day)", start, end)
    meetings = discover_meeting_dates(start, end, workers=workers, fetcher=f)
    log.info("discovered %d meetings in %.0fs (%d HTTP requests so far)", len(meetings),
             time.time() - t0, f.n_requests)
    done = {(r["date"], r["venue"]) for r in conn.execute(
        "SELECT date, venue FROM meetings WHERE status IN ('results','abandoned')")}
    todo = [m for m in meetings if force or m not in done]
    log.info("%d meetings already complete, %d to scrape", len(meetings) - len(todo), len(todo))
    stats = {"meetings": len(meetings), "scraped": 0, "races": 0, "failed": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(collect_meeting_results, d, v, f): (d, v) for d, v in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            d, v = futs[fu]
            try:
                bundle = fu.result()
                if bundle is None:
                    log.warning("%s %s: no results page", d, v)
                    stats["failed"].append((d, v, "no data"))
                    continue
                expected = None
                if not _is_immutable(d):
                    expected = _expected_n_races(conn, d, v, f)
                n = write_meeting_results(conn, bundle, expected)
                stats["scraped"] += 1
                stats["races"] += n
                st = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?",
                                  (d, v)).fetchone()["status"]
                log.info("[%d/%d] %s %s: %d races (%s)  elapsed %.0fs req=%d", i, len(todo), d, v,
                         n, st, time.time() - t0, f.n_requests)
            except Exception as e:
                log.exception("meeting %s %s failed: %s", d, v, e)
                stats["failed"].append((d, v, repr(e)))
    log.info("backfill done: %s (%.0fs)", {k: (v if k != "failed" else len(v)) for k, v in stats.items()},
             time.time() - t0)
    return stats


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hkjc.scrape")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("start")
    b.add_argument("end")
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--force", action="store_true")
    m = sub.add_parser("meeting")
    m.add_argument("date")
    m.add_argument("venue", nargs="?")
    m.add_argument("--refresh", action="store_true")
    c = sub.add_parser("card")
    c.add_argument("date", nargs="?")
    c.add_argument("venue", nargs="?")
    sub.add_parser("next")
    dsc = sub.add_parser("discover")
    dsc.add_argument("start")
    dsc.add_argument("end")
    ap.add_argument("--db", default=None, help="SQLite path (default: $HKJC_DB or data/hkjc.db)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    conn = dbmod.connect(a.db)
    if a.cmd == "backfill":
        st = backfill(conn, a.start, a.end, workers=a.workers, force=a.force)
        for x in st["failed"]:
            print("FAILED", *x)
    elif a.cmd == "meeting":
        print(scrape_meeting_results(conn, a.date, a.venue, refresh=True if a.refresh else None))
    elif a.cmd == "card":
        if a.date:
            n = scrape_racecard(conn, a.date, a.venue)
            print(n)
        else:
            nm = next_meeting(conn)
            print("next meeting:", nm)
            if nm:
                print(scrape_racecard(conn, nm["date"], nm["venue"]))
    elif a.cmd == "next":
        print(next_meeting(conn))
    elif a.cmd == "discover":
        ms = discover_meeting_dates(a.start, a.end)
        for x in ms:
            print(*x)
        print(len(ms), "meetings")


if __name__ == "__main__":
    main()
