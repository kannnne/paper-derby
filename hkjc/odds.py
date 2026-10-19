"""Live HKJC pari-mutuel odds + meeting info via the bet.hkjc.com GraphQL backend.

READ-ONLY public data, PAPER TRADING ONLY - this module never logs in and never
places bets.

Endpoint: POST https://info.cld.hkjc.com/graphql/base/   (gzip response)
The server only accepts whitelisted query documents, so the query texts below
are byte-for-byte copies of what the bet.hkjc.com front-end sends:
  * ODDS_QUERY    == tests/fixtures/odds_query.json["query"]
  * MEETING_QUERY == tests/fixtures/odds_meeting_query.json["query"]
    (the front-end's ``raceMeetings`` query; same text is shipped in the npm
    package hkjc-api 1.0.5, dist/query/horseRacingQuery.js ``horseQuery``).
Only ``variables`` may be changed.  Verified live on 2026-10-19.

Observed response semantics (2026-10-19):
  * pool ``status``: DEFINED (not open) -> START_SELL (selling) -> ... -> PAYOUT;
    ``sellStatus``: STOP_SELL / START_SELL.  Not-yet-open pools have
    ``oddsNodes == []`` and ``lastUpdateTime == ""``.
  * ``oddsValue`` is a string: "4.1", "10", or "SCR" for a scratched horse (and
    for every QIN/QPL pair containing it).  After the race the final odds
    remain available.
  * combString: "01" (WIN/PLA), "01,07" (QIN/QPL).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import requests

ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://bet.hkjc.com",
    "Referer": "https://bet.hkjc.com/",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"),
}
DEFAULT_POOLS = ("WIN", "PLA", "QPL", "QIN")
MIN_INTERVAL_S = 0.5            # <= 2 requests / second
ORDERED_POOLS = {"FCT", "TCE", "QTT"}

try:
    from zoneinfo import ZoneInfo
    HKT = ZoneInfo("Asia/Hong_Kong")
except Exception:  # pragma: no cover - tzdata missing
    HKT = timezone(timedelta(hours=8))

# --------------------------------------------------------------------- queries
ODDS_QUERY = (
    "query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {\n"
    "  raceMeetings(date: $date, venueCode: $venueCode) {\n"
    "    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {\n"
    "      id\n      status\n      sellStatus\n      oddsType\n      lastUpdateTime\n"
    "      guarantee\n      minTicketCost\n      name_en\n      name_ch\n"
    "      leg {\n        number\n        races\n      }\n"
    "      cWinSelections {\n        composite\n        name_ch\n        name_en\n        starters\n      }\n"
    "      oddsNodes {\n        combString\n        oddsValue\n        hotFavourite\n        oddsDropValue\n"
    "        bankerOdds {\n          combString\n          oddsValue\n        }\n      }\n"
    "    }\n  }\n}"
)

MEETING_QUERY = """fragment raceFragment on Race {
  id
  no
  status
  raceName_en
  raceName_ch
  postTime
  country_en
  country_ch
  distance
  wageringFieldSize
  go_en
  go_ch
  ratingType
  raceTrack {
    description_en
    description_ch
  }
  raceCourse {
    description_en
    description_ch
    displayCode
  }
  claCode
  raceClass_en
  raceClass_ch
  judgeSigns {
    value_en
  }
}

fragment racingBlockFragment on RaceMeeting {
  jpEsts: pmPools(
    oddsTypes: [WIN, PLA, TCE, TRI, FF, QTT, DT, TT, SixUP]
    filters: ["jackpot", "estimatedDividend"]
  ) {
    leg {
      number
      races
    }
    oddsType
    jackpot
    estimatedDividend
    mergedPoolId
  }
  poolInvs: pmPools(
    oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
  ) {
    id
    leg {
      races
    }
  }
  penetrometerReadings(filters: ["first"]) {
    reading
    readingTime
  }
  hammerReadings(filters: ["first"]) {
    reading
    readingTime
  }
  changeHistories(filters: ["top3"]) {
    type
    time
    raceNo
    runnerNo
    horseName_ch
    horseName_en
    jockeyName_ch
    jockeyName_en
    scratchHorseName_ch
    scratchHorseName_en
    handicapWeight
    scrResvIndicator
  }
}

query raceMeetings($date: String, $venueCode: String) {
  timeOffset {
    rc
  }
  activeMeetings: raceMeetings {
    id
    venueCode
    date
    status
    races {
      no
      postTime
      status
      wageringFieldSize
    }
  }
  raceMeetings(date: $date, venueCode: $venueCode) {
    id
    status
    venueCode
    date
    totalNumberOfRace
    currentNumberOfRace
    dateOfWeek
    meetingType
    totalInvestment
    country {
      code
      namech
      nameen
      seq
    }
    races {
      ...raceFragment
      runners {
        id
        no
        standbyNo
        status
        name_ch
        name_en
        horse {
          id
          code
        }
        color
        barrierDrawNumber
        handicapWeight
        currentWeight
        currentRating
        internationalRating
        gearInfo
        racingColorFileName
        allowance
        trainerPreference
        last6run
        saddleClothNo
        trumpCard
        priority
        finalPosition
        deadHeat
        winOdds
        jockey {
          code
          name_en
          name_ch
        }
        trainer {
          code
          name_en
          name_ch
        }
      }
    }
    obSt: pmPools(oddsTypes: [WIN, PLA]) {
      leg {
        races
      }
      oddsType
      comingleStatus
    }
    poolInvs: pmPools(
      oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
    ) {
      id
      leg {
        number
        races
      }
      status
      sellStatus
      oddsType
      investment
      mergedPoolId
      lastUpdateTime
    }
    ...racingBlockFragment
    pmPools(oddsTypes: []) {
      id
    }
    jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
      instNo
    }
    tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
      instNo
    }
  }
}"""

# --------------------------------------------------------------------- transport
_lock = threading.Lock()
_last_request = [0.0]


class OddsAPIError(RuntimeError):
    pass


def _post(body: dict, session: requests.Session | None = None, timeout: float = 20,
          retries: int = 2) -> dict:
    """POST one GraphQL document, rate-limited to <= 2 req/s, with retries."""
    sess = session or requests
    err = None
    for attempt in range(retries + 1):
        with _lock:
            wait = MIN_INTERVAL_S - (time.monotonic() - _last_request[0])
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.monotonic()
        try:
            r = sess.post(ENDPOINT, json=body, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if data.get("errors"):
                raise OddsAPIError(f"GraphQL errors: {data['errors']}")
            return data
        except (requests.RequestException, ValueError) as e:
            err = e
            time.sleep(1.0 + attempt)
    raise OddsAPIError(f"odds request failed: {err}")


# --------------------------------------------------------------------- parsing
def normalise_comb(pool: str, comb_string: str) -> str:
    """'01,07' -> '1,7'; unordered pools sorted ascending, ordered pools kept."""
    parts = [p.strip() for p in str(comb_string).replace("-", ",").split(",") if p.strip()]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return str(comb_string)
    if pool not in ORDERED_POOLS:
        nums.sort()
    return ",".join(str(n) for n in nums)


def parse_odds_value(v):
    """'4.1' -> 4.1; 'SCR' / '---' / '' / None -> None."""
    if v is None:
        return None
    try:
        x = float(str(v).replace(",", "").strip())
    except ValueError:
        return None
    return x if x > 0 else None


class PoolOdds(dict):
    """dict[pool][combination] = odds (float or None) with extra attributes:
    ``scratched`` (set of horse numbers marked SCR), ``meta`` ({pool: {status,
    sell_status, last_update}}), ``race_no``."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.scratched: set[int] = set()
        self.meta: dict[str, dict] = {}
        self.race_no: int | None = None


def parse_pools(resp: dict, race_no: int | None = None) -> PoolOdds:
    """Parse an ODDS_QUERY response into :class:`PoolOdds`."""
    out = PoolOdds()
    out.race_no = race_no
    meetings = ((resp or {}).get("data") or {}).get("raceMeetings") or []
    for m in meetings:
        for p in m.get("pmPools") or []:
            pool = p.get("oddsType")
            if not pool:
                continue
            book = out.setdefault(pool, {})
            out.meta[pool] = {"status": p.get("status"), "sell_status": p.get("sellStatus"),
                              "last_update": p.get("lastUpdateTime") or None,
                              "n_nodes": len(p.get("oddsNodes") or [])}
            for n in p.get("oddsNodes") or []:
                comb = normalise_comb(pool, n.get("combString", ""))
                raw = n.get("oddsValue")
                book[comb] = parse_odds_value(raw)
                if str(raw).strip().upper() == "SCR" and pool in ("WIN", "PLA"):
                    try:
                        out.scratched.add(int(comb))
                    except ValueError:
                        pass
    return out


def fetch_pools(date: str, venue: str, race_no: int, pools=DEFAULT_POOLS,
                session: requests.Session | None = None) -> PoolOdds:
    """Live odds of one race: dict[pool][combination] = odds (None if SCR/---).
    Scratched horses are in ``result.scratched``; pool status in ``result.meta``.
    Pools not open yet come back as empty dicts."""
    body = {"operationName": "racing",
            "variables": {"date": date, "venueCode": venue, "raceNo": int(race_no),
                          "oddsTypes": list(pools)},
            "query": ODDS_QUERY}
    return parse_pools(_post(body, session), race_no=int(race_no))


def _hk_minute(ts: str | None) -> str | None:
    """'2026-10-21T18:40:00+08:00' -> '2026-10-21T18:40' (HK time)."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            dt = dt.astimezone(HKT)
        return dt.strftime("%Y-%m-%dT%H:%M")
    except ValueError:
        return ts[:16]


def _to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _surface(desc: str | None) -> str | None:
    d = (desc or "").upper()
    if not d:
        return None
    if "TURF" in d:
        return "TURF"
    if "ALL WEATHER" in d or "AWT" in d or "DIRT" in d:
        return "AWT"
    return desc


def parse_meeting(resp: dict, date: str | None = None, venue: str | None = None) -> dict | None:
    """Parse a MEETING_QUERY response.  Picks the meeting matching date/venue
    (or the first returned).  Returns None when no meeting is found."""
    data = (resp or {}).get("data") or {}
    meetings = data.get("raceMeetings") or []
    pick = None
    for m in meetings:
        if (date is None or m.get("date") == date) and (venue is None or m.get("venueCode") == venue):
            pick = m
            break
    if pick is None:
        return None
    races = []
    scratched: dict[int, list[int]] = {}
    for r in sorted(pick.get("races") or [], key=lambda r: _to_int(r.get("no")) or 0):
        rno = _to_int(r.get("no"))
        runners = []
        for ru in r.get("runners") or []:
            status = (ru.get("status") or "").strip()
            st_low = status.lower()
            standby = st_low.startswith("standby") or bool((ru.get("standbyNo") or "").strip())
            is_scr = st_low.startswith("scratch")
            hn = _to_int(ru.get("no"))
            if is_scr and hn is not None and not standby:
                scratched.setdefault(rno, []).append(hn)
            runners.append({
                "horse_no": hn, "standby_no": _to_int(ru.get("standbyNo")),
                "status": status, "scratched": is_scr, "standby": standby,
                "horse_id": (ru.get("horse") or {}).get("code"),
                "name_ch": ru.get("name_ch"), "name_en": ru.get("name_en"),
                "draw": _to_int(ru.get("barrierDrawNumber")),
                "weight": _to_int(ru.get("handicapWeight")),
                "body_weight": _to_int(ru.get("currentWeight")),
                "rating": _to_int(ru.get("currentRating")),
                "gear": ru.get("gearInfo") or None, "last6": ru.get("last6run") or None,
                "jockey_id": (ru.get("jockey") or {}).get("code"),
                "jockey_name": (ru.get("jockey") or {}).get("name_ch"),
                "trainer_id": (ru.get("trainer") or {}).get("code"),
                "trainer_name": (ru.get("trainer") or {}).get("name_ch"),
                "final_position": _to_int(ru.get("finalPosition")) or None,
                "dead_heat": bool(ru.get("deadHeat")),
                "win_odds": parse_odds_value(ru.get("winOdds")),
            })
        races.append({
            "race_no": rno, "status": r.get("status"),
            "post_time": _hk_minute(r.get("postTime")),
            "distance": _to_int(r.get("distance")),
            "field_size": _to_int(r.get("wageringFieldSize")),
            "race_class": r.get("raceClass_ch"), "race_name": r.get("raceName_ch"),
            "surface": _surface((r.get("raceTrack") or {}).get("description_en")),
            "course": (r.get("raceCourse") or {}).get("displayCode"),
            "going": r.get("go_ch") or None,
            "runners": runners,
        })
    posts = [r["post_time"] for r in races if r["post_time"]]
    return {
        "date": pick.get("date"), "venue": pick.get("venueCode"),
        "status": pick.get("status"), "meeting_type": pick.get("meetingType"),
        "n_races": _to_int(pick.get("totalNumberOfRace")) or len(races),
        "current_race": _to_int(pick.get("currentNumberOfRace")),
        "first_post_time": min(posts) if posts else None,
        "races": races, "scratched": scratched,
        "change_histories": pick.get("changeHistories") or [],
        "active_meetings": [{"date": m.get("date"), "venue": m.get("venueCode"),
                             "status": m.get("status"), "n_races": len(m.get("races") or [])}
                            for m in data.get("activeMeetings") or []],
    }


def fetch_meeting(date: str | None = None, venue: str | None = None,
                  session: requests.Session | None = None) -> dict | None:
    """Meeting info (race list, post times, race status, runners incl. scratchings).
    With no date/venue returns the first currently active meeting."""
    variables = {}
    if date:
        variables["date"] = date
    if venue:
        variables["venueCode"] = venue
    body = {"operationName": "raceMeetings", "variables": variables, "query": MEETING_QUERY}
    return parse_meeting(_post(body, session), date, venue)


# --------------------------------------------------------------------- database
def now_hk() -> str:
    return datetime.now(HKT).strftime("%Y-%m-%dT%H:%M:%S")


def _race_numbers(conn, date: str, venue: str, session) -> list[int]:
    try:
        info = fetch_meeting(date, venue, session=session)
        if info and info["races"]:
            return [r["race_no"] for r in info["races"] if r["race_no"]]
    except OddsAPIError:
        pass
    row = conn.execute("SELECT n_races FROM meetings WHERE date=? AND venue=?", (date, venue)).fetchone()
    if row and row[0]:
        return list(range(1, int(row[0]) + 1))
    row = conn.execute("SELECT MAX(race_no) FROM races WHERE date=? AND venue=?", (date, venue)).fetchone()
    if row and row[0]:
        return list(range(1, int(row[0]) + 1))
    return []          # unknown: probe until an empty race


def snapshot_meeting(conn, date: str, venue: str, pools=DEFAULT_POOLS,
                     race_nos=None, session: requests.Session | None = None,
                     snapshot_at: str | None = None) -> int:
    """Fetch odds for every race of a meeting and write them to ``odds_snapshots``
    (snapshot_at = now, HK time 'YYYY-MM-DDTHH:MM:SS').  Odds of scratched
    combinations are stored as NULL.  Returns the number of rows written
    (0 when betting is not open yet)."""
    session = session or requests.Session()
    snapshot_at = snapshot_at or now_hk()
    probe = False
    if race_nos is None:
        race_nos = _race_numbers(conn, date, venue, session)
        if not race_nos:
            race_nos, probe = range(1, 15), True
    rows = []
    for rno in race_nos:
        odds = fetch_pools(date, venue, rno, pools, session=session)
        if probe and not odds:
            break
        for pool, book in odds.items():
            for comb, val in book.items():
                rows.append((date, venue, int(rno), pool, comb, val, snapshot_at))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO odds_snapshots (date, venue, race_no, pool, combination, odds, snapshot_at) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        conn.commit()
    return len(rows)


def latest_odds(conn, date: str, venue: str) -> dict[int, dict[str, dict[str, float | None]]]:
    """Most recent snapshot per (race, pool): {race_no: {pool: {combination: odds}}}."""
    cur = conn.execute(
        """SELECT o.race_no, o.pool, o.combination, o.odds
           FROM odds_snapshots o
           JOIN (SELECT race_no, pool, MAX(snapshot_at) AS ts FROM odds_snapshots
                 WHERE date=? AND venue=? GROUP BY race_no, pool) l
             ON o.race_no=l.race_no AND o.pool=l.pool AND o.snapshot_at=l.ts
           WHERE o.date=? AND o.venue=?
           ORDER BY o.race_no, o.pool""", (date, venue, date, venue))
    out: dict[int, dict[str, dict[str, float | None]]] = {}
    for race_no, pool, comb, odds in cur.fetchall():
        out.setdefault(int(race_no), {}).setdefault(pool, {})[comb] = odds
    return out
