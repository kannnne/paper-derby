"""Pure HTML -> dict parsers for HKJC results and racecard pages (zh-hk).

No network or DB access here.  See SPEC.md for ID / code conventions.

Main entry points
-----------------
parse_results(html)  -> dict(race, runners, dividends, meeting_race_nos,
                             meeting_date, meeting_venue, complete, no_data, ...)
parse_racecard(html) -> dict(race, runners, meeting_race_nos, schedule, ...)
"""
from __future__ import annotations

import json
import re
from fractions import Fraction
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

VENUE_MAP = {"沙田": "ST", "跑馬地": "HV"}

# Chinese pool name -> SPEC pool code
POOL_MAP = {
    "獨贏": "WIN",
    "位置": "PLA",
    "連贏": "QIN",
    "位置Q": "QPL",
    "二重彩": "FCT",
    "三重彩": "TCE",
    "單T": "TRI",
    "四連環": "FF",
    "四重彩": "QTT",
    "孖寶": "DBL",
    "三寶": "TBL",
    "六環彩": "SIXUP",
    "孖T": "DT",
    "三T": "TT",
}
UNORDERED_POOLS = {"QIN", "QPL", "TRI", "FF"}
ORDERED_POOLS = {"FCT", "TCE", "QTT"}
MULTI_RACE_POOLS = {"DBL", "TBL", "SIXUP", "DT", "TT"}

# Margin abbreviations (English + Chinese) -> lengths
MARGIN_WORDS = {
    "N": 0.05, "NOSE": 0.05, "鼻": 0.05, "鼻位": 0.05, "馬鼻位": 0.05,
    "SH": 0.1, "SHD": 0.1, "短頭": 0.1, "短頭位": 0.1, "短馬頭位": 0.1,
    "HD": 0.2, "H": 0.2, "頭": 0.2, "頭位": 0.2, "馬頭位": 0.2,
    "NK": 0.3, "頸": 0.3, "頸位": 0.3, "馬頸位": 0.3,
}
DEAD_HEAT_WORDS = ("平頭馬", "DH")
REFUND_TEXT = "退款"

WITHDRAWN_STATUSES = {"WV", "WV-A", "WX", "WX-A", "WXNR"}

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- helpers
def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def clean(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = s.get_text(" ", strip=True)
    return _WS.sub(" ", s.replace("\xa0", " ")).strip()


def to_int(s):
    s = clean(s).replace(",", "")
    m = re.fullmatch(r"[+-]?\d+", s)
    return int(s) if m else None


def to_float(s):
    s = clean(s).replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_time(s):
    """'1:40.50' -> 100.5 ; '59.80' -> 59.8 ; '(1:16.63)' -> 76.63 ; '1.38.91' -> 98.91."""
    s = clean(s).strip("()")
    if not s or s in ("---", "-"):
        return None
    m = re.fullmatch(r"(?:(\d+)[:.])?(\d{1,2})\.(\d{1,2})", s)
    if not m:
        return None
    mins = int(m.group(1)) if m.group(1) else 0
    return round(mins * 60 + int(m.group(2)) + int(m.group(3)) / (10 ** len(m.group(3))), 2)


def parse_lbw(s, finish_pos=None):
    """Lengths behind winner.  '---' -> 0 for the winner, else None."""
    raw = clean(s)
    if finish_pos == 1:
        return 0.0
    if raw in ("", "-"):
        return None
    if raw == "---":
        # a placed horse (not the winner) with '---' crossed the line first and was
        # relegated by the stewards (e.g. 2024-11-20 HV R7) -> 0 lengths behind the post
        return 0.0 if finish_pos else None
    # leading '-' (e.g. '-短馬頭位' on a promoted winner): margin relative to 1st past post
    t = raw.lstrip("-").upper().replace(" ", "")
    raw = raw.lstrip("-")
    if t in MARGIN_WORDS:
        return MARGIN_WORDS[t]
    if raw in MARGIN_WORDS:
        return MARGIN_WORDS[raw]
    m = re.fullmatch(r"(\d+)(?:-(\d+)/(\d+))?", t)
    if m:
        v = float(m.group(1))
        if m.group(2):
            v += int(m.group(2)) / int(m.group(3))
        return round(v, 4)
    m = re.fullmatch(r"(\d+)/(\d+)", t)
    if m:
        return round(float(Fraction(int(m.group(1)), int(m.group(2)))), 4)
    return None  # incl. '平頭馬' (dead heat): resolved against the tied horse in parse_results


def _qs(href: str, key: str):
    if not href:
        return None
    q = parse_qs(urlparse(href).query)
    for k, v in q.items():
        if k.lower() == key.lower() and v:
            return v[0]
    return None


def _brand_from_horse_link(a):
    hid = _qs(a.get("href", ""), "horseid") if a else None
    if hid:
        return hid.split("_")[-1]
    return None


def _header_key(td) -> str:
    return clean(td).replace(" ", "")


def _race_nos_from_nav(soup) -> list[int]:
    """Race numbers listed in the js_racecard nav table (links + current)."""
    nos = set()
    tbl = soup.find("table", class_="js_racecard")
    if not tbl:
        return []
    for img in tbl.find_all("img"):
        m = re.search(r"racecard_rt_(\d+)(?:_o)?\.gif", img.get("src", ""))
        if m:
            nos.add(int(m.group(1)))
    for a in tbl.find_all("a", href=True):
        n = _qs(a["href"], "RaceNo")
        if n and n.isdigit():
            nos.add(int(n))
    return sorted(nos)


def _current_race_no(soup):
    """Race shown on the page = the nav image without a link ('racecard_rt_N_o.gif')."""
    tbl = soup.find("table", class_="js_racecard")
    if tbl:
        for img in tbl.find_all("img"):
            m = re.search(r"racecard_rt_(\d+)_o\.gif", img.get("src", ""))
            if m:
                return int(m.group(1))
    return None


def _nav_venue(soup):
    tbl = soup.find("table", class_="js_racecard")
    if tbl:
        for a in tbl.find_all("a", href=True):
            v = _qs(a["href"], "Racecourse")
            if v:
                return v.upper()
        t = clean(tbl)
        for zh, code in VENUE_MAP.items():
            if zh in t:
                return code
    return None


def _ddmmyyyy_to_iso(s):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s or "")
    if not m:
        return None
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"


def parse_dropdown_dates(html: str) -> list[str]:
    """Dates listed in the results-page meeting dropdown (ISO, newest first)."""
    soup = _soup(html)
    out = []
    for opt in soup.select("select#selectId option"):
        val = opt.get("value") or ""
        try:
            d = json.loads(val).get("date")
        except (ValueError, AttributeError):
            d = clean(opt)
        iso = _ddmmyyyy_to_iso(d)
        if iso and iso not in out:
            out.append(iso)
    return out


def normalise_combination(pool: str, combo: str) -> str:
    c = clean(combo).replace(" ", "").replace("，", ",")
    c = c.replace("任何組合", "F").replace("任何次序", "M")
    if pool in MULTI_RACE_POOLS:
        legs = c.split("/")
        norm = []
        for leg in legs:
            parts = [p for p in leg.split(",") if p]
            if all(p.isdigit() for p in parts) and len(parts) > 1:
                parts = [str(x) for x in sorted(int(p) for p in parts)]
            norm.append(",".join(parts))
        return "/".join(norm)
    parts = [p for p in c.split(",") if p]
    if pool in UNORDERED_POOLS and all(p.isdigit() for p in parts):
        parts = [str(x) for x in sorted(int(p) for p in parts)]
    return ",".join(parts)


SKIP_POOLS = ("騎師王", "練馬師王")   # jockey / trainer challenge: no per-$10 dividend on page


def pool_code(name: str):
    n = clean(name).replace(" ", "")
    # leg-number prefix of multi-race pools: '第一口孖寶', '第二口三寶', '第四口孖T'
    n = re.sub(r"^第[一二三四五六七八九十\d]+口", "", n)
    if n in POOL_MAP:
        return POOL_MAP[n]
    # e.g. "孖寶第1口" / "三T(安慰獎)" style variants: map prefix + suffix
    for zh in sorted(POOL_MAP, key=len, reverse=True):
        if n.startswith(zh):
            rest = n[len(zh):]
            code = POOL_MAP[zh]
            if not rest:
                return code
            if "安慰" in rest:
                return code + "_CONS"
            m = re.search(r"(\d+)", rest)
            if m:
                return f"{code}{m.group(1)}"
            return code + "_" + rest
    return None


# ---------------------------------------------------------------- results
def _parse_race_tab(soup):
    race = {}
    tab = soup.find("div", class_="race_tab")
    if not tab:
        return race
    head = clean(tab.find("thead")) if tab.find("thead") else ""
    m = re.search(r"第\s*(\d+)\s*場", head)
    if m:
        race["race_no"] = int(m.group(1))
    m = re.search(r"\((\d+)\)", head)
    race["season_race_index"] = int(m.group(1)) if m else None

    rows = tab.find("tbody").find_all("tr") if tab.find("tbody") else []
    cells = [[clean(td) for td in tr.find_all("td", recursive=False)] for tr in rows]
    cells = [c for c in cells if any(c)]
    # row 0: class / distance / band | 場地狀況 | going
    # row 1: race name             | 賽道     | surface/course
    # row 2: prize                 | 時間     | cumulative times
    # row 3: ''                    | 分段時間 | sectionals
    for c in cells:
        label = c[1].replace(" ", "") if len(c) > 1 else ""
        if label.startswith("場地狀況"):
            race.update(parse_class_line(c[0]))
            race["going"] = c[2] if len(c) > 2 else None
        elif label.startswith("賽道"):
            race["race_name"] = c[0] or None
            race.update(parse_track(c[2] if len(c) > 2 else ""))
        elif label.startswith("時間"):
            race["prize"] = to_int(re.sub(r"[^\d]", "", c[0])) if c[0] else None
            times = [parse_time(x) for x in c[2:] if x]
            times = [t for t in times if t is not None]
            race["cumulative_times"] = times
            race["winning_time_s"] = times[-1] if times else None
        elif label.startswith("分段時間"):
            pass
    # sectionals: main split value is first text node of each td (sub-splits are in a div)
    for tr in rows:
        tds = tr.find_all("td", recursive=False)
        if len(tds) > 1 and clean(tds[1]).replace(" ", "").startswith("分段時間"):
            secs = []
            for td in tds[2:]:
                main = td.find(string=True, recursive=False)
                v = to_float(main) if main else None
                if v is None:
                    v = to_float(clean(td).split(" ")[0]) if clean(td) else None
                if v is not None:
                    secs.append(v)
            race["sectional_times"] = secs
    return race


def parse_class_line(s: str) -> dict:
    """'第五班 - 1650米 - (40-0)' / '一級賽 - 2000米' / '新馬賽 - 1200米'."""
    out = {"race_class": None, "distance": None, "rating_band": None}
    parts = [p.strip() for p in re.split(r"\s+-\s+", clean(s)) if p.strip()]
    cls = []
    for p in parts:
        m = re.fullmatch(r"(\d{3,4})\s*米", p)
        if m:
            out["distance"] = int(m.group(1))
            continue
        m = re.fullmatch(r"\((.+)\)", p)
        if m:
            out["rating_band"] = m.group(1).strip()
            continue
        cls.append(p)
    out["race_class"] = " - ".join(cls) or None
    return out


def parse_track(s: str) -> dict:
    """'草地 - "A" 賽道' -> TURF/A ; '全天候跑道' -> AWT."""
    s = clean(s)
    surface = None
    if "全天候" in s:
        surface = "AWT"
    elif "草地" in s:
        surface = "TURF"
    m = re.search(r'"([^"]+)"', s)
    course = m.group(1).strip() if m else None
    return {"surface": surface, "course": course, "track_raw": s or None}


def _results_header(table):
    thead = table.find("thead")
    if not thead:
        return []
    return [_header_key(td) for td in thead.find_all("td")]


RESULT_COLS = {
    "名次": "finish_status",
    "馬號": "horse_no",
    "馬名": "horse",
    "騎師": "jockey",
    "練馬師": "trainer",
    "實際負磅": "act_weight",
    "排位體重": "body_weight",
    "檔位": "draw",
    "頭馬距離": "lbw",
    "沿途走位": "running_pos",
    "完成時間": "finish_time",
    "獨贏賠率": "win_odds",
}


def _finish_pos(status: str):
    m = re.match(r"^(\d+)", status or "")
    return int(m.group(1)) if m else None


def _parse_runner_row(tr, cols):
    tds = tr.find_all("td", recursive=False)
    r = {}
    for key, td in zip(cols, tds):
        field = RESULT_COLS.get(key)
        if not field:
            continue
        if field == "horse":
            a = td.find("a")
            txt = clean(td)
            m = re.search(r"\(([A-Z]{1,2}\d{2,4})\)", txt)
            r["horse_id"] = m.group(1) if m else _brand_from_horse_link(a)
            r["horse_name"] = clean(a) if a else re.sub(r"\(.*?\)", "", txt).strip() or None
        elif field == "jockey":
            a = td.find("a")
            r["jockey_id"] = _qs(a.get("href", ""), "jockeyid") if a else None
            r["jockey_name"] = re.sub(r"\s*\([+-]?\d+\)\s*$", "", clean(td)) or None
        elif field == "trainer":
            a = td.find("a")
            r["trainer_id"] = _qs(a.get("href", ""), "trainerid") if a else None
            r["trainer_name"] = clean(td) or None
        elif field == "running_pos":
            divs = td.find_all("div")
            inner = [clean(d) for d in divs if not d.find("div")]
            vals = [x for x in inner if x] if divs else clean(td).split(" ")
            vals = [v for v in vals if v and v != "---"]
            r["running_pos"] = " ".join(vals) or None
        elif field == "finish_status":
            r["finish_status"] = clean(td)
        elif field == "horse_no":
            r["horse_no"] = to_int(td)
            r["_horse_no_raw"] = clean(td)
        elif field in ("act_weight", "body_weight", "draw"):
            r[field] = to_int(td)
        elif field == "lbw":
            r["lbw"] = clean(td) or None
        elif field == "finish_time":
            r["finish_time_s"] = parse_time(td)
        elif field == "win_odds":
            r["win_odds"] = to_float(td)
    return r


def parse_results(html: str) -> dict:
    soup = _soup(html)
    text = clean(soup.get_text(" "))
    out = {
        "race": {}, "runners": [], "dividends": [], "meeting_race_nos": [],
        "meeting_date": None, "meeting_venue": None, "no_data": False,
        "complete": False, "full_table": False, "unknown_pools": [],
    }
    sel = soup.find("div", class_="raceMeeting_select")
    head = clean(sel.find("span")) if sel and sel.find("span") else ""
    m = re.search(r"賽事日期:\s*(\d{1,2}/\d{1,2}/\d{4})\s*(\S+)?", head)
    if m:
        out["meeting_date"] = _ddmmyyyy_to_iso(m.group(1))
        out["meeting_venue"] = VENUE_MAP.get((m.group(2) or "").strip())
    if out["meeting_venue"] is None:
        out["meeting_venue"] = _nav_venue(soup)
    out["meeting_race_nos"] = _race_nos_from_nav(soup)
    out["dropdown_dates"] = parse_dropdown_dates(html)

    perf = soup.find("div", class_="performance")
    out["cancelled"] = "宣佈取消" in text and perf is None   # 此場賽事宣佈取消
    if "沒有相關資料" in text and not perf and not out["cancelled"]:
        out["no_data"] = True
        return out

    race = _parse_race_tab(soup)
    if race.get("race_no") is None:
        race["race_no"] = _current_race_no(soup)
    race["date"] = out["meeting_date"]
    race["venue"] = out["meeting_venue"]

    runners = []
    table = perf.find("table") if perf else None
    if table is not None:
        cols = _results_header(table)
        out["columns"] = cols
        out["full_table"] = "獨贏賠率" in cols and "檔位" in cols
        tbody = table.find("tbody")
        for tr in (tbody.find_all("tr", recursive=False) if tbody else []):
            r = _parse_runner_row(tr, cols)
            if not r.get("horse_no") and not r.get("horse_id"):
                continue
            status = r.get("finish_status") or ""
            for w in DEAD_HEAT_WORDS:
                if w in status and w != "DH":
                    status = re.sub(r"\s*" + w, " DH", status).strip()
            r["finish_status"] = status
            r["finish_pos"] = _finish_pos(status)
            withdrawn = status.upper() in WITHDRAWN_STATUSES or status.upper().startswith("W")
            if not status and out["full_table"] and r.get("draw") is None and r.get("finish_time_s") is None:
                withdrawn = True  # blank placing + all '---' = withdrawn (seen on same-day pages)
            r["scratched"] = 1 if withdrawn else 0
            r["lbw_num"] = parse_lbw(r.get("lbw"), r["finish_pos"]) if r["finish_pos"] else None
            r.pop("_horse_no_raw", None)
            runners.append(r)
    # dead heat margin '平頭馬' = same margin as the horse it tied with
    for r in runners:
        if r["finish_pos"] and r["lbw_num"] is None and r.get("lbw") in DEAD_HEAT_WORDS:
            tied = [x["lbw_num"] for x in runners if x is not r and x["finish_pos"] == r["finish_pos"]
                    and x["lbw_num"] is not None]
            r["lbw_num"] = 0.0 if r["finish_pos"] == 1 else (tied[0] if tied else None)
    out["runners"] = runners
    race["field_size"] = sum(1 for r in runners if not r["scratched"]) if runners else None

    # dividends
    divs, unknown = _parse_dividends(soup)
    out["dividends"] = divs
    out["unknown_pools"] = unknown
    out["race"] = race
    # abandoned / void race: nobody placed and every pool refunded (e.g. 2023-10-08 ST typhoon)
    # (multi-race pools may still pay with an 'F' leg, e.g. 2025-11-15 ST R8 declared VOID)
    out["abandoned"] = bool(runners) and not any(r["finish_pos"] for r in runners) and \
        any(d["pool"] == "WIN" and d["combination"] == "REFUND" for d in divs)
    # void race: cancelled before running (page has only dividends: single-race pools
    # refunded, multi-race pools may pay with 'F' for the void leg) or abandoned.
    out["void"] = bool(out["cancelled"] or out["abandoned"])
    out["complete"] = bool(out["full_table"] and runners and any(d["pool"] == "WIN" for d in divs)
                           and any(r["finish_pos"] == 1 for r in runners))
    return out


def _dividend_per10(raw: str):
    """'1,602.00' -> 1602.0 ; '144,163.30/$1.0' (quoted per $1 unit) -> 1441633.0 per $10."""
    m = re.fullmatch(r"([\d,]+(?:\.\d+)?)\s*/\s*\$\s*([\d.]+)", clean(raw))
    if m:
        unit = float(m.group(2))
        return round(float(m.group(1).replace(",", "")) * 10.0 / unit, 2) if unit else None
    return to_float(raw)


def _parse_dividends(soup):
    divs, unknown = [], []
    box = soup.find("div", class_="dividend_tab")
    if not box:
        return divs, unknown
    table = box.find("table")
    if not table:
        return divs, unknown
    tbody = table.find("tbody")
    pool = None
    seen = set()
    for tr in (tbody.find_all("tr") if tbody else []):
        tds = tr.find_all("td", recursive=False)
        if len(tds) >= 3:
            pool_name = clean(tds[0])
            if pool_name.replace(" ", "").startswith(SKIP_POOLS):
                pool = None
                continue
            pool = pool_code(pool_name)
            if pool is None:
                unknown.append(pool_name)
                pool = "?" + pool_name
            combo_td, div_td = tds[1], tds[2]
        elif len(tds) == 2 and pool:
            combo_td, div_td = tds[0], tds[1]
        else:
            continue
        combo_raw = clean(combo_td)
        div_raw = clean(div_td)
        if not combo_raw:
            continue
        if REFUND_TEXT in div_raw:
            # whole pool refunded: stored as combination 'REFUND', dividend = stake back (10 per $10)
            combo, dividend = "REFUND", 10.0
        else:
            combo = normalise_combination(pool, combo_raw)
            dividend = _dividend_per10(div_raw)
        key = (pool, combo)
        if key in seen:
            # identical (pool, combination) repeated - keep first, never silently merge
            continue
        seen.add(key)
        divs.append({"pool": pool, "combination": combo, "dividend": dividend,
                     "combination_raw": combo_raw, "dividend_raw": div_raw})
    return divs, unknown


# ---------------------------------------------------------------- racecard
RACECARD_COLS = {
    "馬匹編號": "horse_no",
    "6次近績": "last6",
    "馬名": "horse",
    "烙號": "brand",
    "負磅": "weight",
    "騎師": "jockey",
    "檔位": "draw",
    "練馬師": "trainer",
    "評分": "rating",
    "評分+/-": "rating_change",
    "排位體重": "body_weight",
    "排位體重+/-": "body_weight_change",
    "馬齡": "age",
    "性別": "sex",
    "上賽距今日數": "days_since_last",
    "配備": "gear",
    "國際評分": "intl_rating",
    "今季獎金": "season_stakes",
    "優先參賽次序": "priority",
    "可能超磅": "overweight",
}


def parse_racecard_info(text: str) -> dict:
    """The 4-line race header block of a racecard page."""
    out = {}
    lines = [clean(x) for x in re.split(r"\n", text) if clean(x)]
    blob = " | ".join(lines)
    m = re.search(r"第\s*(\d+)\s*場\s*-\s*([^|]+)", blob)
    if m:
        out["race_no"] = int(m.group(1))
        out["race_name"] = m.group(2).strip()
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日,\s*星期.,\s*(\S+?),\s*(\d{1,2}):(\d{2})", blob)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out["date"] = f"{y:04d}-{mo:02d}-{d:02d}"
        out["venue"] = VENUE_MAP.get(m.group(4))
        out["post_time"] = f"{out['date']}T{int(m.group(5)):02d}:{m.group(6)}"
    for ln in lines:
        if "米" in ln and ("草地" in ln or "全天候" in ln):
            out.update({k: v for k, v in parse_track(ln).items()})
            md = re.search(r"(\d{3,4})米", ln)
            out["distance"] = int(md.group(1)) if md else None
            parts = [p.strip() for p in ln.split(",")]
            tail = [p for p in parts[1:] if not re.search(r"米|賽道", p)]
            out["going"] = tail[-1] if tail else None
        if ln.startswith("獎金"):
            mp = re.search(r"\$([\d,]+)", ln)
            out["prize"] = int(mp.group(1).replace(",", "")) if mp else None
            mr = re.search(r"評分:\s*([\d]+-[\d]+)", ln)
            out["rating_band"] = mr.group(1) if mr else None
            parts = [p.strip() for p in ln.split(",")]
            out["race_class"] = parts[-1] if len(parts) >= 2 and parts[-1] not in ("-", "") else None
    return out


def parse_racecard(html: str) -> dict:
    soup = _soup(html)
    out = {"race": {}, "runners": [], "meeting_race_nos": _race_nos_from_nav(soup),
           "schedule": [], "no_data": False}
    # race header block
    info_div = None
    for div in soup.find_all("div", class_="f_fs13"):
        t = div.get_text("\n", strip=True)
        if re.search(r"第\s*\d+\s*場", t) and "年" in t and "月" in t:
            info_div = div
            break
    race = parse_racecard_info(info_div.get_text("\n", strip=True)) if info_div else {}
    if race.get("venue") is None:
        race["venue"] = _nav_venue(soup)
    out["race"] = race

    # meeting schedule (date_title buttons)
    year = int(race["date"][:4]) if race.get("date") else None
    dt = soup.find("div", class_="date_title")
    if dt:
        for a in dt.find_all("a"):
            t = clean(a)
            m = re.search(r"(\d{1,2})月(\d{1,2})日\s*-\s*(\S+)", t)
            href = a.get("href")
            d = None
            if href and _qs(href, "racedate"):
                rd = _qs(href, "racedate").replace("-", "/")
                p = rd.split("/")
                if len(p) == 3:
                    d = f"{int(p[0]):04d}-{int(p[1]):02d}-{int(p[2]):02d}" if len(p[0]) == 4 else _ddmmyyyy_to_iso(rd)
            elif m and year:
                d = f"{year:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            if m:
                out["schedule"].append({"date": d, "venue": VENUE_MAP.get(m.group(3)),
                                        "current": href is None})

    table = soup.find("table", class_="starter")
    if table is None:
        out["no_data"] = True
        return out
    thead = table.find("thead")
    cols = [_header_key(td) for td in thead.find_all("td")] if thead else []
    out["columns"] = cols
    runners = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < len(cols) - 2:
            continue
        r = {}
        scratched = "退出" in clean(tr)
        for key, td in zip(cols, tds):
            f = RACECARD_COLS.get(key)
            if not f:
                continue
            val = clean(td)
            if f == "horse":
                a = td.find("a")
                r["horse_name"] = clean(a) if a else val.replace("(退出)", "").strip()
                r["_brand_link"] = _brand_from_horse_link(a)
            elif f == "brand":
                r["horse_id"] = val if re.fullmatch(r"[A-Z]{1,2}\d{2,4}", val) else None
            elif f == "jockey":
                a = td.find("a")
                r["jockey_id"] = _qs(a.get("href", ""), "jockeyid") if a else None
                mm = re.search(r"\(([+-]?\d+)\)\s*$", val)
                r["jockey_allowance"] = abs(int(mm.group(1))) if mm else 0
                name = re.sub(r"\s*\([+-]?\d+\)\s*$", "", val)
                r["jockey_name"] = name if name and name != "-" else None
            elif f == "trainer":
                a = td.find("a")
                r["trainer_id"] = _qs(a.get("href", ""), "trainerid") if a else None
                r["trainer_name"] = val if val and val != "-" else None
            elif f in ("horse_no", "weight", "draw", "rating", "rating_change", "body_weight",
                       "age", "days_since_last", "body_weight_change", "intl_rating", "priority"):
                r[f] = to_int(val)
            elif f == "season_stakes":
                r[f] = to_int(val)
            elif f in ("last6", "gear", "sex", "overweight"):
                r[f] = val if val and val != "-" else None
        if not r.get("horse_id"):
            r["horse_id"] = r.get("_brand_link")
        r.pop("_brand_link", None)
        r["scratched"] = 1 if scratched else 0
        if r.get("horse_no") is None:
            continue
        runners.append(r)
    out["runners"] = runners
    race["field_size"] = sum(1 for r in runners if not r["scratched"])
    return out
