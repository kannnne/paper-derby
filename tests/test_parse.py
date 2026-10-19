"""Parser tests against saved HKJC pages (values checked by eye against the page text)."""
from pathlib import Path

import pytest

from hkjc.parse import (normalise_combination, parse_dropdown_dates, parse_lbw, parse_racecard,
                        parse_results, parse_time, pool_code)

FX = Path(__file__).parent / "fixtures"


def load(name):
    return (FX / name).read_text(encoding="utf-8")


def runner(res, horse_no):
    return next(r for r in res["runners"] if r["horse_no"] == horse_no)


def divs(res, pool):
    return [(d["combination"], d["dividend"]) for d in res["dividends"] if d["pool"] == pool]


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize("s,exp", [
    ("1:40.50", 100.5), ("59.80", 59.8), ("(1:16.63)", 76.63), ("1.38.91", 98.91),
    ("---", None), ("", None),
])
def test_parse_time(s, exp):
    assert parse_time(s) == exp


@pytest.mark.parametrize("s,pos,exp", [
    ("---", 1, 0.0), ("---", None, None), ("1-1/4", 2, 1.25), ("13-1/4", 12, 13.25),
    ("3/4", 2, 0.75), ("N", 2, 0.05), ("NOSE", 2, 0.05), ("SH", 2, 0.1), ("HD", 2, 0.2),
    ("NK", 2, 0.3), ("鼻", 2, 0.05), ("短頭", 2, 0.1), ("頭", 2, 0.2), ("頸", 2, 0.3),
    ("2", 4, 2.0), ("10-1/4", 11, 10.25),
])
def test_parse_lbw(s, pos, exp):
    assert parse_lbw(s, pos) == exp


def test_pool_and_combination_normalisation():
    assert pool_code("位置Q") == "QPL"
    assert pool_code("單T") == "TRI"
    assert pool_code("孖寶") == "DBL"
    assert pool_code("六環彩") == "SIXUP"
    assert normalise_combination("QPL", "12,1") == "1,12"
    assert normalise_combination("TRI", "12,1,11") == "1,11,12"
    assert normalise_combination("FF", "12, 7,1,11") == "1,7,11,12"
    assert normalise_combination("TCE", "1,12,11") == "1,12,11"     # ordered pool kept
    assert normalise_combination("QTT", "1,12,11,7") == "1,12,11,7"
    assert normalise_combination("DBL", "4/7") == "4/7"
    assert normalise_combination("TT", "3,1,2/6,5,4/9,8,7") == "1,2,3/4,5,6/7,8,9"


# ------------------------------------------------------------------ results 2026-10-14 HV R1
@pytest.fixture(scope="module")
def hv1():
    return parse_results(load("results_20261014_HV_R1.html"))


def test_results_race_header(hv1):
    race = hv1["race"]
    assert hv1["meeting_date"] == "2026-10-14"
    assert hv1["meeting_venue"] == "HV"
    assert hv1["meeting_race_nos"] == list(range(1, 10))
    assert race["race_no"] == 1
    assert race["season_race_index"] == 99
    assert race["race_class"] == "第五班"
    assert race["rating_band"] == "40-0"
    assert race["distance"] == 1650
    assert race["surface"] == "TURF"
    assert race["course"] == "A"
    assert race["going"] == "好地"
    assert race["race_name"] == "德己立讓賽"
    assert race["prize"] == 875000
    assert race["winning_time_s"] == 100.5
    assert race["sectional_times"] == [28.29, 24.21, 24.13, 23.87]
    assert race["field_size"] == 12
    assert hv1["complete"] and hv1["full_table"] and not hv1["no_data"]


def test_results_runners(hv1):
    assert len(hv1["runners"]) == 12
    w = runner(hv1, 1)
    assert w == {**w, "finish_pos": 1, "finish_status": "1", "horse_id": "L045",
                 "horse_name": "開心旺財", "jockey_id": "PZ", "jockey_name": "潘頓",
                 "trainer_id": "WDJ", "trainer_name": "韋達", "act_weight": 135,
                 "body_weight": 1057, "draw": 7, "lbw": "---", "lbw_num": 0.0,
                 "running_pos": "1 1 1 1", "finish_time_s": 100.5, "win_odds": 4.2, "scratched": 0}
    r12 = runner(hv1, 12)
    assert (r12["finish_pos"], r12["horse_id"], r12["jockey_id"], r12["trainer_id"]) == (2, "K216", "KRW", "EDJ")
    assert (r12["lbw_num"], r12["running_pos"], r12["finish_time_s"], r12["win_odds"]) == (1.25, "3 3 4 2", 100.7, 8.7)
    last = runner(hv1, 2)
    assert (last["finish_pos"], last["horse_id"], last["lbw"], last["lbw_num"], last["win_odds"]) == \
        (12, "K265", "13-1/4", 13.25, 27.0)
    assert sorted(r["finish_pos"] for r in hv1["runners"]) == list(range(1, 13))


def test_results_dividends(hv1):
    assert divs(hv1, "WIN") == [("1", 42.0)]
    assert divs(hv1, "PLA") == [("1", 17.5), ("12", 28.0), ("11", 14.0)]
    assert divs(hv1, "QIN") == [("1,12", 206.5)]
    assert divs(hv1, "QPL") == [("1,12", 73.0), ("1,11", 37.0), ("11,12", 59.0)]
    assert divs(hv1, "FCT") == [("1,12", 394.0)]
    assert divs(hv1, "TCE") == [("1,12,11", 1602.0)]
    assert divs(hv1, "TRI") == [("1,11,12", 246.0)]
    assert divs(hv1, "FF") == [("1,7,11,12", 1739.0)]
    assert divs(hv1, "QTT") == [("1,12,11,7", 23259.0)]
    assert hv1["unknown_pools"] == []


def test_results_dropdown(hv1):
    dd = parse_dropdown_dates(load("results_20261014_HV_R1.html"))
    assert dd[0] == "2026-10-21"          # upcoming meeting is listed first
    assert "2026-10-14" in dd and "2025-07-26" in dd
    assert dd[-1] == "2025-07-26"


def test_results_no_data():
    r = parse_results(load("results_nodata_20230911.html"))
    assert r["no_data"] is True
    assert r["runners"] == [] and r["dividends"] == []


# ------------------------------------------------------------------ racecard
@pytest.fixture(scope="module")
def card():
    return parse_racecard(load("racecard_20261021_HV_R1.html"))


def test_racecard_header(card):
    race = card["race"]
    assert race == {**race, "date": "2026-10-21", "venue": "HV", "race_no": 1,
                    "post_time": "2026-10-21T18:40", "distance": 1650, "surface": "TURF",
                    "course": "B", "race_class": "第四班", "rating_band": "60-40",
                    "prize": 1170000, "race_name": "競賽精神讓賽"}
    assert card["meeting_race_nos"] == list(range(1, 10))
    assert {"date": "2026-10-21", "venue": "HV", "current": True} in card["schedule"]
    assert {"date": "2026-10-19", "venue": "ST", "current": False} in card["schedule"]


def test_racecard_runners(card):
    assert len(card["runners"]) == 11
    r1 = card["runners"][0]
    assert r1 == {**r1, "horse_no": 1, "horse_id": "K362", "horse_name": "太子",
                  "jockey_id": "OJM", "jockey_name": "奧爾民", "trainer_id": "NPC",
                  "trainer_name": "伍鵬志", "weight": 135, "draw": 7, "rating": 60,
                  "rating_change": -1, "body_weight": 1076, "last6": "7/12/4/2/8/5",
                  "gear": "B", "age": 5, "sex": "閹", "days_since_last": 14, "scratched": 0}
    r3 = card["runners"][2]   # apprentice claim shown as '黃寶妮 (-7)'
    assert (r3["horse_id"], r3["jockey_id"], r3["jockey_name"], r3["jockey_allowance"]) == \
        ("J079", "WPN", "黃寶妮", 7)
    assert (r3["rating"], r3["rating_change"], r3["gear"], r3["days_since_last"]) == (58, -2, "TT", 14)
    assert all(r["scratched"] == 0 for r in card["runners"])


def test_racecard_scratched():
    c = parse_racecard(load("racecard_20261019_ST_R8.html"))
    race = c["race"]
    assert (race["date"], race["venue"], race["race_no"], race["post_time"]) == \
        ("2026-10-19", "ST", 8, "2026-10-19T16:05")
    assert (race["race_class"], race["rating_band"], race["course"], race["going"]) == \
        ("二級賽", None, "C+3", "好地")
    assert c["meeting_race_nos"] == list(range(1, 12))
    scr = [r for r in c["runners"] if r["scratched"]]
    assert [(r["horse_no"], r["horse_id"], r["horse_name"]) for r in scr] == [(4, "K296", "白鷺金剛")]
    assert scr[0]["jockey_id"] is None and scr[0]["draw"] is None
    assert len(c["runners"]) == 12 and race["field_size"] == 11


# ------------------------------------------------------------------ edge cases (older seasons)
def test_awt_race_sha_tin_2023():
    r = parse_results(load("results_20230917_ST_R5_awt.html"))
    race = r["race"]
    assert (race["date"], race["venue"], race["race_no"], race["season_race_index"]) == ("2023-09-17", "ST", 5, 23)
    assert (race["surface"], race["course"], race["distance"], race["race_class"], race["rating_band"]) == \
        ("AWT", None, 1200, "第四班", "60-35")
    assert race["race_name"] == "東華三院挑戰盃（讓賽）"
    assert (race["prize"], race["winning_time_s"], race["sectional_times"]) == (1170000, 69.35, [23.26, 22.38, 23.71])
    w = runner(r, 10)
    assert (w["finish_pos"], w["horse_id"], w["draw"], w["running_pos"], w["finish_time_s"], w["win_odds"]) == \
        (1, "G020", 12, "9 6 1", 69.35, 38.0)
    assert (runner(r, 8)["lbw"], runner(r, 8)["lbw_num"]) == ("頸位", 0.3)     # neck
    assert divs(r, "WIN") == [("10", 387.0)]
    assert divs(r, "QPL") == [("8,10", 285.0), ("2,10", 284.0), ("2,8", 111.5)]
    assert divs(r, "QTT") == [("10,8,2,4", 245751.0)]
    assert divs(r, "DBL") == [("5/10", 5035.0), ("5/8", 188.5)]      # 第四口孖寶 (dead heat in leg 1)
    assert divs(r, "TBL") == [("1/5/10", 92348.0), ("1/5/8", 6921.0)]


def test_dead_heat_for_third():
    r = parse_results(load("results_20230910_ST_R7_deadheat.html"))
    a, b = runner(r, 5), runner(r, 7)
    assert (a["finish_status"], a["finish_pos"], a["lbw_num"]) == ("3 DH", 3, 1.5)
    assert (b["finish_status"], b["finish_pos"], b["lbw_num"]) == ("3 DH", 3, 1.5)
    assert [x["finish_pos"] for x in r["runners"]] == [1, 2, 3, 3, 5, 6, 7, 8, 9, 10]
    assert divs(r, "PLA") == [("8", 12.0), ("9", 49.5), ("5", 17.5), ("7", 10.1)]
    assert divs(r, "QPL") == [("8,9", 87.5), ("5,8", 25.5), ("7,8", 10.5), ("5,9", 111.0), ("7,9", 43.0),
                              ("5,7", 29.5)]
    assert divs(r, "TCE") == [("8,9,5", 1068.0), ("8,9,7", 424.0)]
    assert divs(r, "TRI") == [("5,8,9", 336.0), ("7,8,9", 95.0)]
    assert divs(r, "QTT") == [("8,9,5,7", 2806.0), ("8,9,7,5", 2426.0)]
    assert divs(r, "FF") == [("5,7,8,9", 379.0)]


def test_withdrawn_horses():
    r = parse_results(load("results_20230910_ST_R8_withdrawn.html"))
    assert r["race"]["field_size"] == 10 and len(r["runners"]) == 12
    for no, hid in ((6, "E009"), (7, "E043")):
        x = runner(r, no)
        assert (x["finish_status"], x["finish_pos"], x["scratched"], x["horse_id"]) == ("WV", None, 1, hid)
        assert x["draw"] is None and x["win_odds"] is None and x["finish_time_s"] is None
    assert x["jockey_id"] is not None and x["act_weight"] == 132
    last = runner(r, 4)
    assert (last["finish_pos"], last["lbw_num"], last["finish_time_s"]) == (10, 16.25, 85.34)
    assert divs(r, "DT") == [("5,8,9/1,8,9", 35551.0), ("7,8,9/1,8,9", 11210.0)]


def test_fell_and_unseated():
    r = parse_results(load("results_20231111_ST_R7_fell_unseated.html"))
    assert (r["race"]["course"], r["race"]["race_class"], r["race"]["rating_band"]) == ("A+3", "第一班", "110-85")
    fe, ur = runner(r, 1), runner(r, 7)
    assert (fe["finish_status"], fe["finish_pos"], fe["scratched"], fe["win_odds"], fe["running_pos"]) == \
        ("FE", None, 0, 4.5, "7 6 6")
    assert (ur["finish_status"], ur["finish_pos"], ur["scratched"], ur["win_odds"]) == ("UR", None, 0, 14.0)
    assert fe["lbw_num"] is None and fe["finish_time_s"] is None
    assert r["race"]["field_size"] == 8          # non-finishers still count as starters
    assert divs(r, "WIN") == [("5", 115.5)]


def test_dnf_and_dead_heat_hv():
    r = parse_results(load("results_20240417_HV_R9_dnf.html"))
    d = runner(r, 5)
    assert (d["finish_status"], d["finish_pos"], d["scratched"], d["win_odds"], d["running_pos"]) == \
        ("DNF", None, 0, 6.2, "7 7 12")
    assert (runner(r, 6)["finish_status"], runner(r, 10)["finish_status"]) == ("3 DH", "3 DH")
    assert len(divs(r, "PLA")) == 4 and len(divs(r, "QPL")) == 6
    assert divs(r, "SIXUP") == [("9,11/3,4/1,3,8/2,6,10/3,8/4,11", 2374.0), ("11/3/3/2/8/11", 2464371.0)]
    assert not any(x["pool"].startswith("?") for x in r["dividends"])   # 騎師王/練馬師王 skipped


def test_pulled_up():
    r = parse_results(load("results_20240121_ST_R2_pulledup.html"))
    p = runner(r, 11)
    assert (p["finish_status"], p["finish_pos"], p["win_odds"], p["running_pos"], p["scratched"]) == \
        ("PU", None, 44.0, "2 2 14", 0)
    assert r["race"]["field_size"] == 14


def test_griffin_race():
    r = parse_results(load("results_20240414_ST_R1_griffin.html"))
    race = r["race"]
    assert (race["race_class"], race["distance"], race["rating_band"], race["race_name"]) == \
        ("新馬賽", 1000, None, "王府井平磅賽")
    assert (race["season_race_index"], race["prize"], race["winning_time_s"]) == (578, 950000, 56.42)
    assert r["meeting_race_nos"] == list(range(1, 12))
    w = runner(r, 6)
    assert (w["horse_id"], w["finish_time_s"], w["win_odds"]) == ("J019", 56.42, 8.9)
    assert divs(r, "QTT") == [("6,7,4,2", 3135.0)]


def test_cancelled_race():
    r = parse_results(load("results_20241113_HV_R7_cancelled.html"))
    assert r["cancelled"] and r["void"] and not r["complete"]
    assert r["race"]["race_no"] == 7 and r["runners"] == []
    assert divs(r, "WIN") == [("REFUND", 10.0)]
    assert divs(r, "DBL") == [("7/F", 27.0)]
    assert divs(r, "DT") == [("7,10,11/F", 2672.0)]
    assert divs(r, "TT") == [("1,5,7/7,10,11/F", 256366.0)]


def test_abandoned_meeting_race():
    r = parse_results(load("results_20231008_ST_R1_abandoned.html"))
    assert r["abandoned"] and r["void"] and not r["complete"]
    assert r["race"]["season_race_index"] == 0 and len(r["runners"]) == 8
    assert all(x["finish_pos"] is None for x in r["runners"])
    assert {d["combination"] for d in r["dividends"]} == {"REFUND"}


def test_void_race_with_multi_leg_payout():
    r = parse_results(load("results_20251115_ST_R8_void.html"))
    assert r["void"] and r["abandoned"]
    assert {x["finish_status"] for x in r["runners"]} == {"VOID", "WV"}   # WV = withdrawn pre-race
    assert divs(r, "DBL") == [("9/F", 32.0)]


def test_dividend_quoted_per_dollar():
    r = parse_results(load("results_20231022_ST_R9_qtt_per1.html"))
    assert divs(r, "QTT") == [("12,10,3,14", 1441633.0)]   # '144,163.30/$1.0' -> per $10


def test_relegated_first_past_post():
    r = parse_results(load("results_20241120_HV_R7_relegated.html"))
    w, s = runner(r, 4), runner(r, 12)
    assert (w["finish_pos"], w["lbw"], w["lbw_num"], w["finish_time_s"]) == (1, "-短馬頭位", 0.0, 99.62)
    assert (s["finish_pos"], s["lbw"], s["lbw_num"], s["finish_time_s"]) == (2, "---", 0.0, 99.6)
    assert divs(r, "WIN") == [("4", 345.0)]


def test_took_no_part_is_a_starter():
    r = parse_results(load("results_20240218_ST_R8_tnp.html"))
    t = runner(r, 11)
    assert (t["finish_status"], t["finish_pos"], t["scratched"], t["win_odds"]) == ("TNP", None, 0, 2.2)


def test_disqualified_after_payout():
    r = parse_results(load("results_20250212_ST_R8_disq.html"))
    d = runner(r, 12)
    assert (d["finish_status"], d["finish_pos"], d["scratched"], d["finish_time_s"]) == ("DISQ", None, 0, 70.18)
    # dividends were paid on the original placings (12 second) -> settle from dividends, not finish_pos
    assert ("12", 17.5) in divs(r, "PLA")


def test_wxnr_non_runner():
    r = parse_results(load("results_20250219_HV_R2_wxnr.html"))
    x = runner(r, 3)
    assert (x["finish_status"], x["scratched"], x["finish_pos"], x["win_odds"]) == ("WXNR", 1, None, None)
    assert r["race"]["field_size"] == 11


def test_same_day_withdrawn_blank_status():
    r = parse_results(load("results_20261019_ST_R8_sameday_withdrawn.html"))
    x = runner(r, 4)
    assert (x["horse_id"], x["finish_status"], x["scratched"], x["jockey_id"], x["trainer_id"]) == \
        ("K296", "", 1, "FEL", "NM")
    assert (r["race"]["race_class"], r["race"]["rating_band"], r["race"]["course"]) == ("二級賽", None, "C+3")
    assert r["race"]["field_size"] == 11 and r["complete"]
    assert divs(r, "WIN") == [("7", 88.5)]


def test_preliminary_reduced_table():
    r = parse_results(load("results_20261019_ST_R9_prelim.html"))
    assert r["columns"] == ["名次", "馬號", "馬名", "騎師"]
    assert not r["full_table"] and not r["complete"]
    assert [(x["finish_pos"], x["horse_no"], x["horse_id"]) for x in r["runners"]] == \
        [(1, 6, "L336"), (2, 1, "L232"), (3, 8, "L519"), (4, 12, "K283")]
    assert divs(r, "WIN") == [("6", 93.5)]
    assert divs(r, "PLA") == [("6", 26.0), ("1", 14.0), ("8", 15.0)]
