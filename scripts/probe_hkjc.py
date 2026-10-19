"""Cloud connectivity probe: can this machine reach HKJC results, racecard and odds API?"""
import json
import sys

import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
ok = True
for name, url in [
    ("results", "https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=2026/10/19&Racecourse=ST&RaceNo=1"),
    ("racecard", "https://racing.hkjc.com/zh-hk/local/information/racecard"),
]:
    try:
        r = requests.get(url, headers=UA, timeout=30)
        good = r.status_code == 200 and "名次" in r.text if name == "results" else r.status_code == 200 and "排位表" in r.text
        print(f"{name}: HTTP {r.status_code} final={r.url} bytes={len(r.content)} ok={good}")
        ok &= good
    except Exception as e:
        print(f"{name}: ERROR {e}")
        ok = False
try:
    from hkjc import odds
    m = odds.fetch_meeting()
    print("odds meeting:", json.dumps({k: m.get(k) for k in ("date", "venue", "status", "n_races", "first_post_time")}, ensure_ascii=False))
    p = odds.fetch_pools(m["date"], m["venue"], 1, pools=("WIN",))
    print("odds WIN race 1 entries:", len(p.get("WIN", {})), "meta:", getattr(p, "meta", None))
except Exception as e:
    print("odds: ERROR", e)
    ok = False
print("PROBE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
