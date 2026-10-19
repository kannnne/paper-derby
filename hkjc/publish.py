"""Public dashboard exporter (GitHub Pages static site in ``site/``).  SIMULATION ONLY.

    export_site(conn, out_dir=None) -> list[Path]
    publish(conn, message=None) -> dict          # export + git commit (+ push when enabled)

Files written (formats per SPEC_V2 "Dashboard"; additions are marked [+] below):

``site/data/index.json``
    {updated_at, disclaimer, next_meeting:{date, venue, first_post_time, [+]venue_zh, [+]n_races},
     meetings:[{date, venue, status, file, [+]venue_zh, [+]n_races, [+]has_scores, [+]has_decisions,
                [+]has_picks}],
     bots:[{id, name, style, description, bankroll, start_bankroll,
            live:{n_meetings, stake, returns, profit, roi, hit_rate, [+]n_tickets, [+]open_stake},
            [+]no (saddle-cloth number 1..10)}],
     [+]score_explanation (中文, from scores.SCORE_EXPLANATION_ZH when available),
     [+]sub_score_labels {key: 中文}, [+]start_bankroll, [+]simulation_only: true}
    status: 'pre' | 'results' | [+]'abandoned'.

``site/data/meetings/YYYY-MM-DD_VV.json``
    {date, venue, venue_zh, status, generated_at, odds_snapshot_at, model_version,
     [+]scores_run_id, [+]decisions_run_id,
     races:[{race_no, post_time, race_class, distance, surface, course, going, race_name, [+]rating_band,
             horses:[{horse_no, horse_name, jockey, trainer, draw, weight, score, rank, p_win, p_place,
                      win_odds, pla_odds, sub_scores:{form, jockey, trainer, combo, draw, fit, class, market},
                      reasons:[...], scratched,
                      result:{finish_pos, final_win_odds, [+]finish_status, [+]lbw}|null}],
             decisions:[{bot_id, action, structure, selections, stake, reason, payout|null, profit|null,
                         [+]race_no, [+]ticket_status, [+]parlay_leg (row is one leg of an all-up ticket whose
                         P&L is reported once, on the race_no-0 meeting decision)}],
             result:{placings:[{pos, horse_no, horse_name, [+]jockey, [+]finish_status, [+]win_odds}],
                     dividends:{WIN, PLA, QIN, QPL, TRI, FF, [+]FCT, [+]TCE, [+]QTT: [[comb, div]]}}|null}],
     [+]meeting_decisions: [... same shape as decisions, race_no 0 = 過關 decisions ...],
     bots:[{bot_id, stake, payout, profit, [+]n_tickets, [+]settled,
            tickets:[{strategy, legs, total_stake, payout, reason, [+]profit, [+]status, [+]n_combos,
                      [+]unit_stake, [+]ticket_id, [+]race_no (0 = 過關 / multi-race ticket),
                      [+]structure (e.g. "QPL 複式3", "PLA 6x63"), [+]selections,
                      [+]leg_hits:[bool|null per leg], [+]hit (bool once settled, else null)}],
            [+]skips:[{race_no (0 = 過關決定), structure, reason}]}],
     [+]picks: 今日精選 | null (see below)}
    payout/profit are null while a ticket is open.  Void (replaced) tickets are left out.
    (The old 「賽後補做」 ``note`` field is gone: meeting files carry no such notice any more.)

    [+] ``picks`` (今日精選, 第四版「每種玩法各一個」): the picks of hkjc.picks.make_picks as stored
    (table ``picks``) by the pre-meeting run whose ``daily_picks`` robot decisions are current
    (fallback: the latest stored picks), internal fields stripped, results added:
      {title:"今日精選", generated_at, method_note, odds_note, legend, sections:[{section, section_zh}],
       total_stake, n_items, n_star, total_payout|null, total_profit|null,
       items:[{rank, section (PLA_6x63 QPL_2x3 WIN PLA QIN QPL FCT TRI TCE FF QTT DBL TBL), section_zh,
               star (⭐ 值博: passes the tuned threshold), kind:"single"|"multi_race"|"allup", race_no, races,
               post_time, pool, pool_zh, structure_zh, legs:[{race_no, bankers:[{horse_no, horse_name, score}],
               selections:[...], positions:[[horse_no...]]|null, [pool_zh], [hit]}], n_combos, unit_stake, stake,
               hit_prob, exp_roi, est_return_if_hit, odds_source:"即時賠率"|"估計賠率", grade:"精選", reason,
               result:{hit, payout, profit, detail, hypothetical}|null}],
       banker:{race_no, post_time, horse_no, horse_name, jockey, score, p_place, pla_odds, reason,
               result:{finish_pos, placed}|null}|null}
    Results come from the daily_picks robot's settled tickets (it buys every item); an item the robot
    could not buy (e.g. 資金不足) gets a hypothetical result from the published dividends
    (hypothetical=true) and keeps the totals null.  total_payout is set once every item is settled.
    picks is null when no picks were stored for the meeting.

[+] ``site/data/bots/<bot_id>.json``
    {bot_id, n_tickets, tickets:[{date, venue, venue_zh, ...same ticket fields as above}]}  (newest first;
    the robot page reads this one file instead of every meeting file)

``site/data/bot_history.json``
    {updated_at, start_bankroll, bots:[{bot_id, live:[{date, venue, stake, payout, profit, bankroll}]}]}
    live = one point per settled meeting (bankroll after that meeting, start_bankroll before the first).

Live paper results only: backtest figures (SPEC_V2 index.json ``bots[].backtest`` and the backtest
curves in bot_history.json) are deliberately NOT exported any more - the public dashboard shows
live (paper) results only.  ``data/bot_backtest.json`` (hkjc.bots.backtest_bots) stays an internal
research file and is never read here.

Only the latest run per meeting is exported: the latest post-meeting run when the meeting has
results and such a run produced data, else the latest pre-meeting run (robot decisions: each
robot's latest run).  A meeting file whose content is unchanged apart from generated_at is not
rewritten (keeps the site's git history small).  No personal data (paths,
e-mails, account names) is ever written: only public racing data and model/robot output.

Publishing: ``data/publish.json`` = {"enabled": false, "remote": "", "branch": "main"} (optional
"author_name"/"author_email" for the commit identity; defaults to a neutral bot identity so no
personal e-mail ends up in the public history).  ``site/`` is its own git repo.  publish() never
raises: any failure is logged as a warning and reported in the returned dict.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .db import ROOT

log = logging.getLogger("hkjc.publish")

HKT = ZoneInfo("Asia/Hong_Kong")
SITE_DIR = ROOT / "site"
PUBLISH_CONFIG = ROOT / "data" / "publish.json"
START_BANKROLL = 10_000.0
DEFAULT_PUBLISH_CONFIG = {"enabled": False, "remote": "", "branch": "main"}
DEFAULT_AUTHOR = ("Paper Derby Bot", "paper-bot@localhost.invalid")   # neutral, not a club name

VENUE_ZH = {"ST": "沙田", "HV": "跑馬地"}
DISCLAIMER = ("模擬研究，非投注建議；未滿18歲不得投注；博彩要有節制，"
              "如需協助可致電平和基金熱線 1834 633。")
SUB_SCORE_KEYS = ["form", "jockey", "trainer", "combo", "draw", "fit", "class", "market"]
SUB_SCORE_LABELS = {"form": "往績", "jockey": "騎師", "trainer": "練馬師", "combo": "騎練",
                    "draw": "檔位", "fit": "適應", "class": "班次負磅", "market": "市場"}
RESULT_POOLS = ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT"]

# Used only when hkjc/bots.py is unavailable (order = saddle-cloth number).
FALLBACK_BOTS = [
    {"id": "place_only", "name_zh": "位置專家", "style_zh": "只買位置",
     "description_zh": "只買單場位置，每場最多 2 隻；入三甲機率 × 位置賠率夠門檻先買。"},
    {"id": "fixed_combo", "name_zh": "固定組合", "style_zh": "過關組合",
     "description_zh": "每個賽馬日固定 1 條位置 6x63 過關 + 3 條位置Q 2x3 過關。"},
    {"id": "adaptive", "name_zh": "按場自動", "style_zh": "按場揀玩法",
     "description_zh": "每場按模型機率同賠率計預期回報，揀最好嘅玩法；唔夠門檻就唔買。"},
    {"id": "win_value", "name_zh": "獨贏價值", "style_zh": "獨贏",
     "description_zh": "勝率 × 獨贏賠率高過門檻先買獨贏。"},
    {"id": "qpl_box", "name_zh": "位置Q 複式", "style_zh": "位置Q",
     "description_zh": "模型前三名集中時，買位置Q 複式 3 隻。"},
    {"id": "qin_banker", "name_zh": "連贏膽拖", "style_zh": "連贏",
     "description_zh": "有明顯首選時，1 膽拖 N 腳連贏。"},
    {"id": "tri_banker", "name_zh": "單T 膽拖", "style_zh": "單T",
     "description_zh": "有兩隻強馬時，2 膽拖 N 腳單T。"},
    {"id": "pla_parlay", "name_zh": "位置過關", "style_zh": "位置過關",
     "description_zh": "揀最有信心嘅 3 場首選，位置 3x4 過關。"},
    {"id": "favourite", "name_zh": "跟大熱", "style_zh": "對照組",
     "description_zh": "對照組：每場大熱門獨贏 + 位置各 $10。"},
    {"id": "monkey", "name_zh": "隨機猴子", "style_zh": "對照組",
     "description_zh": "對照組：每場隨機揀一隻買位置 $10（以場次編號做種子，可重現）。"},
]

FALLBACK_SCORE_EXPLANATION = (
    "綜合評分（0–100）由模型估計嘅勝出機率轉換而成：以該馬勝率對比同場「平均馬」（1/出賽匹數）"
    "嘅倍數取對數再映射到 0–100，所以同場排序同勝率一致，而且唔同場次之間可以比較。"
    "50 分大約代表平均水平，每高約 15 分代表勝出機會大約倍增。"
    "子評分（往績、騎師、練馬師、騎練、檔位、適應、班次負磅、市場）係同場內相對分數，"
    "只用嚟解釋評分來源，唔係獨立預測。評分只係模型輸出，唔保證準確，並非投注建議。")


# ------------------------------------------------------------------ helpers
def _now() -> str:
    return datetime.now(HKT).replace(tzinfo=None).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(HKT).strftime("%Y-%m-%d")


def _clean(x):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None, floats rounded."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):   # numpy scalar
        try:
            x = x.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 4)
    return str(x)


def _f(x, nd=None):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, nd) if nd is not None else v
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return None if v is None else int(v)


def _json(s, default):
    if s is None:
        return default
    if isinstance(s, (list, dict)):
        return s
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


_TERMS = [(re.compile(r"(\d+)戰(\d+)上名"), r"\1戰\2次入三甲"), (re.compile("戰未上名"), "戰未入過三甲"),
          (re.compile("(近年|檔位)上名率"), r"\1入三甲率"), (re.compile("上名率"), "入三甲機率"),
          (re.compile("上名機會"), "入三甲機會")]


def _terms(obj):
    """Older DB text (reasons) used 上名率; the site now says 入三甲機率 / 入三甲率."""
    if isinstance(obj, str):
        for pat, rep in _TERMS:
            obj = pat.sub(rep, obj)
        return obj
    if isinstance(obj, list):
        return [_terms(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _terms(v) for k, v in obj.items()}
    return obj


def _write_json(path: Path, obj, ignore=()) -> Path:
    """Write ``obj`` as compact JSON.  ``ignore``: top-level keys (e.g. generated_at) that do
    not count as a change - when everything else equals the existing file it is left
    untouched, so re-publishing does not rewrite (and git-commit) every old meeting file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = _terms(_clean(obj))
    if ignore and path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and isinstance(obj, dict) and \
                    {k: v for k, v in old.items() if k not in ignore} == \
                    {k: v for k, v in obj.items() if k not in ignore}:
                return path
        except (OSError, ValueError):
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def _rid_prefix(date, venue):
    return f"{date.replace('-', '')}-{venue}-"


def _table_exists(conn, name) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _cols(conn, table) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


# ------------------------------------------------------------------ robots metadata
def bot_meta() -> list[dict]:
    """[{id, name, style, description, no}] from hkjc.bots.BOTS (fallback: SPEC_V2 table)."""
    src = FALLBACK_BOTS
    try:
        from . import bots as _bots   # written by another agent; optional
        if getattr(_bots, "BOTS", None):
            src = _bots.BOTS
    except Exception as e:   # pragma: no cover - depends on the other module's state
        log.debug("hkjc.bots unavailable (%s); using fallback robot list", e)
    out = []
    for i, b in enumerate(src, 1):
        out.append({"id": b["id"], "no": i,
                    "name": b.get("name_zh") or b.get("name") or b["id"],
                    "style": b.get("style_zh") or b.get("style") or "",
                    "description": b.get("description_zh") or b.get("description") or ""})
    return out


def sub_score_labels() -> dict:
    try:
        from . import scores as _scores
        names = getattr(_scores, "SUB_SCORE_NAMES_ZH", None)
        if names:
            return {k: str(names.get(k, SUB_SCORE_LABELS.get(k, k))) for k in SUB_SCORE_KEYS}
    except Exception as e:   # pragma: no cover
        log.debug("hkjc.scores unavailable (%s)", e)
    return dict(SUB_SCORE_LABELS)


def score_explanation() -> str:
    try:
        from . import scores as _scores
        txt = getattr(_scores, "SCORE_EXPLANATION_ZH", None)
        if txt:
            return str(txt)
    except Exception as e:   # pragma: no cover
        log.debug("hkjc.scores unavailable (%s)", e)
    return FALLBACK_SCORE_EXPLANATION


# ------------------------------------------------------------------ meetings / runs
def next_meeting(conn) -> dict | None:
    today = _today()
    r = conn.execute("SELECT date, venue, first_post_time, n_races FROM meetings WHERE date >= ? AND "
                     "COALESCE(status,'') NOT IN ('results','abandoned') ORDER BY date, venue LIMIT 1",
                     (today,)).fetchone()
    if not r:
        return None
    fpt = r["first_post_time"] or conn.execute(
        "SELECT MIN(post_time) FROM races WHERE date=? AND venue=?", (r["date"], r["venue"])).fetchone()[0]
    n = r["n_races"] or conn.execute("SELECT COUNT(DISTINCT race_no) FROM racecard WHERE date=? AND venue=?",
                                     (r["date"], r["venue"])).fetchone()[0]
    return {"date": r["date"], "venue": r["venue"], "venue_zh": VENUE_ZH.get(r["venue"], r["venue"]),
            "first_post_time": fpt, "n_races": _i(n)}


def _meeting_status(conn, date, venue) -> str:
    r = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?", (date, venue)).fetchone()
    st = r[0] if r else None
    if st == "results":
        return "results"
    if st == "abandoned":
        return "abandoned"
    return "pre"


def _pick_run(cands, status) -> str | None:
    """cands: [(run_id, kind, ts)].  Latest post run if results, else latest pre run, else latest."""
    if not cands:
        return None
    cands = sorted(cands, key=lambda c: (c[2] or ""), reverse=True)
    if status == "results":
        post = [c for c in cands if (c[1] or "").startswith("post")]
        if post:
            return post[0][0]
    pre = [c for c in cands if (c[1] or "").startswith("pre")]
    return (pre or cands)[0][0]


def _score_runs(conn, date, venue):
    return [(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT h.run_id, u.kind, COALESCE(u.created_at, MAX(h.created_at)) FROM horse_scores h "
        "LEFT JOIN runs u ON u.run_id=h.run_id WHERE h.race_id LIKE ? GROUP BY h.run_id",
        (_rid_prefix(date, venue) + "%",))]


def _decision_runs(conn, date, venue):
    return [(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT d.run_id, u.kind, COALESCE(u.created_at, MAX(d.created_at)) FROM bot_decisions d "
        "LEFT JOIN runs u ON u.run_id=d.run_id WHERE d.date=? AND d.venue=? GROUP BY d.run_id",
        (date, venue))]


def meetings_to_export(conn) -> list[tuple[str, str]]:
    keys = set()
    for r in conn.execute("SELECT DISTINCT date, venue FROM bot_decisions"):
        keys.add((r[0], r[1]))
    for r in conn.execute("SELECT DISTINCT substr(race_id,1,8), substr(race_id,10,2) FROM horse_scores"):
        d = r[0]
        keys.add((f"{d[:4]}-{d[4:6]}-{d[6:8]}", r[1]))
    nm = next_meeting(conn)
    if nm:
        keys.add((nm["date"], nm["venue"]))
    return sorted(keys, reverse=True)


# ------------------------------------------------------------------ meeting file
def _predictions(conn, run_ids, date, venue):
    """{race_id: {horse_no: (p_win, p_place, model_version)}} for the first run that has rows."""
    pref = _rid_prefix(date, venue) + "%"
    for rid in [r for r in run_ids if r]:
        rows = conn.execute("SELECT race_id, horse_no, p_win, p_place, model_version FROM predictions "
                            "WHERE run_id=? AND race_id LIKE ?", (rid, pref)).fetchall()
        if rows:
            return _pred_map(rows)
    r = conn.execute("SELECT p.run_id FROM predictions p JOIN runs u ON u.run_id=p.run_id "
                     "WHERE u.kind LIKE 'pre%' AND p.race_id LIKE ? ORDER BY u.created_at DESC LIMIT 1",
                     (pref,)).fetchone()
    if r:
        rows = conn.execute("SELECT race_id, horse_no, p_win, p_place, model_version FROM predictions "
                            "WHERE run_id=? AND race_id LIKE ?", (r[0], pref)).fetchall()
        return _pred_map(rows)
    return {}


def _pred_map(rows):
    out = {}
    for r in rows:
        out.setdefault(r[0], {})[int(r[1])] = (_f(r[2]), _f(r[3]), r[4])
    return out


def _latest_odds(conn, date, venue):
    try:
        from .odds import latest_odds
        return latest_odds(conn, date, venue) or {}
    except Exception as e:   # pragma: no cover
        log.warning("latest_odds failed: %s", e)
        return {}


def _ticket_rows(conn, date, venue):
    if "bot_id" not in _cols(conn, "paper_tickets"):
        return []
    return conn.execute("SELECT * FROM paper_tickets WHERE date=? AND venue=? AND bot_id IS NOT NULL "
                        "AND status != 'void' ORDER BY ticket_id", (date, venue)).fetchall()


def _reason_txt(s) -> str:
    """Decision reason as shown on the site: older rows said 「見第0場（跨場決定）」; the meeting page
    section is now called 「過關決定」."""
    s = str(s or "")
    return s.replace("見第0場跨場決定", "見「過關決定」").replace("見第0場", "見「過關決定」")


def _strategy_structure(strategy, legs) -> str:
    code = str(strategy or "").split(":")[-1]
    pool = ((legs[0].get("pool") if legs else None) or code.split("_")[0] or "").upper()
    rest = code.split("_", 1)[1] if "_" in code else ""
    return (pool + " " + rest).strip() or str(strategy or "")


def _leg_hit(race, leg, strategy=""):
    """Did this leg of a ticket come in?  PLA: a selection placed; QPL: two selections both placed
    (a paid 位置Q pair); WIN: a selection won.  None while the race has no result or for other pools
    (the ticket-level ``hit`` then tells)."""
    if not race or not race.get("result"):
        return None
    pool = (leg.get("pool") or str(strategy or "").split(":")[-1].split("_")[0] or "").upper()
    sel = set()
    for x in list(leg.get("selections") or []) + list(leg.get("bankers") or []):
        try:
            sel.add(int(x))
        except (TypeError, ValueError):
            pass
    if pool == "PLA":
        placed = _placed_set(race)
        return None if placed is None else bool(sel & placed)
    if pool == "QPL":
        pairs = _qpl_pairs(race)
        return None if pairs is None else any(p <= sel for p in pairs)
    if pool == "WIN":
        wins = {int(c) for c, _d in (race["result"].get("dividends") or {}).get("WIN") or [] if str(c).isdigit()}
        return bool(sel & wins) if wins else None
    return None


def _ticket_json(t) -> dict:
    status = t["status"]
    settled = status == "settled"
    reason = t["reason"] if "reason" in t.keys() else None
    return {"ticket_id": t["ticket_id"], "strategy": t["strategy"], "legs": _json(t["legs"], []),
            "n_combos": _i(t["n_combos"]), "unit_stake": _f(t["unit_stake"], 2),
            "total_stake": _f(t["total_stake"], 2),
            "payout": _f(t["payout"], 2) if settled else None,
            "profit": _f(t["profit"], 2) if settled else None,
            "status": status, "reason": reason or ""}


# ------------------------------------------------------------------ 今日精選 (daily picks)
PICKS_BOT = "daily_picks"
PICKS_TITLE = "今日精選"


def _picks_run(conn, date, venue):
    """run_id of the daily_picks robot's latest decisions for the meeting (None if it never ran)."""
    if not _table_exists(conn, "bot_decisions"):
        return None
    r = conn.execute("SELECT run_id FROM bot_decisions WHERE bot_id=? AND date=? AND venue=? "
                     "ORDER BY created_at DESC, run_id DESC LIMIT 1", (PICKS_BOT, date, venue)).fetchone()
    return r[0] if r else None


def _stored_picks(conn, date, venue):
    if not _table_exists(conn, "picks"):
        return None
    from . import picks as _picks
    run = _picks_run(conn, date, venue)
    p = _picks.load_picks(conn, date, venue, run) if run else None
    return p if p is not None else _picks.load_picks(conn, date, venue)


def _pick_tickets(conn, date, venue) -> dict:
    """{pick_key: ticket row} of the daily_picks robot's non-void tickets of the meeting."""
    if "bot_id" not in _cols(conn, "paper_tickets"):
        return {}
    out = {}
    for t in conn.execute("SELECT * FROM paper_tickets WHERE date=? AND venue=? AND bot_id=? AND status != 'void' "
                          "ORDER BY ticket_id", (date, venue, PICKS_BOT)):
        key = ((_json(t["detail"], {}) or {}).get("notes") or {}).get("pick_key")
        if key:
            out[key] = t
    return out


def _ticket_pool(t) -> str:
    legs = t.get("legs") or []
    pool = (t.get("pool") or (legs[0].get("pool") if legs else None) or "").upper()
    if not pool:
        s = str(t.get("strategy") or "").upper()
        pool = "QPL" if "QPL" in s else "PLA" if "PLA" in s else ""
    return pool


def _placed_set(race) -> set | None:
    """Horses that placed (paid a 位置 dividend) in a race; None while there is no result."""
    res = race.get("result") if race else None
    if not res:
        return None
    pla = (res.get("dividends") or {}).get("PLA") or []
    out = set()
    for comb, _div in pla:
        try:
            out.add(int(str(comb).strip()))
        except (TypeError, ValueError):
            pass
    if out:
        return out
    fin = [h for h in race.get("horses", []) if h.get("result") and h["result"].get("finish_pos")]
    if not fin:
        return None
    k = 3 if len(fin) >= 7 else 2
    return {h["horse_no"] for h in fin if h["result"]["finish_pos"] <= k}


def _qpl_pairs(race) -> set | None:
    res = race.get("result") if race else None
    if not res:
        return None
    out = set()
    for comb, _div in (res.get("dividends") or {}).get("QPL") or []:
        try:
            a, b = (int(x) for x in str(comb).split(","))
            out.add(frozenset((a, b)))
        except (TypeError, ValueError):
            pass
    if out:
        return out
    placed = _placed_set(race)
    if placed is None:
        return None
    pl = sorted(placed)
    return {frozenset((a, b)) for i, a in enumerate(pl) for b in pl[i + 1:]}


def _horse_result(race, horse):
    """{finish_pos, placed} once the race has a result, else None."""
    placed = _placed_set(race)
    if placed is None:
        return None
    r = (horse or {}).get("result") or {}
    scratched = bool((horse or {}).get("scratched"))
    return {"finish_pos": _i(r.get("finish_pos")),
            "placed": None if scratched else (horse or {}).get("horse_no") in placed}


def _races_done(byrn, item) -> bool:
    return all((byrn.get(int(rn)) or {}).get("result") for rn in item.get("races") or [item["race_no"]])


def _mr_leg_hits(byrn, legs):
    """DBL/TBL legs: earlier legs hit when a selection won, the last leg when one ran 1st or 2nd."""
    out = []
    for i, l in enumerate(legs):
        race = byrn.get(int(l["race_no"])) or {}
        need = 2 if i == len(legs) - 1 else 1
        pos = {int(h["horse_no"]): (h.get("result") or {}).get("finish_pos") for h in race.get("horses", [])}
        sel = [int(h["horse_no"]) for h in l.get("selections") or []]
        if not any(v for v in pos.values()):
            out.append(None)
            continue
        out.append(any(pos.get(h) is not None and pos[h] <= need for h in sel))
    return out


def build_picks(conn, date, venue, races) -> dict | None:
    """「今日精選」 block of a meeting file (第四版, sections + ⭐) from the picks stored by the
    pre-meeting run whose daily_picks robot decisions are current, with results from the
    daily_picks robot's settled tickets (an unbought item: hypothetical result from the published
    dividends, ``result.hypothetical = true``).  None when the meeting has no stored picks."""
    stored = _stored_picks(conn, date, venue)
    if not stored:
        return None
    from . import picks as _picks
    byrn = {int(r["race_no"]): r for r in races or []}
    horse = {(rn, int(h["horse_no"])): h for rn, r in byrn.items() for h in r.get("horses", [])}
    tickets = _pick_tickets(conn, date, venue)
    data = _picks.DBData(conn, date, venue)
    items = []
    all_done, tot_pay = True, 0.0
    for it in stored.get("items") or []:
        res = None
        t = tickets.get(it.get("key"))
        leg_hits = None
        if t is not None and t["status"] == "settled":
            det = _json(t["detail"], {}) or {}
            nw = int(det.get("n_winning_units") or 0)
            pay, prof = _f(t["payout"], 2), _f(t["profit"], 2)
            res = {"hit": nw > 0, "payout": pay, "profit": prof,
                   "detail": (f"中 {nw} 注，派彩 ${pay:,.1f}" if nw else "冇中"), "hypothetical": False}
        elif t is None and _races_done(byrn, it) and it.get("legs_raw"):
            try:
                r = _picks.settle_item(it, data)
                res = {"hit": r["hit"], "payout": r["payout"], "profit": r["profit"],
                       "detail": "機械人冇落呢項（例如資金不足），假設落注：" + r["detail"],
                       "hypothetical": True}
                leg_hits = r.get("leg_hits")
            except Exception as e:      # pragma: no cover - never break the export
                log.warning("picks result failed for %s: %s", it.get("key"), e)
        if res is None or res.get("hypothetical"):
            all_done = False
        else:
            tot_pay += res["payout"] or 0.0
        pub = {k: v for k, v in it.items() if k not in ("ticket", "legs_raw", "key", "family", "spec")}
        legs = []
        for i, l in enumerate(pub.get("legs") or []):
            rn = int(l["race_no"])
            l = dict(l)
            for k in ("bankers", "selections"):
                l[k] = [{**h, "horse_name": (horse.get((rn, int(h["horse_no"]))) or {}).get("horse_name")
                         or h.get("horse_name") or "",
                         "score": _f((horse.get((rn, int(h["horse_no"]))) or {}).get("score"), 1)
                         if (horse.get((rn, int(h["horse_no"]))) or {}).get("score") is not None
                         else h.get("score")} for h in l.get(k) or []]
            if res is not None:
                if leg_hits is None and it.get("kind") == "multi_race":
                    leg_hits = _mr_leg_hits(byrn, pub.get("legs") or [])
                if leg_hits is not None and i < len(leg_hits):
                    l["hit"] = leg_hits[i]
                elif len(pub.get("legs") or []) == 1:
                    l["hit"] = bool(res["hit"])
            legs.append(l)
        pub["legs"] = legs
        pub["result"] = res
        items.append(pub)
    sel = items
    meeting_done = bool(byrn) and all(r.get("result") for r in byrn.values())
    total = round(sum(i["stake"] or 0 for i in sel), 2)
    if sel and all_done:
        total_payout = round(tot_pay, 2)
    elif not sel and meeting_done:
        total_payout = 0.0
    else:
        total_payout = None
    banker = stored.get("banker")
    if banker:
        banker = dict(banker)
        rn = int(banker["race_no"])
        h = horse.get((rn, int(banker["horse_no"])))
        if h:
            banker["horse_name"] = h.get("horse_name") or banker.get("horse_name") or ""
            banker["jockey"] = h.get("jockey") or banker.get("jockey") or ""
            banker["score"] = _f(h.get("score"), 1) if h.get("score") is not None else banker.get("score")
            banker["result"] = _horse_result(byrn.get(rn), h)
        banker["post_time"] = banker.get("post_time") or (byrn.get(rn) or {}).get("post_time")
    out = {k: v for k, v in stored.items() if k not in ("race_notes", "n_candidates", "items", "banker")}
    for it in items:
        it["post_time"] = it.get("post_time") or (byrn.get(int(it["races"][0])) or {}).get("post_time")
    out.update({"title": PICKS_TITLE, "items": items, "banker": banker, "n_items": len(sel), "total_stake": total,
                "n_star": sum(1 for i in sel if i.get("star")),
                "total_payout": total_payout,
                "total_profit": None if total_payout is None else round(total_payout - total, 2)})
    return out


def export_meeting(conn, date, venue, bots_order=None) -> dict:
    status = _meeting_status(conn, date, venue)
    order = {b: i for i, b in enumerate(bots_order or [])}
    s_run = _pick_run(_score_runs(conn, date, venue), status)
    d_run = _pick_run(_decision_runs(conn, date, venue), status)
    pref = _rid_prefix(date, venue)

    races = conn.execute("SELECT * FROM races WHERE date=? AND venue=? ORDER BY race_no", (date, venue)).fetchall()
    race_nos = [r["race_no"] for r in races]
    card_nos = [r[0] for r in conn.execute("SELECT DISTINCT race_no FROM racecard WHERE date=? AND venue=? "
                                           "ORDER BY race_no", (date, venue))]
    race_info = {r["race_no"]: r for r in races}
    all_nos = sorted(set(race_nos) | set(card_nos))

    card = {}
    for r in conn.execute("SELECT * FROM racecard WHERE date=? AND venue=?", (date, venue)):
        card[(r["race_no"], r["horse_no"])] = r
    runners = {}
    for r in conn.execute("SELECT u.*, r.race_no FROM runners u JOIN races r USING(race_id) "
                          "WHERE r.date=? AND r.venue=?", (date, venue)):
        runners[(r["race_no"], r["horse_no"])] = r

    scores = {}
    if s_run:
        for r in conn.execute("SELECT * FROM horse_scores WHERE run_id=? AND race_id LIKE ?", (s_run, pref + "%")):
            scores[(int(r["race_id"][-2:]), int(r["horse_no"]))] = r
    preds = _predictions(conn, [s_run, d_run], date, venue)
    odds = _latest_odds(conn, date, venue)
    snap = conn.execute("SELECT MAX(snapshot_at) FROM odds_snapshots WHERE date=? AND venue=?",
                        (date, venue)).fetchone()[0]

    tickets = {t["ticket_id"]: t for t in _ticket_rows(conn, date, venue)}
    all_tickets = {}
    if "bot_id" in _cols(conn, "paper_tickets"):
        all_tickets = {t["ticket_id"]: t for t in conn.execute(
            "SELECT * FROM paper_tickets WHERE date=? AND venue=?", (date, venue))}

    decisions, tdec, skips = {}, {}, {}
    if d_run:
        # each robot's rows of its latest run (bots.run_bots replaces a robot's rows per meeting,
        # so robots re-decided by a later run never hide the others' decisions)
        rows = conn.execute("SELECT * FROM bot_decisions WHERE date=? AND venue=?", (date, venue)).fetchall()
        latest = {}
        for r in rows:
            k = (r["created_at"] or "", r["run_id"] or "")
            if k > latest.get(r["bot_id"], ("", "")):
                latest[r["bot_id"]] = k
        rows = [r for r in rows if (r["created_at"] or "", r["run_id"] or "") == latest[r["bot_id"]]]
        rows = sorted(rows, key=lambda r: (order.get(r["bot_id"], 99), r["bot_id"], r["structure"] or ""))
        for r in rows:
            t = all_tickets.get(r["ticket_id"]) if r["ticket_id"] is not None else None
            tstat = t["status"] if t is not None else None
            payout = profit = None
            # a per-race row pointing at a multi-race (all-up) ticket: the ticket's P&L is shown once,
            # on the meeting-level (race_no 0) decision, not on every leg
            leg = t is not None and int(r["race_no"]) > 0 and len(_json(t["legs"], [])) > 1
            if leg:
                pass
            elif t is not None and tstat == "settled":
                payout, profit = _f(t["payout"], 2), _f(t["profit"], 2)
            elif t is not None and tstat == "void":
                payout, profit = _f(t["total_stake"], 2), 0.0
            if r["ticket_id"] is not None and not leg and r["action"] == "bet":
                tdec.setdefault(r["ticket_id"], (r["structure"] or "", _json(r["selections"], [])))
            if r["action"] != "bet":
                skips.setdefault(r["bot_id"], []).append({"race_no": int(r["race_no"]),
                                                          "structure": r["structure"] or "",
                                                          "reason": _reason_txt(r["reason"])})
            decisions.setdefault(int(r["race_no"]), []).append({
                "bot_id": r["bot_id"], "race_no": int(r["race_no"]), "action": r["action"],
                "structure": r["structure"] or "", "selections": _json(r["selections"], []),
                "stake": _f(r["stake"], 2) or 0.0, "reason": _reason_txt(r["reason"]),
                "payout": payout, "profit": profit, "ticket_status": tstat, "parlay_leg": bool(leg)})

    model_version = None
    out_races = []
    for rn in all_nos:
        ri = race_info.get(rn)
        rid = f"{pref}{rn:02d}"
        hnos = sorted({h for (r_, h) in card if r_ == rn} | {h for (r_, h) in runners if r_ == rn})
        pr = preds.get(rid, {})
        horses = []
        for h in hnos:
            c, u, s = card.get((rn, h)), runners.get((rn, h)), scores.get((rn, h))
            src = c if c is not None else u
            pw, pp, mv = pr.get(h, (None, None, None))
            model_version = model_version or mv
            scratched = bool((c["scratched"] if c is not None else 0) or (u["scratched"] if u is not None else 0))
            sub = _json(s["sub_scores"], {}) if s is not None else {}
            res = None
            if u is not None and (u["finish_pos"] is not None or u["finish_status"]):
                res = {"finish_pos": _i(u["finish_pos"]), "final_win_odds": _f(u["win_odds"]),
                       "finish_status": u["finish_status"], "lbw": u["lbw"]}
            ro = odds.get(int(rn), {}) if odds else {}
            horses.append({
                "horse_no": int(h),
                "horse_name": (u["horse_name"] if u is not None and u["horse_name"] else src["horse_name"]) or "",
                "jockey": (u["jockey_name"] if u is not None and u["jockey_name"] else src["jockey_name"]) or "",
                "trainer": (u["trainer_name"] if u is not None and u["trainer_name"] else src["trainer_name"]) or "",
                "draw": _i(c["draw"] if c is not None else u["draw"]),
                "weight": _i(c["weight"] if c is not None else u["act_weight"]),
                "score": _f(s["score"], 1) if s is not None else None,
                "rank": _i(s["rank_in_race"]) if s is not None else None,
                "p_win": pw, "p_place": pp,
                "win_odds": _f((ro.get("WIN") or {}).get(str(h))),
                "pla_odds": _f((ro.get("PLA") or {}).get(str(h))),
                "sub_scores": {k: _f(sub.get(k), 1) for k in SUB_SCORE_KEYS},
                "reasons": [str(x) for x in (_json(s["reasons"], []) if s is not None else [])],
                "scratched": scratched,
                "result": res,
            })
        if not any(x["rank"] for x in horses):   # no scores: rank by model p_win
            ranked = sorted([x for x in horses if x["p_win"] is not None and not x["scratched"]],
                            key=lambda x: -x["p_win"])
            for i, x in enumerate(ranked, 1):
                x["rank"] = i
        out_races.append({
            "race_no": int(rn),
            "post_time": ri["post_time"] if ri is not None else None,
            "race_class": ri["race_class"] if ri is not None else None,
            "rating_band": ri["rating_band"] if ri is not None else None,
            "distance": _i(ri["distance"]) if ri is not None else None,
            "surface": ri["surface"] if ri is not None else None,
            "course": ri["course"] if ri is not None else None,
            "going": ri["going"] if ri is not None else None,
            "race_name": ri["race_name"] if ri is not None else None,
            "horses": horses,
            "decisions": decisions.get(int(rn), []),
            "result": _race_result(conn, rid, horses),
        })

    byrn = {r["race_no"]: r for r in out_races}
    bots = {}
    for t in tickets.values():
        b = bots.setdefault(t["bot_id"], {"bot_id": t["bot_id"], "stake": 0.0, "payout": 0.0, "profit": 0.0,
                                          "n_tickets": 0, "settled": True, "tickets": []})
        tj = _ticket_json(t)
        legs = tj["legs"] or []
        struct, sels = tdec.get(t["ticket_id"], (None, None))
        tj["race_no"] = int(legs[0]["race_no"]) if len(legs) == 1 else 0
        tj["structure"] = struct or _strategy_structure(t["strategy"], legs)
        tj["selections"] = sels if sels else (legs[0].get("selections", []) if len(legs) == 1 else
                                              [{"race_no": l["race_no"], "selections": l.get("selections", [])}
                                               for l in legs])
        tj["leg_hits"] = [_leg_hit(byrn.get(int(l["race_no"])), l, t["strategy"]) for l in legs]
        tj["hit"] = (tj["payout"] or 0) > 0 if tj["status"] == "settled" else None
        b["tickets"].append(tj)
        b["n_tickets"] += 1
        b["stake"] += tj["total_stake"] or 0.0
        if tj["status"] == "settled":
            b["payout"] += tj["payout"] or 0.0
        else:
            b["settled"] = False
    for b in bots.values():
        b["stake"] = round(b["stake"], 2)
        if b["settled"]:
            b["payout"] = round(b["payout"], 2)
            b["profit"] = round(b["payout"] - b["stake"], 2)
        else:
            b["payout"] = b["profit"] = None
    for bid in {d["bot_id"] for ds in decisions.values() for d in ds} - set(bots):
        bots[bid] = {"bot_id": bid, "stake": 0.0, "payout": 0.0 if status == "results" else None,
                     "profit": 0.0 if status == "results" else None, "n_tickets": 0,
                     "settled": status == "results", "tickets": []}
    for b in bots.values():
        b["tickets"].sort(key=lambda t: (t["race_no"] or 99, t["ticket_id"] or 0))
        b["skips"] = sorted(skips.get(b["bot_id"], []), key=lambda x: (x["race_no"] or 99))
    bot_list = sorted(bots.values(), key=lambda b: (order.get(b["bot_id"], 99), b["bot_id"]))

    return {"date": date, "venue": venue, "venue_zh": VENUE_ZH.get(venue, venue), "status": status,
            "generated_at": _now(), "odds_snapshot_at": snap, "model_version": model_version,
            "scores_run_id": s_run, "decisions_run_id": d_run,
            "races": out_races, "meeting_decisions": decisions.get(0, []), "bots": bot_list,
            "picks": build_picks(conn, date, venue, out_races)}


def _race_result(conn, race_id, horses) -> dict | None:
    rows = conn.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id=?", (race_id,)).fetchall()
    fin = sorted([h for h in horses if h["result"] and h["result"]["finish_pos"]],
                 key=lambda h: h["result"]["finish_pos"])
    if not rows and not fin:
        return None
    divs = {}
    for p in RESULT_POOLS:
        lst = [[r["combination"], _f(r["dividend"], 2)] for r in rows if r["pool"] == p]
        if lst:
            divs[p] = lst
    placings = [{"pos": h["result"]["finish_pos"], "horse_no": h["horse_no"], "horse_name": h["horse_name"],
                 "jockey": h["jockey"], "finish_status": h["result"]["finish_status"],
                 "win_odds": h["result"]["final_win_odds"]} for h in fin if h["result"]["finish_pos"] <= 4]
    return {"placings": placings, "dividends": divs}


# ------------------------------------------------------------------ robots (live)
def _is_hit(t) -> bool:
    det = _json(t["detail"], {})
    if isinstance(det, dict) and "n_winning_units" in det:
        try:
            return int(det["n_winning_units"]) > 0
        except (TypeError, ValueError):
            pass
    return (t["payout"] or 0) > 0


def _live_stats(conn, bot_ids):
    has_bot = "bot_id" in _cols(conn, "paper_tickets")
    ledger_bot = "bot_id" in _cols(conn, "bankroll_ledger")
    out, hist = {}, {}
    for bid in bot_ids:
        rows = conn.execute("SELECT * FROM paper_tickets WHERE bot_id=? AND status != 'void' "
                            "ORDER BY date, ticket_id", (bid,)).fetchall() if has_bot else []
        settled = [r for r in rows if r["status"] == "settled"]
        open_ = [r for r in rows if r["status"] == "open"]
        stake = sum(r["total_stake"] or 0 for r in settled)
        ret = sum(r["payout"] or 0 for r in settled)
        start = START_BANKROLL
        bal = None
        if ledger_bot:
            first = conn.execute("SELECT amount, ticket_id FROM bankroll_ledger WHERE bot_id=? ORDER BY id LIMIT 1",
                                 (bid,)).fetchone()
            if first is not None and first[1] is None:
                start = float(first[0])
            last = conn.execute("SELECT balance FROM bankroll_ledger WHERE bot_id=? ORDER BY id DESC LIMIT 1",
                                (bid,)).fetchone()
            bal = float(last[0]) if last is not None else None
        open_stake = sum(r["total_stake"] or 0 for r in open_)
        if bal is None:
            bal = start + (ret - stake) - open_stake
        meets = {}
        finished = {(r[0], r[1]) for r in conn.execute(
            "SELECT DISTINCT d.date, d.venue FROM bot_decisions d JOIN meetings m ON m.date=d.date AND "
            "m.venue=d.venue WHERE d.bot_id=? AND m.status IN ('results','abandoned')", (bid,))}
        for r in settled:
            m = meets.setdefault((r["date"], r["venue"]), [0.0, 0.0])
            m[0] += r["total_stake"] or 0
            m[1] += r["payout"] or 0
        curve, cum = [], start
        for (d, v) in sorted(meets):
            st, py = meets[(d, v)]
            cum += py - st
            curve.append({"date": d, "venue": v, "stake": round(st, 2), "payout": round(py, 2),
                          "profit": round(py - st, 2), "bankroll": round(cum, 2)})
        out[bid] = {"bankroll": round(bal, 2), "start_bankroll": round(start, 2), "live": {
            "n_meetings": len(set(meets) | finished), "stake": round(stake, 2), "returns": round(ret, 2),
            "profit": round(ret - stake, 2), "roi": (ret / stake - 1) if stake else None,
            "hit_rate": (sum(1 for r in settled if _is_hit(r)) / len(settled)) if settled else None,
            "n_tickets": len(settled), "open_stake": round(open_stake, 2)}}
        hist[bid] = curve
    return out, hist


# ------------------------------------------------------------------ export
def export_site(conn, out_dir=None, demo: bool = False) -> list[Path]:
    """Write index.json, meetings/*.json and bot_history.json under ``<out_dir>/data``.
    Returns the written paths.  ``out_dir`` defaults to the project's ``site/``.
    ``demo=True`` marks index.json as demonstration data (the site then shows a 示範數據 banner)."""
    out_dir = Path(out_dir) if out_dir is not None else SITE_DIR
    data = out_dir / "data"
    mdir = data / "meetings"
    mdir.mkdir(parents=True, exist_ok=True)
    meta = bot_meta()
    ids = [b["id"] for b in meta]
    # robots seen in the DB but not in the metadata list (e.g. renamed) are still exported
    if "bot_id" in _cols(conn, "paper_tickets"):
        extra = [r[0] for r in conn.execute(
            "SELECT DISTINCT bot_id FROM paper_tickets WHERE bot_id IS NOT NULL "
            "UNION SELECT DISTINCT bot_id FROM bot_decisions") if r[0] not in ids]
        for i, bid in enumerate(sorted(extra), len(meta) + 1):
            meta.append({"id": bid, "no": i, "name": bid, "style": "", "description": ""})
            ids.append(bid)

    written, metas, bot_tickets = [], [], {}
    for date, venue in meetings_to_export(conn):
        m = export_meeting(conn, date, venue, bots_order=ids)
        fname = f"{date}_{venue}.json"
        written.append(_write_json(mdir / fname, m, ignore=("generated_at",)))
        for b in m["bots"]:
            for t in b["tickets"]:
                bot_tickets.setdefault(b["bot_id"], []).append(
                    {"date": date, "venue": venue, "venue_zh": VENUE_ZH.get(venue, venue), **t})
        metas.append({"date": date, "venue": venue, "venue_zh": VENUE_ZH.get(venue, venue),
                      "status": m["status"], "file": f"data/meetings/{fname}", "n_races": len(m["races"]),
                      "has_scores": any(h["score"] is not None for r in m["races"] for h in r["horses"]),
                      "has_decisions": any(r["decisions"] for r in m["races"]) or bool(m["meeting_decisions"]),
                      "has_picks": bool(m.get("picks"))})
    keep = {p.name for p in written}
    for p in mdir.glob("*.json"):
        if p.name not in keep:
            p.unlink()
    # one file per robot with all its (non-void) tickets, newest meeting first: the robot page
    # loads this instead of every meeting file (hundreds after a season)
    bdir = data / "bots"
    bfiles = []
    for bid in ids:
        rows = sorted(bot_tickets.get(bid, []), key=lambda t: (t["date"], t["venue"], t["ticket_id"] or 0),
                      reverse=True)
        bfiles.append(_write_json(bdir / f"{bid}.json", {"bot_id": bid, "n_tickets": len(rows), "tickets": rows}))
    keep_b = {p.name for p in bfiles}
    for p in bdir.glob("*.json"):
        if p.name not in keep_b:
            p.unlink()
    written += bfiles

    live, live_hist = _live_stats(conn, ids)
    bots = []
    for b in meta:   # live (paper) results only - no backtest figures on the public site
        lv = live[b["id"]]
        bots.append({**b, "bankroll": lv["bankroll"], "start_bankroll": lv["start_bankroll"],
                     "live": lv["live"]})
    now = _now()
    index = {"updated_at": now, "disclaimer": DISCLAIMER, "simulation_only": True,
             "start_bankroll": START_BANKROLL, "next_meeting": next_meeting(conn), "meetings": metas,
             "bots": bots, "score_explanation": score_explanation(), "sub_score_labels": sub_score_labels()}
    if demo:
        index["demo"] = True
    written.append(_write_json(data / "index.json", index))
    history = {"updated_at": now, "start_bankroll": START_BANKROLL,
               "bots": [{"bot_id": bid, "live": live_hist[bid]} for bid in ids]}
    written.append(_write_json(data / "bot_history.json", history))
    return written


# ------------------------------------------------------------------ publish
def load_config(path=None) -> dict:
    path = Path(path) if path is not None else PUBLISH_CONFIG
    cfg = dict(DEFAULT_PUBLISH_CONFIG)
    try:
        if path.exists():
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(DEFAULT_PUBLISH_CONFIG, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as e:
        log.warning("publish config %s unreadable (%s): publishing disabled", path.name, e)
        cfg["enabled"] = False
    return cfg


def _git(site: Path, *args, timeout=120, ident=None):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "", "SSH_ASKPASS": "",
           "GIT_SSH_COMMAND": os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=15")}
    pre = []
    if ident:
        pre = ["-c", f"user.name={ident[0]}", "-c", f"user.email={ident[1]}"]
        # env identity beats user.name/email: never let a personal GIT_AUTHOR_* leak into public history
        env.update(GIT_AUTHOR_NAME=ident[0], GIT_AUTHOR_EMAIL=ident[1],
                   GIT_COMMITTER_NAME=ident[0], GIT_COMMITTER_EMAIL=ident[1])
    return subprocess.run(["git", *pre, "-C", str(site), *args], capture_output=True, text=True,
                          timeout=timeout, env=env)


def _ensure_repo(site: Path, branch: str):
    if (site / ".git").exists():
        return
    r = _git(site, "init", "-b", branch)
    if r.returncode != 0:     # very old git without -b
        r = _git(site, "init")
        if r.returncode != 0:
            raise RuntimeError(f"git init failed: {r.stderr.strip()[:300]}")
        _git(site, "symbolic-ref", "HEAD", f"refs/heads/{branch}")


def publish(conn, message=None, *, site_dir=None, config_path=None) -> dict:
    """Export the site, commit it in ``site/`` (its own git repo) and push when enabled.

    Returns {status, files, committed, pushed, error}; status is one of
    'exported' (push disabled / no remote), 'pushed', 'export_failed', 'commit_failed',
    'push_failed'.  Never raises."""
    site = Path(site_dir) if site_dir is not None else SITE_DIR
    res = {"status": "exported", "files": [], "committed": False, "pushed": False, "error": None}
    try:
        files = export_site(conn, site)
        res["files"] = [str(p.relative_to(site)) for p in files]
    except Exception as e:
        log.warning("site export failed: %s", e, exc_info=True)
        return {**res, "status": "export_failed", "error": str(e)}
    cfg = load_config(config_path)
    if cfg.get("git") is False or os.environ.get("HKJC_CLOUD"):
        # unified-repo mode: site/ is part of the main repo; GitHub Actions commits + deploys Pages
        return res
    branch = str(cfg.get("branch") or "main")
    ident = (cfg.get("author_name") or DEFAULT_AUTHOR[0], cfg.get("author_email") or DEFAULT_AUTHOR[1])
    try:
        _ensure_repo(site, branch)
        r = _git(site, "add", "-A")
        if r.returncode != 0:
            raise RuntimeError(f"git add failed: {r.stderr.strip()[:300]}")
        if _git(site, "status", "--porcelain").stdout.strip():
            msg = message or f"更新數據 {_now()[:16].replace('T', ' ')}"
            r = _git(site, "-c", "commit.gpgsign=false", "commit", "-q", "-m", msg, ident=ident)
            if r.returncode != 0:
                raise RuntimeError(f"git commit failed: {(r.stderr or r.stdout).strip()[:300]}")
            res["committed"] = True
    except Exception as e:
        log.warning("site commit failed: %s", e)
        return {**res, "status": "commit_failed", "error": str(e)}
    remote = str(cfg.get("remote") or "").strip()
    if not cfg.get("enabled") or not remote:
        return res
    try:
        r = _git(site, "push", remote, f"HEAD:refs/heads/{branch}", timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"git push failed: {r.stderr.strip()[:300]}")
        res.update(status="pushed", pushed=True)
    except Exception as e:
        log.warning("site push failed: %s", e)
        res.update(status="push_failed", error=str(e))
    return res
