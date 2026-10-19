"""Strategy-grid backtest on real HKJC dividends.  PAPER / RESEARCH ONLY.

main(conn, start=None, end=None, source=None) -> dict
    Runs the grid for ``source`` ('market' or a predictions run_id) - or, when
    None, for 'market' plus every walk-forward run ('wf-%') in ``predictions`` -
    writes ``backtest_results`` and ``reports/回測報告.md``.
run_grid(conn, source='market', start=None, end=None, strategies=None) -> dict
    {'bets': DataFrame (one row per bet), 'strategies': list, 'meetings': DataFrame}
summarise(bets, strategies, meetings, ...) -> DataFrame of metrics per strategy.

Predictions sources
-------------------
* 'market': p_win = normalised 1/final win odds per race, p_place = Harville
  (strategy.harville).  NB final odds are only known at the off -> look-ahead.
* '<run_id>': p_win / p_place from the predictions table.

Every bet is settled with the published per-$10 dividends via hkjc.bets
(combo_outcome / settle_allup), scratched horses refunded, dead heats paid per row.
Single-race bets are skipped when the pool was not offered for the race (e.g. QPL
with < 7 starters - known before the race); void/abandoned races are excluded.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import bets, strategy as st

BT_VERSION = "grid-v1"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "回測報告.md"
POOLS = ("WIN", "PLA", "QIN", "QPL", "FCT", "TCE", "TRI", "FF", "QTT")
UNIT = 10.0
MIN_BETS = 50          # minimum bets for a strategy to be ranked in the out-of-sample check
IS_SEASONS = ("2024-25",)
OOS_SEASONS = ("2025-26", "2026-27")
# HKJC Local Pools page (share paid as dividends 82.5% / 80.5% / 77% / 75%):
# https://special.hkjc.com/e-win/en-US/betting-info/racing/beginners-guide/local-pools/
TAKEOUT = {"WIN": "17.5%", "PLA": "17.5%", "QIN": "17.5%", "QPL": "17.5%", "FCT": "19.5%",
           "TCE": "25%", "TRI": "23%", "FF": "25%", "QTT": "25%", "過關": "各池扣率逐關疊加"}


def season_of(date: str) -> str:
    """HKJC season runs Sept -> July: '2024-10-01' -> '2024-25'."""
    y, m = int(date[:4]), int(date[5:7])
    if m < 8:
        y -= 1
    return f"{y}-{(y + 1) % 100:02d}"


# ----------------------------------------------------------------- strategy grid
def default_strategies(source: str = "market") -> list[dict]:
    S: list[dict] = []

    def add(name, family, **kw):
        S.append({"name": name, "family": family, **kw})

    # A. the two fixed products under every race-selection rule
    pla_rules = ("confidence", "value", "first6", "last6", "fixed")
    qpl_rules = (("confidence", "chrono"), ("confidence", "rank"), ("consecutive", "chrono"),
                 ("value", "chrono"), ("first6", "chrono"), ("last6", "chrono"))
    if source == "market":            # model/market edge is identically 1 -> 'value' == 'confidence'
        pla_rules = tuple(r for r in pla_rules if r != "value")
        qpl_rules = tuple(r for r in qpl_rules if r[0] != "value")
    for rule in pla_rules:
        add(f"PLA 6x63 [{rule}]", "固定組合", kind="allup", pool="PLA", n_legs=6, spec="6x63",
            hpl=1, rule=rule, fixed=[2, 3, 4, 5, 6, 7])
    for rule, pairing in qpl_rules:
        tag = rule if pairing == "chrono" else f"{rule}-配對按信心"
        add(f"3xQPL 2x3 [{tag}]", "固定組合", kind="multi", pool="QPL", n_tickets=3, lpt=2,
            spec="2x3", hpl=3, rule=rule, pairing=pairing)
    # B. all-up alternatives (confidence rule)
    for spec in ("6x1", "6x7", "6x22", "6x42", "6x57"):
        add(f"PLA {spec}", "過關結構", kind="allup", pool="PLA", n_legs=6, spec=spec, hpl=1,
            rule="confidence")
    for spec in ("3x1", "3x4", "3x7"):
        add(f"PLA {spec}", "過關結構", kind="allup", pool="PLA", n_legs=3, spec=spec, hpl=1,
            rule="confidence")
    for spec in ("4x11", "4x15"):
        add(f"PLA {spec}", "過關結構", kind="allup", pool="PLA", n_legs=4, spec=spec, hpl=1,
            rule="confidence")
    add("3xQPL 2x1", "過關結構", kind="multi", pool="QPL", n_tickets=3, lpt=2, spec="2x1", hpl=3,
        rule="confidence", pairing="chrono")
    for h in (2, 4):
        add(f"3xQPL 2x3 每關{h}隻", "過關結構", kind="multi", pool="QPL", n_tickets=3, lpt=2,
            spec="2x3", hpl=h, rule="confidence", pairing="chrono")
    # C. single-race grid
    singles = [{"pool": "WIN", "top": 1}, {"pool": "PLA", "top": 1}, {"pool": "PLA", "top": 2}]
    for p in ("QIN", "QPL"):
        singles += [{"pool": p, "box": n} for n in range(2, 6)]
        singles += [{"pool": p, "bankers": 1, "legs": n} for n in range(2, 7)]
    singles += [{"pool": "TRI", "box": n} for n in range(3, 7)]
    singles += [{"pool": "TRI", "bankers": 1, "legs": n} for n in range(3, 8)]
    singles += [{"pool": "TRI", "bankers": 2, "legs": n} for n in range(2, 7)]
    singles += [{"pool": "FF", "box": n} for n in range(4, 8)]
    singles += [{"pool": "FF", "bankers": 1, "legs": n} for n in range(4, 8)]
    singles += [{"pool": "FF", "bankers": 2, "legs": n} for n in range(3, 7)]
    singles += [{"pool": "TCE", "box": 3}, {"pool": "TCE", "box": 4}]
    singles += [{"pool": "TCE", "bankers": 1, "legs": n, "mode": "single"} for n in (3, 4, 5)]
    singles += [{"pool": "TCE", "bankers": 1, "legs": n, "mode": "multiple"} for n in (3, 4)]
    singles += [{"pool": "TCE", "positions": [[1, 2], [1, 2, 3], [1, 2, 3, 4]]}]
    if source != "market":
        singles += [{"pool": "WIN", "value_ev": 1.1}, {"pool": "WIN", "value_ev": 1.3}]
    for sp in singles:
        add(st.single_spec_name(sp), "單場", kind="single", sp=sp, filt=None)
    # D. filter dimension on a core subset
    core = [{"pool": "WIN", "top": 1}, {"pool": "PLA", "top": 1}, {"pool": "QIN", "box": 2},
            {"pool": "QPL", "box": 3}, {"pool": "QPL", "bankers": 1, "legs": 3},
            {"pool": "TRI", "bankers": 1, "legs": 4}, {"pool": "FF", "bankers": 1, "legs": 5},
            {"pool": "TCE", "bankers": 1, "legs": 4, "mode": "single"}]
    filts = [("pw", 0.25), ("pw", 0.35)]
    if source != "market":
        filts += [("edge", 1.1), ("edge", 1.25)]
    for sp in core:
        for f in filts:
            lab = f"首選勝率≥{f[1]:.2f}" if f[0] == "pw" else f"模型/市場≥{f[1]:.2f}"
            add(f"{st.single_spec_name(sp)}（{lab}）", "篩選", kind="single", sp=sp, filt=f)
    for i, s in enumerate(S):
        s["sid"] = i
    return S


# ----------------------------------------------------------------- data loading
def _q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def available_sources(conn) -> list[str]:
    rows = conn.execute("SELECT DISTINCT run_id FROM predictions WHERE run_id LIKE 'wf-%' "
                        "ORDER BY run_id").fetchall()
    return [r[0] for r in rows]


def load_data(conn, source="market", start=None, end=None) -> list[dict]:
    """Chronological list of meetings: {date, venue, season, races: {race_no: {...}}}."""
    w, prm = ["m.status='results'"], []
    if start:
        w.append("r.date>=?"); prm.append(start)
    if end:
        w.append("r.date<=?"); prm.append(end)
    races = _q(conn, "SELECT r.race_id, r.date, r.venue, r.race_no, r.surface, r.distance, r.race_class "
                     "FROM races r JOIN meetings m ON m.date=r.date AND m.venue=r.venue "
                     f"WHERE {' AND '.join(w)} ORDER BY r.date, r.race_no", prm)
    if races.empty:
        return []
    d0, d1 = races["date"].min(), races["date"].max()
    run = _q(conn, "SELECT u.race_id, u.horse_no, u.win_odds, u.scratched, u.finish_pos FROM runners u "
                   "JOIN races r ON r.race_id=u.race_id WHERE r.date BETWEEN ? AND ?", (d0, d1))
    divs = _q(conn, "SELECT d.race_id, d.pool, d.combination, d.dividend FROM dividends d "
                    "JOIN races r ON r.race_id=d.race_id WHERE r.date BETWEEN ? AND ? "
                    f"AND d.pool IN ({','.join('?' * len(POOLS))})", (d0, d1, *POOLS))
    run = run[run["race_id"].isin(set(races["race_id"]))]
    fin = run[run["finish_pos"].notna()].groupby("race_id").apply(
        lambda g: dict(zip(g["horse_no"].astype(int), g["finish_pos"].astype(int))), include_groups=False).to_dict()
    scr = run[run["scratched"] == 1].groupby("race_id")["horse_no"].apply(lambda s: {int(x) for x in s}).to_dict()
    live = run[(run["scratched"] == 0) & run["win_odds"].notna() & (run["win_odds"] > 0)].copy()
    live["p_mkt"] = live.groupby("race_id")["win_odds"].transform(lambda s: st.market_probs(s.values))
    live["p_mkt_place"] = live.groupby("race_id")["p_mkt"].transform(lambda s: st.harville(s.values))
    if source == "market":
        live["p_win"], live["p_place"] = live["p_mkt"], live["p_mkt_place"]
    else:
        pr = _q(conn, "SELECT race_id, horse_no, p_win, p_place FROM predictions WHERE run_id=?", (source,))
        if pr.empty:
            raise ValueError(f"no predictions for run_id {source!r}")
        live = live.merge(pr, on=["race_id", "horse_no"], how="inner")
    div_idx = {}
    for rid, g in divs.groupby("race_id"):
        rows = [(p, c, d) for p, c, d in zip(g["pool"], g["combination"], g["dividend"]) if d is not None and pd.notna(d)]
        idx = bets.index_dividends(rows)
        if pool_live(idx, "WIN"):           # race run and WIN pool not refunded
            div_idx[rid] = idx
    info = races.set_index("race_id")
    meetings: dict[tuple, dict] = {}
    for rid, g in live.groupby("race_id", sort=False):
        if rid not in div_idx or len(g) < 4:
            continue                          # void / abandoned / no results / tiny field
        r = info.loc[rid]
        key = (r["date"], r["venue"])
        m = meetings.setdefault(key, {"date": r["date"], "venue": r["venue"],
                                      "season": season_of(r["date"]), "races": {}})
        g = g.sort_values("horse_no")
        df = pd.DataFrame({"race_no": int(r["race_no"]), "horse_no": g["horse_no"].astype(int).values,
                           "p_win": g["p_win"].astype(float).values,
                           "p_place": g["p_place"].astype(float).values,
                           "win_odds": g["win_odds"].astype(float).values,
                           "p_mkt": g["p_mkt"].astype(float).values,
                           "p_mkt_place": g["p_mkt_place"].astype(float).values})
        m["races"][int(r["race_no"])] = {
            "race_id": rid, "surface": r["surface"], "df": df, "idx": div_idx[rid],
            "scr": scr.get(rid, set()), "orders": st.race_orders(df), "finish": fin.get(rid, {}),
            "profile": st.race_profile(df)}
    return [meetings[k] for k in sorted(meetings)]


# ----------------------------------------------------------------- settlement helpers
def pool_live(idx, pool) -> bool:
    """Pool offered and not refunded in this race (bets.REFUND_KEY marks refunds)."""
    rows = idx.get(pool)
    return bool(rows) and bets.REFUND_KEY not in rows


def settle_combos(pool, combos, idx, scratched, unit=UNIT):
    """Fast equivalent of bets.settle_single with a pre-built dividend index.
    Returns (stake, payout, n_winning_units)."""
    pay, nwin = 0.0, 0
    for c in combos:
        status, factor, _ = bets.combo_outcome(pool, c, idx, scratched)
        pay += unit * factor
        nwin += status == "win"
    return unit * len(combos), pay, nwin


def _filter_ok(race, filt):
    if not filt:
        return True
    o = race["orders"]
    top = o["p_win"][0]
    df = race["df"]
    row = df[df["horse_no"] == top].iloc[0]
    if filt[0] == "pw":
        return row["p_win"] >= filt[1]
    if filt[0] == "edge":
        return row["p_mkt"] > 0 and row["p_win"] / row["p_mkt"] >= filt[1]
    raise ValueError(filt)


def _meeting_frame(m, pool):
    """Prediction frame of the races of a meeting where ``pool`` was offered."""
    dfs = [r["df"] for r in m["races"].values() if pool_live(r["idx"], pool)]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _allup_tickets(s, frame):
    if s["kind"] == "allup":
        t = st.build_allup(frame, pool=s["pool"], n_legs=s["n_legs"], spec=s["spec"],
                           horses_per_leg=s["hpl"], race_rule=s["rule"], fixed_races=s.get("fixed"),
                           unit_stake=UNIT, strategy=s["name"], _prepped=True)
        return [t] if t else []
    return st.build_multi_allup(frame, pool=s["pool"], n_tickets=s["n_tickets"], legs_per_ticket=s["lpt"],
                                spec=s["spec"], horses_per_leg=s["hpl"], race_rule=s["rule"],
                                pairing=s.get("pairing", "chrono"), unit_stake=UNIT,
                                strategy=s["name"], _prepped=True)


def run_grid(conn, source="market", start=None, end=None, strategies=None, meetings=None) -> dict:
    """Chronological per-meeting simulation of every strategy.  One row per bet
    (a multi-ticket product such as 3 x QPL 2x3 counts as ONE bet per meeting)."""
    t0 = time.time()
    meetings = meetings if meetings is not None else load_data(conn, source, start, end)
    strategies = strategies or default_strategies(source)
    rec = defaultdict(list)

    def put(s, mi, m, surface, stake, pay, nwin, rn=0):
        rec["race_no"].append(rn)
        rec["sid"].append(s["sid"]); rec["mi"].append(mi); rec["date"].append(m["date"])
        rec["venue"].append(m["venue"]); rec["season"].append(m["season"])
        rec["surface"].append(surface); rec["stake"].append(stake); rec["payout"].append(pay)
        rec["hit"].append(nwin > 0)

    allups = [s for s in strategies if s["kind"] in ("allup", "multi")]
    singles = [s for s in strategies if s["kind"] == "single"]
    for mi, m in enumerate(meetings):
        races = m["races"]
        frames = {}
        for s in allups:
            pool = s["pool"]
            if pool not in frames:
                frames[pool] = _meeting_frame(m, pool)
            if frames[pool] is None:
                continue
            tickets = _allup_tickets(s, frames[pool])
            if not tickets:
                continue
            stake = pay = 0.0
            nwin = 0
            surf = set()
            for t in tickets:
                dbr = {l["race_no"]: races[l["race_no"]]["idx"] for l in t["legs"]}
                scr = {l["race_no"]: races[l["race_no"]]["scr"] for l in t["legs"]}
                res = bets.settle_allup(t["pool"], t["legs"], t["spec"], dbr, UNIT, scr)
                assert abs(res["stake"] - t["total_stake"]) < 1e-6
                stake += res["stake"]; pay += res["payout"]; nwin += res["n_winning_units"]
                surf |= {races[l["race_no"]]["surface"] for l in t["legs"]}
            put(s, mi, m, surf.pop() if len(surf) == 1 else "MIX", stake, pay, nwin)
        for rn in sorted(races):
            race = races[rn]
            fcache = {}
            for s in singles:
                sp = s["sp"]
                pool = sp["pool"]
                if not pool_live(race["idx"], pool):
                    continue                  # pool not offered / refunded in this race
                f = s.get("filt")
                if f:
                    if f not in fcache:
                        fcache[f] = _filter_ok(race, f)
                    if not fcache[f]:
                        continue
                leg = st.single_selection(None, sp, orders=race["orders"])
                if leg is None:
                    continue
                combos = st.leg_combos(leg)
                stake, pay, nwin = settle_combos(pool, combos, race["idx"], race["scr"])
                put(s, mi, m, race["surface"], stake, pay, nwin, rn)
    bets_df = pd.DataFrame(rec)
    mt = pd.DataFrame([{"mi": i, "date": m["date"], "venue": m["venue"], "season": m["season"],
                        "n_races": len(m["races"])} for i, m in enumerate(meetings)])
    return {"bets": bets_df, "strategies": strategies, "meetings": mt, "source": source,
            "seconds": time.time() - t0, "meeting_data": meetings}


# ----------------------------------------------------------------- metrics
def _max_drawdown(profit):
    cum = np.cumsum(profit)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum)) if len(cum) else 0.0


def _longest_losing(profit):
    best = cur = 0
    for p in profit:
        cur = cur + 1 if p < 0 else 0
        best = max(best, cur)
    return best


def _group_profit(g, col):
    out = {}
    for k, x in g.groupby(col):
        stake = x["stake"].sum()
        out[str(k)] = {"n": int(len(x)), "stake": round(float(stake), 2),
                       "profit": round(float(x["payout"].sum() - stake), 2),
                       "roi": round(float(x["payout"].sum() / stake - 1), 4) if stake else None}
    return out


def summarise(bets_df, strategies, meetings, seasons=None, n_boot=2000, seed=0) -> pd.DataFrame:
    """Metrics per strategy (optionally restricted to ``seasons``)."""
    b = bets_df
    mt = meetings
    if seasons is not None:
        b = b[b["season"].isin(seasons)]
        mt = mt[mt["season"].isin(seasons)]
    mis = mt["mi"].to_numpy()
    pos = {m: i for i, m in enumerate(mis)}
    M = len(mis)
    S = len(strategies)
    stake_m = np.zeros((S, M)); ret_m = np.zeros((S, M))
    if len(b) and M:
        agg = b.groupby(["sid", "mi"])[["stake", "payout"]].sum().reset_index()
        agg = agg[agg["mi"].isin(pos)]
        r = agg["sid"].to_numpy(); c = np.array([pos[x] for x in agg["mi"]])
        stake_m[r, c] = agg["stake"].to_numpy(); ret_m[r, c] = agg["payout"].to_numpy()
    lo = np.full(S, np.nan); hi = np.full(S, np.nan)
    if M:
        rng = np.random.default_rng(seed)
        W = rng.multinomial(M, np.full(M, 1.0 / M), size=n_boot).T.astype(float)   # (M, B)
        sb, rb = stake_m @ W, ret_m @ W
        with np.errstate(invalid="ignore", divide="ignore"):
            roi_b = np.where(sb > 0, rb / sb - 1, np.nan)
        ok = ~np.all(np.isnan(roi_b), axis=1)
        lo[ok] = np.nanpercentile(roi_b[ok], 2.5, axis=1)
        hi[ok] = np.nanpercentile(roi_b[ok], 97.5, axis=1)
    rows = []
    groups = dict(tuple(b.groupby("sid"))) if len(b) else {}
    for s in strategies:
        g = groups.get(s["sid"])
        base = {"sid": s["sid"], "strategy": s["name"], "family": s["family"]}
        if g is None or not len(g):
            rows.append({**base, "n_bets": 0, "n_hits": 0, "hit_rate": np.nan, "stake": 0.0,
                         "returns": 0.0, "roi": np.nan, "profit": 0.0, "max_dd": 0.0, "lose_streak": 0,
                         "ci_lo": np.nan, "ci_hi": np.nan, "avg_stake": np.nan, "best_payout": 0.0,
                         "profitable_rate": np.nan, "by_season": {}, "by_venue": {}, "by_surface": {},
                         "first": None, "last": None})
            continue
        prof = (g["payout"] - g["stake"]).to_numpy()
        stake, ret = float(g["stake"].sum()), float(g["payout"].sum())
        rows.append({**base, "n_bets": int(len(g)), "n_hits": int(g["hit"].sum()),
                     "hit_rate": float(g["hit"].mean()), "stake": stake, "returns": ret,
                     "roi": ret / stake - 1 if stake else np.nan, "profit": ret - stake,
                     "max_dd": _max_drawdown(prof), "lose_streak": _longest_losing(prof),
                     "ci_lo": lo[s["sid"]], "ci_hi": hi[s["sid"]],
                     "avg_stake": stake / len(g), "best_payout": float(g["payout"].max()),
                     "profitable_rate": float((prof > 0).mean()),
                     "by_season": _group_profit(g, "season"), "by_venue": _group_profit(g, "venue"),
                     "by_surface": _group_profit(g, "surface"),
                     "first": g["date"].min(), "last": g["date"].max()})
    return pd.DataFrame(rows)


def sanity(meetings) -> dict:
    """Market-favourite sanity numbers + data consistency checks."""
    n = wins = 0
    stake = ret = 0.0
    over = []
    match = tot = 0
    for m in meetings:
        for r in m["races"].values():
            df = r["df"]
            fav = df.sort_values(["win_odds", "horse_no"]).iloc[0]
            div = r["idx"]["WIN"]
            n += 1
            stake += 10
            d = div.get(str(int(fav["horse_no"])))
            if d:
                wins += 1
                ret += d
            over.append(float((1.0 / df["win_odds"]).sum()))
            if len(div) == 1:                      # WIN dividend ~ 10 x final odds? (skip dead heats)
                for combo, dv in div.items():
                    o = df.loc[df["horse_no"] == int(combo), "win_odds"]
                    if len(o) and dv:
                        tot += 1
                        # odds >= 10 are displayed truncated to integers -> allow 10% slack
                        match += abs(o.iloc[0] * 10 - dv) <= max(0.51, 0.1 * dv)
    return {"races": n, "fav_win_rate": wins / n if n else None,
            "fav_roi": ret / stake - 1 if stake else None,
            "avg_overround": float(np.mean(over)) if over else None,
            "implied_takeout": 1 - 1 / float(np.mean(over)) if over else None,
            "win_div_matches_odds": match / tot if tot else None}


# ----------------------------------------------------------------- persistence
def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def write_results(conn, source, tables: dict, strategies) -> int:
    """tables: {period_label: summary DataFrame}.  Replaces earlier rows of this
    BT_VERSION + source."""
    by_sid = {s["sid"]: s for s in strategies}
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for period, df in tables.items():
        for r in df.to_dict("records"):
            s = by_sid[r["sid"]]
            cfg = {k: v for k, v in s.items() if k not in ("sid", "name", "family")}
            extra = {k: _clean(r[k]) for k in ("profit", "lose_streak", "ci_lo", "ci_hi", "avg_stake",
                                                 "best_payout", "profitable_rate")}
            extra.update(by_season=r["by_season"], by_venue=r["by_venue"], by_surface=r["by_surface"])
            params = {"bt": BT_VERSION, "source": source, "period": period, "family": s["family"],
                      "config": cfg, "metrics": extra}
            rows.append((s["name"], json.dumps(params, ensure_ascii=False, default=str), r["first"], r["last"],
                         int(r["n_bets"]), int(r["n_hits"]), _clean(r["hit_rate"]), _clean(r["stake"]),
                         _clean(r["returns"]), _clean(r["roi"]), _clean(r["max_dd"]), now))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM backtest_results WHERE json_extract(params,'$.bt')=? "
                     "AND json_extract(params,'$.source')=?", (BT_VERSION, source))
        conn.executemany("INSERT INTO backtest_results (strategy, params, period_start, period_end, n_bets, "
                         "n_hits, hit_rate, stake, returns, roi, max_drawdown, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


# ----------------------------------------------------------------- report
def _p(x, signed=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _m(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:,.0f}"


def _table(df, cols=None, title_col="策略"):
    L = [f"| {title_col} | 注數 | 命中率 | 盈利率 | 平均每注 | 總投注 | 總回報 | ROI | ROI 95%區間 | 盈虧 | 最大回撤 | 最長連敗 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in df.to_dict("records"):
        if not r["n_bets"]:
            L.append(f"| {r['strategy']} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {r['strategy']} | {r['n_bets']} | {_p(r['hit_rate'], False)} | {_p(r['profitable_rate'], False)} | {_m(r['avg_stake'])} | "
                 f"{_m(r['stake'])} | {_m(r['returns'])} | **{_p(r['roi'])}** | "
                 f"{_p(r['ci_lo'])} ~ {_p(r['ci_hi'])} | {_m(r['profit'])} | {_m(r['max_dd'])} | {r['lose_streak']} |")
    return L


def _breakdown(df, names, key, label):
    sub = df[df["strategy"].isin(names)]
    keys = sorted({k for d in sub[key] for k in d})
    L = [f"| 策略 | " + " | ".join(f"{label}{k} ROI (注數)" for k in keys) + " |",
         "|---|" + "---:|" * len(keys)]
    for r in sub.to_dict("records"):
        cells = []
        for k in keys:
            v = r[key].get(k)
            cells.append("-" if not v else f"{_p(v['roi'])} ({v['n']})")
        L.append(f"| {r['strategy']} | " + " | ".join(cells) + " |")
    return L


SRC_LABEL = {"market": "市場基準（最終獨贏賠率隱含機率 + Harville）"}


def _source_section(res, sm_all, sm_is, sm_oos, san):
    src = res["source"]
    mt = res["meetings"]
    L = [f"## 預測來源：`{src}` — {SRC_LABEL.get(src, '模型 walk-forward 預測')}", ""]
    L += [f"- 期間：{mt['date'].min()} 至 {mt['date'].max()}，{len(mt)} 個賽馬日，"
          f"{int(mt['n_races'].sum())} 場有效賽事（已剔除取消/無派彩場次）。",
          f"- 每季賽馬日：" + "，".join(f"{k} {v}" for k, v in mt.groupby('season').size().items()),
          f"- 單位注碼 $10；派彩按馬會每 $10 派彩計；回測耗時 {res['seconds']:.0f} 秒。", ""]
    if san:
        L += ["### 數據核對 (sanity check)", "",
              "| 項目 | 數值 | 預期 |", "|---|---:|---|",
              f"| 市場大熱門 (最低獨贏賠率) 勝出率 | {_p(san['fav_win_rate'], False)} | 約 30–35% |",
              f"| 每場買大熱門獨贏 ROI | {_p(san['fav_roi'])} | 約 −10% 至 −25% |",
              f"| 平均 Σ(1/獨贏賠率) (overround) | {san['avg_overround']:.3f} | 約 1.21 (扣 17.5%) |",
              f"| 由賠率反推扣數 | {_p(san['implied_takeout'], False)} | 17.5% |",
              f"| 獨贏派彩 ≈ 最終賠率×10 (容許 10%，因 10 倍以上賠率只顯示整數) | {_p(san['win_div_matches_odds'], False)} | ≈100% |", ""]
    fam = lambda df, f: df[df["family"] == f]
    L += ["### 一、兩個固定組合（每個賽馬日一次）", "",
          "PLA 6x63：6 場各揀 1 隻（位置機率最高），63 注 = $630。3×QPL 2x3：6 場分 3 條，每場 3 隻，"
          "每條 15 注，共 45 注 = $450。`[]` 內為揀場方法：confidence=信心最高、value=模型相對市場最抵、"
          "first6=頭六場、last6=尾六場、fixed=第2–7場、consecutive=信心最高嘅相連兩場一組。", ""]
    L += _table(fam(sm_all, "固定組合")) + [""]
    L += ["### 二、過關結構比較（信心揀場）", ""]
    L += _table(pd.concat([sm_all[sm_all["strategy"].isin(["PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]"])],
                           fam(sm_all, "過關結構")])) + [""]
    L += ["### 三、單場玩法網格（每場一注）", "",
          "排名：WIN/QIN/TRI/FF/TCE 按勝出機率，PLA/QPL 按位置機率；膽 = 排第 1（或 1–2）嘅馬，"
          "腳 = 跟住排名嘅馬。", ""]
    single = fam(sm_all, "單場").copy()
    single["_pool"] = single["strategy"].str.split(" ").str[0]
    for pool, g in single.groupby("_pool", sort=False):
        L += [f"**{pool}**（扣數約 {TAKEOUT.get(pool, '-')}）", ""] + _table(g) + [""]
    L += ["### 四、篩選條件（只喺符合條件嘅場次落注）", ""] + _table(fam(sm_all, "篩選")) + [""]
    key = ["PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]", "WIN top1", "PLA top1",
           "QPL 複式3", "TRI 膽1x腳4", "FF 膽1x腳5"]
    L += ["### 五、分季 / 場地 / 跑道", ""] + _breakdown(sm_all, key, "by_season", "") + [""]
    L += _breakdown(sm_all, key, "by_venue", "") + [""] + _breakdown(sm_all, key, "by_surface", "") + [""]
    # out-of-sample
    L += ["### 六、樣本外檢驗（防止「揀中咗運氣」）", "",
          f"一共測試咗 {len(sm_all)} 個策略。策略越多，純粹靠運氣出現一個「好」ROI 嘅機會越大，"
          "所以喺 2024-25 季（樣本內，至少 50 注）揀出最好嘅策略，再睇佢喺 2025-26 及 2026-27 季至今（樣本外）嘅表現。", ""]
    ok = sm_is[sm_is["n_bets"] >= MIN_BETS].sort_values("roi", ascending=False)
    oos = sm_oos.set_index("sid")
    L += ["| 樣本內排名 | 策略 | 樣本內注數 | 樣本內 ROI | 樣本內 95%區間 | 樣本外注數 | 樣本外 ROI | 樣本外 95%區間 |",
          "|---:|---|---:|---:|---|---:|---:|---|"]
    for i, r in enumerate(ok.head(10).to_dict("records"), 1):
        o = oos.loc[r["sid"]]
        L.append(f"| {i} | {r['strategy']} | {r['n_bets']} | {_p(r['roi'])} | {_p(r['ci_lo'])} ~ {_p(r['ci_hi'])} | "
                 f"{o['n_bets']} | **{_p(o['roi'])}** | {_p(o['ci_lo'])} ~ {_p(o['ci_hi'])} |")
    j = ok[["sid", "roi"]].merge(sm_oos[["sid", "roi", "n_bets"]], on="sid", suffixes=("_is", "_oos"))
    j = j[j["n_bets"] >= MIN_BETS]
    rho = j["roi_is"].rank().corr(j["roi_oos"].rank()) if len(j) > 3 else np.nan
    n_pos_is = int((ok["roi"] > 0).sum())
    n_pos_oos = int((sm_oos[sm_oos["n_bets"] >= MIN_BETS]["roi"] > 0).sum())
    n_sig = int((sm_all[sm_all["n_bets"] >= MIN_BETS]["ci_lo"] > 0).sum())
    L += ["", f"- 全期 95% 區間下限高於 0（即統計上顯著賺錢）嘅策略：**{n_sig}/{len(sm_all)}**。"
          f"喺 {len(sm_all)} 個策略入面，就算全部真實 ROI 都係負數，純運氣都可能有一兩個區間高於 0。",
          f"- 樣本內 ROI 與樣本外 ROI 嘅等級相關 (Spearman)：**{rho:.2f}**"
          f"（接近 0 = 樣本內嘅排名對將來冇乜預測力；正數多數係因為唔同玩法扣數唔同、組合越多越蝕，"
          f"屬結構性差異，唔代表揀中咗有優勢嘅策略）。" if np.isfinite(rho) else "",
          f"- 樣本內正回報策略：{n_pos_is}/{len(ok)}；樣本外正回報策略：{n_pos_oos}/"
          f"{int((sm_oos['n_bets'] >= MIN_BETS).sum())}。", ""]
    return L, {"rho": rho, "best_is": ok.head(1).to_dict("records"),
               "best_is_oos": (oos.loc[ok.iloc[0]["sid"]].to_dict() if len(ok) else None)}


def _conclusions(results):
    """Plain-language conclusion from the first (preferably model) source."""
    L = ["## 結論（白話版）", ""]
    for src, d in results.items():
        sm = d["all"]
        name = SRC_LABEL.get(src, f"模型 `{src}`")
        L.append(f"### 以 {name} 計")
        L.append("")
        g = sm[sm["n_bets"] >= 50]
        single = g[g["family"] == "單場"]
        if len(single):
            hr = single.sort_values("hit_rate", ascending=False).head(3)
            rr = single.sort_values("roi", ascending=False).head(3)
            L.append("1. **邊種玩法命中率高 / 回報高？** 命中率最高："
                     + "、".join(f"{r['strategy']} ({_p(r['hit_rate'], False)}，ROI {_p(r['roi'])})" for r in hr.to_dict("records"))
                     + "。ROI 最高："
                     + "、".join(f"{r['strategy']} ({_p(r['roi'])}，95%區間 {_p(r['ci_lo'])}~{_p(r['ci_hi'])})" for r in rr.to_dict("records"))
                     + "。命中率高嘅通常係位置/位置Q（易中但派彩細），回報率反而受扣數同大熱門偏差影響。")
        fx = sm[sm["family"] == "固定組合"].sort_values("roi", ascending=False)
        if len(fx):
            pla = fx[fx["strategy"].str.startswith("PLA")]
            qpl = fx[fx["strategy"].str.startswith("3xQPL")]
            L.append("2. **過關點揀場？** PLA 6x63 各揀場方法 ROI："
                     + "、".join(f"{r['strategy'].split('[')[1][:-1]} {_p(r['roi'])}" for r in pla.to_dict("records"))
                     + "；3×QPL 2x3："
                     + "、".join(f"{r['strategy'].split('[')[1][:-1]} {_p(r['roi'])}" for r in qpl.to_dict("records"))
                     + "。如果各方法嘅 95% 區間大幅重疊，即係差別多數係運氣。")
        al = sm[sm["family"].isin(["過關結構"]) | sm["strategy"].isin(["PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]"])]
        if len(al):
            b = al.sort_values("roi", ascending=False)
            L.append("3. **過關結構？** 由好到差："
                     + "、".join(f"{r['strategy']} {_p(r['roi'])}" for r in b.to_dict("records"))
                     + "。關數越多、注項越集中喺高關數組合，每多一關就多扣一次彩池扣數，長線回報通常越差但偶然派彩越大。")
            pt = sm[sm["strategy"] == "PLA top1"]
            if len(pt) and np.isfinite(pt.iloc[0]["roi"]):
                f = 1 + float(pt.iloc[0]["roi"])
                L.append(f"   - 點解位置過關咁蝕？單場買位置首選每 $1 平均返 {f:.3f}；過關係將每關回報相乘："
                         f"2 關 ≈ {f ** 2:.2f}、3 關 ≈ {f ** 3:.2f}、6 關 ≈ {f ** 6:.2f}。"
                         "6x63 大部份注項係 3 關或以上，所以整體回報率遠低過單場。")
        L.append("4. **膽拖定複式？幾多膽幾多腳？**（每個彩池最好嘅一款複式 vs 最好嘅一款膽拖）")
        for pool in ("QIN", "QPL", "TRI", "FF", "TCE"):
            gp = single[single["strategy"].str.startswith(pool + " ")]
            bx = gp[gp["strategy"].str.contains("複式")].sort_values("roi", ascending=False).head(1)
            bk = gp[gp["strategy"].str.contains("膽")].sort_values("roi", ascending=False).head(1)
            if len(bx) and len(bk):
                a, c = bx.iloc[0], bk.iloc[0]
                L.append(f"   - {pool}：複式 {a['strategy']} ROI {_p(a['roi'])}（命中 {_p(a['hit_rate'], False)}，"
                         f"區間 {_p(a['ci_lo'])}~{_p(a['ci_hi'])}） vs 膽拖 {c['strategy']} ROI {_p(c['roi'])}"
                         f"（命中 {_p(c['hit_rate'], False)}，區間 {_p(c['ci_lo'])}~{_p(c['ci_hi'])}）。")
        oo = d.get("oos_info") or {}
        if oo.get("best_is"):
            bi = oo["best_is"][0]
            bo = oo["best_is_oos"] or {}
            L.append(f"5. **樣本外：** 2024-25 季表現最好嘅係 {bi['strategy']}（ROI {_p(bi['roi'])}），"
                     f"去到 2025-26 季之後變成 {_p(bo.get('roi'))}（{bo.get('n_bets', 0)} 注）。"
                     f"樣本內外排名相關度 {oo['rho']:.2f}。")
        L.append("")
    L += ["### 總括", "",
          "- 馬會每個彩池先扣 17.5%–25%，再加上過關時逐關疊加，任何「機械式」玩法嘅長線期望值都係負數；"
          "回測入面 95% 區間完全高於 0 嘅策略先值得再跟進，而且要喺樣本外再驗證。",
          "- 命中率同回報率係兩回事：位置、位置Q 命中率高但派彩低；三T、四連環命中率低但一次派彩大，"
          "盈虧好波動（睇「最大回撤」同「最長連敗」）。",
          "- 揀場方法、膽拖 vs 複式之間嘅差距，好多時細過 95% 區間嘅闊度 —— 即係未有足夠證據話邊個一定較好。",
          "- 模型同市場（最終賠率）比較：如果模型策略唔能夠穩定贏過市場基準，即係模型未有真正優勢。", ""]
    return L


def render_report(results: dict) -> str:
    L = ["# 香港賽馬投注策略回測報告", "",
         f"_產生時間：{datetime.now():%Y-%m-%d %H:%M}（{BT_VERSION}）_", "",
         "> **重要聲明**：本報告只屬歷史數據模擬（paper trade），並非投資或投注建議。過往表現不代表將來。", "",
         "## 方法及注意事項", "",
         "- 用真實馬會派彩（每 $10 派彩）逐場結算：平頭馬按每個派彩組合各自派彩；退出馬有關組合退款（過關則該關當 1 倍過關）；"
         "彩池無開售（例如少於 7 匹馬嘅位置Q）嘅場次單場注不落、過關不選。",
         "- **用咗最終獨贏賠率**：市場基準嘅機率同「模型/市場」值博篩選都係用收市賠率，"
         "實際上開跑前 2 小時嘅賠率會唔同（有偷看未來嘅成份），所以回測結果會偏樂觀。",
         "- 馬會扣數（馬會 Local Pools 頁面）：獨贏/位置/連贏/位置Q 17.5%、二重彩 19.5%、單T 23%、三重彩/四連環/四重彩 25%；過關每關都要扣，"
         "所以長線要贏錢非常困難。",
         "- ROI = 總回報 ÷ 總投注 − 1；命中率 = 有中獎組合嘅注數比例（過關只要有任何一個組合中就算）；"
         "盈利率 = 派彩多過投注額嘅注數比例；最長連敗 = 連續蝕錢嘅注數。"
         "ROI 95% 區間用賽馬日重抽樣 (bootstrap, 2000 次)。",
         "- 多重比較風險：測試咗過百個策略，最好嗰個嘅 ROI 一定偏高（純運氣都會有幾個好睇嘅）。"
         "所以第六節用 2024-25 季揀最好嘅策略，再睇 2025-26 季之後嘅樣本外表現。", ""]
    sections = []
    for src, d in results.items():
        sec, oinfo = _source_section(d["res"], d["all"], d["is"], d["oos"], d["sanity"])
        d["oos_info"] = oinfo
        sections += sec
    # cross-source comparison of key strategies (common seasons only)
    if len(results) > 1:
        L += ["## 模型 vs 市場基準（同一期間：2024-25 季起）", "",
              "| 策略 | " + " | ".join(f"`{s}` ROI (注數)" for s in results) + " |",
              "|---|" + "---:|" * len(results)]
        names = ["PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]", "WIN top1", "PLA top1", "QIN 複式2",
                 "QPL 複式3", "QPL 膽1x腳3", "TRI 複式4", "TRI 膽1x腳4", "FF 膽1x腳5", "TCE 膽1x腳4(膽定首位)"]
        for n in names:
            cells = []
            for s, d in results.items():
                c = d["common"]
                r = c[c["strategy"] == n]
                cells.append("-" if not len(r) else f"{_p(r.iloc[0]['roi'])} ({r.iloc[0]['n_bets']})")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L.append("")
    L += _conclusions(results)
    L += adaptive_report(results)
    L += sections
    return "\n".join(L)



# ================================================================= ADAPTIVE backtest
AD_POOLS_CAL = ("PLA", "QIN", "QPL", "TRI", "FF")
AD_GRID = [(tau, h) for tau in (-0.05, 0.0, 0.1, 0.2, 0.4) for h in (0.05, 0.15, 0.30)]
AU_GRID = [(pl, qh, e) for pl, qh in ((0.45, 0.30), (0.55, 0.40), (0.65, 0.50)) for e in (1.0, 1.05, 1.10)]
AD_MIN_IS_BETS = 100          # single-race: a setting must bet >= 100 races in 2024-25
AU_MIN_IS_BETS = 20           # all-up: >= 20 meetings with a parlay in 2024-25
P1_BUCKETS = [(0, .20, "p1<20%"), (.20, .30, "20–30%"), (.30, .40, "30–40%"), (.40, 1.01, "p1≥40%")]
GAP_BUCKETS = [(-1, .05, "差距<5%"), (.05, .15, "5–15%"), (.15, 1.01, "差距≥15%")]


def _bucket(x, buckets):
    for lo, hi, lab in buckets:
        if lo <= x < hi:
            return lab
    return buckets[-1][2]


def _structure_cat(spec) -> str:
    if "top" in spec:
        return f"{spec['pool']} 單選" if spec["top"] == 1 else f"{spec['pool']} 兩隻"
    if "box" in spec:
        return f"{spec['pool']} 複式"
    if "bankers" in spec:
        return f"{spec['pool']} {spec['bankers']}膽拖"
    return spec["pool"]


def fit_lambdas(meetings, seasons=IS_SEASONS, grid=None):
    """Stern/Henery-style discount for 2nd / 3rd place: maximise the likelihood of
    the observed 2nd (given 1st) and 3rd (given 1st, 2nd) under p**lam weights."""
    grid = grid if grid is not None else np.round(np.arange(0.5, 1.31, 0.05), 2)
    data = []
    for m in meetings:
        if m["season"] not in seasons:
            continue
        for r in m["races"].values():
            fin, df = r["finish"], r["df"]
            pos = {}
            for h, f in fin.items():
                pos.setdefault(f, []).append(h)
            if any(len(pos.get(k, [])) != 1 for k in (1, 2, 3)):
                continue                              # dead heat / missing
            p = dict(zip(df["horse_no"], df["p_win"].astype(float)))
            h1, h2, h3 = pos[1][0], pos[2][0], pos[3][0]
            if not all(p.get(h, 0) > 0 for h in (h1, h2, h3)):
                continue
            data.append((np.array(list(p.values())), p[h1], p[h2], p[h3]))
    if not data:
        return 1.0, 1.0, 0

    def ll(lam, k):
        tot = 0.0
        for arr, a, b, c in data:
            q = arr ** lam
            if k == 2:
                tot += lam * np.log(b) - np.log(q.sum() - a ** lam)
            else:
                tot += lam * np.log(c) - np.log(q.sum() - a ** lam - b ** lam)
        return tot
    l2 = float(max(grid, key=lambda g: ll(g, 2)))
    l3 = float(max(grid, key=lambda g: ll(g, 3)))
    return l2, l3, len(data)


def fit_calibration(meetings, seasons=IS_SEASONS, lam2=1.0, lam3=1.0):
    """log(actual dividend) = a + b*log(raw market estimate) per pool, fitted on the
    winning combinations of ``seasons`` (a combination's dividend is set by the pool
    money, independent of the result, so regressing on winners is unbiased)."""
    xs = {p: [] for p in AD_POOLS_CAL}
    for m in meetings:
        if m["season"] not in seasons:
            continue
        for r in m["races"].values():
            est = st.DividendEstimator(r["df"], None, None, lam2, lam3)
            for pool in AD_POOLS_CAL:
                if not pool_live(r["idx"], pool):
                    continue
                for combo, d in r["idx"][pool].items():
                    if not d or combo == bets.REFUND_KEY:
                        continue
                    raw = est.raw_market(pool, combo)
                    if raw:
                        xs[pool].append((np.log(raw), np.log(d)))
    cal, stats = {}, {}
    for pool, v in xs.items():
        if len(v) < 30:
            continue
        a = np.array(v)
        b, c = np.polyfit(a[:, 0], a[:, 1], 1)
        cal[pool] = (float(c), float(b))
        stats[pool] = {"n": len(v), "a": float(c), "b": float(b),
                       "median_ratio": float(np.median(np.exp(a[:, 1] - a[:, 0])))}
    return cal, stats


def adaptive_precompute(meetings, lam2=1.0, lam3=1.0, calib=None):
    """Per race: every candidate's (exp_roi, hit, stake) and its actual settlement;
    per meeting: all-up leg candidates.  Thresholds are applied later."""
    cfg = {"lam2": lam2, "lam3": lam3, "calib": calib or {}}
    races, allup = [], []
    for mi, m in enumerate(meetings):
        infos = {"PLA": [], "QPL": []}
        for rn in sorted(m["races"]):
            r = m["races"][rn]
            df = r["df"]
            probs = st.RaceProbs(df["horse_no"], df["p_win"].to_numpy(), lam2, lam3)
            est = st.DividendEstimator(df, None, cfg["calib"], lam2, lam3)
            cands = st.race_candidates(df, None, cfg, probs=probs, est=est)
            out = []
            for c in cands:
                if not pool_live(r["idx"], c["pool"]):
                    continue
                stake, pay, nwin = settle_combos(c["pool"], c["combos"], r["idx"], r["scr"])
                out.append({"name": c["name"], "cat": _structure_cat(c["spec"]), "spec": c["spec"],
                            "exp_roi": c["exp_roi"], "hit": c["hit"], "stake": stake,
                            "payout": pay, "won": nwin > 0})
            races.append({"mi": mi, "race_no": rn, "surface": r["surface"], "profile": r["profile"],
                          "cands": out})
            for pool in infos:
                if not pool_live(r["idx"], pool):
                    continue
                li = st.allup_leg_info(df, pool, probs, est, 3)
                if li:
                    infos[pool].append(li)
        allup.append(infos)
    return {"races": races, "allup": allup}


def adaptive_records(pre, meetings, tau, h_min, max_stake=200.0):
    rows = []
    for r in pre["races"]:
        ok = [c for c in r["cands"] if c["exp_roi"] >= tau and c["hit"] >= h_min and c["stake"] <= max_stake]
        if not ok:
            continue
        c = max(ok, key=lambda c: (round(c["exp_roi"], 12), c["hit"], -c["stake"]))
        m = meetings[r["mi"]]
        pr = r["profile"]
        rows.append({"mi": r["mi"], "date": m["date"], "venue": m["venue"], "season": m["season"],
                     "surface": r["surface"], "race_no": r["race_no"], "stake": c["stake"],
                     "payout": c["payout"], "hit": c["won"], "structure": c["name"], "cat": c["cat"],
                     "pool": c["spec"]["pool"], "n_bankers": c["spec"].get("bankers", 0),
                     "n_legs": c["spec"].get("legs", c["spec"].get("box", c["spec"].get("top", 0))),
                     "exp_roi": c["exp_roi"], "hit_prob": c["hit"], "p1": pr["p1"], "gap": pr["gap"],
                     "edge1": pr.get("edge1")})
    return pd.DataFrame(rows)


def allup_records(pre, meetings, pla_min_p, qpl_min_hit, min_edge, cache=None):
    cache = {} if cache is None else cache
    cfg = {"pla_min_p": pla_min_p, "qpl_min_hit": qpl_min_hit, "min_edge": min_edge, "tau": 0.0}
    rows = []
    for mi, infos in enumerate(pre["allup"]):
        m = meetings[mi]
        tickets = st.adaptive_allup_from_legs(infos, cfg)
        if not tickets:
            continue
        stake = pay = 0.0
        nwin = 0
        desc = []
        for t in tickets:
            key = (mi, t["pool"], t["spec"], tuple((l["race_no"], tuple(l["selections"])) for l in t["legs"]))
            if key not in cache:
                dbr = {l["race_no"]: m["races"][l["race_no"]]["idx"] for l in t["legs"]}
                scr = {l["race_no"]: m["races"][l["race_no"]]["scr"] for l in t["legs"]}
                res = bets.settle_allup(t["pool"], t["legs"], t["spec"], dbr, UNIT, scr)
                cache[key] = (res["stake"], res["payout"], res["n_winning_units"])
            a, b, c = cache[key]
            stake += a; pay += b; nwin += c
            desc.append(f"{t['pool']} {t['spec']}")
        rows.append({"mi": mi, "date": m["date"], "venue": m["venue"], "season": m["season"],
                     "surface": "MIX", "race_no": 0, "stake": stake, "payout": pay, "hit": nwin > 0,
                     "structure": " + ".join(desc), "n_tickets": len(tickets),
                     "n_legs": max(len(t["legs"]) for t in tickets)})
    return pd.DataFrame(rows)


def _quick(df, seasons=None):
    if df is None or not len(df):
        return {"n": 0, "stake": 0.0, "roi": np.nan, "hit": np.nan, "profit": 0.0}
    d = df if seasons is None else df[df["season"].isin(seasons)]
    if not len(d):
        return {"n": 0, "stake": 0.0, "roi": np.nan, "hit": np.nan, "profit": 0.0}
    stake = d["stake"].sum()
    return {"n": int(len(d)), "stake": float(stake), "roi": float(d["payout"].sum() / stake - 1),
            "hit": float(d["hit"].mean()), "profit": float(d["payout"].sum() - stake)}


def adaptive_backtest(res) -> dict:
    """Fit lam/calibration and tune thresholds on 2024-25 ONLY, then evaluate all
    seasons (OOS = 2025-26 + 2026-27)."""
    meetings = res["meeting_data"]
    t0 = time.time()
    lam2, lam3, n_lam = fit_lambdas(meetings)
    calib, cal_stats = fit_calibration(meetings, IS_SEASONS, lam2, lam3)
    pre = adaptive_precompute(meetings, lam2, lam3, calib)
    n_races_is = sum(1 for r in pre["races"] if meetings[r["mi"]]["season"] in IS_SEASONS)
    n_races_oos = sum(1 for r in pre["races"] if meetings[r["mi"]]["season"] in OOS_SEASONS)
    grid = []
    recs = {}
    for tau, h in AD_GRID:
        d = adaptive_records(pre, meetings, tau, h)
        recs[(tau, h)] = d
        grid.append({"tau": tau, "h_min": h, "is": _quick(d, IS_SEASONS), "oos": _quick(d, OOS_SEASONS)})
    # the robot never takes a bet it believes is negative-EV: only tau >= 0 is eligible
    # (negative tau rows are shown as diagnostics)
    elig = [g for g in grid if g["is"]["n"] >= AD_MIN_IS_BETS and g["tau"] >= 0]
    tuned = max(elig, key=lambda g: g["is"]["roi"]) if elig else None
    if tuned:
        tuned = {**tuned, "enabled": tuned["is"]["roi"] > 0}   # loses even in-sample -> robot skips
    au_grid, au_recs, cache = [], {}, {}
    for pl, qh, e in AU_GRID:
        d = allup_records(pre, meetings, pl, qh, e, cache)
        au_recs[(pl, qh, e)] = d
        au_grid.append({"pla_min_p": pl, "qpl_min_hit": qh, "min_edge": e,
                        "is": _quick(d, IS_SEASONS), "oos": _quick(d, OOS_SEASONS)})
    au_elig = [g for g in au_grid if g["is"]["n"] >= AU_MIN_IS_BETS]
    au_tuned = max(au_elig, key=lambda g: g["is"]["roi"]) if au_elig else None
    if au_tuned:
        au_tuned = {**au_tuned, "enabled": au_tuned["is"]["roi"] > 0}
    return {"lam2": lam2, "lam3": lam3, "n_lam": n_lam, "calib": calib, "cal_stats": cal_stats,
            "grid": grid, "tuned": tuned, "recs": recs, "au_grid": au_grid, "au_tuned": au_tuned,
            "au_recs": au_recs, "n_races_is": n_races_is, "n_races_oos": n_races_oos,
            "n_races": len(pre["races"]), "seconds": time.time() - t0}


def _append_adaptive(res, ad):
    """Add the adaptive settings to the strategy list / bets frame so that they get
    the same metrics, CIs and backtest_results rows as the fixed grid."""
    S = list(res["strategies"])
    frames = [res["bets"]]
    sid = len(S)

    def add(name, family, cfg, d):
        nonlocal sid
        S.append({"sid": sid, "name": name, "family": family, "kind": "adaptive", **cfg})
        if d is not None and len(d):
            x = d[["mi", "date", "venue", "season", "surface", "stake", "payout", "hit", "race_no"]].copy()
            x["sid"] = sid
            frames.append(x)
        sid += 1
    if ad["tuned"]:
        t = ad["tuned"]
        add(f"ADAPTIVE 按場自動 (τ={t['tau']:+.2f}, h≥{t['h_min']:.2f}){'' if t['enabled'] else ' [樣本內蝕→停用]'}", "自動",
            {"tau": t["tau"], "h_min": t["h_min"], "tuned_on": "2024-25", "lam2": ad["lam2"], "lam3": ad["lam3"],
             "calib": ad["calib"]}, ad["recs"][(t["tau"], t["h_min"])])
    if ad["au_tuned"]:
        t = ad["au_tuned"]
        key = (t["pla_min_p"], t["qpl_min_hit"], t["min_edge"])
        add(f"ADAPTIVE_ALLUP 自動過關 (PLA≥{key[0]:.2f}, QPL≥{key[1]:.2f}, edge≥{key[2]:.2f})"
            f"{'' if t['enabled'] else ' [樣本內蝕→停用]'}", "自動",
            {"pla_min_p": key[0], "qpl_min_hit": key[1], "min_edge": key[2], "tuned_on": "2024-25"},
            ad["au_recs"][key])
    for (tau, h), d in ad["recs"].items():
        add(f"自動單場網格 τ={tau:+.2f} h≥{h:.2f}", "自動網格", {"tau": tau, "h_min": h}, d)
    for key, d in ad["au_recs"].items():
        add(f"自動過關網格 PLA≥{key[0]:.2f} QPL≥{key[1]:.2f} edge≥{key[2]:.2f}", "自動網格",
            {"pla_min_p": key[0], "qpl_min_hit": key[1], "min_edge": key[2]}, d)
    bets_df = pd.concat(frames, ignore_index=True)
    return S, bets_df


def _ad_table_rows(d, by, min_n=1):
    L = []
    if d is None or not len(d):
        return L
    tot = len(d)
    for k, g in d.groupby(by, sort=False):
        if len(g) < min_n:
            continue
        stake = g["stake"].sum()
        L.append((k, len(g), len(g) / tot, g["hit"].mean(), g["payout"].sum() / stake - 1, g["payout"].sum() - stake))
    return sorted(L, key=lambda x: -x[1])


def _adaptive_section(src, d):
    """Markdown for one source (Traditional Chinese)."""
    ad = d["adaptive"]
    sm_all, sm_is, sm_oos = d["all"], d["is"], d["oos"]
    L = [f"### 來源 `{src}`", ""]
    L += [f"- 喺 2024-25 季估計：第二名折減 λ2 = {ad['lam2']:.2f}、第三名 λ3 = {ad['lam3']:.2f}"
          f"（1.00 = 純 Harville；{ad['n_lam']} 場）。",
          "- 派彩估算校正（2024-25 季勝出組合，log(實際派彩) = a + b·log(市場估算)）："
          + "；".join(f"{p} a={v['a']:+.2f} b={v['b']:.2f}（中位實際/估算 {v['median_ratio']:.2f}，{v['n']} 個）"
                     for p, v in ad["cal_stats"].items()), ""]
    # tuning grid
    L += ["**單場自動策略：參數網格（只用 2024-25 季揀參數；樣本外只作報告）**", "",
          f"（✔ = 揀中設定：τ ≥ 0、樣本內至少 {AD_MIN_IS_BETS} 注之中樣本內 ROI 最高；τ 負數只作參考，機械人唔會落預期蝕錢嘅注）", "",
          "| τ (最低預期 ROI) | h_min (最低命中機率) | 樣本內注數 | 佔場次 | 樣本內 ROI | 樣本外注數 | 佔場次 | 樣本外 ROI |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    t = ad["tuned"]
    for g in ad["grid"]:
        mark = " ✔" if t and g["tau"] == t["tau"] and g["h_min"] == t["h_min"] else ""
        L.append(f"| {g['tau']:+.2f}{mark} | {g['h_min']:.2f} | {g['is']['n']} | "
                 f"{g['is']['n'] / max(ad['n_races_is'], 1) * 100:.0f}% | {_p(g['is']['roi'])} | {g['oos']['n']} | "
                 f"{g['oos']['n'] / max(ad['n_races_oos'], 1) * 100:.0f}% | {_p(g['oos']['roi'])} |")
    L.append("")
    # tuned vs fixed
    names = [s for s in sm_all["strategy"] if s.startswith("ADAPTIVE")] + \
        ["PLA 6x63 [confidence]", "3xQPL 2x3 [confidence]", "WIN top1", "QPL 複式3"]
    L += ["**已調校嘅自動策略 vs 固定組合**", ""]
    for lab, sm in (("樣本內 2024-25", sm_is), ("樣本外 2025-26 起", sm_oos)):
        L += [f"_{lab}_", ""] + _table(sm[sm["strategy"].isin(names)]) + [""]
    if not t:
        L += [f"- 單場：冇任何參數組合喺 2024-25 季落注 ≥ {AD_MIN_IS_BETS} 場 —— 即係模型睇唔到值博嘅場次，機械人會全部「唔落」。", ""]
    else:
        if not t["enabled"]:
            L += [f"- ⚠️ 單場：樣本內最好嘅設定 ROI 都係 {_p(t['is']['roi'])}（蝕錢），所以呢個來源嘅自動策略應該**停用**（全部唔落）；"
                  "下面列出佢揀咗咩玩法只係用嚟理解模型行為。", ""]
        rec = ad["recs"][(t["tau"], t["h_min"])]
        oos = rec[rec["season"].isin(OOS_SEASONS)]
        allp = rec
        L += [f"**揀咗咩玩法（已調校設定，2024-25 季起全部 {len(allp)} 注；樣本外 {len(oos)} 注）**", "",
              "| 玩法類別 | 注數 | 佔比 | 命中率 | ROI | 盈虧 |", "|---|---:|---:|---:|---:|---:|"]
        for k, n, sh, hit, roi, prof in _ad_table_rows(allp, "cat"):
            L.append(f"| {k} | {n} | {sh * 100:.0f}% | {hit * 100:.1f}% | {_p(roi)} | {_m(prof)} |")
        L += ["", "| 具體結構 | 注數 | 命中率 | ROI |", "|---|---:|---:|---:|"]
        for k, n, sh, hit, roi, prof in _ad_table_rows(allp, "structure")[:15]:
            L.append(f"| {k} | {n} | {hit * 100:.1f}% | {_p(roi)} |")
        L.append("")
        # by profile bucket
        rp = pd.DataFrame([{"p1b": _bucket(r["profile"]["p1"], P1_BUCKETS),
                            "gapb": _bucket(r["profile"]["gap"], GAP_BUCKETS)}
                           for r in d["res"]["_ad_profiles"]])
        allp = allp.assign(p1b=allp["p1"].map(lambda x: _bucket(x, P1_BUCKETS)),
                           gapb=allp["gap"].map(lambda x: _bucket(x, GAP_BUCKETS)))
        L += ["**按場次形態（模型首選勝率 p1 × 首兩名差距）：落注比例、最常揀嘅玩法、ROI**", "",
              "| p1 | 差距 | 場數 | 落注% | 最常揀 | 注數 | 命中率 | ROI |", "|---|---|---:|---:|---|---:|---:|---:|"]
        for _, _, b1 in P1_BUCKETS:
            for _, _, b2 in GAP_BUCKETS:
                nr = int(((rp["p1b"] == b1) & (rp["gapb"] == b2)).sum())
                g = allp[(allp["p1b"] == b1) & (allp["gapb"] == b2)]
                if not nr:
                    continue
                if len(g):
                    mode = g["cat"].value_counts().index[0]
                    roi = g["payout"].sum() / g["stake"].sum() - 1
                    L.append(f"| {b1} | {b2} | {nr} | {len(g) / nr * 100:.0f}% | {mode} | {len(g)} | "
                             f"{g['hit'].mean() * 100:.1f}% | {_p(roi)} |")
                else:
                    L.append(f"| {b1} | {b2} | {nr} | 0% | 唔落 | 0 | - | - |")
        L.append("")
    at = ad["au_tuned"]
    L += ["**自動過關：參數網格（2024-25 揀；PLA 門檻 = 首選入三甲機率、QPL 門檻 = 三隻腳中位置Q機率、edge = 模型/市場）**", "",
          "| PLA≥ | QPL≥ | edge≥ | 樣本內 有過關賽馬日 | 樣本內 ROI | 樣本外 有過關賽馬日 | 樣本外 ROI |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ad["au_grid"]:
        mark = " ✔" if at and (g["pla_min_p"], g["qpl_min_hit"], g["min_edge"]) == \
            (at["pla_min_p"], at["qpl_min_hit"], at["min_edge"]) else ""
        L.append(f"| {g['pla_min_p']:.2f}{mark} | {g['qpl_min_hit']:.2f} | {g['min_edge']:.2f} | {g['is']['n']} | "
                 f"{_p(g['is']['roi'])} | {g['oos']['n']} | {_p(g['oos']['roi'])} |")
    L.append("")
    if at and not at["enabled"]:
        L += [f"- ⚠️ 過關：樣本內最好嘅設定 ROI 都係 {_p(at['is']['roi'])}，應該**停用**。", ""]
    if at:
        rec = ad["au_recs"][(at["pla_min_p"], at["qpl_min_hit"], at["min_edge"])]
        L += ["| 過關結構（已調校） | 次數 | 命中率 | ROI |", "|---|---:|---:|---:|"]
        for k, n, sh, hit, roi, prof in _ad_table_rows(rec, "structure")[:12]:
            L.append(f"| {k} | {n} | {hit * 100:.1f}% | {_p(roi)} |")
        L.append("")
    else:
        L += [f"- 過關：冇參數組合喺 2024-25 季有 ≥ {AU_MIN_IS_BETS} 個賽馬日合資格 —— 即係唔落過關。", ""]
    return L


def _expost_bucket_table(d):
    """Ex-post ROI of representative fixed structures by race profile bucket
    (2024-25 onwards).  Descriptive only - the most promising cell is subject to
    the multiple-comparison caveat."""
    res = d["res"]
    b = res["bets"]
    S = {s["sid"]: s["name"] for s in res["strategies"]}
    reps = ["WIN top1", "PLA top1", "QPL 複式3", "QPL 膽1x腳3", "QIN 膽1x腳3", "QIN 複式3",
            "TRI 膽1x腳4", "TRI 膽2x腳4", "TRI 複式4", "FF 膽2x腳4"]
    sid = {v: k for k, v in S.items() if v in reps}
    prof = {(p["mi"], p["race_no"]): p["profile"] for p in res["_ad_profiles"]}
    x = b[b["sid"].isin(sid.values()) & b["season"].isin(IS_SEASONS + OOS_SEASONS)].copy()
    if not len(x):
        return [], {}, x
    x["p1b"] = [_bucket(prof[(m, r)]["p1"], P1_BUCKETS) for m, r in zip(x["mi"], x["race_no"])]
    x["gapb"] = [_bucket(prof[(m, r)]["gap"], GAP_BUCKETS) for m, r in zip(x["mi"], x["race_no"])]
    x["name"] = x["sid"].map(S)
    L = ["| p1 | 差距 | 場數 | " + " | ".join(reps) + " | 最好 |", "|---|---|---:|" + "---:|" * len(reps) + "---|"]
    best = {}
    for _, _, b1 in P1_BUCKETS:
        for _, _, b2 in GAP_BUCKETS:
            g = x[(x["p1b"] == b1) & (x["gapb"] == b2)]
            if not len(g):
                continue
            cells, rois = [], {}
            n_r = g[g["name"] == "WIN top1"].shape[0]
            for nm in reps:
                h = g[g["name"] == nm]
                if len(h) >= 30:
                    roi = h["payout"].sum() / h["stake"].sum() - 1
                    rois[nm] = roi
                    cells.append(_p(roi))
                else:
                    cells.append("-")
            bn = max(rois, key=rois.get) if rois else None
            ci = _boot_ci(g[g["name"] == bn]) if bn else (np.nan, np.nan)
            best[(b1, b2)] = (bn, rois.get(bn) if bn else None, n_r, ci)
            L.append(f"| {b1} | {b2} | {n_r} | " + " | ".join(cells) + f" | {bn or '-'} |")
    return L, best, x


def _boot_ci(g, n_boot=1000, seed=0):
    """95% bootstrap CI of ROI resampling bets (races)."""
    if g is None or len(g) < 10:
        return (np.nan, np.nan)
    st_, pa = g["stake"].to_numpy(), g["payout"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    r = pa[idx].sum(1) / st_[idx].sum(1) - 1
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def _profile_cols(d, x):
    prof = {(p["mi"], p["race_no"]): p["profile"] for p in d["res"]["_ad_profiles"]}
    p1 = [prof[(m, r)]["p1"] for m, r in zip(x["mi"], x["race_no"])]
    gap = [prof[(m, r)]["gap"] for m, r in zip(x["mi"], x["race_no"])]
    return p1, gap


GROUPS = {"單選（獨贏/位置首選）": ["WIN top1", "PLA top1"],
          "膽拖（位置Q/連贏/單T/四連環）": ["QPL 膽1x腳3", "QIN 膽1x腳3", "TRI 膽1x腳4", "TRI 膽2x腳4", "FF 膽2x腳4"],
          "複式（位置Q/連贏/單T）": ["QPL 複式3", "QIN 複式3", "TRI 複式4"]}


def _profile_type(p1, gap):
    if p1 >= 0.30 and gap >= 0.15:
        return "一匹獨大 (p1≥30% 且拋離≥15%)"
    if gap < 0.05:
        return "實力接近 (首兩名差距<5%)"
    return "其他"


def adaptive_report(results) -> list[str]:
    L = ["## 按場自動決定策略", "",
         "用家意見：唔應該一早固定用唔用膽、幾多膽幾多腳，應該按每場馬嘅評分決定落唔落、點樣落。"
         "所以機械人加咗兩個「自動」產品（同固定嘅 PLA 6x63 + 3×QPL 2x3 一齊跑）：", "",
         "1. **ADAPTIVE（單場）**：每場用模型勝出機率（廣義 Harville，第二、三名用 λ 折減）計出每個候選玩法"
         "（獨贏、位置、位置Q 複式/1膽拖2–5腳、連贏 複式/1膽拖、單T 1膽拖3–6腳/2膽拖2–5腳/複式4–5、四連環 2膽拖3–5腳）"
         "嘅命中機率同預期回報 = Σ 組合機率 × 估計派彩。估計派彩：賽前有即時賠率（獨贏/位置/連贏/位置Q）就用即時賠率；"
         "冇就用市場（獨贏賠率）隱含機率扣除彩池扣數（WIN/PLA/QIN/QPL 17.5%、TRI 23%、FF 25%，"
         "來源：馬會 Local Pools 頁面），再用 2024-25 季實際派彩校正。揀預期 ROI 最高而且 ≥ τ、命中機率 ≥ h_min 嘅玩法；"
         "冇一個合格就**唔落**。每場上限 $200。",
         "2. **ADAPTIVE_ALLUP（過關）**：只用合資格場次（首選入三甲機率夠高 + 模型比市場更睇好）砌位置/位置Q 過關，"
         "關數（2–6 關）同過關方式（NxM）按預期值揀；合資格場次少過 2 場就唔過關。", "",
         "> **偷看未來警告**：回測用嘅係**最終**獨贏賠率去估派彩同計「模型/市場」差距；實際賽前 2 小時嘅賠率會變，"
         "而且最終賠率已經包含晒最後一刻嘅資金流向，所以回測數字係**偏樂觀**嘅上限。"
         "τ、h_min、過關門檻全部只用 2024-25 季調校，2025-26 季起嘅結果係真正樣本外。", ""]
    expost = {}
    for src, d in results.items():
        if d.get("adaptive") is None:
            continue
        L += _adaptive_section(src, d)
        tb, best, x = _expost_bucket_table(d)
        expost[src] = (best, x, d)
        if tb:
            L += ["**事後分析：唔同場次形態下各固定玩法嘅 ROI（2024-25 季起；只作描述，格仔多有運氣成份）**", ""] + tb + [""]
    L += ["### 白話結論：幾時用膽、幾時複式、幾時唔落", ""]
    for src, (best, x, d) in expost.items():
        if src == "market" or not len(x):
            continue
        L.append(f"**以 `{src}` 計（2024-25 季起，每場一注，$10 單位）：**")
        L.append("")
        x = x.assign(ptype=[_profile_type(prof_p1, prof_gap) for prof_p1, prof_gap in
                            zip(*_profile_cols(d, x))])
        L += ["| 場次形態 | " + " | ".join(GROUPS) + " |", "|---|" + "---:|" * len(GROUPS)]
        for pt in ("一匹獨大 (p1≥30% 且拋離≥15%)", "實力接近 (首兩名差距<5%)", "其他"):
            cells = []
            for gname, members in GROUPS.items():
                g = x[(x["ptype"] == pt) & x["name"].isin(members)]
                if len(g) < 50:
                    cells.append("-")
                    continue
                lo, hi = _boot_ci(g)
                cells.append(f"{_p(g['payout'].sum() / g['stake'].sum() - 1)} ({_p(lo)}~{_p(hi)})")
            n_r = x[(x["ptype"] == pt) & (x["name"] == "WIN top1")].shape[0]
            L.append(f"| {pt}（{n_r} 場） | " + " | ".join(cells) + " |")
        L.append("")
        for (b1, b2), (bn, roi, n, ci) in best.items():
            if bn is None:
                continue
            lo, hi = ci
            if roi is not None and roi > 0 and np.isfinite(lo) and lo > 0:
                verdict = "95% 區間高於 0，值得留意，但係事後揀出嚟，要再睇樣本外"
            elif roi is not None and roi > 0:
                verdict = f"表面賺錢但 95% 區間 {_p(lo)}~{_p(hi)} 包括 0 —— 證據不足，當運氣"
            else:
                verdict = "連最好嘅玩法都蝕 —— 唔落"
            L.append(f"- {b1}、{b2}（{n} 場）：事後最好係 {bn}（ROI {_p(roi)}）→ {verdict}。")
        ad = d["adaptive"]
        t = ad["tuned"]
        if t and t.get("enabled"):
            rec = ad["recs"][(t["tau"], t["h_min"])]
            share = len(rec[rec["season"].isin(OOS_SEASONS)]) / max(ad["n_races_oos"], 1)
            nb = rec["n_bankers"].fillna(0)
            legs = rec.loc[nb > 0, "n_legs"]
            L += ["", f"- 自動策略（已調校）只喺約 {share * 100:.0f}% 場次落注；其中用膽拖 {(nb > 0).mean() * 100:.0f}%、"
                  f"複式 {(rec['cat'].str.contains('複式')).mean() * 100:.0f}%、單選 {(rec['cat'].str.contains('單選')).mean() * 100:.0f}%。"
                  + (f"膽拖平均 {legs.mean():.1f} 隻腳（{(legs <= 3).mean() * 100:.0f}% 係 3 隻腳或以下）。" if len(legs) else "")]
            ro = _quick(rec, OOS_SEASONS)
            L.append(f"- 樣本外（2025-26 起）：{ro['n']} 注，ROI {_p(ro['roi'])}，命中率 {_p(ro['hit'], False)}。")
        L.append("")
    L += ["**總括（答用家問題）**", "",
          "- 「應唔應該每場都固定用膽 / 固定腳數？」—— 唔應該。上面每種場次形態最好嘅玩法都唔同，"
          "而且大部份格仔連最好嘅玩法都係蝕；按場揀（包括**唔落**）至少可以避開預期最差嘅場次。",
          "- 膽拖 vs 複式：睇上面「場次形態」表 —— 一匹獨大嘅場次，膽拖同複式嘅差距細過 95% 區間；"
          "實力接近嘅場次，用膽好易全軍覆沒。數據未有足夠證據話一定邊個好，所以由每場嘅預期值去揀，而唔係固定一個做法。",
          "- 腳數：腳越多命中率越高，但每多一隻腳就多一堆低機率組合、扣數照扣，預期回報通常下降；自動策略揀出嚟嘅多數係少腳。",
          "- 市場基準（`market`）本身冇優勢：用市場機率計，預期 ROI 約等於負扣數，自動策略喺 τ ≥ 0 下幾乎完全唔落 —— 呢個係正確反應。"
          "`wf-fundamental-v1`（唔用賠率嘅模型）過份自信，自動策略喺樣本內已經大蝕，應該停用。",
          "- 以上所有數字都受「最終賠率」偷看未來影響，而且網格同分組多，請以樣本外表現為準，唔好當成穩賺方法。", ""]
    return L


# ----------------------------------------------------------------- entry points
def evaluate(conn, source="market", start=None, end=None, write=True, adaptive=True) -> dict:
    res = run_grid(conn, source, start, end)
    ad = adaptive_backtest(res) if adaptive else None
    if ad:
        res["strategies"], res["bets"] = _append_adaptive(res, ad)
        res["_ad_profiles"] = [{"mi": mi, "race_no": rn, "profile": r["profile"]}
                               for mi, m in enumerate(res["meeting_data"]) for rn, r in m["races"].items()]
    S, b, mt = res["strategies"], res["bets"], res["meetings"]
    out = {"res": res, "adaptive": ad,
           "all": summarise(b, S, mt),
           "is": summarise(b, S, mt, seasons=IS_SEASONS),
           "oos": summarise(b, S, mt, seasons=OOS_SEASONS),
           "common": summarise(b, S, mt, seasons=IS_SEASONS + OOS_SEASONS),
           "sanity": sanity(res["meeting_data"]) if source == "market" else None}
    if write:
        write_results(conn, source, {"all": out["all"], "is_2024-25": out["is"],
                                     "oos_2025+": out["oos"]}, S)
    return out


def main(conn, start=None, end=None, source=None, report_path=REPORT_PATH, write=True) -> dict:
    """Run the backtest grid for one or all prediction sources and write the report."""
    sources = [source] if source else ["market"] + available_sources(conn)
    results = {}
    for s in sources:
        t0 = time.time()
        results[s] = evaluate(conn, s, start, end, write=write)
        print(f"[backtest] {s}: {len(results[s]['res']['meetings'])} meetings, "
              f"{len(results[s]['res']['bets'])} bets, {time.time() - t0:.0f}s")
    md = render_report(results)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(md, encoding="utf-8")
        try:     # keep the 每日精選 section (hkjc.picks.backtest_picks) in the regenerated report
            from . import picks as _picks
            if _picks.BACKTEST_PATH.exists():
                _picks.write_report_section(json.loads(_picks.BACKTEST_PATH.read_text(encoding="utf-8")),
                                            report_path)
        except Exception as e:  # pragma: no cover - the grid report must not fail
            print(f"[backtest] picks section not re-added: {e}")
        print(f"[backtest] report written: {report_path}")
    for d in results.values():
        d["res"].pop("meeting_data", None)
    return {"sources": sources, "report_md": md, "results": results}
