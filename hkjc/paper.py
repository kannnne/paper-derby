"""Paper-trading robot (模擬投注).  SIMULATION ONLY - never places real bets.

place_tickets(conn, run_id, date, venue, tickets, ..., bot_id=None, reason=None) -> list[ticket_id]
void_open(conn, date, venue, bot_id=None, keep_run_id=None, note=None) -> list[ticket_id]
settle(conn, date) -> int                     (idempotent; every bot + legacy)
bankroll(conn, bot_id=None) -> float
bot_summary(conn) -> list[dict]               (one row per robot of hkjc.bots.BOTS)
summary_md(conn) -> str                       (Traditional Chinese)

Multi-bot: every robot (hkjc/bots.py) has its own ledger (bankroll_ledger.bot_id) and
its tickets carry paper_tickets.bot_id / reason.  bot_id NULL = the legacy single
robot (pipeline products PLA_6x63 / QPL_2x3 / ADAPTIVE); legacy and bot ledgers
never mix.

Ledger (bankroll_ledger): seeded once with the starting bankroll (default
HK$10,000); every placed ticket debits ``total_stake``; settlement credits the
payout; a ticket replaced by a newer run for the same meeting is voided and its
stake credited back.  ``balance`` is always previous balance + amount.

Settlement uses the ``dividends`` table (HK$ per $10) through bets.settle_allup /
bets.settle_single; scratched runners (runners.scratched) are refunded / carried
forward (rule 3.10).  A leg race that was abandoned (no races row or no
finishers once the meeting is final) is treated as refunded (factor 1.0).
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from . import bets

HKT = ZoneInfo("Asia/Hong_Kong")
DEFAULT_BANKROLL = 10_000.0
FINAL_STATUSES = ("results", "abandoned")
_VOID_ROWS = [("VOID", "-", None)]       # makes every combo 'pool_refund' (factor 1.0)


def _now():
    return datetime.now(HKT).replace(tzinfo=None).isoformat(timespec="seconds")


def _race_id(date, venue, race_no):
    return f"{date.replace('-', '')}-{venue}-{int(race_no):02d}"


def _last_balance(conn, bot_id=None):
    row = conn.execute("SELECT balance FROM bankroll_ledger WHERE bot_id IS ? ORDER BY id DESC LIMIT 1",
                       (bot_id,)).fetchone()
    return None if row is None else float(row[0])


def _ledger(conn, ticket_id, amount, note, ts=None, bot_id=None):
    prev = _last_balance(conn, bot_id)
    if prev is None:
        raise RuntimeError(f"ledger not seeded (bot_id={bot_id!r})")
    bal = round(prev + float(amount), 2)
    conn.execute("INSERT INTO bankroll_ledger (ts, ticket_id, amount, balance, note, bot_id) "
                 "VALUES (?,?,?,?,?,?)",
                 (ts or _now(), ticket_id, round(float(amount), 2), bal, note, bot_id))
    return bal


def seed_bankroll(conn, amount: float = DEFAULT_BANKROLL, bot_id=None) -> bool:
    """Seed the ledger (of ``bot_id``; None = legacy) once.  True if a seed row was written."""
    if conn.execute("SELECT 1 FROM bankroll_ledger WHERE bot_id IS ? LIMIT 1", (bot_id,)).fetchone():
        return False
    conn.execute("INSERT INTO bankroll_ledger (ts, ticket_id, amount, balance, note, bot_id) "
                 "VALUES (?,?,?,?,?,?)",
                 (_now(), None, float(amount), float(amount), "起始資金 seed", bot_id))
    conn.commit()
    return True


def _void_open_tx(conn, date, venue, bot_id, keep_run_id, note, ts):
    """(inside a transaction) void + refund the open tickets of one meeting/robot."""
    q = ("SELECT ticket_id, total_stake FROM paper_tickets WHERE date=? AND venue=? AND status='open' "
         "AND bot_id IS ?")
    prm = [date, venue, bot_id]
    if keep_run_id is not None:
        q += " AND COALESCE(run_id, '') != ?"
        prm.append(keep_run_id or "")
    ids = []
    for old in conn.execute(q, prm).fetchall():
        conn.execute("UPDATE paper_tickets SET status='void', payout=?, profit=0, settled_at=? "
                     "WHERE ticket_id=?", (old[1], ts, old[0]))
        _ledger(conn, old[0], old[1], note, ts, bot_id)
        ids.append(old[0])
    return ids


def void_open(conn, date, venue, bot_id=None, keep_run_id=None, note=None) -> list[int]:
    """Void every still-open ticket of meeting (date, venue) placed by ``bot_id``
    (None = legacy tickets only) and refund the stake to that robot's ledger.
    ``keep_run_id``: tickets of this run are kept.  Returns the voided ticket ids."""
    ts = _now()
    try:
        conn.execute("BEGIN IMMEDIATE")
        ids = _void_open_tx(conn, date, venue, bot_id, keep_run_id, note or "作廢退回 (重新決定)", ts)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ids


def void_settled(conn, date, venue, bot_id, note=None) -> list[int]:
    """Void SETTLED tickets of one meeting/robot (e.g. a robot's picks re-done with a new rule):
    the ledger gets back the stake and gives back the payout (one row, amount = stake - payout),
    the ticket becomes 'void' (payout = stake, profit 0, like void_open).  Returns the ticket ids."""
    ts = _now()
    ids = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        for t in conn.execute("SELECT ticket_id, total_stake, payout FROM paper_tickets WHERE date=? AND venue=? "
                              "AND status='settled' AND bot_id IS ?", (date, venue, bot_id)).fetchall():
            stake, pay = float(t[1]), float(t[2] or 0.0)
            conn.execute("UPDATE paper_tickets SET status='void', payout=?, profit=0, settled_at=? WHERE ticket_id=?",
                         (stake, ts, t[0]))
            _ledger(conn, t[0], round(stake - pay, 2), note or "作廢退回（已結算注項重做）", ts, bot_id)
            ids.append(t[0])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ids


def _spec_of(strategy: str, ticket: dict):
    if ticket.get("spec") is not None or "spec" in ticket:
        return ticket.get("spec")
    tail = strategy.rsplit("_", 1)[-1]
    try:
        bets.allup_formula(tail)
        return tail
    except ValueError:
        return None


def place_tickets(conn, run_id, date, venue, tickets, starting_bankroll: float = DEFAULT_BANKROLL,
                  replace: bool = True, bot_id=None, reason=None) -> list[int]:
    """Record tickets as open paper bets and debit the ledger (of ``bot_id``).

    replace=True: every still-open ticket of the same meeting *and the same robot*
    (bot_id; None = legacy tickets) placed by an earlier (different) run is voided
    (stake refunded) so each meeting has one live set - whatever strategies the new
    run produces (e.g. a re-run whose ADAPTIVE product now skips every race must not
    leave the old ADAPTIVE tickets live).  Tickets of the same ``run_id`` are kept, so
    one run may place its products in several calls.
    ``reason``: 中文落注原因 stored in paper_tickets.reason (a ticket's own
    ``t['reason']`` wins)."""
    seed_bankroll(conn, starting_bankroll, bot_id)
    ids: list[int] = []
    ts = _now()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if replace and tickets:
            _void_open_tx(conn, date, venue, bot_id, run_id, f"作廢退回 (被 {run_id} 取代)", ts)
        for t in tickets:
            pool = t.get("pool") or (t["legs"][0].get("pool") if t["legs"] else None)
            spec = _spec_of(t["strategy"], t)
            legs = [{**leg, "pool": leg.get("pool", pool)} for leg in t["legs"]]
            if pool in bets.MULTI_RACE_POOLS:
                n = bets.multi_race_units(pool, legs)
            elif spec:
                n = bets.ticket_units(pool, legs, spec)
            else:
                leg = legs[0]
                n = bets.count_units(leg["pool"], leg.get("selections", ()), leg.get("bankers", ()),
                                     leg.get("positions"), leg.get("mode", "multiple"))
            unit = float(t.get("unit_stake", 10.0))
            stake = round(unit * n, 2)
            if t.get("n_combos") is not None and int(t["n_combos"]) != n:
                raise ValueError(f"ticket n_combos {t['n_combos']} != computed {n}")
            detail = {"pool": pool, "spec": spec, "notes": t.get("notes", {})}
            cur = conn.execute(
                "INSERT INTO paper_tickets (run_id, date, venue, strategy, ticket_no, legs, unit_stake, "
                "n_combos, total_stake, status, created_at, detail, bot_id, reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, date, venue, t["strategy"], t.get("ticket_no"),
                 json.dumps(legs, ensure_ascii=False), unit, n, stake, "open", ts,
                 json.dumps(detail, ensure_ascii=False, default=str), bot_id,
                 t.get("reason") or reason))
            tid = cur.lastrowid
            _ledger(conn, tid, -stake, f"落注 {t['strategy']} #{t.get('ticket_no')}", ts, bot_id)
            ids.append(tid)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ids


def _race_data(conn, race_id):
    """(status, dividend rows, scratched set): status 'ok' / 'void' / 'missing'.

    Settlement keys off the dividends table only (finish_pos can disagree after a
    disqualification).  A WIN-pool refund row (void race; such races have no
    races/runners rows) -> 'void' (every bet refunded / carried forward)."""
    rows = conn.execute("SELECT pool, combination, dividend FROM dividends WHERE race_id=?",
                        (race_id,)).fetchall()
    scr = {int(r[0]) for r in conn.execute(
        "SELECT horse_no FROM runners WHERE race_id=? AND scratched=1", (race_id,))}
    good = [(p, c, d) for p, c, d in rows if p in bets.POOL_SIZE]
    idx = bets.index_dividends(good)
    win = idx.get("WIN") or {}
    if bets.REFUND_KEY in win:
        return "void", _VOID_ROWS, scr
    if any(v for k, v in win.items()):
        return "ok", good, scr
    return "missing", good, scr


def _favourite(conn, race_id):
    """Lowest final win odds among the starters of a race (the multi-race pools' substitute)."""
    r = conn.execute("SELECT horse_no FROM runners WHERE race_id=? AND COALESCE(scratched,0)=0 "
                     "AND win_odds IS NOT NULL AND win_odds > 0 ORDER BY win_odds, horse_no LIMIT 1",
                     (race_id,)).fetchone()
    return None if r is None else int(r[0])


def _settle_multi_race(conn, t, pool, legs, scr, void_legs, unit):
    """DBL / TBL ticket: dividend rows live on the LAST leg's race_id (SPEC.md).  Withdrawn
    selections take the leg favourite as substitute (bets.settle_multi_race); a void leg or a
    pool without dividend rows refunds every unit."""
    last = int(legs[-1]["race_no"])
    rows = [] if void_legs else conn.execute(
        "SELECT pool, combination, dividend FROM dividends WHERE race_id=? AND pool=?",
        (_race_id(t["date"], t["venue"], last), pool)).fetchall()
    subs = {int(l["race_no"]): _favourite(conn, _race_id(t["date"], t["venue"], l["race_no"])) for l in legs}
    return bets.settle_multi_race(pool, legs, [tuple(r) for r in rows], unit_stake=unit,
                                  scratched_by_leg=scr, substitutes=subs)


def settle(conn, date) -> int:
    """Settle all open tickets dated ``date`` (legacy and every robot; each payout is
    credited to the ticket's own ledger).  Returns the number settled.

    A ticket is settled once every leg race has dividends, or the meeting is
    final ('results'/'abandoned') - legs without results are then void/refunded.
    Safe to call repeatedly."""
    open_t = conn.execute("SELECT * FROM paper_tickets WHERE date=? AND status='open' ORDER BY ticket_id",
                          (date,)).fetchall()
    n_done = 0
    for t in open_t:
        mst = conn.execute("SELECT status FROM meetings WHERE date=? AND venue=?",
                           (t["date"], t["venue"])).fetchone()
        final = bool(mst) and mst[0] in FINAL_STATUSES
        legs = json.loads(t["legs"])
        detail0 = json.loads(t["detail"]) if t["detail"] else {}
        pool = detail0.get("pool") or legs[0].get("pool")
        spec = detail0.get("spec")
        divs, scr, void_legs, pending = {}, {}, [], False
        for leg in legs:
            rn = int(leg["race_no"])
            st, rows, s = _race_data(conn, _race_id(t["date"], t["venue"], rn))
            if st == "ok":
                divs[rn], scr[rn] = rows, s
            elif st == "void":
                divs[rn], scr[rn] = rows, s
                void_legs.append(rn)
            elif final:
                divs[rn], scr[rn] = _VOID_ROWS, s
                void_legs.append(rn)
            else:
                pending = True
        if pending:
            continue
        unit = float(t["unit_stake"])
        if pool in bets.MULTI_RACE_POOLS:
            res = _settle_multi_race(conn, t, pool, legs, scr, void_legs, unit)
        elif spec:
            res = bets.settle_allup(pool, legs, spec, divs, unit_stake=unit, scratched_by_race=scr)
        else:
            leg = legs[0]
            rn = int(leg["race_no"])
            combos = bets.single_combos(leg.get("pool", pool), leg.get("selections", ()),
                                        leg.get("bankers", ()), leg.get("positions"),
                                        leg.get("mode", "multiple"))
            res = bets.settle_single(leg.get("pool", pool), combos, divs[rn], unit_stake=unit,
                                     scratched=scr.get(rn, ()))
        payout = round(res["payout"], 2)
        stake = float(t["total_stake"])
        if abs(res["stake"] - stake) > 0.005:
            raise ValueError(f"ticket {t['ticket_id']}: stake mismatch {res['stake']} vs {stake}")
        detail = {**detail0, "void_legs": void_legs,
                  "n_winning_units": res["n_winning_units"],
                  "n_refunded_units": res["n_refunded_units"],
                  "combinations": res["detail"]}
        ts = _now()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE paper_tickets SET status='settled', payout=?, profit=?, settled_at=?, detail=? "
                "WHERE ticket_id=? AND status='open'",
                (payout, round(payout - stake, 2), ts, json.dumps(detail, ensure_ascii=False, default=str),
                 t["ticket_id"]))
            if cur.rowcount == 1:
                _ledger(conn, t["ticket_id"], payout, f"派彩 {t['strategy']} #{t['ticket_no']}", ts,
                        t["bot_id"])
                n_done += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return n_done


def bankroll(conn, bot_id=None) -> float:
    """Current paper bankroll of ``bot_id`` (None = legacy robot): last ledger balance,
    or the default starting bankroll if that ledger is unseeded."""
    b = _last_balance(conn, bot_id)
    return DEFAULT_BANKROLL if b is None else b


# ----------------------------------------------------------------- summary
def _fmt(x, nd=0):
    if x is None:
        return "-"
    return f"{x:,.{nd}f}"


def _money(x, nd=2):
    """Signed money: 1234.5 -> '$1,234.50', -704.08 -> '-$704.08'."""
    if x is None:
        return "-"
    return ("-" if x < 0 else "") + f"${abs(x):,.{nd}f}"


def _pct(x):
    return "-" if x is None else f"{x * 100:+.1f}%"


def _is_hit(r) -> bool:
    """A ticket 'hits' when at least one unit won a dividend (refunds don't count)."""
    try:
        return int(json.loads(r["detail"] or "{}").get("n_winning_units", 0)) > 0
    except (ValueError, TypeError):
        return (r["payout"] or 0) > 0


def _stats(rows):
    stake = sum(r["total_stake"] for r in rows)
    ret = sum(r["payout"] or 0 for r in rows)
    hits = sum(1 for r in rows if _is_hit(r))
    best = max((r["payout"] or 0 for r in rows), default=0)
    cum, peak = 0.0, 0.0
    for r in rows:
        cum += (r["payout"] or 0) - r["total_stake"]
        peak = max(peak, cum)
    return {"n": len(rows), "stake": stake, "ret": ret, "profit": ret - stake,
            "roi": (ret / stake - 1) if stake else None,
            "hit": hits / len(rows) if rows else None, "best": best, "dd": peak - cum}


def _meeting_dd(rows):
    """(max drawdown, current drawdown) of the cumulative profit, meeting by meeting."""
    by = {}
    for r in rows:
        by[(r["date"], r["venue"])] = by.get((r["date"], r["venue"]), 0.0) + (r["payout"] or 0) - r["total_stake"]
    cum = peak = mdd = 0.0
    for k in sorted(by):
        cum += by[k]
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return round(mdd, 2), round(peak - cum, 2)


def bot_summary(conn) -> list[dict]:
    """Per robot (hkjc.bots.BOTS order, then any other bot_id found in the DB):
    {bot_id, name, bankroll, start_bankroll, n_meetings, n_tickets, n_open, open_stake,
     stake, returns, profit, roi, hit_rate, best_payout, drawdown (max, HK$),
     current_drawdown}.  Only settled tickets count towards stake/returns; a meeting
    counts once it has a settled ticket or a decision row on a finished meeting."""
    from . import bots as _bots          # local import: bots imports paper
    known = [(b["id"], b["name_zh"], float(b.get("start_bankroll", DEFAULT_BANKROLL))) for b in _bots.BOTS]
    ids = {k for k, _, _ in known}
    for (bid,) in conn.execute("SELECT DISTINCT bot_id FROM paper_tickets WHERE bot_id IS NOT NULL "
                               "UNION SELECT DISTINCT bot_id FROM bankroll_ledger WHERE bot_id IS NOT NULL"):
        if bid not in ids:
            known.append((bid, bid, DEFAULT_BANKROLL))
            ids.add(bid)
    out = []
    for bid, name, start0 in known:
        rows = conn.execute("SELECT * FROM paper_tickets WHERE bot_id=? AND status='settled' "
                            "ORDER BY date, ticket_id", (bid,)).fetchall()
        seed = conn.execute("SELECT amount FROM bankroll_ledger WHERE bot_id=? ORDER BY id LIMIT 1",
                            (bid,)).fetchone()
        start = float(seed[0]) if seed else start0
        bal = _last_balance(conn, bid)
        op = conn.execute("SELECT COUNT(*), COALESCE(SUM(total_stake),0) FROM paper_tickets "
                          "WHERE bot_id=? AND status='open'", (bid,)).fetchone()
        mtgs = {(r["date"], r["venue"]) for r in rows}
        mtgs |= {(r[0], r[1]) for r in conn.execute(
            "SELECT DISTINCT d.date, d.venue FROM bot_decisions d JOIN meetings m "
            "ON m.date=d.date AND m.venue=d.venue WHERE d.bot_id=? AND m.status IN (?,?)",
            (bid, *FINAL_STATUSES))}
        s = _stats(rows)
        mdd, cdd = _meeting_dd(rows)
        out.append({"bot_id": bid, "name": name, "bankroll": round(start if bal is None else bal, 2),
                    "start_bankroll": start, "n_meetings": len(mtgs), "n_tickets": s["n"],
                    "n_open": int(op[0]), "open_stake": round(float(op[1]), 2),
                    "stake": round(s["stake"], 2), "returns": round(s["ret"], 2),
                    "profit": round(s["profit"], 2),
                    "roi": None if s["roi"] is None else round(s["roi"], 4),
                    "hit_rate": None if s["hit"] is None else round(s["hit"], 4),
                    "best_payout": round(s["best"], 2), "drawdown": mdd, "current_drawdown": cdd})
    return out


def _bots_md(conn) -> list[str]:
    if not conn.execute("SELECT 1 FROM bankroll_ledger WHERE bot_id IS NOT NULL LIMIT 1").fetchone() and \
            not conn.execute("SELECT 1 FROM paper_tickets WHERE bot_id IS NOT NULL LIMIT 1").fetchone():
        return []
    L = ["## 10 隻機械人", "",
         "| 機械人 | 現時資金 | 賽馬日 | 飛數 | 投注額 | 派彩 | 盈虧 | 回報率 | 命中率 | 最高派彩 | 最大回撤 | 未結算 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in bot_summary(conn):
        hit = "-" if b["hit_rate"] is None else f"{b['hit_rate'] * 100:.1f}%"
        L.append(f"| {b['name']} (`{b['bot_id']}`) | ${_fmt(b['bankroll'], 2)} | {b['n_meetings']} | "
                 f"{b['n_tickets']} | {_fmt(b['stake'])} | {_fmt(b['returns'])} | {_money(b['profit'], 0)} | "
                 f"{_pct(b['roi'])} | {hit} | {_fmt(b['best_payout'])} | {_fmt(b['drawdown'])} | "
                 f"{b['n_open']} 飛 / ${_fmt(b['open_stake'])} |")
    L.append("")
    return L


def summary_md(conn) -> str:
    """Traditional-Chinese markdown summary of the paper-trading account."""
    seed = conn.execute("SELECT amount FROM bankroll_ledger WHERE bot_id IS NULL ORDER BY id LIMIT 1").fetchone()
    start = float(seed[0]) if seed else DEFAULT_BANKROLL
    bal = bankroll(conn)
    settled = conn.execute("SELECT * FROM paper_tickets WHERE status='settled' AND bot_id IS NULL "
                           "ORDER BY date, ticket_id").fetchall()
    n_open = conn.execute("SELECT COUNT(*), COALESCE(SUM(total_stake),0) FROM paper_tickets "
                          "WHERE status='open' AND bot_id IS NULL").fetchone()
    L = ["# 模擬投注總結 (Paper Trade)", "",
         "> 只屬模擬投注，並無真實落注；內容不構成任何投資或投注建議。", ""]
    bot_lines = _bots_md(conn)
    L += bot_lines
    has_legacy = bool(seed) or conn.execute("SELECT 1 FROM paper_tickets WHERE bot_id IS NULL LIMIT 1").fetchone()
    if bot_lines:
        if not has_legacy:
            return "\n".join(L)
        L += ["## 舊有單一機械人（固定組合 + ADAPTIVE）", ""]
    s = _stats(settled)
    hit_txt = "-" if s["hit"] is None else f"{s['hit'] * 100:.1f}%"
    L += ["## 整體", "",
          "| 項目 | 數值 |", "|---|---|",
          f"| 起始資金 | ${_fmt(start, 2)} |",
          f"| 現時資金 | ${_fmt(bal, 2)} |",
          f"| 已結算飛數 | {s['n']} |",
          f"| 總投注額 | ${_fmt(s['stake'], 2)} |",
          f"| 總派彩 | ${_fmt(s['ret'], 2)} |",
          f"| 淨盈虧 | {_money(s['profit'])} |",
          f"| 回報率 (ROI) | {_pct(s['roi'])} |",
          f"| 命中率 (有中獎組合飛數比例) | {hit_txt} |",
          f"| 最高單飛派彩 | ${_fmt(s['best'], 2)} |",
          f"| 現時回撤 (由高位計) | ${_fmt(s['dd'], 2)} |",
          f"| 未結算 | {n_open[0]} 飛 / ${_fmt(n_open[1], 2)} |", ""]
    by = {}
    for r in settled:
        by.setdefault(r["strategy"], []).append(r)
    L += ["## 按策略", "",
          "| 策略 | 飛數 | 投注額 | 派彩 | 盈虧 | 回報率 | 命中率 | 最高派彩 | 現時回撤 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in sorted(by):
        x = _stats(by[k])
        L.append(f"| {k} | {x['n']} | {_fmt(x['stake'])} | {_fmt(x['ret'])} | {_fmt(x['profit'])} | "
                 f"{_pct(x['roi'])} | {x['hit'] * 100:.1f}% | {_fmt(x['best'])} | {_fmt(x['dd'])} |")
    if not by:
        L.append("| (未有已結算注項) | | | | | | | | |")
    L += ["", "## 最近 10 個賽馬日", "",
          "| 日期 | 場地 | 飛數 | 投注額 | 派彩 | 盈虧 | 命中飛數 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    mtg = {}
    for r in settled:
        mtg.setdefault((r["date"], r["venue"]), []).append(r)
    for key in sorted(mtg)[-10:][::-1]:
        x = _stats(mtg[key])
        L.append(f"| {key[0]} | {'沙田' if key[1] == 'ST' else '跑馬地' if key[1] == 'HV' else key[1]} | "
                 f"{x['n']} | {_fmt(x['stake'])} | {_fmt(x['ret'])} | {_fmt(x['profit'])} | "
                 f"{sum(1 for r in mtg[key] if _is_hit(r))} |")
    if not mtg:
        L.append("| (未有) | | | | | | |")
    L.append("")
    return "\n".join(L)
