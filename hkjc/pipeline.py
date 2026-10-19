"""Pre-meeting and post-meeting workflows called by cli.py (reports in Traditional Chinese)."""
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from . import bots, model, paper, publish, scores, strategy
from .odds import latest_odds

log = logging.getLogger("hkjc.pipeline")

VENUE_ZH = {"ST": "沙田", "HV": "跑馬地"}
SURFACE_ZH = {"TURF": "草地", "AWT": "全天候(泥地)"}
DISCLAIMER = ("> ⚠️ 本報告只供模擬投注 (paper trade) 研究用途，並非投資或投注建議。"
              "馬會各彩池抽水約 17.5%–25%，長期回測顯示大部分策略為負回報。博彩要有節制。")


def _pct(x, nd=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{nd}f}%"


def _num(x, nd=1):
    return "-" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(冇資料)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def _ticket_lines(tickets, names) -> str:
    out = []
    for t in tickets:
        legs = []
        for leg in t["legs"]:
            sel = "、".join(f"{h} {names.get((leg['race_no'], int(h)), '')}".strip() for h in leg.get("selections", []))
            if leg.get("bankers"):
                bk = "、".join(f"{h} {names.get((leg['race_no'], int(h)), '')}".strip() for h in leg["bankers"])
                sel = f"膽 {bk}　拖 {sel}"
            if leg.get("positions"):
                sel = " / ".join(",".join(str(h) for h in pos) for pos in leg["positions"])
            pool = leg.get("pool") or t.get("pool") or ""
            legs.append(f"第{leg['race_no']}場 {pool}: {sel}")
        status = ""
        if t.get("payout") is not None:
            status = f" → 派彩 ${t['payout']:,.2f}（盈虧 {t['profit']:+,.2f}）"
        kind = t.get("spec") or (t.get("notes") or {}).get("structure") or t.get("pool") or ""
        # ADAPTIVE tickets are numbered by race number: say so instead of "第N條"
        label = (f"{t['strategy']}（第{t['legs'][0]['race_no']}場）" if t["strategy"] == "ADAPTIVE" and t["legs"]
                 else f"{t['strategy']} 第{t.get('ticket_no', 1)}條")
        out.append(f"- **{label}**（{kind}，"
                   f"{t['n_combos']} 注 × ${t['unit_stake']:.0f} = ${t['total_stake']:,.0f}）{status}\n  "
                   + "；".join(legs))
    return "\n".join(out) + "\n" if out else "_(冇投注)_\n"


def _backtest_hint(conn) -> str:
    rows = conn.execute(
        "SELECT strategy, params, n_bets, roi FROM backtest_results "
        "WHERE strategy IN ('PLA 6x63 [confidence]', '3xQPL 2x3 [confidence]')").fetchall()
    out = []
    for r in rows:
        p = json.loads(r["params"])
        if p.get("period") != "all":
            continue
        m = p.get("metrics", {})
        out.append(f"- {r['strategy']}（來源 {p.get('source')}）：{r['n_bets']} 個賽馬日，ROI {_pct(r['roi'])}，"
                   f"有錢賺嘅賽馬日 {_pct(m.get('profitable_rate'))}")
    if not out:
        return ""
    return "### 回測提示（固定組合過往表現）\n" + "\n".join(sorted(out)) + "\n詳見 `reports/回測報告.md`。\n"


def _adaptive_md(pred, odds, config) -> str:
    """Per-race ADAPTIVE decision table (按場自動決定)."""
    try:
        cfg = ((config or {}).get("ADAPTIVE") or {})
        plan = strategy.adaptive_plan(pred, odds=odds or None, cfg=cfg)
    except Exception as e:  # report must not fail because of the adaptive table
        log.warning("adaptive plan failed: %s", e)
        return ""
    rows = []
    for p in plan:
        pr, ch, best = p["profile"], p["chosen"], p["best"]
        if ch:
            leg = ch["leg"]
            sel = ("膽 " + ",".join(map(str, leg["bankers"])) + " 拖 " if leg.get("bankers") else "") + \
                ",".join(map(str, leg.get("selections", [])))
            dec = f"{ch['name']}（{sel}，${ch['stake']:.0f}）"
            er, hit = ch["exp_roi"], ch["hit"]
        else:
            tau = {**strategy.ADAPTIVE_DEFAULT, **cfg}["tau"]
            why = "命中機率太低" if best and best["exp_roi"] >= tau else "預期回報唔夠"
            dec = "唔落" + (f"（{why}；最好係 {best['name']} 預期 {best['exp_roi'] * 100:+.0f}%、"
                           f"命中 {best['hit'] * 100:.0f}%）" if best else "（冇賠率估唔到）")
            er, hit = None, None
        rows.append({"場次": p["race_no"], "首選": pr["top_horse"], "p1": _pct(pr["p1"]), "p2": _pct(pr["p2"]),
                     "差距": _pct(pr["gap"]), "模型/市場": _num(pr.get("edge1"), 2), "決定": dec,
                     "預期ROI": _pct(er), "命中機率": _pct(hit)})
    if not rows:
        return ""
    return ("## 按場自動決定（ADAPTIVE）\n"
            "每場按模型機率 + 賠率計每個玩法嘅預期回報同命中機率，揀預期 ROI 最高而且過門檻嘅玩法；"
            "冇合格就唔落。詳見 `reports/回測報告.md`「按場自動決定策略」。\n\n" + _table(pd.DataFrame(rows)))


def _bots_md(decisions, names, conn) -> str:
    out = []
    for b in bots.BOTS:
        ds = [d for d in decisions if d["bot_id"] == b["id"]]
        bets = [d for d in ds if d["action"] == "bet" and d.get("stake")]
        stake = sum(d["stake"] or 0 for d in bets)
        out.append(f"### {b['name_zh']}（{b['style_zh']}）— 落 {len(bets)} 注，${stake:,.0f}；"
                   f"資金 ${paper.bankroll(conn, bot_id=b['id']):,.2f}")
        for d in ds:
            if d["action"] == "bet" and d.get("stake"):
                where = f"第{d['race_no']}場" if d["race_no"] else "過關"
                out.append(f"- ✅ {where} {d.get('structure') or ''}：{d['reason']}")
        skipped = [d for d in ds if d["action"] != "bet"]
        if skipped:
            out.append("- 唔買：" + "；".join(
                (f"第{d['race_no']}場 " if d["race_no"] else "全日 ") + d["reason"] for d in skipped[:12]))
        out.append("")
    return "\n".join(out) + "\n"


def _picks_md(picks) -> str:
    """「## 今日精選」: the fixed_combo robot's tickets (位置 6x63 + 3 條位置Q 2x3) + 全日最穩 + 值博之選."""
    if not picks:
        return "## 今日精選\n_(未有精選：模型未有預測／固定組合未有決定)_\n"
    def tm(ts):
        return f" {ts[11:16]}" if ts and len(ts) >= 16 else ""
    def money(x):
        return "-" if x is None else f"${x:,.2f}".replace(".00", "")
    out = ["## 今日精選", f"_{picks.get('note') or ''}_", ""]
    p6 = picks.get("pla_6x63")
    if p6:
        out.append(f"**位置 6x63（每場 1 隻，{p6.get('n_combos') or 63} 注 × $10 = {money(p6['stake'])}）**")
        for l in p6["legs"]:
            res = ""
            if l.get("result"):
                res = f" → 第{l['result']['finish_pos'] or '-'}名 {'✅' if l['result']['placed'] else '❌'}"
            out.append(f"- 第{l['race_no']}場{tm(l.get('post_time'))}　{l['horse_no']}號 {l['horse_name']}"
                       f"（{l.get('jockey') or '-'}）評分 {_num(l.get('score'), 0)}｜上名率 {_pct(l.get('p_place'))}"
                       f"｜位置賠率 {_num(l.get('pla_odds'))}{res}")
        if p6.get("payout") is not None:
            out.append(f"- 中 {p6.get('legs_hit')}/{len(p6['legs'])} 關 → 派彩 {money(p6['payout'])}（盈虧 {p6['profit']:+,.2f}）")
    else:
        out.append("**位置 6x63**：今日冇買")
    out.append("")
    q = picks.get("qpl_2x3") or []
    if q:
        out.append(f"**位置Q 2x3 × {len(q)} 條（每場 3 隻，每條 15 注 × $10 = $150）**")
        for t in q:
            legs = "；".join(f"第{l['race_no']}場{tm(l.get('post_time'))} " +
                            "、".join(f"{h['horse_no']} {h['horse_name']}" for h in l["horses"]) +
                            ("" if l.get("hit_pairs") is None else
                             (" ✅ 中 " + "、".join("-".join(map(str, pr)) for pr in l["hit_pairs"]) if l["hit_pairs"] else " ❌"))
                            for l in t["legs"])
            res = "" if t.get("payout") is None else f" → 派彩 {money(t['payout'])}（盈虧 {t['profit']:+,.2f}）"
            out.append(f"- 第{t['ticket_no']}條：{legs}{res}")
    else:
        out.append("**位置Q 2x3**：今日冇買")
    out.append("")
    b = picks.get("banker")
    if b:
        out.append(f"**全日最穩**：第{b['race_no']}場 {b['horse_no']}號 {b['horse_name']} — {b['reason']}")
    v = picks.get("value") or []
    if v:
        out.append("**值博之選**：")
        out += [f"- 第{x['race_no']}場 {x['horse_no']}號 {x['horse_name']} — {x['reason']}" for x in v]
    else:
        out.append(f"**值博之選**：{picks.get('value_note') or '冇'}")
    out.append("")
    tot = f"**合共 {money(picks.get('total_stake'))}（模擬）**"
    if picks.get("total_payout") is not None:
        tot += f"　派彩 {money(picks['total_payout'])}，盈虧 {picks['total_profit']:+,.2f}"
    out += [tot, ""]
    return "\n".join(out) + "\n"


def _pre_picks(conn, date, venue, decisions) -> str:
    """Picks for the pre-meeting report: this run's fixed_combo tickets (also in a report-only run,
    where nothing is written) on top of the meeting's exported horses.  Never raises."""
    try:
        meeting = publish.export_meeting(conn, date, venue)
        fc = [dict(d["ticket"]) for d in decisions
              if d.get("bot_id") == publish.PICKS_BOT and d.get("ticket") and d.get("action") == "bet"]
        picks = publish.build_picks(meeting["races"], fc) if fc else meeting.get("picks")
        return _picks_md(picks)
    except Exception:
        log.exception("今日精選 failed; report continues without it")
        return ""


# ---------------------------------------------------------------------- pre-meeting
def pre_meeting(conn, run_id, date, venue, config=None, place_bets=True) -> dict:
    """place_bets=False (scheduler woke up too late, < 15 min before race 1): the
    report and predictions are still produced, but no paper tickets are placed."""
    odds = latest_odds(conn, date, venue)
    pred = model.predict_card(conn, date, venue, kind="combined", odds=odds or None)
    if pred.empty:
        raise RuntimeError(f"no racecard predictions for {date} {venue}")
    card = pd.read_sql_query(
        "SELECT race_no, horse_no, horse_name, jockey_name, trainer_name, draw, weight, rating, scratched "
        "FROM racecard WHERE date=? AND venue=?", conn, params=(date, venue))
    pred = pred.drop(columns=[c for c in ("horse_name",) if c in pred.columns]).merge(
        card, on=["race_no", "horse_no"], how="left")
    pred = pred[pred["scratched"].fillna(0) == 0].copy()

    def _o(rn, pool, h):
        v = (odds.get(int(rn), {}).get(pool, {}) or {}).get(str(int(h)))
        return v
    pred["win_odds_live"] = [_o(r, "WIN", h) for r, h in zip(pred.race_no, pred.horse_no)]
    pred["pla_odds"] = [_o(r, "PLA", h) for r, h in zip(pred.race_no, pred.horse_no)]

    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR REPLACE INTO predictions (run_id, race_id, horse_no, p_win, p_place, model_version, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(run_id, r.race_id, int(r.horse_no), float(r.p_win), float(r.p_place), r.model_version, now)
         for r in pred.itertuples()])
    conn.commit()

    # 馬匹評分 + 理由
    try:
        sc = scores.score_meeting(conn, date, venue, pred_df=pred, odds=odds or None)
        scores.save_scores(conn, run_id, sc)
        pred = pred.merge(sc[["race_no", "horse_no", "score", "rank_in_race", "reasons"]],
                          on=["race_no", "horse_no"], how="left")
    except Exception:
        log.exception("scoring failed; continuing without scores")
        pred["score"], pred["reasons"] = float("nan"), None

    # 10 隻機械人各自決定（每場都有買/唔買原因）
    if place_bets:
        decisions = bots.run_bots(conn, run_id, date, venue, pred, odds or None)
    else:
        decisions = []
        for b in bots.BOTS:
            try:
                decisions += bots.decide(b, pred, odds or None, paper.bankroll(conn, bot_id=b["id"]))
            except Exception:     # report-only run: a robot's bug must not lose the report
                log.exception("robot %s failed (report-only run)", b["id"])
    tickets = [dict(d["ticket"], bot_id=d["bot_id"]) for d in decisions if d.get("ticket")]
    ids = [d.get("ticket_id") for d in decisions if d.get("ticket_id")]
    names = {(int(r.race_no), int(r.horse_no)): r.horse_name for r in pred.itertuples()}

    snap = conn.execute("SELECT MAX(snapshot_at) FROM odds_snapshots WHERE date=? AND venue=?",
                        (date, venue)).fetchone()[0]
    races = conn.execute("SELECT race_no, post_time, distance, surface, course, race_class, rating_band, race_name "
                         "FROM races WHERE date=? AND venue=? ORDER BY race_no", (date, venue)).fetchall()
    md = [f"# {date} {VENUE_ZH.get(venue, venue)} 賽前投注參考\n",
          f"生成時間：{now}　｜　賠率快照：{snap or '未有（賠率未開售，只用模型）'}　｜　模型：{pred['model_version'].iloc[0]}\n",
          DISCLAIMER + "\n",
          _pre_picks(conn, date, venue, decisions),
          ("## 模擬投注（機械人已記錄）\n" if place_bets else
           "## 參考組合（⚠️ 排程太遲執行，已過落注時限：只供參考，機械人冇記錄模擬注）\n"),
          _bots_md(decisions, names, conn),
          _backtest_hint(conn),
          _adaptive_md(pred, odds, config),
          "## 各場預測\n",
          "欄位：勝率/上名率＝模型估計；合理獨贏＝1/勝率；位置EV＝上名率×位置賠率（>1 代表賠率高過模型估值）。\n"]
    for rc in races:
        g = pred[pred.race_no == rc["race_no"]].sort_values("p_place", ascending=False)
        if g.empty:
            continue
        t = pd.DataFrame({
            "馬號": g.horse_no.astype(int), "馬名": g.horse_name, "騎師": g.jockey_name, "練馬師": g.trainer_name,
            "檔位": g.draw.astype("Int64"), "負磅": g.weight.astype("Int64"), "勝率": g.p_win.map(_pct), "上名率": g.p_place.map(_pct),
            "評分": g["score"].map(lambda x: _num(x, 0)),
            "理由": g["reasons"].map(lambda r: "；".join(r) if isinstance(r, list) else ""),
            "合理獨贏": (1 / g.p_win).map(lambda x: _num(x)), "獨贏賠率": g.win_odds_live.map(_num),
            "位置賠率": g.pla_odds.map(_num),
            "位置EV": (g.p_place * g.pla_odds).map(lambda x: _num(x, 2)),
        })
        md.append(f"### 第{rc['race_no']}場 {rc['post_time'][11:] if rc['post_time'] else ''}　"
                  f"{rc['race_class'] or ''} {rc['distance'] or ''}米 {SURFACE_ZH.get(rc['surface'], rc['surface'] or '')}"
                  f" {('「' + rc['course'] + '」賽道') if rc['course'] else ''}　{rc['race_name'] or ''}\n")
        md.append(_table(t))
    md.append(_long_term_md(conn, venue, date))
    return {"report_md": "\n".join(md), "notes": {"tickets": ids, "n_pred": len(pred), "odds_snapshot": snap,
                                                  "placed": bool(place_bets)}}


# --------------------------------------------------------------------- post-meeting
def _hist(conn, venue, before, days=365):
    since = (datetime.strptime(before, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    return pd.read_sql_query(
        "SELECT r.date, r.venue, r.distance, r.surface, r.course, u.* FROM runners u JOIN races r USING(race_id) "
        "WHERE r.venue=? AND r.date>=? AND r.date<=? AND u.scratched=0", conn, params=(venue, since, before))


def _rates(df, keys, min_n):
    df = df.assign(win=(df.finish_pos == 1).astype(int), plc=(df.finish_pos <= 3).astype(int))
    g = df.groupby(keys).agg(出賽=("win", "size"), 頭馬=("win", "sum"), 上名=("plc", "sum")).reset_index()
    g = g[g["出賽"] >= min_n]
    g["勝出率"] = g["頭馬"] / g["出賽"]
    g["上名率"] = g["上名"] / g["出賽"]
    return g.sort_values(["上名率", "出賽"], ascending=False)


def _long_term_md(conn, venue, date) -> str:
    h = _hist(conn, venue, (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"))
    if h.empty:
        return ""
    out = [f"## {VENUE_ZH.get(venue, venue)} 近 365 日長期參考\n"]
    h = h.rename(columns={"jockey_name": "騎師", "trainer_name": "練馬師"})
    j = _rates(h, ["騎師"], 40).head(10)
    t = _rates(h, ["練馬師"], 40).head(10)
    jt = _rates(h, ["騎師", "練馬師"], 12).head(12)
    for title, d in (("騎師（≥40 騎）", j), ("練馬師（≥40 匹次）", t), ("騎練組合（≥12 次）", jt)):
        d = d.copy()
        d["勝出率"] = d["勝出率"].map(_pct); d["上名率"] = d["上名率"].map(_pct)
        out += [f"### {title}\n", _table(d)]
    h = h.assign(檔位組=pd.cut(h.draw, [0, 3, 6, 9, 14], labels=["1-3", "4-6", "7-9", "10+"]),
                 路程=h.distance.astype("Int64").astype(str) + "米 " + h.surface.map(SURFACE_ZH).fillna(""))
    d = _rates(h.dropna(subset=["檔位組"]), ["路程", "檔位組"], 20)
    p = d.pivot_table(index="路程", columns="檔位組", values="上名率", observed=False)
    p = p.apply(lambda c: c.map(_pct)).reset_index()
    out += ["### 檔位上名率（按路程/跑道）\n", _table(p)]
    return "\n".join(out)


def post_meeting(conn, run_id, date, venue, retrain=True) -> dict:
    races = pd.read_sql_query("SELECT * FROM races WHERE date=? AND venue=? ORDER BY race_no", conn,
                              params=(date, venue))
    run = pd.read_sql_query(
        "SELECT u.*, r.race_no FROM runners u JOIN races r USING(race_id) WHERE r.date=? AND r.venue=?",
        conn, params=(date, venue))
    pre = conn.execute("SELECT run_id FROM runs WHERE kind='pre_bet' AND date=? AND venue=? "
                       "ORDER BY created_at DESC LIMIT 1", (date, venue)).fetchone()
    pred = pd.DataFrame()
    if pre:
        pred = pd.read_sql_query("SELECT race_id, horse_no, p_win, p_place FROM predictions WHERE run_id=?",
                                 conn, params=(pre["run_id"],))
    md = [f"# {date} {VENUE_ZH.get(venue, venue)} 賽後分析\n", DISCLAIMER + "\n", "## 逐場賽果 vs 模型\n"]
    rows, hit_top1_place, hit_qpl, n_races = [], 0, 0, 0
    for rc in races.itertuples():
        g = run[(run.race_id == rc.race_id) & (run.scratched == 0)].sort_values("finish_pos")
        top3 = g[g.finish_pos <= 3]
        fav = g.sort_values("win_odds").head(1)
        placed = set(top3.horse_no.astype(int))
        pick, qpl3 = "-", "-"
        if not pred.empty:
            pg = pred[pred.race_id == rc.race_id].sort_values("p_place", ascending=False)
            if len(pg):
                n_races += 1
                p1 = int(pg.horse_no.iloc[0])
                ok = p1 in placed
                hit_top1_place += ok
                pick = f"{p1}{'✅' if ok else '❌'}"
                q = set(pg.horse_no.astype(int).head(3))
                nq = len(q & placed)
                hit_qpl += nq >= 2
                qpl3 = f"{sorted(q)} 中{nq}"
        rows.append({
            "場": rc.race_no, "班次/路程": f"{rc.race_class or ''} {rc.distance}米 {SURFACE_ZH.get(rc.surface, '')}",
            "頭馬": " ".join(f"{int(x.horse_no)}{x.horse_name}" for x in top3.head(1).itertuples()),
            "騎師/練馬師": " ".join(f"{x.jockey_name}/{x.trainer_name}" for x in top3.head(1).itertuples()),
            "檔位": ",".join(str(int(x)) for x in top3.draw.head(3) if pd.notna(x)),
            "頭三名": ",".join(str(int(x)) for x in top3.horse_no),
            "獨贏賠率": _num(top3.win_odds.iloc[0]) if len(top3) else "-",
            "大熱": (f"{int(fav.horse_no.iloc[0])}({_num(fav.win_odds.iloc[0])})"
                     f"{'✅' if len(fav) and int(fav.horse_no.iloc[0]) in placed else '❌'}") if len(fav) else "-",
            "模型首選上名": pick, "模型前三(位置Q)": qpl3,
        })
    md.append(_table(pd.DataFrame(rows)))
    if n_races:
        md.append(f"模型首選上名：{hit_top1_place}/{n_races}（{_pct(hit_top1_place / n_races)}）；"
                  f"前三揀中位置Q：{hit_qpl}/{n_races}\n")

    st = run[run.scratched == 0].assign(win=lambda d: (d.finish_pos == 1).astype(int),
                                         plc=lambda d: (d.finish_pos <= 3).astype(int))
    for title, keys in (("騎師", ["jockey_name"]), ("練馬師", ["trainer_name"]),
                        ("騎練組合（有上名）", ["jockey_name", "trainer_name"])):
        g = st.groupby(keys).agg(出賽=("win", "size"), 頭馬=("win", "sum"), 上名=("plc", "sum")).reset_index()
        g = g[g["上名"] > 0].sort_values(["頭馬", "上名"], ascending=False).head(12)
        g = g.rename(columns={"jockey_name": "騎師", "trainer_name": "練馬師"})
        md += [f"## 當日{title}\n", _table(g)]
    wins = st[st.win == 1].merge(races[["race_id", "distance", "surface"]], on="race_id")
    if len(wins):
        md.append("## 當日頭馬檔位 / 路程 / 跑道\n")
        md.append(_table(pd.DataFrame({
            "路程": wins.distance.astype(str) + "米", "跑道": wins.surface.map(SURFACE_ZH),
            "檔位": wins.draw.astype("Int64"), "獨贏賠率": wins.win_odds,
        }).sort_values("路程")))

    tk = conn.execute("SELECT * FROM paper_tickets WHERE date=? AND venue=? AND status!='void' ORDER BY strategy, ticket_no",
                      (date, venue)).fetchall()
    tickets = []
    for t in tk:      # pool / spec / notes (structure) live in the detail JSON
        det = json.loads(t["detail"] or "{}")
        tickets.append(dict(t, legs=json.loads(t["legs"]), pool=det.get("pool"), spec=det.get("spec"),
                            notes=det.get("notes") or {}))
    names = {(int(r.race_no), int(r.horse_no)): r.horse_name for r in run.itertuples()}
    md += ["## 模擬投注結算\n", _ticket_lines(tickets, names)]
    if tickets:
        stake = sum(t["total_stake"] for t in tickets)
        pay = sum(t["payout"] or 0 for t in tickets)
        md.append(f"當日成本 ${stake:,.0f}，派彩 ${pay:,.2f}，盈虧 {pay - stake:+,.2f}\n")
    md.append(paper.summary_md(conn))

    notes = {}
    if retrain:
        nxt = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        for kind in ("combined", "fundamental"):
            try:
                m = model.latest_model(conn, nxt, kind=kind)
                notes[f"model_{kind}"] = m.model_version
            except Exception as e:  # keep the report even if retraining fails
                log.exception("retrain %s failed", kind)
                notes[f"model_{kind}_error"] = str(e)
    return {"report_md": "\n".join(md), "notes": notes}
