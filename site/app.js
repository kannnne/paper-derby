/* 跑道機械人 · 模擬賽馬研究 — static dashboard (no build step, no external libraries).
   Data: data/index.json, data/meetings/<date>_<venue>.json (lazy), data/bot_history.json (lazy). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ constants
  // Original saddle-cloth palette (background, text) — not any racing club's colours.
  var CLOTH = [
    ["#f4f1e8", "#111111"], ["#d7263d", "#ffffff"], ["#1b4fd8", "#ffffff"], ["#f5c400", "#111111"],
    ["#13804a", "#ffffff"], ["#141414", "#ffd400"], ["#ef7d1a", "#111111"], ["#e86aa6", "#111111"],
    ["#18a3a3", "#0b1a1a"], ["#7b3fd1", "#ffffff"], ["#8a5a2b", "#ffffff"], ["#9aa4b1", "#111111"],
    ["#a3d12e", "#111111"], ["#5c0f1d", "#ffffff"]
  ];
  // Robot colours (saddle-cloth number 1..10 always shown with the colour).
  var BOT_COLORS = [
    ["#3987e5", "#ffffff"], ["#d95926", "#ffffff"], ["#199e70", "#ffffff"], ["#c98500", "#111111"],
    ["#d55181", "#ffffff"], ["#2fa32f", "#ffffff"], ["#9085e9", "#111111"], ["#e66767", "#111111"],
    ["#9aa4b1", "#111111"], ["#8a5a2b", "#ffffff"]
  ];
  var VENUE_ZH = { ST: "沙田", HV: "跑馬地" };
  var SURFACE_ZH = { TURF: "草地", AWT: "全天候跑道" };
  var POOL_ZH = { WIN: "獨贏", PLA: "位置", QIN: "連贏", QPL: "位置Q", FCT: "二重彩", TRI: "單T", TCE: "三重彩", FF: "四連環", QTT: "四重彩" };
  var POOL_ORDER = ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "TCE", "FF", "QTT"];
  var SUB_KEYS = ["form", "jockey", "trainer", "combo", "draw", "fit", "class", "market"];
  var SUB_SHORT = { form: "往績", jockey: "騎師", trainer: "練馬", combo: "騎練", draw: "檔位", fit: "適應", "class": "班次", market: "市場" };
  var WEEK = ["日", "一", "二", "三", "四", "五", "六"];
  var STATUS_ZH = { pre: "賽前", results: "已有賽果", abandoned: "取消" };

  var app = document.getElementById("app");
  var state = { index: null, meetings: {}, history: null, timers: [], prefs: { mode: "live", sort: "no" }, cleanup: [] };
  try {
    var saved = JSON.parse(localStorage.getItem("pd-prefs") || "{}");
    if (saved && typeof saved === "object") {
      if (saved.mode === "live" || saved.mode === "backtest") state.prefs.mode = saved.mode;
      if (saved.sort === "no" || saved.sort === "score") state.prefs.sort = saved.sort;
    }
  } catch (e) { /* storage unavailable: defaults */ }
  function savePrefs() { try { localStorage.setItem("pd-prefs", JSON.stringify(state.prefs)); } catch (e) { /* ignore */ } }

  // ------------------------------------------------------------------ formatting (never render null/NaN)
  function ok(v) { return v !== null && v !== undefined && !(typeof v === "number" && !isFinite(v)); }
  function esc(s) {
    return String(ok(s) ? s : "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  var DASH = "—";
  function money(v, nd) {
    if (!ok(v)) return DASH;
    nd = nd === undefined ? (Math.abs(v) >= 1000 || Number.isInteger(v) ? 0 : 2) : nd;
    var s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: nd, maximumFractionDigits: nd });
    return (v < 0 ? "−$" : "$") + s;
  }
  function signedMoney(v) {
    if (!ok(v)) return DASH;
    var s = money(Math.abs(v));
    return (v > 0.004 ? "+" : v < -0.004 ? "−" : "") + s;
  }
  function equity(b) {
    if (!b || !ok(b.bankroll)) return null;
    return b.bankroll + ((b.live && ok(b.live.open_stake)) ? b.live.open_stake : 0);
  }
  function pct(v, nd, signed) {
    if (!ok(v)) return DASH;
    nd = nd === undefined ? 1 : nd;
    var s = (Math.abs(v) * 100).toFixed(nd) + "%";
    if (Math.abs(v) * 100 < 0.5 * Math.pow(10, -nd)) v = 0;
    if (!signed) return (v < 0 ? "−" : "") + s;
    return (v > 0 ? "+" : v < 0 ? "−" : "±") + s;
  }
  function num(v, nd) { return ok(v) ? Number(v).toFixed(nd === undefined ? 0 : nd) : DASH; }
  function oddsTxt(v) {
    if (!ok(v)) return null;
    return v >= 100 ? String(Math.round(v)) : (Math.round(v * 10) / 10).toFixed(v % 1 === 0 && v >= 10 ? 0 : 1);
  }
  function plSpan(v) {
    if (!ok(v)) return '<span class="muted">' + DASH + "</span>";
    var cls = v > 0.004 ? "up" : v < -0.004 ? "down" : "flat";
    var lbl = v > 0.004 ? "賺" : v < -0.004 ? "蝕" : "打和";
    return '<span class="pl ' + cls + '" title="' + lbl + '"><span class="sr-only">' + lbl + "</span>" + signedMoney(v) + "</span>";
  }
  function parseHK(ts) {
    if (!ts) return null;
    var s = String(ts);
    if (s.length === 10) s += "T00:00";
    if (s.length === 16) s += ":00";
    var d = new Date(s + "+08:00");
    return isNaN(d.getTime()) ? null : d;
  }
  function dateZh(date) {
    var d = parseHK(date);
    if (!d) return esc(date);
    var p = date.split("-");
    var wd = new Date(Date.UTC(+p[0], +p[1] - 1, +p[2])).getUTCDay();
    return p[0] + "年" + (+p[1]) + "月" + (+p[2]) + "日（星期" + WEEK[wd] + "）";
  }
  function shortDate(date) { if (!date) return DASH; var p = String(date).split("-"); return (+p[1]) + "/" + (+p[2]); }
  function hhmm(ts) { return ts && String(ts).length >= 16 ? String(ts).slice(11, 16) : DASH; }
  function meetingId(m) { return m.date + "_" + m.venue; }
  function meetingName(m) { return dateZh(m.date) + " " + esc(m.venue_zh || VENUE_ZH[m.venue] || m.venue); }

  function cloth(no, size) {
    var c = CLOTH[((+no || 1) - 1) % CLOTH.length];
    return '<span class="cloth ' + (size || "") + '" style="--c:' + c[0] + ";--t:" + c[1] + '" aria-label="' + esc(no) + '號">' + esc(no) + "</span>";
  }
  function botInfo(id) {
    var bots = (state.index && state.index.bots) || [];
    for (var i = 0; i < bots.length; i++) if (bots[i].id === id) return bots[i];
    return { id: id, no: 0, name: id, style: "", description: "" };
  }
  function botColor(b) { return BOT_COLORS[((b.no || 1) - 1) % BOT_COLORS.length]; }
  function botToken(b, size) {
    var c = botColor(b);
    return '<span class="bot-token ' + (size || "") + '" style="--c:' + c[0] + ";--t:" + c[1] + '" aria-hidden="true">' + esc(b.no || "?") + "</span>";
  }

  // ------------------------------------------------------------------ data
  function getJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " → HTTP " + r.status);
      return r.json();
    });
  }
  function loadIndex() {
    if (state.index) return Promise.resolve(state.index);
    return getJSON("data/index.json").then(function (d) { state.index = d; return d; });
  }
  function loadMeeting(id) {
    if (state.meetings[id]) return Promise.resolve(state.meetings[id]);
    var m = (state.index.meetings || []).filter(function (x) { return meetingId(x) === id; })[0];
    var file = m ? m.file : "data/meetings/" + id + ".json";
    return getJSON(file).then(function (d) { state.meetings[id] = d; return d; });
  }
  function loadHistory() {
    if (state.history) return Promise.resolve(state.history);
    return getJSON("data/bot_history.json").then(function (d) { state.history = d; return d; })
      .catch(function () { state.history = { bots: [] }; return state.history; });
  }

  // ------------------------------------------------------------------ router
  function clearTimers() {
    state.timers.forEach(function (t) { clearInterval(t); });
    state.timers = [];
    state.cleanup.forEach(function (f) { try { f(); } catch (e) { /* ignore */ } });
    state.cleanup = [];
  }
  function setNav(key) {
    document.querySelectorAll(".nav a").forEach(function (a) {
      if (a.getAttribute("data-nav") === key) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  function route() {
    clearTimers();
    var h = (location.hash || "#/").replace(/^#\/?/, "");
    var parts = h.split("/").filter(Boolean).map(decodeURIComponent);
    loadIndex().then(function (idx) {
      renderChrome(idx);
      var p0 = parts[0] || "";
      if (p0 === "meeting") { setNav("meeting"); return pageMeeting(parts[1], parts[2]); }
      if (p0 === "bots") { setNav("bots"); return pageBots(); }
      if (p0 === "bot") { setNav("bots"); return pageBot(parts[1]); }
      if (p0 === "about") { setNav("about"); return pageAbout(); }
      setNav("home");
      return pageHome();
    }).catch(function (e) {
      app.innerHTML = '<div class="panel pad empty">暫時載入唔到數據，請稍後再試。<br><span class="small">' + esc(e.message) + "</span></div>";
      if (window.console) console.warn(e);
    });
  }
  function renderChrome(idx) {
    var f = document.getElementById("foot-updated");
    if (f) f.textContent = "數據更新：" + (idx.updated_at ? String(idx.updated_at).replace("T", " ").slice(0, 16) + "（香港時間）" : DASH);
    var nav = document.querySelector('.nav a[data-nav="meeting"]');
    var def = defaultMeeting(idx);
    if (nav && def) nav.setAttribute("href", "#/meeting/" + meetingId(def));
  }
  function defaultMeeting(idx) {
    var ms = idx.meetings || [];
    if (idx.next_meeting) {
      var n = ms.filter(function (m) { return m.date === idx.next_meeting.date && m.venue === idx.next_meeting.venue; })[0];
      if (n) return n;
    }
    return ms[0] || null;
  }
  function focusMain() { try { app.focus({ preventScroll: true }); } catch (e) { /* ignore */ } }
  function demoBanner(idx) {
    return idx.demo ? '<div class="banner" role="note"><b>示範數據：</b>以下機械人決定同注項係用真實排位表同模型預測砌出嚟嘅示範，用嚟展示網站版面；正式紀錄會喺系統每次賽前／賽後自動更新。</div>' : "";
  }

  // ------------------------------------------------------------------ home
  function pageHome() {
    var idx = state.index;
    var nm = idx.next_meeting;
    var nmFile = nm && (idx.meetings || []).filter(function (m) { return m.date === nm.date && m.venue === nm.venue; })[0];
    var html = demoBanner(idx);
    html += '<section class="hero" aria-label="下一個賽馬日">';
    if (nm) {
      html += '<div class="lower-third"><div class="lt-tag"><span>下一個賽馬日</span></div><div class="lt-body"><span>' +
        esc(nm.venue_zh || VENUE_ZH[nm.venue] || nm.venue) + (nm.venue === "HV" ? " · 夜賽" : " · 日賽") + "</span></div></div>";
      html += '<h1 style="margin-top:12px">' + dateZh(nm.date) + "</h1>";
      html += '<div class="meta">第一場開跑 <b>' + hhmm(nm.first_post_time) + "</b>（香港時間）" +
        (ok(nm.n_races) ? " · 共 " + nm.n_races + " 場" : "") + "</div>";
      html += '<div class="countdown" id="countdown" aria-live="off"></div>';
      if (nmFile) html += '<a class="cta" href="#/meeting/' + meetingId(nmFile) + '">睇排位、評分同機械人決定 →</a>';
    } else {
      html += '<h1>暫時未有已公佈嘅賽馬日</h1><div class="meta">排位表公佈後會自動更新。</div>';
    }
    html += "</section>";

    html += '<section class="section" aria-labelledby="lb-h"><div class="section-head"><span class="kicker">LIVE</span><h2 id="lb-h">機械人賽道排行榜</h2><span class="spacer"></span>' +
      '<div class="seg" role="group" aria-label="排行榜數據">' +
      '<button type="button" data-mode="live" aria-pressed="' + (state.prefs.mode === "live") + '">實盤（模擬）</button>' +
      '<button type="button" data-mode="backtest" aria-pressed="' + (state.prefs.mode === "backtest") + '">回測</button></div></div>' +
      '<div id="track"></div><div id="lb-table" class="section"></div></section>';

    html += '<section class="section" aria-labelledby="curve-h"><div class="section-head"><h2 id="curve-h">資金曲線</h2><span class="muted small" id="curve-sub"></span></div>' +
      '<div class="panel pad"><div id="curve"></div></div></section>';

    html += '<section class="section" aria-labelledby="mtg-h"><div class="section-head"><h2 id="mtg-h">賽馬日紀錄</h2></div><div class="grid" id="mtg-list" style="grid-template-columns:repeat(auto-fill,minmax(240px,1fr))"></div></section>';
    app.innerHTML = html;

    if (nm) startCountdown(document.getElementById("countdown"), parseHK(nm.first_post_time || nm.date));
    app.querySelectorAll(".seg [data-mode]").forEach(function (b) {
      b.addEventListener("click", function () {
        state.prefs.mode = b.getAttribute("data-mode"); savePrefs();
        app.querySelectorAll(".seg [data-mode]").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        drawBoard(); drawCurves();
      });
    });
    var ml = document.getElementById("mtg-list");
    var MAX_CARDS = 12;   // a season has ~90 meetings: older ones via the meeting page's selector
    var allM = idx.meetings || [];
    ml.innerHTML = allM.slice(0, MAX_CARDS).map(function (m) {
      return '<a class="panel pad" style="text-decoration:none;display:block" href="#/meeting/' + meetingId(m) + '">' +
        '<div style="display:flex;justify-content:space-between;gap:8px;align-items:center"><b>' + shortDate(m.date) + " " + esc(m.venue_zh || VENUE_ZH[m.venue]) + "</b>" +
        '<span class="chip status-' + esc(m.status) + '">' + esc(STATUS_ZH[m.status] || m.status) + "</span></div>" +
        '<div class="muted small" style="margin-top:4px">' + dateZh(m.date) + (ok(m.n_races) ? " · " + m.n_races + " 場" : "") + "</div></a>";
    }).join("") || '<div class="empty">未有紀錄</div>';
    if (allM.length > MAX_CARDS) {
      ml.insertAdjacentHTML("afterend", '<p class="small muted">只列出最近 ' + MAX_CARDS + " 個賽馬日（共 " + allM.length +
        ' 個）；較早嘅可以喺<a href="#/meeting/' + meetingId(allM[MAX_CARDS]) + '">賽馬日頁</a>用選單揀。</p>');
    }

    drawBoard();
    loadHistory().then(drawCurves);
  }

  function startCountdown(el, target) {
    if (!el || !target) return;
    function cell(v, lbl) {
      var s = String(v).padStart(2, "0");
      return '<div class="cd-cell"><div class="cd-digits">' + s.split("").map(function (d) { return '<span class="flip">' + d + "</span>"; }).join("") +
        '</div><div class="cd-label">' + lbl + "</div></div>";
    }
    function tick() {
      var ms = target.getTime() - Date.now();
      if (ms <= 0) { el.innerHTML = '<span class="chip gold">賽事進行中／已開跑</span>'; return; }
      var s = Math.floor(ms / 1000);
      var d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
      el.innerHTML = cell(d, "日") + cell(h, "時") + cell(m, "分") + cell(sec, "秒");
      el.setAttribute("aria-label", "距離開跑 " + d + " 日 " + h + " 小時 " + m + " 分");
    }
    tick();
    state.timers.push(setInterval(tick, 1000));
  }

  function boardRows(mode) {
    var bots = (state.index.bots || []).slice();
    var rows = bots.map(function (b) {
      var v, sub;
      if (mode === "backtest") {
        var bt = b.backtest;
        v = bt ? (ok(bt.oos_roi) ? bt.oos_roi : ok(bt.roi) ? bt.roi : null) : null;
        sub = bt ? (ok(bt.oos_roi) ? "樣本外 ROI " + pct(bt.oos_roi, 1, true) : "ROI " + pct(bt.roi, 1, true)) : "未有回測";
      } else {
        v = equity(b);
        var gain = ok(v) && b.start_bankroll ? v / b.start_bankroll - 1 : null;
        sub = money(v) + "（" + pct(gain, 1, true) + "）";
      }
      return { bot: b, v: v, sub: sub };
    });
    rows.sort(function (a, b) {
      if (a.v === null && b.v === null) return a.bot.no - b.bot.no;
      if (a.v === null) return 1;
      if (b.v === null) return -1;
      return b.v - a.v || a.bot.no - b.bot.no;
    });
    return rows;
  }

  function drawBoard() {
    var mode = state.prefs.mode;
    var el = document.getElementById("track");
    if (!el) return;
    var rows = boardRows(mode);
    var vals = rows.filter(function (r) { return r.v !== null; }).map(function (r) { return r.v; });
    if (!vals.length) {
      el.innerHTML = '<div class="panel empty">' + (mode === "backtest" ? "回測數據未有／計算緊，稍後再嚟睇。" : "機械人未有實盤紀錄。") + "</div>";
    } else {
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      if (mode === "live") { lo = Math.min(lo, state.index.start_bankroll || 10000); hi = Math.max(hi, state.index.start_bankroll || 10000); }
      var span = hi - lo;
      var html = '<div class="track" role="list" aria-label="機械人排位">';
      rows.forEach(function (r, i) {
        var b = r.bot;
        var x = r.v === null ? 4 : span < 1e-9 ? 50 : 8 + 84 * (r.v - lo) / span;
        html += '<div class="lane" role="listitem"><div class="pos" aria-label="第' + (i + 1) + '位">' + (i + 1) + "</div>" +
          '<a class="who" href="#/bot/' + esc(b.id) + '">' + botToken(b, "sm") + '<span style="min-width:0"><b>' + esc(b.name) + "</b><small>" + esc(r.sub) + "</small></span></a>" +
          '<div class="run"><div class="runner" data-x="' + x.toFixed(1) + '" title="' + esc(b.name) + '">' +
          '<span class="trail" style="width:' + Math.max(0, x * 0.6).toFixed(0) + 'px"></span>' + botToken(b) + "</div></div></div>";
      });
      html += "</div>";
      html += '<div class="track-note">' + (mode === "live"
        ? "跑道位置 = 現時模擬資金（起步 " + money(state.index.start_bankroll || 10000) + "，未結算注碼按本金計）；一樣資金時並排。"
        : "跑道位置 = 樣本外回測回報率（2025-26 季起，門檻冇用呢段數據調）。回測用最終賠率，結果偏樂觀，唔代表將來表現。") + "</div>";
      el.innerHTML = html;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          el.querySelectorAll(".runner").forEach(function (r) { r.style.left = r.getAttribute("data-x") + "%"; });
        });
      });
    }
    var t = document.getElementById("lb-table");
    if (!t) return;
    var head, body;
    if (mode === "backtest") {
      head = "<th>#</th><th>機械人</th><th>回測期</th><th class=r>投注額</th><th class=r>全期 ROI</th><th class=r>樣本外 ROI</th><th class=r>樣本外 95% 區間</th><th class=r>賽馬日</th>";
      body = rows.map(function (r, i) {
        var bt = r.bot.backtest || {};
        var ci = bt.roi_ci && ok(bt.roi_ci[0]) && ok(bt.roi_ci[1]) ? pct(bt.roi_ci[0], 0, true) + " 至 " + pct(bt.roi_ci[1], 0, true) : DASH;
        return "<tr><td>" + (i + 1) + "</td><td>" + botCell(r.bot) + "</td><td class=small>" + esc(bt.period || DASH) +
          "</td><td class='r num'>" + money(bt.stake) + "</td><td class='r num'>" + pct(bt.roi, 1, true) + "</td><td class='r num'><b>" + pct(bt.oos_roi, 1, true) +
          "</b></td><td class='r num small'>" + ci + "</td><td class='r num'>" + (ok(bt.n_meetings) ? bt.n_meetings : DASH) + "</td></tr>";
      }).join("");
    } else {
      head = "<th>#</th><th>機械人</th><th class=r>資金</th><th class=r>盈虧</th><th class=r>ROI</th><th class=r>命中率</th><th class=r>賽馬日</th>";
      body = rows.map(function (r, i) {
        var lv = r.bot.live || {};
        return "<tr><td>" + (i + 1) + "</td><td>" + botCell(r.bot) + "</td><td class='r num'><b>" + money(equity(r.bot)) +
          "</b></td><td class=r>" + plSpan(lv.profit) + "</td><td class='r num'>" + pct(lv.roi, 1, true) + "</td><td class='r num'>" + pct(lv.hit_rate, 0) +
          "</td><td class='r num'>" + (ok(lv.n_meetings) ? lv.n_meetings : DASH) + "</td></tr>";
      }).join("");
    }
    t.innerHTML = '<div class="panel scroll-x"><table class="tbl"><caption>' + (mode === "backtest" ? "回測（歷史模擬）" : "實盤（模擬投注，已結算注項）") +
      "</caption><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>";
  }
  function botCell(b) {
    return '<a href="#/bot/' + esc(b.id) + '" style="display:inline-flex;align-items:center;gap:8px;text-decoration:none">' + botToken(b, "sm") + "<b>" + esc(b.name) + "</b></a>";
  }

  // ------------------------------------------------------------------ curves
  function curveSeries(mode, onlyBot) {
    var hist = state.history || { bots: [] };
    var start = hist.start_bankroll || (state.index && state.index.start_bankroll) || 10000;
    var series = [];
    (hist.bots || []).forEach(function (h) {
      if (onlyBot && h.bot_id !== onlyBot) return;
      var b = botInfo(h.bot_id);
      var pts = (mode === "backtest" ? h.backtest : h.live) || [];
      pts = pts.filter(function (p) { return ok(p.bankroll) && p.date; });
      if (!pts.length) return;
      series.push({ id: h.bot_id, bot: b, points: pts });
    });
    // x domain = union of dates (ordinal), with a start point before the first meeting
    var dates = {};
    series.forEach(function (s) { s.points.forEach(function (p) { dates[p.date] = 1; }); });
    var xs = Object.keys(dates).sort();
    var pos = {};
    var lead = mode === "live" ? 1 : 0;
    xs.forEach(function (d, i) { pos[d] = i + lead; });
    series.forEach(function (s) {
      var pts = s.points.map(function (p) { return { x: pos[p.date], y: p.bankroll, date: p.date, venue: p.venue }; });
      s.xy = lead ? [{ x: 0, y: start, date: null }].concat(pts) : pts;
    });
    return { series: series, xs: lead ? [null].concat(xs) : xs, start: start };
  }

  function drawCurves() {
    var el = document.getElementById("curve");
    if (!el) return;
    var mode = state.prefs.mode;
    var sub = document.getElementById("curve-sub");
    if (sub) sub.textContent = mode === "backtest" ? "回測（樣本外，2025-26 季起）：每個賽馬日後嘅模擬資金" : "實盤：每個已結算賽馬日後嘅模擬資金";
    var cs = curveSeries(mode);
    if (!cs.series.length) {
      el.innerHTML = '<div class="empty">' + (mode === "backtest" ? "未有回測資金曲線。" : "未有已結算嘅賽馬日，資金曲線會喺第一個賽馬日結算後出現。") + "</div>";
      return;
    }
    // default highlight: top 3 by final bankroll
    var ranked = cs.series.slice().sort(function (a, b) { return b.xy[b.xy.length - 1].y - a.xy[a.xy.length - 1].y; });
    var hl = state.highlight && state.highlight.length ? state.highlight : ranked.slice(0, 3).map(function (s) { return s.id; });
    lineChart(el, cs, hl, function (id) {
      var cur = hl.slice();
      var i = cur.indexOf(id);
      if (i >= 0) cur.splice(i, 1); else { cur.push(id); if (cur.length > 3) cur.shift(); }
      state.highlight = cur;
      drawCurves();
    });
  }

  function stratZh(code) {
    if (!code) return DASH;
    return String(code).split("_").map(function (t) { return POOL_ZH[t] || ({ FAV: "大熱", MONKEY: "隨機", ADAPTIVE: "按場自動" })[t] || t; }).join(" ");
  }
  function lineChart(el, cs, highlight, onToggle, single) {
    var W = Math.max(300, el.clientWidth || 600);
    var narrow = W < 560;
    var H = narrow ? 240 : 300;
    var m = { l: narrow ? 50 : 62, r: narrow ? 12 : 104, t: 12, b: 28 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var all = [];
    cs.series.forEach(function (s) { s.xy.forEach(function (p) { all.push(p.y); }); });
    var lo = Math.min.apply(null, all.concat([cs.start])), hi = Math.max.apply(null, all.concat([cs.start]));
    var pad = (hi - lo) * 0.08 || Math.max(50, cs.start * 0.02);
    lo -= pad; hi += pad;
    var nx = cs.xs.length - 1 || 1;
    function X(x) { return m.l + iw * x / nx; }
    function Y(y) { return m.t + ih * (1 - (y - lo) / (hi - lo)); }
    var ticks = niceTicks(lo, hi, narrow ? 4 : 5);
    var svg = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="機械人資金曲線">';
    ticks.forEach(function (t) {
      svg += '<line class="grid-line" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(t).toFixed(1) + '" y2="' + Y(t).toFixed(1) + '"/>' +
        '<text class="axis-text" x="' + (m.l - 8) + '" y="' + (Y(t) + 4).toFixed(1) + '" text-anchor="end">' + compactMoney(t) + "</text>";
    });
    svg += '<line class="base-line" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(cs.start).toFixed(1) + '" y2="' + Y(cs.start).toFixed(1) + '"/>';
    // x labels: start + a few dates
    var step = Math.max(1, Math.ceil(nx / (narrow ? 3 : 6)));
    for (var i = 0; i <= nx; i += step) {
      svg += '<text class="axis-text" x="' + X(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="' + (i === 0 ? "start" : i === nx ? "end" : "middle") + '">' +
        (cs.xs[i] === null ? "起步" : esc(nx > 40 ? String(cs.xs[i]).slice(0, 7) : shortDate(cs.xs[i]))) + "</text>";
    }
    var order = cs.series.slice().sort(function (a, b) { return (highlight.indexOf(a.id) >= 0) - (highlight.indexOf(b.id) >= 0); });
    var labels = [];
    order.forEach(function (s) {
      var on = highlight.indexOf(s.id) >= 0;
      var c = botColor(s.bot)[0];
      var d = s.xy.map(function (p, j) { return (j ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1); }).join(" ");
      svg += '<path class="series' + (on ? "" : " dim") + '" d="' + d + '"' + (on ? ' stroke="' + c + '"' : "") + "/>";
      if (on) {
        if (s.xy.length <= 12) s.xy.forEach(function (p) { svg += '<circle cx="' + X(p.x).toFixed(1) + '" cy="' + Y(p.y).toFixed(1) + '" r="4" fill="' + c + '" stroke="var(--panel)" stroke-width="2"/>'; });
        var last = s.xy[s.xy.length - 1];
        labels.push({ y: Y(last.y), x: X(last.x), s: s, c: c });
      }
    });
    if (!narrow) {
      labels.sort(function (a, b) { return a.y - b.y; });
      for (var k = 1; k < labels.length; k++) if (labels[k].y - labels[k - 1].y < 15) labels[k].y = labels[k - 1].y + 15;
      labels.forEach(function (L) {
        svg += '<text class="end-label" x="' + (L.x + 8).toFixed(1) + '" y="' + (L.y + 4).toFixed(1) + '">' + esc(L.s.bot.no + " " + L.s.bot.name) + "</text>";
      });
    }
    svg += '<line class="xhair" id="xh" x1="0" x2="0" y1="' + m.t + '" y2="' + (H - m.b) + '" style="display:none"/>';
    svg += '<rect x="' + m.l + '" y="' + m.t + '" width="' + iw + '" height="' + ih + '" fill="transparent" id="hit"/>';
    svg += "</svg>";
    var legend = '<div class="legend" role="group" aria-label="揀選要突出嘅機械人（最多 3 隻）">' + cs.series.slice().sort(function (a, b) { return a.bot.no - b.bot.no; }).map(function (s) {
      return '<button type="button" data-id="' + esc(s.id) + '" aria-pressed="' + (highlight.indexOf(s.id) >= 0) + '">' + botToken(s.bot, "sm") + esc(s.bot.name) + "</button>";
    }).join("") + "</div>";
    el.innerHTML = '<div class="chart">' + svg + '<div class="tooltip" role="status"></div></div>' + (single ? "" : legend) +
      '<div class="small muted" style="margin-top:6px">虛線 = 起步資金 ' + money(cs.start) + (single ? "。" : "。撳名牌可以突出顯示（最多 3 隻）。") + "</div>";
    el.querySelectorAll(".legend button").forEach(function (b) { b.addEventListener("click", function () { onToggle(b.getAttribute("data-id")); }); });
    var svgEl = el.querySelector("svg"), tip = el.querySelector(".tooltip"), xh = el.querySelector("#xh"), hit = el.querySelector("#hit");
    function move(ev) {
      var r = svgEl.getBoundingClientRect();
      var px = (ev.clientX - r.left) * W / r.width;
      var xi = Math.max(0, Math.min(nx, Math.round((px - m.l) / iw * nx)));
      xh.setAttribute("x1", X(xi)); xh.setAttribute("x2", X(xi)); xh.style.display = "";
      var rows = cs.series.map(function (s) {
        var p = null;
        for (var j = 0; j < s.xy.length; j++) if (s.xy[j].x <= xi) p = s.xy[j];
        return { s: s, p: p };
      }).filter(function (o) { return o.p; }).sort(function (a, b) { return b.p.y - a.p.y; });
      tip.innerHTML = "<b>" + (cs.xs[xi] === null ? "起步" : esc(cs.xs[xi])) + "</b>" + rows.map(function (o) {
        var on = highlight.indexOf(o.s.id) >= 0;
        return '<div class="row"' + (on ? ' style="font-weight:700"' : "") + '><span><i class="sw" style="background:' + botColor(o.s.bot)[0] + '"></i>' +
          esc(o.s.bot.no + " " + o.s.bot.name) + "</span><span class=num>" + money(o.p.y) + "</span></div>";
      }).join("");
      tip.style.display = "block";
      var left = (X(xi) / W) * r.width + 12;
      if (left + tip.offsetWidth > r.width) left = (X(xi) / W) * r.width - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = "8px";
    }
    function leave() { tip.style.display = "none"; xh.style.display = "none"; }
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerdown", move);
    hit.addEventListener("pointerleave", leave);
    var onResize = debounce(function () { if (document.body.contains(el)) lineChart(el, cs, highlight, onToggle, single); }, 200);
    window.addEventListener("resize", onResize);
    state.cleanup.push(function () { window.removeEventListener("resize", onResize); });
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
    return out;
  }
  function compactMoney(v) { return Math.abs(v) >= 10000 ? "$" + (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k" : money(v, 0); }
  function debounce(f, ms) { var t; return function () { clearTimeout(t); t = setTimeout(f, ms); }; }

  // ------------------------------------------------------------------ meeting page
  function pageMeeting(id, raceNo) {
    var idx = state.index;
    if (!id) { var d = defaultMeeting(idx); if (d) { location.replace("#/meeting/" + meetingId(d)); return; } }
    if (!id) { app.innerHTML = '<div class="panel empty">未有賽馬日數據。</div>'; return; }
    app.innerHTML = '<div class="loading">載入賽馬日數據…</div>';
    return loadMeeting(id).then(function (m) { renderMeeting(m, raceNo); }).catch(function () {
      app.innerHTML = '<div class="panel empty">搵唔到呢個賽馬日嘅數據。<br><a href="#/">返首頁</a></div>';
    });
  }

  function renderMeeting(m, raceNo) {
    var idx = state.index;
    var id = meetingId(m);
    var races = m.races || [];
    var rn = parseInt(raceNo, 10);
    if (!races.some(function (r) { return r.race_no === rn; })) rn = races.length ? races[0].race_no : null;
    var opts = (idx.meetings || []).map(function (x) {
      return '<option value="' + meetingId(x) + '"' + (meetingId(x) === id ? " selected" : "") + ">" + shortDate(x.date) + " " + esc(x.venue_zh || VENUE_ZH[x.venue]) +
        "（" + esc(STATUS_ZH[x.status] || x.status) + "）</option>";
    }).join("");
    var html = demoBanner(idx);
    html += '<div class="mtg-head"><h1>' + meetingName(m) + '</h1><span class="chip status-' + esc(m.status) + '">' + esc(STATUS_ZH[m.status] || m.status) + "</span>" +
      '<label class="small muted" style="display:flex;align-items:center;gap:6px">賽馬日 <select class="select" id="mtg-sel">' + opts + "</select></label></div>";
    html += '<div class="mtg-meta"><span>賠率快照：' + (m.odds_snapshot_at ? esc(String(m.odds_snapshot_at).replace("T", " ").slice(0, 16)) : "未開售（只用模型估計）") + "</span>" +
      "<span>模型：" + esc(m.model_version || DASH) + "</span><span>生成時間：" + esc(m.generated_at ? String(m.generated_at).replace("T", " ").slice(0, 16) : DASH) + "</span></div>";

    html += '<nav class="race-tabs" aria-label="場次">' + races.map(function (r) {
      return '<a href="#/meeting/' + id + "/" + r.race_no + '"' + (r.race_no === rn ? ' aria-current="true"' : "") + ">" + r.race_no + "<small>" + hhmm(r.post_time) + "</small></a>";
    }).join("") + "</nav>";
    html += '<div id="race"></div>';
    if ((m.meeting_decisions || []).length) {
      html += '<section class="section"><div class="section-head"><h2>跨場／過關決定</h2><span class="muted small">成個賽馬日一齊計</span></div><div class="dec-grid">' +
        m.meeting_decisions.map(function (d) { return decisionCard(d, m, null); }).join("") + "</div></section>";
    }
    html += meetingBotsSummary(m);
    app.innerHTML = html;
    document.getElementById("mtg-sel").addEventListener("change", function (e) { location.hash = "#/meeting/" + e.target.value; });
    var race = races.filter(function (r) { return r.race_no === rn; })[0];
    if (race) renderRace(document.getElementById("race"), race, m);
    else document.getElementById("race").innerHTML = '<div class="panel empty">排位表未公佈。</div>';
    if (raceNo) { var cur = app.querySelector('.race-tabs [aria-current="true"]'); if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest", inline: "center" }); }
  }

  function meetingBotsSummary(m) {
    var bots = (m.bots || []).slice();
    if (!bots.length) return "";
    var settled = m.status === "results";
    bots.sort(function (a, b) {
      if (settled && ok(a.profit) && ok(b.profit)) return b.profit - a.profit;
      return botInfo(a.bot_id).no - botInfo(b.bot_id).no;
    });
    var rows = bots.map(function (b, i) {
      var info = botInfo(b.bot_id);
      return "<tr><td>" + (settled ? i + 1 : info.no) + "</td><td>" + botCell(info) + "</td><td class=r>" + (ok(b.profit) ? plSpan(b.profit) : '<span class="muted">' + DASH + "</span>") +
        "</td><td class='r num'>" + money(b.stake) + "</td><td class='r num'>" + (ok(b.payout) ? money(b.payout) : '<span class="muted">未結算</span>') +
        "</td><td class='r num'>" + (ok(b.n_tickets) ? b.n_tickets : DASH) + "</td></tr>";
    }).join("");
    return '<section class="section"><div class="section-head"><h2>' + (settled ? "本日機械人戰績" : "本日機械人注項") + "</h2></div>" +
      '<div class="panel scroll-x"><table class="tbl"><thead><tr><th>#</th><th>機械人</th><th class=r>盈虧</th><th class=r>投注額</th><th class=r>派彩</th><th class=r>飛數</th></tr></thead><tbody>' +
      rows + "</tbody></table></div></section>";
  }

  function renderRace(el, r, m) {
    var horses = (r.horses || []).slice();
    var hasScore = horses.some(function (h) { return ok(h.score); });
    var sort = state.prefs.sort;
    horses.sort(function (a, b) {
      if (a.scratched !== b.scratched) return a.scratched ? 1 : -1;
      if (sort === "score" && hasScore) return (ok(b.score) ? b.score : -1) - (ok(a.score) ? a.score : -1);
      if (sort === "finish") return finKey(a) - finKey(b);
      return a.horse_no - b.horse_no;
    });
    var info = [r.race_class, r.rating_band ? "(" + r.rating_band + ")" : null, ok(r.distance) ? r.distance + "米" : null,
      SURFACE_ZH[r.surface] || r.surface, r.course ? "「" + r.course + "」賽道" : null, r.going].filter(Boolean).map(esc).join(" · ");
    var html = '<div class="race-bar"><div class="rb-no">第 ' + r.race_no + ' 場</div><div class="rb-main"><b>' + esc(r.race_name || "") + "</b><span>" + info + "</span></div>" +
      '<div class="rb-time">' + hhmm(r.post_time) + "</div></div>";
    if (r.result) html += finishGraphic(r);
    html += '<div class="section-head section"><h2>排位表 · 評分</h2><span class="spacer"></span><div class="seg" role="group" aria-label="排序">' +
      '<button type="button" data-sort="no" aria-pressed="' + (sort === "no") + '">按馬號</button>' +
      (hasScore ? '<button type="button" data-sort="score" aria-pressed="' + (sort === "score") + '">按評分</button>' : "") +
      (r.result ? '<button type="button" data-sort="finish" aria-pressed="' + (sort === "finish") + '">按名次</button>' : "") + "</div></div>";
    html += '<div class="card-head" aria-hidden="true"><span>馬號</span><span>馬名 / 騎師 / 練馬師</span><span>評分</span><span>子評分</span><span>勝率 / 上名率</span><span>賠率</span></div>';
    html += '<div class="rows">' + horses.map(function (h) { return horseRow(h, r); }).join("") + "</div>";
    if (!hasScore) html += '<p class="small muted">呢場未有評分（模型未計／排位表剛公佈）。</p>';
    html += '<section class="section"><div class="section-head"><h2>10 隻機械人點樣決定</h2><span class="muted small">每場都會寫低買定唔買同原因</span></div>';
    var decs = r.decisions || [];
    html += decs.length ? '<div class="dec-grid">' + decs.map(function (d) { return decisionCard(d, m, r); }).join("") + "</div>"
      : '<div class="panel empty">呢場未有機械人決定（通常喺第一場開跑前約 2 小時出）。</div>';
    html += "</section>";
    if (r.result && r.result.dividends) html += dividendsBlock(r.result.dividends);
    el.innerHTML = html;
    el.querySelectorAll("[data-sort]").forEach(function (b) {
      b.addEventListener("click", function () { state.prefs.sort = b.getAttribute("data-sort"); savePrefs(); renderRace(el, r, m); });
    });
  }
  function finKey(h) { return h.result && ok(h.result.finish_pos) ? h.result.finish_pos : 99 + h.horse_no / 100; }

  function horseRow(h, r) {
    var c = CLOTH[((h.horse_no || 1) - 1) % CLOTH.length];
    var lit = ok(h.score) ? Math.round(h.score / 5) : 0;
    var energy = "";
    for (var i = 0; i < 20; i++) energy += "<i" + (i < lit ? ' class="on"' : "") + "></i>";
    var subs = SUB_KEYS.map(function (k) {
      var v = h.sub_scores ? h.sub_scores[k] : null;
      var lbl = (state.index.sub_score_labels && state.index.sub_score_labels[k]) || SUB_SHORT[k];
      return '<div class="sub" title="' + esc(lbl) + "：" + (ok(v) ? Math.round(v) : "未有") + '"><div class="bar' + (ok(v) ? "" : " na") + '">' +
        (ok(v) ? '<i style="height:' + Math.max(4, Math.min(100, v)) + '%"></i>' : "") + "</div><span>" + SUB_SHORT[k] + "</span></div>";
    }).join("");
    var fin = "";
    if (h.result && ok(h.result.finish_pos)) fin = '<span class="fin-tag' + (h.result.finish_pos <= 3 ? " p" + h.result.finish_pos : "") + '">第' + h.result.finish_pos + "名</span>";
    else if (h.result && h.result.finish_status) fin = '<span class="fin-tag">' + esc(h.result.finish_status) + "</span>";
    var winO = h.win_odds, plaO = h.pla_odds, oddsLabel = "獨贏";
    if (!ok(winO) && h.result && ok(h.result.final_win_odds)) { winO = h.result.final_win_odds; oddsLabel = "最終獨贏"; }
    var reasons = (h.reasons || []).map(function (s) { return '<span class="reason">' + esc(s) + "</span>"; }).join("");
    return '<article class="hrow' + (h.scratched ? " scratched" : "") + (h.rank === 1 && !h.scratched ? " top1" : "") + '" style="--c:' + c[0] + '">' +
      '<div class="c-cloth">' + cloth(h.horse_no, "xl") + (ok(h.rank) && !h.scratched && h.rank <= 3 ? '<span class="rank-tag">評分第' + h.rank + "</span>" : "") + "</div>" +
      '<div class="c-name"><div class="hn">' + esc(h.horse_name || DASH) + (h.scratched ? ' <span class="chip">退出</span>' : "") + fin + "</div>" +
      '<div class="jt">' + esc(h.jockey || DASH) + " / " + esc(h.trainer || DASH) + "</div>" +
      '<div class="dw">檔位 ' + num(h.draw) + " · 負磅 " + num(h.weight) + "</div></div>" +
      (!ok(h.score) && !ok(h.p_win) ? '<div class="c-score muted small">' + (h.scratched ? "退出，冇評分" : "未有評分") + "</div>" :
      '<div class="c-score"><div class="score-line"><span class="score-num">' + (ok(h.score) ? Math.round(h.score) : DASH) + '</span><div class="energy" role="img" aria-label="評分 ' +
      (ok(h.score) ? Math.round(h.score) : "未有") + ' / 100">' + energy + '</div></div><div class="score-cap">綜合評分 / 100</div></div>' +
      '<div class="c-subs"><div class="subs">' + subs + "</div></div>" +
      '<div class="c-prob"><div>勝 <b>' + pct(h.p_win) + "</b></div><div>位 <b>" + pct(h.p_place) + "</b></div></div>") +
      '<div class="c-odds">' + ledBoard([[oddsLabel, winO], ["位置", plaO]]) + "</div>" +
      (reasons ? '<div class="c-reasons">' + reasons + "</div>" : "") + "</article>";
  }

  function ledBoard(rows) {
    return '<div class="led-board">' + rows.map(function (r) {
      var t = oddsTxt(r[1]);
      return '<div class="led-row"><span class="led-k">' + esc(r[0]) + '</span><span class="led-v' + (t ? "" : " off") + '">' +
        (t ? t.split("").map(function (ch) { return "<b>" + esc(ch) + "</b>"; }).join("") : "<b>未開售</b>") + "</span></div>";
    }).join("") + "</div>";
  }

  function finishGraphic(r) {
    var res = r.result, divs = res.dividends || {};
    var win = {}, pla = {};
    (divs.WIN || []).forEach(function (d) { win[d[0]] = d[1]; });
    (divs.PLA || []).forEach(function (d) { pla[d[0]] = d[1]; });
    var top = (res.placings || []).filter(function (p) { return p.pos <= 3; });
    if (!top.length) return "";
    var POS = { 1: "冠軍", 2: "亞軍", 3: "季軍" };
    return '<section class="section" aria-label="賽果"><div class="section-head"><span class="kicker">RESULT</span><h2>賽果</h2></div><div class="finish">' +
      top.map(function (p) {
        var w = win[String(p.horse_no)], pl = pla[String(p.horse_no)];
        return '<div class="fin-row p' + p.pos + '"><div class="place"><span>' + p.pos + "</span></div>" + cloth(p.horse_no) +
          '<div class="fn"><b>' + esc(p.horse_name) + "</b><span>" + esc(POS[p.pos] || "") + " · " + esc(p.jockey || "") + (ok(p.win_odds) ? " · 獨贏賠率 " + oddsTxt(p.win_odds) : "") + "</span></div>" +
          '<div class="divs">' + (ok(w) ? "獨贏 <b>" + money(w, 1) + "</b><br>" : "") + (ok(pl) ? "位置 <b>" + money(pl, 1) + "</b>" : "") + "</div></div>";
      }).join("") + '</div><p class="small muted">派彩以每 $10 計。</p></section>';
  }

  function dividendsBlock(divs) {
    var cells = POOL_ORDER.filter(function (p) { return divs[p] && divs[p].length; }).map(function (p) {
      return '<div class="div-cell"><h4>' + POOL_ZH[p] + "</h4>" + divs[p].map(function (d) {
        return "<div><span>" + esc(d[0]) + "</span><b>" + money(d[1], ok(d[1]) && d[1] >= 1000 ? 0 : 1) + "</b></div>";
      }).join("") + "</div>";
    }).join("");
    return cells ? '<section class="section"><div class="section-head"><h2>派彩</h2><span class="muted small">每 $10</span></div><div class="div-grid">' + cells + "</div></section>" : "";
  }

  function selHtml(sel, raceNo) {
    if (!ok(sel)) return "";
    if (typeof sel === "number" || typeof sel === "string") return cloth(sel, "sm");
    if (Array.isArray(sel)) {
      if (!sel.length) return "";
      if (typeof sel[0] === "object" && sel[0] !== null) {
        return sel.map(function (leg) {
          return '<span class="leg-lbl">' + (ok(leg.race_no) ? "第" + leg.race_no + "場" : "") + "</span>" + selHtml(leg.bankers ? { bankers: leg.bankers, legs: leg.selections || leg.legs } : (leg.selections || leg.legs || leg.horses), leg.race_no);
        }).join(" ");
      }
      if (Array.isArray(sel[0])) return sel.map(function (s, i) { return '<span class="leg-lbl">第' + (i + 1) + "位</span>" + selHtml(s); }).join(" ");
      return sel.map(function (h) { return cloth(h, "sm"); }).join("");
    }
    if (typeof sel === "object") {
      var out = "";
      if (sel.bankers) out += '<span class="leg-lbl">膽</span>' + selHtml(sel.bankers);
      var legs = sel.legs || sel.selections || sel.horses;
      if (legs) out += '<span class="leg-lbl">' + (sel.bankers ? "拖" : "") + "</span>" + selHtml(legs);
      if (sel.positions) out += selHtml(sel.positions);
      if (!out && sel.horse_no) out = cloth(sel.horse_no, "sm");
      return out;
    }
    return "";
  }

  function decisionCard(d, m, r) {
    var b = botInfo(d.bot_id);
    var bet = d.action === "bet";
    var settled = m.status === "results";
    var money_ = "";
    if (bet && (d.parlay_leg || !(d.stake > 0))) {
      money_ = '<div class="money"><span>注碼 <b style="color:var(--text)">計入過關</b></span><span class="muted">盈虧見「跨場／過關決定」</span></div>';
    } else if (bet) {
      money_ = '<div class="money"><span>注碼 <b class="num" style="color:var(--text)">' + money(d.stake) + "</b></span>";
      if (ok(d.profit)) money_ += "<span>派彩 " + money(d.payout) + " · " + plSpan(d.profit) + "</span>";
      else if (settled && d.stake > 0) money_ += '<span class="muted">' + (d.ticket_status === "open" ? "未結算" : DASH) + "</span>";
      else if (d.stake > 0) money_ += '<span class="muted">等賽果</span>';
      money_ += "</div>";
    }
    var sel = selHtml(d.selections, r ? r.race_no : null);
    return '<article class="dec' + (bet ? "" : " is-skip") + '"><div class="dec-top">' + botToken(b, "sm") + "<b>" + esc(b.name) + "</b>" +
      '<span class="chip ' + (bet ? "bet" : "skip") + '">' + (bet ? "買" : "唔買") + "</span></div>" +
      (d.structure && d.structure !== "-" ? '<div class="small muted">' + esc(d.structure) + "</div>" : "") +
      (sel ? '<div class="sel">' + sel + "</div>" : "") +
      '<div class="why">' + esc(d.reason || (bet ? "" : "冇寫原因")) + "</div>" + money_ + "</article>";
  }

  // ------------------------------------------------------------------ robots
  function pageBots() {
    var bots = (state.index.bots || []).slice().sort(function (a, b) { return a.no - b.no; });
    var html = demoBanner(state.index) + '<div class="section-head"><span class="kicker">ROBOTS</span><h1>10 隻模擬投注機械人</h1></div>' +
      '<p class="muted" style="margin-top:0">每隻機械人用同一套模型預測，但玩法同落注門檻唔同；各自有獨立 ' + money(state.index.start_bankroll || 10000) + " 模擬資金。「跟大熱」同「隨機猴子」係對照組。</p>";
    html += '<div class="bot-cards">' + bots.map(function (b) {
      var lv = b.live || {}, bt = b.backtest || {};
      return '<a class="bot-card" href="#/bot/' + esc(b.id) + '" style="--c:' + botColor(b)[0] + '"><div class="bc-top">' + botToken(b, "lg") +
        '<div><b style="font-size:17px">' + esc(b.name) + '</b><div class="small muted">' + esc(b.style) + "</div></div></div>" +
        '<div class="small" style="color:var(--text-2)">' + esc(b.description) + "</div>" +
        '<div class="mini-stats"><span>資金 <b>' + money(equity(b)) + "</b></span><span>實盤 ROI <b>" + pct(lv.roi, 1, true) + "</b></span><span>回測 ROI <b>" +
        pct(bt.roi, 1, true) + "</b></span></div></a>";
    }).join("") + "</div>";
    app.innerHTML = html;
    focusMain();
  }

  function pageBot(id) {
    var b = botInfo(id);
    if (!b || !b.no) { app.innerHTML = '<div class="panel empty">搵唔到呢隻機械人。<a href="#/bots">返機械人列表</a></div>'; return; }
    var lv = b.live || {}, bt = b.backtest;
    var eq = equity(b);
    var gain = ok(eq) && b.start_bankroll ? eq / b.start_bankroll - 1 : null;
    var html = demoBanner(state.index) + '<div class="mtg-head">' + botToken(b, "lg") + '<div style="flex:1;min-width:0"><h1>' + esc(b.name) + '</h1><div class="muted">' +
      esc(b.style) + " · 機械人 " + b.no + " 號</div></div>" +
      '<label class="small muted" style="display:flex;align-items:center;gap:6px">機械人 <select class="select" id="bot-sel">' +
      (state.index.bots || []).map(function (x) { return '<option value="' + esc(x.id) + '"' + (x.id === id ? " selected" : "") + ">" + x.no + " " + esc(x.name) + "</option>"; }).join("") +
      "</select></label></div>";
    html += '<p style="color:var(--text-2)">' + esc(b.description) + "</p>";
    html += '<div class="stats">' +
      stat("現時資金", money(eq), "起步 " + money(b.start_bankroll) + "（" + pct(gain, 1, true) + "）" +
        (lv.open_stake > 0 ? "；其中 " + money(lv.open_stake) + " 喺未結算注項" : "")) +
      stat("實盤盈虧", ok(lv.profit) ? signedMoney(lv.profit) : DASH, "投注 " + money(lv.stake) + " · 派彩 " + money(lv.returns)) +
      stat("實盤 ROI", pct(lv.roi, 1, true), (ok(lv.n_meetings) ? lv.n_meetings : 0) + " 個賽馬日 · " + (ok(lv.n_tickets) ? lv.n_tickets : 0) + " 飛已結算") +
      stat("命中率", pct(lv.hit_rate, 0), "有派彩嘅飛數比例") +
      stat("回測 ROI（全期）", bt ? pct(bt.roi, 1, true) : DASH, bt ? esc(bt.period || "") + (ok(bt.n_meetings) ? " · " + bt.n_meetings + " 個賽馬日" : "") : "未有回測") +
      stat("樣本外 ROI", bt ? pct(bt.oos_roi, 1, true) : DASH, bt && bt.roi_ci && ok(bt.roi_ci[0]) ? "95% 區間 " + pct(bt.roi_ci[0], 0, true) + " 至 " + pct(bt.roi_ci[1], 0, true) + "（" + esc(bt.oos_period || "樣本外") + "）" : "門檻冇用呢段數據調") +
      "</div>" + (bt && bt.caveat ? '<p class="small muted">' + esc(bt.caveat) + "</p>" : "");
    html += '<section class="section"><div class="section-head"><h2>資金曲線</h2><span class="spacer"></span><div class="seg" role="group" aria-label="曲線">' +
      '<button type="button" data-mode="live" aria-pressed="' + (state.prefs.mode === "live") + '">實盤</button>' +
      '<button type="button" data-mode="backtest" aria-pressed="' + (state.prefs.mode === "backtest") + '">回測</button></div></div><div class="panel pad"><div id="curve1"></div></div></section>';
    html += '<section class="section"><div class="section-head"><h2>全部注項</h2><span class="muted small" id="tk-sub"></span></div><div id="tickets"><div class="loading">載入中…</div></div></section>';
    app.innerHTML = html;
    document.getElementById("bot-sel").addEventListener("change", function (e) { location.hash = "#/bot/" + e.target.value; });
    function curve() {
      var el = document.getElementById("curve1");
      if (!el) return;
      var cs = curveSeries(state.prefs.mode, id);
      if (!cs.series.length) { el.innerHTML = '<div class="empty">' + (state.prefs.mode === "backtest" ? "未有回測曲線。" : "未有已結算嘅賽馬日。") + "</div>"; return; }
      lineChart(el, cs, [id], function () {}, true);
    }
    app.querySelectorAll(".seg [data-mode]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.prefs.mode = btn.getAttribute("data-mode"); savePrefs();
        app.querySelectorAll(".seg [data-mode]").forEach(function (x) { x.setAttribute("aria-pressed", String(x === btn)); });
        curve();
      });
    });
    loadHistory().then(curve);
    // one file per robot (data/bots/<id>.json); older exports: gather from every meeting file
    getJSON("data/bots/" + encodeURIComponent(id) + ".json").then(function (d) {
      return (d.tickets || []).map(function (t) { return { m: { date: t.date, venue: t.venue, venue_zh: t.venue_zh }, t: t }; });
    }).catch(function () {
      var ms = state.index.meetings || [];
      return Promise.all(ms.map(function (x) { return loadMeeting(meetingId(x)).catch(function () { return null; }); })).then(function (list) {
        var out = [];
        list.forEach(function (m) {
          if (!m) return;
          (m.bots || []).forEach(function (bb) {
            if (bb.bot_id !== id) return;
            (bb.tickets || []).forEach(function (t) { out.push({ m: m, t: t }); });
          });
        });
        return out;
      });
    }).then(function (rows) {
      var el = document.getElementById("tickets");
      if (!el) return;
      document.getElementById("tk-sub").textContent = rows.length ? "共 " + rows.length + " 飛" : "";
      if (!rows.length) { el.innerHTML = '<div class="panel empty">未有注項。</div>'; return; }
      el.innerHTML = '<div class="panel scroll-x"><table class="tbl"><thead><tr><th>賽馬日</th><th>玩法</th><th>組合</th><th class=r>注碼</th><th class=r>派彩</th><th class=r>盈虧</th><th>原因</th></tr></thead><tbody>' +
        rows.map(function (o) {
          var t = o.t;
          return '<tr><td class="nowrap"><a href="#/meeting/' + meetingId(o.m) + '">' + shortDate(o.m.date) + " " + esc(o.m.venue_zh || VENUE_ZH[o.m.venue]) + "</a></td>" +
            "<td class=nowrap title=\"" + esc(t.strategy) + "\">" + esc(stratZh(t.strategy)) + (ok(t.n_combos) ? '<div class="small muted">' + t.n_combos + " 注 × " + money(t.unit_stake) + "</div>" : "") + "</td>" +
            '<td class="wrap"><div class="sel" style="display:flex;flex-wrap:wrap;gap:4px;align-items:center">' + selHtml(t.legs) + "</div></td>" +
            "<td class='r num'>" + money(t.total_stake) + "</td><td class='r num'>" + (t.status === "settled" ? money(t.payout) : '<span class="muted">未結算</span>') + "</td>" +
            "<td class=r>" + (t.status === "settled" ? plSpan(t.profit) : '<span class="muted">' + DASH + "</span>") + '</td><td class="small wrap">' + esc(t.reason || "") + "</td></tr>";
        }).join("") + "</tbody></table></div>";
    });
    focusMain();
  }
  function stat(k, v, s) { return '<div class="stat"><div class="k">' + esc(k) + '</div><div class="v">' + v + '</div><div class="s">' + s + "</div></div>"; }

  // ------------------------------------------------------------------ about
  function pageAbout() {
    var idx = state.index;
    var labels = idx.sub_score_labels || {};
    var SUB_DESC = {
      form: "近期名次、頭馬距離、速度同表現評級", jockey: "騎師近一年同近期勝出／上名率（按場地、跑道、路程）",
      trainer: "練馬師近一年同近期勝出／上名率", combo: "呢位騎師同練馬師合作嘅往績", draw: "檔位喺呢個路程／跑道嘅歷史優劣",
      fit: "匹馬喺呢個路程、場地、跑道嘅往績", "class": "班次升降、評分、負磅相對同場", market: "即時賠率反映嘅市場看法（未開售時留空）"
    };
    var html = '<div class="prose"><div class="section-head"><span class="kicker">GUIDE</span><h1>說明</h1></div>';
    html += "<h2>評分代表咩？</h2><p>" + esc(idx.score_explanation || "") + "</p>";
    html += '<h2>子評分</h2><div class="sub-explain">' + SUB_KEYS.map(function (k) {
      return "<div><b>" + esc(labels[k] || SUB_SHORT[k]) + "</b><br>" + esc(SUB_DESC[k]) + "</div>";
    }).join("") + '</div><p class="small">子評分係同場馬匹之間嘅相對分數（0 = 同場最差，100 = 同場最好），只用嚟解釋綜合評分從何而來。</p>';
    html += "<h2>方法</h2><ul>" +
      "<li>數據：公開嘅歷屆賽果、派彩、排位表，同賽前公開賠率（開售後）。</li>" +
      "<li>模型：按每場比賽估計每匹馬嘅勝出率（勝）同上名率（位）；每場勝出率加埋等於 100%。</li>" +
      "<li>防止偷看未來：所有特徵只用該場之前嘅資料；回測用 walk-forward（每次只用預測日之前嘅數據訓練）。</li>" +
      "<li>賽前約 2 小時攞賠率、計評分，10 隻機械人各自決定買唔買、買咩，並記錄原因；賽後用正式派彩結算。</li>" +
      "<li>回測用嘅係最終賠率，開跑前 2 小時嘅賠率會唔同，所以回測結果偏樂觀；試過好多策略，好睇嘅結果好可能只係運氣。</li>" +
      "<li>彩池抽水大約 17.5%–25%，過關每一關都要扣；長遠嚟講大部分玩法都係蝕錢。</li></ul>";
    html += "<h2>10 隻機械人</h2><ul>" + (idx.bots || []).map(function (b) {
      return '<li><a href="#/bot/' + esc(b.id) + '">' + b.no + " 號 " + esc(b.name) + "</a>（" + esc(b.style) + "）：" + esc(b.description) + "</li>";
    }).join("") + "</ul>";
    html += "<h2>免責聲明</h2><ul>" +
      "<li><b>" + esc(idx.disclaimer || "") + "</b></li>" +
      "<li>呢度所有投注都係電腦模擬（paper trade），從來冇落過任何真注，亦唔會代任何人落注。</li>" +
      "<li>內容只供研究同學習，唔構成任何投資或投注建議；過往（包括回測）表現唔代表將來。</li>" +
      "<li>本網站與香港賽馬會或任何賽馬機構無關，並非官方網站，亦冇使用其標誌或商標；賽事資料可能有誤，請以官方公佈為準。</li>" +
      "<li>如果覺得賭博影響到你或身邊嘅人，可以致電平和基金熱線 <b>1834 633</b> 尋求協助。</li></ul>";
    html += '<p class="small muted">數據更新：' + esc(idx.updated_at ? String(idx.updated_at).replace("T", " ").slice(0, 16) : DASH) + "（香港時間）</p></div>";
    app.innerHTML = html;
    focusMain();
  }

  // ------------------------------------------------------------------ boot
  var tb = document.getElementById("theme-btn");
  if (tb) tb.addEventListener("click", function () {
    var cur = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = cur;
    try { localStorage.setItem("pd-theme", cur); } catch (e) { /* ignore */ }
  });
  var lastPage = null;
  window.addEventListener("hashchange", function () {
    var key = (location.hash || "#/").split("/").slice(0, 3).join("/");
    var samePage = key === lastPage;
    lastPage = key;
    route();
    if (!samePage) window.scrollTo(0, 0);
  });
  lastPage = (location.hash || "#/").split("/").slice(0, 3).join("/");
  route();
})();
