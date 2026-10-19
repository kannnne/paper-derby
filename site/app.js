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
  // Robot colours (robot number 1..12 always shown with the colour).
  var BOT_COLORS = [
    ["#3987e5", "#ffffff"], ["#d95926", "#ffffff"], ["#199e70", "#ffffff"], ["#c98500", "#111111"],
    ["#d55181", "#ffffff"], ["#2fa32f", "#ffffff"], ["#9085e9", "#111111"], ["#e66767", "#111111"],
    ["#9aa4b1", "#111111"], ["#8a5a2b", "#ffffff"], ["#1fb5c9", "#0b1a1a"], ["#b04fc4", "#ffffff"]
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
  var state = { index: null, meetings: {}, meetingP: {}, history: null, timers: [], prefs: { sort: "no" }, cleanup: [], homeRace: {} };
  try {
    var saved = JSON.parse(localStorage.getItem("pd-prefs") || "{}");
    if (saved && typeof saved === "object") {
      if (saved.sort === "no" || saved.sort === "score") state.prefs.sort = saved.sort;
    }
  } catch (e) { /* storage unavailable: defaults */ }
  function savePrefs() { try { localStorage.setItem("pd-prefs", JSON.stringify({ sort: state.prefs.sort })); } catch (e) { /* ignore */ } }

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
    if (state.meetingP[id]) return state.meetingP[id];
    var m = (state.index.meetings || []).filter(function (x) { return meetingId(x) === id; })[0];
    var file = m ? m.file : "data/meetings/" + id + ".json";
    state.meetingP[id] = getJSON(file).then(function (d) { state.meetings[id] = d; delete state.meetingP[id]; return d; },
      function (e) { delete state.meetingP[id]; throw e; });
    return state.meetingP[id];
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
  function sortedMeetings(idx) {   // newest first
    return (idx.meetings || []).slice().sort(function (a, b) {
      return a.date < b.date ? 1 : a.date > b.date ? -1 : String(a.venue).localeCompare(String(b.venue));
    });
  }
  // Meeting whose horse scores the home page shows: the upcoming meeting once its pre-race run has
  // published scores, otherwise the latest past meeting with scores (null = none yet).
  function scoresMeeting(idx) {
    var ms = sortedMeetings(idx);
    var nm = idx.next_meeting;
    if (!ms.some(function (m) { return typeof m.has_scores === "boolean"; })) return defaultMeeting(idx);   // older export
    if (nm) {
      var n = ms.filter(function (m) { return m.date === nm.date && m.venue === nm.venue && m.has_scores; })[0];
      if (n) return n;
    }
    return ms.filter(function (m) { return m.has_scores && (!nm || m.date <= nm.date); })[0] || null;
  }
  function preRaceEta(nm) {   // 「約 HH:MM」= first post time − 2 h (Hong Kong time)
    var d = nm && parseHK(nm.first_post_time);
    if (!d) return "";
    return "（約 " + new Date(d.getTime() + 6 * 3600e3).toISOString().slice(11, 16) + "）";
  }
  function nextScoresNote(idx) {
    var nm = idx.next_meeting;
    if (!nm) return "評分會喺下一個賽馬日第一場開跑前約 2 小時自動出。";
    return "下一個賽馬日（" + shortDate(nm.date) + " " + esc(nm.venue_zh || VENUE_ZH[nm.venue] || nm.venue) + "）嘅評分會喺第一場開跑前約 2 小時" +
      preRaceEta(nm) + "自動出。";
  }

  function pageHome() {
    var idx = state.index;
    var nm = idx.next_meeting;
    var html = demoBanner(idx) + '<h1 class="sr-only">跑道機械人 · 模擬賽馬研究</h1>';

    // ① 賽馬日紀錄 (newest first) — very top
    html += '<section class="section top-section" aria-labelledby="mtg-h"><div class="section-head"><span class="kicker">MEETINGS</span><h2 id="mtg-h">賽馬日紀錄</h2>' +
      '<span class="muted small">最新排最前</span></div><div class="mtg-list" id="mtg-list"></div><div id="mtg-more"></div></section>';
    // ①b 今日精選 (same meeting file as the scores when both exist)
    html += '<section class="section" aria-labelledby="pk-h"><div class="section-head"><span class="kicker">PICKS</span><h2 id="pk-h">今日精選</h2>' +
      '<span class="muted small" id="pk-sub"></span></div><div id="home-picks"><div class="panel loading">載入精選…</div></div></section>';
    // ② 最新賽馬日馬匹評分 (one meeting file, lazy)
    html += '<section class="section" aria-labelledby="hs-h"><div class="section-head"><span class="kicker">SCORES</span><h2 id="hs-h">最新馬匹評分</h2>' +
      '<span class="muted small" id="hs-sub"></span></div><div id="home-scores"><div class="panel loading">載入評分…</div></div></section>';
    // ③ 機械人賽道排行榜 (race-track lanes only; table + curve live on the robot page)
    html += '<section class="section" aria-labelledby="lb-h"><div class="section-head"><span class="kicker">LIVE</span><h2 id="lb-h">機械人賽道排行榜</h2>' +
      '<span class="muted small">實盤（模擬投注）</span></div><div id="track"></div>' +
      '<div class="cta-row"><a class="cta" href="#/bots">睇全部機械人 →</a></div></section>';
    app.innerHTML = html;

    var shown = scoresMeeting(idx);
    var ml = document.getElementById("mtg-list");
    var MAX_CARDS = 12;   // a season has ~90 meetings: older ones via the meeting page's selector
    var allM = sortedMeetings(idx);
    ml.innerHTML = allM.slice(0, MAX_CARDS).map(function (m) {
      var isNext = nm && m.date === nm.date && m.venue === nm.venue;
      return '<a class="mtg-card' + (isNext ? " is-next" : "") + '" href="#/meeting/' + meetingId(m) + '">' +
        '<div class="mc-top"><b>' + shortDate(m.date) + " " + esc(m.venue_zh || VENUE_ZH[m.venue] || m.venue) + "</b>" +
        '<span class="chip status-' + esc(m.status) + '">' + esc(STATUS_ZH[m.status] || m.status) + "</span></div>" +
        '<div class="muted small">' + dateZh(m.date) + (ok(m.n_races) ? " · " + m.n_races + " 場" : "") + "</div>" +
        '<div class="mc-tags">' + (isNext ? '<span class="chip gold">下一個</span>' : "") +
        (m.has_scores ? '<span class="chip">有評分</span>' : "") + (m.has_decisions ? '<span class="chip">機械人已決定</span>' : "") + "</div></a>";
    }).join("") || '<div class="panel empty">未有紀錄</div>';
    if (allM.length > MAX_CARDS) {
      document.getElementById("mtg-more").innerHTML = '<p class="small muted">只列出最近 ' + MAX_CARDS + " 個賽馬日（共 " + allM.length +
        ' 個）；較早嘅可以喺<a href="#/meeting/' + meetingId(allM[MAX_CARDS]) + '">賽馬日頁</a>用選單揀。</p>';
    }
    drawHomePicks(document.getElementById("home-picks"), picksMeeting(idx));
    drawHomeScores(document.getElementById("home-scores"), shown);
    drawBoard();
  }

  // ------------------------------------------------------------------ 今日精選 (daily picks, from the analysis engine)
  // Meeting whose picks the home page shows: the upcoming meeting once its picks are out,
  // otherwise the latest meeting with picks (null = none yet).
  function picksMeeting(idx) {
    var ms = sortedMeetings(idx);
    var nm = idx.next_meeting;
    if (!ms.some(function (m) { return typeof m.has_picks === "boolean"; })) return null;   // older export
    if (nm) {
      var n = ms.filter(function (m) { return m.date === nm.date && m.venue === nm.venue && m.has_picks; })[0];
      if (n) return n;
    }
    return ms.filter(function (m) { return m.has_picks && (!nm || m.date <= nm.date); })[0] || null;
  }
  function drawHomePicks(el, pick) {
    var idx = state.index;
    var sub = document.getElementById("pk-sub");
    function placeholder() {
      if (sub) sub.textContent = "";
      el.innerHTML = '<div class="panel pad hs-empty"><div class="hs-empty-icon" aria-hidden="true">' + cloth("?", "") + "</div><div><b>未有今日精選</b>" +
        '<p class="muted small">' + nextScoresNote(idx).replace("評分", "評分同精選") + "模擬研究，非投注建議。</p></div></div>";
    }
    if (!pick) { placeholder(); return; }
    loadMeeting(meetingId(pick)).then(function (m) {
      if (!document.body.contains(el)) return;
      if (!m.picks) { placeholder(); return; }
      var nm = idx.next_meeting;
      var isNext = nm && m.date === nm.date && m.venue === nm.venue;
      if (sub) sub.textContent = isNext ? "下一個賽馬日" : "最近一個有精選嘅賽馬日";
      var html = picksCard(m, true);
      if (!isNext && nm) html += '<p class="small muted hs-note">下一個賽馬日（' + shortDate(nm.date) + " " + esc(nm.venue_zh || VENUE_ZH[nm.venue] || nm.venue) +
        "）嘅精選會喺第一場開跑前約 2 小時" + preRaceEta(nm) + "自動出。</p>";
      el.innerHTML = html;
    }).catch(function () {
      if (!document.body.contains(el)) return;
      el.innerHTML = '<div class="panel empty">暫時載入唔到精選。</div>';
    });
  }
  function pkRace(rn, ts) { return '<div class="pk-race"><b>第' + esc(rn) + "場</b><small>" + hhmm(ts) + "</small></div>"; }
  function pkOdds(v) {
    var t = oddsTxt(v);
    return t ? '<b class="pk-led">' + esc(t) + "</b>" : '<b class="pk-led off">未開售</b>';
  }
  function finTxt(res) {
    if (!res) return "";
    return mark(res.placed, "上名", "唔上名") + (ok(res.finish_pos) ? '<span class="pk-fin">第' + res.finish_pos + "名</span>" : "");
  }
  function moneyLine(stake, payout, profit, extra) {
    if (!ok(payout)) return '<span class="pk-meta">' + (extra || "") + '<span class="muted">等賽果</span></span>';
    return '<span class="pk-meta">' + (extra || "") + "派彩 <b class=num>" + money(payout, 2) + "</b> " + plSpan(profit) + "</span>";
  }
  // Dispatcher: 第三版 analysis-driven picks ({items:[...]}) vs the old fixed-set shape (pla_6x63 / qpl_2x3).
  function isOldPicks(p) { return !Array.isArray(p.items) && (p.pla_6x63 !== undefined || p.qpl_2x3 !== undefined); }
  function picksCard(m, home) {
    var p = m.picks;
    if (!p || typeof p !== "object") return "";
    return isOldPicks(p) ? picksCardV1(m, home) : picksCardV2(m, home);
  }

  // ---- 第三版：any pool / structure, single race, multi-race (孖寶/三寶) or all-up (過關)
  var POOL_FAM = { WIN: "win", PLA: "pla", QIN: "qin", QPL: "qin", FCT: "fct", TRI: "tri", TCE: "tri", FF: "f4", QTT: "f4", DBL: "multi", TBL: "multi" };
  var EXTRA_POOL_ZH = { DBL: "孖寶", TBL: "三寶" };
  function poolFam(it) {
    if (it.kind === "allup" || /過關/.test(String(it.pool_zh || ""))) return "allup";
    if (it.kind === "multi_race") return "multi";
    var code = String(it.pool || "").toUpperCase().split(/[^A-Z0-9]/)[0];
    return POOL_FAM[code] || "other";
  }
  function poolName(it) {
    if (it.pool_zh) return String(it.pool_zh);
    var code = String(it.pool || "").toUpperCase();
    return POOL_ZH[code] || EXTRA_POOL_ZH[code] || code || "玩法";
  }
  function probTxt(v) {
    if (!ok(v)) return DASH;
    if (v > 0 && v < 0.001) return "<0.1%";
    return (v * 100).toFixed(v < 0.1 ? 1 : 0) + "%";
  }
  function roiSpan(v) {
    if (!ok(v)) return '<b class="pk-roi muted">' + DASH + "</b>";
    var cls = v > 0.004 ? "up" : v < -0.004 ? "down" : "flat";
    return '<b class="pk-roi ' + cls + ' num">' + pct(v, 1, true) + "</b>";
  }
  function horseLookup(m) {
    var map = {};
    (m.races || []).forEach(function (r) {
      (r.horses || []).forEach(function (h) { map[r.race_no + ":" + h.horse_no] = h; });
    });
    return map;
  }
  function normHorse(x, rn, look, known) {   // number | string | {horse_no, horse_name, score}
    if (!ok(x)) return null;
    var o = typeof x === "object" ? x : { horse_no: x };
    if (!ok(o.horse_no)) return null;
    var k = known && known[o.horse_no], h = look[rn + ":" + o.horse_no] || {};
    return { horse_no: o.horse_no, horse_name: o.horse_name || (k && k.horse_name) || h.horse_name || "", score: ok(o.score) ? o.score : k && ok(k.score) ? k.score : h.score };
  }
  function pkHorse(h) {
    return '<span class="pk-hs" title="' + esc(h.horse_no + "號 " + (h.horse_name || "") + (ok(h.score) ? "（評分 " + Math.round(h.score) + "）" : "")) + '">' +
      cloth(h.horse_no, "sm") + (h.horse_name ? "<small>" + esc(h.horse_name) + "</small>" : "") + "</span>";
  }
  function pkGroup(lbl, cls, hs) {
    if (!hs.length) return "";
    return '<span class="pk-grp' + (cls ? " " + cls : "") + '">' + (lbl ? '<span class="pk-gl">' + esc(lbl) + "</span>" : "") + hs.map(pkHorse).join("") + "</span>";
  }
  function pkLeg(l, it, look, multi) {
    var rn = l.race_no;
    var known = {};
    (l.bankers || []).concat(l.selections || []).forEach(function (x) { if (x && typeof x === "object" && ok(x.horse_no)) known[x.horse_no] = x; });
    function list(a) { return (a || []).map(function (x) { return normHorse(x, rn, look, known); }).filter(Boolean); }
    var body = "";
    if (Array.isArray(l.positions) && l.positions.length) {   // 位置/次序：第1、第2、第3、第4 groups
      body = l.positions.map(function (grp, i) { return pkGroup("第" + (i + 1), "pos", list(Array.isArray(grp) ? grp : [grp])); }).join("");
    } else {
      var bk = list(l.bankers), sel = list(l.selections);
      body = pkGroup(bk.length ? "膽" : "", bk.length ? "banker" : "", bk) + pkGroup(bk.length ? "腳" : "", bk.length ? "leg" : "", sel);
    }
    if (!body) body = '<span class="muted small">未有選擇</span>';
    var head = "";
    if (multi) {
      var lp = l.pool_zh && l.pool_zh !== it.pool_zh ? ' <span class="pk-lg-pool">' + esc(l.pool_zh) + "</span>" : "";
      head = '<span class="pk-lg-r"><b>第' + esc(ok(rn) ? rn : "?") + "場</b>" + lp + mark(l.hit, "呢關中", "呢關唔中") + "</span>";
    }
    return '<div class="pk-lg' + (multi ? " multi" : "") + '">' + head + '<div class="pk-lg-b">' + body + "</div></div>";
  }
  function pkWhere(it, m) {
    var legs = it.legs || [];
    var races = (Array.isArray(it.races) && it.races.length ? it.races : legs.map(function (l) { return l.race_no; })).filter(ok);
    var multi = it.kind === "multi_race" || it.kind === "allup" || races.length > 1;
    var rank = ok(it.rank) ? '<span class="pk-rank" aria-label="第' + esc(it.rank) + '項">' + esc(it.rank) + "</span>" : "";
    if (!multi) {
      var rn = ok(it.race_no) ? it.race_no : races[0];
      return rank + "<b>第" + esc(ok(rn) ? rn : "?") + "場</b><small>" + hhmm(it.post_time) + "</small>";
    }
    var lbl = it.kind === "allup" ? "過關" : poolName(it);
    return rank + "<b>" + esc(lbl) + '</b><small class="pk-wr">' + races.map(function (r) { return "第" + esc(r) + "場"; }).join("<br>") + "</small>" +
      (it.post_time ? "<small>" + hhmm(it.post_time) + "</small>" : "");
  }
  function pkReason(txt) {
    txt = String(ok(txt) ? txt : "");
    if (!txt) return "";
    if (txt.length <= 70) return '<div class="pk-why">' + esc(txt) + "</div>";
    return '<details class="pk-why more"><summary>' + esc(txt.slice(0, 56)) + '…<span class="more-lbl">展開</span></summary><div>' + esc(txt) + "</div></details>";
  }
  function pkItem(it, m, look) {
    var fam = poolFam(it), ref = it.grade === "參考";
    var legs = it.legs || [];
    var multi = it.kind === "multi_race" || it.kind === "allup" || legs.length > 1;
    var res = it.result && typeof it.result === "object" ? it.result : null;
    var st = res ? (res.hit === true ? " is-hit" : res.hit === false ? " is-miss" : "") : "";
    var src = it.odds_source ? '<span class="pk-src' + (it.odds_source === "即時賠率" ? " live" : "") + '">' + esc(it.odds_source) + "</span>" : "";
    var html = '<li class="pk-it fam-' + fam + (ref ? " is-ref" : "") + st + '">';
    html += '<div class="pk-wh">' + pkWhere(it, m) + "</div>";
    html += '<div class="pk-mn"><div class="pk-it-h"><span class="pk-pool fam-' + fam + '">' + esc(poolName(it)) + "</span>" +
      (it.structure_zh ? '<span class="pk-struct">' + esc(it.structure_zh) + "</span>" : "") + src +
      (ref ? '<span class="pk-reftag">未過門檻，只供參考</span>' : "") + "</div>";
    html += '<div class="pk-sel">' + (legs.length ? legs.map(function (l) { return pkLeg(l, it, look, multi); }).join("") : '<span class="muted small">未有選擇</span>') + "</div></div>";
    var combos = ok(it.n_combos) ? it.n_combos + " 注 × " + money(it.unit_stake) : "";
    html += '<div class="pk-mo"><div class="pk-stake">' + (combos ? "<small>" + combos + " =</small>" : "") + '<b class="num">' + money(it.stake) + "</b></div>" +
      '<div class="pk-kv"><span>命中機率</span><b class="num">' + probTxt(it.hit_prob) + "</b></div>" +
      '<div class="pk-kv"><span>預期回報</span>' + roiSpan(it.exp_roi) + "</div>" +
      (ok(it.est_return_if_hit) && !res ? '<div class="pk-kv"><span>中咗約派</span><b class="num">' + money(it.est_return_if_hit, 0) + "</b></div>" : "");
    if (res && ok(res.payout)) {
      html += '<div class="pk-out">' + mark(res.hit, "中", "唔中") + "<span>派彩 <b class=num>" + money(res.payout) + "</b></span>" + plSpan(res.profit) + "</div>";
    } else if (res && res.hit !== undefined && res.hit !== null) {
      html += '<div class="pk-out">' + mark(res.hit, "中", "唔中") + "<span>" + (res.hit ? "中" : "唔中") + "</span></div>";
    } else {
      html += '<div class="pk-out muted small">' + (m.status === "results" ? "未結算" : "等賽果") + "</div>";
    }
    if (res && ok(res.detail) && typeof res.detail !== "object") html += '<div class="pk-detail small muted">' + esc(res.detail) + "</div>";
    html += "</div>";
    html += pkReason(it.reason) + "</li>";
    return html;
  }
  function pkSortKey(it) {
    var t = parseHK(it.post_time);
    return [t ? t.getTime() : Infinity, ok(it.race_no) ? +it.race_no : 99, ok(it.rank) ? +it.rank : 99];
  }
  function pkSort(a, b) {
    var x = pkSortKey(a), y = pkSortKey(b);
    for (var i = 0; i < x.length; i++) if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
  }
  function pkBanker(p) {
    var b = p.banker, v = Array.isArray(p.value) ? p.value : [];
    if (!b && !v.length) return "";
    var html = '<div class="pk-extra">';
    if (b) html += '<div class="pk-x"><span class="pk-x-tag gold">全日最穩</span><div class="pk-x-row">' + cloth(b.horse_no, "sm") +
      "<div><b>第" + esc(b.race_no) + "場 " + esc(b.horse_name) + "</b> " + finTxt(b.result) +
      (ok(b.p_place) ? ' <span class="chip">入三甲機率 ' + pct(b.p_place, 0) + "</span>" : "") + '<div class="small muted">' + esc(b.reason) + "</div></div></div></div>";
    if (v.length) html += '<div class="pk-x"><span class="pk-x-tag">值博之選</span>' + v.map(function (x) {
      return '<div class="pk-x-row">' + cloth(x.horse_no, "sm") + "<div><b>第" + esc(x.race_no) + "場 " + esc(x.horse_name) + "</b>" +
        (ok(x.ev) ? ' <span class="chip">EV ' + num(x.ev, 2) + "</span>" : "") + " " + finTxt(x.result) + '<div class="small muted">' + esc(x.reason) + "</div></div></div>";
    }).join("") + "</div>";
    return html + "</div>";
  }
  function picksCardV2(m, home) {
    var p = m.picks;
    var look = horseLookup(m);
    var all = Array.isArray(p.items) ? p.items.filter(function (x) { return x && typeof x === "object"; }) : [];
    var main = all.filter(function (x) { return x.grade !== "參考"; }).sort(pkSort);
    var extraRefs = [p.reference_items, p.ref_items, p.reference, p.references].filter(Array.isArray)[0] || [];
    var refs = all.filter(function (x) { return x.grade === "參考"; }).concat(extraRefs.filter(function (x) { return x && typeof x === "object"; }))
      .map(function (x) { return x.grade ? x : Object.assign({ grade: "參考" }, x); }).sort(pkSort);
    var nItems = ok(p.n_items) ? p.n_items : main.length;
    var settled = ok(p.total_payout);
    var dp = (state.index.bots || []).filter(function (b) { return b.id === "daily_picks"; })[0];
    var html = '<article class="picks pk2" aria-label="今日精選 ' + esc(m.date) + '">';
    html += '<header class="pk-head"><div class="pk-tag"><span>' + esc(p.title || "今日精選") + '</span></div><div class="pk-title"><b>' + meetingName(m) + "</b>" +
      "<small>" + (dp ? "「" + esc(dp.name) + "」機械人照呢度落注追蹤成績 · " : "") + "按模型分析揀玩法，唔固定配搭</small></div>" +
      '<div class="pk-total"><small>合共注碼</small><b class="num">' + money(ok(p.total_stake) ? p.total_stake : 0, 0) + "</b><small>（模擬）</small></div></header>";
    html += '<div class="pk-sum"><span class="pk-count"><b class="num">' + esc(nItems) + "</b> 項精選</span>" +
      (p.odds_note ? '<span class="pk-sum-i">' + esc(p.odds_note) + "</span>" : "") +
      (p.generated_at ? '<span class="pk-sum-i muted">生成 ' + esc(String(p.generated_at).replace("T", " ").slice(0, 16)) + "</span>" : "") + "</div>";
    if (p.method_note) html += reasonBlock(p.method_note, "pk-method small muted");
    if (settled) {
      html += '<div class="pk-final"><span>全日結算</span><span>派彩 <b class=num>' + money(p.total_payout, 2) + "</b></span>" + plSpan(p.total_profit) +
        (main.length ? '<span class="small muted">中 ' + main.filter(function (x) { return x.result && x.result.hit === true; }).length + " / " + main.length + " 項</span>" : "") + "</div>";
    }
    if (main.length) {
      html += '<ol class="pk-items">' + main.map(function (it) { return pkItem(it, m, look); }).join("") + "</ol>";
    } else {
      html += '<div class="pk-none"><b>今日冇玩法過門檻，冇精選</b>' + (p.note && !/冇精選/.test(String(p.note)) ? '<span class="small muted">' + esc(p.note) + "</span>" : "") + "</div>";
      if (refs.length) {
        html += '<details class="pk-refs"><summary>' + refs.length + ' 項參考（未過門檻，只供參考）<span class="more-lbl">展開</span></summary>' +
          '<ol class="pk-items">' + refs.map(function (it) { return pkItem(it, m, look); }).join("") + "</ol></details>";
      }
    }
    html += pkBanker(p);
    html += '<footer class="pk-foot"><b>合共 ' + money(ok(p.total_stake) ? p.total_stake : 0, 0) + "（模擬）</b>" +
      (settled ? "<span>派彩 " + money(p.total_payout, 2) + " · " + plSpan(p.total_profit) + "</span>" : "") +
      '<span class="pk-disc">模擬研究，非投注建議</span>' +
      (home ? '<a class="hs-link" href="#/meeting/' + meetingId(m) + '">完整賽馬日頁 →</a>' : "") + "</footer>";
    html += '<p class="pk-note small muted">' + (main.length && p.note ? esc(p.note) + " " : "") +
      "命中機率同預期回報由模型估計；預期回報 = 估計派彩 ÷ 注碼 − 1，正數先值博。膽 = 一定要跑入，腳 = 配搭。博彩要有節制。</p>";
    return html + "</article>";
  }

  // ---- old fixed-set picks (位置 6x63 + 位置Q 2x3), kept for older meeting files
  function picksCardV1(m, home) {
    var p = m.picks;
    var p6 = p.pla_6x63, q = p.qpl_2x3 || [];
    var html = '<article class="picks" aria-label="今日精選 ' + esc(m.date) + '">';
    html += '<header class="pk-head"><div class="pk-tag"><span>' + esc(p.title || "今日精選") + '</span></div><div class="pk-title"><b>' + meetingName(m) + "</b>" +
      "<small>每注 $10 · 同「" + esc(botInfo(p.bot_id || "fixed_combo").name) + "」機械人實際買嘅一樣</small></div>" +
      '<div class="pk-total"><small>合共</small><b class="num">' + money(p.total_stake, 0) + "</b><small>（模擬）</small></div></header>";
    if (ok(p.total_payout)) {
      html += '<div class="pk-final"><span>全日結算</span><span>派彩 <b class=num>' + money(p.total_payout, 2) + "</b></span>" + plSpan(p.total_profit) + "</div>";
    }
    // 位置 6x63
    html += '<section class="pk-sec"><div class="pk-sec-h"><h3>位置 6x63（每場 1 隻）</h3>';
    if (p6) {
      html += '<span class="pk-meta">' + (ok(p6.n_combos) ? p6.n_combos : 63) + " 注 × $10 = " + money(p6.stake, 0) + "</span>" +
        moneyLine(p6.stake, p6.payout, p6.profit, ok(p6.legs_hit) ? "中 <b>" + p6.legs_hit + "/" + (p6.legs || []).length + "</b> 關 · " : "") + "</div>";
      html += '<div class="pk-legs" role="list">' + (p6.legs || []).map(function (l) {
        var st = l.result ? (l.result.placed ? " is-hit" : l.result.placed === false ? " is-miss" : "") : "";
        return '<div class="pk-leg' + st + '" role="listitem">' + pkRace(l.race_no, l.post_time) + cloth(l.horse_no) +
          '<div class="pk-name"><b>' + esc(l.horse_name || DASH) + "</b><small>" + esc(l.jockey || DASH) + "</small></div>" +
          '<div class="pk-stats"><span><small>評分</small><b class="num">' + num(l.score) + "</b></span><span><small>入三甲機率</small><b class=num>" + pct(l.p_place, 0) +
          "</b></span><span><small>位置賠率</small>" + pkOdds(l.pla_odds) + "</span></div>" +
          '<div class="pk-res">' + finTxt(l.result) + "</div></div>";
      }).join("") + "</div>";
    } else {
      html += "</div>" + '<p class="small muted">今日固定組合冇買位置 6x63。</p>';
    }
    html += "</section>";
    // 位置Q 2x3 × 3
    html += '<section class="pk-sec"><div class="pk-sec-h"><h3>位置Q 2x3 × ' + (q.length || 3) + " 條（每場 3 隻）</h3>" +
      (q.length ? '<span class="pk-meta">每條 15 注 × $10 = $150</span>' : "") + "</div>";
    if (q.length) {
      html += '<div class="pk-qgrid">' + q.map(function (t) {
        return '<div class="pk-q"><div class="pk-q-h"><b>第' + esc(t.ticket_no) + "條</b>" + moneyLine(t.stake, t.payout, t.profit, ok(t.legs_hit) ? "中 " + t.legs_hit + "/" + (t.legs || []).length + " 關 · " : money(t.stake, 0) + " · ") + "</div>" +
          (t.legs || []).map(function (l) {
            var hit = l.hit_pairs;
            var hp = !ok(hit) ? "" : hit.length ? '<span class="pk-pair ok">' + mark(true, "中位置Q") + "中 " + hit.map(function (pr) { return pr.join("-"); }).join("、") + "</span>"
              : '<span class="pk-pair no">' + mark(false, "唔中") + "冇中</span>";
            return '<div class="pk-q-leg"><div class="pk-q-race"><b>第' + esc(l.race_no) + "場</b><small>" + hhmm(l.post_time) + "</small>" + hp + "</div>" +
              '<div class="pk-q-hs">' + (l.horses || []).map(function (h) {
                var st = h.placed === true ? " is-hit" : h.placed === false ? " is-miss" : "";
                return '<div class="pk-qh' + st + '">' + cloth(h.horse_no, "sm") + '<span class="pk-qn">' + esc(h.horse_name || DASH) + "</span>" +
                  '<span class="pk-qs num"><small>評分</small> ' + num(h.score) + '</span><span class="pk-qs num"><small>入三甲</small> ' + pct(h.p_place, 0) + "</span>" +
                  '<span class="pk-qf">' + (ok(h.finish_pos) ? mark(h.placed, "上名", "唔上名") + '<span class="pk-fin">第' + h.finish_pos + "名</span>" : "") + "</span></div>";
              }).join("") + "</div></div>";
          }).join("") + "</div>";
      }).join("") + "</div>";
    } else {
      html += '<p class="small muted">今日固定組合冇買位置Q 2x3。</p>';
    }
    html += "</section>";
    // 全日最穩 + 值博之選
    var b = p.banker, v = p.value || [];
    html += '<div class="pk-extra">';
    html += '<div class="pk-x"><span class="pk-x-tag gold">全日最穩</span>' + (b ? '<div class="pk-x-row">' + cloth(b.horse_no, "sm") +
      '<div><b>第' + esc(b.race_no) + "場 " + esc(b.horse_name) + "</b> " + finTxt(b.result) + '<div class="small muted">' + esc(b.reason) + "</div></div></div>" : '<div class="small muted">未有</div>') + "</div>";
    html += '<div class="pk-x"><span class="pk-x-tag">值博之選</span>' + (v.length ? v.map(function (x) {
      return '<div class="pk-x-row">' + cloth(x.horse_no, "sm") + '<div><b>第' + esc(x.race_no) + "場 " + esc(x.horse_name) + '</b> <span class="chip">EV ' + num(x.ev, 2) + "</span> " + finTxt(x.result) +
        '<div class="small muted">' + esc(x.reason) + "</div></div></div>";
    }).join("") : '<div class="small muted">' + esc(p.value_note || "冇") + "</div>") + "</div>";
    html += "</div>";
    html += '<footer class="pk-foot"><b>合共 ' + money(p.total_stake, 0) + "（模擬）</b>" +
      (ok(p.total_payout) ? "<span>派彩 " + money(p.total_payout, 2) + " · " + plSpan(p.total_profit) + "</span>" : "") +
      '<span class="pk-disc">模擬研究，非投注建議</span>' +
      (home ? '<a class="hs-link" href="#/meeting/' + meetingId(m) + '">完整賽馬日頁 →</a>' : "") + "</footer>";
    html += '<p class="pk-note small muted">' + esc(p.note || "") + "上名 = 跑入派位置彩嘅名次；入三甲機率 = 模型估計跑入頭三名嘅機率；EV = 入三甲機率 × 位置賠率。博彩要有節制。</p>";
    return html + "</article>";
  }

  // ------------------------------------------------------------------ home: latest horse scores
  function drawHomeScores(el, pick) {
    var idx = state.index;
    var sub = document.getElementById("hs-sub");
    function placeholder() {
      if (sub) sub.textContent = "";
      el.innerHTML = '<div class="panel pad hs-empty"><div class="hs-empty-icon" aria-hidden="true">' + cloth("?", "") + "</div><div><b>未有馬匹評分</b>" +
        '<p class="muted small">' + nextScoresNote(idx) + "到時呢度會自動顯示每場每匹馬嘅評分同理由。</p></div></div>";
    }
    if (!pick) { placeholder(); return; }
    var id = meetingId(pick);
    loadMeeting(id).then(function (m) {
      if (!document.body.contains(el)) return;
      var races = (m.races || []).filter(function (r) { return (r.horses || []).some(function (h) { return ok(h.score); }); });
      if (!races.length) { placeholder(); return; }
      var nm = idx.next_meeting;
      var isNext = nm && m.date === nm.date && m.venue === nm.venue;
      if (sub) sub.textContent = isNext ? "下一個賽馬日" : "最近一個有評分嘅賽馬日";
      var rn = state.homeRace[id];
      if (!races.some(function (r) { return r.race_no === rn; })) {
        rn = races[0].race_no;
        if (m.status === "pre") {   // first race not yet run
          var now = Date.now();
          var up = races.filter(function (r) { var t = parseHK(r.post_time); return t && t.getTime() > now; })[0];
          if (up) rn = up.race_no;
        }
      }
      var html = '<div class="hs-head"><div><b class="hs-title">' + meetingName(m) + '</b> <span class="chip status-' + esc(m.status) + '">' +
        esc(STATUS_ZH[m.status] || m.status) + "</span>" +
        '<div class="muted small">' + races.length + " 場有評分 · 評分生成時間 " + esc(m.generated_at ? String(m.generated_at).replace("T", " ").slice(0, 16) : DASH) +
        "（香港時間）</div></div>" +
        '<a class="hs-link" id="hs-link" href="#/meeting/' + id + "/" + rn + '">完整賽馬日頁 →</a></div>';
      if (!isNext && nm) html += '<p class="small muted hs-note">' + nextScoresNote(idx) + "到時呢度會自動換成新賽馬日。</p>";
      html += '<div class="hs-tabs" role="tablist" aria-label="場次">' + races.map(function (r) {
        return '<button type="button" role="tab" id="hs-tab-' + r.race_no + '" data-rn="' + r.race_no + '" aria-controls="hs-race" aria-selected="' + (r.race_no === rn) + '"' +
          (r.race_no === rn ? "" : ' tabindex="-1"') + ">" + r.race_no + "<small>" + hhmm(r.post_time) + "</small></button>";
      }).join("") + "</div>";
      html += '<div id="hs-race" role="tabpanel" aria-labelledby="hs-tab-' + rn + '"></div>';
      html += '<p class="small muted">評分 0–100 由模型勝出機率計出，50 分約等於平均水平；只供研究參考，並非投注建議。' +
        '子評分、勝率、賠率同 ' + (idx.bots || []).length + ' 隻機械人嘅決定見<a href="#/meeting/' + id + '">賽馬日頁</a>。</p>';
      el.innerHTML = html;
      function show(no, focus) {
        state.homeRace[id] = no;
        var r = races.filter(function (x) { return x.race_no === no; })[0];
        el.querySelectorAll(".hs-tabs [role=tab]").forEach(function (b) {
          var on = +b.getAttribute("data-rn") === no;
          b.setAttribute("aria-selected", String(on));
          if (on) b.removeAttribute("tabindex"); else b.setAttribute("tabindex", "-1");
          if (on && focus) b.focus();
          if (on && b.scrollIntoView && focus !== undefined) b.scrollIntoView({ block: "nearest", inline: "nearest" });
        });
        var panel = document.getElementById("hs-race");
        panel.setAttribute("aria-labelledby", "hs-tab-" + no);
        panel.innerHTML = homeRaceBlock(r);
        var link = document.getElementById("hs-link");
        if (link) link.setAttribute("href", "#/meeting/" + id + "/" + no);
      }
      el.querySelectorAll(".hs-tabs [role=tab]").forEach(function (b, i, all) {
        b.addEventListener("click", function () { show(+b.getAttribute("data-rn"), false); });
        b.addEventListener("keydown", function (e) {
          var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : null;
          if (j === null) return;
          e.preventDefault();
          j = (j + all.length) % all.length;
          show(+all[j].getAttribute("data-rn"), true);
        });
      });
      show(rn);
    }).catch(function () {
      if (!document.body.contains(el)) return;
      el.innerHTML = '<div class="panel empty">暫時載入唔到評分。<a href="#/meeting/' + id + '">去賽馬日頁</a></div>';
    });
  }

  function homeRaceBlock(r) {
    var info = [r.race_class, ok(r.distance) ? r.distance + "米" : null, SURFACE_ZH[r.surface] || r.surface,
      r.course ? "「" + r.course + "」賽道" : null, r.going].filter(Boolean).map(esc).join(" · ");
    var horses = (r.horses || []).slice().sort(function (a, b) {
      var sa = !a.scratched && ok(a.score), sb = !b.scratched && ok(b.score);
      if (sa !== sb) return sa ? -1 : 1;
      if (a.scratched !== b.scratched) return a.scratched ? 1 : -1;
      if (sa) return (ok(a.rank) ? a.rank : 99) - (ok(b.rank) ? b.rank : 99) || b.score - a.score || a.horse_no - b.horse_no;
      return a.horse_no - b.horse_no;
    });
    var html = '<div class="race-bar compact"><div class="rb-no">第 ' + r.race_no + ' 場</div><div class="rb-main"><b>' + esc(r.race_name || "") + "</b><span>" + info + "</span></div>" +
      '<div class="rb-time">' + hhmm(r.post_time) + "</div></div>";
    html += '<div class="srows" role="list">' + horses.map(function (h) {
      var has = !h.scratched && ok(h.score);
      var lit = has ? Math.round(h.score / 5) : 0;
      var energy = "";
      for (var i = 0; i < 20; i++) energy += "<i" + (i < lit ? ' class="on"' : "") + "></i>";
      var fin = "";
      if (h.result && ok(h.result.finish_pos)) fin = '<span class="fin-tag' + (h.result.finish_pos <= 3 ? " p" + h.result.finish_pos : "") + '">第' + h.result.finish_pos + "名</span>";
      var reasons = has ? (h.reasons || []).map(function (s) { return '<span class="reason">' + esc(s) + "</span>"; }).join("") : "";
      var rk = has && ok(h.rank) ? h.rank : null;
      return '<div class="srow' + (h.scratched ? " scratched" : "") + (rk === 1 ? " top1" : "") + '" role="listitem">' +
        '<div class="s-rank' + (rk && rk <= 3 ? " r" + rk : "") + '" title="評分排名">' + (rk ? '<span class="sr-only">評分第</span>' + rk + '<span class="sr-only">位</span>' : DASH) + "</div>" +
        '<div class="s-cloth">' + cloth(h.horse_no) + "</div>" +
        '<div class="s-name"><div class="hn">' + esc(h.horse_name || DASH) + (h.scratched ? ' <span class="chip">退出</span>' : "") + fin + "</div>" +
        '<div class="jt">' + esc(h.jockey || DASH) + " / " + esc(h.trainer || DASH) + ' <span class="dw">· 檔位 ' + num(h.draw) + "</span></div></div>" +
        (has ? '<div class="s-score"><span class="score-num">' + Math.round(h.score) + '</span><div class="energy" role="img" aria-label="評分 ' + Math.round(h.score) + ' / 100">' + energy + "</div></div>"
          : '<div class="s-score muted small">' + (h.scratched ? "退出，冇評分" : "未有評分") + "</div>") +
        (reasons ? '<div class="s-reasons">' + reasons + "</div>" : "") + "</div>";
    }).join("") + "</div>";
    return html;
  }

  function boardRows() {
    var bots = (state.index.bots || []).slice();
    var rows = bots.map(function (b) {
      var v = equity(b);
      var gain = ok(v) && b.start_bankroll ? v / b.start_bankroll - 1 : null;
      return { bot: b, v: ok(v) ? v : null, sub: money(v) + "（" + pct(gain, 1, true) + "）" };
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
    var el = document.getElementById("track");
    if (!el) return;
    var rows = boardRows();
    var vals = rows.filter(function (r) { return r.v !== null; }).map(function (r) { return r.v; });
    if (!vals.length) {
      el.innerHTML = '<div class="panel empty">機械人未有實盤紀錄。</div>';
    } else {
      var start = state.index.start_bankroll || 10000;
      var lo = Math.min.apply(null, vals.concat([start])), hi = Math.max.apply(null, vals.concat([start]));
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
      html += '<div class="track-note">跑道位置 = 現時模擬資金（起步 ' + money(start) + "，未結算注碼按本金計）；一樣資金時並排。</div>";
      el.innerHTML = html;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          el.querySelectorAll(".runner").forEach(function (r) { r.style.left = r.getAttribute("data-x") + "%"; });
        });
      });
    }
    var t = document.getElementById("lb-table");
    if (!t) return;
    var head = "<th>#</th><th>機械人</th><th class=r>資金</th><th class=r>盈虧</th><th class=r>ROI</th><th class=r>命中率</th><th class=r>賽馬日</th>";
    var body = rows.map(function (r, i) {
      var lv = r.bot.live || {};
      return "<tr><td>" + (i + 1) + "</td><td>" + botCell(r.bot) + "</td><td class='r num'><b>" + money(equity(r.bot)) +
        "</b></td><td class=r>" + plSpan(lv.profit) + "</td><td class='r num'>" + pct(lv.roi, 1, true) + "</td><td class='r num'>" + pct(lv.hit_rate, 0) +
        "</td><td class='r num'>" + (ok(lv.n_meetings) ? lv.n_meetings : DASH) + "</td></tr>";
    }).join("");
    t.innerHTML = '<div class="panel scroll-x"><table class="tbl"><caption>實盤（模擬投注，已結算注項）</caption><thead><tr>' + head +
      "</tr></thead><tbody>" + body + "</tbody></table></div>";
  }
  function botCell(b) {
    return '<a href="#/bot/' + esc(b.id) + '" style="display:inline-flex;align-items:center;gap:8px;text-decoration:none">' + botToken(b, "sm") + "<b>" + esc(b.name) + "</b></a>";
  }

  // ------------------------------------------------------------------ curves (live paper results)
  function curveSeries(onlyBot) {
    var hist = state.history || { bots: [] };
    var start = hist.start_bankroll || (state.index && state.index.start_bankroll) || 10000;
    var series = [];
    (hist.bots || []).forEach(function (h) {
      if (onlyBot && h.bot_id !== onlyBot) return;
      var pts = (h.live || []).filter(function (p) { return ok(p.bankroll) && p.date; });
      if (!pts.length) return;
      series.push({ id: h.bot_id, bot: botInfo(h.bot_id), points: pts });
    });
    // x domain = union of dates (ordinal), with a start point before the first meeting
    var dates = {};
    series.forEach(function (s) { s.points.forEach(function (p) { dates[p.date] = 1; }); });
    var xs = Object.keys(dates).sort();
    var pos = {};
    xs.forEach(function (d, i) { pos[d] = i + 1; });
    series.forEach(function (s) {
      s.xy = [{ x: 0, y: start, date: null }].concat(s.points.map(function (p) { return { x: pos[p.date], y: p.bankroll, date: p.date, venue: p.venue }; }));
    });
    return { series: series, xs: [null].concat(xs), start: start };
  }

  function drawCurves() {
    var el = document.getElementById("curve");
    if (!el) return;
    var cs = curveSeries();
    if (!cs.series.length) {
      el.innerHTML = '<div class="empty">未有已結算嘅賽馬日，資金曲線會喺第一個賽馬日結算後出現。</div>';
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
    if (m.picks) html += '<section class="section pk-section" aria-label="今日精選">' + picksCard(m, false) + "</section>";

    html += '<nav class="race-tabs" aria-label="場次">' + races.map(function (r) {
      return '<a href="#/meeting/' + id + "/" + r.race_no + '"' + (r.race_no === rn ? ' aria-current="true"' : "") + ">" + r.race_no + "<small>" + hhmm(r.post_time) + "</small></a>";
    }).join("") + "</nav>";
    html += '<div id="race"></div>';
    html += robotsDay(m);
    app.innerHTML = html;
    document.getElementById("mtg-sel").addEventListener("change", function (e) { location.hash = "#/meeting/" + e.target.value; });
    bindBotJumps(app);
    var race = races.filter(function (r) { return r.race_no === rn; })[0];
    if (race) renderRace(document.getElementById("race"), race, m);
    else document.getElementById("race").innerHTML = '<div class="panel empty">排位表未公佈。</div>';
    if (raceNo) { var cur = app.querySelector('.race-tabs [aria-current="true"]'); if (cur && cur.scrollIntoView) cur.scrollIntoView({ block: "nearest", inline: "center" }); }
  }

  // ------------------------------------------------------------------ meeting: 機械人當日落單 (grouped by robot)
  function structZh(s) {
    var parts = String(s || "").split(" ");
    var p = POOL_ZH[parts[0]];
    if (!p) return String(s || "");
    var rest = parts.slice(1).join(" ");
    if (rest === p) rest = "";
    return p + (rest ? " " + rest : "");
  }
  function mark(v, yes, no) {
    if (v === true) return '<span class="mk ok" title="' + (yes || "中") + '"><span aria-hidden="true">✓</span><span class="sr-only">' + (yes || "中") + "</span></span>";
    if (v === false) return '<span class="mk no" title="' + (no || "唔中") + '"><span aria-hidden="true">✗</span><span class="sr-only">' + (no || "唔中") + "</span></span>";
    return "";
  }
  function reasonBlock(txt, cls) {
    txt = String(txt || "");
    if (!txt) return "";
    if (txt.length <= 38) return '<div class="' + cls + '">' + esc(txt) + "</div>";
    return '<details class="' + cls + ' more"><summary>' + esc(txt.slice(0, 30)) + "…<span class=\"more-lbl\">展開</span></summary><div>" + esc(txt) + "</div></details>";
  }
  function ticketWhere(t) {
    var legs = t.legs || [];
    if (legs.length > 1 || t.race_no === 0) return '<b>過關</b><small>' + legs.length + " 關</small>";
    var rn = legs.length ? legs[0].race_no : t.race_no;
    return "<b>第" + esc(rn) + "場</b>";
  }
  function ticketSel(t) {
    var legs = t.legs || [];
    if (legs.length > 1) {
      return legs.map(function (l, i) {
        var hits = t.leg_hits || [];
        return '<span class="tk-leg"><span class="leg-lbl">第' + esc(l.race_no) + "場</span>" +
          selHtml(l.bankers ? { bankers: l.bankers, legs: l.selections } : (l.selections || [])) + mark(hits[i], "呢關中", "呢關唔中") + "</span>";
      }).join("");
    }
    var sel = t.selections;
    if (!ok(sel) || (Array.isArray(sel) && !sel.length)) sel = legs.length ? (legs[0].bankers ? { bankers: legs[0].bankers, legs: legs[0].selections } : legs[0].selections) : [];
    return selHtml(sel);
  }
  function ticketMoney(t) {
    var out = '<span class="tk-stake num">' + money(t.total_stake) + "</span>";
    if (t.status === "settled") out += '<span class="tk-pay">' + mark(t.hit, "有派彩", "冇派彩") + "派彩 <b class=num>" + money(t.payout) + "</b></span>" + plSpan(t.profit);
    else out += '<span class="muted small">等賽果</span>';
    return out;
  }
  function robotCard(info, b, m) {
    var c = botColor(info);
    var tks = (b && b.tickets) || [];
    var skips = (b && b.skips) || [];
    var settled = b && b.settled && m.status === "results";
    var html = '<article class="rb-card" id="rb-' + esc(info.id) + '" style="--c:' + c[0] + '" tabindex="-1">';
    html += '<header class="rb-head"><a class="rb-who" href="#/bot/' + esc(info.id) + '">' + botToken(info, "sm") + "<span><b>" + esc(info.name) + "</b><small>" + esc(info.style) + "</small></span></a>";
    if (b) {
      html += '<div class="rb-tot"><span>注數 <b class="num">' + tks.length + "</b></span><span>注碼 <b class=num>" + money(b.stake || 0) + "</b></span>" +
        (tks.length ? (ok(b.payout) ? "<span>派彩 <b class=num>" + money(b.payout) + "</b></span><span>" + plSpan(b.profit) + "</span>" : '<span class="muted">等賽果</span>') : "") + "</div>";
    }
    html += "</header>";
    if (!b) return html + '<div class="rb-empty muted small">呢個賽馬日未有決定。</div></article>';
    if (tks.length) {
      html += '<ul class="rb-tickets">' + tks.map(function (t) {
        var st = t.status === "settled" ? (t.hit ? " is-hit" : " is-miss") : "";
        return '<li class="tk' + st + '"><div class="tk-where">' + ticketWhere(t) + "</div>" +
          '<div class="tk-main"><div class="tk-struct">' + esc(structZh(t.structure || t.strategy)) +
          (ok(t.n_combos) && t.n_combos > 1 ? ' <small class="muted">' + t.n_combos + " 注 × " + money(t.unit_stake) + "</small>" : "") + "</div>" +
          '<div class="tk-sel">' + ticketSel(t) + "</div></div>" +
          '<div class="tk-money">' + ticketMoney(t) + "</div>" + reasonBlock(t.reason, "tk-why") + "</li>";
      }).join("") + "</ul>";
    } else {
      html += '<div class="rb-empty small">今日一張單都冇落。</div>';
    }
    if (skips.length) {
      var nr = skips.filter(function (x) { return x.race_no > 0; }).length, n0 = skips.length - nr;
      var lbl = [nr ? nr + " 場唔買" : null, n0 ? "過關決定 " + n0 + " 項唔買" : null].filter(Boolean).join(" · ");
      html += '<details class="rb-skips"><summary>' + lbl + '<span class="more-lbl">睇原因</span></summary><ul>' + skips.map(function (x) {
        return "<li><b>" + (x.race_no > 0 ? "第" + x.race_no + "場" : "過關決定") + "</b>" + (x.structure && x.structure !== "過關腳" && x.structure !== "自動" ? '<span class="muted small"> ' + esc(structZh(x.structure)) + "</span>" : "") +
          "：" + esc(x.reason || "冇寫原因") + "</li>";
      }).join("") + "</ul></details>";
    }
    if (settled === false && m.status === "results" && tks.some(function (t) { return t.status !== "settled"; })) html += '<div class="small muted">部分注項未結算。</div>';
    return html + "</article>";
  }
  function robotsDay(m) {
    var bots = m.bots || [];
    var hasAny = bots.length > 0;
    var by = {};
    bots.forEach(function (b) { by[b.bot_id] = b; });
    var list = (state.index.bots || []).slice().sort(function (a, b) { return a.no - b.no; });
    bots.forEach(function (b) { if (!list.some(function (x) { return x.id === b.bot_id; })) list.push(botInfo(b.bot_id)); });
    var settled = m.status === "results";
    var html = '<section class="section" aria-labelledby="rb-h"><div class="section-head"><span class="kicker">ROBOTS</span><h2 id="rb-h">機械人當日落單</h2>' +
      '<span class="muted small">每隻機械人成日所有注項（單場同過關）擺埋一齊' + (settled ? "，已按賽果結算" : "") + "</span></div>";
    if (!hasAny) return html + '<div class="panel empty">未有機械人決定（通常喺第一場開跑前約 2 小時出）。</div></section>';
    var tot = bots.reduce(function (a, b) { a.s += b.stake || 0; a.n += (b.tickets || []).length; return a; }, { s: 0, n: 0 });
    html += '<p class="small muted rb-sum">' + bots.length + ' 隻機械人合共 ' + tot.n + " 張單、注碼 " + money(tot.s) + "（模擬，每隻獨立資金）。模擬研究，非投注建議。</p>";
    html += '<div class="rb-grid">' + list.map(function (info) { return robotCard(info, by[info.id], m); }).join("") + "</div></section>";
    return html;
  }
  function raceBetChips(r, m) {
    var chips = [];
    (m.bots || []).forEach(function (b) {
      var names = [];
      (b.tickets || []).forEach(function (t) {
        var legs = t.legs || [];
        if (!legs.some(function (l) { return l.race_no === r.race_no; })) return;
        var n = legs.length > 1 ? "過關 " + structZh(t.structure || t.strategy).replace(/ #\d+$/, "") : structZh(t.structure || t.strategy);
        if (names.indexOf(n) < 0) names.push(n);
      });
      if (names.length) {
        var info = botInfo(b.bot_id);
        chips.push('<button type="button" class="bet-chip" data-bot="' + esc(b.bot_id) + '">' + botToken(info, "xs") + "<span>" + esc(info.name) + "</span><small>" + esc(names.join("、")) + "</small></button>");
      }
    });
    if (!chips.length) return "";
    return '<div class="bet-chips" aria-label="本場有落注嘅機械人"><span class="small muted">本場有落注：</span>' + chips.join("") + "</div>";
  }
  function bindBotJumps(root) {
    root.querySelectorAll(".bet-chip[data-bot]").forEach(function (b) {
      b.addEventListener("click", function () {
        var card = document.getElementById("rb-" + b.getAttribute("data-bot"));
        if (!card) return;
        card.scrollIntoView({ block: "start" });
        try { card.focus({ preventScroll: true }); } catch (e) { /* ignore */ }
        card.classList.remove("flash"); void card.offsetWidth; card.classList.add("flash");
      });
    });
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
    html += raceBetChips(r, m);
    if (r.result) html += finishGraphic(r);
    html += '<div class="section-head section"><h2>排位表 · 評分</h2><span class="spacer"></span><div class="seg" role="group" aria-label="排序">' +
      '<button type="button" data-sort="no" aria-pressed="' + (sort === "no") + '">按馬號</button>' +
      (hasScore ? '<button type="button" data-sort="score" aria-pressed="' + (sort === "score") + '">按評分</button>' : "") +
      (r.result ? '<button type="button" data-sort="finish" aria-pressed="' + (sort === "finish") + '">按名次</button>' : "") + "</div></div>";
    html += '<div class="card-head" aria-hidden="true"><span>馬號</span><span>馬名 / 騎師 / 練馬師</span><span>評分</span><span>子評分</span><span>勝率 / 入三甲機率</span><span>賠率</span></div>';
    html += '<div class="rows">' + horses.map(function (h) { return horseRow(h, r); }).join("") + "</div>";
    if (!hasScore) html += '<p class="small muted">呢場未有評分（模型未計／排位表剛公佈）。</p>';
    if (r.result && r.result.dividends) html += dividendsBlock(r.result.dividends);
    el.innerHTML = html;
    bindBotJumps(el);
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
      '<div class="c-prob"><div>勝 <b>' + pct(h.p_win) + "</b></div><div>入三甲 <b>" + pct(h.p_place) + "</b></div></div>") +
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

  // ------------------------------------------------------------------ robots
  function pageBots() {
    var idx = state.index;
    var bots = (idx.bots || []).slice().sort(function (a, b) { return a.no - b.no; });
    var html = demoBanner(idx);
    // ① 機械人賽道排行榜 → ② 實盤排行表 → ③ 資金曲線 → ④ 機械人卡片
    html += '<section class="section top-section" aria-labelledby="lb-h"><div class="section-head"><span class="kicker">LIVE</span><h1 id="lb-h">機械人賽道排行榜</h1>' +
      '<span class="muted small">實盤（模擬投注）</span></div><div id="track"></div><div id="lb-table" class="section"></div></section>';
    html += '<section class="section" aria-labelledby="curve-h"><div class="section-head"><h2 id="curve-h">資金曲線</h2><span class="muted small">實盤：每個已結算賽馬日後嘅模擬資金</span></div>' +
      '<div class="panel pad"><div id="curve"><div class="loading">載入資金曲線…</div></div></div></section>';
    html += '<section class="section" aria-labelledby="bots-h"><div class="section-head"><span class="kicker">ROBOTS</span><h2 id="bots-h">' + bots.length + ' 隻模擬投注機械人</h2></div>' +
      '<p class="muted" style="margin-top:0">每隻機械人用同一套模型預測，但玩法同落注門檻唔同；各自有獨立 ' + money(idx.start_bankroll || 10000) + " 模擬資金。「跟大熱」同「隨機猴子」係對照組。</p>";
    html += '<div class="bot-cards">' + bots.map(function (b) {
      var lv = b.live || {};
      return '<a class="bot-card" href="#/bot/' + esc(b.id) + '" style="--c:' + botColor(b)[0] + '"><div class="bc-top">' + botToken(b, "lg") +
        '<div><b style="font-size:17px">' + esc(b.name) + '</b><div class="small muted">' + esc(b.style) + "</div></div></div>" +
        '<div class="small" style="color:var(--text-2)">' + esc(b.description) + "</div>" +
        '<div class="mini-stats"><span>資金 <b>' + money(equity(b)) + "</b></span><span>盈虧 " + plSpan(lv.profit) + "</span><span>ROI <b>" +
        pct(lv.roi, 1, true) + "</b></span></div></a>";
    }).join("") + "</div></section>";
    app.innerHTML = html;
    drawBoard();
    loadHistory().then(drawCurves);
    focusMain();
  }

  function pageBot(id) {
    var b = botInfo(id);
    if (!b || !b.no) { app.innerHTML = '<div class="panel empty">搵唔到呢隻機械人。<a href="#/bots">返機械人列表</a></div>'; return; }
    var lv = b.live || {};
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
      "</div>";
    html += '<section class="section"><div class="section-head"><h2>資金曲線</h2><span class="muted small">實盤：每個已結算賽馬日後嘅模擬資金</span></div>' +
      '<div class="panel pad"><div id="curve1"></div></div></section>';
    html += '<section class="section"><div class="section-head"><h2>全部注項</h2><span class="muted small" id="tk-sub"></span></div><div id="tickets"><div class="loading">載入中…</div></div></section>';
    app.innerHTML = html;
    document.getElementById("bot-sel").addEventListener("change", function (e) { location.hash = "#/bot/" + e.target.value; });
    function curve() {
      var el = document.getElementById("curve1");
      if (!el) return;
      var cs = curveSeries(id);
      if (!cs.series.length) { el.innerHTML = '<div class="empty">未有已結算嘅賽馬日。</div>'; return; }
      lineChart(el, cs, [id], function () {}, true);
    }
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
      form: "近期名次、頭馬距離、速度同表現評級", jockey: "騎師近一年同近期勝出／入三甲比率（按場地、跑道、路程）",
      trainer: "練馬師近一年同近期勝出／入三甲比率", combo: "呢位騎師同練馬師合作嘅往績", draw: "檔位喺呢個路程／跑道嘅歷史優劣",
      fit: "匹馬喺呢個路程、場地、跑道嘅往績", "class": "班次升降、評分、負磅相對同場", market: "即時賠率反映嘅市場看法（未開售時留空）"
    };
    var html = '<div class="prose"><div class="section-head"><span class="kicker">GUIDE</span><h1>說明</h1></div>';
    html += "<h2>評分代表咩？</h2><p>" + esc(idx.score_explanation || "") + "</p>";
    html += '<h2>子評分</h2><div class="sub-explain">' + SUB_KEYS.map(function (k) {
      return "<div><b>" + esc(labels[k] || SUB_SHORT[k]) + "</b><br>" + esc(SUB_DESC[k]) + "</div>";
    }).join("") + '</div><p class="small">子評分係同場馬匹之間嘅相對分數（0 = 同場最差，100 = 同場最好），只用嚟解釋綜合評分從何而來。</p>';
    html += "<h2>方法</h2><ul>" +
      "<li>數據：公開嘅歷屆賽果、派彩、排位表，同賽前公開賠率（開售後）。</li>" +
      "<li>模型：按每場比賽估計每匹馬嘅勝出率（勝）同入三甲機率（入三甲）；每場勝出率加埋等於 100%。</li>" +
      "<li>防止偷看未來：所有特徵只用該場之前嘅資料，模型每次只用預測日之前嘅數據訓練。</li>" +
      "<li>賽前約 2 小時攞賠率、計評分，" + (idx.bots || []).length + " 隻機械人各自決定買唔買、買咩，並記錄原因；賽後用正式派彩結算。</li>" +
      "<li>彩池抽水大約 17.5%–25%，過關每一關都要扣；長遠嚟講大部分玩法都係蝕錢。</li></ul>";
    html += "<h2>" + (idx.bots || []).length + " 隻機械人</h2><ul>" + (idx.bots || []).map(function (b) {
      return '<li><a href="#/bot/' + esc(b.id) + '">' + b.no + " 號 " + esc(b.name) + "</a>（" + esc(b.style) + "）：" + esc(b.description) + "</li>";
    }).join("") + "</ul>";
    html += "<h2>免責聲明</h2><ul>" +
      "<li><b>" + esc(idx.disclaimer || "") + "</b></li>" +
      "<li>呢度所有投注都係電腦模擬（paper trade），從來冇落過任何真注，亦唔會代任何人落注。</li>" +
      "<li>內容只供研究同學習，唔構成任何投資或投注建議；過往表現唔代表將來。</li>" +
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
