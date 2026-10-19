# HKJC 賽馬分析 + Paper Trade 系統 — 共用規格 (所有代理必須遵守)

Python: `./.venv/bin/python` (3.12). 已裝: pandas numpy requests beautifulsoup4 lxml scikit-learn pytest tabulate.
**不可用 LightGBM**(缺 libomp)，用 `sklearn.ensemble.HistGradientBoostingClassifier` / LogisticRegression。
不要另裝套件，如真係需要先寫低原因。所有代碼放 `hkjc/` package，測試放 `tests/`，用 `pytest` 跑。
只做 paper trade（模擬），絕對唔落真注、唔登入任何戶口。

## 目標
1. 長期 SQLite 資料庫 `data/hkjc.db`，儲存賽果、排位表、賠率快照、預測、模擬投注。
2. 每個賽馬日後分析；下一個同類賽馬日（跑馬地夜馬 HV / 沙田日馬 ST）第一場開跑前 2 小時，攞即時賠率計算並出投注參考。
3. 固定投注組合（paper trade 機械人每個賽馬日落）：
   - **1 條 位置 6x63 過關**：揀 6 場，每場 1 隻，6 關所有組合 (C(6,1)+…+C(6,6)=63 注)。
   - **3 條 位置Q 2x3 過關**：每條 2 場（3 條共 6 場，場次不重複），每場 3 隻 → 每場 3 個位置Q組合；2x3 = 2 個單關 + 1 個二關。每條注數 = 3 + 3 + 3×3 = 15。
   - 每注單位注碼預設 $10（可設定）。派彩以每 $10 計（馬會派彩格式）。
4. 用近兩年數據回測：邊種玩法/結構勝率同回報較高；場次揀法（信心最高 vs 固定場次等）；膽拖 vs 複式；幾多膽拖幾多腳。

## 數據來源（已驗證可用）
- 賽果：`https://racing.hkjc.com/zh-hk/local/information/localresults?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`
  （舊 URL `racing/information/Chinese/Racing/LocalResults.aspx` 會 302 去呢度）。頁內有：班次、路程、場地狀況、賽道(草地/全天候 + "A"/"C+3"等)、獎金、全場馬匹表（名次、馬號、馬名+烙號、騎師id、練馬師id、實際負磅、排位體重、檔位、頭馬距離、沿途走位、完成時間、獨贏賠率）、派彩表（獨贏/位置/連贏/位置Q/二重彩/三重彩/單T/四連環/四重彩/孖寶等）、競賽事件報告。
  樣本: `tests/fixtures/results_20261014_HV_R1.html`。不帶 racedate 會顯示最新賽日；下拉選單只列到 2025/07 起嘅日期，舊日期要自行探測。
- 排位表：`https://racing.hkjc.com/zh-hk/local/information/racecard?racedate=YYYY/MM/DD&Racecourse=HV&RaceNo=N`
  樣本: `tests/fixtures/racecard_20261021_HV_R1.html`（2026-10-21 跑馬地，第1場 18:40）。
- 即時賠率 GraphQL：POST `https://info.cld.hkjc.com/graphql/base/`，headers `Content-Type: application/json`, `Origin: https://bet.hkjc.com`, `Referer: https://bet.hkjc.com/`，curl 要 `--compressed`（回應 gzip）。**查詢字串有白名單**，要用 `tests/fixtures/odds_query.json` 入面完全一樣嘅 query 文字，只改 variables（date `YYYY-MM-DD`, venueCode, raceNo, oddsTypes 例如 ["WIN","PLA","QPL","QIN"]）。賠率通常賽前一日先開售，未開時 oddsNodes 係空。
- 禮貌抓取：每個請求之間 ≥0.4 秒，最多 3 個並行；User-Agent 用正常瀏覽器字串；原始 HTML gzip 快取於 `data/raw/`（重跑唔使再抓）。

## 代碼 ID 規範
- `race_id` = `YYYYMMDD-VV-NN`，例如 `20261014-HV-01`。
- 日期一律 `YYYY-MM-DD` 字串；時間 `YYYY-MM-DDTHH:MM` 香港時間 (Asia/Hong_Kong)。
- venue: `ST` 沙田, `HV` 跑馬地。surface: `TURF` 草地, `AWT` 全天候跑道。
- 彩池代碼: 獨贏 WIN, 位置 PLA, 連贏 QIN, 位置Q QPL, 二重彩 FCT, 三重彩 TCE, 單T TRI, 四連環 FF, 四重彩 QTT, 孖寶 DBL, 三寶 TBL, 六環彩 SIXUP, 孖T DT, 三T TT。組合字串: 不分次序嘅池(QIN/QPL/TRI/FF)馬號升序以逗號分隔；有次序嘅池(FCT/TCE/QTT)按名次次序；跨場池(DBL等)用 `/`。

## 資料庫
Schema 定義喺 `hkjc/db.py`（`connect()`, `init_db()`, `upsert()`）。唔好改現有欄位意義；如要加欄位/表，喺 db.py 加並寫低。

## 模組分工及介面
- `hkjc/parse.py` 純解析（HTML → dict），`hkjc/scrape.py` 抓取+寫庫+回補：
  - `parse_results(html) -> dict(race=..., runners=[...], dividends=[...], meeting_race_nos=[...])`
  - `parse_racecard(html) -> dict(race=..., runners=[...])`
  - `scrape_meeting_results(conn, date, venue=None) -> int`（寫入 meetings/races/runners/dividends，回傳場數）
  - `scrape_racecard(conn, date, venue=None) -> int`
  - `discover_meeting_dates(start, end) -> list[(date, venue)]`
  - `backfill(conn, start, end)`
  - `next_meeting(conn) -> dict(date, venue, first_post_time, n_races)` 或 None
- `hkjc/odds.py`：`fetch_pools(date, venue, race_no, pools) -> dict[pool][combination] = odds`；`snapshot_meeting(conn, date, venue, pools=("WIN","PLA","QPL","QIN")) -> int`
- `hkjc/bets.py`（純函數）：過關組合 (`allup_combos(n_legs, n_per_combo_spec)` 例如 "6x63","2x3")、膽拖/複式組合生成、單場及過關結算（用 dividends 表，per $10，處理平頭馬/多個派彩、退出馬退款）。
- `hkjc/features.py`：`build_training_frame(conn) -> DataFrame`（每匹馬每場一行，所有特徵只用該場**之前**嘅資料，杜絕偷看未來），`build_card_frame(conn, date, venue) -> DataFrame`（未來賽事排位表特徵）。
- `hkjc/model.py`：`train(df, ...)`、`predict(model, df) -> DataFrame[race_id, horse_no, p_win, p_place]`（每場 p_win 總和=1；p_place 合理正規化），walk-forward 預測寫入 `predictions`。
- `hkjc/strategy.py`：由預測+賠率砌飛：`build_tickets(pred_df, odds=None, config) -> list[Ticket]`（Ticket = dict: strategy, ticket_no, legs=[{race_no, selections:[...]}], unit_stake, n_combos, total_stake）。
- `hkjc/paper.py`：`place_tickets(conn, run_id, date, venue, tickets)`、`settle(conn, date)`、`bankroll(conn)`。
- `hkjc/backtest.py`：策略網格回測，結果寫 `backtest_results` 並輸出中文報告。
- `hkjc/cli.py`（統籌者負責，已寫好）：`python -m hkjc backfill|post|pre|tick|backtest|report`。佢會呼叫：
  - `pipeline.pre_meeting(conn, run_id, date, venue) -> dict(report_md=str, notes=dict)`：排位表+賠率已入庫後，建特徵→預測→寫 predictions→砌飛→paper.place_tickets→中文賽前報告（每場預測、賠率、價值、注碼、組合）。
  - `pipeline.post_meeting(conn, run_id, date, venue) -> dict(report_md=str, notes=dict)`：賽果已入庫、模擬注已結算後，出中文賽後分析（當日騎練組合、路程、草地/泥地、檔位表現；模型命中率；模擬注盈虧），並按需要重新訓練模型。
  - `paper.settle(conn, date) -> int`、`paper.summary_md(conn) -> str`。
  - `backtest.main(conn, start=None, end=None)`。

## 質素要求
- 每個模組要有 pytest 測試（用 fixtures 或細規模真實數據），提交前跑過全部測試。
- 解析結果要同網頁對得上（抽樣對數）。
- 回測/特徵嚴禁未來資料洩漏；回測用 walk-forward（只用預測日之前嘅數據訓練）。
- 回測中用最終獨贏賠率屬賽前不可完全得知，要喺報告註明。
- 唔好刪除或覆蓋其他代理嘅檔案；只改自己負責嘅檔案。
