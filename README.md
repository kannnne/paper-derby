# 香港賽馬分析 + 模擬投注（Paper Trade）系統

> ⚠️ **只做模擬投注**：系統從來唔會登入任何馬會戶口、唔會落真注，亦冇儲存任何密碼。
> 所有報告只供研究參考，**並非投資或投注建議**。馬會彩池抽水約 17.5%–25%，回測顯示絕大部分策略長期都係蝕錢。

## 系統做咩

1. **資料庫**：自動抓馬會網站嘅賽果、派彩、排位表，同 bet.hkjc.com 公開嘅即時賠率，儲存喺 `data/hkjc.db`（SQLite）。
2. **賽後分析**：每個賽馬日最後一場開跑後約 45 分鐘，抓賽果 → 結算模擬注 → 出賽後報告（逐場賽果 vs 模型、騎練表現、檔位路程、模擬注盈虧）→ 用最新數據重新訓練模型。
3. **賽前投注參考**：下一個賽馬日第一場開跑前 2 小時，抓排位表 + 即時賠率 → 模型預測每匹馬勝出 / 上名機率 → 計每匹馬「評分」→ 10 隻機械人各自模擬落注（每場都寫低買 / 唔買同原因）→ 出賽前報告。
4. **10 隻模擬投注機械人**（見下面「10 隻機械人」，每隻獨立 HK$10,000 模擬資金，每注 $10）。用戶原本嘅固定組合（1 條位置 6x63 + 3 條位置Q 2x3）由「固定組合」機械人照買。
5. **回測**：用近兩年數據比較唔同玩法、揀場方法、膽拖 vs 複式，結果喺 `reports/回測報告.md`；機械人回測喺 `data/bot_backtest.json`。
6. **公開 Dashboard**（可選）：每次賽前 / 賽後自動更新 `site/` 靜態網站，設定好之後可以推上 GitHub Pages 俾朋友睇（見下面「公開網站（GitHub Pages）」）。

## 資料夾

| 位置 | 內容 |
|---|---|
| `hkjc/` | 程式（`scrape` 抓網頁、`parse` 解析、`odds` 即時賠率、`features` 特徵、`model` 模型、`strategy` 砌飛、`bets` 過關/派彩計算、`paper` 模擬戶口、`backtest` 回測、`pipeline` 賽前/賽後流程、`cli` 指令及排程） |
| `tests/` | 自動測試（`.venv/bin/python -m pytest -q`） |
| `scripts/` | 自動排程安裝 / 移除腳本、launchd 設定檔 |
| `data/hkjc.db` | 資料庫（賽果、排位表、賠率快照、預測、模擬注、資金流水） |
| `data/raw/` | 抓落嚟嘅網頁快取（gzip），重跑唔使再上網 |
| `data/models/` | 訓練好嘅模型（`*_latest.pkl` 係現用版本） |
| `data/logs/` | 執行紀錄：`hkjc.log`（主要）、`launchd.out.log` / `launchd.err.log`（排程輸出） |
| `reports/` | 報告：`YYYY-MM-DD_場地_賽前投注參考.md`、`YYYY-MM-DD_場地_賽後分析.md`、`回測報告.md` |
| `hkjc/scores.py` / `hkjc/bots.py` / `hkjc/publish.py` | 馬匹評分、10 隻機械人、網站數據匯出 + 發佈 |
| `site/` | 公開網站（純靜態：`index.html`、`app.js`、`style.css`，數據喺 `site/data/`）；係一個獨立 git repo，只會推呢個資料夾 |
| `data/publish.json` | 網站發佈設定（預設 `"enabled": false` = 只更新本機 `site/`，唔會上網推送） |

## 指令

喺專案資料夾執行（注意路徑有空格，要加引號）：

```sh
cd "<項目資料夾>"
.venv/bin/python -m hkjc <指令>
```

| 指令 | 作用 |
|---|---|
| `backfill 2024-09-01 2026-10-19` | 回補呢段日子所有本地賽馬日嘅賽果（已完成嘅會跳過） |
| `post [日期]` | 賽後：抓賽果、結算模擬注、出賽後報告、重新訓練。唔寫日期 = 最近一個未完成嘅賽馬日 |
| `pre [--force]` | 賽前：下一個賽馬日嘅排位表 + 即時賠率 → 預測 → 模擬落注 → 賽前報告。已做過就唔會再做；`--force` 會重做（舊嘅未結算模擬注會作廢退回，唔會重複落注；該賽馬日已經有賽果就唔會再做） |
| `tick` | 排程每 10 分鐘自動行一次，自己判斷應該做 `post` 定 `pre`（平時乜都唔做） |
| `backtest [開始 結束]` | 重新跑策略回測，更新 `reports/回測報告.md` |
| `report` | 喺畫面顯示模擬戶口總結（資金、每個策略盈虧、命中率） |

## 自動排程（launchd）

安裝（每 10 分鐘執行一次 `tick`，登入後即時開始）：

```sh
zsh "./scripts/install_scheduler.sh"
```

移除：

```sh
zsh "./scripts/uninstall_scheduler.sh"
```

`tick` 嘅規則：
- **賽前**：第一場開跑前 2 小時至前 15 分鐘之間做一次（例如 2026-10-21 跑馬地第 1 場 18:40 → 大約 16:40 做）。如果部 Mac 瞓咗、遲咗醒，只要仲未到開跑前 15 分鐘都會補做（log 會寫 "late"）。開跑前 15 分鐘之後先醒：只出參考報告，**唔會**記錄模擬注；第一場開跑後就乜都唔做。
- **賽後**：最後一場開跑後 45 分鐘開始，每次 tick 試抓賽果，直到全部賽果齊（"results"）先結算，唔會半途結算。成個賽馬日取消（例如颱風）而馬會網站 2 日後都冇賽果，會標記為取消（abandoned），模擬注全數退回，之後唔再重試。
- 網絡斷咗：該次 tick 只會寫錯誤入 log，下一次 tick 自動再試。
- 同一時間只會有一個 `pre` / `post` / `tick` 執行（`data/logs/hkjc.lock`）。
- 下一個賽馬日資料每小時先上網更新一次，平時 tick 唔會上網。

**部 Mac 要開機而且醒住**排程先會行（launchd 喺 Mac 瞓覺期間唔會執行，醒返之後先補行一次）。如果想賽馬日自動喚醒部 Mac，可以自己喺「系統設定 > 電池 / 能源」（或 `pmset` 排程）設定喚醒時間——系統唔會幫你改呢啲設定。

想睇排程有冇行：`tail -f "./data/logs/hkjc.log"`（每次 tick 會寫一行「next meeting …」）。

## 10 隻機械人

每隻機械人用同一個模型預測，但玩法同門檻唔同；各自有獨立 HK$10,000 模擬資金（資金唔夠就唔買，原因寫「資金不足」）。門檻只用 2024-25 季調校，2025-26 起係樣本外。

| # | id | 名 | 玩法 | 樣本外回測 ROI* |
|---|---|---|---|---:|
| 1 | `place_only` | 位置專家 | 只買單場位置；上名率 × 位置賠率夠門檻先買，每場最多 2 隻 | −15.7% |
| 2 | `fixed_combo` | 固定組合 | 用戶原本：1 條位置 6x63 過關 + 3 條位置Q 2x3 過關，每個賽馬日照買 | −32.6% |
| 3 | `adaptive` | 按場自動 | 每場計幾十種玩法嘅預期回報，揀最好而且過門檻嘅，唔抵就唔買 | −4.8% |
| 4 | `win_value` | 獨贏價值 | 勝率 × 獨贏賠率夠高先買獨贏 | +1.7% |
| 5 | `qpl_box` | 位置Q 複式 | 模型頭三選夠集中先買位置Q 複式 3 隻 | −9.7% |
| 6 | `qin_banker` | 連贏膽拖 | 有明顯首選先買，1 膽拖幾腳 | −13.1% |
| 7 | `tri_banker` | 單T 膽拖 | 兩隻強馬做雙膽拖幾腳 | −5.6% |
| 8 | `pla_parlay` | 位置過關 | 全日最有信心 3 場首選做位置 3x4 過關 | −36.3% |
| 9 | `favourite` | 跟大熱（對照組） | 每場大熱獨贏 + 位置各 $10，完全唔用模型 | −18.0% |
| 10 | `monkey` | 隨機猴子（對照組） | 每場隨機揀一隻買位置 $10（以場次編號做種子，可重現） | −22.0% |
| 11 | `daily_picks` | 每日精選 | 「今日精選」每種玩法各一個，全部照買追蹤成績 | −17.9%** |

\* 2025-26 + 2026-27 樣本外、約 100 個賽馬日（`data/bot_backtest.json`）。\*\* 每日精選回測見 `data/picks_backtest.json`（95% 區間 −32% 至 −3%）。回測用**最終**賠率，比真實賽前 2 小時落注樂觀；區間好闊，正數都好可能只係運氣。

每個賽馬日每隻機械人每場都有一行決定（`bot_decisions` 表，`race_no=0` = 過關 / 跨場決定），原因會寫埋用咗嘅數字，例如「3號 上名率55%，位置賠率1.9 → 期望值1.05 ≥ 門檻0.90，買位置 $10」。

## 今日精選（每種玩法各一個，`hkjc/picks.py`）

- **每種玩法都有一個精選**，玩邊種都有參考：「位置過關 6x63」、「位置Q 2x3 ×3 條」（用戶原本規則）、獨贏（2 個）、位置（2 個）、連贏、位置Q、二重彩、單T、三重彩、四連環、四重彩、孖寶、三寶。
- 每種玩法由模型機率逐場比較各種結構（單注、複式、膽拖、定位），喺預計命中唔太低（每種玩法有下限，例如獨贏 ≥ 15%、位置 ≥ 45%）嘅組合入面揀估計回報最高嗰個；派彩用即時賠率，冇就由獨贏賠率推算並用 2024-25 季派彩校準。
- **⭐ 值博** = 估計回報同預計命中都過 2024-25 季調校嘅門檻（現時只有連贏、二重彩、孖寶、三寶有門檻）；其他只係該玩法當日最好嘅選擇，**未必值博**。
- 「每日精選」機械人每日全部照買（大約 $1,600，其中 $1,080 係兩個固定玩法），長期追蹤每種玩法。樣本外回測：全部精選 −17.9%（95% 區間 −32% 至 −3%），⭐ 值博項目 +10.3%（區間 −39% 至 +78%，即係冇證據有長期優勢）。
- **預覽（未落注）**：馬會一開售賠率，每小時更新評分嘅同時，亦會預覽每隻機械人嘅買賣決定同今日精選（網站標明「預覽（未落注）」，唔計入資金／排行榜）；第一場開跑前約 2 小時先正式落注，正式決定會取代預覽。
- 賽前流程計好精選 → 存入 `picks` 表 → 機械人照買 → 網站同賽前報告「## 今日精選」按玩法分組顯示，賽後顯示每項結果。
- 重跑回測：`.venv/bin/python -c "from hkjc import db, picks; picks.backtest_picks(db.connect())"`（寫 `data/picks_backtest.json` 同 `reports/回測報告.md`「每日精選（分析決定玩法）」一節；網站唔會顯示回測）。

## 馬匹評分（0–100）

- **評分**由模型勝出機率計出：`評分 = 100 ÷ (1 + e^(−(ln(勝出機率 × 出賽匹數) + 0.57) ÷ 0.77))`。同場內評分高低次序**完全等於**模型勝出機率次序；唔同場次、唔同匹數之間都可以比較（大約等於該馬喺過去兩年所有出賽馬匹入面嘅百分位）。
- 參考：勝出機率等於「平均馬」（1 ÷ 出賽匹數）≈ 68 分；兩倍 ≈ 84 分；一半 ≈ 46 分；十分一 ≈ 10 分。
- **子評分**（往績、騎師、練馬師、騎練組合、檔位、路程場地、班次負磅、市場）係同場馬匹之間嘅相對分數（0 = 同場最差，100 = 同場最好），只用嚟解釋評分從何而來；未有賠率時「市場」留空。
- **理由**：每匹馬 2–4 句有數字嘅短句（有正有負），例如「近3仗平均第2.3名」「檔位11偏外」。
- 評分只係模型輸出，唔保證準確，並非投注建議。

## 改機械人門檻 / 每注金額

而家賽前模擬落注全部由 10 隻機械人負責（`hkjc/strategy.py` 嘅 `DEFAULT_CONFIG` 已經唔再用嚟落注，只係回測 / 報告入面「按場自動決定」參考表用）。

- 每隻機械人嘅門檻：`hkjc/bots.py` 嘅 `_CFG`（例如 `place_only` 嘅 `min_ev`、`min_p`、`max_per_race`）
- 每注金額：`hkjc/bots.py` 嘅 `UNIT`（預設 $10；馬會派彩以每 $10 計，改咗會按比例計）
- 每隻機械人起始資金：`hkjc/bots.py` 嘅 `START_BANKROLL`（只喺該機械人第一次落注時設定）
- 改完門檻建議重跑機械人回測（更新網站上嘅回測數字）：`.venv/bin/python -c "from hkjc import db, bots; bots.backtest_bots(db.connect())"`

改完建議跑一次測試：`.venv/bin/python -m pytest -q`。

## 試行（唔影響真資料）

可以用環境變數將資料庫、模型、報告、log 指去副本，例如：

```sh
sqlite3 data/hkjc.db ".backup /tmp/hkjc_copy.db"
HKJC_DB=/tmp/hkjc_copy.db HKJC_MODEL_DIR=/tmp/models HKJC_REPORT_DIR=/tmp/reports HKJC_LOG_DIR=/tmp/logs \
  .venv/bin/python -m hkjc report
```

## 公開網站（GitHub Pages）

網站只會顯示**模擬**數據（模型評分、機械人模擬注、公開賽果派彩），每頁底都有聲明：「模擬研究，非投注建議；未滿18歲不得投注；博彩要有節制，如需協助可致電平和基金熱線 1834 633」，並註明同香港賽馬會無關、並非官方網站。網站唔會包含任何戶口、電郵或本機路徑。

### 本機預覽（唔會上網）

```sh
cd "<項目資料夾>"
.venv/bin/python -c "from hkjc import db, publish; print(len(publish.export_site(db.connect())), 'files')"
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory site
```

然後用瀏覽器開 <http://127.0.0.1:8000/>（Ctrl+C 停止）。平時每次 `pre` / `post` 都會自動更新 `site/data/` 同喺 `site/` 入面 commit（`enabled` 係 `false` 時唔會推送）。

### 第一次設定（以下步驟要你自己做，系統唔會幫你開戶口或者處理密碼）

1. **開 GitHub repo**：登入 GitHub → New repository → 名建議用中性名，例如 `paper-derby`（**唔好**用「HKJC」或者馬會名，免得人以為係官方）→ 揀 **Public** → **唔好**剔「Add a README」（要空 repo）→ Create。注意網址會包含你嘅 GitHub 用戶名。
2. **俾部 Mac 有權推送**（揀一種）：
   - **SSH deploy key（建議，只限呢個 repo）**：
     ```sh
     ssh-keygen -t ed25519 -f ~/.ssh/paper_derby -N "" -C "paper-derby deploy key"
     pbcopy < ~/.ssh/paper_derby.pub
     ```
     去 repo 嘅 Settings → Deploy keys → Add deploy key，貼上，**剔 Allow write access** → Add key。然後喺 `~/.ssh/config` 加：
     ```
     Host github-paper-derby
       HostName github.com
       User git
       IdentityFile ~/.ssh/paper_derby
       IdentitiesOnly yes
     ```
     試吓：`ssh -T github-paper-derby`（第一次會問你信唔信 github.com 嘅 fingerprint，答 yes）。remote 就係 `github-paper-derby:<你的用戶名>/paper-derby.git`。
     （排程係無人看管咁行，所以呢條 key 唔設密碼；佢只可以寫呢一個 repo。）
   - **Fine-grained token**：GitHub → Settings → Developer settings → Fine-grained tokens → Generate：Repository access 只揀 `paper-derby`，Permissions → Contents: **Read and write**。之後喺 Terminal 手動推一次（見第 4 步，remote 用 `https://github.com/<你的用戶名>/paper-derby.git`），git 問密碼時貼 token，macOS 會記入「鑰匙圈」。**唔好**將 token 寫入 remote 網址或者任何檔案。
3. **改 `data/publish.json`**（例子用 SSH）：
   ```json
   {"enabled": true, "remote": "github-paper-derby:<你的用戶名>/paper-derby.git", "branch": "main"}
   ```
4. **第一次手動發佈**：
   ```sh
   cd "<項目資料夾>"
   .venv/bin/python -c "from hkjc import db, publish; print(publish.publish(db.connect(), '首次發佈'))"
   ```
   見到 `'status': 'pushed'` 就得。（用 token 嘅話，第一次改喺 Terminal 行：`git -C site push https://github.com/<你的用戶名>/paper-derby.git HEAD:refs/heads/main`，輸入用戶名 + token，之後自動推送就會用鑰匙圈記住嘅 token。）
5. **開 Pages**：repo → Settings → Pages → Build and deployment → Source 揀 **Deploy from a branch** → Branch 揀 **main**、資料夾 **/ (root)** → Save。
6. 大約 1–5 分鐘後網站會喺：`https://<你的用戶名>.github.io/paper-derby/`

之後每次賽前 / 賽後排程都會自動推送。推送失敗（例如斷網、key 失效）只會喺 `data/logs/hkjc.log` 寫 warning，**唔會**影響模擬落注或者報告；下次發佈會一齊補推。想停：將 `enabled` 改返 `false`（或者喺 GitHub 刪 repo / 關 Pages）。

只有 `site/` 資料夾會推上去（佢係獨立 git repo），`data/hkjc.db`、報告、log、模型都唔會上傳。commit 作者係中性嘅「Paper Derby Bot」，唔會用你嘅名或電郵。網站大細：每個賽馬日約 150 KB，一季約 15 MB，GitHub Pages 冇問題。

## 注意事項（請務必閱讀）

- **回測有偷看未來嘅成份**：回測用嘅係**最終**獨贏賠率，但開跑前 2 小時嘅賠率會唔同，所以回測結果會偏樂觀。
- **抽水**：獨贏 / 位置 / 連贏 / 位置Q 17.5%，二重彩 19.5%，單T 23%，三重彩 / 四連環 / 四重彩 25%；過關每一關都要扣一次，關數越多長線回報越差。
- **冇證據有長期優勢**：回測入面所有固定組合都係負回報（例如位置 6x63 約 −40%、3 條位置Q 2x3 約 −14% 至 −19%）；少數好睇嘅結果好可能只係運氣（測試咗過百個策略）。
- 模型只用賽前已知資料（之前嘅賽果 + 排位表 + 即時賠率），每次預測只會用該日之前嘅數據訓練。
- 只係模擬：唔會落真注，報告內容**唔構成任何投資或投注建議**。博彩要有節制。

## 雲端 24/7 運行（GitHub Actions，現行方式）

- Repo：`kannnne/paper-derby`（公開）。公開網站：https://kannnne.github.io/paper-derby/
- **評分**：馬會一開售賠率（通常賽前一日）就計評分並更新網站，之後大約每小時用最新賠率重計，直到開跑前 2 小時嘅正式賽前流程（落模擬注、出今日精選）。手動：`python -m hkjc scores`。
- `.github/workflows/race-bot.yml` 每 30 分鐘（香港時間 08:00–01:59）喺 GitHub 伺服器跑一次 `python -m hkjc tick`，自動做賽前（第一場前 2 小時）同賽後（最後一場後約 45 分鐘）流程，然後 commit 數據並重新發佈網站。關機都照跑。
- 資料庫以文字形式存喺 `data/db/*.sql`（每次運行前還原、運行後匯出）；模型每次喺雲端重新訓練，唔存入 git。
- 手動觸發：GitHub → Actions → race-bot → Run workflow（command 可填 `tick` / `probe` / `report`）。
- 本機想睇最新數據：`git pull` 之後 `.venv/bin/python -m hkjc.dbdump load data/db data/hkjc.db`。
- Mac 本機排程已停用（避免雙重落注）。如要改返用本機，先喺 GitHub 停用 workflow，再跑 `scripts/install_scheduler.sh`。
- 注意：GitHub 排程繁忙時可能延遲 5–30 分鐘；公開 repo 連續 60 日冇活動會被暫停排程（賽季期間每次賽馬日都有 commit，唔會觸發）。
