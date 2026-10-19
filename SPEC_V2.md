# 第二期規格：10 隻機械人 + 馬匹評分 + 公開 Dashboard（所有代理必須遵守，同 SPEC.md 一齊睇）

仍然只做 paper trade（模擬），絕不落真注。所有公開內容要有「模擬研究、非投注建議、博彩要有節制」聲明。

## 資料庫（已加，見 hkjc/db.py）
- `paper_tickets.bot_id`, `paper_tickets.reason`, `bankroll_ledger.bot_id`（每隻機械人獨立資金，起始 HK$10,000）。舊有 NULL bot_id 嘅紀錄當 legacy（現時真實 DB 冇任何 tickets）。
- `horse_scores(run_id, race_id, horse_no, score, rank_in_race, sub_scores JSON, reasons JSON, created_at)`
- `bot_decisions(run_id, bot_id, date, venue, race_no, action bet/skip, structure, selections JSON, stake, reason, ticket_id, created_at)`；race_no=0 代表跨場（過關）決定。**每隻機械人每場都要有一行**（買或者唔買都要寫原因）。

## 馬匹評分 `hkjc/scores.py`
`score_horses(pred_df, card_df) -> DataFrame[race_id, race_no, horse_no, score, rank_in_race, sub_scores(dict), reasons(list[str])]`
- `pred_df` = model.predict_card 輸出（p_win, p_place…）；`card_df` = features.build_card_frame 輸出（全部特徵）。
- `score` 0–100 綜合評分：同模型 p_win 單調一致（同場內排序必須同 p_win 一樣），而且跨場可比（例如以勝出機率對比「平均馬」1/田數嘅倍數做 log 尺度映射）；寫清楚公式。
- `sub_scores`（0–100，同場內相對）：`form` 往績、`jockey` 騎師、`trainer` 練馬師、`combo` 騎練組合、`draw` 檔位、`fit` 路程/場地/跑道適應、`class` 班次/負磅、`market` 市場（冇賠率時 None）。每項由對應特徵組成。
- `reasons`：2–4 句短中文（繁體），有正有負，具體數字，例如「近3仗平均第2.3名」「潘頓/廖康銘組合近年上名率45%」「檔位11偏外」。
- `save_scores(conn, run_id, df)` 寫 `horse_scores`。

## 10 隻機械人 `hkjc/bots.py`
`BOTS: list[dict(id, name_zh, style_zh, description_zh, config)]`；`decide(bot, pred_df, odds, bankroll) -> list[Decision]`，Decision = dict(bot_id, race_no, action, structure, selections, stake, reason, ticket(None 或 SPEC ticket dict))。
`run_bots(conn, run_id, date, venue, pred_df, odds) -> list[Decision]`：每隻機械人決定 → 有 ticket 就 paper.place_tickets（帶 bot_id、reason）→ 寫 bot_decisions。

| id | 名 | 玩法 |
|---|---|---|
| `place_only` | 位置專家 | **只買位置**（單場位置），每場最多 2 隻，上名率×位置賠率 ≥ 門檻先買 |
| `fixed_combo` | 固定組合 | 用戶原本：1 條位置 6x63 + 3 條位置Q 2x3 |
| `adaptive` | 按場自動 | 現有 ADAPTIVE（按場揀玩法或唔買） |
| `win_value` | 獨贏價值 | 勝率×獨贏賠率 ≥ 門檻嘅獨贏 |
| `qpl_box` | 位置Q 複式 | 前三名集中時買位置Q 複式 3 隻 |
| `qin_banker` | 連贏膽拖 | 有明顯首選時 1 膽拖 N 腳 |
| `tri_banker` | 單T 膽拖 | 兩隻強馬時 2 膽拖 N 腳 |
| `pla_parlay` | 位置過關 | 最有信心 3 場首選 3x4 |
| `favourite` | 跟大熱（對照組） | 每場大熱獨贏+位置各 $10 |
| `monkey` | 隨機猴子（對照組） | 每場隨機揀一隻買位置 $10（以 race_id 做 seed，可重現） |
- 門檻只用 2024-25 季調，2025-26 + 2026-27 做樣本外驗證；冇賠率（未開售）時用估計賠率並喺原因註明。
- 資金不足（餘額 < 注碼）就 skip，原因寫「資金不足」。
- `backtest_bots(conn, source='wf-combined-v1') -> DataFrame`：每隻機械人歷史回測（每個賽馬日 stake/return、累積資金曲線），寫 `data/bot_backtest.json` 供 dashboard 用。

## Dashboard（GitHub Pages 靜態網站）
- 原始碼：`site/`（`index.html`, `app.js`, `style.css`, 資料喺 `site/data/`）。純靜態，無後端；外部 JS 只可用 cdnjs/jsdelivr 固定版本；手機睇都要得；繁體中文。
- `hkjc/publish.py`：`export_site(conn, out_dir=site/) -> list[path]` 由 DB 生成 JSON；`publish(conn, message)` → export + 喺 `site/`（獨立 git repo，remote 由 `data/publish.json` 設定 {"enabled": bool, "remote": "...", "branch": "main"}）commit + push。enabled=false 時只 export 唔 push。推送失敗唔可以令 pre/post 流程失敗。
- JSON 格式：
  - `site/data/index.json`：`{updated_at, disclaimer, next_meeting:{date,venue,first_post_time}, meetings:[{date, venue, status:'pre'|'results', file}], bots:[{id, name, style, description, bankroll, start_bankroll, live:{n_meetings, stake, returns, profit, roi, hit_rate}, backtest:{period, stake, returns, roi, roi_ci:[lo,hi], oos_roi}}]}`
  - `site/data/meetings/YYYY-MM-DD_VV.json`：`{date, venue, venue_zh, status, generated_at, odds_snapshot_at, model_version, races:[{race_no, post_time, race_class, distance, surface, course, going, race_name, horses:[{horse_no, horse_name, jockey, trainer, draw, weight, score, rank, p_win, p_place, win_odds, pla_odds, sub_scores:{...}, reasons:[...], scratched, result:{finish_pos, final_win_odds}|null}], decisions:[{bot_id, action, structure, selections, stake, reason, payout|null, profit|null}], result:{placings:[...], dividends:{WIN:[[comb,div]], PLA:[...], QIN:[...], QPL:[...], TRI:[...], FF:[...]}}|null}], bots:[{bot_id, stake, payout, profit, tickets:[{strategy, legs, total_stake, payout, reason}]}]}`
  - `site/data/bot_history.json`：每隻機械人實盤（paper）資金曲線 + 回測資金曲線。
- 頁面：① 首頁：下一個賽馬日、10 隻機械人排行榜（實盤 + 回測 ROI）、資金曲線圖；② 賽馬日頁：逐場馬匹評分表（評分、子評分條、理由、勝率/上名率、賠率），每場 10 隻機械人決定同原因，賽後顯示賽果、派彩、每隻機械人盈虧；③ 機械人頁：玩法說明、歷史表現、全部注項。
- 公開內容唔可以包含任何個人資料、帳戶、電郵。
- **UI/UX 方向（用戶要求）**：參考電視台直播賽馬畫面嘅排位/字幕設計，同賽馬遊戲感：深草地綠/夜賽深藍底、號碼布式馬號徽章、LED/翻牌式賠率板、直播式「排位表」橫條、評分用能量條/星級、機械人排行榜做成跑道（10 隻機械人似馬匹喺跑道上按資金/ROI 排位，帶號碼布同顏色）、賽後似直播終點字幕顯示頭三名同派彩。動畫要克制、可讀性優先、手機得。原創設計：唔好用馬會 logo/商標/綵衣圖片或抄任何現有遊戲，唔好令人以為係馬會官方網站。
