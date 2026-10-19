# Review: `hkjc/bets.py` and `hkjc/odds.py` (independent, adversarial)

Reviewer date: 2026-10-19. Files reviewed: `hkjc/bets.py`, `hkjc/odds.py`, `tests/test_bets.py`, `tests/test_odds.py`.
Reviewer tests: `tests/test_review_bets.py`. They use their own brute-force generators and their own settler, and import nothing from `bets` for the oracle side.

```
.venv/bin/python -m pytest -q tests/test_bets.py tests/test_odds.py tests/test_review_bets.py
125 passed, 2 skipped, 6 xfailed
```
The 6 `xfail(strict=True)` tests are the confirmed bug below. Once it is fixed they will XPASS, which strict mode reports as a failure, so remove the markers then. The 2 skipped tests are live-API tests, gated by `HKJC_LIVE=1`.

## Summary

| # | Check | Result |
|---|-------|--------|
| B1 | **Settling races where a whole pool was refunded (closed or void)** | **FAIL: confirmed bug** |
| 1 | All-up formula table, 2–6 legs, vs HKJC page | PASS |
| 2 | Combination counts vs chance table, plus brute-force generators | PASS |
| 3 | Settlement vs independent settler (random + real DB data) | PASS (except B1) |
| 4 | Scratching / dead-heat / small-field / QPL<7 rules | PASS for code logic. One wrong assumption in a test comment (N1). QPL<7 FAILS through B1. |
| 5 | odds.py: canonicalisation, SCR/---/empty, query bytes, snapshot, HK time, errors, live R1 | PASS (minor notes M1–M3) |

---

## B1 (CONFIRMED BUG): refunded-pool rows are settled as LOSE

**Where:** `hkjc/bets.py:319-323` (`index_dividends`) and `hkjc/bets.py:342-350` (`combo_outcome`).

**What happens.** When a pool is closed and refunded, the results page shows "-" or 退款. Two cases:
- Rule 3.9(b): QPL with "less than 7 Starters", PLA with fewer than 5 (3 dividends) or 4 (2 dividends).
- Rule 3.15: a void or abandoned race.

These rows reach the dividends table as a *pool-level marker*, not as a per-combination row:
* Legacy rows already in `data/hkjc.db`: `combination='-'`, `dividend=NULL`. There are 10 rows each in WIN/PLA/QIN and 12 in QPL, from races 20230920-HV-01, 20230920-HV-03 (QPL closed after a WV) and all of 2023-10-08 ST (typhoon, abandoned).
* Current `hkjc/parse.py:531` writes `combination='REFUND'`, `dividend=10.0`.

`index_dividends` cannot canonicalise `'-'` or `'REFUND'`. It catches the `ValueError` and stores the raw key, so `idx['QPL'] == {'-': None}`. `combo_outcome` then sees a non-empty pool, finds that the staked key is not in it, and returns `('lose', 0.0)`. The documented assumption at `bets.py:55` ("a None dividend → refund") only applies when the combination key matches, which never happens for these marker rows.

**Reproductions** (all in `tests/test_review_bets.py`, xfail strict):

| Input | Expected (rules) | Actual |
|---|---|---|
| `settle_single("QPL", ["3,4"], [WIN 3=31, QPL REFUND=10.0])` | payout 10 (refund, 3.9(b)) | 0.0, status `lose` |
| QPL 2x3, leg 1 = race with row `QPL '-' NULL`, selections 3,4,2; leg 2 pays 25.5 | 30 + 25.5 + 76.5 = **132.0** | **25.5** |
| PLA 2x3, leg 1 abandoned (all pools `REFUND`), leg 2 PLA pays 11 | 10 + 11 + 11 = 32 (3.15 / 3.10(c)) | 11 |
| Real DB `20230920-HV-01`, `settle_single("QPL", ["3,4"], db_rows)` | 10 | 0 |
| Real DB sweep: 3,640 random PLA 6x63 / QPL 2x3 tickets over 182 meetings | 0 mismatches | **3 mismatches, all 2023-09-20 HV QPL 2x3, each $0 vs $30** |

**Impact.** Every paper or backtest ticket with a leg in a race whose pool was closed or voided is settled as a total loss on that leg. All-up multi-leg units are also wiped out when they should have been carried forward (rule 3.10(c)(ii)). The QPL<7-starter case recurs whenever there is a late scratching from a 7-runner field.

**Suggested fix** (in `bets.py`; handle both representations):
```python
_REFUND_MARKERS = {"-", "REFUND", "退款", ""}
# in index_dividends, when canonical() fails:
key = "__POOL_REFUND__" if str(combo).strip().upper() in _REFUND_MARKERS else str(combo)
# in combo_outcome, after the scratched check:
if not rows or "__POOL_REFUND__" in rows:
    return "pool_refund", 1.0, None
```
Optionally also re-scrape or normalise the legacy `'-'` rows in the DB to the new `'REFUND'` convention. When B1 is fixed, drop the `xfail` markers in `tests/test_review_bets.py`.

---

## 1. All-up formulas: PASS
I fetched the HKJC all-up page myself and parsed its "All Up Combinations" table, which has 41 rows from 2x1 to 6x63. I transcribed it into `AU_PAGE` in the test file.
* `bets.ALLUP_FORMULAS` has exactly the same 41 keys. For every formula, the per-leg-size subset counts equal the page's columns. Each subset is unique, and the subset set equals my brute-force set built from whole C(n,k) layers.
* `allup_subsets(6,'6x63')` has 63 unique subsets, which equals all non-empty bitmasks of 6. `allup_subsets(2,'2x3')` has 3: (0,), (1,), (0,1).
* Minor nit (not a bug): `allup_formula` also accepts the "football-style" 7- and 8-leg formulas, so `settle_allup` will settle a 7-leg racing all-up that HKJC does not offer (maximum 6 races; FCT/TRI maximum 3 per the page). `ALLUP_POOLS` and `ALLUP_MAX_LEGS` are defined but never enforced.

## 2. Combination counts: PASS
* I parsed every non-blank number in the chance-table page: QIN/QPL multiple and 1 banker; Trio multiple, 1 and 2 bankers; First 4 multiple, 1, 2 and 3 bankers; FCT/TCE/QTT multiple plus banker single and multiple. That is 24 parametrised rows. My brute force (`itertools.product` with a filter) matches the page, and `bets.multiple` / `bets.banker` match both.
* I ran randomised brute-force set equality, not just counts, for all 9 pools: box sizes 1–9, and bankers 1..k-1 × legs 0–7 in both modes.
* Observation: the HKJC page leaves a few cells blank, for example QTT 1 banker multiple with 3 legs (should be 24). The module's numbers there agree with the formula.

## 3. Settlement correctness: PASS (except B1)
The independent settler (`my_settle_allup`) uses Fractions and the identity payout = unit × Σ_{S∈formula} Π_{i∈S} F_i, where F_i is the sum of the leg factors. It never enumerates units, so it is structurally different from the module.
* **Random:** 40 seeds × 4 rounds. Each round settles PLA 6x63 (1 horse per leg), QPL 2x3 (3 horses per leg), a random formula/pool/selection size, and single PLA/QPL/WIN bets. The scenarios have 4–14 declared horses, 0–2 scratchings, random dead heats for 1st/2nd/3rd, 2-place fields, QPL closed below 7 starters, unsorted QPL combination strings, and unit stakes of 1/5/10/20. All agree to the cent, and `n_units == ticket_units`.
* **Real DB:** 10 random complete meetings, with 1 PLA 6x63 and 3 QPL 2x3 tickets each (6 distinct races), all matched to the cent. An extended sweep of 3,640 tickets over all 182 complete meetings included 173 legs with a scratched selection and 55 dead-heat legs. Its only mismatches were the 3 B1 cases.
* Settling from the dividends table rather than finishing positions is correct. In DB race 20250212-ST-08, horse 12 was DISQ yet PLA/QPL dividends were paid on it, so any settlement based on `runners.finish_pos` would be wrong there.

## 4. Rules: scratchings, dead heats, small fields, QPL<7
Source: Horse Race Betting Rules s.3, rev. 2026-03-30 (PDF linked from the betting-rules page).
* **3.10(a)** says "Bets involving the Withdrawal will be refunded. For Multiple Bets a Refund will be made only in respect of Bets involving the Withdrawal". The module refunds exactly the combinations containing the scratched horse. PASS.
* **3.10(c)** says "(i) the Bet on a Withdrawal in a one-race combination will be refunded. (ii) The Bet on a Withdrawal in a multi-Leg combination will be carried forward to the next Leg." The module uses leg factor 1.0, which implements both. PASS. Note that the caller must pass `scratched_by_race`; otherwise a scratched horse just has no dividend row and loses. paper.py must source it from `runners.scratched`.
* **3.9(a)**: two Place dividends with 4, 5 or 6 Declared Starters at the commencement of betting, three with 7 or more. The module is data-driven: it pays whatever PLA rows exist. PASS. DB race 20230920-HV-03 (7 entries, 1 WV) paid 2 places, which shows the rows are the right authority.
* **3.11(a)(i)**: a dead heat for 3rd gives 4 PLA dividends. Each staked horse is paid its own row. PASS (real example 20230910-ST-07: PLA 5, 7, 8, 9).
* **3.9(b)**: "Quinella Place less than 7 Starters" closes the pool and refunds all bets. If the pool simply has no rows, the module refunds (pool_refund). With the DB/parse.py marker rows it does not, which is **B1**.
* **N1 (test-comment error, not a code bug).** `tests/test_bets.py:208` claims that in a QPL dead heat for 3rd, "5,9 does NOT win". Rule 3.11(a)(ii)(c) allocates 2/15 to "3rd & 3rd", so the pair of the two dead-heaters *does* win. Real data confirms it: 20230910-ST-07 has 6 QPL rows including 5,9 = 111.0. The code is unaffected because it pays whatever rows exist, but the test's fixture and comment encode a wrong rule. Fix the comment, or add the 3rd&3rd row to that fixture.

## 5. odds.py: PASS
* `ODDS_QUERY` and `MEETING_QUERY` are byte-identical (UTF-8) to the `query` fields of `tests/fixtures/odds_query.json` and `odds_meeting_query.json`.
* Canonicalisation: '01,07' → '1,7'; '07,01' → '1,7' (QIN/QPL ascending); FCT/TCE keep their order; 'SCR', '---', '' and None map to None. The scratched set is built only from SCR, not '---'. An empty `oddsNodes` gives `{}` with `last_update=None`.
* Errors: a live probe with an altered query returned HTTP 200 with `{"errors":[{"message":"Internal server error - WHITELIST_ERROR"}]}`. `_post` raises `OddsAPIError` immediately with no retry, and `snapshot_meeting` propagates it and writes nothing. PASS.
* Idempotency: re-running with the same `snapshot_at` leaves the row count unchanged (INSERT OR REPLACE on a PK that includes `snapshot_at`). The default `snapshot_at` is HK local time (UTC+8), verified against a UTC+8 clock window. `_hk_minute('…T10:40:00Z')` gives `18:40`. PASS.
* **Live check** (1 request): `fetch_pools('2026-10-19','ST',1)` returned WIN odds for 13 runners, all equal to the official results page (racing.hkjc.com localresults 2026/10/19 ST R1): 1=4.1, 2=9, 3=27, 4=5.5, 5=37, 6=5.1 (winner), 7=53, 8=4.6, 9=87, 10=31, 11=57, 12=25, 13=11. Pool status was PAYOUT. That race is not in the DB yet (the backfill has reached 2025-02). The live result also equals fixture `odds_20261019_ST_R1.json`. PASS.
* Minor notes, not bugs:
  * M1: `_race_numbers` (`odds.py:481`) silently swallows an `OddsAPIError` from the meeting query, with no log line.
  * M2: `snapshot_meeting` buffers all races and writes at the end (`odds.py:514`), so an error on race N discards races 1..N-1 already fetched.
  * M3: in probe mode, when nothing is known, a not-yet-open meeting probes all 14 races, because a PoolOdds with empty pool dicts is truthy.
