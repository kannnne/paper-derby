"""Tests for hkjc.strategy (ticket construction).  Expected numbers derived by hand."""
import itertools
from math import comb

import numpy as np
import pandas as pd
import pytest

from hkjc import bets, strategy as st


def _brute_place(p, k):
    p = np.asarray(p, float) / np.sum(p)
    out = np.zeros(len(p))
    for perm in itertools.permutations(range(len(p)), k):
        pr, rem = 1.0, 1.0
        for h in perm:
            pr *= p[h] / rem
            rem -= p[h]
        for h in perm:
            out[h] += pr
    return out


def test_harville_matches_brute_force():
    p = [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07]          # 7 runners -> 3 places
    np.testing.assert_allclose(st.harville(p), _brute_place(p, 3), atol=1e-12)
    assert st.harville(p).sum() == pytest.approx(3.0)
    q = [0.5, 0.3, 0.2, 0.0]                             # 4 runners -> 2 places
    # hand: P(A in top2) = .5 + .3*.5/.7 + .2*.5/.8 = .5 + .2142857 + .125 = .8392857
    assert st.harville(q)[0] == pytest.approx(0.8392857142857, abs=1e-12)
    assert st.harville(q).sum() == pytest.approx(2.0)


def test_market_probs():
    # odds 2, 4, 4 -> 1/o = .5,.25,.25 (sum 1)
    np.testing.assert_allclose(st.market_probs([2.0, 4.0, 4.0]), [0.5, 0.25, 0.25])
    # overround: 1/2+1/2+1/4 = 1.25 -> .4,.4,.2 ; NaN -> 0
    np.testing.assert_allclose(st.market_probs([2.0, 2.0, 4.0, np.nan]), [0.4, 0.4, 0.2, 0.0])


def test_p_at_least_two():
    # three independent .5: P(>=2) = 3*.125 + .125 = .5
    assert st.p_at_least_two([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # .9,.8 : P(both) = .72
    assert st.p_at_least_two([0.9, 0.8]) == pytest.approx(0.72)


def make_pred(n_races=8, field=10, strong=None):
    """Synthetic card: in race r the favourite (horse 1) has p_win = base[r]."""
    strong = strong or {1: .30, 2: .50, 3: .20, 4: .45, 5: .25, 6: .40, 7: .35, 8: .15}
    rows = []
    for r in range(1, n_races + 1):
        top = strong[r]
        rest = (1 - top) / (field - 1)
        ps = [top] + [rest * (1 + 0.1 * (field - 1 - i) - 0.05 * (field - 2)) for i in range(field - 1)]
        ps = np.array(ps) / np.sum(ps)
        pl = st.harville(ps)
        for h, (pw, pp) in enumerate(zip(ps, pl), start=1):
            rows.append({"race_no": r, "horse_no": h, "p_win": pw, "p_place": pp,
                         "win_odds": round(0.82 / pw, 1)})
    return pd.DataFrame(rows)


def test_build_tickets_default_products():
    df = make_pred()
    tk = st.build_tickets(df)
    pla = [t for t in tk if t["strategy"] == "PLA_6x63"]
    qpl = [t for t in tk if t["strategy"] == "QPL_2x3"]
    assert len(pla) == 1 and len(qpl) == 3
    t = pla[0]
    # confidence: favourites' p_win .50(R2) .45(R4) .40(R6) .35(R7) .30(R1) .25(R5) -> top 6
    assert [l["race_no"] for l in t["legs"]] == [1, 2, 4, 5, 6, 7]
    assert all(l["selections"] == [1] for l in t["legs"])
    assert t["n_combos"] == 63 and t["total_stake"] == 630.0 and t["spec"] == "6x63"
    assert t["pool"] == "PLA" and t["unit_stake"] == 10.0
    # QPL: 3 tickets, 6 distinct races, 3 horses -> 3 pairs/leg -> 3+3+9 = 15 units = $150
    races = [l["race_no"] for x in qpl for l in x["legs"]]
    assert len(races) == 6 and len(set(races)) == 6
    for x in qpl:
        assert x["n_combos"] == 15 and x["total_stake"] == 150.0 and x["spec"] == "2x3"
        assert all(len(l["selections"]) == 3 for l in x["legs"])
        assert [l["race_no"] for l in x["legs"]] == sorted(l["race_no"] for l in x["legs"])
    assert [x["ticket_no"] for x in qpl] == [1, 2, 3]


def test_unit_stake_config():
    tk = st.build_tickets(make_pred(), config={"unit_stake": 1, "products": ["PLA_6x63"]})
    assert len(tk) == 1 and tk[0]["total_stake"] == 63.0


def test_race_rules():
    df = make_pred()
    first = st.build_allup(df, race_rule="first6")
    last = st.build_allup(df, race_rule="last6")
    fixed = st.build_allup(df, race_rule="fixed", fixed_races=[8, 3, 5, 1, 2, 7])
    assert [l["race_no"] for l in first["legs"]] == [1, 2, 3, 4, 5, 6]
    assert [l["race_no"] for l in last["legs"]] == [3, 4, 5, 6, 7, 8]
    assert [l["race_no"] for l in fixed["legs"]] == [1, 2, 3, 5, 7, 8]
    assert st.build_allup(df, race_rule="fixed", fixed_races=[1, 2]) is None
    # value: model p vs market p from win_odds.  Make race 8 favourite heavily
    # under-priced in the market -> it must be chosen first by 'value'.
    df2 = df.copy()
    df2.loc[(df2.race_no == 8) & (df2.horse_no == 1), "win_odds"] = 30.0
    val = st.build_allup(df2, race_rule="value")
    assert 8 in [l["race_no"] for l in val["legs"]]
    conf = st.build_allup(df2, race_rule="confidence")
    assert 8 not in [l["race_no"] for l in conf["legs"]]


def test_qpl_pairings_and_consecutive():
    df = make_pred()
    chrono = st.build_multi_allup(df, race_rule="confidence", pairing="chrono")
    rank = st.build_multi_allup(df, race_rule="confidence", pairing="rank")
    # QPL confidence ranks races by P(>=2 of top3 place); with this card the
    # order follows the favourite strength: R2 R4 R6 R7 R1 R5 (R3, R8 weakest)
    chosen = sorted(l["race_no"] for t in chrono for l in t["legs"])
    assert chosen == [1, 2, 4, 5, 6, 7]
    assert [[l["race_no"] for l in t["legs"]] for t in chrono] == [[1, 2], [4, 5], [6, 7]]
    assert [[l["race_no"] for l in t["legs"]] for t in rank] == [[1, 5], [2, 4], [6, 7]]
    cons = st.build_multi_allup(df, race_rule="consecutive")
    pairs = [[l["race_no"] for l in t["legs"]] for t in cons]
    assert all(b == a + 1 for a, b in pairs)
    flat = [x for p in pairs for x in p]
    assert len(set(flat)) == 6
    # adjacent pairs scored by product of confidences; best disjoint triple must
    # avoid the weak races 3 and 8 if possible -> (1,2),(4,5),(6,7)
    assert pairs == [[1, 2], [4, 5], [6, 7]]


def test_pla_value_filter_with_place_odds():
    df = make_pred()
    # place odds: only horse 2 of each race is value (EV = p_place * odds >= 1)
    odds = {}
    for r, g in df.groupby("race_no"):
        odds[r] = {"PLA": {str(h): (1.05 / p if h == 2 else 0.8 / p)
                           for h, p in zip(g.horse_no, g.p_place)}}
    tk = st.build_tickets(df, odds=odds, config={"products": ["PLA_6x63"],
                                                  "PLA_6x63": {"value_filter": True}})
    assert all(l["selections"] == [2] for l in tk[0]["legs"])


def test_too_few_races_returns_nothing():
    df = make_pred(n_races=5)
    assert st.build_allup(df) is None
    assert st.build_multi_allup(df) == []
    assert st.build_tickets(df) == []


def test_scratched_rows_ignored():
    df = make_pred()
    df["scratched"] = 0
    df.loc[(df.race_no == 2) & (df.horse_no == 1), "scratched"] = 1
    t = st.build_allup(df)
    leg2 = [l for l in t["legs"] if l["race_no"] == 2]
    assert not leg2 or leg2[0]["selections"] != [1]


@pytest.mark.parametrize("spec,units", [
    ({"pool": "WIN", "top": 1}, 1),
    ({"pool": "PLA", "top": 2}, 2),
    ({"pool": "QIN", "box": 4}, comb(4, 2)),                       # 6
    ({"pool": "QPL", "bankers": 1, "legs": 5}, 5),
    ({"pool": "TRI", "box": 5}, comb(5, 3)),                       # 10
    ({"pool": "TRI", "bankers": 1, "legs": 4}, comb(4, 2)),        # 6
    ({"pool": "TRI", "bankers": 2, "legs": 6}, 6),
    ({"pool": "FF", "box": 6}, comb(6, 4)),                        # 15
    ({"pool": "FF", "bankers": 1, "legs": 5}, comb(5, 3)),         # 10
    ({"pool": "FF", "bankers": 2, "legs": 4}, comb(4, 2)),         # 6
    ({"pool": "TCE", "box": 4}, 4 * 3 * 2),                        # 24
    ({"pool": "TCE", "bankers": 1, "legs": 4, "mode": "single"}, 4 * 3),        # banker 1st: 12
    ({"pool": "TCE", "bankers": 1, "legs": 3, "mode": "multiple"}, 3 * 3 * 2),  # 3n(n-1) = 18
    # positions {1,2} / {1,2,3} / {1,2,3,4}: 1-2-3,1-2-4,1-3-2,1-3-4,2-1-3,2-1-4,2-3-1,2-3-4 = 8
    ({"pool": "TCE", "positions": [[1, 2], [1, 2, 3], [1, 2, 3, 4]]}, 8),
])
def test_single_ticket_units(spec, units):
    race = make_pred().query("race_no == 2")
    t = st.build_single(race, spec)
    assert t["n_combos"] == units and t["total_stake"] == units * 10
    assert t["spec"] is None and len(t["legs"]) == 1


def test_single_selection_ranks():
    race = make_pred().query("race_no == 2")
    leg = st.single_selection(race, {"pool": "TRI", "bankers": 1, "legs": 3})
    assert leg["bankers"] == [1] and leg["selections"] == [2, 3, 4]
    combos = st.leg_combos(leg)
    assert combos == ["1,2,3", "1,2,4", "1,3,4"]
    assert st.single_selection(race, {"pool": "FF", "box": 11}) is None   # 10 runners only
    # value_ev: horse 5 priced at 200 -> EV = p*200 >> 1.1 -> chosen
    r2 = race.copy()
    r2.loc[r2.horse_no == 5, "win_odds"] = 200.0
    leg = st.single_selection(r2, {"pool": "WIN", "value_ev": 1.1})
    assert leg["selections"] == [5]
    r3 = race.copy()
    r3["win_odds"] = 0.8 / r3["p_win"]          # EV = .8 everywhere -> no bet
    assert st.single_selection(r3, {"pool": "WIN", "value_ev": 1.1}) is None


def test_spec_names_unique():
    from hkjc.backtest import default_strategies
    for src in ("market", "wf-x"):
        names = [s["name"] for s in default_strategies(src)]
        assert len(names) == len(set(names))


# ===================================================================== ADAPTIVE
def _race(p_model, p_market=None, race_no=1, overround=1.0):
    """One race frame: model p_win, market odds = 1 / (p_market * overround)."""
    p_model = np.asarray(p_model, float)
    p_market = p_model if p_market is None else np.asarray(p_market, float)
    pl = st.harville(p_model)
    return pd.DataFrame({"race_no": race_no, "horse_no": np.arange(1, len(p_model) + 1),
                         "p_win": p_model, "p_place": pl,
                         "win_odds": 1.0 / (p_market * overround)})


NOCAL = {"lam2": 1.0, "lam3": 1.0, "calib": {}}


def test_raceprobs_consistency_and_qpl_hit():
    p = [0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.04, 0.03]
    for lam in ((1.0, 1.0), (0.85, 0.7)):
        r = st.RaceProbs(range(1, 9), p, *lam)
        H = range(1, 9)
        assert sum(r.prob("QIN", c) for c in bets.multiple("QIN", H)) == pytest.approx(1.0)
        assert sum(r.prob("QPL", c) for c in bets.multiple("QPL", H)) == pytest.approx(3.0)
        assert sum(r.prob("TRI", c) for c in bets.multiple("TRI", H)) == pytest.approx(1.0)
        assert sum(r.prob("FF", c) for c in bets.multiple("FF", H)) == pytest.approx(1.0)
        assert r.place.sum() == pytest.approx(3.0)
    r = st.RaceProbs(range(1, 9), p)
    np.testing.assert_allclose(r.place, st.harville(p))
    # QPL box {1,2,3}: hit = P(>=2 of them in top 3) - brute force over ordered triples
    brute = 0.0
    pp = np.array(p) / sum(p)
    for a, b, c in itertools.permutations(range(8), 3):
        pr = pp[a] * pp[b] / (1 - pp[a]) * pp[c] / (1 - pp[a] - pp[b])
        if len({a, b, c} & {0, 1, 2}) >= 2:
            brute += pr
    assert r.hit_prob("QPL", ["1,2", "1,3", "2,3"]) == pytest.approx(brute)


def test_ev_arithmetic_by_hand():
    # 7 runners; model = market, QIN live odds 5.0 for 1-2.
    p = [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]
    df = _race(p)
    df = st._ensure_market(df)
    odds = {"QIN": {"1,2": 5.0}, "WIN": {"1": 3.0}}
    cands = {c["name"]: c for c in st.race_candidates(df, odds, NOCAL)}
    # WIN top1: live odds 3.0 -> div $30; P = .4 -> E[return] = .4*30/10 = 1.2 -> ROI +20%
    assert cands["WIN top1"]["exp_roi"] == pytest.approx(0.2)
    assert cands["WIN top1"]["hit"] == pytest.approx(0.4)
    # QIN 1-2: P = .4*.2/.6 + .2*.4/.8 = .133333 + .1 = .233333; div $50 -> E = 1.16667
    q = cands["QIN 複式2"]
    assert q["hit"] == pytest.approx(0.2333333333)
    assert q["exp_roi"] == pytest.approx(0.2333333333 * 5.0 - 1)
    # market-implied dividend without live odds: 10*(1-0.175)/P_market(QIN 1-2) = 8.25/.233333
    est = st.DividendEstimator(df)
    assert est("QIN", "1,2") == pytest.approx(8.25 / 0.2333333333)
    # ... so a model equal to the market has expected ROI = -takeout on every such combo
    assert est.raw_market("TRI", "1,2,3") * st.RaceProbs(df.horse_no, p).prob("TRI", "1,2,3") / 10 \
        == pytest.approx(1 - 0.23)
    # calibration log(div) = a + b log(raw): a=log(2), b=1 doubles the estimate
    est2 = st.DividendEstimator(df, calib={"QIN": (np.log(2.0), 1.0)})
    assert est2("QIN", "1,2") == pytest.approx(2 * 8.25 / 0.2333333333)


def test_decide_race_dominant_horse_bets_banker_or_win():
    # model: horse 1 at 60%; market only 30% -> big edge on horse 1
    p_model = [0.60, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
    p_mkt = np.array([0.30, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08])
    df = _race(p_model, p_mkt / p_mkt.sum(), overround=1.21)
    t = st.decide_race(df, None, {**NOCAL, "tau": 0.0, "h_min": 0.10})
    assert len(t) == 1
    tk = t[0]
    leg = tk["legs"][0]
    assert tk["strategy"] == "ADAPTIVE" and tk["spec"] is None
    # either a straight WIN/PLA on horse 1 or a banker structure with horse 1 as banker
    assert (leg.get("bankers") == [1]) or (leg.get("selections") == [1] and tk["pool"] in ("WIN", "PLA"))
    assert tk["notes"]["exp_roi"] >= 0.0 and tk["notes"]["hit_prob"] >= 0.10
    assert tk["total_stake"] == tk["n_combos"] * 10 <= 200


def test_decide_race_flat_field_skips():
    # 12 equal horses, model == market with 17.5% overround -> every candidate has
    # negative expected ROI (WIN: 1/12 * 12/1.21 = 0.826) -> skip
    p = np.full(12, 1 / 12)
    df = _race(p, overround=1.21)
    assert st.decide_race(df, None, {**NOCAL, "tau": 0.0, "h_min": 0.05}) == []
    cands = st.race_candidates(st._ensure_market(df), None, NOCAL)
    assert cands and max(c["exp_roi"] for c in cands) < 0
    # no odds at all -> nothing can be priced -> skip
    assert st.decide_race(df.drop(columns=["win_odds"]), None, NOCAL) == []


def test_choose_candidate_thresholds():
    c = [{"exp_roi": 0.3, "hit": 0.05, "stake": 10, "n_units": 1},
         {"exp_roi": 0.1, "hit": 0.40, "stake": 30, "n_units": 3},
         {"exp_roi": 0.5, "hit": 0.50, "stake": 600, "n_units": 60}]
    assert st.choose_candidate(c, tau=0.0, h_min=0.0, max_stake=200)["exp_roi"] == 0.3
    assert st.choose_candidate(c, tau=0.0, h_min=0.1, max_stake=200)["exp_roi"] == 0.1
    assert st.choose_candidate(c, tau=0.2, h_min=0.1, max_stake=200) is None
    assert st.choose_candidate(c, tau=0.2, h_min=0.1, max_stake=None)["exp_roi"] == 0.5


def test_best_allup_by_hand():
    leg = lambda rn, E: {"race_no": rn, "E": E, "combos": ["x"], "hit": 0.7}
    # two legs E=1.2: 2x1 -> 1.44 per unit (ROI +44%); 2x3 -> (1.2+1.2+1.44)/3 = 1.28 (+28%)
    use, spec, roi = st.best_allup([leg(1, 1.2), leg(2, 1.2)])
    assert spec == "2x1" and roi == pytest.approx(0.44)
    # E = 0.9 and 1.5: 2x1 1.35 (+35%) vs 2x3 (0.9+1.5+1.35)/3 = 1.25 -> 2x1
    use, spec, roi = st.best_allup([leg(1, 0.9), leg(2, 1.5)])
    assert spec == "2x1" and roi == pytest.approx(0.35)
    # all legs < 1 -> best ROI negative -> no parlay (tau 0)
    assert st.best_allup([leg(1, 0.9), leg(2, 0.95)]) is None
    # three legs 1.1: 3x1 = 1.331 is the max
    use, spec, roi = st.best_allup([leg(1, 1.1), leg(2, 1.1), leg(3, 1.1)])
    assert spec == "3x1" and roi == pytest.approx(0.331)


def test_adaptive_allup_needs_two_qualifying_races():
    # strong race: model 70% vs market 25% on horse 1 -> PLA leg E = P_place * est_div/10 > 1
    p_model = [0.70, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03]
    p_mkt = np.array([0.25, 0.16, 0.14, 0.13, 0.12, 0.10, 0.10])
    strong = _race(p_model, p_mkt / p_mkt.sum(), race_no=1, overround=1.21)
    flat = _race(np.full(10, 0.1), race_no=2, overround=1.21)
    cfg = {**NOCAL, "pla_min_p": 0.5, "qpl_min_hit": 0.3, "min_edge": 1.05}
    assert st.build_adaptive_allup(pd.concat([strong, flat]), None, cfg) == []
    two = pd.concat([strong, strong.assign(race_no=3), flat])
    t = st.build_adaptive_allup(two, None, cfg)
    assert t and all(len(x["legs"]) == 2 and x["spec"] in ("2x1", "2x3") for x in t)
    assert {l["race_no"] for x in t for l in x["legs"]} == {1, 3}


def test_build_tickets_with_adaptive_products():
    p_model = [0.60, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
    p_mkt = np.array([0.30, 0.16, 0.14, 0.12, 0.11, 0.09, 0.08])
    races = [_race(p_model, p_mkt / p_mkt.sum(), race_no=r, overround=1.21) for r in range(1, 4)]
    races += [_race(np.full(12, 1 / 12), race_no=r, overround=1.21) for r in range(4, 9)]
    df = pd.concat(races, ignore_index=True)
    tk = st.build_tickets(df, config={"ADAPTIVE": {**NOCAL, "h_min": 0.1}})
    names = [t["strategy"] for t in tk]
    assert names.count("PLA_6x63") == 1 and names.count("QPL_2x3") == 3      # baseline kept
    ad = [t for t in tk if t["strategy"] == "ADAPTIVE"]
    assert sorted(t["legs"][0]["race_no"] for t in ad) == [1, 2, 3]          # flat races skipped
    only = st.build_tickets(df, config={"products": ["ADAPTIVE"], "ADAPTIVE": NOCAL})
    assert all(t["strategy"] == "ADAPTIVE" for t in only)


def test_race_profile():
    df = _race([0.5, 0.2, 0.1, 0.1, 0.05, 0.05], p_market=[0.25, 0.25, 0.2, 0.1, 0.1, 0.1])
    pr = st.race_profile(st._ensure_market(df))
    assert pr["p1"] == 0.5 and pr["gap"] == pytest.approx(0.3) and pr["top3"] == pytest.approx(0.8)
    assert pr["field"] == 6 and pr["top_horse"] == 1
    assert pr["edge1"] == pytest.approx(2.0)
