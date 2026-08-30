import pytest

from eval.adcc.score.matcher import match_all, match_one


def _gold(t_s, action_class, match_id="m1"):
    return {"t_s": t_s, "action_class": action_class, "compat": frozenset({action_class}), "match_id": match_id}


def _pred(start_s, end_s, action_class, match_id="m1", confidence=0.9):
    return {"start_s": start_s, "end_s": end_s, "action_class": action_class, "match_id": match_id, "confidence": confidence}


class TestBasicTP:
    def test_point_inside_span_is_tp(self):
        gold = [_gold(10, "guard_pass")]
        pred = [_pred(8, 12, "guard_pass")]
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.tp == [(0, 0)]
        assert not outcome.fn_silent and not outcome.fn_misclassified and not outcome.fn_competed_away
        assert not outcome.fp_hallucinated and not outcome.fp_duplicate


class TestToleranceBoundary:
    def test_exactly_at_lower_boundary_is_feasible(self):
        # gold.t_s == pred.start_s - tau_t (inclusive per plan §5.1)
        gold = [_gold(8, "sweep")]
        pred = [_pred(10, 20, "sweep")]  # start_s - tau_t = 10 - 2 = 8
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.tp == [(0, 0)]

    def test_just_outside_lower_boundary_is_infeasible(self):
        gold = [_gold(7, "sweep")]
        pred = [_pred(10, 20, "sweep")]  # 10 - 2 = 8 > 7 -> infeasible
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.tp == []
        assert outcome.fn_silent == [0]

    def test_exactly_at_upper_boundary_is_feasible(self):
        gold = [_gold(22, "sweep")]
        pred = [_pred(10, 20, "sweep")]  # end_s + tau_t = 20 + 2 = 22
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.tp == [(0, 0)]


class TestOneToOneDuplicate:
    def test_two_preds_one_gold_yields_one_tp_one_fp_duplicate(self):
        gold = [_gold(10, "guard_pass")]
        pred = [_pred(9, 11, "guard_pass", confidence=0.6), _pred(9.5, 10.5, "guard_pass", confidence=0.9)]
        outcome = match_one(gold, pred, tau_t=2.0)
        assert len(outcome.tp) == 1
        assert len(outcome.fp_duplicate) == 1
        assert not outcome.fp_hallucinated
        # the winner should be the lower-cost (closer/inside) one; both are inside span here so cost ties at 0 —
        # scipy breaks ties deterministically; just assert exactly one TP + one duplicate total.
        assigned_pred_idx = outcome.tp[0][1]
        other_idx = 1 - assigned_pred_idx
        assert outcome.fp_duplicate == [other_idx]


class TestFNSubtypes:
    def test_fn_silent_when_nothing_time_feasible(self):
        gold = [_gold(100, "guard_pass")]
        pred = [_pred(0, 5, "guard_pass")]
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.fn_silent == [0]
        assert outcome.fn_misclassified == []

    def test_fn_misclassified_when_time_feasible_but_wrong_class(self):
        gold = [_gold(10, "sweep")]
        pred = [_pred(9, 11, "guard_pass")]  # right there in time, wrong class
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.fn_misclassified == [0]
        assert outcome.fn_misclassified_nearest_pred == {0: 0}
        assert outcome.fp_hallucinated == [0]  # the guard_pass pred has no feasible gold of its own class either

    def test_fn_competed_away_when_feasible_pred_lost_to_another_gold(self):
        # Two gold events, both class-compatible with the single pred, both
        # time-feasible; the pred can only win one assignment.
        gold = [_gold(10, "guard_pass"), _gold(10.1, "guard_pass")]
        pred = [_pred(9, 11, "guard_pass")]
        outcome = match_one(gold, pred, tau_t=2.0)
        assert len(outcome.tp) == 1
        assert len(outcome.fn_competed_away) == 1
        assert outcome.fn_silent == [] and outcome.fn_misclassified == []


class TestFPSubtypes:
    def test_fp_hallucinated_when_no_feasible_gold_at_all(self):
        gold = [_gold(10, "guard_pass")]
        pred = [_pred(9, 11, "guard_pass"), _pred(500, 505, "sweep")]
        outcome = match_one(gold, pred, tau_t=2.0)
        assert outcome.tp == [(0, 0)]
        assert outcome.fp_hallucinated == [1]


class TestMatchAll:
    def test_never_matches_across_match_ids(self):
        gold_by_match = {
            "m1": [_gold(10, "guard_pass", match_id="m1")],
            "m2": [_gold(10, "guard_pass", match_id="m2")],
        }
        pred_by_match = {
            "m1": [_pred(9, 11, "guard_pass", match_id="m1")],
            "m2": [_pred(9, 11, "guard_pass", match_id="m2")],
        }
        results = match_all(gold_by_match, pred_by_match, tau_t=2.0)
        assert set(results.keys()) == {"m1", "m2"}
        assert results["m1"].tp == [(0, 0)]
        assert results["m2"].tp == [(0, 0)]

    def test_mistagged_event_triggers_hard_assertion(self):
        gold_by_match = {"m1": [_gold(10, "guard_pass", match_id="WRONG_ID")]}
        pred_by_match = {"m1": []}
        with pytest.raises(AssertionError):
            match_all(gold_by_match, pred_by_match, tau_t=2.0)
