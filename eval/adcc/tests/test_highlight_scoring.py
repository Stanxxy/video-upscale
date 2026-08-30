import json

import pytest

from eval.adcc.score import highlight_scoring as hs
from eval.adcc.score.highlight_scoring import (
    HIGHLIGHT_CLASS,
    effective_pred_events,
    filter_to_highlights,
    load_highlight_gold_by_match,
    mean_abs_center_error,
    original_pred_events,
    render_timeline_marks,
    score_highlights,
)
from eval.adcc.score.matcher import match_all, match_one


def _gold_gt(match_id, t_s_values):
    return {
        match_id: [
            {"t_s": t_s, "action_class": "submission_choke", "match_id": match_id, "compat": frozenset({HIGHLIGHT_CLASS})}
            for t_s in t_s_values
        ]
    }


class TestFilterToHighlights:
    def test_tags_every_prediction_with_highlight_class(self):
        pred_by_match = {"m1": [{"start_s": 10.0, "end_s": 10.0}, {"start_s": 20.0, "end_s": 25.0, "action_class": "junk"}]}
        out = filter_to_highlights(pred_by_match)
        assert all(p["action_class"] == HIGHLIGHT_CLASS for p in out["m1"])

    def test_preserves_other_fields_and_match_id(self):
        pred_by_match = {"m1": [{"start_s": 10.0, "end_s": 10.0, "confidence": 0.9}]}
        out = filter_to_highlights(pred_by_match)
        assert out["m1"][0]["confidence"] == 0.9
        assert out["m1"][0]["match_id"] == "m1"

    def test_never_mutates_the_input(self):
        pred_by_match = {"m1": [{"start_s": 10.0, "end_s": 10.0}]}
        original = json.dumps(pred_by_match, default=str)
        filter_to_highlights(pred_by_match)
        assert json.dumps(pred_by_match, default=str) == original


class TestClassCheckDegradesToNoop:
    """The frozen contract's core claim: with GT compat={'highlight'} and
    every prediction tagged action_class='highlight', matcher.match_one's
    class check is always satisfied, so ONLY temporal feasibility decides
    matching — verified directly against the UNCHANGED matcher.match_all."""

    def test_a_temporally_close_prediction_matches_regardless_of_original_action_class(self):
        gold_by_match = _gold_gt("m1", [100.0])
        raw_preds = {"m1": [{"start_s": 99.0, "end_s": 101.0, "action_class": "anything_at_all"}]}
        pred_by_match = filter_to_highlights(raw_preds)
        outcomes = match_all(gold_by_match, pred_by_match, tau_t=2.0)
        assert outcomes["m1"].tp == [(0, 0)]

    def test_a_temporally_distant_prediction_still_misses(self):
        gold_by_match = _gold_gt("m1", [100.0])
        raw_preds = {"m1": [{"start_s": 500.0, "end_s": 500.0, "action_class": "anything_at_all"}]}
        pred_by_match = filter_to_highlights(raw_preds)
        outcomes = match_all(gold_by_match, pred_by_match, tau_t=2.0)
        assert outcomes["m1"].tp == []
        assert outcomes["m1"].fn_silent == [0]
        assert outcomes["m1"].fp_hallucinated == [0]


class TestScoreHighlightsKnownFixture:
    """A synthetic fixture with a KNOWN TP/FN/FP outcome (frozen contract's
    "highlight scoring on a synthetic fixture with known TP/FN/FP" test
    requirement): 2 gold highlights, 2 predictions — one true positive
    (temporally close), one false negative (nothing near it), one false
    positive (hallucinated, nothing near any gold)."""

    def _fixture(self):
        gold_by_match = _gold_gt("m1", [100.0, 500.0])  # 500.0 will go unmatched -> FN
        raw_preds = {"m1": [
            {"start_s": 99.0, "end_s": 101.0},  # TP against t_s=100
            {"start_s": 900.0, "end_s": 900.0},  # FP hallucinated
        ]}
        return gold_by_match, raw_preds

    def test_known_tp_fn_fp_counts(self):
        gold_by_match, raw_preds = self._fixture()
        result = score_highlights(gold_by_match, raw_preds, tau_t=2.0)
        totals = result["summary"]["totals"]
        assert totals["tp"] == 1
        assert totals["fn"] == 1
        assert totals["fp"] == 1
        assert totals["gold_count"] == 2
        assert totals["pred_count"] == 2

    def test_precision_recall_f1_from_known_counts(self):
        gold_by_match, raw_preds = self._fixture()
        result = score_highlights(gold_by_match, raw_preds, tau_t=2.0)
        micro = result["summary"]["micro"]
        assert micro["precision"] == pytest.approx(0.5)  # 1 tp / (1 tp + 1 fp)
        assert micro["recall"] == pytest.approx(0.5)  # 1 tp / (1 tp + 1 fn)
        assert micro["f1"] == pytest.approx(0.5)

    def test_outcomes_returned_for_downstream_rendering(self):
        gold_by_match, raw_preds = self._fixture()
        result = score_highlights(gold_by_match, raw_preds, tau_t=2.0)
        assert "m1" in result["outcomes"]
        assert result["outcomes"]["m1"].tp == [(0, 0)]


class TestLoadHighlightGoldByMatch:
    def test_attaches_singleton_highlight_compat(self, tmp_path):
        path = tmp_path / "highlight_gold.jsonl"
        path.write_text(json.dumps({
            "match_id": "m1", "video_url": "x", "video_url_md5": "y", "name": "n",
            "events": [{"t_s": 10, "action_class": "sweep", "outcome": "successful"}],
        }) + "\n")
        gold_by_match = load_highlight_gold_by_match(path)
        assert gold_by_match["m1"][0]["compat"] == frozenset({HIGHLIGHT_CLASS})

    def test_zero_highlight_match_yields_empty_list_not_missing_key(self, tmp_path):
        path = tmp_path / "highlight_gold.jsonl"
        path.write_text(json.dumps({
            "match_id": "m_empty", "video_url": "x", "video_url_md5": "y", "name": "n", "events": [],
        }) + "\n")
        gold_by_match = load_highlight_gold_by_match(path)
        assert gold_by_match["m_empty"] == []

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "highlight_gold.jsonl"
        rec = json.dumps({"match_id": "m1", "video_url": "x", "video_url_md5": "y", "name": "n", "events": []})
        path.write_text("\n" + rec + "\n\n")
        gold_by_match = load_highlight_gold_by_match(path)
        assert list(gold_by_match.keys()) == ["m1"]


class TestRenderTimelineMarks:
    def test_marks_carry_correct_status_and_pairs(self):
        gold_by_match, raw_preds = TestScoreHighlightsKnownFixture()._fixture()
        result = score_highlights(gold_by_match, raw_preds, tau_t=2.0)
        marks = render_timeline_marks(
            "m1", result["gold_by_match"]["m1"], result["pred_by_match"]["m1"], result["outcomes"]["m1"],
        )
        gold_statuses = {m["index"]: m["status"] for m in marks["gold_marks"]}
        assert gold_statuses[0] == "tp"
        assert gold_statuses[1] == "fn_silent"
        pred_statuses = {m["index"]: m["status"] for m in marks["pred_marks"]}
        assert pred_statuses[0] == "tp"
        assert pred_statuses[1] == "fp_hallucinated"
        assert marks["pairs"] == [{"gold_index": 0, "pred_index": 0}]

    def test_every_index_covered_no_unclassified_marks(self):
        gold_by_match, raw_preds = TestScoreHighlightsKnownFixture()._fixture()
        result = score_highlights(gold_by_match, raw_preds, tau_t=2.0)
        marks = render_timeline_marks(
            "m1", result["gold_by_match"]["m1"], result["pred_by_match"]["m1"], result["outcomes"]["m1"],
        )
        assert len(marks["gold_marks"]) == 2
        assert len(marks["pred_marks"]) == 2


class TestEffectiveAndOriginalPredEvents:
    """Timestamp critique/refinement (frozen contract §G, LeCun's scoring
    policy): score the EFFECTIVE timestamp (refined_* when refined=true,
    else original_*); both original-only and effective must be derivable
    from ONE persisted file for a before/after comparison."""

    def _enriched(self):
        return [
            {  # accepted refinement -- effective uses refined_*
                "original_start_s": 10.0, "original_end_s": 12.0,
                "refined_start_s": 9.0, "refined_end_s": 11.0, "refined": True,
                "reason": "r1", "action_confirmed": True, "refinement_failed": False,
            },
            {  # refine off / failed -- effective falls back to original_*
                "original_start_s": 50.0, "original_end_s": 52.0,
                "refined_start_s": None, "refined_end_s": None, "refined": False,
                "reason": "r2", "action_confirmed": None, "refinement_failed": False,
            },
        ]

    def test_effective_uses_refined_when_refined_true(self):
        events = effective_pred_events(self._enriched())
        assert events[0] == {"start_s": 9.0, "end_s": 11.0}

    def test_effective_falls_back_to_original_when_not_refined(self):
        events = effective_pred_events(self._enriched())
        assert events[1] == {"start_s": 50.0, "end_s": 52.0}

    def test_original_pred_events_always_uses_original_regardless_of_refined(self):
        events = original_pred_events(self._enriched())
        assert events[0] == {"start_s": 10.0, "end_s": 12.0}
        assert events[1] == {"start_s": 50.0, "end_s": 52.0}

    def test_legacy_flat_shape_falls_back_to_start_s_end_s(self):
        legacy = [{"start_s": 5.0, "end_s": 5.0, "reason": "legacy"}]
        assert effective_pred_events(legacy) == [{"start_s": 5.0, "end_s": 5.0}]
        assert original_pred_events(legacy) == [{"start_s": 5.0, "end_s": 5.0}]

    def test_order_is_preserved_for_index_cross_referencing(self):
        enriched = self._enriched()
        events = effective_pred_events(enriched)
        # 1:1 only when no rejected entries; rejected skips break index correspondence
        # (callers must filter enriched the same way before zipping — Task 5).
        assert len(events) == len(enriched)


class TestEffectivePredEventsSkipsRejected:
    def test_rejected_highlight_is_excluded_never_falls_back_to_original(self):
        enriched = [
            {"original_start_s": 148.0, "original_end_s": 152.0, "refined_start_s": None,
             "refined_end_s": None, "refined": False, "reason": "x", "action_confirmed": None,
             "refinement_failed": False, "rejected": True, "rejection_reason": "out_of_range"},
        ]
        assert hs.effective_pred_events(enriched) == []
        assert hs.original_pred_events(enriched) == []

    def test_valid_and_rejected_mixed_only_the_valid_one_is_scoreable(self):
        enriched = [
            {"original_start_s": 10.0, "original_end_s": 20.0, "refined_start_s": 15.0,
             "refined_end_s": 18.0, "refined": True, "reason": "x", "action_confirmed": True,
             "refinement_failed": False, "rejected": False, "rejection_reason": None},
            {"original_start_s": 148.0, "original_end_s": 152.0, "refined_start_s": None,
             "refined_end_s": None, "refined": False, "reason": "x", "action_confirmed": None,
             "refinement_failed": False, "rejected": True, "rejection_reason": "out_of_range"},
        ]
        assert hs.effective_pred_events(enriched) == [{"start_s": 15.0, "end_s": 18.0}]

    def test_legacy_persisted_highlight_with_no_rejected_key_is_never_filtered(self):
        # Predates this increment entirely -- no rejected key at all.
        legacy = [{"start_s": 10.0, "end_s": 10.0}]
        assert hs.effective_pred_events(legacy) == [{"start_s": 10.0, "end_s": 10.0}]
        assert hs.original_pred_events(legacy) == [{"start_s": 10.0, "end_s": 10.0}]

    def test_all_rejected_match_scores_as_a_real_zero_prediction_match_not_a_crash(self):
        enriched = [
            {"original_start_s": 148.0, "original_end_s": 152.0, "refined_start_s": None,
             "refined_end_s": None, "refined": False, "reason": "x", "action_confirmed": None,
             "refinement_failed": False, "rejected": True, "rejection_reason": "out_of_range"},
        ]
        pred_by_match = {"m1": hs.effective_pred_events(enriched)}
        # action_class required by metrics.summarize's FN path (same shape as _gold_gt).
        gold = {"m1": [{
            "t_s": 10.0,
            "action_class": "submission_choke",
            "match_id": "m1",
            "compat": frozenset({hs.HIGHLIGHT_CLASS}),
        }]}
        result = hs.score_highlights(gold, pred_by_match, tau_t=5.0)
        assert result["summary"]["totals"]["tp"] == 0
        assert result["summary"]["totals"]["fn"] == 1  # the gold event, correctly unmatched
        assert result["summary"]["totals"]["fp"] == 0  # rejected must NOT count as hallucinated FP


class TestMeanAbsCenterError:
    """A small LeCun-requested diagnostic (frozen contract §G) — never a new
    matching pass, just a re-slice of an already-computed MatchOutcome."""

    def test_mean_abs_center_error_over_a_known_tp_pair(self):
        gold_events = [{"t_s": 100.0, "compat": frozenset({HIGHLIGHT_CLASS}), "match_id": "m1"}]
        pred_events = [{"start_s": 96.0, "end_s": 100.0, "action_class": HIGHLIGHT_CLASS, "match_id": "m1"}]
        outcome = match_one(gold_events, pred_events, tau_t=5.0)
        assert outcome.tp == [(0, 0)]
        # pred_center = (96+100)/2 = 98; |98 - 100| = 2
        assert mean_abs_center_error(gold_events, pred_events, outcome) == pytest.approx(2.0)

    def test_returns_none_when_no_tp_pairs(self):
        gold_events = [{"t_s": 100.0, "compat": frozenset({HIGHLIGHT_CLASS}), "match_id": "m1"}]
        pred_events = [{"start_s": 900.0, "end_s": 900.0, "action_class": HIGHLIGHT_CLASS, "match_id": "m1"}]
        outcome = match_one(gold_events, pred_events, tau_t=5.0)
        assert outcome.tp == []
        assert mean_abs_center_error(gold_events, pred_events, outcome) is None

    def test_original_vs_effective_before_after_comparison_is_runnable(self):
        """The LeCun 3-way-verdict use case: compute this metric TWICE,
        once per prediction set, from the SAME persisted enriched
        highlights -- no UI wiring, just confirms the plumbing works
        end-to-end."""
        enriched = [{
            "original_start_s": 90.0, "original_end_s": 90.0,
            "refined_start_s": 98.0, "refined_end_s": 98.0, "refined": True,
            "reason": "r", "action_confirmed": True, "refinement_failed": False,
        }]
        gold_events = [{"t_s": 100.0, "compat": frozenset({HIGHLIGHT_CLASS}), "match_id": "m1"}]

        original_preds = [{**e, "action_class": HIGHLIGHT_CLASS, "match_id": "m1"} for e in original_pred_events(enriched)]
        effective_preds = [{**e, "action_class": HIGHLIGHT_CLASS, "match_id": "m1"} for e in effective_pred_events(enriched)]

        original_outcome = match_one(gold_events, original_preds, tau_t=15.0)
        effective_outcome = match_one(gold_events, effective_preds, tau_t=15.0)

        original_error = mean_abs_center_error(gold_events, original_preds, original_outcome)
        effective_error = mean_abs_center_error(gold_events, effective_preds, effective_outcome)

        # Refinement moved the prediction from 90 (10s off) to 98 (2s off) --
        # the effective (refined) error must be smaller, proving the metric
        # actually differentiates before vs. after.
        assert effective_error < original_error
