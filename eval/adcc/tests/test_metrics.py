from eval.adcc.score.metrics import (
    bootstrap_match_ci,
    filter_scoreable,
    pr_curve_over_confidence,
    prf1,
    run_track,
    summarize,
)
from eval.adcc.score.matcher import match_all

NON_SCOREABLE = frozenset({"transition", "guard_pull"})


def _gold(t_s, action_class, scoreable=True, match_id="m1"):
    return {
        "t_s": t_s, "action_class": action_class, "scoreable": scoreable,
        "compat": frozenset({action_class}), "match_id": match_id,
    }


def _pred(start_s, end_s, action_class, confidence=0.8, match_id="m1"):
    return {"start_s": start_s, "end_s": end_s, "action_class": action_class, "confidence": confidence, "match_id": match_id}


class TestPRF1:
    def test_basic(self):
        p, r, f1 = prf1(tp=8, fp=2, fn=2)
        assert p == 0.8
        assert r == 0.8
        assert abs(f1 - 0.8) < 1e-9

    def test_undefined_precision_when_no_predictions(self):
        p, r, f1 = prf1(tp=0, fp=0, fn=5)
        assert p is None
        assert r == 0.0
        assert f1 is None


class TestFilterThenMatch:
    def test_non_scoreable_gold_with_no_pred_is_not_an_fn_in_track_b(self):
        """A non-scoreable gold (e.g. residual `transition`) with nothing
        nearby must NOT count as an FN once dropped from Track B's
        denominator (filter-then-match, never filter-after-match)."""
        gold_by_match = {"m1": [
            _gold(10, "guard_pass", scoreable=True),
            _gold(50, "transition", scoreable=False),  # nothing predicted anywhere near this
        ]}
        pred_by_match = {"m1": [_pred(9, 11, "guard_pass")]}

        gold_b, pred_b = filter_scoreable(gold_by_match, pred_by_match, NON_SCOREABLE)
        outcomes = match_all(gold_b, pred_b, tau_t=2.0)
        summary = summarize(outcomes, gold_b, pred_b)

        assert summary["totals"]["gold_count"] == 1  # the transition row is GONE from the denominator
        assert summary["totals"]["fn"] == 0
        assert summary["totals"]["tp"] == 1

    def test_non_scoreable_pred_near_real_gold_is_not_a_competing_fp_in_track_b(self):
        """A `transition` prediction sitting right on top of a real
        scoreable gold must NOT be counted as a competing/duplicate FP once
        dropped from Track B's numerator."""
        gold_by_match = {"m1": [_gold(10, "guard_pass", scoreable=True)]}
        pred_by_match = {"m1": [
            _pred(9, 11, "guard_pass"),
            _pred(9, 11, "transition"),  # same window, non-scoreable class
        ]}

        gold_b, pred_b = filter_scoreable(gold_by_match, pred_by_match, NON_SCOREABLE)
        outcomes = match_all(gold_b, pred_b, tau_t=2.0)
        summary = summarize(outcomes, gold_b, pred_b)

        assert summary["totals"]["pred_count"] == 1  # the transition pred is GONE from the numerator
        assert summary["totals"]["fp"] == 0
        assert summary["totals"]["tp"] == 1

    def test_track_a_unfiltered_still_counts_transition_normally(self):
        """Sanity: Track A (no filtering) treats the same transition gold
        event as a normal FN candidate, unlike Track B."""
        gold_by_match = {"m1": [_gold(50, "transition", scoreable=False)]}
        pred_by_match = {"m1": []}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["totals"]["gold_count"] == 1
        assert summary["totals"]["fn"] == 1


class TestMicroMacro:
    def test_micro_aggregates_across_classes(self):
        gold_by_match = {"m1": [
            _gold(10, "guard_pass"),
            _gold(20, "sweep"),
        ]}
        pred_by_match = {"m1": [
            _pred(9, 11, "guard_pass"),
            _pred(500, 505, "sweep"),  # far away -> FN-silent for the sweep gold, FP-hallucinated for this pred
        ]}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["totals"]["tp"] == 1
        assert summary["totals"]["fn"] == 1
        assert summary["totals"]["fp"] == 1
        assert summary["micro"]["precision"] == 0.5
        assert summary["micro"]["recall"] == 0.5

    def test_macro_excludes_undefined_classes_not_coerces_to_zero(self):
        # guard_pass: perfect. sweep: gold exists, never predicted -> precision undefined for sweep.
        gold_by_match = {"m1": [
            _gold(10, "guard_pass"), _gold(20, "sweep"),
        ]}
        pred_by_match = {"m1": [_pred(9, 11, "guard_pass")]}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["per_class"]["sweep"]["precision"] is None
        assert summary["per_class"]["sweep"]["recall"] == 0.0
        assert summary["per_class"]["guard_pass"]["precision"] == 1.0
        # macro precision averages only over classes with a DEFINED precision (guard_pass only)
        assert summary["macro"]["precision"] == 1.0


class TestConfusionMatrix:
    def test_fn_misclassified_attributes_to_nearest_pred_class_not_no_match(self):
        gold_by_match = {"m1": [_gold(10, "sweep")]}
        pred_by_match = {"m1": [_pred(9, 11, "guard_pass")]}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["confusion_matrix"]["sweep"]["guard_pass"] == 1

    def test_fn_silent_attributes_to_no_match_column(self):
        gold_by_match = {"m1": [_gold(10, "sweep")]}
        pred_by_match = {"m1": []}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["confusion_matrix"]["sweep"]["NO_MATCH"] == 1

    def test_fp_hallucinated_attributes_to_no_match_row(self):
        gold_by_match = {"m1": []}
        pred_by_match = {"m1": [_pred(9, 11, "guard_pass")]}
        summary = run_track(gold_by_match, pred_by_match, tau_t=2.0)
        assert summary["confusion_matrix"]["NO_MATCH"]["guard_pass"] == 1


def _mixed_multi_match_dataset():
    """6 matches with varying TP/FP/FN mixes -> real cross-match variance for
    the bootstrap to estimate a non-degenerate CI over."""
    gold_by_match = {
        "m1": [_gold(10, "guard_pass", match_id="m1")],  # TP
        "m2": [_gold(10, "guard_pass", match_id="m2")],  # TP
        "m3": [_gold(10, "guard_pass", match_id="m3")],  # FN (nothing predicted)
        "m4": [_gold(10, "sweep", match_id="m4")],        # TP
        "m5": [_gold(10, "sweep", match_id="m5")],        # FN
        "m6": [_gold(10, "submission_choke", match_id="m6")],  # TP
    }
    pred_by_match = {
        "m1": [_pred(9, 11, "guard_pass", match_id="m1")],
        "m2": [_pred(9, 11, "guard_pass", match_id="m2")],
        "m3": [_pred(500, 505, "guard_pass", match_id="m3")],  # FP-hallucinated, far away
        "m4": [_pred(9, 11, "sweep", match_id="m4")],
        "m5": [],
        "m6": [_pred(9, 11, "submission_choke", match_id="m6")],
    }
    return gold_by_match, pred_by_match


class TestBootstrapMatchCI:
    def test_ci_brackets_the_point_estimate(self):
        gold_by_match, pred_by_match = _mixed_multi_match_dataset()
        result = bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=500, seed=42)

        p_point, r_point, f1_point = result["point"]["precision"], result["point"]["recall"], result["point"]["f1"]
        p_lo, p_hi = result["precision_ci"]
        r_lo, r_hi = result["recall_ci"]
        f1_lo, f1_hi = result["f1_ci"]

        assert p_lo <= p_point <= p_hi
        assert r_lo <= r_point <= r_hi
        assert f1_lo <= f1_point <= f1_hi

    def test_deterministic_under_fixed_seed(self):
        gold_by_match, pred_by_match = _mixed_multi_match_dataset()
        result_a = bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=300, seed=7)
        result_b = bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=300, seed=7)
        assert result_a["precision_ci"] == result_b["precision_ci"]
        assert result_a["recall_ci"] == result_b["recall_ci"]
        assert result_a["f1_ci"] == result_b["f1_ci"]

    def test_different_seed_can_differ(self):
        gold_by_match, pred_by_match = _mixed_multi_match_dataset()
        result_a = bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=300, seed=1)
        result_b = bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=300, seed=2)
        # Same point estimate (full-sample, not bootstrapped) regardless of seed...
        assert result_a["point"] == result_b["point"]
        # ...but the resampled CI bounds are free to differ across seeds.
        assert result_a["recall_ci"] != result_b["recall_ci"] or result_a["precision_ci"] != result_b["precision_ci"]

    def test_empty_match_set_returns_none_ci_not_a_crash(self):
        result = bootstrap_match_ci({}, {}, tau_t=2.0, n_resamples=50, seed=1)
        assert result["precision_ci"] is None
        assert result["recall_ci"] is None
        assert result["f1_ci"] is None

    def test_resample_preserves_match_partition_never_mixes_across_matches(self):
        """A resample draw of match m1 twice must not accidentally let m1's
        gold compete against m3's predictions -- each occurrence gets its own
        synthetic match_id bucket, so match_all's per-match_id partition still
        holds inside every resample."""
        gold_by_match, pred_by_match = _mixed_multi_match_dataset()
        # Just exercise a large n_resamples to shake out any cross-match
        # leakage (which would show up as an assertion error raised from
        # match_all's own tag-consistency check inside bootstrap_match_ci).
        bootstrap_match_ci(gold_by_match, pred_by_match, tau_t=2.0, n_resamples=200, seed=99)


class TestPRCurve:
    def test_higher_confidence_threshold_can_only_reduce_or_hold_recall(self):
        gold_by_match = {"m1": [_gold(10, "guard_pass")]}
        pred_by_match = {"m1": [_pred(9, 11, "guard_pass", confidence=0.4)]}
        curve = pr_curve_over_confidence(gold_by_match, pred_by_match, tau_t=2.0, thresholds=(0.0, 0.5, 1.0))
        by_tau_c = {c["tau_c"]: c for c in curve}
        assert by_tau_c[0.0]["recall"] == 1.0
        assert by_tau_c[0.5]["recall"] == 0.0  # the 0.4-confidence pred is filtered out at tau_c=0.5
        assert by_tau_c[1.0]["recall"] == 0.0
