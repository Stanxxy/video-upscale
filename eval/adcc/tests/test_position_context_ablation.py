from __future__ import annotations

from eval.adcc import position_context_ablation as ablation


def test_balanced_subset_uses_two_frozen_windows_per_action_class():
    windows = ablation.load_selected_windows()

    counts: dict[str, int] = {}
    for window in windows:
        action_class = window["gold_event"]["action_class"]
        counts[action_class] = counts.get(action_class, 0) + 1

    assert counts == {
        "back_take": 2,
        "guard_pass": 2,
        "submission_arm_lock": 2,
        "submission_choke": 2,
        "submission_leg_lock": 2,
        "sweep": 2,
    }
    assert ablation.planned_call_count(windows) == 24
    ablation.validate_call_budget(windows, max_calls=24)


def test_position_treatment_is_advisory_and_baseline_does_not_leak_gold_position():
    window = ablation.load_selected_windows()[0]
    bucket = window["gold_event"]["position_bucket"]

    baseline = ablation.build_prompt(window, include_position_context=False)
    treatment = ablation.build_prompt(window, include_position_context=True)

    assert bucket not in baseline
    assert bucket in treatment
    assert "fallible" in treatment.lower()
    assert "visible body-movement evidence" in treatment
    assert window["gold_event"]["action_class"] not in baseline
    assert window["gold_event"]["action_class"] not in treatment


def test_summary_scores_exact_class_outcome_and_paired_discordance():
    records = [
        {
            "window_id": "w0",
            "gold": {"action_class": "sweep", "outcome": "successful"},
            "baseline": {"status": "ok", "prediction": {"action_class": "transition", "outcome": "successful"}},
            "position_context": {"status": "ok", "prediction": {"action_class": "sweep", "outcome": "successful"}},
        },
        {
            "window_id": "w1",
            "gold": {"action_class": "guard_pass", "outcome": "successful"},
            "baseline": {"status": "ok", "prediction": {"action_class": "guard_pass", "outcome": "successful"}},
            "position_context": {"status": "ok", "prediction": {"action_class": "back_take", "outcome": "successful"}},
        },
        {
            "window_id": "w2",
            "gold": {"action_class": "back_take", "outcome": "successful"},
            "baseline": {"status": "call_failed", "prediction": None},
            "position_context": {"status": "ok", "prediction": {"action_class": "back_take", "outcome": "successful"}},
        },
    ]

    summary = ablation.summarize_records(records)

    assert summary["paired_success_count"] == 2
    assert summary["failed_pair_count"] == 1
    assert summary["baseline"]["exact_action_correct"] == 1
    assert summary["position_context"]["exact_action_correct"] == 1
    assert summary["discordant"]["context_only_correct"] == 1
    assert summary["discordant"]["baseline_only_correct"] == 1
    assert summary["exact_action_net_delta"] == 0
    assert summary["paired_sign_test_p_value"] == 1.0


def test_budget_validation_fails_closed():
    windows = ablation.load_selected_windows()

    try:
        ablation.validate_call_budget(windows, max_calls=23)
    except ValueError as error:
        assert "24" in str(error)
    else:
        raise AssertionError("expected the call budget to fail closed")
