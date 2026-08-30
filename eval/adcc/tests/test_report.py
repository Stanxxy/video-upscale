import json

from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.score.report import _render_report_md, build_report

MAPPING = load_mapping()


def _gold_records():
    return [
        {
            "match_id": "m_dev_1", "video_url": "https://x/1", "video_url_md5": "stem1",
            "name": "A vs B",
            "events": [
                {"t_s": 10, "action_class": "guard_pass", "position_bucket": "open_guard",
                 "outcome": "successful", "scoreable": True, "track": "hard",
                 "raw_sequenceType": "GUARD PASS", "raw_positions": {"starting": "OPEN GUARD", "ending": "SIDE CONTROL"},
                 "raw_movements": "KNEE CUT"},
                {"t_s": 100, "action_class": "submission_choke", "position_bucket": "back_control",
                 "outcome": "successful", "scoreable": True, "track": "hard",
                 "raw_sequenceType": "SUBMISSION", "raw_positions": {"starting": "BACK", "ending": "RNC"},
                 "raw_movements": "N/A"},
                {"t_s": 200, "action_class": "transition", "position_bucket": "open_guard",
                 "outcome": "unclear", "scoreable": False, "track": "soft",
                 "raw_sequenceType": "POSITION ENTRY", "raw_positions": {"starting": "OPEN GUARD", "ending": "TURTLE"},
                 "raw_movements": "CIRCLE"},
            ],
        },
        {
            "match_id": "m_test_1", "video_url": "https://x/2", "video_url_md5": "stem2",
            "name": "C vs D",
            "events": [
                {"t_s": 50, "action_class": "sweep", "position_bucket": "closed_guard",
                 "outcome": "successful", "scoreable": True, "track": "hard",
                 "raw_sequenceType": "SWEEP", "raw_positions": {"starting": "CLOSED GUARD", "ending": "MOUNT"},
                 "raw_movements": "SCISSOR"},
            ],
        },
    ]


def _split_assignment():
    return {
        "seed": 1, "ratios": {"sandbox": 0.7, "dev": 0.15, "test": 0.15},
        "video_counts": {"dev": 1, "test": 1},
        "assignment": {
            "stem1": {"split": "dev", "match_ids": ["m_dev_1"], "event_count": 3},
            "stem2": {"split": "test", "match_ids": ["m_test_1"], "event_count": 1},
        },
    }


def test_build_report_end_to_end_with_synthetic_predictions(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [
            {"start_s": 9, "end_s": 11, "action_class": "guard_pass", "confidence": 0.9, "outcome": "successful"},
            {"start_s": 200, "end_s": 202, "action_class": "transition", "confidence": 0.3},  # near a non-scoreable gold
        ],
    }))
    (predictions_dir / "m_test_1.json").write_text(json.dumps({
        "match_id": "m_test_1",
        "clips": [
            {"start_s": 300, "end_s": 305, "action_class": "guard_pass", "confidence": 0.5},  # hallucinated FP
        ],
    }))

    report = build_report(
        _gold_records(), _split_assignment(), predictions_dir, MAPPING,
        split="all", tau_t=5.0,
    )

    assert set(report["per_split"].keys()) == {"dev", "test"}
    dev = report["per_split"]["dev"]
    # guard_pass matched, transition/transition pair correctly excluded from Track B,
    # submission_choke has zero predictions -> FN.
    assert dev["track_b"]["totals"]["tp"] == 1
    assert dev["track_b"]["totals"]["gold_count"] == 2  # guard_pass + submission_choke (transition dropped)
    assert dev["track_b"]["totals"]["pred_count"] == 1  # transition pred dropped

    test_split = report["per_split"]["test"]
    assert test_split["track_b"]["totals"]["fp"] == 1  # the far-away guard_pass pred is a hallucinated FP
    assert test_split["track_b"]["totals"]["fn"] == 1   # the real sweep gold has nothing nearby

    # report.md renders without raising, and contains both split sections
    md = _render_report_md(report)
    assert "Split: dev" in md
    assert "Split: test" in md
    assert "Named submission misses" in md


def test_build_report_includes_track_b_bootstrap_ci_in_md_and_json(tmp_path):
    """B1: the report headline carries a bootstrap 95% CI (over matches) for
    Track-B micro P/R/F1, and it shows up in the rendered report.md."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [{"start_s": 9, "end_s": 11, "action_class": "guard_pass", "confidence": 0.9}],
    }))

    report = build_report(
        _gold_records(), _split_assignment(), predictions_dir, MAPPING,
        split="dev", tau_t=5.0, bootstrap_n=50, bootstrap_seed=1,
    )
    dev = report["per_split"]["dev"]
    assert "track_b_ci" in dev
    assert dev["track_b_ci"]["n_resamples"] == 50
    assert dev["track_b_ci"]["seed"] == 1
    # point estimate matches the non-bootstrapped Track-B micro precision
    assert dev["track_b_ci"]["point"]["precision"] == dev["track_b"]["micro"]["precision"]

    md = _render_report_md(report)
    assert "95% CI over matches" in md
    assert "n=50 resamples" in md
    assert "seed=1" in md


def test_build_report_surfaces_dev_test_match_count_asymmetry_caveat(tmp_path):
    """B2: dev (1 match) vs test (1 match) here are equal, so build an
    intentionally asymmetric split to prove the caveat line fires."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()

    gold_records = _gold_records() + [
        {
            "match_id": "m_dev_2", "video_url": "https://x/3", "video_url_md5": "stem3",
            "name": "E vs F",
            "events": [
                {"t_s": 30, "action_class": "escape", "position_bucket": "side_control",
                 "outcome": "successful", "scoreable": True, "track": "hard",
                 "raw_sequenceType": "POSITION ESCAPE", "raw_positions": {"starting": "SIDE CONTROL", "ending": "OPEN GUARD"},
                 "raw_movements": "HIP ESCAPE"},
            ],
        },
    ]
    split_assignment = _split_assignment()
    split_assignment["assignment"]["stem3"] = {"split": "dev", "match_ids": ["m_dev_2"], "event_count": 1}

    report = build_report(
        gold_records, split_assignment, predictions_dir, MAPPING,
        split="all", tau_t=5.0, bootstrap_n=20, bootstrap_seed=1,
    )
    assert report["per_split"]["dev"]["match_count"] == 2
    assert report["per_split"]["test"]["match_count"] == 1

    md = _render_report_md(report)
    assert "dev/test match-count asymmetry" in md


def test_build_report_missing_prediction_file_means_all_fn(tmp_path):
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    # no prediction files at all for either match

    report = build_report(_gold_records(), _split_assignment(), predictions_dir, MAPPING, split="dev", tau_t=5.0)
    dev = report["per_split"]["dev"]
    assert dev["track_b"]["totals"]["pred_count"] == 0
    assert dev["track_b"]["totals"]["fn"] == dev["track_b"]["totals"]["gold_count"]


def test_track_b_per_class_table_renders_all_scoreable_classes(tmp_path):
    """Gap 1: report.md must render a per-class P/R/F1 (+TP/FN/FP) table for
    Track B, not just the micro/macro aggregates."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [{"start_s": 9, "end_s": 11, "action_class": "guard_pass", "confidence": 0.9}],
    }))

    report = build_report(_gold_records(), _split_assignment(), predictions_dir, MAPPING, split="dev", tau_t=5.0)
    dev = report["per_split"]["dev"]

    per_class = dev["track_b"]["per_class"]
    assert "guard_pass" in per_class
    assert "submission_choke" in per_class
    assert per_class["guard_pass"]["tp_for_recall"] == 1
    assert per_class["submission_choke"]["fn"] == 1  # never predicted -> FN, no matching pred file entry

    md = _render_report_md(report)
    assert "Per-class P/R/F1 (Track B" in md
    # header carries the counts columns Jobs asked for, not just ratios
    assert "tp (recall-side)" in md and "fp" in md
    assert "| guard_pass |" in md
    assert "| submission_choke |" in md


def test_submission_family_recall_headline_combines_the_three_submission_classes(tmp_path):
    """Gap 2: combined submission_choke+arm_lock+leg_lock recall must be a
    standalone headline number, not something the founder hand-derives from
    the miss list."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    # No prediction anywhere near the gold submission_choke at t_s=100 -> FN.
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [{"start_s": 9, "end_s": 11, "action_class": "guard_pass", "confidence": 0.9}],
    }))

    report = build_report(_gold_records(), _split_assignment(), predictions_dir, MAPPING, split="dev", tau_t=5.0)
    dev = report["per_split"]["dev"]

    sub_recall = dev["submission_family_recall"]
    assert sub_recall == {"tp": 0, "fn": 1, "gold": 1, "recall": 0.0}

    md = _render_report_md(report)
    assert "Submission-family recall" in md
    assert "tp=0/gold=1" in md


def test_submission_family_recall_headline_true_positive_case(tmp_path):
    """Sanity: when the submission IS matched, the headline recall is 1.0,
    not just correctly reporting the miss case above."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [
            {"start_s": 9, "end_s": 11, "action_class": "guard_pass", "confidence": 0.9},
            {"start_s": 99, "end_s": 101, "action_class": "submission_choke", "confidence": 0.8},
        ],
    }))

    report = build_report(_gold_records(), _split_assignment(), predictions_dir, MAPPING, split="dev", tau_t=5.0)
    sub_recall = report["per_split"]["dev"]["submission_family_recall"]
    assert sub_recall == {"tp": 1, "fn": 0, "gold": 1, "recall": 1.0}


def _gold_records_with_position_entry_back_take():
    """Extends the base fixture with a POSITION-ENTRY-derived, scoreable
    back_take gold event (§4.3 rule 1) — needed to exercise the
    POSITION-ENTRY-alone rollup (gap 3), which the base fixture's only
    POSITION ENTRY row (a non-scoreable `transition`) can't."""
    records = _gold_records()
    records[0]["events"].append({
        "t_s": 150, "action_class": "back_take", "position_bucket": "back_control",
        "outcome": "successful", "scoreable": True, "track": "hard",
        "raw_sequenceType": "POSITION ENTRY", "raw_positions": {"starting": "TURTLE", "ending": "BACK"},
        "raw_movements": "HOOKS IN",
    })
    return records


def test_position_entry_rollup_recall_true_positive(tmp_path):
    """Gap 3: a POSITION-ENTRY-derived scoreable event (back_take) that IS
    matched must show up as a TP in the rollup, isolated from the rest of
    Track B, without a new match run (same outcomes, re-sliced)."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({
        "match_id": "m_dev_1",
        "clips": [{"start_s": 149, "end_s": 151, "action_class": "back_take", "confidence": 0.9}],
    }))

    report = build_report(
        _gold_records_with_position_entry_back_take(), _split_assignment(), predictions_dir, MAPPING,
        split="dev", tau_t=5.0,
    )
    dev = report["per_split"]["dev"]
    pe = dev["position_entry_rollup"]
    assert pe["totals"] == {"tp": 1, "fn": 0, "recall": 1.0}
    assert pe["per_class"]["back_take"] == {"tp": 1, "fn": 0, "recall": 1.0}
    # the non-scoreable POSITION-ENTRY `transition` row must NOT leak into this rollup
    assert "transition" not in pe["per_class"]

    md = _render_report_md(report)
    assert "POSITION-ENTRY-alone rollup" in md
    assert "NOT a ship gate" in md
    assert "back_take: recall=1.000" in md


def test_position_entry_rollup_recall_miss_case(tmp_path):
    """Same fixture, no prediction anywhere near the back_take -> FN, recall 0."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_dev_1.json").write_text(json.dumps({"match_id": "m_dev_1", "clips": []}))

    report = build_report(
        _gold_records_with_position_entry_back_take(), _split_assignment(), predictions_dir, MAPPING,
        split="dev", tau_t=5.0,
    )
    pe = report["per_split"]["dev"]["position_entry_rollup"]
    assert pe["totals"] == {"tp": 0, "fn": 1, "recall": 0.0}
    assert pe["per_class"]["back_take"]["fn"] == 1


def test_named_misses_by_class_covers_non_submission_scoreable_classes(tmp_path):
    """Gap 4: the sweep gold event in m_test_1 has nothing predicted nearby
    -> a real FN. It must be individually named under its own class section,
    not folded into an aggregate FN count or only visible under submissions."""
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "m_test_1.json").write_text(json.dumps({"match_id": "m_test_1", "clips": []}))

    report = build_report(_gold_records(), _split_assignment(), predictions_dir, MAPPING, split="held_out", tau_t=5.0)
    test_split = report["per_split"]["test"]

    by_class = test_split["named_misses_by_class"]
    assert "sweep" in by_class
    assert len(by_class["sweep"]) == 1
    miss = by_class["sweep"][0]
    assert miss["match_id"] == "m_test_1"
    assert miss["t_s"] == 50
    assert miss["raw_sequenceType"] == "SWEEP"

    md = _render_report_md(report)
    assert "Named scoreable misses by class" in md
    assert "#### sweep (1)" in md
    assert "match=m_test_1 t_s=50 class=sweep raw=SWEEP" in md
