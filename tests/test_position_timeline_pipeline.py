"""Contracts for the QA-only continuous-position + sparse-technique pipeline."""

from __future__ import annotations

import base64
import json
import io
from datetime import date
from pathlib import Path
import re
from unittest.mock import MagicMock

import pytest
from google.genai import types
from PIL import Image
from pydantic import ValidationError

from service.pipelines import (
    executors,
    gemini_cost,
    highlight_critique,
    models as pipeline_models,
    position_timeline,
    registry,
    simplified_tags,
)
from service.pipelines.executors import RunContext
from service.pipelines.models import (
    HighlightCritiqueConfig,
    HighlightScanConfig,
    PositionCritiqueConfig,
    PositionScanConfig,
    PositionTechniqueCandidatesConfig,
    TechniqueAnalyzeConfig,
    TimelineReconcileConfig,
)
from service.pipelines.position_timeline import (
    POSITION_CRITIQUE_INITIAL_PROMPT,
    POSITION_CRITIQUE_SYSTEM_PROMPT,
    POSITION_SCAN_INITIAL_PROMPT,
    POSITION_SCAN_SYSTEM_PROMPT,
    TECHNIQUE_SCAN_INITIAL_PROMPT,
    TECHNIQUE_SCAN_SYSTEM_PROMPT,
    TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT,
    TECHNIQUE_SYSTEM_PROMPT,
    TECHNIQUE_VALIDATOR_SYSTEM_PROMPT,
    TimelineValidationError,
    attach_techniques_to_intervals,
    build_objective_statistics,
    build_position_critique_prompt,
    build_position_critique_schema,
    build_position_intervals,
    build_position_scan_prompt,
    build_position_scan_schema,
    build_technique_schema,
    build_technique_prompt,
    build_technique_validator_prompt,
    build_technique_validator_schema,
    validate_technique_validator_decision,
    validate_player_references,
)
from service.routes import qa as qa_route
from service.routes.qa import QAPlayerReference


PLAYER_REFS = [
    {
        "player_id": "player_a",
        "player_name": "Athlete A",
        "image_s3_uri_or_url": "https://example.test/a.jpg",
    },
    {
        "player_id": "player_b",
        "player_name": "Athlete B",
        "image_s3_uri_or_url": "https://example.test/b.jpg",
    },
]


def test_loaded_reference_preserves_source_contract_for_downstream_grounding(monkeypatch):
    image_buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(image_buffer, format="JPEG")
    monkeypatch.setattr(qa_route, "_fetch_bytes", lambda source: image_buffer.getvalue())
    source = "https://example.test/a.jpg"
    loaded = qa_route._download_reference_images(
        [
            QAPlayerReference(
                player_id="player_a",
                player_name="Athlete A",
                image_s3_uri_or_url=source,
            )
        ]
    )[0]
    assert loaded["image_s3_uri_or_url"] == source
    assert loaded["image_bytes"] == image_buffer.getvalue()


def test_inline_crop_reference_decodes_without_an_external_url():
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (20, 40, 60)).save(image_buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_buffer.getvalue()).decode()

    loaded = qa_route._download_reference_images(
        [
            QAPlayerReference(
                player_id="player_a",
                player_name="Athlete A",
                image_base64=data_url,
                bounding_box={"x": 11, "y": 22, "w": 33, "h": 44},
                captured_at_ms=12_340,
            )
        ]
    )[0]

    assert loaded["image_s3_uri_or_url"] is None
    assert loaded["image_bytes"] == image_buffer.getvalue()
    assert loaded["mime_type"] == "image/jpeg"
    assert loaded["bounding_box"] == {"x": 11, "y": 22, "w": 33, "h": 44}
    assert loaded["captured_at_ms"] == 12_340


def test_reference_requires_exactly_one_image_source_and_crop_provenance():
    with pytest.raises(ValidationError, match="exactly one reference image source"):
        QAPlayerReference(player_id="player_a", player_name="Athlete A")

    with pytest.raises(ValidationError, match="exactly one reference image source"):
        QAPlayerReference(
            player_id="player_a",
            player_name="Athlete A",
            image_s3_uri_or_url="https://example.test/a.jpg",
            image_base64="data:image/jpeg;base64,YQ==",
        )

    with pytest.raises(ValidationError, match="bounding_box and captured_at_ms"):
        QAPlayerReference(
            player_id="player_a",
            player_name="Athlete A",
            image_base64="data:image/jpeg;base64,YQ==",
        )


def _state(
    position: str,
    role_a: str,
    role_b: str,
    *,
    position_note: str = "",
    inactive_reason: str = "",
    confidence: float = 0.9,
) -> dict:
    return {
        "position": position,
        "player_a_role": role_a,
        "player_b_role": role_b,
        "position_note": position_note,
        "inactive_reason": inactive_reason,
        "confidence": confidence,
        "evidence": "visible body relationship",
    }


def _transition(at_s: float, state_after: dict, **overrides) -> dict:
    transition = {
        "at_s": at_s,
        "corrected_at_s": at_s,
        "confirmed": True,
        "state_after": state_after,
        "critique_note": "confirmed boundary",
    }
    transition.update(overrides)
    return transition


def _all_model_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "model":
                yield child
            yield from _all_model_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_model_values(child)


def test_registry_adds_position_pipeline_without_mutating_v2_shape():
    new_pipeline = registry.get_default("position-timeline-techniques")
    assert [stage.type for stage in new_pipeline.stages] == [
        "video_window",
        "position_scan",
        "position_critique",
        "highlight_scan",
        "position_technique_candidates",
        "highlight_critique",
        "technique_analyze",
        "timeline_reconcile",
    ]

    candidate_stage = next(
        stage
        for stage in new_pipeline.stages
        if stage.type == "position_technique_candidates"
    )
    assert candidate_stage.config == {
        "transition_pre_s": 4.0,
        "transition_post_s": 2.0,
        "merge_tolerance_s": 2.0,
    }
    assert pipeline_models.PositionTechniqueCandidatesConfig.model_validate(
        candidate_stage.config
    )

    v2 = registry.get_default("highlight-scan-critique-analyze")
    assert [stage.type for stage in v2.stages] == [
        "video_window",
        "highlight_scan",
        "highlight_critique",
        "highlight_analyze",
    ]


def test_registry_saves_complete_critique_prompts_for_both_critique_dags():
    position_pipeline = registry.get_default("position-timeline-techniques")
    position_critique = next(
        stage for stage in position_pipeline.stages if stage.type == "position_critique"
    )
    technique_critique = next(
        stage for stage in position_pipeline.stages if stage.type == "highlight_critique"
    )
    legacy_v2_critique = next(
        stage
        for stage in registry.get_default("highlight-scan-critique-analyze").stages
        if stage.type == "highlight_critique"
    )

    assert position_critique.config["system_prompt"] == POSITION_CRITIQUE_SYSTEM_PROMPT
    assert position_critique.config["initial_prompt"] == POSITION_CRITIQUE_INITIAL_PROMPT
    assert technique_critique.config["system_prompt"] == (
        highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT
    )
    assert technique_critique.config["initial_prompt"] == (
        TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT
    )
    for stage in (legacy_v2_critique,):
        assert stage.config["system_prompt"] == highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT
        assert stage.config["initial_prompt"] == highlight_critique.DEFAULT_CRITIQUE_INITIAL_PROMPT
        assert stage.config["fps"] == 1


def test_registry_defaults_every_non_single_shot_model_to_gemini_37_flash():
    for pipeline_id in registry.DEFAULT_PIPELINES:
        pipeline = registry.get_default(pipeline_id).model_dump()
        models = list(_all_model_values(pipeline))
        assert models, pipeline_id
        if pipeline_id == "single-shot":
            assert models == ["gemini-3.5-flash"]
        else:
            assert set(models) == {"gemini-3.7-flash"}, pipeline_id


def test_position_pipeline_budget_estimate_counts_both_scans_and_deep_lanes():
    pipeline = registry.get_default("position-timeline-techniques")
    planned = executors.estimate_run_plan(pipeline, duration_sec=30.0)
    # Position: 1 scan + <=30 critiques (1s budget-only gap).
    # Technique candidates: <=10 scan hits + <=30 active position transitions.
    # Every final candidate may cost critique + analyze + validator.
    assert planned == {
        "planned_frames": 0,
        "planned_windows": 70,
        "planned_gemini_calls": 152,
    }


def test_position_accuracy_prompts_correct_late_boundaries_and_preserve_named_subpositions():
    assert "later than the rough timestamp" in POSITION_CRITIQUE_SYSTEM_PROMPT
    assert "anywhere in the provided local window" in POSITION_CRITIQUE_SYSTEM_PROMPT
    assert "Do not collapse a clearly visible named leg entanglement" in POSITION_SCAN_SYSTEM_PROMPT
    assert PositionCritiqueConfig().forwardpad_s == 6.0


def test_position_critique_config_retains_editable_prompt_fields():
    config = PositionCritiqueConfig(
        system_prompt="edited position critique system",
        initial_prompt="check $proposed_at_s inside $window_start_s-$window_end_s",
    )

    assert config.system_prompt == "edited position critique system"
    assert config.initial_prompt == "check $proposed_at_s inside $window_start_s-$window_end_s"


def test_position_critique_prompt_uses_an_editable_template_with_state_json():
    prompt = build_position_critique_prompt(
        window_start_s=6.0,
        window_end_s=18.0,
        proposed_at_s=10.0,
        state_before={"position": "standing"},
        state_after={"position": "open_guard"},
        override=(
            "CUSTOM $window_start_s $window_end_s $proposed_at_s "
            "$state_before_json -> $state_after_json"
        ),
    )

    assert prompt == (
        'CUSTOM 6.0 18.0 10.0 {"position": "standing"} -> '
        '{"position": "open_guard"}'
    )


def test_position_scan_prompt_supports_competition_and_uncontrolled_training_footage():
    system_prompt = " ".join(POSITION_SCAN_SYSTEM_PROMPT.split())
    initial_prompt = " ".join(build_position_scan_prompt(12.5, 48.0).split())

    assert "formal competition" in system_prompt
    assert "open mat" in system_prompt
    assert "random gym" in system_prompt
    assert "Track exactly the two target athletes" in system_prompt
    assert "Ignore every other pair" in system_prompt
    assert "absolute source-video second" in initial_prompt
    assert "Do not assume the scope begins at a round boundary" in initial_prompt


def test_position_scan_registry_persists_canonical_prompts_in_default_config():
    pipeline = registry.get_default("position-timeline-techniques")
    position_scan = next(stage for stage in pipeline.stages if stage.id == "position_scan")

    assert position_scan.config["system_prompt"] == POSITION_SCAN_SYSTEM_PROMPT
    assert "$start_sec" in position_scan.config["initial_prompt"]
    assert "$end_sec" in position_scan.config["initial_prompt"]
    assert build_position_scan_prompt(
        12.5, 48.0, position_scan.config["initial_prompt"]
    ) == build_position_scan_prompt(12.5, 48.0)


def test_position_scan_offline_html_defaults_match_backend_canonical_prompts():
    html = (Path(__file__).parents[1] / "qa_client" / "vlm_studio.html").read_text()
    system_match = re.search(
        r"const POSITION_SCAN_DEFAULT_SYSTEM_PROMPT = `([\s\S]*?)`;", html
    )
    initial_match = re.search(
        r"const POSITION_SCAN_DEFAULT_INITIAL_PROMPT = `([\s\S]*?)`;", html
    )

    assert system_match is not None
    assert initial_match is not None
    assert system_match.group(1) == POSITION_SCAN_SYSTEM_PROMPT
    assert initial_match.group(1) == POSITION_SCAN_INITIAL_PROMPT


def test_critique_offline_html_defaults_match_backend_canonical_prompts():
    html = (Path(__file__).parents[1] / "qa_client" / "vlm_studio.html").read_text()
    expected = {
        "POSITION_CRITIQUE_DEFAULT_SYSTEM_PROMPT": POSITION_CRITIQUE_SYSTEM_PROMPT,
        "POSITION_CRITIQUE_DEFAULT_INITIAL_PROMPT": POSITION_CRITIQUE_INITIAL_PROMPT,
        "TECHNIQUE_CRITIQUE_DEFAULT_SYSTEM_PROMPT": (
            highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT
        ),
        "TECHNIQUE_CRITIQUE_DEFAULT_INITIAL_PROMPT": (
            TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT
        ),
    }

    for constant_name, backend_prompt in expected.items():
        match = re.search(
            rf"const {constant_name} = `([\s\S]*?)`;",
            html,
        )
        assert match is not None, constant_name
        assert match.group(1) == backend_prompt


def test_position_timeline_all_gemini_nodes_default_to_high_thinking():
    pipeline = registry.get_default("position-timeline-techniques")
    gemini_stage_types = {
        "position_scan",
        "position_critique",
        "highlight_scan",
        "highlight_critique",
        "technique_analyze",
    }

    for stage in pipeline.stages:
        if stage.type in gemini_stage_types:
            assert stage.config["thinking"] == "high", stage.type

    technique_analyze = next(
        stage for stage in pipeline.stages if stage.type == "technique_analyze"
    )
    assert technique_analyze.config["validator_thinking"] == "high"


def test_position_timeline_technique_critique_defaults_to_four_seconds_of_outcome_context():
    pipeline = registry.get_default("position-timeline-techniques")
    technique_critique = next(
        stage for stage in pipeline.stages if stage.type == "highlight_critique"
    )

    assert HighlightCritiqueConfig().critique_forwardpad_s == 0.0
    assert technique_critique.config["critique_forwardpad_s"] == 4.0
    prompt = " ".join(highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT.split())
    assert "visible outcome" in prompt
    assert "unsuccessful" in prompt
    assert "unclear" in prompt


def test_position_node_cost_fails_closed_when_usage_is_missing_and_tracks_future_rate():
    stage = next(
        stage
        for stage in registry.get_default("position-timeline-techniques").stages
        if stage.id == "position_scan"
    )
    missing = gemini_cost.new_node_accumulator(stage)
    gemini_cost.record_event_usage(missing, {"usage_metadata": None})
    missing_row = gemini_cost.node_cost_snapshot(missing)
    assert missing_row["call_count"] == 1
    assert missing_row["pricing_status"] == "usage_unavailable"
    assert missing_row["estimated_usd"] is None
    assert gemini_cost.summarize_node_costs([missing_row])["estimated_usd"] is None

    metered = gemini_cost.new_node_accumulator(stage)
    gemini_cost.record_event_usage(
        metered,
        {
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 10,
                "thoughts_token_count": 20,
                "total_token_count": 130,
            }
        },
    )
    future_row = gemini_cost.node_cost_snapshot(metered, on_date=date(2027, 1, 1))
    assert future_row["rate_effective_from"] == "2027-01-01"
    assert future_row["estimated_usd"] == pytest.approx(0.000375)


def test_technique_scan_requires_one_atomic_event_per_highlight():
    for prompt in (TECHNIQUE_SCAN_SYSTEM_PROMPT, TECHNIQUE_SCAN_INITIAL_PROMPT):
        assert "one intentional technique event per highlight" in prompt
        assert "separate highlights" in prompt
    timeline_pipeline = registry.get_default("position-timeline-techniques")
    technique_scan = next(
        stage for stage in timeline_pipeline.stages if stage.type == "highlight_scan"
    )
    assert technique_scan.config["thinking"] == "high"


def test_technique_analysis_cannot_substitute_a_neighboring_event():
    system_prompt = " ".join(TECHNIQUE_SYSTEM_PROMPT.split())
    validator_prompt = " ".join(TECHNIQUE_VALIDATOR_SYSTEM_PROMPT.split())
    event_prompt = " ".join(
        build_technique_prompt(
            "bottom athlete comes up to top in a sweep",
            start_s=163.0,
            end_s=170.5,
            player_ids=("player_a", "player_b"),
        ).split()
    )
    assert "Do not substitute a later or overlapping technique" in system_prompt
    assert "Do not substitute a neighboring event" in validator_prompt
    assert "The scan-described movement is the target event" in event_prompt


def test_position_context_is_advisory_in_critique_analyze_and_validator_prompts():
    context = {
        "position_before": "open_guard",
        "position_after": "side_control",
        "position_transitions": [{"at_s": 10.0}],
    }
    critique_prompt = highlight_critique.build_critique_prompt(
        window_start_sec=4.0,
        window_end_sec=14.0,
        critique_backpad_s=4.0,
        critique_forwardpad_s=2.0,
        scan_start_sec=8.0,
        scan_end_sec=12.0,
        description="top athlete moves around the legs",
        override=TECHNIQUE_CRITIQUE_POSITION_INITIAL_PROMPT,
        position_context=context,
    )
    analyze_prompt = build_technique_prompt(
        "top athlete moves around the legs",
        start_s=8.0,
        end_s=12.0,
        player_ids=("player_a", "player_b"),
        position_context=context,
    )
    validator_prompt = build_technique_validator_prompt(
        "top athlete moves around the legs",
        {"action_class": "guard_pass"},
        position_context=context,
    )

    for prompt in (critique_prompt, analyze_prompt, validator_prompt):
        assert '"position_before": "open_guard"' in prompt
        assert "fallible" in prompt.lower()
        assert "visible body-movement evidence" in prompt
    assert "$position_context_json" not in critique_prompt


def test_technique_validator_cannot_ditch_a_different_visible_technique():
    normalized_prompt = " ".join(TECHNIQUE_VALIDATOR_SYSTEM_PROMPT.split())
    assert "A different intentional technique is a correction, never a ditch" in normalized_prompt
    schema = build_technique_validator_schema(("player_a", "player_b"))
    assert "intentional_technique_visible" in schema.required
    assert schema.properties["intentional_technique_visible"].type == types.Type.BOOLEAN

    assert validate_technique_validator_decision(
        {"verdict": "correct", "intentional_technique_visible": True},
        context="highlight 1",
    ) == ("correct", True)
    with pytest.raises(TimelineValidationError, match="contradictory ditch"):
        validate_technique_validator_decision(
            {"verdict": "ditch", "intentional_technique_visible": True},
            context="highlight 1",
        )
    with pytest.raises(TimelineValidationError, match="kept an event"):
        validate_technique_validator_decision(
            {"verdict": "agree", "intentional_technique_visible": False},
            context="highlight 1",
        )


def test_exactly_two_distinct_grounded_player_references_are_required():
    assert validate_player_references(PLAYER_REFS) == ("player_a", "player_b")

    with pytest.raises(TimelineValidationError, match="exactly two"):
        validate_player_references(PLAYER_REFS[:1])

    duplicate = [PLAYER_REFS[0], {**PLAYER_REFS[1], "player_id": "player_a"}]
    with pytest.raises(TimelineValidationError, match="distinct"):
        validate_player_references(duplicate)

    inline = [
        {
            "player_id": "player_a",
            "player_name": "Athlete A",
            "image_base64": "data:image/jpeg;base64,YQ==",
        },
        {
            "player_id": "player_b",
            "player_name": "Athlete B",
            "image_base64": "data:image/jpeg;base64,Yg==",
        },
    ]
    assert validate_player_references(inline) == ("player_a", "player_b")


def test_position_scan_schema_is_boundary_based_and_includes_other_plus_inactive():
    schema = build_position_scan_schema()
    assert schema.type == types.Type.OBJECT
    assert set(schema.required) == {"initial_state", "transitions"}
    assert "intervals" not in schema.properties
    state_schema = schema.properties["initial_state"]
    assert set(state_schema.properties["position"].enum) == {
        "standing",
        "closed_guard",
        "open_guard",
        "half_guard",
        "side_control",
        "mount",
        "back_control",
        "turtle",
        "scramble",
        "other",
        "inactive",
    }
    transition_item = schema.properties["transitions"].items
    assert set(transition_item.required) == {"at_s", "state_after", "evidence"}


def test_position_critique_and_technique_schemas_keep_axes_separate():
    critique = build_position_critique_schema()
    assert set(critique.required) == {
        "confirmed",
        "corrected_at_s",
        "state_after",
        "note",
    }

    technique = build_technique_schema(("player_a", "player_b"))
    assert "position" not in technique.properties
    assert set(technique.properties["actor_player_id"].enum) == {
        "player_a",
        "player_b",
        "contested",
        "unclear",
    }
    assert set(technique.required) == {
        "actor_player_id",
        "identity_uncertain",
        "action_class",
        "outcome",
        "specific_technique_guess",
        "confidence",
        "evidence",
    }


def test_boundary_reconciliation_covers_scope_and_preserves_reset_separation():
    initial = _state("standing", "neutral", "neutral")
    half_guard = _state("half_guard", "top", "bottom")
    inactive = _state(
        "inactive",
        "not_applicable",
        "not_applicable",
        inactive_reason="referee reset",
    )
    other = _state(
        "other",
        "bottom",
        "top",
        position_note="50/50 guard with both athletes entangled at the hips",
    )
    transitions = [
        _transition(10.0, half_guard),
        # Identical adjacent state must merge deterministically.
        _transition(30.0, half_guard),
        _transition(40.0, inactive),
        _transition(45.0, half_guard),
        _transition(70.0, other),
    ]

    intervals = build_position_intervals(
        0.0,
        100.0,
        initial,
        transitions,
        player_ids=("player_a", "player_b"),
    )

    assert [(item["start_s"], item["end_s"], item["position"]) for item in intervals] == [
        (0.0, 10.0, "standing"),
        (10.0, 40.0, "half_guard"),
        (40.0, 45.0, "inactive"),
        (45.0, 70.0, "half_guard"),
        (70.0, 100.0, "other"),
    ]
    assert intervals[1]["player_roles"] == {"player_a": "top", "player_b": "bottom"}
    assert intervals[-1]["position_note"].startswith("50/50")
    assert sum(item["duration_s"] for item in intervals) == 100.0
    assert all(left["end_s"] == right["start_s"] for left, right in zip(intervals, intervals[1:]))


def test_position_candidates_preserve_scan_targets_and_add_only_active_transitions():
    intervals = build_position_intervals(
        0.0,
        40.0,
        _state("standing", "neutral", "neutral"),
        [
            _transition(10.0, _state("open_guard", "top", "bottom")),
            _transition(
                20.0,
                _state(
                    "inactive",
                    "not_applicable",
                    "not_applicable",
                    inactive_reason="referee reset",
                ),
            ),
            _transition(22.0, _state("standing", "neutral", "neutral")),
            _transition(32.0, _state("side_control", "top", "bottom")),
        ],
        player_ids=("player_a", "player_b"),
    )
    scanned = [
        {
            "index": 0,
            "start_s": 7.0,
            "end_s": 12.0,
            "description": "Player A changes level and drives Player B to the mat",
            "reasoning": "deliberate whole-body movement",
        }
    ]

    candidates = position_timeline.build_position_aware_candidates(
        scanned,
        intervals,
        scope_start_s=0.0,
        scope_end_s=40.0,
        transition_pre_s=4.0,
        transition_post_s=2.0,
        merge_tolerance_s=2.0,
    )

    assert len(candidates) == 2
    assert candidates[0]["start_s"] == 7.0
    assert candidates[0]["end_s"] == 12.0
    assert candidates[0]["description"] == scanned[0]["description"]
    assert candidates[0]["candidate_sources"] == [
        "technique_scan",
        "position_transition",
    ]
    assert [
        item["at_s"] for item in candidates[0]["position_context"]["position_transitions"]
    ] == [10.0]

    assert candidates[1]["start_s"] == 28.0
    assert candidates[1]["end_s"] == 34.0
    assert candidates[1]["candidate_sources"] == ["position_transition"]
    assert "do not infer" in candidates[1]["description"].lower()
    assert candidates[1]["position_context"]["position_before"] == "standing"
    assert candidates[1]["position_context"]["position_after"] == "side_control"
    assert all(
        item["at_s"] not in {20.0, 22.0}
        for candidate in candidates
        for item in candidate["position_context"]["position_transitions"]
    )


def test_distinct_position_transitions_never_collapse_into_one_scan_candidate():
    intervals = build_position_intervals(
        0.0,
        20.0,
        _state("standing", "neutral", "neutral"),
        [
            _transition(10.0, _state("open_guard", "top", "bottom")),
            _transition(12.0, _state("side_control", "top", "bottom")),
        ],
        player_ids=("player_a", "player_b"),
    )

    candidates = position_timeline.build_position_aware_candidates(
        [{"index": 0, "start_s": 9.0, "end_s": 13.0, "description": "one scan event"}],
        intervals,
        scope_start_s=0.0,
        scope_end_s=20.0,
        transition_pre_s=4.0,
        transition_post_s=2.0,
        merge_tolerance_s=2.0,
    )

    assert len(candidates) == 2
    assert sum(
        candidate["candidate_sources"] == ["technique_scan", "position_transition"]
        for candidate in candidates
    ) == 1
    transition_only = next(
        candidate
        for candidate in candidates
        if candidate["candidate_sources"] == ["position_transition"]
    )
    assert transition_only["transition_at_s"] in {10.0, 12.0}


def test_position_context_tracks_before_after_roles_and_overlapping_intervals():
    intervals = build_position_intervals(
        0.0,
        30.0,
        _state("standing", "neutral", "neutral", confidence=0.8),
        [
            _transition(
                10.0,
                _state("open_guard", "top", "bottom", confidence=0.9),
            ),
            _transition(
                20.0,
                _state("side_control", "top", "bottom", confidence=0.95),
            ),
        ],
        player_ids=("player_a", "player_b"),
    )

    context = position_timeline.build_position_context(intervals, 8.0, 22.0)

    assert context["position_before"] == "standing"
    assert context["position_after"] == "side_control"
    assert context["player_roles_before"] == {
        "player_a": "neutral",
        "player_b": "neutral",
    }
    assert context["player_roles_after"] == {
        "player_a": "top",
        "player_b": "bottom",
    }
    assert context["overlapping_position_interval_indexes"] == [0, 1, 2]
    assert [item["at_s"] for item in context["position_transitions"]] == [10.0, 20.0]
    assert context["confidence_before"] == 0.8
    assert context["confidence_after"] == 0.95


def test_attached_technique_exposes_before_after_context_across_a_boundary():
    intervals = build_position_intervals(
        0.0,
        20.0,
        _state("open_guard", "top", "bottom"),
        [_transition(10.0, _state("side_control", "top", "bottom"))],
        player_ids=("player_a", "player_b"),
    )

    attached = attach_techniques_to_intervals(
        [
            {
                "start_s": 7.0,
                "end_s": 13.0,
                "actor_player_id": "player_a",
                "candidate_sources": ["technique_scan", "position_transition"],
            }
        ],
        intervals,
    )[0]

    assert attached["position_before"] == "open_guard"
    assert attached["position_after"] == "side_control"
    assert attached["overlapping_position_interval_indexes"] == [0, 1]
    assert attached["position_transitions"] == [
        {
            "at_s": 10.0,
            "position_before": "open_guard",
            "position_after": "side_control",
            "player_roles_before": {"player_a": "top", "player_b": "bottom"},
            "player_roles_after": {"player_a": "top", "player_b": "bottom"},
        }
    ]
    assert attached["candidate_sources"] == [
        "technique_scan",
        "position_transition",
    ]


@pytest.mark.asyncio
async def test_position_candidate_node_rebuilds_timeline_and_streams_provenance():
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
        player_references=_loaded_refs(),
    )
    ctx.position_initial_state = _state("standing", "neutral", "neutral")
    ctx.position_transitions = [
        _transition(10.0, _state("open_guard", "top", "bottom")),
        _transition(20.0, _state("side_control", "top", "bottom")),
    ]
    ctx.highlights = [
        {
            "index": 0,
            "start_s": 7.0,
            "end_s": 12.0,
            "description": "Player A drives Player B to the mat",
            "reasoning": "deliberate body movement",
        }
    ]

    events = [
        event
        async for event in executors.position_technique_candidates_node(
            ctx, PositionTechniqueCandidatesConfig().model_dump()
        )
    ]

    candidate_map = events[0]
    assert candidate_map["type"] == "technique_candidate_map"
    assert candidate_map["scan_candidate_count"] == 1
    assert candidate_map["transition_candidate_count"] == 2
    assert candidate_map["merged_candidate_count"] == 1
    assert candidate_map["final_candidate_count"] == 2
    assert ctx.position_intervals[1]["position"] == "open_guard"
    assert ctx.highlights[0]["candidate_sources"] == [
        "technique_scan",
        "position_transition",
    ]
    assert ctx.highlights[1]["candidate_sources"] == ["position_transition"]


def test_objective_statistics_exclude_inactive_and_never_emit_dominance_fields():
    intervals = build_position_intervals(
        0.0,
        100.0,
        _state("standing", "neutral", "neutral"),
        [
            _transition(10.0, _state("half_guard", "top", "bottom")),
            _transition(
                40.0,
                _state(
                    "inactive",
                    "not_applicable",
                    "not_applicable",
                    inactive_reason="referee reset",
                ),
            ),
            _transition(45.0, _state("half_guard", "top", "bottom")),
            _transition(
                70.0,
                _state("other", "bottom", "top", position_note="50/50 guard"),
            ),
        ],
        player_ids=("player_a", "player_b"),
    )
    techniques = [
        {"actor_player_id": "player_a", "start_s": 71.0, "end_s": 74.0},
        {"actor_player_id": "player_b", "start_s": 20.0, "end_s": 22.0},
        {"actor_player_id": "contested", "start_s": 4.0, "end_s": 6.0},
    ]

    stats = build_objective_statistics(
        intervals,
        techniques,
        player_ids=("player_a", "player_b"),
    )

    assert stats["active_duration_s"] == 95.0
    assert stats["inactive_duration_s"] == 5.0
    assert stats["by_position"]["standing"]["duration_s"] == 10.0
    assert stats["by_position"]["half_guard"]["duration_s"] == 55.0
    assert stats["by_position"]["other"]["duration_s"] == 30.0
    assert stats["by_player"]["player_a"]["by_position"]["half_guard"]["top_s"] == 55.0
    assert stats["by_player"]["player_a"]["by_position"]["other"]["bottom_s"] == 30.0
    assert stats["by_player"]["player_b"]["role_totals"]["top_s"] == 30.0
    assert stats["techniques_by_player"] == {"player_a": 1, "player_b": 1, "contested": 1}
    assert "dominant_player" not in stats
    assert "dominance" not in stats
    assert "controller" not in stats


def test_objective_statistics_include_zero_technique_count_for_each_grounded_player():
    intervals = build_position_intervals(
        0,
        10,
        _state("standing", "neutral", "neutral"),
        [],
        player_ids=("player_a", "player_b"),
    )
    stats = build_objective_statistics(
        intervals, [], player_ids=("player_a", "player_b")
    )
    assert stats["techniques_by_player"] == {"player_a": 0, "player_b": 0}


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_state("other", "top", "bottom"), "position_note"),
        (
            _state("inactive", "not_applicable", "not_applicable"),
            "inactive_reason",
        ),
    ],
)
def test_other_and_inactive_require_explanatory_notes(state, message):
    with pytest.raises(TimelineValidationError, match=message):
        build_position_intervals(
            0.0,
            10.0,
            state,
            [],
            player_ids=("player_a", "player_b"),
        )


def test_crossed_or_out_of_scope_corrected_boundaries_fail_closed():
    with pytest.raises(TimelineValidationError, match="strictly increasing"):
        build_position_intervals(
            0.0,
            30.0,
            _state("standing", "neutral", "neutral"),
            [
                _transition(20.0, _state("scramble", "neutral", "neutral")),
                _transition(10.0, _state("mount", "top", "bottom")),
            ],
            player_ids=("player_a", "player_b"),
        )


def _loaded_refs():
    return [
        {
            "player_id": "player_a",
            "player_name": "Athlete A",
            "image_bytes": b"jpeg-a",
            "mime_type": "image/jpeg",
        },
        {
            "player_id": "player_b",
            "player_name": "Athlete B",
            "image_bytes": b"jpeg-b",
            "mime_type": "image/jpeg",
        },
    ]


@pytest.mark.asyncio
async def test_position_critique_node_sends_edited_system_and_initial_prompts(monkeypatch):
    captured = {}

    async def _fake_analyze_chunk(
        self, youtube_url, start_sec, end_sec, previous_context=None, **kwargs
    ):
        captured.update(
            system_instruction=self.system_instruction,
            prompt_text=kwargs["prompt_text"],
            fps=kwargs["fps"],
        )
        return json.dumps(
            {
                "confirmed": True,
                "corrected_at_s": 10.0,
                "state_after": _state("open_guard", "top", "bottom"),
                "note": "confirmed",
            }
        )

    monkeypatch.setattr(
        simplified_tags.SimplifiedTagsTimeAnalyzer,
        "analyze_chunk",
        _fake_analyze_chunk,
    )
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://y",
        start_sec=0,
        end_sec=30,
        gemini_api_key="k",
        request_timeout_ms=1000,
        player_references=_loaded_refs(),
    )
    ctx.position_initial_state = _state("standing", "neutral", "neutral")
    ctx.position_transitions = [
        {
            "index": 0,
            "at_s": 10.0,
            "state_after": _state("open_guard", "top", "bottom"),
        }
    ]
    config = PositionCritiqueConfig(
        fps=15,
        system_prompt="EDITED POSITION CRITIQUE SYSTEM",
        initial_prompt="EDITED $proposed_at_s @ $window_start_s-$window_end_s",
    ).model_dump()

    events = [event async for event in executors.position_critique_node(ctx, config)]

    assert any(event["type"] == "position_critique_result" for event in events)
    assert captured == {
        "system_instruction": "EDITED POSITION CRITIQUE SYSTEM",
        "prompt_text": "EDITED 10.0 @ 6.0-16.0",
        "fps": 15,
    }


@pytest.mark.asyncio
async def test_new_nodes_stream_separate_position_and_technique_results(monkeypatch):
    calls: list[dict] = []

    async def _fake_analyze_chunk(self, youtube_url, start_sec, end_sec, previous_context=None, **kwargs):
        calls.append({"start": start_sec, "end": end_sec, **kwargs})
        prompt = kwargs["prompt_text"]
        if "first state" in prompt:
            return json.dumps(
                {
                    "initial_state": _state("standing", "neutral", "neutral"),
                    "transitions": [
                        {
                            "at_s": 10.0,
                            "state_after": _state("half_guard", "top", "bottom"),
                            "evidence": "player A settles above one trapped leg",
                        }
                    ],
                }
            )
        if "PROPOSED TRANSITION" in prompt:
            return json.dumps(
                {
                    "confirmed": True,
                    "corrected_at_s": 9.5,
                    "state_after": _state("half_guard", "top", "bottom"),
                    "note": "boundary begins when the leg is trapped",
                }
            )
        if "ADVERSARIAL TECHNIQUE REVIEW" in prompt:
            return json.dumps(
                {
                    "verdict": "agree",
                    "intentional_technique_visible": True,
                    "actor_player_id": "player_a",
                    "identity_uncertain": False,
                    "action_class": "guard_pass",
                    "outcome": "unsuccessful",
                    "specific_technique_guess": "knee cut",
                    "confidence": 0.82,
                    "evidence": "player A drives a knee through but guard remains",
                    "reason": "labels survive the opposing case",
                }
            )
        return json.dumps(
            {
                "actor_player_id": "player_a",
                "identity_uncertain": False,
                "action_class": "guard_pass",
                "outcome": "unsuccessful",
                "specific_technique_guess": "knee cut",
                "confidence": 0.8,
                "evidence": "player A drives a knee through the guard",
            }
        )

    monkeypatch.setattr(
        simplified_tags.SimplifiedTagsTimeAnalyzer,
        "analyze_chunk",
        _fake_analyze_chunk,
    )
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())

    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
        player_references=_loaded_refs(),
    )

    scan_events = [
        event
        async for event in executors.position_scan_node(
            ctx,
            PositionScanConfig().model_dump(),
        )
    ]
    assert next(event for event in scan_events if event["type"] == "position_map")[
        "transitions"
    ][0]["at_s"] == 10.0

    critique_events = [
        event
        async for event in executors.position_critique_node(
            ctx,
            PositionCritiqueConfig().model_dump(),
        )
    ]
    assert next(
        event for event in critique_events if event["type"] == "position_critique_result"
    )["corrected_at_s"] == 9.5

    ctx.highlights = [
        {
            "index": 0,
            "start_s": 12.0,
            "end_s": 18.0,
            "corrected_start_s": 11.5,
            "corrected_end_s": 18.5,
            "description": "player A drives a knee between player B's legs",
            "candidate_sources": ["technique_scan", "position_transition"],
            "position_context": {"position_before": "stale_scan_context"},
        }
    ]
    technique_events = [
        event
        async for event in executors.technique_analyze_node(
            ctx,
            TechniqueAnalyzeConfig().model_dump(),
        )
    ]
    result = next(event for event in technique_events if event["type"] == "technique_result")
    assert result["event"]["actor_player_id"] == "player_a"
    assert result["event"]["position"] is None  # attached only by deterministic reconcile
    assert result["event"]["original_bounds"] == [12.0, 18.0]
    assert result["event"]["corrected_bounds"] == [11.5, 18.5]
    assert result["event"]["candidate_sources"] == [
        "technique_scan",
        "position_transition",
    ]
    assert result["event"]["position_context"]["position_before"] == "half_guard"
    assert "stale_scan_context" not in calls[-2]["prompt_text"]
    assert '"position_before": "half_guard"' in calls[-2]["prompt_text"]
    assert '"position_before": "half_guard"' in calls[-1]["prompt_text"]
    assert '"candidate_sources": ["technique_scan", "position_transition"]' in calls[-2]["prompt_text"]
    assert '"candidate_sources": ["technique_scan", "position_transition"]' in calls[-1]["prompt_text"]

    timeline_events = [
        event
        async for event in executors.timeline_reconcile_node(
            ctx,
            TimelineReconcileConfig().model_dump(),
        )
    ]
    timeline = next(event for event in timeline_events if event["type"] == "timeline_result")
    assert timeline["position_intervals"] == [
        {
            **timeline["position_intervals"][0],
            "start_s": 0.0,
            "end_s": 9.5,
            "position": "standing",
        },
        {
            **timeline["position_intervals"][1],
            "start_s": 9.5,
            "end_s": 30.0,
            "position": "half_guard",
        },
    ]
    assert timeline["technique_events"][0]["position_interval_index"] == 1
    assert timeline["technique_events"][0]["position"] == "half_guard"
    assert timeline["statistics"]["active_duration_s"] == 30.0
    assert ctx.final_format == "position-technique-timeline-v1"

    # Every model judgment with player-specific fields receives both labelled refs.
    assert len(calls) == 4
    assert all(len(call["extra_parts"]) == 4 for call in calls)


@pytest.mark.asyncio
async def test_actual_budget_gate_includes_position_calls_before_technique_deep_loop(
    monkeypatch,
):
    executed: list[str] = []

    def _executor(stage_type):
        async def _run(ctx, config):
            executed.append(stage_type)
            if stage_type == "position_scan":
                ctx.position_transitions = [{"index": 0}, {"index": 1}]
            if stage_type == "highlight_scan":
                ctx.highlights = [{"index": 0}, {"index": 1}]
            if False:
                yield {}

        return _run

    pipeline = registry.get_default("position-timeline-techniques")
    for stage in pipeline.stages:
        monkeypatch.setitem(executors.NODE_EXECUTORS, stage.type, _executor(stage.type))

    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    events = [
        event
        async for event in executors.run_pipeline(
            pipeline,
            ctx,
            executors.estimate_run_plan(pipeline, 30.0),
            budget_cap=9,
        )
    ]

    # 1 position scan + 2 position critiques + 1 technique scan + 2*3 deep
    # calls = 10, so the run stops after the real technique map and before
    # technique critique/analyze.
    error = next(event for event in events if event["type"] == "error")
    assert "10" in error["message"]
    assert executed == [
        "video_window",
        "position_scan",
        "position_critique",
        "highlight_scan",
        "position_technique_candidates",
    ]

    with pytest.raises(TimelineValidationError, match="inside the requested scope"):
        build_position_intervals(
            0.0,
            30.0,
            _state("standing", "neutral", "neutral"),
            [_transition(31.0, _state("scramble", "neutral", "neutral"))],
            player_ids=("player_a", "player_b"),
        )


@pytest.mark.parametrize(
    "event",
    [
        {"start_s": -1, "end_s": 3},
        {"start_s": 8, "end_s": 11},
    ],
)
def test_technique_bounds_must_stay_inside_requested_position_scope(event):
    intervals = build_position_intervals(
        0,
        10,
        _state("standing", "neutral", "neutral"),
        [],
        player_ids=("player_a", "player_b"),
    )
    with pytest.raises(TimelineValidationError, match="inside the requested scope"):
        attach_techniques_to_intervals([event], intervals)


@pytest.mark.asyncio
async def test_rejected_technique_critique_is_ditched_without_deep_model_calls(monkeypatch):
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
        player_references=_loaded_refs(),
    )
    ctx.highlights = [
        {
            "index": 0,
            "start_s": 8.0,
            "end_s": 12.0,
            "_critique_status": "rejected",
            "critique_note": "scan description does not match visible movement",
        }
    ]
    events = [
        event
        async for event in executors.technique_analyze_node(
            ctx, TechniqueAnalyzeConfig().model_dump()
        )
    ]
    assert events == [
        {
            "type": "technique_result",
            "highlight_index": 0,
            "status": "ditched",
            "ditch_reason": "scan description does not match visible movement",
            "event": None,
        }
    ]
    assert ctx.technique_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["highlight_scan", "highlight_critique"])
async def test_new_pipeline_never_completes_after_technique_lane_failure(
    monkeypatch, failure_stage
):
    executed: list[str] = []

    def _executor(stage_type):
        async def _run(ctx, config):
            executed.append(stage_type)
            if stage_type == "highlight_scan":
                ctx.highlights = [{"index": 0, "start_s": 5.0, "end_s": 10.0}]
                if failure_stage == "highlight_scan":
                    ctx.highlight_scan_failed = True
                    yield {
                        "type": "error",
                        "stage_id": "highlight_scan",
                        "message": "transport failed",
                    }
            if stage_type == "highlight_critique" and failure_stage == "highlight_critique":
                ctx.highlights[0]["_critique_status"] = "failed"
                ctx.highlights[0]["critique_note"] = "critique transport failed"
                yield {
                    "type": "error",
                    "stage_id": "highlight_critique",
                    "highlight_index": 0,
                    "message": "critique transport failed",
                }
            if False:
                yield {}

        return _run

    pipeline = registry.get_default("position-timeline-techniques")
    for stage in pipeline.stages:
        monkeypatch.setitem(executors.NODE_EXECUTORS, stage.type, _executor(stage.type))
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    events = [
        event
        async for event in executors.run_pipeline(
            pipeline,
            ctx,
            executors.estimate_run_plan(pipeline, 30),
            budget_cap=60,
        )
    ]
    assert "technique_analyze" not in executed
    assert not any(event["type"] == "run_complete" for event in events)


@pytest.mark.asyncio
async def test_valid_zero_highlight_scan_skips_deep_technique_stages_and_completes(monkeypatch):
    executed: list[str] = []

    def _executor(stage_type):
        async def _run(ctx, config):
            executed.append(stage_type)
            if stage_type == "position_scan":
                yield {
                    "type": "position_map",
                    "usage_metadata": {
                        "prompt_token_count": 100,
                        "candidates_token_count": 10,
                        "thoughts_token_count": 20,
                        "total_token_count": 130,
                    },
                }
            if stage_type == "highlight_scan":
                ctx.highlights = []
                yield {
                    "type": "highlight_map",
                    "highlights": [],
                    "usage_metadata": {
                        "prompt_token_count": 200,
                        "candidates_token_count": 20,
                        "thoughts_token_count": 30,
                        "total_token_count": 250,
                    },
                }
            if stage_type == "timeline_reconcile":
                ctx.final_format = "position-technique-timeline-v1"
                ctx.final_clips = []

        return _run

    pipeline = registry.get_default("position-timeline-techniques")
    for stage in pipeline.stages:
        monkeypatch.setitem(executors.NODE_EXECUTORS, stage.type, _executor(stage.type))
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    events = [
        event
        async for event in executors.run_pipeline(
            pipeline,
            ctx,
            executors.estimate_run_plan(pipeline, 30),
            budget_cap=60,
        )
    ]
    assert "highlight_critique" not in executed
    assert "technique_analyze" not in executed
    assert not any(event["type"] == "error" for event in events)
    run_complete = next(event for event in events if event["type"] == "run_complete")
    assert [cost["stage_id"] for cost in run_complete["node_costs"]] == [
        stage.id for stage in pipeline.stages
    ]
    by_stage = {cost["stage_id"]: cost for cost in run_complete["node_costs"]}
    assert by_stage["position_scan"]["call_count"] == 1
    assert by_stage["position_scan"]["thoughts_token_count"] == 20
    assert by_stage["position_scan"]["estimated_usd"] == pytest.approx(0.0001875)
    assert by_stage["highlight_scan"]["estimated_usd"] == pytest.approx(0.0003375)
    assert by_stage["video_window"]["estimated_usd"] == 0.0
    assert by_stage["highlight_critique"]["call_count"] == 0
    assert run_complete["cost_summary"]["estimated_usd"] == pytest.approx(0.000525)
    assert run_complete["cost_summary"]["pricing_tier"] == "standard_paid"


@pytest.mark.asyncio
async def test_technique_critique_can_extend_end_to_visible_outcome_within_forward_window(monkeypatch):
    captured = []

    async def _confirm(self, youtube_url, start_sec, end_sec, previous_context=None, **kwargs):
        captured.append((start_sec, end_sec, kwargs["prompt_text"]))
        return json.dumps(
            {
                "movement_confirmed": True,
                "corrected_start_s": 18.0,
                "corrected_end_s": 33.5,
                "note": "defense completes and the attacker abandons at 33.5s",
            }
        )

    monkeypatch.setattr(
        simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _confirm
    )
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=100,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    ctx.highlights = [
        {
            "index": 0,
            "start_s": 20.0,
            "end_s": 30.0,
            "description": "top athlete attempts a guard pass",
        }
    ]

    events = [
        event
        async for event in executors.highlight_critique_node(
            ctx,
            HighlightCritiqueConfig(
                critique_backpad_s=6.0,
                critique_forwardpad_s=4.0,
            ).model_dump(),
        )
    ]

    assert captured[0][:2] == (14.0, 34.0)
    assert "4.0 s FORWARD" in captured[0][2]
    assert ctx.highlights[0]["corrected_end_s"] == 33.5
    result = next(event for event in events if event["type"] == "highlight_critique_result")
    assert result["corrected_end_s"] == 33.5


@pytest.mark.asyncio
async def test_technique_critique_accepts_forward_window_start_without_coercion(monkeypatch):
    async def _confirm(self, youtube_url, start_sec, end_sec, previous_context=None, **kwargs):
        return json.dumps(
            {
                "movement_confirmed": True,
                "corrected_start_s": 31.0,
                "corrected_end_s": 33.0,
                "note": "the decisive outcome is visible after the rough scan range",
            }
        )

    monkeypatch.setattr(
        simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _confirm
    )
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=100,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    ctx.highlights = [
        {
            "index": 0,
            "start_s": 20.0,
            "end_s": 30.0,
            "description": "top athlete attempts a guard pass",
        }
    ]

    events = [
        event
        async for event in executors.highlight_critique_node(
            ctx,
            HighlightCritiqueConfig(
                critique_backpad_s=6.0,
                critique_forwardpad_s=4.0,
            ).model_dump(),
        )
    ]

    assert ctx.highlights[0]["corrected_start_s"] == 31.0
    assert ctx.highlights[0]["corrected_end_s"] == 33.0
    result = next(event for event in events if event["type"] == "highlight_critique_result")
    assert result["applied"] is True


@pytest.mark.asyncio
async def test_highlight_scan_transport_failure_is_explicitly_recorded(monkeypatch):
    async def _raise(*args, **kwargs):
        raise RuntimeError("scan transport failed")

    monkeypatch.setattr(executors.gemini_retry, "call_with_retry", _raise)
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    events = [
        event
        async for event in executors.highlight_scan_node(
            ctx, HighlightScanConfig().model_dump()
        )
    ]
    assert ctx.highlight_scan_failed is True
    assert events[0]["type"] == "error"
    assert "scan transport failed" in events[0]["message"]


@pytest.mark.asyncio
async def test_highlight_critique_records_rejected_state(monkeypatch):
    async def _reject(self, *args, **kwargs):
        return json.dumps(
            {
                "movement_confirmed": False,
                "corrected_start_s": 4.0,
                "corrected_end_s": 9.0,
                "note": "description does not match the clip",
            }
        )

    monkeypatch.setattr(
        simplified_tags.SimplifiedTagsTimeAnalyzer, "analyze_chunk", _reject
    )
    monkeypatch.setattr(RunContext, "gemini_client", lambda self: MagicMock())
    ctx = RunContext(
        youtube_id="x",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        start_sec=0,
        end_sec=30,
        gemini_api_key="key",
        request_timeout_ms=1_000,
    )
    ctx.highlights = [
        {"index": 0, "start_s": 5.0, "end_s": 10.0, "description": "scan claim"}
    ]
    events = [
        event
        async for event in executors.highlight_critique_node(
            ctx, HighlightCritiqueConfig().model_dump()
        )
    ]
    assert ctx.highlights[0]["_critique_status"] == "rejected"
    result = next(event for event in events if event["type"] == "highlight_critique_result")
    assert result["applied"] is False
    assert result["corrected_start_s"] is None
