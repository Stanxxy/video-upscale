"""``service/pipelines/models.py`` + ``registry.py`` — fail-closed config
validation, structural-node lock, registry default round-trip (build plan §3.1).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from service.pipelines import simplified_tags
from service.pipelines.models import (
    ActorAxisConfig,
    AnalyzeConfig,
    ChunkAnalyzeConfig,
    ChunkSegmentConfig,
    DetectCropConfig,
    HighlightAnalyzeConfig,
    HighlightCritiqueConfig,
    HighlightScanConfig,
    PipelineDef,
    PositionAxisConfig,
    StageDef,
    TechniqueAxisConfig,
    ThinkingQualityMixin,
    ValidatorAxisConfig,
)
from service.pipelines.registry import PipelineValidationError, get_default, list_pipelines, validate_pipeline_def


# --------------------------------------------------------------------------- #
# Registry defaults round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pipeline_id",
    [
        "single-shot", "vision-mimic", "vision-mimic-multi", "vision-mimic-simplified",
        "chunk-segment-tags", "highlight-scan-analyze", "highlight-scan-critique-analyze",
    ],
)
def test_get_default_round_trips_through_json(pipeline_id):
    pdef = get_default(pipeline_id)
    dumped = pdef.model_dump(mode="json")
    reloaded = PipelineDef.model_validate(dumped)
    assert validate_pipeline_def(reloaded).id == pipeline_id


def test_get_default_returns_a_deep_copy_not_the_registry_singleton():
    a = get_default("vision-mimic")
    a.stages[0].enabled = False
    b = get_default("vision-mimic")
    assert b.stages[0].enabled is True  # mutation of `a` must not leak into the registry


def test_get_default_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        get_default("does-not-exist")


def test_list_pipelines_summary_shape():
    summaries = list_pipelines()
    ids = {p["id"] for p in summaries}
    assert {
        "single-shot", "vision-mimic", "vision-mimic-multi", "vision-mimic-simplified",
        "chunk-segment-tags", "highlight-scan-analyze", "highlight-scan-critique-analyze",
    } <= ids
    for p in summaries:
        assert set(p) == {"id", "label", "description"}


# --------------------------------------------------------------------------- #
# chunk-segment-tags — registry defaults + fixed stage set
# --------------------------------------------------------------------------- #
def test_chunk_segment_tags_defaults_fixed_stage_set():
    pdef = get_default("chunk-segment-tags")
    assert [s.type for s in pdef.stages] == [
        "video_window", "chunk_segment", "chunk_analyze", "context_chain", "dedup",
    ]
    validate_pipeline_def(pdef)  # must not raise


def test_chunk_segment_tags_default_config_values():
    pdef = get_default("chunk-segment-tags")
    cs_cfg = ChunkSegmentConfig.model_validate(next(s for s in pdef.stages if s.type == "chunk_segment").config)
    ca_cfg = ChunkAnalyzeConfig.model_validate(next(s for s in pdef.stages if s.type == "chunk_analyze").config)
    assert cs_cfg.min_chunk_s == 5.0
    assert cs_cfg.max_chunk_s == 60.0
    assert cs_cfg.rough_fps == 1.0
    assert cs_cfg.analyze_all is False
    assert ca_cfg.overlap_s == 5.0
    assert ca_cfg.thinking is None


def test_chunk_segment_and_chunk_analyze_are_structural_cannot_disable():
    pdef = get_default("chunk-segment-tags")
    for node_type in ("chunk_segment", "chunk_analyze"):
        p = get_default("chunk-segment-tags")
        next(s for s in p.stages if s.type == node_type).enabled = False
        with pytest.raises(PipelineValidationError, match="structural"):
            validate_pipeline_def(p)
    validate_pipeline_def(pdef)  # unrelated copy untouched, still valid


def test_chunk_segment_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        ChunkSegmentConfig(bogus_field="x")


def test_chunk_analyze_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        ChunkAnalyzeConfig(bogus_field="x")


def test_chunk_segment_off_allowlist_model_rejected():
    pdef = get_default("chunk-segment-tags")
    next(s for s in pdef.stages if s.type == "chunk_segment").config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_chunk_analyze_off_allowlist_model_rejected():
    pdef = get_default("chunk-segment-tags")
    next(s for s in pdef.stages if s.type == "chunk_analyze").config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_chunk_segment_tags_unknown_config_key_rejected():
    pdef = get_default("chunk-segment-tags")
    next(s for s in pdef.stages if s.type == "chunk_segment").config["not_a_real_key"] = 123
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pdef)


def test_chunk_segment_tags_missing_stage_type_rejected():
    pdef = get_default("chunk-segment-tags")
    pdef.stages = [s for s in pdef.stages if s.type != "chunk_analyze"]
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)


# --------------------------------------------------------------------------- #
# highlight-scan-analyze — registry defaults + fixed stage set
# --------------------------------------------------------------------------- #
def test_highlight_scan_analyze_defaults_fixed_stage_set():
    pdef = get_default("highlight-scan-analyze")
    assert [s.type for s in pdef.stages] == ["video_window", "highlight_scan", "highlight_analyze"]
    validate_pipeline_def(pdef)  # must not raise


def test_highlight_scan_analyze_default_config_values():
    pdef = get_default("highlight-scan-analyze")
    hs_cfg = HighlightScanConfig.model_validate(next(s for s in pdef.stages if s.type == "highlight_scan").config)
    ha_cfg = HighlightAnalyzeConfig.model_validate(next(s for s in pdef.stages if s.type == "highlight_analyze").config)
    assert hs_cfg.rough_fps == 1.0
    assert hs_cfg.min_highlight_s == 3.0
    assert hs_cfg.max_highlight_s == 30.0
    assert hs_cfg.system_prompt is None
    assert hs_cfg.initial_prompt is None
    assert ha_cfg.fps == 1
    assert ha_cfg.thinking is None
    assert ha_cfg.preroll_s == 5.0
    assert ha_cfg.postroll_s == 4.0


def test_highlight_scan_and_highlight_analyze_are_structural_cannot_disable():
    for node_type in ("highlight_scan", "highlight_analyze"):
        p = get_default("highlight-scan-analyze")
        next(s for s in p.stages if s.type == node_type).enabled = False
        with pytest.raises(PipelineValidationError, match="structural"):
            validate_pipeline_def(p)
    validate_pipeline_def(get_default("highlight-scan-analyze"))  # unrelated copy untouched, still valid


def test_highlight_scan_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        HighlightScanConfig(bogus_field="x")


def test_highlight_analyze_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(bogus_field="x")


def test_highlight_analyze_fps_only_accepts_1_10_or_15():
    HighlightAnalyzeConfig(fps=1)
    HighlightAnalyzeConfig(fps=10)
    HighlightAnalyzeConfig(fps=15)
    # fps=5 was strictly dominated by fps=10 (build plan "Experiment
    # Findings") and stays excluded even though 15 was added — this is the
    # whole point of the test, not an incidental case.
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(fps=5)
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(fps=2)
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(fps=12)
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(fps=20)


def test_highlight_scan_system_prompt_and_initial_prompt_default_none_and_settable():
    assert HighlightScanConfig().system_prompt is None
    assert HighlightScanConfig().initial_prompt is None
    cfg = HighlightScanConfig(system_prompt="custom system text", initial_prompt="custom initial text")
    assert cfg.system_prompt == "custom system text"
    assert cfg.initial_prompt == "custom initial text"


def test_highlight_scan_off_allowlist_model_rejected():
    pdef = get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_scan").config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_analyze_off_allowlist_model_rejected():
    pdef = get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_analyze").config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_scan_analyze_unknown_config_key_rejected():
    pdef = get_default("highlight-scan-analyze")
    next(s for s in pdef.stages if s.type == "highlight_scan").config["not_a_real_key"] = 123
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pdef)


def test_highlight_scan_analyze_missing_stage_type_rejected():
    pdef = get_default("highlight-scan-analyze")
    pdef.stages = [s for s in pdef.stages if s.type != "highlight_analyze"]
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)


def test_vision_mimic_simplified_defaults():
    pdef = get_default("vision-mimic-simplified")
    assert {s.type for s in pdef.stages} == {
        "frame_sample", "detect_crop", "window", "analyze", "context_chain", "dedup",
    }
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    cfg = AnalyzeConfig.model_validate(analyze_stage.config)
    assert cfg.return_format == "simplified-tags-v1"
    assert cfg.model == "gemini-3.1-flash-lite"
    assert cfg.agent_mode == "single"
    # system_instruction defaults to the bundled taxonomy text (visible/editable
    # in the node's system_instruction card), not None.
    assert cfg.system_instruction == simplified_tags.DEFAULT_TAXONOMY_TEXT
    validate_pipeline_def(pdef)  # must not raise


def test_single_shot_defaults_match_todays_vlm_chat_contract():
    pdef = get_default("single-shot")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    cfg = AnalyzeConfig.model_validate(analyze_stage.config)
    assert cfg.model == "gemini-3.5-flash"
    assert cfg.thinking == "low"
    assert cfg.return_format == "events-v1"
    assert cfg.agent_mode == "single"
    assert cfg.system_instruction is None


def test_analyze_config_system_instruction_defaults_none_and_is_settable():
    assert AnalyzeConfig().system_instruction is None
    cfg = AnalyzeConfig(system_instruction="custom taxonomy text")
    assert cfg.system_instruction == "custom taxonomy text"


def test_vision_mimic_multi_defaults_agent_mode_multi():
    pdef = get_default("vision-mimic-multi")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    cfg = AnalyzeConfig.model_validate(analyze_stage.config)
    assert cfg.agent_mode == "multi"


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #
def test_unknown_pipeline_id_rejected():
    pdef = get_default("vision-mimic")
    pdef.id = "not-a-real-pipeline"
    with pytest.raises(PipelineValidationError, match="Unknown pipeline id"):
        validate_pipeline_def(pdef)


def test_unknown_config_key_rejected_400_equivalent():
    pdef = get_default("vision-mimic")
    detect_stage = next(s for s in pdef.stages if s.type == "detect_crop")
    detect_stage.config["not_a_real_key"] = 123
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pdef)


def test_bad_config_value_type_rejected():
    pdef = get_default("vision-mimic")
    sample_stage = next(s for s in pdef.stages if s.type == "frame_sample")
    sample_stage.config["sample_fps"] = "not-a-number"
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pdef)


def test_config_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DetectCropConfig(padding=-1.0)


def test_analyze_off_allowlist_model_rejected():
    pdef = get_default("single-shot")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    analyze_stage.config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_detect_crop_off_allowlist_model_rejected():
    pdef = get_default("vision-mimic")
    detect_stage = next(s for s in pdef.stages if s.type == "detect_crop")
    detect_stage.config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_analyze_allowlisted_model_swap_is_accepted():
    pdef = get_default("single-shot")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    analyze_stage.config["model"] = "gemini-3.1-pro-preview"
    validate_pipeline_def(pdef)  # must not raise


def test_structural_node_disabled_rejected():
    pdef = get_default("vision-mimic")
    window_stage = next(s for s in pdef.stages if s.type == "window")
    window_stage.enabled = False
    with pytest.raises(PipelineValidationError, match="structural"):
        validate_pipeline_def(pdef)


def test_non_structural_node_can_be_disabled():
    pdef = get_default("vision-mimic")
    dedup_stage = next(s for s in pdef.stages if s.type == "dedup")
    dedup_stage.enabled = False
    # Should NOT raise — dedup is not structural.
    validate_pipeline_def(pdef)


def test_missing_stage_type_rejected():
    pdef = get_default("vision-mimic")
    pdef.stages = [s for s in pdef.stages if s.type != "dedup"]
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)


def test_extra_stage_type_rejected():
    pdef = get_default("single-shot")
    pdef.stages.append(
        StageDef(id="rogue", type="dedup", label="Rogue Dedup", config={})
    )
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)


def test_duplicate_stage_id_rejected():
    pdef = get_default("single-shot")
    pdef.stages[1].id = pdef.stages[0].id
    with pytest.raises(PipelineValidationError, match="Duplicate stage id"):
        validate_pipeline_def(pdef)


def test_simplified_tags_v1_return_format_accepted_for_analyze():
    pdef = get_default("vision-mimic")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    analyze_stage.config["return_format"] = "simplified-tags-v1"
    validate_pipeline_def(pdef)  # must not raise — accepted on ANY vision-mimic-shaped pipeline


def test_simplified_tags_v1_with_multi_agent_rejected():
    pdef = get_default("vision-mimic-simplified")
    analyze_stage = next(s for s in pdef.stages if s.type == "analyze")
    analyze_stage.config["agent_mode"] = "multi"
    with pytest.raises(PipelineValidationError, match="does not support agent_mode='multi'"):
        validate_pipeline_def(pdef)


def test_reorder_within_fixed_set_is_allowed():
    pdef = get_default("vision-mimic")
    pdef.stages = list(reversed(pdef.stages))
    validated = validate_pipeline_def(pdef)
    assert validated.stages[0].type == "dedup"  # order preserved as given, not re-sorted


# --------------------------------------------------------------------------- #
# Node config models — extra="forbid" behavior directly
# --------------------------------------------------------------------------- #
def test_analyze_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        AnalyzeConfig(agent_mode="single", bogus_field="x")


def test_analyze_config_rejects_unknown_agent_mode():
    with pytest.raises(ValidationError):
        AnalyzeConfig(agent_mode="triple")


def test_analyze_config_rejects_unknown_return_format():
    with pytest.raises(ValidationError):
        AnalyzeConfig(return_format="xml-v9")


# =============================================================================== #
# v2 (2026-07-18-highlight-scan-critique-analyze-v2-plan.md) — ThinkingQualityMixin,
# HighlightCritiqueConfig, position/technique/validator sub-configs,
# max_validator_iterations bounds, and the new pipeline's registry wiring.
# =============================================================================== #
def test_thinking_quality_mixin_defaults_none_and_settable():
    m = ThinkingQualityMixin()
    assert m.thinking is None
    assert m.media_resolution is None
    m2 = ThinkingQualityMixin(thinking="high", media_resolution="medium")
    assert m2.thinking == "high" and m2.media_resolution == "medium"


def test_thinking_quality_mixin_rejects_unknown_thinking_level():
    with pytest.raises(ValidationError):
        ThinkingQualityMixin(thinking="ludicrous")


def test_thinking_quality_mixin_rejects_unknown_media_resolution():
    with pytest.raises(ValidationError):
        ThinkingQualityMixin(media_resolution="ultra")


def test_thinking_quality_mixin_rejects_unknown_key():
    with pytest.raises(ValidationError):
        ThinkingQualityMixin(bogus_field="x")


def test_highlight_scan_config_gains_mixin_fields_default_none():
    cfg = HighlightScanConfig()
    assert cfg.thinking is None
    assert cfg.media_resolution is None
    cfg2 = HighlightScanConfig(thinking="low", media_resolution="high")
    assert cfg2.thinking == "low" and cfg2.media_resolution == "high"


def test_highlight_critique_config_defaults_and_bounds():
    cfg = HighlightCritiqueConfig()
    assert cfg.model == "gemini-3.1-flash-lite"
    assert cfg.critique_backpad_s == 6.0
    assert cfg.thinking is None
    assert cfg.media_resolution is None
    HighlightCritiqueConfig(critique_backpad_s=0)  # ge=0 boundary
    HighlightCritiqueConfig(critique_backpad_s=30)  # le=30 boundary
    with pytest.raises(ValidationError):
        HighlightCritiqueConfig(critique_backpad_s=-0.01)
    with pytest.raises(ValidationError):
        HighlightCritiqueConfig(critique_backpad_s=30.01)


def test_highlight_critique_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        HighlightCritiqueConfig(bogus_field="x")


def test_position_technique_validator_axis_configs_default_model_and_mixin_fields():
    pos = PositionAxisConfig()
    tech = TechniqueAxisConfig()
    assert pos.model == "gemini-3.1-flash-lite"
    assert tech.model == "gemini-3.1-flash-lite"
    assert pos.thinking is None and pos.media_resolution is None
    assert tech.thinking is None and tech.media_resolution is None


def test_actor_axis_config_default_model_and_mixin_fields():
    """S12 Phase 1b — the fourth, independent actor axis (design §4.3):
    same defaults discipline as position/technique, NOT the validator's
    "high" media_resolution default (identity attribution is not flagged
    as needing HIGH res)."""
    actor = ActorAxisConfig()
    assert actor.model == "gemini-3.1-flash-lite"
    assert actor.thinking is None and actor.media_resolution is None


def test_actor_axis_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        ActorAxisConfig(bogus_field="x")


def test_validator_axis_config_media_resolution_defaults_high_unlike_other_axes():
    """LeCun design-pass finding: HIGH media_resolution is the untried lever
    for occlusion-limited techniques — reserved for the validator only."""
    val = ValidatorAxisConfig()
    assert val.media_resolution == "high"
    assert val.thinking is None
    # Still overridable, not hardcoded.
    val2 = ValidatorAxisConfig(media_resolution="low")
    assert val2.media_resolution == "low"


def test_position_technique_validator_configs_reject_unknown_key():
    for cls in (PositionAxisConfig, TechniqueAxisConfig, ValidatorAxisConfig):
        with pytest.raises(ValidationError):
            cls(bogus_field="x")


def test_highlight_analyze_config_gains_position_technique_validator_sub_configs():
    cfg = HighlightAnalyzeConfig()
    assert isinstance(cfg.position, PositionAxisConfig)
    assert isinstance(cfg.technique, TechniqueAxisConfig)
    assert isinstance(cfg.actor, ActorAxisConfig)
    assert isinstance(cfg.validator, ValidatorAxisConfig)
    # Retained top-level fields (v2 docstring: deliberately unused by the
    # real per-axis calls now, kept for allowlist/back-compat — see models.py).
    assert cfg.model == "gemini-3.1-flash-lite"
    assert cfg.thinking is None
    assert cfg.fps == 1
    assert cfg.preroll_s == 5.0
    assert cfg.postroll_s == 4.0


def test_highlight_analyze_config_max_validator_iterations_default_and_bounds():
    cfg = HighlightAnalyzeConfig()
    assert cfg.max_validator_iterations == 1  # v1 founder decision: single pass
    HighlightAnalyzeConfig(max_validator_iterations=1)  # ge=1 boundary
    HighlightAnalyzeConfig(max_validator_iterations=5)  # le=5 boundary
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(max_validator_iterations=0)
    with pytest.raises(ValidationError):
        HighlightAnalyzeConfig(max_validator_iterations=6)


def test_highlight_analyze_config_sub_configs_independently_overridable():
    cfg = HighlightAnalyzeConfig(
        position={"model": "gemini-3.5-flash", "thinking": "high"},
        technique={"media_resolution": "medium"},
        actor={"model": "gemini-3.5-flash", "media_resolution": "medium"},
        validator={"media_resolution": "low", "thinking": "high"},
        max_validator_iterations=3,
    )
    assert cfg.position.model == "gemini-3.5-flash"
    assert cfg.position.thinking == "high"
    assert cfg.technique.media_resolution == "medium"
    assert cfg.actor.model == "gemini-3.5-flash"
    assert cfg.actor.media_resolution == "medium"
    assert cfg.validator.media_resolution == "low"  # override away from the "high" default
    assert cfg.validator.thinking == "high"
    assert cfg.max_validator_iterations == 3


# --------------------------------------------------------------------------- #
# highlight-scan-critique-analyze — registry defaults + fixed stage set
# --------------------------------------------------------------------------- #
def test_highlight_scan_critique_analyze_defaults_fixed_stage_set():
    pdef = get_default("highlight-scan-critique-analyze")
    assert [s.type for s in pdef.stages] == [
        "video_window", "highlight_scan", "highlight_critique", "highlight_analyze",
    ]
    validate_pipeline_def(pdef)  # must not raise


def test_highlight_scan_critique_analyze_highlight_scan_analyze_both_registered():
    """ADDITIVE — the v1 pipeline stays registered untouched alongside v2."""
    assert "highlight-scan-analyze" in {p["id"] for p in list_pipelines()}
    assert "highlight-scan-critique-analyze" in {p["id"] for p in list_pipelines()}
    v1 = get_default("highlight-scan-analyze")
    assert [s.type for s in v1.stages] == ["video_window", "highlight_scan", "highlight_analyze"]


def test_highlight_scan_critique_analyze_all_three_gemini_stages_are_structural():
    for node_type in ("highlight_scan", "highlight_critique", "highlight_analyze"):
        p = get_default("highlight-scan-critique-analyze")
        next(s for s in p.stages if s.type == node_type).enabled = False
        with pytest.raises(PipelineValidationError, match="structural"):
            validate_pipeline_def(p)
    validate_pipeline_def(get_default("highlight-scan-critique-analyze"))  # untouched copy still valid


def test_highlight_critique_off_allowlist_model_rejected():
    pdef = get_default("highlight-scan-critique-analyze")
    next(s for s in pdef.stages if s.type == "highlight_critique").config["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_analyze_position_axis_off_allowlist_model_rejected():
    """Nested sub-config model fields are allowlist-validated too (registry.py
    extension), not just the top-level `model` field."""
    pdef = get_default("highlight-scan-critique-analyze")
    ha_stage = next(s for s in pdef.stages if s.type == "highlight_analyze")
    ha_stage.config["position"]["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_analyze_technique_axis_off_allowlist_model_rejected():
    pdef = get_default("highlight-scan-critique-analyze")
    ha_stage = next(s for s in pdef.stages if s.type == "highlight_analyze")
    ha_stage.config["technique"]["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_analyze_validator_axis_off_allowlist_model_rejected():
    pdef = get_default("highlight-scan-critique-analyze")
    ha_stage = next(s for s in pdef.stages if s.type == "highlight_analyze")
    ha_stage.config["validator"]["model"] = "gemini-1.0-nano-experimental"
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pdef)


def test_highlight_critique_unknown_config_key_rejected():
    pdef = get_default("highlight-scan-critique-analyze")
    next(s for s in pdef.stages if s.type == "highlight_critique").config["not_a_real_key"] = 123
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pdef)


def test_highlight_scan_critique_analyze_missing_stage_type_rejected():
    pdef = get_default("highlight-scan-critique-analyze")
    pdef.stages = [s for s in pdef.stages if s.type != "highlight_critique"]
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)


def test_highlight_scan_critique_analyze_extra_stage_type_rejected():
    pdef = get_default("highlight-scan-analyze")  # v1 pipeline, no highlight_critique in its fixed set
    pdef.stages.append(
        StageDef(id="rogue_critique", type="highlight_critique", label="Rogue Critique", config={})
    )
    with pytest.raises(PipelineValidationError, match="fixed stage set"):
        validate_pipeline_def(pdef)
