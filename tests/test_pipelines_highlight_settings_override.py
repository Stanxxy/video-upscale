"""S12 pre-analysis AI settings spec (2026-07-25) §4 —
``service/pipelines/highlight_settings_override.py``.

Covers the ENGINE leg's contract: absent-fields byte-identical regression
(AC1), each override landing on every Gemini-calling node's config incl. the
validator-quality nuance both ways (§2.1), and fps reaching ONLY
``HighlightAnalyzeConfig.fps``.
"""
from __future__ import annotations

from service.models import TrackRequest
from service.pipelines.highlight_settings_override import (
    PIPELINE_ID,
    apply_analysis_settings_override,
    has_any_analysis_setting,
)
from service.pipelines.registry import get_default, validate_pipeline_def


def _request(**kwargs) -> TrackRequest:
    return TrackRequest(bucket="b", key="k.mp4", **kwargs)


def _stage(pipeline, node_type):
    return next(s for s in pipeline.stages if s.type == node_type)


# --------------------------------------------------------------------------- #
# AC1 — absent fields: byte-identical to today's unconditional get_default()
# --------------------------------------------------------------------------- #
def test_all_fields_absent_returns_byte_identical_pipeline_def():
    request = _request()
    assert not has_any_analysis_setting(request)

    baseline = get_default(PIPELINE_ID)
    overridden = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    assert overridden.model_dump() == baseline.model_dump()


def test_all_fields_absent_returns_the_same_object_no_op():
    request = _request()
    pipeline = get_default(PIPELINE_ID)
    result = apply_analysis_settings_override(pipeline, request)
    assert result is pipeline  # true no-op, not merely equal


# --------------------------------------------------------------------------- #
# analysis_model -> every Gemini-calling node's model (scan/critique/
# position/technique/actor/validator)
# --------------------------------------------------------------------------- #
def test_analysis_model_override_lands_on_every_node():
    request = _request(analysis_model="gemini-3.1-pro-preview")
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    assert _stage(pipeline, "highlight_scan").config["model"] == "gemini-3.1-pro-preview"
    assert _stage(pipeline, "highlight_critique").config["model"] == "gemini-3.1-pro-preview"
    ha = _stage(pipeline, "highlight_analyze").config
    for axis in ("position", "technique", "actor", "validator"):
        assert ha[axis]["model"] == "gemini-3.1-pro-preview", axis
    validate_pipeline_def(pipeline)  # must not raise — allowlisted model


def test_analysis_model_off_allowlist_fails_validate_pipeline_def():
    from service.pipelines.registry import PipelineValidationError
    import pytest

    request = _request(analysis_model="gemini-1.0-nano-experimental")
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)
    with pytest.raises(PipelineValidationError, match="Unsupported model"):
        validate_pipeline_def(pipeline)


# --------------------------------------------------------------------------- #
# analysis_thinking -> every Gemini-calling node's thinking
# --------------------------------------------------------------------------- #
def test_analysis_thinking_override_lands_on_every_node():
    request = _request(analysis_thinking="high")
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    assert _stage(pipeline, "highlight_scan").config["thinking"] == "high"
    assert _stage(pipeline, "highlight_critique").config["thinking"] == "high"
    ha = _stage(pipeline, "highlight_analyze").config
    for axis in ("position", "technique", "actor", "validator"):
        assert ha[axis]["thinking"] == "high", axis
    validate_pipeline_def(pipeline)  # must not raise


# --------------------------------------------------------------------------- #
# analysis_media_resolution -> §2.1 validator-quality nuance, both directions
# --------------------------------------------------------------------------- #
def test_analysis_media_resolution_explicit_choice_overrides_validator_high_default():
    request = _request(analysis_media_resolution="low")
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    assert _stage(pipeline, "highlight_scan").config["media_resolution"] == "low"
    assert _stage(pipeline, "highlight_critique").config["media_resolution"] == "low"
    ha = _stage(pipeline, "highlight_analyze").config
    for axis in ("position", "technique", "actor"):
        assert ha[axis]["media_resolution"] == "low", axis
    # The critical nuance: an EXPLICIT "low" choice overrides the validator's
    # baked-in "high" default too — no invisible exception for this one node.
    assert ha["validator"]["media_resolution"] == "low"
    validate_pipeline_def(pipeline)  # must not raise


def test_analysis_media_resolution_absent_leaves_validator_high_default_intact():
    request = _request()  # no quality choice at all
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    ha = _stage(pipeline, "highlight_analyze").config
    assert ha["validator"]["media_resolution"] == "high"  # untouched baked-in default
    for axis in ("position", "technique", "actor"):
        assert ha[axis]["media_resolution"] is None  # untouched None default
    assert _stage(pipeline, "highlight_scan").config["media_resolution"] is None
    assert _stage(pipeline, "highlight_critique").config["media_resolution"] is None


def test_analysis_media_resolution_high_explicit_is_idempotent_with_validator_default():
    """An explicit 'high' choice should also land on every OTHER node (not
    just leave the validator's pre-existing high alone) — proves the
    override is unconditional, not merely a validator-preserving no-op."""
    request = _request(analysis_media_resolution="high")
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    ha = _stage(pipeline, "highlight_analyze").config
    for axis in ("position", "technique", "actor", "validator"):
        assert ha[axis]["media_resolution"] == "high", axis
    assert _stage(pipeline, "highlight_scan").config["media_resolution"] == "high"


# --------------------------------------------------------------------------- #
# analysis_fps -> HighlightAnalyzeConfig.fps ONLY
# --------------------------------------------------------------------------- #
def test_analysis_fps_override_reaches_highlight_analyze_config_only():
    request = _request(analysis_fps=15)
    baseline = get_default(PIPELINE_ID)
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)

    assert _stage(pipeline, "highlight_analyze").config["fps"] == 15
    # Nothing else changed — scan/critique carry no fps field at all, and no
    # other highlight_analyze key was touched.
    assert _stage(pipeline, "highlight_scan").config == _stage(baseline, "highlight_scan").config
    assert _stage(pipeline, "highlight_critique").config == _stage(baseline, "highlight_critique").config
    ha_before, ha_after = _stage(baseline, "highlight_analyze").config, _stage(pipeline, "highlight_analyze").config
    diff_keys = {k for k in ha_after if ha_after[k] != ha_before.get(k)}
    assert diff_keys == {"fps"}
    validate_pipeline_def(pipeline)  # must not raise — 15 is a valid literal


def test_analysis_fps_invalid_value_fails_validate_pipeline_def():
    from service.pipelines.registry import PipelineValidationError
    import pytest

    request = _request(analysis_fps=7)  # not in {1, 10, 15}
    pipeline = apply_analysis_settings_override(get_default(PIPELINE_ID), request)
    with pytest.raises(PipelineValidationError, match="Invalid config"):
        validate_pipeline_def(pipeline)


# --------------------------------------------------------------------------- #
# Opening the panel and changing exactly one control (spec AC2) — only that
# field's value shows up as non-default; the pipeline-level effect is scoped.
# --------------------------------------------------------------------------- #
def test_single_field_set_only_that_settings_field_is_non_none_on_request():
    request = _request(analysis_fps=1)
    assert request.analysis_model is None
    assert request.analysis_media_resolution is None
    assert request.analysis_thinking is None
    assert request.analysis_fps == 1
