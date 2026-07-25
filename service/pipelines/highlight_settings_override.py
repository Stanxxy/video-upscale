"""Pre-analysis AI settings override builder — S12 pre-analysis AI settings
spec (``working_log/plans/2026-07-25-pre-analysis-ai-settings-spec.md``) §4.

Four additive, OPTIONAL ``TrackRequest`` fields (``analysis_model``,
``analysis_media_resolution``, ``analysis_fps``, ``analysis_thinking``) let a
caller of ``POST /track`` pick a GLOBAL model/quality/fps/thinking-level
override for the production ``highlight-scan-critique-analyze`` pipeline
(``worker/highlight_orchestrator.py::run_highlight_job``). Absent (all four
``None``) means "current engine defaults", byte-identical to today's
unconditional ``registry.get_default(PIPELINE_ID)`` call — this module's
``apply_analysis_settings_override`` is a no-op (returns the SAME def
untouched) in that case.

This is the ONE place the override is built, reused by BOTH:
- ``service/routes/jobs.py::create_track_job`` — fail-closed admission-time
  validation (``registry.validate_pipeline_def`` on the overridden def; an
  off-allowlist model or an invalid enum value 400s here, never a silent
  fallback to the default model/level — INS-055 discipline).
- ``service/worker/highlight_orchestrator.py::run_highlight_job`` — the
  actual dispatch-time pipeline construction.

**Scope note (spec §0 vs. this codebase's current state):** the spec's own
framing describes the pipeline as having 5 Gemini-calling nodes (``scan``,
``critique``, and 3 ``highlight_analyze`` axis sub-configs:
``position``/``technique``/``validator``) — written before the S12 Phase 1b
production-wiring design's fourth, independent ``actor`` axis
(``ActorAxisConfig``) landed on this same branch. The spec's OPERATIVE
contract language in §4 is "every node" / "every ``ThinkingQualityMixin``
node", not the enumerated list — so ``actor`` (a ``ThinkingQualityMixin``
node exactly like ``position``/``technique``/``validator`` today) IS included
in the override below. Excluding it would be exactly the kind of "invisible
exception" the spec's own §2.1 validator-quality nuance explicitly forbids,
just for a different node. Flagged here per the spec's own "sanity-check
during implementation" framing for judgment calls of this shape.
"""
from __future__ import annotations

from service.models import TrackRequest
from service.pipelines.models import PipelineDef

# The ONE production v2 pipeline this override targets (matches
# ``worker/highlight_orchestrator.py``'s own ``PIPELINE_ID`` constant —
# duplicated as a literal, not imported, to avoid a routes<->worker import
# edge; both mirror the SAME registered pipeline id string used throughout
# ``service/pipelines/registry.py``).
PIPELINE_ID = "highlight-scan-critique-analyze"

# ``highlight_analyze``'s ThinkingQualityMixin axis sub-configs that receive
# the GLOBAL model/thinking/media_resolution override (see module docstring's
# scope note re: ``actor``). ``highlight_analyze``'s own top-level
# ``model``/``thinking`` fields are NOT touched — they are documented,
# retained-but-UNUSED fields (see ``HighlightAnalyzeConfig``'s docstring),
# kept only for schema/allowlist-check shape parity with older tests; leaving
# them at the pipeline default avoids implying they do something they don't.
_HIGHLIGHT_ANALYZE_AXES: tuple[str, ...] = ("position", "technique", "actor", "validator")


def has_any_analysis_setting(request: TrackRequest) -> bool:
    """True when at least one of the four pre-analysis AI settings is set on
    the request — the single gate both callers (admission-time validation,
    dispatch-time pipeline construction) use to decide whether to touch the
    default pipeline def at all."""
    return (
        request.analysis_model is not None
        or request.analysis_media_resolution is not None
        or request.analysis_fps is not None
        or request.analysis_thinking is not None
    )


def _apply_top_level(config: dict, request: TrackRequest) -> None:
    """Patch one Gemini-calling node's config dict in place (``model``, and —
    for ``ThinkingQualityMixin`` nodes — ``thinking``/``media_resolution``).
    Only fields the request actually set are touched."""
    if request.analysis_model is not None:
        config["model"] = request.analysis_model
    if request.analysis_thinking is not None:
        config["thinking"] = request.analysis_thinking
    if request.analysis_media_resolution is not None:
        config["media_resolution"] = request.analysis_media_resolution


def apply_analysis_settings_override(pipeline: PipelineDef, request: TrackRequest) -> PipelineDef:
    """Return a copy of ``pipeline`` with the request's pre-analysis AI
    settings applied GLOBALLY across every Gemini-calling node's config.

    - All four fields absent -> returns the SAME ``pipeline`` object,
      completely untouched (byte-identical regression guarantee for the
      default path — the vast majority of traffic, spec §4/AC1).
    - ``analysis_model`` -> every Gemini-calling node's ``model`` field
      (``highlight_scan``, ``highlight_critique``, and ``highlight_analyze``'s
      ``position``/``technique``/``actor``/``validator`` sub-configs).
    - ``analysis_media_resolution`` -> every ``ThinkingQualityMixin`` node's
      ``media_resolution`` — INCLUDING overriding ``ValidatorAxisConfig``'s
      baked-in ``media_resolution="high"`` default (spec §2.1: an explicit
      user quality choice is global, no invisible exceptions). When this
      field is absent, every node's existing default (validator's `"high"`
      included) is left completely untouched.
    - ``analysis_fps`` -> ``HighlightAnalyzeConfig.fps`` only — the single
      fps-bearing config; every axis call reads this one shared value
      (``executors.highlight_analyze_node``'s ``cfg.fps``).
    - ``analysis_thinking`` -> every Gemini-calling node's ``thinking`` field
      (same node set as ``analysis_model``).

    Never mutates the input ``PipelineDef`` in place — takes its own
    ``model_copy(deep=True)`` even though ``registry.get_default`` already
    returns a deep copy, so callers never need to reason about aliasing.
    """
    if not has_any_analysis_setting(request):
        return pipeline

    patched = pipeline.model_copy(deep=True)
    for stage in patched.stages:
        if stage.type in ("highlight_scan", "highlight_critique"):
            _apply_top_level(stage.config, request)
        elif stage.type == "highlight_analyze":
            if request.analysis_fps is not None:
                stage.config["fps"] = request.analysis_fps
            for axis_name in _HIGHLIGHT_ANALYZE_AXES:
                axis_cfg = stage.config.get(axis_name)
                if axis_cfg is None:
                    continue
                _apply_top_level(axis_cfg, request)
    return patched
