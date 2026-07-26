"""Default pipeline definitions — ``single-shot``, ``vision-mimic``, ``vision-mimic-multi``.

Registry is architected for N pipelines (build plan §1.4): adding a 4th pipeline
is "add a dict entry", nothing structural. ``get_default`` returns a deep copy so
callers can mutate freely without corrupting the registry singleton.
"""
from __future__ import annotations

from pydantic import ValidationError

from service.pipelines.models import (
    NODE_CONFIG_MODELS,
    STRUCTURAL_NODE_TYPES,
    AnalyzeConfig,
    ChunkAnalyzeConfig,
    ChunkSegmentConfig,
    ContextChainConfig,
    DedupConfig,
    DetectCropConfig,
    FrameSampleConfig,
    HighlightAnalyzeConfig,
    HighlightCritiqueConfig,
    HighlightScanConfig,
    PipelineDef,
    StageDef,
    VideoWindowConfig,
    WindowConfig,
)
from service.pipelines.highlight_scan import DEFAULT_INITIAL_PROMPT, DEFAULT_SYSTEM_PROMPT
from service.pipelines.simplified_tags import DEFAULT_TAXONOMY_TEXT
from service.routes.qa_vlm import DETECT_MODELS, EVENT_MODELS

# Model-id allowlists per node type, reused (not redefined) from qa_vlm.py's
# live-verified per-task lists — same no-silent-swap discipline as
# ``qa_vlm._select_model``: an off-allowlist model is a 400 at validation time,
# never a late Gemini 4xx mid-stream.
_MODEL_ALLOWLIST_BY_NODE_TYPE: dict[str, list[str]] = {
    "analyze": EVENT_MODELS,
    "detect_crop": DETECT_MODELS,
    "chunk_segment": EVENT_MODELS,
    "chunk_analyze": EVENT_MODELS,
    "highlight_scan": EVENT_MODELS,
    "highlight_analyze": EVENT_MODELS,
    "highlight_critique": EVENT_MODELS,
}


def _single_shot() -> PipelineDef:
    return PipelineDef(
        id="single-shot",
        label="Single-shot",
        description=(
            "Today's /qa/vlm-chat behavior recast as a pipeline — native YouTube "
            "video window (no frame sampling), one Gemini call, Event JSON v1. "
            "Zero regression vs the existing QA page."
        ),
        stages=[
            StageDef(
                id="video_window", type="video_window", label="Video Window",
                enabled=True, config=VideoWindowConfig().model_dump(),
            ),
            StageDef(
                id="analyze", type="analyze", label="Analyze",
                enabled=True,
                config=AnalyzeConfig(
                    agent_mode="single",
                    model="gemini-3.5-flash",
                    thinking="low",
                    return_format="events-v1",
                ).model_dump(),
            ),
        ],
    )


def _vision_mimic(pipeline_id: str, label: str, agent_mode: str) -> PipelineDef:
    return PipelineDef(
        id=pipeline_id,
        label=label,
        description=(
            "Gemini-only mimic of the production vision engine: frame-sample -> "
            "per-frame role-labeled detect+crop-to-athletes -> sliding window -> "
            "analyze -> context-chain -> dedup. No tracking model, no upscale — "
            "see the plan's fidelity caveats (§6)."
        ),
        stages=[
            StageDef(
                id="frame_sample", type="frame_sample", label="Sample",
                enabled=True, config=FrameSampleConfig().model_dump(),
            ),
            StageDef(
                id="detect_crop", type="detect_crop", label="Detect + Crop",
                enabled=True, config=DetectCropConfig().model_dump(),
            ),
            StageDef(
                id="window", type="window", label="Window",
                enabled=True, config=WindowConfig().model_dump(),
            ),
            StageDef(
                id="analyze", type="analyze", label="Analyze",
                enabled=True,
                config=AnalyzeConfig(agent_mode=agent_mode).model_dump(),
            ),
            StageDef(
                id="context_chain", type="context_chain", label="Context Chain",
                enabled=True, config=ContextChainConfig().model_dump(),
            ),
            StageDef(
                id="dedup", type="dedup", label="Dedup",
                enabled=True, config=DedupConfig().model_dump(),
            ),
        ],
    )


def _vision_mimic_simplified() -> PipelineDef:
    """``vision-mimic-simplified`` — same fixed stage set as ``vision-mimic``
    (frame_sample -> detect_crop -> window -> analyze -> context_chain ->
    dedup), single-agent only, but the ``analyze`` node tags with the NEW
    experimental 4-axis simplified taxonomy (``return_format=
    "simplified-tags-v1"``, ``service/pipelines/simplified_tags.py``) instead
    of the production ``clips-v1`` schema. QA experiment only — see
    ``working_log/BJJ_TAXONOMY_SIMPLIFIED_GEMINI_TAGGING.md``. The ``analyze``
    node's ``system_instruction`` defaults to the bundled
    ``bjj_taxonomy_simplified.md`` text (still overridable via the node's
    system_instruction card, same as every other pipeline's analyze node)."""
    return PipelineDef(
        id="vision-mimic-simplified",
        label="Vision-engine mimic (simplified taxonomy)",
        description=(
            "Same frame-sample -> detect+crop -> window -> analyze -> context-chain -> "
            "dedup shape as vision-mimic, but the analyze node tags with the NEW "
            "simplified 4-axis taxonomy (position/actor/action_class/outcome + "
            "specific_technique_guess) instead of the production clips-v1 schema. "
            "QA experiment only — no tracking model, no upscale, not routed through "
            "analyzer.py's clips-v1 prompt."
        ),
        stages=[
            StageDef(
                id="frame_sample", type="frame_sample", label="Sample",
                enabled=True, config=FrameSampleConfig().model_dump(),
            ),
            StageDef(
                id="detect_crop", type="detect_crop", label="Detect + Crop",
                enabled=True, config=DetectCropConfig().model_dump(),
            ),
            StageDef(
                id="window", type="window", label="Window",
                enabled=True, config=WindowConfig().model_dump(),
            ),
            StageDef(
                id="analyze", type="analyze", label="Analyze",
                enabled=True,
                config=AnalyzeConfig(
                    agent_mode="single",
                    model="gemini-3.1-flash-lite",
                    return_format="simplified-tags-v1",
                    system_instruction=DEFAULT_TAXONOMY_TEXT,
                ).model_dump(),
            ),
            StageDef(
                id="context_chain", type="context_chain", label="Context Chain",
                enabled=True, config=ContextChainConfig().model_dump(),
            ),
            StageDef(
                id="dedup", type="dedup", label="Dedup",
                enabled=True, config=DedupConfig().model_dump(),
            ),
        ],
    )


def _chunk_segment_tags() -> PipelineDef:
    """``chunk-segment-tags`` — YouTube-URL-native, two-Gemini-phase pipeline,
    SEQUENTIAL analysis (never parallel — continuity over speed, per product
    decision). Fixed stage set: ``video_window -> chunk_segment -> chunk_analyze
    -> context_chain -> dedup``.

    PASS 1 (``chunk_segment``): ONE cheap rough-scan Gemini call classifies the
    whole scoped video into an ordered chunk map (Gracie's BJJ-correct reason
    enum: match_action/restart_reset/stoppage/pre_match/broadcast).

    PASS 2 (``chunk_analyze``): a SEQUENTIAL native-video loop over only the
    ``worth_analysis`` chunks, tagging each with the NEW time-keyed
    ``simplified-tags-time-v1`` format (``service/pipelines/simplified_tags.py``
    — chunk-relative seconds from Gemini, converted to absolute match seconds
    in Python). No frame sampling, no download, no tracking model, no upscale.
    """
    return PipelineDef(
        id="chunk-segment-tags",
        label="Chunk segment + tag (native)",
        description=(
            "YouTube-native two-pass pipeline: one cheap Gemini scan segments the "
            "scope into chunks (match_action/restart_reset/stoppage/pre_match/"
            "broadcast), then a SEQUENTIAL per-chunk native-video analyze pass tags "
            "only the worthy (match_action) chunks with the time-keyed 4-axis "
            "simplified taxonomy. No frame sampling, no tracking model, no upscale."
        ),
        stages=[
            StageDef(
                id="video_window", type="video_window", label="Video Window",
                enabled=True, config=VideoWindowConfig().model_dump(),
            ),
            StageDef(
                id="chunk_segment", type="chunk_segment", label="Chunk Segment",
                enabled=True, config=ChunkSegmentConfig().model_dump(),
            ),
            StageDef(
                id="chunk_analyze", type="chunk_analyze", label="Chunk Analyze",
                enabled=True, config=ChunkAnalyzeConfig().model_dump(),
            ),
            StageDef(
                id="context_chain", type="context_chain", label="Context Chain",
                enabled=True, config=ContextChainConfig().model_dump(),
            ),
            StageDef(
                id="dedup", type="dedup", label="Dedup",
                enabled=True, config=DedupConfig().model_dump(),
            ),
        ],
    )


def _highlight_scan_analyze() -> PipelineDef:
    """``highlight-scan-analyze`` — YouTube-URL-native, two-Gemini-phase
    pipeline, SEQUENTIAL analysis (never parallel — continuity over speed,
    same product decision as ``chunk-segment-tags``). Fixed stage set:
    ``video_window -> highlight_scan -> highlight_analyze``.

    Supersedes (conceptually, NOT in code — ``chunk-segment-tags`` stays
    registered untouched) the mechanical chunk-then-reassemble design: PASS 1
    finds highlights (not a full scene segmentation), PASS 2 analyzes each
    highlight finely with a selectable frame rate and a few seconds of
    surrounding pre/post-roll VIDEO context.

    PASS 1 (``highlight_scan``): ONE cheap rough-scan Gemini call returns an
    ORDERED list of ``{start_s, end_s}`` highlight spans — no reason enum, no
    ``worth_analysis`` (build plan "Product Decision": precision over recall
    accepted for this playground; the human confirms events downstream).

    PASS 2 (``highlight_analyze``): a SEQUENTIAL native-video loop over every
    highlight, each sent as
    ``[start_s - preroll_s, end_s + postroll_s]`` (clamped to the requested
    scope) at a selectable ``fps`` (1, 10, or 15 only — see
    ``HighlightAnalyzeConfig.fps`` for why fps=5 stays excluded). ONE
    taxonomy call per highlight (position + action_class + outcome, a single
    flat verdict — see ``HighlightAnalyzeConfig``'s single-call-cutover
    docstring) plus one independent actor/identity call.

    NO ``context_chain``, NO ``dedup`` stage — pre-roll video replaces the
    lossy text context chain. Event clipping (every emitted event is clipped
    to its highlight's OWN ``[start_s, end_s]``, never the pre/post-roll-
    expanded window) is NOT a full dedup replacement (corrected 2026-07-18,
    evaluator CHANGES-REQUIRED item 2 — earlier text here overclaimed that);
    see ``executors.highlight_analyze_node``'s docstring for the real
    residual boundary-fragmentation risk and how each sub-case is handled
    (one fixed, one accepted playground-scope limitation).
    """
    return PipelineDef(
        id="highlight-scan-analyze",
        label="Highlight scan + analyze (native)",
        description=(
            "YouTube-native two-pass pipeline: one cheap Gemini scan finds highlight "
            "spans worth a closer look (no scene segmentation, no reason enum), then a "
            "SEQUENTIAL per-highlight native-video analyze pass tags each highlight at a "
            "selectable frame rate with a few seconds of pre/post-roll video context. No "
            "context-chain, no dedup stage — event clipping bounds each highlight's own "
            "output; adjacent-highlight boundary fragmentation is a known, accepted "
            "limitation for close-together highlights (see engineering docs)."
        ),
        stages=[
            StageDef(
                id="video_window", type="video_window", label="Video Window",
                enabled=True, config=VideoWindowConfig().model_dump(),
            ),
            StageDef(
                id="highlight_scan", type="highlight_scan", label="Highlight Scan",
                enabled=True, config=HighlightScanConfig().model_dump(),
            ),
            StageDef(
                id="highlight_analyze", type="highlight_analyze", label="Highlight Analyze",
                enabled=True, config=HighlightAnalyzeConfig().model_dump(),
            ),
        ],
    )


def _highlight_scan_critique_analyze() -> PipelineDef:
    """``highlight-scan-critique-analyze`` — v2 build plan
    (``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``). ADDITIVE
    sibling of ``highlight-scan-analyze`` (that pipeline stays registered,
    untouched at the DAG-shape level) with ONE new stage inserted between
    scan and analyze: ``video_window -> highlight_scan -> highlight_critique
    -> highlight_analyze``.

    ``highlight_scan``: unchanged stage TYPE, evolved output (``description``/
    ``reasoning`` per highlight — see ``highlight_scan.build_highlight_schema``).

    ``highlight_critique`` (NEW): per scanned highlight, a backward-padded
    native-video window + the scan's own description, asking Gemini to
    confirm the description and locate the highlight's TRUE (possibly
    earlier) start — corrects Gemini's long-video timestamp drift (Gracie's
    VTG-drift finding). Output is ADDITIVE (``corrected_start_s``/
    ``corrected_end_s``/``critique_note``), never overwrites the scan's own
    bounds.

    ``highlight_analyze``: reads ``corrected_start_s``/``corrected_end_s``
    (falling back to the scan's own ``start_s``/``end_s`` when absent) as the
    AUTHORITATIVE bounds, then runs ONE taxonomy call (position + action_class
    + outcome, a single flat verdict — see ``HighlightAnalyzeConfig``'s
    single-call-cutover docstring / ``executors.highlight_analyze_node``)
    plus one independent actor/identity call. This SAME executor/config also
    serves ``highlight-scan-analyze`` (no ``highlight_critique`` stage there
    — ``corrected_*`` stays absent, so behavior falls back to the scan's own
    bounds unchanged).
    """
    return PipelineDef(
        id="highlight-scan-critique-analyze",
        label="Highlight scan + critique + analyze (native, v2)",
        description=(
            "v2: YouTube-native three-pass pipeline — a cheap Gemini scan finds highlight "
            "spans (with a body-movement description + reasoning), a per-highlight backward-"
            "padded critique call corrects long-video timestamp drift, then a single flat "
            "taxonomy verdict (position + action_class + outcome) plus an independent actor/"
            "identity call tag each highlight. No context-chain, no dedup stage — pre-roll "
            "video + event clipping (against the critique-corrected bounds) replace them."
        ),
        stages=[
            StageDef(
                id="video_window", type="video_window", label="Video Window",
                enabled=True, config=VideoWindowConfig().model_dump(),
            ),
            StageDef(
                id="highlight_scan", type="highlight_scan", label="Highlight Scan",
                enabled=True,
                # v2-ONLY playground pre-fill (2026-07-19): populate this
                # registry stage's own system_prompt/initial_prompt with the
                # bundled constants so GET /qa/pipeline-defaults?id=
                # highlight-scan-critique-analyze returns them POPULATED
                # (playground textareas show the founder's 4-category prompt
                # pre-filled). Deliberately NOT done on ``HighlightScanConfig``'s
                # own pydantic field default (stays None) and NOT done on
                # ``_highlight_scan_analyze()`` (v1)'s registry entry below —
                # v1's playground textarea keeps showing empty/None exactly as
                # before this change. Clearing either field here still falls
                # back through ``resolve_system_prompt``/``build_scan_prompt``
                # to the SAME bundled constants (see ``highlight_scan.py``),
                # so "empty textarea -> None -> bundled default" is unbroken.
                config=HighlightScanConfig(
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    initial_prompt=DEFAULT_INITIAL_PROMPT,
                ).model_dump(),
            ),
            StageDef(
                id="highlight_critique", type="highlight_critique", label="Highlight Critique",
                enabled=True, config=HighlightCritiqueConfig().model_dump(),
            ),
            StageDef(
                id="highlight_analyze", type="highlight_analyze", label="Highlight Analyze",
                enabled=True, config=HighlightAnalyzeConfig().model_dump(),
            ),
        ],
    )


DEFAULT_PIPELINES: dict[str, PipelineDef] = {
    "single-shot": _single_shot(),
    "vision-mimic": _vision_mimic("vision-mimic", "Vision-engine mimic", "single"),
    "vision-mimic-multi": _vision_mimic("vision-mimic-multi", "Vision-engine mimic (multi-agent)", "multi"),
    "vision-mimic-simplified": _vision_mimic_simplified(),
    "chunk-segment-tags": _chunk_segment_tags(),
    "highlight-scan-analyze": _highlight_scan_analyze(),
    "highlight-scan-critique-analyze": _highlight_scan_critique_analyze(),
}

# Fixed stage-type SET per pipeline id (order-independent) — used by fail-closed
# validation to enforce "FIXED stage set per pipeline; enable/disable/reorder
# within that set. No arbitrary node authoring" (build plan §1.1).
PIPELINE_STAGE_TYPES: dict[str, frozenset[str]] = {
    pid: frozenset(s.type for s in pdef.stages) for pid, pdef in DEFAULT_PIPELINES.items()
}


def get_default(pipeline_id: str) -> PipelineDef:
    """Return a deep copy of the named pipeline's default definition.

    Raises KeyError (caller converts to HTTP 404/400) for an unknown id — no
    fabricated/blank default.
    """
    return DEFAULT_PIPELINES[pipeline_id].model_copy(deep=True)


def list_pipelines() -> list[dict]:
    """Summary list for ``GET /qa/pipelines`` — id/label/description only (no
    stage bodies; fetch those via ``GET /qa/pipeline-defaults?id=``)."""
    return [
        {"id": p.id, "label": p.label, "description": p.description}
        for p in DEFAULT_PIPELINES.values()
    ]


class PipelineValidationError(ValueError):
    """Raised by ``validate_pipeline_def`` — the route layer converts this to HTTP 400.
    Never silently drops/coerces a bad definition (fail-closed, build plan §3.1)."""


def validate_pipeline_def(pipeline: PipelineDef) -> PipelineDef:
    """Fail-closed validation of a (possibly frontend-edited) ``PipelineDef``.

    Checks, in order:
    1. ``pipeline.id`` is a known registered pipeline (no arbitrary pipeline ids).
    2. The stage TYPE SET matches that pipeline's fixed set exactly (order and
       per-stage ``enabled``/``label``/``id`` may vary — no arbitrary node
       authoring, no dropped/added node types).
    3. Every stage's ``type`` has a config model (defensive; (1)+(2) already
       guarantee this for registered pipelines).
    4. Every stage's ``config`` dict validates against its node type's Pydantic
       model with ``extra="forbid"`` — unknown key or bad value -> 400.
    5. ``analyze``/``detect_crop`` stages' ``model`` field is checked against the
       corresponding qa_vlm allowlist (``EVENT_MODELS``/``DETECT_MODELS``) — an
       off-allowlist model id is rejected here, not left to a later Gemini 4xx.
    6. No structural node type (``STRUCTURAL_NODE_TYPES``) is disabled.
    7. An ``analyze`` stage with ``return_format="simplified-tags-v1"`` never pairs
       with ``agent_mode="multi"`` (no multi-agent path for that format — fail
       closed at validation time rather than a runtime error mid-stream).

    Returns the SAME ``PipelineDef`` (not a reconstructed one) on success so the
    caller keeps the exact stage order the request specified (reordering within
    the fixed set is allowed, per build plan §1.1).
    """
    if pipeline.id not in PIPELINE_STAGE_TYPES:
        raise PipelineValidationError(
            f"Unknown pipeline id {pipeline.id!r}; must be one of {sorted(PIPELINE_STAGE_TYPES)}."
        )

    expected_types = PIPELINE_STAGE_TYPES[pipeline.id]
    actual_types = frozenset(s.type for s in pipeline.stages)
    if actual_types != expected_types:
        raise PipelineValidationError(
            f"Pipeline {pipeline.id!r} has a fixed stage set {sorted(expected_types)}; "
            f"got {sorted(actual_types)}. Enable/disable/reorder is allowed, adding or "
            f"removing a stage type is not."
        )

    seen_ids: set[str] = set()
    for stage in pipeline.stages:
        if stage.id in seen_ids:
            raise PipelineValidationError(f"Duplicate stage id {stage.id!r} in pipeline {pipeline.id!r}.")
        seen_ids.add(stage.id)

        config_model = NODE_CONFIG_MODELS.get(stage.type)
        if config_model is None:  # pragma: no cover — guarded by the type-set check above
            raise PipelineValidationError(f"Unknown node type {stage.type!r} for stage {stage.id!r}.")

        try:
            validated_config = config_model.model_validate(stage.config)
        except ValidationError as e:
            raise PipelineValidationError(
                f"Invalid config for stage {stage.id!r} (type={stage.type!r}): {e}"
            ) from e

        allowlist = _MODEL_ALLOWLIST_BY_NODE_TYPE.get(stage.type)
        if allowlist is not None and validated_config.model not in allowlist:
            raise PipelineValidationError(
                f"Unsupported model {validated_config.model!r} for stage {stage.id!r} "
                f"(type={stage.type!r}); must be one of {allowlist}."
            )

        # highlight_analyze's `actor` sub-config carries its OWN `model` field
        # (ThinkingQualityMixin subclass) — same no-silent-swap discipline as
        # every top-level `model` field above, extended here since it's
        # nested, not reachable by the generic top-level check. (Position/
        # technique/validator sub-configs were deleted in the 2026-07-26
        # single-call re-scope — see HighlightAnalyzeConfig's docstring.)
        if stage.type == "highlight_analyze":
            axis_cfg = validated_config.actor
            if axis_cfg.model not in EVENT_MODELS:
                raise PipelineValidationError(
                    f"Unsupported model {axis_cfg.model!r} for stage {stage.id!r} "
                    f"(type={stage.type!r}, axis='actor'); must be one of {EVENT_MODELS}."
                )

        if stage.type in STRUCTURAL_NODE_TYPES and not stage.enabled:
            raise PipelineValidationError(
                f"Stage {stage.id!r} (type={stage.type!r}) is structural and cannot be disabled."
            )

        if (
            stage.type == "analyze"
            and validated_config.return_format == "simplified-tags-v1"
            and validated_config.agent_mode == "multi"
        ):
            raise PipelineValidationError(
                f"Stage {stage.id!r}: return_format='simplified-tags-v1' does not support "
                "agent_mode='multi' — the QA-layer 4-axis analyzer (service/pipelines/"
                "simplified_tags.py) has no multi-agent Judge synthesis path. Use "
                "agent_mode='single'."
            )

    return pipeline
