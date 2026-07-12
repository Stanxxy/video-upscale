"""Pipeline definition schema — ``PipelineDef``/``StageDef`` + per-node-type config
models + the fail-closed config allowlist (build plan §3.1/§3.4).

Every config model uses ``model_config = ConfigDict(extra="forbid")`` so an unknown
config key is a hard Pydantic ``ValidationError`` (converted to HTTP 400 by the
route layer) — never a silent drop. Every field default is the PRODUCTION constant
it mirrors (documented per-field) so a pipeline run with unedited defaults is,
modulo the documented fidelity gaps in the plan, as close to production as this
QA harness gets.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal[
    "video_window",
    "frame_sample",
    "detect_crop",
    "window",
    "analyze",
    "context_chain",
    "dedup",
    "chunk_segment",
    "chunk_analyze",
]

# Node types that structurally define the pipeline's shape — the UX layer locks
# these from being disabled; the backend is the authoritative enforcer (fail-closed,
# not just a UI nicety). See build plan §5 acceptance item 8.
#
# ``chunk_segment``/``chunk_analyze`` (the ``chunk-segment-tags`` pipeline's two
# Gemini-calling passes) are BOTH structural: disabling ``chunk_segment`` leaves
# ``chunk_analyze`` with no ``ctx.chunks`` to loop over (dead pipeline), and
# disabling ``chunk_analyze`` leaves the segmentation pass with no consumer —
# same "disabling either = dead pipeline" discipline as ``frame_sample``/``window``.
STRUCTURAL_NODE_TYPES: frozenset[str] = frozenset(
    {"video_window", "frame_sample", "window", "analyze", "chunk_segment", "chunk_analyze"}
)


# --------------------------------------------------------------------------- #
# Per-node-type config models
# --------------------------------------------------------------------------- #
class VideoWindowConfig(BaseModel):
    """``video_window`` — forwards the run request's YouTube scope untouched via
    native ``video_metadata`` offsets (today's ``/qa/vlm-chat`` input path). No
    tunable knobs of its own; present so the DAG has a visible "source" node."""

    model_config = ConfigDict(extra="forbid")


class FrameSampleConfig(BaseModel):
    """``frame_sample`` — server-side raw frame sampling at a fixed fps.

    ``sample_fps`` default 5.0 per plan §4.1 (``stride = round(native_fps / 5)``).
    """

    model_config = ConfigDict(extra="forbid")
    sample_fps: float = Field(default=5.0, gt=0, le=30.0)


class DetectCropConfig(BaseModel):
    """``detect_crop`` — per-frame role-labeled Gemini bbox detect, semantic top-2
    athlete selection, union+padded-square crop, identity-stable EMA smoothing.
    Defaults per plan §4.2-§4.4.

    ``model`` is allowlist-validated against ``qa_vlm.DETECT_MODELS`` at
    ``registry.validate_pipeline_def`` time (400 pre-stream, never a silent
    swap/late Gemini 4xx)."""

    model_config = ConfigDict(extra="forbid")
    model: str = "gemini-3.5-flash"
    padding: float = Field(default=0.2, ge=0.0, le=1.0)
    single_athlete_padding: float = Field(default=0.5, ge=0.0, le=1.0)
    min_crop_fraction: float = Field(default=0.4, gt=0.0, le=1.0)
    ema_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    hysteresis_margin: float = Field(default=0.15, ge=0.0, le=2.0)
    hysteresis_frames: int = Field(default=3, ge=1, le=30)
    crop_target_size: int = Field(default=768, ge=64, le=2048)


class WindowConfig(BaseModel):
    """``window`` — sliding window over crop frames. Defaults mirror production's
    standard mode (``WINDOW_SIZE=30``/``STRIDE=15``, ``upscale_setup.py:177-182``)."""

    model_config = ConfigDict(extra="forbid")
    window_size: int = Field(default=30, ge=1, le=200)
    stride: int = Field(default=15, ge=1, le=200)


class AnalyzeConfig(BaseModel):
    """``analyze`` — the Gemini call node. ``model``/``thinking`` default to the
    production analyzer default (``gemini-3.1-flash-lite``, no thinking_config);
    the ``single-shot`` pipeline overrides these to today's vlm-chat defaults
    (``gemini-3.5-flash``, thinking ``low``) in its own registry entry.
    ``return_format`` selects the response schema (build plan §3.5).

    ``model`` is allowlist-validated against ``qa_vlm.EVENT_MODELS`` at
    ``registry.validate_pipeline_def`` time (400 pre-stream, never a silent
    swap/late Gemini 4xx).

    ``system_instruction`` is the same user-loadable taxonomy/schema override
    qa_vlm.py already supports for ``/qa/vlm-chat`` — ``None``/default reads
    the repo's ``bjj_analysis_taxonomy.md`` exactly as production does; a
    non-empty value REPLACES that text (frame-based path: forwarded as
    ``taxonomy_text=`` to ``BJJTechniqueAnalyzer``/``BJJMultiAgentAnalyzer``;
    video-native path: forwarded as ``GenerateContentConfig.system_instruction``
    exactly like ``qa_vlm.vlm_chat``). Whitespace-only normalizes to ``None``
    (same discipline as ``qa_vlm._clean_system_instruction``).

    ``return_format="simplified-tags-v1"`` is a QA-EXPERIMENT format (NOT
    production clips-v1): a 4-axis taxonomy (position/actor/action_class/
    outcome + specific_technique_guess), built and enum-constrained entirely in
    ``service/pipelines/simplified_tags.py`` — never routed through
    ``analyzer.py``'s clips-v1 prompt. Only supported for ``agent_mode="single"``
    (no multi-agent Judge synthesis path for this format); ``system_instruction``
    for this format defaults (``None``) to the bundled
    ``bjj_taxonomy_simplified.md`` via ``simplified_tags.resolve_system_instruction``
    — same "empty means default" discipline as clips-v1/events-v1 above.
    """

    model_config = ConfigDict(extra="forbid")
    agent_mode: Literal["single", "multi"] = "single"
    model: str = "gemini-3.1-flash-lite"
    thinking: Optional[Literal["off", "low", "medium", "high"]] = None
    return_format: Literal["clips-v1", "events-v1", "simplified-tags-v1", "raw-text"] = "clips-v1"
    system_instruction: Optional[str] = None


class ContextChainConfig(BaseModel):
    """``context_chain`` — toggles whether ``current_context_summary`` is threaded
    from one window's analyze response into the next window's prompt. This node
    does no work of its own; its ``enabled`` flag is read by the ``analyze`` node
    executor (see executors.py) before/while it loops over windows."""

    model_config = ConfigDict(extra="forbid")


class DedupConfig(BaseModel):
    """``dedup`` — final clip merge (``pipeline.deduplicate_clips``, with the
    Brooks R6 ``key_by`` fix: bucket by ``actor_player_id`` by default, not the
    schema's absent ``role`` field).

    ``key_by`` applies to the ``clips-v1`` upstream format only. When the
    upstream ``analyze`` node's ``return_format`` is ``simplified-tags-v1``,
    the dedup executor always buckets by the format's own ``actor`` field
    (there is no ``actor_player_id``/``role`` equivalent to choose between) —
    see ``executors._apply_simplified_tags_dedup_keys``. When the upstream
    format is ``simplified-tags-time-v1`` (the ``chunk-segment-tags``
    pipeline), dedup runs a SECONDS-based merge instead (time-IoU + axis
    identity on ``action_class``+``actor``, keep-higher-confidence) — see
    ``service/pipelines/time_dedup.py``; ``key_by`` is not consulted there
    either. Any other upstream format (``events-v1``, ``raw-text``) is skipped
    with a recorded reason.
    """

    model_config = ConfigDict(extra="forbid")
    key_by: Literal["actor_player_id", "role"] = "actor_player_id"


class ChunkSegmentConfig(BaseModel):
    """``chunk_segment`` — PASS 1 of ``chunk-segment-tags``: ONE cheap Gemini call
    over the whole scoped native video (YouTube URL + ``video_metadata``), sampled
    at ``rough_fps`` (default 1.0, LOW media resolution — detecting scene CLASS,
    not reading grips). Returns an ORDERED chunk list; each chunk's
    ``reason`` is one of the Gracie's BJJ-correct enum (``match_action`` /
    ``restart_reset`` / ``stoppage`` / ``pre_match`` / ``broadcast`` — see
    ``service/pipelines/chunk_segment.py::REASON_VALUES``); ``worth_analysis`` is
    DERIVED in Python (``reason == "match_action"``), never asked of Gemini.

    ``min_chunk_s``/``max_chunk_s`` gate a deterministic bound-repair pass
    (``chunk_segment.repair_chunk_bounds``): sub-min chunks merge into a
    neighbor, over-max chunks split WITH a fixed overlap (so PASS 2's dedup
    seam-heals the split) rather than a naked cut at the boundary.

    ``analyze_all`` (default False) is a LeCun no-prefilter ablation switch: when
    True, PASS 2 (``chunk_analyze``) forces ``worth_analysis=True`` on every
    chunk regardless of ``reason`` — a control run, not a production default.

    ``model`` is allowlist-validated against ``qa_vlm.EVENT_MODELS`` at
    ``registry.validate_pipeline_def`` time (400 pre-stream, never a silent
    swap/late Gemini 4xx) — same no-silent-swap discipline as ``analyze``/
    ``detect_crop``.
    """

    model_config = ConfigDict(extra="forbid")
    model: str = "gemini-3.1-flash-lite"
    min_chunk_s: float = Field(default=5.0, gt=0.0)
    max_chunk_s: float = Field(default=60.0, gt=0.0)
    rough_fps: float = Field(default=1.0, gt=0.0, le=5.0)
    analyze_all: bool = False


class ChunkAnalyzeConfig(BaseModel):
    """``chunk_analyze`` — PASS 2 of ``chunk-segment-tags``: a SEQUENTIAL loop
    (never parallel — continuity over speed, per product decision) over ONLY
    the ``worth_analysis`` chunks from PASS 1. Each chunk is sent as a NATIVE
    YouTube video segment (URL + ``video_metadata`` start/end offsets, with an
    ``overlap_s`` back-overlap into the prior chunk — never a frame download).

    Locked to the ``simplified-tags-time-v1`` response format (chunk-relative
    seconds from Gemini, converted to ABSOLUTE match seconds in Python by
    adding the chunk's actual sent ``start_offset`` — see
    ``executors.chunk_analyze_node``). There is no ``return_format`` field
    here (unlike ``AnalyzeConfig``) because this node has exactly one output
    shape by design.

    ``model`` is allowlist-validated against ``qa_vlm.EVENT_MODELS`` at
    ``registry.validate_pipeline_def`` time — same discipline as ``analyze``/
    ``chunk_segment``.
    """

    model_config = ConfigDict(extra="forbid")
    model: str = "gemini-3.1-flash-lite"
    thinking: Optional[Literal["off", "low", "medium", "high"]] = None
    overlap_s: float = Field(default=5.0, ge=0.0)


NODE_CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "video_window": VideoWindowConfig,
    "frame_sample": FrameSampleConfig,
    "detect_crop": DetectCropConfig,
    "window": WindowConfig,
    "analyze": AnalyzeConfig,
    "context_chain": ContextChainConfig,
    "dedup": DedupConfig,
    "chunk_segment": ChunkSegmentConfig,
    "chunk_analyze": ChunkAnalyzeConfig,
}


# --------------------------------------------------------------------------- #
# Pipeline / stage envelope
# --------------------------------------------------------------------------- #
class StageDef(BaseModel):
    id: str
    type: NodeType
    label: str
    enabled: bool = True
    config: dict = Field(default_factory=dict)


class PipelineDef(BaseModel):
    schema_version: int = 1
    id: str
    label: str
    description: str = ""
    stages: list[StageDef]
