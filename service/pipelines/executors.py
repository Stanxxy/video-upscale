"""Node executors + the streaming run loop (build plan §3.2/§3.4).

``NODE_EXECUTORS`` maps each ``NodeType`` to an async-generator function
``(ctx, config) -> AsyncIterator[dict]`` that mutates a shared ``RunContext`` and
yields NDJSON event payloads (sans envelope — ``run_pipeline`` adds
``stage_start``/``stage_complete`` around each stage and the top-level
``run_start``/``run_complete``).

Structural data flow through ``RunContext``:
- ``video_window`` sets ``ctx.video_scope``.
- ``frame_sample`` sets ``ctx.crops`` (identity pass-through of every sampled frame).
- ``detect_crop`` (if enabled) REPLACES ``ctx.crops`` with real detect+crop results,
  dropping frames with <2 usable persons (never fabricated).
- ``window`` sets ``ctx.windows`` (list of ``CropFrame`` batches).
- ``analyze`` is the one node that actually calls Gemini for match analysis; it
  branches on whether ``ctx.windows`` or ``ctx.video_scope`` is populated, loops
  over windows (if any) threading ``ctx.current_context`` (gated by whether the
  sibling ``context_chain`` stage is enabled), and always sets
  ``ctx.final_clips``/``ctx.final_format`` as the default terminal output.
- ``context_chain`` does no work of its own (a UI/timing marker only — its
  ``enabled`` flag is consulted by ``analyze`` before that loop runs).
- ``dedup`` (if enabled and upstream format == clips-v1) overwrites
  ``ctx.final_clips`` with the deduped list.
- ``chunk_segment``/``chunk_analyze`` are ``chunk-segment-tags``'s own PASS-1/
  PASS-2 pair (``ctx.chunks`` in between); ``highlight_scan``/
  ``highlight_analyze`` are ``highlight-scan-analyze``'s sibling pair
  (``ctx.highlights`` in between) — the newer pipeline has NO ``context_chain``/
  ``dedup`` stages at all (pre-roll video + per-highlight event clipping
  replace them; see ``highlight_analyze_node``'s docstring).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types
from PIL import Image

from analyzer import BJJMultiAgentAnalyzer, BJJTechniqueAnalyzer
from pipeline import deduplicate_clips
from service.pipelines import (
    chunk_segment,
    claude_frames,
    frame_source,
    gemini_cost,
    gemini_retry,
    highlight_axes,
    highlight_critique,
    highlight_scan,
    position_timeline,
    simplified_tags,
    time_dedup,
)
from service.pipelines.models import (
    AnalyzeConfig,
    ChunkAnalyzeConfig,
    ChunkSegmentConfig,
    DedupConfig,
    DetectCropConfig,
    FrameSampleConfig,
    HighlightAnalyzeConfig,
    HighlightCritiqueConfig,
    HighlightScanConfig,
    PipelineDef,
    PositionCritiqueConfig,
    PositionScanConfig,
    TechniqueAnalyzeConfig,
    TimelineReconcileConfig,
    WindowConfig,
)
from service.pipelines.gemini_model_compat import generation_sampling_kwargs
from service.routes.qa_vlm import (
    _clean_system_instruction,
    _event_schema,
    _resolve_stream_url,
    _run_detect,
    thinking_config_for,
)

logger = logging.getLogger("service.pipelines.executors")

# Pre-flight budget guard default (build plan §3.2). Counts multi-agent's 4x
# fan-out + per-frame detect calls. A caller may override per-run (see
# service/routes/qa_pipeline.py's ``budget_cap`` request field) — this is a
# QA-tool safety rail, not a hard product limit.
DEFAULT_BUDGET_CAP = 60

# Base64-encoded crop thumbnails are capped at this long-edge size to keep the
# NDJSON `crops` event payload reasonable — separate from the 768x768 image
# actually sent to Gemini (`DetectCropConfig.crop_target_size`).
_THUMBNAIL_LONG_EDGE = 320


# --------------------------------------------------------------------------- #
# RunContext
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    youtube_id: str
    youtube_url: str
    start_sec: int
    end_sec: int
    gemini_api_key: str
    request_timeout_ms: int
    taxonomy_path: Optional[str] = None
    # Anthropic key for Claude highlight_analyze axis path (empty = not configured).
    anthropic_api_key: str = ""

    # Transient-error retry-with-backoff policy for every QA-layer Gemini call
    # this RunContext drives (detect_crop, analyze, chunk_segment,
    # chunk_analyze). Defaults to GeminiRetryConfig()'s dataclass defaults
    # when not explicitly built from a live ServiceConfig (e.g. existing unit
    # tests that construct RunContext directly) — see
    # service/pipelines/gemini_retry.py.
    retry_config: gemini_retry.GeminiRetryConfig = field(default_factory=gemini_retry.GeminiRetryConfig)

    stream_url: Optional[str] = None
    stream_headers: dict = field(default_factory=dict)
    native_fps: Optional[float] = None

    crops: list = field(default_factory=list)  # list[frame_source.CropFrame]
    windows: Optional[list] = None  # list[list[frame_source.CropFrame]]
    video_scope_active: bool = False

    # ``chunk-segment-tags`` PASS 1 output: finalized chunk dicts
    # ({"index","start_s","end_s","reason","worth_analysis","adjustment"}),
    # set by ``chunk_segment_node``, consumed by ``chunk_analyze_node``. The
    # first non-frame RunContext intermediate — never touches ``native_fps``
    # (this pipeline is entirely native-video; ``native_fps`` stays ``None``
    # end-to-end, same "no unit fabrication" discipline as everywhere else).
    chunks: list = field(default_factory=list)

    # PASS 1's ``ChunkSegmentConfig.max_chunk_s`` — threaded through so PASS
    # 2's ``chunk_analyze_node`` can cap the OVERLAP-EXPANDED analyzed span at
    # the SAME bound the chunk map itself was repaired against (bug fix: the
    # back-overlap used to be added on top of a base chunk that was already
    # AT the cap after ``repair_chunk_bounds``'s over-max split, silently
    # pushing the real Gemini-analyzed window past ``max_chunk_s``). ``None``
    # when ``chunk_analyze_node`` runs without a preceding ``chunk_segment``
    # stage (e.g. directly in a unit test) — the clamp is then a no-op,
    # preserving prior behavior for callers that never set it.
    chunk_max_s: Optional[float] = None

    # ``highlight-scan-analyze`` PASS 1 output: finalized highlight dicts
    # ({"index","start_s","end_s","adjustment"}), set by
    # ``highlight_scan_node``, consumed by ``highlight_analyze_node``. Same
    # "entirely native-video, ``native_fps`` stays None" discipline as
    # ``chunks`` above — this pipeline never samples frames either.
    highlights: list = field(default_factory=list)

    # Grounded inputs and outputs for ``position-timeline-techniques``.
    # References are loaded before streaming starts; the model never receives
    # an unlabelled image or fabricates a player identity.
    player_references: list = field(default_factory=list)
    position_initial_state: Optional[dict] = None
    position_transitions: list = field(default_factory=list)
    position_intervals: list = field(default_factory=list)
    technique_events: list = field(default_factory=list)
    objective_statistics: dict = field(default_factory=dict)
    pipeline_error: Optional[str] = None
    highlight_scan_failed: bool = False

    current_context: str = "Start of match."
    raw_results: list = field(default_factory=list)  # [{"window":, "frames":, "analysis":}]
    degraded_counts: dict = field(default_factory=dict)

    final_clips: list = field(default_factory=list)
    final_format: str = "clips-v1"
    dedup_applied: bool = False
    dedup_skip_reason: Optional[str] = None

    stage_enabled: dict = field(default_factory=dict)  # node type -> enabled

    def gemini_client(self) -> genai.Client:
        return genai.Client(
            api_key=self.gemini_api_key,
            http_options=types.HttpOptions(timeout=self.request_timeout_ms),
        )

    def record_degraded(self, reason: Optional[str]) -> None:
        key = reason or "clean"
        self.degraded_counts[key] = self.degraded_counts.get(key, 0) + 1


def _thumbnail_b64(image: Image.Image) -> str:
    import base64
    import io

    thumb = frame_source.resize_long_edge(image, _THUMBNAIL_LONG_EDGE)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# video_window
# --------------------------------------------------------------------------- #
async def video_window_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    ctx.video_scope_active = True
    yield {"type": "progress", "stage_id": "video_window", "message": f"scope={ctx.start_sec}-{ctx.end_sec}s"}


# --------------------------------------------------------------------------- #
# frame_sample
# --------------------------------------------------------------------------- #
async def frame_sample_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = FrameSampleConfig.model_validate(config)
    ctx.stream_url, ctx.stream_headers = _resolve_stream_url(ctx.youtube_url)
    ctx.native_fps = frame_source.probe_native_fps(ctx.stream_url, ctx.stream_headers)
    sampled = frame_source.sample_frames(
        ctx.stream_url, ctx.stream_headers, ctx.start_sec, ctx.end_sec, ctx.native_fps, cfg.sample_fps,
    )
    ctx.crops = [
        frame_source.CropFrame(
            frame_index=s.frame_index,
            image=frame_source.resize_long_edge(s.image, 768),
            degraded=None,
        )
        for s in sampled
    ]
    yield {
        "type": "progress", "stage_id": "frame_sample",
        "message": f"sampled {len(sampled)} frames at {cfg.sample_fps}fps (native_fps={ctx.native_fps:.2f})",
        "frame_count": len(sampled),
    }


# --------------------------------------------------------------------------- #
# detect_crop
# --------------------------------------------------------------------------- #
async def detect_crop_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = DetectCropConfig.model_validate(config)
    client = ctx.gemini_client()
    tracker = frame_source.AthleteTracker(
        alpha=cfg.ema_alpha, margin=cfg.hysteresis_margin, min_streak=cfg.hysteresis_frames,
    )
    survivors: list = []
    thinking_off = thinking_config_for(cfg.model, "off")

    for i, crop in enumerate(ctx.crops):
        img_w, img_h = crop.image.size
        try:
            raw_objects, _model_used, model_ms = await _run_detect(
                client, crop.image, model=cfg.model, thinking_config=thinking_off,
                prompt=frame_source.MIMIC_DETECT_PROMPT, retry_config=ctx.retry_config,
            )
        except RuntimeError as e:
            yield {"type": "error", "stage_id": "detect_crop", "message": f"frame {crop.frame_index}: {e}"}
            continue

        detections = frame_source.to_detections(raw_objects, img_w, img_h)
        selection = frame_source.select_top2(detections, img_w, img_h)
        selection = tracker.resolve(selection, detections, img_w, img_h)
        ctx.record_degraded(selection.degraded)

        if selection.candidates is None:
            yield {
                "type": "progress", "stage_id": "detect_crop",
                "message": f"frame {crop.frame_index}: dropped (no_detection)",
                "frame_index": crop.frame_index, "degraded": "no_detection",
            }
            continue

        padding = cfg.single_athlete_padding if selection.degraded == "single_athlete" else cfg.padding
        square = tracker.smoothed_crop_box(selection.boxes_px, (img_h, img_w), padding, cfg.min_crop_fraction)
        cropped_img = frame_source.crop_and_resize(crop.image, square, cfg.crop_target_size)
        survivors.append(frame_source.CropFrame(
            frame_index=crop.frame_index, image=cropped_img, degraded=selection.degraded,
        ))
        yield {
            "type": "progress", "stage_id": "detect_crop",
            "message": f"frame {crop.frame_index}: cropped (degraded={selection.degraded})",
            "frame_index": crop.frame_index, "degraded": selection.degraded,
            "timing": {"model_ms": round(model_ms, 1)},
        }

    ctx.crops = survivors


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #
async def window_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = WindowConfig.model_validate(config)
    ctx.windows = frame_source.build_sliding_windows(ctx.crops, cfg.window_size, cfg.stride)
    yield {
        "type": "progress", "stage_id": "window",
        "message": f"{len(ctx.windows)} window(s) from {len(ctx.crops)} surviving frame(s)",
        "window_count": len(ctx.windows),
    }


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def _response_schema_for(return_format: str):
    if return_format == "clips-v1":
        return None  # analyzer.py's own _build_response_schema(player_ids)
    if return_format == "events-v1":
        return _event_schema()
    if return_format == "simplified-tags-v1":
        return simplified_tags.build_response_schema()
    return None  # raw-text — handled specially per code path


async def _analyze_frame_windows(ctx: RunContext, cfg: AnalyzeConfig) -> AsyncIterator[dict]:
    if cfg.return_format == "raw-text":
        yield {
            "type": "error", "stage_id": "analyze",
            "message": "return_format='raw-text' is not supported for a frame-based (vision-mimic) "
                       "analyze node — analyzer.py always requests structured JSON output. Use "
                       "'clips-v1' or 'events-v1'.",
        }
        return

    if cfg.return_format == "simplified-tags-v1" and cfg.agent_mode == "multi":
        # Fail-closed guard mirrored here (also caught earlier at
        # registry.validate_pipeline_def time) — the QA-layer 4-axis analyzer
        # has no multi-agent Judge synthesis path, never silently downgrades
        # to single-agent.
        yield {
            "type": "error", "stage_id": "analyze",
            "message": "return_format='simplified-tags-v1' does not support agent_mode='multi' "
                       "(single-agent only).",
        }
        return

    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)
    response_schema = _response_schema_for(cfg.return_format)

    if cfg.return_format == "simplified-tags-v1":
        # QA-layer direct Gemini call (service/pipelines/simplified_tags.py) —
        # NOT analyzer.py's BJJTechniqueAnalyzer/BJJMultiAgentAnalyzer, whose
        # prompt is hardcoded to the production clips-v1 taxonomy. Reuses the
        # shared Gemini client + thinking_config_for, same as every other
        # analyze path in this module.
        analyzer = simplified_tags.SimplifiedTagsAnalyzer(
            ctx.gemini_client(), model_id=cfg.model, thinking_config=thinking_cfg,
            system_instruction=cfg.system_instruction, retry_config=ctx.retry_config,
        )
    else:
        common_kwargs = dict(
            api_key=ctx.gemini_api_key, taxonomy_path=ctx.taxonomy_path,
            request_timeout_ms=ctx.request_timeout_ms,
            model_id=cfg.model, thinking_config=thinking_cfg,
            taxonomy_text=_clean_system_instruction(cfg.system_instruction),
        )
        analyzer = BJJMultiAgentAnalyzer(**common_kwargs) if cfg.agent_mode == "multi" else BJJTechniqueAnalyzer(**common_kwargs)
    chain_enabled = ctx.stage_enabled.get("context_chain", True)

    for window_no, batch in enumerate(ctx.windows or [], start=1):
        frame_indices = [c.frame_index for c in batch]
        images = [c.image for c in batch]
        yield {
            "type": "crops", "window": window_no, "frame_indices": frame_indices,
            "thumbnails": [
                {"frame_index": c.frame_index, "image_b64": _thumbnail_b64(c.image), "degraded": c.degraded}
                for c in batch
            ],
        }
        yield {"type": "window_start", "window": window_no, "frame_indices": frame_indices,
               "frame_count": len(batch), "native_fps": ctx.native_fps}

        ctx_for_call = ctx.current_context if chain_enabled else "Start of match."
        t0 = time.perf_counter()
        try:
            result_str = await analyzer.analyze_sequence_async(
                images, frame_indices, ctx_for_call, response_schema=response_schema,
            )
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
            yield {"type": "error", "stage_id": "analyze", "message": f"window {window_no}: {e}"}
            continue
        model_ms = (time.perf_counter() - t0) * 1000

        try:
            result_data = json.loads(result_str)
        except (ValueError, TypeError) as e:
            yield {"type": "error", "stage_id": "analyze", "message": f"window {window_no}: non-JSON response: {e}"}
            continue

        if cfg.return_format in ("clips-v1", "simplified-tags-v1"):
            # Both formats share the SAME top-level envelope
            # ({current_context_summary, clips}) by design — simplified-tags-v1
            # is a schema/prompt swap, not an envelope change, so it stays
            # context-chain compatible with zero extra branching here.
            clips = result_data.get("clips", [])
            if chain_enabled and "current_context_summary" in result_data:
                ctx.current_context = result_data["current_context_summary"]
        else:  # events-v1
            clips = result_data if isinstance(result_data, list) else []

        ctx.raw_results.append({"window": window_no, "frames": frame_indices, "analysis": {"clips": clips}})
        yield {
            "type": "window_result", "window": window_no, "format": cfg.return_format,
            "clips": clips, "context_summary": ctx.current_context if chain_enabled else None,
            "timing": {"model_ms": round(model_ms, 1)},
        }

    ctx.final_format = cfg.return_format
    ctx.final_clips = [c for r in ctx.raw_results for c in r["analysis"]["clips"]]


async def _analyze_video_window(ctx: RunContext, cfg: AnalyzeConfig) -> AsyncIterator[dict]:
    model = cfg.model
    thinking_cfg = thinking_config_for(model, cfg.thinking)
    client = ctx.gemini_client()
    system_instruction = _clean_system_instruction(cfg.system_instruction)

    video_part = types.Part(
        file_data=types.FileData(file_uri=ctx.youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{ctx.start_sec}s", end_offset=f"{ctx.end_sec}s", fps=1,
        ),
    )
    if cfg.return_format == "raw-text":
        prompt_text = (
            "Describe the BJJ techniques/events visible in this video clip in free text, "
            "in chronological order with approximate MM:SS timestamps."
        )
        gen_config = types.GenerateContentConfig(
            **generation_sampling_kwargs(model, temperature=0.2),
            thinking_config=thinking_cfg,
            system_instruction=system_instruction,
        )
    else:
        prompt_text = (
            "Detect every distinct BJJ technique/action event in this video clip. "
            "Report event start/end timestamps as MM:SS in the ORIGINAL video's absolute "
            "timeline (not relative to this clipped window). Return ONLY the JSON array "
            "matching the response schema — no prose."
        )
        gen_config = types.GenerateContentConfig(
            **generation_sampling_kwargs(model, temperature=0.2),
            response_mime_type="application/json",
            response_schema=_event_schema(), thinking_config=thinking_cfg,
            system_instruction=system_instruction,
        )

    yield {"type": "window_start", "window": 1, "scope": [ctx.start_sec, ctx.end_sec]}
    t0 = time.perf_counter()
    try:
        response = await gemini_retry.call_with_retry(
            lambda: client.aio.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
                config=gen_config,
            ),
            op_name="analyze:video_window",
            retry_config=ctx.retry_config,
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        yield {"type": "error", "stage_id": "analyze", "message": str(e)}
        return
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    if cfg.return_format == "raw-text":
        ctx.final_format = "raw-text"
        ctx.final_clips = []
        yield {
            "type": "window_result", "window": 1, "format": "raw-text",
            "raw_text": raw_text, "context_summary": None,
            "timing": {"model_ms": round(model_ms, 1)},
        }
        return

    try:
        clips = json.loads(raw_text)
        if not isinstance(clips, list):
            raise ValueError(f"Expected a JSON array, got {type(clips).__name__}")
    except (ValueError, TypeError) as e:
        yield {"type": "error", "stage_id": "analyze", "message": f"non-JSON-array response: {e}"}
        clips = []

    ctx.final_format = "events-v1"
    ctx.final_clips = clips
    ctx.raw_results.append({"window": 1, "frames": [], "analysis": {"clips": clips}})
    yield {
        "type": "window_result", "window": 1, "format": "events-v1",
        "clips": clips, "context_summary": None,
        "timing": {"model_ms": round(model_ms, 1)},
    }


async def analyze_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = AnalyzeConfig.model_validate(config)
    if ctx.windows is not None:
        async for event in _analyze_frame_windows(ctx, cfg):
            yield event
    elif ctx.video_scope_active:
        async for event in _analyze_video_window(ctx, cfg):
            yield event
    else:
        yield {
            "type": "error", "stage_id": "analyze",
            "message": "analyze node has neither ctx.windows nor ctx.video_scope_active set — "
                       "pipeline is missing a source stage (frame_sample+window or video_window).",
        }


# --------------------------------------------------------------------------- #
# chunk_segment — PASS 1 of chunk-segment-tags: one cheap rough-scan Gemini
# call producing an ordered chunk map (see service/pipelines/chunk_segment.py
# for the schema/prompt/bound-repair pure helpers).
# --------------------------------------------------------------------------- #
async def chunk_segment_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = ChunkSegmentConfig.model_validate(config)
    client = ctx.gemini_client()
    thinking_off = thinking_config_for(cfg.model, "off")

    video_part = types.Part(
        file_data=types.FileData(file_uri=ctx.youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{ctx.start_sec}s", end_offset=f"{ctx.end_sec}s", fps=cfg.rough_fps,
        ),
    )
    gen_config = types.GenerateContentConfig(
        **generation_sampling_kwargs(cfg.model, temperature=0.0),
        response_mime_type="application/json",
        response_schema=chunk_segment.build_chunk_segment_schema(),
        thinking_config=thinking_off,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    prompt_text = chunk_segment.build_segment_prompt(ctx.start_sec, ctx.end_sec)

    t0 = time.perf_counter()
    try:
        response = await gemini_retry.call_with_retry(
            lambda: client.aio.models.generate_content(
                model=cfg.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
                config=gen_config,
            ),
            op_name="chunk_segment",
            retry_config=ctx.retry_config,
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        yield {"type": "error", "stage_id": "chunk_segment", "message": str(e)}
        ctx.chunks = []
        return
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    try:
        data = json.loads(raw_text)
        raw_chunks = data.get("chunks", []) if isinstance(data, dict) else []
    except (ValueError, TypeError) as e:
        yield {"type": "error", "stage_id": "chunk_segment", "message": f"non-JSON response: {e}"}
        raw_chunks = []

    ctx.chunks = chunk_segment.finalize_chunks(raw_chunks, cfg.min_chunk_s, cfg.max_chunk_s, cfg.analyze_all)
    ctx.chunk_max_s = cfg.max_chunk_s

    worthy_count = sum(1 for c in ctx.chunks if c["worth_analysis"])
    yield {
        "type": "chunk_map", "chunks": ctx.chunks,
        "timing": {"model_ms": round(model_ms, 1)},
        # Additive-only (ADCC eval dashboard, Task 1): the PASS-1 rough-scan
        # call's prompt/raw response/token usage. Never replaces or renames
        # an existing key — a consumer that only reads "chunks"/"timing"
        # (the QA VLM Studio frontend, existing tests) is byte-identical.
        "prompt_text": prompt_text,
        "raw_response_text": raw_text,
        "usage_metadata": simplified_tags.extract_usage_metadata(response),
    }
    yield {
        "type": "progress", "stage_id": "chunk_segment",
        "message": f"{len(ctx.chunks)} chunk(s), {worthy_count} worth analyzing",
    }


# --------------------------------------------------------------------------- #
# chunk_analyze — PASS 2 of chunk-segment-tags: SEQUENTIAL native-video loop
# over only the worth_analysis chunks from PASS 1.
# --------------------------------------------------------------------------- #
async def chunk_analyze_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = ChunkAnalyzeConfig.model_validate(config)

    if not ctx.chunks:
        yield {
            "type": "error", "stage_id": "chunk_analyze",
            "message": "chunk_analyze node has no ctx.chunks — the chunk_segment stage must run "
                       "first and produce a non-empty chunk map.",
            # No specific chunk to attribute (there are zero) — key present
            # (None) for the same consistent-contract reason as the
            # highlight_analyze no-highlights guard (2026-07-18 fix).
            "chunk_index": None,
        }
        ctx.final_format = "simplified-tags-time-v1"
        ctx.final_clips = []
        return

    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=cfg.model, thinking_config=thinking_cfg,
        retry_config=ctx.retry_config,
    )
    chain_enabled = ctx.stage_enabled.get("context_chain", True)
    context = "Start of match."

    for chunk in ctx.chunks:
        if not chunk["worth_analysis"]:
            yield {
                "type": "progress", "stage_id": "chunk_analyze",
                "message": f"chunk {chunk['index']} skipped (reason={chunk['reason']})",
                "chunk_index": chunk["index"], "reason": chunk["reason"], "skipped_chunk": True,
            }
            continue

        # Native-video back-overlap into the prior chunk (LeCun overlap safety
        # net) — never before ctx.start_sec (the requested scope's own bound),
        # and never past ``ctx.chunk_max_s`` (PASS 1's max_chunk_s) either: the
        # cap is defined on the REAL analyzed span, not just the base chunk
        # bounds the chunk map displays. Without this second floor, a base
        # chunk already AT the cap (e.g. one of repair_chunk_bounds's
        # over-max split pieces) would silently grow past max_chunk_s once
        # overlap_s is subtracted — the exact "chunk over max length" bug.
        # Split pieces already carry their own dedup-healing overlap baked
        # into adjacent pieces' base start_s/end_s (repair_chunk_bounds), so
        # clamping this SEPARATE back-overlap to fit under the same cap loses
        # no seam-healing — it only trims back-overlap room that would have
        # pushed the window over budget.
        effective_start = max(ctx.start_sec, chunk["start_s"] - cfg.overlap_s)
        chunk_end = chunk["end_s"]
        if ctx.chunk_max_s is not None:
            effective_start = max(effective_start, chunk_end - ctx.chunk_max_s)

        yield {
            "type": "chunk_start", "chunk_index": chunk["index"],
            "scope": [effective_start, chunk_end], "reason": chunk["reason"],
        }

        ctx_for_call = context if chain_enabled else "Start of match."
        t0 = time.perf_counter()
        try:
            result_str = await analyzer.analyze_chunk(
                ctx.youtube_url, effective_start, chunk_end, ctx_for_call,
            )
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
            yield {
                "type": "error", "stage_id": "chunk_analyze",
                "message": f"chunk {chunk['index']}: {e}",
                "chunk_index": chunk["index"],
            }
            continue
        model_ms = (time.perf_counter() - t0) * 1000

        try:
            result_data = json.loads(result_str)
        except (ValueError, TypeError) as e:
            yield {
                "type": "error", "stage_id": "chunk_analyze",
                "message": f"chunk {chunk['index']}: non-JSON response: {e}",
                "chunk_index": chunk["index"],
            }
            continue

        # SimplifiedTagsTimeAnalyzer.analyze_chunk() never raises on a Gemini/
        # transport failure — it swallows the exception and returns
        # ``{"error": "..."}`` as its JSON payload so the surrounding
        # sequential loop can keep going. That means a PASS-2 outage parses
        # as perfectly valid JSON with no "clips" key, and naively falling
        # through to ``.get("clips", [])`` would silently render it as
        # "0 events found" — indistinguishable from a real clean chunk. Catch
        # that here and surface it as an explicit error event instead; the
        # chunk is marked errored (not aborted) and the loop CONTINUES to the
        # next chunk, consistent with the other error branches above (chunks
        # are independent units of work in this QA harness).
        if isinstance(result_data, dict) and "error" in result_data:
            yield {
                "type": "error", "stage_id": "chunk_analyze",
                "message": f"chunk {chunk['index']}: PASS-2 Gemini call failed: {result_data['error']}",
                "chunk_index": chunk["index"],
            }
            continue

        raw_clips = result_data.get("clips", []) if isinstance(result_data, dict) else []
        clips = _convert_chunk_relative_to_absolute(raw_clips, effective_start, chunk["index"])

        if chain_enabled and isinstance(result_data, dict) and "current_context_summary" in result_data:
            context = result_data["current_context_summary"]

        ctx.raw_results.append({"window": chunk["index"], "frames": [], "analysis": {"clips": clips}})
        yield {
            "type": "chunk_result", "chunk_index": chunk["index"], "format": "simplified-tags-time-v1",
            "clips": clips, "context_summary": context if chain_enabled else None,
            "timing": {"model_ms": round(model_ms, 1)},
            # Additive-only (ADCC eval dashboard, Task 1): the PASS-2 deep
            # call's prompt/raw response/token usage, read off the analyzer
            # instance's post-call instrumentation slots (never a change to
            # analyze_chunk's return contract — see SimplifiedTagsTimeAnalyzer
            # docstring). Absent (None) whenever analyzer isn't the real
            # SimplifiedTagsTimeAnalyzer (e.g. a test double that replaces
            # analyze_chunk entirely) — existing consumers ignoring these keys
            # are byte-identical.
            "prompt_text": getattr(analyzer, "last_prompt_text", None),
            "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
            "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
        }

    ctx.final_format = "simplified-tags-time-v1"
    ctx.final_clips = [c for r in ctx.raw_results for c in r["analysis"]["clips"]]


def _convert_chunk_relative_to_absolute(raw_clips: list, effective_start: float, chunk_index: int) -> list[dict]:
    """LeCun hard requirement: PASS 2 returns CHUNK-RELATIVE seconds; Python
    (never Gemini — it never sees the match clock) adds the chunk's actual
    sent ``start_offset`` (``effective_start``, which already includes the
    back-overlap) to produce ABSOLUTE match seconds. A clip missing/with
    non-numeric ``start_s``/``end_s`` is DROPPED (logged), never fabricated
    into a plausible absolute time."""
    out: list[dict] = []
    for clip in raw_clips:
        if not isinstance(clip, dict):
            continue
        try:
            rel_start = float(clip["start_s"])
            rel_end = float(clip["end_s"])
        except (KeyError, TypeError, ValueError):
            logger.warning("chunk_analyze: chunk %d dropping clip with malformed start_s/end_s: %r", chunk_index, clip)
            continue
        converted = dict(clip)
        converted["start_s"] = effective_start + rel_start
        converted["end_s"] = effective_start + rel_end
        out.append(converted)
    return out


# --------------------------------------------------------------------------- #
# highlight_scan — PASS 1 of highlight-scan-analyze: one cheap rough-scan
# Gemini call producing an ORDERED list of highlight spans (see
# service/pipelines/highlight_scan.py for the schema/prompt/bound-repair pure
# helpers). Deliberately simpler than chunk_segment_node: no reason enum, no
# worth_analysis derivation.
# --------------------------------------------------------------------------- #
async def highlight_scan_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = HighlightScanConfig.model_validate(config)
    client = ctx.gemini_client()
    # v2 (ThinkingQualityMixin): cfg.thinking/cfg.media_resolution default
    # For the unchanged default model, None preserves the prior budget-zero
    # behavior. Newer level-only models resolve to their lowest supported
    # level; media_resolution None still resolves to MEDIA_RESOLUTION_LOW.
    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)

    video_part = types.Part(
        file_data=types.FileData(file_uri=ctx.youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{ctx.start_sec}s", end_offset=f"{ctx.end_sec}s", fps=cfg.rough_fps,
        ),
    )
    gen_config = types.GenerateContentConfig(
        **generation_sampling_kwargs(cfg.model, temperature=0.0),
        response_mime_type="application/json",
        response_schema=highlight_scan.build_highlight_schema(),
        thinking_config=thinking_cfg,
        media_resolution=simplified_tags.media_resolution_enum(cfg.media_resolution),
        system_instruction=highlight_scan.resolve_system_prompt(cfg.system_prompt),
    )
    prompt_text = highlight_scan.build_scan_prompt(ctx.start_sec, ctx.end_sec, cfg.initial_prompt)

    t0 = time.perf_counter()
    try:
        response = await gemini_retry.call_with_retry(
            lambda: client.aio.models.generate_content(
                model=cfg.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
                config=gen_config,
            ),
            op_name="highlight_scan",
            retry_config=ctx.retry_config,
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        ctx.highlight_scan_failed = True
        yield {"type": "error", "stage_id": "highlight_scan", "message": str(e)}
        ctx.highlights = []
        return
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict) or not isinstance(data.get("highlights"), list):
            ctx.highlight_scan_failed = True
            yield {
                "type": "error",
                "stage_id": "highlight_scan",
                "message": "structured response requires a highlights array",
            }
            raw_highlights = []
        else:
            raw_highlights = data["highlights"]
    except (ValueError, TypeError) as e:
        ctx.highlight_scan_failed = True
        yield {"type": "error", "stage_id": "highlight_scan", "message": f"non-JSON response: {e}"}
        raw_highlights = []

    # scope_start/scope_end (live-bug fix, 2026-07-18): PASS 1 is an LLM and
    # can emit a highlight entirely outside the requested scope (real repro:
    # scope 40-85s, PASS 1 returned index 3 start_s=111/end_s=124) — dropped/
    # clamped here so the highlight_map streamed to the UI is scope-valid and
    # never flows an out-of-scope highlight into highlight_analyze_node's
    # preroll/postroll window math (which could otherwise invert). See
    # highlight_scan.repair_highlight_bounds docstring.
    ctx.highlights = highlight_scan.finalize_highlights(
        raw_highlights, cfg.min_highlight_s, cfg.max_highlight_s,
        scope_start=ctx.start_sec, scope_end=ctx.end_sec,
    )

    yield {
        "type": "highlight_map", "highlights": ctx.highlights,
        "timing": {"model_ms": round(model_ms, 1)},
        "prompt_text": prompt_text,
        "raw_response_text": raw_text,
        "usage_metadata": simplified_tags.extract_usage_metadata(response),
    }
    yield {
        "type": "progress", "stage_id": "highlight_scan",
        "message": f"{len(ctx.highlights)} highlight(s) found",
    }


# --------------------------------------------------------------------------- #
# highlight_critique — shared by highlight-scan-critique-analyze v2 and the
# Position Timeline technique lane. Per scanned
# highlight, sends a backward-padded and optionally forward-padded native-video
# window and asks Gemini to confirm the scan's description, locate the TRUE
# (possibly earlier) start, and retain the visible outcome in the corrected
# end. Mutates ctx.highlights IN PLACE, adding
# corrected_start_s/corrected_end_s/critique_note — ADDITIVE ONLY, never
# overwrites start_s/end_s (see service/pipelines/highlight_critique.py).
# --------------------------------------------------------------------------- #
async def highlight_critique_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = HighlightCritiqueConfig.model_validate(config)

    if not ctx.highlights:
        yield {
            "type": "error", "stage_id": "highlight_critique",
            "message": "highlight_critique node has no ctx.highlights — the highlight_scan stage "
                       "must run first and produce a non-empty highlight map.",
            "highlight_index": None,
        }
        return

    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=cfg.model, thinking_config=thinking_cfg,
        system_instruction=(
            cfg.system_prompt.strip()
            if cfg.system_prompt and cfg.system_prompt.strip()
            else highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT
        ),
        retry_config=ctx.retry_config,
    )
    critique_schema = highlight_critique.build_critique_schema()

    for highlight in ctx.highlights:
        # Backward context finds the setup; optional forward context lets the
        # corrected clip include the visible outcome. Legacy v2 keeps a zero
        # forward default, while Position Timeline opts into four seconds.
        window_start = round(max(ctx.start_sec, highlight["start_s"] - cfg.critique_backpad_s), 3)
        window_end = round(
            min(ctx.end_sec, highlight["end_s"] + cfg.critique_forwardpad_s), 3
        )

        yield {
            "type": "highlight_critique_start", "highlight_index": highlight["index"],
            "scope": [window_start, window_end],
        }

        # Reuse of the offset-quantization + inverted-window guard (INS-107 /
        # the existing highlight_analyze_node fix) — never send Gemini an
        # inverted or zero-length native-video offset.
        if window_start >= window_end:
            highlight["_critique_status"] = "failed"
            highlight["critique_note"] = "computed critique window is invalid"
            yield {
                "type": "error", "stage_id": "highlight_critique",
                "message": (
                    f"highlight {highlight['index']}: computed window "
                    f"[{window_start:.2f}, {window_end:.2f}] is invalid (start >= end) — "
                    "skipping, never sending an inverted/zero-length offset to Gemini."
                ),
                "highlight_index": highlight["index"],
            }
            continue

        t0 = time.perf_counter()
        try:
            result_str = await analyzer.analyze_chunk(
                ctx.youtube_url, window_start, window_end, None,
                response_schema=critique_schema,
                prompt_text=highlight_critique.build_critique_prompt(
                    # Gracie's EXACT token names ($window_start_sec/
                    # $window_end_sec/$critique_backpad_s/$scan_start_sec/
                    # $scan_end_sec/$description) — see
                    # highlight_critique.build_critique_prompt's docstring
                    # for why a name mismatch here would be a silent
                    # leftover-token bug (Template.safe_substitute never
                    # raises), and
                    # tests/test_pipelines_highlight_v2.py's
                    # test_critique_prompt_leaves_no_leftover_tokens for the
                    # regression guard.
                    window_start_sec=window_start, window_end_sec=window_end,
                    critique_backpad_s=cfg.critique_backpad_s,
                    critique_forwardpad_s=cfg.critique_forwardpad_s,
                    scan_start_sec=highlight["start_s"], scan_end_sec=highlight["end_s"],
                    description=highlight.get("description"),
                    override=cfg.initial_prompt,
                ),
                fps=cfg.fps, media_resolution=cfg.media_resolution,
            )
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
            highlight["_critique_status"] = "failed"
            highlight["critique_note"] = str(e)
            yield {
                "type": "error", "stage_id": "highlight_critique",
                "message": f"highlight {highlight['index']}: {e}",
                "highlight_index": highlight["index"],
            }
            continue
        model_ms = (time.perf_counter() - t0) * 1000

        try:
            result_data = json.loads(result_str)
        except (ValueError, TypeError) as e:
            highlight["_critique_status"] = "failed"
            highlight["critique_note"] = f"non-JSON response: {e}"
            yield {
                "type": "error", "stage_id": "highlight_critique",
                "message": f"highlight {highlight['index']}: non-JSON response: {e}",
                "highlight_index": highlight["index"],
            }
            continue

        if isinstance(result_data, dict) and "error" in result_data:
            highlight["_critique_status"] = "failed"
            highlight["critique_note"] = str(result_data["error"])
            yield {
                "type": "error", "stage_id": "highlight_critique",
                "message": f"highlight {highlight['index']}: PASS-2 Gemini call failed: {result_data['error']}",
                "highlight_index": highlight["index"],
            }
            continue

        # ABSOLUTE-seconds contract (see highlight_critique.py module
        # docstring): Gracie's prompt explicitly tells Gemini the window's
        # own absolute match-second bounds and asks for corrected_start_s/
        # corrected_end_s on THAT SAME clock — "never relative to this
        # clip's own 0:00". This is the OPPOSITE of every other
        # analyze_chunk caller in this package (which frame the clip as
        # starting at 0:00 and get chunk-relative seconds back). Do NOT run
        # these values through _convert_chunk_relative_to_absolute — they
        # are already absolute, only sanity-clamped below.
        raw_start = result_data.get("corrected_start_s")
        raw_end = result_data.get("corrected_end_s")
        movement_confirmed = result_data.get("movement_confirmed")

        applied = False
        if movement_confirmed is False:
            # Gate (evaluator LOW 1, 2026-07-18): Gemini explicitly said the
            # description does NOT match this clip — the schema doesn't
            # make corrected_start_s/corrected_end_s conditional on
            # movement_confirmed, so a model can (and did, live) still emit
            # them alongside movement_confirmed=false. Never apply a
            # correction in that case, even if the fields are present and
            # well-formed: the scan's ORIGINAL start_s/end_s stay
            # authoritative (corrected_* left absent, same fallback
            # highlight_analyze_node already uses). Surface the mismatch via
            # critique_note/telemetry instead of silently discarding it.
            highlight["critique_note"] = (
                result_data.get("note") or "movement_confirmed=false — original bounds kept, correction not applied"
            )
            highlight["_critique_status"] = "rejected"
        elif isinstance(raw_start, (int, float)) and isinstance(raw_end, (int, float)):
            # Sanity clamp (engineering decision, not spec-mandated): a
            # correction must land within [window_start, window_end] and have
            # positive duration, else it is DROPPED (never
            # fabricated/coerced into a plausible-looking value) — the
            # highlight then falls back to its original start_s/end_s in
            # highlight_analyze_node (corrected_* stays absent).
            corrected_start = max(window_start, min(float(raw_start), window_end))
            corrected_end = min(window_end, float(raw_end))
            if corrected_end > corrected_start:
                highlight["corrected_start_s"] = corrected_start
                highlight["corrected_end_s"] = corrected_end
                highlight["critique_note"] = result_data.get("note")
                highlight["_critique_status"] = "confirmed" if movement_confirmed is True else "failed"
                applied = True
            else:
                highlight["_critique_status"] = "failed"
                highlight["critique_note"] = result_data.get("note") or "invalid corrected critique bounds"
        else:
            highlight["_critique_status"] = "failed"
            highlight["critique_note"] = result_data.get("note") or "critique did not return valid corrected bounds"

        yield {
            "type": "highlight_critique_result", "highlight_index": highlight["index"],
            "movement_confirmed": result_data.get("movement_confirmed"),
            "corrected_start_s": highlight.get("corrected_start_s"),
            "corrected_end_s": highlight.get("corrected_end_s"),
            "critique_note": highlight.get("critique_note"),
            "applied": applied,
            "timing": {"model_ms": round(model_ms, 1)},
            "prompt_text": getattr(analyzer, "last_prompt_text", None),
            "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
            "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
        }

    corrected_count = sum(1 for h in ctx.highlights if h.get("corrected_start_s") is not None)
    yield {
        "type": "progress", "stage_id": "highlight_critique",
        "message": f"{corrected_count}/{len(ctx.highlights)} highlight(s) corrected",
    }


# --------------------------------------------------------------------------- #
# highlight_analyze — PASS 2 of highlight-scan-analyze (AND
# highlight-scan-critique-analyze — this executor is SHARED): SEQUENTIAL
# native-video loop over every highlight from PASS 1, each sent with
# pre/post-roll context video instead of a threaded text context_chain.
# --------------------------------------------------------------------------- #
def _clip_events_to_highlight_bounds(
    events: list[dict], bound_start: float, bound_end: float,
) -> tuple[list[dict], int]:
    """Bounds every emitted event to its highlight's OWN ``[bound_start,
    bound_end]`` — NOT the pre/post-roll-expanded window it was actually sent
    in (build plan "Target design" — Stage 2).

    **Honest correctness statement (corrected 2026-07-18, evaluator
    CHANGES-REQUIRED item 2 — earlier text here claimed this "replaces dedup"
    /  makes running with no dedup stage unconditionally "safe"; that
    overclaimed).** ``highlight_analyze_node`` sends Gemini a video window
    EXPANDED by ``preroll_s``/``postroll_s`` around each highlight's own
    span, so neighboring highlights' SENT windows routinely overlap in real
    video content even though their own ``[start_s, end_s]`` bounds do not.
    This function DOES prevent one event from satisfying two DIFFERENT
    highlights' clip tests from bounds alone (bounds themselves never
    overlap — merge only extends outward, split pieces are zero-overlap; see
    ``highlight_scan.py``). It does NOT prevent a single real action that
    genuinely spans the overlapping pre/post-roll content from being
    reported — independently, by two different Gemini calls — as two
    adjacent truncated fragments instead of one event. That residual case is
    handled by policy, not by this function: see ``highlight_scan.py``
    module docstring's "Correctness statement" and
    ``highlight_analyze_node``'s docstring below for the two residual
    sub-cases (manufactured split seams — FIXED via preroll/postroll
    clamping; genuinely-close distinct highlights — ACCEPTED, playground
    scope).

    Given events already converted to absolute seconds:
    - An event with ZERO overlap with the bounds (entirely in the pre-roll or
      post-roll region) is DROPPED.
    - An event that PARTIALLY overlaps the bounds (e.g. starts during
      pre-roll but continues into the highlight) is CLIPPED (its
      ``start_s``/``end_s`` truncated to the bounds), not dropped outright —
      the exchange is real and belongs to this highlight, only its reported
      edges overshot into the context video.

    Returns ``(kept_events, dropped_count)``. An event missing/with
    non-numeric ``start_s``/``end_s`` is also dropped (logged) — never
    fabricated, same discipline as ``_convert_chunk_relative_to_absolute``.
    """
    kept: list[dict] = []
    dropped = 0
    for event in events:
        try:
            start_s = float(event["start_s"])
            end_s = float(event["end_s"])
        except (KeyError, TypeError, ValueError):
            logger.warning("highlight_analyze: dropping event with malformed start_s/end_s: %r", event)
            dropped += 1
            continue
        if end_s <= bound_start or start_s >= bound_end:
            # Zero overlap with the highlight's OWN bounds — belongs entirely
            # to the pre-roll/post-roll context, not this highlight.
            dropped += 1
            continue
        clipped = dict(event)
        clipped["start_s"] = max(start_s, bound_start)
        clipped["end_s"] = min(end_s, bound_end)
        kept.append(clipped)
    return kept, dropped


async def _call_axis_json(
    analyzer: Optional["simplified_tags.SimplifiedTagsTimeAnalyzer"], youtube_url: str,
    window_start: float, window_end: float, *,
    response_schema, prompt_text: str, fps: float, media_resolution: Optional[str],
    ctx: Optional[RunContext] = None,
    model_id: Optional[str] = None,
    system_instruction: Optional[str] = None,
    thinking: Optional[str] = None,
    extra_parts: Optional[list[types.Part]] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Shared call+parse+error-check for the position/technique/validator
    axis calls in ``highlight_analyze_node`` — DRYs up the exact
    transport-exception / non-JSON / swallowed-``{"error": ...}`` pattern
    already used by every other Gemini-calling node in this module (never a
    fourth hand-copy of the same three checks). Returns ``(data, None)`` on
    success or ``(None, error_message)`` — never raises.

    When ``model_id`` is a Claude model, routes through
    ``claude_frames.analyze_axis_json`` (frame sample ≤20 images) instead of
    Gemini native video. ``media_resolution`` is Gemini-only and ignored on
    the Claude path.
    """
    if model_id and claude_frames.is_claude_model(model_id):
        if ctx is None:
            return None, "Claude axis call requires RunContext (missing)"
        try:
            data, raw_text = await claude_frames.analyze_axis_json(
                ctx,
                model_id=model_id,
                window_start=window_start,
                window_end=window_end,
                prompt_text=prompt_text,
                system_instruction=system_instruction or "",
                fps=fps,
                thinking=thinking,
            )
        except Exception as e:  # noqa: BLE001 — surface real yt-dlp/ffmpeg/Anthropic errors
            return None, str(e)
        if isinstance(data, dict) and "error" in data:
            return None, f"PASS-2 Claude call failed: {data['error']}"
        # Mirror SimplifiedTagsTimeAnalyzer.last_* so highlight_result.axis_calls
        # still carries raw JSON for the playground validator summary (Claude
        # has no Gemini analyzer instance — we write onto the telemetry holder).
        if analyzer is not None:
            analyzer.last_prompt_text = prompt_text
            analyzer.last_raw_response_text = raw_text
            analyzer.last_usage_metadata = None
        return data, None

    if analyzer is None:
        return None, "Gemini axis call requires SimplifiedTagsTimeAnalyzer (missing)"
    try:
        call_kwargs = {
            "response_schema": response_schema,
            "prompt_text": prompt_text,
            "fps": fps,
            "media_resolution": media_resolution,
        }
        if extra_parts is not None:
            call_kwargs["extra_parts"] = extra_parts
        result_str = await analyzer.analyze_chunk(
            youtube_url, window_start, window_end, None,
            **call_kwargs,
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
        return None, str(e)
    try:
        data = json.loads(result_str)
    except (ValueError, TypeError) as e:
        return None, f"non-JSON response: {e}"
    if isinstance(data, dict) and "error" in data:
        return None, f"PASS-2 Gemini call failed: {data['error']}"
    return data, None


def _effective_highlight_axis_model(cfg: HighlightAnalyzeConfig, axis_model: str) -> str:
    """Top-level Highlight Analyze ``model`` drives all three axis calls when
    it is a Claude id (UI Model dropdown). Otherwise keep per-axis models
    (Gemini playground defaults / independent overrides)."""
    if claude_frames.is_claude_model(cfg.model):
        return cfg.model
    return axis_model


async def highlight_analyze_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """PASS 2 of ``highlight-scan-analyze`` — SHARED, unchanged executor, ALSO
    ``highlight-scan-critique-analyze``'s PASS 3.

    **v2 restructure (build plan
    ``2026-07-18-highlight-scan-critique-analyze-v2-plan.md``, "Node 3";
    Gracie's real prompts, ``v2_node_prompts.md``, integrated 2026-07-18):**
    per highlight this runs an INDEPENDENT ``position`` call and
    ``technique`` call — Gracie: orthogonal, both always fire — never either/
    or — EXACTLY ONCE EACH (LeCun's own cost model, build plan design-pass:
    ``1 + N(3+L)`` — position/technique are a FLAT +2 per highlight; only the
    validator loops L times). Unlike an earlier engineering scaffold of this
    function (which modeled each axis as a LIST of chunk-relative-timestamped
    tags, merged by time-overlap, AND re-ran all three calls every
    iteration), Gracie's real prompts ask for exactly ONE flat verdict per
    axis per highlight — "Decide the ONE position value ... in this clip" /
    "Decide the ONE action_class value and the ONE outcome value ... in this
    clip" — no timestamps, no actor, no confidence score. A bounded
    ``for _ in range(cfg.max_validator_iterations)`` loop THEN re-invokes
    JUST the ``validator`` call (never position/technique again) to
    reconcile those two single, fixed verdicts (adversarial review + a
    strict ditch test — see ``highlight_axes.VALIDATOR_SYSTEM_PROMPT``),
    returning either ``verdict="agree"``/``"disagree"`` (with
    ``final_position``/``final_action_class``/``final_outcome`` — the SAME
    labels on agree, a correction on disagree) or ``verdict="ditch"`` (the
    WHOLE highlight is dropped, zero clips, ``status="ditched"``). The
    validator ``for`` loop is structurally bounded — it cannot hang
    regardless of how the validator answers; exhausting every iteration
    without an ``agree``/``disagree``/``ditch`` verdict falls through to the
    loop's own ``else`` clause, which ditches with an "unresolved
    disagreement" reason. v1 ships ``max_validator_iterations=1`` (single
    pass, founder decision) — an unresolved verdict on that one pass ditches
    immediately via this SAME fallthrough, never a special-cased "v1 vs v2"
    branch.

    A non-ditched highlight's ``clips`` list contains exactly ONE synthesized
    clip spanning the highlight's own AUTHORITATIVE bounds
    (``{start_s, end_s, position, action_class, outcome}`` plus the
    justification/evidence trail) — there is nothing left to clip/convert
    from chunk-relative seconds (none of the three axis calls report
    timestamps at all), so ``_convert_chunk_relative_to_absolute``/
    ``_clip_events_to_highlight_bounds`` (both still used by
    ``chunk_analyze_node`` and directly unit-tested, untouched) are no longer
    called from this function.

    ``corrected_start_s``/``corrected_end_s`` (set by the upstream
    ``highlight_critique`` stage, ABSENT when that stage didn't run — e.g.
    the OLDER ``highlight-scan-analyze`` pipeline, which has no
    ``highlight_critique`` stage at all) are read as the AUTHORITATIVE bounds
    for both window expansion and event clipping, falling back to the scan's
    own ``start_s``/``end_s`` when absent — this is what makes the critique
    stage's correction actually take effect downstream, and what makes this
    shared executor's behavior IDENTICAL to before for any highlight that was
    never critiqued.

    **Residual duplicate/fragmentation risk (evaluator CHANGES-REQUIRED item
    2, 2026-07-18 — read this before assuming "no dedup stage" is a fully
    solved problem):** ``preroll_s``/``postroll_s`` expand each highlight's
    SENT video window beyond its own authoritative bounds, so neighboring
    highlights' sent windows can share real video content even though their
    own bounds never overlap. A single real action that genuinely spans that
    shared content can be independently seen and reported by two different
    Gemini calls; ``_clip_events_to_highlight_bounds`` then truncates each
    report to its own highlight's bounds, yielding two adjacent fragments
    instead of one event. Two sub-cases, handled differently by product
    decision (see ``highlight_scan.py`` module docstring's "Correctness
    statement" for the full reasoning):

    1. **Split siblings of one over-long PASS-1 highlight** (this pipeline's
       own ``max_highlight_s`` bound-repair) — FIXED below: a highlight whose
       ``start_is_synthetic``/``end_is_synthetic`` flag is set (see
       ``highlight_scan.repair_highlight_bounds``) has ``preroll_s``/
       ``postroll_s`` clamped to ZERO on that synthetic edge, so sibling
       calls' sent windows share NO frames across a seam WE manufactured.
    2. **Two genuinely distinct PASS-1 highlights sitting closer together
       than ``preroll_s + postroll_s``** — ACCEPTED, not fixed: real
       pre/post-roll context doing its intended job on two independently
       judged highlights. Playground scope (precision over recall,
       human-in-the-loop confirms events) — see
       ``tests/test_pipelines_executors.py``'s pinned regression test for
       this exact, accepted behavior.
    """
    cfg = HighlightAnalyzeConfig.model_validate(config)

    if not ctx.highlights:
        yield {
            "type": "error", "stage_id": "highlight_analyze",
            "message": "highlight_analyze node has no ctx.highlights — the highlight_scan stage "
                       "must run first and produce a non-empty highlight map.",
            # No specific highlight to attribute (there are zero) — the key is
            # still present (None) so a frontend consumer can rely on
            # `highlight_index` always being a key on every highlight_analyze
            # error event, never only sometimes present (contract fix,
            # 2026-07-18).
            "highlight_index": None,
        }
        ctx.final_format = "simplified-tags-time-v1"
        ctx.final_clips = []
        return

    # Three independent Gemini analyzers when axes use Gemini; Claude models
    # skip SimplifiedTagsTimeAnalyzer (native YouTube video) and route through
    # claude_frames instead. Top-level cfg.model, when Claude, overrides all
    # three axes so the UI Model dropdown is not a no-op (v2 axes own model).
    # Claude still gets a lightweight telemetry holder (same last_* attrs the
    # highlight_result.axis_calls emission reads) so the UI validator summary
    # is not empty — see "No structured validator/axis output captured".
    pos_model = _effective_highlight_axis_model(cfg, cfg.position.model)
    tech_model = _effective_highlight_axis_model(cfg, cfg.technique.model)
    val_model = _effective_highlight_axis_model(cfg, cfg.validator.model)

    def _claude_axis_telem():
        return SimpleNamespace(
            last_prompt_text=None,
            last_raw_response_text=None,
            last_usage_metadata=None,
        )

    if claude_frames.is_claude_model(pos_model):
        position_analyzer = _claude_axis_telem()
    else:
        position_analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
            ctx.gemini_client(), model_id=pos_model,
            thinking_config=thinking_config_for(pos_model, cfg.position.thinking),
            system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT, retry_config=ctx.retry_config,
        )
    if claude_frames.is_claude_model(tech_model):
        technique_analyzer = _claude_axis_telem()
    else:
        technique_analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
            ctx.gemini_client(), model_id=tech_model,
            thinking_config=thinking_config_for(tech_model, cfg.technique.thinking),
            system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT, retry_config=ctx.retry_config,
        )
    if claude_frames.is_claude_model(val_model):
        validator_analyzer = _claude_axis_telem()
    else:
        validator_analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
            ctx.gemini_client(), model_id=val_model,
            thinking_config=thinking_config_for(val_model, cfg.validator.thinking),
            system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT, retry_config=ctx.retry_config,
        )
    position_schema = highlight_axes.build_position_schema()
    technique_schema = highlight_axes.build_technique_schema()
    validator_schema = highlight_axes.build_validator_schema()

    # No context_chain in this pipeline (build plan: "pre-roll video replaces
    # the lossy text context chain") — every highlight is analyzed with a
    # fresh "Start of the match." narrative context; the SimplifiedTagsTimeAnalyzer
    # default already covers that when previous_context is None.
    for highlight in ctx.highlights:
        # Authoritative bounds (v2): corrected_* (set by the upstream
        # highlight_critique stage) OVERRIDE the scan's own start_s/end_s for
        # window math + event clipping — ABSENT (None) whenever that stage
        # didn't run (the older highlight-scan-analyze pipeline, or a
        # highlight the critique call failed/skipped/declined to correct),
        # in which case this falls back to the scan's own bounds, byte-
        # identical to pre-v2 behavior.
        authoritative_start = highlight.get("corrected_start_s")
        if authoritative_start is None:
            authoritative_start = highlight["start_s"]
        authoritative_end = highlight.get("corrected_end_s")
        if authoritative_end is None:
            authoritative_end = highlight["end_s"]

        # Synthetic-seam clamp (evaluator CHANGES-REQUIRED item 2 fix): a
        # synthetic edge is a manufactured internal cut point from THIS
        # pipeline's own max-length split (highlight_scan.repair_highlight_bounds),
        # not a genuine PASS-1-reported boundary. Extending preroll/postroll
        # across it would send two sibling calls OVERLAPPING video content
        # across a seam we created ourselves — clamp to exactly the
        # highlight's own (authoritative) edge there instead. A genuine
        # (non-synthetic) edge still gets the full configured preroll/postroll,
        # clamped only to the requested scope.
        if highlight.get("start_is_synthetic", False):
            window_start = authoritative_start
        else:
            window_start = max(ctx.start_sec, authoritative_start - cfg.preroll_s)
        if highlight.get("end_is_synthetic", False):
            window_end = authoritative_end
        else:
            window_end = min(ctx.end_sec, authoritative_end + cfg.postroll_s)

        # Quantize ONCE, here, to millisecond precision (live-bug fix,
        # 2026-07-18 — real repro: preroll/postroll float subtraction/addition
        # e.g. 33.3 - 5.0 == 28.299999999999997, a 15-fractional-digit float
        # that overflows protobuf Duration's 9-digit limit and Gemini 400s).
        # ``simplified_tags._format_offset_seconds`` already hardens the
        # actual Gemini call against this independently (defense at the
        # serialization boundary), but rounding the SAME window_start/
        # window_end here too means the "highlight_start" scope event, the
        # inverted-window guard below, the analyze_chunk calls, and
        # ``_convert_chunk_relative_to_absolute`` all share one clean value
        # instead of the event/guard seeing raw float noise while only the
        # Gemini call sees a quantized one. The <1ms rounding delta folded
        # into the absolute clip times by ``_convert_chunk_relative_to_absolute``
        # is negligible — far below a video frame (~66ms at 15fps) — and is
        # intentional, not a bug.
        window_start = round(window_start, 3)
        window_end = round(window_end, 3)

        yield {
            "type": "highlight_start", "highlight_index": highlight["index"],
            "scope": [window_start, window_end],
            # "highlight_bounds" keeps its PRE-v2 meaning (the scan's OWN
            # original bounds) unconditionally — never swapped for the
            # corrected values, so an existing consumer reading this key sees
            # byte-identical semantics. "authoritative_bounds" is the NEW,
            # additive key carrying whichever bounds actually drove the
            # window/clipping math above (== highlight_bounds when this
            # highlight was never critiqued/corrected).
            "highlight_bounds": [highlight["start_s"], highlight["end_s"]],
            "authoritative_bounds": [authoritative_start, authoritative_end],
        }

        # Belt-and-suspenders (live-bug fix, 2026-07-18): highlight_scan.py's
        # scope-clamp (repair_highlight_bounds) is the real fix for a PASS-1
        # highlight landing outside the requested scope, but this check
        # guards here too — NEVER send Gemini an inverted or zero-length
        # native-video offset (observed real repro: scope 40-85s, an
        # out-of-scope highlight produced window_start=106 > window_end=85,
        # a genuine Gemini 500 INTERNAL). Skip this highlight entirely
        # (zero Gemini calls spent on it) rather than sending a malformed
        # request.
        if window_start >= window_end:
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": (
                    f"highlight {highlight['index']}: computed window "
                    f"[{window_start:.2f}, {window_end:.2f}] is invalid (start >= end) — "
                    "skipping, never sending an inverted/zero-length offset to Gemini. "
                    "This highlight's bounds are likely outside the requested scope."
                ),
                "highlight_index": highlight["index"],
            }
            continue

        status = "analyzed"
        ditch_reason: Optional[str] = None
        final_position: Optional[str] = None
        final_action_class: Optional[str] = None
        final_outcome: Optional[str] = None
        last_verdict: Optional[str] = None
        model_ms_total = 0.0
        validator_rounds = 0
        description = highlight.get("description")

        # Position + technique are each called EXACTLY ONCE per highlight
        # (build plan's own cost model, LeCun: "1 + N(3+L)" — position and
        # technique contribute a FLAT +2, only the validator (L) loops).
        # This is deliberately OUTSIDE the validator's bounded `for` below —
        # only the validator call itself repeats on disagreement, never a
        # re-ask of position/technique (no feedback-refinement is wired yet
        # for v2's argue-to-agreement loop; v1 never exercises a second
        # round at all, since max_validator_iterations defaults to 1).
        t0 = time.perf_counter()
        pos_data, pos_err = await _call_axis_json(
            position_analyzer, ctx.youtube_url, window_start, window_end,
            response_schema=position_schema, prompt_text=highlight_axes.build_position_prompt(description),
            fps=cfg.fps, media_resolution=cfg.position.media_resolution,
            ctx=ctx, model_id=pos_model,
            system_instruction=highlight_axes.POSITION_SYSTEM_PROMPT,
            thinking=cfg.position.thinking,
        )
        model_ms_total += (time.perf_counter() - t0) * 1000
        if pos_err is not None:
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": f"highlight {highlight['index']}: position call: {pos_err}",
                "highlight_index": highlight["index"],
            }
            continue

        t0 = time.perf_counter()
        tech_data, tech_err = await _call_axis_json(
            technique_analyzer, ctx.youtube_url, window_start, window_end,
            response_schema=technique_schema, prompt_text=highlight_axes.build_technique_prompt(description),
            fps=cfg.fps, media_resolution=cfg.technique.media_resolution,
            ctx=ctx, model_id=tech_model,
            system_instruction=highlight_axes.TECHNIQUE_SYSTEM_PROMPT,
            thinking=cfg.technique.thinking,
        )
        model_ms_total += (time.perf_counter() - t0) * 1000
        if tech_err is not None:
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": f"highlight {highlight['index']}: technique call: {tech_err}",
                "highlight_index": highlight["index"],
            }
            continue

        # ONE flat verdict per axis (see docstring) — never a list to merge.
        position_label = pos_data.get("position") if isinstance(pos_data, dict) else None
        position_justification = pos_data.get("justification") if isinstance(pos_data, dict) else None
        technique_label = tech_data.get("action_class") if isinstance(tech_data, dict) else None
        outcome_label = tech_data.get("outcome") if isinstance(tech_data, dict) else None
        technique_justification = tech_data.get("justification") if isinstance(tech_data, dict) else None

        validator_transport_error: Optional[str] = None
        for iteration in range(1, cfg.max_validator_iterations + 1):
            validator_rounds = iteration

            t0 = time.perf_counter()
            val_data, val_err = await _call_axis_json(
                validator_analyzer, ctx.youtube_url, window_start, window_end,
                response_schema=validator_schema,
                prompt_text=highlight_axes.build_validator_prompt(
                    description=description,
                    position_label=position_label, position_justification=position_justification,
                    technique_label=technique_label, outcome_label=outcome_label,
                    technique_justification=technique_justification,
                ),
                fps=cfg.fps, media_resolution=cfg.validator.media_resolution,
                ctx=ctx, model_id=val_model,
                system_instruction=highlight_axes.VALIDATOR_SYSTEM_PROMPT,
                thinking=cfg.validator.thinking,
            )
            model_ms_total += (time.perf_counter() - t0) * 1000
            if val_err is not None:
                validator_transport_error = val_err
                break

            verdict = val_data.get("verdict") if isinstance(val_data, dict) else None
            last_verdict = verdict
            if verdict == "ditch":
                status = "ditched"
                ditch_reason = (
                    (val_data.get("reason") if isinstance(val_data, dict) else None)
                    or "validator invoked the strict ditch criterion"
                )
                break
            if verdict == "agree":
                status = "analyzed"
                # Both labels survive the adversarial case unchanged —
                # ORIGINAL analyzer labels ship, never whatever (possibly
                # null, possibly stray) values the validator put in
                # final_*. There is nothing to trust from the validator's
                # echo here; the source of truth is the analyzer's own call.
                final_position = position_label
                final_action_class = technique_label
                final_outcome = outcome_label
                break
            if verdict == "disagree":
                status = "analyzed"
                # CONTRACT ENFORCEMENT (evaluator MEDIUM, 2026-07-18):
                # Gracie's prompt names EXACTLY ONE wrong_label — "disagree —
                # one specific label is wrong; say which one". Trusting all
                # three final_* fields (as an earlier version of this code
                # did) let a validator whose adversarial case only doubted
                # ONE axis silently overwrite the OTHER two too — a live-
                # demonstrated risk (validator doubts position, sets
                # wrong_label="action_class", but also returns a changed
                # final_position). Enforce the single-field-correction
                # property structurally: change ONLY the field
                # `wrong_label` names, force the other two back to the
                # ORIGINAL analyzer labels regardless of what the validator
                # put in their final_* slots.
                wrong_label = val_data.get("wrong_label") if isinstance(val_data, dict) else None
                if wrong_label == "position":
                    final_position = val_data.get("final_position") or position_label
                    final_action_class = technique_label
                    final_outcome = outcome_label
                elif wrong_label == "action_class":
                    final_position = position_label
                    final_action_class = val_data.get("final_action_class") or technique_label
                    final_outcome = outcome_label
                elif wrong_label == "outcome":
                    final_position = position_label
                    final_action_class = technique_label
                    final_outcome = val_data.get("final_outcome") or outcome_label
                else:
                    # wrong_label null/absent/unrecognized on a "disagree" —
                    # the validator didn't name which axis it disagrees
                    # with, so there is nothing safe to correct. Treat as
                    # NO CHANGE (keep every original label) rather than
                    # guessing which field to trust; note it for telemetry
                    # rather than silently corrupting a label.
                    final_position = position_label
                    final_action_class = technique_label
                    final_outcome = outcome_label
                    logger.warning(
                        "highlight_analyze: highlight %s validator returned verdict='disagree' "
                        "with no recognized wrong_label (%r) — keeping original labels unchanged.",
                        highlight["index"], wrong_label,
                    )
                break
            # verdict missing/unrecognized (schema enum should prevent this
            # in production — Gemini structured output enforces
            # agree/disagree/ditch — this branch exists for a malformed/mock
            # response): unresolved this round. v1
            # (max_validator_iterations=1) falls through to the for-loop's
            # own `else` below immediately; v2 (>1) would re-invoke JUST the
            # validator again here — no feedback-refinement is wired yet
            # (deferred to v2, per build plan).
        else:
            # Bounded-loop exhaustion (NEVER a `while True` — this `for...else`
            # only fires when every iteration ran to completion without a
            # `break`, i.e. the validator never reached an agree/disagree/
            # ditch verdict within `cfg.max_validator_iterations` rounds).
            # Structurally guaranteed to terminate: a permanently-
            # unresolved validator ditches here after exactly
            # `max_validator_iterations` rounds, never hangs.
            status = "ditched"
            ditch_reason = f"validator disagreement unresolved after {cfg.max_validator_iterations} iteration(s)"

        if validator_transport_error is not None:
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": f"highlight {highlight['index']}: validator call: {validator_transport_error}",
                "highlight_index": highlight["index"],
            }
            continue

        if status == "ditched":
            clips: list[dict] = []  # contract: a ditched highlight has zero clips by design
        else:
            # ONE synthesized clip spanning the highlight's own AUTHORITATIVE
            # bounds — neither axis call reports its own timestamps (Gracie's
            # real prompts: "you do not need to report timestamps"), so there
            # is nothing to clip/convert here; the window IS the event.
            clips = [{
                "start_s": authoritative_start, "end_s": authoritative_end,
                "position": final_position, "action_class": final_action_class, "outcome": final_outcome,
            }]

        ctx.raw_results.append({"window": highlight["index"], "frames": [], "analysis": {"clips": clips}})
        yield {
            "type": "highlight_result", "highlight_index": highlight["index"], "format": "simplified-tags-time-v1",
            "clips": clips,
            # "Ditched" contract (build plan item 6): sibling to "clips" on
            # THIS event, never nested inside the simplified-tags-time-v1
            # clip schema itself.
            "status": status, "ditch_reason": ditch_reason,
            "validator_rounds": validator_rounds, "verdict": last_verdict,
            "timing": {"model_ms": round(model_ms_total, 1)},
            # Per-axis telemetry (additive) — three independent calls replace
            # the single pre-v2 call, so there is no longer one
            # prompt_text/raw_response_text/usage_metadata triple; each
            # axis's own analyzer instance carries its OWN last-call
            # instrumentation (simplified_tags.SimplifiedTagsTimeAnalyzer),
            # surfaced here per-axis for debugging/playground display.
            "axis_calls": {
                "position": {
                    "prompt_text": getattr(position_analyzer, "last_prompt_text", None),
                    "raw_response_text": getattr(position_analyzer, "last_raw_response_text", None),
                    "usage_metadata": getattr(position_analyzer, "last_usage_metadata", None),
                },
                "technique": {
                    "prompt_text": getattr(technique_analyzer, "last_prompt_text", None),
                    "raw_response_text": getattr(technique_analyzer, "last_raw_response_text", None),
                    "usage_metadata": getattr(technique_analyzer, "last_usage_metadata", None),
                },
                "validator": {
                    "prompt_text": getattr(validator_analyzer, "last_prompt_text", None),
                    "raw_response_text": getattr(validator_analyzer, "last_raw_response_text", None),
                    "usage_metadata": getattr(validator_analyzer, "last_usage_metadata", None),
                },
            },
        }

    ctx.final_format = "simplified-tags-time-v1"
    ctx.final_clips = [c for r in ctx.raw_results for c in r["analysis"]["clips"]]


# --------------------------------------------------------------------------- #
# position-timeline-techniques — grounded continuous state + sparse events.
# --------------------------------------------------------------------------- #
def _timeline_player_ids(ctx: RunContext) -> tuple[str, str]:
    return position_timeline.validate_player_references(ctx.player_references)


def _timeline_analyzer(
    ctx: RunContext,
    *,
    model_id: str,
    thinking: Optional[str],
    system_instruction: str,
) -> simplified_tags.SimplifiedTagsTimeAnalyzer:
    return simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(),
        model_id=model_id,
        thinking_config=thinking_config_for(model_id, thinking),
        system_instruction=system_instruction,
        retry_config=ctx.retry_config,
    )


def _set_pipeline_error(ctx: RunContext, stage_id: str, message: str) -> dict:
    ctx.pipeline_error = message
    return {"type": "error", "stage_id": stage_id, "message": message}


async def position_scan_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """One whole-scope call producing an initial state + transition boundaries."""
    cfg = PositionScanConfig.model_validate(config)
    try:
        player_ids = _timeline_player_ids(ctx)
        reference_parts = position_timeline.build_reference_parts(ctx.player_references)
    except position_timeline.TimelineValidationError as error:
        yield _set_pipeline_error(ctx, "position_scan", str(error))
        return

    analyzer = _timeline_analyzer(
        ctx,
        model_id=cfg.model,
        thinking=cfg.thinking,
        system_instruction=(
            cfg.system_prompt.strip()
            if cfg.system_prompt and cfg.system_prompt.strip()
            else position_timeline.POSITION_SCAN_SYSTEM_PROMPT
        ),
    )
    data, error = await _call_axis_json(
        analyzer,
        ctx.youtube_url,
        ctx.start_sec,
        ctx.end_sec,
        response_schema=position_timeline.build_position_scan_schema(),
        prompt_text=position_timeline.build_position_scan_prompt(
            ctx.start_sec, ctx.end_sec, cfg.initial_prompt
        ),
        fps=cfg.rough_fps,
        media_resolution=cfg.media_resolution,
        ctx=ctx,
        model_id=cfg.model,
        system_instruction=position_timeline.POSITION_SCAN_SYSTEM_PROMPT,
        thinking=cfg.thinking,
        extra_parts=reference_parts,
    )
    if error is not None:
        yield _set_pipeline_error(ctx, "position_scan", f"position scan call failed: {error}")
        return
    if not isinstance(data, dict) or not isinstance(data.get("transitions"), list):
        yield _set_pipeline_error(
            ctx,
            "position_scan",
            "position scan response requires initial_state and a transitions array",
        )
        return

    transitions = []
    for index, item in enumerate(data["transitions"]):
        if not isinstance(item, dict):
            yield _set_pipeline_error(
                ctx, "position_scan", f"position scan transition[{index}] must be an object"
            )
            return
        transitions.append(
            {
                "index": index,
                "at_s": item.get("at_s"),
                "state_after": item.get("state_after"),
                "scan_evidence": item.get("evidence"),
                "confirmed": None,
                "corrected_at_s": None,
                "critique_note": None,
            }
        )

    # Validate the rough map before spending per-transition critique calls.
    rough_transitions = [
        {
            "at_s": item["at_s"],
            "corrected_at_s": item["at_s"],
            "confirmed": True,
            "state_after": item["state_after"],
        }
        for item in transitions
    ]
    try:
        position_timeline.build_position_intervals(
            ctx.start_sec,
            ctx.end_sec,
            data.get("initial_state") or {},
            rough_transitions,
            player_ids=player_ids,
        )
    except position_timeline.TimelineValidationError as validation_error:
        yield _set_pipeline_error(
            ctx, "position_scan", f"invalid position scan response: {validation_error}"
        )
        return

    ctx.position_initial_state = dict(data["initial_state"])
    ctx.position_transitions = transitions
    yield {
        "type": "position_map",
        "initial_state": ctx.position_initial_state,
        "transitions": copy.deepcopy(ctx.position_transitions),
        "player_ids": list(player_ids),
        "timing": {},
        "prompt_text": getattr(analyzer, "last_prompt_text", None),
        "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
        "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
    }


async def position_critique_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """Confirm/correct each proposed position boundary without inventing ranges."""
    cfg = PositionCritiqueConfig.model_validate(config)
    if ctx.position_initial_state is None:
        yield _set_pipeline_error(
            ctx, "position_critique", "position_critique requires a successful position_scan"
        )
        return
    try:
        player_ids = _timeline_player_ids(ctx)
        reference_parts = position_timeline.build_reference_parts(ctx.player_references)
    except position_timeline.TimelineValidationError as error:
        yield _set_pipeline_error(ctx, "position_critique", str(error))
        return

    system_instruction = (
        cfg.system_prompt.strip()
        if cfg.system_prompt and cfg.system_prompt.strip()
        else position_timeline.POSITION_CRITIQUE_SYSTEM_PROMPT
    )
    analyzer = _timeline_analyzer(
        ctx,
        model_id=cfg.model,
        thinking=cfg.thinking,
        system_instruction=system_instruction,
    )
    state_before = dict(ctx.position_initial_state)
    for transition in ctx.position_transitions:
        proposed_at = round(float(transition["at_s"]), 3)
        window_start = round(max(ctx.start_sec, proposed_at - cfg.backpad_s), 3)
        window_end = round(min(ctx.end_sec, proposed_at + cfg.forwardpad_s), 3)
        if window_start >= window_end:
            yield _set_pipeline_error(
                ctx,
                "position_critique",
                f"transition {transition['index']} has an invalid critique window",
            )
            return
        data, error = await _call_axis_json(
            analyzer,
            ctx.youtube_url,
            window_start,
            window_end,
            response_schema=position_timeline.build_position_critique_schema(),
            prompt_text=position_timeline.build_position_critique_prompt(
                window_start_s=window_start,
                window_end_s=window_end,
                proposed_at_s=proposed_at,
                state_before=state_before,
                state_after=transition["state_after"],
                override=cfg.initial_prompt,
            ),
            fps=cfg.fps,
            media_resolution=cfg.media_resolution,
            ctx=ctx,
            model_id=cfg.model,
            system_instruction=system_instruction,
            thinking=cfg.thinking,
            extra_parts=reference_parts,
        )
        if error is not None:
            yield _set_pipeline_error(
                ctx,
                "position_critique",
                f"transition {transition['index']} critique call failed: {error}",
            )
            return
        if not isinstance(data, dict) or not isinstance(data.get("confirmed"), bool):
            yield _set_pipeline_error(
                ctx,
                "position_critique",
                f"transition {transition['index']} critique returned an invalid object",
            )
            return

        confirmed = data["confirmed"]
        transition["confirmed"] = confirmed
        transition["critique_note"] = str(data.get("note") or "").strip()
        if confirmed:
            corrected_at = data.get("corrected_at_s")
            if not isinstance(corrected_at, (int, float)) or isinstance(corrected_at, bool):
                yield _set_pipeline_error(
                    ctx,
                    "position_critique",
                    f"transition {transition['index']} confirmed without numeric corrected_at_s",
                )
                return
            corrected_at = round(float(corrected_at), 3)
            if not window_start <= corrected_at <= window_end:
                yield _set_pipeline_error(
                    ctx,
                    "position_critique",
                    f"transition {transition['index']} corrected_at_s lies outside its critique window",
                )
                return
            transition["corrected_at_s"] = corrected_at
            transition["state_after"] = data.get("state_after")
            state_before = dict(transition["state_after"] or {})

        yield {
            "type": "position_critique_result",
            "transition_index": transition["index"],
            "original_at_s": transition["at_s"],
            "confirmed": confirmed,
            "corrected_at_s": transition["corrected_at_s"],
            "state_after": transition["state_after"],
            "critique_note": transition["critique_note"],
            "prompt_text": getattr(analyzer, "last_prompt_text", None),
            "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
            "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
        }

    try:
        position_timeline.build_position_intervals(
            ctx.start_sec,
            ctx.end_sec,
            ctx.position_initial_state,
            ctx.position_transitions,
            player_ids=player_ids,
        )
    except position_timeline.TimelineValidationError as validation_error:
        yield _set_pipeline_error(
            ctx, "position_critique", f"invalid corrected position map: {validation_error}"
        )


def _validated_technique_payload(
    data: dict,
    *,
    player_ids: tuple[str, str],
    context: str,
) -> dict:
    allowed_actors = {*player_ids, *position_timeline.TECHNIQUE_ACTOR_SENTINELS}
    actor = data.get("actor_player_id")
    if actor not in allowed_actors:
        raise position_timeline.TimelineValidationError(
            f"{context}: actor_player_id must be grounded to {sorted(allowed_actors)}"
        )
    action = data.get("action_class")
    if action not in simplified_tags.ACTION_CLASS_VALUES:
        raise position_timeline.TimelineValidationError(
            f"{context}: unsupported action_class {action!r}"
        )
    outcome = data.get("outcome")
    if outcome not in simplified_tags.OUTCOME_VALUES:
        raise position_timeline.TimelineValidationError(
            f"{context}: unsupported outcome {outcome!r}"
        )
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise position_timeline.TimelineValidationError(f"{context}: confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise position_timeline.TimelineValidationError(
            f"{context}: confidence must be between 0 and 1"
        )
    if not isinstance(data.get("identity_uncertain"), bool):
        raise position_timeline.TimelineValidationError(
            f"{context}: identity_uncertain must be boolean"
        )
    return {
        "actor_player_id": actor,
        "identity_uncertain": data["identity_uncertain"],
        "action_class": action,
        "outcome": outcome,
        "specific_technique_guess": str(data.get("specific_technique_guess") or "").strip(),
        "confidence": confidence,
        "evidence": str(data.get("evidence") or "").strip(),
    }


async def technique_analyze_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """Ground each critiqued highlight as a sparse technique-only event."""
    cfg = TechniqueAnalyzeConfig.model_validate(config)
    try:
        player_ids = _timeline_player_ids(ctx)
        reference_parts = position_timeline.build_reference_parts(ctx.player_references)
    except position_timeline.TimelineValidationError as error:
        yield _set_pipeline_error(ctx, "technique_analyze", str(error))
        return

    analyzer = _timeline_analyzer(
        ctx,
        model_id=cfg.model,
        thinking=cfg.thinking,
        system_instruction=position_timeline.TECHNIQUE_SYSTEM_PROMPT,
    )
    validator = _timeline_analyzer(
        ctx,
        model_id=cfg.model,
        thinking=cfg.validator_thinking,
        system_instruction=position_timeline.TECHNIQUE_VALIDATOR_SYSTEM_PROMPT,
    )
    technique_schema = position_timeline.build_technique_schema(player_ids)
    validator_schema = position_timeline.build_technique_validator_schema(player_ids)
    ctx.technique_events = []

    for highlight in ctx.highlights:
        critique_status = highlight.get("_critique_status")
        if critique_status == "rejected":
            yield {
                "type": "technique_result",
                "highlight_index": highlight["index"],
                "status": "ditched",
                "ditch_reason": highlight.get("critique_note") or "technique critique rejected the scanned movement",
                "event": None,
            }
            continue
        if critique_status == "failed":
            yield _set_pipeline_error(
                ctx,
                "technique_analyze",
                f"highlight {highlight['index']} has no valid critique result: "
                f"{highlight.get('critique_note') or 'unknown critique failure'}",
            )
            return
        original_start = float(highlight["start_s"])
        original_end = float(highlight["end_s"])
        authoritative_start = float(highlight.get("corrected_start_s", original_start))
        authoritative_end = float(highlight.get("corrected_end_s", original_end))
        if authoritative_end <= authoritative_start:
            yield _set_pipeline_error(
                ctx,
                "technique_analyze",
                f"highlight {highlight['index']} has non-positive authoritative bounds",
            )
            return
        window_start = round(max(ctx.start_sec, authoritative_start - cfg.preroll_s), 3)
        window_end = round(min(ctx.end_sec, authoritative_end + cfg.postroll_s), 3)
        yield {
            "type": "highlight_start",
            "highlight_index": highlight["index"],
            "scope": [window_start, window_end],
            "highlight_bounds": [original_start, original_end],
            "authoritative_bounds": [authoritative_start, authoritative_end],
        }
        prompt = position_timeline.build_technique_prompt(
            highlight.get("description"),
            start_s=authoritative_start,
            end_s=authoritative_end,
            player_ids=player_ids,
        )
        data, error = await _call_axis_json(
            analyzer,
            ctx.youtube_url,
            window_start,
            window_end,
            response_schema=technique_schema,
            prompt_text=prompt,
            fps=cfg.fps,
            media_resolution=cfg.media_resolution,
            ctx=ctx,
            model_id=cfg.model,
            system_instruction=position_timeline.TECHNIQUE_SYSTEM_PROMPT,
            thinking=cfg.thinking,
            extra_parts=reference_parts,
        )
        if error is not None:
            yield _set_pipeline_error(
                ctx,
                "technique_analyze",
                f"highlight {highlight['index']} technique call failed: {error}",
            )
            return
        try:
            proposed = _validated_technique_payload(
                data or {}, player_ids=player_ids, context=f"highlight {highlight['index']} analysis"
            )
        except position_timeline.TimelineValidationError as validation_error:
            yield _set_pipeline_error(ctx, "technique_analyze", str(validation_error))
            return

        validated, error = await _call_axis_json(
            validator,
            ctx.youtube_url,
            window_start,
            window_end,
            response_schema=validator_schema,
            prompt_text=position_timeline.build_technique_validator_prompt(
                highlight.get("description"), proposed
            ),
            fps=cfg.fps,
            media_resolution=cfg.validator_media_resolution,
            ctx=ctx,
            model_id=cfg.model,
            system_instruction=position_timeline.TECHNIQUE_VALIDATOR_SYSTEM_PROMPT,
            thinking=cfg.validator_thinking,
            extra_parts=reference_parts,
        )
        if error is not None:
            yield _set_pipeline_error(
                ctx,
                "technique_analyze",
                f"highlight {highlight['index']} validator call failed: {error}",
            )
            return
        try:
            verdict, intentional_visible = (
                position_timeline.validate_technique_validator_decision(
                    validated if isinstance(validated, dict) else {},
                    context=f"highlight {highlight['index']}",
                )
            )
        except position_timeline.TimelineValidationError as validation_error:
            yield _set_pipeline_error(ctx, "technique_analyze", str(validation_error))
            return
        axis_calls = {
            "technique": {
                "prompt_text": getattr(analyzer, "last_prompt_text", None),
                "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
                "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
            },
            "validator": {
                "prompt_text": getattr(validator, "last_prompt_text", None),
                "raw_response_text": getattr(validator, "last_raw_response_text", None),
                "usage_metadata": getattr(validator, "last_usage_metadata", None),
            },
        }
        if verdict == "ditch":
            yield {
                "type": "technique_result",
                "highlight_index": highlight["index"],
                "status": "ditched",
                "ditch_reason": str(validated.get("reason") or "validator found no technique event"),
                "event": None,
                "axis_calls": axis_calls,
            }
            continue
        try:
            final_payload = _validated_technique_payload(
                validated,
                player_ids=player_ids,
                context=f"highlight {highlight['index']} validator",
            )
        except position_timeline.TimelineValidationError as validation_error:
            yield _set_pipeline_error(ctx, "technique_analyze", str(validation_error))
            return

        event = {
            "start_s": round(authoritative_start, 3),
            "end_s": round(authoritative_end, 3),
            "position": None,
            "position_note": "",
            **final_payload,
            "original_bounds": [round(original_start, 3), round(original_end, 3)],
            "corrected_bounds": (
                [round(authoritative_start, 3), round(authoritative_end, 3)]
                if "corrected_start_s" in highlight or "corrected_end_s" in highlight
                else None
            ),
            "scan_description": highlight.get("description"),
            "scan_reasoning": highlight.get("reasoning"),
            "critique_note": highlight.get("critique_note"),
            "validator_verdict": verdict,
            "validator_reason": str(validated.get("reason") or "").strip(),
        }
        ctx.technique_events.append(event)
        yield {
            "type": "technique_result",
            "highlight_index": highlight["index"],
            "status": "analyzed",
            "ditch_reason": None,
            "event": copy.deepcopy(event),
            "axis_calls": axis_calls,
        }


async def timeline_reconcile_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """Build exact intervals, attach sparse events, and aggregate objective time."""
    TimelineReconcileConfig.model_validate(config)
    if ctx.position_initial_state is None:
        yield _set_pipeline_error(
            ctx, "timeline_reconcile", "timeline reconciliation requires position scan output"
        )
        return
    try:
        player_ids = _timeline_player_ids(ctx)
        ctx.position_intervals = position_timeline.build_position_intervals(
            ctx.start_sec,
            ctx.end_sec,
            ctx.position_initial_state,
            ctx.position_transitions,
            player_ids=player_ids,
        )
        ctx.technique_events = position_timeline.attach_techniques_to_intervals(
            ctx.technique_events, ctx.position_intervals
        )
        ctx.objective_statistics = position_timeline.build_objective_statistics(
            ctx.position_intervals,
            ctx.technique_events,
            player_ids=player_ids,
        )
    except position_timeline.TimelineValidationError as validation_error:
        yield _set_pipeline_error(ctx, "timeline_reconcile", str(validation_error))
        return

    ctx.final_format = "position-technique-timeline-v1"
    ctx.final_clips = copy.deepcopy(ctx.technique_events)
    yield {
        "type": "timeline_result",
        "player_ids": list(player_ids),
        "position_intervals": copy.deepcopy(ctx.position_intervals),
        "technique_events": copy.deepcopy(ctx.technique_events),
        "statistics": copy.deepcopy(ctx.objective_statistics),
    }


# --------------------------------------------------------------------------- #
# context_chain — no-op marker node (its enabled flag is read by `analyze`)
# --------------------------------------------------------------------------- #
async def context_chain_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    return
    yield  # pragma: no cover — makes this an async generator


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def _apply_dedup_bucket_key(raw_results: list, key_by: str) -> list:
    """Brooks R6 fix: ``pipeline.deduplicate_clips`` buckets by ``clip["role"]``,
    but the schema emits ``actor_player_id``. Rather than changing the shared
    function's signature, inject the desired bucketing value into a ``role``
    copy field before calling it — scoped entirely to this QA dedup call path.
    """
    prepared = copy.deepcopy(raw_results)
    for chunk in prepared:
        for clip in chunk.get("analysis", {}).get("clips", []):
            if key_by == "actor_player_id":
                clip["role"] = clip.get("actor_player_id") or clip.get("role", "")
            else:
                clip["role"] = clip.get("role", "")
    return prepared


def _apply_simplified_tags_dedup_keys(raw_results: list) -> list:
    """Adapts ``simplified-tags-v1`` clips onto ``pipeline.deduplicate_clips``'s
    expected keys WITHOUT changing that shared function: it buckets by
    ``clip.get("role", "")`` (per-actor merge) and merges same-category
    overlaps via ``clip.get("action", clip.get("category", ""))``.
    Simplified-tags clips carry ``actor`` (not ``role``) and ``action_class``
    (not ``action``) — inject deep-copied ``role``/``action`` aliases before
    calling the shared function, same call-path-scoped injection discipline as
    the R6 ``_apply_dedup_bucket_key`` fix above (never touching the shared
    function's signature or the clips-v1 call path).

    There is no ``actor_player_id``/``role`` choice to make for this format (it
    has one identity field, ``actor``), so — unlike ``_apply_dedup_bucket_key``
    — this always buckets by ``actor``; ``DedupConfig.key_by`` is not
    consulted here.
    """
    prepared = copy.deepcopy(raw_results)
    for chunk in prepared:
        for clip in chunk.get("analysis", {}).get("clips", []):
            clip["role"] = clip.get("actor", "")
            clip["action"] = clip.get("action_class", "")
    return prepared


async def dedup_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    cfg = DedupConfig.model_validate(config)
    if ctx.final_format == "clips-v1":
        prepared = _apply_dedup_bucket_key(ctx.raw_results, cfg.key_by)
        key_note = f"key_by={cfg.key_by}"
        ctx.final_clips = deduplicate_clips(prepared)
        ctx.dedup_applied = True
        yield {"type": "progress", "stage_id": "dedup", "message": f"{len(ctx.final_clips)} clip(s) after dedup ({key_note})"}
        return
    if ctx.final_format == "simplified-tags-v1":
        prepared = _apply_simplified_tags_dedup_keys(ctx.raw_results)
        key_note = "key_by=actor (fixed for simplified-tags-v1)"
        ctx.final_clips = deduplicate_clips(prepared)
        ctx.dedup_applied = True
        yield {"type": "progress", "stage_id": "dedup", "message": f"{len(ctx.final_clips)} clip(s) after dedup ({key_note})"}
        return
    if ctx.final_format == "simplified-tags-time-v1":
        # Brooks HIGH seam: a REAL seconds-based dedup branch — this pipeline
        # never sets ctx.native_fps (no frame sampling occurs), so frame-keyed
        # `deduplicate_clips` cannot be reused without fabricating a frame
        # rate. `time_dedup.deduplicate_clips_by_time` merges on time-IoU +
        # exact (action_class, actor) identity, keep-higher-confidence.
        ctx.final_clips = time_dedup.deduplicate_clips_by_time(ctx.raw_results)
        ctx.dedup_applied = True
        yield {
            "type": "progress", "stage_id": "dedup",
            "message": f"{len(ctx.final_clips)} clip(s) after time-based dedup (key_by=action_class+actor)",
        }
        return
    ctx.dedup_skip_reason = (
        f"upstream return_format={ctx.final_format!r} not supported by dedup "
        "('clips-v1'/'simplified-tags-v1'/'simplified-tags-time-v1' only)"
    )
    yield {"type": "progress", "stage_id": "dedup", "message": f"skipped: {ctx.dedup_skip_reason}"}


NODE_EXECUTORS = {
    "video_window": video_window_node,
    "frame_sample": frame_sample_node,
    "detect_crop": detect_crop_node,
    "window": window_node,
    "analyze": analyze_node,
    "context_chain": context_chain_node,
    "dedup": dedup_node,
    "chunk_segment": chunk_segment_node,
    "chunk_analyze": chunk_analyze_node,
    "highlight_scan": highlight_scan_node,
    "highlight_analyze": highlight_analyze_node,
    "highlight_critique": highlight_critique_node,
    "position_scan": position_scan_node,
    "position_critique": position_critique_node,
    "technique_analyze": technique_analyze_node,
    "timeline_reconcile": timeline_reconcile_node,
}


def _highlight_per_highlight_call_factor(stage_by_type: dict) -> int:
    """Real Gemini-call factor per highlight for a ``highlight_scan``-based
    pipeline's PASS 2+ (build plan v2 item 8 — "budget gate"). The shared
    ``HighlightAnalyzeConfig`` evolution means every highlight now costs
    ``2 (position + technique) + max_validator_iterations (validator)``
    calls, not one; an optional ``highlight_critique`` stage
    (``highlight-scan-critique-analyze`` only) adds ONE more call per
    highlight on top of that. Single source of truth for BOTH
    ``estimate_run_plan``'s pre-flight upper bound AND ``run_pipeline``'s
    real two-phase post-scan gate — never duplicated (INS-107 discipline).
    Floors at 1 (never 0) so a pipeline shape this function doesn't
    recognize can never silently disable the gate (build plan: "do NOT
    under-count").
    """
    factor = 0
    if "highlight_critique" in stage_by_type:
        factor += 1
    if "highlight_analyze" in stage_by_type:
        ha_cfg = HighlightAnalyzeConfig.model_validate(stage_by_type["highlight_analyze"].config)
        factor += 2 + ha_cfg.max_validator_iterations
    if "technique_analyze" in stage_by_type:
        # One grounded technique call + one single-pass technique validator.
        factor += 2
    return max(1, factor)


# --------------------------------------------------------------------------- #
# Budget estimation (pre-flight guard) — no Gemini calls, pure arithmetic.
# --------------------------------------------------------------------------- #
def estimate_run_plan(pipeline: PipelineDef, duration_sec: float) -> dict:
    """Returns ``{"planned_frames", "planned_windows", "planned_gemini_calls"}``.

    An UPPER BOUND: real ``detect_crop`` frame drops (no_detection) can only
    reduce the actual window/analyze-call count below this estimate, never
    exceed it — safe to gate on. Same discipline for ``chunk-segment-tags``:
    the worth_analysis prefilter can only REDUCE the real chunk_analyze call
    count below the estimate below, never exceed it.
    """
    stage_by_type = {s.type: s for s in pipeline.stages}

    if "position_scan" in stage_by_type and "technique_analyze" in stage_by_type:
        ps_cfg = PositionScanConfig.model_validate(stage_by_type["position_scan"].config)
        hs_cfg = HighlightScanConfig.model_validate(stage_by_type["highlight_scan"].config)
        max_transitions = max(1, math.ceil(duration_sec / ps_cfg.min_transition_gap_s))
        max_highlights = max(1, math.ceil(duration_sec / hs_cfg.min_highlight_s))
        per_highlight_factor = _highlight_per_highlight_call_factor(stage_by_type)
        return {
            "planned_frames": 0,
            "planned_windows": max_transitions + max_highlights,
            "planned_gemini_calls": (
                1 + max_transitions + 1 + max_highlights * per_highlight_factor
            ),
        }

    if "chunk_segment" in stage_by_type:
        # Brooks budget-guard seam: a REAL upper bound for this pipeline
        # (1 rough-scan call + at most one chunk_analyze call per
        # theoretically-smallest-possible chunk), NOT the flat "1" a
        # non-frame pipeline falls through to below (that flat estimate is
        # the exact bypass this branch exists to close for chunk-segment-tags).
        cs_cfg = ChunkSegmentConfig.model_validate(stage_by_type["chunk_segment"].config)
        max_chunks = max(1, math.ceil(duration_sec / cs_cfg.min_chunk_s))
        return {
            "planned_frames": 0,
            "planned_windows": max_chunks,
            "planned_gemini_calls": 1 + max_chunks,
        }

    if "highlight_scan" in stage_by_type:
        # Same real-upper-bound discipline as the chunk_segment branch above
        # (1 rough-scan call + at most one highlight-branch call PER
        # theoretically-smallest-possible highlight) — highlight_scan has no
        # worth_analysis prefilter, but PASS 1's own judgment (build plan:
        # "not a full scene segmentation") means the REAL highlight count is
        # typically far below this worst case in practice.
        #
        # v2 (build plan item 8): the per-highlight factor is no longer a
        # flat "1" — the shared HighlightAnalyzeConfig evolution (position +
        # technique + validator-iterations calls) and the optional
        # highlight_critique stage both add real per-highlight Gemini spend.
        # See _highlight_per_highlight_call_factor (single source of truth,
        # shared with run_pipeline's real two-phase gate below).
        hs_cfg = HighlightScanConfig.model_validate(stage_by_type["highlight_scan"].config)
        max_highlights = max(1, math.ceil(duration_sec / hs_cfg.min_highlight_s))
        per_highlight_factor = _highlight_per_highlight_call_factor(stage_by_type)
        return {
            "planned_frames": 0,
            "planned_windows": max_highlights,
            "planned_gemini_calls": 1 + max_highlights * per_highlight_factor,
        }

    analyze_cfg = AnalyzeConfig.model_validate(stage_by_type["analyze"].config)
    multiplier = 4 if analyze_cfg.agent_mode == "multi" else 1

    if "frame_sample" in stage_by_type:
        fs_cfg = FrameSampleConfig.model_validate(stage_by_type["frame_sample"].config)
        planned_frames = max(0, round(duration_sec * fs_cfg.sample_fps))
        detect_calls = planned_frames if stage_by_type["detect_crop"].enabled else 0
        w_cfg = WindowConfig.model_validate(stage_by_type["window"].config)
        planned_windows = frame_source.count_sliding_windows(planned_frames, w_cfg.window_size, w_cfg.stride)
        analyze_calls = planned_windows * multiplier
        return {
            "planned_frames": planned_frames,
            "planned_windows": planned_windows,
            "planned_gemini_calls": detect_calls + analyze_calls,
        }

    return {"planned_frames": 0, "planned_windows": 1, "planned_gemini_calls": multiplier}


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #
async def run_pipeline(
    pipeline: PipelineDef, ctx: RunContext, planned: dict, budget_cap: Optional[int] = None,
) -> AsyncIterator[dict]:
    """``budget_cap`` (optional): the EFFECTIVE cap (``body.budget_cap`` or
    ``DEFAULT_BUDGET_CAP``) computed by the caller (``qa_pipeline.pipeline_run``).

    Consulted by TWO two-phase gates below — ``chunk-segment-tags`` (Task 1,
    "durable budget fix") and ``highlight-scan-analyze`` (same pattern,
    reapplied verbatim: evaluator CHANGES-REQUIRED item 1, 2026-07-18). Both
    pipelines' pre-stream guard in ``qa_pipeline.py`` deliberately does NOT
    hard-block on their pathological worst-case pre-flight estimate
    (``1 + ceil(duration/min_chunk_s)`` / ``1 + ceil(duration/min_highlight_s)``
    — e.g. a real ~5min match already worst-cases to 101 planned highlight
    calls at the default ``min_highlight_s=3.0``, over ``DEFAULT_BUDGET_CAP``
    of 60, despite spending far fewer calls in practice). Instead, the run is
    allowed to START and stream PASS 1's real output (``chunk_map`` /
    ``highlight_map``), and THIS loop gates the REAL PASS-2 call count
    (worthy-chunk count / highlight count) against ``budget_cap`` immediately
    after PASS 1 completes — aborting with an ``error`` event BEFORE the
    sequential PASS-2 deep loop (``chunk_analyze`` / ``highlight_analyze``)
    spends a single Gemini call if the real number is over cap. Every other
    pipeline is untouched: they have neither a ``chunk_segment`` nor a
    ``highlight_scan`` stage, so neither branch ever fires for them, and their
    pre-flight guard in ``qa_pipeline.py`` is unchanged (still hard-blocks
    before streaming starts on the worst-case estimate).
    """
    total_t0 = time.perf_counter()
    ctx.stage_enabled = {s.type: s.enabled for s in pipeline.stages}
    stage_by_type = {s.type: s for s in pipeline.stages}
    # Position Timeline exposes a backend-authoritative per-node ledger. The
    # runner is the only layer that can reliably attribute repeated and nested
    # call telemetry to the DAG stage that owns it.
    cost_accumulators = (
        {stage.id: gemini_cost.new_node_accumulator(stage) for stage in pipeline.stages}
        if pipeline.id == "position-timeline-techniques"
        else None
    )

    yield {
        "type": "run_start", "pipeline_id": pipeline.id,
        "planned_frames": planned["planned_frames"],
        "planned_windows": planned["planned_windows"],
        "planned_gemini_calls": planned["planned_gemini_calls"],
    }

    per_stage_timing: dict[str, float] = {}
    for stage in pipeline.stages:
        if (
            pipeline.id == "position-timeline-techniques"
            and stage.type in {"highlight_critique", "technique_analyze"}
            and not ctx.highlights
            and not ctx.highlight_scan_failed
        ):
            per_stage_timing[stage.id] = 0.0
            yield {
                "type": "stage_start",
                "stage_id": stage.id,
                "stage_type": stage.type,
                "skipped": True,
                "reason": "technique scan found zero highlights",
            }
            yield {
                "type": "stage_complete",
                "stage_id": stage.id,
                "skipped": True,
                "timing": {"stage_ms": 0.0},
                **(
                    {
                        "cost": gemini_cost.node_cost_snapshot(
                            cost_accumulators[stage.id]
                        )
                    }
                    if cost_accumulators is not None
                    else {}
                ),
            }
            continue
        if not stage.enabled:
            per_stage_timing[stage.id] = 0.0
            if stage.type == "dedup":
                ctx.dedup_skip_reason = "stage disabled"
            yield {"type": "stage_start", "stage_id": stage.id, "stage_type": stage.type, "skipped": True}
            yield {
                "type": "stage_complete",
                "stage_id": stage.id,
                "skipped": True,
                "timing": {"stage_ms": 0.0},
                **(
                    {
                        "cost": gemini_cost.node_cost_snapshot(
                            cost_accumulators[stage.id]
                        )
                    }
                    if cost_accumulators is not None
                    else {}
                ),
            }
            continue

        yield {"type": "stage_start", "stage_id": stage.id, "stage_type": stage.type, "skipped": False}
        stage_t0 = time.perf_counter()
        executor = NODE_EXECUTORS[stage.type]
        async for event in executor(ctx, stage.config):
            if cost_accumulators is not None:
                gemini_cost.record_event_usage(cost_accumulators[stage.id], event)
            yield event
        stage_ms = (time.perf_counter() - stage_t0) * 1000
        per_stage_timing[stage.id] = round(stage_ms, 1)
        yield {
            "type": "stage_complete",
            "stage_id": stage.id,
            "skipped": False,
            "timing": {"stage_ms": round(stage_ms, 1)},
            **(
                {
                    "cost": gemini_cost.node_cost_snapshot(
                        cost_accumulators[stage.id]
                    )
                }
                if cost_accumulators is not None
                else {}
            ),
        }

        if ctx.pipeline_error is not None:
            return

        if pipeline.id == "position-timeline-techniques" and stage.type == "highlight_scan" and ctx.highlight_scan_failed:
            return

        if pipeline.id == "position-timeline-techniques" and stage.type == "highlight_critique":
            failed = [h for h in ctx.highlights if h.get("_critique_status") == "failed"]
            if failed:
                yield {
                    "type": "error",
                    "stage_id": "highlight_critique",
                    "message": (
                        f"Technique critique failed for {len(failed)} highlight(s); aborting before "
                        "technique analysis so uncritiqued scan bounds are never substituted."
                    ),
                }
                return

        if stage.type == "position_scan" and budget_cap is not None:
            # One position scan has already run; each proposed boundary costs
            # one critique, and one later technique scan call is guaranteed.
            position_lane_calls = 1 + len(ctx.position_transitions) + 1
            if position_lane_calls > budget_cap:
                yield {
                    "type": "error",
                    "stage_id": "position_critique",
                    "message": (
                        f"Position scan found {len(ctx.position_transitions)} transition(s): "
                        f"1 scan + {len(ctx.position_transitions)} critiques + 1 technique scan "
                        f"= {position_lane_calls} calls, exceeding budget cap {budget_cap}. "
                        "Aborting before position critiques. Narrow the scope or raise the cap."
                    ),
                }
                return

        if stage.type == "chunk_segment" and budget_cap is not None:
            worthy_count = sum(1 for c in ctx.chunks if c.get("worth_analysis"))
            if worthy_count > budget_cap:
                yield {
                    "type": "error", "stage_id": "chunk_analyze",
                    "message": (
                        f"Actual worthy-chunk count ({worthy_count}) from the chunk map exceeds "
                        f"the budget cap ({budget_cap}) — aborting BEFORE the deep per-chunk "
                        "analysis loop (zero Gemini calls spent on chunk_analyze). Raise "
                        "min_chunk_s (fewer, larger chunks), narrow the run scope, or pass a "
                        "higher Budget cap."
                    ),
                }
                logger.warning(
                    "run_pipeline: chunk-segment-tags aborting before deep loop — "
                    "worthy_count=%d > budget_cap=%d", worthy_count, budget_cap,
                )
                return

        if stage.type == "highlight_scan" and budget_cap is not None:
            # Same two-phase gate as chunk_segment above, reapplied verbatim
            # for highlight-scan-analyze (evaluator CHANGES-REQUIRED item 1).
            #
            # v2 (build plan item 8): highlight_scan has no worth_analysis
            # prefilter at all — EVERY highlight PASS 1 returns is planned
            # PASS-2+ spend — but the REAL per-highlight Gemini-call count is
            # no longer 1 (the shared HighlightAnalyzeConfig evolution +
            # optional highlight_critique stage — see
            # _highlight_per_highlight_call_factor, the SAME single source of
            # truth estimate_run_plan's pre-flight estimate uses). Gating raw
            # highlight_count against budget_cap (as before) would UNDER-
            # COUNT the real spend now that each highlight costs `factor`
            # calls, not 1 — the plan explicitly forbids that.
            highlight_count = len(ctx.highlights)
            per_highlight_factor = _highlight_per_highlight_call_factor(stage_by_type)
            if pipeline.id == "position-timeline-techniques":
                real_call_count = (
                    1
                    + len(ctx.position_transitions)
                    + 1
                    + highlight_count * per_highlight_factor
                )
            else:
                real_call_count = highlight_count * per_highlight_factor
            if real_call_count > budget_cap:
                yield {
                    "type": "error", "stage_id": (
                        "technique_analyze" if pipeline.id == "position-timeline-techniques" else "highlight_analyze"
                    ),
                    "message": (
                        f"Actual run call count for {highlight_count} highlight(s) at per-highlight "
                        f"factor {per_highlight_factor} is {real_call_count}, exceeding the budget cap "
                        f"({budget_cap}) — aborting BEFORE the deep per-highlight analysis loop "
                        "(zero Gemini calls spent on highlight_critique/highlight_analyze). Raise "
                        "min_highlight_s (fewer, larger highlights), lower max_validator_iterations, "
                        "narrow the run scope, or pass a higher Budget cap."
                    ),
                }
                logger.warning(
                    "run_pipeline: highlight-scan-analyze/highlight-scan-critique-analyze aborting "
                    "before deep loop — highlight_count=%d x factor=%d = %d > budget_cap=%d",
                    highlight_count, per_highlight_factor, real_call_count, budget_cap,
                )
                return

    total_ms = (time.perf_counter() - total_t0) * 1000
    run_complete_event = {
        "type": "run_complete",
        "clips": ctx.final_clips,
        "format": ctx.final_format,
        "native_fps": ctx.native_fps,
        "dedup_applied": ctx.dedup_applied,
        "dedup_skip_reason": ctx.dedup_skip_reason,
        "degraded_counts": ctx.degraded_counts,
        "per_stage_timing": per_stage_timing,
        "timing": {"total_ms": round(total_ms, 1)},
    }
    if ctx.final_format == "position-technique-timeline-v1":
        node_costs = [
            gemini_cost.node_cost_snapshot(cost_accumulators[stage.id])
            for stage in pipeline.stages
        ]
        run_complete_event.update(
            {
                "position_intervals": copy.deepcopy(ctx.position_intervals),
                "technique_events": copy.deepcopy(ctx.technique_events),
                "statistics": copy.deepcopy(ctx.objective_statistics),
                "node_costs": node_costs,
                "cost_summary": gemini_cost.summarize_node_costs(node_costs),
            }
        )
    yield run_complete_event
