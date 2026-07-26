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
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types
from PIL import Image

from analyzer import BJJMultiAgentAnalyzer, BJJTechniqueAnalyzer
from shared_lib.models.simplified_taxonomy import AXIS2_ACTOR_SENTINELS
from pipeline import deduplicate_clips
from service.pipelines import (
    chunk_segment,
    frame_source,
    gemini_retry,
    highlight_axes,
    highlight_critique,
    highlight_scan,
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
    WindowConfig,
)
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

    current_context: str = "Start of match."
    raw_results: list = field(default_factory=list)  # [{"window":, "frames":, "analysis":}]
    degraded_counts: dict = field(default_factory=dict)

    final_clips: list = field(default_factory=list)
    final_format: str = "clips-v1"
    dedup_applied: bool = False
    dedup_skip_reason: Optional[str] = None

    stage_enabled: dict = field(default_factory=dict)  # node type -> enabled

    # S12 Phase 1b production wiring design §2.1: populated ONLY by the
    # production orchestrator (service/worker/highlight_ingest.py) when
    # ``youtube_url`` actually holds a Gemini Files API URI (``files/xxxx``)
    # instead of a YouTube URL — Files API's ``file_data.mime_type`` is
    # documented "Required" for non-YouTube sources. ``None`` for every
    # QA-playground caller (byte-identical prior behavior — YouTube URLs
    # need no mime_type).
    video_mime_type: Optional[str] = None

    # S12 Phase 1b production wiring design §4.2: populated ONCE during the
    # HIGHLIGHT_INGEST stage from ``TrackRequest.athlete_bindings[].s3_key``
    # — each entry ``{"player_id", "player_name", "image_bytes", "mime_type"}``.
    # Empty list for every QA-playground caller (the actor axis then degrades
    # to a sentinel-only enum — see highlight_axes.build_actor_schema).
    player_references: list = field(default_factory=list)

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
    thinking_off = types.ThinkingConfig(thinking_budget=0)

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
            temperature=0.2, thinking_config=thinking_cfg, system_instruction=system_instruction,
        )
    else:
        prompt_text = (
            "Detect every distinct BJJ technique/action event in this video clip. "
            "Report event start/end timestamps as MM:SS in the ORIGINAL video's absolute "
            "timeline (not relative to this clipped window). Return ONLY the JSON array "
            "matching the response schema — no prose."
        )
        gen_config = types.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json",
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
    thinking_off = types.ThinkingConfig(thinking_budget=0)

    video_part = types.Part(
        file_data=types.FileData(file_uri=ctx.youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{ctx.start_sec}s", end_offset=f"{ctx.end_sec}s", fps=cfg.rough_fps,
        ),
    )
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
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
    # None -> thinking_config_for(..., None) == ThinkingConfig(thinking_budget=0)
    # and simplified_tags.media_resolution_enum(None) == MEDIA_RESOLUTION_LOW,
    # i.e. BYTE-IDENTICAL to this node's prior hardcoded values — now
    # overridable per the build plan's per-node thinking+quality decision.
    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)

    video_part = types.Part(
        file_data=types.FileData(file_uri=ctx.youtube_url, mime_type=ctx.video_mime_type),
        video_metadata=types.VideoMetadata(
            start_offset=f"{ctx.start_sec}s", end_offset=f"{ctx.end_sec}s", fps=cfg.rough_fps,
        ),
    )
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
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
        yield {"type": "error", "stage_id": "highlight_scan", "message": str(e)}
        ctx.highlights = []
        return
    model_ms = (time.perf_counter() - t0) * 1000

    raw_text = response.text or ""
    try:
        data = json.loads(raw_text)
        raw_highlights = data.get("highlights", []) if isinstance(data, dict) else []
    except (ValueError, TypeError) as e:
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
# highlight_critique — NEW v2 stage (highlight-scan-critique-analyze ONLY),
# sitting between highlight_scan and highlight_analyze. Per scanned
# highlight, sends a BACKWARD-padded native-video window and asks Gemini to
# confirm the scan's description and locate the highlight's TRUE (possibly
# earlier) start + a corrected end. Mutates ctx.highlights IN PLACE, adding
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
        system_instruction=highlight_critique.DEFAULT_CRITIQUE_SYSTEM_PROMPT,
        retry_config=ctx.retry_config,
    )
    critique_schema = highlight_critique.build_critique_schema()

    for highlight in ctx.highlights:
        # BACKWARD-padded window only (Gracie: search for the earlier true
        # setup cue) — never extends past the highlight's own scanned end.
        window_start = round(max(ctx.start_sec, highlight["start_s"] - cfg.critique_backpad_s), 3)
        window_end = round(highlight["end_s"], 3)

        yield {
            "type": "highlight_critique_start", "highlight_index": highlight["index"],
            "scope": [window_start, window_end],
        }

        # Reuse of the offset-quantization + inverted-window guard (INS-107 /
        # the existing highlight_analyze_node fix) — never send Gemini an
        # inverted or zero-length native-video offset.
        if window_start >= window_end:
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
                    scan_start_sec=highlight["start_s"], scan_end_sec=highlight["end_s"],
                    description=highlight.get("description"),
                ),
                fps=1, media_resolution=cfg.media_resolution,
                # INS-140 fix: Files API requires `file_data.mime_type` for a
                # non-YouTube (production ingest) URI — was silently omitted
                # here (only highlight_scan_node set it correctly before this
                # fix). `None` for the QA playground (byte-identical, a
                # YouTube URL needs none).
                mime_type=ctx.video_mime_type,
            )
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error, no fabrication
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
            yield {
                "type": "error", "stage_id": "highlight_critique",
                "message": f"highlight {highlight['index']}: non-JSON response: {e}",
                "highlight_index": highlight["index"],
            }
            continue

        if isinstance(result_data, dict) and "error" in result_data:
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
        elif isinstance(raw_start, (int, float)) and isinstance(raw_end, (int, float)):
            # Sanity clamp (engineering decision, not spec-mandated): a
            # correction must land within [window_start, highlight's own
            # end_s] and have positive duration, else it is DROPPED (never
            # fabricated/coerced into a plausible-looking value) — the
            # highlight then falls back to its original start_s/end_s in
            # highlight_analyze_node (corrected_* stays absent).
            corrected_start = max(window_start, min(float(raw_start), highlight["end_s"]))
            corrected_end = min(highlight["end_s"], float(raw_end))
            if corrected_end > corrected_start:
                highlight["corrected_start_s"] = corrected_start
                highlight["corrected_end_s"] = corrected_end
                highlight["critique_note"] = result_data.get("note")
                applied = True

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
    analyzer: "simplified_tags.SimplifiedTagsTimeAnalyzer", youtube_url: str,
    window_start: float, window_end: float, *,
    response_schema, prompt_text: str, fps: int, media_resolution: Optional[str],
    mime_type: Optional[str] = None, extra_parts: Optional[list] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Shared call+parse+error-check for the taxonomy/actor axis calls in
    ``highlight_analyze_node`` — DRYs up the exact transport-exception /
    non-JSON / swallowed-``{"error": ...}`` pattern already used by every
    other Gemini-calling node in this module (never a hand-copy of the same
    three checks). Returns ``(data, None)`` on success or
    ``(None, error_message)`` — never raises.

    ``mime_type`` (INS-140 fix, S12 Phase 1b production wiring design §2.1):
    forwarded to ``analyze_chunk`` unconditionally — ``None`` for the QA
    playground (byte-identical prior behavior, no mime_type needed for a
    YouTube URL), ``ctx.video_mime_type`` for the production ingest path
    (Files API URIs, where ``file_data.mime_type`` is documented "Required").

    ``extra_parts`` (additive, S12 Phase 1b — the actor axis's inline
    reference-image ``Part``s): ``None`` for the taxonomy call (byte-identical
    prior behavior)."""
    try:
        result_str = await analyzer.analyze_chunk(
            youtube_url, window_start, window_end, None,
            response_schema=response_schema, prompt_text=prompt_text,
            fps=fps, media_resolution=media_resolution, mime_type=mime_type, extra_parts=extra_parts,
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


async def highlight_analyze_node(ctx: RunContext, config: dict) -> AsyncIterator[dict]:
    """PASS 2 of ``highlight-scan-analyze`` — SHARED, unchanged executor, ALSO
    ``highlight-scan-critique-analyze``'s PASS 3.

    **Single-call cutover (2026-07-26-engine13-rescope-single-call-cutover.md
    AC1-3, OQ1) — replaces the position/technique/validator triple.** LeCun's
    5 cutover gates (``VERDICTS_V2.md``, qualified-model record) ran the
    two-call+validator design against a single flat taxonomy call TWICE, at
    two model tiers, and the two-call design lost by a WIDER margin at the
    stronger model (Gate 1: hit_T-hit_S -1 -> -5, every disagreement favoring
    single-call) while the validator's HIGH ``media_resolution`` default
    actively regressed a correct answer (Gate 2: net_delta=-1). Per highlight
    this now runs exactly ONE taxonomy call
    (``highlight_axes.build_single_call_schema``/``build_single_call_prompt``
    — position + action_class + outcome together, ONE flat verdict, no
    timestamps) plus the UNCHANGED, independent actor/identity call (Gate 1/2
    did not touch this axis). There is no more adversarial reconciliation
    pass and no more ditch authority — every highlight whose taxonomy call
    succeeds reaches a synthesized clip; ``status`` is kept on the emitted
    event, constant at ``"analyzed"``, purely for wire-contract stability
    with ``service/worker/highlight_orchestrator.py``'s existing
    ``status == "ditched"`` branch (which simply never fires anymore).

    A synthesized highlight's ``clips`` list contains exactly ONE clip
    spanning the highlight's own AUTHORITATIVE bounds
    (``{start_s, end_s, position, action_class, outcome}`` plus the actor
    axis's resolved identity fields) — there is nothing left to clip/convert
    from chunk-relative seconds (the taxonomy call reports no timestamps at
    all), so ``_convert_chunk_relative_to_absolute``/
    ``_clip_events_to_highlight_bounds`` (both still used by
    ``chunk_analyze_node`` and directly unit-tested, untouched) are not
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

    (The outer-720s-chunk seam this residual note does NOT cover — a
    coarser-grained boundary than any one highlight — is addressed
    separately, by 45s backward overlap + seam dedup, in
    ``service/worker/highlight_orchestrator.py``.)
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

    # Two independent analyzers — the single taxonomy call and the actor
    # call — each with its OWN thinking/media_resolution/model. Built ONCE,
    # reused across every highlight (same discipline as the single
    # `analyzer` instance the pre-v2 version of this function built once).
    taxonomy_analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=cfg.model,
        thinking_config=thinking_config_for(cfg.model, cfg.thinking),
        system_instruction=highlight_axes.SINGLE_CALL_SYSTEM_PROMPT, retry_config=ctx.retry_config,
    )
    # S12 Phase 1b (design §4.1/§4.3): second, independent actor axis — a
    # non-looped call, same construction pattern as the taxonomy call.
    actor_analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=cfg.actor.model,
        thinking_config=thinking_config_for(cfg.actor.model, cfg.actor.thinking),
        system_instruction=highlight_axes.ACTOR_SYSTEM_PROMPT, retry_config=ctx.retry_config,
    )
    taxonomy_schema = highlight_axes.build_single_call_schema()
    # Dynamic per-job enum (design §4.3) — built ONCE per node call from
    # ctx.player_references, which is constant across the whole job (every
    # chunk's highlight_analyze_node invocation shares the same job-level
    # RunContext.player_references).
    actor_player_choices = [
        p.get("player_id") for p in ctx.player_references if p.get("player_id")
    ]
    actor_schema = highlight_axes.build_actor_schema(actor_player_choices)
    actor_reference_parts = [
        types.Part(
            inline_data=types.Blob(
                data=p["image_bytes"], mime_type=p.get("mime_type") or "image/jpeg",
            ),
        )
        for p in ctx.player_references
        if p.get("image_bytes")
    ]

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

        model_ms_total = 0.0
        description = highlight.get("description")

        # ONE taxonomy call per highlight (position + action_class + outcome
        # together — see highlight_axes.build_single_call_schema), replacing
        # the deleted position+technique+validator triple.
        t0 = time.perf_counter()
        tax_data, tax_err = await _call_axis_json(
            taxonomy_analyzer, ctx.youtube_url, window_start, window_end,
            response_schema=taxonomy_schema, prompt_text=highlight_axes.build_single_call_prompt(description),
            fps=cfg.fps, media_resolution=cfg.media_resolution,
            # INS-140 fix — see _call_axis_json's mime_type docstring.
            mime_type=ctx.video_mime_type,
        )
        model_ms_total += (time.perf_counter() - t0) * 1000
        if tax_err is not None:
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": f"highlight {highlight['index']}: taxonomy call: {tax_err}",
                "highlight_index": highlight["index"],
            }
            continue

        # ONE flat verdict — never a list to merge (see module docstring).
        position_label = tax_data.get("position") if isinstance(tax_data, dict) else None
        action_class_label = tax_data.get("action_class") if isinstance(tax_data, dict) else None
        outcome_label = tax_data.get("outcome") if isinstance(tax_data, dict) else None
        taxonomy_justification = tax_data.get("justification") if isinstance(tax_data, dict) else None

        # S12 Phase 1b (design §4.1/§4.4): actor axis call — flat, once per
        # highlight (every highlight the taxonomy call succeeded on — there
        # is no more validator/ditch authority to gate this on).
        t0 = time.perf_counter()
        actor_data, actor_err = await _call_axis_json(
            actor_analyzer, ctx.youtube_url, window_start, window_end,
            response_schema=actor_schema,
            prompt_text=highlight_axes.build_actor_prompt(description, ctx.player_references),
            fps=cfg.fps, media_resolution=cfg.actor.media_resolution,
            # INS-140 fix — see _call_axis_json's mime_type docstring.
            mime_type=ctx.video_mime_type,
            extra_parts=actor_reference_parts,
        )
        model_ms_total += (time.perf_counter() - t0) * 1000
        if actor_err is not None:
            # No fallback identity — same treatment as the taxonomy call
            # failing: skip this highlight entirely, zero clip emitted
            # (design §4.4).
            yield {
                "type": "error", "stage_id": "highlight_analyze",
                "message": f"highlight {highlight['index']}: actor call: {actor_err}",
                "highlight_index": highlight["index"],
            }
            continue

        actor_value = actor_data.get("actor") if isinstance(actor_data, dict) else None
        identity_uncertain_raw = (
            actor_data.get("identity_uncertain") if isinstance(actor_data, dict) else None
        )
        actor_justification = actor_data.get("justification") if isinstance(actor_data, dict) else None
        player_lookup = {
            p.get("player_id"): p.get("player_name")
            for p in ctx.player_references if p.get("player_id")
        }

        if actor_value in AXIS2_ACTOR_SENTINELS:
            # A sentinel IS an uncertainty signal by definition (design
            # §4.4) — forced True regardless of the model's own flag.
            resolved_player_id = None
            resolved_player_name = None
            resolved_identity_uncertain = True
            resolved_actor_sentinel = actor_value
        elif actor_value in player_lookup:
            resolved_player_id = actor_value
            resolved_player_name = player_lookup[actor_value]
            resolved_identity_uncertain = (
                bool(identity_uncertain_raw) if identity_uncertain_raw is not None else None
            )
            resolved_actor_sentinel = None
        else:
            # Missing/unrecognized actor value — the schema enum prevents
            # this for a real Gemini structured-output call; this branch
            # exists for a malformed/mock response. No fallback identity
            # (never "assume top player" / "assume last-seen identity") —
            # flows through as None, same permissive "unset stays unset"
            # treatment already given to the taxonomy call's own
            # possibly-absent fields.
            resolved_player_id = None
            resolved_player_name = None
            resolved_identity_uncertain = (
                bool(identity_uncertain_raw) if identity_uncertain_raw is not None else None
            )
            resolved_actor_sentinel = None

        # Notes: taxonomy + actor justification text, concatenated (S12
        # Phase 1b design §5.2) — free text, no schema risk.
        notes = " | ".join(bit for bit in (taxonomy_justification, actor_justification) if bit)

        # ONE synthesized clip spanning the highlight's own AUTHORITATIVE
        # bounds — neither call reports its own timestamps, so there is
        # nothing to clip/convert here; the window IS the event.
        clips = [{
            "start_s": authoritative_start, "end_s": authoritative_end,
            "position": position_label, "action_class": action_class_label, "outcome": outcome_label,
            "player_id": resolved_player_id, "player_name": resolved_player_name,
            "identity_uncertain": resolved_identity_uncertain, "actor_sentinel": resolved_actor_sentinel,
            "notes": notes,
        }]

        ctx.raw_results.append({"window": highlight["index"], "frames": [], "analysis": {"clips": clips}})
        yield {
            "type": "highlight_result", "highlight_index": highlight["index"], "format": "simplified-tags-time-v1",
            "clips": clips,
            # "status" is kept, constant at "analyzed", purely for wire-
            # contract stability — there is no more validator/ditch
            # authority (see this function's docstring). A future consumer
            # relying on this key unconditionally being present is
            # unaffected by the removal.
            "status": "analyzed",
            "timing": {"model_ms": round(model_ms_total, 1)},
            # Per-axis telemetry (additive) — two independent calls now,
            # not four; each axis's own analyzer instance carries its OWN
            # last-call instrumentation (simplified_tags.SimplifiedTagsTimeAnalyzer),
            # surfaced here per-axis for debugging/playground display.
            "axis_calls": {
                "taxonomy": {
                    "prompt_text": getattr(taxonomy_analyzer, "last_prompt_text", None),
                    "raw_response_text": getattr(taxonomy_analyzer, "last_raw_response_text", None),
                    "usage_metadata": getattr(taxonomy_analyzer, "last_usage_metadata", None),
                },
                "actor": {
                    "prompt_text": getattr(actor_analyzer, "last_prompt_text", None),
                    "raw_response_text": getattr(actor_analyzer, "last_raw_response_text", None),
                    "usage_metadata": getattr(actor_analyzer, "last_usage_metadata", None),
                },
            },
        }

    ctx.final_format = "simplified-tags-time-v1"
    ctx.final_clips = [c for r in ctx.raw_results for c in r["analysis"]["clips"]]


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
}


def _highlight_per_highlight_call_factor(stage_by_type: dict) -> int:
    """Real Gemini-call factor per highlight for a ``highlight_scan``-based
    pipeline's PASS 2+ (build plan v2 item 8 — "budget gate"; S12 Phase 1b
    production wiring design §4.1 extends this for the actor axis).

    **2026-07-26 single-call re-scope:** every highlight now costs ``2``
    calls — ONE taxonomy call (position + action_class + outcome) + ONE
    actor call — down from ``3 (position + technique + actor) +
    max_validator_iterations`` (the deleted position/technique/validator
    triple, see ``HighlightAnalyzeConfig``'s docstring); an optional
    ``highlight_critique`` stage (``highlight-scan-critique-analyze`` only)
    adds ONE more call per highlight on top of that. Single source of truth
    for BOTH ``estimate_run_plan``'s pre-flight upper bound AND
    ``run_pipeline``'s real two-phase post-scan gate — never duplicated
    (INS-107 discipline). Floors at 1 (never 0) so a pipeline shape this
    function doesn't recognize can never silently disable the gate (build
    plan: "do NOT under-count").
    """
    factor = 0
    if "highlight_critique" in stage_by_type:
        factor += 1
    if "highlight_analyze" in stage_by_type:
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
        # flat "1" — the shared HighlightAnalyzeConfig's taxonomy + actor
        # calls (2, since the 2026-07-26 single-call re-scope) and the
        # optional highlight_critique stage both add real per-highlight
        # Gemini spend. See _highlight_per_highlight_call_factor (single
        # source of truth, shared with run_pipeline's real two-phase gate
        # below).
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

    yield {
        "type": "run_start", "pipeline_id": pipeline.id,
        "planned_frames": planned["planned_frames"],
        "planned_windows": planned["planned_windows"],
        "planned_gemini_calls": planned["planned_gemini_calls"],
    }

    per_stage_timing: dict[str, float] = {}
    for stage in pipeline.stages:
        if not stage.enabled:
            per_stage_timing[stage.id] = 0.0
            if stage.type == "dedup":
                ctx.dedup_skip_reason = "stage disabled"
            yield {"type": "stage_start", "stage_id": stage.id, "stage_type": stage.type, "skipped": True}
            yield {"type": "stage_complete", "stage_id": stage.id, "skipped": True, "timing": {"stage_ms": 0.0}}
            continue

        yield {"type": "stage_start", "stage_id": stage.id, "stage_type": stage.type, "skipped": False}
        stage_t0 = time.perf_counter()
        executor = NODE_EXECUTORS[stage.type]
        async for event in executor(ctx, stage.config):
            yield event
        stage_ms = (time.perf_counter() - stage_t0) * 1000
        per_stage_timing[stage.id] = round(stage_ms, 1)
        yield {"type": "stage_complete", "stage_id": stage.id, "skipped": False, "timing": {"stage_ms": round(stage_ms, 1)}}

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
            real_call_count = highlight_count * per_highlight_factor
            if real_call_count > budget_cap:
                yield {
                    "type": "error", "stage_id": "highlight_analyze",
                    "message": (
                        f"Actual highlight count ({highlight_count}) x per-highlight call factor "
                        f"({per_highlight_factor}) = {real_call_count} exceeds the budget cap "
                        f"({budget_cap}) — aborting BEFORE the deep per-highlight analysis loop "
                        "(zero Gemini calls spent on highlight_critique/highlight_analyze). Raise "
                        "min_highlight_s (fewer, larger highlights), narrow the run scope, "
                        "or pass a higher Budget cap."
                    ),
                }
                logger.warning(
                    "run_pipeline: highlight-scan-analyze/highlight-scan-critique-analyze aborting "
                    "before deep loop — highlight_count=%d x factor=%d = %d > budget_cap=%d",
                    highlight_count, per_highlight_factor, real_call_count, budget_cap,
                )
                return

    total_ms = (time.perf_counter() - total_t0) * 1000
    yield {
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
