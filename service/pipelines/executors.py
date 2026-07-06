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
from pipeline import deduplicate_clips
from service.pipelines import chunk_segment, frame_source, simplified_tags, time_dedup
from service.pipelines.models import (
    AnalyzeConfig,
    ChunkAnalyzeConfig,
    ChunkSegmentConfig,
    DedupConfig,
    DetectCropConfig,
    FrameSampleConfig,
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
    thinking_off = types.ThinkingConfig(thinking_budget=0)

    for i, crop in enumerate(ctx.crops):
        img_w, img_h = crop.image.size
        try:
            raw_objects, _model_used, model_ms = await _run_detect(
                client, crop.image, model=cfg.model, thinking_config=thinking_off,
                prompt=frame_source.MIMIC_DETECT_PROMPT,
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
            system_instruction=cfg.system_instruction,
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
        response = await client.aio.models.generate_content(
            model=model, contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
            config=gen_config,
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
        response = await client.aio.models.generate_content(
            model=cfg.model,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_text), video_part])],
            config=gen_config,
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

    worthy_count = sum(1 for c in ctx.chunks if c["worth_analysis"])
    yield {
        "type": "chunk_map", "chunks": ctx.chunks,
        "timing": {"model_ms": round(model_ms, 1)},
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
        }
        ctx.final_format = "simplified-tags-time-v1"
        ctx.final_clips = []
        return

    thinking_cfg = thinking_config_for(cfg.model, cfg.thinking)
    analyzer = simplified_tags.SimplifiedTagsTimeAnalyzer(
        ctx.gemini_client(), model_id=cfg.model, thinking_config=thinking_cfg,
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
        # net) — never before ctx.start_sec (the requested scope's own bound).
        effective_start = max(ctx.start_sec, chunk["start_s"] - cfg.overlap_s)
        chunk_end = chunk["end_s"]

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
            yield {"type": "error", "stage_id": "chunk_analyze", "message": f"chunk {chunk['index']}: {e}"}
            continue
        model_ms = (time.perf_counter() - t0) * 1000

        try:
            result_data = json.loads(result_str)
        except (ValueError, TypeError) as e:
            yield {"type": "error", "stage_id": "chunk_analyze", "message": f"chunk {chunk['index']}: non-JSON response: {e}"}
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
}


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

    Only consulted by the ``chunk-segment-tags`` two-phase gate below (Task 1,
    "durable budget fix"): the pre-stream guard in ``qa_pipeline.py``
    deliberately does NOT hard-block this pipeline on its pathological
    worst-case estimate (``1 + ceil(duration/min_chunk_s)`` — e.g. a real
    ~10min match plans 128 worst-case but spends ~15-25 calls in practice, once
    the PASS-1 ``worth_analysis`` prefilter runs). Instead, the run is allowed
    to START and stream the ``chunk_map`` (PASS 1's real output), and THIS loop
    gates the REAL worthy-chunk count (== the actual planned PASS-2 deep-call
    count) against ``budget_cap`` immediately after the ``chunk_segment``
    stage completes — aborting with an ``error`` event BEFORE the sequential
    ``chunk_analyze`` deep loop spends a single Gemini call if the real number
    is over cap. Every other pipeline is untouched: they have no
    ``chunk_segment`` stage, so this branch never fires for them, and their
    pre-flight guard in ``qa_pipeline.py`` is unchanged (still hard-blocks
    before streaming starts on the worst-case estimate).
    """
    total_t0 = time.perf_counter()
    ctx.stage_enabled = {s.type: s.enabled for s in pipeline.stages}

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
