"""Sequential upscale + analysis (setup half)."""
import asyncio
import json
import logging
import os
import threading
import time

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_upscale_started, should_flush_analysis
from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import ProcessingMode, TrackRequest

from pipeline import deduplicate_clips
from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint
from service.worker.helpers import _make_s3
from service.worker.progress import _make_worker_state
from service.worker.stages.upscale_jpeg import JpegWriterPool

logger = logging.getLogger("service.worker")


def _upscale_analysis_setup(
    video_path: str,
    tracking_json_path: str,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    *,
    job_id: str,
    jobs_store: JobsStore,
    loop: asyncio.AbstractEventLoop,
    tracking_s3_key: str,
    progress_cb=None,
) -> dict:
    import cv2
    from PIL import Image
    from utils import get_union_box, get_padded_square_box

    # -- Load tracking data --------------------------------------------------
    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    output_bucket = request.output_bucket or request.bucket
    s3_for_writes = _make_s3(config)
    total_tracking_frames = len(tracking_data.get("frames", []))

    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    jpeg_pool = JpegWriterPool()
    _submit_jpeg_write = jpeg_pool.submit
    _drain_jpeg_writes = jpeg_pool.drain
    # -- Initialize restorer -------------------------------------------------
    # M4: fast mode uses BicubicRestorer (no neural network; negligible latency).
    if request.processing_mode == ProcessingMode.FAST:
        from restorer import BicubicRestorer
        restorer = BicubicRestorer()
        logger.info("Job %s upscale: fast mode — using BicubicRestorer (LANCZOS4)", job_id)
    elif request.method == "diffusion":
        from diffusion_restorer import DiffusionRestorer
        restorer = DiffusionRestorer()
    else:
        from restorer import RealESRGANRestorer
        restorer = RealESRGANRestorer(config.model_path)

    # -- Download player reference images (optional) -------------------------
    # Prefer the human-confirmed athlete_bindings (carry track_id + player_id, ordered by
    # track_id) so Gemini can ground actor_player_id. Fall back to LEGACY player_references
    # (unlabelled, no player_id) only when no bindings are supplied.
    player_ref_images = None
    if request.athlete_bindings:
        from PIL import Image as _PILImage
        import io

        player_ref_images = []
        s3_for_refs = _make_s3(config)
        ref_bucket = request.bucket  # references stored in same bucket
        for binding in sorted(request.athlete_bindings, key=lambda b: b.track_id):
            if not binding.s3_key:
                logger.warning(
                    "Athlete binding for player %s (track %s) has no s3_key; skipping ref image",
                    binding.player_id, binding.track_id,
                )
                continue
            try:
                ref_resp = s3_for_refs.get_object(ref_bucket, binding.s3_key)
                ref_data = ref_resp["Body"].read()
                img = _PILImage.open(io.BytesIO(ref_data))
                player_ref_images.append({
                    "track_id": binding.track_id,
                    "player_id": binding.player_id,
                    "player_name": binding.player_name or "Unknown",
                    "image": img,
                })
                logger.info(
                    "Downloaded player reference for %s (player_id %s, track %s): %s",
                    binding.player_name, binding.player_id, binding.track_id, binding.s3_key,
                )
            except Exception as e:
                logger.warning(
                    "Failed to download player reference %s: %s",
                    binding.s3_key, e,
                )
        if not player_ref_images:
            player_ref_images = None
    elif request.player_references:
        # LEGACY: superseded by athlete_bindings. Unlabelled refs carry no player_id,
        # so actor_player_id cannot be enum-constrained on this path.
        from PIL import Image as _PILImage
        import io

        player_ref_images = []
        s3_for_refs = _make_s3(config)
        ref_bucket = request.bucket  # references stored in same bucket
        for ref in request.player_references:
            try:
                ref_resp = s3_for_refs.get_object(ref_bucket, ref["s3_key"])
                ref_data = ref_resp["Body"].read()
                img = _PILImage.open(io.BytesIO(ref_data))
                player_ref_images.append({
                    "player_name": ref.get("player_name", "Unknown"),
                    "image": img,
                })
                logger.info(
                    "Downloaded player reference for %s: %s",
                    ref.get("player_name"), ref["s3_key"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to download player reference %s: %s",
                    ref.get("s3_key"), e,
                )
        if not player_ref_images:
            player_ref_images = None

    # -- Initialize analyser (optional) --------------------------------------
    analyze_fn = None
    if config.gemini_api_key:
        taxonomy_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "bjj_analysis_taxonomy.md",
        )
        if request.analyzer_mode == "multi":
            from analyzer import BJJMultiAgentAnalyzer, analyze_sequence_sync
            _analyzer = BJJMultiAgentAnalyzer(
                config.gemini_api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=config.gemini_request_timeout_ms,
            )
            analyze_fn = lambda frames, indices, ctx: analyze_sequence_sync(
                _analyzer, frames, indices, ctx,
                player_references=player_ref_images,
            )
        else:
            from analyzer import BJJTechniqueAnalyzer
            _analyzer = BJJTechniqueAnalyzer(
                config.gemini_api_key,
                taxonomy_path=taxonomy_path,
                request_timeout_ms=config.gemini_request_timeout_ms,
            )
            analyze_fn = lambda frames, indices, ctx: _analyzer.analyze_sequence(
                frames, indices, ctx,
                player_references=player_ref_images,
            )

    # -- Sliding-window analysis state ---------------------------------------
    sliding_buffer = []
    analysis_results = []
    current_context = "Start of match."
    # M4 F2: fast mode uses non-overlapping windows (WINDOW_SIZE=STRIDE=20)
    # to halve the window count vs standard overlapping windows (30/15).
    if request.processing_mode == ProcessingMode.FAST:
        WINDOW_SIZE = 20
        STRIDE = 20
    else:
        WINDOW_SIZE = 30
        STRIDE = 15

    # -- Resume from analysis checkpoint if available ------------------------
    resume_start_frame = 0
    if request.analysis_raw_s3_key:
        try:
            s3_for_resume = _make_s3(config)
            raw_bucket = request.output_bucket or request.bucket
            prev_raw = s3_for_resume.download_json(raw_bucket, request.analysis_raw_s3_key)
            if prev_raw and isinstance(prev_raw, list):
                analysis_results = prev_raw
                # Find last window's max frame to resume from
                last_window = prev_raw[-1] if prev_raw else None
                if last_window and "frames" in last_window:
                    resume_start_frame = max(last_window["frames"]) + 1
                # Restore context
                if last_window and "analysis" in last_window:
                    ctx = last_window["analysis"].get("current_context_summary", "")
                    if ctx:
                        current_context = ctx
                logger.info(
                    "Resumed analysis from checkpoint: %d windows, resuming from frame %d",
                    len(analysis_results), resume_start_frame,
                )
        except Exception as e:
            logger.warning("Failed to load analysis checkpoint: %s", e)

    # -- Write analysis_started checkpoint -----------------------------------
    started_ws = _make_worker_state(
        progress_percent=55.0,
        current_frame=resume_start_frame,
        total_frames=total_tracking_frames,
        stage_progress_fraction=(
            resume_start_frame / max(total_tracking_frames, 1)
        ),
    )
    try:
        asyncio.run_coroutine_threadsafe(
            jobs_store.write_checkpoint(
                job_id, PipelineStage.UPSCALE_ANALYZE, False,
                build_upscale_started(
                    tracking_s3_key=tracking_s3_key,
                    analysis_raw_s3_key=request.analysis_raw_s3_key or None,
                    resume_from_frame=resume_start_frame,
                    analysis_window_count=len(analysis_results),
                    analysis_current_context=current_context,
                    worker_state=started_ws,
                ),
            ),
            loop,
        ).result(timeout=10)
    except Exception as e:
        logger.warning("analysis_started checkpoint write failed: %s", e)

    method_prefix = {
        "diffusion": "diff_", "swinir": "swinir_", "hat": "hat_",
    }.get(request.method, "esrgan_")

    _ret = {k: v for k, v in locals().items() if not k.startswith('__')}
    _ret['jpeg_pool'] = jpeg_pool
    return _ret

