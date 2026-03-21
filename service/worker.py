"""
Unified pipeline worker: detect → verify → track → upscale → upload → publish.

Orchestrates the full job lifecycle, bridging the synchronous tracking thread
with the async WebSocket layer via asyncio.run_coroutine_threadsafe().
"""
import asyncio
import base64
import json
import logging
import os
import shutil
from uuid import uuid4

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import JobCancelledError, JobStatus, TrackRequest
from service.s3 import S3Client
from service.sns import SNSPublisher
from service.ws_manager import WSManager

logger = logging.getLogger(__name__)


async def run_job(
    job_id: str,
    request: TrackRequest,
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    ws_manager: WSManager,
):
    """Run the full tracking + upscale pipeline for a single job."""
    work_dir = os.path.join(config.temp_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)
    loop = asyncio.get_event_loop()

    try:
        # ============================================================
        # STAGE 1: DOWNLOAD (0–10%)
        # ============================================================
        await job_store.update_job(
            job_id, status=JobStatus.DOWNLOADING, progress_percent=2.0,
        )
        await ws_manager.send_progress(job_id, 0, 0, "downloading", 2.0)

        s3 = _make_s3(config)
        s3.ensure_bucket(request.bucket)

        video_path = await loop.run_in_executor(
            None,
            s3.download_file,
            request.bucket,
            request.key,
            os.path.join(work_dir, "video.mp4"),
        )

        await job_store.update_job(job_id, progress_percent=10.0)
        await ws_manager.send_progress(job_id, 0, 0, "downloading", 10.0)

        if _is_cancelled(job_id, job_store):
            return

        # Parse time range BEFORE detection so start_frame is available
        start_frame, end_frame = _parse_time_range(
            video_path, request.start_time, request.end_time,
        )

        # ============================================================
        # STAGE 2: DETECT + HUMAN VERIFY (10–15%)
        # ============================================================
        if request.box_a is not None and request.box_b is not None:
            # Boxes provided in request — skip detection entirely
            box_a = request.box_a
            box_b = request.box_b
            logger.info(
                "Job %s using provided boxes (skipping detection): "
                "box_a=%s box_b=%s",
                job_id, box_a, box_b,
            )
            await job_store.update_job(
                job_id, status=JobStatus.DETECTING, progress_percent=15.0,
            )
            await ws_manager.send_progress(job_id, 0, 0, "detecting", 15.0)
        else:
            await job_store.update_job(
                job_id, status=JobStatus.DETECTING, progress_percent=10.0,
            )
            await ws_manager.send_progress(job_id, 0, 0, "detecting", 10.0)

            # Import heavy ML deps lazily
            from service.tracking_runner import run_detect, capture_frame_jpeg

            # Detect at the start frame (first frame of the requested range)
            frame_idx = start_frame
            candidates = await loop.run_in_executor(
                None,
                lambda: run_detect(
                    video_path,
                    frame_idx=frame_idx,
                    threshold=request.detection_threshold,
                    yolo_model=request.yolo_model,
                ),
            )

            frame_jpeg = await loop.run_in_executor(
                None, lambda: capture_frame_jpeg(video_path, frame_idx),
            )
            if frame_jpeg is None:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video")

            # VLLM pre-selection if >2 candidates
            suggested_boxes = None
            if len(candidates) > 2:
                try:
                    from service.vllm_selector import suggest_athletes
                    suggested_boxes = await suggest_athletes(
                        frame_jpeg, candidates, config,
                    )
                except Exception as e:
                    logger.warning("VLLM pre-selection failed: %s", e)

            # ALWAYS send to human for verification
            frame_b64 = base64.b64encode(frame_jpeg).decode()
            await ws_manager.send_detection_needed(
                job_id,
                frame_idx=frame_idx,
                frame_b64=frame_b64,
                reason="initial",
                candidates=candidates,
                suggested_boxes=suggested_boxes,
            )
            await job_store.update_job(
                job_id,
                status=JobStatus.WAITING_FOR_DETECTION,
                progress_percent=12.0,
            )

            detection = await ws_manager.wait_for_detection(
                job_id, timeout=config.detection_timeout,
            )
            if detection is None:
                raise RuntimeError("Detection timed out or was cancelled by client")

            box_a = detection["box_a"]
            box_b = detection["box_b"]

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 3: TRACKING (15–55%)
        # ============================================================
        await job_store.update_job(
            job_id, status=JobStatus.TRACKING, progress_percent=15.0,
        )
        await ws_manager.send_progress(job_id, 0, 0, "tracking", 15.0)

        from service.tracking_runner import run_tracking_job

        # Build sync progress callback -> async WS
        # Tracking spans 15%–55% (40% range)
        def tracking_progress_cb(frames_done: int, total: int):
            pct = 15.0 + (frames_done / max(total, 1)) * 40.0
            asyncio.run_coroutine_threadsafe(
                _update_tracking_progress(
                    job_id, frames_done, total, pct, job_store, ws_manager,
                ),
                loop,
            )

        # Build sync detection callback -> async WS + VLLM
        detection_cb = _make_detection_cb(
            job_id, loop, ws_manager, config,
        )

        tracking_output_dir = os.path.join(work_dir, "tracking")
        eff_step_size = request.step_size or config.tracking_step_size
        eff_max_history = request.max_history or config.tracking_max_history
        eff_max_missing_frames = (
            request.max_missing_frames
            if request.max_missing_frames is not None
            else config.tracking_max_missing_frames
        )
        logger.info(
            "Job %s tracking config: step_size=%s max_history=%s "
            "max_missing_frames=%s",
            job_id,
            eff_step_size,
            eff_max_history,
            eff_max_missing_frames,
        )
        tracking_json_path = await loop.run_in_executor(
            None,
            lambda: run_tracking_job(
                video_path,
                box_a,
                box_b,
                tracking_output_dir,
                sam2_model_id=request.sam2_model,
                yolo_model=request.yolo_model,
                detection_threshold=request.detection_threshold,
                start_frame=start_frame,
                end_frame=end_frame,
                step_size=eff_step_size,
                max_history=eff_max_history,
                max_missing_frames=eff_max_missing_frames,
                progress_cb=tracking_progress_cb,
                detection_cb=detection_cb,
                should_stop=lambda: job_store.is_cancelled(job_id),
            ),
        )

        # ==============================================================
        # TRACKING-ONLY SHORT-CIRCUIT (skip_upscale=True)
        # Upload tracking results to S3 then mark completed.
        # ==============================================================
        if request.skip_upscale:
            await job_store.update_job(
                job_id, status=JobStatus.UPLOADING, progress_percent=80.0,
            )
            await ws_manager.send_progress(job_id, 0, 0, "uploading", 80.0)

            output_bucket = request.output_bucket or request.bucket
            base_key = os.path.splitext(request.key)[0]
            tracking_result_key = f"{base_key}_tracked.json"
            tracked_video_key = f"{base_key}_tracked.mp4"

            s3.ensure_bucket(output_bucket)

            # Upload tracking JSON
            with open(tracking_json_path) as f:
                tracking_data = json.load(f)
            await loop.run_in_executor(
                None, s3.upload_json, tracking_data,
                output_bucket, tracking_result_key,
            )

            # Upload tracked video (if it exists)
            tracked_video = os.path.join(
                tracking_output_dir, "tracked_output.mp4",
            )
            uploaded_video_key = None
            if os.path.isfile(tracked_video):
                await loop.run_in_executor(
                    None, s3.upload_file, tracked_video,
                    output_bucket, tracked_video_key, "video/mp4",
                )
                uploaded_video_key = tracked_video_key

            await job_store.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_percent=100.0,
                result_bucket=output_bucket,
                result_key=tracking_result_key,
            )
            await ws_manager.send_completed(
                job_id,
                result_bucket=output_bucket,
                result_key=tracking_result_key,
                tracking_key=tracking_result_key,
                annotated_video_key=uploaded_video_key,
            )
            logger.info(
                "Job %s completed (tracking only → s3://%s/%s)",
                job_id, output_bucket, tracking_result_key,
            )
            return

        # ============================================================
        # STAGE 4: UPSCALE + ANALYSIS — second pass (55–80%)
        # ============================================================
        await job_store.update_job(
            job_id, status=JobStatus.UPSCALING, progress_percent=55.0,
        )
        await ws_manager.send_progress(job_id, 0, 0, "upscaling", 55.0)

        async def _update_upscale_progress(pct: float):
            await job_store.update_job(job_id, progress_percent=pct)
            await ws_manager.send_progress(job_id, 0, 0, "upscaling", pct)

        def _upscale_progress(pct_within_stage: float):
            overall = 55.0 + pct_within_stage * 25.0  # 55%–80%
            asyncio.run_coroutine_threadsafe(
                _update_upscale_progress(overall),
                loop,
            )

        analysis_result, fps = await loop.run_in_executor(
            None,
            lambda: _run_upscale_analysis(
                video_path, tracking_json_path, config, request, work_dir,
                progress_cb=_upscale_progress,
            ),
        )

        output_bucket = request.output_bucket or request.bucket
        base_key = os.path.splitext(request.key)[0]
        tracking_result_key = f"{base_key}_tracked.json"
        analysis_result_key = f"{base_key}_analysis.json"
        annotated_video_key = f"{base_key}_annotated.mp4"

        # ============================================================
        # STAGE 4.5: ANNOTATE VIDEO (80–85%)
        # ============================================================
        annotated_video_path = None
        tracked_video_path = os.path.join(tracking_output_dir, "tracked_output.mp4")

        if analysis_result and os.path.isfile(tracked_video_path):
            await job_store.update_job(job_id, progress_percent=80.0)
            await ws_manager.send_progress(job_id, 0, 0, "annotating", 80.0)

            from service.video_annotator import annotate_video
            annotated_path = os.path.join(work_dir, "annotated_output.mp4")
            try:
                annotated_video_path = await loop.run_in_executor(
                    None,
                    lambda: annotate_video(
                        tracked_video_path, analysis_result,
                        annotated_path, fps, start_frame,
                    ),
                )
            except Exception as e:
                logger.warning("Video annotation failed (non-fatal): %s", e)
        elif os.path.isfile(tracked_video_path):
            # No analysis but tracked video exists — upload the tracked video
            annotated_video_path = tracked_video_path

        await job_store.update_job(job_id, progress_percent=85.0)
        await ws_manager.send_progress(job_id, 0, 0, "annotating", 85.0)

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 5: UPLOAD (85–90%)
        # ============================================================
        await job_store.update_job(
            job_id, status=JobStatus.UPLOADING, progress_percent=85.0,
        )
        await ws_manager.send_progress(job_id, 0, 0, "uploading", 85.0)

        s3.ensure_bucket(output_bucket)

        # Upload tracking JSON
        with open(tracking_json_path) as f:
            tracking_data = json.load(f)
        await loop.run_in_executor(
            None,
            s3.upload_json,
            tracking_data,
            output_bucket,
            tracking_result_key,
        )

        # Upload analysis JSON (if upscaling was done)
        if analysis_result is not None:
            await loop.run_in_executor(
                None,
                s3.upload_json,
                analysis_result,
                output_bucket,
                analysis_result_key,
            )

        # Upload annotated video
        if annotated_video_path and os.path.isfile(annotated_video_path):
            await loop.run_in_executor(
                None,
                s3.upload_file,
                annotated_video_path,
                output_bucket,
                annotated_video_key,
                "video/mp4",
            )

        await job_store.update_job(job_id, progress_percent=90.0)

        if _is_cancelled(job_id, job_store):
            return

        # ============================================================
        # STAGE 6: PUBLISH SNS (90–95%) — optional
        # ============================================================
        topic_arn = request.sns_topic_arn or config.sns_topic_arn
        event_count = 0
        if topic_arn and analysis_result is not None:
            try:
                await job_store.update_job(
                    job_id, status=JobStatus.PUBLISHING, progress_percent=92.0,
                )
                await ws_manager.send_progress(job_id, 0, 0, "uploading", 92.0)

                video_id = request.video_id or uuid4()
                sns = SNSPublisher(
                    config.aws_region,
                    topic_arn,
                    endpoint_url=config.s3_endpoint_url or None,
                    access_key_id=config.aws_access_key_id or None,
                    secret_access_key=config.aws_secret_access_key or None,
                )
                result_uri = f"s3://{output_bucket}/{analysis_result_key}"
                event_count = sns.publish_events(
                    analysis_result, video_id, fps,
                    job_id=job_id, result_s3_uri=result_uri,
                )
            except Exception as e:
                logger.warning("SNS publish failed (non-fatal): %s", e)
                await ws_manager.send_progress(job_id, 0, 0, "uploading", 95.0)

        # ============================================================
        # COMPLETED
        # ============================================================
        result_key = analysis_result_key if analysis_result else tracking_result_key
        await job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            result_bucket=output_bucket,
            result_key=result_key,
        )
        await ws_manager.send_completed(
            job_id,
            result_bucket=output_bucket,
            result_key=result_key,
            tracking_key=tracking_result_key,
            annotated_video_key=annotated_video_key if annotated_video_path else None,
        )
        logger.info("Job %s completed (%d SNS events published)", job_id, event_count)

    except asyncio.CancelledError:
        logger.info("Job %s cancelled (client disconnected)", job_id)
        await job_store.update_job(job_id, status=JobStatus.CANCELLED)

    except JobCancelledError:
        logger.info("Job %s cancelled (DELETE requested)", job_id)
        await job_store.update_job(job_id, status=JobStatus.CANCELLED)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        await job_store.update_job(
            job_id, status=JobStatus.FAILED, error_message=str(e),
        )
        await ws_manager.send_error(job_id, str(e))

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_s3(config: ServiceConfig) -> S3Client:
    return S3Client(
        region=config.aws_region,
        endpoint_url=config.s3_endpoint_url or None,
        access_key_id=config.aws_access_key_id or None,
        secret_access_key=config.aws_secret_access_key or None,
    )


def _is_cancelled(job_id: str, job_store: InMemoryJobStore) -> bool:
    if job_store.is_cancelled(job_id):
        logger.info("Job %s cancelled", job_id)
        return True
    return False


async def _update_tracking_progress(
    job_id: str,
    frames_done: int,
    total: int,
    pct: float,
    job_store: InMemoryJobStore,
    ws_manager: WSManager,
):
    await job_store.update_job(
        job_id,
        progress_percent=pct,
        current_frame=frames_done,
        total_frames=total,
    )
    await ws_manager.send_progress(job_id, frames_done, total, "tracking", pct)


def _make_detection_cb(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    ws_manager: WSManager,
    config: ServiceConfig,
):
    """
    Create a sync detection_callback for run_tracking's detection_callback parameter.

    This bridges the synchronous tracking thread to the async event loop for
    WebSocket communication and VLLM calls.
    """

    def detection_cb(
        reason: str, frame_jpeg: bytes, **kwargs
    ) -> tuple | None:
        yolo_detections = kwargs.get("yolo_detections", [])
        global_frame_idx = kwargs.get("frame_idx", 0)

        async def _async_flow():
            # Convert YOLO detections to candidate format
            candidates = [
                {
                    "candidate_id": i,
                    "box": d["box"],
                    "confidence": d["confidence"],
                }
                for i, d in enumerate(yolo_detections)
            ]

            # VLLM pre-filter if >2 candidates
            suggested_boxes = None
            if len(candidates) > 2:
                try:
                    from service.vllm_selector import suggest_athletes
                    suggested_boxes = await suggest_athletes(
                        frame_jpeg, candidates, config,
                    )
                except Exception as e:
                    logger.warning("VLLM mid-tracking suggestion failed: %s", e)

            # Send to human via WebSocket
            frame_b64 = base64.b64encode(frame_jpeg).decode()
            await ws_manager.send_detection_needed(
                job_id,
                frame_idx=global_frame_idx,
                frame_b64=frame_b64,
                reason=reason,
                candidates=candidates,
                suggested_boxes=suggested_boxes,
            )

            # Wait for human response
            return await ws_manager.wait_for_detection(
                job_id, timeout=config.detection_timeout,
            )

        future = asyncio.run_coroutine_threadsafe(_async_flow(), loop)
        try:
            result = future.result(timeout=config.detection_timeout + 10)
        except Exception as e:
            logger.warning("Mid-tracking detection callback failed: %s", e)
            return None

        if result is None:
            return None
        return (result["box_a"], result["box_b"])

    return detection_cb


def _run_upscale_analysis(
    video_path: str,
    tracking_json_path: str,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    progress_cb=None,
):
    """
    Second-pass: read tracking.json, re-open video, crop + upscale + analyse.

    Runs AFTER tracking is complete so SAM2/RTMPose are fully unloaded and
    the upscaler can use all available GPU memory.

    Returns (analysis_result, fps):
      - analysis_result: dict with clips or None if no Gemini key
      - fps: video frame rate
    """
    import cv2
    from PIL import Image
    from utils import get_union_box, get_padded_square_box

    # -- Load tracking data --------------------------------------------------
    with open(tracking_json_path) as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # -- Initialize restorer -------------------------------------------------
    if request.method == "diffusion":
        from diffusion_restorer import DiffusionRestorer
        restorer = DiffusionRestorer()
    else:
        from restorer import RealESRGANRestorer
        restorer = RealESRGANRestorer(config.model_path)

    # -- Download player reference images (optional) -------------------------
    player_ref_images = None
    if request.player_references:
        from PIL import Image as _PILImage
        import io

        player_ref_images = []
        s3_for_refs = _make_s3(config)
        ref_bucket = request.bucket  # references stored in same bucket
        for ref in request.player_references:
            try:
                ref_resp = s3_for_refs.client.get_object(Bucket=ref_bucket, Key=ref["s3_key"])
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
            os.path.dirname(os.path.dirname(__file__)),
            "bjj_analysis_taxonomy.md",
        )
        if request.analyzer_mode == "multi":
            from analyzer import BJJMultiAgentAnalyzer, analyze_sequence_sync
            _analyzer = BJJMultiAgentAnalyzer(
                config.gemini_api_key, taxonomy_path=taxonomy_path,
            )
            analyze_fn = lambda frames, indices, ctx: analyze_sequence_sync(
                _analyzer, frames, indices, ctx,
                player_references=player_ref_images,
            )
        else:
            from analyzer import BJJTechniqueAnalyzer
            _analyzer = BJJTechniqueAnalyzer(
                config.gemini_api_key, taxonomy_path=taxonomy_path,
            )
            analyze_fn = lambda frames, indices, ctx: _analyzer.analyze_sequence(
                frames, indices, ctx,
                player_references=player_ref_images,
            )

    # Load root pipeline.py explicitly — test_tracking/pipeline.py shadows it
    # on sys.path after tracking/__init__.py inserts test_tracking/ at position 0.
    import importlib.util as _ilu
    _root_pipeline_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pipeline.py",
    )
    _root_pipeline_spec = _ilu.spec_from_file_location(
        "_root_pipeline", _root_pipeline_path,
    )
    _root_pipeline_mod = _ilu.module_from_spec(_root_pipeline_spec)
    _root_pipeline_spec.loader.exec_module(_root_pipeline_mod)
    _deduplicate_clips = _root_pipeline_mod.deduplicate_clips

    # -- Sliding-window analysis state ---------------------------------------
    sliding_buffer = []
    analysis_results = []
    current_context = "Start of match."
    WINDOW_SIZE = 30
    STRIDE = 15

    method_prefix = {
        "diffusion": "diff_", "swinir": "swinir_", "hat": "hat_",
    }.get(request.method, "esrgan_")

    def _analyze_window(window):
        nonlocal current_context
        batch_frames = [x[1] for x in window]
        batch_indices = [x[0] for x in window]
        chunk_idx = len(analysis_results) + 1
        logger.info(
            "Analyzing window %d (frames %d-%d, %d images)",
            chunk_idx, batch_indices[0], batch_indices[-1], len(batch_frames),
        )
        try:
            result_str = analyze_fn(
                batch_frames, batch_indices, current_context,
            )
            logger.info(
                "Window %d raw response length: %d chars",
                chunk_idx, len(result_str) if result_str else 0,
            )
            logger.debug("Window %d raw response: %.500s", chunk_idx, result_str)
            result_data = json.loads(result_str)

            if "error" in result_data:
                logger.error(
                    "Window %d: Gemini returned error: %s",
                    chunk_idx, result_data["error"],
                )
            elif "clips" not in result_data:
                logger.warning(
                    "Window %d: Gemini response missing 'clips' key. Keys: %s",
                    chunk_idx, list(result_data.keys()),
                )
            else:
                logger.info(
                    "Window %d: Gemini returned %d clips",
                    chunk_idx, len(result_data["clips"]),
                )

            if "current_context_summary" in result_data:
                current_context = result_data["current_context_summary"]
            analysis_results.append({
                "window": chunk_idx,
                "frames": batch_indices,
                "analysis": result_data,
            })
        except Exception as e:
            logger.error(
                "Analysis window %d failed: %s", chunk_idx, e, exc_info=True,
            )
            analysis_results.append({"window": chunk_idx, "raw_error": str(e)})

    # -- Main upscale loop ---------------------------------------------------
    frames = tracking_data.get("frames", [])
    total = len(frames)
    processed = 0

    try:
        for entry in frames:
            frame_idx = entry["frame_idx"]
            athletes = entry.get("athletes", [])

            if frame_idx % request.sampling_rate != 0:
                processed += 1
                continue
            if not athletes:
                processed += 1
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                logger.warning("Could not read frame %d", frame_idx)
                processed += 1
                continue

            h, w = frame_bgr.shape[:2]
            boxes = [a["box"] for a in athletes]
            union_box = get_union_box(boxes)
            square_box = get_padded_square_box(
                union_box, padding=0.2, img_shape=(h, w),
            )
            x1, y1, x2, y2 = square_box
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                processed += 1
                continue

            try:
                if request.method == "diffusion":
                    h_crop, w_crop = crop.shape[:2]
                    if max(h_crop, w_crop) > 768:
                        scale = 768 / max(h_crop, w_crop)
                        crop = cv2.resize(
                            crop,
                            (int(w_crop * scale), int(h_crop * scale)),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    enhanced = restorer.enhance(crop, strength=0.5)
                else:
                    enhanced = restorer.enhance(crop, target_size=1024)
            except Exception as e:
                logger.warning("Upscale failed at frame %d: %s", frame_idx, e)
                processed += 1
                continue

            out_path = os.path.join(
                output_dir, f"{method_prefix}frame_{frame_idx:06d}.jpg",
            )
            cv2.imwrite(out_path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Analysis buffer
            if analyze_fn is not None:
                img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sliding_buffer.append((frame_idx, pil_img))

                if len(sliding_buffer) >= WINDOW_SIZE:
                    _analyze_window(sliding_buffer[:WINDOW_SIZE])
                    del sliding_buffer[:STRIDE]

                    with open(
                        os.path.join(output_dir, "analysis_raw.json"), "w",
                    ) as f:
                        json.dump(analysis_results, f, indent=2)

            processed += 1
            if progress_cb and total > 0:
                progress_cb(processed / total)
    finally:
        cap.release()

    # -- Finalize analysis ---------------------------------------------------
    if analyze_fn is None:
        logger.info("Upscale pass complete (no Gemini key — skipping analysis)")
        return None, fps

    logger.info(
        "Finalize: %d windows analysed, %d frames in remaining buffer",
        len(analysis_results), len(sliding_buffer),
    )
    if sliding_buffer:
        _analyze_window(sliding_buffer)

    ok_windows = sum(
        1 for r in analysis_results
        if "analysis" in r and "clips" in r["analysis"]
    )
    err_windows = sum(1 for r in analysis_results if "raw_error" in r)
    api_err_windows = sum(
        1 for r in analysis_results
        if "analysis" in r and "error" in r.get("analysis", {})
    )
    logger.info(
        "Finalize summary: %d total windows, %d with clips, "
        "%d with raw_error, %d with Gemini API error",
        len(analysis_results), ok_windows, err_windows, api_err_windows,
    )

    final_clips = _deduplicate_clips(analysis_results)
    mode = "Multi-Agent" if request.analyzer_mode == "multi" else "Single-Agent"
    result = {
        "match_summary": f"Analysis generated via {mode}",
        "clips": final_clips,
        "fps": fps,
    }
    with open(os.path.join(output_dir, "analysis_final.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, fps


def _parse_time_range(
    video_path: str,
    start_time: str | None,
    end_time: str | None,
) -> tuple[int, int | None]:
    """Convert MM:SS or HH:MM:SS time strings to frame indices."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    def _to_seconds(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)

    start_frame = 0
    if start_time:
        start_frame = int(_to_seconds(start_time) * fps)

    end_frame = None
    if end_time:
        end_frame = int(_to_seconds(end_time) * fps)

    return start_frame, end_frame
