"""Mid-track detection suspend callback factory."""
import asyncio
import logging
import os

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import build_track_mid_loss
from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.s3 import S3Client
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend

from service.worker.gpu import _load_partial_tracking_dict
from service.worker.progress import (
    _make_worker_state,
    _tracking_progress_pct_clip,
)

logger = logging.getLogger("service.worker")

def _make_detection_cb(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    jobs_store: JobsStore,
    s3: S3Client,
    config: ServiceConfig,
    request: TrackRequest,
    work_dir: str,
    *,
    clip_start_frame: int,
    clip_total_frames: int,
    progress_floor: float,
):
    """
    Create a sync detection_callback for run_tracking's detection_callback parameter.

    Instead of blocking on a client connection, this writes a checkpoint to
    Keyspaces and returns None, which signals the tracking loop to stop.
    The job will be resumed later via /resume with corrected bounding boxes.
    """

    def detection_cb(
        reason: str, frame_jpeg: bytes, **kwargs
    ) -> tuple | None:
        yolo_detections = kwargs.get("yolo_detections", [])
        global_frame_idx = kwargs.get("frame_idx", 0)

        async def _async_suspend():
            # Convert YOLO detections to candidate format
            candidates = [
                {
                    "candidate_id": i,
                    "box": d["box"],
                    "confidence": d["confidence"],
                }
                for i, d in enumerate(yolo_detections)
            ]

            # Optional Gemini hint if >2 candidates
            suggested_boxes = None
            if len(candidates) > 2:
                try:
                    from service.vllm_selector import suggest_athletes
                    suggested_boxes = await suggest_athletes(
                        frame_jpeg, candidates, config,
                    )
                except Exception as e:
                    logger.warning("Mid-track athlete suggestion (Gemini) failed: %s", e)

            # Upload detection frame to S3 — put_object takes raw bytes;
            # upload_file would treat the JPEG bytes as a local path.
            frame_s3_key = f"checkpoints/{job_id}/frame_{global_frame_idx}.jpg"
            await loop.run_in_executor(
                None,
                lambda: s3.put_object(
                    request.bucket,
                    frame_s3_key,
                    frame_jpeg,
                    "image/jpeg",
                ),
            )

            # Save partial tracking to S3 if available. Use the same bucket
            # the resume route reads from (output_bucket when set).
            tracking_json_path = os.path.join(work_dir, "tracking", "tracking.json")
            partial_key: str | None = None
            if os.path.isfile(tracking_json_path):
                try:
                    partial_key = f"checkpoints/{job_id}/partial_tracking.json"
                    partial_data = _load_partial_tracking_dict(tracking_json_path)
                    upload_bucket = request.output_bucket or request.bucket
                    await loop.run_in_executor(
                        None, s3.upload_json, partial_data, upload_bucket, partial_key,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to save partial tracking during mid-track suspend: %s", e,
                    )
                    partial_key = None

            mid_clip_done, mid_pct = _tracking_progress_pct_clip(
                global_frame_idx, clip_start_frame, clip_total_frames, progress_floor,
            )
            mid_loss_ws = _make_worker_state(
                progress_percent=mid_pct,
                current_frame=mid_clip_done,
                total_frames=clip_total_frames,
                stage_progress_fraction=(
                    mid_clip_done / max(clip_total_frames, 1)
                ),
            )
            await jobs_store.write_checkpoint(
                job_id, PipelineStage.TRACK, False,
                build_track_mid_loss(
                    frame_idx=global_frame_idx,
                    frame_s3_key=frame_s3_key,
                    frame_bucket=request.bucket,
                    candidates=candidates,
                    suggested_boxes=suggested_boxes,
                    partial_tracking_s3_key=partial_key,
                    resume_from_frame=global_frame_idx,
                    worker_state=mid_loss_ws,
                ),
            )
            await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)

        future = asyncio.run_coroutine_threadsafe(_async_suspend(), loop)
        try:
            future.result(timeout=60)
        except Exception as e:
            logger.warning("Mid-tracking suspend failed: %s", e)
            return None

        raise HumanVerificationSuspend()

    return detection_cb
