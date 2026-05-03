import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    WorkerStateSnapshot,
    build_cancellation_checkpoint,
    build_verified_boxes_checkpoint,
    next_unprocessed_frame,
    select_correction_checkpoint,
)
from service.models import TrackRequest, TrackResponse, JobResponse, ResumeRequest, JobStatus
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.worker import run_job
from service.config import ServiceConfig

_QA_HTML = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "qa_client", "index.html"
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Set during app lifespan via init_routes()
_config: ServiceConfig | None = None
_job_store: InMemoryJobStore | None = None
_jobs_store: JobsStore | None = None
_job_semaphore: asyncio.Semaphore | None = None
_instance_id: str = ""
_active_tasks: dict[str, asyncio.Task] = {}  # job_id -> task


def init_routes(
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    jobs_store: JobsStore,
    instance_id: str = "",
):
    global _config, _job_store, _jobs_store, _job_semaphore, _instance_id
    _config = config
    _job_store = job_store
    _jobs_store = jobs_store
    _job_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
    _instance_id = instance_id


async def _run_with_semaphore(job_id: str, request: TrackRequest):
    _jobs_store.register_owned_job(job_id)
    try:
        async with _job_semaphore:
            await run_job(job_id, request, _config, _job_store, _jobs_store)
    finally:
        _jobs_store.unregister_owned_job(job_id)


async def _cleanup_orphaned_tasks():
    """Remove completed tasks from the active-tasks dict."""
    for jid, task in list(_active_tasks.items()):
        if task.done():
            _active_tasks.pop(jid, None)


def _require_write(ok: bool, operation: str) -> None:
    if not ok:
        raise HTTPException(500, f"Failed to persist {operation}")


def _schedule_job(job_id: str, request: TrackRequest) -> None:
    task = asyncio.create_task(_run_with_semaphore(job_id, request))
    _active_tasks[job_id] = task


# ---------------------------------------------------------------------------
# QA client
# ---------------------------------------------------------------------------

@router.get("/", include_in_schema=False)
async def qa_client():
    return FileResponse(_QA_HTML, media_type="text/html")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

# TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.
# The track request should ONLY be used for starting a new job. NOT for resuming a job.
@router.post("/track", response_model=TrackResponse)
async def create_track_job(request: TrackRequest):
    # Proactively clean up completed tasks before checking capacity
    await _cleanup_orphaned_tasks()

    if _job_semaphore.locked():
        raise HTTPException(429, "Server is at capacity. Try again later.")

    job = await _job_store.create_job(request)
    job_id = job.job_id

    # Persist to Keyspaces (user_id="" for engine-direct jobs; the analysis
    # service populates it when submitting via POST /tracking/analyze)

    # TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.
    _require_write(
        await _jobs_store.create_lifecycle(
            job_id,
            request.video_id,
            request.user_id or "",
            "",
            owner_instance_id=_instance_id,
        ),
        "job lifecycle",
    )
    _require_write(
        await _jobs_store.save_request(job_id, request.model_dump_json()),
        "job request",
    )
    if request.video_id:
        _require_write(
            await _jobs_store.set_latest(str(request.video_id), job_id, JobState.PENDING),
            "latest job",
        )

    # Start the job immediately (no WS handshake needed)
    _schedule_job(job_id, request)

    return TrackResponse(job_id=job_id, status="pending")


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    # Try Keyspaces first
    lifecycle = await _jobs_store.get_lifecycle(job_id)
    if lifecycle:
        return JobResponse(
            job_id=lifecycle["job_id"],
            status=lifecycle["job_state"].lower(),
            progress_percent=lifecycle.get("progress_percent", 0.0),
            current_frame=lifecycle.get("current_frame"),
            total_frames=lifecycle.get("total_frames"),
            error_message=lifecycle.get("error_message"),
            created_at=str(lifecycle.get("started_at", "")),
            updated_at=str(lifecycle.get("updated_at", "")),
        )
    # Fall back to in-memory
    job = await _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    lifecycle = await _jobs_store.get_lifecycle(job_id)
    if lifecycle and lifecycle.get("replacement_job_id"):
        raise HTTPException(
            409,
            f"Job has replacement {lifecycle['replacement_job_id']}; cancel latest job instead",
        )

    job = await _job_store.get_job(job_id)
    if job is None and lifecycle is None:
        raise HTTPException(404, "Job not found")

    if job is not None:
        await _job_store.set_cancelled(job_id)

    frame_idx = lifecycle.get("current_frame", 0) if lifecycle else 0
    total_frames = lifecycle.get("total_frames", 0) if lifecycle else 0
    progress_percent = lifecycle.get("progress_percent", 0.0) if lifecycle else 0.0
    cancel_ws = WorkerStateSnapshot(
        progress_percent=progress_percent,
        current_frame=frame_idx,
        total_frames=total_frames,
        stage_progress_fraction=(
            (frame_idx / total_frames) if total_frames > 0 else 0.0
        ),
    )
    _require_write(
        await _jobs_store.write_checkpoint(
            job_id,
            PipelineStage.TRACK,
            False,
            build_cancellation_checkpoint(
                reason="user_cancelled",
                frame_idx=frame_idx,
                progress_percent=progress_percent,
                worker_state=cancel_ws,
            ),
        ),
        "cancellation checkpoint",
    )
    _require_write(
        await _jobs_store.set_state(job_id, JobState.CANCELLED),
        "job cancellation state",
    )

    # Stop the running job task so tracking (and later stages) exit promptly
    task = _active_tasks.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        logger.info("Cancelled active task for job %s", job_id)
    return {"status": "cancelled", "job_id": job_id}


# ---------------------------------------------------------------------------
# SSE endpoint -- this is used to stream job events to the QA client.
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/events")
async def job_events_sse(job_id: str):
    """Server-Sent Events endpoint that tails the Keyspaces job_lifecycle row."""

    async def event_generator():
        last_state = None
        last_pct = -1.0
        consecutive_failures = 0
        tick = 0
        while True:
            try:
                lifecycle = await _jobs_store.get_lifecycle(job_id)
                if not lifecycle:
                    yield f"event: job_error\ndata: {{\"message\": \"Job not found\"}}\n\n"
                    return

                consecutive_failures = 0  # reset on success
                state = lifecycle["job_state"]
                stage = lifecycle.get("stage", "")
                pct = lifecycle.get("progress_percent", 0.0)

                # TODO: to reduce CPU burden, we should go with sleep 1 second instead of checking without sleep.

                # Only send when something changed
                if state != last_state or abs(pct - last_pct) >= 0.5:
                    last_state = state
                    last_pct = pct

                    if state == JobState.AWAITING_CORRECTION.value:
                        for stage_name in (PipelineStage.DETECT, PipelineStage.TRACK):
                            cp = await _jobs_store.get_checkpoint(job_id, stage_name)
                            if cp and cp.get("checkpoint_data", {}).get("pending_detection"):
                                det = cp["checkpoint_data"]["pending_detection"]
                                yield f"event: detection_needed\ndata: {json.dumps(det)}\n\n"
                                break
                    elif state == JobState.COMPLETED.value:
                        yield f"event: completed\ndata: {{\"job_id\": \"{job_id}\"}}\n\n"
                        return
                    elif state in (JobState.FAILED.value, JobState.CANCELLED.value):
                        msg = lifecycle.get("error_message", state)
                        yield f"event: job_error\ndata: {{\"message\": \"{msg}\"}}\n\n"
                        return
                    elif state == JobState.INTERRUPTED.value:
                        yield f"event: interrupted\ndata: {{\"job_id\": \"{job_id}\"}}\n\n"
                        return
                    else:
                        progress = {
                            "type": "progress",
                            "state": stage,
                            "percent": round(pct, 1),
                            "frame_idx": lifecycle.get("current_frame", 0),
                            "total_frames": lifecycle.get("total_frames", 0),
                        }
                        yield f"event: progress\ndata: {json.dumps(progress)}\n\n"

            except Exception as e:
                consecutive_failures += 1
                logger.warning("SSE error for job %s (failure %d): %s", job_id, consecutive_failures, e)
                if consecutive_failures >= 30:
                    yield f"event: job_error\ndata: {{\"message\": \"Service unavailable\"}}\n\n"
                    return

            await asyncio.sleep(1.0)
            tick += 1
            # Send keepalive comment every 15s to prevent proxy idle timeouts
            if tick % 15 == 0:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Detection response (async human-in-the-loop)
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/detection_response")
async def submit_detection_response(job_id: str, body: ResumeRequest):
    """Submit corrected bounding boxes for an AWAITING_CORRECTION job.

    Creates a new resume job with the provided boxes and returns its ID.
    """
    lifecycle = await _jobs_store.get_lifecycle(job_id)
    if not lifecycle:
        raise HTTPException(404, "Job not found")
    if lifecycle["job_state"] != JobState.AWAITING_CORRECTION.value:
        raise HTTPException(409, f"Job state is {lifecycle['job_state']}, not AWAITING_CORRECTION")
    if lifecycle.get("replacement_job_id"):
        raise HTTPException(
            409,
            f"Job already has replacement {lifecycle['replacement_job_id']}",
        )

    # Read the original request params
    request_json = await _jobs_store.get_request(job_id)
    if not request_json:
        raise HTTPException(404, "Original request not found")

    orig_request = json.loads(request_json)

    checkpoints = await _jobs_store.get_all_checkpoints(job_id)
    correction_stage, correction_cp = select_correction_checkpoint(checkpoints)


    # Build resume request
    resume_params = {
        **orig_request,
        "box_a": body.box_a,
        "box_b": body.box_b,
    }

    # If mid-track loss, include partial tracking resume hints
    if correction_cp.get("partial_tracking_s3_key"):
        resume_params["resume_from_job_id"] = job_id
        resume_params["resume_tracking_s3_key"] = correction_cp["partial_tracking_s3_key"]
        resume_params["resume_from_frame"] = next_unprocessed_frame(correction_cp)

    # Create new job
    new_request = TrackRequest(**resume_params)
    new_job = await _job_store.create_job(new_request)
    new_job_id = new_job.job_id

    video_id = orig_request.get("video_id") or lifecycle.get("video_id", "")
    origin_job_id = lifecycle.get("origin_job_id") or job_id
    _require_write(
        await _jobs_store.create_lifecycle(
            new_job_id,
            str(video_id),
            lifecycle.get("user_id", ""),
            origin_job_id=origin_job_id,
            parent_job_id=job_id,
            owner_instance_id=_instance_id,
        ),
        "replacement job lifecycle",
    )
    _require_write(
        await _jobs_store.save_request(new_job_id, new_request.model_dump_json()),
        "replacement job request",
    )
    claimed = await _jobs_store.claim_replacement(job_id, new_job_id)
    if not claimed:
        await _jobs_store.set_state(new_job_id, JobState.CANCELLED, sync_latest=False)
        raise HTTPException(409, "Job replacement claim was already taken")

    if video_id:
        _require_write(
            await _jobs_store.set_latest(str(video_id), new_job_id, JobState.PENDING),
            "latest replacement job",
        )

    _require_write(
        await _jobs_store.set_state(job_id, JobState.CANCELLED, sync_latest=False),
        "source job cancellation state",
    )

    # New job starts post-detect-verification at 15% progress; tracking has not yet begun.
    verified_ws = WorkerStateSnapshot(
        progress_percent=15.0,
        current_frame=0,
        total_frames=0,
        stage_progress_fraction=1.0,
    )
    _require_write(
        await _jobs_store.write_checkpoint(
            new_job_id,
            PipelineStage.TRACK,
            False,
            build_verified_boxes_checkpoint(
                body.box_a, body.box_b, correction_stage,
                worker_state=verified_ws,
            ),
        ),
        "verified boxes checkpoint",
    )

    # Start the new job
    _schedule_job(new_job_id, new_request)

    return {"status": "resumed", "job_id": new_job_id, "origin_job_id": job_id}

# This is called by /checkpoints/{job_id}/correct endpoint in analysis service.
@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, body: ResumeRequest):
    """Deliver corrected bounding boxes to a job waiting for detection.

    Delegates to the detection_response endpoint logic: creates a new
    resume job with the corrected boxes.
    """
    return await submit_detection_response(job_id, body)


async def recover_interrupted_job(lifecycle: dict) -> None:
    """Create and schedule a replacement job for an interrupted worker job."""
    job_id = lifecycle["job_id"]
    if lifecycle.get("replacement_job_id"):
        return

    request_json = await _jobs_store.get_request(job_id)
    if not request_json:
        raise RuntimeError(f"Original request not found for interrupted job {job_id}")

    resume_params = json.loads(request_json)
    checkpoints = await _jobs_store.get_all_checkpoints(job_id)
    _, correction_cp = select_correction_checkpoint(checkpoints)
    if correction_cp.get("partial_tracking_s3_key"):
        resume_params["resume_from_job_id"] = job_id
        resume_params["resume_tracking_s3_key"] = correction_cp["partial_tracking_s3_key"]
        resume_params["resume_from_frame"] = next_unprocessed_frame(correction_cp)

    new_request = TrackRequest(**resume_params)
    new_job = await _job_store.create_job(new_request)
    new_job_id = new_job.job_id

    video_id = resume_params.get("video_id") or lifecycle.get("video_id", "")
    origin_job_id = lifecycle.get("origin_job_id") or job_id
    _require_write(
        await _jobs_store.create_lifecycle(
            new_job_id,
            str(video_id),
            lifecycle.get("user_id", ""),
            origin_job_id=origin_job_id,
            parent_job_id=job_id,
            owner_instance_id=_instance_id,
        ),
        "recovery replacement lifecycle",
    )
    _require_write(
        await _jobs_store.save_request(new_job_id, new_request.model_dump_json()),
        "recovery replacement request",
    )
    claimed = await _jobs_store.claim_replacement(
        job_id,
        new_job_id,
        expected_state=JobState.INTERRUPTED,
    )
    if not claimed:
        await _jobs_store.set_state(new_job_id, JobState.CANCELLED, sync_latest=False)
        return

    if video_id:
        _require_write(
            await _jobs_store.set_latest(str(video_id), new_job_id, JobState.PENDING),
            "latest recovery replacement",
        )
    _schedule_job(new_job_id, new_request)


# ---------------------------------------------------------------------------
# Health & debug
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/debug/memory")
async def debug_memory():
    """Return GPU/MPS memory usage for QA verification of model offload."""
    result = {"gpu_available": False, "allocated_mb": 0, "reserved_mb": 0}
    if torch.cuda.is_available():
        result["gpu_available"] = True
        result["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        result["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        result["gpu_available"] = True
        try:
            result["allocated_mb"] = round(torch.mps.current_allocated_memory() / 1024 / 1024, 1)
        except Exception:
            pass
    return result
