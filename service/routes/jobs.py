"""Track job CRUD endpoints."""

import logging

from fastapi import HTTPException

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import WorkerStateSnapshot, build_cancellation_checkpoint
from service.guardrails import admit_new_analysis, check_kill_switch
from service.models import TrackRequest, TrackResponse, JobResponse
from service.routes import state as route_state
import service.routes as routes_pkg

logger = logging.getLogger("service.routes")


async def create_track_job(request: TrackRequest):
    # G7 kill switch: stop all new GPU/Gemini spend without killing the process.
    check_kill_switch(route_state._config)

    # Proactively clean up completed tasks before checking capacity so the
    # in-flight count reflects only running + queued jobs.
    await routes_pkg._cleanup_orphaned_tasks()

    # G4 (bounded queue admission) + G7 (daily cap). _active_tasks holds every
    # scheduled job — the one holding the GPU semaphore plus any waiting on it.
    # When in_flight < capacity the new job is admitted and queues on the
    # semaphore (serialized GPU access); beyond capacity we reject with 429.
    # admit_new_analysis increments the per-UTC-day counter on success.
    admit_new_analysis(route_state._config, in_flight=len(route_state._active_tasks))

    job = await route_state._job_store.create_job(request)
    job_id = job.job_id

    # Persist to Keyspaces (user_id="" for engine-direct jobs; the analysis
    # service populates it when submitting via POST /tracking/analyze)

    # TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.
    route_state._require_write(
        await route_state._jobs_store.create_lifecycle(
            job_id,
            request.video_id,
            request.user_id or "",
            "",
            owner_instance_id=route_state._instance_id,
        ),
        "job lifecycle",
    )
    route_state._require_write(
        await route_state._jobs_store.save_request(job_id, request.model_dump_json()),
        "job request",
    )
    if request.video_id:
        route_state._require_write(
            await route_state._jobs_store.set_latest(
                str(request.video_id), job_id, JobState.PENDING,
            ),
            "latest job",
        )

    # Start the job immediately (no WS handshake needed)
    routes_pkg._schedule_job(job_id, request)

    return TrackResponse(job_id=job_id, status="pending")


async def get_job(job_id: str):
    # Try Keyspaces first
    lifecycle = await route_state._jobs_store.get_lifecycle(job_id)
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
    job = await route_state._job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


async def cancel_job(job_id: str):
    lifecycle = await route_state._jobs_store.get_lifecycle(job_id)
    if lifecycle and lifecycle.get("replacement_job_id"):
        raise HTTPException(
            409,
            f"Job has replacement {lifecycle['replacement_job_id']}; cancel latest job instead",
        )

    job = await route_state._job_store.get_job(job_id)
    if job is None and lifecycle is None:
        raise HTTPException(404, "Job not found")

    if job is not None:
        await route_state._job_store.set_cancelled(job_id)

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
    route_state._require_write(
        await route_state._jobs_store.write_checkpoint(
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
    route_state._require_write(
        await route_state._jobs_store.set_state(job_id, JobState.CANCELLED),
        "job cancellation state",
    )

    # Stop the running job task so tracking (and later stages) exit promptly
    task = route_state._active_tasks.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        logger.info("Cancelled active task for job %s", job_id)
    return {"status": "cancelled", "job_id": job_id}
