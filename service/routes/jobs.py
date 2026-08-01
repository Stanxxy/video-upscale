"""Track job CRUD endpoints."""

import json
import logging

from fastapi import HTTPException

from service.analysis_settings import AnalysisSettingsValidationError, resolve_analysis_settings
from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import WorkerStateSnapshot, build_cancellation_checkpoint
from service.models import AdmittedTrackRequest, TrackRequest, TrackResponse, JobResponse
from service.routes import state as route_state
import service.routes as routes_pkg

logger = logging.getLogger("service.routes")


async def _analysis_settings_diagnostics(job_id: str) -> dict:
    """Expose only the persisted R4 audit envelope, never storage credentials."""
    request_json = await route_state._jobs_store.get_request(job_id)
    if not request_json:
        return {}
    payload = json.loads(request_json)
    required = {
        "capability_schema_version",
        "requested_analysis_settings",
        "effective_analysis_config",
    }
    if not required.intersection(payload):
        return {}  # Historic pre-R4 completed job; no snapshot was recorded.
    try:
        admitted = AdmittedTrackRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Persisted analysis settings snapshot is invalid",
        ) from exc
    return {
        "capability_schema_version": admitted.capability_schema_version,
        "requested_analysis_settings": admitted.requested_analysis_settings,
        "effective_analysis_config": admitted.effective_analysis_config,
    }


async def create_track_job(request: TrackRequest):
    try:
        admitted_request = resolve_analysis_settings(request)
    except AnalysisSettingsValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # G7 cost guardrails: kill switch then daily cap (NEW analyses only).
    route_state._check_kill_switch()
    route_state._admit_daily()

    # Proactively clean up completed tasks before checking capacity
    await routes_pkg._cleanup_orphaned_tasks()

    if routes_pkg._job_semaphore.locked():
        raise HTTPException(429, "Server is at capacity. Try again later.")

    job = await route_state._job_store.create_job(admitted_request)
    job_id = job.job_id

    # Persist to Keyspaces (user_id="" for engine-direct jobs; the analysis
    # service populates it when submitting via POST /tracking/analyze)

    # TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.
    route_state._require_write(
        await route_state._jobs_store.create_lifecycle(
            job_id,
            admitted_request.video_id,
            admitted_request.user_id or "",
            "",
            owner_instance_id=route_state._instance_id,
            # S12 Phase 1b (design §1.1/§6.2): v2 is THE production path —
            # every job created via POST /track is a highlight_v2 job.
            # Existing rows created before this column existed read back as
            # "tracking" (see jobs_store/lifecycle.py::get_lifecycle).
            pipeline_kind="highlight_v2",
        ),
        "job lifecycle",
    )
    route_state._require_write(
        await route_state._jobs_store.save_request(job_id, admitted_request.model_dump_json()),
        "job request",
    )
    if admitted_request.video_id:
        route_state._require_write(
            await route_state._jobs_store.set_latest(
                str(admitted_request.video_id), job_id, JobState.PENDING,
            ),
            "latest job",
        )

    # Start the job immediately; clients receive lifecycle updates through
    # the Keyspaces-backed SSE projection.
    routes_pkg._schedule_job(job_id, admitted_request)

    return TrackResponse(job_id=job_id, status="pending")


async def get_job(job_id: str):
    diagnostics = await _analysis_settings_diagnostics(job_id)
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
            **diagnostics,
            created_at=str(lifecycle.get("started_at", "")),
            updated_at=str(lifecycle.get("updated_at", "")),
        )
    # Fall back to in-memory
    job = await route_state._job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobResponse(**job.model_dump(), **diagnostics)


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
