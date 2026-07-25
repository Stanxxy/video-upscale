"""Track job CRUD endpoints."""

import logging

from fastapi import HTTPException

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import WorkerStateSnapshot, build_cancellation_checkpoint
from service.models import TrackRequest, TrackResponse, JobResponse
from service.pipelines import registry
from service.pipelines.highlight_settings_override import (
    PIPELINE_ID as HIGHLIGHT_PIPELINE_ID,
    apply_analysis_settings_override,
    has_any_analysis_setting,
)
from service.routes import state as route_state
import service.routes as routes_pkg

logger = logging.getLogger("service.routes")


def _validate_analysis_settings(request: TrackRequest) -> None:
    """S12 pre-analysis AI settings spec §4/§5 AC3 — fail-closed, never a
    silent default swap (INS-055). Reuses the SAME allowlist +
    ``registry.validate_pipeline_def`` machinery the QA pipeline registry
    already uses: build the overridden default pipeline def and run it
    through the real validator. An off-allowlist ``analysis_model`` or an
    invalid ``analysis_media_resolution``/``analysis_fps``/
    ``analysis_thinking`` value both fail the same way — HTTP 400 with a
    clear message (allowlist echoed for the model case) — no new,
    parallel validation logic. No-op when all four settings are absent."""
    if not has_any_analysis_setting(request):
        return
    overridden = apply_analysis_settings_override(registry.get_default(HIGHLIGHT_PIPELINE_ID), request)
    try:
        registry.validate_pipeline_def(overridden)
    except registry.PipelineValidationError as e:
        raise HTTPException(400, str(e)) from e


async def create_track_job(request: TrackRequest):
    # G7 cost guardrails: kill switch then daily cap (NEW analyses only).
    route_state._check_kill_switch()
    # Validate BEFORE consuming a daily-cap admission slot — a request that's
    # going to 400 on an off-allowlist model/invalid setting shouldn't cost
    # the caller (or the project) an admission they can't use.
    _validate_analysis_settings(request)
    route_state._admit_daily()

    # Proactively clean up completed tasks before checking capacity
    await routes_pkg._cleanup_orphaned_tasks()

    if routes_pkg._job_semaphore.locked():
        raise HTTPException(429, "Server is at capacity. Try again later.")

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
            # S12 Phase 1b (design §1.1/§6.2): v2 is THE production path —
            # every job created via POST /track is a highlight_v2 job.
            # Existing rows created before this column existed read back as
            # "tracking" (see jobs_store/lifecycle.py::get_lifecycle).
            pipeline_kind="highlight_v2",
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
