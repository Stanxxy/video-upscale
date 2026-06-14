"""Shared replacement-job creation for detection handoff and recovery."""

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    WorkerStateSnapshot,
    build_replaced_by_new_job,
    build_verified_boxes_checkpoint,
    resume_plan_to_request_fields,
    worker_state_from,
)
from service.models import ResumeRequest, TrackRequest
from service.tracking_chain_merge import preflight_resume_tracking_overrides
from service.routes import state as route_state
import service.routes as routes_pkg


@dataclass(frozen=True)
class ReplacementJobResult:
    new_job_id: str
    new_request: TrackRequest
    source_job_id: str


def build_resume_params(
    source_job_id: str,
    orig_request: dict,
    checkpoints: list,
    resume_plan,
    *,
    body: ResumeRequest | None = None,
) -> dict:
    """Build TrackRequest kwargs from original request and checkpoint resume plan."""
    track_overrides = dict(resume_plan.track_request_overrides)
    bucket = (orig_request.get("output_bucket") or orig_request.get("bucket") or "").strip()
    if bucket and track_overrides.get("resume_tracking_s3_key") and route_state._config:
        track_overrides = preflight_resume_tracking_overrides(
            track_overrides, checkpoints, route_state._s3_client(), bucket,
        )

    if body is not None:
        resume_params = {
            **orig_request,
            # LEGACY: box_a/box_b superseded by player_mapping (track_id<->player_id). Remove once all paths consume bindings.
            "box_a": body.box_a,
            "box_b": body.box_b,
            # Stream 0b: carry the confirmed obj_id->player_id binding onto the
            # replacement job so tracking re-seeds init_boxes from it (no flip).
            "player_mapping": body.player_mapping,
            **track_overrides,
            **resume_plan_to_request_fields(resume_plan),
        }
    else:
        resume_params = dict(orig_request)
        resume_params.update(track_overrides)
        resume_params.update(resume_plan_to_request_fields(resume_plan))

    if "resume_tracking_s3_key" in resume_params:
        resume_params["resume_from_job_id"] = source_job_id
    return resume_params


async def create_replacement_job(
    source_job_id: str,
    lifecycle: dict,
    resume_params: dict,
    checkpoints: list,
    correction_stage: PipelineStage | None,
    *,
    claim_expected_state: JobState | None = None,
    on_claim_failure: Literal["raise_409", "cancel_and_return"] = "raise_409",
    lifecycle_operation: str = "replacement job lifecycle",
    request_operation: str = "replacement job request",
    latest_operation: str = "latest replacement job",
    old_checkpoint_operation: str = "old job replacement checkpoint",
    cancel_source_job: bool = False,
    verified_boxes: tuple[list, list] | None = None,
) -> ReplacementJobResult | None:
    """Persist and schedule a replacement job; shared by detection handoff and recovery."""
    new_request = TrackRequest(**resume_params)
    new_job = await route_state._job_store.create_job(new_request)
    new_job_id = new_job.job_id

    video_id = resume_params.get("video_id") or lifecycle.get("video_id", "")
    origin_job_id = lifecycle.get("origin_job_id") or source_job_id
    seed_ws = worker_state_from(checkpoints) or {}
    route_state._require_write(
        await route_state._jobs_store.create_lifecycle(
            new_job_id,
            str(video_id),
            lifecycle.get("user_id", ""),
            origin_job_id=origin_job_id,
            parent_job_id=source_job_id,
            owner_instance_id=route_state._instance_id,
            progress_percent=seed_ws.get("progress_percent", 0.0),
            current_frame=seed_ws.get("current_frame", 0),
            total_frames=seed_ws.get("total_frames", 0),
        ),
        lifecycle_operation,
    )
    route_state._require_write(
        await route_state._jobs_store.save_request(new_job_id, new_request.model_dump_json()),
        request_operation,
    )

    claim_kwargs = {"job_id": source_job_id, "replacement_job_id": new_job_id}
    if claim_expected_state is not None:
        claim_kwargs["expected_state"] = claim_expected_state
    claimed = await route_state._jobs_store.claim_replacement(**claim_kwargs)
    if not claimed:
        await route_state._jobs_store.set_state(new_job_id, JobState.CANCELLED, sync_latest=False)
        if on_claim_failure == "raise_409":
            raise HTTPException(409, "Job replacement claim was already taken")
        return None

    if video_id:
        route_state._require_write(
            await route_state._jobs_store.set_latest(str(video_id), new_job_id, JobState.PENDING),
            latest_operation,
        )

    old_ws_dict = worker_state_from(checkpoints) or {
        "progress_percent": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "stage_progress_fraction": 0.0,
    }
    old_ws = WorkerStateSnapshot(**old_ws_dict)
    terminal_stage = correction_stage or PipelineStage.TRACK
    route_state._require_write(
        await route_state._jobs_store.write_checkpoint(
            source_job_id,
            terminal_stage,
            True,
            build_replaced_by_new_job(
                replacement_job_id=new_job_id,
                worker_state=old_ws,
            ),
        ),
        old_checkpoint_operation,
    )

    if cancel_source_job:
        route_state._require_write(
            await route_state._jobs_store.set_state(
                source_job_id, JobState.CANCELLED, sync_latest=False,
            ),
            "source job cancellation state",
        )

    if verified_boxes is not None:
        box_a, box_b = verified_boxes
        verified_ws = WorkerStateSnapshot(
            progress_percent=15.0,
            current_frame=0,
            total_frames=0,
            stage_progress_fraction=1.0,
        )
        route_state._require_write(
            await route_state._jobs_store.write_checkpoint(
                new_job_id,
                PipelineStage.TRACK,
                False,
                build_verified_boxes_checkpoint(
                    box_a, box_b, correction_stage,
                    worker_state=verified_ws,
                ),
            ),
            "verified boxes checkpoint",
        )

    routes_pkg._schedule_job(new_job_id, new_request)
    return ReplacementJobResult(
        new_job_id=new_job_id,
        new_request=new_request,
        source_job_id=source_job_id,
    )
