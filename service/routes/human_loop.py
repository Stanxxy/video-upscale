"""Human-in-the-loop detection and resume endpoints."""

import asyncio
import base64
import json
import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import build_resume_plan, select_correction_checkpoint
from service.models import ResumeRequest
from service.routes import state as route_state
from service.routes.resume_job_factory import build_resume_params, create_replacement_job

logger = logging.getLogger("service.routes")


async def get_detection_frame(job_id: str):
    """Return the detection frame as base64 for the QA client canvas.

    The frame JPEG is stored in S3 at ``pending_detection.frame_s3_key``.
    This endpoint fetches it and returns ``{frame_b64: <base64>}``.
    """
    # Find the pending_detection checkpoint (DETECT or TRACK stage)
    det: dict | None = None
    for stage_name in (PipelineStage.DETECT, PipelineStage.TRACK):
        cp = await route_state._jobs_store.get_checkpoint(job_id, stage_name)
        if cp:
            pd = cp.get("checkpoint_data", {}).get("pending_detection")
            if pd:
                det = pd
                break

    if not det:
        raise HTTPException(404, "No pending detection for this job")

    bucket = det.get("frame_bucket", "")
    key = det.get("frame_s3_key", "")
    if not bucket or not key:
        raise HTTPException(404, "Detection frame S3 location not recorded")

    try:
        s3 = route_state._s3_client()
        jpeg_bytes = await asyncio.get_event_loop().run_in_executor(
            None, lambda: s3.get_object(bucket, key)
        )
        frame_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return JSONResponse({"frame_b64": frame_b64, "frame_idx": det.get("frame_idx", 0)})
    except Exception as e:
        logger.warning("get_detection_frame: S3 fetch failed job=%s: %s", job_id, e)
        raise HTTPException(502, f"Could not fetch frame from S3: {e}")


async def submit_detection_response(job_id: str, body: ResumeRequest):
    """Submit corrected bounding boxes for an AWAITING_CORRECTION job.

    Creates a new resume job with the provided boxes and returns its ID.
    """
    # G7: resume must honor the kill switch, but does NOT count against the daily cap.
    route_state._check_kill_switch()

    lifecycle = await route_state._jobs_store.get_lifecycle(job_id)
    if not lifecycle:
        raise HTTPException(404, "Job not found")
    if lifecycle["job_state"] != JobState.AWAITING_CORRECTION.value:
        raise HTTPException(409, f"Job state is {lifecycle['job_state']}, not AWAITING_CORRECTION")
    if lifecycle.get("replacement_job_id"):
        raise HTTPException(
            409,
            f"Job already has replacement {lifecycle['replacement_job_id']}",
        )

    request_json = await route_state._jobs_store.get_request(job_id)
    if not request_json:
        raise HTTPException(404, "Original request not found")

    orig_request = json.loads(request_json)
    checkpoints = await route_state._jobs_store.get_all_checkpoints(job_id)
    correction_stage, _ = select_correction_checkpoint(checkpoints)
    resume_plan = build_resume_plan(checkpoints)
    resume_params = build_resume_params(
        job_id, orig_request, checkpoints, resume_plan, body=body,
    )

    result = await create_replacement_job(
        job_id,
        lifecycle,
        resume_params,
        checkpoints,
        correction_stage,
        cancel_source_job=True,
        verified_boxes=(body.box_a, body.box_b),
    )
    assert result is not None

    return {"status": "resumed", "job_id": result.new_job_id, "origin_job_id": job_id}


async def resume_job(job_id: str, body: ResumeRequest):
    """Deliver corrected bounding boxes to a job waiting for detection.

    Delegates to the detection_response endpoint logic: creates a new
    resume job with the corrected boxes.
    """
    return await submit_detection_response(job_id, body)
