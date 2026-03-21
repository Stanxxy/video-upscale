import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from service.models import TrackRequest, TrackResponse, JobResponse, ResumeRequest, JobStatus
from service.job_store import InMemoryJobStore
from service.worker import run_job
from service.config import ServiceConfig
from service.ws_manager import WSManager

_QA_HTML = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "qa_client", "index.html"
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Set during app lifespan via init_routes()
_config: ServiceConfig | None = None
_job_store: InMemoryJobStore | None = None
_ws_manager: WSManager | None = None
_job_semaphore: asyncio.Semaphore | None = None
_active_tasks: dict[str, asyncio.Task] = {}  # job_id -> task


def init_routes(
    config: ServiceConfig,
    job_store: InMemoryJobStore,
    ws_manager: WSManager,
):
    global _config, _job_store, _ws_manager, _job_semaphore
    _config = config
    _job_store = job_store
    _ws_manager = ws_manager
    _job_semaphore = asyncio.Semaphore(config.max_concurrent_jobs)


async def _run_with_semaphore(job_id: str, request: TrackRequest):
    async with _job_semaphore:
        await run_job(job_id, request, _config, _job_store, _ws_manager)


async def _cleanup_orphaned_tasks():
    """Cancel tasks whose WebSocket is no longer connected."""
    for jid, task in list(_active_tasks.items()):
        if task.done():
            _active_tasks.pop(jid, None)
            continue
        if not _ws_manager.is_connected(jid):
            logger.info("Cleaning up orphaned task for job %s", jid)
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            _active_tasks.pop(jid, None)


# ---------------------------------------------------------------------------
# QA client
# ---------------------------------------------------------------------------

@router.get("/", include_in_schema=False)
async def qa_client():
    return FileResponse(_QA_HTML, media_type="text/html")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.post("/track", response_model=TrackResponse)
async def create_track_job(request: TrackRequest):
    # Proactively clean up completed/orphaned tasks before checking capacity
    await _cleanup_orphaned_tasks()

    if _job_semaphore.locked():
        raise HTTPException(429, "Server is at capacity. Try again later.")

    job = await _job_store.create_job(request)
    ws_url = f"ws://localhost:{_config.service_port}/ws/{job.job_id}"
    return TrackResponse(job_id=job.job_id, ws_url=ws_url, status="pending")


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    job = await _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    job = await _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    await _job_store.set_cancelled(job_id)
    # Stop the running job task so tracking (and later stages) exit promptly
    task = _active_tasks.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        logger.info("Cancelled active task for job %s", job_id)
    return {"status": "cancelled", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, body: ResumeRequest):
    """Deliver corrected bounding boxes to a job waiting for detection.

    This is the REST equivalent of sending a ``detection_response`` message
    over the WebSocket.  It allows async correction: the user can come back
    hours later and provide boxes via a regular HTTP call.
    """
    job = await _job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    terminal_states = {
        JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED,
    }
    if job.status in terminal_states:
        raise HTTPException(
            409,
            f"Job is in terminal state '{job.status}' and cannot be resumed",
        )

    if job.status != JobStatus.WAITING_FOR_DETECTION:
        raise HTTPException(
            409,
            f"Job is in state '{job.status}', not 'waiting_for_detection'",
        )

    # Deliver the detection payload (same shape the WS handler uses)
    payload = {
        "type": "detection_response",
        "box_a": body.box_a,
        "box_b": body.box_b,
    }
    if body.player_mapping:
        payload["player_mapping"] = body.player_mapping

    _ws_manager.deliver_detection(job_id, payload)
    logger.info("Resume endpoint delivered detection for job %s", job_id)

    return {"status": "resumed", "job_id": job_id}


@router.get("/health")
async def health():
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    job = await _job_store.get_job(job_id)
    if job is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
    _ws_manager.register(job_id, websocket)

    # Retrieve the original request and start the pipeline
    request = _job_store.get_request(job_id)
    if request is None:
        await websocket.send_json({"type": "error", "message": "Request data lost"})
        await websocket.close()
        _ws_manager.unregister(job_id)
        return

    task = asyncio.create_task(_run_with_semaphore(job_id, request))
    _active_tasks[job_id] = task

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "detection_response":
                _ws_manager.deliver_detection(job_id, msg)
            elif msg_type == "detection_cancelled":
                _ws_manager.cancel_detection(job_id)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception as e:
        logger.warning("WebSocket error for job %s: %s", job_id, e)
    finally:
        _ws_manager.unregister(job_id)
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        _active_tasks.pop(job_id, None)
