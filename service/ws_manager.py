"""
Per-job WebSocket lifecycle, message routing, and detection request/response exchange.

The tracking loop runs synchronously in a worker thread. When it needs human input
(detection_callback), it blocks. The WebSocket handler runs in the asyncio event loop.
This module bridges the two via asyncio.Event for synchronization.
"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSManager:
    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._detection_events: dict[str, asyncio.Event] = {}
        self._detection_responses: dict[str, Optional[dict]] = {}

    # -- Connection lifecycle --

    def register(self, job_id: str, websocket: WebSocket) -> None:
        self._connections[job_id] = websocket
        self._detection_events[job_id] = asyncio.Event()
        self._detection_responses[job_id] = None

    def unregister(self, job_id: str) -> None:
        self._connections.pop(job_id, None)
        self._detection_events.pop(job_id, None)
        self._detection_responses.pop(job_id, None)

    def is_connected(self, job_id: str) -> bool:
        return job_id in self._connections

    # -- Outbound messages (server -> client) --

    async def _send(self, job_id: str, payload: dict) -> None:
        ws = self._connections.get(job_id)
        if ws is None:
            return
        try:
            await ws.send_json(payload)
        except Exception as e:
            logger.warning("WS send failed for job %s: %s", job_id, e)

    async def send_progress(
        self,
        job_id: str,
        frame_idx: int,
        total_frames: int,
        state: str,
        percent: float,
    ) -> None:
        await self._send(job_id, {
            "type": "progress",
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "state": state,
            "percent": round(percent, 1),
        })

    async def send_detection_needed(
        self,
        job_id: str,
        frame_idx: int,
        frame_b64: str,
        reason: str,
        candidates: list[dict] | None = None,
        suggested_boxes: dict | None = None,
    ) -> None:
        msg = {
            "type": "detection_needed",
            "frame_idx": frame_idx,
            "frame_b64": frame_b64,
            "reason": reason,
        }
        if candidates is not None:
            msg["candidates"] = candidates
        if suggested_boxes is not None:
            msg["suggested_boxes"] = suggested_boxes
        await self._send(job_id, msg)

    async def send_completed(
        self,
        job_id: str,
        result_bucket: str,
        result_key: str,
        tracking_key: str | None = None,
        annotated_video_key: str | None = None,
    ) -> None:
        msg = {
            "type": "completed",
            "result_bucket": result_bucket,
            "result_key": result_key,
        }
        if tracking_key:
            msg["tracking_key"] = tracking_key
        if annotated_video_key:
            msg["annotated_video_key"] = annotated_video_key
        await self._send(job_id, msg)

    async def send_error(self, job_id: str, message: str) -> None:
        await self._send(job_id, {"type": "error", "message": message})

    # -- Detection request/response synchronization --

    async def wait_for_detection(
        self, job_id: str, timeout: float
    ) -> Optional[dict]:
        """Block until detection_response arrives or timeout. Returns None on timeout/cancel."""
        event = self._detection_events.get(job_id)
        if event is None:
            return None
        event.clear()
        self._detection_responses[job_id] = None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Detection timeout for job %s", job_id)
            return None
        return self._detection_responses.get(job_id)

    def deliver_detection(self, job_id: str, payload: dict) -> None:
        """Called by WS route when client sends detection_response."""
        self._detection_responses[job_id] = payload
        event = self._detection_events.get(job_id)
        if event is not None:
            event.set()

    def cancel_detection(self, job_id: str) -> None:
        """Called when detection_cancelled received. Unblocks wait with None."""
        self._detection_responses[job_id] = None
        event = self._detection_events.get(job_id)
        if event is not None:
            event.set()
