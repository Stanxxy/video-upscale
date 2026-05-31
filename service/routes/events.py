"""SSE job event streaming endpoint."""

import asyncio
import json
import logging

from fastapi.responses import StreamingResponse

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.routes import state as route_state

logger = logging.getLogger("service.routes")


async def job_events_sse(job_id: str):
    """Server-Sent Events endpoint that tails the Keyspaces job_lifecycle row."""

    async def event_generator():
        last_state = None
        last_pct = -1.0
        consecutive_failures = 0
        tick = 0
        while True:
            try:
                lifecycle = await route_state._jobs_store.get_lifecycle(job_id)
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
                            cp = await route_state._jobs_store.get_checkpoint(job_id, stage_name)
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
