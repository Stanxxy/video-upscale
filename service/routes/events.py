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
                        # S12 Phase 1b (design §1.2): additive, v2-only keys
                        # — only present once run_highlight_job has actually
                        # written them (chunk_index/chunks_total/
                        # highlights_found_so_far). A dormant tracking job's
                        # lifecycle row never has these columns set, so this
                        # branch is a no-op for it — byte-identical prior
                        # payload shape for any consumer that only reads
                        # state/percent/frame_idx/total_frames.
                        chunk_index = lifecycle.get("chunk_index")
                        if chunk_index is not None:
                            progress["chunk_index"] = chunk_index
                        chunks_total = lifecycle.get("chunks_total")
                        if chunks_total is not None:
                            progress["chunks_total"] = chunks_total
                        highlights_found_so_far = lifecycle.get("highlights_found_so_far")
                        if highlights_found_so_far is not None:
                            progress["highlights_found_so_far"] = highlights_found_so_far
                        # Item 11.5 — additive per-job attribution metrics,
                        # written once (the job's final progress write,
                        # before terminal state) as a JSON blob.
                        attribution_metrics_json = lifecycle.get("attribution_metrics_json")
                        if attribution_metrics_json:
                            try:
                                progress["attribution_metrics"] = json.loads(attribution_metrics_json)
                            except (ValueError, TypeError):
                                logger.warning(
                                    "SSE: unparseable attribution_metrics_json for job %s", job_id,
                                )
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
