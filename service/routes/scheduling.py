"""Background job scheduling helpers."""

import asyncio
import logging

from service.models import AdmittedTrackRequest
from service.worker import run_highlight_job
from service.routes import state as route_state

logger = logging.getLogger("service.routes")


async def _run_with_semaphore(job_id: str, request: AdmittedTrackRequest):
    """Dispatch to ``run_highlight_job`` — the v2 highlight-scan-critique-
    analyze production path (S12 Phase 1b design §1.1). This is a SINGLE
    call-site swap, not a runtime feature flag: v2 is THE production path
    (founder decision 2), so a permanent ``if use_tracking_pipeline`` branch
    here would be surface area nobody asked for. Reversibility comes from
    source control (revert this swap), not a config knob.

    ``worker/orchestrator.py::run_job`` (the tracking pipeline) is no
    longer called from here — it stays fully importable and its own
    existing unit tests call it directly, unaffected (decision 2, deprecate
    in place, never deleted)."""
    route_state._jobs_store.register_owned_job(job_id)
    try:
        async with route_state._job_semaphore:
            await run_highlight_job(
                job_id,
                request,
                route_state._config,
                route_state._job_store,
                route_state._jobs_store,
            )
    finally:
        route_state._jobs_store.unregister_owned_job(job_id)


async def _cleanup_orphaned_tasks():
    """Remove completed tasks from the active-tasks dict."""
    for jid, task in list(route_state._active_tasks.items()):
        if task.done():
            route_state._active_tasks.pop(jid, None)


def _schedule_job(job_id: str, request: AdmittedTrackRequest) -> None:
    task = asyncio.create_task(_run_with_semaphore(job_id, request))
    route_state._active_tasks[job_id] = task

    def _log_uncaught(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error(
                "Background job task %s exited with uncaught exception",
                job_id,
                exc_info=exc,
            )

    task.add_done_callback(_log_uncaught)
