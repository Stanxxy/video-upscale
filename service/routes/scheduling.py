"""Background job scheduling helpers."""

import asyncio
import logging

from service.models import TrackRequest
from service.worker import run_job
from service.routes import state as route_state

logger = logging.getLogger("service.routes")


async def _run_with_semaphore(job_id: str, request: TrackRequest):
    route_state._jobs_store.register_owned_job(job_id)
    try:
        async with route_state._job_semaphore:
            await run_job(
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


def _schedule_job(job_id: str, request: TrackRequest) -> None:
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
