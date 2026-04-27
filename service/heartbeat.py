"""Background asyncio task that writes heartbeats for all owned jobs."""

import asyncio
import logging

from service.jobs_store import JobsStore

logger = logging.getLogger(__name__)


class HeartbeatTask:
    def __init__(self, jobs_store: JobsStore, instance_id: str, interval: float = 5.0):
        self._store = jobs_store
        self._instance_id = instance_id
        self._interval = interval
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                for job_id in list(self._store.owned_jobs):
                    await self._store.heartbeat(job_id, self._instance_id)
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
