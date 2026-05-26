"""Scheduled recovery manager for stale worker-owned jobs."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

from service.analysis_keyspaces_enums import JobState
from service.jobs_store import JobsStore

logger = logging.getLogger(__name__)

RecoverJobCallback = Callable[[dict], Awaitable[None]]


class RecoveryManager:
    def __init__(
        self,
        jobs_store: JobsStore,
        instance_id: str,
        *,
        interval: float = 30.0,
        stale_after: float = 90.0,
        heartbeat_bucket_hours: int = 24,
        recover_job: RecoverJobCallback | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self._store = jobs_store
        self._instance_id = instance_id
        self._interval = interval
        self._stale_after = stale_after
        self._heartbeat_bucket_hours = max(1, int(heartbeat_bucket_hours))
        self._recover_job = recover_job
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        logger.info(
            "RecoveryManager started: instance=%s interval=%ss stale_after=%ss bucket_hours=%s",
            self._instance_id,
            self._interval,
            self._stale_after,
            self._heartbeat_bucket_hours,
        )
        while True:
            try:
                await self.reconcile_once()
            except Exception as e:
                logger.warning(
                    "Recovery manager reconcile failed: %s", e, exc_info=True,
                )
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def reconcile_once(self, *, stale_after_override: float | None = None) -> None:
        """Run one reconciliation pass.

        ``stale_after_override`` (seconds) replaces ``self._stale_after`` for this
        single pass when not ``None``. Passing ``0.0`` treats any heartbeat strictly
        in the past as stale — used by the lifespan bootstrap to claim orphaned
        ``RUNNING`` / ``INTERRUPTED`` rows from a previous process before the
        periodic loop's conservative window would otherwise wait 90+s.

        The override only affects the heartbeat-staleness threshold; the recovery
        index bucket scan window is governed by ``heartbeat_bucket_hours`` and is
        unchanged here.
        """
        now = self._now_fn()
        stale_after = (
            self._stale_after if stale_after_override is None
            else float(stale_after_override)
        )
        stale_before = now - timedelta(seconds=stale_after)
        buckets = self.heartbeat_buckets_for_scan(
            now, hours=self._heartbeat_bucket_hours,
        )
        candidates = await self._store.list_stale_recovery_candidates(
            buckets, stale_before,
        )
        logger.info(
            "Recovery tick: candidates=%d stale_after=%ss buckets=%d",
            len(candidates),
            stale_after,
            len(buckets),
        )
        for candidate in candidates:
            job_id = candidate["job_id"]
            lifecycle = await self._store.get_lifecycle(job_id)
            if not lifecycle:
                logger.debug("Recovery skip job %s: no lifecycle row", job_id)
                continue
            job_state = lifecycle.get("job_state")
            if job_state not in (JobState.RUNNING.value, JobState.INTERRUPTED.value):
                logger.debug(
                    "Recovery skip job %s: state-not-eligible (%s)", job_id, job_state,
                )
                continue
            if lifecycle.get("replacement_job_id"):
                logger.debug(
                    "Recovery skip job %s: replacement-set (%s)",
                    job_id,
                    lifecycle.get("replacement_job_id"),
                )
                continue
            last_heartbeat_at = lifecycle.get("last_heartbeat_at")
            if last_heartbeat_at and last_heartbeat_at >= stale_before:
                logger.debug(
                    "Recovery skip job %s: heartbeat-fresh (%s >= %s)",
                    job_id,
                    last_heartbeat_at,
                    stale_before,
                )
                continue

            owner_instance_id = lifecycle.get("owner_instance_id", "")
            expected_state = JobState(job_state)
            claimed = await self._store.claim_job_for_recovery(
                job_id,
                self._instance_id,
                owner_instance_id,
                expected_state=expected_state,
                expected_last_heartbeat_at=last_heartbeat_at,
            )
            if not claimed:
                logger.info("Recovery claim lost for job %s", job_id)
                continue

            if expected_state == JobState.RUNNING:
                ok = await self._store.set_state(
                    job_id,
                    JobState.INTERRUPTED,
                    error_message="Worker heartbeat stale; scheduling recovery",
                )
                if not ok:
                    logger.warning("Failed to mark job %s interrupted", job_id)
                    continue
            if self._recover_job:
                logger.info(
                    "Recovery: claimed and dispatched job_id=%s from owner=%s heartbeat=%s",
                    job_id,
                    owner_instance_id,
                    last_heartbeat_at,
                )
                await self._recover_job(lifecycle)

    @staticmethod
    def heartbeat_buckets_for_scan(now: datetime, *, hours: int = 24) -> list[str]:
        """Distinct ``YYYYMMDDHH`` buckets for ``now`` and the prior ``hours - 1`` UTC hours.

        Index rows are partitioned by the hour of ``last_heartbeat_at`` at write time; after a
        long outage the stale row may live in a bucket far behind "now", so recovery must scan
        a trailing window (default 24h) rather than only the latest two hours.
        """
        h = max(1, int(hours))
        anchor = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
        seen: set[str] = set()
        ordered: list[str] = []
        for i in range(h):
            b = (anchor - timedelta(hours=i)).strftime("%Y%m%d%H")
            if b not in seen:
                seen.add(b)
                ordered.append(b)
        return ordered


Reconciler = RecoveryManager
