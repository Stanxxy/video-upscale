"""Startup reconciler that transitions orphaned jobs to INTERRUPTED.

TODO(PRODUCTION-BLOCKER): This is currently a stub. The bounce-proof guarantee
requires scanning Keyspaces for rows where last_heartbeat_at is stale and
job_state is still RUNNING, then transitioning them to INTERRUPTED + emitting
SNS notifications. Options:
  1. Persist owned job_ids to a local file; on restart, read + reconcile those IDs.
  2. Add a GSI/secondary index on owner_instance_id (Keyspaces supports this).
  3. Use a bounded time-window scan via the video_analysis_latest_job table
     (small number of rows) and cross-check each job's heartbeat.
Until this is implemented, orphaned jobs from engine restarts will remain
in RUNNING state and the analysis service must detect them via stale heartbeats
when it reads job_lifecycle for status queries.
"""

import logging

from service.jobs_store import JobsStore

logger = logging.getLogger(__name__)


class Reconciler:
    def __init__(self, jobs_store: JobsStore, instance_id: str):
        self._store = jobs_store
        self._instance_id = instance_id

    async def run_on_startup(self) -> None:
        """Check for orphaned jobs from a previous process crash.

        Currently a stub — see module docstring for implementation options.
        """
        logger.info(
            "Reconciler: startup check (instance=%s) — stub, no orphan scan yet",
            self._instance_id,
        )
