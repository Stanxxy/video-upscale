"""Job recovery index Keyspaces operations."""

from datetime import datetime, timezone
from typing import Any

from service.analysis_keyspaces_enums import JobState, parse_job_state
from service.jobs_store._utils import as_utc_aware


class RecoveryIndexMixin:
    _client: Any
    _ks: str

    async def upsert_recovery_index(self, lifecycle: dict[str, Any]) -> bool:
        last_heartbeat_at = lifecycle.get("last_heartbeat_at") or datetime.now(timezone.utc)
        heartbeat_bucket = self.heartbeat_bucket_for(last_heartbeat_at)
        state = parse_job_state(lifecycle.get("job_state"))
        recovery_state = (
            "ACTIVE"
            if state in (JobState.PENDING, JobState.RUNNING, JobState.INTERRUPTED)
            else "AWAITING_CORRECTION"
            if state == JobState.AWAITING_CORRECTION
            else "TERMINAL"
        )
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.job_recovery_index "
            f"(recovery_state, heartbeat_bucket, last_heartbeat_at, job_id, "
            f"owner_instance_id, video_id, job_state, updated_at) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        )
        return await self._client.execute_write(q, [
            recovery_state,
            heartbeat_bucket,
            last_heartbeat_at,
            lifecycle["job_id"],
            lifecycle.get("owner_instance_id", ""),
            lifecycle.get("video_id", ""),
            state.value,
            now,
        ])

    async def remove_recovery_index(
        self,
        job_id: str,
        recovery_state: str,
        heartbeat_bucket: str,
        last_heartbeat_at: datetime,
    ) -> bool:
        q = (
            f"DELETE FROM {self._ks}.job_recovery_index "
            f"WHERE recovery_state = %s AND heartbeat_bucket = %s "
            f"AND last_heartbeat_at = %s AND job_id = %s"
        )
        return await self._client.execute_write(q, [
            recovery_state, heartbeat_bucket, last_heartbeat_at, job_id,
        ])

    async def list_active_recovery_index_rows_newest_first(
        self,
        heartbeat_buckets: list[str],
        *,
        limit_per_bucket: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return recent rows from ACTIVE recovery partitions (newest heartbeats first)."""
        results: list[dict[str, Any]] = []
        q = (
            f"SELECT job_id, video_id, job_state, owner_instance_id, last_heartbeat_at "
            f"FROM {self._ks}.job_recovery_index "
            f"WHERE recovery_state = %s AND heartbeat_bucket = %s "
            f"ORDER BY last_heartbeat_at DESC LIMIT %s"
        )
        for bucket in heartbeat_buckets:
            rows = await self._client.execute(
                q, ["ACTIVE", bucket, limit_per_bucket],
            )
            for r in rows:
                results.append({
                    "job_id": r.job_id,
                    "video_id": getattr(r, "video_id", None) or "",
                    "job_state": getattr(r, "job_state", None) or "",
                    "owner_instance_id": getattr(r, "owner_instance_id", None) or "",
                    "last_heartbeat_at": as_utc_aware(
                        getattr(r, "last_heartbeat_at", None),
                    ),
                    "heartbeat_bucket": bucket,
                })
        return results

    async def list_stale_recovery_candidates(
        self,
        heartbeat_buckets: list[str],
        stale_before: datetime,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        q = (
            f"SELECT * FROM {self._ks}.job_recovery_index "
            f"WHERE recovery_state = %s AND heartbeat_bucket = %s "
            f"AND last_heartbeat_at < %s"
        )
        for bucket in heartbeat_buckets:
            rows = await self._client.execute(q, ["ACTIVE", bucket, stale_before])
            for r in rows:
                results.append({
                    "job_id": r.job_id,
                    "video_id": r.video_id or "",
                    "job_state": r.job_state or "",
                    "owner_instance_id": r.owner_instance_id or "",
                    "last_heartbeat_at": as_utc_aware(r.last_heartbeat_at),
                    "heartbeat_bucket": r.heartbeat_bucket,
                    "updated_at": as_utc_aware(r.updated_at),
                })
        return results
