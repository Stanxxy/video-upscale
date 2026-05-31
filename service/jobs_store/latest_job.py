"""video_analysis_latest_job Keyspaces operations."""

from datetime import datetime, timezone
from typing import Any

from service.analysis_keyspaces_enums import JobState

from service.jobs_store._utils import as_utc_aware


class LatestJobMixin:
    _client: Any
    _ks: str

    async def set_latest(
        self, video_id: str, job_id: str, job_state: JobState
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.video_analysis_latest_job "
            f"(video_id, job_id, job_state, updated_at) "
            f"VALUES (%s, %s, %s, %s)"
        )
        return await self._client.execute_write(q, [video_id, job_id, job_state.value, now])

    async def get_latest(self, video_id: str) -> dict[str, Any] | None:
        q = (
            f"SELECT job_id, job_state, updated_at "
            f"FROM {self._ks}.video_analysis_latest_job WHERE video_id = %s"
        )
        rows = await self._client.execute(q, [video_id])
        if not rows:
            return None
        r = rows[0]
        return {
            "job_id": r.job_id,
            "job_state": r.job_state,
            "updated_at": as_utc_aware(r.updated_at),
        }
