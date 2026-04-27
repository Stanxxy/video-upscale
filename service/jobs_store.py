"""Persistent jobs store backed by Keyspaces (Cassandra).

Operates on the same four tables as shared_lib's AnalysisJobsStore.
Returns plain dicts (string job_state / stage values) for routes.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from service.analysis_keyspaces_enums import (
    JobState,
    PipelineStage,
    parse_job_state,
    parse_pipeline_stage_optional,
    parse_pipeline_stage_strict,
    states_that_sync_latest_job_row,
    states_with_completed_at,
)
from service.keyspaces_client import KeyspacesClient

logger = logging.getLogger(__name__)


class JobsStore:
    """Async facade over Keyspaces for the 4 analysis-job tables."""

    def __init__(self, client: KeyspacesClient) -> None:
        self._client = client
        self._ks = client.keyspace  # e.g. "video_analysis"
        self.owned_jobs: set[str] = set()

    def register_owned_job(self, job_id: str) -> None:
        self.owned_jobs.add(job_id)

    def unregister_owned_job(self, job_id: str) -> None:
        self.owned_jobs.discard(job_id)

    async def create_lifecycle(
        self,
        job_id: str,
        video_id: str,
        user_id: str,
        origin_job_id: str | None = None,
        owner_instance_id: str = "",
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.job_lifecycle "
            f"(job_id, video_id, user_id, origin_job_id, job_state, stage, "
            f"progress_percent, current_frame, total_frames, stage_message, "
            f"error_message, owner_instance_id, last_heartbeat_at, started_at, "
            f"updated_at, completed_at) "
            f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        )
        return await self._client.execute_write(q, [
            job_id, video_id, user_id, origin_job_id, JobState.PENDING.value, "",
            0.0, 0, 0, "", "", owner_instance_id, now, now, now, None,
        ])

    async def get_lifecycle(self, job_id: str) -> dict[str, Any] | None:
        q = f"SELECT * FROM {self._ks}.job_lifecycle WHERE job_id = %s"
        rows = await self._client.execute(q, [job_id])
        if not rows:
            return None
        r = rows[0]
        st = parse_job_state(r.job_state)
        stg = parse_pipeline_stage_optional(r.stage)
        return {
            "job_id": r.job_id,
            "video_id": r.video_id or "",
            "user_id": r.user_id or "",
            "origin_job_id": r.origin_job_id,
            "job_state": st.value,
            "stage": stg.value if stg else "",
            "progress_percent": r.progress_percent or 0.0,
            "current_frame": r.current_frame or 0,
            "total_frames": r.total_frames or 0,
            "stage_message": r.stage_message or "",
            "error_message": r.error_message or "",
            "owner_instance_id": r.owner_instance_id or "",
            "last_heartbeat_at": r.last_heartbeat_at,
            "started_at": r.started_at,
            "updated_at": r.updated_at,
            "completed_at": r.completed_at,
        }

    async def update_progress(
        self,
        job_id: str,
        stage: PipelineStage,
        progress_percent: float,
        current_frame: int = 0,
        total_frames: int = 0,
        stage_message: str = "",
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"stage = %s, progress_percent = %s, "
            f"current_frame = %s, total_frames = %s, stage_message = %s, "
            f"last_heartbeat_at = %s, updated_at = %s "
            f"WHERE job_id = %s"
        )
        return await self._client.execute_write(q, [
            stage.value, progress_percent,
            current_frame, total_frames, stage_message,
            now, now, job_id,
        ])

    async def set_state(
        self,
        job_id: str,
        state: JobState,
        error_message: str = "",
    ) -> bool:
        now = datetime.now(timezone.utc)
        completed_at = now if state in states_with_completed_at() else None
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"job_state = %s, error_message = %s, "
            f"updated_at = %s, completed_at = %s "
            f"WHERE job_id = %s"
        )
        ok = await self._client.execute_write(q, [
            state.value, error_message, now, completed_at, job_id,
        ])
        if ok and state in states_that_sync_latest_job_row():
            try:
                lifecycle = await self.get_lifecycle(job_id)
                if lifecycle and lifecycle.get("video_id"):
                    await self.set_latest(lifecycle["video_id"], job_id, state)
            except Exception as e:
                logger.warning("Failed to sync latest_job for %s: %s", job_id, e)
        return ok

    async def heartbeat(self, job_id: str, owner_instance_id: str) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"UPDATE {self._ks}.job_lifecycle SET "
            f"last_heartbeat_at = %s, owner_instance_id = %s "
            f"WHERE job_id = %s"
        )
        return await self._client.execute_write(q, [now, owner_instance_id, job_id])

    async def save_request(self, job_id: str, request_json: str) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.job_request_params "
            f"(job_id, request_json, created_at) VALUES (%s, %s, %s)"
        )
        return await self._client.execute_write(q, [job_id, request_json, now])

    async def get_request(self, job_id: str) -> str | None:
        q = f"SELECT request_json FROM {self._ks}.job_request_params WHERE job_id = %s"
        rows = await self._client.execute(q, [job_id])
        if not rows:
            return None
        return rows[0].request_json

    async def write_checkpoint(
        self,
        job_id: str,
        stage_name: PipelineStage,
        completed: bool,
        data: dict[str, Any],
    ) -> bool:
        now = datetime.now(timezone.utc)
        q = (
            f"INSERT INTO {self._ks}.job_stage_checkpoints "
            f"(job_id, stage_name, completed, checkpoint_data, updated_at) "
            f"VALUES (%s, %s, %s, %s, %s)"
        )
        return await self._client.execute_write(q, [
            job_id, stage_name.value, completed, json.dumps(data), now,
        ])

    async def get_checkpoint(
        self, job_id: str, stage_name: PipelineStage
    ) -> dict[str, Any] | None:
        q = (
            f"SELECT * FROM {self._ks}.job_stage_checkpoints "
            f"WHERE job_id = %s AND stage_name = %s"
        )
        rows = await self._client.execute(q, [job_id, stage_name.value])
        if not rows:
            return None
        r = rows[0]
        data: dict[str, Any] = {}
        if r.checkpoint_data:
            try:
                data = json.loads(r.checkpoint_data)
            except (json.JSONDecodeError, TypeError):
                pass
        sn = parse_pipeline_stage_strict(r.stage_name, label="get_checkpoint")
        return {
            "job_id": r.job_id,
            "stage_name": sn.value,
            "completed": r.completed or False, # what does this flag mean? Global completed or stage completed????
            "checkpoint_data": data,
            "updated_at": r.updated_at,
        }

    async def get_all_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        q = f"SELECT * FROM {self._ks}.job_stage_checkpoints WHERE job_id = %s"
        rows = await self._client.execute(q, [job_id])
        if not rows:
            return []
        result: list[dict[str, Any]] = []
        for r in rows:
            data: dict[str, Any] = {}
            if r.checkpoint_data:
                try:
                    data = json.loads(r.checkpoint_data)
                except (json.JSONDecodeError, TypeError):
                    pass
            sn = parse_pipeline_stage_strict(
                r.stage_name, label=f"list checkpoints {job_id}",
            )
            result.append({
                "job_id": r.job_id,
                "stage_name": sn.value,
                "completed": r.completed or False,
                "checkpoint_data": data,
                "updated_at": r.updated_at,
            })
        return result

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
        return {"job_id": r.job_id, "job_state": r.job_state, "updated_at": r.updated_at}

    async def list_running_jobs(self, owner_instance_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job_id in list(self.owned_jobs):
            row = await self.get_lifecycle(job_id)
            if row and row.get("job_state") in (
                JobState.PENDING.value,
                JobState.RUNNING.value,
                JobState.AWAITING_CORRECTION.value,
            ):
                results.append(row)
        return results
