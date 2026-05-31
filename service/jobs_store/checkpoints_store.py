"""Job stage checkpoint Keyspaces operations."""

import json
from datetime import datetime, timezone
from typing import Any

from service.analysis_keyspaces_enums import (
    PipelineStage,
    parse_pipeline_stage_strict,
)

from service.jobs_store._utils import as_utc_aware


class CheckpointsStoreMixin:
    _client: Any
    _ks: str

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
            "completed": r.completed or False,
            "checkpoint_data": data,
            "updated_at": as_utc_aware(r.updated_at),
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
                "updated_at": as_utc_aware(r.updated_at),
            })
        return result
