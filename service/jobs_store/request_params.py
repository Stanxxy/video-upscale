"""Job request params Keyspaces operations."""

from datetime import datetime, timezone
from typing import Any


class RequestParamsMixin:
    _client: Any
    _ks: str

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
