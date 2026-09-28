"""Read-only Keyspaces/S3 checkpoint evidence; host supplies clients and authority.

No schema provisioning, checkpoint writes, leases, provider or engine service imports.
"""

import asyncio
import hashlib
import re
from uuid import UUID

from bjj_analysis_core.execution import StepCheckpointV1
from bjj_analysis_core.requests import strict_json
from cassandra import ConsistencyLevel
from cassandra.policies import FallthroughRetryPolicy
from cassandra.query import SimpleStatement
from shared_lib.models.analysis_contracts import PrivateArtifactRef
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024


class CheckpointIntegrityError(ValueError):
    pass


def _target(keyspace, table):
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", v)
        for v in (keyspace, table)
    ):
        raise ValueError("explicit safe CQL identifiers required")
    return f"{keyspace}.{table}"


class KeyspacesCheckpointReader:
    def __init__(
        self,
        run,
        session,
        s3_client,
        *,
        bucket,
        keyspace,
        guard,
        table="analysis_execution_checkpoints",
        timeout=10,
    ):
        self.run = RegisteredAIRunV1.model_validate(run)
        self.target = _target(keyspace, table)
        if not isinstance(bucket, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket
        ):
            raise ValueError("explicit private S3 bucket required")
        if not callable(guard):
            raise ValueError("current-authority/lease guard required")
        self.session, self.s3, self.bucket = session, s3_client, bucket
        self.guard, self.timeout = guard, timeout

    def _key(self, step_id, revision, digest):
        return (
            f"analysis-checkpoints/v1/{self.run.spec.analysis_id}/{self.run.spec.run_id}/"
            f"{step_id}/{revision}/{digest}.json"
        )

    def _execute_sync(self, query, values, *, conditional=False):
        if conditional or not query.startswith("SELECT "):
            raise ValueError("checkpoint observer only accepts read queries")
        statement = SimpleStatement(
            query,
            consistency_level=ConsistencyLevel.LOCAL_QUORUM,
            serial_consistency_level=(
                ConsistencyLevel.LOCAL_SERIAL if conditional else None
            ),
            retry_policy=FallthroughRetryPolicy(),
            is_idempotent=False,
        )
        # Result iteration may fetch pages; keep all driver I/O off the event loop.
        return list(self.session.execute(statement, values, timeout=self.timeout))

    async def _execute(self, query, values, *, conditional=False):
        return await asyncio.to_thread(
            self._execute_sync, query, values, conditional=conditional
        )

    def _read_bytes(self, ref):
        response = self.s3.get_object(Bucket=self.bucket, Key=ref.key)
        stream = response["Body"]
        try:
            if response.get("ContentLength", 0) > MAX_CHECKPOINT_BYTES:
                raise CheckpointIntegrityError("checkpoint exceeds byte limit")
            parts, size = [], 0
            while True:
                chunk = stream.read(min(1024 * 1024, MAX_CHECKPOINT_BYTES + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_CHECKPOINT_BYTES:
                    raise CheckpointIntegrityError("checkpoint exceeds byte limit")
                parts.append(chunk)
            data = b"".join(parts)
            if hashlib.sha256(data).hexdigest() != ref.sha256:
                raise CheckpointIntegrityError("checkpoint checksum mismatch")
            return data
        finally:
            stream.close()

    async def _decode(self, step_id, row):
        revision = row.record_revision
        if type(revision) is not int or revision < 1:
            raise CheckpointIntegrityError("invalid checkpoint revision")
        ref = PrivateArtifactRef(
            bucket=self.bucket, key=row.artifact_key, sha256=row.artifact_sha256
        )
        if ref.key != self._key(step_id, revision, ref.sha256):
            raise CheckpointIntegrityError("checkpoint pointer outside registered run")
        data = await asyncio.to_thread(self._read_bytes, ref)
        record = StepCheckpointV1.model_validate(strict_json(data))
        if (
            record.step_id != step_id
            or record.progress.run_id != self.run.spec.run_id
            or record.revision != revision
            or record.input_sha256 != row.input_sha256
        ):
            raise CheckpointIntegrityError("checkpoint payload differs from pointer")
        return record

    async def read(self, step_id):
        step_id = UUID(str(step_id))
        await self.guard()
        rows = await self._execute(
            f"SELECT record_revision, artifact_key, artifact_sha256, input_sha256 FROM {self.target} "
            "WHERE run_id=%s AND step_id=%s AND revision=0",
            [self.run.spec.run_id, step_id],
        )
        if not rows:
            retained = await self._execute(
                f"SELECT revision FROM {self.target} WHERE run_id=%s AND step_id=%s AND revision>0 LIMIT 1",
                [self.run.spec.run_id, step_id],
            )
            if retained:
                # A concurrent first commit may land between the two reads.
                # Re-read its head before distinguishing corruption from a race.
                rows = await self._execute(
                    f"SELECT record_revision,artifact_key,artifact_sha256,input_sha256 FROM {self.target} "
                    "WHERE run_id=%s AND step_id=%s AND revision=0",
                    [self.run.spec.run_id, step_id],
                )
                if not rows:
                    raise CheckpointIntegrityError(
                        "retained history has no checkpoint head"
                    )
        if len(rows) > 1:
            raise CheckpointIntegrityError("multiple checkpoint heads")
        result = None if not rows else await self._decode(step_id, rows[0])
        await self.guard()
        return result

    async def history_page(self, step_id, *, after_revision=0, limit=100):
        if (
            type(after_revision) is not int
            or after_revision < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("invalid checkpoint history page")
        step_id = UUID(str(step_id))
        head = await self.read(step_id)
        if head is None:
            if after_revision:
                raise CheckpointIntegrityError("history cursor has no head")
            return (), None
        if after_revision > head.revision:
            raise CheckpointIntegrityError("history cursor exceeds head")
        rows = await self._execute(
            f"SELECT revision,record_revision,artifact_key,artifact_sha256,input_sha256 FROM {self.target} "
            "WHERE run_id=%s AND step_id=%s AND revision>%s AND revision<=%s LIMIT %s",
            [self.run.spec.run_id, step_id, after_revision, head.revision, limit],
        )
        expected = list(
            range(after_revision + 1, min(head.revision, after_revision + limit) + 1)
        )
        if [row.revision for row in rows] != expected or any(
            row.revision != row.record_revision for row in rows
        ):
            raise CheckpointIntegrityError("committed checkpoint history has a gap")
        result = tuple([await self._decode(step_id, row) for row in rows])
        if any(r.input_sha256 != head.input_sha256 for r in result):
            raise CheckpointIntegrityError("history inputs differ from head")
        await self.guard()
        cursor = (
            result[-1].revision
            if result and result[-1].revision < head.revision
            else None
        )
        return result, cursor
