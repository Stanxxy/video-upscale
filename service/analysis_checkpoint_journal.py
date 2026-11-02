"""Engine-owned checkpoint writes; shared observation logic is read-only."""

import asyncio
import hashlib
import logging

from bjj_analysis_core.execution import StepCheckpointV1
from bjj_analysis_observer.checkpoints import (
    MAX_CHECKPOINT_BYTES,
    CheckpointIntegrityError,
    KeyspacesCheckpointReader,
    _target,
)
from botocore.exceptions import ClientError
from cassandra import ConsistencyLevel
from cassandra.policies import FallthroughRetryPolicy
from cassandra.query import SimpleStatement
from shared_lib.models.analysis_contracts import PrivateArtifactRef

LOG = logging.getLogger(__name__)
# Logged once per process: evidence of which outcome protocol the store speaks.
_implicit_outcome_logged = False


def checkpoint_table_ddl(keyspace, table="analysis_execution_checkpoints"):
    target = _target(keyspace, table)
    return f"""CREATE TABLE IF NOT EXISTS {target} (
        run_id uuid, step_id uuid, revision bigint, record_revision bigint,
        artifact_key text, artifact_sha256 text, input_sha256 text,
        PRIMARY KEY ((run_id, step_id), revision)
    ) WITH CLUSTERING ORDER BY (revision ASC)"""


class KeyspacesCheckpointJournal(KeyspacesCheckpointReader):
    def _execute_sync(self, query, values, *, conditional=False):
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

    def _put_bytes(self, checkpoint):
        data = checkpoint.model_dump_json().encode()
        if len(data) > MAX_CHECKPOINT_BYTES:
            raise CheckpointIntegrityError("checkpoint exceeds byte limit")
        digest = hashlib.sha256(data).hexdigest()
        ref = PrivateArtifactRef(
            bucket=self.bucket,
            key=self._key(checkpoint.step_id, checkpoint.revision, digest),
            sha256=digest,
        )
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=ref.key,
                Body=data,
                ContentType="application/json",
                ACL="private",
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            # Exact immutable-object reuse, not recovery by assuming success.
            if self._read_bytes(ref) != data:
                raise CheckpointIntegrityError(
                    "immutable checkpoint bytes changed"
                ) from error
        return ref

    async def compare_and_swap(self, expected_revision, checkpoint):
        record = StepCheckpointV1.model_validate(checkpoint)
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 1
        ):
            raise ValueError("expected revision must be null or positive integer")
        if (
            record.revision != (expected_revision or 0) + 1
            or record.progress.run_id != self.run.spec.run_id
        ):
            raise CheckpointIntegrityError("invalid checkpoint run/revision")
        await self.guard()
        # Guard input/identity continuity as well as the head's CAS revision.
        previous = await self.read(record.step_id)
        if (None if previous is None else previous.revision) != expected_revision:
            return False
        if previous is not None and record.input_sha256 != previous.input_sha256:
            raise CheckpointIntegrityError("checkpoint inputs changed")
        ref = await asyncio.to_thread(self._put_bytes, record)
        await self.guard()
        columns = "run_id,step_id,revision,record_revision,artifact_key,artifact_sha256,input_sha256"
        history = f"INSERT INTO {self.target} ({columns}) VALUES (%s,%s,%s,%s,%s,%s,%s) IF NOT EXISTS;"
        values = [
            self.run.spec.run_id,
            record.step_id,
            record.revision,
            record.revision,
            ref.key,
            ref.sha256,
            record.input_sha256,
        ]
        if previous is None:
            head = f"INSERT INTO {self.target} ({columns}) VALUES (%s,%s,0,%s,%s,%s,%s) IF NOT EXISTS;"
            values += [
                self.run.spec.run_id,
                record.step_id,
                record.revision,
                ref.key,
                ref.sha256,
                record.input_sha256,
            ]
        else:
            head = (
                f"UPDATE {self.target} SET record_revision=%s,artifact_key=%s,artifact_sha256=%s,input_sha256=%s "
                "WHERE run_id=%s AND step_id=%s AND revision=0 IF record_revision=%s AND input_sha256=%s;"
            )
            values += [
                record.revision,
                ref.key,
                ref.sha256,
                record.input_sha256,
                self.run.spec.run_id,
                record.step_id,
                expected_revision,
                record.input_sha256,
            ]
        rows = await self._execute(
            f"BEGIN BATCH {history} {head} APPLY BATCH", values, conditional=True
        )
        # A lease may be revoked while the driver is blocked. The committed audit
        # row remains, but a denied return must not lead straight to a paid call.
        await self.guard()
        # Failed conditional batches may return multiple current rows. An
        # explicit homogeneous boolean result is success or ordinary contention.
        flags = [getattr(row, "applied", None) for row in rows]
        if flags and all(type(flag) is bool for flag in flags):
            if len(set(flags)) != 1:
                raise CheckpointIntegrityError("ambiguous conditional-write outcome")
            return flags[0]
        if any(flag is not None for flag in flags):
            raise CheckpointIntegrityError("malformed conditional-write outcome")
        # Amazon Keyspaces applies a conditional logged batch without returning
        # Cassandra's [applied] row. The committed head is then the outcome.
        return await self._committed_outcome(record, ref, len(rows))

    async def _committed_outcome(self, record, ref, returned_rows):
        """Decide a conditional write the store did not report, from its committed head.

        Ours: the head points at exactly this revision's immutable artifact.
        Lost race: a competing writer already committed this revision or later.
        Anything else means the store neither reported nor committed the write.
        """
        global _implicit_outcome_logged
        heads = await self._execute(
            f"SELECT record_revision, artifact_key, artifact_sha256 FROM {self.target} "
            "WHERE run_id=%s AND step_id=%s AND revision=0",
            [self.run.spec.run_id, record.step_id],
        )
        await self.guard()
        if len(heads) > 1:
            raise CheckpointIntegrityError("multiple checkpoint heads")
        head = heads[0] if heads else None
        current = None if head is None else head.record_revision
        if current is not None and type(current) is not int:
            raise CheckpointIntegrityError("invalid checkpoint revision")
        if not _implicit_outcome_logged:
            _implicit_outcome_logged = True
            LOG.info(
                "conditional checkpoint batch returned %d rows without [applied]; "
                "outcomes are read from the committed head",
                returned_rows,
            )
        if current == record.revision and (head.artifact_key, head.artifact_sha256) == (
            ref.key,
            ref.sha256,
        ):
            return True
        if current is not None and current >= record.revision:
            return False
        raise CheckpointIntegrityError(
            "conditional write returned no outcome and did not commit"
        )
