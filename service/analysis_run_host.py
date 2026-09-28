"""Opt-in durable Gemini run coordination, not an active worker/scheduler.

Uses the existing Keyspaces checkpoint journal. External operations are NEVER
reclaimed through StepRunner.deterministic's replay-safe calculation policy.
"""

import hashlib
from datetime import datetime, timezone
from uuid import uuid5

from bjj_analysis_core.execution import (
    CheckpointConflict,
    RecoveryRequired,
    StepCheckpointV1,
)
from bjj_analysis_core.executor import ExecutionResultV1, execute_pipeline
from bjj_analysis_core.requests import canonical_json, load_config
from pydantic import AwareDatetime, TypeAdapter
from shared_lib.models.analysis_contracts import (
    CallUsageV1,
    NodeProgressV1,
    PrivateArtifactRef,
)

from service.analysis_gemini import UploadedVideo


class DurableGeminiRun:
    def __init__(
        self,
        run,
        config_bytes,
        *,
        journal,
        provider,
        sink,
        guard,
        mime_type,
        clock=None,
    ):
        self.run, self.config = load_config(run, config_bytes)
        if journal.run != self.run or provider.run != self.run or not callable(guard):
            raise ValueError(
                "matching run ports and live authority/lease guard required"
            )
        self.config_bytes, self.journal, self.provider = config_bytes, journal, provider
        self.sink, self.guard, self.mime_type = sink, guard, mime_type
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self):
        now = TypeAdapter(AwareDatetime).validate_python(self.clock())
        if now < self.run.registered_at:
            raise ValueError("host checkpoint time precedes registration")
        return now

    def _identity(self, node):
        if node not in {"provider_upload", "provider_cleanup", "dag_complete"}:
            raise ValueError("unknown host operation")
        step = uuid5(self.run.spec.run_id, f"host:v1:{node}")
        digest = hashlib.sha256(
            canonical_json(
                dict(
                    host_version=1,
                    run=self.run.model_dump(mode="json"),
                    provider=self.provider.name,
                    mime_type=self.mime_type,
                    operation=node,
                )
            ).encode()
        ).hexdigest()
        return step, digest

    async def _read(self, node):
        await self.guard()
        step, digest = self._identity(node)
        record = await self.journal.read(step)
        if record is not None:
            record = StepCheckpointV1.model_validate(record)
            if (
                record.step_id != step
                or record.input_sha256 != digest
                or record.progress.node_id != node
                or record.request is not None
            ):
                raise RecoveryRequired("host checkpoint identity changed")
        await self.guard()
        return record

    def _reservation(self, node, previous):
        step, digest = self._identity(node)
        attempt = 1 if previous is None else previous.progress.attempt + 1
        revision = 1 if previous is None else previous.revision + 1
        now = self._now()
        return StepCheckpointV1(
            step_id=step,
            input_sha256=digest,
            revision=revision,
            phase="running",
            request=None,
            progress=NodeProgressV1(
                run_id=self.run.spec.run_id,
                execution_id=step,
                node_id=node,
                attempt=attempt,
                sequence=revision,
                status="running",
                started_at=now,
                updated_at=now,
                total_items=1,
            ),
            usage=CallUsageV1(
                run_id=self.run.spec.run_id,
                execution_id=step,
                call_id=uuid5(step, f"attempt:{attempt}"),
                recorded_at=now,
                kind="deterministic",
                usage_status="not_applicable",
                cost_status="not_applicable",
                cost_nano_usd=0,
            ),
        )

    async def _once(self, node, result_type, operation, *, replay_safe=False):
        previous = await self._read(node)
        adapter = TypeAdapter(result_type)
        if previous is not None and previous.phase == "succeeded":
            return adapter.validate_json(previous.result_json)
        if previous is not None and not replay_safe:
            raise RecoveryRequired(
                f"{node} outcome uncertain; reconcile before another external action"
            )
        record = self._reservation(node, previous)
        expected = None if previous is None else previous.revision
        if not await self.journal.compare_and_swap(expected, record):
            raise CheckpointConflict("another worker reserved host operation")
        await self.guard()  # includes revocation during the reservation commit
        # Any error/cancellation leaves the running reservation intentionally.
        # Never guess whether upload/delete reached the remote service.
        result = adapter.validate_python(await operation(record.usage.call_id))
        return await self._commit_result(record, result_type, result)

    async def _commit_result(self, record, result_type, result):
        await self.guard()
        adapter = TypeAdapter(result_type)
        result = adapter.validate_python(result)
        finished = self._now()
        progress = record.progress.model_dump()
        progress.update(
            sequence=record.revision + 1,
            status="succeeded",
            completed_items=1,
            updated_at=finished,
            finished_at=finished,
        )
        complete = StepCheckpointV1.model_validate(
            {
                **record.model_dump(),
                "revision": record.revision + 1,
                "phase": "succeeded",
                "progress": progress,
                "result_json": adapter.dump_json(result).decode(),
            }
        )
        if not await self.journal.compare_and_swap(record.revision, complete):
            raise CheckpointConflict("host operation result could not be committed")
        await self.guard()
        return result

    async def reconcile_upload(self, receipt):
        """Trusted explicit recovery under a live guard, not automatic retry."""
        if await self._read("provider_cleanup") is not None:
            raise RecoveryRequired("cleanup began; cannot recover upload")
        record = await self._read("provider_upload")
        if record is None or record.phase not in {"running", "succeeded"}:
            raise RecoveryRequired("upload reservation required")
        if record.phase == "succeeded":
            return await self.ensure_upload()
        result = await self.provider.reconcile_upload(
            record.usage.call_id,
            receipt,
            mime_type=self.mime_type,
            not_before=record.progress.started_at,
        )
        result = await self._commit_result(record, UploadedVideo, result)
        self.provider.bind_upload(result)
        await self.guard()
        return result

    async def reconcile_cleanup(self, receipt):
        complete = await self._read("dag_complete")
        if complete is None or complete.phase != "succeeded":
            raise RecoveryRequired("cleanup requires a durably completed DAG")
        TypeAdapter(ExecutionResultV1).validate_json(complete.result_json)
        record = await self._read("provider_cleanup")
        if record is None or record.phase not in {"running", "succeeded"}:
            raise RecoveryRequired("cleanup reservation required")
        if record.phase == "succeeded":
            return TypeAdapter(PrivateArtifactRef).validate_json(record.result_json)
        upload = await self._read("provider_upload")
        if upload is None or upload.phase != "succeeded":
            raise RecoveryRequired("cleanup has no committed upload target")
        self.provider.bind_upload(
            TypeAdapter(UploadedVideo).validate_json(upload.result_json)
        )
        result = await self.provider.reconcile_cleanup(
            record.usage.call_id,
            receipt,
            not_before=record.progress.started_at,
        )
        result = await self._commit_result(record, PrivateArtifactRef, result)
        self.provider.uploaded = None
        await self.guard()
        return result

    async def ensure_upload(self):
        if await self._read("provider_cleanup") is not None:
            raise RecoveryRequired(
                "cleanup began; this run cannot acquire provider media again"
            )

        async def upload(operation_id):
            return await self.provider.upload(operation_id, mime_type=self.mime_type)

        result = await self._once("provider_upload", UploadedVideo, upload)
        self.provider.bind_upload(result)
        await self.guard()
        return result

    async def execute(self, *, retry_failed=False):
        complete = await self._read("dag_complete")
        if complete is not None and complete.phase == "succeeded":
            # A deleted provider file must not force a new upload or inference
            # when the complete DAG result was already durably acknowledged.
            return TypeAdapter(ExecutionResultV1).validate_json(complete.result_json)
        await self.ensure_upload()
        result = await execute_pipeline(
            self.run,
            self.config_bytes,
            journal=self.journal,
            provider=self.provider,
            sink=self.sink,
            guard=self.guard,
            retry_failed=retry_failed,
            clock=self.clock,
        )

        async def retain_result(operation_id):
            return result

        # Only this pure result marker may be reconstructed after interruption.
        # Core step checkpoints still control any model work during resume.
        return await self._once(
            "dag_complete", ExecutionResultV1, retain_result, replay_safe=True
        )

    async def cleanup(self):
        complete = await self._read("dag_complete")
        if complete is None or complete.phase != "succeeded":
            raise RecoveryRequired("cleanup requires a durably completed DAG")
        TypeAdapter(ExecutionResultV1).validate_json(complete.result_json)
        upload = await self._read("provider_upload")
        if upload is None or upload.phase != "succeeded":
            raise RecoveryRequired("cleanup has no committed upload target")
        self.provider.bind_upload(
            TypeAdapter(UploadedVideo).validate_json(upload.result_json)
        )

        async def remove(operation_id):
            return await self.provider.delete_upload(operation_id)

        return await self._once("provider_cleanup", PrivateArtifactRef, remove)
