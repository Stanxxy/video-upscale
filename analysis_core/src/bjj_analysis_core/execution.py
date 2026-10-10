"""Checkpoint-aware step execution; host I/O is explicit, never imported.

This is not a production storage adapter. Journal implementations MUST provide
durable strong reads and atomic CAS + retained history, privately externalizing
large payloads. A plain overwriting stage snapshot is not a conforming journal.
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import UUID, uuid5

from bjj_analysis_core.requests import (
    PreparedCallV1,
    canonical_json,
    load_config,
    parse_response,
    prepare_call,
    prompt_bundle,
    strict_json,
)
from pydantic import AwareDatetime, StrictBool, StrictStr, TypeAdapter, model_validator
from shared_lib.models.analysis_contracts import (
    CallUsageV1,
    Identifier,
    NodeProgressV1,
    PositiveInt,
    Sha256,
    VersionedAnalysisContract,
)
from shared_lib.models.analysis_run_record import RegisteredAIRunV1


class ProviderReplyV1(VersionedAnalysisContract):
    text: StrictStr  # Malformed/empty JSON must still retain usage before parsing.
    usage: CallUsageV1


class StepCheckpointV1(VersionedAnalysisContract):
    step_id: UUID
    input_sha256: Sha256
    revision: PositiveInt
    phase: Literal["running", "responded", "succeeded", "failed", "cancelled"]
    progress: NodeProgressV1
    usage: CallUsageV1
    request: PreparedCallV1 | None
    reply: ProviderReplyV1 | None = None
    result_json: StrictStr | None = None
    retryable: StrictBool = False

    @model_validator(mode="after")
    def consistent_checkpoint(self):
        expected = "running" if self.phase == "responded" else self.phase
        if self.progress.status != expected or self.progress.sequence != self.revision:
            raise ValueError("checkpoint phase/revision disagrees with progress")
        if self.progress.execution_id != self.step_id:
            raise ValueError("checkpoint execution identity differs")
        if (
            self.usage.run_id != self.progress.run_id
            or self.usage.execution_id != self.step_id
        ):
            raise ValueError("checkpoint usage identity differs")
        if self.usage.call_id != uuid5(
            self.step_id, f"attempt:{self.progress.attempt}"
        ):
            raise ValueError("checkpoint usage attempt differs")
        if (self.request is None) != (self.usage.kind == "deterministic"):
            raise ValueError("checkpoint kind differs from request")
        if self.phase == "responded" and self.reply is None:
            raise ValueError("responded checkpoint requires raw reply")
        if self.phase == "succeeded" and self.result_json is None:
            raise ValueError("success needs retained result")
        if self.retryable and self.phase != "failed":
            raise ValueError("only known failures can be retryable")
        return self


class CheckpointJournal(Protocol):
    async def read(self, step_id: UUID) -> StepCheckpointV1 | None: ...

    async def compare_and_swap(
        self, expected_revision: int | None, checkpoint: StepCheckpointV1
    ) -> bool:
        """Atomically replace expected head AND append immutable revision history.

        Return false on contention, propagate unavailable/unknown commit outcomes.
        Retain all attempts and replies; do not silently expire active run history.
        """
        ...


class Provider(Protocol):
    name: str

    async def generate(
        self,
        run: RegisteredAIRunV1,
        request: PreparedCallV1,
        execution_id: UUID,
        call_id: UUID,
    ) -> ProviderReplyV1:
        """One dispatch only; no hidden retries. Verify source/crop bytes first.

        Translate provider media clocks to the declared absolute-ms contract.
        Bind actual usage to supplied IDs; preserve unknown cost as null. A real
        adapter must be qualified before deployment; this protocol is not one.
        """
        ...


class CheckpointConflict(RuntimeError):
    pass


class RecoveryRequired(RuntimeError):
    pass


class ExecutionCancelled(RuntimeError):
    pass


class NotDispatched(RuntimeError):
    """Adapter attests that no provider request left this process."""


class StepRunner:
    def __init__(
        self,
        run,
        config_bytes,
        journal,
        provider,
        *,
        retry_failed=False,
        reissue_unknown=False,
        guard=None,
        clock=None,
    ):
        self.run, self.config = load_config(run, config_bytes)
        self.config_bytes, self.journal, self.provider = config_bytes, journal, provider
        self.retry_failed, self.guard = retry_failed, guard
        # Owner-authorized resume only: a model call whose answer never arrived is
        # sent again (a possible duplicate charge), bounded by max_call_attempts.
        self.reissue_unknown = reissue_unknown
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def now(self):
        now = TypeAdapter(AwareDatetime).validate_python(self.clock())
        if now < self.run.registered_at:
            raise ValueError("execution time precedes registration")
        return now

    async def _save(self, previous, record):
        record = StepCheckpointV1.model_validate(record)
        expected = None if previous is None else previous.revision
        if not await self.journal.compare_and_swap(expected, record):
            raise CheckpointConflict("checkpoint changed concurrently")
        return record

    async def _begin(self, node, item, request, inputs, total):
        if self.guard is not None:
            await self.guard()  # host checks current authority, cancellation/lease
        step_id = uuid5(self.run.spec.run_id, f"step:v1:{node}:{item or ''}")
        digest = hashlib.sha256(
            canonical_json(
                dict(
                    run=self.run.model_dump(mode="json"),
                    inputs=inputs,
                    request=(
                        None if request is None else request.model_dump(mode="json")
                    ),
                    provider=None if request is None else self.provider.name,
                    total=total,
                )
            ).encode()
        ).hexdigest()
        previous = await self.journal.read(step_id)
        if previous is not None:
            previous = StepCheckpointV1.model_validate(previous)
            if previous.step_id != step_id or previous.input_sha256 != digest:
                raise RecoveryRequired(
                    "checkpoint inputs changed; never reinterpret old work"
                )
            if previous.phase in ("succeeded", "responded"):
                return previous
            if previous.phase == "running" and request is not None and (
                not self.reissue_unknown
                or previous.progress.attempt >= self.config.max_call_attempts
            ):
                raise RecoveryRequired(
                    "provider dispatch outcome unknown; reconcile before retry"
                )
            # An owner resume may also re-send a call whose dispatch raised with an
            # unknown outcome (DISPATCH_UNCERTAIN): same risk as an unanswered call.
            unknown_outcome = (
                self.reissue_unknown
                and request is not None
                and previous.phase == "failed"
                and previous.progress.error_code == "DISPATCH_UNCERTAIN"
            )
            if previous.phase in ("failed", "cancelled") and (
                not (unknown_outcome or (self.retry_failed and previous.retryable))
                or previous.progress.attempt >= self.config.max_call_attempts
            ):
                raise RecoveryRequired(
                    "failed/cancelled work needs authorized bounded recovery"
                )
            # Only pure/idempotent deterministic work can reclaim an in-flight
            # step without provider reconciliation. CAS fences the old finisher.
        now = self.now()
        attempt = 1 if previous is None else previous.progress.attempt + 1
        revision = 1 if previous is None else previous.revision + 1
        usage = CallUsageV1(
            run_id=self.run.spec.run_id,
            execution_id=step_id,
            call_id=uuid5(step_id, f"attempt:{attempt}"),
            recorded_at=now,
            kind="deterministic" if request is None else "model",
            usage_status="not_applicable" if request is None else "unknown",
            cost_status="not_applicable" if request is None else "unknown",
            cost_nano_usd=0 if request is None else None,
            provider=None if request is None else self.provider.name,
            model=None if request is None else request.config.model,
        )
        progress = NodeProgressV1(
            run_id=self.run.spec.run_id,
            execution_id=step_id,
            node_id=node,
            item_id=item,
            attempt=attempt,
            sequence=revision,
            status="running",
            started_at=now,
            updated_at=now,
            total_items=total,
        )
        return await self._save(
            previous,
            StepCheckpointV1(
                step_id=step_id,
                input_sha256=digest,
                revision=revision,
                phase="running",
                progress=progress,
                usage=usage,
                request=request,
            ),
        )

    async def _update(self, record, phase, **changes):
        now = self.now()
        terminal = phase in ("succeeded", "failed", "cancelled")
        revision = record.revision + 1
        progress = NodeProgressV1.model_validate(
            {
                **record.progress.model_dump(),
                "sequence": revision,
                "status": "running" if phase == "responded" else phase,
                "updated_at": now,
                "finished_at": now if terminal else None,
                "completed_items": (
                    record.progress.total_items if phase == "succeeded" else 0
                ),
                "error_code": changes.pop("error_code", None),
            }
        )
        return await self._save(
            record,
            {
                **record.model_dump(),
                "revision": revision,
                "phase": phase,
                "progress": progress,
                **changes,
            },
        )

    async def model(self, request, *, item=None):
        request = PreparedCallV1.model_validate(request)
        bundle = prompt_bundle()
        initial = strict_json(request.initial_prompt)
        # Re-run the request builder's stage/window/context checks. Advisory
        # content is trusted persisted upstream output, not browser parameters.
        try:
            rebuilt = prepare_call(
                self.run,
                self.config_bytes,
                request.node_id,
                input_window=request.input_window,
                context=initial["stage_context"],
            )
        except (KeyError, ValueError, TypeError) as error:
            raise RecoveryRequired("prepared stage input is invalid") from error
        expected_initial = strict_json(rebuilt.initial_prompt)
        actual_initial = dict(initial)
        advisory = actual_initial.pop("advisory_positions", None)
        expected_initial.pop("advisory_positions")
        if actual_initial != expected_initial or (
            advisory is not None
            and request.node_id
            in ("position_scan", "technique_scan", "position_critique")
        ):
            raise RecoveryRequired(
                "prepared prompt context differs from admitted stage"
            )
        if (
            request.config_sha256 != self.run.spec.effective_config.sha256
            or request.prompt_bundle_sha256 != self.config.prompt_bundle_sha256
            or request.config != getattr(self.config, request.node_id)
            or request.system_prompt != bundle["prompts"][request.node_id]
            or request.response_schema != bundle["schemas"][request.node_id]
            or not self.run.spec.scope.contains(request.input_window)
            or initial.get("scope") != self.run.spec.scope.model_dump(mode="json")
            or initial.get("observation_window")
            != request.input_window.model_dump(mode="json")
            or initial.get("source_time_unit") != "absolute_milliseconds"
            or initial.get("ordered_players")
            != [
                dict(player_id=a.player_id, display_name=a.display_name)
                for a in self.run.spec.athletes
            ]
        ):
            raise RecoveryRequired(
                "prepared call differs from admitted run/configuration"
            )
        record = await self._begin(request.node_id, item, request, {}, 1)
        if record.phase == "succeeded":
            parsed = parse_response(request, record.reply.text)
            if parsed.model_dump_json() != record.result_json:
                raise RecoveryRequired(
                    "saved model result differs from retained response"
                )
            return parsed
        if record.phase == "running":
            try:
                reply = ProviderReplyV1.model_validate(
                    await self.provider.generate(
                        self.run, request, record.step_id, record.usage.call_id
                    )
                )
            except NotDispatched:
                await self._update(
                    record, "failed", error_code="NOT_DISPATCHED", retryable=True
                )
                raise
            except asyncio.CancelledError:
                await self._update(record, "cancelled")
                raise
            except Exception:
                await self._update(record, "failed", error_code="DISPATCH_UNCERTAIN")
                raise
            # Persist raw text even if JSON/semantics are invalid. Never wrap
            # persistence failures in a provider retry; an unknown commit stops.
            record = await self._update(record, "responded", reply=reply)
        reply = record.reply
        usage = reply.usage
        if (
            usage.run_id != self.run.spec.run_id
            or usage.execution_id != record.step_id
            or usage.call_id != record.usage.call_id
            or usage.kind != "model"
            or usage.provider != self.provider.name
            or usage.model != request.config.model
            or not record.progress.started_at <= usage.recorded_at <= self.now()
        ):
            await self._update(record, "failed", error_code="USAGE_IDENTITY_MISMATCH")
            raise RecoveryRequired("provider usage does not belong to this attempt")
        try:
            parsed = parse_response(request, reply.text)
        except ValueError:
            await self._update(
                record,
                "failed",
                usage=usage,
                error_code="INVALID_MODEL_OUTPUT",
                retryable=True,
            )
            raise
        record = await self._update(
            record, "succeeded", usage=usage, result_json=parsed.model_dump_json()
        )
        return parsed

    async def deterministic(
        self, node, inputs, operation, result_type, *, item=None, total=1
    ):
        """Operation gets its persisted start time; it must be pure or idempotent."""
        record = await self._begin(node, item, None, inputs, total)
        adapter = TypeAdapter(result_type)
        if record.phase == "succeeded":
            return adapter.validate_json(record.result_json)
        try:
            result = adapter.validate_python(
                await operation(record.progress.started_at)
            )
        except asyncio.CancelledError:
            await self._update(record, "cancelled")
            raise
        except Exception:
            await self._update(
                record, "failed", error_code="DETERMINISTIC_STEP_FAILED", retryable=True
            )
            raise
        await self._update(
            record, "succeeded", result_json=adapter.dump_json(result).decode()
        )
        return result
