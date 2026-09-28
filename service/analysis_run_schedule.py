"""Opt-in new-run discovery using isolated existing Keyspaces index partitions.

Discovery is a hint. Immutable registration/current authority and lifecycle CAS
still own dispatch. No auto-resume, new admission, expiry deletion or schema DDL.
"""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from cassandra import ConsistencyLevel
from cassandra.policies import FallthroughRetryPolicy
from cassandra.query import SimpleStatement
from shared_lib.infrastructure.dag_admission_relay import run_io

from service.analysis_checkpoint_journal import _target

INDEX_STATE = "POSITION_NEW_RUN_V1"


def _stamp(value):
    if not isinstance(value, datetime):
        raise ValueError("timestamp required")
    # Native CQL timestamps return naive UTC.
    value = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


@dataclass(frozen=True)
class ScheduleCursor:
    timestamp: datetime
    job_id: str


@dataclass(frozen=True)
class ScheduledAttempt:
    run_id: UUID | None
    status: str


@dataclass(frozen=True)
class SchedulePage:
    shard: int
    attempts: tuple[ScheduledAttempt, ...]
    next_cursor: ScheduleCursor | None
    error_code: str | None = None


class PositionRunSchedule:
    index_state = INDEX_STATE

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self.index = _target(dispatcher.keyspace, "job_recovery_index")
        self.lifecycle = _target(dispatcher.keyspace, "job_lifecycle")

    async def _execute(self, query, values, *, conditional=False):
        statement = SimpleStatement(
            query,
            consistency_level=ConsistencyLevel.LOCAL_QUORUM,
            serial_consistency_level=(
                ConsistencyLevel.LOCAL_SERIAL if conditional else None
            ),
            retry_policy=FallthroughRetryPolicy(),
            is_idempotent=False,
        )

        def execute():
            return list(
                self.dispatcher.session.execute(
                    statement, values, timeout=self.dispatcher.checkpoint_timeout
                )
            )

        return await run_io(execute)

    @staticmethod
    def _bucket(shard):
        if type(shard) is not int or not 0 <= shard < 16:
            raise ValueError("bounded shard required")
        return f"position-new-v1#{shard:02d}"

    def _key(self, run):
        return [
            self.index_state,
            self._bucket(run.spec.run_id.int % 16),
            _stamp(run.registered_at),
            str(run.spec.run_id),
        ]

    async def _run(self, run_id):
        run = await self.dispatcher.proposals.resolve_delivery_run(run_id)
        self.dispatcher._version(run)
        await self.dispatcher._authority(run)
        return run

    @staticmethod
    def _validate_candidate(row, run):
        if (
            row.job_id != str(run.spec.run_id)
            or _stamp(row.last_heartbeat_at) != _stamp(run.registered_at)
            or row.video_id != run.spec.video_id
            or row.job_state != "PENDING"
            or row.owner_instance_id != ""
        ):
            raise ValueError("discovery entry differs from registered run")

    async def enqueue(self, run_id):
        """Trusted admission handoff: confirm durable discoverability, not execution.

        Retry a lost acknowledgment with the same registered UUID. No lifecycle row
        is needed yet: discovery survives a crash before dispatcher registration.
        """
        run = await self._run(UUID(str(run_id)))
        key = self._key(run)
        rows = await self._execute(
            f"INSERT INTO {self.index} (recovery_state,heartbeat_bucket,last_heartbeat_at,job_id,"
            "owner_instance_id,video_id,job_state,updated_at) VALUES (%s,%s,%s,%s,'',%s,'PENDING',%s) IF NOT EXISTS",
            key + [run.spec.video_id, _stamp(run.registered_at)],
            conditional=True,
        )
        if len(rows) != 1 or type(getattr(rows[0], "applied", None)) is not bool:
            raise RuntimeError("discovery commit unconfirmed")
        saved = await self._execute(
            f"SELECT * FROM {self.index} WHERE recovery_state=%s AND heartbeat_bucket=%s AND last_heartbeat_at=%s AND job_id=%s",
            key,
        )
        if len(saved) != 1:
            # A concurrent scheduler may have completed/removed this entry; caller
            # must reconcile, never infer that an absent read is durable admission.
            raise RuntimeError("discovery entry not confirmed")
        self._validate_candidate(saved[0], run)
        await self.dispatcher._authority(run)
        return run.spec.run_id

    async def _state(self, run):
        rows = await self._execute(
            f"SELECT * FROM {self.lifecycle} WHERE job_id=%s", [str(run.spec.run_id)]
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("ambiguous lifecycle")
        row = rows[0]
        if (row.video_id, row.user_id, row.pipeline_kind) != (
            run.spec.video_id,
            run.spec.owner.principal_id,
            run.spec.workflow_id,
        ):
            raise ValueError("lifecycle identity mismatch")
        return row

    async def _retire_completed(self, run):
        await self.dispatcher._authority(run)
        state = await self._state(run)
        if state is None or state.job_state != "COMPLETED":
            raise RuntimeError("completion not durable")
        await self._execute(
            f"DELETE FROM {self.index} WHERE recovery_state=%s AND heartbeat_bucket=%s AND last_heartbeat_at=%s AND job_id=%s",
            self._key(run),
        )
        await self.dispatcher._authority(run)

    async def _process(self, shard, row):
        run_id = UUID(row.job_id)
        if run_id.int % 16 != shard:
            raise ValueError("run in wrong shard")
        run = await self._run(run_id)
        self._validate_candidate(row, run)
        state = await self._state(run)
        if state is not None and state.job_state == "COMPLETED":
            await self._retire_completed(run)
            return ScheduledAttempt(run_id, "completed")
        if state is not None and state.job_state != "PENDING":
            if state.job_state == "RUNNING":
                expires = _stamp(state.last_heartbeat_at) + timedelta(
                    seconds=self.dispatcher.lease_seconds
                )
                status = (
                    "running"
                    if datetime.now(timezone.utc) < expires
                    else "explicit_recovery_required"
                )
            elif state.job_state in {
                "INTERRUPTED",
                "FAILED",
                "CANCELLED",
                "AWAITING_CORRECTION",
            }:
                status = "explicit_recovery_required"
            else:
                raise ValueError("unknown lifecycle state")
            return ScheduledAttempt(run_id, status)
        # Never pass recover/retry_failed here. A competing scheduler still must
        # win the lifecycle claim, and expired work is never silently reclaimed.
        await self.dispatcher.dispatch(run_id)
        await self._retire_completed(run)
        return ScheduledAttempt(run_id, "completed")

    async def process_page(self, shard, *, after=None, limit=10, stop=None):
        bucket = self._bucket(shard)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("bounded page size required")
        values = [self.index_state, bucket]
        condition = ""
        if after is not None:
            if not isinstance(after, ScheduleCursor) or not isinstance(
                after.job_id, str
            ):
                raise ValueError("internal schedule cursor required")
            condition = " AND (last_heartbeat_at,job_id)>(%s,%s)"
            values += [_stamp(after.timestamp), after.job_id]
        try:
            rows = await self._execute(
                f"SELECT * FROM {self.index} WHERE recovery_state=%s AND heartbeat_bucket=%s{condition} LIMIT %s",
                values + [limit + 1],
            )
        except Exception:
            return SchedulePage(shard, (), after, "discovery_unavailable")
        attempts = []
        for row in rows[:limit]:
            if stop is not None and stop.is_set():
                return SchedulePage(shard, tuple(attempts), after)
            try:
                result = await self._process(shard, row)
            except Exception:
                try:
                    run_id = UUID(row.job_id)
                except (ValueError, TypeError, AttributeError):
                    run_id = None
                result = ScheduledAttempt(run_id, "unconfirmed")
            attempts.append(result)
        cursor = (
            ScheduleCursor(
                _stamp(rows[limit - 1].last_heartbeat_at), rows[limit - 1].job_id
            )
            if len(rows) > limit
            else None
        )
        return SchedulePage(shard, tuple(attempts), cursor)

    async def run(self, stop, *, observe, interval_seconds, page_size=10):
        if (
            not callable(observe)
            or not math.isfinite(interval_seconds)
            or not 0 < interval_seconds <= 60
        ):
            raise ValueError("observer and bounded polling interval required")
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise ValueError("bounded page size required")
        cursors = {shard: None for shard in range(16)}
        while not stop.is_set():
            for shard in range(16):
                if stop.is_set():
                    return
                page = await self.process_page(
                    shard, after=cursors[shard], limit=page_size, stop=stop
                )
                await observe(page)
                cursors[shard] = page.next_cursor
            try:
                await asyncio.wait_for(stop.wait(), interval_seconds)
            except TimeoutError:
                pass


class OwnerPositionRunSchedule(PositionRunSchedule):
    """Explicit approved owner lane; never reads the older unpartitioned hints."""

    def __init__(self, dispatcher, *, owner_kind):
        if owner_kind not in {"account", "session"}:
            raise ValueError("explicit account/session scheduling lane required")
        super().__init__(dispatcher)
        self.owner_kind = owner_kind
        self.index_state = f"POSITION_NEW_RUN_{owner_kind.upper()}_V1"

    def _bucket(self, shard):
        super()._bucket(shard)  # Preserve strict bounded-shard validation.
        return f"position-new-{self.owner_kind}-v1#{shard:02d}"

    async def _run(self, run_id):
        run = await self.dispatcher.proposals.resolve_delivery_run(run_id)
        if run.spec.owner.kind != self.owner_kind:
            raise PermissionError("registered owner differs from scheduling lane")
        self.dispatcher._version(run)
        await self.dispatcher._authority(run)
        return run
