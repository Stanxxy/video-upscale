"""Read-only lifecycle/heartbeat composition; not a public API or atomic snapshot."""

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bjj_analysis_observer.checkpoints import _target
from bjj_analysis_observer.progress import (
    ProgressEvidenceError,
    ProgressView,
    read_progress,
)
from cassandra import ConsistencyLevel
from cassandra.policies import FallthroughRetryPolicy
from cassandra.query import SimpleStatement
from pydantic import AwareDatetime, TypeAdapter


class StatusEvidenceError(ProgressEvidenceError):
    pass


class StatusObservationChanged(StatusEvidenceError):
    """Lifecycle changed across the read; caller may explicitly request a fresh read."""


@dataclass(frozen=True)
class LifecycleView:
    recorded_state: str
    heartbeat_freshness: str
    last_heartbeat_at: datetime | None
    lease_expires_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class StatusView:
    observed_at: datetime
    lifecycle: LifecycleView
    progress: ProgressView


@dataclass(frozen=True)
class _Lifecycle:
    state: str
    owner: str
    started: datetime
    heartbeat: datetime
    updated: datetime
    completed: datetime | None

    def identity(self):
        return self.state, self.owner, self.started, self.completed


def _timestamp(value):
    if not isinstance(value, datetime):
        raise StatusEvidenceError("missing or invalid lifecycle timestamp")
    # The CQL driver returns naive UTC timestamps, at millisecond precision.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class KeyspacesLifecycleReader:
    def __init__(self, run, session, *, keyspace, timeout):
        self.run, self.session = run, session
        self.target = _target(keyspace, "job_lifecycle")
        self.timeout = timeout

    async def read(self):
        statement = SimpleStatement(
            f"SELECT * FROM {self.target} WHERE job_id=%s",
            consistency_level=ConsistencyLevel.LOCAL_QUORUM,
            retry_policy=FallthroughRetryPolicy(),
            is_idempotent=True,
        )
        rows = await asyncio.to_thread(
            lambda: list(
                self.session.execute(
                    statement, [str(self.run.spec.run_id)], timeout=self.timeout
                )
            )
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise StatusEvidenceError("ambiguous lifecycle evidence")
        row, spec = rows[0], self.run.spec
        if (
            row.video_id != spec.video_id
            or row.user_id != spec.owner.principal_id
            or row.pipeline_kind != spec.workflow_id
        ):
            raise StatusEvidenceError("lifecycle identity differs from admitted run")
        state, owner = row.job_state, row.owner_instance_id
        if state not in {"PENDING", "RUNNING", "INTERRUPTED", "COMPLETED"}:
            raise StatusEvidenceError("unsupported position lifecycle state")
        if (state == "PENDING" and owner != "") or (
            state != "PENDING"
            and (
                not isinstance(owner, str) or not owner.startswith("position-attempt:")
            )
        ):
            raise StatusEvidenceError("invalid lifecycle attempt identity")
        value = _Lifecycle(
            state,
            owner,
            _timestamp(row.started_at),
            _timestamp(row.last_heartbeat_at),
            _timestamp(row.updated_at),
            None if row.completed_at is None else _timestamp(row.completed_at),
        )
        if not value.started <= value.heartbeat <= value.updated:
            raise StatusEvidenceError("inconsistent lifecycle timestamp order")
        if (state == "COMPLETED" and value.completed != value.updated) or (
            state != "COMPLETED" and value.completed is not None
        ):
            raise StatusEvidenceError("inconsistent lifecycle completion evidence")
        return value


async def read_status(
    journal,
    *,
    session,
    keyspace,
    guard,
    lease_seconds,
    timeout=10,
    max_item_reads=2000,
    clock=None,
):
    if (
        not callable(guard)
        or not math.isfinite(lease_seconds)
        or lease_seconds <= 0
        or not math.isfinite(timeout)
        or not 0 < timeout < lease_seconds
    ):
        raise ValueError("authority and finite lease longer than I/O timeout required")
    clock = clock or (lambda: datetime.now(timezone.utc))

    def now():
        return (
            TypeAdapter(AwareDatetime).validate_python(clock()).astimezone(timezone.utc)
        )

    started = now()
    reader = KeyspacesLifecycleReader(
        journal.run, session, keyspace=keyspace, timeout=timeout
    )
    await guard()
    before = await reader.read()
    await guard()
    progress = await read_progress(journal, guard=guard, max_item_reads=max_item_reads)
    await guard()
    after = await reader.read()
    await guard()
    observed = now()  # includes all awaited authorization and checkpoint I/O
    if observed < started:
        raise StatusEvidenceError("status clock moved backwards")
    if (before is None) != (after is None) or (
        before is not None and before.identity() != after.identity()
    ):
        raise StatusObservationChanged("lifecycle changed during progress read")
    if after is not None:
        if after.heartbeat < before.heartbeat or after.updated < before.updated:
            raise StatusEvidenceError("lifecycle timestamps moved backwards")
        if before.updated > observed or after.updated > observed:
            raise StatusEvidenceError("lifecycle timestamp is in the future")
    nodes = {node.node_id: node for node in progress.nodes}
    if after is None or after.state == "PENDING":
        if any(node.checkpoints for node in progress.nodes):
            raise StatusEvidenceError("checkpoints require an attempted lifecycle")
    elif after.state == "COMPLETED":
        if nodes["dag_complete"].recorded_status != "succeeded" or nodes[
            "provider_cleanup"
        ].recorded_status not in {"not_started", "succeeded"}:
            raise StatusEvidenceError(
                "completed lifecycle lacks completed host evidence"
            )
    expires = (
        after.heartbeat + timedelta(seconds=lease_seconds)
        if after is not None and after.state == "RUNNING"
        else None
    )
    freshness = "not_running"
    if expires is not None:
        freshness = "expired" if observed >= expires else "recent"
    return StatusView(
        observed,
        LifecycleView(
            "NOT_REGISTERED" if after is None else after.state,
            freshness,
            None if after is None else after.heartbeat,
            expires,
            None if after is None else after.completed,
        ),
        progress,
    )
