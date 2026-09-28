"""Opt-in position-DAG worker leases on the existing job_lifecycle table.

Not registered with legacy recovery/heartbeat tasks. A unique owner token belongs
to one attempt, never to a process; all mutations compare that token and heartbeat.
"""

import asyncio
import math
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from cassandra import ConsistencyLevel
from cassandra.policies import FallthroughRetryPolicy
from cassandra.query import SimpleStatement
from pydantic import AwareDatetime, TypeAdapter
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

from service.analysis_checkpoint_journal import _target
from service.analysis_media import DeterministicPreparationRejected


class LeaseLost(RuntimeError):
    pass


class KeyspacesRunLease:
    def __init__(
        self,
        run,
        session,
        *,
        keyspace,
        authority,
        lease_seconds,
        clock=None,
        timeout=10,
    ):
        self.run = RegisteredAIRunV1.model_validate(run)
        self.target = _target(keyspace, "job_lifecycle")
        if (
            not callable(authority)
            or not math.isfinite(lease_seconds)
            or lease_seconds <= 0
            or not math.isfinite(timeout)
            or not 0 < timeout < lease_seconds
        ):
            raise ValueError(
                "authority and finite lease longer than I/O timeout required"
            )
        self.session, self.authority = session, authority
        self.lease_seconds, self.timeout = lease_seconds, timeout
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.token = "position-attempt:" + uuid4().hex
        self.acquired = False
        self.closed = False
        self._lock = asyncio.Lock()
        self._last_now = None

    def _now(self):
        now = (
            TypeAdapter(AwareDatetime)
            .validate_python(self.clock())
            .astimezone(timezone.utc)
        )
        # CQL timestamp precision is milliseconds, including CAS comparisons.
        now = now.replace(microsecond=now.microsecond // 1000 * 1000)
        if self._last_now is not None and now < self._last_now:
            raise LeaseLost("worker clock moved backwards")
        self._last_now = now
        return now

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
            return list(self.session.execute(statement, values, timeout=self.timeout))

        return await asyncio.to_thread(execute)

    @staticmethod
    def _applied(rows):
        if len(rows) != 1 or type(getattr(rows[0], "applied", None)) is not bool:
            raise LeaseLost("missing conditional lifecycle outcome")
        return rows[0].applied

    async def _read(self):
        rows = await self._execute(
            f"SELECT * FROM {self.target} WHERE job_id=%s", [str(self.run.spec.run_id)]
        )
        if len(rows) != 1:
            raise LeaseLost("registered lifecycle row required")
        row = rows[0]
        spec = self.run.spec
        if (
            row.video_id != spec.video_id
            or row.user_id != spec.owner.principal_id
            or row.pipeline_kind != spec.workflow_id
        ):
            raise LeaseLost("lifecycle identity differs from admitted run")
        return row

    @staticmethod
    def _heartbeat(row):
        value = row.last_heartbeat_at
        if not isinstance(value, datetime):
            raise LeaseLost("missing lifecycle heartbeat")
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    async def register(self):
        """Explicit idempotent registration; never overwrites an existing run."""
        await self.authority(self.run)
        now, spec = self._now(), self.run.spec
        rows = await self._execute(
            f"INSERT INTO {self.target} "
            "(job_id,video_id,user_id,pipeline_kind,job_state,owner_instance_id,"
            "last_heartbeat_at,started_at,updated_at,stage,progress_percent) "
            "VALUES (%s,%s,%s,%s,'PENDING','',%s,%s,%s,'',0) IF NOT EXISTS",
            [
                str(spec.run_id),
                spec.video_id,
                spec.owner.principal_id,
                spec.workflow_id,
                now,
                now,
                now,
            ],
            conditional=True,
        )
        self._applied(rows)
        await self._read()
        await self.authority(self.run)

    def _live(self, row):
        now, heartbeat = self._now(), self._heartbeat(row)
        if heartbeat > now or now >= heartbeat + timedelta(seconds=self.lease_seconds):
            raise LeaseLost("worker lease expired or clock is inconsistent")
        if row.job_state != "RUNNING" or row.owner_instance_id != self.token:
            raise LeaseLost("worker no longer owns the running attempt")

    async def guard(self):
        if not self.acquired or self.closed:
            raise LeaseLost("attempt has no live lease")
        await self.authority(self.run)
        row = await self._read()
        self._live(row)
        await self.authority(self.run)
        if not self.acquired or self.closed:
            raise LeaseLost("attempt closed during authority check")
        self._live(row)  # authority I/O must not return an already expired lease

    async def acquire(self, *, recover=False):
        async with self._lock:
            if self.acquired or self.closed:
                raise LeaseLost("lease object is single-attempt")
            await self.authority(self.run)
            row = await self._read()
            now, heartbeat = self._now(), self._heartbeat(row)
            eligible = row.job_state == "PENDING"
            if recover:
                eligible |= row.job_state == "INTERRUPTED" or (
                    row.job_state == "RUNNING"
                    and now >= heartbeat + timedelta(seconds=self.lease_seconds)
                )
            if not eligible or heartbeat > now:
                raise LeaseLost("job is not eligible for this claim")
            try:
                applied = await self._mutate(
                    row,
                    "job_state='RUNNING',owner_instance_id=%s,"
                    "last_heartbeat_at=%s,updated_at=%s,completed_at=null",
                    [self.token, now, now],
                )
                if not applied:
                    raise LeaseLost("another worker won the lifecycle claim")
                self.acquired = True
                await self.guard()
            except BaseException:
                # An unknown commit never authorizes dispatch or reuse of this token.
                self.closed = True
                raise

    async def _mutate(self, row, assignment, values):
        return self._applied(
            await self._execute(
                f"UPDATE {self.target} SET {assignment} WHERE job_id=%s "
                "IF owner_instance_id=%s AND job_state=%s AND last_heartbeat_at=%s "
                "AND video_id=%s AND user_id=%s AND pipeline_kind=%s",
                values
                + [
                    str(self.run.spec.run_id),
                    row.owner_instance_id,
                    row.job_state,
                    row.last_heartbeat_at,
                    self.run.spec.video_id,
                    self.run.spec.owner.principal_id,
                    self.run.spec.workflow_id,
                ],
                conditional=True,
            )
        )

    async def heartbeat(self):
        async with self._lock:
            await self.guard()
            row = await self._read()
            self._live(row)
            now = self._now()
            if not await self._mutate(
                row, "last_heartbeat_at=%s,updated_at=%s", [now, now]
            ):
                raise LeaseLost("heartbeat lost ownership")
            await self.guard()

    async def finish(self, *, completed):
        """Caller supplies success only after DurableGeminiRun.execute returns."""
        async with self._lock:
            await self.guard()
            row = await self._read()
            self._live(row)
            now = self._now()
            state = "COMPLETED" if completed else "INTERRUPTED"
            try:
                if not await self._mutate(
                    row,
                    "job_state=%s,updated_at=%s,completed_at=%s",
                    [state, now, now if completed else None],
                ):
                    raise LeaseLost("terminal transition lost ownership")
            finally:
                self.closed = True


async def _drain(task):
    """Do not let repeated caller cancellation abandon resource teardown."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if not task.cancelled():
        task.exception()  # retrieve; original outcome is handled by the supervisor


async def run_position_attempt(
    lease, host_factory, *, heartbeat_seconds, recover=False, retry_failed=False
):
    """Run one explicit attempt, supervising media teardown and durable completion.

    host_factory(guard) is an async context manager. It must pass this exact guard
    to the host/journal/sink and adapt it for the provider/media run argument. It
    owns resource teardown. No scheduler registration, automatic recovery or cleanup.
    """
    if (
        not math.isfinite(heartbeat_seconds)
        or not 0 < heartbeat_seconds < lease.lease_seconds / 3
    ):
        raise ValueError("heartbeat interval must be below one third of lease duration")
    await lease.acquire(recover=recover)
    stop = asyncio.Event()

    async def pulse():
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), heartbeat_seconds)
            except TimeoutError:
                await lease.heartbeat()

    async def execute():
        async with host_factory(lease.guard) as host:
            if host.run != lease.run:
                raise ValueError("worker host differs from leased run")
            await lease.guard()
            return await host.execute(retry_failed=retry_failed)

    heart = asyncio.create_task(pulse())
    work = asyncio.create_task(execute())
    try:
        done, _ = await asyncio.wait((work, heart), return_when=asyncio.FIRST_COMPLETED)
        if heart in done:
            await heart  # propagate heartbeat/authority failure
            raise LeaseLost("heartbeat stopped before worker completion")
        result = await work
        stop.set()
        await heart
        await lease.finish(completed=True)
        return result
    except DeterministicPreparationRejected:
        # This marker is emitted only before the host yields to provider work.
        # The failing host task has already unwound its media/provider contexts;
        # stop and join heartbeats before making one guarded CAS transition.
        stop.set()
        await heart
        try:
            await lease.finish(completed=False)
        except BaseException:
            # A lost owner, failed authority check or uncertain Cassandra CAS must
            # remain uncertain. Never force an INTERRUPTED transition.
            lease.closed = True
            raise
        raise
    except BaseException:
        # Stop new work and drain provider/media contexts before releasing ownership.
        # Latch failure before cancellation: an in-flight guard must not return
        # permission during teardown after a failed heartbeat or caller cancel.
        lease.closed = True
        work.cancel()
        await _drain(work)
        stop.set()
        await _drain(heart)
        raise
    finally:
        # Deliberately do not invent FAILED/CANCELLED semantics for uncertain work.
        # On errors leave RUNNING until expiry; explicit recovery reads checkpoints.
        lease.closed = True
