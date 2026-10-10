"""Owner-requested recovery of one stalled run in an approved owner lane.

The scheduler never reclaims expired work by itself. This queue is the explicit
path: the account host asks for one run, the lifecycle row decides whether the
run is stalled, and a supervised worker resumes it from its checkpoints with
``dispatch(recover=True, retry_failed=True, reissue_unknown=True)``: a model
call interrupted before its answer arrived is sent again. Retries stay bounded
by each step's ``max_call_attempts``; the lifecycle CAS prevents a double claim.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from service.analysis_checkpoint_journal import _target

from service.analysis_run_schedule import (
    OwnerPositionRunSchedule,
    ScheduledAttempt,
    _stamp,
)
from service.analysis_run_worker import LeaseLost

LOG = logging.getLogger(__name__)

STALLED = "stalled"
NOT_STALLED = "not_stalled"
NOT_RECOVERABLE = "not_recoverable"
LIMIT_REACHED = "limit_reached"
ACCEPTED = "accepted"
# Owner resumes per run: one after a stop, one more if the resumed run stops again.
RESUME_LIMIT = 2
RESUMES_TABLE = "analysis_run_resumes"


def resumes_table_ddl(keyspace):
    """Per-run resume budget; ordinals are claimed with lightweight transactions."""
    return (
        f"CREATE TABLE IF NOT EXISTS {_target(keyspace, RESUMES_TABLE)} ("
        "job_id text, ordinal int, command_id text, requested_at timestamp, "
        "PRIMARY KEY (job_id, ordinal))"
    )


class RecoveryUnavailable(RuntimeError):
    """The recovery queue cannot take another request right now."""


class OwnerRunRecovery:
    def __init__(self, schedule, *, capacity=8, clock=None):
        if not isinstance(schedule, OwnerPositionRunSchedule):
            raise ValueError("an approved owner scheduling lane is required")
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("bounded recovery capacity required")
        self.schedule = schedule
        self.resumes = _target(schedule.dispatcher.keyspace, RESUMES_TABLE)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.queue = asyncio.Queue(capacity)
        self.pending = set()

    @property
    def dispatcher(self):
        return self.schedule.dispatcher

    def _classify(self, state):
        if state is None or state.job_state in {"PENDING", "COMPLETED"}:
            return NOT_STALLED
        if state.job_state == "INTERRUPTED":
            return STALLED
        if state.job_state == "RUNNING":
            expires = _stamp(state.last_heartbeat_at) + timedelta(
                seconds=self.dispatcher.lease_seconds
            )
            return STALLED if self.clock() >= expires else NOT_STALLED
        if state.job_state in {"FAILED", "CANCELLED", "AWAITING_CORRECTION"}:
            return NOT_RECOVERABLE
        raise ValueError("unknown lifecycle state")

    async def eligibility(self, run_id):
        """``stalled``, ``not_stalled`` or ``not_recoverable`` for this owner lane."""
        run = await self.schedule._run(UUID(str(run_id)))
        return self._classify(await self.schedule._state(run))

    async def _claimed(self, run_id):
        rows = await self.schedule._execute(
            f"SELECT ordinal, command_id FROM {self.resumes} WHERE job_id=%s",
            [str(run_id)],
        )
        return {row.ordinal: row.command_id for row in rows}

    @staticmethod
    def _result(outcome, ordinal=None):
        return {"outcome": outcome, "ordinal": ordinal, "limit": RESUME_LIMIT}

    async def status(self, run_id):
        """Eligibility plus the run's resume budget (used of ``RESUME_LIMIT``)."""
        run_id = UUID(str(run_id))
        eligibility = await self.eligibility(run_id)
        return {"eligibility": eligibility, "used": len(await self._claimed(run_id)), "limit": RESUME_LIMIT}

    async def request(self, run_id, command_id):
        """Claim one resume and queue the run.

        Returns ``{"outcome", "ordinal", "limit"}``; ``ordinal`` is None on refusal.

        A replayed ``command_id`` returns its original ordinal and never claims
        another resume; a refused or unqueued request leaves the budget unchanged.
        """
        run_id, command_id = UUID(str(run_id)), str(UUID(str(command_id)))
        run = await self.schedule._run(run_id)
        for _ in range(3):
            claimed = await self._claimed(run_id)
            for ordinal, existing in claimed.items():
                if existing == command_id:
                    return self._result(ACCEPTED, ordinal)
            eligibility = self._classify(await self.schedule._state(run))
            if eligibility != STALLED:
                return self._result(eligibility)
            if len(claimed) >= RESUME_LIMIT:
                return self._result(LIMIT_REACHED)
            ordinal = len(claimed) + 1
            rows = await self.schedule._execute(
                f"INSERT INTO {self.resumes} (job_id,ordinal,command_id,requested_at) "
                "VALUES (%s,%s,%s,%s) IF NOT EXISTS",
                [str(run_id), ordinal, command_id, self.clock()],
                conditional=True,
            )
            if len(rows) != 1 or type(getattr(rows[0], "applied", None)) is not bool:
                raise RuntimeError("resume claim unconfirmed")
            if not rows[0].applied:
                continue  # A concurrent request took this ordinal; re-read.
            if run_id not in self.pending:
                try:
                    self.queue.put_nowait(run_id)
                except asyncio.QueueFull as error:
                    await self.schedule._execute(
                        f"DELETE FROM {self.resumes} WHERE job_id=%s AND ordinal=%s IF command_id=%s",
                        [str(run_id), ordinal, command_id],
                        conditional=True,
                    )
                    raise RecoveryUnavailable("recovery queue is full") from error
                self.pending.add(run_id)
            return self._result(ACCEPTED, ordinal)
        raise RecoveryUnavailable("resume claim kept conflicting")

    async def _recover(self, run_id):
        try:
            run = await self.schedule._run(run_id)
            await self.dispatcher.dispatch(
                run_id, recover=True, retry_failed=True, reissue_unknown=True
            )
            await self.schedule._retire_completed(run)
            return ScheduledAttempt(run_id, "completed")
        except LeaseLost:
            # Another owner already holds a fresh lease; nothing was resumed.
            return ScheduledAttempt(run_id, "lease_lost")
        except Exception as error:
            # Class and engine message only; provider payloads never reach this log.
            LOG.warning("owner recovery of %s stopped: %s: %s", run_id, type(error).__name__, str(error)[:200])
            return ScheduledAttempt(run_id, "unconfirmed")
        finally:
            self.pending.discard(run_id)

    async def run(self, stop, *, observe):
        if not callable(observe):
            raise ValueError("recovery observer required")
        stopped = asyncio.ensure_future(stop.wait())
        try:
            while not stop.is_set():
                take = asyncio.ensure_future(self.queue.get())
                done, _ = await asyncio.wait(
                    {take, stopped}, return_when=asyncio.FIRST_COMPLETED
                )
                if take not in done:
                    take.cancel()
                    return
                await observe(await self._recover(take.result()))
        finally:
            stopped.cancel()
