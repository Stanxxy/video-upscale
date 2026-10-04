"""Owner-requested recovery of one stalled run in an approved owner lane.

The scheduler never reclaims expired work by itself. This queue is the explicit
path: the account host asks for one run, the lifecycle row decides whether the
run is stalled, and a supervised worker resumes it from its checkpoints with
``dispatch(recover=True, retry_failed=True, reissue_unknown=True)``: a model
call interrupted before its answer arrived is sent again. Retries stay bounded
by each step's ``max_call_attempts``; the lifecycle CAS prevents a double claim.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import UUID

from service.analysis_run_schedule import (
    OwnerPositionRunSchedule,
    ScheduledAttempt,
    _stamp,
)
from service.analysis_run_worker import LeaseLost

STALLED = "stalled"
NOT_STALLED = "not_stalled"
NOT_RECOVERABLE = "not_recoverable"
ACCEPTED = "accepted"


class RecoveryUnavailable(RuntimeError):
    """The recovery queue cannot take another request right now."""


class OwnerRunRecovery:
    def __init__(self, schedule, *, capacity=8, clock=None):
        if not isinstance(schedule, OwnerPositionRunSchedule):
            raise ValueError("an approved owner scheduling lane is required")
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("bounded recovery capacity required")
        self.schedule = schedule
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

    async def request(self, run_id):
        run_id = UUID(str(run_id))
        if run_id in self.pending:
            return ACCEPTED
        eligibility = await self.eligibility(run_id)
        if eligibility != STALLED:
            return eligibility
        if run_id in self.pending:
            return ACCEPTED
        try:
            self.queue.put_nowait(run_id)
        except asyncio.QueueFull as error:
            raise RecoveryUnavailable("recovery queue is full") from error
        self.pending.add(run_id)
        return ACCEPTED

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
        except Exception:
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
