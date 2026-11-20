import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from shared_lib.infrastructure.dag_internal_transport import DagServiceKey
from shared_lib.infrastructure.dag_service_runtime import DagListenerSettings
from service.analysis_run_recovery import (
    ACCEPTED,
    LIMIT_REACHED,
    NOT_RECOVERABLE,
    NOT_STALLED,
    RESUME_LIMIT,
    STALLED,
    OwnerRunRecovery,
    RecoveryUnavailable,
    resumes_table_ddl,
)
from service.analysis_run_schedule import OwnerPositionRunSchedule
from service.analysis_run_worker import LeaseLost
from service.current_account_dag_server import (
    current_account_engine_service_runtime,
    current_account_run_recovery,
)
from tests.test_current_account_dag_server import (
    AccountOnlyDagRunAuthority,
    _account_client,
    _dispatcher,
)

NOW = datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc)


class FakeSchedule(OwnerPositionRunSchedule):
    """Real lane semantics with in-memory run/lifecycle reads."""

    def __init__(self, dispatcher, states):
        super().__init__(dispatcher, owner_kind="account")
        self.states, self.retired, self.claims = states, [], {}

    async def _execute(self, query, values, *, conditional=False):
        """The resumes table, with Cassandra lightweight-transaction results."""
        if query.startswith("SELECT ordinal"):
            return [SimpleNamespace(ordinal=o, command_id=c) for (j, o), c in sorted(self.claims.items()) if j == values[0]]
        if query.startswith("INSERT"):
            key = (values[0], values[1])
            if key in self.claims:
                return [SimpleNamespace(applied=False)]
            self.claims[key] = values[2]
            return [SimpleNamespace(applied=True)]
        if query.startswith("DELETE"):
            key = (values[0], values[1])
            applied = self.claims.get(key) == values[2]
            if applied:
                del self.claims[key]
            return [SimpleNamespace(applied=applied)]
        raise AssertionError(query)

    async def _run(self, run_id):
        if run_id not in self.states:
            raise PermissionError("registered owner differs from scheduling lane")
        return SimpleNamespace(spec=SimpleNamespace(run_id=run_id))

    async def _state(self, run):
        return self.states[run.spec.run_id]

    async def _retire_completed(self, run):
        self.retired.append(run.spec.run_id)


def _state(job_state, *, heartbeat_age=0):
    return SimpleNamespace(
        job_state=job_state, last_heartbeat_at=NOW - timedelta(seconds=heartbeat_age)
    )


def _recovery(states, **kwargs):
    dispatcher = _dispatcher(AccountOnlyDagRunAuthority(_account_client()))
    dispatcher.lease_seconds = 60
    dispatcher.calls = []

    async def dispatch(run_id, **flags):
        dispatcher.calls.append((run_id, flags))

    dispatcher.dispatch = dispatch
    schedule = FakeSchedule(dispatcher, states)
    return OwnerRunRecovery(schedule, clock=lambda: NOW, **kwargs), schedule, dispatcher


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, NOT_STALLED),
        (_state("PENDING"), NOT_STALLED),
        (_state("COMPLETED"), NOT_STALLED),
        (_state("RUNNING", heartbeat_age=30), NOT_STALLED),
        (_state("RUNNING", heartbeat_age=61), STALLED),
        (_state("INTERRUPTED"), STALLED),
        (_state("FAILED"), NOT_RECOVERABLE),
        (_state("CANCELLED"), NOT_RECOVERABLE),
        (_state("AWAITING_CORRECTION"), NOT_RECOVERABLE),
    ],
)
def test_eligibility_follows_the_lifecycle_row(state, expected):
    run_id = uuid4()
    recovery, _, _ = _recovery({run_id: state})
    assert asyncio.run(recovery.eligibility(run_id)) == expected


def test_only_stalled_runs_are_queued_and_a_replayed_command_keeps_its_ordinal():
    stalled, live = uuid4(), uuid4()
    recovery, schedule, _ = _recovery(
        {stalled: _state("RUNNING", heartbeat_age=120), live: _state("RUNNING")}
    )
    command = uuid4()

    async def exercise():
        assert await recovery.request(live, uuid4()) == {"outcome": NOT_STALLED, "ordinal": None, "limit": RESUME_LIMIT}
        assert await recovery.request(stalled, command) == {"outcome": ACCEPTED, "ordinal": 1, "limit": RESUME_LIMIT}
        assert await recovery.request(stalled, command) == {"outcome": ACCEPTED, "ordinal": 1, "limit": RESUME_LIMIT}
        assert (await recovery.status(stalled))["used"] == 1
        return recovery.queue.qsize()

    assert asyncio.run(exercise()) == 1
    assert not any(job == str(live) for job, _ in schedule.claims)


def test_two_resumes_per_run_then_limit_reached():
    run_id = uuid4()
    recovery, _, _ = _recovery({run_id: _state("INTERRUPTED")})

    async def exercise():
        assert await recovery.request(run_id, uuid4()) == {"outcome": ACCEPTED, "ordinal": 1, "limit": RESUME_LIMIT}
        recovery.pending.clear()  # the first resume ran and the run stopped again
        assert await recovery.request(run_id, uuid4()) == {"outcome": ACCEPTED, "ordinal": 2, "limit": RESUME_LIMIT}
        assert await recovery.request(run_id, uuid4()) == {"outcome": LIMIT_REACHED, "ordinal": None, "limit": RESUME_LIMIT}
        status = await recovery.status(run_id)
        assert status == {"eligibility": STALLED, "used": RESUME_LIMIT, "limit": RESUME_LIMIT}

    asyncio.run(exercise())


def test_a_concurrent_claim_moves_to_the_next_ordinal():
    run_id = uuid4()
    recovery, schedule, _ = _recovery({run_id: _state("INTERRUPTED")})
    schedule.claims[(str(run_id), 1)] = "racing-command"
    original = schedule._execute
    calls = {"select": 0}

    async def racing(query, values, *, conditional=False):
        if query.startswith("SELECT ordinal") and calls["select"] == 0:
            calls["select"] += 1
            return []  # first read missed the racing claim
        return await original(query, values, conditional=conditional)

    schedule._execute = racing
    assert asyncio.run(recovery.request(run_id, uuid4())) == {"outcome": ACCEPTED, "ordinal": 2, "limit": RESUME_LIMIT}


def test_a_full_queue_releases_the_claim_and_reports_unavailable():
    first, second = uuid4(), uuid4()
    recovery, schedule, _ = _recovery(
        {first: _state("INTERRUPTED"), second: _state("INTERRUPTED")}, capacity=1
    )

    async def exercise():
        await recovery.request(first, uuid4())
        with pytest.raises(RecoveryUnavailable):
            await recovery.request(second, uuid4())
        assert second not in recovery.pending
        assert (await recovery.status(second))["used"] == 0

    asyncio.run(exercise())


def test_another_owners_run_is_refused_by_the_lane():
    recovery, _, _ = _recovery({})
    with pytest.raises(PermissionError):
        asyncio.run(recovery.request(uuid4(), uuid4()))


def test_resumes_table_ddl_is_keyspace_scoped():
    ddl = resumes_table_ddl("bjj_s39")
    assert "bjj_s39.analysis_run_resumes" in ddl and "PRIMARY KEY (job_id, ordinal)" in ddl


def test_worker_resumes_with_recovery_flags_and_retires_the_completed_run():
    run_id = uuid4()
    recovery, schedule, dispatcher = _recovery({run_id: _state("INTERRUPTED")})
    observed = []

    async def exercise():
        stop = asyncio.Event()

        async def observe(attempt):
            observed.append(attempt)
            stop.set()

        await recovery.request(run_id, uuid4())
        await asyncio.wait_for(recovery.run(stop, observe=observe), 2)

    asyncio.run(exercise())
    assert dispatcher.calls == [(run_id, {"recover": True, "retry_failed": True, "reissue_unknown": True})]
    assert schedule.retired == [run_id]
    assert [(a.run_id, a.status) for a in observed] == [(run_id, "completed")]
    assert run_id not in recovery.pending


def test_a_lost_lease_is_reported_without_retiring_the_run():
    run_id = uuid4()
    recovery, schedule, dispatcher = _recovery({run_id: _state("INTERRUPTED")})

    async def lose(run_id, **_flags):
        raise LeaseLost("fresh owner")

    dispatcher.dispatch = lose
    observed = []

    async def exercise():
        stop = asyncio.Event()

        async def observe(attempt):
            observed.append(attempt.status)
            stop.set()

        await recovery.request(run_id, uuid4())
        await asyncio.wait_for(recovery.run(stop, observe=observe), 2)

    asyncio.run(exercise())
    assert observed == ["lease_lost"] and schedule.retired == []


def test_worker_returns_promptly_when_stopped_while_idle():
    recovery, _, _ = _recovery({})

    async def exercise():
        stop = asyncio.Event()
        task = asyncio.create_task(recovery.run(stop, observe=lambda _a: asyncio.sleep(0)))
        await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 1)

    asyncio.run(exercise())


def _runtime(dispatcher, **kwargs):
    return current_account_engine_service_runtime(
        dispatcher,
        SimpleNamespace(publish=lambda **_kwargs: {"MessageId": "not-used"}),
        listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
        enqueue_key=DagServiceKey("current-account-enqueue", "e" * 40, "account"),
        enqueue_audience="current-account-enqueue",
        observe_schedule=lambda _page: asyncio.sleep(0),
        observe_delivery=lambda _page: asyncio.sleep(0),
        on_failure=lambda _name: asyncio.sleep(0),
        interval_seconds=1,
        **kwargs,
    )


def test_runtime_adds_the_recovery_worker_only_for_this_dispatchers_account_lane():
    dispatcher = _dispatcher(AccountOnlyDagRunAuthority(_account_client()))
    recovery = current_account_run_recovery(dispatcher)
    runtime = _runtime(dispatcher, recovery=recovery, observe_recovery=lambda _a: asyncio.sleep(0))
    assert set(runtime.workers) == {"account_schedule", "account_delivery", "account_recovery"}

    other = current_account_run_recovery(_dispatcher(AccountOnlyDagRunAuthority(_account_client())))
    with pytest.raises(ValueError, match="share this dispatcher"):
        _runtime(dispatcher, recovery=other, observe_recovery=lambda _a: asyncio.sleep(0))
    with pytest.raises(ValueError, match="observer"):
        _runtime(dispatcher, recovery=recovery)
