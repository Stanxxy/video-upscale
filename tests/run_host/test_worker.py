import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from test_host import case

from service.analysis_run_worker import (
    KeyspacesRunLease,
    LeaseLost,
    run_position_attempt,
)
from service.analysis_media import DeterministicPreparationRejected


@pytest.fixture
def leases(checkpoint_database, inputs, resolver):
    session, keyspace = checkpoint_database
    # Existing job_lifecycle columns, not a new production schema/migration.
    session.execute(f"""CREATE TABLE IF NOT EXISTS {keyspace}.job_lifecycle (
        job_id text PRIMARY KEY, video_id text, user_id text, pipeline_kind text,
        job_state text, owner_instance_id text, last_heartbeat_at timestamp,
        started_at timestamp, updated_at timestamp, completed_at timestamp,
        stage text, progress_percent double
    )""")
    authority = resolver.guard

    def make(**kwargs):
        return KeyspacesRunLease(
            inputs[0],
            session,
            keyspace=keyspace,
            authority=authority,
            lease_seconds=30,
            timeout=5,
            **kwargs,
        )

    return make


@pytest.mark.asyncio
async def test_registration_does_not_overwrite_running_or_terminal(leases):
    first, second = leases(), leases()
    await first.register()
    await first.acquire()
    await second.register()
    assert (await second._read()).owner_instance_id == first.token
    with pytest.raises(LeaseLost):
        await second.acquire()
    await first.finish(completed=True)
    await second.register()
    assert (await second._read()).job_state == "COMPLETED"
    with pytest.raises(LeaseLost):
        await leases().acquire(recover=True)


@pytest.mark.asyncio
async def test_simultaneous_claims_have_one_owner(leases):
    first, second = leases(), leases()
    await first.register()
    barrier = asyncio.Barrier(2)

    def racing(lease):
        save = lease._mutate

        async def mutate(*args):
            await asyncio.wait_for(barrier.wait(), 5)
            return await save(*args)

        lease._mutate = mutate

    racing(first)
    racing(second)
    results = await asyncio.gather(
        first.acquire(), second.acquire(), return_exceptions=True
    )
    assert sum(r is None for r in results) == 1
    assert sum(isinstance(r, LeaseLost) for r in results) == 1
    row = await first._read()
    assert row.owner_instance_id in {first.token, second.token}
    assert first.token != second.token


@pytest.mark.asyncio
async def test_takeover_fences_old_heartbeat_and_terminal_write(leases):
    now = datetime.now(timezone.utc)
    old = leases(clock=lambda: now)
    await old.register()
    await old.acquire()
    now += timedelta(seconds=31)
    replacement = leases(clock=lambda: now)
    with pytest.raises(LeaseLost):
        await replacement.acquire()  # recovery is explicit
    await replacement.acquire(recover=True)
    with pytest.raises(LeaseLost):
        await old.heartbeat()
    with pytest.raises(LeaseLost):
        await old.finish(completed=True)
    assert (await replacement._read()).owner_instance_id == replacement.token
    await replacement.finish(completed=True)


@pytest.mark.asyncio
async def test_stale_terminal_cas_cannot_overwrite_replacement(leases):
    now = datetime.now(timezone.utc)
    old = leases(clock=lambda: now)
    await old.register()
    await old.acquire()
    mutate = old._mutate
    replacement = leases(clock=lambda: now)

    async def takeover(row, assignment, values):
        nonlocal now
        now += timedelta(seconds=31)
        await replacement.acquire(recover=True)
        return await mutate(row, assignment, values)

    old._mutate = takeover
    with pytest.raises(LeaseLost, match="terminal transition"):
        await old.finish(completed=True)
    assert (await replacement._read()).job_state == "RUNNING"
    await replacement.guard()


@pytest.mark.asyncio
async def test_interrupted_cas_cannot_overwrite_replacement_owner(leases):
    now = datetime.now(timezone.utc)
    old = leases(clock=lambda: now)
    await old.register()
    await old.acquire()
    mutate = old._mutate
    replacement = leases(clock=lambda: now)

    async def takeover(row, assignment, values):
        nonlocal now
        now += timedelta(seconds=31)
        await replacement.acquire(recover=True)
        return await mutate(row, assignment, values)

    old._mutate = takeover
    with pytest.raises(LeaseLost, match="terminal transition"):
        await old.finish(completed=False)
    row = await replacement._read()
    assert row.job_state == "RUNNING"
    assert row.owner_instance_id == replacement.token
    await replacement.guard()


@pytest.mark.asyncio
async def test_pre_provider_preparation_rejection_interrupts_after_context_drain(leases):
    lease = leases()
    await lease.register()
    drained = False
    original_finish = lease.finish

    async def checked_finish(*, completed):
        assert completed is False
        assert drained is True
        assert lease.closed is False
        await original_finish(completed=completed)

    lease.finish = checked_finish

    @asynccontextmanager
    async def reject_before_yield(guard):
        nonlocal drained
        try:
            raise DeterministicPreparationRejected("sanitized fixture rejection")
            yield None
        finally:
            drained = True

    with pytest.raises(
        DeterministicPreparationRejected, match="sanitized fixture rejection"
    ):
        await run_position_attempt(
            lease, reject_before_yield, heartbeat_seconds=0.01
        )
    row = await lease._read()
    assert row.job_state == "INTERRUPTED"
    assert row.completed_at is None
    with pytest.raises(LeaseLost):
        await leases().acquire()
    recovery = leases()
    await recovery.acquire(recover=True)
    await recovery.finish(completed=False)


@pytest.mark.asyncio
async def test_renewal_racing_stale_takeover_blocks_claim(leases):
    now = datetime.now(timezone.utc)
    old = leases(clock=lambda: now)
    await old.register()
    await old.acquire()
    candidate = leases(clock=lambda: now + timedelta(seconds=31))
    mutate = candidate._mutate

    async def renewal(row, assignment, values):
        nonlocal now
        now += timedelta(seconds=20)
        await old.heartbeat()
        return await mutate(row, assignment, values)

    candidate._mutate = renewal
    with pytest.raises(LeaseLost, match="another worker"):
        await candidate.acquire(recover=True)
    await old.guard()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_lost_claim_ack_never_authorizes_dispatch(leases, cancel):
    lease = leases()
    await lease.register()
    mutate, entered = lease._mutate, asyncio.Event()

    async def lost(*args):
        await mutate(*args)
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise TimeoutError("claim acknowledgement lost")

    lease._mutate = lost
    task = asyncio.create_task(lease.acquire())
    await asyncio.wait_for(entered.wait(), 5)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert (await lease._read()).owner_instance_id == lease.token
    with pytest.raises(LeaseLost):
        await lease.guard()
    with pytest.raises(LeaseLost):
        await lease.acquire(recover=True)


@pytest.mark.asyncio
async def test_missing_malformed_and_wrong_identity_fail_closed(leases):
    lease = leases()
    with pytest.raises(LeaseLost, match="registered lifecycle"):
        await lease.acquire()
    await lease.register()
    await lease._execute(
        f"UPDATE {lease.target} SET user_id='intruder' WHERE job_id=%s",
        [str(lease.run.spec.run_id)],
    )
    with pytest.raises(LeaseLost, match="identity"):
        await lease.acquire()
    with pytest.raises(LeaseLost, match="identity"):
        await lease.register()
    for rows in ([], [object()]):
        with pytest.raises(LeaseLost):
            lease._applied(rows)


@pytest.mark.asyncio
async def test_backwards_clock_and_expired_lease_cannot_renew(leases):
    now = datetime.now(timezone.utc)
    lease = leases(clock=lambda: now)
    await lease.register()
    await lease.acquire()
    now -= timedelta(seconds=1)
    with pytest.raises(LeaseLost, match="backwards"):
        await lease.guard()
    now += timedelta(seconds=32)
    with pytest.raises(LeaseLost, match="expired"):
        await lease.heartbeat()


@asynccontextmanager
async def worker_case(leases, resolver, inputs, s3, journal):
    lease = leases()
    await lease.register()
    observed = {}

    @asynccontextmanager
    async def factory(guard):
        original = resolver.guard

        async def run_guard(run):
            assert run == lease.run
            await guard()

        resolver.guard = run_guard
        journal.guard = guard
        try:
            async with case(resolver, inputs, s3, journal) as (host, wire, published):
                host.guard = guard
                observed.update(host=host, wire=wire, published=published)
                yield host
        finally:
            resolver.guard = original
            observed["closed"] = True

    yield lease, factory, observed


@pytest.mark.asyncio
async def test_supervisor_real_dag_completes_only_after_durable_host_and_teardown(
    leases, resolver, inputs, s3, journal
):
    async with worker_case(leases, resolver, inputs, s3, journal) as (
        lease,
        factory,
        seen,
    ):
        finish = lease.finish

        async def checked_finish(**kwargs):
            assert seen["closed"]
            assert (await seen["host"]._read("dag_complete")).phase == "succeeded"
            await finish(**kwargs)

        lease.finish = checked_finish
        result = await run_position_attempt(lease, factory, heartbeat_seconds=0.1)
        assert result.techniques.techniques == ()
        assert len(seen["wire"].generates()) == 2
        assert [c[0] for c in seen["published"]] == ["positions", "techniques"]
        assert (await lease._read()).job_state == "COMPLETED"
        with pytest.raises(LeaseLost):
            await lease.guard()


@pytest.mark.asyncio
async def test_supervisor_heartbeat_failure_cancels_and_closes_media(
    leases, resolver, inputs, s3, journal
):
    async with worker_case(leases, resolver, inputs, s3, journal) as (
        lease,
        factory,
        seen,
    ):
        entered = asyncio.Event()

        @asynccontextmanager
        async def blocked(guard):
            async with factory(guard) as host:

                async def execute(**kwargs):
                    entered.set()
                    await asyncio.Event().wait()

                host.execute = execute
                yield host

        async def failed_heartbeat():
            await entered.wait()
            raise TimeoutError("heartbeat store unavailable")

        lease.heartbeat = failed_heartbeat
        with pytest.raises(TimeoutError, match="heartbeat store"):
            await run_position_attempt(lease, blocked, heartbeat_seconds=0.01)
        assert seen["closed"] and seen["wire"].calls == []
        assert (await lease._read()).job_state == "RUNNING"
        assert lease.closed


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_teardown(leases):
    lease = leases()
    await lease.register()
    entered, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def factory(guard):
        class Host:
            run = lease.run

            async def execute(self, **kwargs):
                entered.set()
                await asyncio.Event().wait()

        try:
            yield Host()
        finally:
            draining.set()
            await release.wait()

    task = asyncio.create_task(
        run_position_attempt(lease, factory, heartbeat_seconds=0.01)
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    await asyncio.wait_for(draining.wait(), 5)
    assert lease.closed
    with pytest.raises(LeaseLost):
        await lease.guard()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lease.closed and (await lease._read()).job_state == "RUNNING"


@pytest.mark.asyncio
async def test_worker_failure_retains_partial_result_and_no_completed_lifecycle(
    leases, resolver, inputs, s3, journal
):
    async with worker_case(leases, resolver, inputs, s3, journal) as (
        lease,
        factory,
        seen,
    ):

        @asynccontextmanager
        async def failed(guard):
            async with factory(guard) as host:
                generate = host.provider.generate

                async def fail_technique(run, request, execution_id, call_id):
                    if request.node_id == "technique_scan":
                        seen["wire"].status = 503
                    return await generate(run, request, execution_id, call_id)

                host.provider.generate = fail_technique
                yield host

        with pytest.raises(RuntimeError, match="HTTP 503"):
            await run_position_attempt(lease, failed, heartbeat_seconds=0.1)
        assert seen["closed"]
        assert [c[0] for c in seen["published"]] == ["positions"]
        assert (await lease._read()).job_state == "RUNNING"
        assert not any(c[0] == "DELETE" for c in seen["wire"].calls)


@pytest.mark.asyncio
async def test_terminal_commit_loss_preserves_completion_without_second_attempt(leases):
    lease = leases()
    await lease.register()
    await lease.acquire()
    mutate = lease._mutate

    async def lost(*args):
        await mutate(*args)
        raise TimeoutError("completion acknowledgement lost")

    lease._mutate = lost
    with pytest.raises(TimeoutError):
        await lease.finish(completed=True)
    assert lease.closed and (await lease._read()).job_state == "COMPLETED"
    with pytest.raises(LeaseLost):
        await leases().acquire(recover=True)


@pytest.mark.asyncio
async def test_invalid_heartbeat_interval_never_claims(leases):
    lease = leases()
    await lease.register()
    for interval in (0, -1, 10, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            await run_position_attempt(lease, None, heartbeat_seconds=interval)
    assert (await lease._read()).job_state == "PENDING"


@pytest.mark.asyncio
@pytest.mark.parametrize("close", [False, True])
async def test_final_authority_io_cannot_return_expired_or_closed_lease(leases, close):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    lease = leases(clock=lambda: now)
    await lease.register()
    await lease.acquire()
    calls = 0

    async def slow_authority(run):
        nonlocal now, calls
        calls += 1
        if calls == 2:
            if close:
                lease.closed = True
            else:
                now += timedelta(seconds=30)

    lease.authority = slow_authority
    with pytest.raises(LeaseLost, match="closed|expired"):
        await lease.guard()


@pytest.mark.asyncio
async def test_completed_dag_survives_lifecycle_commit_loss_without_more_inference(
    leases, resolver, inputs, s3, journal
):
    async with worker_case(leases, resolver, inputs, s3, journal) as (
        lease,
        factory,
        seen,
    ):
        mutate = lease._mutate

        async def unavailable(row, assignment, values):
            if values and values[0] == "COMPLETED":
                raise TimeoutError("lifecycle result not committed")
            return await mutate(row, assignment, values)

        lease._mutate = unavailable
        with pytest.raises(TimeoutError, match="not committed"):
            await run_position_attempt(lease, factory, heartbeat_seconds=0.1)
        assert len(seen["wire"].generates()) == 2
        assert (await lease._read()).job_state == "RUNNING"
        replacement = leases(
            clock=lambda: datetime.now(timezone.utc) + timedelta(seconds=31)
        )
        result = await run_position_attempt(
            replacement, factory, heartbeat_seconds=0.1, recover=True
        )
        assert result.techniques.techniques == ()
        assert seen["wire"].calls == [] and seen["published"] == []
        assert (await replacement._read()).job_state == "COMPLETED"
