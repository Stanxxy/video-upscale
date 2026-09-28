import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.analysis_run_schedule import (
    INDEX_STATE,
    PositionRunSchedule,
    ScheduleCursor,
    SchedulePage,
)
from service.analysis_run_worker import KeyspacesRunLease


@pytest.fixture
def schedule_table(checkpoint_database):
    session, keyspace = checkpoint_database
    ddl = (
        Path(__file__).resolve().parents[2]
        / "scripts/keyspaces/migrations/002_job_recovery_index.cql"
    ).read_text()
    session.execute(
        ddl.replace(
            "video_analysis.job_recovery_index", f"{keyspace}.job_recovery_index"
        )
    )
    try:
        yield
    finally:
        # The keyspace is module-scoped; retained poison hints must not leak into
        # another test that happens to generate a UUID in the same fixed shard.
        # This table belongs only to this fixture in its disposable UUID keyspace.
        session.execute(f"DROP TABLE {keyspace}.job_recovery_index")


def schedule(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    return PositionRunSchedule(dispatch), run, wire


async def confirmed_page(queue, shard):
    # Keep the production error boundary, but expose unexpected causes to the test.
    process, errors = queue._process, []

    async def inspected(*args):
        try:
            return await process(*args)
        except Exception as exc:
            errors.append(exc)
            raise

    queue._process = inspected
    try:
        page = await queue.process_page(shard)
    finally:
        queue._process = process
    assert errors == [], [repr(error) for error in errors]
    assert page.error_code is None
    return page


@pytest.mark.asyncio
async def test_enqueue_is_durable_idempotent_and_isolated_from_legacy(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    assert await queue.enqueue(run.spec.run_id) == run.spec.run_id
    assert await queue.enqueue(run.spec.run_id) == run.spec.run_id
    key = queue._key(run)
    rows = await queue._execute(
        f"SELECT * FROM {queue.index} WHERE recovery_state=%s AND heartbeat_bucket=%s",
        key[:2],
    )
    assert len(rows) == 1 and rows[0].job_id == str(run.spec.run_id)
    assert INDEX_STATE not in {"ACTIVE", "TERMINAL", "AWAITING_CORRECTION"}
    assert await queue._state(run) is None and wire.calls == []


@pytest.mark.asyncio
async def test_restarted_scheduler_runs_dag_retires_hint_and_completed_reenqueue_never_reruns(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    restarted = PositionRunSchedule(queue.dispatcher)
    page = await restarted.process_page(run.spec.run_id.int % 16)
    assert [a.status for a in page.attempts] == ["completed"]
    assert (
        len(wire.generates()) == 2
        and (await queue._state(run)).job_state == "COMPLETED"
    )
    assert (await queue.process_page(run.spec.run_id.int % 16)).attempts == ()
    count = len(wire.calls)
    await queue.enqueue(run.spec.run_id)
    assert (await queue.process_page(run.spec.run_id.int % 16)).attempts[
        0
    ].status == "completed"
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_two_schedulers_cannot_dispatch_duplicate_paid_work(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    other = PositionRunSchedule(queue.dispatcher)
    barrier = asyncio.Barrier(2)

    def race(target):
        process = target._process

        async def together(*args):
            await asyncio.wait_for(barrier.wait(), 5)
            return await process(*args)

        target._process = together

    race(queue)
    race(other)
    results = await asyncio.gather(
        queue.process_page(run.spec.run_id.int % 16),
        other.process_page(run.spec.run_id.int % 16),
    )
    assert any(p.attempts[0].status == "completed" for p in results)
    assert len(wire.generates()) == 2


@pytest.mark.asyncio
async def test_expired_running_attempt_is_reported_not_automatically_recovered(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    lease = KeyspacesRunLease(
        run,
        queue.dispatcher.session,
        keyspace=queue.dispatcher.keyspace,
        authority=queue.dispatcher._authority,
        lease_seconds=60,
    )
    await lease.register()
    await lease.acquire()
    candidates = await queue._execute(
        f"SELECT * FROM {queue.index} WHERE recovery_state=%s AND heartbeat_bucket=%s",
        queue._key(run)[:2],
    )
    assert (
        await queue._process(run.spec.run_id.int % 16, candidates[0])
    ).status == "running"
    page = await confirmed_page(queue, run.spec.run_id.int % 16)
    assert page.attempts[0].status == "running"
    await queue._execute(
        f"UPDATE {queue.lifecycle} SET last_heartbeat_at=%s WHERE job_id=%s",
        [datetime.now(timezone.utc) - timedelta(seconds=61), str(run.spec.run_id)],
    )
    page = await queue.process_page(run.spec.run_id.int % 16)
    assert page.attempts[0].status == "explicit_recovery_required" and wire.calls == []


@pytest.mark.asyncio
async def test_uncertain_model_attempt_stays_discoverable_without_paid_retry(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    wire.status = 503
    page = await queue.process_page(run.spec.run_id.int % 16)
    assert page.attempts[0].status == "unconfirmed" and len(wire.generates()) == 1
    await queue._execute(
        f"UPDATE {queue.lifecycle} SET last_heartbeat_at=%s WHERE job_id=%s",
        [datetime.now(timezone.utc) - timedelta(seconds=61), str(run.spec.run_id)],
    )
    page = await queue.process_page(run.spec.run_id.int % 16)
    assert page.attempts[0].status == "explicit_recovery_required"
    assert len(wire.generates()) == 1


@pytest.mark.asyncio
async def test_lost_enqueue_ack_reuses_exact_discovery_entry(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    execute = queue._execute

    async def lost(query, *args, **kwargs):
        result = await execute(query, *args, **kwargs)
        if query.startswith("INSERT"):
            raise TimeoutError("index committed, acknowledgment lost")
        return result

    queue._execute = lost
    with pytest.raises(TimeoutError):
        await queue.enqueue(run.spec.run_id)
    queue._execute = execute
    await queue.enqueue(run.spec.run_id)
    rows = await queue._execute(
        f"SELECT * FROM {queue.index} WHERE recovery_state=%s AND heartbeat_bucket=%s",
        queue._key(run)[:2],
    )
    assert len(rows) == 1 and wire.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-1, 0])
async def test_corrupt_candidate_does_not_block_later_page(
    setup_dispatch, schedule_table, offset
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    state, bucket, timestamp, _ = queue._key(run)
    await queue._execute(
        f"INSERT INTO {queue.index} (recovery_state,heartbeat_bucket,last_heartbeat_at,job_id) VALUES (%s,%s,%s,%s)",
        [state, bucket, timestamp + timedelta(seconds=offset), "!invalid-uuid"],
    )
    page = await queue.process_page(run.spec.run_id.int % 16, limit=1)
    assert page.attempts[0].run_id is None and page.attempts[0].status == "unconfirmed"
    assert page.next_cursor is not None
    next_page = await queue.process_page(
        run.spec.run_id.int % 16, after=page.next_cursor, limit=1
    )
    assert next_page.attempts[0].status == "completed" and len(wire.generates()) == 2


@pytest.mark.asyncio
async def test_access_revocation_retains_hint_without_dispatch(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    setup_dispatch[-1]()
    page = await queue.process_page(run.spec.run_id.int % 16)
    assert page.attempts[0].status == "unconfirmed" and wire.calls == []
    assert await queue._state(run) is None


@pytest.mark.asyncio
async def test_completion_cleanup_failure_is_repaired_without_inference(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    retire = queue._retire_completed

    async def failed(run):
        raise TimeoutError("discovery removal not acknowledged")

    queue._retire_completed = failed
    assert (await queue.process_page(run.spec.run_id.int % 16)).attempts[
        0
    ].status == "unconfirmed"
    assert (await queue._state(run)).job_state == "COMPLETED"
    count = len(wire.calls)
    queue._retire_completed = retire
    assert (await queue.process_page(run.spec.run_id.int % 16)).attempts[
        0
    ].status == "completed"
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_query_error_retains_cursor_and_reports_missing_evidence(
    setup_dispatch, schedule_table
):
    queue, _, _ = schedule(setup_dispatch)

    async def unavailable(*args, **kwargs):
        raise TimeoutError("store unavailable")

    queue._execute = unavailable
    cursor = ScheduleCursor(datetime.now(timezone.utc), "prior")
    page = await queue.process_page(0, after=cursor)
    assert page.error_code == "discovery_unavailable" and page.next_cursor == cursor


@pytest.mark.asyncio
async def test_stopped_loop_does_not_dispatch_or_query(setup_dispatch, schedule_table):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    stop = asyncio.Event()
    stop.set()

    async def unexpected(page):
        pytest.fail("stopped loop observed work")

    await queue.run(stop, observe=unexpected, interval_seconds=1)
    assert wire.calls == [] and await queue._state(run) is None


@pytest.mark.asyncio
async def test_loop_retains_cursor_across_passes_then_restarts_at_eof(
    setup_dispatch, schedule_table
):
    queue, _, _ = schedule(setup_dispatch)
    stop, seen, reports = asyncio.Event(), [], []
    cursor = ScheduleCursor(datetime.now(timezone.utc), "last")

    async def page(shard, *, after, **kwargs):
        if shard == 0:
            seen.append(after)
        return SchedulePage(shard, (), cursor if after is None else None)

    async def observe(report):
        reports.append(report)
        if len(reports) == 33:
            stop.set()

    queue.process_page = page
    await queue.run(stop, observe=observe, interval_seconds=0.001)
    assert seen == [None, cursor, None]


@pytest.mark.asyncio
async def test_cancelled_dispatch_drains_media_and_retains_discovery(
    setup_dispatch, schedule_table
):
    queue, run, wire = schedule(setup_dispatch)
    await queue.enqueue(run.spec.run_id)
    original = queue.dispatcher._host
    entered = asyncio.Event()

    @asynccontextmanager
    async def stalled(candidate, guard):
        async with original(candidate, guard) as host:
            generate = host.provider.generate

            async def blocked(*args, **kwargs):
                entered.set()
                await asyncio.Event().wait()
                return await generate(*args, **kwargs)

            host.provider.generate = blocked
            yield host

    queue.dispatcher._host = stalled
    task = asyncio.create_task(queue.process_page(run.spec.run_id.int % 16))
    await asyncio.wait_for(entered.wait(), 15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(queue.dispatcher.scratch_root.iterdir()) == []
    assert (await queue._state(run)).job_state == "RUNNING"
    page = await confirmed_page(queue, run.spec.run_id.int % 16)
    assert page.attempts[0].status == "running" and len(wire.generates()) == 0
