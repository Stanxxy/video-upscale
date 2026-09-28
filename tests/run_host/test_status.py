import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from bjj_analysis_observer import lifecycle as analysis_status
from bjj_analysis_observer.lifecycle import (
    StatusEvidenceError,
    StatusObservationChanged,
)
from shared_lib.infrastructure.annotation_store import AnnotationNotFound
from test_dispatch import dispatch_tables, setup_dispatch
from test_progress import nodes, positive_transport
from test_worker import leases

from service.analysis_gemini import GeminiBoundaryError
from service.analysis_run_worker import KeyspacesRunLease


async def registered(dispatch, run, *, running=False):
    lease = KeyspacesRunLease(
        run,
        dispatch.session,
        keyspace=dispatch.keyspace,
        authority=dispatch._authority,
        lease_seconds=dispatch.lease_seconds,
        timeout=dispatch.checkpoint_timeout,
    )
    await lease.register()
    if running:
        await lease.acquire()
    return lease


def update(dispatch, run, assignment, values):
    # Only the UUID-namespaced disposable native fixture table.
    dispatch.session.execute(
        f"UPDATE {dispatch.keyspace}.job_lifecycle SET {assignment} WHERE job_id=%s",
        values + [str(run.spec.run_id)],
    )


def fixed_clock(monkeypatch, now):
    original = analysis_status.read_status

    async def read(*args, **kwargs):
        return await original(*args, **kwargs, clock=lambda: now)

    monkeypatch.setattr("service.analysis_run_dispatch.read_status", read)


@pytest.mark.asyncio
async def test_unregistered_pending_and_interrupted_are_not_running(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    view = await dispatch.status(run.spec.run_id)
    assert view.lifecycle.recorded_state == "NOT_REGISTERED"
    assert view.lifecycle.last_heartbeat_at is None
    assert view.lifecycle.heartbeat_freshness == "not_running"
    lease = await registered(dispatch, run)
    pending = await dispatch.status(run.spec.run_id)
    assert pending.lifecycle.recorded_state == "PENDING"
    assert pending.lifecycle.heartbeat_freshness == "not_running"
    assert pending.lifecycle.lease_expires_at is None
    await lease.acquire()
    await lease.finish(completed=False)
    interrupted = await dispatch.status(run.spec.run_id)
    assert interrupted.lifecycle.recorded_state == "INTERRUPTED"
    assert interrupted.lifecycle.heartbeat_freshness == "not_running"
    assert interrupted.lifecycle.completed_at is None
    assert wire.calls == [] and list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "offset,freshness", [(59.999, "recent"), (60, "expired"), (61, "expired")]
)
async def test_lease_expiry_boundary_is_read_only(
    setup_dispatch, monkeypatch, offset, freshness
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=True)
    before = await lease._read()
    heartbeat = before.last_heartbeat_at.replace(tzinfo=timezone.utc)
    fixed_clock(monkeypatch, heartbeat + timedelta(seconds=offset))
    view = await dispatch.status(run.spec.run_id)
    assert view.lifecycle.recorded_state == "RUNNING"
    assert view.lifecycle.heartbeat_freshness == freshness
    assert view.lifecycle.lease_expires_at == heartbeat + timedelta(seconds=60)
    assert await lease._read() == before
    assert wire.calls == [] and list(dispatch.scratch_root.iterdir()) == []
    assert "position-attempt:" not in str(asdict(view))


@pytest.mark.asyncio
async def test_status_reads_real_active_dag_and_completed_lifecycle(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    entered, release = asyncio.Event(), asyncio.Event()
    positive_transport(dispatch, run, wire, entered, release)
    task = asyncio.create_task(dispatch.dispatch(run.spec.run_id))
    wait = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait(
            (wait, task), timeout=30, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
        assert wait in done
        view = await dispatch.status(run.spec.run_id)
        assert view.lifecycle.recorded_state == "RUNNING"
        assert view.lifecycle.heartbeat_freshness == "recent"
        node = nodes(view.progress)["position_critique"]
        assert (node.recorded_status, node.completed_items, node.total_items) == (
            "running",
            0,
            1,
        )
        release.set()
        await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        wait.cancel()
        await asyncio.gather(task, wait, return_exceptions=True)
    count = len(wire.calls)
    view = await dispatch.status(run.spec.run_id)
    assert view.lifecycle.recorded_state == "COMPLETED"
    assert view.lifecycle.heartbeat_freshness == "not_running"
    assert view.lifecycle.completed_at is not None
    assert nodes(view.progress)["dag_complete"].recorded_status == "succeeded"
    assert nodes(view.progress)["provider_cleanup"].recorded_status == "not_started"
    assert len(wire.calls) == count and list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
async def test_uncertain_provider_failure_preserves_node_and_lifecycle_distinction(
    setup_dispatch,
    monkeypatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original = dispatch.transport

    async def handler(request):
        if ":generateContent" in str(request.url):
            return httpx.Response(503, json={"error": "uncertain"})
        return await original.handle_async_request(request)

    dispatch.transport = httpx.MockTransport(handler)
    with pytest.raises(GeminiBoundaryError, match="HTTP 503; raw receipt retained"):
        await dispatch.dispatch(run.spec.run_id)
    view = await dispatch.status(run.spec.run_id)
    assert view.lifecycle.recorded_state == "RUNNING"
    assert view.lifecycle.heartbeat_freshness == "recent"
    node = nodes(view.progress)["position_scan"]
    assert node.recorded_status == "failed"
    assert node.checkpoints[0].error_code == "DISPATCH_UNCERTAIN"
    before = dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    ).one()
    count = len(wire.calls)
    fixed_clock(monkeypatch, view.lifecycle.lease_expires_at)
    stale = await dispatch.status(run.spec.run_id)
    assert stale.lifecycle.recorded_state == "RUNNING"
    assert stale.lifecycle.heartbeat_freshness == "expired"
    assert stale.progress.nodes == view.progress.nodes
    assert len(wire.calls) == count
    assert (
        dispatch.session.execute(
            f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
            [str(run.spec.run_id)],
        ).one()
        == before
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("video_id", "wrong-video", "identity differs"),
        ("user_id", "wrong-owner", "identity differs"),
        ("pipeline_kind", "wrong-workflow", "identity differs"),
        ("job_state", "FAILED", "unsupported"),
        ("owner_instance_id", "", "attempt identity"),
        ("last_heartbeat_at", None, "timestamp"),
        ("completed_at", datetime(2026, 1, 1), "completion evidence"),
    ],
)
async def test_corrupt_lifecycle_fails_closed(setup_dispatch, field, value, message):
    dispatch, run, _, _, _, _ = setup_dispatch
    await registered(dispatch, run, running=True)
    update(dispatch, run, f"{field}=%s", [value])
    with pytest.raises(StatusEvidenceError, match=message):
        await dispatch.status(run.spec.run_id)


@pytest.mark.asyncio
async def test_future_heartbeat_and_clock_rollback_are_not_recent(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=True)
    row = await lease._read()
    heartbeat = row.last_heartbeat_at.replace(tzinfo=timezone.utc)
    fixed_clock(monkeypatch, heartbeat - timedelta(seconds=1))
    with pytest.raises(StatusEvidenceError, match="future"):
        await dispatch.status(run.spec.run_id)
    original = analysis_status.read_status
    times = iter([heartbeat + timedelta(seconds=2), heartbeat + timedelta(seconds=1)])

    async def backward(*args, **kwargs):
        return await original(*args, **kwargs, clock=lambda: next(times))

    monkeypatch.setattr("service.analysis_run_dispatch.read_status", backward)
    with pytest.raises(StatusEvidenceError, match="clock moved backwards"):
        await dispatch.status(run.spec.run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["claim", "owner", "finish"])
async def test_lifecycle_change_during_progress_requires_new_observation(
    setup_dispatch, monkeypatch, change
):
    dispatch, run, _, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=change != "claim")
    original = analysis_status.read_progress

    async def changed(*args, **kwargs):
        result = await original(*args, **kwargs)
        if change == "claim":
            await lease.acquire()
        elif change == "owner":
            update(
                dispatch,
                run,
                "owner_instance_id=%s",
                ["position-attempt:" + uuid4().hex],
            )
        else:
            await lease.finish(completed=False)
        return result

    monkeypatch.setattr(analysis_status, "read_progress", changed)
    with pytest.raises(StatusObservationChanged, match="changed during"):
        await dispatch.status(run.spec.run_id)


@pytest.mark.asyncio
async def test_same_attempt_heartbeat_renewal_is_allowed(setup_dispatch, monkeypatch):
    dispatch, run, _, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=True)
    before = await lease._read()
    original = analysis_status.read_progress

    async def renew(*args, **kwargs):
        result = await original(*args, **kwargs)
        await lease.heartbeat()
        return result

    monkeypatch.setattr(analysis_status, "read_progress", renew)
    view = await dispatch.status(run.spec.run_id)
    assert view.lifecycle.last_heartbeat_at > before.last_heartbeat_at.replace(
        tzinfo=timezone.utc
    )
    assert view.lifecycle.heartbeat_freshness == "recent"


@pytest.mark.asyncio
async def test_expiry_is_evaluated_after_final_authority_io(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=True)
    heartbeat = (await lease._read()).last_heartbeat_at.replace(tzinfo=timezone.utc)
    current = heartbeat + timedelta(seconds=59)
    original_status = analysis_status.read_status
    original_read = analysis_status.KeyspacesLifecycleReader.read
    reads = 0

    async def read(reader):
        nonlocal reads
        result = await original_read(reader)
        reads += 1
        return result

    async def guard(candidate):
        nonlocal current
        assert candidate == run
        if reads == 2:
            current = heartbeat + timedelta(seconds=60)

    async def status(*args, **kwargs):
        return await original_status(*args, **kwargs, clock=lambda: current)

    dispatch.authority = guard
    monkeypatch.setattr(analysis_status.KeyspacesLifecycleReader, "read", read)
    monkeypatch.setattr("service.analysis_run_dispatch.read_status", status)
    assert (
        await dispatch.status(run.spec.run_id)
    ).lifecycle.heartbeat_freshness == "expired"


@pytest.mark.asyncio
async def test_final_authority_revocation_returns_no_status(
    setup_dispatch, monkeypatch
):
    dispatch, run, _, _, _, revoke = setup_dispatch
    original = analysis_status.KeyspacesLifecycleReader.read
    count = 0

    async def read(reader):
        nonlocal count
        result = await original(reader)
        count += 1
        if count == 2:
            revoke()
        return result

    monkeypatch.setattr(analysis_status.KeyspacesLifecycleReader, "read", read)
    with pytest.raises(PermissionError, match="revoked"):
        await dispatch.status(run.spec.run_id)


@pytest.mark.asyncio
async def test_completed_requires_durable_dag_evidence(setup_dispatch):
    dispatch, run, _, _, _, _ = setup_dispatch
    lease = await registered(dispatch, run, running=True)
    await lease.finish(completed=True)
    with pytest.raises(StatusEvidenceError, match="completed host evidence"):
        await dispatch.status(run.spec.run_id)


@pytest.mark.asyncio
async def test_unknown_run_is_not_an_empty_status(setup_dispatch):
    dispatch, _, wire, _, _, _ = setup_dispatch
    with pytest.raises(AnnotationNotFound):
        await dispatch.status(uuid4())
    assert wire.calls == []
