import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
from bjj_analysis_core.execution import RecoveryRequired
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.analysis_run_dispatch import ProviderActions
from service.analysis_run_worker import LeaseLost


def state(dispatch, run):
    return dispatch.session.execute(
        f"SELECT * FROM {dispatch.keyspace}.job_lifecycle WHERE job_id=%s",
        [str(run.spec.run_id)],
    ).one()


def expire(dispatch, run):
    dispatch.session.execute(
        f"UPDATE {dispatch.keyspace}.job_lifecycle SET last_heartbeat_at=%s WHERE job_id=%s",
        [datetime.now(timezone.utc) - timedelta(seconds=61), str(run.spec.run_id)],
    )


async def interrupted(dispatch, run, node):
    original, receipts = dispatch._host, {}

    @asynccontextmanager
    async def fault(candidate, guard):
        async with original(candidate, guard) as host:
            save, retain = host.journal.compare_and_swap, host.provider.artifacts.retain

            async def retained(kind, identity, value):
                ref = await retain(kind, identity, value)
                receipts[kind] = ref
                return ref

            async def lost(expected, record):
                if record.progress.node_id == node and record.phase == "succeeded":
                    raise TimeoutError("host completion not committed")
                return await save(expected, record)

            host.journal.compare_and_swap = lost
            host.provider.artifacts.retain = retained
            yield host

    dispatch._host = fault
    try:
        with pytest.raises(TimeoutError):
            await dispatch.dispatch(
                run.spec.run_id,
                provider_actions=ProviderActions(cleanup=node == "provider_cleanup"),
            )
    finally:
        dispatch._host = original
    assert state(dispatch, run).job_state == "RUNNING"
    assert list(dispatch.scratch_root.iterdir()) == []
    return receipts["upload_finalize" if node == "provider_upload" else "file_delete"]


@pytest.mark.asyncio
async def test_explicit_cleanup_runs_inside_live_lease_after_durable_outbox(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original, checked = dispatch._host, []

    @asynccontextmanager
    async def observed(candidate, guard):
        async with original(candidate, guard) as host:
            remove = host.provider.delete_upload

            async def check(operation_id):
                await guard()
                assert state(dispatch, run).job_state == "RUNNING"
                assert (await host._read("dag_complete")).phase == "succeeded"
                for lane in ("positions", "techniques"):
                    publication, _ = await dispatch.outbox.read(run.spec.run_id, lane)
                    assert publication is not None
                checked.append(True)
                return await remove(operation_id)

            host.provider.delete_upload = check
            yield host

    dispatch._host = observed
    await dispatch.dispatch(
        run.spec.run_id, provider_actions=ProviderActions(cleanup=True)
    )
    assert checked == [True]
    assert state(dispatch, run).job_state == "COMPLETED"
    assert len(wire.generates()) == 2
    assert sum(call[0] == "DELETE" for call in wire.calls) == 1
    assert list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("node", ["provider_upload", "provider_cleanup"])
async def test_receipt_recovery_reclaims_expired_lease_and_completes_without_repeating_operation(
    setup_dispatch, node
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    receipt = await interrupted(dispatch, run, node)
    first_owner = state(dispatch, run).owner_instance_id
    actions = (
        ProviderActions(upload_receipt=receipt)
        if node == "provider_upload"
        else ProviderActions(cleanup=True, cleanup_receipt=receipt)
    )
    count = len(wire.calls)
    with pytest.raises(LeaseLost):
        await dispatch.dispatch(run.spec.run_id, recover=True, provider_actions=actions)
    assert len(wire.calls) == count
    expire(dispatch, run)
    await dispatch.dispatch(run.spec.run_id, recover=True, provider_actions=actions)
    current = state(dispatch, run)
    assert current.job_state == "COMPLETED" and current.owner_instance_id != first_owner
    assert len(wire.generates()) == 2
    assert sum(call[0] == "POST" and "/upload/" in call[1] for call in wire.calls) == 2
    assert sum(call[0] == "DELETE" for call in wire.calls) == (
        node == "provider_cleanup"
    )
    if node == "provider_cleanup":
        assert len(wire.calls) == count
    assert list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
async def test_pending_cleanup_cannot_be_bypassed_by_ordinary_recovery(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    receipt = await interrupted(dispatch, run, "provider_cleanup")
    count = len(wire.calls)
    expire(dispatch, run)
    with pytest.raises(RecoveryRequired, match="pending cleanup"):
        await dispatch.dispatch(run.spec.run_id, recover=True)
    assert state(dispatch, run).job_state == "RUNNING" and len(wire.calls) == count
    expire(dispatch, run)
    with pytest.raises(RecoveryRequired):
        await dispatch.dispatch(
            run.spec.run_id,
            recover=True,
            provider_actions=ProviderActions(cleanup=True),
        )
    assert state(dispatch, run).job_state == "RUNNING" and len(wire.calls) == count
    expire(dispatch, run)
    await dispatch.dispatch(
        run.spec.run_id,
        recover=True,
        provider_actions=ProviderActions(cleanup=True, cleanup_receipt=receipt),
    )
    assert state(dispatch, run).job_state == "COMPLETED" and len(wire.calls) == count


@pytest.mark.asyncio
async def test_failed_technique_never_triggers_requested_cleanup(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original = dispatch._host

    @asynccontextmanager
    async def fault(candidate, guard):
        async with original(candidate, guard) as host:
            generate = host.provider.generate

            async def fail(run, request, execution_id, call_id):
                if request.node_id == "technique_scan":
                    wire.status = 503
                return await generate(run, request, execution_id, call_id)

            host.provider.generate = fail
            yield host

    dispatch._host = fault
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await dispatch.dispatch(
            run.spec.run_id, provider_actions=ProviderActions(cleanup=True)
        )
    assert state(dispatch, run).job_state == "RUNNING"
    publication, _ = await dispatch.outbox.read(run.spec.run_id, "positions")
    assert publication is not None
    assert not any(call[0] == "DELETE" for call in wire.calls)


@pytest.mark.asyncio
async def test_revoked_recovery_never_claims_or_uses_receipt(setup_dispatch):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    receipt = await interrupted(dispatch, run, "provider_upload")
    expire(dispatch, run)
    owner, count = state(dispatch, run).owner_instance_id, len(wire.calls)
    revoke()
    with pytest.raises(PermissionError):
        await dispatch.dispatch(
            run.spec.run_id,
            recover=True,
            provider_actions=ProviderActions(upload_receipt=receipt),
        )
    assert state(dispatch, run).owner_instance_id == owner and len(wire.calls) == count


@pytest.mark.asyncio
async def test_terminal_run_is_not_reopened_for_cleanup(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    count = len(wire.calls)
    with pytest.raises(LeaseLost):
        await dispatch.dispatch(
            run.spec.run_id,
            recover=True,
            provider_actions=ProviderActions(cleanup=True),
        )
    assert state(dispatch, run).job_state == "COMPLETED" and len(wire.calls) == count


@pytest.mark.asyncio
async def test_receipt_requires_explicit_recovery_before_lifecycle_registration(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    with pytest.raises(ValueError, match="recover=True"):
        await dispatch.dispatch(
            run.spec.run_id,
            provider_actions=ProviderActions(upload_receipt=run.spec.source),
        )
    assert state(dispatch, run) is None and wire.calls == []


@pytest.mark.parametrize(
    "options",
    [
        dict(cleanup=1),
        dict(cleanup_receipt={"bucket": "bucket", "key": "key", "sha256": "0" * 64}),
        dict(
            upload_receipt={"bucket": "bucket", "key": "key", "sha256": "0" * 64},
            cleanup=True,
            cleanup_receipt={"bucket": "bucket", "key": "key", "sha256": "0" * 64},
        ),
    ],
)
def test_invalid_action_combinations_are_rejected(options):
    with pytest.raises(ValueError):
        ProviderActions(**options)


@pytest.mark.asyncio
async def test_cancelled_cleanup_drains_resources_and_does_not_complete(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    original, entered = dispatch._host, asyncio.Event()

    @asynccontextmanager
    async def fault(candidate, guard):
        async with original(candidate, guard) as host:

            async def blocked(operation_id):
                await guard()
                entered.set()
                await asyncio.Event().wait()

            host.provider.delete_upload = blocked
            yield host

    dispatch._host = fault
    task = asyncio.create_task(
        dispatch.dispatch(
            run.spec.run_id, provider_actions=ProviderActions(cleanup=True)
        )
    )
    await asyncio.wait_for(entered.wait(), 20)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    dispatch._host = original
    assert state(dispatch, run).job_state == "RUNNING"
    assert list(dispatch.scratch_root.iterdir()) == []
    count = len(wire.calls)
    expire(dispatch, run)
    with pytest.raises(RecoveryRequired):
        await dispatch.dispatch(
            run.spec.run_id,
            recover=True,
            provider_actions=ProviderActions(cleanup=True),
        )
    assert len(wire.calls) == count


@pytest.mark.asyncio
async def test_outbox_failure_blocks_cleanup_until_explicit_safe_recovery(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    prepare = dispatch.outbox.prepare

    async def unavailable(run_id, component, revision):
        if component == "techniques":
            raise TimeoutError("publication intent not committed")
        return await prepare(run_id, component, revision)

    dispatch.outbox.prepare = unavailable
    with pytest.raises(TimeoutError):
        await dispatch.dispatch(
            run.spec.run_id, provider_actions=ProviderActions(cleanup=True)
        )
    assert not any(call[0] == "DELETE" for call in wire.calls)
    assert state(dispatch, run).job_state == "RUNNING"
    dispatch.outbox.prepare = prepare
    expire(dispatch, run)
    await dispatch.dispatch(
        run.spec.run_id,
        recover=True,
        retry_failed=True,
        provider_actions=ProviderActions(cleanup=True),
    )
    assert state(dispatch, run).job_state == "COMPLETED"
    assert (
        len(wire.generates()) == 2
        and sum(call[0] == "DELETE" for call in wire.calls) == 1
    )


@pytest.mark.asyncio
async def test_authority_revoked_during_receipt_recovery_stops_before_model_work(
    setup_dispatch,
):
    dispatch, run, wire, _, _, revoke = setup_dispatch
    receipt = await interrupted(dispatch, run, "provider_upload")
    expire(dispatch, run)
    original = dispatch._host

    @asynccontextmanager
    async def fault(candidate, guard):
        async with original(candidate, guard) as host:
            request = host.provider._request

            async def denied(*args, **kwargs):
                result = await request(*args, **kwargs)
                revoke()
                return result

            host.provider._request = denied
            yield host

    dispatch._host = fault
    with pytest.raises(PermissionError):
        await dispatch.dispatch(
            run.spec.run_id,
            recover=True,
            provider_actions=ProviderActions(upload_receipt=receipt),
        )
    assert [call[0] for call in wire.calls] == ["POST", "POST", "GET"]
    assert state(dispatch, run).job_state == "RUNNING"
    assert list(dispatch.scratch_root.iterdir()) == []
