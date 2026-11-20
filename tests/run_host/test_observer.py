import asyncio
from uuid import uuid4

import pytest
from bjj_analysis_observer.observer import AuthorizedRunObserver
from shared_lib.infrastructure.annotation_store import AnnotationNotFound, encode_item
from shared_lib.models.analysis_contracts import PrincipalRef
from test_dispatch import dispatch_tables, setup_dispatch
from test_worker import leases

from service.analysis_run_worker import KeyspacesRunLease


def observer(dispatch, authority):
    return AuthorizedRunObserver(
        dispatch.proposals,
        dispatch.session,
        dispatch.s3,
        keyspace=dispatch.keyspace,
        bucket=dispatch.bucket,
        authorize=authority,
        lease_seconds=dispatch.lease_seconds,
        timeout=dispatch.checkpoint_timeout,
    )


@pytest.mark.asyncio
async def test_account_host_reads_completed_run_with_no_engine_call(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    await dispatch.dispatch(run.spec.run_id)
    count, calls = len(wire.calls), []

    async def authority(candidate, actor):
        assert candidate == run and actor == run.spec.owner
        calls.append(actor)

    view = await observer(dispatch, authority).status(
        run.spec.analysis_id, run.spec.run_id, actor=run.spec.owner
    )
    assert view.lifecycle.recorded_state == "COMPLETED"
    assert view.progress.run_id == run.spec.run_id and len(calls) > 2
    assert len(wire.calls) == count
    assert list(dispatch.scratch_root.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", ["principal", "kind", "analysis", "run"])
async def test_host_does_not_replace_requester_with_saved_owner(setup_dispatch, wrong):
    dispatch, run, wire, _, _, _ = setup_dispatch
    calls = []

    async def authority(candidate, actor):
        calls.append(actor)

    actor = run.spec.owner
    if wrong == "principal":
        actor = PrincipalRef(kind="account", principal_id=str(uuid4()))
    elif wrong == "kind":
        actor = PrincipalRef(kind="session", principal_id=actor.principal_id)
    with pytest.raises(AnnotationNotFound):
        await observer(dispatch, authority).status(
            uuid4() if wrong == "analysis" else run.spec.analysis_id,
            uuid4() if wrong == "run" else run.spec.run_id,
            actor=actor,
        )
    assert not calls and wire.calls == []


@pytest.mark.asyncio
async def test_host_reauthorizes_during_reads_and_does_not_return_after_revocation(
    setup_dispatch,
):
    dispatch, run, wire, _, _, _ = setup_dispatch
    count = 0

    async def authority(candidate, actor):
        nonlocal count
        count += 1
        if count == 5:
            raise AnnotationNotFound("host video access revoked")

    with pytest.raises(AnnotationNotFound, match="revoked"):
        await observer(dispatch, authority).status(
            run.spec.analysis_id, run.spec.run_id, actor=run.spec.owner
        )
    assert wire.calls == []


@pytest.mark.asyncio
async def test_host_observation_cancellation_never_claims_work(setup_dispatch):
    dispatch, run, wire, _, _, _ = setup_dispatch
    entered = asyncio.Event()

    async def authority(candidate, actor):
        entered.set()
        await asyncio.Future()

    task = asyncio.create_task(
        observer(dispatch, authority).status(
            run.spec.analysis_id, run.spec.run_id, actor=run.spec.owner
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire.calls == []


@pytest.mark.asyncio
async def test_session_owner_is_distinct_from_account_and_expires_without_ip_recovery(
    setup_dispatch,
):
    dispatch, account_run, wire, _, annotations, _ = setup_dispatch
    actor = PrincipalRef(kind="session", principal_id=str(uuid4()))
    source = await annotations.metadata(
        account_run.spec.analysis_id, actor=account_run.spec.owner
    )
    metadata = source.model_copy(
        update={"analysis_id": uuid4(), "scope_id": uuid4(), "owner": actor}
    )
    await annotations.create(metadata, actor=actor)
    spec = account_run.spec.model_copy(
        update={
            "run_id": uuid4(),
            "analysis_id": metadata.analysis_id,
            "scope_id": metadata.scope_id,
            "owner": actor,
        }
    )
    run = await dispatch.proposals.register_run(spec, actor=actor)

    async def authority(candidate, viewer):
        assert candidate == run and viewer == actor

    async def lease_authority(candidate):
        await authority(candidate, actor)

    lease = KeyspacesRunLease(
        run,
        dispatch.session,
        keyspace=dispatch.keyspace,
        authority=lease_authority,
        lease_seconds=60,
        timeout=10,
    )
    await lease.register()
    await lease.acquire()
    host = observer(dispatch, authority)
    status = await host.status(spec.analysis_id, spec.run_id, actor=actor)
    assert status.lifecycle.recorded_state == "RUNNING"
    assert status.lifecycle.heartbeat_freshness == "recent"
    for stranger in (
        PrincipalRef(kind="account", principal_id=actor.principal_id),
        PrincipalRef(kind="session", principal_id=str(uuid4())),
    ):
        with pytest.raises(AnnotationNotFound):
            await host.status(spec.analysis_id, spec.run_id, actor=stranger)
    annotations.client.update_item(
        TableName=annotations.tables.metadata,
        Key=encode_item({"analysis_id": str(spec.analysis_id)}),
        UpdateExpression="SET expires_at = :expired",
        ExpressionAttributeValues=encode_item({":expired": 1}),
    )
    with pytest.raises(AnnotationNotFound):
        await host.status(spec.analysis_id, spec.run_id, actor=actor)
    assert wire.calls == []
