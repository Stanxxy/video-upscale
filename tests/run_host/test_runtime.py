"""Native engine runtime: signed ingress, actual DAG/store/queue, fixture provider."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagInternalEndpoint,
    DagRunAuthority,
    DagServiceKey,
)
from shared_lib.infrastructure.dag_service_runtime import DagListenerSettings
from test_delivery import delivery_queue, receive
from test_dispatch import dispatch_tables, setup_dispatch
from test_schedule import schedule_table
from test_worker import leases

from service.dag_runtime import engine_service_runtime


@pytest.mark.parametrize(
    "models,key",
    [
        ({}, "fixture"),
        ({"fixture-model": []}, "fixture"),
        ({"fixture-model": [""]}, "fixture"),
        ({"fixture-model": ["r1"]}, ""),
    ],
)
def test_missing_qualification_or_credential_fails_before_listener_creation(
    models, key
):
    # Pure configuration rejection, before any storage/transport is constructed.
    dispatcher = SimpleNamespace(qualified_models=models, api_key=key)
    with pytest.raises(ValueError):
        engine_service_runtime(
            dispatcher,
            None,
            listener_settings=None,
            enqueue_keys=[],
            enqueue_audience="unused",
            observe_schedule=None,
            observe_delivery=None,
            on_failure=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_observer", [None, "schedule", "delivery"])
async def test_signed_engine_runtime_executes_and_delivers_or_fails_closed(
    setup_dispatch, schedule_table, delivery_queue, fail_observer
):
    dispatcher, run, wire, components, _, _ = setup_dispatch
    sns, sqs, topic, queue = delivery_queue
    dispatcher.outbox.topic_arn = topic
    original_authority = dispatcher.authority
    access_keys = [
        DagServiceKey("account-access", "a" * 32, "account"),
        DagServiceKey("session-access", "s" * 32, "session"),
    ]
    enqueue_keys = [
        DagServiceKey("account-enqueue", "b" * 32, "account"),
        DagServiceKey("session-enqueue", "t" * 32, "session"),
    ]
    authorized = []

    async def authorize(run_id, owner):
        assert run_id == run.spec.run_id and owner == run.spec.owner.kind
        await original_authority(run)
        authorized.append(run_id)

    access_app = DagInternalEndpoint(
        action="authorize",
        audience="native-access",
        keys=access_keys,
        handler=authorize,
    )
    clients = {
        key.owner_kind: DagInternalClient(
            "https://fixture.internal",
            key=key,
            audience="native-access",
            transport=httpx.ASGITransport(app=access_app),
        )
        for key in access_keys
    }
    dispatcher.authority = DagRunAuthority(**clients)
    delivered, completed = asyncio.Event(), asyncio.Event()
    failures, acknowledgments = [], []

    async def observe_schedule(page):
        if fail_observer == "schedule":
            raise RuntimeError("controlled schedule observer failure")
        assert page.error_code is None
        for attempt in page.attempts:
            assert attempt.status == "completed"
            completed.set()

    async def observe_delivery(page):
        if fail_observer == "delivery":
            raise RuntimeError("controlled delivery observer failure")
        for attempt in page.attempts:
            assert attempt.status == "acknowledged"
            acknowledgments.append(attempt)
        if len(acknowledgments) == 2:
            delivered.set()

    async def failed(component):
        failures.append(component)

    runtime = engine_service_runtime(
        dispatcher,
        sns,
        listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
        enqueue_keys=enqueue_keys,
        enqueue_audience="native-enqueue",
        observe_schedule=observe_schedule,
        observe_delivery=observe_delivery,
        on_failure=failed,
        interval_seconds=0.1,
    )

    async def work():
        async with runtime.lifespan():
            if fail_observer:
                await asyncio.Event().wait()
            else:
                key = next(
                    k for k in enqueue_keys if k.owner_kind == run.spec.owner.kind
                )
                client = DagInternalClient(
                    f"http://127.0.0.1:{runtime.bound_port}",
                    key=key,
                    audience="native-enqueue",
                    local_test=True,
                )
                try:
                    assert (
                        await client.call("enqueue", run.spec.run_id) == run.spec.run_id
                    )
                    await asyncio.wait_for(completed.wait(), 30)
                    await asyncio.wait_for(delivered.wait(), 30)
                finally:
                    await client.aclose()

    try:
        if fail_observer:
            with pytest.raises(ExceptionGroup):
                await asyncio.wait_for(work(), 10)
            assert failures == [fail_observer] and wire.calls == []
        else:
            await work()
            assert runtime.state == "stopped" and failures == []
            messages = await receive(sqs, queue, 2)
            assert {m.payload.component for m in messages} == {
                "positions",
                "techniques",
            }
            for message in messages:
                await components.ingest(
                    run.spec.analysis_id, message, actor=run.spec.owner
                )
            assert len(wire.generates()) == 2 and len(authorized) >= 3
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", runtime.bound_port)
    finally:
        for client in clients.values():
            await client.aclose()
