"""Two native stores and real DAG/persistence; only provider I/O is a fixture."""

import asyncio
from contextlib import AsyncExitStack, contextmanager
from types import SimpleNamespace

import httpx
import pytest
from conftest import inputs as make_inputs
from shared_lib.models.analysis_run_record import RegisteredAIRunV1
from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagInternalEndpoint,
    DagRunAuthority,
    DagServiceKey,
)
from shared_lib.infrastructure.dag_service_runtime import DagListenerSettings
from test_dispatch import dispatch_tables, setup_dispatch
from test_delivery import delivery_queue, receive
from test_schedule import schedule_table
from test_worker import leases

from service.analysis_run_schedule import OwnerPositionRunSchedule, PositionRunSchedule
from service.dag_runtime import partitioned_engine_service_runtime


@pytest.mark.parametrize("shared", ["runs", "outbox"])
def test_owner_store_aliases_fail_before_runtime_creation(shared):
    # Configuration-only rejection; no native clients or worker substitutes.
    dispatchers = {
        owner: SimpleNamespace(
            proposals=SimpleNamespace(
                tables=SimpleNamespace(runs="same" if shared == "runs" else owner)
            ),
            outbox=SimpleNamespace(
                table=SimpleNamespace(name="same" if shared == "outbox" else owner)
            ),
        )
        for owner in ("account", "session")
    }
    with pytest.raises(ValueError, match="separate owner .* stores required"):
        partitioned_engine_service_runtime(
            dispatchers,
            dict.fromkeys(dispatchers),
            listener_settings=None,
            enqueue_keys=[],
            enqueue_audience="unused",
            observe_schedule=None,
            observe_delivery=None,
            on_failure=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("composition", ["runtime", "entrypoint_factory"])
async def test_two_store_signed_runtime_isolates_partitions_and_delivery(
    setup_dispatch,
    schedule_table,
    delivery_queue,
    s3,
    media_bytes,
    checkpoint_database,
    leases,
    tmp_path,
    monkeypatch,
    composition,
):
    errors = []
    process = OwnerPositionRunSchedule._process

    async def inspected(self, *args):
        try:
            return await process(self, *args)
        except Exception as error:
            errors.append(repr(error))
            raise

    monkeypatch.setattr(OwnerPositionRunSchedule, "_process", inspected)
    # Reuse production-shaped fixture construction, not a fake store/resolver.
    with (
        contextmanager(dispatch_tables.__wrapped__)(s3) as second_tables,
        contextmanager(delivery_queue_fixture)(s3) as second_queue,
    ):
        second_inputs = make_inputs.__wrapped__(s3, media_bytes)
        record = second_inputs[0].model_dump()
        record["spec"]["owner"] = {"kind": "session", "principal_id": "session-owner"}
        second_inputs = (RegisteredAIRunV1.model_validate(record), *second_inputs[1:])
        (tmp_path / "session").mkdir()
        second = await setup_dispatch_fixture(
            second_tables,
            second_inputs,
            media_bytes,
            s3,
            checkpoint_database,
            leases,
            tmp_path / "session",
        )
        cases = {"account": setup_dispatch, "session": second}
        queues = {"account": delivery_queue, "session": second_queue}
        dispatchers = {owner: case[0] for owner, case in cases.items()}
        access_keys = [
            DagServiceKey("account-access", "a" * 32, "account"),
            DagServiceKey("session-access", "s" * 32, "session"),
        ]
        enqueue_keys = [
            DagServiceKey("account-enqueue", "b" * 32, "account"),
            DagServiceKey("session-enqueue", "t" * 32, "session"),
        ]
        original = {
            owner: dispatcher.authority for owner, dispatcher in dispatchers.items()
        }

        async def authorize(run_id, owner):
            run = cases[owner][1]
            assert run.spec.run_id == run_id
            await original[owner](run)

        access_app = DagInternalEndpoint(
            action="authorize",
            audience="owner-access",
            keys=access_keys,
            handler=authorize,
        )
        clients = {
            key.owner_kind: DagInternalClient(
                "https://authority.internal",
                key=key,
                audience="owner-access",
                transport=httpx.ASGITransport(app=access_app),
            )
            for key in access_keys
        }
        authority = DagRunAuthority(**clients)
        for owner, dispatcher in dispatchers.items():
            dispatcher.authority = authority
            dispatcher.outbox.topic_arn = queues[owner][2]
        schedules = {
            owner: OwnerPositionRunSchedule(d, owner_kind=owner)
            for owner, d in dispatchers.items()
        }
        assert schedules["account"].index_state != schedules["session"].index_state
        for owner, schedule in schedules.items():
            run = cases[owner][1]
            legacy = PositionRunSchedule(dispatchers[owner])
            await legacy.enqueue(run.spec.run_id)
            await schedule.enqueue(run.spec.run_id)
            # Replay/restart uses exactly the same owner partition.
            await OwnerPositionRunSchedule(
                dispatchers[owner], owner_kind=owner
            ).enqueue(run.spec.run_id)
            legacy_rows = await legacy._execute(
                f"SELECT * FROM {legacy.index} WHERE recovery_state=%s AND heartbeat_bucket=%s AND last_heartbeat_at=%s AND job_id=%s",
                legacy._key(run),
            )
            assert len(legacy_rows) == 1

        completed, delivered = set(), {"account": [], "session": []}
        done = asyncio.Event()

        def ready():
            if completed == {"account", "session"} and all(
                len(v) == 2 for v in delivered.values()
            ):
                done.set()

        async def observe_schedule(owner, page):
            assert page.error_code is None
            for attempt in page.attempts:
                assert attempt.status == "completed", errors
                assert attempt.run_id == cases[owner][1].spec.run_id
                completed.add(owner)
            ready()

        async def observe_delivery(owner, page):
            for attempt in page.attempts:
                assert attempt.status == "acknowledged"
                assert attempt.run_id == cases[owner][1].spec.run_id
                delivered[owner].append(attempt)
            ready()

        failures = []

        async def failed(component):
            failures.append(component)

        runtime = partitioned_engine_service_runtime(
            dispatchers,
            {owner: q[0] for owner, q in queues.items()},
            listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
            enqueue_keys=enqueue_keys,
            enqueue_audience="owner-enqueue",
            observe_schedule=observe_schedule,
            observe_delivery=observe_delivery,
            on_failure=failed,
            interval_seconds=0.1,
        )
        owned = AsyncExitStack()
        closed = []
        if composition == "entrypoint_factory":
            import service.dag_server as server
            from service.dag_server import runtime_factory

            prepare = server.prepare_dispatchers
            composed = {}

            def observed_prepare(*args, **kwargs):
                result = prepare(*args, **kwargs)
                composed.update(result)
                return result

            monkeypatch.setattr(server, "prepare_dispatchers", observed_prepare)

            settings = dict(
                lanes={
                    owner: dict(
                        annotations=d.proposals.annotations.tables,
                        proposals=d.proposals.tables,
                        publication=d.outbox.table,
                        artifact_bucket=d.bucket,
                        allowed_buckets=d.allowed_buckets,
                        scratch_root=d.scratch_root,
                        topic_arn=d.outbox.topic_arn,
                        access_origin="https://authority.internal",
                        access_audience="owner-access",
                        access_key=access_keys[index],
                        enqueue_key=enqueue_keys[index],
                    )
                    for index, (owner, d) in enumerate(dispatchers.items())
                },
                keyspace=checkpoint_database[1],
                qualified_models=setup_dispatch[0].qualified_models,
                api_key="fixture-secret",
                max_source_bytes=10_000_000,
                listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
                enqueue_audience="owner-enqueue",
            )

            def native_clients(settings, stack):
                # Fixture owns these live clients; production owns/creates its own.
                stack.callback(closed.append, True)
                return dict(
                    dynamodb=setup_dispatch[0].proposals.annotations.client,
                    s3=s3[0],
                    sns=delivery_queue[0],
                    session=checkpoint_database[0],
                )

            async def observe(kind, owner, page):
                await (observe_schedule if kind == "schedule" else observe_delivery)(
                    owner, page
                )

            from botocore.exceptions import ClientError
            from shared_lib.infrastructure.result_publication_table import (
                ResultPublicationTable,
            )
            from uuid import uuid4

            invalid = {
                **settings,
                "lanes": {
                    **settings["lanes"],
                    "session": {
                        **settings["lanes"]["session"],
                        "publication": ResultPublicationTable(
                            "s39_absent_" + uuid4().hex
                        ),
                    },
                },
            }
            with pytest.raises(ClientError):
                async with runtime_factory(invalid, native_clients=native_clients)(
                    None, failed
                ):
                    pytest.fail("missing table must prevent runtime creation")
            assert closed == [True] and composed == {}
            assert all(case[2].generates() == [] for case in cases.values())
            closed.clear()

            runtime = await owned.enter_async_context(
                runtime_factory(
                    settings,
                    native_clients=native_clients,
                    access_transports={
                        owner: httpx.ASGITransport(app=access_app) for owner in cases
                    },
                    transports={owner: d.transport for owner, d in dispatchers.items()},
                    observe=observe,
                    interval_seconds=0.1,
                )(None, failed)
            )
            # Use the actual newly composed dispatchers for the remaining assertions.
            dispatchers = composed
            schedules = {
                owner: OwnerPositionRunSchedule(d, owner_kind=owner)
                for owner, d in dispatchers.items()
            }
        assert dispatchers["account"].slots is dispatchers["session"].slots
        for key in enqueue_keys:

            async def lose_ack(scope, receive, send):
                async def discarded(message):
                    if message["type"] == "http.response.start":
                        assert message["status"] == 204

                await runtime.application(scope, receive, discarded)
                raise httpx.ReadTimeout("fixture: committed enqueue ACK lost")

            run_id = cases[key.owner_kind][1].spec.run_id
            lost = DagInternalClient(
                "https://engine.internal",
                key=key,
                audience="owner-enqueue",
                transport=httpx.ASGITransport(app=lose_ack),
            )
            try:
                with pytest.raises(httpx.ReadTimeout):
                    await lost.call("enqueue", run_id)
            finally:
                await lost.aclose()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=runtime.application),
                base_url="https://engine.internal",
            ) as signed:
                from shared_lib.infrastructure.dag_internal_transport import (
                    signed_headers,
                )

                run_id = cases[key.owner_kind][1].spec.run_id
                path = f"/internal/dag/runs/{run_id}/enqueue"
                response = await signed.post(
                    path,
                    content=b"",
                    headers=signed_headers(key, "owner-enqueue", path),
                )
                assert response.status_code == 204
            schedule = schedules[key.owner_kind]
            rows = await schedule._execute(
                f"SELECT * FROM {schedule.index} WHERE recovery_state=%s AND heartbeat_bucket=%s AND last_heartbeat_at=%s AND job_id=%s",
                schedule._key(cases[key.owner_kind][1]),
            )
            assert len(rows) == 1

        # Observe real reads before workers start: a session credential must never
        # probe account storage, even when its requested UUID exists there.
        reads = []
        for owner, dispatcher in dispatchers.items():
            original_resolve = dispatcher.proposals.resolve_delivery_run

            def observed_resolver(owner, resolve):
                async def observed(run_id):
                    reads.append((owner, run_id))
                    return await resolve(run_id)

                return observed

            monkeypatch.setattr(
                dispatcher.proposals,
                "resolve_delivery_run",
                observed_resolver(owner, original_resolve),
            )
        wrong_lane = DagInternalClient(
            "https://engine.internal",
            key=enqueue_keys[1],
            audience="owner-enqueue",
            transport=httpx.ASGITransport(app=runtime.application),
        )
        try:
            with pytest.raises(PermissionError):
                await wrong_lane.call("enqueue", cases["account"][1].spec.run_id)
            assert reads == [("session", cases["account"][1].spec.run_id)]
        finally:
            await wrong_lane.aclose()
        # Hold the actual shared permit: neither lane may generate while occupied.
        await dispatchers["account"].slots.acquire()
        try:
            async with runtime.lifespan():
                session_client = DagInternalClient(
                    f"http://127.0.0.1:{runtime.bound_port}",
                    key=enqueue_keys[1],
                    audience="owner-enqueue",
                    local_test=True,
                )
                try:
                    with pytest.raises(PermissionError):
                        await session_client.call(
                            "enqueue", cases["account"][1].spec.run_id
                        )
                finally:
                    await session_client.aclose()
                assert all(case[2].generates() == [] for case in cases.values())
                dispatchers["account"].slots.release()
                await asyncio.wait_for(done.wait(), 45)
            for owner, case in cases.items():
                assert len(case[2].generates()) == 2
                messages = await receive(queues[owner][1], queues[owner][3], 2)
                assert all(
                    message.run_id == case[1].spec.run_id for message in messages
                )
                assert (
                    await schedules[owner].process_page(case[1].spec.run_id.int % 16)
                ).attempts == ()
                legacy = PositionRunSchedule(dispatchers[owner])
                legacy_rows = await legacy._execute(
                    f"SELECT * FROM {legacy.index} WHERE recovery_state=%s AND heartbeat_bucket=%s AND last_heartbeat_at=%s AND job_id=%s",
                    legacy._key(case[1]),
                )
                assert len(legacy_rows) == 1  # No implicit legacy discovery/retirement.
            assert failures == []
        finally:
            await owned.aclose()
            for client in clients.values():
                await client.aclose()
        if composition == "entrypoint_factory":
            assert closed == [True]


# Aliases preserve access to the fixture factories when test arguments shadow names.
setup_dispatch_fixture = setup_dispatch.__wrapped__
delivery_queue_fixture = delivery_queue.__wrapped__


@pytest.mark.asyncio
async def test_scheduler_cancellation_drains_native_driver_before_return(
    setup_dispatch,
    schedule_table,
):
    import threading
    from types import SimpleNamespace

    dispatcher = setup_dispatch[0]
    queue = OwnerPositionRunSchedule(dispatcher, owner_kind="account")
    original = dispatcher.session
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        rows = list(original.execute(*args, **kwargs))
        finished.set()
        return rows

    dispatcher.session = SimpleNamespace(execute=delayed)
    task = asyncio.create_task(
        queue._execute(
            f"SELECT * FROM {queue.index} WHERE recovery_state=%s AND heartbeat_bucket=%s",
            [queue.index_state, queue._bucket(0)],
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        dispatcher.session = original
