import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from shared_lib.infrastructure.dag_internal_transport import (
    DagInternalClient,
    DagServiceKey,
    signed_headers,
)
from shared_lib.infrastructure.dag_service_runtime import DagListenerSettings
from shared_lib.infrastructure.result_publication_table import ResultPublicationTable
from service.current_account_dag_server import (
    AccountOnlyDagRunAuthority,
    current_account_position_dispatcher,
    current_account_engine_service_runtime,
)


def _account_client(owner="account", secret="a" * 40):
    return DagInternalClient(
        "http://127.0.0.1:1",
        key=DagServiceKey("current-account-access", secret, owner),
        audience="current-account-access",
        local_test=True,
    )


class EmptySession:
    def __init__(self):
        self.calls = 0

    def execute(self, *_args, **_kwargs):
        self.calls += 1
        return []


class EmptyOutbox:
    def __init__(self, proposals):
        self.proposals = proposals
        self.table = SimpleNamespace(name="current_publications")
        self.calls = 0

    async def pending_candidates_page(self, _shard, *, after=None, limit=20):
        self.calls += 1
        return [], None


def _dispatcher(authority):
    proposals = SimpleNamespace(tables=SimpleNamespace(runs="current_runs"))
    return SimpleNamespace(
        authority=authority,
        proposals=proposals,
        outbox=EmptyOutbox(proposals),
        bucket="current-artifacts",
        allowed_buckets=frozenset(
            {"current-source", "current-artifacts", "current-references"}
        ),
        qualified_models={"gemini-3.8-flash": ["gemini-3.8-flash"]},
        api_key="explicit-test-provider-key",
        session=EmptySession(),
        keyspace="current_account_test",
        checkpoint_timeout=1,
    )


def _runtime(dispatcher, *, enqueue_owner="account", enqueue_secret="e" * 40):
    return current_account_engine_service_runtime(
        dispatcher,
        SimpleNamespace(publish=lambda **_kwargs: {"MessageId": "not-used"}),
        listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
        enqueue_key=DagServiceKey(
            "current-account-enqueue", enqueue_secret, enqueue_owner
        ),
        enqueue_audience="current-account-enqueue",
        observe_schedule=lambda _page: asyncio.sleep(0),
        observe_delivery=lambda _page: asyncio.sleep(0),
        on_failure=lambda _name: asyncio.sleep(0),
        interval_seconds=1,
    )


def test_account_authority_contains_only_account_and_denies_other_owner():
    client = _account_client()
    authority = AccountOnlyDagRunAuthority(client)
    assert authority.clients == {"account": client}

    run = SimpleNamespace(
        spec=SimpleNamespace(owner=SimpleNamespace(kind="session"), run_id=uuid4())
    )
    with pytest.raises(PermissionError, match="unsupported authority"):
        asyncio.run(authority(run))

    session_client = _account_client(owner="session", secret="s" * 40)
    with pytest.raises(ValueError, match="account-bound"):
        AccountOnlyDagRunAuthority(session_client)


def test_runtime_binds_single_account_schedule_delivery_and_enqueue_key():
    authority = AccountOnlyDagRunAuthority(_account_client())
    dispatcher = _dispatcher(authority)
    runtime = _runtime(dispatcher)

    assert set(runtime.workers) == {"account_schedule", "account_delivery"}
    assert runtime.application.action == "enqueue"
    assert runtime.application.audience == "current-account-enqueue"
    assert [key.owner_kind for key in runtime.application.keys.values()] == ["account"]

    async def exercise_workers():
        schedule_pages = []
        delivery_pages = []

        async def observe_schedule(page):
            schedule_pages.append(page)

        async def observe_delivery(page):
            delivery_pages.append(page)

        observed_runtime = current_account_engine_service_runtime(
            dispatcher,
            SimpleNamespace(publish=lambda **_kwargs: {"MessageId": "not-used"}),
            listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
            enqueue_key=DagServiceKey(
                "current-account-enqueue", "e" * 40, "account"
            ),
            enqueue_audience="current-account-enqueue",
            observe_schedule=observe_schedule,
            observe_delivery=observe_delivery,
            on_failure=lambda _name: asyncio.sleep(0),
            interval_seconds=1,
        )
        schedule_stop = asyncio.Event()
        schedule_task = asyncio.create_task(
            observed_runtime.workers["account_schedule"](schedule_stop)
        )
        while len(schedule_pages) < 16:
            await asyncio.sleep(0.001)
        schedule_stop.set()
        await schedule_task

        delivery_stop = asyncio.Event()
        delivery_task = asyncio.create_task(
            observed_runtime.workers["account_delivery"](delivery_stop)
        )
        while len(delivery_pages) < 16:
            await asyncio.sleep(0.001)
        delivery_stop.set()
        await delivery_task
        assert dispatcher.session.calls == 16
        assert dispatcher.outbox.calls == 16
        assert all(page.error_code is None for page in schedule_pages)
        assert all(page.error_code is None for page in delivery_pages)

    asyncio.run(exercise_workers())


def test_private_enqueue_endpoint_rejects_valid_session_credential():
    authority = AccountOnlyDagRunAuthority(_account_client())
    runtime = _runtime(_dispatcher(authority))
    session_key = DagServiceKey("session-enqueue", "s" * 40, "session")

    async def request():
        import httpx

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=runtime.application),
            base_url="http://current-engine.test",
        ) as client:
            path = f"/internal/dag/runs/{uuid4()}/enqueue"
            response = await client.post(
                path,
                headers=signed_headers(
                    session_key,
                    "current-account-enqueue",
                    path,
                ),
                content=b"",
            )
            return response

    response = asyncio.run(request())
    assert response.status_code == 401


def test_runtime_rejects_non_account_or_shared_credentials_and_missing_resources():
    authority = AccountOnlyDagRunAuthority(_account_client())
    dispatcher = _dispatcher(authority)

    with pytest.raises(ValueError, match="distinct account enqueue"):
        _runtime(dispatcher, enqueue_secret="a" * 40)
    with pytest.raises(ValueError, match="account enqueue"):
        _runtime(dispatcher, enqueue_owner="session")

    dispatcher.allowed_buckets = frozenset({"current-source"})
    with pytest.raises(ValueError, match="resources are incomplete"):
        _runtime(dispatcher)

    session_only = _dispatcher(SimpleNamespace(clients={"session": object()}))
    with pytest.raises(ValueError, match="account-only signed authority"):
        _runtime(session_only)


def test_dispatcher_binds_current_source_artifact_and_reference_buckets(tmp_path):
    from shared_lib.infrastructure.admitted_dynamo_client import AdmittedDynamoClient

    binding = SimpleNamespace(table_name="current_publications")
    identity = SimpleNamespace(
        coverage_table="current_control",
        table_bindings=(binding,),
    )
    admission = SimpleNamespace(
        namespace_id="current-namespace",
        incarnation="current-incarnation",
        admission_generation=1,
        manifest_sha256="current-manifest",
        admitted_writer_kinds=("annotation",),
        identity=identity,
    )
    capability = SimpleNamespace(
        namespace_id=admission.namespace_id,
        namespace_incarnation=admission.incarnation,
        admission_generation=admission.admission_generation,
        manifest_sha256=admission.manifest_sha256,
        writer_kinds=admission.admitted_writer_kinds,
    )
    facade = AdmittedDynamoClient(
        object(), SimpleNamespace(namespace=identity.coverage_table),
        admission=admission, capability=capability,
    )
    proposals = SimpleNamespace(
        annotations=SimpleNamespace(client=facade),
        tables=SimpleNamespace(runs="current_runs"),
    )
    components = SimpleNamespace(proposals=proposals)
    dispatcher = current_account_position_dispatcher(
        proposals=proposals,
        components=components,
        publication_table=ResultPublicationTable("current_publications"),
        topic_arn="arn:aws:sns:us-east-1:000000000000:current-account-results",
        authority=AccountOnlyDagRunAuthority(_account_client()),
        session=EmptySession(),
        s3_client=SimpleNamespace(get_object=lambda **_kwargs: None),
        keyspace="current_account_test",
        source_bucket="current-source",
        artifact_bucket="current-artifacts",
        reference_bucket="current-references",
        scratch_root=tmp_path,
        max_source_bytes=128 * 1024 * 1024,
        api_key="explicit-test-provider-key",
        qualified_models={"gemini-3.8-flash": ["gemini-3.8-flash"]},
    )
    assert dispatcher.allowed_buckets == frozenset(
        {"current-source", "current-artifacts", "current-references"}
    )
    assert dispatcher.bucket == "current-artifacts"
    assert dispatcher.outbox.table.name == "current_publications"
    runtime = _runtime(dispatcher)
    assert set(runtime.workers) == {"account_schedule", "account_delivery"}
    assert runtime.application.action == "enqueue"

    for invalid_bucket in ("", "unapproved-artifacts"):
        dispatcher.bucket = invalid_bucket
        with pytest.raises(ValueError, match="resources are incomplete"):
            _runtime(dispatcher)


def test_dispatcher_rejects_wrong_authority_before_building_resources(tmp_path):
    proposals = SimpleNamespace(annotations=object())
    with pytest.raises(ValueError, match="current account authority"):
        current_account_position_dispatcher(
            proposals=proposals,
            components=SimpleNamespace(proposals=proposals),
            publication_table=ResultPublicationTable("current_publications"),
            topic_arn="arn:aws:sns:us-east-1:000000000000:current-account-results",
            authority=object(),
            session=EmptySession(),
            s3_client=SimpleNamespace(get_object=lambda **_kwargs: None),
            keyspace="current_account_test",
            source_bucket="current-source",
            artifact_bucket="current-artifacts",
            reference_bucket="current-references",
            scratch_root=tmp_path,
            max_source_bytes=128 * 1024 * 1024,
            api_key="explicit-test-provider-key",
            qualified_models={"gemini-3.8-flash": ["gemini-3.8-flash"]},
        )
