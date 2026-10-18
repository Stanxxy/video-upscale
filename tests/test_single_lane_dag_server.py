import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from shared_lib.infrastructure.dag_internal_transport import (
    DagServiceKey,
    signed_headers,
)
from shared_lib.infrastructure.dag_service_runtime import DagListenerSettings
from service.current_account_dag_server import (
    AccountOnlyDagRunAuthority,
    current_account_engine_service_runtime,
)
from service.single_lane_dag_server import (
    SingleLaneDagRunAuthority,
    single_lane_authority_client,
    single_lane_engine_service_runtime,
    single_lane_run_recovery,
)
from tests.test_current_account_dag_server import _dispatcher


def _session_authority(secret="a" * 40):
    return single_lane_authority_client(
        "http://127.0.0.1:1",
        access_key=DagServiceKey("trial-session-access", secret, "session"),
        audience="trial-session-access",
        local_test=True,
    )


def _runtime(dispatcher, *, enqueue_owner="session", enqueue_secret="e" * 40, **extra):
    return single_lane_engine_service_runtime(
        dispatcher,
        SimpleNamespace(publish=lambda **_kwargs: {"MessageId": "not-used"}),
        listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
        enqueue_key=DagServiceKey("trial-session-enqueue", enqueue_secret, enqueue_owner),
        enqueue_audience="trial-session-enqueue",
        observe_schedule=lambda _page: asyncio.sleep(0),
        observe_delivery=lambda _page: asyncio.sleep(0),
        on_failure=lambda _name: asyncio.sleep(0),
        interval_seconds=1,
        **extra,
    )


async def _enqueue(runtime, key):
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.application),
        base_url="http://trial-engine.test",
    ) as client:
        path = f"/internal/dag/runs/{uuid4()}/enqueue"
        return await client.post(
            path,
            headers=signed_headers(key, "trial-session-enqueue", path),
            content=b"",
        )


def test_session_authority_contains_only_session_and_denies_account_runs():
    authority = _session_authority()
    assert authority.owner_kind == "session"
    assert set(authority.clients) == {"session"}

    run = SimpleNamespace(
        spec=SimpleNamespace(owner=SimpleNamespace(kind="account"), run_id=uuid4())
    )
    with pytest.raises(PermissionError, match="unsupported authority"):
        asyncio.run(authority(run))

    with pytest.raises(TypeError, match="signed authority client"):
        SingleLaneDagRunAuthority(object())


def test_session_runtime_binds_session_workers_recovery_and_enqueue_key():
    dispatcher = _dispatcher(_session_authority())
    recovery = single_lane_run_recovery(dispatcher)
    runtime = _runtime(
        dispatcher,
        recovery=recovery,
        observe_recovery=lambda _page: asyncio.sleep(0),
    )

    assert recovery.schedule.owner_kind == "session"
    assert set(runtime.workers) == {"session_schedule", "session_delivery", "session_recovery"}
    assert runtime.application.audience == "trial-session-enqueue"
    assert [key.owner_kind for key in runtime.application.keys.values()] == ["session"]


def test_session_enqueue_endpoint_rejects_a_valid_account_credential():
    runtime = _runtime(_dispatcher(_session_authority()))
    account_key = DagServiceKey("account-enqueue", "k" * 40, "account")

    assert asyncio.run(_enqueue(runtime, account_key)).status_code == 401


def test_session_runtime_rejects_account_shared_or_incomplete_configuration():
    dispatcher = _dispatcher(_session_authority())
    with pytest.raises(ValueError, match="distinct session enqueue"):
        _runtime(dispatcher, enqueue_secret="a" * 40)
    with pytest.raises(ValueError, match="session enqueue"):
        _runtime(dispatcher, enqueue_owner="account")

    dispatcher.allowed_buckets = frozenset({"current-source"})
    with pytest.raises(ValueError, match="session dispatcher resources are incomplete"):
        _runtime(dispatcher)

    with pytest.raises(ValueError, match="single-lane signed authority"):
        _runtime(_dispatcher(SimpleNamespace(clients={"session": object()})))


def test_account_binding_refuses_a_session_lane_dispatcher():
    dispatcher = _dispatcher(_session_authority())
    with pytest.raises(ValueError, match="account-only signed authority"):
        current_account_engine_service_runtime(
            dispatcher,
            SimpleNamespace(publish=lambda **_kwargs: None),
            listener_settings=DagListenerSettings("127.0.0.1", 0, local_test=True),
            enqueue_key=DagServiceKey("account-enqueue", "e" * 40, "account"),
            enqueue_audience="account-enqueue",
            observe_schedule=lambda _page: asyncio.sleep(0),
            observe_delivery=lambda _page: asyncio.sleep(0),
            on_failure=lambda _name: asyncio.sleep(0),
        )
    assert not isinstance(dispatcher.authority, AccountOnlyDagRunAuthority)
