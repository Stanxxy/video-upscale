"""Tests for post-restart drain of orphaned PENDING jobs (Keyspaces + local worker)."""

import json
from unittest.mock import AsyncMock

import pytest

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.models import TrackRequest
from service.analysis_settings import resolve_analysis_settings
from service.analysis_keyspaces_enums import JobState
from service import routes as routes_mod


def _minimal_track_request() -> dict:
    return {"bucket": "b", "key": "k.mp4"}


def _admitted_track_request() -> dict:
    return resolve_analysis_settings(
        TrackRequest(**_minimal_track_request()),
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_hydrate_job_registers_fixed_id():
    store = InMemoryJobStore()
    req = TrackRequest(**_minimal_track_request())
    a = await store.hydrate_job("fixed-id", req)
    b = await store.hydrate_job("fixed-id", req)
    assert a.job_id == b.job_id == "fixed-id"
    assert store.get_request("fixed-id") is req


@pytest.mark.asyncio
async def test_drain_schedules_pending_after_claim(monkeypatch):
    job_store = InMemoryJobStore()
    jobs_store = AsyncMock()
    jobs_store.list_active_recovery_index_rows_newest_first = AsyncMock(
        return_value=[
            {"job_id": "j-repl", "video_id": "v1"},
            {"job_id": "j-repl", "video_id": "v1"},
        ],
    )
    jobs_store.get_lifecycle = AsyncMock(
        return_value={
            "job_id": "j-repl",
            "video_id": "v1",
            "job_state": "PENDING",
            "owner_instance_id": "old-owner",
            "replacement_job_id": None,
        },
    )
    jobs_store.get_request = AsyncMock(
        return_value=json.dumps(_admitted_track_request()),
    )
    jobs_store.claim_pending_job_takeover = AsyncMock(return_value=True)

    scheduled: list[tuple[str, TrackRequest]] = []

    def fake_schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append((job_id, request))

    cfg = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    routes_mod.init_routes(cfg, job_store, jobs_store, instance_id="new-1")
    monkeypatch.setattr(routes_mod, "_schedule_job", fake_schedule)

    await routes_mod.drain_orphan_pending_jobs_on_startup("new-1")

    jobs_store.claim_pending_job_takeover.assert_awaited_once()
    assert len(scheduled) == 1
    assert scheduled[0][0] == "j-repl"
    assert scheduled[0][1].bucket == "b"
    assert (await job_store.get_job("j-repl")).job_id == "j-repl"


@pytest.mark.asyncio
async def test_drain_skips_non_pending(monkeypatch):
    job_store = InMemoryJobStore()
    jobs_store = AsyncMock()
    jobs_store.list_active_recovery_index_rows_newest_first = AsyncMock(
        return_value=[{"job_id": "j1"}],
    )
    jobs_store.get_lifecycle = AsyncMock(
        return_value={
            "job_id": "j1",
            "job_state": "RUNNING",
            "owner_instance_id": "x",
            "replacement_job_id": None,
        },
    )

    scheduled: list[str] = []

    def fake_schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append(job_id)

    cfg = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    routes_mod.init_routes(cfg, job_store, jobs_store, instance_id="srv")
    monkeypatch.setattr(routes_mod, "_schedule_job", fake_schedule)

    await routes_mod.drain_orphan_pending_jobs_on_startup("srv")

    assert scheduled == []
    jobs_store.claim_pending_job_takeover.assert_not_called()


@pytest.mark.asyncio
async def test_drain_skips_when_claim_lost(monkeypatch):
    job_store = InMemoryJobStore()
    jobs_store = AsyncMock()
    jobs_store.list_active_recovery_index_rows_newest_first = AsyncMock(
        return_value=[{"job_id": "j1"}],
    )
    jobs_store.get_lifecycle = AsyncMock(
        return_value={
            "job_id": "j1",
            "job_state": "PENDING",
            "owner_instance_id": "other",
            "replacement_job_id": None,
        },
    )
    jobs_store.get_request = AsyncMock(return_value=json.dumps(_admitted_track_request()))
    jobs_store.claim_pending_job_takeover = AsyncMock(return_value=False)

    scheduled: list[str] = []

    def fake_schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append(job_id)

    cfg = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    routes_mod.init_routes(cfg, job_store, jobs_store, instance_id="srv")
    monkeypatch.setattr(routes_mod, "_schedule_job", fake_schedule)

    await routes_mod.drain_orphan_pending_jobs_on_startup("srv")

    assert scheduled == []


@pytest.mark.asyncio
async def test_drain_marks_pre_r4_request_failed_after_claim(monkeypatch):
    job_store = InMemoryJobStore()
    jobs_store = AsyncMock()
    jobs_store.list_active_recovery_index_rows_newest_first = AsyncMock(
        return_value=[{"job_id": "old-job"}],
    )
    jobs_store.get_lifecycle = AsyncMock(
        return_value={
            "job_id": "old-job",
            "job_state": "PENDING",
            "owner_instance_id": "old-owner",
            "replacement_job_id": None,
        },
    )
    jobs_store.get_request = AsyncMock(return_value=json.dumps(_minimal_track_request()))
    jobs_store.claim_pending_job_takeover = AsyncMock(return_value=True)
    jobs_store.set_state = AsyncMock(return_value=True)
    routes_mod.init_routes(ServiceConfig(), job_store, jobs_store, instance_id="new-owner")
    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda *args: scheduled.append(args))

    await routes_mod.drain_orphan_pending_jobs_on_startup("new-owner")

    assert scheduled == []
    jobs_store.set_state.assert_awaited_once()
    args = jobs_store.set_state.await_args
    assert args.args[:2] == ("old-job", JobState.FAILED)
    assert "effective analysis settings snapshot" in args.kwargs["error_message"]
