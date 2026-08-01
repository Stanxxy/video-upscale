"""Characterization tests for POST /track, GET /job/{id}, and GET /health."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from service import routes as routes_mod
from service.analysis_keyspaces_enums import JobState
from service.models import TrackRequest, TrackResponse
from service.routes import scheduling as scheduling_mod


@pytest.fixture()
def scheduled_jobs(monkeypatch):
    scheduled: list[tuple[str, TrackRequest]] = []

    def _schedule(job_id: str, request: TrackRequest) -> None:
        scheduled.append((job_id, request))

    async def _cleanup() -> None:
        return None

    monkeypatch.setattr(routes_mod, "_schedule_job", _schedule)
    # S12 Phase 1b opportunistic fix (item 15): create_track_job AWAITS
    # _cleanup_orphaned_tasks — the prior sync `lambda: None` stand-in was a
    # latent TypeError ("NoneType can't be used in 'await' expression"),
    # masked until now by this same test's earlier 422 (an invalid
    # video_id UUID, fixed above) short-circuiting before this call was ever
    # reached.
    monkeypatch.setattr(routes_mod, "_cleanup_orphaned_tasks", _cleanup)
    return scheduled


@pytest.mark.asyncio
async def test_post_track_creates_lifecycle_and_schedules_worker(
    service_client, service_components, scheduled_jobs,
):
    _, _, jobs_store = service_components
    # S12 Phase 1b opportunistic fix (item 15): this test's `video_id` was
    # never a valid UUID ("vid-123") — TrackRequest.video_id: Optional[UUID]
    # rejects it with a 422 (root cause was a UUID-format bug in the test
    # fixture, not the S3-mocking gap the design doc's §7.1 disposition list
    # otherwise describes for this file's cohort).
    video_id = "12345678-1234-5678-1234-567812345678"
    payload = {
        "bucket": "test-bucket",
        "key": "videos/match.mp4",
        "video_id": video_id,
        "user_id": "user-456",
    }

    resp = await service_client.post("/track", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]
    assert job_id

    lifecycle = await jobs_store.get_lifecycle(job_id)
    assert lifecycle is not None
    # The route passes request.video_id (a UUID object) straight through to
    # create_lifecycle; the mock jobs_store stores it as-is (the REAL
    # JobsStore.create_lifecycle str()-converts it before the CQL write —
    # see jobs_store/lifecycle.py).
    assert str(lifecycle["video_id"]) == video_id
    assert lifecycle["job_state"] == "PENDING"
    # S12 Phase 1b (design §1.1): v2 is THE production path — every job
    # created via POST /track is a highlight_v2 job.
    assert lifecycle["pipeline_kind"] == "highlight_v2"

    request_json = await jobs_store.get_request(job_id)
    assert request_json is not None
    assert "videos/match.mp4" in request_json

    assert jobs_store._latest[video_id]["job_id"] == job_id
    assert len(scheduled_jobs) == 1
    assert scheduled_jobs[0][0] == job_id


def test_track_response_uses_rest_ack_without_websocket_url():
    response = TrackResponse(job_id="job-1")

    assert response.model_dump() == {"job_id": "job-1", "status": "pending"}


@pytest.mark.asyncio
async def test_run_with_semaphore_dispatches_to_run_highlight_job(monkeypatch, service_components):
    """S12 Phase 1b (design §1.1, item 15): the single call-site swap —
    _run_with_semaphore must dispatch to run_highlight_job, never the
    dormant tracking pipeline's run_job."""
    config, job_store, jobs_store = service_components
    run_highlight_job_mock = AsyncMock()
    run_job_mock = AsyncMock()
    monkeypatch.setattr(scheduling_mod, "run_highlight_job", run_highlight_job_mock)
    monkeypatch.setattr("service.worker.orchestrator.run_job", run_job_mock)

    request = TrackRequest(bucket="b", key="k.mp4")
    await scheduling_mod._run_with_semaphore("job-1", request)

    run_highlight_job_mock.assert_awaited_once_with("job-1", request, config, job_store, jobs_store)
    run_job_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_job_reads_keyspaces_lifecycle(service_client, service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-abc", "vid", "user", progress_percent=42.5)
    await jobs_store.set_state("job-abc", JobState.RUNNING)

    resp = await service_client.get("/job/job-abc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-abc"
    assert body["status"] == "running"
    assert body["progress_percent"] == 42.5


@pytest.mark.asyncio
async def test_get_job_exposes_only_the_persisted_analysis_settings_envelope(
    service_client,
    service_components,
):
    from service.analysis_settings import resolve_analysis_settings

    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-r4", "vid", "user")
    admitted = resolve_analysis_settings(
        TrackRequest(
            bucket="private-bucket",
            key="private-key.mp4",
            analysis_media_resolution="medium",
        ),
    )
    await jobs_store.save_request("job-r4", admitted.model_dump_json())

    response = await service_client.get("/job/job-r4")

    assert response.status_code == 200
    body = response.json()
    assert body["capability_schema_version"] == 1
    assert body["requested_analysis_settings"] == {
        "analysis_media_resolution": "medium",
    }
    assert body["effective_analysis_config"]["scan"]["media_resolution"] == "medium"
    assert "bucket" not in body
    assert "key" not in body


@pytest.mark.asyncio
async def test_get_job_returns_null_diagnostics_for_historical_pre_r4_request(
    service_client,
    service_components,
):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-pre-r4", "vid", "user")
    await jobs_store.save_request(
        "job-pre-r4",
        TrackRequest(bucket="historic-bucket", key="historic.mp4").model_dump_json(),
    )

    response = await service_client.get("/job/job-pre-r4")

    assert response.status_code == 200
    body = response.json()
    assert body["capability_schema_version"] is None
    assert body["requested_analysis_settings"] is None
    assert body["effective_analysis_config"] is None
    assert "bucket" not in body
    assert "key" not in body


@pytest.mark.asyncio
async def test_get_job_fails_visibly_for_partial_r4_analysis_settings_envelope(
    service_client,
    service_components,
):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-partial-r4", "vid", "user")
    await jobs_store.save_request(
        "job-partial-r4",
        '{"bucket":"private-bucket","key":"private.mp4",'
        '"capability_schema_version":1}',
    )

    response = await service_client.get("/job/job-partial-r4")

    assert response.status_code == 500
    assert response.json()["detail"] == "Persisted analysis settings snapshot is invalid"


@pytest.mark.asyncio
async def test_get_health_returns_ok(service_client):
    resp = await service_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_post_track_returns_429_when_at_capacity(
    service_client, monkeypatch,
):
    sem = MagicMock()
    sem.locked.return_value = True
    monkeypatch.setattr(routes_mod, "_job_semaphore", sem)

    resp = await service_client.post(
        "/track",
        json={"bucket": "b", "key": "v.mp4"},
    )
    assert resp.status_code == 429
    assert "capacity" in resp.json()["detail"].lower()
