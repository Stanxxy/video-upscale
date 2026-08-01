"""Tests for POST /jobs/{job_id}/resume and /jobs/{job_id}/detection_response.

These tests validate the Keyspaces-backed suspend/resume flow:
  - POST /jobs/{id}/resume delegates to detection_response
  - POST /jobs/{id}/detection_response creates a new job from checkpoint

Shared fixtures live in ``tests/conftest.py``:
  - ``mock_jobs_store`` — fresh in-memory mock per test.
  - ``service_components`` — ``(config, job_store, jobs_store)`` triple.
  - ``service_client`` — ASGI client wired to the routes.
"""

import json

import pytest
import pytest_asyncio

from service import routes as routes_mod
from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.analysis_settings import resolve_analysis_settings
from service.models import TrackRequest


@pytest.fixture(autouse=True)
def _noop_schedule_job(monkeypatch):
    """Resume routes schedule the worker which touches S3 (ensure_bucket); CI may
    have no LocalStack. These tests assert HTTP + persisted request_json only."""
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: None)


@pytest_asyncio.fixture()
async def awaiting_job(service_components):
    """Create a job in AWAITING_CORRECTION state with checkpoint data."""
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="test-bucket", key="test/video.mp4")
    job = await job_store.create_job(req)
    job_id = job.job_id

    # Simulate job reaching AWAITING_CORRECTION
    await jobs_store.create_lifecycle(job_id, "test-video-id", "test-user")
    await jobs_store.save_request(job_id, req.model_dump_json())
    await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)
    await jobs_store.write_checkpoint(job_id, PipelineStage.DETECT, False, {
        "pending_detection": {
            "frame_idx": 0,
            "candidates": [{"candidate_id": 0, "box": [10, 20, 100, 200]}],
            "reason": "initial",
        }
    })
    return job_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_creates_resume_job(
    service_client, awaiting_job, service_components,
):
    """POST /jobs/{id}/detection_response creates a new job and returns its ID."""
    _, _, jobs_store = service_components
    job_id = awaiting_job

    resp = await service_client.post(
        f"/jobs/{job_id}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resumed"
    assert body["origin_job_id"] == job_id
    new_job_id = body["job_id"]
    assert new_job_id != job_id

    # New job should exist in Keyspaces
    new_lifecycle = await jobs_store.get_lifecycle(new_job_id)
    assert new_lifecycle is not None
    assert new_lifecycle["origin_job_id"] == job_id
    assert new_lifecycle["parent_job_id"] == job_id
    assert jobs_store._latest["test-video-id"]["job_id"] == new_job_id

    # Old job should be CANCELLED
    old_lifecycle = await jobs_store.get_lifecycle(job_id)
    assert old_lifecycle["job_state"] == "CANCELLED"
    assert old_lifecycle["replacement_job_id"] == new_job_id


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_rejects_duplicate_resume(
    service_client, awaiting_job, service_components,
):
    """409 when an awaiting job already has a replacement job."""
    _, _, jobs_store = service_components
    await jobs_store.set_replacement(awaiting_job, "replacement-job-id")

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 409
    assert "already has replacement" in resp.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_rejects_lost_replacement_claim(
    service_client, awaiting_job, service_components,
):
    """409 when another worker wins the replacement claim race."""
    _, _, jobs_store = service_components
    jobs_store._claim_replacement_result = False

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 409
    assert "replacement claim" in resp.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_uses_track_resume_cursor(
    service_client, awaiting_job, service_components,
):
    """Mid-track correction resumes from the next unprocessed frame."""
    _, _, jobs_store = service_components
    await jobs_store.write_checkpoint(awaiting_job, PipelineStage.TRACK, False, {
        "frame_count": 50,
        "resume_cursor": {"frame_idx": 51},
        "partial_tracking_s3_key": "checkpoints/original/partial_tracking.json",
        "pending_detection": {
            "frame_idx": 51,
            "candidates": [{"candidate_id": 0, "box": [10, 20, 100, 200]}],
            "reason": "identity_lost",
        },
    })

    resp = await service_client.post(
        f"/jobs/{awaiting_job}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )

    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]
    request_json = await jobs_store.get_request(new_job_id)
    resumed_request = json.loads(request_json)
    assert resumed_request["resume_from_job_id"] == awaiting_job
    assert resumed_request["resume_tracking_s3_key"] == (
        "checkpoints/original/partial_tracking.json"
    )
    assert resumed_request["resume_from_frame"] == 51


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_resume_delegates_to_detection_response(service_client, awaiting_job):
    """POST /jobs/{id}/resume delegates to detection_response logic."""
    job_id = awaiting_job

    resp = await service_client.post(
        f"/jobs/{job_id}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resumed"
    assert body["job_id"] != job_id


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_writes_replaced_by_new_job_on_old(
    service_client, awaiting_job, service_components,
):
    """The old job receives a final replaced_by_new_job checkpoint with
    completed=True and the new job's id under artifacts."""
    _, _, jobs_store = service_components
    job_id = awaiting_job
    # Seed a track row carrying worker_state so build_replaced_by_new_job can
    # snapshot the old job's progress.
    await jobs_store.write_checkpoint(job_id, PipelineStage.TRACK, False, {
        "schema_version": 1, "pending_detection": None, "artifacts": {},
        "worker_state": {
            "progress_percent": 35.0, "current_frame": 7432,
            "total_frames": 21600, "stage_progress_fraction": 0.34,
        },
    })

    resp = await service_client.post(
        f"/jobs/{job_id}/detection_response",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]

    # Find the replaced_by_new_job row on the OLD job.
    replaced = [
        cp for (jid, _), cp in jobs_store._checkpoints.items()
        if jid == job_id
        and cp["checkpoint_data"].get("reason") == "replaced_by_new_job"
    ]
    assert len(replaced) == 1
    cp = replaced[0]
    assert cp["completed"] is True
    data = cp["checkpoint_data"]
    assert data["schema_version"] == 1
    assert data["artifacts"]["replacement_job_id"] == new_job_id
    assert data["worker_state"]["current_frame"] == 7432


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_forwards_upscale_artifacts_via_overrides(
    service_client, service_components,
):
    """If the awaiting job has an upscale_analyze checkpoint, the resume
    request inherits analysis_raw_s3_key + window_count + current_context
    and uses the END_OF_TRACKING_SENTINEL for resume_from_frame."""
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="v.mp4")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid", "u")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.AWAITING_CORRECTION)
    await jobs_store.write_checkpoint(job.job_id, PipelineStage.DETECT, False, {
        "schema_version": 1,
        "pending_detection": {"reason": "initial", "frame_idx": 0},
        "artifacts": {},
        "worker_state": {
            "progress_percent": 10.0, "current_frame": 0,
            "total_frames": 0, "stage_progress_fraction": 0.0,
        },
    })
    await jobs_store.write_checkpoint(job.job_id, PipelineStage.UPSCALE_ANALYZE, False, {
        "schema_version": 1, "pending_detection": None,
        "reason": "analysis_window_completed",
        "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
        "analysis_current_context": "north-south",
        "artifacts": {
            "tracking_s3_key": "checkpoints/orig/tracking.json",
            "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
        },
        "worker_state": {
            "progress_percent": 67.5, "current_frame": 9120,
            "total_frames": 21600, "stage_progress_fraction": 0.5,
        },
    })

    resp = await service_client.post(
        f"/jobs/{job.job_id}/resume",
        json={"box_a": [1, 2, 3, 4], "box_b": [5, 6, 7, 8]},
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]

    rec = json.loads(await jobs_store.get_request(new_job_id))
    assert rec["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    assert rec["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"
    assert rec["analysis_window_count"] == 12
    assert rec["analysis_current_context"] == "north-south"
    # END_OF_TRACKING_SENTINEL = 10**9 — well past any plausible video frame
    assert rec["resume_from_frame"] >= 10**9


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_resume_seeds_new_lifecycle_progress_from_old_worker_state(
    service_client, awaiting_job, service_components,
):
    """The new lifecycle row should start at the OLD job's last-known
    progress_percent / current_frame / total_frames so the SSE stream
    does not regress to 0%."""
    _, _, jobs_store = service_components
    job_id = awaiting_job
    await jobs_store.write_checkpoint(job_id, PipelineStage.TRACK, False, {
        "schema_version": 1,
        "pending_detection": None,
        "artifacts": {},
        "worker_state": {
            "progress_percent": 35.0,
            "current_frame": 7432,
            "total_frames": 21600,
            "stage_progress_fraction": 0.34,
        },
    })

    resp = await service_client.post(
        f"/jobs/{job_id}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]

    new_lc = await jobs_store.get_lifecycle(new_job_id)
    assert new_lc["progress_percent"] == 35.0
    assert new_lc["current_frame"] == 7432
    assert new_lc["total_frames"] == 21600


# ---------------------------------------------------------------------------
# Recovery path — recover_interrupted_job (Task 8 + Task 9 wiring)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_interrupted_job_writes_replaced_row_and_forwards_artifacts(
    service_components,
):
    """recover_interrupted_job:
      - composes resume_params via build_resume_overrides (forwards
        analysis_raw_s3_key + window_count + current_context for upscale crash),
      - writes replaced_by_new_job row with completed=True on the OLD job,
      - seeds the NEW lifecycle row from the OLD worker_state.
    """
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="folder/v.mp4")
    job = await job_store.create_job(req)
    old_job_id = job.job_id
    await jobs_store.create_lifecycle(old_job_id, "vid-77", "user-77")
    await jobs_store.save_request(old_job_id, req.model_dump_json())
    # Simulate INTERRUPTED job that crashed mid-upscale.
    await jobs_store.set_state(old_job_id, JobState.INTERRUPTED)
    await jobs_store.write_checkpoint(
        old_job_id, PipelineStage.TRACK, False, {
            "schema_version": 1, "pending_detection": None,
            "artifacts": {"tracking_s3_key": "checkpoints/orig/tracking.json"},
            "worker_state": {
                "progress_percent": 55.0, "current_frame": 21600,
                "total_frames": 21600, "stage_progress_fraction": 1.0,
            },
        },
    )
    await jobs_store.write_checkpoint(
        old_job_id, PipelineStage.UPSCALE_ANALYZE, False, {
            "schema_version": 1, "pending_detection": None,
            "reason": "analysis_window_completed",
            "resume_cursor": {"frame_idx": 9120, "analysis_window_count": 12},
            "analysis_current_context": "north-south",
            "artifacts": {
                "tracking_s3_key": "checkpoints/orig/tracking.json",
                "analysis_raw_s3_key": "checkpoints/orig/analysis_raw.json",
            },
            "worker_state": {
                "progress_percent": 67.5, "current_frame": 9120,
                "total_frames": 21600, "stage_progress_fraction": 0.5,
            },
        },
    )

    lifecycle = await jobs_store.get_lifecycle(old_job_id)
    await recover_interrupted_job(lifecycle)

    # The OLD job got a replaced_by_new_job row with completed=True.
    new_job_id = (await jobs_store.get_lifecycle(old_job_id))["replacement_job_id"]
    assert new_job_id and new_job_id != old_job_id
    replaced = [
        cp for (jid, _), cp in jobs_store._checkpoints.items()
        if jid == old_job_id
        and cp["checkpoint_data"].get("reason") == "replaced_by_new_job"
    ]
    assert len(replaced) == 1
    cp = replaced[0]
    assert cp["completed"] is True
    assert cp["checkpoint_data"]["artifacts"]["replacement_job_id"] == new_job_id
    # OLD job's worker_state was snapshotted into the row (latest stage = upscale_analyze)
    assert cp["checkpoint_data"]["worker_state"]["progress_percent"] == 67.5
    assert cp["checkpoint_data"]["worker_state"]["current_frame"] == 9120

    # The NEW request was saved with upscale-crash overrides forwarded.
    rec = json.loads(await jobs_store.get_request(new_job_id))
    assert rec["resume_tracking_s3_key"] == "checkpoints/orig/tracking.json"
    assert rec["analysis_raw_s3_key"] == "checkpoints/orig/analysis_raw.json"
    assert rec["analysis_window_count"] == 12
    assert rec["analysis_current_context"] == "north-south"
    assert rec["resume_from_frame"] >= 10**9  # END_OF_TRACKING_SENTINEL
    assert rec["resume_from_job_id"] == old_job_id

    # The NEW lifecycle row was seeded from the OLD worker_state.
    new_lc = await jobs_store.get_lifecycle(new_job_id)
    assert new_lc["origin_job_id"] == old_job_id
    assert new_lc["parent_job_id"] == old_job_id
    assert new_lc["progress_percent"] == 67.5
    assert new_lc["current_frame"] == 9120
    assert new_lc["total_frames"] == 21600


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_not_found(service_client):
    """404 when job_id does not exist in Keyspaces."""
    resp = await service_client.post(
        "/jobs/nonexistent-id/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_recover_interrupted_job_skips_spawn_when_pipeline_terminal(
    service_components,
):
    """Stale INTERRUPTED row with publish terminal checkpoint → mark COMPLETED, no spawn."""
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="k.mp4")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-term", "u")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)
    await jobs_store.write_checkpoint(
        job.job_id,
        PipelineStage.PUBLISH,
        True,
        {
            "schema_version": 1,
            "pending_detection": None,
            "reason": "publish_completed",
            "artifacts": {
                "sns_topic_arn": "arn:aws:sns:x",
                "sns_event_count": 1,
                "sns_completion_sent": True,
            },
            "worker_state": {
                "progress_percent": 100.0,
                "current_frame": 0,
                "total_frames": 0,
                "stage_progress_fraction": 1.0,
            },
        },
    )

    lifecycle = await jobs_store.get_lifecycle(job.job_id)
    await recover_interrupted_job(lifecycle)

    lc = await jobs_store.get_lifecycle(job.job_id)
    assert lc["job_state"] == JobState.COMPLETED.value


@pytest.mark.asyncio
async def test_recover_interrupted_job_highlight_v2_resumes_same_job_id_no_replacement(
    service_components, monkeypatch,
):
    """S12 Phase 1b (design §6.3, item 17): a pipeline_kind="highlight_v2"
    interrupted job resumes the SAME job_id — no create_replacement_job
    call, no new job_id, no replacement_job_id written on the old row."""
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = resolve_analysis_settings(TrackRequest(bucket="b", key="folder/v.mp4"))
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-v2", "u", pipeline_kind="highlight_v2")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)
    await jobs_store.write_checkpoint(
        job.job_id, PipelineStage.HIGHLIGHT_CHUNK, True,
        {
            "schema_version": 1, "pending_detection": None,
            "artifacts": {"gemini_file_uri": "files/abc"},
            "chunk_index": 1, "chunks_total": 3,
            "worker_state": {
                "progress_percent": 33.0, "current_frame": 0,
                "total_frames": 0, "stage_progress_fraction": 1.0,
            },
        },
    )

    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: scheduled.append(job_id))

    lifecycle = await jobs_store.get_lifecycle(job.job_id)
    assert lifecycle["pipeline_kind"] == "highlight_v2"
    await recover_interrupted_job(lifecycle)

    assert scheduled == [job.job_id]  # SAME job_id, no replacement

    lc_after = await jobs_store.get_lifecycle(job.job_id)
    assert not lc_after.get("replacement_job_id")  # no replacement-job chain
    assert lc_after["job_state"] == JobState.INTERRUPTED.value  # unchanged — resume owns the transition


@pytest.mark.asyncio
async def test_recover_interrupted_job_absent_pipeline_kind_uses_tracking_replacement_flow(
    service_components, monkeypatch,
):
    """Back-compat: a row created before pipeline_kind existed (absent ->
    "tracking" per jobs_store.get_lifecycle) takes the OLD replacement-job
    path, unchanged."""
    from service.routes import recover_interrupted_job

    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="folder/v.mp4")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "vid-tracking", "u")  # no pipeline_kind kwarg
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    await jobs_store.set_state(job.job_id, JobState.INTERRUPTED)
    await jobs_store.write_checkpoint(
        job.job_id, PipelineStage.TRACK, False,
        {
            "schema_version": 1, "pending_detection": None,
            "artifacts": {"tracking_s3_key": "checkpoints/orig/tracking.json"},
            "worker_state": {
                "progress_percent": 55.0, "current_frame": 21600,
                "total_frames": 21600, "stage_progress_fraction": 1.0,
            },
        },
    )

    scheduled = []
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: scheduled.append(job_id))

    lifecycle = await jobs_store.get_lifecycle(job.job_id)
    assert lifecycle["pipeline_kind"] == "tracking"
    await recover_interrupted_job(lifecycle)

    new_job_id = (await jobs_store.get_lifecycle(job.job_id))["replacement_job_id"]
    assert new_job_id and new_job_id != job.job_id  # REPLACEMENT job chain, unchanged tracking behavior
    assert scheduled == [new_job_id]


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_detection_response_wrong_state(service_client, service_components):
    """409 when job is not in AWAITING_CORRECTION state."""
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="b", key="k")
    job = await job_store.create_job(req)
    await jobs_store.create_lifecycle(job.job_id, "", "")
    await jobs_store.save_request(job.job_id, req.model_dump_json())
    # State is PENDING, not AWAITING_CORRECTION

    resp = await service_client.post(
        f"/jobs/{job.job_id}/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 409
