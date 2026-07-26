"""Stream 0b — the confirmed player_mapping is threaded resume -> replacement job.

The human-confirmed obj_id->player_id binding
``player_mapping { "1": player_id_A, "2": player_id_B }`` travels from the
ResumeRequest onto the replacement TrackRequest so tracking re-seeds init_boxes
from the confirmed binding (track_ids never flip). Without it, the field stays
None and the legacy positional box_a/box_b seeding applies.
"""

import inspect
import json

import pytest
import pytest_asyncio

from service import routes as routes_mod
from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.models import ResumeRequest, TrackRequest


@pytest.fixture(autouse=True)
def _noop_schedule_job(monkeypatch):
    monkeypatch.setattr(routes_mod, "_schedule_job", lambda job_id, request: None)


@pytest_asyncio.fixture()
async def awaiting_job(service_components):
    _, job_store, jobs_store = service_components
    req = TrackRequest(bucket="test-bucket", key="test/video.mp4")
    job = await job_store.create_job(req)
    job_id = job.job_id
    await jobs_store.create_lifecycle(job_id, "test-video-id", "test-user")
    await jobs_store.save_request(job_id, req.model_dump_json())
    await jobs_store.set_state(job_id, JobState.AWAITING_CORRECTION)
    await jobs_store.write_checkpoint(job_id, PipelineStage.TRACK, False, {
        "frame_count": 50,
        "resume_cursor": {"frame_idx": 51},
        "partial_tracking_s3_key": "checkpoints/original/partial_tracking.json",
        "pending_detection": {
            "frame_idx": 51,
            "candidates": [{"candidate_id": 0, "box": [10, 20, 100, 200]}],
            "reason": "tracking_lost",
        },
    })
    return job_id


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_resume_threads_player_mapping_into_replacement_request(
    service_client, awaiting_job, service_components,
):
    """A resume carrying player_mapping persists it on the new job's request."""
    _, _, jobs_store = service_components
    job_id = awaiting_job

    resp = await service_client.post(
        f"/jobs/{job_id}/resume",
        json={
            "box_a": [10, 20, 100, 200],
            "box_b": [300, 20, 400, 200],
            "player_mapping": {"1": "player-A", "2": "player-B"},
        },
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]

    rec = json.loads(await jobs_store.get_request(new_job_id))
    assert rec["player_mapping"] == {"1": "player-A", "2": "player-B"}


@pytest.mark.asyncio
@pytest.mark.skip(reason="S12 Phase 1b decision 2: human_loop.py routes unregistered from the production app surface (item 16) — this test hits the HTTP route and now 404s. Quarantined, not deleted; re-registering the routes and un-skipping this test is one clean pair of actions if the dormant path is ever revived.")
async def test_resume_without_player_mapping_leaves_field_none(
    service_client, awaiting_job, service_components,
):
    """Legacy resume (no mapping) -> replacement request has player_mapping None;
    tracking falls back to the legacy positional box_a/box_b seeding."""
    _, _, jobs_store = service_components
    job_id = awaiting_job

    resp = await service_client.post(
        f"/jobs/{job_id}/resume",
        json={"box_a": [10, 20, 100, 200], "box_b": [300, 20, 400, 200]},
    )
    assert resp.status_code == 200
    new_job_id = resp.json()["job_id"]

    rec = json.loads(await jobs_store.get_request(new_job_id))
    assert rec.get("player_mapping") is None


def test_resume_request_model_carries_player_mapping():
    body = ResumeRequest(
        box_a=[0, 0, 1, 1], box_b=[2, 2, 3, 3],
        player_mapping={"1": "a", "2": "b"},
    )
    assert body.player_mapping == {"1": "a", "2": "b"}


def test_run_tracking_consumes_player_mapping_kwarg():
    """The tracking entrypoints expose player_mapping so the confirmed binding can
    re-seed init_boxes (guards the seam against accidental signature drift)."""
    from service.tracking_runner import run_tracking_job
    from tracking_pipeline.hybrid.run_tracking import run_tracking

    assert "player_mapping" in inspect.signature(run_tracking_job).parameters
    assert "player_mapping" in inspect.signature(run_tracking).parameters


def test_init_boxes_seeded_from_confirmed_binding():
    """Stream 0b seeding: with a confirmed player_mapping, init_boxes is keyed by
    the confirmed obj_ids (obj_id 1 -> box_a, 2 -> box_b); without it, positional.

    Mirrors the construction in run_tracking.run_tracking so the override+fallback
    is asserted without booting SAM2/GPU.
    """
    box_a, box_b = [0, 0, 1, 1], [2, 2, 3, 3]
    obj_id_to_box = {1: box_a, 2: box_b}

    def seed(player_mapping):
        if player_mapping:
            init_boxes = {
                int(obj_id): obj_id_to_box[int(obj_id)]
                for obj_id in player_mapping
                if int(obj_id) in obj_id_to_box
            }
            return init_boxes or {1: box_a, 2: box_b}
        return {1: box_a, 2: box_b}

    # Confirmed binding -> seeded by obj_id keys.
    assert seed({"1": "player-A", "2": "player-B"}) == {1: box_a, 2: box_b}
    # Legacy (no mapping) -> positional fallback.
    assert seed(None) == {1: box_a, 2: box_b}
    # Empty/garbage mapping -> legacy fallback, never empty init_boxes.
    assert seed({}) == {1: box_a, 2: box_b}
