"""S12 Phase 1b — GET /jobs/{job_id}/events SSE additive progress keys
(item 18, design §1.2/item 11.5).
"""
from __future__ import annotations

import json

import pytest

from service.analysis_keyspaces_enums import JobState
from service.routes.events import job_events_sse


async def _first_event(response) -> str:
    chunk = await response.body_iterator.__anext__()
    await response.body_iterator.aclose()
    return chunk


def _parse_sse(chunk: str) -> tuple[str, dict]:
    lines = chunk.strip("\n").split("\n")
    event_line = next(line for line in lines if line.startswith("event: "))
    data_line = next(line for line in lines if line.startswith("data: "))
    return event_line[len("event: "):], json.loads(data_line[len("data: "):])


@pytest.mark.asyncio
async def test_progress_event_stays_byte_identical_when_v2_fields_absent(service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-1", "vid", "user")
    await jobs_store.set_state("job-1", JobState.RUNNING)

    response = await job_events_sse("job-1")
    event_name, data = _parse_sse(await _first_event(response))

    assert event_name == "progress"
    assert set(data) == {"type", "state", "percent", "frame_idx", "total_frames"}


@pytest.mark.asyncio
async def test_progress_event_includes_chunk_index_and_highlights_found_so_far(service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-1", "vid", "user", pipeline_kind="highlight_v2")
    await jobs_store.set_state("job-1", JobState.RUNNING)
    jobs_store._lifecycles["job-1"].update({
        "progress_percent": 42.0,
        "chunk_index": 2,
        "chunks_total": 5,
        "highlights_found_so_far": 7,
    })

    response = await job_events_sse("job-1")
    event_name, data = _parse_sse(await _first_event(response))

    assert event_name == "progress"
    assert data["chunk_index"] == 2
    assert data["chunks_total"] == 5
    assert data["highlights_found_so_far"] == 7
    # frame-counting fields still present (default 0) — never removed.
    assert data["frame_idx"] == 0
    assert data["total_frames"] == 0


@pytest.mark.asyncio
async def test_progress_event_includes_coach_facing_phase_when_persisted(service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-1", "vid", "user", pipeline_kind="highlight_v2")
    await jobs_store.set_state("job-1", JobState.RUNNING)
    jobs_store._lifecycles["job-1"].update({
        "progress_percent": 42.0,
        "stage_message": "detecting",
    })

    response = await job_events_sse("job-1")
    event_name, data = _parse_sse(await _first_event(response))

    assert event_name == "progress"
    assert data["phase"] == "detecting"


@pytest.mark.asyncio
async def test_progress_event_includes_attribution_metrics_when_present(service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-1", "vid", "user", pipeline_kind="highlight_v2")
    await jobs_store.set_state("job-1", JobState.RUNNING)
    metrics = {"player_id_counts": {"p1": 3, "p2": 2}, "sentinel_count": 1, "identity_uncertain_count": 2}
    jobs_store._lifecycles["job-1"].update({
        "progress_percent": 100.0,
        "attribution_metrics_json": json.dumps(metrics),
    })

    response = await job_events_sse("job-1")
    event_name, data = _parse_sse(await _first_event(response))

    assert data["attribution_metrics"] == metrics


@pytest.mark.asyncio
async def test_progress_event_malformed_attribution_metrics_json_does_not_crash(service_components):
    _, _, jobs_store = service_components
    await jobs_store.create_lifecycle("job-1", "vid", "user", pipeline_kind="highlight_v2")
    await jobs_store.set_state("job-1", JobState.RUNNING)
    jobs_store._lifecycles["job-1"].update({
        "progress_percent": 10.0,
        "attribution_metrics_json": "{not valid json",
    })

    response = await job_events_sse("job-1")
    event_name, data = _parse_sse(await _first_event(response))

    assert event_name == "progress"
    assert "attribution_metrics" not in data
