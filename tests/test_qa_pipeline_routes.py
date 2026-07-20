"""``service/routes/qa_pipeline.py`` — ``GET /qa/pipelines``, ``GET
/qa/pipeline-defaults``, ``POST /qa/pipeline-run`` (streaming NDJSON).

Mirrors ``test_qa_vlm.py``'s pattern: a minimal FastAPI app mounting only the
pipeline routes + ``init_routes`` for ``route_state._config``, no production
router/Cassandra/torch dependency beyond what importing ``service.routes.qa_vlm``
already pulls in (documented in ``service/qa_app.py``).
"""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.pipelines import registry
from service.routes import qa_pipeline as qa_pipeline_route
from service.routes.state import init_routes

YT_ID = "jNQXAC9IVRw"
YT_URL = f"https://www.youtube.com/watch?v={YT_ID}"


def _make_app(gemini_api_key: str) -> FastAPI:
    config = ServiceConfig(detection_timeout=86400.0, s3_endpoint_url="http://localhost:4566")
    object.__setattr__(config, "gemini_api_key", gemini_api_key)
    init_routes(config, InMemoryJobStore(), None, instance_id="qa-pipeline-test")
    app = FastAPI()
    app.get("/qa/pipelines")(qa_pipeline_route.list_pipelines)
    app.get("/qa/pipeline-defaults")(qa_pipeline_route.pipeline_defaults)
    app.post("/qa/pipeline-run")(qa_pipeline_route.pipeline_run)
    return app


@pytest_asyncio.fixture()
async def client_with_key():
    app = _make_app("fake-gemini-key")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture()
async def client_without_key():
    app = _make_app("")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------- #
# GET /qa/pipelines, GET /qa/pipeline-defaults
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_list_pipelines(client_with_key):
    resp = await client_with_key.get("/qa/pipelines")
    assert resp.status_code == 200
    ids = {p["id"] for p in resp.json()["pipelines"]}
    assert {"single-shot", "vision-mimic", "vision-mimic-multi"} <= ids


@pytest.mark.asyncio
async def test_pipeline_defaults_known_id(client_with_key):
    resp = await client_with_key.get("/qa/pipeline-defaults", params={"id": "vision-mimic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "vision-mimic"
    assert [s["type"] for s in body["stages"]] == [
        "frame_sample", "detect_crop", "window", "analyze", "context_chain", "dedup",
    ]


@pytest.mark.asyncio
async def test_pipeline_defaults_unknown_id_404(client_with_key):
    resp = await client_with_key.get("/qa/pipeline-defaults", params={"id": "nope"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /qa/pipeline-run — fail-closed gates BEFORE streaming
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_run_missing_gemini_key_400(client_without_key):
    pdef = registry.get_default("single-shot").model_dump(mode="json")
    resp = await client_without_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "Gemini API key" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_invalid_pipeline_def_400(client_with_key):
    pdef = registry.get_default("single-shot").model_dump(mode="json")
    pdef["stages"][0]["type"] = "dedup"  # violates single-shot's fixed stage set
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "fixed stage set" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_off_allowlist_model_400(client_with_key):
    pdef = registry.get_default("single-shot").model_dump(mode="json")
    analyze_stage = next(s for s in pdef["stages"] if s["type"] == "analyze")
    analyze_stage["config"]["model"] = "gemini-1.0-nano-experimental"
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "Unsupported model" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_over_budget_400(client_with_key):
    pdef = registry.get_default("vision-mimic").model_dump(mode="json")
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 600},  # 10 minutes -> way over cap
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "budget cap" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_over_budget_allowed_with_explicit_override(client_with_key, monkeypatch):
    """An explicit ``budget_cap`` override raises the ceiling for one run."""
    pdef = registry.get_default("single-shot").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": planned["planned_gemini_calls"]}
        yield {"type": "run_complete", "clips": []}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
        "pipeline": pdef,
        "budget_cap": 1,
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_pipeline_defaults_chunk_segment_tags(client_with_key):
    resp = await client_with_key.get("/qa/pipeline-defaults", params={"id": "chunk-segment-tags"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chunk-segment-tags"
    assert [s["type"] for s in body["stages"]] == [
        "video_window", "chunk_segment", "chunk_analyze", "context_chain", "dedup",
    ]


# --------------------------------------------------------------------------- #
# chunk-segment-tags — fail-closed pre-stream (Brooks seams)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_structural_disable_400(client_with_key):
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    next(s for s in pdef["stages"] if s["type"] == "chunk_segment")["enabled"] = False
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "structural" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_analyze_structural_disable_400(client_with_key):
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    next(s for s in pdef["stages"] if s["type"] == "chunk_analyze")["enabled"] = False
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "structural" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_off_allowlist_model_400(client_with_key):
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    next(s for s in pdef["stages"] if s["type"] == "chunk_segment")["config"]["model"] = "gemini-1.0-nano-experimental"
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "Unsupported model" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_unknown_config_key_400(client_with_key):
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    next(s for s in pdef["stages"] if s["type"] == "chunk_analyze")["config"]["not_a_real_key"] = 1
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "Invalid config" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_tags_worst_case_estimate_does_not_pre_stream_block(client_with_key, monkeypatch):
    """Task 1 (durable budget fix): chunk-segment-tags' pathological
    worst-case pre-flight estimate (1 + ceil(duration/min_chunk_s) — 300s /
    5s min_chunk -> 61 planned calls > DEFAULT_BUDGET_CAP of 60) must NOT
    hard-block the run pre-stream anymore — that estimate false-blocks even a
    plain 5-minute match whose REAL PASS-2 call count is far lower once the
    worth_analysis prefilter runs. The run must be allowed to start (200,
    streaming) — the two-phase gate on the ACTUAL worthy-chunk count lives
    inside run_pipeline (see test_pipelines_executors.py), not here."""
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": planned["planned_gemini_calls"]}
        yield {"type": "chunk_map", "chunks": []}
        yield {"type": "run_complete", "clips": [], "format": "simplified-tags-time-v1"}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    # 300s / 5s min_chunk -> 60 chunks + 1 = 61 planned calls > DEFAULT_BUDGET_CAP (60) —
    # the OLD pre-flight guard would have 400'd here; the new one must not.
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 300},
        "pipeline": pdef,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert [ln["type"] for ln in lines] == ["run_start", "chunk_map", "run_complete"]


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_tags_still_hard_blocks_over_12min_scope_cap(client_with_key):
    """The v1 hard scope cap (~12min, chunk_segment.check_scope_cap) is
    UNCHANGED and still fires pre-stream regardless of the worst-case-estimate
    bypass above — this is the real backstop bounding worst-case spend now
    that the pathological pre-flight estimate no longer gates chunk-segment-tags."""
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 900},  # 15 minutes
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "exceeds the chunk-segment-tags v1 hard cap" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_other_pipelines_pre_flight_guard_unchanged(client_with_key):
    """Sanity: the 4 non-chunk-segment-tags pipelines' pre-flight budget guard
    is untouched by Task 1 — still a hard 400 before streaming starts."""
    pdef = registry.get_default("vision-mimic").model_dump(mode="json")
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 600},  # 10 minutes -> way over cap
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "budget cap" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_tags_hard_scope_cap_400(client_with_key):
    """v1 hard cap (~10-12 min) — checked BEFORE the generic budget guard so
    the caller gets the actionable cap message, not the generic one."""
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 900},  # 15 minutes — over the 12min cap
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "exceeds the chunk-segment-tags v1 hard cap" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_chunk_segment_tags_within_budget_and_cap_streams_ok(client_with_key, monkeypatch):
    pdef = registry.get_default("chunk-segment-tags").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": planned["planned_gemini_calls"]}
        yield {"type": "chunk_map", "chunks": []}
        yield {"type": "run_complete", "clips": [], "format": "simplified-tags-time-v1"}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert [ln["type"] for ln in lines] == ["run_start", "chunk_map", "run_complete"]


# --------------------------------------------------------------------------- #
# highlight-scan-analyze — evaluator CHANGES-REQUIRED item 1 (2026-07-18):
# same two-phase-gate exemption as chunk-segment-tags above, reapplied
# verbatim for this pipeline (it has no worth_analysis prefilter at all, so
# its pathological worst-case pre-flight estimate is even worse).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_defaults_highlight_scan_analyze(client_with_key):
    resp = await client_with_key.get("/qa/pipeline-defaults", params={"id": "highlight-scan-analyze"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "highlight-scan-analyze"
    assert [s["type"] for s in body["stages"]] == ["video_window", "highlight_scan", "highlight_analyze"]


@pytest.mark.asyncio
async def test_pipeline_run_highlight_scan_analyze_worst_case_estimate_does_not_pre_stream_block(client_with_key, monkeypatch):
    """highlight-scan-analyze's pathological worst-case pre-flight estimate
    (1 + ceil(duration/min_highlight_s) — 300s / 3s min_highlight -> 101
    planned calls > DEFAULT_BUDGET_CAP of 60) must NOT hard-block the run
    pre-stream — that estimate false-blocks even a plain 5-minute match whose
    REAL PASS-2 call count (== the real PASS-1 highlight count) is far lower
    in practice. The run must be allowed to start (200, streaming) — the
    two-phase gate on the ACTUAL highlight count lives inside run_pipeline
    (see test_pipelines_executors.py), not here."""
    pdef = registry.get_default("highlight-scan-analyze").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": planned["planned_gemini_calls"]}
        yield {"type": "highlight_map", "highlights": []}
        yield {"type": "run_complete", "clips": [], "format": "simplified-tags-time-v1"}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    # 300s / 3s min_highlight -> 100 highlights + 1 = 101 planned calls >
    # DEFAULT_BUDGET_CAP (60) — the generic pre-flight guard would have 400'd
    # here without the exemption; it must not.
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 300},
        "pipeline": pdef,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert [ln["type"] for ln in lines] == ["run_start", "highlight_map", "run_complete"]


@pytest.mark.asyncio
async def test_pipeline_run_highlight_scan_analyze_within_budget_and_cap_streams_ok(client_with_key, monkeypatch):
    pdef = registry.get_default("highlight-scan-analyze").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": planned["planned_gemini_calls"]}
        yield {"type": "highlight_map", "highlights": []}
        yield {"type": "run_complete", "clips": [], "format": "simplified-tags-time-v1"}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert [ln["type"] for ln in lines] == ["run_start", "highlight_map", "run_complete"]


@pytest.mark.asyncio
async def test_pipeline_run_highlight_scan_analyze_structural_disable_400(client_with_key):
    pdef = registry.get_default("highlight-scan-analyze").model_dump(mode="json")
    next(s for s in pdef["stages"] if s["type"] == "highlight_scan")["enabled"] = False
    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
        "pipeline": pdef,
    })
    assert resp.status_code == 400
    assert "structural" in resp.text


@pytest.mark.asyncio
async def test_pipeline_run_streams_ndjson_lines(client_with_key, monkeypatch):
    pdef = registry.get_default("single-shot").model_dump(mode="json")

    async def _fake_run_pipeline(pipeline, ctx, planned, budget_cap=None):
        yield {"type": "run_start", "planned_gemini_calls": 1}
        yield {"type": "stage_start", "stage_id": "video_window", "stage_type": "video_window", "skipped": False}
        yield {"type": "stage_complete", "stage_id": "video_window", "skipped": False, "timing": {"stage_ms": 1.0}}
        yield {"type": "window_result", "window": 1, "format": "events-v1", "clips": [], "context_summary": None,
               "timing": {"model_ms": 5.0}}
        yield {"type": "run_complete", "clips": [], "format": "events-v1", "timing": {"total_ms": 10.0}}

    monkeypatch.setattr(qa_pipeline_route, "run_pipeline", _fake_run_pipeline)

    resp = await client_with_key.post("/qa/pipeline-run", json={
        "youtube_id": YT_ID, "youtube_url": YT_URL,
        "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
        "pipeline": pdef,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in resp.text.strip().split("\n")]
    assert [ln["type"] for ln in lines] == [
        "run_start", "stage_start", "stage_complete", "window_result", "run_complete",
    ]
