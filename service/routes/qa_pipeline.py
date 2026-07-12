"""QA VLM Studio pipeline endpoints — ``GET /qa/pipelines``, ``GET /qa/pipeline-defaults``,
``POST /qa/pipeline-run`` (build plan §3.2/§3.4).

Mirrors ``qa_vlm.py``'s no-fabrication discipline:
- Missing Gemini key -> 400 (``_require_gemini_key``, reused as-is).
- Unknown pipeline id / malformed stage config / disabled structural stage ->
  400 BEFORE streaming starts (``registry.validate_pipeline_def``).
- A run whose estimated Gemini-call count exceeds the budget cap -> 400 BEFORE
  streaming starts (``executors.estimate_run_plan``) — never silently truncated
  mid-run. **EXCEPTION (Task 1, durable budget fix): ``chunk-segment-tags`` is
  a two-phase gate, not a single pre-flight block** — its own worst-case
  estimate (1 rough-scan call + one chunk_analyze call per theoretically
  smallest chunk) is pathological and false-blocks legitimate short matches
  (a plain 5min match already worst-cases to 61 > the default cap of 60), so
  this pipeline SKIPS the pre-stream hard block (the existing >12min scope cap
  still bounds real spend) and instead streams the real ``chunk_map`` from
  PASS 1, then gates the ACTUAL worthy-chunk count against the cap inside
  ``executors.run_pipeline`` — aborting with an ``error`` NDJSON event before
  PASS 2's deep loop spends a single Gemini call if the real number is over
  cap. Every other pipeline's pre-flight guard is unchanged.
- Real Gemini/transport/ffmpeg errors surface as an ``error`` NDJSON event (the
  stream itself already started 200, so they cannot become an HTTP error code);
  the run continues past a single failed window/frame rather than aborting the
  whole run on one bad call (frame-level and window-level try/except live in
  ``executors.py``).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from service.pipelines import chunk_segment, gemini_retry, registry
from service.pipelines.executors import DEFAULT_BUDGET_CAP, RunContext, estimate_run_plan, run_pipeline
from service.pipelines.models import PipelineDef
from service.routes import state as route_state
from service.routes.qa_vlm import QAVLMScope, _effective_window, _require_gemini_key

logger = logging.getLogger("service.routes")


class PipelineRunRequest(BaseModel):
    youtube_id: str
    youtube_url: str
    scope: QAVLMScope
    pipeline: PipelineDef
    # Optional override of DEFAULT_BUDGET_CAP for this run only (e.g. a test
    # harness intentionally exercising a larger scope). Omitted -> the default.
    budget_cap: Optional[int] = None


async def list_pipelines():
    return {"pipelines": registry.list_pipelines()}


async def pipeline_defaults(id: str):
    try:
        pdef = registry.get_default(id)
    except KeyError:
        raise HTTPException(
            404, f"Unknown pipeline id {id!r}; must be one of {sorted(registry.PIPELINE_STAGE_TYPES)}."
        )
    return pdef.model_dump()


async def pipeline_run(body: PipelineRunRequest, request: Request):
    _require_gemini_key()
    config = route_state._config
    assert config is not None

    start_sec, end_sec = _effective_window(body.scope)

    try:
        registry.validate_pipeline_def(body.pipeline)
    except registry.PipelineValidationError as e:
        raise HTTPException(400, str(e))

    is_chunk_segment_pipeline = any(s.type == "chunk_segment" for s in body.pipeline.stages)

    # v1 hard scope cap for chunk-segment-tags (fail-closed, no silent
    # truncation, no pre-slice-and-stitch) — checked BEFORE the budget guard
    # so a caller gets the actionable cap message rather than the generic
    # budget-cap message when both would technically fire.
    if is_chunk_segment_pipeline:
        cap_error = chunk_segment.check_scope_cap(end_sec - start_sec)
        if cap_error is not None:
            raise HTTPException(400, cap_error)

    planned = estimate_run_plan(body.pipeline, duration_sec=end_sec - start_sec)
    cap = body.budget_cap if body.budget_cap is not None else DEFAULT_BUDGET_CAP

    # Two-phase gate (Task 1, durable budget fix): chunk-segment-tags' own
    # worst-case estimate (1 + ceil(duration/min_chunk_s)) is pathological — a
    # real ~10min match plans ~128 worst-case but spends ~15-25 calls once the
    # PASS-1 worth_analysis prefilter runs; hard-blocking pre-stream on that
    # worst-case false-blocks even a plain 5min match (1+ceil(300/5)=61>60).
    # For chunk-segment-tags ONLY, skip this pre-stream hard block — the
    # >12min scope cap above already bounds real spend — and let the run
    # START and stream the real chunk_map. The ACTUAL worthy-chunk count is
    # gated against `cap` immediately after PASS 1 completes, INSIDE
    # run_pipeline (executors.py), aborting with an `error` event before the
    # deep PASS-2 loop spends a single Gemini call if the real number is over
    # cap. The other 4 pipelines are UNCHANGED: their pre-flight guard below
    # still hard-blocks before streaming starts.
    if not is_chunk_segment_pipeline and planned["planned_gemini_calls"] > cap:
        raise HTTPException(
            400,
            f"Planned Gemini calls ({planned['planned_gemini_calls']}) exceed the budget cap "
            f"({cap}). Narrow the scope, lower sample_fps/window size, switch agent_mode to "
            f"'single', or pass an explicit higher budget_cap.",
        )

    ctx = RunContext(
        youtube_id=body.youtube_id,
        youtube_url=body.youtube_url,
        start_sec=start_sec,
        end_sec=end_sec,
        gemini_api_key=config.gemini_api_key,
        request_timeout_ms=config.gemini_request_timeout_ms,
        retry_config=gemini_retry.GeminiRetryConfig.from_service_config(config),
    )

    logger.info(
        "pipeline-run: pipeline_id=%s youtube_id=%s window=%d-%ds planned_calls=%d",
        body.pipeline.id, body.youtube_id, start_sec, end_sec, planned["planned_gemini_calls"],
    )

    async def event_stream():
        try:
            async for event in run_pipeline(body.pipeline, ctx, planned, budget_cap=cap):
                yield json.dumps(event) + "\n"
                if await request.is_disconnected():
                    logger.info("pipeline-run: client disconnected mid-run, aborting")
                    return
        except Exception as e:  # noqa: BLE001 — surface a fatal run-loop error as a final NDJSON line
            logger.error("pipeline-run: run loop crashed: %s", e, exc_info=True)
            yield json.dumps({"type": "error", "stage_id": "run", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
