"""QA VLM Studio pipeline endpoints — ``GET /qa/pipelines``, ``GET /qa/pipeline-defaults``,
``POST /qa/pipeline-run`` (build plan §3.2/§3.4).

Mirrors ``qa_vlm.py``'s no-fabrication discipline:
- Missing Gemini key -> 400 (``_require_gemini_key``, reused as-is).
- Unknown pipeline id / malformed stage config / disabled structural stage ->
  400 BEFORE streaming starts (``registry.validate_pipeline_def``).
- A run whose estimated Gemini-call count exceeds the budget cap -> 400 BEFORE
  streaming starts (``executors.estimate_run_plan``) — never silently truncated
  mid-run. **EXCEPTION (Task 1, durable budget fix; reapplied for
  ``highlight-scan-analyze`` per evaluator CHANGES-REQUIRED item 1,
  2026-07-18): ``chunk-segment-tags`` AND ``highlight-scan-analyze`` are each
  a two-phase gate, not a single pre-flight block** — their own worst-case
  estimates (1 rough-scan call + one PASS-2 call per theoretically smallest
  chunk/highlight) are pathological and false-block legitimate short matches
  (chunk-segment-tags: a plain 5min match already worst-cases to 61 > the
  default cap of 60; highlight-scan-analyze: a plain 5min match worst-cases to
  101 > 60 at the default ``min_highlight_s=3.0`` — WORSE, since it has no
  ``worth_analysis`` prefilter at all), so BOTH pipelines SKIP the pre-stream
  hard block (chunk-segment-tags' existing >12min scope cap still bounds its
  real spend; highlight-scan-analyze has no analogous scope cap — see the
  route body for why) and instead stream the real ``chunk_map``/
  ``highlight_map`` from PASS 1, then gate the ACTUAL worthy-chunk/highlight
  count against the cap inside ``executors.run_pipeline`` — aborting with an
  ``error`` NDJSON event before PASS 2's deep loop spends a single Gemini call
  if the real number is over cap. Every other pipeline's pre-flight guard is
  unchanged.
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
    is_highlight_scan_pipeline = any(s.type == "highlight_scan" for s in body.pipeline.stages)

    # v1 hard scope cap for chunk-segment-tags (fail-closed, no silent
    # truncation, no pre-slice-and-stitch) — checked BEFORE the budget guard
    # so a caller gets the actionable cap message rather than the generic
    # budget-cap message when both would technically fire.
    #
    # highlight-scan-analyze has NO analogous scope cap: its PASS-1 rough
    # scan is a single call whose cost scales with duration (same as
    # chunk_segment's), but nothing in the build plan specified a v1 hard
    # cap for this pipeline, and the two-phase gate below already bounds
    # PASS-2 real spend. Revisit if a very long unscoped run turns out to be
    # a real-world footgun (chunk-segment-tags' cap exists because its PASS-1
    # cost model was measured against that exact failure mode).
    if is_chunk_segment_pipeline:
        cap_error = chunk_segment.check_scope_cap(end_sec - start_sec)
        if cap_error is not None:
            raise HTTPException(400, cap_error)

    planned = estimate_run_plan(body.pipeline, duration_sec=end_sec - start_sec)
    cap = body.budget_cap if body.budget_cap is not None else DEFAULT_BUDGET_CAP

    # Two-phase gate (Task 1, durable budget fix; reapplied for
    # highlight-scan-analyze per evaluator CHANGES-REQUIRED item 1,
    # 2026-07-18): both chunk-segment-tags' own worst-case estimate
    # (1 + ceil(duration/min_chunk_s)) and highlight-scan-analyze's
    # (1 + ceil(duration/min_highlight_s), WORSE — no worth_analysis
    # prefilter at all) are pathological — a real ~5min match already
    # worst-cases chunk-segment-tags to 61 and highlight-scan-analyze to 101,
    # both over the default cap of 60, despite spending far fewer calls once
    # PASS 1's real (much smaller) output is known. For these TWO pipelines
    # ONLY, skip this pre-stream hard block — chunk-segment-tags' >12min scope
    # cap above already bounds its real spend — and let the run START and
    # stream the real chunk_map/highlight_map. The ACTUAL worthy-chunk/
    # highlight count is gated against `cap` immediately after PASS 1
    # completes, INSIDE run_pipeline (executors.py), aborting with an `error`
    # event before the deep PASS-2 loop spends a single Gemini call if the
    # real number is over cap. The other pipelines are UNCHANGED: their
    # pre-flight guard below still hard-blocks before streaming starts.
    if (
        not is_chunk_segment_pipeline
        and not is_highlight_scan_pipeline
        and planned["planned_gemini_calls"] > cap
    ):
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
