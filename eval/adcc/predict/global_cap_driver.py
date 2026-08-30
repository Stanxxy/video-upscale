"""GLOBAL cross-match/slice Gemini call cap — the dashboard's cost guarantee
(Task 2 of the ADCC eval cost-control dashboard build).

**Why this exists, and why it's not in ``service/pipelines/**``.**
``executors.run_pipeline``'s existing ``budget_cap`` only gates ONE slice's
PASS-2 (``chunk_analyze``) worthy-chunk count, checked once, right after that
slice's own ``chunk_segment`` stage completes (see
``executors.run_pipeline``'s in-loop check). It has NO notion of a multi-slice,
multi-match BATCH: it does not count PASS-1 calls, does not persist across
slices, and does not exist at all outside a single ``run_pipeline`` call. The
dashboard's real cost guarantee — "the batch aborts the instant the NEXT
Gemini call anywhere in the batch would exceed the operator's cap" — has to
live one layer up, driving ``run_pipeline`` slice-by-slice, match-by-match,
and counting every single call (PASS-1 rough-scan *and* PASS-2 deep-tag) as it
happens. That driver is this module. It is dashboard-backend code
(``eval/adcc/predict/**``), never a modification of ``service/pipelines/**``
— those files stay additive-only per Task 1, untouched here.

**Consumes ``run_pipeline``'s event stream directly** (not
``run_predictions.capture_match_predictions``, which drains the whole
generator internally and only returns a final stitched result — no live event
access, no way to count-and-abort mid-stream). Every event this module emits
is the SAME dict ``run_pipeline`` produced, tagged with ``match_id``/
``slice_index`` plus (for the events that represent one spent Gemini call)
the running ``global_calls_spent``/``global_dollars_spent`` totals — additive
tagging only, never a mutation of an existing key.

**Cap semantics (Jobs's accepted ruling, LeCun ratified):** "at most one
in-flight call may complete." Concretely: the moment a call-completion event
(``chunk_map`` = one spent PASS-1 call; ``chunk_result`` = one spent PASS-2
call; an ``error`` event for stage_id in {chunk_segment, chunk_analyze} = one
spent-but-failed call) pushes ``calls_spent >= cap``, this driver (a) yields
that event to the caller as normal (the call already happened, its result is
real and must be shown), (b) immediately closes the inner ``run_pipeline``
generator (``aclose()``) so it can never reach its next ``await
client.aio.models.generate_content(...)``, and (c) yields a terminal
``cap_reached`` event and returns — no further slice, no further match. The
SAME closing mechanism implements operator STOP (``BatchRunState.stop_requested``):
checked at the identical point, right after each event, so a STOP registers
with at most one more in-flight call ever completing (whatever was already
mid-flight when STOP was clicked), never a new one starting.

**Real-vs-placeholder cost accounting** (LeCun): the instant a call-completion
event carries real ``usage_metadata`` (Task 1's additive emit —
``prompt_token_count``/``candidates_token_count``), this module prices it at
the real published per-token rate. Only when ``usage_metadata`` is entirely
absent (e.g. a test double, or an SDK response that never populated it) does
it fall back to the flat per-call placeholder rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from eval.adcc.predict.run_predictions import MatchTarget
from eval.adcc.predict.scope_policy import Slice
from eval.adcc.predict.stitch import stitch_match_predictions
from eval.adcc.predict.youtube_url import normalize_youtube_watch_url
from service.pipelines import registry
from service.pipelines.executors import RunContext, estimate_run_plan, run_pipeline

__all__ = [
    "PASS1_PLACEHOLDER_USD",
    "PASS2_PLACEHOLDER_USD",
    "INPUT_USD_PER_TOKEN",
    "OUTPUT_USD_PER_TOKEN",
    "PIPELINE_ID",
    "BatchRunState",
    "event_cost_usd",
    "is_call_event",
    "run_batch_with_global_cap",
]

PIPELINE_ID = "chunk-segment-tags"

# LeCun's live-counter placeholder rates — used ONLY until a call's real
# usage_metadata lands (Task 1's additive emit), at which point the real
# published per-token rate below takes over for THAT call.
PASS1_PLACEHOLDER_USD: float = 0.02
PASS2_PLACEHOLDER_USD: float = 0.003

# Published Gemini per-token rates (LeCun): $0.25 / 1M input tokens,
# $1.50 / 1M output tokens.
INPUT_USD_PER_TOKEN: float = 0.25 / 1_000_000
OUTPUT_USD_PER_TOKEN: float = 1.50 / 1_000_000


@dataclass
class BatchRunState:
    """Mutable, single-run bookkeeping shared between the dashboard's
    ``/run`` stream and its ``/stop`` handler (the SAME object — ``/stop``
    flips ``stop_requested`` on the instance the running ``/run`` request is
    reading from; there is exactly one active run at a time, per Jobs's
    spec). Never reused across two different runs — the dashboard app
    constructs a fresh one per ``/run`` call."""

    cap: int
    calls_spent: int = 0
    dollars_spent: float = 0.0
    stop_requested: bool = False
    stop_reason: Optional[str] = None
    # Debugging/audit aid only — never consulted for cap/stop decisions.
    events_seen: int = field(default=0, repr=False)


def event_cost_usd(event: dict) -> float:
    """Real cost from ``usage_metadata`` (Task 1's additive emit) when
    present; otherwise LeCun's flat per-call placeholder for the event's
    pass. Never fabricates a token count — absence of usage_metadata is a
    real, expected condition (e.g. a test double, or an SDK response that
    didn't populate it), not an error."""
    usage = event.get("usage_metadata")
    if usage and usage.get("prompt_token_count") is not None and usage.get("candidates_token_count") is not None:
        return (
            usage["prompt_token_count"] * INPUT_USD_PER_TOKEN
            + usage["candidates_token_count"] * OUTPUT_USD_PER_TOKEN
        )
    return PASS1_PLACEHOLDER_USD if event.get("type") == "chunk_map" else PASS2_PLACEHOLDER_USD


def is_call_event(event: dict) -> bool:
    """True iff this event represents exactly ONE spent Gemini call (whether
    it succeeded or failed) — the only events this driver's global counter
    increments on. ``chunk_map`` = the PASS-1 rough-scan call completed (one
    per slice, always). ``chunk_result`` = one PASS-2 deep-tag call completed
    for one worthy chunk. A ``chunk_segment``/``chunk_analyze`` ``error``
    event ALSO represents a spent call (the request was made and failed) —
    counted so a run of failing calls cannot spend unboundedly without ever
    tripping the cap. Every other event type (progress/stage_start/
    stage_complete/run_start/run_complete/skipped-chunk progress) spent
    nothing and is never counted."""
    if event.get("type") in ("chunk_map", "chunk_result"):
        return True
    return event.get("type") == "error" and event.get("stage_id") in ("chunk_segment", "chunk_analyze")


async def _drive_one_slice(
    target: MatchTarget, sl: Slice, *, gemini_api_key: str, request_timeout_ms: int,
    state: BatchRunState, per_slice_budget_cap: Optional[int],
) -> AsyncIterator[dict]:
    """Drives exactly one slice's ``run_pipeline`` to completion OR until the
    global cap/stop fires, tagging every event with match_id/slice_index and
    (for call-completion events) the running global totals. The LAST event
    yielded is either a real ``run_complete`` (untouched, slice ran clean to
    its own end) tagged with ``slice_clips``, or nothing further — the caller
    inspects ``state`` after this generator is exhausted to know why it
    stopped early."""
    pipeline_def = registry.get_default(PIPELINE_ID)
    # LIVE-BUG FIX (2026-07-18): target.video_url is the ADCC gold snapshot's
    # raw /embed/<id> YouTube URL (INS-084) — Gemini's native file_uri
    # ingestion only accepts the canonical watch?v=<id> form (see
    # youtube_url.py's module docstring for the full incident writeup).
    # Normalized HERE, upstream of RunContext, so service/pipelines/
    # executors.py (co-tenant, byte-untouchable) receives an already-good
    # URL and needs zero changes. Fail-closed: a malformed URL raises here,
    # BEFORE any Gemini call — zero spend on a bad URL.
    ctx = RunContext(
        youtube_id=target.video_url_md5, youtube_url=normalize_youtube_watch_url(target.video_url),
        start_sec=int(sl.start_s), end_sec=int(round(sl.end_s)),
        gemini_api_key=gemini_api_key, request_timeout_ms=request_timeout_ms,
    )
    planned = estimate_run_plan(pipeline_def, duration_sec=float(ctx.end_sec - ctx.start_sec))

    yield {
        "type": "slice_start", "match_id": target.match_id, "slice_index": sl.slice_index,
        "scope": [ctx.start_sec, ctx.end_sec],
    }

    gen = run_pipeline(pipeline_def, ctx, planned, budget_cap=per_slice_budget_cap)
    try:
        async for event in gen:
            tagged = dict(event, match_id=target.match_id, slice_index=sl.slice_index)
            state.events_seen += 1

            if is_call_event(event):
                state.calls_spent += 1
                state.dollars_spent += event_cost_usd(event)
                tagged["global_calls_spent"] = state.calls_spent
                tagged["global_dollars_spent"] = round(state.dollars_spent, 4)

            yield tagged

            # Cap/stop check happens AFTER yielding the event that just
            # completed (the call already happened; its result is real and
            # must reach the caller) but BEFORE the generator is allowed to
            # resume toward its next `await generate_content(...)`.
            if is_call_event(event) and state.calls_spent >= state.cap:
                return
            if state.stop_requested:
                return
    finally:
        await gen.aclose()


async def run_batch_with_global_cap(
    targets: list[MatchTarget],
    slices_by_match: dict[str, list[Slice]],
    *,
    gemini_api_key: str,
    request_timeout_ms: int,
    state: BatchRunState,
    per_slice_budget_cap: Optional[int] = None,
) -> AsyncIterator[dict]:
    """Drives the WHOLE batch (every match, every slice, in order) through
    ``_drive_one_slice``, stopping the instant ``state.calls_spent >= state.cap``
    or ``state.stop_requested`` becomes true — checked BEFORE starting every
    new slice (belt) and INSIDE every slice's own event loop (suspenders, see
    ``_drive_one_slice``). Zero Gemini calls happen once either condition is
    true, at any granularity (never just "no new match", also "no new
    slice", also "no new call within an in-progress slice").

    Yields NDJSON-ready dicts: ``batch_start``, ``match_start``, every
    (tagged) ``run_pipeline`` event, ``match_complete`` (per match, once all
    its slices finish clean), and exactly ONE terminal event —
    ``cap_reached``, ``run_stopped``, or ``batch_complete``.
    """
    yield {"type": "batch_start", "match_count": len(targets), "cap": state.cap}

    all_match_results: dict[str, dict] = {}

    for target in targets:
        if state.stop_requested:
            yield {"type": "run_stopped", "reason": state.stop_reason or "stop requested",
                   "calls_spent": state.calls_spent, "dollars_spent": round(state.dollars_spent, 4)}
            return
        if state.calls_spent >= state.cap:
            yield {"type": "cap_reached", "calls_spent": state.calls_spent, "cap": state.cap}
            return

        slices = slices_by_match[target.match_id]
        yield {"type": "match_start", "match_id": target.match_id, "slice_count": len(slices)}

        per_slice_clips: list[list[dict]] = []
        for sl in slices:
            slice_clips: list[dict] = []
            async for event in _drive_one_slice(
                target, sl, gemini_api_key=gemini_api_key, request_timeout_ms=request_timeout_ms,
                state=state, per_slice_budget_cap=per_slice_budget_cap,
            ):
                if event.get("type") == "run_complete":
                    slice_clips = event.get("clips", [])
                yield event
            per_slice_clips.append(slice_clips)

            if state.stop_requested:
                yield {"type": "run_stopped", "reason": state.stop_reason or "stop requested",
                       "calls_spent": state.calls_spent, "dollars_spent": round(state.dollars_spent, 4)}
                return
            if state.calls_spent >= state.cap:
                yield {"type": "cap_reached", "calls_spent": state.calls_spent, "cap": state.cap}
                return

        result = stitch_match_predictions(target.match_id, per_slice_clips)
        all_match_results[target.match_id] = result
        yield {"type": "match_complete", "match_id": target.match_id, "clip_count": len(result["clips"])}

    yield {
        "type": "batch_complete", "calls_spent": state.calls_spent,
        "dollars_spent": round(state.dollars_spent, 4), "results": all_match_results,
    }
