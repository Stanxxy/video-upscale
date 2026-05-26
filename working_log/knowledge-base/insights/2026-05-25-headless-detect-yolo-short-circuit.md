---
date: 2026-05-25
category: insight
tags: [service, parallel-tracking, tracking, ml, regression-prevention]
status: active
---

# `_detect_and_request_boxes` short-circuit in headless / parallel-segment mode

## Context

`tracking_pipeline/hybrid_tracking.py:_detect_and_request_boxes`
(`hybrid_tracking.py:724`) runs on every SAM2 track-loss frame. In the
service path it does three things:

1. lazy-load the YOLO detector (`YOLO26Detector(...)`),
2. run a forward pass to get candidate person bounding boxes, and
3. dispatch the boxes through `detection_callback(...)` so a human can
   confirm / pick / cancel before tracking resumes.

**Status after the 2026-05-25 refactor:** the short-circuit is now a
**defense-in-depth safety net**, not a production hot path. Production
tracking is always sequential (`run_job` in `service/worker.py` always
passes a non-None `detection_cb` built by `_make_detection_cb`), so this
branch only fires for CLI/headless callers (e.g. `tracking_pipeline.cli`
or the future ad-hoc analysis script flows). The earlier PR-2 design
that explicitly relied on `detection_cb=None` inside a K-segment
parallel-tracking helper (`_run_parallel_segments`) was deleted because
it violated the mid-track-loss V1 contract
(`CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` §`track`,
`JOB_ROTATION_HANDOFF_AND_RESUME.md` §4.1). Keeping the short-circuit
nonetheless prevents the YOLO-reload thrash regression from ever
returning if some new headless caller is added later.

## Root cause

Pre-fix, `_detect_and_request_boxes` checked `detection_callback` **only
after** loading YOLO and running a forward pass. When the callback was
`None` the function returned `None` without surfacing the local `detector`
to the outer loop, so on the **very next** track-loss frame the outer loop
would call `_detect_and_request_boxes` again and YOLO would be reloaded
from disk — repeatedly, several times per second under sustained occlusion.
Logs filled with `[detect] Loading yolo26m.pt on mps (persistent)...` every
few frames during parallel tracking, wasting wall-clock and Apple-Silicon
unified memory.

This was a pure-waste hot path: YOLO detections were computed, formatted
into JPEG-encoded payloads for the (non-existent) callback, and immediately
discarded — there was no consumer.

## Fix (PR https://github.com/Stanxxy/video-upscale/pull/2, commit `92e8240`)

One-line short-circuit at the **top** of `_detect_and_request_boxes`
(`tracking_pipeline/hybrid_tracking.py:742-747`):

```python
if detection_callback is None:
    # Parallel-segment mode / headless run: nothing to do with YOLO output.
    # Return None to signal track loss; caller handles via max_missing_frames.
    # Do NOT try CLI select_boxes — not available in server context.
    print(f"  Frame {global_idx}: Track lost, no detection_callback — continuing")
    return None
```

The caller already handles `None` returns via the existing
`max_missing_frames` budget in `hybrid_tracking.py`'s main tracking loop, so
the parallel path naturally lets SAM2 drift through the gap until it
re-locks or the segment ends — exactly the behavior we want for headless
parallel tracking.

The detailed docstring at `hybrid_tracking.py:732-741` documents why the
short-circuit exists, cross-references the parallel path, and notes the
old log-flood symptom so future readers don't try to "restore" YOLO
re-detection in headless mode.

## What did NOT change

- The sequential service path (`detection_callback != None`) is fully
  preserved — YOLO loads on demand, forwards, and dispatches to the
  human-in-the-loop callback as before. Same behavior, same
  `HumanVerificationSuspend` semantics (`hybrid_tracking.py:768-769`).
  After the 2026-05-25 refactor, **this is the only production path**.
- `max_missing_frames` budget is unchanged.
- The detector cache lifecycle (lazy load, persistent across calls when a
  callback exists) is unchanged.

## Verification

- 2 new tests in `tests/test_human_verification_suspend.py` cover the
  short-circuit branch (returns `None` immediately when callback is `None`)
  and confirm YOLO is not instantiated.
- Full engine suite **269/269** passing post-fix.

## Companion insights

- [Parallel-upscale progress aggregator pattern](2026-05-25-parallel-upscale-progress-aggregator.md) — the post-refactor home of the aggregator (upscale stage, not tracking).
- [`run_coroutine_threadsafe` must add_done_callback](2026-05-25-run-coroutine-threadsafe-must-add-done-callback.md) — pure-improvement companion fix shipped in the same PR.

## Primary files

- `tracking_pipeline/hybrid_tracking.py:724-772` — `_detect_and_request_boxes` with the headless short-circuit at the top.
- `service/worker.py:_make_detection_cb` — the **only** production callsite that wires `detection_cb` (always non-None) after the refactor.
- `tests/test_human_verification_suspend.py` — regression guard (3 cases including the headless short-circuit).
