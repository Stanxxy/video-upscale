---
date: 2026-05-25
category: insight
tags: [service, parallel-upscale, keyspaces, progress, asyncio, regression-prevention]
status: active
---

# Parallel-upscale progress aggregator pattern (`PipelineStage.UPSCALE_ANALYZE`)

## Context

The vision engine splits the UPSCALE_ANALYZE stage across `N` memory-bounded
segments when the clip is too large to fit in a single segment **or** when
`config.standard_segments > 1`. Each segment runs in its own thread executor
against its own `_run_upscale_analysis` invocation. Without an aggregator,
each segment's `progress_cb` would be ignored (`progress_cb=None` in the
prior implementation), and the lifecycle row's `progress_percent` would
stay frozen in the 55%-80% band for the full upscale duration — exactly the
"frozen progress bar" pattern that originally motivated this aggregator for
tracking.

The earlier draft of this insight described the pattern at the **tracking**
stage. That implementation was rejected during contract review (PR #2
revision) because parallel tracking with `detection_cb=None` violates the
mid-track-loss V1 contract (`CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` §`track`,
`JOB_ROTATION_HANDOFF_AND_RESUME.md` §4.1). Post-refactor:
**tracking is always sequential; the aggregator now lives in the
upscale-analyze stage only**, where there are no per-segment human-in-the-
loop semantics to honor and the V1 contract is unaffected (each
`_run_upscale_analysis` segment writes its own `upscale_analyze` checkpoint
rows independently per the addendum).

## Where it lives

`service/worker.py:_run_parallel_upscale` (post-refactor function name).
Called from `run_job` after the sequential tracking stage completes, when
`_use_parallel_upscale` is true (multiple segments AND not a resume AND not
`skip_upscale`).

## Design

1. **Shared `dict[int, int] frames_done`**, seeded with `{i: 0 for i in
   range(n_total)}`. Each segment owns its own key; thread-safe **without a
   lock** because integer assignment to a distinct dict key is atomic under
   the GIL.
2. **Per-segment `progress_cb` closure** (`_make_seg_progress_cb`) converts
   the per-segment processed/total fraction reported by
   `_run_upscale_analysis` (`service/worker.py:~2384`) into an absolute
   frames-done count: `done = round(fraction * seg_len)`. Each closure does
   exactly one assignment.
3. **asyncio heartbeat task** (`_aggregator_task`) running on the engine's
   main loop. It awakes every `LIFECYCLE_HEARTBEAT_INTERVAL`
   (`service/worker.py:63`, 1.0s) and calls
   `_aggregate_and_write_lifecycle`.
4. **Aggregated upscale-band progress**: `pct = 55.0 + (sum(frames_done) /
   total_seg_frames) * 25.0`, floored by `progress_floor` (so replacement
   jobs cannot regress) and capped at `80.0` (the band ceiling — the
   stage-complete write at 80% owns the boundary cleanly).
5. **Clean shutdown**: `aggregator_stop.set()` + `aggregator.cancel()` +
   `await aggregator` guarantees the task is reaped before
   `_run_parallel_upscale` returns, then a **final aggregated write**
   captures any frames produced after the last 1Hz tick.

## What the aggregator does NOT do

- It NEVER writes `set_state(...)` — job state transitions remain owned by
  the worker and `_make_detection_cb` (see
  `tests/test_parallel_upscale_progress.py::test_parallel_upscale_aggregator_does_not_write_checkpoints`).
- It NEVER writes `jobs_store.write_checkpoint(...)` — V1 checkpoint rows
  are owned by `_run_upscale_analysis` per
  `CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` §`stage_name == upscale_analyze`.
  Each segment's `_run_upscale_analysis` independently writes its own
  `analysis_started` / `analysis_window_completed` rows.
- It NEVER writes outside the 55-80 band — the ceiling clamp is explicit.

## Why this is engine-side only

The SSE controller and frontend hook are unchanged. The whole contract
change is internal to the engine: `progress_percent` on the Keyspaces
lifecycle row now advances during parallel upscale instead of being
pinned at 55%. `PipelineStage.UPSCALE_ANALYZE` (not `TRACK`) is the only
stage written by the aggregator, matching the band.

## Verification

- New tests (`tests/test_parallel_upscale_progress.py`, 4 cases):
  monotonic in-band progress, ceiling clamp at 80%, no checkpoint or
  set_state writes from the aggregator, clean aggregator shutdown on
  segment failure.
- Companion: `tests/test_tracking_always_sequential.py` asserts the
  inverse — when conditions would have triggered the old parallel-tracking
  path, tracking still goes sequential and emits a V1
  `build_track_mid_loss` envelope on simulated mid-track loss.
- Full engine suite **271/271** passing post-refactor.

## Companion insights

- [`run_coroutine_threadsafe` must `add_done_callback`](2026-05-25-run-coroutine-threadsafe-must-add-done-callback.md) — pure improvement, applies anywhere the worker schedules fire-and-forget coroutines.
- [Headless-mode `_detect_and_request_boxes` short-circuit](2026-05-25-headless-detect-yolo-short-circuit.md) — defense-in-depth: production tracking always passes a non-None `detection_cb` after this refactor, so the short-circuit is for CLI/headless tests only.

## Future upgrade

The "pre-scan segmented parallel tracking" requirement under
`working_log/knowledge-base/requirements/2026-05-25-prescan-segmented-parallel-tracking.md`
describes how a pre-scan stage could segment the video at known-good
boundaries so tracking could also run in parallel. That work is
intentionally deferred — this insight covers the
**upscale-only** parallelism that ships in this PR.

## Primary files

- `service/worker.py:_run_parallel_upscale` — the helper plus aggregator.
- `service/worker.py:_pct_at_least` — the floor helper shared with the
  sequential paths.
- `service/segment_runner.py` — `compute_segment_ranges` and
  `merge_analysis_results` (identity-stitching helpers are no longer
  needed since tracking is sequential).
- `tests/test_parallel_upscale_progress.py` — regression guards (4
  cases).
- `tests/test_tracking_always_sequential.py` — inverse regression guard
  (sequential mid-track loss writes V1 envelope).
