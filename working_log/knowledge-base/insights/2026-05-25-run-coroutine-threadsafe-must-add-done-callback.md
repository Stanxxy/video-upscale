---
date: 2026-05-25
category: insight
tags: [service, asyncio, keyspaces, progress, regression-prevention]
status: active
---

# `run_coroutine_threadsafe` requires `add_done_callback` for fire-and-forget paths

## Context

The vision engine schedules best-effort heartbeats (lifecycle row updates,
upscale progress writes, partial-tracking uploads) from worker threads back
onto the asyncio loop via `asyncio.run_coroutine_threadsafe(coro, loop)`. The
calling thread does **not** block on `future.result()` because heartbeats are
best-effort — losing one is fine.

The trap: a `concurrent.futures.Future` returned by
`run_coroutine_threadsafe` **silently swallows the wrapped coroutine's
exception** if nobody ever inspects it. There is no warning, no log, no
process exit — `asyncio` will not even surface it as an "exception was never
retrieved" warning the way it does for `asyncio.Task`, because the cross-loop
Future is consumed in a different module.

This is what kept job `b75d43e0-5739-4ed4-aedd-b50012a5b62a` un-diagnosed for
so long: even after the aggregator pattern was in place, **if** the Keyspaces
UPDATE inside `_aggregate_and_write_lifecycle` had failed for a transient
reason (network blip, throttling, serialization error), the engine log would
have stayed clean while the UI stayed frozen — identical user-visible
symptoms to "no aggregator at all", with zero log evidence to distinguish
them.

## Fix (PR https://github.com/Stanxxy/video-upscale/pull/2, commit `92e8240`)

Two new helpers in `service/worker.py`:

1. `_log_progress_future_failure(future, *, context)` at
   `service/worker.py:66-87` — `add_done_callback` body. Calls
   `future.exception()` inside a `try/except` so a cancelled Future doesn't
   re-raise, and `logger.error(..., exc_info=exc)` only when there's a real
   exception. Safe to attach to **any** fire-and-forget Future.

2. `_schedule_background_coro(coro, loop, *, context)` at
   `service/worker.py:90-101` — wraps the
   `run_coroutine_threadsafe` + `add_done_callback` boilerplate. Returns the
   Future so callers that want to cancel can; the failure hook is attached
   regardless. Forcing callers through this helper makes it impossible to
   "forget" the failure logging.

Two existing fire-and-forget sites were migrated:

- `service/worker.py:724-735` — sequential `tracking_progress_cb` heartbeat
  (replaces the prior bare `asyncio.run_coroutine_threadsafe` call).
- `service/worker.py:1046-1052` — `_upscale_progress` heartbeat in
  `_run_upscale_analysis` scheduling.

The aggregator inside `_run_parallel_segments` runs on the engine loop
itself (it is an `asyncio.create_task`) and does its own
`try/except` around the Keyspaces writes (`service/worker.py:2626-2644`), so
it needs no `_schedule_background_coro` wrapping — but it follows the same
"log, never raise" pattern as `_log_progress_future_failure`. The docstring
at line 2638 explicitly notes "Match `_log_progress_future_failure`
semantics: log, never raise."

## Rule

**Any** new `asyncio.run_coroutine_threadsafe(...)` call in this codebase
must either:

- be scheduled via `_schedule_background_coro(...)`, or
- have the developer immediately call `future.result()` (which propagates the
  exception synchronously), or
- attach `add_done_callback(_log_progress_future_failure)` explicitly with a
  context string.

A bare `run_coroutine_threadsafe` that is neither awaited nor wrapped is a
regression and reviewers should reject it.

## Verification

- New tests (`tests/test_worker_background_coro.py`, 3 cases): coroutine
  succeeds, coroutine raises (failure is logged), Future is cancelled
  (callback handles `CancelledError` gracefully).
- Full engine suite **269/269** passing post-fix.

## Companion insights

- [Parallel-segment progress aggregator pattern](2026-05-25-parallel-segment-progress-aggregator.md) — the root visible bug this hook protects.

## Primary files

- `service/worker.py:66-101` — `_log_progress_future_failure` + `_schedule_background_coro`.
- `service/worker.py:724-735` — sequential tracking heartbeat call site.
- `service/worker.py:1046-1052` — upscale-progress heartbeat call site.
- `service/worker.py:2626-2644` — aggregator's in-loop equivalent (try/except log-never-raise).
- `tests/test_worker_background_coro.py` — regression guard.
