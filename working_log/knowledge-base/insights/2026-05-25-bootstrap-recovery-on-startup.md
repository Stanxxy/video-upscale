---
date: 2026-05-25
category: insight
tags: [service, recovery, reconciler, keyspaces, bootstrap-recovery, observability, regression-prevention]
status: active
---

# Bootstrap recovery on startup (auto-resume latency fix)

## Problem

Users reported the vision engine "never auto-resumes" jobs after the
service is restarted mid-tracking. The recovery code in `develop` and on
PR #2 was identical (and correct) — this was a **UX latency bug**, not a
regression.

Two pre-existing issues stacked:

1. **90-120 s recovery latency after restart.** `RecoveryManager(stale_after=90, interval=30)`
   in `service/reconciler.py` skips any candidate whose
   `last_heartbeat_at >= now − 90 s`. On a fresh restart, the previous
   process's last heartbeat is usually only a few seconds old, so the
   first periodic tick after restart skips the orphaned row. Recovery
   then waits for the 90 s window plus up to one 30 s tick before the
   replacement is created — felt by the operator as "broken."

2. **Silent failures inside `recover_interrupted_job`.** Lifecycle write
   errors raised `HTTPException(500)` (intended for the manual
   `POST .../resume` HTTP path); the reconciler's outer `except Exception`
   logged the message at `WARNING` **without `exc_info`**. Real stack
   traces never landed in `service.log`, so any genuine recovery failure
   was invisible.

## Solution

### 1. Single-shot bootstrap sweep at lifespan startup

New `bootstrap_recovery_on_startup(instance_id, recover_job, *, heartbeat_bucket_hours)`
in `service/routes.py` runs **once** during FastAPI lifespan, immediately
after `drain_orphan_pending_jobs_on_startup` and **before**
`HeartbeatTask.start()` and `RecoveryManager.start()`.

It constructs a one-shot `RecoveryManager` and calls
`reconcile_once(stale_after_override=0.0)`. The override is a new keyword
argument on `RecoveryManager.reconcile_once`:

```python
async def reconcile_once(self, *, stale_after_override: float | None = None) -> None:
```

When `stale_after_override=0`, `stale_before = now`, so any
recovery-index candidate whose lifecycle `last_heartbeat_at` is strictly
in the past is considered stale. Because this process has **not yet
heartbeated anything** (no `register_owned_job` has been called, no
`HeartbeatTask` is running), that set is exactly the orphaned
`RUNNING` / `INTERRUPTED` rows from the previous owner.

The override only affects the heartbeat-staleness threshold; the
recovery-index bucket scan window (`heartbeat_bucket_hours`) is
unchanged, and the periodic loop continues to use the conservative
`self._stale_after` default. Periodic defaults are deliberately kept
conservative so a future multi-instance deployment does not race
healthy peer instances.

### 2. Observability fixes

- `RecoveryManager._run` — `exc_info=True` on the failure log; INFO line
  at startup: `RecoveryManager started: instance=<id> interval=<s> stale_after=<s> bucket_hours=<n>`.
- `RecoveryManager.reconcile_once` — INFO one-liner per tick with
  candidate count; DEBUG-level skip reasons (`state-not-eligible`,
  `replacement-set`, `heartbeat-fresh`); INFO `Recovery: claimed and
  dispatched job_id=<id> from owner=<old> heartbeat=<old_dt>` on
  successful dispatch.
- `HeartbeatTask._run` — `exc_info=True` on failure log; INFO startup
  line `HeartbeatTask started: instance=<id> interval=<s>`.
- `recover_interrupted_job` — body wrapped in `try/except Exception`
  that logs `recover_interrupted_job FAILED for job_id=<id>` with
  `exc_info=True` and re-raises so callers still see the failure.
- `service/app.py` — INFO `RecoveryManager + HeartbeatTask started`
  after `recovery.start()` so operators can confirm both background
  loops are running.

## Safety analysis

`stale_after_override=0` is **only safe under the current
single-instance-per-process assumption** that the codebase makes today
(one engine per VM, ownership tracked via `owner_instance_id`).

In the current design:

- The bootstrap runs strictly before the local `HeartbeatTask` starts.
- The local process owns no jobs at that moment.
- The conditional claim CAS (`claim_job_for_recovery`) still verifies
  `expected_state` (`RUNNING` / `INTERRUPTED`), `expected_owner_instance_id`,
  and `expected_last_heartbeat_at`, so even with `stale_before = now`
  the bootstrap cannot trample a row that another writer modified in
  the same window.

If multi-instance is ever added (multiple vision engines fronting the
same Keyspaces cluster), this bootstrap **must** add a peer-coordination
guard before merging — for example, only treat heartbeats as stale when
no known peer's heartbeat ticked in the same window, or fall back to
the periodic `stale_after=90s` policy when peers are detected. The
periodic loop defaults are intentionally left untouched so that
transition is not blocked by this change.

## Verification

- `./venv/bin/python -m pytest tests/test_reconciler.py tests/test_bootstrap_recovery.py tests/test_startup_pending_drain.py -v` → 17 passed.
- Full `./venv/bin/python -m pytest tests/` → 275 passed, 1 pre-existing
  `test_restorer_batch.py::test_enhance_batch_mixed_size_semantic`
  failure (MPS FP16 numerical drift; out of scope per evaluator note).
- `tests/test_bootstrap_recovery.py` covers: (a) fresh-heartbeat orphan
  is claimed, (b) default periodic `reconcile_once()` leaves the same
  row alone (control case), (c) `recover_job` exception is logged with
  `exc_info` and does **not** propagate out of
  `bootstrap_recovery_on_startup`, (d) the scan/complete INFO log lines
  emit so operators can confirm the sweep ran.
- `tests/test_reconciler.py::test_reconcile_once_stale_after_override_picks_up_fresh_heartbeats`
  proves the override keyword behaves correctly and the default path is
  unchanged.

## Primary files

- `service/reconciler.py` — `reconcile_once(stale_after_override=...)`,
  observability log lines.
- `service/routes.py` — `bootstrap_recovery_on_startup`,
  `recover_interrupted_job` error wrapper.
- `service/heartbeat.py` — observability log lines.
- `service/app.py` — bootstrap wired into lifespan between
  `drain_orphan_pending_jobs_on_startup` and the background loops.
- `tests/test_bootstrap_recovery.py` — new.
- `tests/test_reconciler.py` — extended.

## References

- PR #2 (`fix/parallel-segment-progress-and-yolo-thrash`).
- Commit `57cb33f` (Gemini default model bump) — bootstrap commit lands
  on top of this.
- Contract: `working_log/contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md`
  §5 (Automatic recovery) and §5.1 (Process restart — orphan PENDING rows).
- Prior insight: `2026-05-01-lifecycle-resume-recovery-implementation.md`
  (`RecoveryManager` design).
- Prior insight: `2026-05-10-recovery-index-bucket-scan-window.md`
  (bucket scan window is independent of `stale_after`).
