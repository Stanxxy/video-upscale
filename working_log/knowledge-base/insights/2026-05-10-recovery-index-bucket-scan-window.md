---
date: 2026-05-10
category: insight
tags: [service, recovery, reconciler, keyspaces, job_recovery_index, ops]
status: active
---

# Recovery index bucket scan window (overnight stale jobs)

## Context

`job_recovery_index` rows are partitioned by `(recovery_state, heartbeat_bucket, last_heartbeat_at, job_id)` where `heartbeat_bucket` is the UTC calendar hour of `last_heartbeat_at` at write time (`JobsStore.heartbeat_bucket_for`).

The scheduled reconciler (`RecoveryManager.reconcile_once`) discovers stale **ACTIVE** candidates via `JobsStore.list_stale_recovery_candidates`, which queries **one partition per `heartbeat_bucket` value** passed in.

## Problem (historical)

Originally only the **current and previous** hour buckets were scanned. If a worker crashed **many hours** before the next service start (for example overnight), the last durable index row for that job could sit in a bucket **older than those two partitions**. The job’s `job_lifecycle` row could still read `RUNNING`, but the reconciler **never saw** the stale index entry, so no `INTERRUPTED` handoff and no replacement scheduling occurred.

## Current behavior

- **`RecoveryManager.heartbeat_buckets_for_scan(now, hours=N)`** returns **N** distinct `YYYYMMDDHH` strings: `now`’s hour in UTC, then each prior hour, going back **N − 1** additional hours (inclusive window of **N** hours of wall clock).
- **`recovery_heartbeat_bucket_hours`** on `ServiceConfig` (default **24**, max **168**). Environment: **`BJJ_RECOVERY_HEARTBEAT_BUCKET_HOURS`** (with the usual `BJJ_` prefix from pydantic-settings).
- **`RecoveryManager`** is constructed with `heartbeat_bucket_hours=config.recovery_heartbeat_bucket_hours` from `service/app.py`.
- **`drain_orphan_pending_jobs_on_startup`** receives the same value so **orphan `PENDING`** discovery uses the same trailing window (it calls `list_active_recovery_index_rows_newest_first` per bucket).

## Operational notes

- **Cost:** Each reconcile interval runs **one `SELECT` per distinct bucket** in the window (default 24 queries every 30 seconds). Increasing the window toward **168** improves visibility for very old stuck rows but **linearly increases** Keyspaces read load on each tick.
- **This does not change** who is eligible for recovery: lifecycle must still be **`RUNNING` or `INTERRUPTED`**, heartbeat older than **`stale_after`** (default 90s), no **`replacement_job_id`**, and claim CAS must succeed. **`AWAITING_CORRECTION`**, **`FAILED`**, and terminal states are unchanged.

## Verification

- `pytest tests/test_reconciler.py -v` includes `test_reconcile_once_finds_stale_job_in_bucket_many_hours_behind` and `test_reconcile_once_narrow_bucket_hours_misses_distant_partition`.

## Primary files

- `service/config.py` — `recovery_heartbeat_bucket_hours`
- `service/reconciler.py` — `heartbeat_buckets_for_scan`, constructor `heartbeat_bucket_hours`
- `service/app.py` — passes config into `RecoveryManager` and `drain_orphan_pending_jobs_on_startup`
- `service/routes.py` — `drain_orphan_pending_jobs_on_startup(..., heartbeat_bucket_hours=...)`
