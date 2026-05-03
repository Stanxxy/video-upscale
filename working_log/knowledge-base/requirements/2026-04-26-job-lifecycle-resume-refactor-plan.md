---
date: 2026-04-26
category: requirement
tags: [service, lifecycle, resume, crash-recovery, keyspaces, governance]
status: active
---

# Job Lifecycle Resume Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make job lifecycle, manual correction resume, user cancellation, and worker crash recovery durable through Keyspaces and S3 checkpoint artifacts.

**Architecture:** Manual resume is only for `AWAITING_CORRECTION` jobs and always creates a new job; the replaced old job becomes terminal `CANCELLED` with a replacement pointer. Crash recovery is owned by a vision-engine scheduled recovery manager that reads a Keyspaces recovery index/table, marks stale `RUNNING` jobs `INTERRUPTED`, claims ownership with a conditional update on `owner_instance_id`, then creates a replacement job from durable checkpoint/request state. In-memory worker state remains an execution optimization, never the source of truth.

**Tech Stack:** FastAPI, asyncio workers, Amazon Keyspaces/Cassandra, S3, pytest, existing `service.jobs_store.JobsStore`, `service.worker.run_job`, and `service.routes`.

---

## Architecture Implementation Plan

Non-negotiable invariants:

- `video_analysis_latest_job` always points to the latest job for a video, including running, awaiting-correction, interrupted, cancelled, failed, and completed jobs.
- User-initiated cancellation is terminal: lifecycle becomes `CANCELLED` and no replacement active task is spawned.
- Detection-correction offload is not user cancellation: the old job remains `AWAITING_CORRECTION` until the client calls resume.
- Manual correction resume creates a replacement job with a new job ID.
- After manual correction resume creates the replacement job, the old `AWAITING_CORRECTION` job becomes terminal `CANCELLED`.
- `origin_job_id` points to the root job in the chain.
- `parent_job_id` points to the immediate predecessor.
- `replacement_job_id` points from an old job to the job that replaced it.
- Stale `RUNNING` jobs transition to `INTERRUPTED` before automatic recovery creates a replacement job.
- Automatic recovery is handled by the vision engine, not the companion video analysis backend.
- Recovery requires a scheduled/background manager, not only a one-time startup scan.
- `job_stage_checkpoints.completed` means the whole job is completed, not merely stage-completed.
- Checkpoint artifacts live under a dedicated checkpoint bucket/prefix keyed by job ID and expire within 48 hours.

Rejected alternatives:

- Reusing the same job ID for resume is rejected because the required behavior is a new job per manual or automatic resume.
- Using `/resume` for crash recovery is rejected because `/resume` is only for `AWAITING_CORRECTION` jobs with corrected boxes.
- Relying on `JobsStore.owned_jobs` for recovery is rejected because it is process-local and lost on crash.
- Delegating monitoring/recovery to the companion video analysis backend is rejected because that service should only handle instant request/response behavior, not monitor processes.
- Treating `stage` and `status` as checkpoint-data fields is rejected because they duplicate `job_lifecycle`.

## File Structure

- Modify `service/models.py`: add response/request fields if route responses need lineage or replacement pointers.
- Modify `service/analysis_keyspaces_enums.py`: keep current states unless a superseded terminal state is explicitly added later.
- Modify `service/jobs_store.py`: add lifecycle lineage fields, replacement pointer helpers, checkpoint selection helpers, cancellation checkpoint helper, conditional claim helpers, and recovery discovery helpers.
- Modify `service/routes.py`: fix manual resume lineage, detect/track checkpoint selection, duplicate resume rejection, and Keyspaces-only cancellation.
- Modify `service/worker.py`: standardize checkpoints, fix mid-track detection frame upload, add stage-specific checkpoint writes, and wire analysis checkpoint artifacts.
- Modify `service/reconciler.py`: replace stub with a scheduled recovery manager for stale-job interruption and automatic replacement scheduling.
- Modify `service/heartbeat.py`: keep heartbeat behavior, but ensure ownership handoff logic does not rely on local `owned_jobs` for recovery discovery.
- Create `service/checkpoints.py`: define checkpoint schema helpers, stage ordering, artifact key helpers, and resume cursor builders.
- Modify `tests/test_resume_endpoint.py`: update route tests for lineage, duplicate resume, and detect/track checkpoint selection.
- Create `tests/test_job_cancellation.py`: cover user cancellation for active and Keyspaces-only jobs.
- Create `tests/test_checkpoint_schema.py`: cover checkpoint schema helper output and stage ordering.
- Create `tests/test_reconciler.py`: cover stale `RUNNING` to `INTERRUPTED` to replacement-job scheduling with conditional ownership claim.

## Task 1: Schema And Store Contract

**Files:**

- Modify: `service/jobs_store.py`
- Test: `tests/test_resume_endpoint.py`
- **Step 1: Confirm Keyspaces schema migration outside this repo**

Required lifecycle columns:

```sql
ALTER TABLE video_analysis.job_lifecycle ADD parent_job_id text;
ALTER TABLE video_analysis.job_lifecycle ADD replacement_job_id text;
```

Required recovery index/table:

```sql
CREATE TABLE video_analysis.job_recovery_index (
  recovery_state text,
  heartbeat_bucket text,
  last_heartbeat_at timestamp,
  job_id text,
  owner_instance_id text,
  video_id text,
  job_state text,
  updated_at timestamp,
  PRIMARY KEY ((recovery_state, heartbeat_bucket), last_heartbeat_at, job_id)
) WITH CLUSTERING ORDER BY (last_heartbeat_at ASC, job_id ASC);
```

Use `recovery_state` values such as `ACTIVE`, `AWAITING_CORRECTION`, and `TERMINAL`. The scheduled recovery manager only scans `ACTIVE` buckets. `heartbeat_bucket` should be a coarse time bucket, such as `YYYYMMDDHH`, so the manager can scan the current and previous buckets without full-table scans. If Keyspaces secondary indexes are approved for the deployment, add a secondary-index access path on `owner_instance_id` or `job_state` only for bounded operational lookups; do not rely on unbounded scans.

If the actual keyspace/table names differ in the companion backend, update both services together before code uses these columns.

- **Step 2: Write failing lineage test**

Add a route/store test asserting the replacement lifecycle has root origin and immediate parent:

```python
assert new_lifecycle["origin_job_id"] == job_id
assert new_lifecycle["parent_job_id"] == job_id
old_lifecycle = await jobs_store.get_lifecycle(job_id)
assert old_lifecycle["replacement_job_id"] == new_job_id
```

Run:

```bash
source venv/bin/activate && pytest tests/test_resume_endpoint.py -v
```

Expected before implementation: fails because `parent_job_id` and `replacement_job_id` are not supported and persisted `origin_job_id` currently points to the new job.

- **Step 3: Extend `JobsStore.create_lifecycle`**

Add optional `parent_job_id` and `replacement_job_id` parameters. Insert and read those columns in `create_lifecycle()` and `get_lifecycle()`.

- **Step 4: Add replacement pointer helper**

Add:

```python
async def set_replacement(self, job_id: str, replacement_job_id: str) -> bool:
    ...
```

It updates `job_lifecycle.replacement_job_id` and `updated_at`.

- **Step 5: Add recovery index helpers**

Add helpers to upsert and remove recovery index rows whenever lifecycle state, heartbeat, owner, or latest progress changes:

```python
async def upsert_recovery_index(self, lifecycle: dict) -> bool:
    ...

async def remove_recovery_index(self, job_id: str, recovery_state: str, heartbeat_bucket: str) -> bool:
    ...

async def list_stale_recovery_candidates(self, heartbeat_buckets: list[str], stale_before: datetime) -> list[dict]:
    ...
```

Terminal states should either move to `TERMINAL` with short retention or be removed from active recovery scans.

- **Step 6: Verify store contract**

Run:

```bash
source venv/bin/activate && pytest tests/test_resume_endpoint.py -v
```

Expected after implementation: lineage assertions pass.

## Task 2: Manual Correction Resume

**Files:**

- Modify: `service/routes.py`
- Modify: `service/checkpoints.py`
- Test: `tests/test_resume_endpoint.py`
- **Step 1: Create checkpoint selection tests**

Cover these cases:

- Initial detection: only `detect.pending_detection` exists, resume creates a new job with corrected boxes.
- Mid-track detection: `track.pending_detection` plus `partial_tracking_s3_key` exists, resume includes `resume_tracking_s3_key` and `resume_from_frame` as the next unprocessed frame.
- Duplicate resume: old job already has `replacement_job_id`, endpoint returns conflict and does not create another job.
- **Step 2: Create `service/checkpoints.py`**

Define:

```python
STAGE_ORDER = [
    PipelineStage.DOWNLOAD,
    PipelineStage.DETECT,
    PipelineStage.TRACK,
    PipelineStage.UPSCALE_ANALYZE,
    PipelineStage.ANNOTATE,
    PipelineStage.UPLOAD,
    PipelineStage.PUBLISH,
]
```

Add helper functions:

- `checkpoint_by_stage(checkpoints)`
- `select_correction_checkpoint(checkpoints)`
- `next_unprocessed_frame(checkpoint_data)`
- `build_verified_boxes_checkpoint(box_a, box_b, source_stage)`
- **Step 3: Fix resume lineage**

In `submit_detection_response`, create the new lifecycle with:

```python
origin_job_id = lifecycle.get("origin_job_id") or job_id
parent_job_id = job_id
```

Then call `set_replacement(job_id, new_job_id)` after successful new lifecycle creation.

- **Step 4: Keep `/resume` scoped to `AWAITING_CORRECTION`**

Keep the existing 409 behavior for states other than `AWAITING_CORRECTION`.

- **Step 5: Update old-job terminal behavior**

After replacement creation, set the old job to terminal `CANCELLED` and persist `replacement_job_id=new_job_id`. The new job becomes the latest job for the video.

## Task 3: User Cancellation

**Files:**

- Modify: `service/routes.py`
- Modify: `service/jobs_store.py`
- Test: `tests/test_job_cancellation.py`
- **Step 1: Write cancellation tests**

Cover:

- Active in-memory job: sets cancellation event, cancels task if present, writes lifecycle `CANCELLED`.
- Keyspaces-only job: no in-memory job exists, lifecycle still becomes `CANCELLED`.
- Job with existing replacement: cancellation returns conflict or targets only the latest replacement job.
- Cancellation writes final checkpoint with `reason` and `resume_cursor`.
- **Step 2: Update `DELETE /job/{job_id}` lookup**

Use Keyspaces first, then in-memory fallback. Do not return 404 when a lifecycle row exists.

- **Step 3: Add cancellation checkpoint writer**

Write a final checkpoint with:

```json
{
  "schema_version": 1,
  "reason": "user_cancelled",
  "resume_cursor": {"frame_idx": <last_known_frame_or_zero>},
  "progress_percent": <last_lifecycle_progress>
}
```

- **Step 4: Guarantee no replacement task**

Cancellation must not call resume logic and must not create a new active task.

## Task 4: Standard Checkpoint Schema

**Files:**

- Create: `service/checkpoints.py`
- Modify: `service/worker.py`
- Test: `tests/test_checkpoint_schema.py`
- **Step 1: Define schema version and common fields**

Common checkpoint data:

```json
{
  "schema_version": 1,
  "progress_percent": 0.0,
  "start_frame": 0,
  "frame_count": 0,
  "resume_cursor": {"frame_idx": 0},
  "reason": "stage_progress",
  "input_artifacts": {},
  "output_artifacts": {}
}
```

Do not include `stage` or `status` inside checkpoint data.

- **Step 2: Define pending detection schema**

Required nested shape:

```json
{
  "pending_detection": {
    "frame_idx": 0,
    "frame_s3_key": "checkpoints/<job_id>/frame_0.jpg",
    "frame_bucket": "<bucket>",
    "candidates": [],
    "suggested_boxes": null,
    "reason": "initial"
  }
}
```

- **Step 3: Define artifact semantics**

Use:

- `input_artifacts`: durable inputs needed to resume the stage, such as source video key, partial tracking key, raw analysis key, or player reference keys.
- `output_artifacts`: durable outputs produced by the stage, such as tracking JSON key, analysis JSON key, annotated video key, SNS publish metadata, or checkpoint frame key.
- **Step 4: Update worker writes**

Replace ad hoc checkpoint dictionaries in `worker.py` with helper-built schemas for detect, track, upscale/analyze, annotate, upload, publish, and cancellation.

## Task 5: Tracking And Detection Checkpoints

**Files:**

- Modify: `service/worker.py`
- Test: `tests/test_resume_endpoint.py`
- **Step 1: Fix mid-track frame upload**

Replace `s3.upload_file(frame_jpeg, ...)` with `s3.put_object(request.bucket, frame_s3_key, frame_jpeg, "image/jpeg")`.

- **Step 2: Store next unprocessed frame**

For mid-track detection, checkpoint `resume_cursor.frame_idx` and route `resume_from_frame` should use the next unprocessed frame.

- **Step 3: Store global and relative frame indices**

Partial tracking JSON should preserve global `frame_idx` and include relative frame indices for selected clip-range resume.

- **Step 4: Keep corrected boxes frame-scoped**

When resuming mid-track, corrected boxes apply to the resume frame only and should not rewrite earlier athlete identity state.

## Task 6: Upscale/Analysis Recovery

**Files:**

- Modify: `service/worker.py`
- Modify: `service/checkpoints.py`
- Test: `tests/test_checkpoint_schema.py`
- **Step 1: Upload raw analysis checkpoint artifacts**

When `_run_upscale_analysis` writes local `analysis_raw.json`, upload it to the checkpoint bucket/prefix and write a checkpoint containing:

```json
{
  "resume_cursor": {"frame_idx": <next_frame>, "analysis_window_count": <count>},
  "reason": "analysis_window_completed",
  "output_artifacts": {"analysis_raw_s3_key": "<key>"}
}
```

- **Step 2: Wire resume request fields**

Automatic recovery should set `analysis_raw_s3_key`, `analysis_window_count`, and `analysis_current_context` when the selected checkpoint contains those values.

- **Step 3: Decide upscaled-frame upload policy**

Use coroutine upload only if profiling shows acceptable resource use. If too expensive, recover by recomputing upscaled frames from tracking and source video.

## Task 7: Automatic Crash Recovery Manager

**Files:**

- Modify: `service/reconciler.py`
- Modify: `service/jobs_store.py`
- Modify: `service/routes.py` or `service/app.py`
- Test: `tests/test_reconciler.py`
- **Step 1: Add recovery discovery**

Create and maintain `job_recovery_index` from `JobsStore` writes. The index is the vision engine's durable queue for monitor/recovery work. The companion video analysis backend should not run the monitor loop.

- **Step 2: Mark stale `RUNNING` jobs `INTERRUPTED`**

A job is stale when `job_state == RUNNING` and `last_heartbeat_at` is older than the configured timeout.

- **Step 3: Claim ownership conditionally**

Use a conditional update on `owner_instance_id` before creating the replacement job. The expected behavior is single-winner recovery when multiple workers start.

- **Step 4: Create automatic replacement job**

Load original request and latest resumable checkpoint, build a replacement `TrackRequest`, create new lifecycle with root `origin_job_id` and immediate `parent_job_id`, set old `replacement_job_id`, update latest-job pointer, and schedule `_run_with_semaphore`.

- **Step 5: Do not use manual `/resume` for crash recovery**

Crash recovery must run from startup/background reconciler logic.

- **Step 6: Run recovery on a schedule**

Replace one-shot `Reconciler.run_on_startup()` with a background manager started from `service/app.py` lifespan, similar to `HeartbeatTask`.

Recommended loop:

```python
class RecoveryManager:
    def __init__(self, jobs_store, instance_id, interval=30.0, stale_after=90.0):
        ...

    async def _run(self):
        while True:
            await self.reconcile_once()
            await asyncio.sleep(self._interval)
```

On startup, call `reconcile_once()` immediately, then continue every 30 seconds. The heartbeat interval is currently 5 seconds, so a 90-second stale threshold gives missed-heartbeat tolerance without delaying recovery too long.

- **Step 7: Use secondary index only for bounded operational lookup**

If a Keyspaces secondary index is added, use it to accelerate bounded owner/state lookups, not as the primary recovery queue. The primary recovery scan should remain the bucketed `job_recovery_index` query.

## Task 8: Upload Stage Split And Retention

**Files:**

- Modify: `service/worker.py`
- Modify: `service/jobs_store.py`
- Test: `tests/test_checkpoint_schema.py`
- **Step 1: Split upload checkpoints**

Write separate checkpoint updates for:

- tracking JSON uploaded
- analysis JSON uploaded
- annotated video uploaded
- SNS publish attempted/sent
- **Step 2: Add 48-hour retention metadata**

Store `expires_at` in checkpoint data or artifact metadata for checkpoint frames, partial tracking, raw analysis, and annotated recovery artifacts.

- **Step 3: Keep SNS non-idempotent for first refactor**

Do not add dedupe keys in this implementation pass.

## Task 9: Test And Verification Pass

**Files:**

- Modify: tests listed above
- **Step 1: Run targeted tests**

```bash
source venv/bin/activate && pytest tests/test_resume_endpoint.py tests/test_job_cancellation.py tests/test_checkpoint_schema.py tests/test_reconciler.py -v
```

- **Step 2: Run service test suite**

```bash
source venv/bin/activate && pytest tests/ -v
```

- **Step 3: Manual lifecycle smoke test**

Using local/dev Keyspaces-compatible infra:

- Submit `/track`.
- Confirm `video_analysis_latest_job` points to the submitted job.
- Force initial detection correction and confirm SSE reports waiting for correction.
- Call `/resume` and confirm new job has `origin_job_id=<root>`, `parent_job_id=<old>`, and old job has `replacement_job_id=<new>`.
- Simulate stale `RUNNING` heartbeat and confirm reconciler marks `INTERRUPTED`, creates a replacement, and updates latest job.
- Confirm the scheduled recovery manager repeats the scan without any companion-backend monitor process.
- Call `DELETE /job/{job_id}` on a Keyspaces-only job and confirm lifecycle is terminal `CANCELLED`.

## Governance Handoff Marker

- architecture_review_required: true
- Impact reasons: [api-contract, data-model, infra-runtime]
- Non-negotiable constraints:
  - Manual `/resume` only handles `AWAITING_CORRECTION`.
  - Crash recovery is automatic and replacement-job based.
  - User cancellation is terminal and never spawns replacement work.
  - `video_analysis_latest_job` tracks the latest job for a video across all states.
  - Recovery monitoring is owned by the vision engine through `job_recovery_index`; the companion backend does not run monitor processes.
- Impacted areas:
  - Frontend: UI/SSE behavior for waiting-for-correction state.
  - Backend services: vision engine routes, worker, reconciler, companion video analysis backend lifecycle reads.
  - Data/storage/events: Keyspaces lifecycle/checkpoint/latest-job tables, S3 checkpoint artifacts, SNS publish stage.
  - Infrastructure/runtime: stale heartbeat detection, ownership claim, checkpoint retention.

## Remaining Decisions Before Code

1. Confirm exact `heartbeat_bucket` granularity for `job_recovery_index` (`YYYYMMDDHH` is recommended).
2. Confirm whether the secondary index should be on `owner_instance_id`, `job_state`, or both, based on Keyspaces deployment limits and query patterns.

