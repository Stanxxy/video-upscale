---
date: 2026-05-03
category: insight
tags: [service, worker, checkpoints, schema, recovery, tests, implementation]
status: active
---

# Worker Checkpoint Standardization — Implementation

## Context
Follows the plan in `working_log/knowledge-base/requirements/2026-05-02-worker-checkpoint-standardization-plan.md` and the artifact contract in `contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`. The goal of this batch was to make every checkpoint write conform to the V1 envelope (`schema_version`, `pending_detection`, `artifacts`, `worker_state`), back the conformance with worker-integration tests, and persist enough durable state under `artifacts.*` plus enough in-memory progress under `worker_state.*` for crash recovery to resume tracking, upscale/analysis, upload, and SNS publish without re-doing finished work or regressing `progress_percent`.

## Content
Implemented behavior (per task):

- **Task 1**: `service/checkpoints.py` exposes `WorkerStateSnapshot`, `make_envelope`, per-stage builders (`build_download_completed`, `build_detect_initial_pending`, `build_track_progress`, `build_track_mid_loss`, `build_track_completed`, `build_upscale_started`, `build_upscale_window_progress`, `build_annotate_completed`, `build_upload_incremental`, `build_publish_completed`, `build_replaced_by_new_job`), the migrated builders (`build_verified_boxes_checkpoint`, `build_cancellation_checkpoint`), the unified `build_resume_overrides`, the `worker_state_from` helper, and `END_OF_TRACKING_SENTINEL = 10**9`.
- **Task 2**: Shared `tests/conftest.py` with `make_mock_jobs_store()` factory and `mock_jobs_store` / `service_components` / `service_app` / `service_client` fixtures. Replaces duplicated factories in `test_resume_endpoint.py` and `test_job_cancellation.py`.
- **Task 3**: `service/worker.py` writes the V1 download checkpoint after the S3 download, the V1 initial-detect pending checkpoint when no boxes are provided, and the V1 mid-track loss checkpoint via `build_track_mid_loss`. Also fixes `s3.upload_file(frame_jpeg, ...)` → `s3.put_object(bucket, key, body, content_type)` for raw bytes, and switches the mid-track partial-tracking upload to `request.output_bucket or request.bucket` so the resume route reads from the same bucket.
- **Task 4**: Track stage standardization. Adds `_update_tracking_progress_with_partial` async helper called by `tracking_progress_cb` on two cadences: 1-second lifecycle heartbeat (existing) and 30-second partial-tracking S3 upload + V1 `track_progress` checkpoint with `artifacts.partial_tracking_s3_key` + `artifacts.resume_from_frame`. Replaces the inline completed-tracking dict with `build_track_completed`.
- **Task 5**: `skip_upscale=True` short-circuit re-writes the track row with `artifacts.tracking_s3_key` after the upload and writes a terminal `upload` row with `completed=True` and `worker_state.progress_percent=100.0`.
- **Task 6**: Adds `should_flush_analysis(window_count, every_n=5)` and `_flush_analysis_checkpoint` async helper. `_run_upscale_analysis` writes `analysis_started` at the top, periodic flushes every 5 windows, and a final flush after the buffer drains. Plumbs `(job_id, jobs_store, loop, tracking_s3_key)` through. Refactors `run_job` to upload tracking JSON BEFORE entering upscale (so the upscale checkpoint can reference the durable `artifacts.tracking_s3_key`), re-write the track row with the durable key, and write the first `upload` row marking `tracking_uploaded`. Removes the redundant tracking JSON upload from stage 5.
- **Task 7**: Stage 5 writes `build_upload_incremental` rows additively as analysis JSON and annotated video land. Stage 4.5 writes `build_annotate_completed` after the annotated video upload. Stage 6 writes `build_publish_completed` with `completed=True` as the terminal pipeline row.
- **Task 8**: `submit_detection_response` and `recover_interrupted_job` use `build_resume_overrides(checkpoints)` to compose `TrackRequest` overrides — including the upscale-crash case (sentinel `resume_from_frame` + `analysis_raw_s3_key` / `analysis_window_count` / `analysis_current_context` forwarding). Both routes write a terminal `replaced_by_new_job` checkpoint to the OLD job with `completed=True` and the OLD job's last-known `worker_state` recorded for analytics.
- **Task 9**: `JobsStore.create_lifecycle` accepts optional `progress_percent` / `current_frame` / `total_frames` kwargs (default 0/0/0). Both routes seed the new lifecycle row from `worker_state_from(old_checkpoints)` so SSE / progress streams do not regress when a job is replaced.
- **Task 10**: Cancellation tests assert the V1 envelope shape on the `user_cancelled` row.

## Verification

- Per-task TDD cycle: every task wrote the failing test first (verified RED), then minimal implementation (verified GREEN), then committed.
- Final sweep after evaluator-fix round 2: `source venv/bin/activate && pytest tests/ -v` → **154 tests pass** in 50.2s.
- Round 1 baseline at Task 10 was 147; round 2 added 7 tests covering H1 (recover_interrupted_job direct test), M1 (`_run_upscale_analysis` orchestration), and M4 (5 cadence-flag unit tests).
- Per-task targeted sweeps after each commit: 44 → 50 → 51 → 53 → 54 → 56 → 57 → 154 (lifecycle + checkpoint + worker + service + taxonomy tests).

## Rationale

Three principles drove the design:

1. **Single envelope, single resume helper.** `make_envelope` + `build_resume_overrides` mean manual resume and automatic crash recovery share the same logic and any new TrackRequest field that needs to be forwarded only has to land in one place.
2. **S3 keys live under `artifacts`, scalar progress at the root.** Cross-stage consistency for analysis-service readers and recovery readers without per-stage special-casing.
3. **`worker_state` on every write** so a replacement job's progress matches the OLD job's last-known progress. SSE clients keep advancing instead of resetting to 0%.

The 30-second `track_progress` cadence and 5-window upscale flush cadence balance Keyspaces / S3 write volume against recovery granularity.

## Impact

Primary files:

- `service/checkpoints.py` (envelope + builders + resume_overrides + worker_state_from + should_flush_analysis + END_OF_TRACKING_SENTINEL)
- `service/worker.py` (download/detect/track/upscale_analyze/annotate/upload/publish standardization + `_make_worker_state`, `_update_tracking_progress_with_partial`, `_flush_analysis_checkpoint` helpers + mid-track frame-upload bug fix)
- `service/routes.py` (`build_resume_overrides` wiring + `replaced_by_new_job` writes + lifecycle progress seeding)
- `service/jobs_store.py` (`create_lifecycle` extended with progress kwargs)
- `tests/conftest.py` (shared fixtures)
- `tests/test_checkpoint_schema.py` (29 unit tests)
- `tests/test_worker_checkpoints.py` (15 worker-integration tests including 5 cadence-flag unit tests added in evaluator round 2)
- `tests/test_resume_endpoint.py` (11 route tests, includes the round-2 `recover_interrupted_job` direct test)
- `tests/test_job_cancellation.py` (3 route tests, V1 envelope assertions)

Companion contracts (sharable with `bjj-vision-backend`):

- `contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`
- `working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`

## Remaining scope

- Owner approval / governance review on the V1 addendum (artifact-key additions, `worker_state` block).
- Coordinated rollout with the analysis service: it can already read the V1 envelope it wrote (no schema change), but the new artifact keys (`tracking_s3_key`, `analysis_raw_s3_key`, etc.) and `worker_state` block are forward-compatible.
- A `skip_tracking: bool` field on `TrackRequest` is deferred to V2; V1 uses the sentinel `resume_from_frame = 10**9` instead.
- SNS dedupe / idempotency keys are deferred per the requirements doc.
