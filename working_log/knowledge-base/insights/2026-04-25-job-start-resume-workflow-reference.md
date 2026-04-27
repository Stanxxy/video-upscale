---
date: 2026-04-25
category: insight
tags: [service, lifecycle, resume, crash-recovery, keyspaces, refactor]
status: active
---

# Job Start and Resume Workflow Reference

## Context
The vision engine currently accepts analysis/tracking work through `POST /track`, persists lifecycle metadata in Keyspaces, suspends when detection correction is needed, and exposes `POST /jobs/{job_id}/resume` as an alias for the detection-response path.

Target expectation for future refactor:

- A submitted job should run until either tracking needs detection correction or the worker/service crashes.
- Durable worker state must be recoverable from AWS infrastructure, not from local process memory.
- If tracking needs correction, the current job should stop, release worker resources, and wait for the client to call the resume endpoint.
- Resume should create a new job from durable checkpoint state.
- If a worker crashes, a new worker should be able to continue from the last durable checkpoint.
- Manual resume endpoints are expected only for jobs in `AWAITING_CORRECTION`; stale `RUNNING`/`INTERRUPTED` crash-recovery jobs should be restored automatically by worker/reconciler logic.

## Current Code Workflow

New job start:

- `service/routes.py` `POST /track` creates an in-memory job through `InMemoryJobStore.create_job`.
- It creates a Keyspaces lifecycle row with `JobState.PENDING`, saves the serialized `TrackRequest`, and writes `video_analysis_latest_job` when `video_id` is present.
- It schedules `_run_with_semaphore(job_id, request)` as an in-process `asyncio.Task`.
- `_run_with_semaphore` registers the job in `JobsStore.owned_jobs`, runs `run_job`, then unregisters the job in a `finally` block.

Worker progression:

- `service/worker.py` sets the Keyspaces lifecycle state to `RUNNING`.
- Stages are currently linear: download, detect/verify, track, optional upscale/analyze, annotate, upload, publish, complete.
- Progress writes update both the in-memory job and the Keyspaces lifecycle row.
- Initial detection without provided boxes writes a `detect` checkpoint with `pending_detection`, sets state to `AWAITING_CORRECTION`, raises `JobSuspendedError`, releases resources, and deletes the local work directory.
- Mid-track detection callback writes a `track` checkpoint with `pending_detection`, tries to upload `partial_tracking.json`, sets state to `AWAITING_CORRECTION`, returns `None` to stop tracking, and `run_job` raises `JobSuspendedError`.
- Final cleanup attempts to save partial tracking only for failed/cancelled jobs that are not already `AWAITING_CORRECTION`.

Resume from correction:

- `POST /jobs/{job_id}/resume` delegates to `submit_detection_response`.
- The route requires the original lifecycle state to be `AWAITING_CORRECTION`.
- It reloads the original serialized request from Keyspaces.
- It reads all checkpoints and builds `cp_map = {cp["stage_name"]: cp.get("checkpoint_data", {}) for cp in checkpoints}`. This is only a dictionary keyed by checkpoint stage name, not a stage-ordering model. The current code only uses `cp_map["track"]` for resume hints; future behavior should consider both `detect` and `track` checkpoints.
- It creates a new in-memory job and a new Keyspaces lifecycle row.
- It stores the corrected boxes in a new `track` checkpoint for the new job.
- It marks the old job `CANCELLED`, schedules a new in-process task, and returns the new job ID.

Expected cancellation/correction workflow to validate:

- Initial detection is expected to occur when boxes are not provided. This is the special case where the first SAM2 boxes must be supplied before tracking can begin.
- Mid-track detection correction is optional and only occurs when tracking loses confidence or needs renewed boxes.
- Both initial and mid-track correction can share the same `ResumeRequest` shape because the human/client is providing SAM2 bounding boxes in both cases.
- The durable difference should be in checkpoint context, not necessarily in the request model: `detect.pending_detection.reason == "initial"` for initial correction; `track.pending_detection` plus partial tracking/cursor artifacts for mid-track correction.
- When correction is needed, worker compute should stop and release resources. The lifecycle marker should remain clear enough for the client to know manual correction is required. The current code uses `AWAITING_CORRECTION` before resume and changes the old job to `CANCELLED` only after a replacement job is created.
- Terminology for future refactor: `terminate` means the job lifecycle becomes terminal and must not be resumed; `cancel/offload` means shift active worker compute out of memory while preserving resumability through durable state. User-initiated `DELETE /job/{job_id}` should terminate the job as `CANCELLED` and must not spawn a replacement active task. Detection-correction offload should normally leave lifecycle as `AWAITING_CORRECTION` until the client resumes with corrected boxes.

Heartbeat and startup recovery:

- `service/heartbeat.py` heartbeats only jobs currently present in `JobsStore.owned_jobs`.
- `JobsStore.owned_jobs` is a local process set, not durable.
- `service/reconciler.py` explicitly states startup recovery is a production-blocker stub. It does not scan Keyspaces, mark stale jobs `INTERRUPTED`, or resume work.
- `JobsStore.list_running_jobs` only iterates `owned_jobs`, so it cannot discover jobs after process restart.

## Suspicious Bugs and Mismatches

- Resumed lifecycle origin appears wrong in lifecycle persistence, not in the response. `submit_detection_response(job_id, body)` receives the old job ID from `/jobs/{job_id}/resume`, and the returned response correctly reports `"origin_job_id": job_id`. However, the Keyspaces write calls `create_lifecycle(new_job_id, ..., origin_job_id=new_job_id, ...)`, so the persisted lifecycle row points to itself instead of the old job.
- Correction-needed jobs should stay `AWAITING_CORRECTION` until manual resume. After manual resume creates a replacement job, the old job should become terminal `CANCELLED` and include `replacement_job_id` so backend/UI can follow the chain.
- Crash recovery is not implemented. A crashed worker's local in-memory job, cancellation event, owned job set, task, and local work directory are lost.
- `/resume` is intentionally scoped to `AWAITING_CORRECTION` jobs with corrected boxes. The mismatch is not endpoint scope; the missing piece is automatic restoration for stale `RUNNING`/`INTERRUPTED` jobs from Keyspaces checkpoints.
- Checkpoint schema is not standardized. Detect, track, upscale/analyze, and publish write different shapes and often mark `completed=False` even after a stage is effectively complete.
- Initial detection correction and mid-track correction can reasonably share `ResumeRequest` because both provide SAM2 bounding boxes. The implementation risk is that checkpoint data must distinguish the context: initial correction has no prior tracking artifact, while mid-track correction should include partial tracking/cursor data when available.
- `player_mapping` is accepted in `ResumeRequest` but is not applied to the resumed `TrackRequest` or checkpoint.
- The mid-track detection callback uses `s3.upload_file(frame_jpeg, ...)` with raw JPEG bytes. `service/s3.py` defines `upload_file(local_path: str, ...)` and delegates to `boto3.client.upload_file`, which expects a filename/local path. Initial detection correctly uses `put_object(..., body=frame_jpeg, ...)`. Mid-track frame upload should likely use `put_object` too, or write bytes to a temp file before `upload_file`.
- `partial_bucket` during resume is `request.output_bucket or request.bucket`, but the mid-track suspend upload currently uses `request.bucket`; jobs with a distinct output bucket may read from the wrong bucket.
- Completed tracking checkpoint writes only `start_frame` and `frame_count`; it does not include a durable tracking S3 key until later upload paths.
- Stage 4 analysis has resume fields in `TrackRequest` (`analysis_raw_s3_key`, `analysis_window_count`, `analysis_current_context`). `_run_upscale_analysis` only restores prior raw analysis windows when `analysis_raw_s3_key` is already present on the request. Current worker code writes `analysis_raw.json` locally during processing but does not upload it as a checkpoint artifact during the stage, does not write a checkpoint containing the raw-analysis S3 key/window cursor/context, and does not build those resume fields automatically during crash recovery.
- `DELETE /job/{job_id}` only consults in-memory job state, so it cannot terminate a Keyspaces-only job. Here "Keyspaces-only" means a job row exists in persistent tables but no current worker process has a local in-memory `JobResponse`, cancellation event, or active task for it.
- Capacity checks and task scheduling are per-process. They do not provide distributed single-GPU ownership if multiple engine workers run against the same Keyspaces tables.

## Answered Requirements For Refactor

1. Use a new job ID for every manual correction resume and every automatic crash-recovery resume.
2. `origin_job_id` should point to the root/original job in the chain.
3. Add `parent_job_id` to point to the immediate predecessor job.
4. Add an explicit replacement pointer so callers can follow old job to replacement job.
5. `video_analysis_latest_job` should always point to the latest job for a video, whether running, awaiting correction, interrupted, cancelled, failed, or completed.
6. While an old job awaits correction and no replacement exists, `video_analysis_latest_job` should still point to that old `AWAITING_CORRECTION` job.
7. `job_stage_checkpoints.completed` means the whole job is completed, so restoration can distinguish incomplete resumable checkpoints from completed jobs.
8. Manual `/resume` is only for `AWAITING_CORRECTION` jobs. Stale `RUNNING` jobs should first transition to `INTERRUPTED`, then be automatically restored.
9. New worker ownership should use a conditional update on `owner_instance_id`.
10. User-initiated cancellation should terminally set lifecycle `CANCELLED`; no replacement/resume active task should be spawned.
11. `DELETE /job/{job_id}` should support Keyspaces-only jobs: jobs present in persistent tables but not owned by any current in-memory worker task.
12. A final cancellation checkpoint should include cancellation reason and last known cursor.
13. Correction-needed jobs should normally remain `AWAITING_CORRECTION` before manual resume. If the user actively cancels, lifecycle should remain `CANCELLED` because the backend/UI reads Keyspaces.
14. Initial detection and mid-track detection can use the same `ResumeRequest`; both provide bounding boxes for SAM2. Initial detection is expected when no boxes exist, while mid-track detection is optional.
15. Corrected boxes at mid-track apply only to the resume frame.
16. Resume checkpoint lookup should primarily use the supplied `job_id`; when resolving from `video_id`, use `video_analysis_latest_job`; if neither is available, use fixed stage ordering.
17. Fixed stage ordering is required for checkpoint selection.
18. For a resumed replacement job, verified boxes may be restored from both `detect` and `track` checkpoint contexts, depending on the job ID and checkpoint source.
19. Pending detection checkpoint data requires `frame_idx`, `frame_s3_key`, `frame_bucket`, `candidates`, `suggested_boxes`, and `reason`.
20. All stages should include enough cursor data to restore progress. `progress_percent`, `resume_cursor`, `reason`, and `schema_version` are desired checkpoint fields. `stage` and `status` are unnecessary in checkpoint data because they duplicate `job_lifecycle`.
21. `input_artifacts` and `output_artifacts` are useful fields but need precise definitions per stage.
22. For tracking resume, `resume_from_frame` should be the next unprocessed frame.
23. Partial tracking JSON should store global frame indices and relative indices for the selected clip range.
24. Upscale/analysis checkpointing should include `frame_idx` and `reason`. Upscaled frames may be uploaded to S3 asynchronously if resource use is acceptable.
25. Upload stage should split durable checkpoints for tracking JSON, analysis JSON, annotated video, and SNS.
26. Dedicated checkpoint bucket/prefix should hold durable checkpoint artifacts, keyed by job ID.
27. Local source-video caching is desirable for faster resume, but durable recovery must still work from S3.
28. Checkpoint artifact retention should be 48 hours max.
29. SNS idempotency is not required for the first refactor.
30. `player_mapping` should not alter resumed request state. At most, it can supplement missing `player_references` for the job.
31. Keep `/track` as an engine-direct endpoint.
32. If correction is submitted for an old job after a newer resume job already exists, return an error and continue the existing resumed job.
33. Enforce one active root-job chain per video.
34. SSE should tell the UI the job is waiting for correction.
35. After manual resume creates a replacement job, the old `AWAITING_CORRECTION` job should become `CANCELLED`.
36. Crash recovery discovery should be owned by the vision engine through a new Keyspaces recovery index/table and secondary-index access pattern. The video analysis backend should not run monitor/recovery processes; it should handle instant request/response work only.
37. The vision engine needs a scheduled/background recovery manager, not only a one-time startup check. It should periodically scan the recovery index for stale running jobs, mark them `INTERRUPTED`, claim ownership, and create replacement jobs.

## Remaining Clarifications

- Define exact `input_artifacts` and `output_artifacts` keys per stage.
- Confirm whether adding `parent_job_id` and `replacement_job_id` requires Keyspaces schema/table migration in both this repo and the video analysis backend.

## Impact

Primary files:

- `service/routes.py`
- `service/worker.py`
- `service/jobs_store.py`
- `service/models.py`
- `service/heartbeat.py`
- `service/reconciler.py`

The future refactor should treat durable lifecycle, checkpoint schema, worker leasing, and resume job lineage as one design problem. The current implementation has useful pieces, but it still relies on in-memory execution state for ownership, cancellation, scheduling, and recovery.
