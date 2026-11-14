# Job lifecycle, recovery and cancellation

**Status:** contract — vision engine (`whole-video-analysis`) + `video_analysis_and_annotation_service`
**Related:** `CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md`, backend `CHECKPOINT_DATA_SCHEMA_V1.md`.

## One job, one `job_id`

A job keeps its `job_id` for its whole life. There is no replacement-job chain and no human-in-the-loop resume: `POST /track` is the only entry point.

## Keyspaces rows

- `job_lifecycle` (per `job_id`): `job_state` is one of `PENDING`, `RUNNING`, `INTERRUPTED`, `COMPLETED`, `FAILED`, `CANCELLED`; plus `stage`, `progress_percent`, `stage_message`, `error_message`, `pipeline_kind` (`highlight_v2`) and the chunk/attribution progress columns.
- `video_analysis_latest_job` (per `video_id`): the `job_id` that owns the video. Only `COMPLETED`, `FAILED`, `CANCELLED` and `INTERRUPTED` transitions sync it; `PENDING` and `RUNNING` do not. Read `job_lifecycle` for authoritative state.
- `job_stage_checkpoints`: durable resume inputs (see the checkpoint addendum).

## Recovery

- **Stale worker.** `RecoveryManager` claims stale `RUNNING` and `INTERRUPTED` rows. `recover_interrupted_job` re-invokes `run_highlight_job` for the **same** `job_id`, skipping completed chunks and reusing a non-expired Gemini upload.
- **Orphan `PENDING` rows.** At startup `drain_orphan_pending_jobs_on_startup` claims `PENDING` rows with a CAS on `(job_state, owner_instance_id)` and re-schedules them from the saved request.
- **Unrecoverable.** A recovered request without a valid effective analysis settings snapshot is set to `FAILED` with an explicit `error_message`.

## Historical rows from the retired tracking pipeline

Rows with `pipeline_kind` NULL or `"tracking"`, or in `AWAITING_CORRECTION`, are never resumed.
- Recovery and the pending drain set them to `FAILED` with "This job ran on the retired local tracking pipeline and cannot be resumed. Start a new analysis."
- `GET /job/{job_id}` reads an `AWAITING_CORRECTION` row as `failed` with that message.
- `replacement_job_id` may still be set on old rows. Cancelling a row that has one returns 409; cancel the replacement.

## Cancellation

`DELETE /job/{job_id}` writes a `cancellation` checkpoint, sets `CANCELLED`, and cancels the local asyncio task if it is running.
