# Checkpoint artifacts addendum (full per-stage reference)

**Date:** 2026-05-02 (revised 2026-05-03, 2026-05-10)
**Status:** draft contract — vision engine + `video_analysis_and_annotation_service`
**Companion to:** `CHECKPOINT_DATA_SCHEMA_V1.md`
**Storage:** Amazon Keyspaces `video_analysis.job_stage_checkpoints.checkpoint_data` (JSON text).
**Duplicate of:** `bjj-proj/whole-video-analysis/working_log/contracts/bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` (edit in one place, copy for engine repo).

## Purpose

`CHECKPOINT_DATA_SCHEMA_V1.md` defines the V1 envelope (`schema_version`, `pending_detection`, `artifacts`) and the `detect` / `track` artifact shapes. This addendum:

1. Mirrors the `detect` / `track` shapes here for one-stop reference (backend `CHECKPOINT_DATA_SCHEMA_V1.md` remains the source of truth — this file MUST stay in sync).
2. Specifies artifact keys for every remaining vision-engine pipeline stage (`download`, `upscale_analyze`, `annotate`, `upload`, `publish`).
3. Adds a cross-cutting `worker_state` block that EVERY stage write MUST populate so a replacement job can seed `job_lifecycle.progress_percent`, `current_frame`, and `total_frames` without reading the old lifecycle row.
4. Specifies the resume-parameter forwarding rules so manual resume and automatic recovery rebuild a complete `TrackRequest` from checkpoint state alone.

All stages keep the same V1 envelope:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "artifacts": { ... },
  "worker_state": { ... }
}
```

## Cross-cutting principles

### S3 keys live under `artifacts`

Anything stored in S3 lives under `artifacts`. Stage progress data (frame cursors, window counts, per-stage scalar context) MAY live at the root level beside `artifacts`, but durable S3 keys MUST be inside `artifacts`.

### `completed` means whole-job complete

`completed` (the `job_stage_checkpoints.completed` column) means the **whole job** is finished — only the terminal pipeline write (typically `publish`) sets it to `true`. Stage rows written along the way always use `completed=false`.

When a replacement job is created from an `AWAITING_CORRECTION` job (manual resume) or a stale `RUNNING`/`INTERRUPTED` job (automatic recovery), the old job MUST also receive a final checkpoint write with `completed=true`; the work has been handed off, so the old job is logically complete from the pipeline's point of view, even though its lifecycle state is `CANCELLED` or `INTERRUPTED`.

### `worker_state` block (REQUIRED on every stage write)

Every checkpoint write MUST include a `worker_state` block snapshotting the in-memory worker's progress at the time of the write:

```json
"worker_state": {
  "progress_percent": 35.0,
  "current_frame": 1200,
  "total_frames": 3600,
  "stage_progress_fraction": 0.5
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `progress_percent` | float | yes | The overall pipeline progress (0..100) the worker last wrote to `job_lifecycle.progress_percent` for this job. Recovery uses this verbatim to seed the replacement job's lifecycle row, so the SSE stream does not show progress dropping back to 0%. |
| `current_frame` | int | yes | Worker's current frame counter at the time of the write. Mirrors `job_lifecycle.current_frame`. Stages that don't process frames (download, detect, annotate, upload, publish) write `0`. |
| `total_frames` | int | yes | Worker's total frame target. Mirrors `job_lifecycle.total_frames`. Stages without frame work write `0`. |
| `stage_progress_fraction` | float | yes | Progress within the current stage as 0.0..1.0. Lets recovery show stage-level progress without recomputing percentages. |

Recovery readers MUST seed the replacement lifecycle row from `worker_state.progress_percent` / `current_frame` / `total_frames`. They MUST NOT reset to 0 (which would make the SSE stream regress).

## Resume parameter forwarding

Manual resume (`POST /jobs/{job_id}/resume`) and automatic recovery (`reconciler → recover_interrupted_job`) MUST both rebuild a complete replacement `TrackRequest` by combining (a) the original request stored in `job_request_params.request_json` with (b) checkpoint-derived overrides. The override rules are:

| Crash/correction location | Source checkpoint(s) | TrackRequest fields to override |
|---------------------------|----------------------|----------------------------------|
| Initial detection (no boxes) | `detect.pending_detection` + resume request body | `box_a`, `box_b` from request body. |
| Mid-track detection loss (`AWAITING_CORRECTION`) | `track.pending_detection`, `track.artifacts.partial_tracking_s3_key`, `track.artifacts.resume_from_frame` | `box_a`, `box_b` from request body; `resume_tracking_s3_key`, `resume_from_frame`, `resume_from_job_id` from checkpoint. |
| Mid-track crash (no `pending_detection`, automatic recovery) | Latest `track` row with `reason="tracking_progress"` — `track.artifacts.partial_tracking_s3_key`, `track.artifacts.resume_from_frame` | `resume_tracking_s3_key`, `resume_from_frame` from checkpoint. Original `box_a`/`box_b` from `job_request_params.request_json`. |
| Crash during/after upscale_analyze | `upscale_analyze.artifacts.tracking_s3_key`, `upscale_analyze.artifacts.analysis_raw_s3_key`, `upscale_analyze.resume_cursor`, `upscale_analyze.analysis_current_context` | `resume_tracking_s3_key` (skip re-tracking — tracking is complete), `resume_from_frame` set to the V1 sentinel `END_OF_TRACKING_SENTINEL = 10**9` to short-circuit the tracking pass, `analysis_raw_s3_key`, `analysis_window_count`, `analysis_current_context`. |
| Crash during annotate / upload / publish | Whichever later-stage checkpoint exists, plus `upscale_analyze` for analysis context | Same as upscale_analyze case; the worker re-runs annotate/upload/publish from already-uploaded inputs without redoing analysis. |

The plumbing is symmetric to how `submit_detection_response` already forwards `resume_tracking_s3_key` / `resume_from_frame` for mid-track loss. The vision engine MUST expose a single helper that returns the override dict for any latest-checkpoint state so manual-resume and recovery share the same logic.

## `stage_name == download`

Lightweight progress-only checkpoint. The video is downloaded fresh per job and not yet cached durably; no S3 artifact is written.

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "download_completed",
  "artifacts": {},
  "worker_state": {
    "progress_percent": 10.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 1.0
  }
}
```

## `stage_name == detect`

Mirrored from `CHECKPOINT_DATA_SCHEMA_V1.md` (backend is source of truth) with `worker_state` added.

### Initial detection waiting for human input

Written when no boxes are provided and the worker captures a candidate frame for the user to verify. The worker then suspends and the job lifecycle becomes `AWAITING_CORRECTION`.

```json
{
  "schema_version": 1,
  "pending_detection": {
    "reason": "initial",
    "frame_idx": 0,
    "frame_s3_key": "checkpoints/{job_id}/frame_0.jpg",
    "frame_bucket": "bjj-video-analysis",
    "candidates": [
      {"candidate_id": 0, "box": [10, 20, 100, 200], "confidence": 0.91}
    ],
    "suggested_boxes": [[10, 20, 100, 200], [300, 20, 400, 200]]
  },
  "artifacts": {},
  "worker_state": {
    "progress_percent": 10.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 0.0
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `pending_detection.reason` | string | yes | `initial` for first-frame detection. |
| `pending_detection.frame_idx` | int | yes | Frame index for the checkpoint frame. |
| `pending_detection.frame_s3_key` | string | yes | Key within `frame_bucket`. |
| `pending_detection.frame_bucket` | string | no | Default: deployment bucket (`VIDEO_STORAGE_BUCKET`). |
| `pending_detection.candidates` | array | yes | Engine-defined detection candidates. Domain fields are engine-owned. |
| `pending_detection.suggested_boxes` | object \| array \| null | no | Optional model hint for the verifier: either two boxes `[[x1,y1,x2,y2],[…]]` or an object `{ "athlete_a": [...], "athlete_b": [...] }` (plus optional `suggestion_model` / `vllm_model`). Never applied automatically. |

### Verified-boxes after manual resume

Written by the resume route when the user submits corrected boxes. The new job starts with this row already in place so the worker can skip the detection step.

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "detection_correction_resume",
  "source_stage": "detect",
  "verified_box_a": [10, 20, 100, 200],
  "verified_box_b": [300, 20, 400, 200],
  "artifacts": {},
  "worker_state": {
    "progress_percent": 15.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 1.0
  }
}
```

`verified_box_a` / `verified_box_b` are kept at the root level (not under `artifacts`) because they are not S3 references — they are inline geometry. Only the `detect` and `track` stages emit them.

## `stage_name == track`

Mirrored from `CHECKPOINT_DATA_SCHEMA_V1.md` with `worker_state` added and the partial-tracking S3 key and resume frame moved under `artifacts` per the cross-cutting principle.

### Tracking in progress (periodic, every 30s of wall-clock tracking)

Written periodically as tracking advances. The worker uploads the current `tracking.json` to S3 as `checkpoints/{job_id}/partial_tracking.json` and records the S3 key plus the next-frame cursor under `artifacts`. This is the durable resume point for crashes that happen mid-track **without** a `pending_detection` event (the common recovery-manager case).

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "tracking_progress",
  "resume_cursor": {"frame_idx": 1200},
  "artifacts": {
    "partial_tracking_s3_key": "checkpoints/{job_id}/partial_tracking.json",
    "resume_from_frame": 1200
  },
  "worker_state": {
    "progress_percent": 35.0,
    "current_frame": 1200,
    "total_frames": 3600,
    "stage_progress_fraction": 0.5
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `artifacts.partial_tracking_s3_key` | string | yes | S3 key of the partial tracking JSON uploaded at this checkpoint moment. Recovery uses this to seed `TrackRequest.resume_tracking_s3_key`. |
| `artifacts.resume_from_frame` | int | yes | Mirror of `frames_done` — the next frame the worker would have processed. Recovery copies this into `TrackRequest.resume_from_frame`. |

`progress_percent` for the track stage is computed as `15.0 + (current_frame / total_frames) * 40.0` (track stage spans 15%-55%). The 30-second cadence is independent of the 1-second `job_lifecycle` heartbeat write.

### Mid-track detection loss

Written when SAM2/RTMPose/YOLO loses identity mid-video. Includes `pending_detection` so the lifecycle moves to `AWAITING_CORRECTION`, plus partial-tracking artifacts so resume can skip already-tracked frames.

**Engine behavior (normative):** After this checkpoint and `AWAITING_CORRECTION` are durable, the vision engine **must end the current `run_tracking` invocation** (no further frames in the same pass). That guarantees the superseded job releases worker capacity so the **replacement** `job_id` from manual resume can start immediately. See `JOB_ROTATION_HANDOFF_AND_RESUME.md` §4.1.

```json
{
  "schema_version": 1,
  "pending_detection": {
    "reason": "tracking_lost",
    "frame_idx": 7432,
    "frame_s3_key": "checkpoints/{job_id}/frame_7432.jpg",
    "frame_bucket": "bjj-video-analysis",
    "candidates": [
      {"candidate_id": 0, "box": [10, 20, 100, 200], "confidence": 0.78}
    ],
    "suggested_boxes": [[10, 20, 100, 200], [300, 20, 400, 200]]
  },
  "resume_cursor": {"frame_idx": 7432},
  "artifacts": {
    "partial_tracking_s3_key": "checkpoints/{job_id}/partial_tracking.json",
    "resume_from_frame": 7432
  },
  "worker_state": {
    "progress_percent": 35.0,
    "current_frame": 7432,
    "total_frames": 21600,
    "stage_progress_fraction": 0.34
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `pending_detection.reason` | string | yes | `tracking_lost`. Synonyms `tracking_loss` MUST be normalized to `tracking_lost` in new writes. |
| `artifacts.partial_tracking_s3_key` | string | yes | S3 key of the merged partial tracking JSON. The resume request copies this to `TrackRequest.resume_tracking_s3_key`. |
| `artifacts.resume_from_frame` | int | yes | Next *video* frame index after the last successfully tracked frame. The resume request copies this to `TrackRequest.resume_from_frame`. |
| `resume_cursor.frame_idx` | int | yes | Mirror of `artifacts.resume_from_frame` — kept at root for consumers that look at `resume_cursor` uniformly across stages. |

### Track stage completed

Written after `run_tracking_job` returns successfully. Initially without `tracking_s3_key` (the upload happens later); the row is **re-written** post-upload to add `artifacts.tracking_s3_key`.

Pre-upload write:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "track_completed",
  "start_frame": 0,
  "frame_count": 21600,
  "artifacts": {},
  "worker_state": {
    "progress_percent": 55.0,
    "current_frame": 21600,
    "total_frames": 21600,
    "stage_progress_fraction": 1.0
  }
}
```

Post-upload re-write:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "track_completed",
  "start_frame": 0,
  "frame_count": 21600,
  "artifacts": {
    "tracking_s3_key": "{base_key}_tracked.json"
  },
  "worker_state": {
    "progress_percent": 55.0,
    "current_frame": 21600,
    "total_frames": 21600,
    "stage_progress_fraction": 1.0
  }
}
```

## `stage_name == upscale_analyze`

Written periodically as the worker analyses sliding windows.

### Frame cadence — important for cursor interpretation

The upscale/analysis stage does **not** process every video frame. It iterates over `tracking_data["frames"]`, where each entry's `frame_idx` is a *video* frame index that tracking chose to record (controlled by `TrackRequest.step_size` / `ServiceConfig.tracking_step_size`). Inside the loop, frames are filtered further by `frame_idx % TrackRequest.sampling_rate == 0` (default `sampling_rate=1`).

Each surviving frame is upscaled and pushed into a sliding buffer. When the buffer reaches `WINDOW_SIZE = 30` upscaled frames, the worker analyses that window with Gemini and then drops `STRIDE = 15` frames from the buffer head. So one new analyzed window is produced every 15 upscaled frames (≈ every `15 * sampling_rate * step_size` video frames).

`resume_cursor.frame_idx` is the **next tracking-recorded video frame index** to process, computed as `max(last_window["frames"]) + 1`. The worker resume code at `_run_upscale_analysis` skips any tracking entries with `frame_idx < resume_cursor.frame_idx`, so the cursor does not have to align to a sampling-rate or step-size boundary — the worker re-applies the filter.

### Periodic write (every 5 windows + once at final flush)

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "analysis_window_completed",
  "resume_cursor": {
    "frame_idx": 9120,
    "analysis_window_count": 12
  },
  "analysis_current_context": "white belt entered guard...",
  "artifacts": {
    "tracking_s3_key": "checkpoints/{job_id}/tracking.json",
    "analysis_raw_s3_key": "checkpoints/{job_id}/analysis_raw.json"
  },
  "worker_state": {
    "progress_percent": 67.5,
    "current_frame": 9120,
    "total_frames": 21600,
    "stage_progress_fraction": 0.5
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `reason` | string | yes | `analysis_window_completed` for periodic writes; `analysis_started` for the very first write before any windows. |
| `resume_cursor.frame_idx` | int | yes | Next *tracking-recorded* video frame index (`max(last_window["frames"]) + 1`). The worker re-applies sampling-rate / step-size filters when it resumes — callers do not need to align this to a multiple. |
| `resume_cursor.analysis_window_count` | int | yes | Count of analysis windows already persisted in `analysis_raw_s3_key`. |
| `analysis_current_context` | string | yes | Latest `current_context_summary` carried forward by Gemini analysis. Empty string before any window. |
| `artifacts.tracking_s3_key` | string | yes | Pointer to the tracking JSON used as input for upscale/analysis. |
| `artifacts.analysis_raw_s3_key` | string | yes | Pointer to the periodically-uploaded raw analysis array. |

`progress_percent` for upscale_analyze is `55.0 + (processed_tracking_entries / total_tracking_entries) * 25.0` (stage spans 55%-80%).

Crash-recovery readers MUST use these three checkpoint fields to populate `TrackRequest.analysis_raw_s3_key`, `TrackRequest.analysis_window_count`, and `TrackRequest.analysis_current_context` on the replacement request, AND set `TrackRequest.resume_tracking_s3_key` from `artifacts.tracking_s3_key` (so the replacement job skips re-running tracking).

## `stage_name == annotate`

Optional stage — present only when `analysis_result` exists and a tracked video is on disk. Lighter coverage is acceptable; this is recoverable by re-running annotate from upstream artifacts.

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "annotate_completed",
  "artifacts": {
    "annotated_video_s3_key": "{base_key}_annotated.mp4"
  },
  "worker_state": {
    "progress_percent": 85.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 1.0
  }
}
```

`artifacts.annotated_video_s3_key` is omitted when annotation failed (non-fatal); readers should treat its absence as "annotation skipped or failed, regenerate during resume."

## `stage_name == upload`

Written **incrementally** — one update per artifact landed. Recovery readers should treat the row as a join over the artifact keys present so far. `worker_state.progress_percent` ramps 85% → 90% across the three sub-writes.

After the tracking JSON lands:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "tracking_uploaded",
  "artifacts": {
    "tracking_s3_key": "{base_key}_tracked.json"
  },
  "worker_state": {
    "progress_percent": 86.6,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 0.33
  }
}
```

After the analysis JSON lands (worker overwrites the row, additive on `artifacts`):

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "analysis_uploaded",
  "artifacts": {
    "tracking_s3_key": "{base_key}_tracked.json",
    "analysis_s3_key": "{base_key}_analysis.json"
  },
  "worker_state": {
    "progress_percent": 88.3,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 0.66
  }
}
```

After the annotated video lands:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "annotated_video_uploaded",
  "artifacts": {
    "tracking_s3_key": "{base_key}_tracked.json",
    "analysis_s3_key": "{base_key}_analysis.json",
    "annotated_video_s3_key": "{base_key}_annotated.mp4"
  },
  "worker_state": {
    "progress_percent": 90.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 1.0
  }
}
```

Recovery code reads the latest row and resumes from the first artifact key not yet present. Cassandra/Keyspaces overwrites the previous JSON value when `(job_id, stage_name)` repeats, so the latest row contains the cumulative state.

## `stage_name == publish`

Terminal stage for the standard path. Also the only place `completed=true` is set on a successful run.

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "publish_completed",
  "artifacts": {
    "sns_topic_arn": "arn:aws:sns:...",
    "sns_event_count": 12,
    "sns_completion_sent": true
  },
  "worker_state": {
    "progress_percent": 100.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 1.0
  }
}
```

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `artifacts.sns_topic_arn` | string | yes | The topic actually published to. |
| `artifacts.sns_event_count` | int | yes | Number of `VideoEventWithCandidates` events published. |
| `artifacts.sns_completion_sent` | bool | yes | True once the completion event was sent. |

If a job crashes after partial SNS publish, the replacement job re-publishes from scratch. Idempotency is **not** required for V1; downstream consumers must tolerate duplicate events keyed by `job_id`.

## Replacement-on-correction final write

When the manual resume route or the recovery manager spawns a replacement job, it MUST write one terminal checkpoint to the **old** job:

```json
{
  "schema_version": 1,
  "pending_detection": null,
  "reason": "replaced_by_new_job",
  "artifacts": {
    "replacement_job_id": "<new_job_id>"
  },
  "worker_state": {
    "progress_percent": 35.0,
    "current_frame": 7432,
    "total_frames": 21600,
    "stage_progress_fraction": 0.34
  }
}
```

The old row's `completed` column is `true` (job-level — the work is complete from the old job's standpoint). The lifecycle row remains `CANCELLED` (manual resume) or `INTERRUPTED` (recovery, then transitioned by recovery code).

`worker_state` carries the OLD job's progress at the time of replacement so analytics and SSE clients can see how far the old chain advanced.

The `stage_name` for this terminal write is the stage the old job was last working on (`detect` or `track` typically). It does not introduce a new pipeline stage.

## Reader/writer responsibilities

| Component | Responsibility |
|-----------|----------------|
| Vision engine worker | Write all stage checkpoints in V1 envelope including `worker_state`; populate `artifacts.*` per this addendum; write the final `replaced_by_new_job` row when handing off. |
| Vision engine resume route / recovery manager | Read `pending_detection` for correction; read `artifacts.partial_tracking_s3_key` / `artifacts.resume_from_frame` for tracking resume; read `upscale_analyze` artifacts to forward `analysis_raw_s3_key`, `analysis_window_count`, `analysis_current_context`, and `tracking_s3_key→resume_tracking_s3_key` for upscale/analysis resume; seed the new lifecycle row's `progress_percent` / `current_frame` / `total_frames` from the latest checkpoint's `worker_state`. |
| Analysis service | Read `pending_detection` and `artifacts.*.s3_key` for SSE/GET-checkpoint responses. Treat unknown artifact keys as forward-compatible. May read `worker_state.progress_percent` to surface stage-level progress to UIs without subscribing to the lifecycle row. |

## Versioning

This addendum stays at `schema_version = 1` because it adds only optional sibling keys under `artifacts` and the `worker_state` block (which V1 already declared as engine-owned and extensible at the envelope level — though strictly speaking `worker_state` is a new top-level key; V1 readers MUST ignore unknown top-level keys, which the spec already requires). A breaking change to `artifacts` semantics or a `worker_state` schema break would require bumping `schema_version`.

## Non-goals

- SNS dedupe / idempotency keys (deferred per requirements doc #29).
- Durable upscaled-frame uploads (deferred — recover by recompute).
- Source-video caching across resumes (deferred — re-download).
- Telling apart "tracking is fully complete and we crashed during upscale" from "tracking partially complete" via a dedicated TrackRequest field. V1 reuses `resume_tracking_s3_key` + a `resume_from_frame` past `end_frame` to make the tracking pass a no-op. A future revision MAY add `skip_tracking: bool` for clarity.
