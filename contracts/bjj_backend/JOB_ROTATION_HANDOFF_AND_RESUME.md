# Job rotation: cancellation, handoff, and resume

**Date:** 2026-05-10  
**Status:** contract — vision engine (`whole-video-analysis`) + `video_analysis_and_annotation_service`  
**Scope:** How `job_id` changes across human-in-the-loop resume and crash recovery, how Keyspaces rows are updated, and how a caller should resolve `video_id` → active work → `progress_percent`.  
**Duplicate:** `working_log/contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md` (mirror; keep body in sync when editing).  
**Related:** `CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md` (checkpoint shapes and `replaced_by_new_job`), `CHECKPOINT_DATA_SCHEMA_V1.md` (backend source of truth for V1 envelope).

---

## 1. Why this document exists

The vision engine **does not keep updating** the original `job_lifecycle` row after work is handed off to a **replacement** job. Progress and heartbeats move to the **new** `job_id`. Any service that maps `video_id` → `job_id` and then reads `job_lifecycle` **must** follow rotation rules or it will appear “stuck” (for example at **10%** on the pre-resume row while tracking runs elsewhere).

---

## 2. Keyspaces tables (relevant columns)

### 2.1 `job_lifecycle` (per `job_id`)

Authoritative fields for **that** job row:

| Field | Meaning |
|--------|--------|
| `job_state` | `PENDING`, `RUNNING`, `AWAITING_CORRECTION`, `INTERRUPTED`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `progress_percent`, `current_frame`, `total_frames`, `stage`, … | Updated by the worker while **this** `job_id` is active |
| `origin_job_id` | First job in the chain (stable across replacements when set correctly) |
| `parent_job_id` | Immediate predecessor when this row was created as a **replacement** |
| `replacement_job_id` | Set on the **old** row when a replacement has been **claimed**; points to the new `job_id` |

**Tracking stage — `current_frame` / `total_frames` (clip-global):** While the worker is in tracking, these fields (and matching `worker_state` in V1 checkpoints) use the **full requested clip**, not the remaining segment after a mid-track resume. **`total_frames`** = `resolved_end_frame − clip_start_frame`. **`current_frame`** is a **1-based** count along that clip from absolute `frame_idx` (`min(global_idx − clip_start + 1, total_frames)`). **Reconcile / resume** still read **checkpoint artifacts** (`resume_from_frame` as an **absolute** global index, partial JSON, etc.); lifecycle frame counts are for **UI progress** and do not control recovery.

`set_state(..., CANCELLED)` on the old job **does not** clear or bump `progress_percent`; the old row can retain the last percentage from before handoff (e.g. **10%** at initial detect suspend).

### 2.2 `video_analysis_latest_job` (per `video_id`)

Small mapping row: **`video_id` → `job_id`** plus `job_state` and `updated_at`.

- **Use this row to discover which `job_id` currently owns pipeline work** for a video (when `video_id` was present at job creation / resume).
- **Do not treat `job_state` on this row as always fresh.** It is only updated when lifecycle transitions pass `states_that_sync_latest_job_row()` in the engine (`COMPLETED`, `FAILED`, `CANCELLED`, `INTERRUPTED`, `AWAITING_CORRECTION`). **`RUNNING` and `PENDING` do not sync** to `video_analysis_latest_job`. After a handoff the row may show `PENDING` for the new `job_id` while `job_lifecycle` for that same id is already `RUNNING`.

**Rule:** For authoritative state and progress, always read **`job_lifecycle` for the `job_id` returned by `get_latest(video_id)`** (or after following `replacement_job_id`).

### 2.3 `job_stage_checkpoints`

Durable pipeline history and resume inputs. A terminal row with `reason == "replaced_by_new_job"` and `completed == true` on the **old** job marks pipeline handoff (see checkpoint contract).

---

## 3. Job states and who moves them

| State | Typical meaning |
|--------|------------------|
| `AWAITING_CORRECTION` | Worker stopped; human/client must submit boxes (initial or mid-track). |
| `CANCELLED` | Terminal. For **handoff**, the **superseded** job is cancelled **after** the replacement is created; progress on that row is **not** advanced further. |
| `INTERRUPTED` | Reconciler marked stale heartbeat (was `RUNNING`) or job was already interrupted. Recovery may attach a **replacement** while this row can stay `INTERRUPTED` with `replacement_job_id` set. |
| `RUNNING` | Worker owns the job in some process. |

---

## 4. Manual resume / detection handoff (human-in-the-loop)

**Endpoints:** `POST /jobs/{job_id}/detection_response` and `POST /jobs/{job_id}/resume` (alias). Implemented as `submit_detection_response` in `service/routes.py`.

**Preconditions:**

- Lifecycle for `job_id` is `AWAITING_CORRECTION`.
- `replacement_job_id` is **null** (single handoff winner).

**Sequence (normative):**

1. Load stored `TrackRequest` JSON and all checkpoints for **old** `job_id`.
2. Build `TrackRequest` for the **new** job (`new_job_id`): original params + corrected `box_a` / `box_b` + checkpoint-derived resume fields (`build_resume_plan` / `resume_plan_to_request_fields`).
3. `INSERT` new **`job_lifecycle`** row for `new_job_id` with `parent_job_id = job_id`, `origin_job_id` carried from the old row (or the old id if absent), and `progress_percent` / frames **seeded** from the latest checkpoint `worker_state` (so UI does not jump backward to 0%).
4. **Claim handoff** on the old row: conditional `UPDATE job_lifecycle SET replacement_job_id = new_job_id WHERE job_id = old AND replacement_job_id IS NULL AND job_state = AWAITING_CORRECTION` (`claim_replacement`). If the claim loses, the new row is cancelled and the API returns **409**.
5. If `video_id` is non-empty: **`video_analysis_latest_job` is set to `(video_id → new_job_id, job_state = PENDING)`** — this is the handoff pointer for the analysis service.
6. Old job: write terminal **`replaced_by_new_job`** checkpoint (`completed = true`) on the old `job_id`.
7. Old job: `set_state(CANCELLED, sync_latest=False)` so cancelling the superseded job **does not** repoint `video_analysis_latest_job` back to the dead job.

**Response:** Returns `job_id: new_job_id` (the replacement). Callers **must** treat this as the active pipeline id for subsequent polling/SSE (or rely on `get_latest` as below).

### 4.1 Mid-track suspend and instant `job_id` rotation

When the worker hits **mid-track** (or BLACKOUT) human verification, it writes the `track` checkpoint with `pending_detection`, sets lifecycle to **`AWAITING_CORRECTION`**, and **stops the tracking loop immediately** in the same process (no per-frame RE-ID spin after suspend). The original job’s worker task then finishes and **releases the concurrency semaphore** so the **replacement** job created by this resume flow can start at once.

**Implication for `video_analysis_and_annotation_service` (and any poller):** Do **not** assume the pre-resume `job_id` stays `RUNNING` while tracking “winds down.” After `POST /jobs/{job_id}/resume` returns **`new_job_id`**, switch immediately to that id for **`job_lifecycle`**, progress, heartbeats, and **SSE** (`GET /jobs/{job_id}/events`). Prefer **`get_latest(video_id).job_id`** after each resume response so UI and backends stay aligned with the handoff row (`video_analysis_latest_job` is updated in step 5 above).

---

## 5. Automatic recovery (stale worker / crash)

**Trigger:** `RecoveryManager` in `service/reconciler.py` on a timer; claims stale `RUNNING` / `INTERRUPTED` rows and invokes `recover_interrupted_job` in `service/routes.py`.

**High level:**

1. If checkpoints show the pipeline is already terminal-complete, the **same** `job_id` may be marked `COMPLETED` and **no** replacement is spawned.
2. Otherwise a **new** `job_id` is created (same pattern as manual resume: stored request + checkpoint overrides, including upscale/track resume fields).
3. **`claim_replacement`** runs with **`expected_state = INTERRUPTED`** (not `AWAITING_CORRECTION`). It still sets **`replacement_job_id`** on the **old** row; it does **not** flip the old row to `CANCELLED` (the old row typically remains **`INTERRUPTED`** with a pointer to the new job).
4. If `video_id` is non-empty: **`set_latest(video_id, new_job_id, PENDING)`** — same as manual handoff.
5. Terminal **`replaced_by_new_job`** checkpoint on the **old** `job_id`.
6. Schedule worker for `new_job_id`.

### 5.1 Process restart — orphan `PENDING` rows

The asyncio worker is **in-memory** only. `RecoveryManager` scans `job_recovery_index` but, by design, only hands **stale `RUNNING` / `INTERRUPTED`** lifecycles to `recover_interrupted_job`. After manual resume (section 4), the replacement row is inserted as **`PENDING`** and the worker is scheduled in-process; if the engine **restarts** before `run_job` flips that row to **`RUNNING`**, no local task survives and the reconciler **does not** pick up pure `PENDING` rows.

On FastAPI lifespan startup the engine runs **`drain_orphan_pending_jobs_on_startup`** (`service/routes.py`, invoked from `service/app.py`): it reads recent **ACTIVE** `job_recovery_index` partitions (newest rows first, deduped by `job_id`), re-fetches **`job_lifecycle`**, and for rows still **`PENDING`** without `replacement_job_id`, performs a CAS **`UPDATE ... IF job_state = PENDING AND owner_instance_id = <expected>`**, loads **`job_request_params`**, hydrates the in-memory job store, and calls the same **`_schedule_job`** path as a fresh submit.

---

## 6. User cancellation

**`DELETE /job/{job_id}`:** If `replacement_job_id` is already set on that lifecycle row, the API returns **409** and instructs the client to cancel the **latest** job instead — the superseded row is not the active pipeline owner.

Normal cancel: lifecycle → `CANCELLED` and, when `sync_latest` applies, `video_analysis_latest_job` can be updated for that `job_id` / `video_id` per engine rules.

---

## 7. Recommended resolution algorithm (video analysis service)

Use this to pick the correct `job_id` and `progress_percent` for a `video_id` (or to validate an assumed `job_id`).

```
function resolveActiveJob(video_id):
    latest = get_latest(video_id)   // video_analysis_latest_job; may be null
    if latest is null:
        return { job_id: null, lifecycle: null, note: "no mapping; cannot infer active job from video_id alone" }

    job_id = latest.job_id
    lifecycle = get_lifecycle(job_id)

    // Follow replacement chain if caller still holds an older id
    while lifecycle and lifecycle.replacement_job_id:
        job_id = lifecycle.replacement_job_id
        lifecycle = get_lifecycle(job_id)

    return { job_id, lifecycle }

progress = lifecycle.progress_percent  // authoritative
state = lifecycle.job_state
```

**Additional checks:**

1. **Prefer `get_latest(video_id).job_id`** over any client-stored `job_id` after resume/recovery responses.
2. After **mid-track resume** (`AWAITING_CORRECTION` → `POST .../resume`), assume **instant rotation** (§4.1): the superseded job is no longer executing tracking; use the returned `new_job_id` immediately.
3. If you only have a **legacy** `job_id`, read `replacement_job_id` until null, then read **`job_lifecycle`** for that final id.
4. **Do not** infer progress from `video_analysis_latest_job.job_state` alone; read **`job_lifecycle`** for the resolved `job_id`.
5. If `video_id` was **missing** on engine job creation / resume, **`set_latest` was skipped**; `get_latest` may be stale or absent — you must persist engine-returned `job_id` from `POST /track` / resume JSON.

**Starting from a known `job_id` only (no `video_id` mapping):** load `get_lifecycle(job_id)` and apply the same `while (replacement_job_id)` walk. A superseded row is usually `CANCELLED` but still carries `replacement_job_id` to the active replacement.

---

## 8. SSE (`GET /jobs/{job_id}/events`)

Events are keyed by the **`job_id` in the URL**. After handoff, subscribing to the **old** `job_id` will not show replacement progress. Subscribe using the **returned** replacement id or re-resolve via `get_latest(video_id)`.

---

## 9. Implementation references (non-normative)

| Piece | Location |
|--------|-----------|
| Resume plan composition (`build_resume_plan`) | `service/checkpoints.py` — prefers durable **`tracking_s3_key`** over **`partial_tracking_s3_key`** when both exist; scans all checkpoint rows via `resolve_best_tracking_keys_from_checkpoints` (`service/tracking_chain_merge.py`). |
| S3 preflight before replacement `TrackRequest` persist | `service/routes.py` — `preflight_resume_tracking_overrides` after `build_resume_plan` in `submit_detection_response` and `recover_interrupted_job` |
| Chain merge before upscale | `service/tracking_chain_merge.py` — `consolidate_tracking_json_with_job_chain`; `service/worker.py` |
| Manual handoff + `set_latest` + `sync_latest=False` cancel | `service/routes.py` — `submit_detection_response` |
| Automatic recovery | `service/routes.py` — `recover_interrupted_job`; `service/reconciler.py` — `RecoveryManager` |
| Startup orphan `PENDING` drain | `service/routes.py` — `drain_orphan_pending_jobs_on_startup`; `service/app.py` — lifespan; `service/jobs_store.py` — `list_active_recovery_index_rows_newest_first`, `claim_pending_job_takeover` |
| Lifecycle progress writes | `service/worker.py` — `run_job`, `update_progress` via `JobsStore` |
| Claim replacement CAS | `service/jobs_store.py` — `claim_replacement` |
| Latest row upsert | `service/jobs_store.py` — `set_latest` |
| Which states sync `latest_job` | `service/analysis_keyspaces_enums.py` — `states_that_sync_latest_job_row` |

---

## 10. Changelog

| Date | Change |
|------|--------|
| 2026-05-11 | Document resume routing: full-vs-partial preference, checkpoint-wide key scan, S3 HEAD preflight on handoff, chain merge hook reference. |
| 2026-05-10 | Mid-track suspend: tracking exits immediately after checkpoint; semaphore releases so replacement runs without delay; analysis service should rotate to returned `job_id` / `get_latest` at once (§4.1). |
| 2026-05-09 | Startup drain re-schedules orphaned **`PENDING`** jobs after process restart (CAS takeover + `job_request_params` reload); complements stale `RUNNING`/`INTERRUPTED` recovery. |
| 2026-05-09 | Document clip-global `current_frame` / `total_frames` during tracking; lifecycle frames are UI-only for recovery. |
| 2026-05-09 | Initial contract for job rotation and analysis-service resolution rules. |
