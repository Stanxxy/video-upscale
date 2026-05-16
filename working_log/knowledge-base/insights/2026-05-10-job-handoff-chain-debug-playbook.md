---
date: 2026-05-10
category: insight
tags: [service, lifecycle, resume, keyspaces, checkpoints, s3, debugging, ops]
status: active
---

# Job handoff chain debugging playbook (Keyspaces + S3)

## Key Insight / Takeaway

When investigating wrong resume inputs, **tail-only `partial_tracking.json`**, or **404 on partial objects**, walk the full **`job_lifecycle` replacement chain** and validate **per-job** `job_request_params`, **latest checkpoint per stage**, and **S3 artifacts**. Resume routing in `build_resume_plan` **prefers `partial_tracking_s3_key` over full `tracking_s3_key`** when both are present on the latest track checkpoint—so a stale or incomplete partial can dominate the full merged history.

**Normative contract:** `contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md` (rotation fields, `video_analysis_latest_job`, terminal `replaced_by_new_job` checkpoints).

---

## 1. Handoff chain tracing in Keyspaces (`job_lifecycle`)

**Goal:** Know every `job_id` that participated in the pipeline for a clip, and which row is active.

**Fields (per row):**

| Field | Use |
|--------|-----|
| `origin_job_id` | First job in the logical chain (stable across replacements when set correctly). |
| `parent_job_id` | Immediate predecessor when this row was created as a replacement. |
| `replacement_job_id` | On the **old** row: points to the **new** job after handoff is claimed (nullable until then). |
| `video_id` | Join key to `video_analysis_latest_job` (when present at job creation); see contract §2.2 / §7. |

**Two equivalent walks:**

1. **Leaf → root:** Start from a known `job_id` (or `get_latest(video_id).job_id`), read `parent_job_id` repeatedly until null/absent — collects the chain backward in time.
2. **Root → leaf:** Start from `origin_job_id` or earliest job, follow `replacement_job_id` forward until null — matches the contract’s “follow replacement until active” loop (`JOB_ROTATION_HANDOFF_AND_RESUME.md` §7).

**Do not** assume a single row holds live progress after resume: progress and heartbeats move to the **replacement** `job_id` (contract §1).

---

## 2. `TrackRequest` inspection (`job_request_params`)

For **each** `job_id` in the chain, load persisted params (typically `job_request_params.request_json` or your ORM’s equivalent) and verify resume-related fields:

| Field | Why |
|--------|-----|
| `start_time` / `end_time` | Clip bounds for the job’s logical segment vs full clip. |
| `resume_tracking_s3_key` | Which S3 object the worker will open for tracking continuation or skip-tracking load. |
| `resume_from_frame` | Absolute/global resume cursor (see contract: lifecycle frame counts are UI-oriented; this drives recovery). |
| `resume_from_job_id` | Lineage hint when present (cross-job resume bookkeeping). |
| `analysis_raw_s3_key` | Upscale/analysis continuation cursor artifact. |

Misalignment here (e.g. partial key from job A stored on job B’s row after handoff) usually shows up **before** S3: compare against checkpoints on the **same** `job_id`.

---

## 3. Checkpoint inspection (`job_stage_checkpoints`)

**Grouping:** For each `job_id`, take the **latest row per `stage_name` by `updated_at`** (not necessarily insert order if rows are rewritten).

**Artifacts to read:**

| Stage / context | Keys |
|-----------------|------|
| `track` | `artifacts.partial_tracking_s3_key`, `artifacts.tracking_s3_key`, `artifacts.resume_from_frame` (and root-level fallbacks per `service/checkpoints.py` builders). |
| `upscale_analyze` | `artifacts.analysis_raw_s3_key`, `artifacts.tracking_s3_key` (used when upscale dominates resume). |

**Terminal handoff:** A row with `reason == "replaced_by_new_job"` and `completed == true` on the **old** job marks pipeline handoff. For **TRACK** rows in that terminal state, expect **`replacement_job_id`** (and minimal artifact carryover)—do not assume a full artifact set on that row alone.

---

## 4. S3 artifact audit (per job in chain)

For each `job_id`, validate durable objects under the checkpoint prefix (pattern may vary by deployment; typical):

```bash
aws s3 cp "s3://<bucket>/checkpoints/<job_id>/partial_tracking.json" -   # or head/first bytes
# Optional: same for final tracking outputs, e.g. *_tracked.json paths from checkpoint artifacts
```

**Checks:**

- **HTTP 404 / NoSuchKey:** Missing partial for that job—resume paths that depend on `partial_tracking_s3_key` will break or fall back incorrectly.
- **JSON validity:** `python -m json.tool` or parse in a REPL.
- **Content stats:** Min/max `frame_idx` (or equivalent per schema), row/array length, monotonicity—confirms “tail only” vs full history.

Repeat for **every** job in the chain; a gap on an intermediate job is a common source of “resume thinks it has data but upscale sees only a tail.”

---

## 5. Root causes summary (from investigation)

These patterns explain **tail-only partials**, **wrong resume key**, and **analysis/upscale seeing truncated history**:

1. **Streaming partial overwrite:** The worker opens streaming `partial_tracking.json` (or equivalent) with **`"w"` per tracking run**, so each segment overwrites the object; **merge into a single durable history happens only when tracking completes** (or on a separate merge path if implemented). Mid-handoff jobs therefore often leave **only the last segment** in partial.
2. **`build_resume_plan` + checkpoint scan:** `build_resume_plan` uses **`resolve_best_tracking_keys_from_checkpoints`** (`service/tracking_chain_merge.py`) over **all** checkpoint rows (newest-first for full keys) so a terminal **`replaced_by_new_job`** TRACK row does not hide **`tracking_s3_key`**. Branch order: **full `tracking_s3_key` before partial-only** (skip-tracking path beats mid-track partial tail).
3. **Resume preflight:** `preflight_resume_tracking_overrides` runs on **`POST .../detection_response`** and **`recover_interrupted_job`**: HEAD the planned `resume_tracking_s3_key`; if missing, try fallback keys from checkpoints; inconclusive HEAD (network) leaves overrides unchanged.
4. **Chain merge before upscale:** **`consolidate_tracking_json_with_job_chain`** (still documented separately).
5. **Missing partial in chain:** If an intermediate job never wrote a partial (404) but downstream resume still references it (or inherits bad overrides), recovery/reconciliation must **validate existence** and apply **fallback keys** or **multi-source merge**.

---

## Related code references (non-exhaustive)

- `service/checkpoints.py` — `build_resume_plan`, `latest_checkpoint_data_by_stage`, `ResumePlan`.
- `service/tracking_chain_merge.py` — `resolve_best_tracking_keys_from_checkpoints`, `preflight_resume_tracking_overrides`, `consolidate_tracking_json_with_job_chain`.
- `service/routes.py` — resume / recovery handoff calls preflight before persisting replacement `TrackRequest`.
- `service/worker.py` — tracking partial upload / checkpoint writes, post-upload `tracking_s3_key` on track row.
- `contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md` — rotation and resolution algorithm.
