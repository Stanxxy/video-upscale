---
date: 2026-05-10
category: requirement
tags: [service, lifecycle, resume, checkpoints, s3, tracking, handoff, merge]
status: "implemented (see service/tracking_chain_merge.py; worker pre-upscale hook)"
---

# Chain artifact merge and durable tracking resume

**Status:** Core merge + resume routing follow-ups (full-before-partial plan, S3 preflight on handoff, `S3Client.object_exists`) implemented 2026-05-11.

**Related:** `contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md`, `working_log/knowledge-base/insights/2026-05-10-job-handoff-chain-debug-playbook.md`, `service/checkpoints.py` (`build_resume_plan`).

---

## Problem statement

Job rotation (manual resume or crash recovery) creates a **chain** of `job_id`s linked by `parent_job_id` / `replacement_job_id` / `origin_job_id`. Durable tracking state is spread across:

- Per-job **`partial_tracking.json`** (often **tail-only** for the segment that job ran),
- Per-job **final** `*_tracked.json` / `tracking.json` keys after upload,
- Checkpoint rows that may reference keys from **predecessor** jobs.

Downstream stages (upscale, analysis) expect a **single coherent frame-indexed history**. Today, partials can be incomplete, overwritten, or preferred by resume logic over fuller artifacts—leading to **truncated history**, **404 resume**, or **redundant re-tracking** if each job naïvely duplicates full clips.

---

## Goals

1. **2.1 Missing/broken partial (404):** Before trusting `resume_tracking_s3_key` / `partial_tracking_s3_key`, **validate object existence** (and optionally schema/min frame range). Define **fallback order** (e.g. full `tracking_s3_key` from latest completed track upload, predecessor job partials, explicit merge output). **Reconcile** when the checkpoint or request points at a missing object for the referenced `job_id`.

2. **2.2 Tail-only partials vs merged history:** Avoid relying on a single per-job partial as “the” tracking history when the handoff chain spans multiple jobs. **Proposed approach:** **merge multiple S3 sources across the handoff chain** so upscale/analysis sees **full history** without redundant per-job duplication:

   - Walk lineage via **`origin_job_id` / `parent_job_id`** (or forward via `replacement_job_id`) to enumerate jobs in order.
   - For each job, collect **durable partials** and **final `_tracked.json` / tracking segments** where present.
   - **Dedupe by `frame_idx`** (or canonical record key); **deterministic merge order** (e.g. ascending `frame_idx`, tie-break by chain position / job creation time).
   - Emit either a **materialized merged object** in S3 (single key for downstream) or a **virtual merge** step consumed only by upscale/analysis—choice is an implementation decision (see open questions).

---

## Non-goals

- Changing **normative Keyspaces rotation semantics** in the backend contract (this plan consumes them; it does not redefine `replacement_job_id` rules).
- Replacing **`video_analysis_latest_job`** with a new discovery mechanism (still use `get_latest` + lifecycle walk per contract).
- Full **distributed locking** or multi-worker GPU scheduling (single-job concurrency assumptions remain).
- **Guaranteed idempotent SNS publish** (known gap elsewhere).
- Implementing **merge** in a follow-up PR — **initial chain merge** shipped 2026-05-10 in `service/tracking_chain_merge.py`.

---

## Phased rollout

### Phase 1 — Design

- Finalize **merge semantics**: record schema, `frame_idx` authority, behavior on overlaps vs gaps.
- Decide **materialized merged artifact** vs **on-the-fly merge** in worker vs dedicated batch step.
- Specify **validation API**: head/object-exists, min/max frame probe, JSON schema version.
- Align with **`build_resume_plan`** overrides: when merged output exists, ensure partial preference does not trump validated full history (policy table).

### Phase 2 — Worker and checkpoint behavior

- Add **pre-resume validation** (404 handling, fallback keys).
- Introduce **merge pass** (or incremental merge on handoff) so replacement jobs inherit **full deduped history** where required.
- Ensure **checkpoint writes** reference keys that exist or are explicitly queued for creation.

### Phase 3 — Checkpoint schema (if needed)

- Optional new artifact keys, e.g. `merged_tracking_s3_key`, `merge_manifest_job_ids`, `merge_schema_version`.
- Document in `contracts/bjj_backend/` addendum if schema ships.

### Phase 4 — Tests

- Unit: merge dedupe + ordering with synthetic JSONL-like frames.
- Integration: mocked S3 + chain of 3+ jobs; partial-only on middle job; full key on last; upscale receives full index range.
- Regression: **404 partial** triggers fallback without crashing worker.

### Phase 5 — Rollout risks

- **S3 cost/latency:** merging large clips; may need streaming merge or background job.
- **Consistency:** merge must not read partially written objects; consider ETag/version checks.
- **Backward compatibility:** old checkpoints without merge keys must still resume via legacy path until migrated.
- **Operator visibility:** log merge provenance (`job_id` list, source keys) for debugging.

---

## Resolved decisions (2026-05-10)

1. **Keyspaces query vs lineage blob:** Walk the chain via **`parent_job_id`** (N `get_lifecycle` calls) — acceptable.
2. **Source of truth for “full” tracking:** Prefer **`artifacts.tracking_s3_key`** (post-upload `*_tracked.json`) when present in checkpoint history; fallback **`partial_tracking_s3_key`**. Scan **newest-first** so terminal **`replaced_by_new_job`** rows do not hide older **`track_completed`** keys.
3. **Where merge runs:** Inside **`run_job`**, immediately after the existing partial+local merge and **before** track-completed / pre-upscale upload — implemented as **`consolidate_tracking_json_with_job_chain`** in **`service/tracking_chain_merge.py`**.
4. **Dedupe rule:** **Last writer wins** on duplicate **`frame_idx`** (merge order: root ancestor → … → parent ancestor → **current** local `tracking.json`).
5. **Outputs:** **Only** the leaf job’s merged JSON is written locally and uploaded as **`{base}_tracked.json`** — no extra per-ancestor materialized keys.

## Open questions (remaining)

1. **Contract with `bjj-vision-backend`:** Display **per-segment** provenance vs **final merged only** (UX).
2. **Operator tooling:** Optional HEAD preflight on resume keys without blocking worker (future hardening).

---

## Acceptance criteria (for future implementation PR)

- [x] **Chain merge** before upscale: `consolidate_tracking_json_with_job_chain` in `service/worker.py` after inline partial+new merge; full `tests/test_tracking_chain_merge.py`.
- [x] **Preflight** `resume_tracking_s3_key`: `preflight_resume_tracking_overrides` on detection_response + recovery; HEAD miss → fallback keys; inconclusive HEAD → unchanged overrides.
- [ ] E2E: multi-job handoff + assert min/max `frame_idx` — **deferred** (requires LocalStack or live S3 in CI).
- [x] **Documentation:** playbook + this requirement updated.
