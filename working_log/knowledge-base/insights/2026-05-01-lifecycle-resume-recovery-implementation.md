---
date: 2026-05-01
category: insight
tags: [service, lifecycle, resume, recovery, keyspaces, implementation]
status: active
---

# Lifecycle Resume and Recovery Implementation

## Context
The first implementation batch for the job lifecycle refactor landed after the plan in `working_log/knowledge-base/requirements/2026-04-26-job-lifecycle-resume-refactor-plan.md`. The goal of this batch was to make manual correction resume, cancellation, and stale worker recovery durable enough for PM verification while leaving broader worker checkpoint standardization for later work.

## Content
Implemented behavior:

- Manual correction resume creates a replacement job with a new job ID.
- Replacement lifecycle rows persist `origin_job_id` as the root job and `parent_job_id` as the immediate predecessor.
- Source jobs persist `replacement_job_id` and become terminal `CANCELLED` after replacement creation.
- Replacement creation uses `claim_replacement` so duplicate resume requests cannot both win.
- `video_analysis_latest_job` remains pointed at the replacement job instead of being overwritten by source-job cancellation.
- `DELETE /job/{job_id}` supports Keyspaces-only jobs and writes a final cancellation checkpoint.
- `RecoveryManager` runs as a scheduled vision-engine background manager.
- Recovery discovery uses `job_recovery_index` candidates, then re-checks current lifecycle state and heartbeat before claiming.
- Recovery handles stale `RUNNING` jobs and already-`INTERRUPTED` jobs without replacements.
- Recovery claims are conditional on `owner_instance_id`, `job_state`, and `last_heartbeat_at`.
- Core lifecycle/heartbeat writes propagate recovery-index write failure instead of silently losing recovery visibility.

Verification evidence:

- `source venv/bin/activate && pytest tests/test_resume_endpoint.py tests/test_job_cancellation.py tests/test_reconciler.py tests/test_jobs_store.py -v`
- Result: `16 passed`
- `ReadLints` on edited service/test files reported no diagnostics.
- Code-review/evaluator loop converged for this batch and recommended PM verification, while noting remaining broader checkpoint work.

## Rationale
The implementation moves the service away from process-local ownership as the source of truth. Manual resume and crash recovery now use explicit lineage and conditional claims, which protects the backend/UI from following stale jobs and gives workers a durable path to recover abandoned work.

## Impact
Primary files:

- `service/checkpoints.py`
- `service/jobs_store.py`
- `service/routes.py`
- `service/reconciler.py`
- `service/app.py`
- `tests/test_resume_endpoint.py`
- `tests/test_job_cancellation.py`
- `tests/test_reconciler.py`
- `tests/test_jobs_store.py`

Remaining scope:

- Apply/coordinate Keyspaces schema migration for `parent_job_id`, `replacement_job_id`, and `job_recovery_index`.
- Standardize worker-produced checkpoint schemas across detect, track, upscale/analyze, annotate, upload, and publish.
- Fix worker mid-track frame upload and add tests against actual worker-produced checkpoint shapes.
- Complete durable artifact recovery for tracking JSON, raw analysis windows, annotated video, and SNS publish stage.
