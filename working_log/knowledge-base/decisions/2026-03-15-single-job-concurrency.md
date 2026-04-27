---
date: 2026-03-15
category: decision
tags: [service, architecture, concurrency, gpu]
status: active
---

# Single-Job Concurrency Model

## Context
The service runs on a GPU machine. ML models (RF-DETR, SAM2, Real-ESRGAN) cannot
safely share GPU memory across concurrent jobs without OOM errors.

## Content
The FastAPI service (`service/app.py`) enforces a single-active-job constraint:
only one analysis job runs at a time. Job state is tracked in two stores:
- `service/job_store.py` (in-memory job status for active process execution)
- `service/jobs_store.py` (Keyspaces-backed lifecycle/checkpoint persistence)

Incoming `POST /track` requests are rejected with HTTP 429 while the semaphore is locked.

A resume endpoint (`POST /jobs/{job_id}/resume`) allows restarting an interrupted
detection step without re-running the full pipeline (added in commit 0145829).

## Rationale
Simpler than a GPU memory partitioning scheme. The analysis pipeline is typically
fast enough that single-job serialization doesn't bottleneck the bjj-vision platform
given expected usage volumes.

## Impact
- `service/job_store.py` — job state tracking
- `service/jobs_store.py` — lifecycle/checkpoint persistence
- `service/worker.py` — enforces single-job constraint
- `service/routes.py` — /jobs, /jobs/{id}, /jobs/{id}/resume endpoints
- `service/reconciler.py` — startup orphan reconciliation is currently a stub (known gap)
- Do NOT add parallel job support without explicit GPU memory budget analysis
