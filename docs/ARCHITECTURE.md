# Vision Engine Architecture

This document describes the `whole-video-analysis` vision engine as it
stands on `develop` (post engine#18). It is the human-readable companion
to the OpenAPI contract in [`contracts/service-openapi.yaml`](../contracts/service-openapi.yaml)
and the checkpoint/handoff docs in [`contracts/bjj_backend/`](../contracts/bjj_backend/).

## System view

```mermaid
flowchart LR
    subgraph Consumers
        BE[FastAPI backend services]
        QA[QA client / vlm_studio.html]
    end

    subgraph Engine["whole-video-analysis (FastAPI)"]
        APP[service/app.py<br/>lifespan: recovery + orphan drain]
        QAAPP[service/qa_app.py<br/>lifespan-free QA app]
        ROUTES[service/routes/*<br/>jobs, events, resume, qa, qa_pipeline, qa_vlm]

        subgraph Worker["Worker orchestration"]
            HL[highlight_orchestrator.py<br/>run_highlight_job (default)]
            LEGACY[orchestrator.py<br/>run_job (legacy GPU profile)]
            STAGES[stages/*<br/>download, track, upscale, annotate, upload, publish]
            PIPES[service/pipelines/*<br/>DAG framework: executors, frame_source,<br/>simplified_tags, chunk_segment, highlight_scan/critique]
        end

        TRACK[tracking_pipeline/<br/>RF-DETR + SAM2 + DINOv2 re-ID<br/>tracking/ public shim]
    end

    subgraph Storage["External boundaries"]
        KS[(Keyspaces<br/>job_lifecycle + checkpoints<br/>service/jobs_store/*)]
        S3[(S3<br/>videos, tracking JSON, crops,<br/>annotated videos, ref images)]
        SNS[(SNS<br/>analysis_complete events)]
        GEM[Gemini API<br/>native video + Files API]
    end

    BE -->|REST / SSE| ROUTES
    QA -->|REST| QAAPP
    QAAPP --> PIPES
    ROUTES -->|schedule job| HL
    ROUTES -->|legacy /jobs| LEGACY
    HL --> STAGES
    LEGACY --> STAGES
    STAGES --> PIPES
    STAGES --> TRACK
    HL --> KS
    LEGACY --> KS
    STAGES --> S3
    HL -->|analysis_complete| SNS
    PIPES --> GEM
```

## Entry points

- **Production service** — `service/app.py` builds the FastAPI app and wires
  the full router. Its `lifespan` drains orphan pending jobs and bootstraps
  recovery so a restart can claim jobs whose owner died mid-run.
- **Standalone QA app** — `service/qa_app.py` mounts the QA routes
  (`qa_vlm`, `qa`, `qa_pipeline`) without the Keyspaces/SAM2 startup
  requirements, so the QA harness runs without the production stack.

## Worker orchestration

Two orchestrators exist behind the routes:

1. **`service/worker/highlight_orchestrator.py::run_highlight_job`** is the
   default path for the Gemini-native highlight pipeline. It ingests a source
   (YouTube or Files API URI), runs the `highlight-scan-critique-analyze`
   DAG, reconciles per-chunk candidates with seam dedup and majority-vote
   attribution, then publishes `analysis_complete` via SNS.
2. **`service/worker/orchestrator.py::run_job`** is the legacy tracking +
   upscale + analyze pipeline (`track → upscale → annotate → upload →
   publish`), retained for the optional GPU profile. It is the only path that
   drives `tracking_pipeline` with the sequential `detection_cb` contract.

Both orchestrators share `service/worker/context.py` (`WorkerRunContext`),
`service/worker/progress.py` (lifecycle progress helpers),
`service/worker/gpu.py` (model-release + partial tracking JSON loader), and
the `stages/` package.

## Pipeline DAG framework

`service/pipelines/` implements a declarative DAG framework
(`registry.py`, `models.py`, `executors.py`) that lets QA and production
share the exact Gemini-calling code. Node implementations include:

- `frame_source.py` — frame sampling from a local video.
- `simplified_tags.py` — 4-axis simplified-tags tagging.
- `chunk_segment.py` + `time_dedup.py` — chunk segmentation and
  seconds-native dedup.
- `highlight_scan.py`, `highlight_critique.py`, `highlight_axes.py` — the
  v2 highlight scan/critique/analyze pipeline nodes.
- `gemini_retry.py` / `gemini_upload.py` — Gemini transport and Files API
  upload mechanics.

## Tracking pipeline

`tracking_pipeline/` is the hybrid tracking implementation (RF-DETR detection,
SAM2 mask propagation, DINOv2 + color re-identification, pose, smoothing,
state machine) with `tracking/` as the public re-export shim. It is only
reachable from the legacy GPU orchestrator path and its own CLI
(`tracking_pipeline/pipeline.py`).

## Persistence and events

- **Keyspaces** — `service/jobs_store/` is the durable facade for
  `job_lifecycle`, checkpoints, latest-job pointers, and the recovery index.
  The engine is the sole durable progress writer; the backend projects these
  rows through SSE.
- **S3** — `service/s3.py` wraps object storage for input videos, tracking
  JSON, crops, annotated videos, and athlete reference images.
- **SNS** — `service/sns.py` + `service/taxonomy_mapper.py` publish
  `analysis_complete` events with axis-native fields and dual-emit legacy
  fields for backward-compatible consumers.
- **Config** — `service/config.py` is the single env-driven configuration
  surface (S3/SNS endpoints, Gemini settings, segment bounds, timeouts).

## Contracts

`contracts/` is the single source of truth for cross-service contracts:

- `service-openapi.yaml` — canonical REST contract (replaces the legacy
  root `API.md` human notes).
- `bjj_backend/` — checkpoint artifact addendum, job rotation/handoff, and
  the S3 tracking-JSON artifact layout.

The old `working_log/contracts/` mirror was removed in the 2026-08-15
clean-code simplification; `contracts/` is now the only copy.
