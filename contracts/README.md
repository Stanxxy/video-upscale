# Contracts — cross-service sharing

Normative API and handoff documents for the vision engine (`whole-video-analysis`) and BJJ vision main services (e.g. `video_analysis_and_annotation_service` in `bjj-proj`).

**Sharing path:** This `contracts/` directory is the single source consumers copy or submodule from. It is the only copy in the engine repo — no `working_log/contracts/` mirror exists.

---

## API

| Document | Description |
|----------|-------------|
| [service-openapi.yaml](./service-openapi.yaml) | OpenAPI spec for the vision engine REST API (port 9001) |

Human-readable API notes also live in [API.md](../API.md) at the repo root.

---

## Backend handoff (`bjj_backend/`)

| Document | Description |
|----------|-------------|
| [TRACKING_JSON_S3_ARTIFACTS.md](./bjj_backend/TRACKING_JSON_S3_ARTIFACTS.md) | **How to read tracking JSON from S3** — keys, job chains, schema (boxes, keypoints, confidence), sample data |
| [JOB_ROTATION_HANDOFF_AND_RESUME.md](./bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md) | `job_id` rotation, `video_analysis_latest_job`, manual resume vs crash recovery |
| [CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md](./bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md) | V1 checkpoint envelope and per-stage `artifacts.*` S3 pointers |

Companion (backend monorepo): `CHECKPOINT_DATA_SCHEMA_V1.md` — V1 envelope source of truth; keep artifact addendum in sync.

---

## Quick pointer for agents

- **Need per-frame boxes / pose?** → [TRACKING_JSON_S3_ARTIFACTS.md](./bjj_backend/TRACKING_JSON_S3_ARTIFACTS.md) — use `{base_key}_tracked.json`, not a single stale `job_id` partial.
- **Need active `job_id` or progress after resume?** → [JOB_ROTATION_HANDOFF_AND_RESUME.md](./bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md).
- **Need S3 keys from a checkpoint row?** → [CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md](./bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md).
