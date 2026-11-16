# Contracts — cross-service sharing

Normative API and handoff documents for the vision engine (`whole-video-analysis`) and BJJ vision main services (e.g. `video_analysis_and_annotation_service` in `bjj-proj`).

**Sharing path:** This `contracts/` directory is the single source consumers copy or submodule from. It is the only copy in the engine repo — no `working_log/contracts/` mirror exists.

---

## API

| Document | Description |
|----------|-------------|
| [service-openapi.yaml](./service-openapi.yaml) | OpenAPI spec for the vision engine REST API (port 9001) |

---

## Backend handoff (`bjj_backend/`)

| Document | Description |
|----------|-------------|
| [JOB_ROTATION_HANDOFF_AND_RESUME.md](./bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md) | Job lifecycle, crash recovery, cancellation, retired tracking rows |
| [CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md](./bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md) | V1 checkpoint envelope and the highlight stages |

Companion (backend monorepo): `CHECKPOINT_DATA_SCHEMA_V1.md` — V1 envelope source of truth; keep artifact addendum in sync.

---

## Quick pointer for agents

- **Need the active `job_id` or progress?** → [JOB_ROTATION_HANDOFF_AND_RESUME.md](./bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md).
- **Need a checkpoint row's shape?** → [CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md](./bjj_backend/CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md).
