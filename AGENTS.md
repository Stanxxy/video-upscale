# Agent Handbook — whole-video-analysis (Vision Engine)

This repo is the **vision engine** for the BJJ Vision platform. It handles:
- Hybrid athlete tracking (RF-DETR detection + SAM2 mask propagation)
- BJJ technique analysis via Gemini API (single-agent and multi-agent modes)
- Video annotation and upscaling
- REST API (FastAPI, default port 8000 via `BJJ_PORT`) consumed by `bjj-vision-backend`

The companion platform monorepo lives at `/Users/stanliu/Documents/bjj-proj/`.

---

## Development Conventions

- Feel free to challenge unclear or unreasonable requirements
- Always remove legacy code when making updates — no dead code
- Use the latest libraries; search the web if unsure of current API
- Default Gemini flash model: `gemini-2.0-flash-preview` (keep updated)
- Use `working_log/` for planning docs, `temp_data/` for intermediate outputs

---

## Directory Structure

```
whole-video-analysis/
├── service/               # FastAPI service (default port 8000)
│   ├── app.py             # Entry point + lifespan
│   ├── routes/            # REST endpoints package
│   ├── worker/            # Async job orchestrator package
│   ├── checkpoints/       # V1 checkpoint builders + resume plan
│   ├── jobs_store/        # Keyspaces persistence facade
│   ├── models.py          # Pydantic schemas
│   ├── job_store.py       # In-memory job state
│   ├── config.py          # Env/config management
│   ├── tracking_runner.py # RF-DETR + SAM2 orchestration
│   ├── video_annotator.py # Video annotation
│   ├── taxonomy_mapper.py # Pipeline → frontend enum bridging
│   ├── vllm_selector.py   # Gemini athlete hint (>2 YOLO candidates)
│   ├── s3.py              # S3 operations
│   └── sns.py             # SNS event publishing
├── tracking_pipeline/     # Installable package (hybrid tracking implementation)
│   ├── hybrid/            # SAM2 loop modules (orchestrator, loop, output, …)
│   ├── hybrid_tracking.py # Public run_tracking shim
│   ├── select_boxes/      # Human box-selection UI (web + cv2)
│   ├── detect.py          # RF-DETR detection
│   ├── sam2_manager.py    # SAM2 propagation
│   ├── identity_manager.py# DINOv2 + color re-ID
│   ├── state_machine.py     # Scramble/cut/fade state
│   ├── pipeline.py        # CLI orchestrator (run: python -m tracking_pipeline)
│   └── ...                # Pose, video I/O, smoothing, etc.
├── tracking/              # Public API re-exports (from tracking_pipeline)
│   └── __init__.py
├── contracts/             # OpenAPI + backend checkpoint contracts
├── pyproject.toml         # setuptools: packages tracking + tracking_pipeline
├── tests/                 # pytest unit tests
├── qa_client/             # Manual QA client scripts
├── working_log/           # Planning docs, knowledge base
│   └── knowledge-base/    # Design decisions, insights, requirements
├── service.sh             # Service start/stop/restart
├── AGENTS.md              # This file
├── API.md                 # Legacy REST reference (see contracts/service-openapi.yaml)
└── bjj_analysis_taxonomy.md # Taxonomy enum source of truth
```

---

## Service Lifecycle

```bash
./service.sh start      # Start the FastAPI service (default port 8000)
./service.sh stop       # Stop the service
./service.sh restart    # Restart
./service.sh status     # Show running status
```

Health check:
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Infrastructure dependencies:
- **AWS** (production S3/SNS): configure credentials and region via `BJJ_AWS_*` env vars
- **Gemini API**: key in `.env` as `GEMINI_API_KEY`

---

## Running Tests

```bash
cd /Users/stanliu/Documents/bjj-proj/whole-video-analysis
source venv/bin/activate
pytest tests/ -v
```

---

## Knowledge Base

Design decisions, technical insights, and requirements are tracked in:

```
working_log/knowledge-base/
├── decisions/    # Architecture and technology choices with rationale
├── insights/     # Debugging discoveries, gotchas, ML tuning notes
├── requirements/ # Feature requirements with acceptance criteria
├── scratch/      # Temporary agent working memory (gitignored)
├── mistakes/     # ENG/OPS/QA mistake logs with quick-fix commands
└── INDEX.md      # Auto-maintained index — update when adding entries
```

**Always read INDEX.md before starting work** — it surfaces known pitfalls and explains why things are built the way they are.

Use the `bjj-team-meta` skill to add new entries at the end of substantial work.

---

## Scratch Paper (In-Session Memory Offload)

Agents may write intermediate notes to `working_log/knowledge-base/scratch/` to free
up in-context working memory during long sessions. Files there are gitignored and
disposable. See `working_log/knowledge-base/scratch/README.md` for naming conventions.

---

## Team Skills

Canonical **bjj-team-*** skill definitions live in the **umbrella** workspace at `../.claude/skills/` (relative to this repo when checked out under `bjj-proj/`). Copy or symlink those folders into your local Claude/Cursor skills path if your environment only loads user-level skills.

| Skill | Trigger |
|-------|---------|
| `bjj-team-product-manager` | Unclear requirements, new features, scope clarification |
| `bjj-team-designer` | UI/UX review (QA client, annotated video output) |
| `bjj-team-engineer` | Implementation, TDD, backend patterns |
| `bjj-team-evaluator` | Architecture, quality, and test-rigor review loop |
| `bjj-team-operator` | Service lifecycle, codebase hygiene (includes vision-engine artifact scans) |
| `bjj-team-meta` | Save design decisions, insights, requirements |

### Workflow Chains

**Feature flow:** PM → Engineer/Designer ↔ Evaluator (mandatory loop) → PM verification → Meta

**Bug fix flow:** Operator (env, if needed) → PM → Engineer/Designer ↔ Evaluator → PM verification → Meta

**Infrastructure/cleanup:** Operator → Meta (when noteworthy)

---

## Key Design Decisions

See `working_log/knowledge-base/INDEX.md` for the full list. Critical ones to know:

1. **Hybrid tracking**: RF-DETR + SAM2 + DINOv2 re-ID — see `decisions/2026-03-15-rfdetr-sam2-hybrid-tracking.md`
2. **Taxonomy mapper**: bridges Gemini output to frontend enums — see `decisions/2026-03-15-taxonomy-mapper-frontend-bridge.md`
3. **Single-job concurrency**: GPU constraint, intentional — see `decisions/2026-03-15-single-job-concurrency.md`
4. **tracking/ + tracking_pipeline**: package layout and public imports — see `decisions/2026-03-15-tracking-package-shim.md`
