# Agent Handbook — whole-video-analysis (Vision Engine)

This repo is the **vision engine** for the BJJ Vision platform. It handles:
- Gemini analysis of match video: position, actor, action class, outcome (the specific move lives in notes)
- The account analysis DAG (`analysis_core`) and the `/track` highlight orchestrator
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
│   ├── worker/            # Highlight orchestrator + ingest stage
│   ├── pipelines/         # Gemini DAG nodes (executors, scan/critique, frame_source, ...)
│   ├── checkpoints/       # V1 checkpoint builders + highlight resume plan
│   ├── jobs_store/        # Keyspaces persistence facade
│   ├── dag_server.py, analysis_run_*.py, result_*.py  # account DAG host
│   ├── models.py          # Pydantic schemas
│   ├── job_store.py       # In-memory job state
│   ├── config.py          # Env/config management
│   ├── taxonomy_mapper.py # Axis vocabulary + axis-only candidate
│   ├── s3.py              # S3 operations
│   └── sns.py             # SNS event publishing
├── analysis_core/         # DAG prompts, outputs, presets
├── analysis_observer/     # Lifecycle/checkpoint observer
├── analyzer.py            # Gemini sliding-window analyzer (QA identity routes)
├── contracts/             # OpenAPI + backend checkpoint contracts
├── tests/                 # pytest unit tests
├── qa_client/             # Manual QA client scripts
├── working_log/           # Planning docs, knowledge base
├── service.sh             # Service start/stop/restart
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

See `working_log/knowledge-base/INDEX.md` for the full list. The local vision-model workflow (hybrid athlete tracking, upscaling, annotation, human box correction) was removed on 2026-11-13; its decisions are history only.
