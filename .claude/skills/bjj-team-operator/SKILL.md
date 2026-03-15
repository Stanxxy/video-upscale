# Operator - Service Lifecycle & Codebase Health

## Trigger

**Mode 1 — Service Lifecycle**: Starting/stopping services, health checks, deployment issues, service misbehavior, environment problems.

**Mode 2 — Codebase Hygiene**: "clean up", "remove cruft", "find dead code", "what can we delete", "codebase health", "disk usage", or after a major refactor/migration.

---

## Mode 1: Service Lifecycle

### Startup Scripts

Use the scripts in `scripts/` for all service lifecycle operations:

| Script | Purpose |
|--------|---------|
| `scripts/start-dev-backend.sh [profile]` | Start backend services (default: all) |
| `scripts/start-dev-frontend.sh` | Start both frontend apps |
| `scripts/stop-dev-backend.sh` | Stop all backend services |
| `scripts/stop-dev-frontend.sh` | Stop all frontend apps |
| `scripts/check-health.sh` | Health check all services |

#### Backend Startup Examples
```bash
./scripts/start-dev-backend.sh              # Start all backend services
./scripts/start-dev-backend.sh core         # Start core only (auth, video_management, statistics)
./scripts/start-dev-backend.sh --no-standalone  # Skip standalone analyzer backend
```

The backend script uses `scripts/service_manager.py` with config from `scripts/service_config.yaml`. You can also use the service manager directly:
```bash
cd bjj-vision-backend
poetry run python scripts/service_manager.py --profile core --action start
poetry run python scripts/service_manager.py --action status
poetry run python scripts/service_manager.py --action stop
poetry run python scripts/service_manager.py --service user_auth --action restart
```

### Service Ports & Profiles

Defined in `scripts/service_config.yaml`:

| Service | Config Key | Port | Profiles |
|---------|-----------|------|----------|
| User Authentication | `user_auth` | 8001 | core, all |
| Video Acquisition | `video_acquisition` | 8002 | discovery, all |
| Video Management | `video_management` | 8003 | core, video, all |
| Payment | `payment` | 8004 | all |
| Statistics Management | `statistics` | 8005 | core, all |
| Video Discovery | `video_discovery` | 8006 | discovery, all |
| Video Analysis & Annotation | `video_analysis` | 8008 | analysis, all |
| Standalone Analyzer Backend | (separate) | 9000 | standalone |

| Profile | Services |
|---------|----------|
| **core** | user_auth, video_management, statistics |
| **discovery** | video_discovery, video_acquisition |
| **analysis** | video_analysis |
| **video** | video_management, video_acquisition, video_discovery, video_analysis |
| **all** | all 7 services above |
| **standalone** | standalone analyzer backend only (port 9000) |

### Frontend Apps

Started via `scripts/start-dev-frontend.sh`:

| App | Port | Directory |
|-----|------|-----------|
| BJJ Vision Dashboard | 3000 | `bjj-vision-frontend/` |
| BJJ Video Analyzer | 5174 | `bjj-video-analyzer/` (proxies `/api` to `localhost:9000`) |

### Logs & PIDs

- **Logs**: `logs/` (per-service log files)
- **PIDs**: `.pids/` (per-service PID files, auto-cleaned on stop)

### Health Check
```bash
curl -s http://localhost:{PORT}/health | python -m json.tool
```

### Troubleshooting

See `references/service-runbook.md` for detailed troubleshooting steps.

#### Quick Fixes
- **Port conflict**: `lsof -i :{PORT}` then `kill {PID}` — or the startup script auto-kills stale port holders
- **Missing env**: `cp .env-tmplt .env` and fill values
- **Import errors**: `cd bjj-vision-backend && poetry install`
- **shared_lib not found**: Reinstall with `poetry install` (path dependency)
- **Supabase connection**: Check `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- **LocalStack**: Verify `http://100.79.167.101:4566` is reachable
- **Standalone venv missing**: `cd standalone-analyzer-backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

### Infrastructure Dependencies
- **LocalStack** (S3, DynamoDB): `http://100.79.167.101:4566`
- **Supabase**: Cloud-hosted PostgreSQL + Auth
- **Cassandra**: `100.79.167.101:9042`, keyspace `video_analysis`

### Service Lifecycle Gates
- Service responds to `/health` endpoint
- No unhandled exceptions in startup logs (check `logs/{service}.log`)
- Required infrastructure (LocalStack, Supabase) is reachable
- Any operational fix is documented in mistakes log for future reference

---

## Mode 2: Codebase Hygiene

### Scope

Two project roots are in scope:

| Project | Path | Description |
|---------|------|-------------|
| bjj-proj | `/Users/stanliu/Documents/bjj-proj/` | Main monorepo (frontend + backend + tools) |
| whole-video-analysis | `/Users/stanliu/Documents/whole-video-analysis/` | Vision engine (YOLO/SAM2/Gemini pipeline) |

### Workflow

1. **Snapshot** — Record `git status` and total file/dir counts for both projects
2. **Scan** — Run scan commands for each cleanup category below; collect results into a report
3. **Report** — Present findings grouped by category with sizes, last-modified dates, and recommended action (delete / archive / keep)
4. **Confirm** — User reviews and approves deletions (or excludes items)
5. **Delete** — Remove approved items; use `git rm` for tracked files, `rm -rf` for untracked
6. **Verify** — Re-run snapshot; confirm file counts dropped and `git status` is clean (or only shows intended deletions)

### Cleanup Categories

#### 1. Experiment Outputs & Analysis Runs
Intermediate results from model runs, analysis pipeline outputs, and experiment logs.

**bjj-proj scan targets:**
```
bjj-proj/temp_data/**                         # Intermediate files (mp3, pdf, png, screenshots)
bjj-proj/working_log/*.md                     # Stale planning docs (check git log for staleness)
```

**whole-video-analysis scan targets:**
```
whole-video-analysis/runs/                    # YOLO/SAM2 experiment run outputs
whole-video-analysis/output*/                 # Analysis pipeline output directories
whole-video-analysis/experiments/             # Experiment result directories
whole-video-analysis/*.mp4                    # Root-level video files (test inputs)
whole-video-analysis/*.json                   # Root-level JSON outputs
whole-video-analysis/*.csv                    # Root-level CSV outputs
```

#### 2. Test Artifacts
QA screenshots, browser test artifacts, Python caches, pytest artifacts.

**Scan patterns (both projects):**
```
**/__pycache__/                               # Python bytecode caches
**/.pytest_cache/                             # Pytest caches
**/node_modules/.cache/                       # Node build caches
**/*.pyc                                      # Compiled Python files
```

**bjj-proj specific:**
```
bjj-proj/temp_data/qa-screenshots/            # QA verification screenshots
bjj-proj/**/test-results/                     # Test result artifacts
```

**whole-video-analysis specific:**
```
whole-video-analysis/test_tracking/           # Test tracking outputs (~33MB active code + ~400MB artifacts)
whole-video-analysis/tests/output*/           # Test output directories
```

**IMPORTANT for test_tracking/**: The `test_tracking/` directory contains BOTH active runtime code (imported by `tracking/__init__.py`) AND large artifact subdirectories. Scan subdirectories individually — never delete `test_tracking/` wholesale. Safe to delete:
- `test_tracking/crops/`
- `test_tracking/frames/`
- `test_tracking/visualization/`
- `test_tracking/checkpoints/` (if not actively resuming a run)

#### 3. Stale Working Logs & Planning Docs
Documentation that has been superseded or completed.

**Scan commands:**
```bash
# Find working log files not modified in 30+ days
find bjj-proj/working_log/ -name "*.md" -mtime +30 -type f

# Check if planning docs reference completed/abandoned work
grep -rl "COMPLETED\|ABANDONED\|SUPERSEDED" bjj-proj/working_log/
```

**Decision rule**: If the content is captured in git history or knowledge-base, the working log file can be deleted. If it contains unique architectural decisions not elsewhere, move to `working_log/knowledge-base/decisions/`.

#### 4. Dead Helping Scripts
Scripts in `bjj-vision-backend/helping_scripts/` that reference deleted services or completed one-off tasks.

**Scan commands:**
```bash
# Scripts referencing old service names
grep -rl "dummy-backend-svc\|poc-backend-svc\|my-react-app\|react-video-editor" bjj-vision-backend/helping_scripts/

# Scripts not modified in 60+ days
find bjj-vision-backend/helping_scripts/ -name "*.py" -mtime +60 -type f
find bjj-vision-backend/helping_scripts/ -name "*.sh" -mtime +60 -type f
```

**Decision rule**: If a script references a deleted directory or service, it's dead. If it's a one-off migration script that already ran successfully, it's dead.

#### 5. Temp / PID / Log / Cache Files
Runtime detritus that accumulates during development.

**Scan patterns (both projects):**
```
**/*.pid                                      # PID files from service runners
**/*.log                                      # Log files (except intentional ones)
**/.DS_Store                                  # macOS metadata
**/*.egg-info/                                # Python package metadata
**/dist/                                      # Build output (unless actively deployed)
**/build/                                     # Build output
**/.cursor/debug.log                          # IDE debug logs
```

**whole-video-analysis specific:**
```
whole-video-analysis/*.pid                    # Stale PID files
whole-video-analysis/nohup.out                # Background process output
whole-video-analysis/.env.bak*                # Backup env files
```

#### 6. Integration Debt (Old Names, Stale References)
Code or config referencing renamed/deleted directories and services.

**Scan commands:**
```bash
# References to old directory names (should be zero after migration)
grep -rn "dummy-backend-svc\|poc-backend-svc\|my-react-app" --include="*.py" --include="*.ts" --include="*.tsx" --include="*.json" --include="*.md" --include="*.yaml" --include="*.yml"

# References to "tracking service" (renamed to "vision engine")
grep -rn "tracking.service\|tracking_service" bjj-vision-backend/ --include="*.py" | grep -v __pycache__ | grep -v .pyc

# Stale TODO/FIXME/HACK comments older than 60 days (check git blame)
grep -rn "TODO\|FIXME\|HACK" bjj-vision-backend/ --include="*.py" | head -30
```

**Decision rule**: Old name references in active code = update them. Old name references in deleted/stale files = delete the files. TODO comments referencing completed work = remove the comment.

#### 7. Unused Model Files
ML model weights that are duplicated, broken, or from abandoned experiments.

**Scan commands:**
```bash
# Find all model weight files
find whole-video-analysis/ -name "*.pt" -o -name "*.pth" -o -name "*.onnx" -o -name "*.engine" -o -name "*.safetensors" | xargs ls -lh

# Check for duplicates (same filename in multiple dirs)
find whole-video-analysis/ \( -name "*.pt" -o -name "*.pth" \) -exec basename {} \; | sort | uniq -d

# Find model files in test directories (likely copies)
find whole-video-analysis/test_tracking/ -name "*.pt" -o -name "*.pth" | xargs ls -lh
```

**SAFETY**: Model files are expensive to re-download. Always confirm before deleting. Flag any model file >50MB for explicit user approval.

### Safety Rules

1. **List before delete** — Always show what will be deleted with sizes before removing anything
2. **Never delete active code** — If a file is imported by any non-test module, it's active. Verify with `grep -r "import\|from.*import" --include="*.py"`
3. **Confirm model files** — Any `.pt`, `.pth`, `.onnx`, `.safetensors` file requires explicit user confirmation
4. **Confirm large deletions** — Any single item >10MB requires explicit user confirmation
5. **Git-tracked files** — Use `git rm` for tracked files to keep history clean
6. **Never touch .env files** — Environment files may contain secrets; leave them alone
7. **Preserve .gitignore** — Review but never delete `.gitignore` files
8. **Check git blame for age** — Don't rely only on filesystem mtime; use `git log -1 --format=%ci -- {file}` for accurate last-modified dates
9. **No cleanup during active work** — If `git status` shows uncommitted changes in a target area, skip that area and warn the user

### Hygiene Gates

**Before cleanup:**
- [ ] `git status` is clean (or user acknowledges uncommitted changes)
- [ ] File/directory counts recorded for both projects
- [ ] Disk usage recorded (`du -sh` on both project roots)

**After cleanup:**
- [ ] File counts decreased by expected amount
- [ ] No active imports broken (quick `grep` verification on deleted files)
- [ ] `git status` shows only intended deletions
- [ ] Disk savings reported (before vs after `du -sh`)

---

## Scratch Paper & Mistakes Log
- Use `working_log/knowledge-base/scratch/` for intermediate debugging notes during a session
- Record operational issues in `working_log/knowledge-base/mistakes/OPS-xxx-*.md` with quick-fix commands
- Review past mistakes before troubleshooting — the fix might already be documented
