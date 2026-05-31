# Clean-code refactor: mechanical package split (2026-05-31)

**Status:** implemented on branch `feature/vision-engine-clean-code-refactor`

## Summary

Mechanical decomposition of the vision engine to enforce **≤500 LOC per production Python file** without changing HTTP APIs, Keyspaces/S3/SNS write semantics, checkpoint JSON shapes, or job rotation behavior.

## Package layout

| Area | Before | After |
|------|--------|-------|
| Worker | `service/worker.py` (~2728 LOC) | `service/worker/` package (orchestrator, stages, callbacks, progress) |
| Routes | `service/routes.py` (~807 LOC) | `service/routes/` package |
| Checkpoints | `service/checkpoints.py` (~577 LOC) | `service/checkpoints/` (envelope, builders, query, resume) |
| JobsStore | `service/jobs_store.py` (~529 LOC) | `service/jobs_store/` mixins |
| Hybrid tracking | `tracking_pipeline/hybrid_tracking.py` (~902 LOC) | `tracking_pipeline/hybrid/` + 15-line shim |
| Select boxes | `select_boxes.py` (~534 LOC) | `select_boxes/` web + cv2 modules |
| Advanced tracking | `advanced_tracking.py` (~524 LOC) | `advanced_tracking/` models + tracker |

## Preserved contracts

- OpenAPI v2 paths and response models (`contracts/service-openapi.yaml`)
- V1 checkpoint envelope and `build_resume_plan` precedence
- Keyspaces write ordering, `video_analysis_latest_job`, recovery index
- Public imports: `from service.worker import run_job`, `from tracking_pipeline.hybrid_tracking import run_tracking`, `from service.routes import router, init_routes`

## Regression gates

Characterization tests added before/at start of refactor:

- `tests/test_routes_track_and_status.py`
- `tests/test_worker_checkpoint_sequence_golden.py`
- `tests/test_s3_artifact_key_layout.py`
- `tests/test_tracking_public_api.py`
- `tests/regression/test_metrics.py`

Run full suite: `./venv/bin/python3.13 -m pytest tests/ -v` (collection is slow due to torch/service imports).

Line-count gate:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('.')
exclude = {'venv', '__pycache__', '.pytest_cache', 'bjj_vision_engine.egg-info'}
for path in sorted(root.rglob('*.py')):
    if any(part in exclude for part in path.parts): continue
    n = len(path.read_text(errors='ignore').splitlines())
    if n > 500 and not str(path).startswith('tests/'): print(n, path)
PY
```

## Notes

- `API.md` remains legacy; canonical contract is `contracts/service-openapi.yaml`.
- Pytest startup can take several minutes on this repo; prefer targeted test files during iterative extraction.
