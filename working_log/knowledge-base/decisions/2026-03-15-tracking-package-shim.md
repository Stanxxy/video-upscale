---
date: 2026-03-15
category: decision
tags: [python, packaging, tracking, naming]
status: active
---

# tracking/ Public API + tracking_pipeline Package

## Context
The hybrid loop lives in `tracking_pipeline/hybrid_tracking.py` (renamed from `tracking.py`)
so it does not collide with the top-level `tracking/` **package**, which re-exports a stable
service import surface (`from tracking import run_tracking`).

`tracking_pipeline/` is a normal Python package (relative imports, `pyproject.toml`).

## Content
- **`tracking_pipeline/`** — implementation (`hybrid_tracking`, `detect`, `pipeline`, etc.).
  Run the tracking CLI from repo root: `python -m tracking_pipeline` or
  `python -m tracking_pipeline.pipeline` (same arguments as before).
- **`tracking/__init__.py`** — thin re-exports from `tracking_pipeline` (no `sys.path` mutation).
- **`pyproject.toml`** — declares installable packages `tracking` and `tracking_pipeline`
  (`pip install -e .` optional; repo root on `PYTHONPATH` / cwd is enough for local runs).

`tracking_pipeline/pipeline.py` uses relative imports (e.g. `from .hybrid_tracking import run_tracking`),
so there is no import cycle with `tracking`.

## Rationale
Proper packaging removes `sys.path.insert(0, …)` (which shadowed the repo-root `pipeline.py` and
forced `importlib` workarounds in `service/worker.py`). Service code keeps the short `from tracking import …` imports.

## Impact
- `tracking/__init__.py`, `tracking_pipeline/__init__.py`, `pyproject.toml`
- `service/tracking_runner.py`, `service/worker.py` — plain imports.
- CLI: prefer `python -m tracking_pipeline` from repo root.
