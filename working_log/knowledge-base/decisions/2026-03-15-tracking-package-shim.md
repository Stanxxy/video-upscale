---
date: 2026-03-15
category: decision
tags: [python, packaging, tracking, naming]
status: active
---

# tracking/ Package Shim to Avoid Module Name Collision

## Context
The main tracking code lives in `test_tracking/tracking.py`. But `test_tracking/` is
not a proper Python package, and importing `from test_tracking.tracking import ...`
is awkward in service code. A `tracking/` package would naturally shadow the `tracking.py`
file inside it (Python name collision).

## Content
`tracking/__init__.py` is a shim that:
1. Adds `test_tracking/` to `sys.path`
2. Uses `importlib` to load `tracking.py` as `_tracking_module` (avoiding the name clash)
3. Re-exports `run_tracking`, `detect_persons`, `run_pipeline` at the `tracking` package level

Service code imports cleanly: `from tracking import run_tracking`.

## Rationale
Avoids renaming `test_tracking/tracking.py` (would break the CLI pipeline) while
providing a clean import surface for the service. The shim is 10 lines and easy to reason about.

## Impact
- `tracking/__init__.py` — the shim (do not delete)
- `service/tracking_runner.py` — imports from `tracking`
- If `test_tracking/tracking.py` is renamed, update the shim's importlib call
