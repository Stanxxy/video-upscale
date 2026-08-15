# Vision Engine Clean-Code Simplification Plan

This plan governs the isolated simplification work on
`whole-video-analysis` (the BJJ Vision vision engine). It deliberately stays
out of the original checkout until the feature branch is reviewed and merged.

## Objective

Reduce accidental complexity, dead code, duplicate documentation, and brittle
tests in the vision engine while preserving its observable behavior and its
external contracts.

## Scope

- Repository: `whole-video-analysis`
- Branch: `feature/vision-engine-simplify`
- Worktree: `/Users/stanliu/bjj-proj/whole-video-analysis-wt-s32`

## Non-Goals

- Do **not** change REST routes, request/response schemas, SNS event shapes,
  Keyspaces persistence semantics, or pipeline public behavior.
- Do **not** change external package boundaries (`tracking`, `tracking_pipeline`)
  unless the change is a pure deletion of genuinely unused code and all tests
  still pass.
- Do **not** upgrade libraries or add new dependencies.
- Do **not** introduce fallback logic or broaden exception handling.
- Do **not** touch the original `/Users/stanliu/bjj-proj/whole-video-analysis`
  checkout.

## Clean-Code Principles

1. Delete dead code and stale comments rather than preserving them.
2. Consolidate duplicated assets; prefer one canonical copy.
3. Prefer small, single-responsibility modules and clear names.
4. Keep tests meaningful. Remove tests that merely duplicate other tests or
   only exercise an unavailable environment. Prefer fakes at ML/network
   boundaries over heavyweight integration tests in the unit suite.
5. Preserve architecture documentation for humans, not just code.

## Known Baseline Problems

The baseline was collected with:

```bash
cd /Users/stanliu/bjj-proj/whole-video-analysis-wt-s32
PYTHONPATH=.:/Users/stanliu/bjj-proj/bjj-vision-backend/shared_lib/src \
  /Users/stanliu/bjj-proj/whole-video-analysis/venv/bin/python -m pytest -q
```

Observations:

- `service/worker/gpu.py::_ensure_models_released` calls
  `torch.mps.empty_cache()` whenever `torch.mps` exists, even when MPS is
  unavailable. On this machine that segfaults tests that run the worker
  orchestrator. Fix: check `torch.backends.mps.is_available()` before calling
  `empty_cache()`.
- `tests/test_parallel_upscale_progress.py` has four tests that reach the real
  `RealESRGANRestorer` and fail because `RealESRGAN_x4plus.pth` is absent. The
  intended unit under test is the progress aggregator, not the model loader.
  Rewrite with a fake/patcher at the restorer boundary, or remove only if the
  same behavior is already covered elsewhere.
- `tests/test_service_script.py` has two tests that bind local TCP sockets and
  fail under the sandbox with `PermissionError`. They are environment-dependent
  integration tests, not clean unit tests. Remove them unless they can be
  converted to deterministic pure-function tests without local sockets.

## Suggested Simplification Targets

The implementer should inspect and decide based on evidence:

- Root-level legacy CLI modules (`main.py`, `pipeline.py`, `analyzer.py`,
  `restorer.py`, `diffusion_restorer.py`, `utils.py`) and whether any are still
  reachable from the service or external contracts.
- Duplicate `contracts/` versus `working_log/contracts/` copies.
- `.claude.archived/` stale skill copies if they are untracked/ignored or
  clearly superseded.
- Legacy code and comments marked `LEGACY`, `deprecated`, or `TODO` that no
  longer serve a live path.
- Over-large modules with multiple responsibilities (especially
  `service/pipelines/executors.py`, `service/routes/qa_vlm.py`,
  `service/worker/highlight_orchestrator.py`) — refactor only when the change
  improves clarity without altering behavior.
- Redundant or near-duplicate tests and helpers.

## Required Deliverables

1. Simplified code committed to this worktree branch.
2. A human-readable architecture graph at `docs/ARCHITECTURE.md` using Mermaid
   (or an equivalent clear diagram) showing the service, tracking pipeline,
   storage/event boundaries, and worker orchestration.
3. A simplification report at `docs/SIMPLIFICATION.md` documenting:
   - before/after file and line counts where meaningful
   - dead code removed and why
   - test changes and why
   - architecture decisions/rulings
   - verification evidence
4. Test suite green for the deterministic unit surface, with any remaining
   environment-only skips explicitly justified.

## Verification Command

```bash
cd /Users/stanliu/bjj-proj/whole-video-analysis-wt-s32
PYTHONPATH=.:/Users/stanliu/bjj-proj/bjj-vision-backend/shared_lib/src \
  /Users/stanliu/bjj-proj/whole-video-analysis/venv/bin/python -m pytest -q
```

The shared library is intentionally resolved from the canonical checked-out
source at `bjj-vision-backend/shared_lib/src`, not from a stale editable
install.
