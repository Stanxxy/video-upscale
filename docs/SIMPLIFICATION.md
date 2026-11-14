# Vision Engine Simplification

## 2026-11-13: local vision workflow removed

The engine keeps only its Gemini analysis paths. Decision record:
`working_log/plans/2026-11-13-legacy-taxonomy-decommission.md` (umbrella repo).

**Removed**
- The hybrid tracking package and its public shim, the tracking/segment/annotate runners, the athlete-hint selector, the human-correction route, the worker orchestrator and its tracking/upscale/annotate/upload/publish stages and callbacks.
- Root `main.py`, `pipeline.py` and the two restorer modules, the regression oracle, the legacy GPU requirements file, `pyproject.toml` (it only packaged the tracking packages) and the stride/upscale config.
- Replacement-job resume (the resume factory, the tracking-chain merge, the resume plan, the `AWAITING_CORRECTION` state and the `UPSCALE_ANALYZE` stage), `tracking_s3_uri` and `pending_detection`.
- The legacy dual-emit SNS path and the technique shortlist resolver.
- `API.md` and the tracking JSON contract (they documented the removed inputs).

**Wire changes**: `TrackRequest` no longer carries the tracking model selector, the `resume_*` pointers, `player_references`, `athlete_bindings` or other tracking-only fields; extra fields from older callers are ignored. The `/track` highlight ingest no longer fetches athlete reference crops, so its actor axis runs without reference images.

**Historical rows**: rows with `pipeline_kind` NULL or `tracking`, or state `AWAITING_CORRECTION`, are failed with an explicit retired-pipeline message and never resumed (`contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md`).

**Moved**: the numpy box helpers `frame_source` uses are now `service/pipelines/box_geometry.py`.

**Verification**: the deployed entry points (`service.dag_server`, `analysis_run_dispatch`, `current_account_dag_server`, `service.app`) import with no removed module and no local vision package.

## 2026-08-15: clean-code pass (summary)

A behaviour-preserving cleanup removed an archived skill fork, a duplicate contract mirror, one-shot scripts, stale docs and ~2.9 MB of sample data, and fixed latent wrong-boundary test mocks. Its detail is in git history.
