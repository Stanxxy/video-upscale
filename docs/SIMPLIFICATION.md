# Vision Engine Simplification

## 2026-11-13: local vision workflow removed

The engine keeps only its Gemini analysis paths. Decision record:
`working_log/plans/2026-11-13-legacy-taxonomy-decommission.md` (umbrella repo).

**Removed**
- `tracking_pipeline/`, `tracking/`, the tracking/segment/annotate runners, `vllm_selector`, the human-loop route, the worker orchestrator and its tracking/upscale/annotate/upload/publish stages and callbacks.
- Root `main.py`, `pipeline.py`, `restorer.py`, `diffusion_restorer.py`, the regression oracle, `requirements-legacy-gpu.txt`, `pyproject.toml` (it only packaged the tracking packages) and the SAM2/stride/upscale config.
- Replacement-job resume (`resume_job_factory`, `tracking_chain_merge`, resume plan, `AWAITING_CORRECTION`, `UPSCALE_ANALYZE`), `tracking_s3_uri` and `pending_detection`.
- The legacy dual-emit SNS path and the technique shortlist resolver.
- `API.md` and the tracking JSON contract (they documented the removed inputs).

**Wire changes**: `TrackRequest` no longer carries `sam2_model`, `resume_*`, `player_references`, `athlete_bindings` or other tracking-only fields; extra fields from older callers are ignored. The `/track` highlight ingest no longer fetches athlete reference crops, so its actor axis runs without reference images.

**Historical rows**: rows with `pipeline_kind` NULL or `tracking`, or state `AWAITING_CORRECTION`, are failed with an explicit retired-pipeline message and never resumed (`contracts/bjj_backend/JOB_ROTATION_HANDOFF_AND_RESUME.md`).

**Moved**: the numpy box helpers `frame_source` uses are now `service/pipelines/box_geometry.py`.

**Verification**: the deployed entry points (`service.dag_server`, `analysis_run_dispatch`, `current_account_dag_server`, `service.app`) import with no removed module and no local vision package.

## 2026-08-15: clean-code pass (summary)

A behaviour-preserving cleanup removed an archived skill fork, a duplicate contract mirror, one-shot scripts, stale docs and ~2.9 MB of sample data, and fixed latent wrong-boundary test mocks. Its detail is in git history.
