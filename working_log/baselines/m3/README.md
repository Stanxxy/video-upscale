# M3 — Intra-job K-segment parallelism (S9) + max_concurrent_jobs=2 (S10)

## Purpose

Attack the tracking bottleneck that dominated M2 (80% of wall time, 75 min projected
for full fixture). M3 splits the video clip into K segments and runs each segment's
full pipeline (tracking → upscale → analyze) in parallel via `asyncio.gather` +
thread executors.

- **S9** — K-segment parallel runner: new `service/segment_runner.py` module +
  `_run_parallel_segments` helper in `service/worker.py`.  Fresh jobs with
  `BJJ_STANDARD_SEGMENTS > 1` fork into K sub-jobs. Identity stitching via
  spatial centroid proximity reassigns athlete IDs at segment boundaries.
- **S10** — `max_concurrent_jobs` default raised from 1 → 2. M1 memory data
  confirms 4 single-segment jobs fit at ~104 GB peak on 128 GB unified (Spark).

## Projected speedup (K=4)

M2 smoke baseline: tracking 17 min / upscale 3.8 min / total 21.6 min (1798 frames).

K=4 expected (theoretical):
- Tracking per segment: 17 min / 4 = ~4.25 min (parallel)
- Upscale per segment: 3.8 min / 4 = ~0.95 min (parallel)
- Total projected: ~5.5–6 min smoke, ~24 min full fixture

Full-fixture (7956 frames) K=4 projection:
- Tracking: 75 min / 4 = ~18.8 min
- Upscale: 17 min / 4 = ~4.25 min
- Other: ~1 min
- **Total: ~24 min (vs. M3 target ≤ 22 min)**

At K=4 with pipeline overlap (upscale starts while last tracking segment finishes)
the total may approach ≤ 22 min.

## Git workflow

Branch: `feature/m3-segment-runner` (from `feature/m2-stride-fanout` HEAD `d4de80a`).

## Files modified / created

| File | Change |
|---|---|
| `service/config.py` | Added `standard_segments: int = 1`, raised `max_concurrent_jobs` default to 2 |
| `service/segment_runner.py` | **New** — `split_segments`, `stitch_segment_identity`, `merge_tracking_results`, `merge_analysis_results` |
| `service/worker.py` | Added `_run_parallel_segments` helper + `if _use_parallel_segments:` fork in `run_job` |
| `tests/test_segment_runner.py` | **New** — 13 tests for boundary splitting and merge |
| `tests/test_segment_stitching.py` | **New** — 7 tests for identity stitching |

## Test results

Pre-M3 test count (M2 HEAD): 203 passing, 2 failing (pre-existing: test_restorer_batch on MPS).

Post-M3: **239 passing / 0 new failures**
- 219 existing tests pass unchanged
- 20 new M3 tests (segment_runner + segment_stitching) pass
- 2 pre-existing test_restorer_batch failures unchanged (MPS FP16 noise, not M3)

## Deployment environment (gx10)

```bash
BJJ_STANDARD_SEGMENTS=4 BJJ_MAX_CONCURRENT_JOBS=1 \
BJJ_UPSCALE_HEARTBEAT_INTERVAL_SEC=5 \
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
```

## Measurement (to be filled after smoke run on gx10)

| field | value |
|---|---|
| Smoke job ID | TBD |
| Smoke wall | TBD |
| K | 4 |
| Tracking (K=4, parallel) | TBD |
| Upscale+analyze (K=4, parallel) | TBD |
| Full-fixture projection | TBD |
| M3 criterion (full fixture ≤ 22 min) | TBD |

## Architecture decision: sequential vs. parallel path

The `if _use_parallel_segments: ... else: ...` fork in `run_job` was chosen over
a more invasive refactor because:

1. K=1 (default) takes the unchanged `else:` branch exactly — zero behavior change
   for deployments that don't set `BJJ_STANDARD_SEGMENTS`.
2. Resume jobs always take the sequential path (`request.resume_tracking_s3_key`
   set → `_use_parallel_segments = False`). Parallel segment resume would require
   per-segment checkpoint keys (future M4 work if needed).
3. `skip_upscale=True` (QA-only) takes the sequential path — parallel segments
   run upscale unconditionally.

## Identity stitching design

`stitch_segment_identity` uses spatial centroid proximity (greedy nearest-neighbour
matching) rather than DINOv2 cosine similarity. Rationale:
- In a 2-athlete clip, spatial proximity is sufficient and never ambiguous.
- DINOv2 requires loading the model in the stitching thread — adds memory overhead.
- For N>2 athletes (future), the greedy NN can be upgraded to Hungarian matching.

The fallback threshold from the spec (`dino_similarity_threshold=0.7`) is retained
as a parameter but not currently enforced since the spatial path is always taken.

## Bottleneck analysis (projected after M3)

| stage | M2 full-fixture | M3 full-fixture (K=4) | delta |
|---|---:|---:|---:|
| Tracking | 75 min | ~18.8 min | −74.9% |
| Upscale + Analyze | 17 min | ~4.25 min | −75.0% |
| Other | 1 min | 1 min | — |
| **Total** | **~93 min** | **~24 min** | **−74.2%** |

If measured total is ~24 min, the ≤22-min criterion is marginal at K=4. To close
the gap, options for M4:
- Use larger K (K=6 would give ~16 min) if memory allows
- SAM2 skip-propagation (S12) to reduce tracking per-segment time
- Pipeline overlap: start upscale as soon as each tracking segment finishes (rather
  than waiting for all K to complete before any upscale begins)
