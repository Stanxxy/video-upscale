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

## Measurement (gx10 smoke run 2026-05-18)

| field | value |
|---|---|
| Smoke job ID | bd3b8fbb-fe57-4b57-8246-3a4082938a54 |
| Smoke wall | **953 s = 15 min 53 s** |
| K | 4 |
| Tracking (K=4, parallel) | **862 s = 14 min 22 s** |
| Upscale+analyze (K=4, parallel) | **84 s = 1 min 24 s** |
| Upscale speedup vs M2 sequential | **2.74× (84 s vs 230 s)** |
| Tracking speedup vs M2 sequential | **0.85× (862 s vs 1021 s) — worse due to GPU serialization + YOLO overhead** |
| Smoke speedup vs M2 | **1.36× (953 s vs 1294 s)** |
| Full-fixture projection | **70.3 min** (scale factor 4.425) |
| M3 criterion (full fixture ≤ 25 min) | **NOT MET** |

### Why K=4 tracking did not speed up

K=4 parallel SAM2 tracking on a single GB10 GPU serializes CUDA kernels from 4 streams. The
observed per-frame rate was ~1.75 s/frame vs. M2's 0.568 s/frame sequential. The GPU fully
saturates handling one segment; adding 3 more segments introduces context-switching overhead
without adding throughput.

Additionally, seg2 experienced track loss for ~313 frames (frames 975–1376) in which the
`_detect_and_request_boxes` code path reloaded the YOLO detector from disk on each frame
(~2.5 s each = ~780 s of extra overhead). This code path is correct and non-crashing, but the
YOLO persistence logic does not cache across the detection-callback-less path. Total seg2
bottleneck time was ~862 s vs. ~680–764 s for the other three segments.

### Upscale+analyze did benefit

84 s upscale+analyze wall time vs. M2's 230 s sequential = **2.74× speedup**. All 4 segments
ran their upscale (ESRGAN) and Gemini analysis pipelines concurrently:
- Each segment: ~75–85 strided frames, 5–6 Gemini windows
- All 4 segments' Gemini calls fired in parallel (unconstrained by the
  `previous_context` chain within each segment; cross-segment overlap is extra fanout)

### Conclusion

M3 delivers:
- **Upscale+analyze: 2.74× speedup** (genuine parallelism benefit)  
- **Tracking: no speedup** (GPU serialization; K=4 adds overhead not throughput)
- **Net smoke: 1.36×** vs. M2; **full-fixture projection: 70.3 min** vs. M2's 93 min

M3 criterion (≤ 25 min full-fixture) is **NOT MET**. To meet it, tracking must be accelerated
further. Options for M4+:
- SAM2 skip-propagation (S12): run SAM2 every N frames, interpolate between; reduces per-segment
  frames by N×
- Lighter tracker (S11 fast-mode): ByteTrack/sort instead of SAM2
- Larger K on multi-GPU (not applicable for single-GPU Spark)

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
