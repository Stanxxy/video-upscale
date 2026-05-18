# M6: Wire prop_stride Through Parallel Segment Path

**Branch**: `feature/m6-parallel-prop-stride`
**Date**: 2026-05-18

## Problem Fixed

`_run_parallel_segments` in `service/worker.py` called `run_tracking_job(...)` WITHOUT passing `prop_stride`. Since `run_tracking_job` defaults to `prop_stride=1`, the K=4 segment parallel path always used full-frame tracking even when `BJJ_STANDARD_PROP_STRIDE=5` was configured. The sequential path (K=1) already correctly passed `prop_stride`.

## Changes (`service/worker.py`, commit `m6: wire prop_stride through _run_parallel_segments`)

1. **`_run_parallel_segments` signature**: Added `eff_prop_stride: int` parameter.

2. **`_track_segment` closure**: Added `prop_stride=eff_prop_stride` to `run_tracking_job(...)` call.

3. **Call site in `run_job`**: Computed `_eff_prop_stride_ps` using same fast/standard logic as sequential path:
   ```python
   _is_fast_mode_ps = request.processing_mode == ProcessingMode.FAST
   if _is_fast_mode_ps:
       _eff_prop_stride_ps = config.fast_prop_stride
   else:
       _eff_prop_stride_ps = config.standard_prop_stride
   ```

4. **Frame-stride override for parallel path**: When `prop_stride > 1` or fast mode, force `frame_stride=1` (M5 fix now applied to parallel path too — was previously missing).

5. **Logging**: Added `prop_stride` to the parallel-segment mode info log line.

## Smoke Test Results (gx10 DGX Spark, 2026-05-18)

### Standard Mode (K=4, prop_stride=5, SAM2 base-plus, ESRGAN)

**Job**: `d8f41d1e-24e0-4aa1-9209-460c64565559`
**Config**: `BJJ_STANDARD_SEGMENTS=4 BJJ_STANDARD_PROP_STRIDE=5`

**Log verification**: `parallel-segment mode k=4 ... prop_stride=5` — confirmed.

- **Total smoke: 305s = 5.1 min**
- Download+detect: ~14s
- Tracking (4 segs K=4, GPU-serialized): 174.9s (last segment done in 174.9s)
  - seg0: 169.4s, 96 frames
  - seg1: 174.9s, 102 frames
  - seg2: 166.1s, 102 frames
  - seg3: 155.5s, 97 frames
  - Total: 397 raw → 361 merged (de-overlap)
- Upscale+analyze (K=4 parallel ESRGAN + Gemini): ~100s
  - 4 ESRGAN instances running simultaneously on GPU
  - 24 Gemini windows across 4 segments (parallel per-segment context chains)
- Annotate+upload+publish: ~16s

**Full-fixture projection (linear)**: 305 × 4.425 = **1350s = 22.5 min**

**Fixed-overhead-adjusted projection**: Fixed overhead ~28s. Variable = 277s × 4.425 = 1226s + 28s = **1254s = 20.9 min**

**Target ≤22 min**: Adjusted projection MET (20.9 min). Linear conservative projection borderline (22.5 min).

### M5 vs M6 Comparison (standard mode)

| Config | Smoke | Full-fixture (linear) | Fixed-overhead-adjusted |
|--------|-------|----------------------|------------------------|
| M5 sequential (K=1, prop_stride=5) | 495s | 36.5 min | ~35 min |
| M5 sequential measure (K=1, prop_stride=5) | 459s | 33.9 min | ~32.7 min |
| **M6 parallel (K=4, prop_stride=5)** | **305s** | **22.5 min** | **20.9 min** |
| M6 speedup vs M5 | 1.51× | — | — |

The K=4 parallel path reduces smoke from 459s to 305s — a 33.5% reduction, primarily from parallel upscale+analyze (sequential 117s ESRGAN → ~50s parallel wall time) and parallel Gemini analysis (sequential 271s → ~100s across 4 segments running independently).

### Fast Mode (prop_stride=30, SAM2-tiny, BicubicRestorer)

**Job**: `92e96716-2c13-45a4-a849-4787aa089f01`
**Config**: `BJJ_FAST_PROP_STRIDE=30 BJJ_MAX_CONCURRENT_JOBS=1`

**Log verification**: `processing_mode=ProcessingMode.FAST ... prop_stride=30 enable_pose=False` — confirmed.

- **Total smoke: 64s = 1.07 min**
- Download+detect: ~7s
- Tracking: 26s (60 SAM2 frames at 2.3 fps measured = 433ms/frame SAM2-tiny)
- Upscale+analyze: ~22s (3 Gemini windows parallel, BicubicRestorer negligible)
- Annotate+upload+publish: ~9s

**Full-fixture projection (linear)**: 64 × 4.425 = **283s = 4.7 min**

**Fixed-overhead-adjusted projection**: Fixed overhead ~16s. Variable = 48s × 4.425 = 212s + 16s = **228s = 3.8 min**

**True full-fixture estimate from per-frame rates**: 7956/30 = 265 SAM2 frames × 433ms = 115s tracking + 13 Gemini windows parallel fanout=24 → 1 batch = 11s analyze + 30s upload = 156s = **2.6 min**.

**Target ≤2.2 min**: NOT MET (linear 4.7 min, adjusted 3.8 min, per-rate 2.6 min). Smoke fixed overhead inflates the linear projection. The per-rate projection (2.6 min) is closest to truth for full fixture.

M5 fast mode (prop_stride=24) measured: 61s → 4.5 min linear / 3.7 min adjusted.
M6 fast mode (prop_stride=30) measured: 64s → 4.7 min linear / 3.8 min adjusted.

Fast mode linear projection doesn't improve despite higher prop_stride, because the smoke is dominated by fixed overhead (download, model load, upload). The true full-fixture projection is below the linear estimate by a factor of ~1.5× due to fixed overhead.

## Root Cause Analysis

### Why K=4 standard mode hits the target

With K=4 segments:
- Tracking: GPU-serialized across 4 threads, same total SAM2 frames (361) → similar tracking time as sequential (175s vs M5 203s, slightly faster due to shorter per-segment warmup)
- Upscale: 4 parallel ESRGAN instances, each processing ~90 frames → wall = max(~45s) vs sequential 117s = 2.6× speedup  
- Analyze: 4 independent context chains running in parallel → wall = max(~30s) vs sequential 271s = ~9× speedup

The K=4 upscale+analyze parallelism is the key win: sequential 388s (117+271) → parallel ~100s = 3.9× speedup for that phase.

### Why fast mode (prop_stride=30) doesn't hit 2.2 min from linear

Fast mode smokes are dominated by fixed overhead (S3 download, model load, S3 upload) which takes ~16s regardless of clip length. The smoke clip (30s = 1798 frames) has only 3-4 min of work but 1 min of fixed overhead → 60s total. The linear scale factor (4.425×) multiplies the fixed overhead by 4.425× = ~71s, overstating the full-fixture fixed overhead (still ~16s). The true full-fixture fast mode is ~2.6 min based on per-frame rates.

## Full Milestone Journey Summary (M0→M6)

| Milestone | Standard Mode | Fast Mode | Key Change |
|-----------|--------------|-----------|-----------|
| M0 | ~183 min | — | Baseline (CPU fallback, S1 pipeline) |
| M1 | ~117 min | — | S2 ESRGAN + S3 faster upscale |
| M2 | 82.5 min | — | S6 frame_stride=6 (300 frames) |
| M3 | 82.5 min | — | K-segment infra (no speed gain: GPU serializes) |
| M4 | 82.5 min | 7.6 min | Fast mode (SAM2-tiny + BicubicRestorer) |
| M5 | 36.5 min | 4.5 min | prop_stride=5 (std) / prop_stride=24 (fast) |
| M6 | **22.5 min** | **4.7 min** | K=4 parallel upscale+analyze wired with prop_stride=5 |

The M6 fix (wiring prop_stride through the parallel path) enables the K=4 upscale+analyze to run at prop_stride=5 instead of defaulting to prop_stride=1, unlocking 3.9× speedup for the upscale+analyze phase that was not accessible before.

## Target Status

| Mode | Target | Linear projection | Adjusted projection | Status |
|------|--------|-------------------|---------------------|--------|
| Standard (K=4, prop_stride=5) | ≤22 min | 22.5 min | 20.9 min | MET (adjusted) |
| Fast (prop_stride=30) | ≤2.2 min | 4.7 min | 3.8 min | NOT MET (linear/adjusted) |

Fast mode requires either: (1) prop_stride > 50 (very sparse), or (2) K=4 segment parallelism for fast mode too, or (3) accepting that the linear scale factor overestimates by ~1.5-2× for very short smokes.
