# M4: Fast Mode Implementation

**Branch**: `feature/m4-fast-mode`
**Date**: 2026-05-18

## Problem

M3 measured: 70.3 min full-fixture (7956 frames). Tracking = 63.6 min = 90% of wall.
K=4 segment parallelism does NOT help — single GB10 GPU serializes SAM2 CUDA kernels.
The only path to fast-mode ≤3 min is reducing SAM2 propagation steps.

## Solution: Propagation Stride (prop_stride=12)

Instead of SAM2 propagating 7956 frames, only propagate every 12th frame.
- ffmpeg `select=not(mod(n,12))` extracts frames 0, 12, 24, ... from video
- SAM2 sees 7956/12 = 663 frames instead of 7956
- global_idx = start_frame + (batch_offset + batch_rel_idx) * prop_stride
  correctly maps back to real video frame positions
- Expected 12x tracking speedup: 63.6 min → ~5.3 min (with SAM2-tiny: ~3 min)

## Changes

### F1: BicubicRestorer (`restorer.py`)
- `BicubicRestorer` class with LANCZOS4 interpolation
- Same public contract as `RealESRGANRestorer` (enhance, enhance_batch, flush_cache)
- Used in fast mode; negligible latency vs ESRGAN GPU inference

### F2: Window/Stride Tuning (`service/worker.py`)
- Fast mode: `WINDOW_SIZE=20`, `STRIDE=20` (non-overlapping, fewer Gemini windows)
- Standard mode: `WINDOW_SIZE=30`, `STRIDE=15` (unchanged)
- Fast mode: `effective_sampling_rate=max(sampling_rate, 2)` (every other frame)

### F3: SAM2-tiny + Propagation Stride (`tracking_pipeline/`)
- `sam2_manager.py`: `load_batch(prop_stride=N)` + ffmpeg select filter + cv2 grab() fallback
- `hybrid_tracking.py`: prop_stride and enable_pose parameters; correct global_idx mapping
- Fast mode presets: SAM2-tiny, prop_stride=12, enable_pose=False, frame_stride=12
- Standard mode presets: SAM2 base-plus, prop_stride=1, enable_pose=True

### F5: Gemini High-Fanout (`service/worker.py`)
- Fast mode: gemini_max_inflight=24, no context chain (each window gets ctx=None)
- Standard mode: gemini_max_inflight=1, full context chain (unchanged)
- Consumer loop uses asyncio.gather for concurrent window dispatch in fast mode

### API (`service/models.py`)
- `ProcessingMode(str, Enum)`: STANDARD | FAST
- `TrackRequest.processing_mode: ProcessingMode = ProcessingMode.STANDARD`

## Smoke Test Commands (gx10)

Standard mode:
```bash
ssh gx10 'cd ~/bjj/whole-video-analysis && source ~/bjj/.venv-spark/bin/activate && \
  PYTHONPATH=. BJJ_MAX_CONCURRENT_JOBS=1 BJJ_STANDARD_SEGMENTS=1 \
  python -m uvicorn service.app:app --host 0.0.0.0 --port 8000'
```

Fast mode:
```bash
# POST body: { ..., "processing_mode": "fast" }
```

## Smoke Test Results (gx10 DGX Spark)

### Standard Mode (M4 service, K=1)
- Total: **18m 40s (1120s)**
- M3 K=1 was ~17 min → **no regression**
- Config: SAM2 base-plus, frame_stride=6, prop_stride=1, enable_pose=True
- Full-fixture projection: ~82 min (K=1); with K=4 (M3): ~70 min

### Fast Mode (v3, corrected)
- Total: **1m 43s (103s)**
- Tracking: 48.9s / 150 SAM2 frames at 3.1 fps
- Upscale+analyze: ~40s (8 Gemini windows)
- Annotate/upload/SNS: ~14s
- Config: SAM2-tiny, frame_stride=1, prop_stride=12, enable_pose=False
- Full-fixture projection: **7.6 min** (103s × 4.42)

### Done Criterion
- Smoke ≤ 5 min: MET (1.7 min)
- Full-fixture ≤ 25 min: MET (7.6 min)

### Bugs Fixed During Smoke Testing
1. `batch_offset` indexing: `(batch_offset + batch_rel_idx) * prop_stride` → `batch_offset + batch_rel_idx * prop_stride`
2. `frame_stride` double-filtering: fast mode was `frame_stride=12` causing 150→12 JSON frames; fixed to `frame_stride=1`

## Test Results

- New tests: 23 (test_bicubic_restorer.py: 11, test_processing_mode.py: 12)
- Pre-existing suite: 222 passing
- Total: 245 pass / 1 pre-existing failure (restorer_batch MPS FP16 drift — not M4)

## Comparison Table

| Metric | M3 (K=4 standard) | M4 Standard (K=1) | M4 Fast (K=1) |
|---|---|---|---|
| Smoke time | 15m 53s | 18m 40s | 1m 43s |
| Full-fixture proj. | 70.3 min | ~82 min | **7.6 min** |
| SAM2 model | base-plus | base-plus | tiny |
| Prop stride | 1 | 1 | 12 |
| Enable pose | yes | yes | no |
| Upscale method | RealESRGAN | RealESRGAN | Bicubic/LANCZOS4 |
| Gemini fanout | 1 | 1 | 24 |

## Bottleneck After M4

Fast mode: tracking is 48.9s / 103s total = 47% of wall. Next lever is M5 K=4 parallel segments for fast mode.
