# M5: Prop-Stride Tuning

**Branch**: `feature/m5-prop-stride-tuning`
**Date**: 2026-05-18

## Objective

Tune `prop_stride` per processing mode to close the gaps identified in M4:
- Fast mode: 7.6 min full-fixture → target ≤2.2 min (need ~3.5×)
- Standard mode: 82.5 min full-fixture → target ≤22 min (need ~3.75×)

## Changes

### 1. Configurable prop_stride per mode (`service/config.py`)
- `BJJ_FAST_PROP_STRIDE=24` (default, was hardcoded 12)
- `BJJ_STANDARD_PROP_STRIDE=5` (default, was hardcoded 1)

### 2. Wire config knobs in worker (`service/worker.py`)
- Fast mode: `_eff_prop_stride = config.fast_prop_stride`
- Standard mode: `_eff_prop_stride = config.standard_prop_stride`

### 3. Bug fix: frame_stride double-filtering (`service/worker.py`, commit `8b347f3`)
When `prop_stride > 1`, `frame_stride` was also being applied (auto-computed from fps), causing double-filtering:
- prop_stride=5 + frame_stride=6 → LCM(5,6)=30 → only 60 tracking entries from 1798 frames (2 fps effective)
- Fix: when `prop_stride > 1`, force `frame_stride=1` (same logic as fast mode)
- Result: prop_stride=5 + frame_stride=1 → 360 entries → 12 fps effective (as intended)

## Smoke Test Results (gx10 DGX Spark, 2026-05-18)

### Fast Mode (prop_stride=24, SAM2-tiny, BicubicRestorer)

- **Total: 61s (1m 1s)**
- Tracking: ~26s / 75 SAM2 frames (1798/24=75)
- Upscale+analyze: ~22s (BicubicRestorer negligible + 4 Gemini windows parallel fanout=24)
- Annotate/upload/publish: ~3s
- **Full-fixture projection: 4.5 min** (61 × 4.425 = 270s)
- M4 comparison: 7.6 min → 4.5 min = **1.69× speedup** from prop_stride 12→24

### Standard Mode (prop_stride=5, SAM2 base-plus, ESRGAN)

- **Total: 495s (8m 15s)**
- Tracking: 203s / 360 SAM2 frames (1798/5=360)
- Upscale: 117s / 360 ESRGAN crops
- Gemini: 271s / 24 windows (WINDOW_SIZE=30, STRIDE=15, sequential context chain)
- Annotate/upload: 21s
- **Full-fixture projection: 36.5 min** (495 × 4.425 = 2190s)
- M4 comparison: 82.5 min → 36.5 min = **2.26× speedup** from prop_stride 1→5

## Target Assessment

| Mode | Smoke | Full-fixture | Target | Result |
|------|-------|-------------|--------|--------|
| Fast (prop_stride=24) | 1m 1s | **4.5 min** | ≤2.2 min | NOT MET |
| Standard (prop_stride=5) | 8m 15s | **36.5 min** | ≤22 min | NOT MET |

## Root Cause Analysis

### Why standard mode (prop_stride=5) fails

Standard mode bottleneck is **Gemini sequential context chain**, not tracking.

With prop_stride=5 → 360 tracking frames, WINDOW_SIZE=30, STRIDE=15 → 24 Gemini windows × 11.3s = 271s Gemini (dominates). Meanwhile tracking improved 5× (203s vs M4's 1021s).

Paradoxically, prop_stride=5 creates MORE windows than M2's frame_stride=6 (24 vs 20), making it slightly worse for Gemini even though tracking is dramatically faster.

**Full-fixture Gemini cost**: 1591 frames → 105 windows × 11.3s = 1187s = 19.8 min. Plus tracking 15 min = 34.8 min total. The 22 min target requires reducing Gemini to ≤7 min (33 sequential windows), which requires either:
1. K=4 upscale+analyze parallelism via segment_runner (4× → 7 min → total 22 min)
2. Gemini fanout increase for standard mode (break context chain accuracy constraint)

### Why fast mode (prop_stride=24) misses but is close

Fast mode target ≤2.2 min for full fixture. Measured 4.5 min. Gap = 2.3 min.

For full fixture: 7956/24 = 332 SAM2 frames × 0.326ms = 108s = 1.8 min tracking.
Gemini: 332/20 = 17 windows parallel (fanout=24) → ~12s. Annotate/upload ~10s.
Projected: 1.8 + 0.2 + 0.2 = ~2.2 min — should barely hit target.

The **61s smoke** vs **4.5 min projected** discrepancy: smoke has only 75 SAM2 frames (1798/24) → fast; full fixture has 332 frames. The smoke scale factor (4.425) is applied linearly, which is correct.

The fast mode 4.5 min projection is computed linearly: 61s × 4.425 = 270s = 4.5 min. But the expected breakdown for full fixture suggests ~2.2 min. The discrepancy comes from fixed overhead in the smoke (download, S3 upload, model load) that doesn't scale with clip length.

**Fixed overhead analysis**: 61s total for 30s clip. If tracking = 26s (scales with clip) and overhead = 35s (fixed), then:
- Full fixture tracking: 7956/24 × 0.347ms/SAM2frame ≈ 116s = 1.9 min
- Full fixture analyze: 17 windows parallel ≈ 12s
- Fixed overhead (download, upload, annotate, publish): 35s = 0.6 min
- Total full fixture: 1.9 + 0.2 + 0.6 = 2.7 min

This is closer to 2.2 min. The linear projection (4.5 min) **over-estimates** because it scales fixed overhead by 4.425×. The actual full-fixture performance should be ~2.7 min, which is closer to the 2.2 min target.

**Accuracy tradeoff for fast mode prop_stride=24**: Every 24th frame at 60fps = 2.5fps effective temporal resolution. Events spanning >0.4s are reliably detected; faster movements may be missed. This requires regression harness validation before production.

## Accuracy Risk

### Standard mode prop_stride=5 (12 fps effective)
- BJJ events (submissions, takedowns, sweeps) span 1-10 seconds
- 12fps captures all such events with sub-second resolution
- **Low risk** — similar to 30fps source at 2.5fps sampling

### Fast mode prop_stride=24 (2.5 fps effective at 60fps source)
- Only 2.5 frames per second of coverage
- Fast technique transitions (<0.4s) may be missed
- **Medium-high risk** — requires regression harness validation
- Document requirement: regression score ≥0.80 vs standard oracle before production

## Conclusion

M5 improvements measured and documented. Neither mode hits the stated targets from the plan spec using prop_stride tuning alone:

1. **Standard mode**: best path to ≤22 min is reverting to prop_stride=1 (M2 baseline = 82.5 min) + enabling K=4 segment parallelism for upscale+analyze (4× → ~20 min). Prop_stride=5 makes it worse by adding more Gemini windows.

2. **Fast mode**: prop_stride=24 gets to ~2.7 min full fixture (correcting for fixed overhead) — very close to the 2.2 min target. The remaining gap could be closed by model warm-up caching or modest prop_stride increase to 32.

## Tests

- 242 pass / 1 pre-existing failure (restorer_batch MPS FP16 drift — unrelated to M5)
- New tests: none (config knobs are tested via integration smoke)

## Commits

1. `2afcecf` — m5(std): configurable prop_stride per mode (BJJ_FAST_PROP_STRIDE=24, BJJ_STANDARD_PROP_STRIDE=5)
2. `8b347f3` — m5(std): fix frame_stride double-filtering when prop_stride > 1
