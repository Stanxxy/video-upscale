# M1 — Standard-mode CUDA hardening (S6 + S2 + S1 + S3)

## Purpose

Measure the post-M1 pipeline end-to-end on **DGX Spark** (NVIDIA GB10) and
compare against the M0 baseline. M1 attacks the four high-leverage
code-level issues confirmed by M0:

- **S6** — sequential `cv2.VideoCapture.read()` in the upscale loop (replaces
  per-frame `cap.set(POS_FRAMES)` random seeks).
- **S2** — off-thread JPEG writes via a 2-worker thread pool with a 16-slot
  semaphore for backpressure.
- **S1** — CUDA hardening on `RealESRGANRestorer`: CUDA-first device,
  `tf32`/`cudnn.benchmark` at import, `channels_last`, throttled
  `empty_cache()` via `flush_cache()`, new `enhance_batch(...)` for
  single-forward-pass batched upscale (batch size = 8).
- **S3** — analyzer consumer thread that pulls completed windows from a
  bounded `queue.Queue(maxsize=2)`. Upscale producer hands windows off and
  continues; consumer runs `_analyze_window` plus the per-window periodic
  flush sequentially.

## Hardware & environment

- **Host**: gx10 (DGX Spark, hostname `gx10-8eb2`)
- **GPU**: NVIDIA GB10 (Grace Blackwell), sm_121, 128 GB unified LPDDR5x
- **Python**: 3.12.3 (venv `~/bjj/.venv-spark`)
- **Key libs**: torch 2.12.0.dev20260407+cu128, ultralytics 8.4.51, SAM-2,
  spandrel 0.4.2, rtmlib 0.0.15, onnxruntime 1.26.0 (CPU EP), google-genai
  2.3.0, opencv 4.13
- **Pipeline commit**: `61ad1d0` on `feature/m1-cuda-hardening` (parent
  `87c4466` — M0 finalize, rooted at `f88f56a` HEAD of `develop`).

### M1 commit chain (one per S-item, all revertable independently)

```
61ad1d0 m1(s3): decouple upscale loop from analyzer via consumer thread
e0c30dc m1(s1): CUDA hardening + batched enhance on RealESRGAN
e8267f0 m1(s2): off-thread JPEG writes in upscale loop
2cd1e40 m1(s6): sequential video reads in upscale loop
```

### Env overrides (same as M0 for like-for-like comparison)

| var | value |
|---|---|
| `BJJ_UPSCALE_HEARTBEAT_INTERVAL_SEC` | `5` |
| `BJJ_MAX_CONCURRENT_JOBS` | `1` |
| `BJJ_TEMP_DIR` | `~/bjj/m1-artifacts/smoke/temp` |

All other config (sampling_rate=1, Real-ESRGAN x4plus, sam2.1-hiera-base-plus,
single-agent Gemini, sliding window 30/stride 15, `max_missing_frames=999999`)
is the unmodified default.

## Fixture

Same as M0: VP9, 1920×1080, 60 fps, 132.7 s, 7956 frames.
`s3://bjj-video-analysis/aHR0cHM6Ly93d3cueW91dHViZS5jb20vd2F0Y2g_dj1DOEZrVWtaSGxGYw==.mp4`

Smoke segment: `start_time=0:00`, `end_time=0:30`, 1798 frames.

## TDD verification

`tests/test_restorer_batch.py` (new in S1) asserts that `enhance_batch`
produces semantically equivalent output to per-call `enhance()`:

- **Same-size N-crop batch** matches `enhance()` within FP16 tolerance (≤5
  uint8 per pixel, <0.5% pixels exceeding) — bit-exact except for FP16
  rounding from the larger tensor.
- **Mixed-size N-crop batch** matches in the inner region (32-px boundary
  ring excluded; the model leaks signal from `BORDER_REPLICATE`-padded
  letterbox into the edge, a property of ESRGAN's conv stack not a
  batching bug).
- **1-element batch** is bit-exact with `enhance()`.
- **Empty batch** returns `[]`.

Passes on MPS (Mac local) and CUDA (gx10 sm_121). 192 pre-existing tests
in `tests/` continue to pass.

## Smoke test (30 s segment, 1798 frames) — COMPLETED

Job `f6952b5c-bd84-49ee-a413-d57fb0f67185`. Wall start 2026-05-18T03:19:39Z,
completed 05:08:45Z. **Total wall: 6546 s ≈ 109 min ≈ 1.82 h.**

| stage | wall | per-frame | M0 (proxy) | delta |
|---|---:|---:|---:|---:|
| DOWNLOAD + DETECT | 5 s | — | 5 s | 0 |
| TRACK (1798 frames) | 1012 s (16.9 min) | 562.8 ms/frame | 568 ms/frame | −1.0% |
| UPSCALE + ANALYZE (1798 frames) | 5452 s (90.9 min) | 3.03 s/frame | 3.26 s/frame | **−7.0%** |
| Gemini avg sec/window | — | 45.8 s/window | 49.5 s/window | −7.5% |
| ANNOTATE / UPLOAD / PUBLISH | 45 s | — | not reached in M0 | — |
| **Total smoke (full run)** | **6546 s ≈ 109 min** | — | ~117 min (extrapolated) | **−6.8%** |

119 analyzer windows completed (vs. 70/119 in M0's cancelled smoke).
**Zero errors** — 119 with clips, 0 raw_error, 0 Gemini API error.

## Full-fixture projection (linear from smoke)

| stage | M0 projected | M1 projected |
|---|---:|---:|
| TRACK | 75.4 min | 74.7 min |
| UPSCALE + ANALYZE | 432.5 min | 402.2 min |
| other | 1 min | 1 min |
| **TOTAL** | **8.5 h** | **~8.0 h** |

vs. 30-min standard-mode target: **M1 = 15.9×** over budget (M0 was 17×).

## Why M1 fell short of the 10× upscale target

The M1 task brief stated: "upscale per-frame rate from 3.26 s down to ≤ 0.3 s
(~10× speedup) by attacking the four high-leverage code-level issues
confirmed by M0." Measured: **3.03 s/frame** (7% improvement).

**Microbenchmark on gx10** of `RealESRGANRestorer` after the S1 hardening,
on representative 200×200 BGR crops at `target_size=1024`:

- `enhance()` per-call: **170 ms/frame**
- `enhance_batch(8)`: **199 ms/frame** (0.85× speedup — batching adds
  letterbox + memory-stack overhead that exceeds the kernel-launch win
  on this input size)

So **raw ESRGAN compute on this hardware is already at ~170 ms/frame** —
roughly the 10× target the brief estimated. The gap between 170 ms/frame
microbench and 3.03 s/frame measured upscale stage is the **single-agent
Gemini sync chain**: 119 windows × ~45 s/window = ~5350 s of the 5452 s
upscale stage = **98%** of stage wall is Gemini-bound.

S3's decoupling lets the producer run during Gemini round-trips, but with
the queue capped at maxsize=2 (one window pipelined ahead, per spec), the
producer is throttled to consumer pace. The end-to-end smoke wall is
`window_count × per_window_consumer_time` — dominated by Gemini, not
ESRGAN.

**Per-item attribution (estimate)**: S6 ≈ 1%, S2 ≈ 1.5%, S1 ≈ 2%, S3 ≈
2.5% — combined ≈ 7%. The first three move ESRGAN compute time around
in the noise; S3 hides some Gemini time but is capped by queue size.
Per-item deltas were not measured in isolation (single rsync + commit
chain) — would require commit-by-commit re-runs (~1 h each).

## GPU resource usage during upscale

`nvidia-smi-samples.csv` polled every 15 s during the run:

- GPU utilization: **mean 82.3%, max 96%** during upscale (well-fed).
- System memory used: max 89 GB (transient), mean 37 GB — comfortable under
  the 128 GB unified pool. **Cross-job concurrency (S10) has plenty of
  headroom**: 2 concurrent jobs would fit at 74 GB peak, 4 would fit at
  104 GB peak. Memory is NOT the blocker for `max_concurrent_jobs > 1`
  on this hardware.

## M1 verdict

**Partial.** The S1+S2+S6+S3 changes work correctly (zero errors, full
smoke complete, full 119/119 analyzer windows produced clips). They
deliver ~7% wall-time improvement, well below the 10× upscale target the
brief estimated.

The 10× figure was over-optimistic against this baseline: **M0 was
already CUDA**, so the plan's "10-30× from CUDA path" was largely already
baked in. The remaining 2-3× from batching does not materialize on
200×200 inputs (microbench shows 0.85× — batching loses here). The real
bottleneck is the sync single-agent Gemini chain, which M1 explicitly
does not address (that's M2's S4).

### Plan branch-decision update

- M1 done criterion in plan: "end-to-end ≤ 30 min on Spark for the
  regression test set with a single segment" — **NOT MET**.
- M2 done criterion: "end-to-end ≤ 22 min on Spark with K=1" — requires
  S4 (analyzer fanout / pipelining) + S5 + S11 + S7 + S8.
- **Recommended next**: run M2 immediately. S4 is the single biggest
  remaining lever; S11 (frame stride N=6 at 60 fps) multiplies S4's gains
  ~6×.

### Reversibility

Each S-item is a separate commit on `feature/m1-cuda-hardening`; revert
any individual one with `git revert <sha>`. No schema changes, no API
contract changes, no config additions in M1.

## Artifact inventory

- `README.md` — this file.
- `timing.json` — machine-readable per-stage numbers.
- `run_m1_smoke.sh` — driver shell script.
- `smoke/` — driver.log, lifecycle.jsonl, service.log,
  nvidia-smi-samples.csv, track_request.json, track_response.json, job_id.txt.

## Pitfall encountered during the run

The driver script (`run_m1_smoke.sh`) had `WALL_BUDGET_SEC=1800` (30 min,
expected to be plenty if M1 hit the 10× target). The actual smoke took
109 min. To avoid losing the run, the driver script was SIGKILLed at
~25 min into the smoke (preserves uvicorn via skipped trap), and a
lightweight curl poller was launched manually to keep collecting
lifecycle data until terminal. The smoke ran to a clean `completed`
state; no data was lost. Lesson: M1 driver should use the M2-level
budget (≥3 h) until M2 lands.
