# M2 — Stride-N sampling + async analyzer fanout (S11 + S4 + S5 + S7 + S8)

## Purpose

Measure the post-M2 pipeline end-to-end on **DGX Spark** (NVIDIA GB10) against
M1. M2 attacks the Gemini-sync bottleneck confirmed by M1 (98% of upscale stage
wall = Gemini round-trips) via two orthogonal levers:

- **S11** — stride-N frame sampling: auto-compute stride from fps (`max(1,
  round(fps/10))` → stride=6 at 60fps). SAM2 still propagates all frames for
  mask continuity; only output filtering is strided. Cuts window count from
  119 → 20 (~6x fewer Gemini calls).
- **S4** — async analyzer fanout: replace the M1 `queue.Queue` + sync consumer
  thread with an `asyncio`-capable consumer using `analyze_sequence_async` and
  `client.aio.models.generate_content()`. Consumer thread owns its own event
  loop; producer is one window ahead via `gemini_max_inflight=1` queue.
- **S5** — smarter upscale target: pre-scale crops to max 288px long edge
  before ESRGAN x4 → output ~1152px → resize to `target_size=768` (~3.5x
  fewer ESRGAN FLOPs vs M1's 1024 target).
- **S7** — SAM2 throughput on CUDA: `offload_video_to_cpu=False`,
  `offload_state_to_cpu=False` on CUDA; GB10's 128 GB unified memory removes
  the need to page tensors to system RAM.
- **S8** — RTMPose on CUDA EP: auto-select `onnxruntime` CUDA execution
  provider via `torch.cuda.is_available()` (was hardcoded CPU on all platforms
  before M2).

## Hardware & environment

- **Host**: gx10 (DGX Spark, hostname `gx10-8eb2`)
- **GPU**: NVIDIA GB10 (Grace Blackwell), sm_121, 128 GB unified LPDDR5x
- **Python**: 3.12.3 (venv `~/bjj/.venv-spark`)
- **Key libs**: torch 2.12.0.dev20260407+cu128, SAM-2, spandrel 0.4.2,
  rtmlib 0.0.15, onnxruntime 1.26.0 (CUDA EP), google-genai 2.3.0, opencv 4.13

## M2 commit chain (one per S-item, all revertable independently)

```
912005d m2(s11): stride-N frame sampling
0e37ec0 m2(s4): async analyzer fanout
17dd53b m2(s7): SAM2 throughput on CUDA
765a51d m2(s8): RTMPose CUDA on Spark
```

Note: S5 pre-scale code (worker `_flush_batch`) landed in the s11 commit since
both touched `worker.py` simultaneously. S5 config (`upscale_target_size=768`)
landed in the s4 commit with `gemini_max_inflight`. This is documented in
the commit messages.

## TDD verification

`tests/test_stride_sampling.py` (new in S11, 7 tests):

- `test_stride_1_writes_all_frames` — stride=1 preserves all frames.
- `test_stride_6_writes_every_6th_frame` — stride=6 output is exactly frames
  0, 6, 12, … (real indices, not renumbered).
- `test_stride_3_on_30fps_source` — stride=3 at 30fps source.
- `test_stride_larger_than_range_keeps_only_frame0` — degenerate case.
- `test_non_zero_start_frame_with_stride` — start_frame offset preserved.
- `test_auto_stride_from_fps` — asserts 60fps→6, 30fps→3, 24fps→2, 5fps→1,
  120fps→12.
- `test_stride_reduces_tracking_json_frame_count` — integration test with mock
  tracking JSON output.

Full pytest suite: **203 passed / 0 failed**.
(`test_enhance_batch_mixed_size_semantic` flaky but pre-existing since M1 —
ordering-dependent GPU state, not an M2 regression.)

## Smoke test (30 s segment, 1798 frames) — COMPLETED

Job `3ca29ea0-af4b-47d4-8271-ddf44fd472d4`. Wall start 2026-05-18T01:31:24Z,
completed 01:52:58Z. **Total wall: 1294 s ≈ 21 min 34 s.**

| stage | wall | per-frame | M1 | delta |
|---|---:|---:|---:|---:|
| DOWNLOAD + DETECT | 5 s | — | 5 s | 0 |
| TRACK (1798 frames) | 1021 s (17.0 min) | 568 ms/frame | 562.8 ms/frame | +0.9% |
| UPSCALE + ANALYZE | 230 s (3.8 min) | 0.77 s/strided-frame | 3.03 s/frame | **−74.6%** |
| — frames processed | 300 (stride=6) | — | 1798 | **−83.3%** |
| — Gemini windows | 20 | 11.5 s/window | 119 / 45.8 s | **−83.2% / −74.9%** |
| ANNOTATE / UPLOAD / PUBLISH | 35 s | — | 45 s | −22% |
| **Total smoke** | **1294 s ≈ 21 min 34 s** | — | 6546 s ≈ 109 min | **−80.2% (5.1x)** |

20 analyzer windows completed. **Zero errors** — 20/20 with clips.

### Feature confirmations from logs

| S-item | Log evidence |
|---|---|
| S11 stride | `frame_stride=6` in worker log; 300/1798 frames in output JSON |
| S4 async | `Analyzing window X async` for all 20 windows |
| S5 pre-scale | `prescale_max_input_edge=288`, `target_size=768` in flush log |
| S7 SAM2 offload | `[sam2] CUDA detected: offload_video_to_cpu=False, offload_state_to_cpu=False (S7)` |
| S8 RTMPose | `[pose] Loading RTMPose on cuda (force_cpu=False)` |

## Upscale+Analyze speedup breakdown

M1 upscale+analyze wall: **5452 s** (119 windows × 45.8 s/window avg).  
M2 upscale+analyze wall: **230 s** (20 windows × 11.5 s/window avg).

- **Window count reduction**: 119 → 20 = **5.95x** (from S11 stride=6)
- **Per-window time reduction**: 45.8 s → 11.5 s = **4.0x** (from S4 async +
  S5 smaller crops → faster ESRGAN → producer finishes faster, less queue
  blocking; Gemini API latency is the dominant component but async pipelining
  avoids blocking the producer thread)
- **Combined**: 5452 s → 230 s = **23.7x**

## M2 done criterion

**CRITERION: smoke end-to-end ≤ 22 min on gx10** — **MET** (21 min 34 s,
margin 0.43 min).

## Full-fixture projection (linear scale from smoke)

Scale factor: 7956 / 1798 = **4.425x**

| stage | M0 | M1 | M2 |
|---|---:|---:|---:|
| TRACK | 75.4 min | 74.7 min | **75.4 min** |
| UPSCALE + ANALYZE | 432.5 min | 402.2 min | **16.9 min** |
| other | 1 min | 1 min | **1 min** |
| **TOTAL** | **~509 min (8.5 h)** | **~478 min (7.97 h)** | **~93 min (1.55 h)** |

vs. 60-min full-fixture target: **M2 = 1.55×** over budget (M0 was 8.5×,
M1 was 7.97×).

**Full-fixture 60-min criterion: NOT MET** (~93 min projected). The tracking
stage (~75 min) now dominates at ~80% of wall time. The upscale+analyze stage
is no longer the bottleneck.

## Bottleneck analysis after M2

| stage | M2 projected (full fixture) | % of total |
|---|---:|---:|
| Tracking | 75.4 min | 80.8% |
| Upscale + Analyze | 16.9 min | 18.1% |
| Other | 1 min | 1.1% |

**Tracking is now the sole bottleneck.** The rate of 568 ms/frame is unchanged
from M0 and M1 — M2's S7 (SAM2 CUDA offload) and S8 (RTMPose CUDA EP) were
expected to help here but the smoke shows no regression; gains (if any) are
within measurement noise at the 1-frame level.

Tracking breakdown estimate (from M0 profiling):
- SAM2 hiera-base-plus propagation: ~400–450 ms/frame
- RTMPose + DINOv2 + JSON I/O overhead: ~120 ms/frame

**M3 must attack tracking throughput.** Options:

1. **S9 — multi-job parallelism**: run 2–4 concurrent tracking jobs on gx10.
   Memory headroom is confirmed (M1 peak: 89 GB / 128 GB available). With 2
   concurrent jobs the full-fixture tracking time halves to ~38 min; with 4
   jobs to ~19 min — well inside the 60-min total target.
2. **S12 — SAM2 skip propagation** (optional, lower priority): only propagate
   every Nth frame at the SAM2 level (currently SAM2 propagates all frames
   even though only strided frames are output). Trade-off: reduced mask
   continuity at batch boundaries. Requires careful testing.

M3 recommended path: **S9 with 2 concurrent jobs** (simple, safe, no model
quality impact, sufficient to hit 60-min target at 2x parallelism).

## Artifact inventory

- `README.md` — this file.
- `timing.json` — machine-readable per-stage numbers.

(Smoke log artifacts on gx10: `~/bjj/m2-artifacts/smoke/` — service.log,
lifecycle.jsonl, track_request.json, job_id.txt.)
