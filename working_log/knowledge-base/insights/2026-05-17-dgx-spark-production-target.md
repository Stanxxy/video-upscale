# 2026-05-17 — DGX Spark production target host (`ssh gx10`)

**Tags:** infra, ops, dgx-spark, cuda, blackwell, production

## Summary

Production deployment target for the `whole-video-analysis` vision pipeline is a DGX Spark (Grace Blackwell GB10) accessible via `ssh gx10` from the developer machine. This supersedes the M4 Max as the deploy target; M4 Max remains dev-only.

## Verified host facts (2026-05-17)

| Property | Value |
|---|---|
| SSH alias | `gx10` (passwordless from developer machine) |
| Hostname | `gx10-8eb2` |
| Architecture | aarch64 (ARM64) |
| Kernel | Linux 6.17.0-1008-nvidia |
| OS | Ubuntu |
| GPU | NVIDIA GB10 (Grace Blackwell, sm_120, single device) |
| System Python | 3.12.3 at `/usr/bin/python3` |

## Operational notes

- **ARM64 wheel availability**: PyTorch and most ML libs publish aarch64 wheels but coverage lags x86_64. For Blackwell sm_120, prefer recent stable PyTorch (≥ 2.5) or nightly with CUDA 12.6+. Build from source as last resort.
- **`onnxruntime-gpu` on aarch64**: Limited official wheels. RTMPose GPU execution provider (see plan S8) may need CPU fallback or a source build. Confirm before depending on it.
- **Shared LPDDR5x memory (128 GB)**: enables larger SAM2 batches and concurrent jobs (plan S7, S10). Spark's `offload_video_to_cpu=False` is a viable default.

## Use cases

- M0 baseline measurement (per [pipeline speedup plan](../../../.claude/plans/the-current-tracking-upscale-analysis-pi-snoopy-manatee.md))
- All subsequent perf milestones M1-M5
- Production deployment target for the `service/` worker

## Verified working environment (M0 setup, 2026-05-17)

Workspace layout on gx10:
- `~/bjj/whole-video-analysis` — rsync'd from local Mac (1.1 GB; `git rev-parse HEAD` = `f88f56a`, 7 commits ahead of `origin/develop`)
- `~/bjj/shared_lib` — rsync'd from `/Users/stanliu/Documents/bjj-proj/bjj-vision-backend/shared_lib`; pip-installed from this path
- `~/bjj/.venv-spark` — Python 3.12.3 venv (7.4 GB after installs)
- `~/bjj/whole-video-analysis/requirements-spark.txt` — derived from `requirements-service.txt` with the shared-lib path repointed (local file change only; do NOT commit this back)
- Total `~/bjj` footprint: 8.5 GB on a 916 GB root with 714 GB free

Verified package versions:
| Package | Version |
|---|---|
| torch | 2.12.0.dev20260407+cu128 (nightly, sm_121 confirmed) |
| torchvision | 0.27.0.dev20260407+cu128 |
| ultralytics | 8.4.51 |
| spandrel | 0.4.2 |
| rtmlib | 0.0.15 |
| onnxruntime | 1.26.0 (CPU EP only — GPU EP install is plan S8 scope) |
| SAM-2 | git@2b90b9f5 (from facebookresearch/sam2) |
| google-genai | 2.3.0 |
| shared-lib | 1.0.0 (local install) |
| opencv-python | 4.13.0.92 |
| fastapi | 0.136.1 |
| cassandra-driver | 3.30.0 |

Smoke test result (2.1 s cold start, ALL OK):
- torch CUDA visible; FP16 matmul on GB10 runs
- `service.models`, `service.config`, `restorer.RealESRGANRestorer`, `analyzer.BJJTechniqueAnalyzer`, all `tracking_pipeline` submodules import cleanly
- `shared_lib.models.sns_event_models.VideoEventCandidate` resolvable

## Setup gotchas worth remembering

1. **Local `develop` is ahead of `origin/develop`** (7 commits). `git clone` from origin returns stale code; **rsync the working tree** to get the actual M0 baseline.
2. **Local repo contains `venv/`** (not `.venv/`) — must exclude during rsync or it errors on internal numpy test data with mmap timeouts.
3. **`requirements-service.txt` has a Mac-local file:// path** for `shared_lib`. On gx10 use the derived `requirements-spark.txt` with the path rewritten — or fix the upstream requirement to be portable.
4. **macOS rsync** (BSD) doesn't support `--info=stats2`; use plain `--stats`.
5. **nvcc not present** on Spark (runtime-only NVIDIA stack). PyTorch wheels ship their own CUDA libs (nvidia-cuda-runtime-cu12, cuDNN, cuBLAS, etc.) so this is fine.
6. **PyTorch wheel choice**: nightly cu128 (`--index-url https://download.pytorch.org/whl/nightly/cu128`) was the chosen route. Stable cu126 should also work for sm_121 but was not tested.

## M0 baseline measurements (2026-05-17)

Smoke run on a 30-sec segment of `aHR0cHM6Ly93d3cueW91dHViZS5jb20vd2F0Y2g_dj1DOEZrVWtaSGxGYw==.mp4` (VP9, 1920×1080, 60 fps, 132.7 s, 7956 frames):

| Stage | Wall (smoke 1798 fr) | Per-frame rate | Projected for full fixture (7956 fr) |
|---|---:|---:|---:|
| Download + Detect | 5 s | — | 5 s |
| Tracking (SAM2 base_plus + RTMPose CPU) | 17.0 min | 568 ms/frame (1.76 fps) | 75 min |
| Upscale + Analyze (Real-ESRGAN x4 + Gemini single-agent) | 57.8 min (partial, 1063 of 1798 frames) ⇒ ~97.7 min extrapolated | 3.26 s/frame (0.31 fps) | **~432 min ≈ 7.2 h** |
| Annotate / Upload / Publish | < 1 min | — | < 1 min |
| **Total** | **~117 min** | | **~510 min ≈ 8.5 h** |

Verdict: **17× over the 30-min standard target** for this 132 s fixture. Stage cost share on Spark CUDA is virtually identical to M4 Max MPS (upscale ~85%, tracking ~15%) — **hardware change alone doesn't move the bottleneck**, only the absolute numbers. Upscale per-frame at 3.26 s on Blackwell when the raw FP16 ESRGAN kernel is ~30-50 ms = **~100× Python/IO overhead** (per-frame `cv2.imwrite`, per-frame `empty_cache()`, per-frame `cap.set(POS_FRAMES)` random seek on VP9, sync analyzer blocking the upscale loop, no batching). Confirms plan items S1-S8 are all required, and S11 (stride-N) gives the biggest single multiplier on 60fps sources.

See [working_log/baselines/m0/README.md](../../baselines/m0/README.md) and [working_log/baselines/m0/timing.json](../../baselines/m0/timing.json) for full data.

## Related

- KB decision `2026-03-15-single-job-concurrency` — to be superseded by S10 (cross-job concurrency tuned per Spark capacity)
- Repo: `https://github.com/Stanxxy/video-upscale.git`
