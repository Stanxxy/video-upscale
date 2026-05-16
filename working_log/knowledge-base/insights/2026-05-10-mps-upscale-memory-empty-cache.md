---
date: 2026-05-10
category: insight
tags: [service, upscale, pytorch, mps, memory, restorer, ops]
status: active
---

# MPS unified memory during RealESRGAN upscale (empty_cache mitigation)

## Context

After tracking completes, `_run_upscale_analysis` in `service/worker.py` runs per-frame crops through **`RealESRGANRestorer`** (`restorer.py`) or **`DiffusionRestorer`** (`diffusion_restorer.py`) on **Apple Silicon MPS** when available. Gemini analysis uses a **bounded** sliding window of PIL images (30-frame window, 15-frame stride); **`analysis_results`** only grows by small JSON blobs per window.

## Symptom

Activity Monitor showed **process RSS climbing into multiple GB** early in the upscale stage (often within the first few enhanced frames). Heartbeats (`buffer=N`) reflected only the **analysis sliding buffer**, not “all upscaled frames kept in RAM.”

## Root cause (validated with runtime probes)

Memory pressure was **not** primarily from Gemini payloads or unbounded lists of upscaled tensors. Instrumentation showed:

- **Baseline** before the upscale loop could already be hundreds of MB (loaded tracking JSON, Python process, model weights).
- **The largest step-up** occurred on the **first `enhance()` calls**: RSS jumped by **on the order of ~1 GB** while **`analysis_windows` was still 0** — i.e. before any Gemini window ran.
- RSS **did not** spike because of **`analysis_results`** (per-window JSON was tiny).
- **Apple MPS uses unified memory**: PyTorch’s allocator retains scratch buffers for throughput; without releasing cache, the OS sees **very high resident memory** even though application-level structures (sliding buffer) are small.

## Mitigation (implemented)

After each successful `enhance()`, release accelerator cache:

- **`torch.mps.empty_cache()`** when `device.type == "mps"` (guard with `hasattr(torch.mps, "empty_cache")` for older PyTorch).
- **`torch.cuda.empty_cache()`** when on CUDA (same pattern for fragmentation/cache retention).

Implemented in:

- `restorer.py` — `RealESRGANRestorer.enhance`
- `diffusion_restorer.py` — `DiffusionRestorer.enhance`

**Trade-off:** Slight potential throughput cost versus returning cached blocks to the driver so **RSS stays closer to steady-state** across thousands of frames.

## What did *not* require changing

- **`service/worker.py`** sliding-window logic for Gemini (already bounded).
- **Checkpoint / `analysis_raw.json`** persistence (orthogonal to GPU cache retention).

## If memory is still high after this

Consider **tiling** large crops, lowering **`target_size`**, or **CPU inference** for upscale (slow but avoids unified-memory spikes). Those are separate knobs from `empty_cache`.

## Primary files

- `restorer.py` — `RealESRGANRestorer.enhance` (MPS/CUDA cache release)
- `diffusion_restorer.py` — `DiffusionRestorer.enhance` (same)
- `service/worker.py` — `_run_upscale_analysis` (orchestration only)
