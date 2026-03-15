---
date: 2026-03-15
category: insight
tags: [tracking, ml, identity, dinov2, reid]
status: active
---

# DINOv2 + Color Histogram Re-ID Strategy

## Context
BJJ matches involve two athletes who regularly occlude each other, change position
rapidly, and may temporarily leave frame. Re-identification (re-ID) after occlusion
or scene cuts must be fast and reliable without a dedicated re-ID dataset.

## Content
Identity re-ID uses a two-factor score:
1. **DINOv2 cosine similarity** — robust to pose changes, captures gi texture/appearance
2. **Color histogram intersection** — fast, good for distinguishing gi colors (e.g. white vs dark)

Both scores are combined with a weighted sum. Thresholds are tuned in `test_tracking/identity_manager.py`.

## Rationale
A dedicated re-ID model (e.g. OSNet) would require a BJJ-specific dataset. DINOv2
zero-shot features + color histograms achieve acceptable performance without training.

## Impact
- `test_tracking/identity_manager.py` — re-ID implementation
- If re-ID failure rate is high, tune `SIMILARITY_THRESHOLD` in that file first
- Adding a trained re-ID model would slot in as a replacement for the DINOv2 path
