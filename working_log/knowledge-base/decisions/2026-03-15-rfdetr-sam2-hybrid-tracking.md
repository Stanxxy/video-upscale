---
date: 2026-03-15
category: decision
tags: [tracking, architecture, ml, rfdetr, sam2]
status: active
---

# RF-DETR + SAM2 Hybrid Tracking Architecture

## Context
The tracking pipeline needs to follow two athletes throughout a BJJ match. Pure
detection-per-frame is too slow for real-time annotation; pure mask propagation
(SAM2 alone) drifts under occlusion. A hybrid approach was chosen.

## Content
The tracking architecture uses RF-DETR for initial person detection and periodic
re-detection, combined with SAM2.1 for continuous mask propagation between keyframes.
Identity association across cuts and scrambles uses DINOv2 feature embeddings + color
histogram similarity scores.

Key files:
- `tracking_pipeline/detect.py` — RF-DETR wrapper
- `tracking_pipeline/sam2_manager.py` — SAM2 propagation wrapper
- `tracking_pipeline/identity_manager.py` — DINOv2 + color histogram re-ID
- `tracking_pipeline/tracking.py` — main hybrid tracking loop
- `tracking_pipeline/state_machine.py` — handles scrambles / scene cuts / fades

## Rationale
SAM2 alone loses identity under occlusion (common in BJJ). RF-DETR provides high-recall
re-detection anchors that SAM2 then propagates efficiently. DINOv2 visual features are
robust to positional changes; color histogram catches gi color differences quickly.

## Impact
All service tracking jobs go through `service/tracking_runner.py` which delegates to
`tracking/__init__.py` shim → `tracking_pipeline/tracking.py`. Any change to the tracking
algorithm lives in `tracking_pipeline/`.
