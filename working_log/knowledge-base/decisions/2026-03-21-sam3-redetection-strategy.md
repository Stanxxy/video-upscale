---
date: 2026-03-21
category: decision
tags: [tracking, architecture, ml, sam3, redetection]
status: planned
---

# SAM3 for Mid-Tracking Re-Detection

## Context
New hardware supports SAM3. Initial question was whether SAM3's text-prompted
segmentation could remove the manual box-selection step entirely. Experiments
with video clips showed SAM3 initial detection accuracy is unstable, so it is
not suitable as a drop-in replacement for YOLO26 at pipeline startup.

However, SAM3 is a strong candidate for mid-tracking re-detection (when a track
is lost during a scramble), where its mask-based approach handles ground-fighting
poses better than YOLO26 bounding box regression.

## Decision
**SAM3 is scoped to mid-tracking re-detection only (opt-in, default off).**

- Initial athlete detection: YOLO26 (unchanged — stable, production-ready)
- Re-detection on track loss (RE_ID_MODE): SAM3 via `use_sam3_redetection=True` flag
- Over-segmentation (referees): handled by the existing `vllm_selector.suggest_athletes()` + human-confirm flow — no new components needed

## Key Files
- `tracking_pipeline/sam3_detector.py` — **to be created**: SAM3 wrapper with same interface as `YOLO26Detector.detect_persons()` → returns `[{"box", "confidence", "track_id"}]`
- `tracking_pipeline/hybrid_tracking.py` — `_detect_and_request_boxes()` gains `use_sam3_redetection` param (lines 360, 495: the two re-detection call sites)
- `tracking_pipeline/pipeline.py` — threads flag through `run_pipeline()` → `run_tracking()`
- `service/models.py` — `TrackRequest.use_sam3_redetection: bool = False`
- `service/worker.py` — passes flag to `run_pipeline()`

## Rationale
Track loss events in BJJ typically occur during scrambles and ground positions
where athletes are heavily occluded or overlapping. SAM3's text-prompted mask
segmentation is more robust to these poses than YOLO26's bounding box regression.
Keeping it as an opt-in flag means zero risk to the production YOLO26 + SAM2 pipeline.

## Branch
`feature/sam3-redetection` from `feature/taxonomy-mapper` (not master — experimental).

## Impact
No existing behaviour changes unless `use_sam3_redetection=True` is set. The full
implementation plan is at `/Users/stanliu/.claude/plans/glimmering-squishing-seahorse.md`.
