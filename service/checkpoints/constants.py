"""Checkpoint package constants."""

from service.analysis_keyspaces_enums import PipelineStage

STAGE_ORDER = [
    PipelineStage.DOWNLOAD,
    PipelineStage.DETECT,
    PipelineStage.TRACK,
    PipelineStage.UPSCALE_ANALYZE,
    PipelineStage.ANNOTATE,
    PipelineStage.UPLOAD,
    PipelineStage.PUBLISH,
]

# S12 Phase 1b (design §6.1) — the v2 highlight-scan-critique-analyze job's
# own, much shorter stage order. Deliberately NOT shoehorned into
# STAGE_ORDER above (that list encodes the TRACKING pipeline's specific
# stage sequence).
HIGHLIGHT_STAGE_ORDER = [
    PipelineStage.HIGHLIGHT_INGEST,
    PipelineStage.HIGHLIGHT_CHUNK,
    PipelineStage.HIGHLIGHT_PUBLISH,
]

# Sentinel used to short-circuit the tracking pass when resuming after an
# upscale/analysis crash — see CHECKPOINT_ARTIFACTS_V1_ADDENDUM.md "Resume
# parameter forwarding". Any value past plausible video lengths works.
END_OF_TRACKING_SENTINEL = 10**9
