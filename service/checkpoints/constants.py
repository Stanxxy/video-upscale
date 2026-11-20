"""Checkpoint package constants."""

from service.analysis_keyspaces_enums import PipelineStage

# The v2 highlight-scan-critique-analyze job's stage order.
HIGHLIGHT_STAGE_ORDER = [
    PipelineStage.HIGHLIGHT_INGEST,
    PipelineStage.HIGHLIGHT_CHUNK,
    PipelineStage.HIGHLIGHT_PUBLISH,
]
