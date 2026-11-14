"""Portable computation only. Hosts own providers, persistence and authorization."""

CORE_VERSION = "0.3.0"
WORKFLOW_ID = "position-timeline-technique-events"
# Same core and DAG with the mma.v1 prompt pack (bjj_analysis_core.packs).
MMA_WORKFLOW_ID = "mma-position-timeline-technique-events"
# MMA with a windowed strike census instead of the per-strike critique/analyze/validate
# chain; grappling attacks keep that chain (bjj_analysis_core.strikes).
MMA_STRIKES_WORKFLOW_ID = "mma-windowed-strikes"
