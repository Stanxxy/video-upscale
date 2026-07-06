"""VLM Studio pipeline framework — QA-only surface.

Turns the single-call QA VLM Studio into a pipeline runner: a fixed set of
node types (per the build plan) assembled into named ``PipelineDef``s
(``single-shot``, ``vision-mimic``, ``vision-mimic-multi``), editable within
their fixed stage set (enable/disable/reorder/config-edit — no arbitrary node
authoring), executed by a streaming NDJSON run loop.

See working_log/plans/2026-07-03-vlm-studio-pipeline-dag-build-plan.md for the
authoritative design this package implements.

Nothing here is imported by production (``service/app.py``, ``service/worker/**``,
``pipeline.py:process_video``) — this is a QA-only surface mounted by
``service/qa_app.py``.
"""
