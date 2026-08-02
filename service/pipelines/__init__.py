"""Shared VLM pipeline framework for QA and production analysis.

Turns the single-call QA VLM Studio into a pipeline runner: a fixed set of
node types (per the build plan) assembled into named ``PipelineDef``s
(``single-shot``, ``vision-mimic``, ``vision-mimic-multi``), editable within
their fixed stage set (enable/disable/reorder/config-edit — no arbitrary node
authoring), executed by a streaming NDJSON run loop.

See working_log/plans/2026-07-03-vlm-studio-pipeline-dag-build-plan.md for the
authoritative design this package implements.

The QA routes drive this framework through ``service/qa_app.py``. Production's
Gemini-native highlight worker also imports ``executors.run_pipeline`` and the
dependency-neutral ``frame_dedup.deduplicate_clips`` leaf; modules must therefore
keep production import boundaries free of optional legacy GPU dependencies.
"""
