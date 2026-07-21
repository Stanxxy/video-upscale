-- S12 Phase 1b production wiring (design §1.2 SSE progress shape + LeCun's
-- attribution-metrics measurement plumbing, item 11.5): additive
-- job_lifecycle columns carried by the v2 highlight orchestrator's own
-- progress writes only — absent/NULL for every dormant tracking-pipeline
-- job (which never calls the new update_highlight_chunk_progress method).
-- Replace video_analysis with KEYSPACE_NAME if different.
-- Idempotent: re-ADD fails until driver/script ignores duplicate errors.

ALTER TABLE video_analysis.job_lifecycle ADD chunk_index int;
ALTER TABLE video_analysis.job_lifecycle ADD chunks_total int;
ALTER TABLE video_analysis.job_lifecycle ADD highlights_found_so_far int;
ALTER TABLE video_analysis.job_lifecycle ADD attribution_metrics_json text;
