-- S12 Phase 1b production wiring (design §6.2): pipeline_kind column on
-- job_lifecycle, distinguishing the dormant tracking pipeline from the v2
-- highlight-scan-critique-analyze production path ("tracking" | "highlight_v2").
-- Replace video_analysis with KEYSPACE_NAME if different.
-- Idempotent: re-ADD fails until driver/script ignores duplicate errors.

ALTER TABLE video_analysis.job_lifecycle ADD pipeline_kind text;
