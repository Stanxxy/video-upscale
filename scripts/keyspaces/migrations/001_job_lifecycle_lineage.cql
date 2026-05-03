-- Job recovery schema: lineage columns on job_lifecycle.
-- Replace video_analysis with KEYSPACE_NAME if different.
-- Idempotent: re-ADD fails until driver/script ignores duplicate errors.

ALTER TABLE video_analysis.job_lifecycle ADD parent_job_id text;
ALTER TABLE video_analysis.job_lifecycle ADD replacement_job_id text;
