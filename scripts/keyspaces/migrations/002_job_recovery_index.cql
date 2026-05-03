-- Recovery index for stale RUNNING discovery (partitioned by recovery_state + heartbeat_bucket).
-- Replace video_analysis with KEYSPACE_NAME if different.

CREATE TABLE IF NOT EXISTS video_analysis.job_recovery_index (
  recovery_state text,
  heartbeat_bucket text,
  last_heartbeat_at timestamp,
  job_id text,
  owner_instance_id text,
  video_id text,
  job_state text,
  updated_at timestamp,
  PRIMARY KEY ((recovery_state, heartbeat_bucket), last_heartbeat_at, job_id)
) WITH CLUSTERING ORDER BY (last_heartbeat_at ASC, job_id ASC);
