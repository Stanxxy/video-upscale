# Checkpoint artifacts (V1 envelope, highlight job)

**Status:** contract — vision engine + `video_analysis_and_annotation_service`
**Companion to:** backend `CHECKPOINT_DATA_SCHEMA_V1.md` (V1 envelope source of truth)
**Storage:** Keyspaces `video_analysis.job_stage_checkpoints.checkpoint_data` (JSON text), `PRIMARY KEY (job_id, stage_name)`.

Every write is an UPSERT on `(job_id, stage_name)`: the latest row carries the cumulative state. The caller reads the latest row, merges, and writes the full state.

## Envelope

```json
{
  "schema_version": 1,
  "artifacts": { },
  "worker_state": {
    "progress_percent": 35.0,
    "current_frame": 0,
    "total_frames": 0,
    "stage_progress_fraction": 0.5
  }
}
```

`worker_state` is required on every write. Durable external pointers live under `artifacts`; stage scalars sit beside it. Readers ignore unknown keys. `completed` (the column) is `true` only for terminal writes.

## Stages

| `stage_name` | When | Contents |
|---|---|---|
| `highlight_ingest` | After the S3 download and Gemini Files API upload | `reason: highlight_ingest_completed`; `artifacts.gemini_file_uri`, `gemini_file_name`, `gemini_file_mime_type`, `gemini_file_expiration` (ISO-8601). A resume reuses the upload unless it expires within 10 minutes. |
| `highlight_chunk` | After each outer chunk's scan, critique and analyze | `reason: highlight_chunk_completed`; top-level `chunk_index`, `chunks_total`, `highlights_scanned`, `highlights_analyzed`, `highlights_ditched`, `highlights_published` (always 0); `artifacts.gemini_file_uri`, `clips_by_chunk` (full merged map keyed by `str(chunk_index)`), `highlights_collected_by_chunk`. Resume continues at `1 + max(completed chunk_index)`. |
| `highlight_publish` | Per published candidate, then terminal | Progress: `reason: highlight_publish_candidate`, `artifacts.candidate_key`, `event_index`, `published_candidate_keys` (full cumulative list). Terminal: `reason: highlight_publish_completed`, `artifacts.sns_topic_arn`, `sns_event_count`, `sns_completion_sent`, `result_s3_uri`. |
| `cancellation` | On `DELETE /job/{job_id}` | `reason: user_cancelled`, `resume_cursor.frame_idx`, `progress_percent`. |

A crash after a partial SNS publish re-publishes the remainder; consumers tolerate duplicate events keyed by `job_id`.

## Retired stage names

`download`, `detect`, `track`, `upscale_analyze`, `annotate`, `upload` and `publish` belonged to the retired local tracking pipeline. Rows carrying them are read-only history: the engine never resumes them, and the lifecycle reader reports a retired stage with an explanatory `stage_message`.
