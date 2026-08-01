# Vision Engine REST API

This document describes the REST and Server-Sent Events contract exposed by
the vision engine. Jobs begin when `POST /track` is accepted; there is no
WebSocket handshake.

## Quick start

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

The video must already exist at the requested S3/LocalStack bucket and key.

## `POST /track`

Submit a new highlight-v2 analysis job. The response is an immediate REST
acknowledgement; the worker is scheduled by the service.

```json
{
  "bucket": "bjj-video-analysis",
  "key": "videos/match1.mp4",
  "output_bucket": "bjj-video-analysis"
}
```

Response:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

Persisted analysis settings are resolved at admission and stored with the
request. The engine owns the analysis lifecycle and writes progress to the
Keyspaces `job_lifecycle` row.

## `GET /job/{job_id}`

Read the current durable job snapshot. This endpoint is useful for a direct
status inspection; it is not a browser progress fallback for the SSE stream.

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress_percent": 46.0,
  "current_frame": 0,
  "total_frames": 0,
  "result_bucket": null,
  "result_key": null,
  "error_message": null
}
```

## `GET /jobs/{job_id}/events`

Subscribe to the durable Keyspaces lifecycle projection over native
Server-Sent Events (`text/event-stream`). The annotation backend may authorize
and transport this projection to the browser; the browser never connects to
Keyspaces directly and never switches to REST polling when SSE reconnects.

Progress payloads for highlight-v2 use coach-facing phases:

```json
{
  "type": "progress",
  "phase": "detecting",
  "percent": 46.0,
  "chunk_index": 1,
  "chunks_total": 4,
  "highlights_found_so_far": 7
}
```

The engine writes these durable fields:

- `stage`: internal engine stage (`highlight_ingest`, `highlight_chunk`, or
  `highlight_publish`)
- `stage_message`: `preparing`, `detecting`, `finalizing`, `completed`, or
  `error`
- `progress_percent`: monotonic whole-job percentage (`0–10`, `10–90`,
  `90–99`, then `100`)
- `chunk_index`, `chunks_total`: present while detecting
- `highlights_found_so_far`: unique detections after incremental seam dedup

The stream ends with `completed` or `job_error`. A dropped connection should
reconnect while preserving the last durable snapshot in the consumer UI.

## `DELETE /job/{job_id}`

Cancel an active job:

```json
{"status": "cancelled", "job_id": "550e8400-e29b-41d4-a716-446655440000"}
```

If the lifecycle has a replacement job, cancel the latest replacement instead.

## Human-in-the-loop resume

Use the REST endpoints below when a dormant tracking job enters a correction
state. They create a replacement lifecycle row and schedule the replacement;
no WebSocket client is involved.

- `POST /jobs/{job_id}/detection_response`
- `POST /jobs/{job_id}/resume`

## Local configuration

| Variable | Default | Description |
| --- | --- | --- |
| `SERVICE_HOST` | `0.0.0.0` | Bind address |
| `SERVICE_PORT` | `8000` | Port |
| `S3_ENDPOINT_URL` | unset | LocalStack endpoint when testing locally |
| `AWS_ACCESS_KEY_ID` | `test` | Local AWS-compatible credential |
| `AWS_SECRET_ACCESS_KEY` | `test` | Local AWS-compatible credential |
| `GEMINI_API_KEY` | unset | Gemini API key for real analysis |
