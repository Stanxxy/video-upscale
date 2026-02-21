# BJJ Video Analysis Service - API Documentation

Base URL: `http://<host>:9001`

---

## Endpoints

### 1. Submit Analysis Job

**`POST /jobs`**

Triggers a video upscale + BJJ technique analysis pipeline. The request is accepted immediately and processing runs in the background. Poll `GET /jobs/{job_id}` for status.

**Request Body**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video_id` | UUID | Yes | - | UUID identifying this video. Passed through to SNS event payloads as `VideoEventWithCandidates.video_id`. |
| `video_s3_bucket` | string | Yes | - | S3 bucket containing the input video file. |
| `video_s3_key` | string | Yes | - | S3 object key for the input video (e.g. `videos/match_001.mp4`). |
| `tracking_json_s3_bucket` | string | Yes | - | S3 bucket containing the tracking JSON file. |
| `tracking_json_s3_key` | string | Yes | - | S3 object key for the tracking JSON (e.g. `tracking/match_001.json`). |
| `output_s3_bucket` | string | Yes | - | S3 bucket where the analysis result JSON will be uploaded. |
| `output_s3_key` | string | Yes | - | S3 object key for the output analysis JSON (e.g. `results/match_001/analysis.json`). |
| `sns_topic_arn` | string | No | Server default | SNS topic ARN for publishing detected events. Overrides the server's `BJJ_SNS_TOPIC_ARN` env var. If neither is set, no events are published. |
| `sampling_rate` | integer | No | `1` | Process every Nth frame. Must be >= 1. Higher values speed up processing but reduce temporal resolution. |
| `analyzer_mode` | string | No | `"single"` | `"single"` for single-agent (Gemini Thinking model) or `"multi"` for multi-agent system (3 specialist agents + judge). |
| `method` | string | No | `"esrgan"` | Enhancement model. One of: `"esrgan"`, `"swinir"`, `"hat"`, `"diffusion"`. |

**Example Request**

```bash
curl -X POST http://localhost:9001/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_s3_bucket": "my-input-bucket",
    "video_s3_key": "videos/match_001.mp4",
    "tracking_json_s3_bucket": "my-input-bucket",
    "tracking_json_s3_key": "tracking/match_001.json",
    "output_s3_bucket": "my-output-bucket",
    "output_s3_key": "results/match_001/analysis.json",
    "sampling_rate": 5,
    "analyzer_mode": "single"
  }'
```

**Response** — `202 Accepted`

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "progress": 0.0,
  "message": "",
  "result_s3_uri": null,
  "error": null,
  "created_at": "2026-02-14T10:30:00.000000+00:00",
  "updated_at": "2026-02-14T10:30:00.000000+00:00"
}
```

**Error Responses**

| Status | Condition |
|--------|-----------|
| `422` | Invalid request body (missing required fields, invalid types, `sampling_rate < 1`). |
| `429` | Server is at capacity (another job is already running). Retry later. |

---

### 2. Get Job Status

**`GET /jobs/{job_id}`**

Poll the current status and progress of a submitted job.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | The job ID returned by `POST /jobs`. |

**Example Request**

```bash
curl http://localhost:9001/jobs/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response** — `200 OK`

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "progress": 0.45,
  "message": "enhancing (45%)",
  "result_s3_uri": null,
  "error": null,
  "created_at": "2026-02-14T10:30:00.000000+00:00",
  "updated_at": "2026-02-14T10:30:15.000000+00:00"
}
```

**Job Status Values**

| Status | Description |
|--------|-------------|
| `queued` | Job accepted, waiting to start. |
| `downloading` | Downloading video and tracking JSON from S3. |
| `processing` | Running upscale + analysis pipeline. `progress` updates during this phase. |
| `uploading` | Uploading analysis result JSON to S3. |
| `publishing` | Publishing detected events to SNS. |
| `completed` | Job finished successfully. `result_s3_uri` contains the output location. |
| `failed` | Job failed. `error` contains the error message. |

**Completed Job Response**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "progress": 1.0,
  "message": "Completed. 12 clips detected, 12 events published.",
  "result_s3_uri": "s3://my-output-bucket/results/match_001/analysis.json",
  "error": null,
  "created_at": "2026-02-14T10:30:00.000000+00:00",
  "updated_at": "2026-02-14T10:45:30.000000+00:00"
}
```

**Failed Job Response**

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "failed",
  "progress": 0.05,
  "message": "Downloading input files from S3",
  "result_s3_uri": null,
  "error": "An error occurred (NoSuchKey) when calling the GetObject operation",
  "created_at": "2026-02-14T10:30:00.000000+00:00",
  "updated_at": "2026-02-14T10:30:02.000000+00:00"
}
```

**Error Responses**

| Status | Condition |
|--------|-----------|
| `404` | Job ID not found. |

---

### 3. Health Check

**`GET /health`**

Returns service health and GPU availability.

**Example Request**

```bash
curl http://localhost:9001/health
```

**Response** — `200 OK`

```json
{
  "status": "ok",
  "gpu_available": true
}
```

---

## SNS Event Schema

When a job completes, each detected BJJ event is published as a separate SNS message to the configured topic. Downstream consumers (e.g. via SQS subscription) receive one `VideoEventWithCandidates` per message.

**Message Body**

```json
{
  "id": "b3e9cd05-8ed2-42d3-b5fa-25e9b925f6b3",
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "start_time": "00:01:24",
  "end_time": "00:01:38",
  "event_candidates": [
    {
      "id": "5e64af47-6f14-48ed-91bc-8f5c099d1528",
      "role": "athlete in white gi",
      "skill_name": "Double Leg Takedown",
      "category": "STANDUP_GAME",
      "confidence": 0.91
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Auto-generated unique ID for this event. |
| `video_id` | UUID | The `video_id` from the original job request. |
| `start_time` | string | Event start timestamp in `HH:MM:SS` format, derived from frame number and video FPS. |
| `end_time` | string | Event end timestamp in `HH:MM:SS` format. |
| `event_candidates` | array | Always contains exactly 1 candidate for each event. |
| `event_candidates[].id` | UUID | Auto-generated unique ID for this candidate. |
| `event_candidates[].role` | string | Which athlete performs the technique (e.g. `"athlete in white gi"`, `"top athlete"`). |
| `event_candidates[].skill_name` | string | The specific technique name (e.g. `"Armbar"`, `"Double Leg Takedown"`). |
| `event_candidates[].category` | string | One of: `STANDUP_GAME`, `GUARD_PLAY`, `GUARD_PASSING`, `POSITIONAL_DOMINANCE`, `SUBMISSION_OFFENSE`, `DEFENSE_ESCAPES`. |
| `event_candidates[].confidence` | float | Confidence score from 0.0 to 1.0. |

**SNS MessageAttributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `category` | String | The event category (e.g. `SUBMISSION_OFFENSE`). Use this for SNS filter policies on SQS subscriptions. |
| `event_type` | String | Always `"bjj_event_detected"`. |

---

## S3 Output Schema

The analysis result JSON uploaded to `output_s3_bucket/output_s3_key` has this structure:

```json
{
  "match_summary": "Analysis generated via Single-Agent",
  "clips": [
    {
      "start_frame": 120,
      "end_frame": 150,
      "category": "SUBMISSION_OFFENSE",
      "specific_technique": "Armbar",
      "role": "athlete in white gi",
      "reasoning": "Biomechanical analysis of the technique...",
      "confidence": 0.92
    }
  ],
  "fps": 30.0
}
```

---

## Tracking JSON Input Format

The tracking JSON referenced by `tracking_json_s3_key` must follow this structure:

```json
{
  "video": "video_filename.mp4",
  "frames": [
    {
      "frame": 121,
      "timestamp": 4.033,
      "athletes": [
        {
          "track_id": 1,
          "box": [x1, y1, x2, y2],
          "keypoints": [[x, y], ...]
        }
      ]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `frames[].frame` | integer | Frame number in the video. |
| `frames[].timestamp` | float | Timestamp in seconds. |
| `frames[].athletes[].track_id` | integer | Persistent athlete tracking ID. |
| `frames[].athletes[].box` | array[4] | Bounding box as `[x1, y1, x2, y2]` (top-left to bottom-right). |
| `frames[].athletes[].keypoints` | array | 17 keypoints in COCO pose format, each as `[x, y]`. |

---

## Concurrency

The service processes **one job at a time** (GPU-bound). If a job is already running, `POST /jobs` returns `429 Too Many Requests`. Clients should retry with backoff.

## Interactive Docs

When the service is running, OpenAPI docs are available at:
- Swagger UI: `http://<host>:9001/docs`
- ReDoc: `http://<host>:9001/redoc`
