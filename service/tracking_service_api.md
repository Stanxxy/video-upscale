# BJJ Tracking Service — API Contract

This document describes the REST + WebSocket API exposed by the tracking service (`service/app.py`). It is intended for external AI agents or human developers building clients.

## Quick Start

```bash
# Start the service
uvicorn service.app:app --host 0.0.0.0 --port 8000

# Or run directly
python -m service.app
```

## Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_HOST` | `0.0.0.0` | Host to bind |
| `SERVICE_PORT` | `8000` | Port to bind |
| `S3_ENDPOINT_URL` | _(unset)_ | Optional custom S3 endpoint (leave empty for AWS production) |
| `DETECTION_TIMEOUT` | `300.0` | Seconds to wait for manual detection response |
| `AWS_ACCESS_KEY_ID` | `test` | S3 credentials |
| `AWS_SECRET_ACCESS_KEY` | `test` | S3 credentials |

## REST Endpoints

### `GET /health`

Health check.

**Response** `200`:
```json
{"status": "healthy", "timestamp": "2025-01-15T12:00:00"}
```

---

### `POST /track`

Submit a new tracking job. The video must already be in S3.

**Request body** (`application/json`):
```json
{
  "bucket": "my-bucket",
  "key": "videos/match1.mp4",
  "output_bucket": null,
  "sam2_model": "facebook/sam2.1-hiera-base-plus"
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `bucket` | yes | — | S3 bucket containing the input video |
| `key` | yes | — | S3 object key of the input video |
| `output_bucket` | no | same as `bucket` | Bucket for result upload |
| `sam2_model` | no | `facebook/sam2.1-hiera-base-plus` | SAM2 model identifier |

**Response** `200`:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "ws_url": "ws://localhost:8000/ws/550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

After receiving the response, the client **must** connect to `ws_url` to start processing. The tracking job only begins when the WebSocket connects.

---

### `GET /job/{job_id}`

Poll job status.

**Response** `200`:
```json
{
  "job_id": "550e8400-...",
  "status": "processing",
  "progress_percent": 42.5,
  "current_frame": 850,
  "total_frames": 2000,
  "result_bucket": null,
  "result_key": null,
  "error_message": null
}
```

**Status values**: `pending`, `downloading`, `processing`, `waiting_for_detection`, `uploading`, `completed`, `failed`, `cancelled`.

---

### `DELETE /job/{job_id}`

Cancel a running job and clean up temporary files.

**Response** `200`:
```json
{"status": "cancelled", "job_id": "550e8400-..."}
```

---

## WebSocket Protocol

### Endpoint: `ws://{host}:{port}/ws/{job_id}`

Bidirectional JSON messages over WebSocket. Connect after `POST /track`.

### Server → Client Messages

#### `progress`
Sent periodically during tracking.
```json
{
  "type": "progress",
  "frame_idx": 150,
  "total_frames": 2000,
  "state": "tracking",
  "percent": 7.5
}
```

#### `manual_detection_required`
Sent when the tracker needs human/agent input to identify athletes in a frame.
```json
{
  "type": "manual_detection_required",
  "frame_idx": 0,
  "frame_base64": "<base64-encoded JPEG>",
  "reason": "initial"
}
```

`reason` values: `"initial"` (first frame), `"transition"` (scene change), `"tracking_lost"` (tracker lost both athletes).

The client **must** respond with either `detection_response` or `detection_cancelled` within the configured timeout (default 5 minutes).

#### `completed`
Sent when the job finishes successfully.
```json
{
  "type": "completed",
  "result_bucket": "my-bucket",
  "result_key": "videos/match1_tracked.json"
}
```

#### `error`
Sent on failure.
```json
{
  "type": "error",
  "message": "Tracking failed - no result produced"
}
```

### Client → Server Messages

#### `detection_response`
Reply to `manual_detection_required`. Each box is `[x1, y1, x2, y2]` in pixel coordinates.
```json
{
  "type": "detection_response",
  "box_a": [100, 200, 300, 500],
  "box_b": [400, 180, 600, 520]
}
```

#### `detection_cancelled`
Cancel the detection request (tracker will use fallback or fail).
```json
{"type": "detection_cancelled"}
```

#### `ping`
Keepalive. Server responds with `{"type": "pong"}`.
```json
{"type": "ping"}
```

## Job Lifecycle

```
POST /track          →  job created (pending)
WS connect           →  tracking starts
                         ├── downloading
                         ├── processing
                         │     ├── progress messages
                         │     └── manual_detection_required (0..N times)
                         ├── uploading
                         └── completed / failed
WS disconnect
GET /job/{id}        →  poll final status + result location
```

## S3 Layout

- **Input**: `s3://{bucket}/{key}` — MP4 video
- **Output**: `s3://{bucket}/{base}_tracked.json` — JSON tracking result

When `S3_ENDPOINT_URL` is set (local emulators), the service can auto-create missing buckets.
In AWS production, buckets should be pre-provisioned and `S3_ENDPOINT_URL` should remain unset.

## Python Client SDK

A ready-made client is available in `service_client/`:

```python
from service_client import TrackingClient

client = TrackingClient("http://localhost:8000")

# Check health
print(client.health())

# Submit job
job = client.submit("my-bucket", "videos/match.mp4")

# Run with WebSocket (blocks until done)
def handle_detection(msg):
    # msg["frame_base64"] contains the JPEG frame
    # Return bounding boxes or None to cancel
    return {"box_a": [100, 200, 300, 500], "box_b": [400, 180, 600, 520]}

result = client.run_job(
    job["ws_url"],
    on_detection=handle_detection,
    on_progress=lambda m: print(f"{m['percent']}%"),
)
print(f"Result at s3://{result['result_bucket']}/{result['result_key']}")
```
