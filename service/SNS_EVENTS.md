# SNS Event Schema — Video Analysis Events

This document describes the events published to SNS by the video analysis service after processing a video. Use it to build an SQS consumer that writes these events into a database.

## Overview

```
Video Analysis Service
        │
        ▼
   SNS Topic (video_analysis_events)
        │
        ├──► SQS Queue (your consumer)
        │         │
        │         ▼
        │    Process messages, write to DB
        │
        └──► (other subscribers)
```

When a video analysis job completes, the service publishes **N + 1** messages to the SNS topic:

1. **N `bjj_event_detected` messages** — one per detected technique/clip
2. **1 `analysis_complete` message** — boundary signal indicating all events for this video have been sent

## SNS Topic

- **Default ARN:** `arn:aws:sns:us-east-1:000000000000:video_analysis_events`
- Can be overridden per-job via the `sns_topic_arn` field in the job request

## SQS Message Envelope

SNS wraps each message before delivering to SQS. Your consumer receives this envelope:

```json
{
  "Type": "Notification",
  "MessageId": "sns-message-id",
  "TopicArn": "arn:aws:sns:us-east-1:...:video_analysis_events",
  "Message": "<JSON string — the actual payload, see schemas below>",
  "MessageAttributes": { ... },
  "Timestamp": "2025-01-15T12:00:00.000Z"
}
```

**Important:** The `Message` field is a JSON-encoded **string**, not a nested object. You must parse it:

```python
import json
body = json.loads(sqs_message["Body"])        # parse the SQS envelope
payload = json.loads(body["Message"])          # parse the SNS message payload
event_type = body["MessageAttributes"]["event_type"]["Value"]
```

---

## Event Type 1: `bjj_event_detected`

Published once per detected technique clip. This is the primary event your DB consumer will store.

### MessageAttributes

| Attribute      | DataType | Description                              | Example                  |
|----------------|----------|------------------------------------------|--------------------------|
| `event_type`   | String   | Always `"bjj_event_detected"`            | `"bjj_event_detected"`   |
| `category`     | String   | Technique category (see categories below)| `"SUBMISSION_OFFENSE"`   |
| `event_index`  | Number   | 1-based index of this event in the batch | `"3"`                    |
| `total_events` | Number   | Total events for this video              | `"12"`                   |

### Message Body Schema

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "video_id": "11111111-2222-3333-4444-555555555555",
  "start_time": "00:01:23",
  "end_time": "00:01:45",
  "event_candidates": [
    {
      "id": "f6e5d4c3-b2a1-0987-fedc-ba0987654321",
      "role": "athlete in white gi",
      "skill_name": "Double Leg Takedown",
      "category": "STANDUP_GAME",
      "confidence": 0.85
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique event ID (auto-generated) |
| `video_id` | UUID | The video this event belongs to. Use this to group events per video. |
| `start_time` | String | Start timestamp in `HH:MM:SS` format |
| `end_time` | String | End timestamp in `HH:MM:SS` format |
| `event_candidates` | Array | Always contains exactly 1 candidate in the current implementation |
| `event_candidates[].id` | UUID | Unique candidate ID (auto-generated) |
| `event_candidates[].role` | String | Description of the athlete performing the technique |
| `event_candidates[].skill_name` | String | Specific technique name (e.g., "Armbar from Guard") |
| `event_candidates[].category` | String | One of the 6 technique categories (see below) |
| `event_candidates[].confidence` | Float | Confidence score, 0.0 to 1.0 |

### Technique Categories

| Category | Description |
|----------|-------------|
| `STANDUP_GAME` | Takedowns, leg kicks, striking exchanges |
| `GUARD_PLAY` | Guard passes, sweep attempts, open/closed guard work |
| `GUARD_PASSING` | Guard progression, side control transitions |
| `POSITIONAL_DOMINANCE` | Mount control, knee on belly, back control |
| `SUBMISSION_OFFENSE` | Armbar, triangle, leg lock attempts |
| `DEFENSE_ESCAPES` | Bridge escapes, reversals, defensive movements |

---

## Event Type 2: `analysis_complete`

Published **once** after all `bjj_event_detected` messages for a video. This is your **boundary signal** — when you receive this, you know all events for the video have been sent.

### MessageAttributes

| Attribute    | DataType | Description                       | Example                |
|--------------|----------|-----------------------------------|------------------------|
| `event_type` | String   | Always `"analysis_complete"`      | `"analysis_complete"`  |

### Message Body Schema

```json
{
  "video_id": "11111111-2222-3333-4444-555555555555",
  "job_id": "abc123",
  "total_event_count": 12,
  "result_s3_uri": "s3://output-bucket/results/video-123.json"
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | UUID | The video this completion signal belongs to |
| `job_id` | String | The processing job ID (can be used to query `GET /jobs/{job_id}` for full status) |
| `total_event_count` | Integer | Number of `bjj_event_detected` events published. Can be 0. |
| `result_s3_uri` | String or null | S3 URI of the full analysis JSON result |

---

## SQS Filter Policies

Set up filter policies on your SQS subscription depending on what your consumer needs:

**Receive all events (recommended for DB consumer):**

```json
{}
```

No filter — receives both `bjj_event_detected` and `analysis_complete` messages.

**Receive only clip events:**

```json
{
  "event_type": ["bjj_event_detected"]
}
```

**Receive only completion signals:**

```json
{
  "event_type": ["analysis_complete"]
}
```

**Receive only specific categories + completion:**

```json
{
  "event_type": ["bjj_event_detected", "analysis_complete"],
  "category": [{"exists": false}, "SUBMISSION_OFFENSE", "STANDUP_GAME"]
}
```

Note: `{"exists": false}` allows the `analysis_complete` message through since it does not carry a `category` attribute.

---

## Consumer Implementation Guide

### Recommended Processing Flow

```
receive SQS message
    │
    ├─ event_type == "bjj_event_detected"
    │       │
    │       ▼
    │   Upsert event into DB
    │   (deduplicate by event `id` or by `video_id` + `event_index`)
    │
    └─ event_type == "analysis_complete"
            │
            ▼
        Mark video analysis as complete in DB
        Trigger any downstream processing
        (e.g., notify client, generate summary)
```

### Handling Out-of-Order Delivery

SNS/SQS does **not** guarantee message ordering. The `analysis_complete` message may arrive **before** some `bjj_event_detected` messages. Handle this:

**Strategy: Use `total_events` to verify completeness**

```python
def handle_message(message):
    event_type = get_event_type(message)

    if event_type == "bjj_event_detected":
        payload = parse_payload(message)
        total = int(get_message_attribute(message, "total_events"))
        upsert_event(payload)
        check_completeness(payload["video_id"], total)

    elif event_type == "analysis_complete":
        payload = parse_payload(message)
        mark_completion_received(payload["video_id"], payload["total_event_count"])
        check_completeness(payload["video_id"], payload["total_event_count"])


def check_completeness(video_id, expected_total):
    """Called after every message. Triggers finalization only when
    all clip events AND the completion signal have been received."""
    received_count = count_events_in_db(video_id)
    completion_received = is_completion_received(video_id)

    if received_count >= expected_total and completion_received:
        finalize_video_analysis(video_id)
```

### Deduplication

SQS may deliver the same message more than once. Deduplicate by:

- **Event `id` field** (UUID) — natural primary key for `bjj_event_detected` messages
- **`(video_id, event_index)`** — alternative composite key using the MessageAttribute

### Suggested DB Schema

```sql
-- Track video analysis completion
CREATE TABLE video_analysis_status (
    video_id            UUID PRIMARY KEY,
    job_id              VARCHAR(64),
    total_event_count   INTEGER NOT NULL,
    received_count      INTEGER DEFAULT 0,
    completion_received BOOLEAN DEFAULT FALSE,
    result_s3_uri       TEXT,
    finalized           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);
```

### Zero-Event Case

When a video produces no detected techniques, the service publishes **only** the `analysis_complete` message with `total_event_count: 0`. Your consumer should handle this by marking the video as complete immediately.
