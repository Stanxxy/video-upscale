"""Unit tests: AnalysisCompleteEvent carries tracking_s3_uri (Gap B).

The vision engine must publish the per-frame tracking JSON location
(`..._tracked.json`) on the completion boundary event so the backend can
serve geometry from GET /api/tracking/frames/{video_id}. These tests pin
the field name (`tracking_s3_uri`) and prove the published URI matches the
key the upload stage derives from the SAME base_key.
"""
from __future__ import annotations

import json
import os
from uuid import uuid4

from service.sns import SNSPublisher


def _base_key(input_key: str) -> str:
    """Mirror of upload.py base_key derivation (os.path.splitext)."""
    return os.path.splitext(input_key)[0]


class _FakeSNSClient:
    """Captures every publish() call instead of hitting AWS."""

    def __init__(self):
        self.published = []

    def publish(self, *, TopicArn, Message, MessageAttributes):
        self.published.append(
            {
                "TopicArn": TopicArn,
                "Message": Message,
                "MessageAttributes": MessageAttributes,
            }
        )
        return {"MessageId": "fake"}


def _make_publisher() -> SNSPublisher:
    pub = SNSPublisher.__new__(SNSPublisher)
    pub.client = _FakeSNSClient()
    pub.topic_arn = "arn:aws:sns:us-east-1:000000000000:bjj-events"
    return pub


def _completion_message(client: _FakeSNSClient) -> dict:
    for call in client.published:
        attrs = call["MessageAttributes"]
        if attrs.get("event_type", {}).get("StringValue") == "analysis_complete":
            return json.loads(call["Message"])
    raise AssertionError("no analysis_complete boundary event was published")


def test_completion_event_carries_tracking_s3_uri():
    pub = _make_publisher()
    output_bucket = "bjj-video-analysis"
    input_key = "videos/match.mp4"
    base_key = _base_key(input_key)
    tracking_uri = f"s3://{output_bucket}/{base_key}_tracked.json"
    result_uri = f"s3://{output_bucket}/{base_key}_analysis.json"

    pub.publish_events(
        {"clips": [{"action": "guard_pass", "technique": "other"}]},
        uuid4(),
        fps=30.0,
        job_id="job-1",
        result_s3_uri=result_uri,
        tracking_s3_uri=tracking_uri,
    )

    completion = _completion_message(pub.client)
    # Field name MUST be exactly tracking_s3_uri (cross-repo invariant).
    assert "tracking_s3_uri" in completion
    assert completion["tracking_s3_uri"] == tracking_uri
    assert completion["tracking_s3_uri"].endswith("_tracked.json")
    # result_s3_uri still flows independently and points at the analysis JSON.
    assert completion["result_s3_uri"] == result_uri
    assert completion["result_s3_uri"].endswith("_analysis.json")


def test_tracking_uri_matches_upload_key_derivation():
    """The published tracking URI must equal what upload.py would upload to.

    upload.py builds: tracking_result_key = f"{base_key}_tracked.json"
    and uploads to output_bucket. publish.py builds the URI from the SAME
    ctx.output_bucket / ctx.tracking_result_key. We reconstruct both sides
    from the single base_key here and assert byte-identity.
    """
    pub = _make_publisher()
    output_bucket = "bjj-video-analysis"
    input_key = "users/u42/whole_video/raw.mp4"
    base_key = _base_key(input_key)

    # What upload.py persists on ctx.tracking_result_key:
    upload_tracking_key = f"{base_key}_tracked.json"
    # What publish.py composes into the SNS event:
    published_tracking_uri = f"s3://{output_bucket}/{upload_tracking_key}"

    pub.publish_events(
        {"clips": []},
        uuid4(),
        fps=24.0,
        job_id="job-2",
        result_s3_uri=f"s3://{output_bucket}/{base_key}_analysis.json",
        tracking_s3_uri=published_tracking_uri,
    )

    completion = _completion_message(pub.client)
    assert completion["tracking_s3_uri"] == (
        f"s3://{output_bucket}/{base_key}_tracked.json"
    )


def test_tracking_uri_omitted_yields_none():
    """When no tracking URI is supplied the field is explicitly null,
    not the empty string (mirrors result_s3_uri handling)."""
    pub = _make_publisher()
    pub.publish_events(
        {"clips": []},
        uuid4(),
        fps=30.0,
        job_id="job-3",
        result_s3_uri="",
    )
    completion = _completion_message(pub.client)
    assert completion["tracking_s3_uri"] is None
