"""S12 Phase 1b — service/sns.py axis-only path (item 12, design §5).

shared_lib 1.3.0 (installed) shipped the relaxed VideoEventCandidate
(§5.1/§8.1) — end-to-end construction through the REAL model now succeeds.
The publish-layer methods are ALSO tested independently of that (they never
validate the ``event``/``clip`` shape themselves — pure serialize-and-publish).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional
from uuid import uuid4

import pytest
from pydantic import ValidationError

from service.sns import SNSPublisher, clip_to_axis_only_event, seconds_to_timestamp


@dataclass
class _DuckCandidate:
    role: str
    player_name: Optional[str] = None
    player_id: Optional[str] = None
    track_id: Optional[int] = None
    identity_uncertain: Optional[bool] = None
    action: Optional[str] = None
    technique: Optional[str] = None
    result: Optional[str] = None
    confidence: Optional[float] = None
    notes: str = ""
    schema_version: int = 1
    axis1_position: Optional[List[str]] = None
    axis3_action: Optional[List[str]] = None
    axis4_outcome: Optional[str] = None
    actor_sentinel: Optional[str] = None
    technique_shortlist: Optional[str] = None
    technique_guess: Optional[str] = None


class _FakeSNSClient:
    def __init__(self):
        self.published = []

    def publish(self, *, TopicArn, Message, MessageAttributes):
        self.published.append(
            {"TopicArn": TopicArn, "Message": Message, "MessageAttributes": MessageAttributes},
        )
        return {"MessageId": "fake"}


def _make_publisher() -> SNSPublisher:
    pub = SNSPublisher.__new__(SNSPublisher)
    pub.client = _FakeSNSClient()
    pub.topic_arn = "arn:aws:sns:us-east-1:000000000000:bjj-events"
    return pub


def _clip(**overrides) -> dict:
    base = {
        "start_s": 65.0, "end_s": 130.0,
        "position": "mount", "action_class": "submission_arm_lock", "outcome": "successful",
        "player_id": "p1", "player_name": "Alice",
        "identity_uncertain": False, "actor_sentinel": None,
        "notes": "j | j | e",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# seconds_to_timestamp — pure, always testable.
# --------------------------------------------------------------------------- #
def test_seconds_to_timestamp_basic():
    assert seconds_to_timestamp(65.0) == "00:01:05"
    assert seconds_to_timestamp(3661.0) == "01:01:01"
    assert seconds_to_timestamp(0.0) == "00:00:00"


def test_seconds_to_timestamp_none_defaults_to_zero():
    assert seconds_to_timestamp(None) == "00:00:00"


# --------------------------------------------------------------------------- #
# clip_to_axis_only_event — mapping logic (real shared_lib 1.3.0 class).
# UNDERLYING logic (seconds_to_timestamp, build_axis_only_candidate's own
# field mapping) is independently covered above / in
# tests/test_taxonomy_mapper_axis_only.py.
# --------------------------------------------------------------------------- #
def test_clip_to_axis_only_event_duck_candidate_rejected_by_outer_model():
    """VideoEventWithCandidates.event_candidates: List[VideoEventCandidate]
    is strictly typed against the real shared_lib class — a duck-typed
    candidate_cls is rejected at the OUTER model boundary too (pydantic
    model_type check), independent of shared_lib's axis-only relaxation."""
    with pytest.raises(ValidationError, match="VideoEventCandidate"):
        clip_to_axis_only_event(_clip(), uuid4(), candidate_cls=_DuckCandidate)


def test_clip_to_axis_only_event_real_construction_succeeds():
    event = clip_to_axis_only_event(_clip(), uuid4())
    assert event.start_time == "00:01:05"
    assert event.end_time == "00:02:10"
    assert event.event_candidates[0].schema_version == 3
    assert event.event_candidates[0].action is None


# --------------------------------------------------------------------------- #
# 2026-07-26 single-call cutover (AC2/AC3) — zero dual-emit, schema_version=3
# axis-only for every event on the single-call path: no `action`/`technique`/
# `result`/`confidence` legacy keys anywhere in the outgoing SNS payload.
# --------------------------------------------------------------------------- #
def test_clip_to_axis_only_event_publish_payload_has_no_legacy_keys():
    """Real end-to-end shape check: build the event from a single-call-shaped
    clip dict, dump it exactly as SNSPublisher.publish_axis_only_event would
    serialize it, and confirm no `action`/`technique`/`result`/`confidence`
    key carries a non-None/legacy value anywhere in the candidate payload."""
    event = clip_to_axis_only_event(_clip(), uuid4())
    message = event.model_dump(mode="json")

    assert message["event_candidates"][0]["schema_version"] == 3
    for legacy_key in ("action", "technique", "result", "confidence"):
        assert message["event_candidates"][0][legacy_key] is None

    # Round-trips through the EXACT publisher serialization path too (never
    # a second, divergent serializer for the real vs. test-inspected shape).
    publisher = _make_publisher()
    publisher.publish_axis_only_event(event, event_index=1)
    published_message = json.loads(publisher.client.published[0]["Message"])
    assert published_message["event_candidates"][0]["schema_version"] == 3
    for legacy_key in ("action", "technique", "result", "confidence"):
        assert published_message["event_candidates"][0][legacy_key] is None


# --------------------------------------------------------------------------- #
# SNSPublisher.publish_axis_only_event / publish_analysis_complete — never
# validate the event/clip shape themselves (serialize-and-publish only), so
# these are fully testable against 1.2.0 with a lightweight duck event.
# --------------------------------------------------------------------------- #
def test_publish_axis_only_event_publishes_bjj_event_detected():
    publisher = _make_publisher()
    duck_event = SimpleNamespace(model_dump=lambda mode="json": {"video_id": "v1", "event_candidates": []})

    publisher.publish_axis_only_event(duck_event, event_index=3)

    assert len(publisher.client.published) == 1
    call = publisher.client.published[0]
    assert call["MessageAttributes"]["event_type"]["StringValue"] == "bjj_event_detected"
    assert call["MessageAttributes"]["event_index"]["StringValue"] == "3"
    assert json.loads(call["Message"]) == {"video_id": "v1", "event_candidates": []}


def test_publish_analysis_complete_omits_tracking_s3_uri():
    publisher = _make_publisher()

    publisher.publish_analysis_complete(
        uuid4(), "job-1", total_event_count=5, result_s3_uri="s3://bucket/base_key_v2_events.json",
    )

    assert len(publisher.client.published) == 1
    call = publisher.client.published[0]
    assert call["MessageAttributes"]["event_type"]["StringValue"] == "analysis_complete"
    body = json.loads(call["Message"])
    assert body["result_s3_uri"] == "s3://bucket/base_key_v2_events.json"
    assert body["tracking_s3_uri"] is None
    assert body["total_event_count"] == 5


def test_publish_analysis_complete_no_result_uri_stays_none():
    publisher = _make_publisher()

    publisher.publish_analysis_complete(uuid4(), "job-1", total_event_count=0)

    body = json.loads(publisher.client.published[0]["Message"])
    assert body["result_s3_uri"] is None
    assert body["tracking_s3_uri"] is None
