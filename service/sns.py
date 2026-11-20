import boto3
import json
import logging
from uuid import UUID
from service.models import AnalysisCompleteEvent, VideoEventWithCandidates
from service import taxonomy_mapper

logger = logging.getLogger(__name__)


def seconds_to_timestamp(seconds: float) -> str:
    """Convert an ALREADY match-absolute seconds value to HH:MM:SS (highlight
    clips carry ``start_s``/``end_s`` directly)."""
    total_seconds = max(0, int(seconds or 0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def clip_to_axis_only_event(clip: dict, video_id: UUID, *, candidate_cls=None) -> VideoEventWithCandidates:
    """Transform ONE ``executors.highlight_analyze_node`` synthesized clip
    dict into a ``schema_version=3`` ``VideoEventWithCandidates`` — the v2
    terminal shape (design §5.2). ``start_time``/``end_time`` derive from
    the clip's OWN ``start_s``/``end_s`` (already the highlight's
    authoritative, match-absolute bounds — see ``highlight_analyze_node``'s
    docstring), never re-derived here.

    ``candidate_cls`` (testability seam, forwarded verbatim to
    ``taxonomy_mapper.build_axis_only_candidate``): defaults to the real
    shared_lib ``VideoEventCandidate``; tests may inject a duck-typed
    stand-in to exercise the mapping logic independently of the model."""
    candidate = taxonomy_mapper.build_axis_only_candidate(
        clip, str(video_id), candidate_cls=candidate_cls,
    )
    return VideoEventWithCandidates(
        video_id=video_id,
        start_time=seconds_to_timestamp(clip.get("start_s", 0.0)),
        end_time=seconds_to_timestamp(clip.get("end_s", 0.0)),
        event_candidates=[candidate],
    )


class SNSPublisher:
    def __init__(
        self,
        region: str,
        topic_arn: str,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        kwargs = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if access_key_id:
            kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["aws_secret_access_key"] = secret_access_key
        self.client = boto3.client("sns", **kwargs)
        self.topic_arn = topic_arn

    def publish_axis_only_event(self, event: "VideoEventWithCandidates", *, event_index: int = 1) -> None:
        """Publish ONE highlight's candidate event as it is analyzed (incremental,
        not batched at end-of-job)."""
        message = event.model_dump(mode="json")
        self.client.publish(
            TopicArn=self.topic_arn,
            Message=json.dumps(message, default=str),
            MessageAttributes={
                "event_type": {
                    "DataType": "String",
                    "StringValue": "bjj_event_detected",
                },
                "event_index": {
                    "DataType": "Number",
                    "StringValue": str(event_index),
                },
            },
        )

    def publish_analysis_complete(
        self,
        video_id: UUID,
        job_id: str,
        total_event_count: int,
        result_s3_uri: str | None = None,
    ) -> None:
        """Terminal ``analysis_complete`` for a v2 job."""
        completion = AnalysisCompleteEvent(
            video_id=video_id,
            job_id=job_id,
            total_event_count=total_event_count,
            result_s3_uri=result_s3_uri or None,
        )
        self.client.publish(
            TopicArn=self.topic_arn,
            Message=json.dumps(completion.model_dump(mode="json"), default=str),
            MessageAttributes={
                "event_type": {
                    "DataType": "String",
                    "StringValue": "analysis_complete",
                },
            },
        )
