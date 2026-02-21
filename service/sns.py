import boto3
import json
from uuid import UUID, uuid4
from service.models import VideoEventWithCandidates, VideoEventCandidate, AnalysisCompleteEvent


def frame_to_timestamp(frame: int, fps: float) -> str:
    """Convert frame number to HH:MM:SS timestamp."""
    if fps <= 0:
        return "00:00:00"
    total_seconds = int(frame / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def clip_to_event(clip: dict, video_id: UUID, fps: float) -> VideoEventWithCandidates:
    """Transform a pipeline clip dict into a VideoEventWithCandidates."""
    candidate = VideoEventCandidate(
        role=clip.get("role", "Unknown"),
        skill_name=clip.get("specific_technique", "Unknown"),
        category=clip.get("category", "UNKNOWN"),
        confidence=clip.get("confidence", 0.0),
    )
    return VideoEventWithCandidates(
        video_id=video_id,
        start_time=frame_to_timestamp(clip.get("start_frame", 0), fps),
        end_time=frame_to_timestamp(clip.get("end_frame", 0), fps),
        event_candidates=[candidate],
    )


class SNSPublisher:
    def __init__(self, region: str, topic_arn: str):
        self.client = boto3.client("sns", region_name=region)
        self.topic_arn = topic_arn

    def publish_events(
        self, analysis: dict, video_id: UUID, fps: float,
        job_id: str = "", result_s3_uri: str = "",
    ) -> int:
        """Publish each clip as a VideoEventWithCandidates SNS message,
        followed by an analysis_complete boundary event. Returns clip count."""
        clips = analysis.get("clips", [])
        total = len(clips)
        count = 0

        for idx, clip in enumerate(clips, start=1):
            event = clip_to_event(clip, video_id, fps)
            message = event.model_dump(mode="json")
            self.client.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(message, default=str),
                MessageAttributes={
                    "category": {
                        "DataType": "String",
                        "StringValue": clip.get("category", "UNKNOWN"),
                    },
                    "event_type": {
                        "DataType": "String",
                        "StringValue": "bjj_event_detected",
                    },
                    "event_index": {
                        "DataType": "Number",
                        "StringValue": str(idx),
                    },
                    "total_events": {
                        "DataType": "Number",
                        "StringValue": str(total),
                    },
                },
            )
            count += 1

        # Publish completion boundary event
        completion = AnalysisCompleteEvent(
            video_id=video_id,
            job_id=job_id,
            total_event_count=total,
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

        return count
