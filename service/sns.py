import boto3
import json
import logging
from uuid import UUID, uuid4
from service.models import VideoEventWithCandidates, VideoEventCandidate, AnalysisCompleteEvent
from service import taxonomy_mapper

logger = logging.getLogger(__name__)


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Convert frame index to HH:MM:SS timestamp."""
    if fps <= 0:
        return "00:00:00"
    total_seconds = int(frame_idx / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _bindings_by_player_id(athlete_bindings) -> dict:
    """Index athlete_bindings by player_id for O(1) track_id/name resolution."""
    out = {}
    for b in (athlete_bindings or []):
        # Support both AthleteBinding pydantic objects and plain dicts.
        pid = getattr(b, "player_id", None) if not isinstance(b, dict) else b.get("player_id")
        if pid:
            out[pid] = b
    return out


def clip_to_event(
    clip: dict, video_id: UUID, fps: float, athlete_bindings=None,
) -> VideoEventWithCandidates:
    """Transform a pipeline clip dict into a VideoEventWithCandidates.

    Identity is grounded: Gemini emits ``actor_player_id`` (a real player_id from the
    confirmed bindings). We resolve track_id/player_name from the binding. gi-color /
    top-bottom live only in ``reasoning`` and become the ``role`` *descriptor*.

    Simplified-4-axis-taxonomy adoption (T2, 2026-07-12, D5): ``clip`` now carries
    ``axis1_position``/``axis3_action``/``axis4_outcome``/``technique_guess`` (Gemini's
    ENUM-constrained ``analyzer._build_response_schema`` output) instead of the old
    free-string ``action``/``technique``. This is the T2 DUAL-EMIT seam (plan §7 risk
    #2): the emitted ``VideoEventCandidate`` carries BOTH the new axis fields
    (``schema_version=2``) AND legacy ``action``/``technique``/``result`` — derived
    from the new axes via ``taxonomy_mapper.dual_emit_legacy_fields``, never read
    directly off the clip dict — so the backend's legacy-enum search validators
    (``event_filter_utils.py``) never see an out-of-enum value.
    """
    axis1_position = taxonomy_mapper.sanitize_axis1_position(clip.get("axis1_position"))
    axis3_action = taxonomy_mapper.sanitize_axis3_action(clip.get("axis3_action"))
    axis4_outcome = taxonomy_mapper.sanitize_axis4_outcome(clip.get("axis4_outcome"))
    technique_guess = clip.get("technique_guess") or None
    technique_shortlist = taxonomy_mapper.resolve_technique_shortlist(technique_guess, axis3_action)

    legacy = taxonomy_mapper.dual_emit_legacy_fields(axis3_action, axis4_outcome, technique_shortlist)
    action = legacy["action"]
    technique = legacy["technique"]
    result = legacy["result"]

    # D6 (2026-07-12 taxonomy adoption, no eval gate / flip-and-monitor): `transition`
    # (axis3) and `scramble` (axis1) are LeCun's named "new slop-bucket" regression
    # signal — log-queryable is the whole requirement, no dashboard needed.
    if "transition" in axis3_action:
        logger.info("taxonomy_monitor: axis3_action includes 'transition' video_id=%s", video_id)
    if "scramble" in axis1_position:
        logger.info("taxonomy_monitor: axis1_position includes 'scramble' video_id=%s", video_id)

    by_pid = _bindings_by_player_id(athlete_bindings)
    player_id = clip.get("actor_player_id")
    binding = by_pid.get(player_id)
    if binding is not None and isinstance(binding, dict):
        player_name = binding.get("player_name")
        track_id = binding.get("track_id")
    elif binding is not None:
        player_name = getattr(binding, "player_name", None)
        track_id = getattr(binding, "track_id", None)
    else:
        player_name = None
        track_id = None

    # role becomes a descriptor only (truncated reasoning), never the identity.
    role_descriptor = (clip.get("reasoning", "") or "")[:80] or "Unknown"

    candidate = VideoEventCandidate(
        role=role_descriptor,
        player_name=player_name,
        action=action,
        technique=technique,
        result=result,
        confidence=clip.get("confidence", 0.0),
        notes=technique_guess or "",
        schema_version=2,
        axis1_position=axis1_position,
        axis3_action=axis3_action,
        axis4_outcome=axis4_outcome,
        technique_shortlist=technique_shortlist,
        technique_guess=technique_guess,
    )
    # Grounded identity fields. The installed shared VideoEventCandidate may not yet
    # declare these (Stream 3); stash them so publish_events can inject the keys into
    # the emitted JSON, matching the backend listener / from_dynamodb_dict contract.
    candidate._grounded_identity = {  # type: ignore[attr-defined]
        "player_id": player_id,
        "track_id": track_id,
        "identity_uncertain": bool(clip.get("identity_uncertain", False)),
    }

    return VideoEventWithCandidates(
        video_id=video_id,
        start_time=frame_to_timestamp(clip.get("start_frame", 0), fps),
        end_time=frame_to_timestamp(clip.get("end_frame", 0), fps),
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

    def publish_events(
        self, analysis: dict, video_id: UUID, fps: float,
        job_id: str = "", result_s3_uri: str = "",
        tracking_s3_uri: str = "",
        athlete_bindings=None,
    ) -> int:
        """Publish each clip as a VideoEventWithCandidates SNS message,
        followed by an analysis_complete boundary event. Returns clip count."""
        clips = analysis.get("clips", [])
        total = len(clips)
        count = 0

        for idx, clip in enumerate(clips, start=1):
            event = clip_to_event(clip, video_id, fps, athlete_bindings=athlete_bindings)
            message = event.model_dump(mode="json")
            # Inject grounded identity keys onto each candidate so the emitted JSON matches
            # the (Stream 3) VideoEventCandidate contract: player_id, track_id, identity_uncertain.
            # The backend listener / from_dynamodb_dict consumes these directly.
            for cand_model, cand_json in zip(
                event.event_candidates, message.get("event_candidates", [])
            ):
                grounded = getattr(cand_model, "_grounded_identity", None)
                if grounded:
                    cand_json["player_id"] = grounded.get("player_id")
                    cand_json["track_id"] = grounded.get("track_id")
                    cand_json["identity_uncertain"] = grounded.get("identity_uncertain")
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
            tracking_s3_uri=tracking_s3_uri or None,
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
