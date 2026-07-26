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


def seconds_to_timestamp(seconds: float) -> str:
    """Convert an ALREADY match-absolute seconds value to HH:MM:SS — the v2
    (S12 Phase 1b) sibling of ``frame_to_timestamp`` above: v2 highlight
    clips carry ``start_s``/``end_s`` directly (no frame index / fps to
    divide), so this is a trivial format, never a frame-based conversion."""
    total_seconds = max(0, int(seconds or 0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


# --------------------------------------------------------------------------- #
# S12 Phase 1b — axis-only path (item 12, design §5). NEW, parallel to
# clip_to_event/publish_events above — NEVER calls dual_emit_legacy_fields.
# shared_lib 1.3.0 (installed) shipped VideoEventCandidate's axis-only
# relaxation (§5.1/§8.1) — this path is unblocked for production.
# --------------------------------------------------------------------------- #
def clip_to_axis_only_event(clip: dict, video_id: UUID, *, candidate_cls=None) -> VideoEventWithCandidates:
    """Transform ONE ``executors.highlight_analyze_node`` synthesized clip
    dict into a ``schema_version=3`` ``VideoEventWithCandidates`` — the v2
    terminal shape (design §5.2). ``start_time``/``end_time`` derive from
    the clip's OWN ``start_s``/``end_s`` (already the highlight's
    authoritative, match-absolute bounds — see ``highlight_analyze_node``'s
    docstring), never re-derived here.

    ``candidate_cls`` (testability seam, forwarded verbatim to
    ``taxonomy_mapper.build_axis_only_candidate`` — see that function's own
    docstring): defaults to the REAL shared_lib 1.3.0 ``VideoEventCandidate``,
    which already constructs successfully with axis-only ``None`` legacy
    values; tests may still inject a duck-typed stand-in to exercise the
    MAPPING logic independently of the real model's own field constraints."""
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

    # ----------------------------------------------------------------- #
    # S12 Phase 1b — v2 per-highlight publish (design §5.3). Publishes
    # INCREMENTALLY as each highlight is analyzed (one call per highlight),
    # NOT batched at end-of-job like publish_events above — lower latency
    # to the analyzer's first pending candidate, smaller blast radius on a
    # mid-job crash. Reuses the SAME boto3 client/topic_arn this instance
    # was constructed with.
    # ----------------------------------------------------------------- #
    def publish_axis_only_event(self, event: "VideoEventWithCandidates", *, event_index: int = 1) -> None:
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
        """Terminal ``analysis_complete`` for a v2 job — ``result_s3_uri``
        only (decision 3: v2 has no tracking artifact; ``tracking_s3_uri``
        is never populated, unlike ``publish_events``'s own terminal
        event above)."""
        completion = AnalysisCompleteEvent(
            video_id=video_id,
            job_id=job_id,
            total_event_count=total_event_count,
            result_s3_uri=result_s3_uri or None,
            tracking_s3_uri=None,
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
