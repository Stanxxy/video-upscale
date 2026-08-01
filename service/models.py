from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum
from typing import Dict, Literal, Optional, List
from uuid import UUID, uuid4
from shared_lib.models.sns_event_models import VideoEventCandidate, VideoEventWithCandidates  # noqa: F401


class AnalyzerMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class ProcessingMode(str, Enum):
    """Processing mode controlling speed/quality tradeoff.

    STANDARD: full-quality SAM2 base-plus + RealESRGAN + RTMPose + sequential Gemini.
    FAST: SAM2-tiny + propagation stride + bicubic upscale + no pose + parallel Gemini.
    """
    STANDARD = "standard"
    FAST = "fast"


class JobCancelledError(Exception):
    """Raised when tracking is stopped early via should_stop (e.g. DELETE /job/{id})."""


class JobSuspendedError(Exception):
    """Raised when the job needs human input (bounding box correction).

    The worker catches this to cleanly release models and persist
    checkpoint state to Keyspaces + S3 before exiting.
    """


class AthleteBinding(BaseModel):
    """Canonical, persisted form of the human-confirmed track_id↔player_id binding.

    Produced by the bounding-box / point-click athlete correction model
    (``player_mapping`` is the same binding one hop upstream). This is the
    authoritative identity key carried through tracking → analysis → SNS so
    track_ids never flip and Gemini grounds ``actor_player_id`` to a real player.
    """
    track_id: int
    player_id: str
    player_name: str
    box: Optional[List[float]] = None  # [x1,y1,x2,y2] for SAM2 init_boxes seeding
    s3_key: Optional[str] = None  # player-references/<vid>/<pid>.jpg


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DETECTING = "detecting"
    WAITING_FOR_DETECTION = "waiting_for_detection"
    TRACKING = "tracking"
    UPSCALING = "upscaling"
    UPLOADING = "uploading"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_CORRECTION = "awaiting_correction"
    INTERRUPTED = "interrupted"


class TrackRequest(BaseModel):
    """Request body for POST /track."""
    bucket: str
    key: str
    output_bucket: Optional[str] = None
    user_id: Optional[str] = None
    sam2_model: str = "facebook/sam2.1-hiera-base-plus"

    # Tracking config
    detection_threshold: float = 0.5
    yolo_model: str = "yolo26m"
    start_time: Optional[str] = None  # MM:SS or HH:MM:SS
    end_time: Optional[str] = None
    # LEGACY: superseded by athlete_bindings (track_id↔player_id). Remove once all paths consume bindings.
    box_a: Optional[List[float]] = None  # [x1,y1,x2,y2] — skip detection if provided
    # LEGACY: superseded by athlete_bindings (track_id↔player_id). Remove once all paths consume bindings.
    box_b: Optional[List[float]] = None  # [x1,y1,x2,y2] — skip detection if provided
    step_size: Optional[int] = None
    max_history: Optional[int] = None
    max_missing_frames: Optional[int] = None

    # Upscale + analysis config (optional)
    skip_upscale: bool = False
    video_id: Optional[UUID] = None
    sampling_rate: int = Field(default=1, ge=1)
    analyzer_mode: AnalyzerMode = AnalyzerMode.SINGLE
    method: str = "esrgan"
    sns_topic_arn: Optional[str] = None

    # R4: optional user overrides. Qualification/default policy remains owned
    # by service.analysis_settings; these transport fields intentionally do
    # not embed a second allowlist in the request schema.
    analysis_model: Optional[str] = None
    analysis_media_resolution: Optional[str] = None
    analysis_fps: Optional[int] = None
    analysis_thinking: Optional[str] = None

    # LEGACY: superseded by athlete_bindings (track_id↔player_id). Remove once all paths consume bindings.
    # Unlabelled refs in request order; no track_id/player_id pairing. Use athlete_bindings instead.
    player_references: Optional[List[Dict[str, str]]] = None

    # Canonical human-confirmed identity binding (track_id↔player_id↔box). When present this is the
    # single source of truth: derive init_boxes and labelled refs from it. N-athlete ready.
    athlete_bindings: Optional[List[AthleteBinding]] = None

    # Stream 0b: human-confirmed obj_id -> player_id binding, one hop upstream of
    # athlete_bindings. { "1": player_id_A, "2": player_id_B }. AUTHORITATIVE
    # track_id<->player_id on the resume path. When present (and athlete_bindings
    # is absent) init_boxes is seeded from this confirmed binding (obj_id "1" ->
    # box_a, "2" -> box_b) so track_ids never flip across the job chain. Threaded
    # from ResumeRequest into the replacement TrackRequest by build_resume_params.
    player_mapping: Optional[Dict[str, str]] = None

    # M2: stride-N frame sampling (S11)
    # 0 = auto: max(1, round(fps / 10)) computed from detected source fps.
    # e.g. 60fps → stride=6 (keeps every 6th frame, ~300 frames from 1800)
    # e.g. 30fps → stride=3
    frame_stride: int = Field(default=0, ge=0, description="Track/upscale every Nth frame. 0=auto from fps.")

    # M4: processing mode (fast vs standard)
    processing_mode: ProcessingMode = ProcessingMode.STANDARD

    # Resume from checkpoint (Keyspaces-backed suspend/resume)
    resume_from_job_id: Optional[str] = None  # job_id to load checkpoints from Keyspaces
    resume_tracking_s3_key: Optional[str] = None  # S3 key for partial tracking JSON
    resume_from_frame: Optional[int] = None  # frame index to resume tracking from
    # Stage 4 resume hints (analysis window checkpoint)
    analysis_raw_s3_key: Optional[str] = None
    analysis_window_count: Optional[int] = None
    analysis_current_context: Optional[str] = None

    # Recovery hints — populated from prior job checkpoints when replacing INTERRUPTED runs
    resume_existing_upload_tracking_key: Optional[str] = None
    resume_existing_upload_analysis_key: Optional[str] = None
    resume_existing_upload_annotated_key: Optional[str] = None
    resume_terminal_publish_done: bool = False


class AnalysisSettingsRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    media_resolution: str
    analysis_fps: int
    thinking: str


class AnalysisSettingsMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: tuple[str, ...]
    media_resolution: tuple[str, ...]
    thinking: tuple[str, ...]
    analysis_fps: tuple[str, ...]


class AnalysisSettingsCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    qualified_models: tuple[str, ...]
    media_resolutions: tuple[str, ...]
    analysis_fps: tuple[int, ...]
    thinking_levels: tuple[str, ...]
    recommended: AnalysisSettingsRecommendation
    mapping: AnalysisSettingsMapping


class EffectiveAnalysisStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    media_resolution: str
    thinking: str
    fps: Optional[int] = None


class EffectiveAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    scan: EffectiveAnalysisStageConfig
    critique: EffectiveAnalysisStageConfig
    taxonomy: EffectiveAnalysisStageConfig
    actor: EffectiveAnalysisStageConfig

    @model_validator(mode="after")
    def enforce_global_mapping(self) -> "EffectiveAnalysisConfig":
        """Reject persisted snapshots that violate the public R4 mapping."""
        stages = (self.scan, self.critique, self.taxonomy, self.actor)
        for field_name in ("model", "media_resolution", "thinking"):
            if len({getattr(stage, field_name) for stage in stages}) != 1:
                raise ValueError(f"{field_name} must be global across every analysis stage")

        if self.scan.fps is not None or self.critique.fps is not None:
            raise ValueError("analysis_fps may target taxonomy and actor only")
        if self.taxonomy.fps is None or self.taxonomy.fps != self.actor.fps:
            raise ValueError("taxonomy and actor analysis_fps must be identical")
        return self


class AdmittedTrackRequest(TrackRequest):
    """Engine-resolved request persisted and scheduled after admission."""

    capability_schema_version: Literal[1]
    requested_analysis_settings: Dict[str, str | int]
    effective_analysis_config: EffectiveAnalysisConfig


class ResumeRequest(BaseModel):
    """Request body for POST /jobs/{job_id}/resume — delivers corrected bounding boxes."""
    # LEGACY: box_a/box_b superseded by player_mapping (track_id<->player_id). Remove once all paths consume bindings.
    box_a: List[float]  # [x1, y1, x2, y2] (xyxy) bounding box for athlete A
    box_b: List[float]  # [x1, y1, x2, y2] (xyxy) bounding box for athlete B
    # Stream 0b: human-confirmed obj_id -> player_id binding
    # { "1": player_id_A, "2": player_id_B }. AUTHORITATIVE track_id<->player_id;
    # threaded onto the replacement TrackRequest and used to seed init_boxes
    # (obj_id "1" -> box_a, "2" -> box_b) so track_ids never flip on resume.
    player_mapping: Optional[Dict[str, str]] = None


class TrackResponse(BaseModel):
    """Response for POST /track."""
    job_id: str
    ws_url: str = ""  # TODO: deprecated — kept for backward compat; SSE is the new transport
    status: str = "pending"


class JobResponse(BaseModel):
    """Response for GET /job/{job_id}."""
    job_id: str
    status: str
    progress_percent: float = 0.0
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    result_bucket: Optional[str] = None
    result_key: Optional[str] = None
    error_message: Optional[str] = None
    capability_schema_version: Optional[int] = None
    requested_analysis_settings: Optional[Dict[str, str | int]] = None
    effective_analysis_config: Optional[EffectiveAnalysisConfig] = None
    created_at: str
    updated_at: str


# SNS event models — imported from shared_lib (VideoEventCandidate, VideoEventWithCandidates)

class AnalysisCompleteEvent(BaseModel):
    video_id: UUID
    job_id: str
    total_event_count: int
    result_s3_uri: Optional[str] = None
    tracking_s3_uri: Optional[str] = None
