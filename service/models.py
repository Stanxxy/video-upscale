from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum
from typing import Dict, List, Literal, Optional
from uuid import UUID
from shared_lib.models.sns_event_models import VideoEventCandidate, VideoEventWithCandidates  # noqa: F401


class AnalyzerMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class AthleteBinding(BaseModel):
    """Reference crop for one athlete; Gemini grounds ``actor_player_id`` to it.

    ``track_id`` is still sent by the backend; it is accepted and unused.
    """
    player_id: str
    player_name: str
    s3_key: Optional[str] = None  # player-references/<vid>/<pid>.jpg
    track_id: Optional[int] = None


class JobStatus(str, Enum):
    """In-memory job status for jobs this process started."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrackRequest(BaseModel):
    """Request body for POST /track."""
    bucket: str
    key: str
    output_bucket: Optional[str] = None
    user_id: Optional[str] = None
    start_time: Optional[str] = None  # MM:SS or HH:MM:SS
    end_time: Optional[str] = None
    video_id: Optional[UUID] = None
    athlete_bindings: Optional[List[AthleteBinding]] = None
    sampling_rate: int = Field(default=1, ge=1)
    analyzer_mode: AnalyzerMode = AnalyzerMode.SINGLE
    sns_topic_arn: Optional[str] = None

    # R4: optional user overrides. Qualification/default policy remains owned
    # by service.analysis_settings; these transport fields intentionally do
    # not embed a second allowlist in the request schema.
    analysis_model: Optional[str] = None
    analysis_media_resolution: Optional[str] = None
    analysis_fps: Optional[int] = None
    analysis_thinking: Optional[str] = None


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


class TrackResponse(BaseModel):
    """Response for POST /track."""
    job_id: str
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
