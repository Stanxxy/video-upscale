from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, Optional, List
from uuid import UUID, uuid4
from shared_lib.models.sns_event_models import VideoEventCandidate, VideoEventWithCandidates  # noqa: F401


class AnalyzerMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class JobCancelledError(Exception):
    """Raised when tracking is stopped early via should_stop (e.g. DELETE /job/{id})."""


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


class TrackRequest(BaseModel):
    """Request body for POST /track."""
    bucket: str
    key: str
    output_bucket: Optional[str] = None
    sam2_model: str = "facebook/sam2.1-hiera-base-plus"

    # Tracking config
    detection_threshold: float = 0.5
    yolo_model: str = "yolo26m"
    start_time: Optional[str] = None  # MM:SS or HH:MM:SS
    end_time: Optional[str] = None
    box_a: Optional[List[float]] = None  # [x1,y1,x2,y2] — skip detection if provided
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

    # Player reference images for athlete identification
    player_references: Optional[List[Dict[str, str]]] = None

    # Resume from partial tracking (Phase 2 — mid-tracking checkpoint resume)
    resume_tracking_s3_key: Optional[str] = None  # S3 key for partial tracking JSON
    resume_from_frame: Optional[int] = None  # frame index to resume tracking from


class ResumeRequest(BaseModel):
    """Request body for POST /jobs/{job_id}/resume — delivers corrected bounding boxes."""
    box_a: List[float]  # [x1, y1, x2, y2] (xyxy) bounding box for athlete A
    box_b: List[float]  # [x1, y1, x2, y2] (xyxy) bounding box for athlete B
    player_mapping: Optional[Dict[str, str]] = None  # optional name mapping


class TrackResponse(BaseModel):
    """Response for POST /track."""
    job_id: str
    ws_url: str
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
    created_at: str
    updated_at: str


# SNS event models — imported from shared_lib (VideoEventCandidate, VideoEventWithCandidates)

class AnalysisCompleteEvent(BaseModel):
    video_id: UUID
    job_id: str
    total_event_count: int
    result_s3_uri: Optional[str] = None
