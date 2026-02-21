from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timezone


class AnalyzerMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRequest(BaseModel):
    video_id: UUID
    video_s3_bucket: str
    video_s3_key: str
    tracking_json_s3_bucket: str
    tracking_json_s3_key: str
    output_s3_bucket: str
    output_s3_key: str
    sns_topic_arn: Optional[str] = None
    sampling_rate: int = Field(default=1, ge=1)
    analyzer_mode: AnalyzerMode = AnalyzerMode.SINGLE
    method: str = "esrgan"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    result_s3_uri: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class VideoEventCandidate(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: str
    skill_name: str
    category: str
    confidence: float


class VideoEventWithCandidates(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    video_id: UUID
    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    event_candidates: List[VideoEventCandidate]


class AnalysisCompleteEvent(BaseModel):
    video_id: UUID
    job_id: str
    total_event_count: int
    result_s3_uri: Optional[str] = None
