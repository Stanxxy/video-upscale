"""Shared state passed between pipeline stage runners."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.s3 import S3Client


@dataclass
class WorkerRunContext:
    job_id: str
    request: TrackRequest
    config: ServiceConfig
    job_store: InMemoryJobStore
    jobs_store: JobsStore
    work_dir: str
    loop: asyncio.AbstractEventLoop
    progress_floor: float = 0.0
    clip_start_frame: int = 0
    clip_end_resolved: int = 1
    clip_total_frames: int = 1
    tracking_start_frame: int = 0
    end_frame: int | None = None
    video_path: str | None = None
    s3: S3Client | None = None
    box_a: list | None = None
    box_b: list | None = None
    tracking_json_path: str | None = None
    tracking_output_dir: str | None = None
    frame_count: int = 0
    use_parallel_upscale: bool = False
    k_needed: int = 1
    output_bucket: str | None = None
    base_key: str | None = None
    tracking_result_key: str | None = None
    analysis_result_key: str | None = None
    annotated_video_key: str | None = None
    analysis_result: dict | None = None
    fps: float | None = None
    annotated_video_path: str | None = None
    event_count: int = 0
    pipeline_complete: bool = field(default=False, repr=False)
