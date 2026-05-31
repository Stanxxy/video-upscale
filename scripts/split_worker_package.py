#!/usr/bin/env python3
"""Mechanically split service/worker.py into service/worker/ package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "service" / "worker.py"
PKG = ROOT / "service" / "worker"


def lines(a: int, b: int) -> str:
    """1-based inclusive line slice from source."""
    all_lines = SRC.read_text().splitlines(keepends=True)
    return "".join(all_lines[a - 1 : b])


def write(rel: str, header: str, body: str) -> None:
    path = PKG / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body)


DOC = '"""\nUnified pipeline worker package (split from service/worker.py).\n"""\n'

LOG_HEADER = """import logging

logger = logging.getLogger("service.worker")

"""

PROGRESS_HEADER = """import asyncio
import logging
import time

from service.checkpoints import WorkerStateSnapshot
from service.worker._log import logger

PARTIAL_UPLOAD_INTERVAL = 30.0
LIFECYCLE_HEARTBEAT_INTERVAL = 1.0

"""

GPU_HEADER = """import gc
import json
import re

from service.worker._log import logger

"""

HELPERS_HEADER = """from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.s3 import S3Client
from service.worker._log import logger

"""

ANALYSIS_CKPT_HEADER = """import asyncio
import json
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_upscale_window_progress
from service.jobs_store import JobsStore
from service.s3 import S3Client
from service.worker._log import logger
from service.worker.progress import _make_worker_state

"""

TRACKING_PROGRESS_HEADER = """from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_track_progress
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.s3 import S3Client
from service.worker._log import logger
from service.worker.gpu import _load_partial_tracking_dict
from service.worker.progress import _make_worker_state

"""

DETECTION_HEADER = """import asyncio
import os

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import build_track_mid_loss
from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.s3 import S3Client
from tracking_pipeline.human_verification_suspend import HumanVerificationSuspend
from service.worker._log import logger
from service.worker.gpu import _load_partial_tracking_dict
from service.worker.progress import (
    _make_worker_state,
    _tracking_progress_pct_clip,
)

"""

UPSCALE_PARSE_HEADER = """from service.worker._log import logger

"""

UPSCALE_ANALYSIS_HEADER = """import asyncio
import json
import os
import threading
import time

from service.analysis_keyspaces_enums import PipelineStage
from service.checkpoints import build_upscale_started, should_flush_analysis
from service.config import ServiceConfig
from service.jobs_store import JobsStore
from service.models import ProcessingMode, TrackRequest
from service.worker._log import logger
from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint
from service.worker.helpers import _make_s3
from service.worker.progress import _make_worker_state
from pipeline import deduplicate_clips

"""

PARALLEL_HEADER = """import asyncio
import json
import os

from service.analysis_keyspaces_enums import PipelineStage
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import TrackRequest
from service.worker._log import logger
from service.worker.progress import LIFECYCLE_HEARTBEAT_INTERVAL, _pct_at_least
from service.worker.stages.upscale_analysis import _run_upscale_analysis

"""

ORCHESTRATOR_HEADER = """import asyncio
import json
import logging
import os
import shutil
import time
from uuid import uuid4

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    END_OF_TRACKING_SENTINEL,
    build_annotate_completed,
    build_detect_initial_pending,
    build_download_completed,
    build_publish_completed,
    build_tracking_started,
    build_track_completed,
    build_track_progress,
    build_upload_incremental,
    build_verified_boxes_checkpoint,
)
from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.jobs_store import JobsStore
from service.models import JobCancelledError, JobSuspendedError, JobStatus, ProcessingMode, TrackRequest
from service.s3 import S3Client
from service.sns import SNSPublisher
from service.tracking_chain_merge import consolidate_tracking_json_with_job_chain
from service.worker._log import logger
from service.worker.callbacks.detection import _make_detection_cb
from service.worker.gpu import _ensure_models_released, _load_partial_tracking_dict
from service.worker.helpers import _is_cancelled, _make_s3
from service.worker.progress import (
    LIFECYCLE_HEARTBEAT_INTERVAL,
    PARTIAL_UPLOAD_INTERVAL,
    _clip_done_inclusive_through_global,
    _make_worker_state,
    _pct_at_least,
    _resolved_clip_end_and_total,
    _schedule_background_coro,
    _track_completed_clip_worker_state,
    _tracking_progress_flags,
    _tracking_progress_pct_clip,
    _update_tracking_progress_with_partial,
    _video_frame_cap,
)
from service.worker.stages.parallel_upscale import _run_parallel_upscale
from service.worker.stages.upscale import _parse_time_range, _run_upscale_analysis

"""


def main() -> None:
    write("_log.py", LOG_HEADER, "")

    progress_body = lines(66, 236) + "\n\n" + lines(1396, 1477)
    progress_header = PROGRESS_HEADER.replace(
        "from service.checkpoints import WorkerStateSnapshot\n",
        "import os\n\n"
        "from service.analysis_keyspaces_enums import PipelineStage\n"
        "from service.checkpoints import WorkerStateSnapshot, build_track_progress\n"
        "from service.job_store import InMemoryJobStore\n"
        "from service.jobs_store import JobsStore\n"
        "from service.models import TrackRequest\n"
        "from service.s3 import S3Client\n"
        "from service.worker.gpu import _load_partial_tracking_dict\n",
    )
    write("progress.py", progress_header, progress_body)

    write("gpu.py", GPU_HEADER, lines(238, 286))

    write("helpers.py", HELPERS_HEADER, lines(1332, 1345))

    write(
        "callbacks/analysis_checkpoint.py",
        ANALYSIS_CKPT_HEADER,
        lines(1348, 1393),
    )

    write(
        "callbacks/detection.py",
        DETECTION_HEADER,
        lines(1480, 1595),
    )

    write("stages/upscale_parse.py", UPSCALE_PARSE_HEADER, lines(2440, 2468))

    write(
        "stages/upscale_analysis.py",
        UPSCALE_ANALYSIS_HEADER,
        lines(1598, 2437),
    )

    write(
        "stages/parallel_upscale.py",
        PARALLEL_HEADER,
        lines(2487, 2728),
    )

    write("orchestrator.py", ORCHESTRATOR_HEADER, lines(292, 1326))

    init = '''"""Unified pipeline worker: detect → verify → track → upscale → upload → publish."""
from service.sns import SNSPublisher
from service.worker._log import logger
from service.worker.callbacks.analysis_checkpoint import _flush_analysis_checkpoint
from service.worker.callbacks.detection import _make_detection_cb
from service.worker.gpu import _ensure_models_released, _load_partial_tracking_dict
from service.worker.helpers import _is_cancelled, _make_s3
from service.worker.orchestrator import run_job
from service.worker.progress import (
    LIFECYCLE_HEARTBEAT_INTERVAL,
    PARTIAL_UPLOAD_INTERVAL,
    _log_progress_future_failure,
    _schedule_background_coro,
    _tracking_progress_flags,
    _update_tracking_progress,
    _update_tracking_progress_with_partial,
)
from service.worker.stages.parallel_upscale import _run_parallel_upscale
from service.worker.stages.upscale import _parse_time_range, _run_upscale_analysis

__all__ = [
    "PARTIAL_UPLOAD_INTERVAL",
    "LIFECYCLE_HEARTBEAT_INTERVAL",
    "SNSPublisher",
    "logger",
    "run_job",
    "_make_s3",
    "_parse_time_range",
    "_run_upscale_analysis",
    "_make_detection_cb",
    "_flush_analysis_checkpoint",
    "_update_tracking_progress_with_partial",
    "_update_tracking_progress",
    "_run_parallel_upscale",
    "_ensure_models_released",
    "_load_partial_tracking_dict",
    "_log_progress_future_failure",
    "_schedule_background_coro",
    "_tracking_progress_flags",
    "_is_cancelled",
]
'''
    (PKG / "__init__.py").write_text(init)

    upscale_init = '''from service.worker.stages.upscale_analysis import _run_upscale_analysis
from service.worker.stages.upscale_parse import _parse_time_range

__all__ = ["_run_upscale_analysis", "_parse_time_range"]
'''
    (PKG / "stages" / "upscale.py").write_text(upscale_init)

    # Remove monolithic module
    SRC.unlink()

    print("Split complete. Line counts:")
    for p in sorted(PKG.rglob("*.py")):
        n = len(p.read_text().splitlines())
        flag = " OVER 500" if n > 500 else ""
        print(f"  {p.relative_to(ROOT)}: {n}{flag}")


if __name__ == "__main__":
    main()
