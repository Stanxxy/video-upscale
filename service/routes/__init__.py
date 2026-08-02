"""FastAPI route package — preserves ``from service.routes import router, init_routes``."""

from fastapi import APIRouter

from service.models import AnalysisSettingsCapabilities, TrackResponse, JobResponse
from service.routes.health import health, qa_client
from service.routes.jobs import cancel_job, create_track_job, get_job
from service.routes.analysis_settings import get_analysis_settings
from service.routes.active_jobs import get_active_jobs, ActiveJobsResponse
from service.routes.events import job_events_sse
# S12 Phase 1b (design §6.4, item 16): human_loop.py's three routes are
# UNREGISTERED from the production app surface — decision 2, "removed from
# production" means the HTTP surface literally 404s. The handler functions
# (get_detection_frame/resume_job/submit_detection_response) stay in
# human_loop.py, fully importable for direct unit testing or a future SAM3-
# scout-gated revival; they are simply never imported/wired here anymore.
from service.routes.recovery import (
    bootstrap_recovery_on_startup,
    drain_orphan_pending_jobs_on_startup,
    recover_interrupted_job,
)
from service.routes.qa import identity_analyze
from service.routes.qa_vlm import vlm_chat, vlm_image
from service.routes.state import init_routes

router = APIRouter()

# QA client
router.get("/", include_in_schema=False)(qa_client)

# Track job CRUD
# TODO: user id needs to be passed in from the request. update TrackRequest model to include user_id.
# The track request should ONLY be used for starting a new job. NOT for resuming a job.
router.post("/track", response_model=TrackResponse)(create_track_job)
router.get("/job/{job_id}", response_model=JobResponse)(get_job)
router.delete("/job/{job_id}")(cancel_job)
router.get(
    "/analysis-settings/capabilities",
    response_model=AnalysisSettingsCapabilities,
)(get_analysis_settings)

# Active jobs list — read-only, used by the release GPU-guard
router.get("/jobs/active", response_model=ActiveJobsResponse)(get_active_jobs)

# SSE
router.get("/jobs/{job_id}/events")(job_events_sse)

# Human-in-the-loop (S12 Phase 1b, design §6.4) — UNREGISTERED, not deleted.
# These three paths now 404 (route-not-found) in production; the dormant
# tracking pipeline's correction/detect-suspend flow has no production
# entrypoint left. See service/routes/human_loop.py for the still-importable
# handlers and tests/test_human_loop_routes_unregistered.py for the 404
# regression guard.

# Health
router.get("/health")(health)

# QA harness — Stream 2 Gemini identity-grounding live gate (analyzer only, no GPU)
router.post("/qa/identity-analyze")(identity_analyze)

# QA VLM Studio — pure-Gemini event chat + single-frame image generation/segmentation
router.post("/qa/vlm-chat")(vlm_chat)
router.post("/qa/vlm-image")(vlm_image)


def __getattr__(name: str):
    if name == "_schedule_job":
        from service.routes.scheduling import _schedule_job

        return _schedule_job
    if name == "_cleanup_orphaned_tasks":
        from service.routes.scheduling import _cleanup_orphaned_tasks

        return _cleanup_orphaned_tasks
    if name == "_job_semaphore":
        from service.routes import state as route_state

        return route_state._job_semaphore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "router",
    "init_routes",
    "recover_interrupted_job",
    "drain_orphan_pending_jobs_on_startup",
    "bootstrap_recovery_on_startup",
    "_schedule_job",
    "_cleanup_orphaned_tasks",
    "_job_semaphore",
]
