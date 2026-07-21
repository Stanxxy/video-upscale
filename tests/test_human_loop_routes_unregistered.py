"""S12 Phase 1b — human_loop.py routes are UNREGISTERED from the production
app surface (item 16, design §6.4/decision 2). These three paths must now
404 (route-not-found), not 200/500 — the handler functions themselves stay
importable in service/routes/human_loop.py for direct unit testing or a
future SAM3-scout-gated revival.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_get_detection_frame_route_404s(service_client):
    resp = await service_client.get("/jobs/some-job/detection_frame")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_post_detection_response_route_404s(service_client):
    resp = await service_client.post(
        "/jobs/some-job/detection_response",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_post_resume_route_404s(service_client):
    resp = await service_client.post(
        "/jobs/some-job/resume",
        json={"box_a": [0, 0, 1, 1], "box_b": [0, 0, 1, 1]},
    )
    assert resp.status_code == 404


def test_human_loop_handlers_remain_importable_for_direct_unit_testing():
    """Fencing discipline (design §6.4): route UNREGISTRATION, never
    deletion — the handlers stay importable."""
    from service.routes.human_loop import (
        get_detection_frame,
        resume_job,
        submit_detection_response,
    )

    assert callable(get_detection_frame)
    assert callable(resume_job)
    assert callable(submit_detection_response)
