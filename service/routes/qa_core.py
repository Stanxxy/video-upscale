"""VLM Studio routes for the production-core runner (``service.qa_core``)."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from service import qa_core
from service.routes.qa_vlm import _require_gemini_key

_runs: Optional[qa_core.CoreRuns] = None


def _core() -> qa_core.CoreRuns:
    global _runs
    if _runs is None:
        try:
            settings = qa_core.Settings.from_env(_require_gemini_key())
            _runs = qa_core.CoreRuns(settings)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from None
    return _runs


def _job(run_id):
    core = _core()
    if run_id not in core.jobs:
        raise HTTPException(status_code=404, detail="unknown run")
    return core


class EstimateIn(BaseModel):
    sport: str
    start_ms: int
    end_ms: int
    overrides: dict[str, qa_core.NodeOverride] = {}


async def core_estimate(body: EstimateIn):
    if body.sport not in qa_core.SPORTS:
        raise HTTPException(status_code=400, detail="sport must be bjj or mma")
    if body.end_ms <= body.start_ms:
        raise HTTPException(status_code=400, detail="scope end must follow its start")
    try:
        config = qa_core.build_config(body.sport, body.overrides)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return dict(config=config.model_dump(mode="json"), scans=qa_core.estimate(config, body.start_ms, body.end_ms))


async def core_media(youtube_id: str, download: bool = False):
    core = _core()
    try:
        path = await asyncio.to_thread(qa_core.media_path, core.settings, youtube_id, download=download)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    if not path.exists():
        return dict(youtube_id=youtube_id, ready=False)
    duration = await asyncio.to_thread(qa_core.probe_duration_ms, path)
    return dict(youtube_id=youtube_id, ready=True, duration_ms=duration, path=str(path))


async def core_frame(youtube_id: str, t_ms: int = Query(ge=0)):
    core = _core()
    path = qa_core.media_path(core.settings, youtube_id, download=False)
    if not path.exists():
        raise HTTPException(status_code=409, detail="fetch the video first")
    try:
        png = await asyncio.to_thread(qa_core.crop_png, path, t_ms, (0, 0, 1, 1), full=True)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"no frame: {error}") from None
    return Response(content=png, media_type="image/png")


async def core_video(youtube_id: str):
    core = _core()
    try:
        path = qa_core.media_path(core.settings, youtube_id, download=False)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    if not path.exists():
        raise HTTPException(status_code=404, detail="fetch the video first")
    # The exact file the run analyses (range requests for seeking).
    return FileResponse(path, media_type="video/mp4")


async def core_create(body: qa_core.CoreRunIn):
    core = _core()
    if body.end_ms <= body.start_ms:
        raise HTTPException(status_code=400, detail="scope end must follow its start")
    if not qa_core.media_path(core.settings, body.youtube_id, download=False).exists():
        raise HTTPException(status_code=409, detail="fetch the video first")
    try:
        qa_core.build_config(body.sport, body.overrides)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return core.status(core.create(body).run_id)


async def core_list():
    return dict(runs=_core().list())


async def core_status(run_id: str):
    return _job(run_id).status(run_id)


async def core_result(run_id: str):
    return _job(run_id).result(run_id)


async def core_resume(run_id: str):
    core = _job(run_id)
    try:
        core.resume(run_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    return core.status(run_id)


async def core_cancel(run_id: str):
    core = _job(run_id)
    core.cancel(run_id)
    return core.status(run_id)


async def core_fighter_crop(run_id: str, index: int):
    core = _job(run_id)
    path = core.jobs[run_id].folder / f"fighter-{index}.png"
    if index not in (0, 1) or not path.exists():
        raise HTTPException(status_code=404, detail="no crop")
    return Response(content=path.read_bytes(), media_type="image/png")


def mount(app):
    app.post("/qa/core/estimate")(core_estimate)
    app.get("/qa/core/media/{youtube_id}")(core_media)
    app.get("/qa/core/frame/{youtube_id}")(core_frame)
    app.get("/qa/core/video/{youtube_id}")(core_video)
    app.get("/qa/core/runs")(core_list)
    app.post("/qa/core/runs")(core_create)
    app.get("/qa/core/runs/{run_id}")(core_status)
    app.get("/qa/core/runs/{run_id}/result")(core_result)
    app.post("/qa/core/runs/{run_id}/resume")(core_resume)
    app.post("/qa/core/runs/{run_id}/cancel")(core_cancel)
    app.get("/qa/core/runs/{run_id}/fighter/{index}")(core_fighter_crop)
