"""VLM Studio routes for the production-core runner (``service.qa_core``)."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from service import qa_core
from service import sequence_review as seqreview
from service import strike_review as review
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
    parallel_position_checks: Optional[bool] = None


async def core_estimate(body: EstimateIn):
    if body.sport not in qa_core.SPORTS:
        raise HTTPException(status_code=400, detail="unknown workflow: " + body.sport)
    if body.end_ms <= body.start_ms:
        raise HTTPException(status_code=400, detail="scope end must follow its start")
    try:
        config = qa_core.build_config(body.sport, body.overrides, body.parallel_position_checks)
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
        qa_core.build_config(body.sport, body.overrides, body.parallel_position_checks)
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


def _run_strikes(core, run_id):
    status = _job(run_id).status(run_id)
    if status["state"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"run {run_id} has not succeeded")
    techniques = (core.result(run_id).get("techniques") or {}).get("techniques") or []
    return status["request"], techniques


async def strike_review_pairs():
    return dict(pairs=review.load_pairs(_core().settings.root))


async def strike_review(a: str, b: str):
    """Both runs' techniques over the part of the video both analysed, matched, with labels."""
    core = _core()
    req_a, tech_a = _run_strikes(core, a)
    req_b, tech_b = _run_strikes(core, b)
    if req_a["youtube_id"] != req_b["youtube_id"]:
        raise HTTPException(status_code=400, detail="the two runs analysed different videos")
    start, end = max(req_a["start_ms"], req_b["start_ms"]), min(req_a["end_ms"], req_b["end_ms"])
    if end <= start:
        raise HTTPException(status_code=400, detail="the two runs do not overlap")
    items = review.match(review.techniques(tech_a, start, end), review.techniques(tech_b, start, end))
    return dict(
        youtube_id=req_a["youtube_id"], start_ms=start, end_ms=end,
        fighters=[f["name"] for f in req_a["fighters"]], fighters_b=[f["name"] for f in req_b["fighters"]],
        a=dict(run_id=a, label=req_a.get("label"), sport=req_a["sport"]),
        b=dict(run_id=b, label=req_b.get("label"), sport=req_b["sport"]),
        items=items, labels=review.load_labels(core.settings.root, req_a["youtube_id"]),
    )


async def strike_label(youtube_id: str, body: review.LabelIn):
    core = _core()
    if not qa_core.YOUTUBE_ID.fullmatch(youtube_id):
        raise HTTPException(status_code=400, detail="invalid YouTube id")
    return dict(labels=review.save_label(core.settings.root, youtube_id, body))


async def sequence_review(run: str):
    """One run's sequences with this video's sequence labels."""
    core = _core()
    req, events = _run_strikes(core, run)
    return dict(
        youtube_id=req["youtube_id"], start_ms=req["start_ms"], end_ms=req["end_ms"],
        fighters=[f["name"] for f in req["fighters"]],
        run=dict(run_id=run, label=req.get("label"), sport=req["sport"]),
        sequences=seqreview.sequences(events), labels=seqreview.load_labels(core.settings.root, req["youtube_id"]),
    )


async def sequence_label(youtube_id: str, body: seqreview.SequenceLabelIn):
    core = _core()
    if not qa_core.YOUTUBE_ID.fullmatch(youtube_id):
        raise HTTPException(status_code=400, detail="invalid YouTube id")
    return dict(labels=seqreview.save_label(core.settings.root, youtube_id, body))


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
    app.get("/qa/core/strike-review/pairs")(strike_review_pairs)
    app.get("/qa/core/strike-review")(strike_review)
    app.put("/qa/core/strike-labels/{youtube_id}")(strike_label)
    app.get("/qa/core/sequence-review")(sequence_review)
    app.put("/qa/core/sequence-labels/{youtube_id}")(sequence_label)
