"""QA VLM Studio endpoints (``POST /qa/vlm-chat``, ``POST /qa/vlm-image``).

These tests pin the request-construction contract with a MOCKED Gemini client —
NO real network call, NO real yt-dlp/ffmpeg invocation. They verify:
- the Gemini request is built with the correct ``video_metadata`` offsets
  (including the timepoint -> 1s-window widening),
- history turns are mapped to Gemini ``Content`` objects in order,
- a non-JSON-array chat response surfaces as an EMPTY events list + parse_error
  (no fabrication),
- out-of-window events are flagged, not dropped,
- a real Gemini transport error surfaces as 502 (not swallowed),
- an empty Gemini key yields 400 for both endpoints,
- ``/qa/vlm-image`` propagates a real frame-extraction (yt-dlp/ffmpeg) failure
  as 502 rather than returning a placeholder image,
- ``image_task="generate"`` is unchanged (still returns ``image_b64``/``mime``),
- ``image_task="detect"`` returns clean ``objects`` (with malformed ``box_2d``
  entries dropped, never fabricated) and falls back to the secondary model on
  a 404/model-unavailable error,
- ``image_task="segment"`` runs detect first, then recovers polygon masks from
  a tint-highlight image when possible, and DEGRADES to boxes-only
  (``degraded.masks_missing``) rather than 502ing when tint recovery finds
  nothing usable.
"""
from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.routes import init_routes, router
from service.routes import qa_vlm as qa_vlm_route
from tests.conftest import make_mock_jobs_store

YT_ID = "dQw4w9WgXcQ"
YT_URL = f"https://www.youtube.com/watch?v={YT_ID}"


def _make_app(gemini_api_key: str) -> FastAPI:
    config = ServiceConfig(detection_timeout=86400.0, s3_endpoint_url="http://localhost:4566")
    object.__setattr__(config, "gemini_api_key", gemini_api_key)
    init_routes(config, InMemoryJobStore(), make_mock_jobs_store())
    app = FastAPI()
    app.include_router(router)
    return app


@pytest_asyncio.fixture()
async def client_with_key():
    app = _make_app("fake-gemini-key")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _fake_client(response_text: str):
    """Build a fake genai.Client whose aio.models.generate_content returns response_text."""
    fake_response = MagicMock()
    fake_response.text = response_text
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
    return fake_client, fake_response


def _fake_image_response(png_bytes: bytes, text: str | None = None):
    """Build a fake Gemini response carrying one inline-data image part (+ optional text)."""
    part = MagicMock()
    part.inline_data.data = png_bytes
    part.text = text
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _fake_client_sequence(effects: list):
    """Build a fake genai.Client whose generate_content returns/raises ``effects`` in order."""
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=effects)
    return fake_client


class _FakeModelNotFoundError(Exception):
    """Minimal stand-in for google.genai.errors.ClientError with a 404 code."""

    code = 404


# --------------------------------------------------------------------------- #
# /qa/vlm-chat
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_vlm_chat_happy_path_builds_request_and_returns_events(monkeypatch, client_with_key):
    events_payload = [
        {
            "start": "00:10", "end": "00:14", "label": "guard pass",
            "action_category": "guard_pass", "technique": "knee_cut_pass",
            "actor": "athlete in blue gi", "description": "clean pass", "confidence": 0.82,
        },
    ]
    fake_client, fake_response = _fake_client(json.dumps(events_payload))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID,
            "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 30},
            "prompt": "List techniques in this window.",
            "history": [{"role": "user", "text": "hi"}, {"role": "model", "text": "ok"}],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == qa_vlm_route.CHAT_MODEL
    assert body["parse_error"] is None
    assert len(body["events"]) == 1
    ev = body["events"][0]
    assert ev["label"] == "guard pass"
    assert ev["start_sec"] == 10 and ev["end_sec"] == 14
    assert ev["out_of_window"] is False

    # Inspect the actual call made to Gemini.
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == qa_vlm_route.CHAT_MODEL
    contents = call_kwargs["contents"]
    # 2 history turns + 1 current user turn (text + video part)
    assert len(contents) == 3
    assert contents[0].role == "user" and contents[0].parts[0].text == "hi"
    assert contents[1].role == "model" and contents[1].parts[0].text == "ok"
    video_part = contents[2].parts[1]
    assert video_part.file_data.file_uri == YT_URL
    assert video_part.video_metadata.start_offset == "0s"
    assert video_part.video_metadata.end_offset == "30s"
    assert video_part.video_metadata.fps == 1

    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_vlm_chat_timepoint_widens_to_one_second_window(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID,
            "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 42, "end_sec": 42},
            "prompt": "What's happening at this instant?",
            "history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["notes"] == "Effective window: 42-43s"

    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    video_part = call_kwargs["contents"][-1].parts[1]
    assert video_part.video_metadata.start_offset == "42s"
    assert video_part.video_metadata.end_offset == "43s"


@pytest.mark.asyncio
async def test_vlm_chat_non_json_array_response_is_empty_events_not_fabricated(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("not json at all")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["events"] == []
    assert body["raw"] == "not json at all"
    assert body["parse_error"] is not None


@pytest.mark.asyncio
async def test_vlm_chat_out_of_window_event_is_flagged_not_dropped(monkeypatch, client_with_key):
    events_payload = [{"start": "05:00", "end": "05:05", "label": "far away", "description": "d", "confidence": 0.5}]
    fake_client, _ = _fake_client(json.dumps(events_payload))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    events = resp.json()["events"]
    assert len(events) == 1  # not dropped
    assert events[0]["out_of_window"] is True


@pytest.mark.asyncio
async def test_vlm_chat_gemini_transport_error_surfaces_as_502(monkeypatch, client_with_key):
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("deadline exceeded"))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert resp.status_code == 502
    assert "deadline exceeded" in resp.text


@pytest.mark.asyncio
async def test_vlm_chat_empty_key_returns_400():
    app = _make_app("")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/qa/vlm-chat",
            json={
                "youtube_id": YT_ID, "youtube_url": YT_URL,
                "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
                "prompt": "x", "history": [],
            },
        )
    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.text


# --------------------------------------------------------------------------- #
# /qa/vlm-image
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_vlm_image_generate_returns_composited_image(monkeypatch, client_with_key):
    frame = Image.new("RGB", (16, 12), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    gen_png = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(gen_png, format="PNG")
    part = MagicMock()
    part.inline_data.data = gen_png.getvalue()
    part.text = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    fake_response = MagicMock()
    fake_response.candidates = [candidate]
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Nano-banana this", "image_task": "generate",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == qa_vlm_route.IMAGE_MODEL
    assert base64.b64decode(body["image_b64"])  # decodes without error

    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == qa_vlm_route.IMAGE_MODEL
    assert call_kwargs["config"].response_modalities == ["TEXT", "IMAGE"]


@pytest.mark.asyncio
async def test_vlm_image_frame_extraction_failure_surfaces_as_502_no_placeholder(monkeypatch, client_with_key):
    def _boom(url, start_sec):
        raise RuntimeError("yt-dlp could not resolve a direct stream URL")

    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", _boom)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "x", "image_task": "generate",
        },
    )
    assert resp.status_code == 502
    assert "yt-dlp" in resp.text


@pytest.mark.asyncio
async def test_vlm_image_detect_returns_valid_objects(monkeypatch, client_with_key):
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [
        {"box_2d": [100, 100, 900, 900], "label": "blue gi athlete", "confidence": 0.91},
    ]
    fake_client, _ = _fake_client(json.dumps(detect_entries))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 45, "end_sec": 45},
            "prompt": "Detect athletes.", "image_task": "detect",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == qa_vlm_route.DETECT_MODEL
    assert body["frame_mime"] == "image/png"
    assert body["frame_time_sec"] == 45
    assert base64.b64decode(body["frame_b64"])  # decodes without error
    assert len(body["objects"]) == 1
    obj = body["objects"][0]
    assert obj["label"] == "blue gi athlete"
    assert obj["box_2d"] == [100, 100, 900, 900]
    assert obj["confidence"] == pytest.approx(0.91)
    assert "mask" not in obj

    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == qa_vlm_route.DETECT_MODEL
    config = call_kwargs["config"]
    assert config.temperature == 0.0
    assert config.thinking_config.thinking_budget == 0
    assert config.max_output_tokens == 4096
    assert config.response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_vlm_image_detect_drops_malformed_box_2d_not_fabricated(monkeypatch, client_with_key):
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [
        {"box_2d": ["not", "a", "number", "box"], "label": "bad_box"},
        {"box_2d": [0, 0, 1500, 500], "label": "out_of_range_box"},  # >1000
        {"box_2d": [0, 0, 500], "label": "wrong_length_box"},  # only 3 elements
        {"box_2d": [100, 100, 900, 900], "label": "good athlete", "confidence": 0.5},
    ]
    fake_client, _ = _fake_client(json.dumps(detect_entries))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Detect athletes.", "image_task": "detect",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["objects"]) == 1  # the 3 malformed entries are dropped, not fabricated/coerced
    assert body["objects"][0]["label"] == "good athlete"


@pytest.mark.asyncio
async def test_vlm_image_detect_falls_back_on_404_model_unavailable(monkeypatch, client_with_key):
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    good_entries = [{"box_2d": [100, 100, 900, 900], "label": "athlete"}]
    fallback_response = MagicMock()
    fallback_response.text = json.dumps(good_entries)
    fake_client = _fake_client_sequence([_FakeModelNotFoundError("model not found"), fallback_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Detect athletes.", "image_task": "detect",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == qa_vlm_route.DETECT_MODEL_FALLBACK
    assert len(body["objects"]) == 1

    assert fake_client.aio.models.generate_content.call_count == 2
    first_call, second_call = fake_client.aio.models.generate_content.call_args_list
    assert first_call.kwargs["model"] == qa_vlm_route.DETECT_MODEL
    assert second_call.kwargs["model"] == qa_vlm_route.DETECT_MODEL_FALLBACK


@pytest.mark.asyncio
async def test_vlm_image_segment_recovers_mask_from_tint(monkeypatch, client_with_key):
    """Detect finds one subject; the tint image actually contains a magenta wash
    over roughly that region, so the polygon-recovery step should attach a mask
    and NOT degrade."""
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "blue gi athlete", "confidence": 0.8}]
    detect_response = MagicMock()
    detect_response.text = json.dumps(detect_entries)

    tint_img = Image.new("RGB", (100, 100), (5, 5, 5))
    draw = ImageDraw.Draw(tint_img)
    draw.rectangle([20, 20, 60, 60], fill=(255, 0, 255))  # pure magenta wash, well above min-area
    tint_buf = io.BytesIO()
    tint_img.save(tint_buf, format="PNG")
    tint_response = _fake_image_response(tint_buf.getvalue())

    fake_client = _fake_client_sequence([detect_response, tint_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("degraded") is None
    assert len(body["objects"]) == 1
    obj = body["objects"][0]
    assert obj["label"] == "blue gi athlete"
    assert obj["mask"]["kind"] == "polygon"
    assert len(obj["mask"]["points"]) >= 3
    assert qa_vlm_route.SEGMENT_TINT_MODEL in body["model"]
    assert "EXPERIMENTAL" in body["notes"]

    tint_call_kwargs = fake_client.aio.models.generate_content.call_args_list[1].kwargs
    assert tint_call_kwargs["model"] == qa_vlm_route.SEGMENT_TINT_MODEL
    assert tint_call_kwargs["config"].response_modalities == ["TEXT", "IMAGE"]


@pytest.mark.asyncio
async def test_vlm_image_segment_no_mask_degrades_to_boxes_not_502(monkeypatch, client_with_key):
    """Detect finds a subject, but the tint image comes back with NO recoverable
    tint (e.g. Gemini ignored the wash instruction) — this must degrade to the
    real detect boxes with an explicit ``degraded`` flag, never a 502, and
    never a fabricated mask."""
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "blue gi athlete"}]
    detect_response = MagicMock()
    detect_response.text = json.dumps(detect_entries)

    # Tint image identical to the original frame — no tint wash present at all.
    tint_buf = io.BytesIO()
    Image.new("RGB", (100, 100), (5, 5, 5)).save(tint_buf, format="PNG")
    tint_response = _fake_image_response(tint_buf.getvalue())

    fake_client = _fake_client_sequence([detect_response, tint_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
        },
    )
    assert resp.status_code == 200, resp.text  # degrade, never 502
    body = resp.json()
    assert body["degraded"]["masks_missing"] is True
    assert len(body["objects"]) == 1
    obj = body["objects"][0]
    assert obj["label"] == "blue gi athlete"
    assert obj["box_2d"] == [100, 100, 900, 900]  # real detect box, unchanged
    assert "mask" not in obj  # no fabricated mask


@pytest.mark.asyncio
async def test_vlm_image_segment_more_subjects_than_palette_leaves_overflow_box_only(
    monkeypatch, client_with_key
):
    """Detect finds 5 subjects but the tint palette only has 4 distinct colors.

    Regression for evaluator Finding 1: colors must NEVER be cycled/reused across
    subjects, since `_extract_tint_polygons` keys its recovered contours by color
    name and would otherwise hand two different subjects the SAME polygon/box
    (a misattributed, effectively fabricated mask). The 5th subject must be left
    box-only (its real detect box, no mask) and the response must explicitly
    flag `degraded.masks_missing` with a palette-overflow reason.
    """
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [
        {"box_2d": [0, 0, 100, 100], "label": "athlete A"},
        {"box_2d": [100, 100, 200, 200], "label": "athlete B"},
        {"box_2d": [200, 200, 300, 300], "label": "athlete C"},
        {"box_2d": [300, 300, 400, 400], "label": "athlete D"},
        {"box_2d": [400, 400, 500, 500], "label": "athlete E (overflow)"},
    ]
    detect_response = MagicMock()
    detect_response.text = json.dumps(detect_entries)

    # Paint 4 distinct, well-separated, non-overlapping tint washes — one per
    # palette color (magenta/cyan/yellow/green) — so all 4 tinted subjects each
    # recover their OWN polygon rather than colliding.
    tint_img = Image.new("RGB", (100, 100), (5, 5, 5))
    draw = ImageDraw.Draw(tint_img)
    draw.rectangle([5, 5, 25, 25], fill=(255, 0, 255))    # magenta
    draw.rectangle([5, 75, 25, 95], fill=(0, 255, 255))   # cyan
    draw.rectangle([75, 5, 95, 25], fill=(255, 255, 0))   # yellow
    draw.rectangle([75, 75, 95, 95], fill=(0, 255, 0))    # green
    tint_buf = io.BytesIO()
    tint_img.save(tint_buf, format="PNG")
    tint_response = _fake_image_response(tint_buf.getvalue())

    fake_client = _fake_client_sequence([detect_response, tint_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
        },
    )
    assert resp.status_code == 200, resp.text  # degrade, never 502
    body = resp.json()

    assert len(body["objects"]) == 5

    tinted = body["objects"][:4]
    overflow = body["objects"][4]

    # All 4 tinted subjects recovered a mask, each from its own color region.
    for obj in tinted:
        assert obj.get("mask") is not None, obj
        assert obj["mask"]["kind"] == "polygon"

    # No two objects (tinted or not) ever share an identical recovered polygon —
    # the core regression check: distinct subjects must never collide on one mask.
    polygons = [tuple(map(tuple, obj["mask"]["points"])) for obj in tinted]
    assert len(set(polygons)) == len(polygons) == 4

    # The 5th subject (beyond the 4-color palette) is box-only: its real detect
    # box is preserved verbatim, and it gets NO mask (never fabricated).
    assert overflow["label"] == "athlete E (overflow)"
    assert overflow["box_2d"] == [400, 400, 500, 500]
    assert "mask" not in overflow

    # The response makes the palette-overflow degrade explicit rather than
    # silently dropping data.
    assert body["degraded"]["masks_missing"] is True
    assert "more subjects (5)" in body["degraded"]["reason"]
    assert "tint palette (4)" in body["degraded"]["reason"]
    assert "1 left box-only" in body["degraded"]["reason"]

    # Only the tintable subjects were ever offered to the tint model — the
    # tint prompt must never mention a color assignment for the overflow subject.
    tint_call_kwargs = fake_client.aio.models.generate_content.call_args_list[1].kwargs
    tint_prompt = tint_call_kwargs["contents"][0]
    assert "athlete E (overflow)" not in tint_prompt


@pytest.mark.asyncio
async def test_vlm_image_segment_no_subjects_detected_returns_empty_objects(monkeypatch, client_with_key):
    """Detect finds nothing — segment must skip the tint call entirely and
    return a valid 200 with an empty ``objects`` list (not a 502)."""
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_response = MagicMock()
    detect_response.text = json.dumps([])
    fake_client = _fake_client_sequence([detect_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["objects"] == []
    assert fake_client.aio.models.generate_content.call_count == 1  # no tint call made


@pytest.mark.asyncio
async def test_vlm_image_detect_frame_extraction_failure_surfaces_as_502(monkeypatch, client_with_key):
    def _boom(url, start_sec):
        raise RuntimeError("yt-dlp could not resolve a direct stream URL")

    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", _boom)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "x", "image_task": "detect",
        },
    )
    assert resp.status_code == 502
    assert "yt-dlp" in resp.text


@pytest.mark.asyncio
async def test_vlm_image_non_timepoint_scope_returns_400(client_with_key):
    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "image_task": "generate",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_vlm_image_empty_key_returns_400():
    app = _make_app("")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/qa/vlm-image",
            json={
                "youtube_id": YT_ID, "youtube_url": YT_URL,
                "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
                "prompt": "x", "image_task": "generate",
            },
        )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# Model selection, thinking mapping, and timing (model-comparison additions)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_vlm_chat_requested_allowlisted_model_is_used(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
            "model": "gemini-3.1-pro-preview",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "gemini-3.1-pro-preview"
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.1-pro-preview"
    # pro-preview cannot disable thinking; default (thinking omitted) == "off" -> forced low.
    assert call_kwargs["config"].thinking_config.thinking_level == "LOW"


@pytest.mark.asyncio
async def test_vlm_chat_model_not_in_allowlist_returns_400(client_with_key):
    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
            "model": "gpt-4o",
        },
    )
    assert resp.status_code == 400
    assert "gpt-4o" in resp.text


@pytest.mark.asyncio
async def test_vlm_chat_response_includes_timing(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    timing = resp.json()["timing"]
    assert isinstance(timing["total_ms"], (int, float))
    assert isinstance(timing["model_ms"], (int, float))
    assert timing["total_ms"] >= timing["model_ms"] >= 0


@pytest.mark.asyncio
async def test_vlm_chat_and_detect_timing_keys_match_ms_contract_exactly(monkeypatch, client_with_key):
    """Pins the frozen ``timing`` shape (all keys suffixed ``_ms``, values in
    milliseconds) so a future regression back to a ``_sec`` shape (the exact
    bug the QA client's ``timingLine()`` once had) fails a test instead of
    silently rendering 1000x-wrong durations in the UI.

    - ``/qa/vlm-chat`` (no frame to extract): timing keys == {total_ms, model_ms}.
    - ``/qa/vlm-image`` detect (extracts a frame first): timing keys include
      total_ms, model_ms, frame_extract_ms.
    """
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    chat_resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert chat_resp.status_code == 200, chat_resp.text
    assert set(chat_resp.json()["timing"].keys()) == {"total_ms", "model_ms"}

    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)
    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "athlete"}]
    fake_detect_client, _ = _fake_client(json.dumps(detect_entries))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_detect_client)

    detect_resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Detect athletes.", "image_task": "detect",
        },
    )
    assert detect_resp.status_code == 200, detect_resp.text
    detect_timing_keys = set(detect_resp.json()["timing"].keys())
    assert {"total_ms", "model_ms", "frame_extract_ms"} <= detect_timing_keys
    assert all(k.endswith("_ms") for k in detect_timing_keys)


def test_thinking_config_for_off_on_flash_is_explicit_budget_zero():
    cfg = qa_vlm_route.thinking_config_for("gemini-3.5-flash", "off")
    assert cfg.thinking_budget == 0
    assert cfg.thinking_level is None


def test_thinking_config_for_off_or_omitted_on_pro_preview_forces_level_low():
    # pro CANNOT disable thinking; both "off" and omitted (None) map to its
    # lowest available level rather than a budget Gemini would 400 on.
    for ui_level in ("off", None):
        cfg = qa_vlm_route.thinking_config_for("gemini-3.1-pro-preview", ui_level)
        assert cfg.thinking_level == "LOW"
        assert cfg.thinking_budget is None


def test_thinking_config_for_high_sets_thinking_level_high():
    cfg = qa_vlm_route.thinking_config_for("gemini-3.5-flash", "high")
    assert cfg.thinking_level == "HIGH"
    assert cfg.thinking_budget is None


def test_thinking_config_for_image_task_is_always_none_regardless_of_level():
    for model in qa_vlm_route.IMAGE_MODELS:
        for ui_level in ("off", "low", "medium", "high", None):
            assert qa_vlm_route.thinking_config_for(model, ui_level) is None


@pytest.mark.asyncio
async def test_vlm_image_detect_model_not_in_allowlist_returns_400(monkeypatch, client_with_key):
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "x", "image_task": "detect", "model": "not-a-real-model",
        },
    )
    assert resp.status_code == 400
    assert "not-a-real-model" in resp.text


@pytest.mark.asyncio
async def test_vlm_image_detect_requested_model_and_thinking_high_bumps_max_tokens(monkeypatch, client_with_key):
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "athlete"}]
    fake_client, _ = _fake_client(json.dumps(detect_entries))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Detect athletes.", "image_task": "detect",
            "model": "gemini-3.1-flash-lite", "thinking": "high",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "gemini-3.1-flash-lite"
    assert body["timing"]["frame_extract_ms"] >= 0
    assert body["timing"]["model_ms"] >= 0

    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.1-flash-lite"
    assert call_kwargs["config"].thinking_config.thinking_level == "HIGH"
    assert call_kwargs["config"].max_output_tokens == 8192  # bumped so thoughts don't eat the JSON budget


@pytest.mark.asyncio
async def test_vlm_image_generate_thinking_field_is_ignored(monkeypatch, client_with_key):
    """``image_task="generate"`` never gets a thinking config, even if the
    caller sends a ``thinking`` value — image models reject thinking outright."""
    frame = Image.new("RGB", (16, 12), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    gen_png = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(gen_png, format="PNG")
    part = MagicMock()
    part.inline_data.data = gen_png.getvalue()
    part.text = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    fake_response = MagicMock()
    fake_response.candidates = [candidate]
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Nano-banana this", "image_task": "generate", "thinking": "high",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].thinking_config is None
    timing = resp.json()["timing"]
    assert timing["frame_extract_ms"] >= 0
    assert timing["model_ms"] >= 0


@pytest.mark.asyncio
async def test_vlm_image_segment_model_selects_tint_model_but_not_internal_detect(monkeypatch, client_with_key):
    """``model`` on a segment request selects the TINT image model only — the
    internal detect-for-subjects call must still use the fixed default detect
    model (never the request's image model, which detect wouldn't accept
    anyway)."""
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "blue gi athlete"}]
    detect_response = MagicMock()
    detect_response.text = json.dumps(detect_entries)

    tint_img = Image.new("RGB", (100, 100), (5, 5, 5))
    draw = ImageDraw.Draw(tint_img)
    draw.rectangle([20, 20, 60, 60], fill=(255, 0, 255))
    tint_buf = io.BytesIO()
    tint_img.save(tint_buf, format="PNG")
    tint_response = _fake_image_response(tint_buf.getvalue())

    fake_client = _fake_client_sequence([detect_response, tint_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
            "model": "gemini-3-pro-image",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "gemini-3-pro-image" in body["model"]
    assert qa_vlm_route.DETECT_MODEL in body["model"]
    assert body["timing"]["detect_model_ms"] >= 0
    assert body["timing"]["tint_model_ms"] >= 0

    detect_call, tint_call = fake_client.aio.models.generate_content.call_args_list
    assert detect_call.kwargs["model"] == qa_vlm_route.DETECT_MODEL  # internal step, unaffected by request model
    assert tint_call.kwargs["model"] == "gemini-3-pro-image"


# --------------------------------------------------------------------------- #
# system_instruction (user-loadable taxonomy/schema) — vlm-chat + vlm-image generate
# --------------------------------------------------------------------------- #
def test_clean_system_instruction_normalizes_empty_and_whitespace_to_none():
    assert qa_vlm_route._clean_system_instruction(None) is None
    assert qa_vlm_route._clean_system_instruction("") is None
    assert qa_vlm_route._clean_system_instruction("   \n\t  ") is None
    assert qa_vlm_route._clean_system_instruction("  # Taxonomy\n- armbar  ") == "# Taxonomy\n- armbar"


@pytest.mark.asyncio
async def test_vlm_chat_system_instruction_passed_through_to_config(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
            "system_instruction": "# BJJ Taxonomy\n- armbar\n- triangle_choke",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == "# BJJ Taxonomy\n- armbar\n- triangle_choke"


@pytest.mark.asyncio
async def test_vlm_chat_whitespace_only_system_instruction_becomes_none(monkeypatch, client_with_key):
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
            "system_instruction": "   \n  ",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction is None


@pytest.mark.asyncio
async def test_vlm_chat_omitted_system_instruction_is_backward_compatible(monkeypatch, client_with_key):
    """A request that omits ``system_instruction`` entirely (pre-feature contract)
    must behave exactly as before — Gemini config gets ``system_instruction=None``."""
    fake_client, _ = _fake_client("[]")
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-chat",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "range", "start_sec": 0, "end_sec": 10},
            "prompt": "x", "history": [],
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction is None


@pytest.mark.asyncio
async def test_vlm_image_generate_system_instruction_passed_through_to_config(monkeypatch, client_with_key):
    frame = Image.new("RGB", (16, 12), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    gen_png = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(gen_png, format="PNG")
    part = MagicMock()
    part.inline_data.data = gen_png.getvalue()
    part.text = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    fake_response = MagicMock()
    fake_response.candidates = [candidate]
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Nano-banana this", "image_task": "generate",
            "system_instruction": "# BJJ Taxonomy\n- armbar",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == "# BJJ Taxonomy\n- armbar"


@pytest.mark.asyncio
async def test_vlm_image_detect_system_instruction_is_not_applied(monkeypatch, client_with_key):
    """``system_instruction`` must NOT reach the detect config — the detect
    prompt is a fixed internal implementation detail, not user-steerable."""
    frame = Image.new("RGB", (200, 150), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "athlete"}]
    fake_client, _ = _fake_client(json.dumps(detect_entries))
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Detect athletes.", "image_task": "detect",
            "system_instruction": "# BJJ Taxonomy\n- armbar",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    config = call_kwargs["config"]
    # GenerateContentConfig for detect never sets system_instruction at all.
    assert getattr(config, "system_instruction", None) is None


@pytest.mark.asyncio
async def test_vlm_image_segment_system_instruction_is_not_applied(monkeypatch, client_with_key):
    """``system_instruction`` must NOT reach either the internal detect step or
    the tint-highlight step of segment — both are fixed implementation details."""
    frame = Image.new("RGB", (100, 100), (5, 5, 5))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    detect_entries = [{"box_2d": [100, 100, 900, 900], "label": "blue gi athlete"}]
    detect_response = MagicMock()
    detect_response.text = json.dumps(detect_entries)

    tint_img = Image.new("RGB", (100, 100), (5, 5, 5))
    draw = ImageDraw.Draw(tint_img)
    draw.rectangle([20, 20, 60, 60], fill=(255, 0, 255))
    tint_buf = io.BytesIO()
    tint_img.save(tint_buf, format="PNG")
    tint_response = _fake_image_response(tint_buf.getvalue())

    fake_client = _fake_client_sequence([detect_response, tint_response])
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "segment", "image_task": "segment",
            "system_instruction": "# BJJ Taxonomy\n- armbar",
        },
    )
    assert resp.status_code == 200, resp.text
    detect_call, tint_call = fake_client.aio.models.generate_content.call_args_list
    assert getattr(detect_call.kwargs["config"], "system_instruction", None) is None
    assert getattr(tint_call.kwargs["config"], "system_instruction", None) is None


@pytest.mark.asyncio
async def test_vlm_image_generate_omitted_system_instruction_is_backward_compatible(monkeypatch, client_with_key):
    frame = Image.new("RGB", (16, 12), (10, 20, 30))
    monkeypatch.setattr(qa_vlm_route, "_extract_youtube_frame", lambda url, start_sec: frame)

    gen_png = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(gen_png, format="PNG")
    part = MagicMock()
    part.inline_data.data = gen_png.getvalue()
    part.text = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    fake_response = MagicMock()
    fake_response.candidates = [candidate]
    fake_client = MagicMock()
    fake_client.aio.models.generate_content = AsyncMock(return_value=fake_response)
    monkeypatch.setattr(qa_vlm_route, "_gemini_client", lambda: fake_client)

    resp = await client_with_key.post(
        "/qa/vlm-image",
        json={
            "youtube_id": YT_ID, "youtube_url": YT_URL,
            "scope": {"type": "timepoint", "start_sec": 5, "end_sec": 5},
            "prompt": "Nano-banana this", "image_task": "generate",
        },
    )
    assert resp.status_code == 200, resp.text
    call_kwargs = fake_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction is None


# --------------------------------------------------------------------------- #
# MM:SS parsing unit
# --------------------------------------------------------------------------- #
def test_parse_mmss():
    assert qa_vlm_route._parse_mmss("01:30") == 90
    assert qa_vlm_route._parse_mmss("1:02:03") == 3723
    assert qa_vlm_route._parse_mmss("not-a-time") is None
    assert qa_vlm_route._parse_mmss(None) is None
