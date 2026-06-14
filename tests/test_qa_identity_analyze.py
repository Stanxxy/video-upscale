"""QA harness for Stream 2 Gemini identity-grounding (``POST /qa/identity-analyze``).

These tests pin the QA endpoint contract with a MOCKED analyzer, MOCKED frame
extraction, and MOCKED reference-image download — NO real Gemini call, NO real
S3. They verify the endpoint returns clips + a validation block + frame previews,
that an out-of-enum ``actor_player_id`` is reported as ``all_in_enum=false`` with
the offender listed (no fabrication), and that an empty Gemini key yields 400.

GEMINI_API_KEY aliasing is also covered: a founder ``.env`` using the plain
``GEMINI_API_KEY`` name must populate ``config.gemini_api_key``.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.routes import init_routes, router
from service.routes import qa as qa_route
from tests.conftest import make_mock_jobs_store


PID_A = "1732d982-dec6-45c6-9425-880658820025"
PID_B = "dcbec0b6-3c71-4c73-9ea9-b44fc8bf92f7"

DEMO_VIDEO = "s3://bjj-video-analysis/users/u/v/clip_001.mp4"


def _refs_payload():
    return [
        {"player_id": PID_A, "player_name": "sliu1143", "image_s3_uri_or_url": "s3://b/ref-a.jpg"},
        {"player_id": PID_B, "player_name": "Stan Liu", "image_s3_uri_or_url": "s3://b/ref-b.jpg"},
    ]


def _make_app(gemini_api_key: str) -> FastAPI:
    config = ServiceConfig(
        detection_timeout=86400.0,
        s3_endpoint_url="http://localhost:4566",
    )
    # Force the key deterministically (bypass any ambient env/.env for the test).
    object.__setattr__(config, "gemini_api_key", gemini_api_key)
    init_routes(config, InMemoryJobStore(), make_mock_jobs_store())
    app = FastAPI()
    app.include_router(router)
    return app


@pytest_asyncio.fixture()
async def _patch_io(monkeypatch):
    """Stub out video download, frame extraction, and ref-image download.

    The analyzer is patched per-test (different responses). Frame extraction returns
    a couple of tiny real PIL images so thumbnailing exercises the real code path.
    """
    monkeypatch.setattr(qa_route, "_download_video_to_temp", lambda src: "/tmp/qa-fake.mp4")

    frames = [Image.new("RGB", (32, 24), (10, 20, 30)), Image.new("RGB", (32, 24), (40, 50, 60))]
    monkeypatch.setattr(qa_route, "_extract_frames", lambda path, lo, hi: (frames, [0, 5]))

    def _fake_refs(player_references):
        return [
            {"player_id": r.player_id, "player_name": r.player_name, "image": Image.new("RGB", (8, 8))}
            for r in player_references
        ]

    monkeypatch.setattr(qa_route, "_download_reference_images", _fake_refs)
    # os.remove on the fake path would raise; make it a no-op via monkeypatch on the module.
    monkeypatch.setattr(qa_route.os, "remove", lambda p: None)
    yield


def _patch_analyzer(monkeypatch, response_obj):
    """Patch BJJTechniqueAnalyzer so analyze_sequence returns the given JSON object."""
    captured = {}

    class _FakeAnalyzer:
        def __init__(self, api_key, **kwargs):
            captured["api_key"] = api_key

        def analyze_sequence(self, frames, indices, previous_context=None, player_references=None):
            captured["frames"] = frames
            captured["player_references"] = player_references
            return json.dumps(response_obj)

    import analyzer as analyzer_mod
    monkeypatch.setattr(analyzer_mod, "BJJTechniqueAnalyzer", _FakeAnalyzer)
    return captured


@pytest_asyncio.fixture()
async def client_with_key():
    app = _make_app("fake-gemini-key")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --------------------------------------------------------------------------- #
# Happy path: clips + validation + frames_preview
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_endpoint_returns_clips_validation_and_previews(monkeypatch, _patch_io, client_with_key):
    _patch_analyzer(monkeypatch, {
        "current_context_summary": "guard work",
        "clips": [{
            "start_frame": 0, "end_frame": 5, "action": "guard_pass",
            "technique": "knee_cut_pass", "specific_technique": "Knee Cut Pass",
            "actor_player_id": PID_A, "identity_uncertain": False,
            "reasoning": "white gi on top driving the pass", "confidence": 0.8,
        }],
    })

    resp = await client_with_key.post(
        "/qa/identity-analyze",
        json={"video_s3_uri_or_url": DEMO_VIDEO, "player_references": _refs_payload()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["clips"]) == 1
    clip = body["clips"][0]
    assert clip["actor_player_id"] == PID_A
    assert clip["action"] == "guard_pass"
    assert clip["reasoning"].startswith("white gi")

    val = body["validation"]
    assert val["provided_player_ids"] == [PID_A, PID_B]
    assert val["all_in_enum"] is True
    assert val["offenders"] == []

    # Frame previews are real base64 JPEG data URIs (human eyeball gate).
    assert len(body["frames_preview"]) == 2
    assert all(p.startswith("data:image/jpeg;base64,") for p in body["frames_preview"])


# --------------------------------------------------------------------------- #
# Out-of-enum actor_player_id → all_in_enum=false + offender listed
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_out_of_enum_actor_player_id_is_flagged(monkeypatch, _patch_io, client_with_key):
    # NOTE: the real analyzer would RAISE on out-of-enum. To prove the endpoint's
    # OWN validation block also catches it, the fake analyzer returns the bad id
    # WITHOUT raising; the endpoint must mark it all_in_enum=false + offender [0].
    _patch_analyzer(monkeypatch, {
        "clips": [
            {"action": "mount", "actor_player_id": PID_A, "reasoning": "ok"},
            {"action": "submission_attempt", "actor_player_id": "athlete in black gi",
             "reasoning": "gi-color leaked into identity"},
        ],
    })

    resp = await client_with_key.post(
        "/qa/identity-analyze",
        json={"video_s3_uri_or_url": DEMO_VIDEO, "player_references": _refs_payload()},
    )
    assert resp.status_code == 200, resp.text
    val = resp.json()["validation"]
    assert val["all_in_enum"] is False
    assert val["offenders"] == [1]  # second clip escaped the enum


# --------------------------------------------------------------------------- #
# Strict analyzer ValueError (real-engine behaviour) → FAIL block, not 500
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_strict_validator_raise_surfaces_as_fail(monkeypatch, _patch_io, client_with_key):
    class _RaisingAnalyzer:
        def __init__(self, api_key, **kwargs):
            pass

        def analyze_sequence(self, *a, **k):
            raise ValueError("Gemini returned out-of-enum actor_player_id='x'; allowed=[...]")

    import analyzer as analyzer_mod
    monkeypatch.setattr(analyzer_mod, "BJJTechniqueAnalyzer", _RaisingAnalyzer)

    resp = await client_with_key.post(
        "/qa/identity-analyze",
        json={"video_s3_uri_or_url": DEMO_VIDEO, "player_references": _refs_payload()},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["clips"] == []
    assert body["validation"]["all_in_enum"] is False
    assert "out-of-enum" in body["validation"]["error"]
    # Previews still returned so the human can still eyeball the clip.
    assert len(body["frames_preview"]) == 2


# --------------------------------------------------------------------------- #
# Empty API key → 400 (no fabrication)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_empty_api_key_returns_400(_patch_io):
    app = _make_app("")  # no key
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/qa/identity-analyze",
            json={"video_s3_uri_or_url": DEMO_VIDEO, "player_references": _refs_payload()},
        )
    assert resp.status_code == 400
    assert "GEMINI_API_KEY" in resp.text


# --------------------------------------------------------------------------- #
# Real Gemini transport error surfaced (analyzer returns {"error": ...}) → 502
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_gemini_error_object_surfaces_502(monkeypatch, _patch_io, client_with_key):
    _patch_analyzer(monkeypatch, {"error": "deadline exceeded talking to Gemini"})

    resp = await client_with_key.post(
        "/qa/identity-analyze",
        json={"video_s3_uri_or_url": DEMO_VIDEO, "player_references": _refs_payload()},
    )
    assert resp.status_code == 502
    assert "deadline exceeded" in resp.text


# --------------------------------------------------------------------------- #
# Validation helper unit
# --------------------------------------------------------------------------- #
def test_build_validation_lists_offenders():
    parsed = {"clips": [
        {"actor_player_id": PID_A},
        {"actor_player_id": None},
        {"actor_player_id": PID_B},
        {"actor_player_id": "gi-color"},
    ]}
    val = qa_route._build_validation(parsed, [PID_A, PID_B])
    assert val["all_in_enum"] is False
    assert val["offenders"] == [1, 3]


def test_parse_s3_uri():
    assert qa_route._parse_s3_uri("s3://bucket/a/b/c.mp4") == ("bucket", "a/b/c.mp4")
    with pytest.raises(ValueError):
        qa_route._parse_s3_uri("s3://bucket-only")


# --------------------------------------------------------------------------- #
# GEMINI_API_KEY aliasing — plain (un-prefixed) env populates config
# --------------------------------------------------------------------------- #
def test_plain_gemini_api_key_env_is_aliased(monkeypatch):
    monkeypatch.delenv("BJJ_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "plain-env-key")
    cfg = ServiceConfig(_env_file=None)
    assert cfg.gemini_api_key == "plain-env-key"


def test_prefixed_gemini_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("BJJ_GEMINI_API_KEY", "prefixed-key")
    monkeypatch.setenv("GEMINI_API_KEY", "plain-key")
    cfg = ServiceConfig(_env_file=None)
    assert cfg.gemini_api_key == "prefixed-key"
