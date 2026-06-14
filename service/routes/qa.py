"""QA harness endpoint for Stream 2 Gemini identity-grounding.

``POST /qa/identity-analyze`` runs ONLY the Gemini analyzer (no GPU, no SAM2
tracking, no upscale) on a short clip plus labelled player references, and
returns the grounded result for the human eyeball gate in the QA client.

This is the reusable live gate for Stream 2 (``actor_player_id`` structured
output). It deliberately bypasses the tracking pipeline: it extracts a handful
of evenly-spaced RAW full frames from the clip, downloads the reference images,
builds a ``BJJTechniqueAnalyzer`` with ``config.gemini_api_key``, and calls
``analyze_sequence(frames, indices, player_references=...)`` directly.

No fabrication / no fallback data: if the Gemini key is missing → 400; if the
analyzer errors, the REAL error is surfaced (502).
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tempfile
import urllib.request
from typing import Optional

import cv2
from fastapi import HTTPException
from PIL import Image
from pydantic import BaseModel

from service.routes import state as route_state

logger = logging.getLogger("service.routes")

# How many evenly-spaced frames to feed Gemini for the QA gate. Small by design:
# this is an identity spot-check, not the full sliding-window pipeline.
QA_FRAME_COUNT = 6
# Thumbnail long edge (px) for the human-eyeball preview returned to the client.
QA_PREVIEW_LONG_EDGE = 480


class QAPlayerReference(BaseModel):
    player_id: str
    player_name: str
    image_s3_uri_or_url: str


class QAIdentityAnalyzeRequest(BaseModel):
    video_s3_uri_or_url: str
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    player_references: list[QAPlayerReference]


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key/path`` → (bucket, key). Raises on malformed input."""
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed s3 URI (need s3://bucket/key): {uri!r}")
    return bucket, key


def _is_s3_uri(value: str) -> bool:
    return value.startswith("s3://")


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _fetch_bytes(source: str) -> bytes:
    """Download bytes from an s3:// URI (engine S3 client) or an http(s) URL.

    Surfaces the real underlying error to the caller — no silent empty bytes.
    """
    if _is_s3_uri(source):
        bucket, key = _parse_s3_uri(source)
        s3 = route_state._s3_client()
        return s3.get_object(bucket, key)
    if _is_http_url(source):
        with urllib.request.urlopen(source) as resp:  # noqa: S310 (QA tool, operator-supplied URL)
            return resp.read()
    raise ValueError(
        f"Unsupported source {source!r}; expected s3://bucket/key or http(s):// URL"
    )


def _download_video_to_temp(source: str) -> str:
    """Download the clip to a temp .mp4 and return the local path."""
    data = _fetch_bytes(source)
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="qa_identity_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _extract_frames(video_path: str, start_frame: Optional[int], end_frame: Optional[int]):
    """Extract ``QA_FRAME_COUNT`` evenly-spaced RGB PIL frames from the clip window.

    Returns ``(frames, frame_indices)``. Raises ValueError when the clip cannot be
    read or yields no frames — never returns fabricated frames.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Clip has no readable frames (decode failed or empty video).")

        lo = max(0, start_frame if start_frame is not None else 0)
        hi = min(total - 1, end_frame if end_frame is not None else total - 1)
        if hi < lo:
            raise ValueError(
                f"Invalid frame window: start_frame={start_frame} > end_frame={end_frame} "
                f"(clip has {total} frames)."
            )

        span = hi - lo
        count = min(QA_FRAME_COUNT, span + 1)
        if count <= 1:
            indices = [lo]
        else:
            step = span / (count - 1)
            indices = [int(round(lo + i * step)) for i in range(count)]
            # Deduplicate while preserving order (tiny clips).
            seen: set[int] = set()
            indices = [i for i in indices if not (i in seen or seen.add(i))]

        frames = []
        kept_indices = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                logger.warning("QA identity-analyze: could not read frame %d; skipping", idx)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            kept_indices.append(idx)

        if not frames:
            raise ValueError("Could not decode any frame in the requested window.")
        return frames, kept_indices
    finally:
        cap.release()


def _thumbnail_data_uri(img: Image.Image) -> str:
    """Downscale a PIL frame and return it as a base64 JPEG data URI for the client."""
    thumb = img.copy()
    thumb.thumbnail((QA_PREVIEW_LONG_EDGE, QA_PREVIEW_LONG_EDGE))
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _download_reference_images(player_references: list[QAPlayerReference]) -> list[dict]:
    """Download each reference image into the labelled shape the analyzer expects.

    Shape mirrors ``upscale_setup`` bindings path: ``{player_id, player_name, image}``.
    Any download/decode failure surfaces as a 502 — no skipping (a missing ref would
    silently weaken the identity grounding the gate is meant to verify).
    """
    refs = []
    for ref in player_references:
        try:
            data = _fetch_bytes(ref.image_s3_uri_or_url)
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:  # noqa: BLE001 — surface the real error
            raise HTTPException(
                502,
                f"Failed to load reference image for {ref.player_name} "
                f"({ref.image_s3_uri_or_url}): {e}",
            )
        refs.append({
            "player_id": ref.player_id,
            "player_name": ref.player_name,
            "image": img,
        })
    return refs


def _build_validation(parsed: dict, provided_player_ids: list[str]) -> dict:
    """Strict enum validation surfaced to the human gate.

    ``all_in_enum`` is True iff every clip's ``actor_player_id`` is one of the
    provided player_ids. Offenders are the clip indices whose grounded identity
    escaped the enum (the failure Stream 2 must never re-introduce).
    """
    allowed = set(provided_player_ids)
    offenders = []
    for i, clip in enumerate(parsed.get("clips", []) or []):
        actor = clip.get("actor_player_id")
        if actor is None or actor not in allowed:
            offenders.append(i)
    return {
        "provided_player_ids": provided_player_ids,
        "all_in_enum": len(offenders) == 0,
        "offenders": offenders,
    }


async def identity_analyze(body: QAIdentityAnalyzeRequest):
    """Run ONLY the Gemini identity-grounding analyzer on a clip + labelled refs."""
    import json

    from analyzer import BJJTechniqueAnalyzer

    config = route_state._config
    assert config is not None
    if not config.gemini_api_key:
        raise HTTPException(
            400,
            "Gemini API key is not configured. Set GEMINI_API_KEY (or BJJ_GEMINI_API_KEY) "
            "in the engine .env and restart the service.",
        )

    if not body.player_references:
        raise HTTPException(400, "player_references must contain at least one labelled player.")

    loop = asyncio.get_event_loop()

    # 1) Download clip + extract a handful of raw frames (NO tracking pipeline).
    video_path = await loop.run_in_executor(
        None, _download_video_to_temp, body.video_s3_uri_or_url
    )
    try:
        frames, frame_indices = await loop.run_in_executor(
            None, _extract_frames, video_path, body.start_frame, body.end_frame
        )
    except ValueError as e:
        raise HTTPException(502, f"Frame extraction failed: {e}")
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass

    # 2) Download labelled reference images.
    player_references = await loop.run_in_executor(
        None, _download_reference_images, body.player_references
    )
    provided_player_ids = [r["player_id"] for r in player_references]

    # 3) Run the analyzer directly. Strict enum validation can RAISE — we want the
    #    human gate to SEE out-of-enum identities, so catch that ValueError and fold
    #    it into the validation block rather than 500-ing.
    analyzer = BJJTechniqueAnalyzer(api_key=config.gemini_api_key)

    def _run_analyzer():
        return analyzer.analyze_sequence(
            frames, frame_indices, previous_context=None,
            player_references=player_references,
        )

    try:
        result_str = await loop.run_in_executor(None, _run_analyzer)
    except ValueError as e:
        # Out-of-enum actor_player_id is a HARD validation failure in the analyzer.
        # Surface it to the gate as a FAIL with the raw message — not a fabricated pass.
        logger.warning("QA identity-analyze: strict validation rejected response: %s", e)
        result_str = None
        validation_error = str(e)
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        logger.error("QA identity-analyze: analyzer call failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Gemini analyzer call failed: {e}")
    else:
        validation_error = None

    frames_preview = await loop.run_in_executor(
        None, lambda: [_thumbnail_data_uri(f) for f in frames]
    )

    if result_str is None:
        # Strict validator raised: report a FAIL with the offending detail and no clips.
        return {
            "clips": [],
            "validation": {
                "provided_player_ids": provided_player_ids,
                "all_in_enum": False,
                "offenders": [],
                "error": validation_error,
            },
            "frames_preview": frames_preview,
            "frame_indices": frame_indices,
        }

    try:
        parsed = json.loads(result_str)
    except (ValueError, TypeError) as e:
        raise HTTPException(502, f"Analyzer returned non-JSON response: {e}: {result_str[:500]!r}")

    if isinstance(parsed, dict) and parsed.get("error"):
        # The analyzer caught a Gemini/transport error and returned {"error": ...}.
        raise HTTPException(502, f"Gemini analyzer error: {parsed['error']}")

    clips = []
    for clip in parsed.get("clips", []) or []:
        clips.append({
            "action": clip.get("action"),
            "technique": clip.get("technique"),
            "specific_technique": clip.get("specific_technique"),
            "actor_player_id": clip.get("actor_player_id"),
            "identity_uncertain": clip.get("identity_uncertain"),
            "reasoning": clip.get("reasoning"),
            "confidence": clip.get("confidence"),
        })

    return {
        "clips": clips,
        "validation": _build_validation(parsed, provided_player_ids),
        "frames_preview": frames_preview,
        "frame_indices": frame_indices,
    }
