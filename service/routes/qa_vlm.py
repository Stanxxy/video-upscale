"""QA VLM Studio endpoints — thin server-side proxies to Gemini for the QA client.

``POST /qa/vlm-chat`` and ``POST /qa/vlm-image`` exist ONLY because a browser
cannot call Gemini directly (CORS) and image-segmentation needs server-side
frame pixels. Both hold the Gemini API key server-side (``config.gemini_api_key``,
same alias rules as ``/qa/identity-analyze``) and otherwise do the absolute
minimum: build the Gemini request, call it, and hand back the raw-ish result.

No fabrication / no fallback data:
- Missing Gemini key -> 400.
- Real Gemini/transport error -> 502 with the real message.
- Unparseable structured output -> the raw text is returned with an explicit
  ``parse_error`` and an EMPTY events list — never invented events.
- Frame extraction (yt-dlp + ffmpeg) failure -> 502 with the real message, no
  placeholder image.
- ``vlm-image`` with ``image_task in {"detect", "segment"}`` never fabricates
  boxes/masks: malformed detect entries are dropped, and a ``segment`` call
  whose tint recovery finds no usable mask degrades to boxes-only
  (``degraded.masks_missing``) rather than 502ing or inventing a mask.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import subprocess
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import HTTPException
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from service.routes import state as route_state

logger = logging.getLogger("service.routes")

CHAT_MODEL = "gemini-3.1-flash-lite"
IMAGE_MODEL = "gemini-3.1-flash-image"

# Box-only detect tier (reliable; no mask). 404/unavailable falls back once.
DETECT_MODEL = "gemini-3.5-flash"
DETECT_MODEL_FALLBACK = "gemini-3.1-flash-lite"

# Coarse tint-highlight segment tier — EXPERIMENTAL. Gemini 3.x has no
# segmentation head (see INS-049); instead of asking for pixel masks directly
# we ask Nano Banana 2 to paint a translucent, distinctly-hued wash over each
# detected subject, then recover an approximate silhouette polygon from the
# returned image via classic HSV color thresholding (OpenCV).
SEGMENT_TINT_MODEL = "gemini-3.1-flash-image"

# One tint per detected subject, cycled if there are more subjects than
# colors. Hue ranges are OpenCV HSV (H: 0-179) and were tuned against a tint
# painted OVER existing pixels (incl. dark rashguards), hence the loose S/V
# floors applied at threshold time rather than baked in here.
_TINT_SPECS = [
    {"name": "magenta", "hex": "#FF00FF", "hue_range": (135, 172)},
    {"name": "cyan", "hex": "#00FFFF", "hue_range": (80, 100)},
    {"name": "yellow", "hex": "#FFFF00", "hue_range": (25, 35)},
    {"name": "green", "hex": "#00FF00", "hue_range": (45, 75)},
]


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class QAVLMScope(BaseModel):
    type: Literal["timepoint", "range"]
    start_sec: int
    end_sec: int


class QAVLMHistoryTurn(BaseModel):
    role: Literal["user", "model"]
    text: str


class QAVLMChatRequest(BaseModel):
    youtube_id: str
    youtube_url: str
    scope: QAVLMScope
    prompt: str
    history: list[QAVLMHistoryTurn] = []
    event_schema_version: str = "v1"


class QAVLMImageRequest(BaseModel):
    youtube_id: str
    youtube_url: str
    scope: QAVLMScope
    prompt: str
    image_task: Literal["generate", "detect", "segment"]
    history: list[QAVLMHistoryTurn] = []


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _require_gemini_key() -> str:
    config = route_state._config
    assert config is not None
    if not config.gemini_api_key:
        raise HTTPException(
            400,
            "Gemini API key is not configured. Set GEMINI_API_KEY (or BJJ_GEMINI_API_KEY) "
            "in the engine .env and restart the service.",
        )
    return config.gemini_api_key


def _gemini_client() -> genai.Client:
    config = route_state._config
    assert config is not None
    return genai.Client(
        api_key=_require_gemini_key(),
        http_options=types.HttpOptions(timeout=config.gemini_request_timeout_ms),
    )


def _effective_window(scope: QAVLMScope) -> tuple[int, int]:
    """Resolve the [start_sec, end_sec) window Gemini should look at.

    ``timepoint`` scope is widened to a minimum 1-second window (Gemini's
    ``video_metadata`` offsets need start < end); ``range`` is used as-is.
    """
    start = scope.start_sec
    if scope.type == "timepoint":
        end = max(start + 1, scope.end_sec)
    else:
        end = scope.end_sec
    if end <= start:
        raise HTTPException(
            400, f"Invalid scope window: start_sec={start} must be < end_sec={end}"
        )
    return start, end


def _history_to_contents(history: list[QAVLMHistoryTurn]) -> list[types.Content]:
    return [
        types.Content(role=turn.role, parts=[types.Part(text=turn.text)])
        for turn in history
    ]


# --------------------------------------------------------------------------- #
# /qa/vlm-chat — event/time-range analysis, NO download
# --------------------------------------------------------------------------- #
def _event_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.ARRAY,
        description="Detected BJJ events/techniques within the requested time window.",
        items=types.Schema(
            type=types.Type.OBJECT,
            required=["start", "end", "label", "description", "confidence"],
            properties={
                "start": types.Schema(
                    type=types.Type.STRING,
                    description="Event start timestamp as MM:SS (absolute video time, NOT relative to the clipped window).",
                ),
                "end": types.Schema(
                    type=types.Type.STRING,
                    description="Event end timestamp as MM:SS (absolute video time, NOT relative to the clipped window).",
                ),
                "label": types.Schema(type=types.Type.STRING, description="Short human-readable event label."),
                "action_category": types.Schema(
                    type=types.Type.STRING,
                    description="Coarse BJJ action category (e.g. takedown, guard_pass, submission_attempt). Optional.",
                ),
                "technique": types.Schema(
                    type=types.Type.STRING,
                    description="Specific technique name (e.g. armbar, double_leg). Optional.",
                ),
                "actor": types.Schema(
                    type=types.Type.STRING,
                    description="Free-text description of which athlete performed it (e.g. gi color/position). Optional.",
                ),
                "description": types.Schema(type=types.Type.STRING, description="Free-text rationale/description."),
                "confidence": types.Schema(type=types.Type.NUMBER, description="Confidence 0.0-1.0.", format="float"),
            },
        ),
    )


def _parse_mmss(value: str) -> Optional[int]:
    """Parse ``MM:SS`` or ``H:MM:SS`` into total seconds. Returns None on malformed input."""
    if not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    try:
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
    except ValueError:
        return None
    return None


def _annotate_window_membership(events: list[dict], start_sec: int, end_sec: int) -> list[dict]:
    """Attach ``start_sec``/``end_sec``/``out_of_window`` to each event (validation, not fabrication).

    An event whose parsed MM:SS timestamps fall outside ``[start_sec, end_sec]`` is
    flagged, never dropped — the human QA reviewer decides what to do with it.
    """
    annotated = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        s = _parse_mmss(ev.get("start"))
        e = _parse_mmss(ev.get("end"))
        out_of_window = (
            s is None or e is None or s < start_sec or e > end_sec
        )
        annotated.append({**ev, "start_sec": s, "end_sec": e, "out_of_window": out_of_window})
    return annotated


async def vlm_chat(body: QAVLMChatRequest):
    """Run a single Gemini structured-output call over a YouTube time window."""
    _require_gemini_key()
    start_sec, end_sec = _effective_window(body.scope)

    video_part = types.Part(
        file_data=types.FileData(file_uri=body.youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{start_sec}s",
            end_offset=f"{end_sec}s",
            fps=1,
        ),
    )
    prompt_text = (
        f"{body.prompt}\n\n"
        "Report event start/end timestamps as MM:SS in the ORIGINAL video's absolute "
        "timeline (not relative to this clipped window). Return ONLY the JSON array "
        "matching the response schema — no prose."
    )
    contents = _history_to_contents(body.history)
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt_text), video_part]))

    client = _gemini_client()
    try:
        logger.info(
            "vlm-chat: youtube_id=%s window=%d-%ds model=%s",
            body.youtube_id, start_sec, end_sec, CHAT_MODEL,
        )
        response = await client.aio.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=_event_schema(),
            ),
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        logger.error("vlm-chat: Gemini call failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Gemini call failed: {e}")

    raw_text = response.text or ""
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
        parse_error = None
    except (ValueError, TypeError) as e:
        logger.warning("vlm-chat: non-JSON-array response: %s", e)
        parsed = []
        parse_error = str(e)

    events = _annotate_window_membership(parsed, start_sec, end_sec)

    return {
        "events": events,
        "notes": f"Effective window: {start_sec}-{end_sec}s",
        "raw": raw_text,
        "model": CHAT_MODEL,
        "parse_error": parse_error,
    }


# --------------------------------------------------------------------------- #
# /qa/vlm-image — single-frame grab + image generation / segmentation
# --------------------------------------------------------------------------- #
def _resolve_stream_url(youtube_url: str) -> tuple[str, dict]:
    """Resolve a direct media stream URL + required HTTP headers via yt-dlp.

    Raises RuntimeError (never returns a placeholder) if yt-dlp cannot resolve
    a playable stream — e.g. missing dependency, private/removed video, or
    YouTube throttling.
    """
    import yt_dlp

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)

    url = info.get("url")
    headers = info.get("http_headers") or {}
    if not url:
        for fmt in reversed(info.get("formats") or []):
            if fmt.get("url"):
                url = fmt["url"]
                headers = fmt.get("http_headers") or headers
                break
    if not url:
        raise RuntimeError(f"yt-dlp could not resolve a direct stream URL for {youtube_url!r}")
    return url, headers


def _grab_frame_via_ffmpeg(stream_url: str, headers: dict, start_sec: int) -> bytes:
    """Extract ONE PNG frame at ``start_sec`` from a resolved stream URL via ffmpeg.

    Raises RuntimeError with ffmpeg's real stderr on failure — no placeholder frame.
    """
    cmd = ["ffmpeg", "-y"]
    if headers:
        header_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        cmd += ["-headers", header_str]
    cmd += [
        "-ss", str(max(0, start_sec)),
        "-i", stream_url,
        "-frames:v", "1",
        "-f", "image2",
        "-vcodec", "png",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg frame extraction failed (rc={proc.returncode}): {stderr_tail}")
    return proc.stdout


def _extract_youtube_frame(youtube_url: str, start_sec: int) -> Image.Image:
    stream_url, headers = _resolve_stream_url(youtube_url)
    png_bytes = _grab_frame_via_ffmpeg(stream_url, headers, start_sec)
    img = Image.open(io.BytesIO(png_bytes))
    img.load()
    return img.convert("RGB")


def _encode_frame_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_inline_image_and_text(response) -> tuple[Optional[bytes], Optional[str]]:
    """Pull the first inline image + concatenated text parts out of a Gemini response.

    Duplicated (not shared) with the ``generate`` task's inline-extraction on
    purpose — ``generate`` must stay byte-for-byte unchanged.
    """
    image_bytes = None
    notes = None
    candidates = response.candidates or []
    for part in (candidates[0].content.parts if candidates and candidates[0].content else []):
        if part.inline_data is not None and part.inline_data.data:
            image_bytes = part.inline_data.data
        elif part.text:
            notes = (notes + part.text) if notes else part.text
    return image_bytes, notes


# --------------------------------------------------------------------------- #
# detect tier — reliable box-only detection (no mask)
# --------------------------------------------------------------------------- #
_DETECT_PROMPT = (
    "Detect every distinct person/athlete visible in this image (and any other notable "
    "object relevant to the scene). Return a JSON list; each entry must have box_2d "
    "[ymin, xmin, ymax, xmax] normalized to integers 0-1000, a short human-readable label "
    "(e.g. gi color/position), and a confidence score between 0 and 1."
)


def _detect_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            required=["box_2d", "label"],
            properties={
                "box_2d": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.INTEGER),
                    description="[ymin, xmin, ymax, xmax] normalized to 0-1000.",
                ),
                "label": types.Schema(type=types.Type.STRING, description="Short label, e.g. gi color/position."),
                "confidence": types.Schema(
                    type=types.Type.NUMBER, format="float", description="Confidence 0.0-1.0 (optional)."
                ),
            },
        ),
    )


def _is_model_unavailable(e: Exception) -> bool:
    """True for a 404/model-not-found style error — worth one fallback retry."""
    code = getattr(e, "code", None)
    if code == 404:
        return True
    msg = str(e).lower()
    return "not found" in msg or "not_found" in msg


def _valid_box_2d(box) -> Optional[list[int]]:
    """Coerce ``box_2d`` to 4 ints in [0, 1000], or None if malformed (never fabricated)."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        vals = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    if any(v < 0 or v > 1000 for v in vals):
        return None
    return vals


def _parse_detect_entries(raw_entries: list) -> list[dict]:
    """Validate raw Gemini detect entries into clean objects.

    Entries with a missing/malformed ``box_2d`` or ``label`` are dropped
    (logged, not fabricated) rather than coerced into something invented.
    """
    objects = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        box = _valid_box_2d(entry.get("box_2d"))
        label = entry.get("label")
        if box is None or not isinstance(label, str) or not label.strip():
            logger.warning("vlm-image detect: dropping malformed entry %r", entry)
            continue
        obj: dict = {"label": label, "box_2d": box}
        confidence = entry.get("confidence")
        if isinstance(confidence, (int, float)):
            obj["confidence"] = float(confidence)
        objects.append(obj)
    return objects


async def _run_detect(client: genai.Client, frame_img: Image.Image) -> tuple[list[dict], str]:
    """Call the detect-tier model (with one 404/unavailable fallback retry).

    Raises RuntimeError on any unrecoverable Gemini/transport error or an
    unparseable response — the caller turns that into a 502.
    """
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=4096,
        response_schema=_detect_schema(),
    )
    model = DETECT_MODEL
    try:
        response = await client.aio.models.generate_content(
            model=model, contents=[_DETECT_PROMPT, frame_img], config=config,
        )
    except Exception as e:  # noqa: BLE001 — fall back once on model-unavailable, else re-raise
        if not _is_model_unavailable(e):
            raise RuntimeError(f"Gemini detect call failed ({model}): {e}") from e
        logger.warning(
            "vlm-image detect: %s unavailable (%s), retrying with %s", model, e, DETECT_MODEL_FALLBACK
        )
        model = DETECT_MODEL_FALLBACK
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=[_DETECT_PROMPT, frame_img], config=config,
            )
        except Exception as e2:  # noqa: BLE001 — surface the real Gemini/transport error
            raise RuntimeError(f"Gemini detect call failed ({model}): {e2}") from e2

    raw_text = response.text or ""
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"Gemini detect returned non-JSON response: {e}: {raw_text[:500]!r}"
        ) from e

    return _parse_detect_entries(parsed), model


# --------------------------------------------------------------------------- #
# segment tier — EXPERIMENTAL coarse tint-highlight + polygon recovery
# --------------------------------------------------------------------------- #
def _build_tint_prompt(assignments: list[tuple[str, dict]]) -> str:
    lines = [
        "Keep this photo EXACTLY as-is at the same resolution — do NOT move, redraw, or "
        "regenerate any person. Apply ONLY translucent 60%-opacity pure color washes ON "
        "TOP of the existing pixels over the subjects listed below; their real texture "
        "and silhouette edges must stay visible underneath the wash. Paint nothing else, "
        "and keep each subject's wash a single flat color (no gradients, no blending "
        "between subjects)."
    ]
    for label, spec in assignments:
        lines.append(
            f'- Apply a translucent 60%-opacity pure-{spec["name"]} ({spec["hex"]}) color '
            f'wash over "{label}".'
        )
    return "\n".join(lines)


def _extract_tint_polygons(
    tint_img: Image.Image, orig_size: tuple[int, int], color_specs: list[dict]
) -> dict[str, dict]:
    """Recover an approximate silhouette polygon per tint color via HSV thresholding.

    Returns ``{color_name: {"points": [[x,y],...] (0-1000), "box_2d": [ymin,xmin,ymax,xmax] (0-1000)}}``
    only for colors whose tint actually produced a large-enough contour — colors
    with no usable contour are simply absent from the result (silent per-object
    fallback; the caller leaves that subject box-only).
    """
    w, h = orig_size
    resized = tint_img.convert("RGB").resize((w, h))
    arr = np.array(resized)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_area = 0.001 * h * w

    results: dict[str, dict] = {}
    for spec in color_specs:
        h_lo, h_hi = spec["hue_range"]
        # Loose S/V floors: tint painted over dark rashguards/gis depresses value.
        lower = np.array([h_lo, 55, 50], dtype=np.uint8)
        upper = np.array([h_hi, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
        xs, ys = approx[:, 0], approx[:, 1]
        box_2d = [
            int(round(int(ys.min()) / h * 1000)),
            int(round(int(xs.min()) / w * 1000)),
            int(round(int(ys.max()) / h * 1000)),
            int(round(int(xs.max()) / w * 1000)),
        ]
        points = [[int(round(px / w * 1000)), int(round(py / h * 1000))] for px, py in approx]
        results[spec["name"]] = {"points": points, "box_2d": box_2d}
    return results


async def vlm_image(body: QAVLMImageRequest):
    """Grab one YouTube frame server-side, then run Gemini image generation or segmentation."""
    _require_gemini_key()
    if body.scope.type != "timepoint":
        raise HTTPException(400, "vlm-image requires scope.type == 'timepoint'")
    start_sec = body.scope.start_sec

    loop = asyncio.get_event_loop()
    try:
        frame_img = await loop.run_in_executor(None, _extract_youtube_frame, body.youtube_url, start_sec)
    except Exception as e:  # noqa: BLE001 — surface the real yt-dlp/ffmpeg error, no placeholder frame
        logger.error("vlm-image: frame extraction failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Frame extraction failed (yt-dlp/ffmpeg): {e}")

    client = _gemini_client()

    if body.image_task == "generate":
        try:
            response = await client.aio.models.generate_content(
                model=IMAGE_MODEL,
                contents=[body.prompt, frame_img],
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
            logger.error("vlm-image generate: Gemini call failed: %s", e, exc_info=True)
            raise HTTPException(502, f"Gemini image generation failed: {e}")

        image_bytes = None
        notes = None
        candidates = response.candidates or []
        for part in (candidates[0].content.parts if candidates and candidates[0].content else []):
            if part.inline_data is not None and part.inline_data.data:
                image_bytes = part.inline_data.data
            elif part.text:
                notes = (notes + part.text) if notes else part.text
        if image_bytes is None:
            raise HTTPException(502, "Gemini image model returned no image data (no inline_data part).")

        return {
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "mime": "image/png",
            "notes": notes,
            "model": IMAGE_MODEL,
        }

    frame_b64 = _encode_frame_png(frame_img)
    base_response = {
        "frame_b64": frame_b64,
        "frame_mime": "image/png",
        "frame_time_sec": start_sec,
    }

    if body.image_task == "detect":
        try:
            objects, model_used = await _run_detect(client, frame_img)
        except RuntimeError as e:  # noqa: BLE001 — surface the real Gemini/transport error
            logger.error("vlm-image detect: %s", e, exc_info=True)
            raise HTTPException(502, str(e))

        return {
            **base_response,
            "objects": objects,
            "model": model_used,
            "notes": f"{len(objects)} object(s) detected." if objects else "No objects detected.",
        }

    # image_task == "segment" — EXPERIMENTAL coarse tint-highlight + polygon recovery.
    # Step 1: detect (reliable boxes + labels + subject count).
    try:
        objects, detect_model_used = await _run_detect(client, frame_img)
    except RuntimeError as e:  # noqa: BLE001 — surface the real Gemini/transport error
        logger.error("vlm-image segment (detect phase): %s", e, exc_info=True)
        raise HTTPException(502, str(e))

    if not objects:
        return {
            **base_response,
            "objects": [],
            "model": detect_model_used,
            "notes": (
                "No subjects detected; nothing to tint-segment "
                "(EXPERIMENTAL coarse tint-highlight segmentation)."
            ),
        }

    # Step 2: assign one distinct tint color per subject and ask Nano Banana 2
    # to paint translucent washes over the ORIGINAL frame (no redraw).
    #
    # IMPORTANT: never cycle/reuse tint colors across subjects. Two subjects
    # sharing a color would collide in `_extract_tint_polygons` (which keys
    # extracted contours by color name and keeps only the largest one),
    # silently handing one subject's mask/box to another — a misattributed,
    # effectively fabricated result. If there are more subjects than palette
    # colors, the overflow subjects are left box-only (their real detect box
    # is still returned; they simply get no mask) and `degraded.masks_missing`
    # is set below to make that explicit.
    tintable_count = min(len(objects), len(_TINT_SPECS))
    overflow_count = len(objects) - tintable_count
    assignments = list(zip(objects[:tintable_count], _TINT_SPECS[:tintable_count]))
    tint_prompt = _build_tint_prompt([(obj["label"], spec) for obj, spec in assignments])

    try:
        tint_response = await client.aio.models.generate_content(
            model=SEGMENT_TINT_MODEL,
            contents=[tint_prompt, frame_img],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as e:  # noqa: BLE001 — surface the real Gemini/transport error
        logger.error("vlm-image segment (tint phase): Gemini call failed: %s", e, exc_info=True)
        raise HTTPException(502, f"Gemini tint-segmentation call failed: {e}")

    tint_image_bytes, tint_notes = _extract_inline_image_and_text(tint_response)

    final_objects = [dict(obj) for obj in objects]
    degraded = None

    if tint_image_bytes is None:
        # Real Gemini call succeeded but produced no image part — this is a
        # "masks missing" degrade, not a transport failure. Real detect boxes
        # are still returned (never a 502 for a missing mask).
        degraded = {
            "masks_missing": True,
            "reason": "Gemini tint model returned no image data",
        }
    else:
        try:
            tint_img = await loop.run_in_executor(
                None, lambda: Image.open(io.BytesIO(tint_image_bytes)).convert("RGB")
            )
        except Exception as e:  # noqa: BLE001 — degrade to boxes, don't 502 on a bad tint image
            logger.warning("vlm-image segment: tint image decode failed: %s", e)
            tint_img = None

        if tint_img is None:
            degraded = {
                "masks_missing": True,
                "reason": "Gemini tint image could not be decoded",
            }
        else:
            try:
                extracted = await loop.run_in_executor(
                    None,
                    _extract_tint_polygons,
                    tint_img,
                    frame_img.size,
                    [spec for _, spec in assignments],
                )
            except Exception as e:  # noqa: BLE001 — degrade to boxes on an extraction crash
                logger.warning(
                    "vlm-image segment: tint polygon extraction crashed: %s", e, exc_info=True
                )
                extracted = {}

            masks_found = 0
            for i, (_, spec) in enumerate(assignments):
                hit = extracted.get(spec["name"])
                if hit:
                    final_objects[i]["mask"] = {"kind": "polygon", "points": hit["points"]}
                    final_objects[i]["box_2d"] = hit["box_2d"]  # tint-derived bbox is tighter
                    masks_found += 1

            if masks_found == 0:
                degraded = {
                    "masks_missing": True,
                    "reason": "tint extraction produced no usable masks",
                }

    if overflow_count > 0:
        overflow_reason = (
            f"more subjects ({len(objects)}) than tint palette ({len(_TINT_SPECS)}); "
            f"{overflow_count} left box-only"
        )
        if degraded is None:
            degraded = {"masks_missing": True, "reason": overflow_reason}
        else:
            degraded["masks_missing"] = True
            degraded["reason"] = f"{degraded['reason']}; {overflow_reason}"

    notes = (
        f"EXPERIMENTAL coarse tint-highlight segmentation ({detect_model_used} boxes + "
        f"{SEGMENT_TINT_MODEL} tint recovery). Masks are approximate silhouettes recovered "
        "from a translucent color wash via HSV thresholding — not pixel-precise segmentation."
    )
    if tint_notes:
        notes += f" Gemini notes: {tint_notes}"

    return {
        **base_response,
        "objects": final_objects,
        "degraded": degraded,
        "model": f"{detect_model_used}+{SEGMENT_TINT_MODEL}",
        "notes": notes,
    }
