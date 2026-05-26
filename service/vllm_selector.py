"""
Gemini-based athlete pre-selection (optional hint when YOLO returns >2 persons).

Uses the Google Gen AI SDK (same stack as analyzer.py). The suggestion is a HINT
for the human verifier — it is never applied directly. Any failure returns None.
"""
import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROMPT = """\
You are analyzing a Brazilian Jiu-Jitsu (BJJ) match frame. I will provide an image with numbered bounding boxes overlaid on detected persons. Your task is to identify exactly which two boxes belong to the competing athletes (not referees in black/white uniforms, not coaches at ringside, not audience members).

Respond ONLY with valid JSON in this exact format:
{"athlete_a": <box_number>, "athlete_b": <box_number>}

Where box_number is one of the provided candidate IDs. Choose the two individuals actively engaged in the match. If you cannot determine with confidence, respond: {"uncertain": true}
"""


async def suggest_athletes(
    frame_jpeg: bytes,
    candidates: list[dict],
    config,
) -> Optional[dict[str, Any]]:
    """
    Ask Gemini to suggest which two candidates are the athletes.

    Args:
        frame_jpeg: Raw JPEG bytes of the detection frame.
        candidates: List of {"candidate_id": int, "box": [...], "confidence": float}.
        config: ServiceConfig (needs gemini_api_key, gemini_athlete_suggest_model).

    Returns:
        {"athlete_a": [...], "athlete_b": [...], "suggestion_model": str}
        or None on any failure or missing API key.
    """
    if not candidates:
        return None

    if not getattr(config, "gemini_api_key", ""):
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai not installed; athlete suggestion unavailable")
        return None

    model_id = getattr(
        config, "gemini_athlete_suggest_model", "gemini-3.1-flash-lite",
    )

    try:
        annotated_jpeg = _annotate_frame(frame_jpeg, candidates)

        candidate_desc = "\n".join(
            f"  Box {c['candidate_id']}: confidence={c['confidence']:.2f}, "
            f"coords=[{', '.join(str(round(v)) for v in c['box'])}]"
            for c in candidates
        )
        user_text = f"Candidates:\n{candidate_desc}\n\nWhich two are the competing athletes?"

        client = genai.Client(
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(getattr(config, "gemini_request_timeout_ms", 600_000)),
            ),
        )
        image_part = types.Part.from_bytes(
            data=annotated_jpeg, mime_type="image/jpeg",
        )

        _timeout_s = int(getattr(config, "gemini_request_timeout_ms", 600_000)) / 1000.0
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_id,
                contents=[_PROMPT + "\n\n" + user_text, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=128,
                ),
            ),
            timeout=_timeout_s,
        )

        raw_text = (response.text or "").strip()
        return _parse_response(raw_text, candidates, model_id)

    except Exception as e:
        logger.warning(
            "Gemini suggest_athletes failed (%s): %s — model=%s",
            type(e).__name__, e, model_id,
        )
        return None


def _annotate_frame(frame_jpeg: bytes, candidates: list[dict]) -> bytes:
    """Draw candidate IDs on a copy of the frame and re-encode as JPEG."""
    import cv2
    import numpy as np

    buf = np.frombuffer(frame_jpeg, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return frame_jpeg

    for c in candidates:
        box = c["box"]
        cid = c["candidate_id"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, str(cid), (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    ok, buf_out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf_out.tobytes() if ok else frame_jpeg


def _parse_response(
    text: str,
    candidates: list[dict],
    model_name: str,
) -> Optional[dict[str, Any]]:
    """Parse Gemini JSON response and validate box indices."""
    text = text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Gemini returned non-JSON: %r", text[:500])
        return None

    if data.get("uncertain"):
        logger.info("Gemini indicated uncertainty — no suggestion provided")
        return None

    id_a = data.get("athlete_a")
    id_b = data.get("athlete_b")

    if id_a is None or id_b is None or id_a == id_b:
        logger.warning("Gemini response missing or duplicate IDs: %r", data)
        return None

    cid_map = {c["candidate_id"]: c["box"] for c in candidates}
    if id_a not in cid_map or id_b not in cid_map:
        logger.warning("Gemini returned unknown candidate IDs %s/%s", id_a, id_b)
        return None

    box_a = cid_map[id_a]
    box_b = cid_map[id_b]

    if (box_a[2] - box_a[0]) <= 0 or (box_a[3] - box_a[1]) <= 0:
        return None
    if (box_b[2] - box_b[0]) <= 0 or (box_b[3] - box_b[1]) <= 0:
        return None

    return {
        "athlete_a": box_a,
        "athlete_b": box_b,
        "suggestion_model": model_name,
        # Back-compat for stored checkpoints / older clients
        "vllm_model": model_name,
    }
