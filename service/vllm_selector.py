"""
VLLM-based athlete pre-selection.

Uses LM Studio's OpenAI-compatible endpoint (Qwen VL 8B by default) to
suggest which two bounding boxes correspond to the competing athletes when
YOLO26 returns more than 2 candidates.

The result is a HINT for the human verifier — it is never applied directly.
Any failure (network, parse, validation) returns None so the caller can
proceed with unfiltered YOLO26 candidates.
"""
import base64
import json
import logging
from typing import Optional

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
) -> Optional[dict]:
    """
    Ask the local VLLM to suggest which two candidates are the athletes.

    Args:
        frame_jpeg: Raw JPEG bytes of the detection frame.
        candidates: List of {"candidate_id": int, "box": [...], "confidence": float}.
        config: ServiceConfig (needs vllm_base_url, vllm_model, vllm_timeout_sec).

    Returns:
        {"athlete_a": [x1,y1,x2,y2], "athlete_b": [x1,y1,x2,y2], "vllm_model": str}
        or None on any failure.
    """
    if not candidates:
        return None

    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed; VLLM pre-filter unavailable")
        return None

    try:
        # Build annotated image showing candidate IDs
        annotated_jpeg = _annotate_frame(frame_jpeg, candidates)
        frame_b64 = base64.b64encode(annotated_jpeg).decode()

        candidate_desc = "\n".join(
            f"  Box {c['candidate_id']}: confidence={c['confidence']:.2f}, "
            f"coords=[{', '.join(str(round(v)) for v in c['box'])}]"
            for c in candidates
        )
        user_text = f"Candidates:\n{candidate_desc}\n\nWhich two are the competing athletes?"

        client = openai.AsyncOpenAI(
            base_url=config.vllm_base_url,
            api_key="lm-studio",
        )

        response = await client.chat.completions.create(
            model=config.vllm_model,
            messages=[
                {
                    "role": "system",
                    "content": _PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            max_tokens=64,
            temperature=0.0,
            timeout=config.vllm_timeout_sec,
        )

        raw_text = response.choices[0].message.content.strip()
        return _parse_response(raw_text, candidates, config.vllm_model)

    except Exception as e:
        logger.warning("VLLM suggest_athletes failed (%s): %s", type(e).__name__, e)
        return None


def _annotate_frame(frame_jpeg: bytes, candidates: list[dict]) -> bytes:
    """Draw candidate IDs on a copy of the frame and re-encode as JPEG."""
    import cv2
    import numpy as np

    buf = np.frombuffer(frame_jpeg, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return frame_jpeg

    h, w = frame.shape[:2]
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
) -> Optional[dict]:
    """Parse VLLM JSON response and validate box indices."""
    # Strip markdown fences if present
    text = text.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("VLLM returned non-JSON: %r", text)
        return None

    if data.get("uncertain"):
        logger.info("VLLM indicated uncertainty — no suggestion provided")
        return None

    id_a = data.get("athlete_a")
    id_b = data.get("athlete_b")

    if id_a is None or id_b is None or id_a == id_b:
        logger.warning("VLLM response missing or duplicate IDs: %r", data)
        return None

    cid_map = {c["candidate_id"]: c["box"] for c in candidates}
    if id_a not in cid_map or id_b not in cid_map:
        logger.warning("VLLM returned unknown candidate IDs %s/%s", id_a, id_b)
        return None

    box_a = cid_map[id_a]
    box_b = cid_map[id_b]

    # Basic sanity: boxes must have positive area
    if (box_a[2] - box_a[0]) <= 0 or (box_a[3] - box_a[1]) <= 0:
        return None
    if (box_b[2] - box_b[0]) <= 0 or (box_b[3] - box_b[1]) <= 0:
        return None

    return {
        "athlete_a": box_a,
        "athlete_b": box_b,
        "vllm_model": model_name,
    }
