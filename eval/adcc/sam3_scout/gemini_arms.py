"""Gemini driver for SAM3 scout Arms A/B/C.

Hard locks (founder):
  model = gemini-3.5-flash
  fps = 10
  frame_quality = high
  never gemini-3.1-flash-lite
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from eval.adcc.sam3_scout import (
    FORBIDDEN_GEMINI_MODELS,
    SCOUT_FPS,
    SCOUT_FRAME_QUALITY,
    SCOUT_GEMINI_MODEL,
)

SCOUT_SYSTEM_PROMPT = (
    "You are scoring a short Brazilian Jiu-Jitsu match clip. "
    "Return only the structured JSON schema requested. "
    "Timestamps are clip-relative seconds starting at 0."
)

SCOUT_USER_PROMPT = (
    "Identify the primary scoreable action in this clip. "
    "Return a JSON array of events with fields: "
    "start_s, end_s, action_class, outcome. "
    "action_class must be one of: "
    "submission_arm_lock, submission_choke, submission_leg_lock, submission_unclear, "
    "sweep, back_take, guard_pass, escape, transition, takedown_attempt, scramble. "
    "outcome must be one of: successful, unsuccessful, unclear. "
    "Prefer one event when the clip centers on a single gold action."
)


class ScoutGeminiError(ValueError):
    pass


def assert_scout_gemini_contract(model: str, fps: float, frame_quality: str) -> None:
    """Reject forbidden models / wrong knobs before any API spend."""
    if model in FORBIDDEN_GEMINI_MODELS or "flash-lite" in model:
        raise ScoutGeminiError(
            f"forbidden Gemini model {model!r}; scout requires {SCOUT_GEMINI_MODEL!r} only"
        )
    if model != SCOUT_GEMINI_MODEL:
        raise ScoutGeminiError(f"scout model must be {SCOUT_GEMINI_MODEL!r}, got {model!r}")
    if float(fps) != float(SCOUT_FPS):
        raise ScoutGeminiError(f"scout fps must be {SCOUT_FPS}, got {fps!r}")
    if frame_quality != SCOUT_FRAME_QUALITY:
        raise ScoutGeminiError(f"scout frame_quality must be {SCOUT_FRAME_QUALITY!r}, got {frame_quality!r}")


@dataclass
class ArmCallResult:
    arm: str
    clip_id: str
    model: str
    fps: float
    frame_quality: str
    video_path: str
    events: list[dict]
    raw_response_text: str
    usage_metadata: dict
    model_ms: float
    dollars_estimate: Optional[float]


def _media_resolution_high(types_mod):
    return types_mod.MediaResolution.MEDIA_RESOLUTION_HIGH


def run_arm_on_local_video(
    *,
    arm: str,
    clip_id: str,
    video_path: str,
    gemini_api_key: Optional[str] = None,
    model: str = SCOUT_GEMINI_MODEL,
    fps: float = SCOUT_FPS,
    frame_quality: str = SCOUT_FRAME_QUALITY,
    window_start_abs_s: float = 0.0,
) -> ArmCallResult:
    """Upload a local MP4 and run one Gemini judgment call.

    Returned event timestamps are converted to **match-absolute** seconds by
    adding ``window_start_abs_s`` (the gold window start on the source VOD).
    """
    assert_scout_gemini_contract(model, fps, frame_quality)
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GMINI_API_KEY")
    if not api_key:
        raise ScoutGeminiError("GEMINI_API_KEY / GMINI_API_KEY not set")
    if not Path(video_path).is_file():
        raise ScoutGeminiError(f"video not found: {video_path}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=video_path)
    # Wait until ACTIVE if the SDK exposes state
    for _ in range(60):
        meta = client.files.get(name=uploaded.name)
        state = getattr(meta, "state", None)
        if state is None or str(state).endswith("ACTIVE") or str(state) == "ACTIVE":
            break
        time.sleep(1.0)

    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "start_s": types.Schema(type=types.Type.NUMBER),
                "end_s": types.Schema(type=types.Type.NUMBER),
                "action_class": types.Schema(type=types.Type.STRING),
                "outcome": types.Schema(type=types.Type.STRING),
            },
            required=["start_s", "end_s", "action_class", "outcome"],
        ),
    )
    video_part = types.Part(
        file_data=types.FileData(file_uri=uploaded.uri, mime_type=getattr(uploaded, "mime_type", None) or "video/mp4"),
        video_metadata=types.VideoMetadata(fps=float(fps)),
    )
    gen_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=SCOUT_SYSTEM_PROMPT,
        media_resolution=_media_resolution_high(types),
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=SCOUT_USER_PROMPT), video_part])],
        config=gen_config,
    )
    model_ms = (time.perf_counter() - t0) * 1000.0
    raw = response.text or ""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ScoutGeminiError(f"non-JSON response for {clip_id}/{arm}: {e}") from e
    if not isinstance(parsed, list):
        raise ScoutGeminiError(f"expected JSON array for {clip_id}/{arm}, got {type(parsed).__name__}")

    events: list[dict] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ScoutGeminiError(f"item {i} not an object: {item!r}")
        try:
            start_s = float(item["start_s"]) + float(window_start_abs_s)
            end_s = float(item["end_s"]) + float(window_start_abs_s)
        except (KeyError, TypeError, ValueError) as e:
            raise ScoutGeminiError(f"item {i} bad timestamps: {item!r}") from e
        events.append({
            "start_s": start_s,
            "end_s": end_s,
            "action_class": str(item["action_class"]),
            "outcome": str(item.get("outcome", "unclear")),
            "arm": arm,
            "clip_id": clip_id,
        })

    usage: dict[str, Any] = {}
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        usage = {
            "prompt_token_count": getattr(um, "prompt_token_count", None),
            "candidates_token_count": getattr(um, "candidates_token_count", None),
            "total_token_count": getattr(um, "total_token_count", None),
        }

    return ArmCallResult(
        arm=arm,
        clip_id=clip_id,
        model=model,
        fps=float(fps),
        frame_quality=frame_quality,
        video_path=video_path,
        events=events,
        raw_response_text=raw,
        usage_metadata=usage,
        model_ms=model_ms,
        dollars_estimate=None,
    )


def run_all_arms_for_clip(
    *,
    clip: dict,
    arm_a_video: str,
    arm_b_video: str,
    arm_c_video: str,
    gemini_api_key: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Arms A/B/C and optionally persist a cost/result JSON."""
    results = {}
    for arm, path in (("A", arm_a_video), ("B", arm_b_video), ("C", arm_c_video)):
        results[arm] = asdict(
            run_arm_on_local_video(
                arm=arm,
                clip_id=clip["clip_id"],
                video_path=path,
                gemini_api_key=gemini_api_key,
                window_start_abs_s=float(clip["window_start_s"]),
            )
        )
    payload = {"clip_id": clip["clip_id"], "match_id": clip["match_id"], "arms": results}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{clip['clip_id']}_gemini_arms.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        payload["path"] = str(path)
    return payload
