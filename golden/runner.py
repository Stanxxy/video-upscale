"""One census call per window with an on-disk cache: a rerun only calls windows that are missing or failed."""
from __future__ import annotations

import json
from pathlib import Path

from golden import census, media
from golden.openrouter import complete

ATTEMPTS = 2


def call_window(entry: dict, scope: census.Scope, index: int, video: Path, crops, path: Path) -> dict:
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("ok"):
            return cached
    request = census.build_request(scope, index)
    w = request["window"]
    clip = media.cut(video, w["start_ms"], w["end_ms"]).read_bytes()
    messages = census.chat_messages(request, clip, crops)
    attempts, record = [], None
    for _ in range(ATTEMPTS):
        try:
            reply = complete(entry["model"], messages, schema=request["response_schema"], reasoning=entry.get("reasoning", "high"))
        except Exception as error:  # provider and network errors are recorded, then retried once
            attempts.append(dict(error=f"{type(error).__name__}: {str(error)[:300]}"))
            continue
        try:
            parsed = census.parse(request, reply["text"], index)
            record = dict(ok=True, window=w, index=index, reply=reply, census=parsed.model_dump(mode="json"), attempts=attempts)
            break
        except Exception as error:
            attempts.append(dict(error=f"invalid answer: {type(error).__name__}: {str(error)[:300]}", reply=reply))
    if record is None:
        record = dict(ok=False, window=w, index=index, attempts=attempts)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(path)
    return record


def record_cost(rec: dict) -> float:
    return sum((a.get("reply") or {}).get("cost") or 0 for a in rec.get("attempts", [])) + ((rec.get("reply") or {}).get("cost") or 0)
