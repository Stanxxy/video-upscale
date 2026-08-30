"""Build a viewer JSON for ``viewer/index.html``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.adcc.sam3_scout.advisory_metrics import compute_advisory_metrics


def build_viewer_entry(
    *,
    clip_id: str,
    raw_mp4: str,
    mask_preview_mp4: str | None,
    arm_b_mp4: str | None,
    arm_c_mp4: str | None,
    tracks_jsonl: str | None = None,
    advisory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if advisory is None and tracks_jsonl:
        advisory = compute_advisory_metrics(tracks_jsonl)
    return {
        "clip_id": clip_id,
        "videos": {
            "A": raw_mp4,
            "mask": mask_preview_mp4,
            "B": arm_b_mp4,
            "C": arm_c_mp4,
        },
        "advisory": advisory or {},
    }


def write_viewer_bundle(entries: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clips": entries}
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
