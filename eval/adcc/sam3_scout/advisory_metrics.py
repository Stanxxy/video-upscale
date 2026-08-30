"""Soft advisory metrics for SAM3 tracks (non-blocking — never a stop-gate)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compute_advisory_metrics(tracks_jsonl: str | Path) -> dict[str, Any]:
    rows = []
    with open(tracks_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        return {
            "n_frames": 0,
            "frames_with_detections": 0,
            "detection_coverage": 0.0,
            "mean_instances": 0.0,
            "frames_with_ge_2": 0,
            "ge_2_coverage": 0.0,
            "advisory_notes": ["empty tracks"],
        }
    with_det = sum(1 for r in rows if r.get("boxes"))
    ge2 = sum(1 for r in rows if len(r.get("boxes") or []) >= 2)
    mean_inst = sum(len(r.get("boxes") or []) for r in rows) / n
    notes = []
    cov = with_det / n
    if cov < 0.5:
        notes.append("low detection coverage (<50%) — eyeball carefully")
    if ge2 / n < 0.3:
        notes.append("few frames with ≥2 athletes — union crop may wobble")
    if not notes:
        notes.append("advisory OK — still requires founder eyeball")
    return {
        "n_frames": n,
        "frames_with_detections": with_det,
        "detection_coverage": round(cov, 4),
        "mean_instances": round(mean_inst, 3),
        "frames_with_ge_2": ge2,
        "ge_2_coverage": round(ge2 / n, 4),
        "advisory_notes": notes,
    }
