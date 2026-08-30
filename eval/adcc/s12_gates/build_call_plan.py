"""Full-batch dry-run CALL PLAN — enumerates EVERY individual Gemini call the
full 216-call batch would make (window id, arm, axis, model, fps,
media_resolution, ``ingestion_path``, ordinal) as one JSON artifact, with
ZERO additional Gemini calls (pure enumeration over the already-frozen
manifests). Written for Bezos's adversarial pass; ``--execute`` is
authorized separately and is NOT gated by this script at all.

**Regenerated 2026-07-26 (CEO instruction, item 6) after the §G.1 Files API
reroute lane was built and smoke-verified** (`files_api_reroute.py`): the 12
calls belonging to the 3 rerouted windows (Gate-1 `w37`; Gate-2
`occ0`/`occ11`) now carry ``"ingestion_path": "files_api"`` and NO
``known_risk`` — the smoke test (2026-07-26) directly confirmed a
Files-API-uploaded, file-relative-offset call against this exact video
succeeds (2/2 real Gemini calls, ``position``/``technique`` against
`occ0`, both `status="ok"`, `technique` result EXACTLY matched gold:
`submission_arm_lock`/`unsuccessful`). Every other window/clip keeps
``"ingestion_path": "youtube_uri"`` and no `known_risk` flag (the original
`pwGbW5GZgfc` 400-on-every-offset finding was specific to that one video,
confirmed via 6 direct SDK probes during the 2026-07-25 smoke test — no
other corpus video has shown this failure mode).

Run from the repo root:

    .venv/bin/python -m eval.adcc.s12_gates.build_call_plan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.adcc.s12_gates import files_api_reroute as reroute
from eval.adcc.s12_gates import gate_common as gc


def _ingestion_fields(window_id: str) -> tuple[str, "str | None"]:
    """Returns ``(ingestion_path, known_risk)`` for a window/clip. The 3
    §G.1-rerouted windows get ``"files_api"`` + a downgraded, non-blocking
    note (smoke-confirmed working, not a live risk anymore); everything
    else gets ``"youtube_uri"`` + no risk flag."""
    if window_id in reroute.REROUTE_WINDOW_IDS:
        return "files_api", (
            "REROUTED (§G.1) — this window's video (pwGbW5GZgfc) 400s on every native YouTube "
            "ingestion offset/fps; routed through the Files API upload lane instead. Smoke-verified "
            "2026-07-26: 2/2 real Gemini calls (position/technique) against occ0 succeeded via "
            "files_api ingestion with file-relative video_metadata offsets, technique result exactly "
            "matched gold. Not a live risk for the full batch — informational tag only."
        )
    return "youtube_uri", None


def _gate1_calls(manifest: dict) -> list[dict]:
    entries = []
    for w in manifest["windows"]:
        ingestion_path, known_risk = _ingestion_fields(w["window_id"])
        base = {
            "gate": "gate1", "window_id": w["window_id"], "match_id": w["match_id"],
            "gate5_straddler": w["gate5_straddler"], "gold_action_class": w["gold_event"]["action_class"],
            "video_url_watch": w["video_url_watch"],
            "window_start_s": w["window_start_s"], "window_end_s": w["window_end_s"],
            "model": gc.MODEL_ID, "fps": gc.FPS, "thinking": None,
            "ingestion_path": ingestion_path, "known_risk": known_risk,
        }
        entries.append({**base, "arm": "arm_s", "axis": "arm_s", "media_resolution": None})
        entries.append({**base, "arm": "arm_t", "axis": "position", "media_resolution": None})
        entries.append({**base, "arm": "arm_t", "axis": "technique", "media_resolution": None})
        entries.append({**base, "arm": "arm_t", "axis": "validator", "media_resolution": "high"})
    return entries


def _gate2_calls(manifest: dict) -> list[dict]:
    entries = []
    for c in manifest["clips"]:
        ingestion_path, known_risk = _ingestion_fields(c["window_id"])
        base = {
            "gate": "gate2", "window_id": c["window_id"], "match_id": c["match_id"],
            "gold_action_class": c["gold_event"]["action_class"], "gold_outcome": c["gold_event"]["outcome"],
            "video_url_watch": c["video_url_watch"],
            "window_start_s": c["window_start_s"], "window_end_s": c["window_end_s"],
            "model": gc.MODEL_ID, "fps": gc.FPS, "thinking": None,
            "ingestion_path": ingestion_path, "known_risk": known_risk,
        }
        entries.append({**base, "arm": "gate2", "axis": "position", "media_resolution": None})
        entries.append({**base, "arm": "gate2", "axis": "technique", "media_resolution": None})
        entries.append({**base, "arm": "gate2", "axis": "validator_low", "media_resolution": "low"})
        entries.append({**base, "arm": "gate2", "axis": "validator_high", "media_resolution": "high"})
    return entries


def build_plan() -> dict:
    gate1_manifest = gc.read_json(gc.GATE1_MANIFEST_PATH)
    gate2_manifest = gc.read_json(gc.GATE2_MANIFEST_PATH)
    calls = _gate1_calls(gate1_manifest) + _gate2_calls(gate2_manifest)
    for i, c in enumerate(calls, start=1):
        c["ordinal"] = i

    files_api_count = sum(1 for c in calls if c["ingestion_path"] == "files_api")
    known_risk_count = sum(1 for c in calls if c["known_risk"])
    return {
        "description": (
            "Full-batch dry-run call plan for Gates 1 & 2 — zero Gemini calls spent producing this "
            "file. Enumerates every planned call (window id, arm, axis, model, fps, media_resolution, "
            "ingestion_path, ordinal) for Bezos's adversarial pass. --execute is a separate, "
            "founder-gated action not controlled by this file."
        ),
        "gate1_call_count": len(gate1_manifest["windows"]) * 4,
        "gate2_call_count": len(gate2_manifest["clips"]) * 4,
        "total_call_count": len(calls),
        "files_api_call_count": files_api_count,
        "youtube_uri_call_count": len(calls) - files_api_count,
        "known_risk_call_count": known_risk_count,
        "known_risk_summary": (
            f"{known_risk_count} of {len(calls)} planned calls carry an informational note "
            "(§G.1 reroute — smoke-confirmed working 2026-07-26, not a live risk). Zero planned calls "
            "have an UNRESOLVED known risk as of this regeneration."
            if known_risk_count else "none — all previously-flagged calls were resolved by the §G.1 reroute fix."
        ),
        "gate1_manifest_sha256": gate1_manifest["manifest_sha256"],
        "gate2_manifest_sha256": gate2_manifest["manifest_sha256"],
        "calls": calls,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(gc.S12_GATES_DIR / "full_batch_call_plan.json"))
    args = parser.parse_args(argv)

    plan = build_plan()
    gc.write_json(Path(args.out), plan)
    print(f"wrote {args.out} — {plan['total_call_count']} planned calls "
          f"({plan['gate1_call_count']} gate1 + {plan['gate2_call_count']} gate2), "
          f"{plan['files_api_call_count']} files_api / {plan['youtube_uri_call_count']} youtube_uri, "
          f"{plan['known_risk_call_count']} flagged known_risk (informational only)", file=sys.stderr)


if __name__ == "__main__":
    main()
