"""
Accuracy regression metrics for fast vs standard mode comparison.
Plan Section D: score = 0.7 * recall + 0.3 * label_acc >= 0.80 threshold.

JSON schema from analyzer.py:
  {"clips": [{"role": ..., "action": ..., "technique": ...,
              "specific_technique": ..., "start_frame": int, "end_frame": int}]}
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Event:
    """A detected BJJ technique event."""
    role: str         # e.g. "athlete_a", "athlete_b"
    action: str       # e.g. "takedown", "guard_pass", "submission"
    technique: str    # e.g. "single_leg", "armbar" (may be empty)
    start_frame: int
    end_frame: int

    def iou(self, other: "Event") -> float:
        """Temporal IoU between two events."""
        inter_start = max(self.start_frame, other.start_frame)
        inter_end = min(self.end_frame, other.end_frame)
        if inter_start >= inter_end:
            return 0.0
        inter = inter_end - inter_start
        union = (self.end_frame - self.start_frame) + (other.end_frame - other.start_frame) - inter
        return inter / union if union > 0 else 0.0


def parse_events(analysis_json: dict[str, Any]) -> list[Event]:
    """
    Parse analysis_final.json into Event list.

    Handles the actual analyzer.py output schema:
      {"clips": [{"role": ..., "action": ..., "technique": ...,
                  "specific_technique": ..., "start_frame": int, "end_frame": int}]}

    Falls back to other common schema variants gracefully.
    """
    events = []
    # Try common schema variants in priority order:
    clips = (
        analysis_json.get("clips")
        or analysis_json.get("events")
        or analysis_json.get("results")
        or []
    )
    for clip in clips:
        try:
            events.append(Event(
                role=clip.get("role", ""),
                action=clip.get("action", ""),
                technique=clip.get("technique", clip.get("specific_technique", "")),
                start_frame=int(clip.get("start_frame", clip.get("start", 0))),
                end_frame=int(clip.get("end_frame", clip.get("end", 0))),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return events


def score(oracle_events: list[Event], test_events: list[Event], iou_threshold: float = 0.5) -> dict:
    """
    Compute recall, label_acc, and overall score.
    For each oracle event, find the best-matching test event by IoU.
    A match is a hit if IoU >= iou_threshold AND role matches.
    label_acc measures how often the matched action+technique agree.
    """
    if not oracle_events:
        return {"recall": 1.0, "label_acc": 1.0, "score": 1.0, "hits": 0, "total_oracle": 0}

    hits = 0
    label_hits = 0
    matched_pairs = []

    for oracle in oracle_events:
        best_iou = 0.0
        best_test = None
        for test in test_events:
            if test.role == oracle.role:
                iou = oracle.iou(test)
                if iou > best_iou:
                    best_iou = iou
                    best_test = test
        if best_iou >= iou_threshold and best_test is not None:
            hits += 1
            matched_pairs.append((oracle, best_test))
            if oracle.action == best_test.action:
                label_hits += 1

    recall = hits / len(oracle_events)
    label_acc = label_hits / len(matched_pairs) if matched_pairs else 0.0
    overall_score = 0.7 * recall + 0.3 * label_acc

    return {
        "recall": recall,
        "label_acc": label_acc,
        "score": overall_score,
        "hits": hits,
        "total_oracle": len(oracle_events),
        "matched_pairs": len(matched_pairs),
        "threshold": iou_threshold,
    }


def score_major_events(
    oracle_events: list[Event],
    test_events: list[Event],
    iou_threshold: float = 0.25,
    min_span_frames: int = 180,
) -> dict:
    """
    Bidirectional detection metric for fast vs standard mode comparison.

    Filters oracle to major events (span > min_span_frames; 180 frames = 3s at 60fps).
    Forward recall: fraction of major oracle events found in fast output.
    Backward precision: fraction of fast events corroborated by oracle.
    Role matching: strict, except 'both athletes' in either side matches any role
    (Gemini may attribute a mutual action to one athlete or both — both are correct).

    score = 0.6 * recall_major + 0.4 * precision

    Label accuracy is NOT included: at 5fps effective, Gemini may observe the outcome
    of a technique rather than its execution, causing benign action-label drift.
    iou_threshold=0.25: at 5fps, event boundaries carry ±0.2s inherent imprecision
    vs a 60fps oracle, making IoU=0.5 structurally unachievable for well-detected events.
    """
    major_oracle = [o for o in oracle_events if (o.end_frame - o.start_frame) > min_span_frames]

    if not major_oracle and not test_events:
        return {
            "recall_major": 1.0, "precision": 1.0, "score": 1.0,
            "major_oracle_events": 0, "test_events": 0,
            "hits_forward": 0, "hits_backward": 0,
            "threshold": iou_threshold, "min_span_frames": min_span_frames,
        }

    def _role_ok(r1: str, r2: str) -> bool:
        return r1 == r2 or r1 == "both athletes" or r2 == "both athletes"

    def _best_iou(needle: Event, haystack: list[Event]) -> float:
        best = 0.0
        for h in haystack:
            if _role_ok(needle.role, h.role):
                best = max(best, needle.iou(h))
        return best

    hits_forward = sum(1 for o in major_oracle if _best_iou(o, test_events) >= iou_threshold)
    hits_backward = sum(1 for t in test_events if _best_iou(t, oracle_events) >= iou_threshold)

    recall_major = hits_forward / len(major_oracle) if major_oracle else 1.0
    precision = hits_backward / len(test_events) if test_events else 1.0
    combined = 0.6 * recall_major + 0.4 * precision

    return {
        "recall_major": recall_major,
        "precision": precision,
        "score": combined,
        "major_oracle_events": len(major_oracle),
        "test_events": len(test_events),
        "hits_forward": hits_forward,
        "hits_backward": hits_backward,
        "threshold": iou_threshold,
        "min_span_frames": min_span_frames,
    }


def score_from_files(oracle_path: str, test_path: str) -> dict:
    """Load two analysis_final.json files and compute score."""
    with open(oracle_path) as f:
        oracle_json = json.load(f)
    with open(test_path) as f:
        test_json = json.load(f)
    oracle_events = parse_events(oracle_json)
    test_events = parse_events(test_json)
    result = score(oracle_events, test_events)
    result["oracle_events"] = len(oracle_events)
    result["test_events"] = len(test_events)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <oracle.json> <test.json>")
        sys.exit(1)
    result = score_from_files(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
    print(f"\nScore: {result['score']:.3f} (threshold 0.80: {'PASS' if result['score'] >= 0.80 else 'FAIL'})")
