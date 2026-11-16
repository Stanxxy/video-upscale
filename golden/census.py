"""The engine's sequence census, sent to any registry model through a replaceable provider.

The request is the engine's own: the same system prompt, the same initial prompt keys, the
same response schema, the same 20 s / 5 s-overlap tiling and the same span merge
(``bjj_analysis_core.sequences``). Two differences, forced by sending a cut clip instead of
a Gemini file offset: the clip starts at 0, so one text line gives its source start time;
and the fps is baked into the clip (providers sample frames their own way).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from bjj_analysis_core.mma_outputs import SequenceCensusV1
from bjj_analysis_core.packs import MMA_SEQUENCES
from bjj_analysis_core.requests import prompt_bundle
from bjj_analysis_core.sequences import merge_sequences, sequence_windows
from shared_lib.models.analysis_contracts import TimeRangeMs

WINDOW_MS, OVERLAP_MS = 20000, 5000
PLAYER_IDS = ("fighter-a", "fighter-b")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Scope:
    start_ms: int
    end_ms: int
    fighters: tuple[str, str]  # display names, ordered: fighter-a, fighter-b

    def windows(self) -> tuple[TimeRangeMs, ...]:
        run = SimpleNamespace(spec=SimpleNamespace(scope=TimeRangeMs(start_ms=self.start_ms, end_ms=self.end_ms)))
        config = SimpleNamespace(sequence_window_ms=WINDOW_MS, sequence_overlap_ms=OVERLAP_MS)
        return sequence_windows(run, config)


_BUNDLE = None


def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = prompt_bundle(MMA_SEQUENCES)
    return _BUNDLE


def build_request(scope: Scope, index: int) -> dict:
    """System prompt, initial prompt and schema for one window, as the engine builds them."""
    b = bundle()
    window = scope.windows()[index]
    initial = {
        "source_time_unit": "absolute_milliseconds",
        "scope": {"start_ms": scope.start_ms, "end_ms": scope.end_ms},
        "observation_window": window.model_dump(),
        "ordered_players": [{"player_id": p, "display_name": n} for p, n in zip(PLAYER_IDS, scope.fighters)],
        "position_taxonomy": b["positions"],
        "action_classes": b["action_classes"],
        "outcomes": b["outcomes"],
        **{key: b[key] for key in MMA_SEQUENCES.extra_prompt_keys},
        "stage_context": {"window": window.model_dump(), "window_index": index},
        "advisory_positions": None,
    }
    return dict(
        window=window.model_dump(),
        system_prompt=b["prompts"]["sequence_census"],
        initial_prompt=canonical_json(initial),
        response_schema=b["schemas"]["sequence_census"],
        clip_note=(f"The attached video is this observation window cut from the source video: its time 0 is "
                   f"source time {window.start_ms} ms, so add {window.start_ms} to every clip time and report "
                   f"absolute source milliseconds."),
    )


def chat_messages(request: dict, clip: bytes, crops: list[tuple[str, str, bytes]]) -> list[dict]:
    """OpenAI-compatible messages; crops are (player_id, display_name, png bytes)."""
    content = []
    for player_id, name, png in crops:
        content.append({"type": "text", "text": canonical_json({"reference_player_id": player_id, "display_name": name})})
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})
    content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(clip).decode()}})
    content.append({"type": "text", "text": request["clip_note"]})
    content.append({"type": "text", "text": request["initial_prompt"]})
    return [{"role": "system", "content": request["system_prompt"]}, {"role": "user", "content": content}]


def strip_fences(text: str) -> str:
    """Some providers wrap JSON in a code fence even with a JSON schema; nothing else is repaired."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def parse(request: dict, text: str, index: int) -> SequenceCensusV1:
    """Validate like the engine's parse_response: schema, window index, spans inside the window, actor ids."""
    census = SequenceCensusV1.model_validate(json.loads(strip_fences(text)))
    if census.window_index != index:
        raise ValueError("census answers a different window")
    w = request["window"]
    for s in census.sequences:
        if not w["start_ms"] <= s.start_ms < s.end_ms <= w["end_ms"]:
            raise ValueError("a sequence span falls outside the window")
        if s.actor_player_id is not None and s.actor_player_id not in PLAYER_IDS:
            raise ValueError("unknown actor id")
    return census


def merge(scope: Scope, censuses: list[SequenceCensusV1]):
    run = SimpleNamespace(spec=SimpleNamespace(scope=TimeRangeMs(start_ms=scope.start_ms, end_ms=scope.end_ms)))
    config = SimpleNamespace(sequence_window_ms=WINDOW_MS, sequence_overlap_ms=OVERLAP_MS)
    return merge_sequences(run, config, censuses, SimpleNamespace(positions=()))


def flat(batch) -> list[dict]:
    """Merged sequences as plain dicts (actor a/b, span, steps) for scoring and task building."""
    out = []
    for s in batch.sequences:
        actor = {"fighter-a": "a", "fighter-b": "b"}.get(str(s.actor_player_id) if s.actor_player_id else "", None)
        out.append(dict(
            actor=actor, sentinel=s.actor_sentinel and str(s.actor_sentinel.value if hasattr(s.actor_sentinel, "value") else s.actor_sentinel),
            start=s.bounds.start_ms, end=s.bounds.end_ms, summary=s.summary,
            steps=[dict(action=st.action.value if hasattr(st.action, "value") else st.action,
                        technique=getattr(st, "technique", None),
                        outcome=st.outcome.value if hasattr(st.outcome, "value") else st.outcome,
                        target=getattr(st, "strike_target", None), repeat=getattr(st, "repeat", 1) or 1,
                        landed=getattr(st, "landed", None), evidence=getattr(st, "evidence", None) or "")
                   for st in s.steps],
        ))
    return out


def read_crops(run_dir: Path, names: tuple[str, str]) -> list[tuple[str, str, bytes]]:
    return [(p, n, (run_dir / f"fighter-{i}.png").read_bytes()) for i, (p, n) in enumerate(zip(PLAYER_IDS, names))]
