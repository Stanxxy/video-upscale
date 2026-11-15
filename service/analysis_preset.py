"""Explicit new-run presets; never rewrites an admitted configuration artifact.

Provider qualification remains a separate host input, backed by live evidence.
``preset_bytes`` writes the exact artifact bytes published under
``qualified-presets/<name>-<sha256>.json`` (``config/analysis/*.json``).

Production presets (founder 2026-11-15): Gemini 4 Hydrogen on every model step and
no check steps, for both sports; MMA adds the ground re-scan (who is on top).
The gemini38 presets stay for runs admitted with them and for VLM Studio.
"""

import json
from pathlib import Path

from bjj_analysis_core.outputs import CandidateConfigV1
from bjj_analysis_core.packs import MMA, MMA_GROUND, MMA_SEQUENCES, MMA_STRIKES
from bjj_analysis_core.requests import CoreConfigV1, NodeConfigV1, prompt_bundle_sha256


def gemini38_preset() -> CoreConfigV1:
    node = NodeConfigV1(
        model="gemini-3.8-flash", thinking="high", media_resolution="low", fps=10
    )
    # Checks and validation confirm or reject a proposal in a short window; they
    # run at low thinking (founder 2026-11-14: high thinking made every check
    # 20-40 s and most of the spend). Scans and analysis keep high thinking.
    # Position checks see the scan's own previous state, so they run in parallel
    # (x-mma 2026-11-14: same checkpoint accuracy, 13.0 -> 4.8 min on 3 min of BJJ).
    check = node.model_copy(update={"thinking": "low"})
    return CoreConfigV1(
        prompt_bundle_sha256=prompt_bundle_sha256(),
        position_scan=node.model_copy(update={"fps": 1}),
        position_critique=check,
        technique_scan=node.model_copy(update={"fps": 1}),
        technique_critique=check,
        technique_analyze=node,
        technique_validate=check,
        candidates=CandidateConfigV1(),
        position_critique_context="proposed",
    )


def gemini38_mma_preset() -> CoreConfigV1:
    """MMA v1 (VLM Studio comparison only): the BJJ preset on the MMA pack, 5 fps scan.

    Production MMA runs the windowed strike workflow (``gemini38_mma_strikes_preset``);
    v1 reviews every strike through the per-candidate chain and is kept as its baseline.
    """
    bjj = gemini38_preset()
    return bjj.model_copy(
        update={
            "prompt_bundle_sha256": prompt_bundle_sha256(MMA),
            "technique_scan": bjj.technique_scan.model_copy(update={"fps": 5}),
        }
    )


def gemini38_mma_strikes_preset(*, audit=True) -> CoreConfigV1:
    """The production MMA preset: windowed strike census (workflow mma-windowed-strikes).

    Strikes: one census call per 20 s window at 10 fps (high thinking: it is the
    detector) and, when ``audit``, one low-thinking audit per window. Grappling keeps
    the per-candidate chain on a 2 fps grappling-only scan. Checks run at low thinking.
    """
    node = NodeConfigV1(
        model="gemini-3.8-flash", thinking="high", media_resolution="low", fps=10
    )
    check = node.model_copy(update={"thinking": "low"})
    return CoreConfigV1(
        prompt_bundle_sha256=prompt_bundle_sha256(MMA_STRIKES),
        position_scan=node.model_copy(update={"fps": 1}),
        position_critique=check,
        technique_scan=node.model_copy(update={"fps": 2}),
        technique_critique=check,
        technique_analyze=node,
        technique_validate=check,
        candidates=CandidateConfigV1(),
        strike_census=node,
        strike_audit=check if audit else None,
        knockdown_check=check,
        position_critique_context="proposed",
    )


CHECK_NODES = ("position_critique", "technique_critique", "technique_validate", "strike_audit", "knockdown_check")


def hydrogen_unchecked(config: CoreConfigV1) -> CoreConfigV1:
    """Every model step on gemini-4-hydrogen, no check steps.

    Scans, census and analysis keep their thinking; the first pass stands as-is
    (x-mma 2026-11-15: Hydrogen's checks changed 0 of 205 answers, and without them
    Hydrogen matched or beat the 3.8 Flash workflow on BJJ positions).
    """
    on = lambda node: None if node is None else node.model_copy(update={"model": "gemini-4-hydrogen"})
    return config.model_copy(update={
        **{n: None for n in CHECK_NODES},
        **{n: on(getattr(config, n)) for n in ("position_scan", "technique_scan", "technique_analyze", "strike_census")},
    })


def with_ground_rescan(config: CoreConfigV1, model: str = "gemini-4-hydrogen") -> CoreConfigV1:
    """The windowed MMA preset plus a ground re-scan node.

    Workflow mma-windowed-strikes-ground: each ground phase of the position scan is
    re-scanned at 5 fps in 20 s windows that start 5 s early; the model names the top
    fighter (bjj_analysis_core.grounds). x-mma 2026-11-15: who is on top 71% -> 91%.
    """
    return config.model_copy(update={
        "prompt_bundle_sha256": prompt_bundle_sha256(MMA_GROUND),
        "ground_rescan": NodeConfigV1(model=model, thinking="high", media_resolution="low", fps=5),
    })


def hydrogen_preset() -> CoreConfigV1:
    """The production BJJ preset: Hydrogen, no check steps."""
    return hydrogen_unchecked(gemini38_preset())


def hydrogen_mma_ground_preset() -> CoreConfigV1:
    """The production MMA preset: Hydrogen, no check steps, ground re-scan."""
    return with_ground_rescan(hydrogen_unchecked(gemini38_mma_strikes_preset()))


def hydrogen_mma_sequences_preset() -> CoreConfigV1:
    """MMA technique sequences (VLM Studio experiment; plan 2026-11-15-mma-technique-sequences).

    Hydrogen, no check steps. The position scan (1 fps) and ground re-scan (5 fps) are
    the production MMA preset's; one high-thinking census per 20 s window (5 s overlap)
    at 10 fps returns every fighter's chain of techniques. There is no technique scan,
    analysis or strike census.
    """
    node = NodeConfigV1(model="gemini-4-hydrogen", thinking="high", media_resolution="low", fps=10)
    return CoreConfigV1(
        prompt_bundle_sha256=prompt_bundle_sha256(MMA_SEQUENCES),
        position_scan=node.model_copy(update={"fps": 1}),
        position_critique=None,
        technique_scan=None,
        technique_critique=None,
        technique_analyze=None,
        technique_validate=None,
        candidates=None,
        position_critique_context="proposed",
        ground_rescan=node.model_copy(update={"fps": 5}),
        sequence_census=node,
    )


# Ground re-scan fields are written only into ground presets.
_GROUND_FIELDS = {"ground_rescan", "ground_window_ms", "ground_lookback_ms"}
# Windowed-workflow fields are written only into windowed presets.
_WINDOWED_FIELDS = {
    "strike_census", "strike_audit", "knockdown_check", "strike_window_ms", "strike_overlap_ms",
}

# Sequence fields are written only into sequence presets.
_SEQUENCE_FIELDS = {"sequence_census", "sequence_window_ms", "sequence_overlap_ms"}

# Published artifacts: file name -> generator.
PRESET_FILES = {
    "gemini38.json": gemini38_preset,
    "gemini38-mma-strikes.json": gemini38_mma_strikes_preset,
    "hydrogen.json": hydrogen_preset,
    "hydrogen-mma-ground.json": hydrogen_mma_ground_preset,
}


def preset_bytes(config: CoreConfigV1) -> bytes:
    """The canonical artifact bytes for one preset (indented JSON, trailing newline)."""
    data = config.model_dump(mode="json")
    if config.strike_census is None:
        data = {k: v for k, v in data.items() if k not in _WINDOWED_FIELDS}
    if config.ground_rescan is None:
        data = {k: v for k, v in data.items() if k not in _GROUND_FIELDS}
    if config.sequence_census is None:
        data = {k: v for k, v in data.items() if k not in _SEQUENCE_FIELDS}
    return (json.dumps(data, indent=2) + "\n").encode()


def write_preset_files(root: Path) -> None:
    for name, build in PRESET_FILES.items():
        (root / name).write_bytes(preset_bytes(build()))


if __name__ == "__main__":
    write_preset_files(Path(__file__).resolve().parents[1] / "config" / "analysis")
