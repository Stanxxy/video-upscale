"""Per-sport prompt/vocabulary/schema packs, selected by the admitted workflow.

The BJJ pack is the original workflow, byte for byte: its prompt bundle (and so
its pinned prompt_bundle_sha256) is exactly what this core produced before
sports existed. A run selects its pack by ``workflow_id`` and must carry the
pack's ``taxonomy_id``; no run ever falls through to another sport's pack.
"""

from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping

from shared_lib.models.position_taxonomy import position_taxonomy_manifest
from shared_lib.models.simplified_taxonomy import AXIS3_ACTION, AXIS4_OUTCOME
from shared_lib.models.sport_taxonomy import vocabulary

from bjj_analysis_core import MMA_GROUND_WORKFLOW_ID, MMA_STRIKES_WORKFLOW_ID, MMA_WORKFLOW_ID, WORKFLOW_ID
from bjj_analysis_core import mma_outputs as mma
from bjj_analysis_core.outputs import (
    PositionCritiqueInputV1,
    PositionCritiqueV1,
    PositionScanV1,
    TechniqueAnalysisV1,
    TechniqueCandidateV1,
    TechniqueDecisionV1,
    TechniqueScanV1,
    TemporalCritiqueV1,
)


@dataclass(frozen=True)
class AnalysisPack:
    workflow_id: str
    taxonomy_id: str
    prompts_dir: str
    outputs: Mapping[str, type]
    position_critique_input: type
    technique_candidate: type
    taxonomy: Mapping[str, object]
    # Prompt keys a sport adds to the initial prompt (none for BJJ).
    extra_prompt_keys: tuple
    # A pack that extends another reads the prompts it does not override from there.
    base_prompts_dir: str | None = None

    @property
    def windowed_strikes(self):
        """Strikes come from the per-window census, not the per-candidate chain."""
        return "strike_census" in self.outputs

    @property
    def ground_rescan(self):
        """Ground phases are re-scanned to re-check who is on top (grounds.py)."""
        return "ground_rescan" in self.outputs

    @property
    def position_scan(self):
        return self.outputs["position_scan"]

    @property
    def position_critique(self):
        return self.outputs["position_critique"]

    @property
    def technique_analysis(self):
        return self.outputs["technique_analyze"]

    @property
    def technique_decision(self):
        return self.outputs["technique_validate"]

    def prompt(self, node):
        own = files("bjj_analysis_core").joinpath(self.prompts_dir, f"{node}.txt")
        if not own.is_file() and self.base_prompts_dir is not None:
            own = files("bjj_analysis_core").joinpath(self.base_prompts_dir, f"{node}.txt")
        return own.read_text(encoding="utf-8")


def _outputs(scan, critique, analysis, decision):
    return MappingProxyType(
        {
            "position_scan": scan,
            "position_critique": critique,
            "technique_scan": TechniqueScanV1,
            "technique_critique": TemporalCritiqueV1,
            "technique_analyze": analysis,
            "technique_validate": decision,
        }
    )


BJJ = AnalysisPack(
    workflow_id=WORKFLOW_ID,
    taxonomy_id="bjj.v1",
    prompts_dir="prompts",
    outputs=_outputs(PositionScanV1, PositionCritiqueV1, TechniqueAnalysisV1, TechniqueDecisionV1),
    position_critique_input=PositionCritiqueInputV1,
    technique_candidate=TechniqueCandidateV1,
    taxonomy=MappingProxyType(
        {
            "positions": position_taxonomy_manifest(),
            "action_classes": list(AXIS3_ACTION),
            "outcomes": list(AXIS4_OUTCOME),
        }
    ),
    extra_prompt_keys=(),
)

_MMA_VOCAB = vocabulary("mma.v1").export()
_BJJ_POSITIONS = position_taxonomy_manifest()
MMA = AnalysisPack(
    workflow_id=MMA_WORKFLOW_ID,
    taxonomy_id="mma.v1",
    prompts_dir="prompts_mma",
    outputs=_outputs(
        mma.MmaPositionScanV1,
        mma.MmaPositionCritiqueV1,
        mma.MmaTechniqueAnalysisV1,
        mma.MmaTechniqueDecisionV1,
    ),
    position_critique_input=mma.MmaPositionCritiqueInputV1,
    technique_candidate=mma.MmaTechniqueCandidateV1,
    taxonomy=MappingProxyType(
        {
            "positions": {
                "positions": _MMA_VOCAB["positions"],
                "roles": _MMA_VOCAB["roles"],
                "role_pairs": _MMA_VOCAB["role_pairs"],
                "unknown_role_pair": _BJJ_POSITIONS["unknown_role_pair"],
                "inactive_role_pair": _BJJ_POSITIONS["inactive_role_pair"],
                "interruption_kinds": _MMA_VOCAB["interruption_kinds"],
                "position_modifiers": _MMA_VOCAB["position_modifiers"],
            },
            "action_classes": _MMA_VOCAB["actions"],
            "outcomes": _MMA_VOCAB["outcomes"],
            "strike_classes": _MMA_VOCAB["strike_actions"],
            "strike_targets": _MMA_VOCAB["strike_targets"],
        }
    ),
    extra_prompt_keys=("strike_classes", "strike_targets"),
)

# MMA with a windowed strike census (bjj_analysis_core.strikes): the technique scan
# finds grappling attacks only; strikes are counted per short window, in parallel.
MMA_STRIKES = AnalysisPack(
    workflow_id=MMA_STRIKES_WORKFLOW_ID,
    taxonomy_id="mma.v1",
    prompts_dir="prompts_mma_strikes",
    outputs=MappingProxyType(
        {
            **MMA.outputs,
            "strike_census": mma.StrikeCensusV1,
            "strike_audit": mma.StrikeAuditV1,
            "knockdown_check": mma.KnockdownCheckV1,
        }
    ),
    position_critique_input=mma.MmaPositionCritiqueInputV1,
    technique_candidate=mma.MmaTechniqueCandidateV1,
    taxonomy=MMA.taxonomy,
    extra_prompt_keys=MMA.extra_prompt_keys,
)

# MMA_STRIKES plus the ground re-scan; every other prompt is MMA_STRIKES's own.
MMA_GROUND = AnalysisPack(
    workflow_id=MMA_GROUND_WORKFLOW_ID,
    taxonomy_id="mma.v1",
    prompts_dir="prompts_mma_ground",
    outputs=MappingProxyType({**MMA_STRIKES.outputs, "ground_rescan": mma.GroundRescanV1}),
    position_critique_input=mma.MmaPositionCritiqueInputV1,
    technique_candidate=mma.MmaTechniqueCandidateV1,
    taxonomy=MMA.taxonomy,
    extra_prompt_keys=MMA.extra_prompt_keys,
    base_prompts_dir="prompts_mma_strikes",
)

PACKS = MappingProxyType(
    {pack.workflow_id: pack for pack in (BJJ, MMA, MMA_STRIKES, MMA_GROUND)}
)


def pack_for(run):
    """The admitted run's own pack; an unknown workflow or taxonomy mismatch fails."""
    pack = PACKS.get(run.spec.workflow_id)
    if pack is None:
        raise ValueError("no analysis pack for the admitted workflow")
    if run.spec.taxonomy_id != pack.taxonomy_id:
        raise ValueError("admitted taxonomy does not match its workflow")
    return pack
