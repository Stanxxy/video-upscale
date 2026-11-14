"""MMA provider-facing fields: the BJJ shapes with mma.v1 vocabulary.

Each model narrows or extends its BJJ parent from ``outputs`` so the decision
rules (critique completeness, ditch/agree) are shared. The parse step enforces
the same strike/cage/interruption rules that publication (shared_lib) enforces,
so a malformed provider answer is retried per call instead of failing publish.
"""

from enum import Enum

from pydantic import StrictBool, model_validator
from shared_lib.models.sport_taxonomy import vocabulary

from bjj_analysis_core.outputs import (
    PositionBoundaryV1,
    PositionCritiqueInputV1,
    PositionCritiqueV1,
    PositionScanV1,
    PositionStateV1,
    TechniqueAnalysisV1,
    TechniqueCandidateV1,
    TechniqueDecisionV1,
)

MMA = vocabulary("mma.v1")


def _enum(name, values):
    return Enum(name, {v: v for v in values}, type=str)


MmaPositionCode = _enum("MmaPositionCode", MMA.positions)
MmaPositionRole = _enum("MmaPositionRole", MMA.roles)
MmaTechniqueAction = _enum("MmaTechniqueAction", MMA.actions)
MmaStrikeTarget = _enum("MmaStrikeTarget", MMA.strike_targets)
MmaInterruptionKind = _enum("MmaInterruptionKind", MMA.interruption_kinds)


class MmaPositionStateV1(PositionStateV1):
    position: MmaPositionCode | None
    roles: tuple[MmaPositionRole, MmaPositionRole]
    interruption_kind: MmaInterruptionKind | None
    cage_contact: StrictBool | None

    @model_validator(mode="after")
    def mma_state_fields(self):
        if (self.observation == "inactive") != (self.interruption_kind is not None):
            raise ValueError("interruption_kind is required exactly for inactive states")
        if self.cage_contact is not None and self.observation != "observed":
            raise ValueError("cage_contact applies only to observed states")
        return self


class MmaPositionBoundaryV1(PositionBoundaryV1):
    state_after: MmaPositionStateV1


class MmaPositionScanV1(PositionScanV1):
    initial_state: MmaPositionStateV1
    transitions: tuple[MmaPositionBoundaryV1, ...]


class MmaPositionCritiqueV1(PositionCritiqueV1):
    state_after: MmaPositionStateV1 | None


class MmaPositionCritiqueInputV1(PositionCritiqueInputV1):
    proposed: MmaPositionBoundaryV1
    state_before: MmaPositionStateV1


class MmaTechniqueAnalysisV1(TechniqueAnalysisV1):
    action_classes: tuple[MmaTechniqueAction, ...]
    strike_target: MmaStrikeTarget | None
    significant: StrictBool | None
    knockdown: StrictBool

    @model_validator(mode="after")
    def strike_fields(self):
        if any(MMA.is_strike(action.value) for action in self.action_classes):
            if self.strike_target is None or self.significant is None:
                raise ValueError("a strike needs a target and a significance flag")
        elif self.strike_target is not None or self.significant is not None or self.knockdown:
            raise ValueError("strike fields apply only to strike action classes")
        return self


class MmaTechniqueDecisionV1(TechniqueDecisionV1):
    technique: MmaTechniqueAnalysisV1 | None


class MmaTechniqueCandidateV1(TechniqueCandidateV1):
    proposed: MmaTechniqueAnalysisV1


# --- Windowed strike census (workflow mma-windowed-strikes) ---------------------------
from typing import Literal  # noqa: E402

from pydantic import StrictStr  # noqa: E402
from shared_lib.models.analysis_contracts import (  # noqa: E402
    AnalysisContract,
    Identifier,
    NonNegativeInt,
    TimeRangeMs,
    VersionedAnalysisContract,
)

MmaStrikeAction = _enum("MmaStrikeAction", MMA.strike_actions)


class StrikeV1(AnalysisContract):
    """One thrown strike at its contact (or full-extension) moment."""

    at_ms: NonNegativeInt
    action: MmaStrikeAction
    actor_player_id: Identifier | None
    actor_sentinel: Literal["contested", "unclear"] | None
    strike_target: MmaStrikeTarget
    outcome: Literal["successful", "unsuccessful", "unclear"]
    significant: StrictBool
    knockdown: StrictBool
    evidence: Identifier

    @model_validator(mode="after")
    def one_actor(self):
        if (self.actor_player_id is None) == (self.actor_sentinel is None):
            raise ValueError("a strike names its fighter or a sentinel, not both")
        return self


class StrikeCensusV1(VersionedAnalysisContract):
    window_index: NonNegativeInt
    strikes: tuple[StrikeV1, ...]


class StrikeAuditDecisionV1(AnalysisContract):
    census_index: NonNegativeInt
    verdict: Literal["keep", "drop", "retime"]
    at_ms: NonNegativeInt | None
    reason: Identifier

    @model_validator(mode="after")
    def retime_has_time(self):
        # keep may echo the census time (checked against it at parse); drop ignores it.
        if self.verdict == "retime" and self.at_ms is None:
            raise ValueError("a retime carries its corrected time")
        return self


class StrikeAuditV1(VersionedAnalysisContract):
    window_index: NonNegativeInt
    decisions: tuple[StrikeAuditDecisionV1, ...]
    added: tuple[StrikeV1, ...]


class StrikeWindowInputV1(AnalysisContract):
    window_index: NonNegativeInt
    window: TimeRangeMs


class StrikeAuditInputV1(AnalysisContract):
    window_index: NonNegativeInt
    census: StrikeCensusV1


class StrikeBatchV1(VersionedAnalysisContract):
    """Deterministic merge result: final strikes as MMA technique analyses."""

    strikes: tuple[MmaTechniqueAnalysisV1, ...]
    census_count: NonNegativeInt
    dropped_by_audit: NonNegativeInt
    added_by_audit: NonNegativeInt
    rejected_not_live: NonNegativeInt
    merged_duplicates: NonNegativeInt
    notes: StrictStr = ""
