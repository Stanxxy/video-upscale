"""Provider-facing fields, without model-authored ownership, IDs or acceptance.

Every time is an absolute source-video integer millisecond. No conversion from
legacy geometric top/bottom roles or unstamped outcome guesses is implied.
"""

from typing import Literal

from pydantic import Field, StrictBool, StrictStr, model_validator
from shared_lib.models.analysis_contracts import (
    AnalysisContract,
    Identifier,
    NonNegativeInt,
    TimeRangeMs,
    VersionedAnalysisContract,
)
from shared_lib.models.analysis_results import (
    TechniqueAction,
    TechniqueActorSentinel,
    TechniqueOutcome,
)
from shared_lib.models.position_taxonomy import PositionCode, PositionRole


class PositionStateV1(AnalysisContract):
    observation: Literal["observed", "unobservable", "inactive"]
    position: PositionCode | None
    roles: tuple[PositionRole, PositionRole]
    notes: StrictStr
    evidence: Identifier
    uncertainty_note: Identifier | None
    interruption_note: Identifier | None


class PositionBoundaryV1(AnalysisContract):
    at_ms: NonNegativeInt
    state_after: PositionStateV1


class PositionScanV1(VersionedAnalysisContract):
    initial_state: PositionStateV1
    transitions: tuple[PositionBoundaryV1, ...]


class PositionCritiqueV1(VersionedAnalysisContract):
    transition_index: NonNegativeInt
    confirmed: StrictBool
    corrected_at_ms: NonNegativeInt | None
    state_after: PositionStateV1 | None
    reason: Identifier

    @model_validator(mode="after")
    def complete_decision(self):
        if self.confirmed:
            if self.corrected_at_ms is None or self.state_after is None:
                raise ValueError("confirmed transition needs corrected time and state")
        elif self.corrected_at_ms is not None or self.state_after is not None:
            raise ValueError("rejected transition must have null corrected fields")
        return self


class TechniqueAnalysisV1(VersionedAnalysisContract):
    bounds: TimeRangeMs
    action_classes: tuple[TechniqueAction, ...]
    actor_player_id: Identifier | None
    actor_sentinel: TechniqueActorSentinel | None
    outcome: TechniqueOutcome
    technique_shortlist: Identifier | None
    technique_guess: Identifier | None
    notes: StrictStr
    evidence: Identifier
    outcome_evidence: Identifier
    outcome_bounds: TimeRangeMs | None


class TechniqueDecisionV1(VersionedAnalysisContract):
    candidate_id: Identifier
    verdict: Literal["agree", "correct", "ditch"]
    intentional_technique_visible: StrictBool
    technique: TechniqueAnalysisV1 | None
    reason: Identifier

    @model_validator(mode="after")
    def real_attempt_survives(self):
        if self.verdict == "ditch":
            if self.intentional_technique_visible or self.technique is not None:
                raise ValueError("ditch only a non-event, with null technique")
        elif not self.intentional_technique_visible or self.technique is None:
            raise ValueError("real attempts require a complete surviving technique")
        return self


class TechniqueCandidateV1(AnalysisContract):
    """Persisted host input to validator; agreement must preserve this proposal."""

    candidate_id: Identifier
    description: Identifier
    observation_window: TimeRangeMs
    proposed: TechniqueAnalysisV1


class PositionCritiqueInputV1(AnalysisContract):
    transition_index: NonNegativeInt
    proposed: PositionBoundaryV1
    state_before: PositionStateV1


class TechniqueAnalysisInputV1(AnalysisContract):
    candidate_id: Identifier
    description: Identifier
    bounds: TimeRangeMs
    end_status: Literal["outcome_visible", "window_exhausted"]
    temporal_reason: Identifier


class ScannedTechniqueV1(AnalysisContract):
    bounds: TimeRangeMs
    description: Identifier
    reasoning: Identifier


class TechniqueScanV1(VersionedAnalysisContract):
    events: tuple[ScannedTechniqueV1, ...]


class CandidateConfigV1(AnalysisContract):
    transition_pre_ms: NonNegativeInt = Field(default=4000, le=30000)
    transition_post_ms: NonNegativeInt = Field(default=2000, le=30000)
    merge_tolerance_ms: NonNegativeInt = Field(default=2000, le=15000)
    critique_back_ms: NonNegativeInt = Field(default=6000, le=30000)
    critique_forward_ms: NonNegativeInt = Field(default=4000, le=30000)

    @model_validator(mode="after")
    def positive_transition_window(self):
        if self.transition_pre_ms + self.transition_post_ms == 0:
            raise ValueError("transition window must have positive duration")
        return self


class DiscoveryCandidateV1(AnalysisContract):
    """Host-owned target, before classification; not a publishable annotation."""

    candidate_id: Identifier
    bounds: TimeRangeMs
    description: Identifier
    reasoning: Identifier
    scan_index: NonNegativeInt | None
    transition_at_ms: NonNegativeInt | None
    sources: tuple[Literal["technique_scan", "position_transition"], ...]

    @model_validator(mode="after")
    def exact_provenance(self):
        expected = []
        if self.scan_index is not None:
            expected.append("technique_scan")
        if self.transition_at_ms is not None:
            expected.append("position_transition")
        if not expected or tuple(expected) != self.sources:
            raise ValueError("candidate sources must match its exact provenance")
        return self


class CandidateBatchV1(VersionedAnalysisContract):
    candidates: tuple[DiscoveryCandidateV1, ...]
    scan_count: NonNegativeInt
    transition_count: NonNegativeInt
    matched_count: NonNegativeInt
    provider_call_count: Literal[0] = 0


class TemporalCritiqueInputV1(AnalysisContract):
    candidate: DiscoveryCandidateV1


class TemporalCritiqueV1(VersionedAnalysisContract):
    candidate_id: Identifier
    confirmed: StrictBool
    bounds: TimeRangeMs | None
    end_status: Literal["outcome_visible", "window_exhausted"] | None
    reason: Identifier

    @model_validator(mode="after")
    def explicit_decision(self):
        if self.confirmed:
            if self.bounds is None or self.end_status is None:
                raise ValueError(
                    "confirmed attempt needs corrected bounds and end status"
                )
        elif self.bounds is not None or self.end_status is not None:
            raise ValueError("rejected non-event must have null timing fields")
        return self
