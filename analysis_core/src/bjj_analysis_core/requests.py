"""Auditable provider-neutral requests bound to the admitted config bytes.

The host resolves/checks private source/crop bytes and supplies the Gemini adapter.
This module performs no network I/O and imports no lab, service or database code.
"""

import hashlib
import json
from functools import cache
from typing import Literal

from bjj_analysis_core import CORE_VERSION
from bjj_analysis_core.outputs import (
    CandidateConfigV1,
    PositionCritiqueV1,
    PositionScanV1,
    TechniqueAnalysisInputV1,
    TechniqueAnalysisV1,
    TechniqueDecisionV1,
    TechniqueScanV1,
    TemporalCritiqueInputV1,
    TemporalCritiqueV1,
)
from bjj_analysis_core.packs import BJJ, PACKS, pack_for
from pydantic import Field, StrictStr
from shared_lib.models.analysis_contracts import (
    AnalysisContract,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    TimeRangeMs,
    VersionedAnalysisContract,
)
from shared_lib.models.analysis_results import AnalysisRevisionV1
from shared_lib.models.analysis_run_record import RegisteredAIRunV1

NodeId = Literal[
    "position_scan",
    "position_critique",
    "technique_scan",
    "technique_critique",
    "technique_analyze",
    "technique_validate",
]
# The BJJ pack's node outputs (every pack has the same nodes).
OUTPUTS = BJJ.outputs


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def prompt_bundle(pack=BJJ):
    return {
        "core_version": CORE_VERSION,
        "prompts": {node: pack.prompt(node) for node in pack.outputs},
        **{key: value for key, value in pack.taxonomy.items()},
        "schemas": {node: model.model_json_schema() for node, model in pack.outputs.items()},
    }


def prompt_bundle_sha256(pack=BJJ):
    return hashlib.sha256(canonical_json(prompt_bundle(pack)).encode("utf-8")).hexdigest()


@cache
def _pack_by_bundle():
    return {prompt_bundle_sha256(pack): pack for pack in PACKS.values()}


def pack_for_bundle(bundle_sha256):
    """A prepared call names its pack by the prompt bundle it was built from."""
    try:
        return _pack_by_bundle()[bundle_sha256]
    except KeyError:
        raise ValueError("prepared call uses an unknown prompt bundle") from None


class NodeConfigV1(AnalysisContract):
    model: Identifier
    thinking: Literal["minimal", "low", "medium", "high"] = "high"
    media_resolution: Literal["low", "medium", "high"]
    fps: PositiveInt = Field(le=30)


class CoreConfigV1(VersionedAnalysisContract):
    core_version: Literal[CORE_VERSION] = CORE_VERSION
    prompt_bundle_sha256: Sha256
    position_scan: NodeConfigV1
    position_critique: NodeConfigV1
    technique_scan: NodeConfigV1
    technique_critique: NodeConfigV1
    technique_analyze: NodeConfigV1
    technique_validate: NodeConfigV1
    candidates: CandidateConfigV1
    # Full executor bounds are versioned with the effective run configuration.
    position_critique_back_ms: NonNegativeInt = Field(default=4000, le=30000)
    position_critique_forward_ms: PositiveInt = Field(default=6000, le=30000)
    max_call_attempts: PositiveInt = Field(default=2, le=3)


class PreparedCallV1(VersionedAnalysisContract):
    node_id: NodeId
    config: NodeConfigV1
    config_sha256: Sha256
    prompt_bundle_sha256: Sha256
    input_window: TimeRangeMs
    system_prompt: StrictStr = Field(min_length=1)
    initial_prompt: StrictStr = Field(min_length=1)
    response_schema: dict


def load_config(run, config_bytes):
    """Bind provider and deterministic stages to the same admitted artifact."""
    run = RegisteredAIRunV1.model_validate(run)
    if type(config_bytes) is not bytes:
        raise ValueError("exact configuration artifact bytes required")
    digest = hashlib.sha256(config_bytes).hexdigest()
    if digest != run.spec.effective_config.sha256:
        raise ValueError("configuration bytes differ from admitted run")
    config = CoreConfigV1.model_validate(strict_json(config_bytes))
    if run.spec.core_version != CORE_VERSION or run.spec.workflow_id not in PACKS:
        raise ValueError("core or prompt bundle differs from admitted configuration")
    pack = pack_for(run)
    if config.prompt_bundle_sha256 != prompt_bundle_sha256(pack):
        raise ValueError("core or prompt bundle differs from admitted configuration")
    return run, config


def prepare_call(
    run, config_bytes, node_id, *, input_window, context, advisory_positions=None
):
    """Use persisted stage input, not raw browser configuration or guessed models."""
    run, config = load_config(run, config_bytes)
    pack = pack_for(run)
    if node_id not in pack.outputs:
        raise ValueError("unknown core node")
    bundle = prompt_bundle(pack)
    window = TimeRangeMs.model_validate(input_window)
    if not run.spec.scope.contains(window):
        raise ValueError("model observation window exceeds admitted scope")
    if node_id in ("position_scan", "technique_scan") and window != run.spec.scope:
        raise ValueError("scan must cover the entire selected scope")
    if node_id in ("position_scan", "technique_scan"):
        if context != {} or advisory_positions is not None:
            raise ValueError(
                "scan is visually independent and has no previous-state context"
            )
    else:
        context_model = {
            "position_critique": pack.position_critique_input,
            "technique_critique": TemporalCritiqueInputV1,
            "technique_analyze": TechniqueAnalysisInputV1,
            "technique_validate": pack.technique_candidate,
        }[node_id]
        parsed = context_model.model_validate(context)
        if node_id == "technique_critique":
            from bjj_analysis_core.discovery import critique_window

            if window != critique_window(run, config_bytes, parsed.candidate):
                raise ValueError(
                    "critique must use its admitted padded inspection window"
                )
        if node_id == "technique_analyze" and window != parsed.bounds:
            raise ValueError("analysis must inspect the corrected temporal bounds")
        if (
            node_id == "position_critique"
            and not window.start_ms <= parsed.proposed.at_ms < window.end_ms
        ):
            raise ValueError(
                "proposed transition is outside critique observation window"
            )
        if node_id == "technique_validate" and (
            parsed.observation_window != window
            or not window.contains(parsed.proposed.bounds)
        ):
            raise ValueError(
                "candidate/proposal does not match validator observation window"
            )
        context = parsed.model_dump(mode="json")
    advisory = None
    if advisory_positions is not None:
        if node_id not in (
            "technique_critique",
            "technique_analyze",
            "technique_validate",
        ):
            raise ValueError("only technique nodes take advisory position context")
        from bjj_analysis_core.discovery import position_context

        context_bounds = window
        if node_id == "technique_critique":
            context_bounds = parsed.candidate.bounds
        elif node_id == "technique_validate":
            context_bounds = parsed.proposed.bounds
        advisory = position_context(run, advisory_positions, context_bounds)
    initial = {
        "source_time_unit": "absolute_milliseconds",
        "scope": run.spec.scope.model_dump(),
        "observation_window": window.model_dump(),
        "ordered_players": [
            {"player_id": a.player_id, "display_name": a.display_name}
            for a in run.spec.athletes
        ],
        "position_taxonomy": bundle["positions"],
        "action_classes": bundle["action_classes"],
        "outcomes": bundle["outcomes"],
        "technique_classes": bundle["technique_classes"],
        **{key: bundle[key] for key in pack.extra_prompt_keys},
        "stage_context": context,
        "advisory_positions": advisory,
    }
    return PreparedCallV1(
        node_id=node_id,
        config=getattr(config, node_id),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        prompt_bundle_sha256=config.prompt_bundle_sha256,
        input_window=window,
        system_prompt=bundle["prompts"][node_id],
        initial_prompt=canonical_json(initial),
        response_schema=bundle["schemas"][node_id],
    )


def parse_response(prepared, response_text):
    """Reject malformed JSON/schema/window violations; no repairs or code fences."""
    prepared = PreparedCallV1.model_validate(prepared)
    pack = pack_for_bundle(prepared.prompt_bundle_sha256)
    result = pack.outputs[prepared.node_id].model_validate(strict_json(response_text))
    window = prepared.input_window
    context = strict_json(prepared.initial_prompt)["stage_context"]
    if isinstance(result, PositionScanV1):
        last = window.start_ms
        for boundary in result.transitions:
            if not last < boundary.at_ms < window.end_ms:
                raise ValueError(
                    "scan boundaries must increase inside the observation window"
                )
            last = boundary.at_ms
    elif isinstance(result, PositionCritiqueV1):
        if result.transition_index != context["transition_index"]:
            raise ValueError("critique addresses a different transition")
        if (
            result.confirmed
            and not window.start_ms <= result.corrected_at_ms < window.end_ms
        ):
            raise ValueError("corrected boundary exceeds observation window")
    elif isinstance(result, TechniqueScanV1):
        last_start = window.start_ms
        for event in result.events:
            if not window.contains(event.bounds) or event.bounds.start_ms < last_start:
                raise ValueError(
                    "scan events must be ordered inside observation window"
                )
            last_start = event.bounds.start_ms
    elif isinstance(result, TemporalCritiqueV1):
        if result.candidate_id != context["candidate"]["candidate_id"]:
            raise ValueError("critique addresses a different candidate")
        if result.bounds is not None and not window.contains(result.bounds):
            raise ValueError("temporal correction exceeds observation window")
        if (
            result.end_status == "window_exhausted"
            and result.bounds.end_ms != window.end_ms
        ):
            raise ValueError(
                "exhausted outcome must retain evidence through window end"
            )
    elif isinstance(result, TechniqueAnalysisV1):
        if result.bounds != window:
            raise ValueError("analysis must preserve the authoritative temporal bounds")
    elif isinstance(result, TechniqueDecisionV1):
        if result.candidate_id != context["candidate_id"]:
            raise ValueError("validator addresses a different candidate")
        if result.technique is not None and not window.contains(
            result.technique.bounds
        ):
            raise ValueError("validated technique exceeds observation window")
    return result
