import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest
from bjj_analysis_core.outputs import PositionScanV1
from bjj_analysis_core.requests import (
    CoreConfigV1,
    parse_response,
    prepare_call,
    prompt_bundle,
    prompt_bundle_sha256,
)
from conftest import candidate, decision, state


def test_installed_wheel_contains_prompts_and_no_host_imports(run, configuration):
    import bjj_analysis_core

    root = Path(bjj_analysis_core.__file__).parent
    assert "site-packages" in str(root)
    assert importlib.metadata.version("bjj-analysis-core") == "0.3.0"
    assert len(list((root / "prompts").glob("*.txt"))) == 6
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                n.startswith(
                    (
                        "service",
                        "fastapi",
                        "boto3",
                        "google",
                        "shared_lib.infrastructure",
                    )
                )
                for n in names
            )
    bundle = prompt_bundle()
    assert "back_controller" in bundle["positions"]["roles"]
    assert "submission_choke" in bundle["action_classes"]
    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    assert request.config.thinking == "high" and request.config.fps == 1
    assert request.prompt_bundle_sha256 == prompt_bundle_sha256()
    assert request.response_schema == PositionScanV1.model_json_schema()
    initial = json.loads(request.initial_prompt)
    assert [p["player_id"] for p in initial["ordered_players"]] == ["a", "b"]
    assert initial["source_time_unit"] == "absolute_milliseconds"
    assert (
        "reference_crop" not in request.initial_prompt
        and "bucket" not in request.initial_prompt
    )


def test_required_geometry_and_outcome_rules_are_packaged():
    prompts = prompt_bundle()["prompts"]
    assert "Back mount/rear mount" in prompts["position_scan"]
    assert "back_controller can be underneath" in prompts["position_scan"]
    assert (
        "unobservable" in prompts["position_scan"]
        and "interruption_note" in prompts["position_scan"]
    )
    assert "outcome_bounds" in prompts["technique_analyze"]
    assert "fixed tail" in prompts["technique_analyze"]
    assert "unchanged position" in prompts["technique_analyze"]
    assert "Do not invent a reason to disagree" in prompts["technique_validate"]


@pytest.mark.parametrize(
    "failure",
    [
        "changed_bytes",
        "old_bundle",
        "unknown_node",
        "partial_scope",
        "outside_scope",
        "missing_context",
    ],
)
def test_admitted_request_binding_rejects_drift(run, configuration, failure):
    node, context, window = "position_scan", {}, run.spec.scope
    if failure == "changed_bytes":
        configuration += b" "
    elif failure == "old_bundle":
        data = json.loads(configuration)
        data["prompt_bundle_sha256"] = "0" * 64
        configuration = json.dumps(data).encode()
        run = run.model_copy(
            update={
                "spec": run.spec.model_copy(
                    update={
                        "effective_config": run.spec.effective_config.model_copy(
                            update={"sha256": hashlib.sha256(configuration).hexdigest()}
                        )
                    }
                )
            }
        )
    elif failure == "unknown_node":
        node = "legacy_highlight"
    elif failure == "partial_scope":
        window = dict(start_ms=2000, end_ms=9000)
    elif failure == "outside_scope":
        window = dict(start_ms=0, end_ms=11000)
    elif failure == "missing_context":
        node = "position_critique"
    with pytest.raises(ValueError):
        prepare_call(run, configuration, node, input_window=window, context=context)


@pytest.mark.parametrize(
    "text",
    [
        '{"initial_state":null,"initial_state":{}}',
        '{"transitions":NaN}',
        '{"transitions":Infinity}',
        "```json\n{}\n```",
        "{}",
    ],
)
def test_no_silent_json_repair(run, configuration, text):
    request = prepare_call(
        run, configuration, "position_scan", input_window=run.spec.scope, context={}
    )
    with pytest.raises(ValueError):
        parse_response(request, text)


def test_parse_and_bind_position_critique(run, configuration):
    context = dict(
        transition_index=2,
        proposed=dict(at_ms=4000, state_after=state()),
        state_before=state(),
    )
    request = prepare_call(
        run,
        configuration,
        "position_critique",
        input_window=dict(start_ms=3000, end_ms=5000),
        context=context,
    )
    data = dict(
        transition_index=2,
        confirmed=True,
        corrected_at_ms=4500,
        state_after=state(),
        reason="visible change",
    )
    assert parse_response(request, json.dumps(data)).corrected_at_ms == 4500
    for change in ({"transition_index": 3}, {"corrected_at_ms": 6000}):
        with pytest.raises(ValueError):
            parse_response(request, json.dumps({**data, **change}))


def test_validator_context_and_returned_candidate_bind(run, configuration):
    request = prepare_call(
        run,
        configuration,
        "technique_validate",
        input_window=run.spec.scope,
        context=candidate(),
    )
    assert parse_response(request, json.dumps(decision())).candidate_id == "scan-1"
    with pytest.raises(ValueError):
        parse_response(request, json.dumps(decision(candidate_id="different")))
    assert all(
        getattr(CoreConfigV1.model_validate_json(configuration), node).thinking
        == "high"
        for node in prompt_bundle()["prompts"]
    )
