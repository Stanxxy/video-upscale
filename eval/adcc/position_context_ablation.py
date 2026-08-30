"""Bounded paired test of advisory position context for technique classification.

This is intentionally narrower than the Position Timeline DAG. The checked-in
ADCC corpus supplies fixed nine-second windows, a coarse position bucket, and
an action/outcome label, but no continuous position truth, athlete-reference
images, or exact technique spans. The result is therefore only an oracle-style
context upper bound for classification. It cannot validate position discovery,
candidate recall, actor grounding, or VTG boundary accuracy.

Dry-run by default. A real run is capped at 24 calls and $3 estimated Standard
paid-tier spend unless an even lower cap is supplied.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

from google import genai
from google.genai import types

from service.config import ServiceConfig
from service.pipelines import gemini_cost, gemini_retry, simplified_tags
from service.pipelines.executors import _call_axis_json
from service.routes.qa_vlm import thinking_config_for


MODEL_ID = "gemini-3.7-flash"
THINKING = "high"
FPS = 10
MEDIA_RESOLUTION = None
MAX_EXPERIMENT_CALLS = 24
ABSOLUTE_CALL_CEILING = 60
ABSOLUTE_USD_CEILING = 3.0

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "s12_gates" / "gate1_subset_manifest.json"
OUTPUT_DIR = ROOT / "position_context_ablation"
RESULTS_PATH = OUTPUT_DIR / "results.gemini37flash_high.json"
REPORT_PATH = OUTPUT_DIR / "REPORT.md"

# Two windows from each of the six scoreable action classes. Membership is a
# balanced projection of the already-frozen Gate-1 manifest, selected only by
# class/index—not by any stored model prediction or result.
SELECTED_INDICES = (0, 1, 7, 8, 18, 30, 19, 20, 23, 24, 26, 27)
CONDITIONS = ("baseline", "position_context")

SYSTEM_PROMPT = """
Classify one known intentional Brazilian Jiu-Jitsu action in a short native
video window. Return exactly one action_class and one outcome for the target
action nearest the supplied time anchor. Surrounding setup, counters, or later
actions are context only. Visible body-movement evidence is authoritative.
""".strip()


def build_response_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        required=["action_class", "outcome", "specific_technique_guess", "confidence", "evidence"],
        properties={
            "action_class": types.Schema(
                type=types.Type.STRING, enum=list(simplified_tags.ACTION_CLASS_VALUES)
            ),
            "outcome": types.Schema(
                type=types.Type.STRING, enum=list(simplified_tags.OUTCOME_VALUES)
            ),
            "specific_technique_guess": types.Schema(type=types.Type.STRING),
            "confidence": types.Schema(type=types.Type.NUMBER, format="float"),
            "evidence": types.Schema(type=types.Type.STRING),
        },
    )


def load_selected_windows(manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    by_index = {int(window["index"]): window for window in manifest["windows"]}
    missing = [index for index in SELECTED_INDICES if index not in by_index]
    if missing:
        raise ValueError(f"frozen ADCC manifest is missing selected indexes: {missing}")
    return [dict(by_index[index]) for index in SELECTED_INDICES]


def planned_call_count(windows: Sequence[Mapping]) -> int:
    return len(windows) * len(CONDITIONS)


def validate_call_budget(
    windows: Sequence[Mapping], *, max_calls: int = MAX_EXPERIMENT_CALLS
) -> None:
    calls = planned_call_count(windows)
    if max_calls > ABSOLUTE_CALL_CEILING:
        raise ValueError(
            f"max_calls {max_calls} exceeds the approved {ABSOLUTE_CALL_CEILING}-call ceiling"
        )
    if calls > max_calls:
        raise ValueError(f"planned experiment needs {calls} calls; max_calls is {max_calls}")


def build_prompt(window: Mapping, *, include_position_context: bool) -> str:
    anchor_offset = round(
        float(window["gold_event"]["t_s"]) - float(window["window_start_s"]), 3
    )
    prompt = (
        "The provided clip is a fixed evaluation window. Classify the single intentional "
        f"BJJ action whose outcome anchor is nearest {anchor_offset:.3f} seconds after the "
        "clip begins. Do not report timestamps. Do not substitute a neighboring action."
    )
    if include_position_context:
        bucket = str(window["gold_event"]["position_bucket"])
        prompt += (
            "\n\nADVISORY POSITION CONTEXT (oracle-style corpus input, independently produced "
            f"and fallible): {bucket}. This label may be coarse or wrong. Use it only to "
            "corroborate the action; visible body-movement evidence is authoritative when "
            "the label conflicts with the clip."
        )
    else:
        prompt += "\n\nNo external position label is supplied; judge from the visible clip alone."
    return prompt


def _condition_correct(record: Mapping, condition: str, field: str) -> bool:
    result = record.get(condition) or {}
    prediction = result.get("prediction") or {}
    return result.get("status") == "ok" and prediction.get(field) == record["gold"].get(field)


def _two_sided_sign_test(context_only: int, baseline_only: int) -> float | None:
    discordant = context_only + baseline_only
    if discordant == 0:
        return None
    lower = min(context_only, baseline_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def summarize_records(records: Sequence[Mapping]) -> dict:
    paired = [
        record
        for record in records
        if all((record.get(condition) or {}).get("status") == "ok" for condition in CONDITIONS)
    ]
    failed_pair_count = len(records) - len(paired)
    summaries: dict[str, dict] = {}
    for condition in CONDITIONS:
        action_correct = sum(
            _condition_correct(record, condition, "action_class") for record in paired
        )
        outcome_correct = sum(
            _condition_correct(record, condition, "outcome") for record in paired
        )
        summaries[condition] = {
            "exact_action_correct": action_correct,
            "exact_action_accuracy": action_correct / len(paired) if paired else None,
            "exact_outcome_correct": outcome_correct,
            "exact_outcome_accuracy": outcome_correct / len(paired) if paired else None,
        }

    context_only = sum(
        _condition_correct(record, "position_context", "action_class")
        and not _condition_correct(record, "baseline", "action_class")
        for record in paired
    )
    baseline_only = sum(
        _condition_correct(record, "baseline", "action_class")
        and not _condition_correct(record, "position_context", "action_class")
        for record in paired
    )
    return {
        "window_count": len(records),
        "paired_success_count": len(paired),
        "failed_pair_count": failed_pair_count,
        **summaries,
        "exact_action_net_delta": (
            summaries["position_context"]["exact_action_correct"]
            - summaries["baseline"]["exact_action_correct"]
        ),
        "discordant": {
            "context_only_correct": context_only,
            "baseline_only_correct": baseline_only,
        },
        "paired_sign_test_p_value": _two_sided_sign_test(context_only, baseline_only),
    }


def _cost_rows(records: Sequence[Mapping]) -> tuple[list[dict], dict]:
    rows = []
    for condition in CONDITIONS:
        stage = SimpleNamespace(
            id=condition,
            type="position_context_ablation",
            label="Baseline" if condition == "baseline" else "Position context",
            config={"model": MODEL_ID},
        )
        accumulator = gemini_cost.new_node_accumulator(stage)
        for record in records:
            result = record.get(condition)
            if result is not None:
                gemini_cost.record_event_usage(
                    accumulator, {"usage_metadata": result.get("usage_metadata")}
                )
        rows.append(gemini_cost.node_cost_snapshot(accumulator))
    return rows, gemini_cost.summarize_node_costs(rows)


def _result_envelope(records: Sequence[Mapping]) -> dict:
    cost_rows, cost_summary = _cost_rows(records)
    return {
        "experiment": "adcc_oracle_position_context_ablation_v1",
        "settings": {
            "model": MODEL_ID,
            "thinking": THINKING,
            "fps": FPS,
            "media_resolution": MEDIA_RESOLUTION,
            "selected_manifest_indexes": list(SELECTED_INDICES),
            "planned_call_count": MAX_EXPERIMENT_CALLS,
        },
        "scope_limitations": [
            "coarse position_bucket is supplied as oracle-style context, not predicted by Position Scan",
            "fixed nine-second windows test classification only, not discovery recall or VTG boundaries",
            "ADCC artifact has no two-athlete reference crops, so actor identity is not scored",
            "single call per condition leaves ordinary model sampling variance unresolved",
        ],
        "records": list(records),
        "summary": summarize_records(records),
        "cost_rows": cost_rows,
        "cost_summary": cost_summary,
    }


def render_report(envelope: Mapping) -> str:
    summary = envelope["summary"]
    cost = envelope["cost_summary"]
    baseline = summary["baseline"]
    treatment = summary["position_context"]
    rows = []
    for record in envelope["records"]:
        gold = record["gold"]
        baseline_prediction = (record.get("baseline") or {}).get("prediction") or {}
        context_prediction = (record.get("position_context") or {}).get("prediction") or {}
        rows.append(
            "| {window} | {position} | {gold} | {baseline} | {context} |".format(
                window=record["window_id"],
                position=gold["position_bucket"],
                gold=gold["action_class"],
                baseline=baseline_prediction.get("action_class", record.get("baseline", {}).get("status", "missing")),
                context=context_prediction.get("action_class", record.get("position_context", {}).get("status", "missing")),
            )
        )
    estimated_usd = cost.get("estimated_usd")
    cost_text = "unavailable" if estimated_usd is None else f"${estimated_usd:.6f}"
    return "\n".join(
        [
            "# ADCC oracle-position context ablation",
            "",
            "This bounded paired experiment tests whether supplying a coarse, oracle-style ADCC position bucket helps Gemini 3.7 Flash classify a technique in an already-located fixed window. It does **not** test Position Scan, candidate recall, player identity, or VTG boundaries.",
            "",
            "## Result",
            "",
            f"- Paired successful windows: {summary['paired_success_count']}/{summary['window_count']} (failed pairs: {summary['failed_pair_count']})",
            f"- Baseline exact action: {baseline['exact_action_correct']}/{summary['paired_success_count']}",
            f"- Position-context exact action: {treatment['exact_action_correct']}/{summary['paired_success_count']}",
            f"- Net exact-action delta: {summary['exact_action_net_delta']:+d}",
            f"- Discordant pairs (context-only / baseline-only): {summary['discordant']['context_only_correct']} / {summary['discordant']['baseline_only_correct']}",
            f"- Two-sided paired sign-test p: {summary['paired_sign_test_p_value']}",
            f"- Measured calls: {cost['call_count']}; estimated Standard paid-tier spend: {cost_text}",
            "",
            "## Per-window classifications",
            "",
            "| Window | Supplied position | Gold action | Baseline | Position context |",
            "|---|---|---|---|---|",
            *rows,
            "",
            "## Interpretation boundary",
            "",
            *[f"- {item}" for item in envelope["scope_limitations"]],
            "",
        ]
    )


def write_artifacts(records: Sequence[Mapping], output_path: Path, report_path: Path) -> dict:
    envelope = _result_envelope(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(envelope, indent=2, sort_keys=False) + "\n")
    report_path.write_text(render_report(envelope))
    return envelope


def _load_existing_records(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    envelope = json.loads(output_path.read_text())
    settings = envelope.get("settings") or {}
    if settings.get("model") != MODEL_ID or settings.get("thinking") != THINKING:
        raise ValueError("existing ablation artifact uses different model/thinking settings")
    return {record["window_id"]: record for record in envelope.get("records", [])}


async def _run_condition(
    *, analyzer, window: Mapping, condition: str
) -> dict:
    include_context = condition == "position_context"
    data, error = await _call_axis_json(
        analyzer,
        window["video_url_watch"],
        float(window["window_start_s"]),
        float(window["window_end_s"]),
        response_schema=build_response_schema(),
        prompt_text=build_prompt(window, include_position_context=include_context),
        fps=FPS,
        media_resolution=MEDIA_RESOLUTION,
        ctx=None,
        model_id=MODEL_ID,
        system_instruction=SYSTEM_PROMPT,
        thinking=THINKING,
    )
    return {
        "status": "ok" if error is None else "call_failed",
        "error": error,
        "prediction": data if error is None else None,
        "prompt_text": getattr(analyzer, "last_prompt_text", None),
        "raw_response_text": getattr(analyzer, "last_raw_response_text", None),
        "usage_metadata": getattr(analyzer, "last_usage_metadata", None),
    }


async def run_experiment(args: argparse.Namespace) -> dict:
    windows = load_selected_windows()
    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]
    validate_call_budget(windows, max_calls=args.max_calls)
    if args.max_usd > ABSOLUTE_USD_CEILING:
        raise ValueError(
            f"max_usd {args.max_usd} exceeds the approved ${ABSOLUTE_USD_CEILING:.2f} ceiling"
        )

    existing = _load_existing_records(args.output)
    records_by_id = dict(existing)
    cfg = ServiceConfig()
    if not cfg.gemini_api_key:
        raise RuntimeError("Set GEMINI_API_KEY or BJJ_GEMINI_API_KEY before --execute")
    client = genai.Client(
        api_key=cfg.gemini_api_key,
        http_options=types.HttpOptions(timeout=cfg.gemini_request_timeout_ms),
    )
    retry_config = gemini_retry.GeminiRetryConfig.from_service_config(cfg)
    analyzers = {
        condition: simplified_tags.SimplifiedTagsTimeAnalyzer(
            client,
            model_id=MODEL_ID,
            thinking_config=thinking_config_for(MODEL_ID, THINKING),
            system_instruction=SYSTEM_PROMPT,
            retry_config=retry_config,
        )
        for condition in CONDITIONS
    }
    calls_this_run = 0

    for ordinal, window in enumerate(windows):
        window_id = window["window_id"]
        record = records_by_id.setdefault(
            window_id,
            {
                "window_id": window_id,
                "manifest_index": window["index"],
                "video_url_watch": window["video_url_watch"],
                "window_start_s": window["window_start_s"],
                "window_end_s": window["window_end_s"],
                "gold": dict(window["gold_event"]),
                "call_order": [],
            },
        )
        # Alternate order to balance any systematic first/second-call effect.
        order = CONDITIONS if ordinal % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            if (record.get(condition) or {}).get("status") == "ok":
                continue
            if calls_this_run >= args.max_calls:
                raise RuntimeError("runtime call cap reached before the paired experiment completed")
            record["call_order"].append(condition)
            record[condition] = await _run_condition(
                analyzer=analyzers[condition], window=window, condition=condition
            )
            calls_this_run += 1
            ordered_records = [records_by_id[w["window_id"]] for w in windows if w["window_id"] in records_by_id]
            envelope = write_artifacts(ordered_records, args.output, args.report)
            estimated = envelope["cost_summary"].get("estimated_usd")
            if estimated is None:
                raise RuntimeError("usage/cost metadata unavailable; stopping instead of hiding spend")
            if estimated > args.max_usd:
                raise RuntimeError(
                    f"measured estimated spend ${estimated:.6f} exceeded ${args.max_usd:.2f}; stopped"
                )

    ordered_records = [records_by_id[window["window_id"]] for window in windows]
    return write_artifacts(ordered_records, args.output, args.report)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Spend real Gemini quota")
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--max-calls", type=int, default=MAX_EXPERIMENT_CALLS)
    parser.add_argument("--max-usd", type=float, default=ABSOLUTE_USD_CEILING)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    windows = load_selected_windows()
    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]
    validate_call_budget(windows, max_calls=args.max_calls)
    if not args.execute:
        print(
            json.dumps(
                {
                    "execute": False,
                    "windows": len(windows),
                    "planned_calls": planned_call_count(windows),
                    "model": MODEL_ID,
                    "thinking": THINKING,
                    "max_usd": args.max_usd,
                    "limitations": _result_envelope([])["scope_limitations"],
                },
                indent=2,
            )
        )
        return
    envelope = asyncio.run(run_experiment(args))
    print(json.dumps(envelope["summary"], indent=2))
    print(f"Results: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
