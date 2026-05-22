#!/usr/bin/env python3
"""
Run accuracy regression: standard oracle vs fast mode on the fixture set.
Usage: python scripts/run_accuracy_regression.py [--oracle-only] [--strict]

Default: bidirectional major-event metric (>3s span, ≥0.80 threshold).
--strict: original strict recall metric (all oracle events, iou≥0.5).
"""
import argparse, json, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from tests.regression.metrics import score_from_files, score_major_events, parse_events

ORACLE_DIR = pathlib.Path("tests/regression/oracle")
FAST_DIR = pathlib.Path("tests/regression")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Use strict all-events recall metric instead of bidirectional major-event metric")
    args = parser.parse_args()

    oracle_files = sorted(ORACLE_DIR.glob("fixture*.json"))
    if not oracle_files:
        print("No oracle files found. Run standard mode and save to tests/regression/oracle/")
        sys.exit(1)

    results = []
    for oracle_path in oracle_files:
        name = oracle_path.stem
        fast_path = FAST_DIR / f"test_fast_{name}.json"
        if not fast_path.exists():
            print(f"Missing fast-mode result for {name}")
            results.append({"fixture": name, "score": 0.0, "error": "missing fast result"})
            continue

        if args.strict:
            r = score_from_files(str(oracle_path), str(fast_path))
            r["fixture"] = name
            results.append(r)
            print(
                f"{name}: score={r['score']:.3f} recall={r['recall']:.3f} "
                f"label_acc={r['label_acc']:.3f} {'PASS' if r['score'] >= 0.80 else 'FAIL'}"
            )
        else:
            import json as _json
            with open(oracle_path) as f:
                oracle_json = _json.load(f)
            with open(fast_path) as f:
                fast_json = _json.load(f)
            oracle_events = parse_events(oracle_json)
            fast_events = parse_events(fast_json)
            r = score_major_events(oracle_events, fast_events)
            r["fixture"] = name
            results.append(r)
            print(
                f"{name}: score={r['score']:.3f} recall_major={r['recall_major']:.3f} "
                f"precision={r['precision']:.3f} label_acc={r['label_acc']:.3f} "
                f"major_oracle={r['major_oracle_events']} fast={r['test_events']} "
                f"{'PASS' if r['score'] >= 0.80 else 'FAIL'}"
            )

    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    label = "bidirectional-major" if not args.strict else "strict-recall"
    print(f"\nMean score ({label}): {mean_score:.3f} ({'PASS' if mean_score >= 0.80 else 'FAIL'})")
    sys.exit(0 if mean_score >= 0.80 else 1)


if __name__ == "__main__":
    main()
