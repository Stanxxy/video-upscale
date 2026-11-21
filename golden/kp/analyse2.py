"""Round 2 numbers for the report page, from the run outputs (Mac side; needs the founder labels).

Reads DATA/{idgt1,idgt2}/answers.jsonl, out/<seg>/identity_*.json, posecheck1/, runs/*.jsonl, student/*.json,
slide.json, silver_events.json, idcards/cards.json. Writes DATA/summary2.json.
Usage: analyse2.py DATA
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.gold import EXPECTED, NEGATIVE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1gold import evaluate  # noqa: E402
from idgt import score as id_score  # noqa: E402
from score import score  # noqa: E402

TUNE = ["gold", "cw3", "fa2", "g4a", "g4b"]
HELD = ["cw2b", "cw3b", "fa2b", "fa2c", "g4a2", "g4b2", "bzq", "s597", "tcd", "hqs", "nrk", "u9z"]
RUNS = [("slide_raw", "slide.json", "Hydrogen, sliding windows, high thinking (teacher)", 1.81),
        ("oracle_raw", "oracle.json", "Hydrogen, founder-timed windows, high thinking", None),
        ("lean_h_low_high", "oracle.json", "Hydrogen, low thinking", 0.92),
        ("lean_h_low_med", "oracle.json", "Hydrogen, low thinking, medium image size", 0.57),
        ("lean_f38_high_high", "oracle.json", "Gemini 3.8 Flash, high thinking", 2.06),
        ("lean_f38_low_med", "oracle.json", "Gemini 3.8 Flash, low thinking, medium image size", 0.24),
        ("lean_fl35_low_med", "oracle.json", "Gemini 3.5 Flash-Lite, low thinking, medium image size", 0.08),
        ("annot_fable", "oracle.json", "Fable agent annotator (dense frame sheets, blind)", None),
        ("annot_sonnet", "oracle.json", "Sonnet agent annotator (same sheets)", None),
        ("student_rtmoz_t0.8", "gold_all.json", "Pose-sequence student (trained on silver, other videos)", 0.0),
        ("o2_raw", "oracle.json", "Overlay re-test: frames only", None),
        ("o2_id", "oracle.json", "Overlay re-test: + identity boxes (round 2 tracking)", None),
        ("o2_pose", "oracle.json", "Overlay re-test: + skeletons (round 2 pose)", None)]


def classifier_rows(data):
    out = []
    for run, wins_name, label, cost_min in RUNS:
        wins = json.load(open(data / wins_name))
        gold_ids = {w["id"] for w in wins if w["seg"] == "gold"}
        rows = [json.loads(line) for line in open(data / "runs" / f"{run}.jsonl")]
        rows = [r for r in rows if r["window"] in gold_ids]
        reps = sorted({r.get("rep", 0) for r in rows})
        strict = [evaluate(wins, [r for r in rows if r.get("rep", 0) == k]) for k in reps]
        lenient = [score(wins, [r for r in rows if r.get("rep", 0) == k])[0] for k in reps]
        mean = lambda xs, k: round(float(np.mean([x.get(k, 0) for x in xs])), 2)
        out.append(dict(run=run, label=label, reps=len(reps), cost_per_min=cost_min,
                        kind_f1=mean(strict, "kind_f1"), kind_p=mean(strict, "kind_precision"), kind_r=mean(strict, "kind_recall"),
                        name_f1=mean(strict, "name_f1"), predicted=mean(strict, "predicted"),
                        fighter=mean(lenient, "right_fighter"), kind=mean(lenient, "right_kind"), name=mean(lenient, "name"),
                        blocks=mean(lenient, "defence_counted"), spend=round(sum(r.get("cost", 0) for r in rows), 2)))
    return out


def label_table(data):
    """Each founder label and what the teacher, the oracle-window Hydrogen runs and the annotators said there."""
    def preds(run, wins_name):
        wins = {w["id"]: w for w in json.load(open(data / wins_name))}
        out = []
        for line in open(data / "runs" / f"{run}.jsonl"):
            r = json.loads(line)
            if r["window"] not in wins or wins[r["window"]]["seg"] != "gold":
                continue
            t0 = 230000 + wins[r["window"]]["f0"] * 40
            out += [(t0 + a["time_s"] * 1000, a["fighter"].lower(), a["technique"], r.get("rep", 0)) for a in r["answer"]["attacks"]]
        return out
    src = dict(teacher=preds("slide_raw", "slide.json"), hydrogen=preds("oracle_raw", "oracle.json"),
               fable=preds("annot_fable", "oracle.json"), sonnet=preds("annot_sonnet", "oracle.json"))
    rows = []
    for t, a, n, o in EXPECTED:
        near = lambda P: sorted({p[2] for p in P if abs(p[0] - t) <= 700 and p[1] == a})
        hyd = sum(any(abs(p[0] - t) <= 700 and p[1] == a and p[2] == n for p in src["hydrogen"] if p[3] == k) for k in range(4))
        rows.append(dict(t=round((t - 230000) / 1000, 1), fighter=a, name=n, outcome=o, teacher=near(src["teacher"]),
                         hydrogen=near(src["hydrogen"]), hydrogen_right=hyd, fable=near(src["fable"]), sonnet=near(src["sonnet"])))
    return rows


def students(data):
    curve = defaultdict(list)
    for p in sorted(glob.glob(str(data / "student" / "rtmoz_*_*.json"))):
        d = json.load(open(p))
        curve[d["frac"]].append(d["sweep"]["0.5"])
    out = [dict(frac=f, runs=len(v), recall=round(float(np.mean([x["recall"] for x in v])), 3),
                precision=round(float(np.mean([x["precision"] for x in v])), 3),
                technique_acc=round(float(np.mean([x["technique_acc"] for x in v])), 3)) for f, v in sorted(curve.items())]
    light = data / "student" / "light_1.0_0.json"
    return dict(curve=out, light=json.load(open(light))["sweep"]["0.5"] if light.exists() else None,
                events=len(json.load(open(data / "silver_events.json"))))


def pose_check(data):
    key = json.load(open(data / "posecheck1" / "key.json"))
    rows = [json.loads(line) for f in ("judge_1", "judge_2") for line in open(data / "posecheck1" / f"{f}.jsonl")]
    wrong, right_person = defaultdict(list), defaultdict(int)
    for r in rows:
        for i, v in enumerate(key[r["sheet"]]):
            p = r["panels"][str(i + 1)]
            wrong[v].append(p["a_wrong"] + p["b_wrong"])
            right_person[v] += p["a_on_right_person"] + p["b_on_right_person"]
    n = len(rows)
    out = {v: dict(mean_wrong=round(float(np.mean(w)), 2), on_right_person=right_person[v], of=2 * n) for v, w in wrong.items()}
    for v in ("samask", "skmask"):
        out[v]["better"] = int(sum(a < b for a, b in zip(wrong[v], wrong["base"])))
        out[v]["worse"] = int(sum(a > b for a, b in zip(wrong[v], wrong["base"])))
    return dict(sheets=n, variants=out)


def main(data):
    data = Path(data)
    ident = dict(tuning={d: id_score(data, data / "idgt1", d, TUNE) for d in ("v1", "light", "pruned16", "repvit", "rtmo", "rtmoz")},
                 heldout={d: id_score(data, data / "idgt2", d, HELD) for d in ("light", "rtmoz")})
    spend = {Path(p).stem: round(sum(json.loads(line).get("cost", 0) for line in open(p)), 2)
             for p in glob.glob(str(data / "runs" / "*.jsonl")) if Path(p).stem.startswith(("lean_", "slide_", "o2_"))}
    S = dict(identity=ident, classifier=classifier_rows(data), labels=label_table(data), students=students(data),
             pose=pose_check(data), cards=json.load(open(data / "idcards" / "cards.json")), spend=spend,
             spend_total=round(sum(spend.values()), 2), negatives=len(NEGATIVE))
    json.dump(S, open(data / "summary2.json", "w"), indent=1)
    print(json.dumps(dict(spend_total=S["spend_total"], students=S["students"]["curve"], pose=S["pose"]["variants"])))


if __name__ == "__main__":
    main(sys.argv[1])
