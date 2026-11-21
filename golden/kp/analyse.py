"""Collect every experiment output into summary.json for the report, and export the fine-tune candidate lists.

Usage: analyse.py DATA   (DATA = temp_data/x-mmagold/kp; writes DATA/summary.json and DATA/datasets/*.jsonl)
"""
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from golden.gold import EXPECTED, NEGATIVE, kind  # noqa: E402
from golden.kp.score import score  # noqa: E402

SEGS = {"gold": ("CW R2 3:50-4:52 (founder minute)", "Dark brown shorts", "Light grey shorts", 230000, "XFBseoY4ER0"),
        "cw3": ("CW R3 8:00-9:00", "Dark brown shorts", "Light grey shorts", 480000, "XFBseoY4ER0"),
        "fa2": ("FA R2 9:00-10:00", "Orange shorts", "Black shorts", 540000, "1ACTKcuQZyQ"),
        "g4a": ("G4 elevated corner cam 4:00-5:00", "Red shin guards, grey shorts", "Blue shin guards, green shorts", 240000, "1kO4Of5KWoc"),
        "g4b": ("G4 overhead fisheye 2:00-3:00", "Red shin guards, blue shorts", "Blue shin guards, black tights", 120000, "g9A3tt74DZo")}
CONDS = ["video", "raw", "id", "pose", "trace"]
METRICS = ["right_fighter", "right_kind", "name", "specific_right", "wrong_fighter_only", "defence_counted", "predicted"]


def rows(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def sheet_frames(w):
    if w.get("peak_f") is not None:
        return [int(np.clip(w["peak_f"] + o, w["f0"], w["f1"])) for o in (-12, -6, -2, 0, 4, 8)]
    return [int(x) for x in np.linspace(w["f0"], w["f1"], 6)]


def ablation(D):
    wins = json.load(open(D / "oracle.json"))
    out = {}
    for c in CONDS:
        rs = rows(D / "runs" / f"oracle_{c}.jsonl")
        reps = sorted({r["rep"] for r in rs})
        per = [score(wins, [r for r in rs if r["rep"] == k]) for k in reps]
        conf = defaultdict(Counter)
        for _, cf in per:
            for g, d in cf.items():
                conf[g].update(d)
        out[c] = dict(reps=len(reps), cost=round(sum(r["cost"] for r in rs), 2),
                      latency_s=round(statistics.mean(r["latency_s"] for r in rs), 1),
                      metrics={m: dict(mean=round(statistics.mean(s.get(m, 0) for s, _ in per), 1),
                                       lo=min(s.get(m, 0) for s, _ in per), hi=max(s.get(m, 0) for s, _ in per))
                               for m in METRICS},
                      confusion={g: dict(v) for g, v in conf.items()})
    old = rows(D / "runs" / "oracle_video1fps.jsonl")
    if old:
        s, _ = score(wins, [r for r in old if r["rep"] == 0])
        out["video_1fps_note"] = s
    return out


def endtoend(D):
    wins = json.load(open(D / "sens_gold.json"))
    out = {}
    for c in ("raw", "trace"):
        rs = rows(D / "runs" / f"sens_gold_{c}.jsonl")
        if rs:
            s, cf = score(wins, rs)
            out[c] = dict(metrics=s, confusion=cf, cost=round(sum(r["cost"] for r in rs), 2), windows=len(wins))
    return out


def detection(D):
    C = json.load(open(D / "out/gold/candidates.json"))
    I = json.load(open(D / "out/gold/identity.json"))["frames"]
    strikes = [(t, a, n) for t, a, n, o in EXPECTED if kind(n) in ("punch", "kick", "knee")]
    sweep = []
    for thr in [(4, 3, 3), (5, 4, 4), (6, 5, 5), (7, 6, 6)]:
        T = dict(punch=thr[0], kick=thr[1], knee=thr[2])
        cs = [c for c in C if c["speed_tl_s"] >= T[c["group"]]]
        near = lambda t, c: abs(230000 + c["t_ms"] - t) <= 300
        sweep.append(dict(thresholds=thr, candidates=len(cs),
                          any=sum(any(near(t, c) for c in cs) for t, a, n in strikes),
                          right_fighter=sum(any(near(t, c) and c["fighter"] == a for c in cs) for t, a, n in strikes),
                          right_fighter_limb=sum(any(near(t, c) and c["fighter"] == a and c["group"] == kind(n) for c in cs)
                                                 for t, a, n in strikes)))
    misses = []
    for t, a, n in strikes:
        f = round((t - 230000) / 40)
        cov = sum(("a" in I[str(k)]) + ("b" in I[str(k)]) for k in range(f - 5, f + 6)) / 22
        flags = sorted({x for k in range(f - 5, f + 6) for x in I[str(k)]["flags"]})
        near = [c for c in C if abs(230000 + c["t_ms"] - t) <= 300]
        ok = any(c["fighter"] == a and c["group"] == kind(n) for c in near)
        if ok:
            why = "found"
        elif cov < 0.8 or "merged" in flags:
            why = "tracking: a fighter not tracked (merged mask or lost)"
        elif not near:
            why = "detector: no fast limb motion flagged"
        elif not any(c["fighter"] == a for c in near):
            why = "detector: only the opponent's limb flagged"
        else:
            why = "detector: right fighter, wrong limb"
        misses.append(dict(t=f"{(t // 1000) // 60}:{(t // 1000) % 60:02d}.{(t % 1000) // 100}", fighter=a.upper(),
                           name=n, coverage=round(cov, 2), flags=flags, result=why,
                           near=[f"{c['fighter'].upper()} {c['group']} {c['side']} {c['speed_tl_s']} TL/s" for c in near][:3]))
    return dict(strikes=len(strikes), sweep=sweep, per_strike=misses, reasons=dict(Counter(m["result"] for m in misses)))


def judges(D):
    J = {}
    for p in sorted((D / "judges").glob("*.jsonl")):
        for l in open(p):
            l = l.strip()
            if l:
                try:
                    v = json.loads(l)
                    J[v["window"]] = v
                except json.JSONDecodeError:
                    pass
    return J


def main(D):
    D = Path(D)
    stats = {s: json.load(open(D / "out" / s / "stats.json")) for s in SEGS if (D / "out" / s / "stats.json").exists()}
    J = judges(D)
    allwins = []
    for f in ["oracle.json"] + [f"sens_{s}.json" for s in SEGS]:
        if (D / f).exists():
            allwins += json.load(open(D / f))
    ans = defaultdict(dict)
    for p in (D / "runs").glob("*.jsonl"):
        if "1fps" in p.name:
            continue
        cond = "trace" if p.stem.endswith("trace") else "raw" if p.stem.endswith("raw") else None
        if not cond:
            continue
        for r in rows(p):
            if r["rep"] == 0:
                ans[r["window"]][cond] = r["answer"]["attacks"]
    cands = {s: {c["id"]: c for c in json.load(open(D / "out" / s / "candidates.json"))}
             for s in SEGS if (D / "out" / s / "candidates.json").exists()}
    idents = {s: json.load(open(D / "out" / s / "identity.json"))["frames"] for s in cands}
    cards = []
    for w in allwins:
        seg = w["seg"]
        if seg not in cands or not (D / "packs" / w["id"] / "sheet.jpg").exists():
            continue
        t0 = SEGS[seg][3] + w["f0"] * 40
        truth = []
        if seg == "gold":
            truth = [dict(t=round((t - t0) / 1000, 2), fighter=a.upper(), name=n) for t, a, n, o in EXPECTED
                     if t0 - 300 <= t <= t0 + (w["f1"] - w["f0"]) * 40 + 300]
            truth += [dict(t=round((t - t0) / 1000, 2), fighter=a.upper(), name=f"NOT a {k} (defence)") for t, a, k in NEGATIVE
                      if t0 <= t <= t0 + (w["f1"] - w["f0"]) * 40]
        fl = Counter(x.rsplit("_", 1)[0] if x != "merged" else x for f in range(w["f0"], w["f1"] + 1)
                     for x in idents[seg].get(str(f), {}).get("flags", []))
        cover = sum(("a" in idents[seg].get(str(f), {})) + ("b" in idents[seg].get(str(f), {}))
                    for f in range(w["f0"], w["f1"] + 1)) / (2 * (w["f1"] - w["f0"] + 1))
        mine = [cands[seg][i] for i in w.get("cands", [])]
        cards.append(dict(window=w["id"], seg=seg, kind="oracle" if w["id"].startswith("gold-o") else "sensing",
                          start=f"{(t0 // 1000) // 60}:{(t0 // 1000) % 60:02d}", video=SEGS[seg][4], t0_s=t0 / 1000,
                          length_s=round((w["f1"] - w["f0"]) / 25, 1), identity_flags=dict(fl), coverage=round(cover, 2),
                          candidates=[dict(fighter=c["fighter"].upper(), limb=f"{c['side']} {c['group']}", lead_rear=c["lead_rear"],
                                           speed=c["speed_tl_s"], curvature=c["curvature"], contact=c["contact"],
                                           t=round((c["peak_f"] - w["f0"]) / 25, 2)) for c in mine],
                          gemini=ans.get(w["id"], {}), judge=J.get(w["id"]), truth=truth))
    # judge aggregates per segment
    agg = {}
    for seg in SEGS:
        js = [c["judge"] for c in cards if c["seg"] == seg and c["judge"]]
        if not js:
            continue
        g = lambda path: Counter(str(_get(j, path)) for j in js)
        errs = Counter(e for j in js for e in (j.get("gemini_check") or {}).get("errors", []))
        joints = Counter(f"{b.get('joint')}: {b.get('problem')}" for j in js for b in (j.get("skeleton") or {}).get("bad_joints", []))
        agg[seg] = dict(n=len(js), camera_cut=g("camera_cut"), a_correct=g("identity.a_correct"),
                        b_correct=g("identity.b_correct"), skel_A=g("skeleton.A"), skel_B=g("skeleton.B"),
                        trail=g("trail.focus_follows_limb"), gemini_overlays=g("gemini_check.verdict"),
                        gemini_frames=g("gemini_frames_only_verdict"), gemini_errors=dict(errs),
                        bad_joints=dict(joints.most_common(12)))
    # fine-tune candidate exports
    ds = D / "datasets"
    ds.mkdir(exist_ok=True)
    counts = Counter()
    with open(ds / "pose_corrections.jsonl", "w") as fp, open(ds / "identity_swaps.jsonl", "w") as fi, \
            open(ds / "classifier_disputes.jsonl", "w") as fc, open(ds / "trail_failures.jsonl", "w") as ft:
        for w in allwins:
            j = J.get(w["id"])
            if not j:
                continue
            frames = sheet_frames(w)
            for b in (j.get("skeleton") or {}).get("bad_joints", []):
                k = b.get("frame")
                if isinstance(k, int) and 0 <= k < 6:
                    fp.write(json.dumps(dict(seg=w["seg"], frame=frames[k], fighter=b.get("fighter"), joint=b.get("joint"),
                                             problem=b.get("problem"), window=w["id"])) + "\n"); counts["pose"] += 1
            idn = j.get("identity") or {}
            for k in idn.get("swap_frames") or []:
                if isinstance(k, int) and 0 <= k < 6:
                    fi.write(json.dumps(dict(seg=w["seg"], frame=frames[k], window=w["id"], kind="swap")) + "\n"); counts["identity"] += 1
            for k in idn.get("merged_frames") or []:
                if isinstance(k, int) and 0 <= k < 6:
                    fi.write(json.dumps(dict(seg=w["seg"], frame=frames[k], window=w["id"], kind="merged")) + "\n"); counts["merged"] += 1
            if (j.get("trail") or {}).get("focus_follows_limb") in ("no", "partly"):
                ft.write(json.dumps(dict(seg=w["seg"], window=w["id"], f0=w["f0"], f1=w["f1"], focus=w.get("focus"),
                                         problem=j["trail"].get("problem"))) + "\n"); counts["trail"] += 1
            gc = j.get("gemini_check") or {}
            if gc.get("verdict") in ("wrong", "partly"):
                fc.write(json.dumps(dict(seg=w["seg"], window=w["id"], f0=w["f0"], f1=w["f1"], errors=gc.get("errors"),
                                         judge_attacks=(j.get("event") or {}).get("attacks"),
                                         gemini=ans.get(w["id"], {}).get("trace"))) + "\n"); counts["classifier"] += 1
    # judge calibration: on the oracle windows, does the judge list the founder's attack (same fighter, same kind)?
    jk = lambda t: {"jab": "punch", "cross": "punch", "hook": "punch", "uppercut": "punch", "overhand": "punch",
                    "roundhouse": "kick", "calf_kick": "kick", "teep": "kick", "side_kick": "kick", "front_kick": "kick",
                    "knee": "knee", "takedown": "takedown", "clinch_entry": "clinch"}.get(t, t)
    cal = Counter()
    for w in json.load(open(D / "oracle.json")):
        j = J.get(w["id"])
        if not j:
            continue
        t0 = 230000 + w["f0"] * 40
        atk = (j.get("event") or {}).get("attacks") or []
        for t, a, n, o in EXPECTED:
            if t0 <= t <= t0 + (w["f1"] - w["f0"]) * 40:
                cal["items"] += 1
                same = [x for x in atk if str(x.get("fighter", "")).lower() == a and jk(x.get("technique")) == kind(n)]
                cal["fighter_kind"] += bool(same)
                cal["name"] += any(x.get("technique") == n for x in same)
                cal["other_fighter_only"] += (not same) and any(jk(x.get("technique")) == kind(n) for x in atk)
    cost = sum(r.get("cost", 0) for p in (D / "runs").glob("*.jsonl") for r in rows(p))
    summary = dict(segments={s: dict(label=v[0], A=v[1], B=v[2], video=v[4]) for s, v in SEGS.items()}, stats=stats,
                   ablation=ablation(D), endtoend=endtoend(D), judge_calibration=dict(cal), detection=detection(D), judges=agg,
                   datasets=dict(counts), cost=round(cost, 2), cards=cards)
    json.dump(summary, open(D / "summary.json", "w"), indent=1)
    print("cards", len(cards), "judged", sum(1 for c in cards if c["judge"]), "cost", round(cost, 2), dict(counts))


def _get(d, path):
    for k in path.split("."):
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d


if __name__ == "__main__":
    main(sys.argv[1])
