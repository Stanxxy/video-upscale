"""Gemini Hydrogen as the attack classifier on rendered windows, under one input condition.

Conditions: video (raw clip), raw (frames), id (+ identity boxes), pose (+ skeletons), trace (+ limb trails and the
measured trace facts of the sensing candidates in the window). The answer lists every attack started in the
window: fighter, action, technique, time, outcome. Blocks, parries and guard moves are not attacks.

Runs on the Mac (the key stays here). Usage:
  classify.py WINDOWS_JSON PACK_DIR COND OUT_JSONL [--repeat N] [--cands CANDS_DIR]
"""
import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

MODEL = "gemini-4-hydrogen"
PRICE_IN, PRICE_OUT = 1.50, 7.50        # $ per 1M tokens (thinking billed as output)
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
ACTIONS = ["strike_punch", "strike_kick", "strike_knee", "strike_elbow", "takedown_attempt", "clinch_entry"]
TECHNIQUES = ["jab", "cross", "hook", "uppercut", "overhand", "roundhouse", "calf_kick", "teep", "side_kick",
              "front_kick", "clinch_knee", "straight_knee", "elbow", "double_leg", "single_leg", "body_lock",
              "clinch_entry", "other"]
SCHEMA = {"type": "object", "required": ["attacks"], "properties": {"attacks": {"type": "array", "items": {
    "type": "object", "required": ["fighter", "action", "technique", "time_s", "outcome", "evidence"],
    "properties": {"fighter": {"type": "string", "enum": ["A", "B"]},
                   "action": {"type": "string", "enum": ACTIONS},
                   "technique": {"type": "string", "enum": TECHNIQUES},
                   "time_s": {"type": "number"},
                   "outcome": {"type": "string", "enum": ["landed", "missed", "unclear"]},
                   "evidence": {"type": "string"}}}}}}
DEFS = (
    "Technique definitions. jab: lead-hand straight punch. cross: rear-hand straight punch. hook: bent arm (elbow "
    "near 90 degrees) swinging sideways in an arc. uppercut: rising punch from below. overhand: rear hand looping "
    "over the top and coming down. roundhouse: hip turns over, the shin sweeps sideways into the target; calf_kick: "
    "a low roundhouse to the calf. teep: knee chambers straight up, then the sole pushes straight forward, hips "
    "square. side_kick: side-on, heel thrust along the line. front_kick: snapping kick with the ball of the foot. "
    "Blocks, parries, slips, covering up, guard resets, feints without commitment and stepping are NOT attacks. A "
    "punch travels to the opponent and the arm extends; a block brings the hand to the fighter's own head and "
    "keeps it there while the opponent's limb arrives.")
NOTES = {
    "video": "The video is the window.",
    "raw": "The frames are in time order; each is stamped with its time from the window start.",
    "id": "The frames are in time order, stamped with time. Fighter A has a red box and the letter A, fighter B a "
          "blue box and the letter B, from a tracker. Use the letters for who is who.",
    "pose": "The frames are in time order, stamped with time. Fighter A: red box and red skeleton; fighter B: blue "
            "box and blue skeleton (pose estimation; dots are wrists and ankles). Use the letters for who is who.",
    "trace": "The frames are in time order, stamped with time. Fighter A: red box and red skeleton; fighter B: blue "
             "box and blue skeleton (dots are wrists and ankles). Pale trails show each wrist and ankle path over "
             "the last 0.4 s; a thick yellow trail marks the limb a motion detector flagged. Below the frames are "
             "the motion detector's measurements (from pose tracking; they can be wrong in clinches). Use the "
             "letters for who is who.",
}


def key():
    for line in open(Path(__file__).resolve().parents[3] / "whole-video-analysis" / ".env"):
        if line.startswith("BJJ_GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no Gemini key")


def card(c, t0_ms):
    side = {"L": "left", "R": "right"}[c["side"]]
    lr = f", {c['lead_rear']} side" if c["lead_rear"] else ""
    bits = [f"fighter {c['fighter'].upper()} {side} {'hand' if c['group'] == 'punch' else 'leg' if c['group'] == 'kick' else 'knee'}{lr}",
            f"peak at t={(c['t_ms'] - t0_ms) / 1000:.2f}s", f"speed {c['speed_tl_s']} torso-lengths/s",
            f"moving toward opponent (cos {c['approach']})"]
    e0, e1 = c["extension_deg"]
    if e0 is not None and e1 is not None:
        bits.append(f"{'elbow' if c['group'] == 'punch' else 'knee'} angle {e0:.0f}->{e1:.0f} deg")
    if c["curvature"] is not None:
        bits.append(f"path curvature {c['curvature']} (0 straight, >0.3 arc)")
    if c["group"] == "punch" and c["wrist_above_shoulder_tl"] is not None:
        bits.append(f"wrist {'above' if c['wrist_above_shoulder_tl'] > 0 else 'below'} shoulder by {abs(c['wrist_above_shoulder_tl'])} TL")
        bits.append(f"hand ends {c['hand_to_own_head_tl']} TL from own head")
    if c["group"] != "punch":
        bits.append(f"hip rotation {c['hip_rotation_deg']} deg, knee chamber {c['knee_chamber_tl']} TL above hips")
    bits.append("reaches opponent" if c["contact"] else "does not reach opponent")
    return "; ".join(bits)


def build(win, idx, cond, names, cands):
    d = PACK / win["id"]
    parts = []
    intro = (f"MMA fight. Fighter A wears {names[0]}; fighter B wears {names[1]}. {NOTES[cond]} List every attack "
             f"(strike, takedown attempt, clinch entry) that STARTS inside this {idx['times'][-1]:.1f} s window, in "
             f"time order, with the fighter who throws it. If there is none, return an empty list. {DEFS}")
    parts.append({"text": intro})
    if cond == "video":
        parts.append({"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode((d / "clip.mp4").read_bytes()).decode()},
                      "videoMetadata": {"fps": 12}})   # the default samples 1 fps: 2-3 frames of a 2 s window
    else:
        for i, t in enumerate(idx["times"]):
            parts.append({"text": f"t={t:.2f}s"})
            parts.append({"inlineData": {"mimeType": "image/jpeg",
                                          "data": base64.b64encode((d / f"{cond}_{i:02d}.jpg").read_bytes()).decode()}})
    if cond == "trace":
        t0_ms = round(idx["f0"] * 40)
        mine = [c for c in cands if idx["f0"] <= c["peak_f"] <= idx["f1"]]
        txt = "\n".join("- " + card(c, t0_ms) for c in mine) or "- no fast limb motion detected"
        parts.append({"text": "Motion detector measurements:\n" + txt})
    return parts


def call(parts, api_key):
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": "HIGH"},
                                 "mediaResolution": "MEDIA_RESOLUTION_HIGH"}}
    for attempt in range(4):
        t = time.monotonic()
        r = httpx.post(URL, json=body, headers={"x-goog-api-key": api_key}, timeout=300)
        if r.status_code == 200:
            v = r.json()
            u = v.get("usageMetadata", {})
            tin, tout = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            text = v["candidates"][0]["content"]["parts"][-1]["text"]
            return dict(answer=json.loads(text), tokens_in=tin, tokens_out=tout,
                        cost=round(tin * PRICE_IN / 1e6 + tout * PRICE_OUT / 1e6, 5),
                        latency_s=round(time.monotonic() - t, 1), model_version=v.get("modelVersion"))
        if r.status_code in (429, 500, 503) and attempt < 3:
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")


def main():
    global PACK
    ap = argparse.ArgumentParser()
    ap.add_argument("windows"); ap.add_argument("pack"); ap.add_argument("cond"); ap.add_argument("out")
    ap.add_argument("--repeat", type=int, default=1); ap.add_argument("--cands", default=None)
    ap.add_argument("--names", nargs=2, required=True); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    PACK = Path(a.pack)
    wins = json.load(open(a.windows))
    index = json.load(open(PACK / f"index_{Path(a.windows).stem}.json"))
    cands = {}
    if a.cands:
        for p in Path(a.cands).glob("*/candidates.json"):
            cands[p.parent.name] = json.load(open(p))
    done = set()
    if Path(a.out).exists():
        done = {(json.loads(l)["window"], json.loads(l)["rep"]) for l in open(a.out)}
    api_key = key()
    jobs = [(w, rep) for rep in range(a.repeat) for w in wins if (w["id"], rep) not in done]

    def run(job):
        w, rep = job
        parts = build(w, index[w["id"]], a.cond, a.names, cands.get(w["seg"], []))
        res = call(parts, api_key)
        return dict(window=w["id"], cond=a.cond, rep=rep, **res)

    total = 0.0
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as fh:
        for res in ex.map(run, jobs):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            total += res["cost"]
    print(f"{a.cond}: {len(jobs)} calls, ${total:.2f}")


if __name__ == "__main__":
    main()
