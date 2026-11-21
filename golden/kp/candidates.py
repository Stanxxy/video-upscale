"""Stage 5: strike candidates and their trace features, from the fused limb tracks.

Per fighter and limb (wrists for punches, ankles for kicks, knees for knees): speed relative to the fighter's own
pelvis in torso lengths per second; a candidate is a local speed peak above the limb's threshold whose motion
points toward the opponent (approach cosine). Peaks of the same fighter and limb within 240 ms merge.

Features measured on each candidate (what the trace says, never a technique name): lead or rear side from the
stance, extension (elbow / knee angle opening), path curvature and lateral offset, wrist height against the
shoulder, downward speed at the peak, contact with the opponent's mask, hip-line rotation (kicks), knee chamber
height, and whether the hand ends near the fighter's own head (guard / block).

Usage: candidates.py SEG_DIR [--fps 25]  (reads identity.json, kp_rtmw.npz, limbs.npz, masks_ab.npz;
writes candidates.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

FPS = 25
LIMBS = [9, 10, 7, 8, 15, 16, 13, 14]                    # order in limbs.npz
GROUPS = {"punch": (0, 1), "kick": (4, 5), "knee": (6, 7)}  # indices into LIMBS (L, R)
THRESH = {"punch": 4.0, "kick": 3.0, "knee": 3.0}         # torso lengths / s
CHAIN = {0: (5, 7, 9), 1: (6, 8, 10), 4: (11, 13, 15), 5: (12, 14, 16), 6: (11, 13, 15), 7: (12, 14, 16)}


def smooth(x, k=3):
    out = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - k // 2), min(len(x), i + k // 2 + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w).any(axis=-1)] if w.ndim > 1 else w[~np.isnan(w)]
        out[i] = w.mean(axis=0) if len(w) else np.nan
    return out


def angle(a, b, c):
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1)))) if d > 0 else float("nan")


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main(seg_dir):
    seg = Path(seg_dir)
    kz, lz = np.load(seg / "kp_rtmw.npz"), np.load(seg / "limbs.npz")
    kp, sc, fused, source = kz["kp"], kz["score"], lz["fused"], lz["source"]
    packed = np.load(seg / "masks_ab.npz")["packed"]
    ident = json.load(open(seg / "identity.json"))["frames"]
    n = kp.shape[0]
    pelvis = (kp[:, :, 11] + kp[:, :, 12]) / 2
    shoulders = (kp[:, :, 5] + kp[:, :, 6]) / 2
    tl = np.linalg.norm(shoulders - pelvis, axis=-1)
    tl_med = np.array([np.nanmedian(tl[:, j]) for j in range(2)])
    cands = []
    for j in range(2):
        o = 1 - j
        rel = fused[:, j] - pelvis[:, j, None, :]
        for group, (li_l, li_r) in GROUPS.items():
            for li in (li_l, li_r):
                p = smooth(rel[:, li])
                v = np.gradient(p, axis=0) * FPS / tl_med[j]
                speed = np.linalg.norm(v, axis=-1)
                vabs = np.gradient(smooth(fused[:, j, li]), axis=0)
                target = (kp[:, o, 0] + (kp[:, o, 11] + kp[:, o, 12]) / 2) / 2   # between head and hips
                to_t = target - fused[:, j, li]
                cosv = np.sum(vabs * to_t, -1) / (np.linalg.norm(vabs, axis=-1) * np.linalg.norm(to_t, axis=-1) + 1e-6)
                peaks = []
                for f in range(2, n - 2):
                    s = speed[f]
                    if np.isnan(s) or s < THRESH[group]:
                        continue
                    win = speed[max(0, f - 4):f + 5]
                    if s < np.nanmax(win) or np.isnan(cosv[f]) or cosv[f] < 0.3:
                        continue
                    if peaks and f - peaks[-1] <= 6:
                        continue
                    peaks.append(f)
                for f in peaks:
                    cands.append(dict(fighter="ab"[j], group=group, side="LR"[li - li_l], li=li, peak=f,
                                      speed=float(speed[f]), approach=float(cosv[f])))
    out = []
    for c in sorted(cands, key=lambda c: c["peak"]):
        j, o, f, li = "ab".index(c["fighter"]), 1 - "ab".index(c["fighter"]), c["peak"], c["li"]
        f0 = max(0, f - 6)
        f1 = min(n - 1, f + 4)
        P = fused[f0:f + 1, j, li]
        ok = ~np.isnan(P).any(axis=1)
        curv = lateral = None
        if ok.sum() >= 3:
            Q = P[ok]
            chord = Q[-1] - Q[0]
            L = np.linalg.norm(chord)
            if L > 1:
                nrm = np.array([-chord[1], chord[0]]) / L
                dev = (Q - Q[0]) @ nrm
                curv, lateral = float(np.abs(dev).max() / L), float(dev[np.argmax(np.abs(dev))] / tl_med[j])
        a, b, cc = CHAIN[c["li"]]
        ext0 = angle(kp[f0, j, a], kp[f0, j, b], kp[f0, j, cc])
        ext1 = angle(kp[f, j, a], kp[f, j, b], kp[f, j, cc])
        # stance: the foot nearer the opponent is the lead foot; lead hand is on the same side
        to_opp = pelvis[f, o] - pelvis[f, j]
        dl, dr = [np.dot(kp[f, j, k] - pelvis[f, j], to_opp) for k in (15, 16)]
        lead = None if np.isnan(dl) or np.isnan(dr) else ("L" if dl > dr else "R")
        lead_rear = None if lead is None else ("lead" if lead == c["side"] else "rear")
        wrist_vs_shoulder = None
        if c["group"] == "punch":
            sh = kp[f, j, 5 if c["side"] == "L" else 6]
            wrist_vs_shoulder = float((sh[1] - fused[f, j, li][1]) / tl_med[j])  # + means above the shoulder
        down = float(np.gradient(smooth(fused[:, j, li]), axis=0)[f][1] * FPS / tl_med[j])
        head = kp[f1, j, 0]
        guard = float(np.linalg.norm(fused[f1, j, li] - head) / tl_med[j])
        contact = False
        for g in range(max(0, f - 2), min(n, f + 3)):
            m = np.unpackbits(packed[g, o]).reshape(288, 288).astype(bool)
            x, y = fused[g, j, li]
            if not np.isnan(x) and m[int(np.clip(y / 720 * 288, 0, 287)), int(np.clip(x / 1280 * 288, 0, 287))]:
                contact = True
        hip0 = kp[f0, j, 12] - kp[f0, j, 11]
        hip1 = kp[f, j, 12] - kp[f, j, 11]
        hip_rot = float(np.degrees(np.arctan2(hip0[0] * hip1[1] - hip0[1] * hip1[0], np.dot(hip0, hip1))))
        knee_y = kp[f0:f + 1, j, 13 if c["side"] == "L" else 14, 1]
        chamber = float((pelvis[f, j, 1] - np.nanmin(knee_y)) / tl_med[j]) if c["group"] != "punch" else None
        src = source[f0:f + 1, j, li]
        fl = ident[str(f)].get("flags", [])
        out.append(dict(
            id=f"{seg.name}-{c['fighter']}-{f:05d}-{c['group']}{c['side']}", seg=seg.name, fighter=c["fighter"],
            group=c["group"], side=c["side"], lead_rear=lead_rear, peak_f=f, t_ms=round(f * 1000 / FPS),
            onset_f=f0, end_f=f1, speed_tl_s=r(c["speed"], 1), approach=r(c["approach"]),
            extension_deg=[r(ext0, 0), r(ext1, 0)], curvature=r(curv), lateral_tl=r(lateral),
            wrist_above_shoulder_tl=r(wrist_vs_shoulder), down_speed_tl_s=r(down, 1), hand_to_own_head_tl=r(guard),
            contact=contact, hip_rotation_deg=r(hip_rot, 0), knee_chamber_tl=r(chamber),
            from_tracker_frames=int((src == 2).sum()), pose_frames=int((src == 1).sum()),
            identity_flags=fl))
    json.dump(out, open(seg / "candidates.json", "w"), indent=0)
    print(json.dumps(dict(seg=seg.name, candidates=len(out),
                          per_group={g: sum(c["group"] == g for c in out) for g in GROUPS})))


if __name__ == "__main__":
    main(sys.argv[1])
