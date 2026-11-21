"""Stage 2b: which detected person is fighter A and which is B on every frame, from kit colours at body parts.

Each person already has a skeleton (poseall.py). Colours are read where the skeleton puts the shorts, gloves,
shin guards and head, not over the whole body, so a clinch partner or the canvas pollutes them less and the
stage works with any detector that gives boxes.

1. Parts: median LAB colour of shorts, gloves, shins and head; share of skin pixels on the torso.
2. Camera cuts: a drop in the colour-histogram match between neighbouring frames starts a new shot; continuity
   never carries across a cut.
3. Kit model per video: frames where the two fighters stand apart give pairs. Starting from the colours in the
   kit names, each fighter's part colours are re-estimated from
   those pairs, and each part is weighted by how well it separates the two (identical shorts get weight 0 and the
   gloves or shin guards decide). Who can be a fighter depends on the video: when most large people are
   bare-chested, only bare-chested tracks are, and the pairs are the two largest of them standing apart. In a
   T-shirt fight or under coloured light (skin does not read as skin) the pair is instead one large person who
   looks like A by the kit names and one who looks like B, and the other large people in those frames (referee,
   corner) give a third, "not a fighter" colour model.
4. Per frame: a person's kit evidence is averaged along its track (+-12 frames, same shot); the Hungarian method
   gives at most one person to A and one to B on kit evidence plus box continuity. When exactly two
   fighter-sized candidates are in the frame they are taken as the two fighters, whichever pairing fits better. Tracks that look more like
   the third model than like either fighter are never candidates.

Flags per frame: ``cut``, ``kit_conflict_<slot>``, ``switch_<slot>``, ``merged`` (one fighter box about as big as two).

Usage: identity2.py SEG_DIR DET NAME_A NAME_B [DEBUG_FRAMES]   (reads pose_<DET>.npz; writes identity_<DET>.json,
kp_<DET>.npz; DEBUG_FRAMES "30,390" prints every person considered on those frames)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

KP_OK = 2.0          # RTMW SimCC score for a joint to locate a part
PARTS = ("shorts", "gloves", "shins", "head", "torso")
PART_WORDS = {"shorts": "shorts", "tights": "shorts", "trunks": "shorts", "gloves": "gloves",
              "shin guards": "shins", "shin pads": "shins", "headgear": "head", "head guard": "head"}
NAMED = {"black": (20, 128, 128), "white": (235, 128, 128), "grey": (130, 128, 128), "gray": (130, 128, 128),
         "light grey": (175, 128, 128), "dark brown": (55, 140, 145), "brown": (75, 140, 150),
         "red": (110, 175, 155), "blue": (70, 140, 95), "green": (110, 105, 140), "orange": (150, 165, 180),
         "yellow": (220, 115, 200), "neon yellow": (225, 100, 210)}
SKIN_MIN, FIT_MAX, WINDOW = 0.25, 6.0, 12
CUT_CORR = 0.55      # histogram correlation below this between neighbouring frames is a camera cut


def named_parts(name):
    """{'gloves': LAB, 'shins': LAB} from 'Red gloves, red shin guards'. Unnamed or unknown colours are left out."""
    out = {}
    for chunk in name.lower().split(","):
        for word in sorted(PART_WORDS, key=len, reverse=True):
            if word in chunk:
                rest = chunk.replace(word, " ")
                for key in sorted(NAMED, key=len, reverse=True):
                    if key in rest:
                        out[PART_WORDS[word]] = np.array(NAMED[key], float)
                        break
                break
    return out


def poly_lab(lab, pts):
    """Median LAB inside a polygon (None when it is tiny or off the frame)."""
    pts = np.round(np.asarray(pts)).astype(np.int32)
    x0, y0 = np.maximum(pts.min(0), 0)
    x1, y1 = np.minimum(pts.max(0) + 1, [lab.shape[1], lab.shape[0]])
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [pts - [x0, y0]], 1)
    px = lab[y0:y1, x0:x1][mask > 0]
    return px if len(px) >= 12 else None


def limb_quad(p, q, t0, t1, width):
    """Quad along the segment p->q from fraction t0 to t1, `width` times the segment length wide."""
    d = q - p
    n = np.array([-d[1], d[0]]) * width / 2
    a, b = p + d * t0, p + d * t1
    return [a - n, a + n, b + n, b - n]


def is_skin(px):
    px = px.astype(float)
    return (px[:, 0] >= 35) & (px[:, 0] <= 225) & (px[:, 1] >= 134) & (px[:, 2] >= 134)


def describe(lab, kp, sc):
    """Part colours and torso skin share for one skeleton."""
    ok = sc >= KP_OK
    d = {"skin": None}
    if ok[[5, 6, 11, 12]].all():
        c = kp[[5, 6, 12, 11]].mean(0)
        px = poly_lab(lab, c + (kp[[5, 6, 12, 11]] - c) * 0.7)
        if px is not None:
            d["skin"] = float(is_skin(px).mean())
            d["torso"] = np.median(px, 0)
    if ok[[11, 12]].all() and ok[[13, 14]].any():
        lk = kp[13] if ok[13] else kp[11] + (kp[14] - kp[12])
        rk = kp[14] if ok[14] else kp[12] + (kp[13] - kp[11])
        px = poly_lab(lab, [kp[11], kp[12], kp[12] + (rk - kp[12]) * 0.55, kp[11] + (lk - kp[11]) * 0.55])
        if px is not None:
            d["shorts"] = np.median(px, 0)
    hands = []
    for el, wr in ((7, 9), (8, 10)):
        if ok[el] and ok[wr]:
            px = poly_lab(lab, limb_quad(kp[el], kp[wr], 0.95, 1.45, 0.55))
            if px is not None:
                hands.append(px)
    if hands:
        d["gloves"] = np.median(np.concatenate(hands), 0)
    shins = []
    for kn, an in ((13, 15), (14, 16)):
        if ok[kn] and ok[an]:
            px = poly_lab(lab, limb_quad(kp[kn], kp[an], 0.25, 0.85, 0.28))
            if px is not None:
                shins.append(px)
    if shins:
        d["shins"] = np.median(np.concatenate(shins), 0)
    if ok[[3, 4]].all() and ok[0]:
        c, r = kp[[0, 3, 4]].mean(0), max(np.linalg.norm(kp[3] - kp[4]) * 0.75, 6)
        px = poly_lab(lab, [c + [-r, -1.6 * r], c + [r, -1.6 * r], c + [r, 0.2 * r], c + [-r, 0.2 * r]])
        if px is not None:
            d["head"] = np.median(px, 0)
    return d


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)


class Kit:
    """Two fighters' part colours, the spread of each part, and how much each part separates them."""

    def __init__(self, names):
        self.proto = {p: {} for p in PARTS}
        for slot, name in zip("ab", names):
            for p, lab in named_parts(name).items():
                self.proto[p][slot] = lab
        self.spread = {p: 25.0 for p in PARTS}
        self.weight = {p: 1.0 if len(self.proto[p]) == 2 else 0.0 for p in PARTS}
        self.source, self.other = "names", {}
        self.named = {p for p in PARTS if len(self.proto[p]) == 2}

    def evidence(self, d):
        """(log-odds of being A in units of spread, fit to the nearer kit) over the weighted parts present."""
        num = fit = wsum = 0.0
        for p in PARTS:
            w = self.weight[p]
            if w <= 0 or p not in d:
                continue
            da, db = (np.linalg.norm(d[p] - self.proto[p][s]) / self.spread[p] for s in "ab")
            num, fit, wsum = num + w * (db - da), fit + w * min(da, db), wsum + w
        return (num / wsum, fit / wsum) if wsum else (0.0, None)

    def fighter_margin(self, d):
        """How much closer the person is to a fighter than to the third model, in spreads (None: cannot tell)."""
        gaps = []
        for p, lab in self.other.items():
            if p in d and len(self.proto[p]) == 2:
                near = min(np.linalg.norm(d[p] - self.proto[p][s]) for s in "ab")
                gaps.append((np.linalg.norm(d[p] - lab) - near) / self.spread[p])
        return float(np.mean(gaps)) if gaps else None

    def fit_other(self, dets, idx):
        for p in PARTS:
            xs = np.array([dets[i][p] for i in idx if p in dets[i]])
            if len(xs) >= 20:
                self.other[p] = np.median(xs, 0)

    def refit(self, dets, pairs, keep_named=False):
        """Re-estimate the part colours from pairs that must be the two fighters (5 rounds). With keep_named the
        parts the kit names describe keep their named colours (the pairs were chosen by those very colours, so
        re-estimating them lets a wrong pairing confirm itself); only the other parts are learned."""
        for _ in range(5):
            member = {"a": [], "b": []}
            for i, j in pairs:
                ei, ej = self.evidence(dets[i])[0], self.evidence(dets[j])[0]
                if abs(ei - ej) < 0.5:
                    continue
                first = "a" if ei > ej else "b"
                member[first].append(i)
                member["b" if first == "a" else "a"].append(j)
            if min(len(member["a"]), len(member["b"])) < 60:   # too few pairs: the kit stays as named
                return
            for p in PARTS:
                xs = {s: np.array([dets[i][p] for i in member[s] if p in dets[i]]) for s in "ab"}
                if min(len(xs["a"]), len(xs["b"])) < 20:        # part rarely visible: keep what the names say
                    continue
                if not (keep_named and p in self.named):
                    for s in "ab":
                        self.proto[p][s] = np.median(xs[s], 0)
                dev = np.concatenate([np.linalg.norm(xs[s] - self.proto[p][s], axis=1) for s in "ab"])
                self.spread[p] = max(float(np.median(dev)) * 1.48, 6.0)
                sep = np.linalg.norm(self.proto[p]["a"] - self.proto[p]["b"]) / self.spread[p]
                self.weight[p] = float(np.clip(sep - 1.0, 1.0 if keep_named and p in self.named else 0.0, 4.0))
            self.source = f"{len(member['a'])} apart pairs"

    def summary(self):
        return dict(source=self.source, weight={p: round(w, 2) for p, w in self.weight.items()},
                    proto={p: {s: [round(float(v)) for v in lab] for s, lab in self.proto[p].items()}
                           for p in PARTS if self.weight[p] > 0})


def main(seg_dir, det, name_a, name_b, debug=""):
    seg = Path(seg_dir)
    z = np.load(seg / f"pose_{det}.npz")
    nframes = int(z["nframes"])
    by_frame = defaultdict(list)
    for i, f in enumerate(z["frame"]):
        by_frame[int(f)].append(i)
    kp_all, sc_all = z["kp"].astype(np.float32), z["score"].astype(np.float32)
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    dets, shot, shots, prev_hist = {}, 0, [], None
    for f in range(nframes):
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < CUT_CORR:
            shot += 1
        prev_hist = hist
        shots.append(shot)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for i in by_frame.get(f, []):
            d = describe(lab, kp_all[i], sc_all[i])
            d.update(f=f, tid=int(z["tid"][i]), box=[float(v) for v in z["box"][i]], i=i)
            d["area"] = d["box"][2] * d["box"][3]
            dets[i] = d
    by_track = defaultdict(list)
    for d in dets.values():
        by_track[d["tid"]].append(d)
    kit = Kit((name_a, name_b))
    if not any(kit.weight.values()):
        raise SystemExit(f"no colour in the kit names {name_a!r} / {name_b!r} names a part this stage can read")
    # bare-chested: over the whole track (robust while the torso is hidden in a clinch) or over the track near
    # this frame (a track can drift from the referee onto a fighter)
    for ds in by_track.values():
        fs = np.array([d["f"] for d in ds])
        skin = np.array([np.nan if d["skin"] is None else d["skin"] for d in ds])
        seen = skin[~np.isnan(skin)]
        whole = len(seen) >= 5 and float(np.median(seen)) >= SKIN_MIN
        for d in ds:
            near = skin[(np.abs(fs - d["f"]) <= 2 * WINDOW) & ~np.isnan(skin)]
            d["bare"] = whole or (len(near) >= 3 and float(np.median(near)) >= SKIN_MIN)
    top2 = [i for idx in by_frame.values() for i in sorted(idx, key=lambda i: -dets[i]["area"])[:2]]
    bare_fight = float(np.mean([dets[i]["bare"] for i in top2])) >= 0.3
    pairs, thirds = [], []
    for f, idx in by_frame.items():
        if bare_fight:   # the two largest bare-chested people, standing apart
            c = sorted((i for i in idx if dets[i]["bare"]), key=lambda i: -dets[i]["area"])[:2]
            if len(c) < 2:
                continue
            i, j = c
        else:            # clothed fighters or coloured light: one large person reads as A by the names, one as B
            big = sorted(idx, key=lambda i: -dets[i]["area"])[:4]
            ev = {i: kit.evidence(dets[i]) for i in big}
            ev = {i: e for i, (e, fit) in ev.items() if fit is not None and fit <= 2.0}   # near a named kit
            if len(ev) < 2:
                continue
            i, j = max(ev, key=ev.get), min(ev, key=ev.get)
            if ev[i] < 0.5 or ev[j] > -0.5:
                continue
        if iou(dets[i]["box"], dets[j]["box"]) < 0.15 \
                and min(dets[i]["area"], dets[j]["area"]) > 0.35 * max(dets[i]["area"], dets[j]["area"]):
            pairs.append((i, j))
            small = min(dets[i]["area"], dets[j]["area"])
            thirds += [k for k in idx if k not in (i, j) and dets[k]["area"] > 0.25 * small]
    kit.refit(dets, pairs, keep_named=not bare_fight)
    if not bare_fight:
        kit.fit_other(dets, thirds)
    for d in dets.values():
        ev, fit = kit.evidence(d)
        d["pa"] = float(1 / (1 + np.exp(-ev)))
        d["fit"] = fit
        d["margin"] = kit.fighter_margin(d)
    for ds in by_track.values():
        fs = np.array([d["f"] for d in ds])
        margin = np.array([np.nan if d["margin"] is None else d["margin"] for d in ds])
        for d in ds:
            near = margin[(np.abs(fs - d["f"]) <= 2 * WINDOW) & ~np.isnan(margin)]
            d["fighter"] = d["bare"] if bare_fight else (len(near) == 0 or float(np.median(near)) >= 0)
    track_pa = defaultdict(dict)
    for d in dets.values():
        if d["fit"] is not None:
            track_pa[d["tid"]][d["f"]] = d["pa"]
    single = np.median([dets[i]["area"] for pr in pairs for i in pr]) if pairs else \
        np.median([d["area"] for d in dets.values() if d["fighter"]] or [1.0])
    out, stats = {}, Counter()
    prev, prev_id = {"a": None, "b": None}, {"a": None, "b": None}
    kp = np.full((nframes, 2, 133, 2), np.nan, np.float32)
    score = np.zeros((nframes, 2, 133), np.float32)
    for f in range(nframes):
        frame = {"flags": []}
        if f and shots[f] != shots[f - 1]:
            prev, prev_id = {"a": None, "b": None}, {"a": None, "b": None}
            frame["flags"].append("cut")
        cands = [dets[i] for i in by_frame.get(f, [])
                 if dets[i]["fighter"] and (dets[i]["fit"] is None or dets[i]["fit"] <= FIT_MAX)]
        cands = sorted(cands, key=lambda d: -d["area"])[:4]
        if cands:
            cost = np.zeros((len(cands), 2))
            for i, d in enumerate(cands):
                near = [p for g, p in track_pa[d["tid"]].items() if abs(g - f) <= WINDOW and shots[g] == shots[f]]
                d["pa_track"] = float(np.mean(near)) if near else 0.5   # no kit part visible near this frame
                for j, slot in enumerate("ab"):
                    kit_cost = 1 - d["pa_track"] if slot == "a" else d["pa_track"]
                    cont = 1 - iou(d["box"], prev[slot]) if prev[slot] is not None else 0.5
                    cost[i, j] = 2.0 * kit_cost + 0.7 * cont
            rows, cols = linear_sum_assignment(cost)
            # exactly two fighter-sized candidates are the two fighters: the better of the two pairings stands
            # even when both read as the same kit (one's shorts hidden behind the other)
            pair = len(cands) == 2 and min(d["area"] for d in cands) >= 0.35 * single
            for i, j in zip(rows, cols):
                if cost[i, j] > 1.9 and not pair:   # neither kit nor continuity supports this person as that fighter
                    continue
                d, slot = cands[i], "ab"[j]
                frame[slot] = dict(id=d["tid"], box=[round(v, 1) for v in d["box"]], pa=round(d["pa"], 3),
                                   pa_track=round(d["pa_track"], 3))
                if (slot == "a" and d["pa"] < 0.35) or (slot == "b" and d["pa"] > 0.65):
                    frame["flags"].append(f"kit_conflict_{slot}")
                if prev_id[slot] is not None and prev_id[slot] != d["tid"]:
                    frame["flags"].append(f"switch_{slot}")
                prev[slot], prev_id[slot] = d["box"], d["tid"]
                kp[f, j], score[f, j] = kp_all[d["i"]], sc_all[d["i"]]
            got = [s for s in "ab" if s in frame]
            if len(got) == 1 and frame[got[0]]["box"][2] * frame[got[0]]["box"][3] > 1.6 * single:
                frame["flags"].append("merged")
        if str(f) in debug.split(","):
            for i in by_frame.get(f, []):
                d = dets[i]
                print(f"DEBUG f{f} tid{d['tid']} box{[round(v) for v in d['box']]} skin {d['skin'] and round(d['skin'], 2)} "
                      f"fighter {d['fighter']} pa {d['pa']:.2f} fit {d['fit'] and round(d['fit'], 1)} "
                      f"parts {[p for p in PARTS if p in d]} -> {[s for s in 'ab' if frame.get(s, {}).get('id') == d['tid']]}")
        stats[f"fighters_{sum(s in frame for s in 'ab')}"] += 1
        for fl in frame["flags"]:
            stats[fl if fl in ("merged", "cut") else fl.rsplit("_", 1)[0]] += 1
        out[f] = frame
    np.savez_compressed(seg / f"kp_{det}.npz", kp=kp, score=score)
    summary = dict(frames=nframes, det=det, names=[name_a, name_b], kit=kit.summary(), apart_pairs=len(pairs),
                   bare_fight=bare_fight, third_parts=sorted(kit.other),
                   fighter_tracks=len({d["tid"] for d in dets.values() if d["fighter"]}), all_tracks=len(by_track),
                   cuts=stats["cut"],
                   both_pct=round(stats["fighters_2"] / nframes, 3), one_pct=round(stats["fighters_1"] / nframes, 3),
                   none_pct=round(stats["fighters_0"] / nframes, 3), kit_conflicts=stats["kit_conflict"],
                   switches=stats["switch"], merged=stats["merged"])
    json.dump(dict(summary=summary, frames=out), open(seg / f"identity_{det}.json", "w"))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:6])
