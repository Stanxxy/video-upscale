"""Pose-sequence student: a small temporal network that reads both fighters' skeletons and marks attacks.

It is trained on labelled attack events (silver labels from the Gemini teacher, or people's corrections) and costs
nothing per video at inference. One sequence is one fighter in one segment: his skeleton, centred on his hips,
scaled by his torso length, mirrored so the opponent is always on the +x side, with left/right limbs re-ordered
as lead/rear from his stance (a jab is the lead hand, a cross the rear hand), plus joint velocities and where the
opponent's head and hips are. The network labels every frame with an attack kind (or none) and a technique.

Evaluation is leave-one-video-out: the model that scores a segment never saw any segment of the same video.

Usage: student.py OUT_DIR DET SEGS_JSON EVENTS_JSON RESULT_JSON [--frac 1.0] [--seed 0]
  (OUT_DIR/<seg>/kp_<DET>.npz; EVENTS_JSON rows: seg, f, fighter, kind, tech; --frac trains on that share of the
  training videos' events, for learning curves)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

KINDS = ["none", "punch", "kick", "knee", "takedown", "clinch"]
TECHS = ["jab", "cross", "hook", "uppercut", "overhand", "roundhouse", "calf_kick", "teep", "side_kick", "front_kick",
         "clinch_knee", "straight_knee", "elbow", "double_leg", "single_leg", "body_lock", "clinch_entry", "other"]
ACTION = {"punch": "strike_punch", "kick": "strike_kick", "knee": "strike_knee", "takedown": "takedown_attempt",
          "clinch": "clinch_entry"}
PAIRS = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]   # left, right: shoulder ... ankle
HALF, CROP, EPOCHS = 3, 256, 60     # frames either side of a peak labelled positive; training crop; epochs
MATCH = 10                          # frames: a predicted event matches a labelled one of the same fighter and kind
NFEAT = 63                          # 13 joints x (position, velocity) + opponent head and hips + flags


def features(kp, sc, j):
    """[n, F] features for fighter j (opponent 1-j); rows where j is untracked are zero with the tracked flag off."""
    n = kp.shape[0]
    me = np.where((sc[:, j] >= 2.0)[..., None], kp[:, j], np.nan)[:, :17]
    op = np.where((sc[:, 1 - j] >= 2.0)[..., None], kp[:, 1 - j], np.nan)[:, :17]
    pelvis, opelvis = (me[:, 11] + me[:, 12]) / 2, (op[:, 11] + op[:, 12]) / 2
    tl = np.nanmedian(np.linalg.norm((me[:, 5] + me[:, 6]) / 2 - pelvis, axis=-1))
    if not np.isfinite(tl) or tl < 5:
        return np.zeros((n, NFEAT), np.float32)
    side = np.sign(opelvis[:, 0] - pelvis[:, 0])
    side = np.where(np.isfinite(side) & (side != 0), side, np.nan)
    for i in range(1, n):                      # hold the last known side through gaps
        if np.isnan(side[i]):
            side[i] = side[i - 1]
    side = np.where(np.isnan(side), 1.0, side)
    rel = (me - pelvis[:, None]) / tl
    rel[..., 0] *= side[:, None]
    orel = (op[:, [0, 11]] - pelvis[:, None]) / tl
    orel[..., 0] *= side[:, None]
    lead_left = rel[:, 15, 0] > rel[:, 16, 0]  # the ankle nearer the opponent is the lead leg
    k = 12
    votes = np.array([np.nanmean(np.where(np.isnan(rel[max(0, i - k):i + k + 1, 15, 0]), np.nan,
                                          lead_left[max(0, i - k):i + k + 1])) if np.isfinite(rel[max(0, i - k):i + k + 1, 15, 0]).any()
                      else 1.0 for i in range(n)])
    lead_left = votes >= 0.5
    joints = [rel[:, 0]]
    for left, right in PAIRS:
        joints.append(np.where(lead_left[:, None], rel[:, left], rel[:, right]))
        joints.append(np.where(lead_left[:, None], rel[:, right], rel[:, left]))
    pos = np.stack(joints, axis=1)             # [n, 13, 2]
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / 2 * 25 / 10   # torso lengths per 0.4 s keeps values near 1
    tracked = np.isfinite(pelvis[:, 0]).astype(np.float32)
    otracked = np.isfinite(opelvis[:, 0]).astype(np.float32)
    conf = np.clip(sc[:, j, :17].mean(axis=1) / 6, 0, 1)
    feat = np.concatenate([pos.reshape(n, -1), vel.reshape(n, -1), orel.reshape(n, -1), tracked[:, None],
                           otracked[:, None], conf[:, None], np.isfinite(pos[..., 0]).astype(np.float32)[:, [5, 6, 11, 12]]],
                          axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).clip(-6, 6)


class Net(nn.Module):
    def __init__(self, f, hidden=64):
        super().__init__()
        layers, c = [], f
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, hidden, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2)]
            c = hidden
        self.body = nn.Sequential(*layers)
        self.kind, self.tech = nn.Conv1d(hidden, len(KINDS), 1), nn.Conv1d(hidden, len(TECHS), 1)

    def forward(self, x):   # x: [B, F, T]
        h = self.body(x)
        return self.kind(h), self.tech(h)


def targets(n, events):
    kind, tech = np.zeros(n, np.int64), np.full(n, -100, np.int64)
    for e in events:
        f = int(round(e["f"]))
        lo, hi = max(0, f - HALF), min(n, f + HALF + 1)
        kind[lo:hi] = KINDS.index(e["kind"])
        tech[lo:hi] = TECHS.index(e["tech"]) if e["tech"] in TECHS else TECHS.index("other")
    return kind, tech


def train(seqs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(seqs[0]["x"].shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    counts = np.bincount(np.concatenate([s["kind"] for s in seqs]), minlength=len(KINDS)).astype(float)
    w = torch.tensor(np.clip(counts.sum() / (len(KINDS) * np.maximum(counts, 1)), 0.2, 20), dtype=torch.float32).cuda()
    steps = max(8, len(seqs) * 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EPOCHS * steps)
    net.train()
    for _ in range(EPOCHS * steps):
        xs, ks, ts = [], [], []
        for s in rng.choice(len(seqs), 16):
            q = seqs[s]
            a = rng.integers(0, max(1, len(q["x"]) - CROP))
            x = q["x"][a:a + CROP] + rng.normal(0, 0.03, (min(CROP, len(q["x"]) - a), q["x"].shape[1])).astype(np.float32)
            pad = CROP - len(x)
            xs.append(np.pad(x, ((0, pad), (0, 0)))); ks.append(np.pad(q["kind"][a:a + CROP], (0, pad)))
            ts.append(np.pad(q["tech"][a:a + CROP], (0, pad), constant_values=-100))
        x = torch.tensor(np.stack(xs)).permute(0, 2, 1).cuda()
        k, t = net(x)
        loss = nn.functional.cross_entropy(k, torch.tensor(np.stack(ks)).cuda(), weight=w) \
            + nn.functional.cross_entropy(t, torch.tensor(np.stack(ts)).cuda(), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


def predict(net, x, thr=0.5):
    """Events from one sequence: peaks of the per-kind probability above thr, at least 6 frames apart."""
    net.eval()
    with torch.no_grad():
        k, t = net(torch.tensor(x[None]).permute(0, 2, 1).cuda())
    p, tech = torch.softmax(k[0], 0).cpu().numpy(), t[0].argmax(0).cpu().numpy()
    out = []
    for ki in range(1, len(KINDS)):
        q = p[ki]
        for f in np.argsort(-q):
            if q[f] < thr:
                break
            if all(abs(f - o["f"]) > 6 or o["kind"] != KINDS[ki] for o in out):
                out.append(dict(f=int(f), kind=KINDS[ki], tech=TECHS[tech[f]], p=round(float(q[f]), 3)))
    return out


def match(pred, gold):
    """Greedy one-to-one matching by time for the same seg, fighter and kind. Returns (hits, technique-right)."""
    used, hits, tech = set(), 0, 0
    for p in sorted(pred, key=lambda e: -e["p"]):
        best = None
        for i, g in enumerate(gold):
            if i in used or (g["seg"], g["fighter"], g["kind"]) != (p["seg"], p["fighter"], p["kind"]):
                continue
            d = abs(g["f"] - p["f"])
            if d <= MATCH and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1]); hits += 1; tech += gold[best[1]]["tech"] == p["tech"]
    return hits, tech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("det"); ap.add_argument("segs"); ap.add_argument("events"); ap.add_argument("result")
    ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    segs = json.load(open(a.segs))
    events = [e for e in json.load(open(a.events)) if e["kind"] in KINDS]
    by_seg = defaultdict(list)
    for e in events:
        by_seg[(e["seg"], e["fighter"])].append(e)
    seqs = []
    for seg, info in segs.items():
        z = np.load(Path(a.out_dir) / seg / f"kp_{a.det}.npz")
        for j, fighter in enumerate("ab"):
            x = features(z["kp"], z["score"], j)
            kind, tech = targets(len(x), by_seg[(seg, fighter)])
            seqs.append(dict(seg=seg, fighter=fighter, video=info["source"], x=x, kind=kind, tech=tech))
    videos = sorted({s["video"] for s in seqs})
    rng = np.random.default_rng(a.seed)
    preds = []
    for held in videos:
        tr = [s for s in seqs if s["video"] != held]
        if a.frac < 1.0:   # learning curve: keep a share of the training segments
            keep = set(rng.choice(sorted({s["seg"] for s in tr}), max(1, round(a.frac * len({s["seg"] for s in tr}))), replace=False))
            tr = [s for s in tr if s["seg"] in keep]
        net = train(tr, a.seed)
        for s in (s for s in seqs if s["video"] == held):
            for e in predict(net, s["x"], thr=0.3):
                preds.append(dict(e, seg=s["seg"], fighter=s["fighter"]))
    res = {}
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p = [e for e in preds if e["p"] >= thr]
        hits, tech = match(p, events)
        res[str(thr)] = dict(predicted=len(p), labelled=len(events), hits=hits, recall=round(hits / len(events), 3),
                             precision=round(hits / max(len(p), 1), 3), technique_right=tech,
                             technique_acc=round(tech / max(hits, 1), 3))
    per_kind = Counter(e["kind"] for e in events)
    json.dump(dict(frac=a.frac, seed=a.seed, det=a.det, events=len(events), per_kind=per_kind, sweep=res, preds=preds),
              open(a.result, "w"))
    print(json.dumps(dict(frac=a.frac, seed=a.seed, **{k: (v["recall"], v["precision"], v["technique_acc"]) for k, v in res.items()})))


if __name__ == "__main__":
    main()
