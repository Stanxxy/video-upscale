"""Video student: frozen V-JEPA 2 (ViT-L) embeddings of fighter-centred clips plus a linear probe.

embed  WINDOWS_JSON OUT_NPZ DET   one clip per window and fighter: the crop follows that fighter's boxes over the
                                  window (padded, square), 32 frames, mean-pooled last hidden state (1024-d). The
                                  fighter in the centre is the subject, so the probe can say who attacked.
probe  WINDOWS_JSON EMB_NPZ SEGS_JSON EVENTS_JSON RESULT_JSON [--frac F] [--seed S]
                                  leave-one-video-out logistic probe: which attack kinds and techniques the centred
                                  fighter starts in the window. Result rows: window, fighter, kinds{}, techs{}.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from student import KINDS, TECHS

K = Path.home() / "bjj/x-mmagold/kp"
MODEL, FRAMES = "facebook/vjepa2-vitl-fpc64-256", 32


def embed(windows, out, det):
    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    wins = json.load(open(windows))
    done, cache = {}, {}
    for n, w in enumerate(wins):
        seg = w["seg"]
        if seg not in cache:
            cap = cv2.VideoCapture(str(K / "segs" / f"{seg}.mp4"))
            frames = []
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
            cache = {seg: (frames, json.load(open(K / "out" / seg / f"identity_{det}.json"))["frames"])}
        frames, ident = cache[seg]
        f1 = min(w["f1"], len(frames) - 1)
        H, W = frames[0].shape[:2]
        for s in "ab":
            boxes = [ident[str(f)][s]["box"] for f in range(w["f0"], f1 + 1) if s in ident[str(f)]]
            if len(boxes) < 0.3 * (f1 - w["f0"] + 1):
                continue   # the fighter is not tracked in this window: no clip
            b = np.array(boxes)
            x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
            side = max(x2 - x1, y2 - y1) * 1.3
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x0, y0 = int(np.clip(cx - side / 2, 0, max(W - side, 0))), int(np.clip(cy - side / 2, 0, max(H - side, 0)))
            side = int(min(side, W - x0, H - y0))
            idx = np.linspace(w["f0"], f1, FRAMES).round().astype(int)
            clip = np.stack([cv2.cvtColor(cv2.resize(frames[f][y0:y0 + side, x0:x0 + side], (256, 256)), cv2.COLOR_BGR2RGB)
                             for f in idx])
            inputs = proc(clip, return_tensors="pt").to("cuda")
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
            with torch.no_grad():
                h = model.get_vision_features(**inputs) if hasattr(model, "get_vision_features") else model(**inputs).last_hidden_state
            done[f"{w['id']}:{s}"] = h.float().mean(dim=1)[0].cpu().numpy()
        if n % 50 == 0:
            print(n, len(done), flush=True)
    np.savez(out, **done)
    print("embedded", len(done), "clips")


def probe(a):
    wins = json.load(open(a.windows))
    emb = dict(np.load(a.emb))
    segs = json.load(open(a.segs))
    events = defaultdict(list)
    for e in json.load(open(a.events)):
        events[(e["seg"], e["fighter"])].append(e)
    rows = []
    for w in wins:
        for s in "ab":
            key = f"{w['id']}:{s}"
            if key not in emb:
                continue
            mine = [e for e in events[(w["seg"], s)] if w["f0"] <= e["f"] <= w["f1"]]
            y = np.zeros(len(KINDS) - 1 + len(TECHS), np.float32)
            for e in mine:
                if e["kind"] in KINDS:
                    y[KINDS.index(e["kind"]) - 1] = 1
                y[len(KINDS) - 1 + (TECHS.index(e["tech"]) if e["tech"] in TECHS else TECHS.index("other"))] = 1
            rows.append(dict(key=key, window=w["id"], seg=w["seg"], fighter=s, video=segs[w["seg"]]["source"], x=emb[key], y=y))
    X = np.stack([r["x"] for r in rows]); Y = np.stack([r["y"] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    videos = sorted({r["video"] for r in rows})
    rng = np.random.default_rng(a.seed)
    out = []
    for held in videos:
        tr = np.array([r["video"] != held for r in rows])
        if a.frac < 1.0:
            train_segs = sorted({r["seg"] for r in rows if r["video"] != held})
            keep = set(rng.choice(train_segs, max(1, round(a.frac * len(train_segs))), replace=False))
            tr &= np.array([r["seg"] in keep for r in rows])
        te = np.array([r["video"] == held for r in rows])
        torch.manual_seed(a.seed)
        lin = torch.nn.Linear(X.shape[1], Y.shape[1]).cuda()
        opt = torch.optim.AdamW(lin.parameters(), 1e-3, weight_decay=0.3)
        xt, yt = torch.tensor(X[tr]).cuda(), torch.tensor(Y[tr]).cuda()
        pos = yt.mean(0).clamp(1e-3, 1)
        weight = ((1 - pos) / pos).clamp(1, 20)
        for _ in range(400):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xt), yt, pos_weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = torch.sigmoid(lin(torch.tensor(X[te]).cuda())).cpu().numpy()
        for r, q in zip([r for r, t in zip(rows, te) if t], p):
            out.append(dict(window=r["window"], seg=r["seg"], fighter=r["fighter"],
                            kinds={k: round(float(q[i]), 3) for i, k in enumerate(KINDS[1:])},
                            techs={t: round(float(q[len(KINDS) - 1 + i]), 3) for i, t in enumerate(TECHS)},
                            silver=[k for i, k in enumerate(KINDS[1:]) if r["y"][i] > 0]))
    json.dump(out, open(a.result, "w"))
    tp = fp = fn = 0
    for r in out:
        pred = {k for k, v in r["kinds"].items() if v >= 0.5}
        tp += len(pred & set(r["silver"])); fp += len(pred - set(r["silver"])); fn += len(set(r["silver"]) - pred)
    print(json.dumps(dict(frac=a.frac, seed=a.seed, clips=len(out), silver_window_f1=round(2 * tp / max(2 * tp + fp + fn, 1), 3),
                          precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3))))


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(*sys.argv[2:5])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("windows"); ap.add_argument("emb"); ap.add_argument("segs")
        ap.add_argument("events"); ap.add_argument("result")
        ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
        probe(ap.parse_args())
