"""Identity answer key: numbered person boxes on sampled frames for annotators, and the scorer that uses their answers.

sheets DATA OUT SEG...   every 60th frame of each segment with the union of the SAM 3 and RTMO person boxes numbered
                         (OUT/<seg>_<frame>.jpg, OUT/candidates.json)
score DATA GT DET SEG... scores out/<seg>/identity_<DET>.json against GT (answers.jsonl rows: seg, frame, a, b,
                         a_state, b_state, others). A slot is right when its box overlaps the chosen box (IoU >= 0.5).
                         DET "v1" reads the first-round identity.json.
cards DATA GT DET OUT    pictures of answer-key frames the tracker got wrong (OUT/<seg>_<frame>.jpg, OUT/cards.json)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

STEP, TOP, MIN_H = 60, 7, 0.14
COLOURS = [(60, 220, 255), (255, 160, 60), (120, 255, 120), (255, 110, 255), (255, 255, 90), (90, 150, 255),
           (200, 200, 200)]


def read_mot(path):
    boxes = defaultdict(list)
    for line in open(path):
        f, _, x, y, w, h = line.split(",")[:6]
        boxes[int(f) - 1].append([float(x), float(y), float(w), float(h)])
    return boxes


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-6)


def sheets(data, out, segs):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cands = {}
    for seg in segs:
        mots = [read_mot(data / "out" / seg / f"masks_{d}.mot.txt") for d in ("light", "rtmoz")]
        cap = cv2.VideoCapture(str(data / "segs" / f"{seg}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(STEP // 2, n, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            boxes = []
            for mot in mots:
                for b in sorted(mot.get(f, []), key=lambda b: -b[2] * b[3]):
                    if b[3] >= MIN_H * img.shape[0] and all(iou(b, k) < 0.7 for k in boxes):
                        boxes.append(b)
            boxes = sorted(boxes, key=lambda b: -b[2] * b[3])[:TOP]
            boxes = sorted(boxes, key=lambda b: b[0])
            for i, (x, y, w, h) in enumerate(boxes):
                c = COLOURS[i]
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
                lx, ly = int(x) + 2 + 34 * (i % 2), max(int(y) + 2, 0)
                cv2.rectangle(img, (lx, ly), (lx + 30, ly + 32), c, -1)
                cv2.putText(img, str(i + 1), (lx + 5, ly + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3)
            cv2.imwrite(str(out / f"{seg}_{f:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            cands[f"{seg}_{f:04d}"] = boxes
    json.dump(cands, open(out / "candidates.json", "w"))
    print(len(cands), "frames")


def slot_kinds(pred, r, boxes):
    """(slot, verdict) for fighters A and B on one answer-key frame: right, missing, swapped, third_person, off
    for a boxed fighter; gt_shared / gt_unboxed when the key has no box of his own; false_box / absent_ok when he
    is not in the frame."""
    gt = {s: boxes[r[s] - 1] if r.get(s) else None for s in "ab"}
    others = [boxes[i - 1] for i in r.get("others", [])]
    out = []
    for s, o in (("a", "b"), ("b", "a")):
        state = r[f"{s}_state"]
        p = pred.get(s, {}).get("box")
        if state == "boxed":
            if p is None:
                kind = "missing"
            elif iou(p, gt[s]) >= 0.5:
                kind = "right"
            elif gt[o] is not None and r[f"{o}_state"] == "boxed" and iou(p, gt[o]) >= 0.5:
                kind = "swapped"
            elif any(iou(p, k) >= 0.5 for k in others):
                kind = "third_person"
            else:
                kind = "off"
        elif state in ("shared", "unboxed"):
            kind = f"gt_{state}"
        else:
            kind = "false_box" if p is not None else "absent_ok"
        out.append((s, kind))
    return out


def score(data, gt_dir, det, segs):
    data, gt_dir = Path(data), Path(gt_dir)
    cands = json.load(open(gt_dir / "candidates.json"))
    rows = [json.loads(line) for line in open(gt_dir / "answers.jsonl")]
    out, total = {}, Counter()
    for seg in segs:
        name = "identity.json" if det == "v1" else f"identity_{det}.json"
        frames = json.load(open(data / "out" / seg / name))["frames"]
        c = Counter()
        for r in (r for r in rows if r["seg"] == seg):
            kinds = [k for _, k in slot_kinds(frames.get(str(r["frame"]), {}), r, cands[f"{seg}_{r['frame']:04d}"])]
            c.update(kinds)
            c["frames"] += 1
            c["frame_ok"] += all(k in ("right", "absent_ok") for k in kinds)
        out[seg] = c
        total.update(c)
    out["ALL"] = total
    return {seg: dict(frames=c["frames"], frame_ok=c["frame_ok"],
                      boxed=sum(c[k] for k in ("right", "missing", "swapped", "third_person", "off")), right=c["right"],
                      missing=c["missing"], swapped=c["swapped"], third_person=c["third_person"], off=c["off"],
                      false_box=c["false_box"], absent=c["false_box"] + c["absent_ok"], gt_shared=c["gt_shared"],
                      gt_unboxed=c["gt_unboxed"]) for seg, c in out.items()}


def cards(data, gt_dir, det, out, per_kind=4):
    """Bad-case pictures: answer-key frames the tracker got wrong, with its A (red) and B (blue) boxes drawn."""
    data, gt_dir, out = Path(data), Path(gt_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cands = json.load(open(gt_dir / "candidates.json"))
    rows = [json.loads(line) for line in open(gt_dir / "answers.jsonl")]
    picked, count, ident = [], Counter(), {}
    for r in sorted(rows, key=lambda r: (r["frame"] * 7919) % 1500):   # spread over segments and time
        seg = r["seg"]
        if seg not in ident:
            ident[seg] = json.load(open(data / "out" / seg / f"identity_{det}.json"))["frames"]
        pred = ident[seg].get(str(r["frame"]), {})
        kinds = slot_kinds(pred, r, cands[f"{seg}_{r['frame']:04d}"])
        bad = [k for _, k in kinds if k not in ("right", "absent_ok")]
        if not bad or count[bad[0]] >= per_kind:
            continue
        count[bad[0]] += 1
        cap = cv2.VideoCapture(str(data / "segs" / f"{seg}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, img = cap.read()
        for s, col in (("a", (40, 40, 230)), ("b", (230, 120, 30))):
            if s in pred:
                x, y, w, h = map(int, pred[s]["box"])
                cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
                cv2.putText(img, s.upper(), (x + 6, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        name = f"{seg}_{r['frame']:04d}.jpg"
        cv2.imwrite(str(out / name), cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
        picked.append(dict(img=name, seg=seg, frame=r["frame"], verdicts=dict(kinds), note=r.get("note", ""),
                           flags=pred.get("flags", [])))
    json.dump(picked, open(out / "cards.json", "w"), indent=1)
    print(len(picked), "cards", dict(count))


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        sheets(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == "cards":
        cards(*sys.argv[2:6])
    else:
        res = score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
        for seg, r in res.items():
            print(f"{seg:6s} frames_ok {r['frame_ok']:3d}/{r['frames']:3d}  right {r['right']:3d}/{r['boxed']:3d}  "
                  f"missing {r['missing']:2d} swapped {r['swapped']:2d} third {r['third_person']:2d} off {r['off']:2d} "
                  f"false_box {r['false_box']:2d}/{r['absent']:2d}  shared {r['gt_shared']:2d} unboxed {r['gt_unboxed']:2d}")
