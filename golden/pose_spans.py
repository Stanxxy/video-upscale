"""COCO-17 keypoints (ViTPose+) for the boxed fighters, only on tracking frames inside sequence spans (runs on gx10).

Boxes are the gap-filled tracker boxes (``render_boxes.filled_boxes``); frames are batched on the GPU.
Usage: pose_spans.py VIDEO_10FPS BOXES_JSON SPANS_JSON OUT_JSON [MODEL]
SPANS_JSON: [[start_ms, end_ms], ...]; frames from start-500 ms to end+500 ms are kept.
"""
import json
import sys

import cv2
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from render_boxes import filled_boxes

TRACK_FPS, PAD_MS, BATCH = 10, 500, 16


def main(video, boxes_path, spans_path, out, model="usyd-community/vitpose-plus-huge"):
    dev = "cuda"
    vp = AutoProcessor.from_pretrained(model)
    vm = VitPoseForPoseEstimation.from_pretrained(model).to(dev).eval()
    boxes = filled_boxes(json.load(open(boxes_path)))
    wanted = set()
    for s, e in json.load(open(spans_path)):
        wanted.update(range(int((s - PAD_MS) / 1000 * TRACK_FPS) + 1, int((e + PAD_MS) / 1000 * TRACK_FPS) + 2))
    cap = cv2.VideoCapture(video)
    res, batch, f = {}, [], 0

    def run(batch):
        imgs = [b[1] for b in batch]
        bbs = [b[2] for b in batch]
        inp = vp(imgs, boxes=bbs, return_tensors="pt").to(dev)
        inp["dataset_index"] = torch.zeros(sum(len(b) for b in bbs), dtype=torch.long, device=dev)
        with torch.no_grad():
            o = vm(**inp)
        kps = vp.post_process_pose_estimation(o, boxes=bbs)
        for (frame, _, _, labs), kp in zip(batch, kps):
            res[frame] = {l: dict(xy=k["keypoints"].cpu().numpy().round(1).tolist(), s=k["scores"].cpu().numpy().round(3).tolist())
                          for l, k in zip(labs, kp)}

    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        d = boxes.get(f)
        if f not in wanted or not d:
            continue
        labs = sorted(d)
        batch.append((f, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [d[l] for l in labs], labs))
        if len(batch) == BATCH:
            run(batch)
            batch = []
    if batch:
        run(batch)
    json.dump(res, open(out, "w"))
    print("done", len(res), "of", len(wanted), "wanted frames")


if __name__ == "__main__":
    main(*sys.argv[1:6])
