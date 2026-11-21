"""Stage 3: RTMW-x whole-body keypoints (133, COCO-WholeBody order) for fighter A and B on every frame.

Top-down on each fighter's identity box (TensorRT FP16, both fighters in one batch). A keypoint that lies inside
the opponent's mask and outside its owner's mask is flagged cross-mask: attribution evidence, kept, not hidden.

With --mask-opponent, the opponent's pixels (inside their mask, outside the fighter's own) are painted grey
before the crop, so the model cannot place the fighter's joints on the other body.

Usage: keypoints.py SEG_DIR ENGINE [--mask-opponent]   (reads identity.json, masks_ab.npz; writes kp_rtmw.npz or
kp_rtmw_masked.npz)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

SIZE = (288, 384)    # model input w, h
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])


class Engine:
    def __init__(self, path):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, batch):   # batch: [N, 3, 384, 288] float32
        n = batch.shape[0]
        x = torch.from_numpy(batch).cuda().contiguous()
        self.ctx.set_input_shape("input", tuple(x.shape))
        sx = torch.empty((n, 133, 576), device="cuda")
        sy = torch.empty((n, 133, 768), device="cuda")
        for name, t in (("input", x), ("simcc_x", sx), ("simcc_y", sy)):
            self.ctx.set_tensor_address(name, t.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return sx.cpu().numpy(), sy.cpu().numpy()


def prep(img, box_xywh):
    x, y, w, h = box_xywh
    center, scale = bbox_xyxy2cs(np.array([x, y, x + w, y + h]), padding=1.25)
    crop, scale = top_down_affine(SIZE, scale, center, img)
    return ((crop - MEAN) / STD).transpose(2, 0, 1).astype(np.float32), center, scale


def main(seg_dir, engine_path, *flags):
    masked = "--mask-opponent" in flags
    seg = Path(seg_dir)
    ident = json.load(open(seg / "identity.json"))["frames"]
    packed = np.load(seg / "masks_ab.npz")["packed"]
    eng = Engine(engine_path)
    cap = cv2.VideoCapture(str(seg.parent.parent / "segs" / f"{seg.name}.mp4"))
    n = len(ident)
    kp = np.full((n, 2, 133, 2), np.nan, np.float32)
    sc = np.zeros((n, 2, 133), np.float32)
    cross = np.zeros((n, 2, 133), bool)
    f = 0
    while f < n:
        ok, img = cap.read()
        if not ok:
            break
        fr = ident[str(f)]
        slots = [j for j, s in enumerate(("a", "b")) if s in fr]
        if slots:
            H, W = img.shape[:2]
            masks = [np.unpackbits(packed[f, j]).reshape(288, 288).astype(bool) for j in range(2)]
            srcs = [img] * len(slots)
            if masked:
                srcs = []
                for j in slots:
                    other = cv2.resize((masks[1 - j] & ~masks[j]).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    im = img.copy(); im[other] = 128
                    srcs.append(im)
            items = [prep(srcs[i], fr[("a", "b")[j]]["box"]) for i, j in enumerate(slots)]
            sx, sy = eng(np.stack([it[0] for it in items]))
            locs, scores = get_simcc_maximum(sx, sy)
            for i, j in enumerate(slots):
                _, center, scale = items[i]
                pts = locs[i] / 2.0 / np.array(SIZE) * scale + center - scale / 2
                kp[f, j], sc[f, j] = pts, scores[i]
                mx = np.clip((pts[:, 0] / W * 288).astype(int), 0, 287)
                my = np.clip((pts[:, 1] / H * 288).astype(int), 0, 287)
                own, other = masks[j][my, mx], masks[1 - j][my, mx]
                cross[f, j] = other & ~own
        f += 1
        if f % 500 == 0:
            print(f, flush=True)
    np.savez_compressed(seg / ("kp_rtmw_masked.npz" if masked else "kp_rtmw.npz"), kp=kp, score=sc, cross=cross)
    limbs = [9, 10, 7, 8, 15, 16, 13, 14]
    have = ~np.isnan(kp[:, :, 0, 0])
    print(json.dumps(dict(frames=n, fighter_frames=int(have.sum()),
                          limb_conf_mean=round(float(sc[have][:, limbs].mean()), 3),
                          limb_cross_rate=round(float(cross[have][:, limbs].mean()), 4))))


if __name__ == "__main__":
    main(*sys.argv[1:])
