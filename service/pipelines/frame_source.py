"""Frame sampling + role-labeled detect + semantic top-2 crop (build plan §4).

Reuses (imports, never copies) ``qa_vlm._resolve_stream_url``/``_run_detect``/
``_detect_schema`` and ``utils.get_union_box``/``get_padded_square_box``. The
multi-frame ffmpeg extraction below is NEW code that sits alongside (not inside)
``qa_vlm._grab_frame_via_ffmpeg`` — the existing single-frame path is untouched.

No fabrication: a frame with fewer than 2 detected persons is DROPPED (never a
prior-box carry-forward, never an invented box) — exactly production's
``if not athletes: continue`` (``pipeline.py:process_video`` / ``upscale_setup.py``).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from utils import get_padded_square_box, get_union_box

logger = logging.getLogger("service.pipelines.frame_source")

_ROLE_TOKENS = ("athlete", "referee", "coach", "bystander")

# Role-labeled detect prompt (build plan §4.2) — literal text, do not paraphrase;
# the "athlete:"/"referee:"/"coach:"/"bystander:" prefix contract is what makes
# `parse_role` deterministic.
MIMIC_DETECT_PROMPT = """Detect every person visible in this frame of a Brazilian Jiu-Jitsu (BJJ) / grappling match.

For EACH person, return one entry with:
- "box_2d": [ymin, xmin, ymax, xmax] as integers normalized 0-1000 (the full visible extent of that person's body).
- "label": begin with EXACTLY ONE role token, then a short visual descriptor. Roles:
    "athlete: ..."   -> one of the TWO people actively grappling/competing on the mat
                        (in a stance, in a clinch, or entangled on the ground with the other athlete).
    "referee: ..."   -> the official (often standing over or circling the two athletes, distinct
                        uniform/stripe, not grappling).
    "coach: ..."     -> someone at the mat edge instructing, not grappling.
    "bystander: ..." -> spectators, other athletes waiting, camera/staff, anyone off the action.
  Example labels: "athlete: white gi, top position", "athlete: blue gi, bottom", "referee: black shirt".
- "confidence": 0.0-1.0.

Rules:
- The two competing athletes are the people physically engaged with EACH OTHER. When they are
  entangled and hard to separate, still return a box for each one you can distinguish.
- Do NOT merge two athletes into one box, and do NOT invent a person who is not clearly visible.
- Distinguish the referee from the athletes by engagement: the referee watches and moves around
  the pair but is not grappling. Coaches and spectators are outside the active grappling exchange.
- Prefer distinct gi color / top-vs-bottom / standing-vs-ground descriptors so the two athletes
  are told apart.

Return ONLY the JSON array matching the schema. No prose."""


# --------------------------------------------------------------------------- #
# Sampling (ffmpeg multi-frame extraction)
# --------------------------------------------------------------------------- #
@dataclass
class SampledFrame:
    frame_index: int  # REAL source frame number (round(sample_time * native_fps))
    image: Image.Image


def probe_native_fps(stream_url: str, headers: dict) -> float:
    """ffprobe the resolved stream's video frame rate. Raises RuntimeError (never
    a guessed default) on failure — a wrong assumed fps corrupts every downstream
    frame_index."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0"]
    if headers:
        header_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        cmd += ["-headers", header_str]
    cmd += [stream_url]
    proc = subprocess.run(cmd, capture_output=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffprobe fps probe failed (rc={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='replace')[-1000:]}"
        )
    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    # ffprobe can emit more than one line (e.g. a duplicated r_frame_rate);
    # take the first non-empty line so a multi-line payload like "25/1\n\n25/1"
    # doesn't blow up on split(). Never guess a default — an empty payload raises.
    first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    try:
        if not first:
            raise ValueError("empty frame-rate payload")
        if "/" in first:
            num, den = first.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(first)
    except (ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"ffprobe returned unparseable frame rate {raw!r}: {e}") from e
    if fps <= 0:
        raise RuntimeError(f"ffprobe returned non-positive frame rate {raw!r}")
    return fps


_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def _split_mjpeg_stream(data: bytes) -> list[bytes]:
    """Split a concatenated MJPEG byte stream into individual JPEG frames by
    SOI/EOI marker scanning. Never fabricates a frame boundary — a trailing
    partial frame (no EOI) is dropped, not padded/guessed."""
    frames: list[bytes] = []
    i = 0
    while True:
        start = data.find(_JPEG_SOI, i)
        if start == -1:
            break
        end = data.find(_JPEG_EOI, start + 2)
        if end == -1:
            break
        frames.append(data[start:end + 2])
        i = end + 2
    return frames


def sample_frames(
    stream_url: str,
    headers: dict,
    start_sec: int,
    end_sec: int,
    native_fps: float,
    sample_fps: float,
) -> list[SampledFrame]:
    """Extract frames at ``sample_fps`` over ``[start_sec, end_sec)`` via a SINGLE
    ffmpeg call (fps filter -> MJPEG pipe), one call regardless of window length
    (plan §4.1: "a single ffmpeg call with fps filter is fine").

    ``frame_index`` for the k-th extracted frame is the REAL source frame number
    ``round((start_sec + k / sample_fps) * native_fps)`` — NOT re-indexed — so
    downstream frame->wall-clock conversion stays correct (plan §4.1).

    Raises RuntimeError with ffmpeg's real stderr on failure — no placeholder
    frames (mirrors ``qa_vlm._grab_frame_via_ffmpeg``'s no-fallback contract).
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"sample_frames: end_sec ({end_sec}) must be > start_sec ({start_sec})")

    cmd = ["ffmpeg", "-y"]
    if headers:
        header_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        cmd += ["-headers", header_str]
    cmd += [
        "-ss", str(max(0, start_sec)),
        "-i", stream_url,
        "-t", str(duration),
        "-vf", f"fps={sample_fps}",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "2",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg multi-frame extraction failed (rc={proc.returncode}): {stderr_tail}")

    jpeg_blobs = _split_mjpeg_stream(proc.stdout)
    sampled: list[SampledFrame] = []
    for k, blob in enumerate(jpeg_blobs):
        sample_time = start_sec + (k / sample_fps)
        frame_index = round(sample_time * native_fps)
        import io
        img = Image.open(io.BytesIO(blob))
        img.load()
        sampled.append(SampledFrame(frame_index=frame_index, image=img.convert("RGB")))
    return sampled


# --------------------------------------------------------------------------- #
# Detection -> role parsing
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    box_px: list[float]  # [x1, y1, x2, y2] pixel coords
    role: str  # "athlete" | "referee" | "coach" | "bystander" | "unlabeled"
    label: str
    confidence: float


def parse_role(label: str) -> str:
    prefix = label.split(":", 1)[0].strip().lower()
    return prefix if prefix in _ROLE_TOKENS else "unlabeled"


def to_detections(raw_objects: list[dict], img_w: int, img_h: int) -> list[Detection]:
    """Convert `_run_detect`'s parsed entries (``box_2d`` 0-1000 normalized
    [ymin,xmin,ymax,xmax]) into pixel-space ``Detection``s. Entries whose role
    prefix is unrecognized are kept as ``"unlabeled"`` (excluded from every
    ranking pool downstream, never silently coerced into a role)."""
    out = []
    for obj in raw_objects:
        ymin, xmin, ymax, xmax = obj["box_2d"]
        box_px = [xmin / 1000 * img_w, ymin / 1000 * img_h, xmax / 1000 * img_w, ymax / 1000 * img_h]
        out.append(Detection(
            box_px=box_px,
            role=parse_role(obj["label"]),
            label=obj["label"],
            confidence=float(obj.get("confidence", 0.0)),
        ))
    return out


# --------------------------------------------------------------------------- #
# Top-2 selection (semantic-first, contact-tiebreak, then geometry) — plan §4.3
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    candidates: Optional[list[Detection]]  # the chosen pair (len 2), or None if dropped
    degraded: Optional[str]  # None | "single_athlete" | "role_uncertain" | "no_detection"

    @property
    def boxes_px(self) -> Optional[list[list[float]]]:
        return [d.box_px for d in self.candidates] if self.candidates else None


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contact(a: list[float], b: list[float]) -> float:
    """Contact/overlap proxy for the pair tiebreak (plan §4.3 step 2) — plain IoU.

    Deliberately NOT "intersection over smaller box's area": that variant scores
    a small box fully CONTAINED inside a much larger one as maximal contact,
    which is the wrong signal here (two grappling athletes have roughly
    comparable on-screen scale; a huge box containing a tiny one is far more
    likely a bad/oversized single detection than "two people touching"). Plain
    IoU naturally penalizes that size mismatch via a large union, while still
    being 0 for boxes that don't touch at all.
    """
    return _iou(a, b)


def _score(d: Detection, img_w: int, img_h: int) -> float:
    """``score = area_norm * (0.5 + 0.5*centrality)`` (plan §4.3 step 2)."""
    area_norm = _area(d) / (img_w * img_h) if img_w and img_h else 0.0
    return area_norm * (0.5 + 0.5 * _centrality(d, img_w, img_h))


def _area(d: Detection) -> float:
    x1, y1, x2, y2 = d.box_px
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _centrality(d: Detection, img_w: int, img_h: int) -> float:
    x1, y1, x2, y2 = d.box_px
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    fcx, fcy = img_w / 2, img_h / 2
    max_dist = ((img_w / 2) ** 2 + (img_h / 2) ** 2) ** 0.5
    if max_dist <= 0:
        return 1.0
    dist = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
    return 1.0 - min(1.0, dist / max_dist)


def _deterministic_sort_key(d: Detection, img_w: int, img_h: int):
    """Final tie-break for ordering the 2 SELECTED elements: area desc -> centrality
    desc -> xmin asc (plan §4.3 step 2). Smaller tuple = ranked first."""
    return (-_area(d), -_centrality(d, img_w, img_h), d.box_px[0])


def _best_pair(cands: list[Detection], img_w: int, img_h: int) -> list[Detection]:
    """Enumerate all pairs; prefer max mutual contact, tie-break by combined
    score, then a fully deterministic geometric tie-break (total area desc ->
    total centrality desc -> smaller member's xmin asc), so the winning pair is
    reproducible across runs with identical inputs. Returned pair is ordered via
    ``_deterministic_sort_key`` (area->centrality->xmin) for stable output order.
    """
    best_key = None
    best_pair = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            contact = _contact(a.box_px, b.box_px)
            combined = _score(a, img_w, img_h) + _score(b, img_w, img_h)
            total_area = _area(a) + _area(b)
            total_centrality = _centrality(a, img_w, img_h) + _centrality(b, img_w, img_h)
            min_xmin = min(a.box_px[0], b.box_px[0])
            # All terms "higher is better"; -min_xmin so a SMALLER xmin wins.
            key = (contact, combined, total_area, total_centrality, -min_xmin)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = (a, b)
    a, b = best_pair
    return sorted([a, b], key=lambda d: _deterministic_sort_key(d, img_w, img_h))


def select_top2(detections: list[Detection], img_w: int, img_h: int) -> SelectionResult:
    """Plan §4.3 — semantic-first athlete selection with a 4-branch degrade ladder.
    NEVER fabricates a box: <2 usable persons -> drop (``no_detection``)."""
    athletes = [d for d in detections if d.role == "athlete"]
    persons = [d for d in detections if d.role in _ROLE_TOKENS]

    if len(athletes) >= 2:
        return SelectionResult(candidates=_best_pair(athletes, img_w, img_h), degraded=None)
    if len(athletes) == 1:
        return SelectionResult(candidates=[athletes[0]], degraded="single_athlete")
    if len(persons) >= 2:
        return SelectionResult(candidates=_best_pair(persons, img_w, img_h), degraded="role_uncertain")
    return SelectionResult(candidates=None, degraded="no_detection")


# --------------------------------------------------------------------------- #
# Crop geometry — union + padded square + min-crop-size floor (plan §4.4)
# --------------------------------------------------------------------------- #
def compute_crop_box(
    boxes_px: list[list[float]],
    img_shape: tuple[int, int],
    padding: float,
    min_crop_fraction: float,
) -> list[int]:
    """``union -> padded square -> min-crop-size floor``. The floor NEVER tightens
    an already-larger crop, only expands ones that would zoom in past the floor
    (plan §4.4 step 3: "cap zoom-in, always allow zoom-out")."""
    union = get_union_box(boxes_px)
    square = get_padded_square_box(union, padding=padding, img_shape=img_shape)
    x1, y1, x2, y2 = square
    side = x2 - x1
    img_h, img_w = img_shape
    min_side = min_crop_fraction * min(img_h, img_w)
    if side >= min_side:
        return square
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hs = min_side / 2
    floored_union = [cx - hs, cy - hs, cx + hs, cy + hs]
    return get_padded_square_box(floored_union, padding=0.0, img_shape=img_shape)


@dataclass
class CropFrame:
    """One surviving (never-dropped) frame ready to enter the sliding window —
    either a real detect+crop result, or (when ``detect_crop`` is disabled) an
    identity pass-through of the raw sampled frame."""
    frame_index: int
    image: Image.Image
    degraded: Optional[str] = None


def resize_long_edge(image: Image.Image, target_size: int) -> Image.Image:
    """Resize so the longer edge == ``target_size`` (identity pass-through path
    when ``detect_crop`` is disabled — still bounds token cost without cropping)."""
    w, h = image.size
    if max(w, h) <= 0:
        return image
    scale = target_size / max(w, h)
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def build_sliding_windows(crops: list, window_size: int, stride: int) -> list[list]:
    """Mirror ``pipeline.py:process_video``'s sliding-window accumulation exactly:
    grow a buffer frame-by-frame, emit a full window once it reaches
    ``window_size``, shift by ``stride``, and emit one trailing partial window
    at the end if anything is left over. Generic over any list (used both for
    real ``CropFrame`` lists and for budget-guard frame-count estimation).
    """
    windows: list[list] = []
    buf: list = []
    for item in crops:
        buf.append(item)
        if len(buf) >= window_size:
            windows.append(buf[:window_size])
            buf = buf[stride:]
    if buf:
        windows.append(buf)
    return windows


def count_sliding_windows(n_frames: int, window_size: int, stride: int) -> int:
    """Budget-guard helper: how many windows ``build_sliding_windows`` would
    produce for ``n_frames`` frames, WITHOUT building real frame objects."""
    if n_frames <= 0:
        return 0
    return len(build_sliding_windows(list(range(n_frames)), window_size, stride))


def crop_and_resize(image: Image.Image, square_box: list[int], target_size: int = 768) -> Image.Image:
    """Crop + LANCZOS resize to ``target_size x target_size``. No ESRGAN.

    Frames here originate from ffmpeg/PIL (already RGB) rather than production's
    cv2.VideoCapture (BGR) reads, so the BGR->RGB conversion step in the plan's
    §4.4 step 4 is a no-op in this code path — deliberate, documented deviation
    (no cv2 array round-trip needed since PIL never touched BGR to begin with).
    """
    x1, y1, x2, y2 = square_box
    cropped = image.crop((x1, y1, x2, y2))
    return cropped.resize((target_size, target_size), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Identity stability — IoU re-association + hysteresis + EMA (plan §4.4 step 5)
# --------------------------------------------------------------------------- #
_CONTINUITY_IOU_THRESHOLD = 0.3


def _pair_continuity(new_boxes: list[list[float]], held_boxes: list[list[float]]) -> float:
    """Best-of assignment IoU between this frame's candidate boxes and the held
    boxes (order-agnostic — Gemini gives no persistent identity token).

    Robust to 0/1/2-length inputs: a real 2-vs-2 assignment IoU only makes
    sense when BOTH sides have exactly 2 boxes (the normal, non-degraded
    case). ``AthleteTracker.resolve`` guarantees ``held_boxes`` is always
    ``None`` or length-2 (a degraded <2-box selection never overwrites it —
    see ``resolve``'s early-return), so in practice this never sees a
    mismatched pair via that caller. This function stays defensive anyway
    (never indexes past either list's actual length) so it can be called
    directly — including from tests — with 0- or 1-length inputs without an
    ``IndexError``: a mismatched or degenerate pairing has no real 2-vs-2
    assignment to score, so it's treated as the best available single-box
    IoU (0.0 if either side is empty)."""
    if not new_boxes or not held_boxes:
        return 0.0
    if len(new_boxes) == 2 and len(held_boxes) == 2:
        direct = (_iou(new_boxes[0], held_boxes[0]) + _iou(new_boxes[1], held_boxes[1])) / 2
        crossed = (_iou(new_boxes[0], held_boxes[1]) + _iou(new_boxes[1], held_boxes[0])) / 2
        return max(direct, crossed)
    return max(_iou(nb, hb) for nb in new_boxes for hb in held_boxes)


def _match_box_to_detections(box: list[float], detections: list[Detection]) -> Optional[Detection]:
    """Find the detection in this frame's FULL list that best matches ``box`` by
    IoU (used to re-score a held pair that select_top2 didn't pick this frame)."""
    best = None
    best_iou = 0.0
    for d in detections:
        i = _iou(box, d.box_px)
        if i > best_iou:
            best_iou = i
            best = d
    return best if best_iou > 0 else None


@dataclass
class AthleteTracker:
    """Holds a stable 2-athlete pair across sampled frames with hysteresis, and
    EMA-smooths the resulting crop square's (cx, cy, side) to kill jitter
    (plan §4.4 step 5). One instance per run (per video scope).
    """
    alpha: float = 0.6
    margin: float = 0.15
    min_streak: int = 3

    held_boxes: Optional[list[list[float]]] = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    _smoothed: Optional[tuple[float, float, float]] = field(default=None, init=False)

    def resolve(
        self, selection: SelectionResult, all_detections: list[Detection], img_w: int, img_h: int,
    ) -> SelectionResult:
        """Apply hysteresis to ``selection`` (this frame's independently-best pair)
        against the held pair. Returns a (possibly overridden) ``SelectionResult``
        with the boxes to actually crop this frame. Frames with no candidates
        (``no_detection``) pass through untouched — the tracker holds no state
        across a dropped frame's absence.

        Frames with a DEGRADED <2-box selection (``single_athlete`` — the only
        branch of ``select_top2`` that returns exactly 1 candidate; a
        ``role_uncertain`` selection is always length-2) also pass through
        untouched, same as ``no_detection``: the crop for THIS frame still
        uses the real (never fabricated) single box ``select_top2`` found, but
        the tracker's own ``held_boxes`` pair-continuity anchor is left alone.
        This is the fix for the IndexError regression (a dropped-to-1-box
        frame used to overwrite ``held_boxes`` with a length-1 list, which
        then crashed ``_pair_continuity``'s ``[0]``/``[1]`` indexing on the
        NEXT full 2-athlete frame) — ``held_boxes`` is now a documented
        invariant: always ``None`` or exactly length 2, so a later full-pair
        frame can still re-anchor against the last known-good pair instead of
        losing tracking over one occluded/partial-detection frame. Nothing is
        fabricated either way; only the internal bookkeeping is preserved."""
        if selection.candidates is None:
            return selection

        new_boxes = selection.boxes_px
        if len(new_boxes) != 2:
            return selection

        if self.held_boxes is None:
            self.held_boxes = new_boxes
            self._streak = 0
            return selection

        if _pair_continuity(new_boxes, self.held_boxes) >= _CONTINUITY_IOU_THRESHOLD:
            self.held_boxes = new_boxes
            self._streak = 0
            return selection

        # Challenger scenario: re-score the (still-)held pair against this frame's
        # FULL detection set (not just the top-2 pick) before deciding to swap.
        held_matches = [_match_box_to_detections(b, all_detections) for b in self.held_boxes]
        if any(m is None for m in held_matches):
            # At least one held athlete is not present in this frame's detections
            # at all — nothing faithful to keep holding, accept the new pair
            # immediately (no hysteresis possible over a partially-vanished pair).
            self.held_boxes = new_boxes
            self._streak = 0
            return selection

        held_score = sum(_score(m, img_w, img_h) for m in held_matches)
        new_score = sum(_score(c, img_w, img_h) for c in selection.candidates)

        if new_score > held_score * (1 + self.margin):
            self._streak += 1
            if self._streak >= self.min_streak:
                self.held_boxes = new_boxes
                self._streak = 0
                return selection
            # Challenger hasn't sustained its lead long enough yet — keep holding
            # the SAME identities (re-scored positions), NOT the challenger's pick.
            self.held_boxes = [m.box_px for m in held_matches]
            return SelectionResult(candidates=held_matches, degraded=selection.degraded or "hysteresis_hold")
        self._streak = 0
        self.held_boxes = [m.box_px for m in held_matches]
        return SelectionResult(candidates=held_matches, degraded=selection.degraded)

    def smoothed_crop_box(self, boxes_px: list[list[float]], img_shape: tuple[int, int], padding: float,
                           min_crop_fraction: float) -> list[int]:
        """EMA-smooth (cx, cy, side) of the union+padded-square crop across calls."""
        square = compute_crop_box(boxes_px, img_shape, padding, min_crop_fraction)
        x1, y1, x2, y2 = square
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1)
        if self._smoothed is None:
            self._smoothed = (cx, cy, side)
        else:
            pcx, pcy, pside = self._smoothed
            a = self.alpha
            self._smoothed = (a * cx + (1 - a) * pcx, a * cy + (1 - a) * pcy, a * side + (1 - a) * pside)
        scx, scy, sside = self._smoothed
        img_h, img_w = img_shape
        hs = sside / 2
        floored = get_padded_square_box([scx - hs, scy - hs, scx + hs, scy + hs], padding=0.0, img_shape=img_shape)
        return floored
