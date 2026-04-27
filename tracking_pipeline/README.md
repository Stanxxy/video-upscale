# test_tracking - BJJ Athlete Detection & Tracking

Hybrid tracking pipeline for BJJ athletes: RF-DETR detection + SAM 2.1 mask propagation + DINOv2 re-ID + online MLP classifiers + state machine.

Ported from [bjj-pose-estimation](https://github.com/Stanxy/bjj-pose-estimation), adapted for Mac M4 Max (MPS/CPU).

## Pipeline

```
RF-DETR detect persons on initial frame
  → User verifies/corrects boxes (cv2 UI)
    → SAM2 mask propagation + periodic RF-DETR re-detection
      → Online MLP classifiers (DINOv2 features) for identity tracking
        → State machine handles scrambles, scene cuts, fades
          → DINOv2 + color histogram re-ID on identity loss
            → Annotated video (masks + boxes) + tracking JSON
```

## Setup

```bash
# From project root, activate existing venv
source venv/bin/activate

# Install dependencies
SAM2_BUILD_CUDA=0 pip install sam2    # Native SAM2 (no CUDA extensions)
pip install rtmlib                     # RTMPose for keypoints
pip install scipy                      # Hungarian assignment
pip install -r test_tracking/requirements.txt
```

## Usage

From the **whole-video-analysis** repo root (venv activated, cwd on `PYTHONPATH` or `pip install -e .`):

```bash
# Track a time range with human verification (default)
python -m tracking_pipeline --video path/to/video.mp4 --start_time 0:04 --end_time 0:30

# Auto mode: skip verification, use top 2 RF-DETR detections
python -m tracking_pipeline --video path/to/video.mp4 --start_time 0:04 --end_time 0:30 --auto

# Force CPU if MPS has issues
python -m tracking_pipeline --video path/to/video.mp4 --start_time 0:04 --end_time 0:30 --cpu

# Larger SAM2 model for better masks
python -m tracking_pipeline --video path/to/video.mp4 --sam2_model large

# Detection only (YOLO26 on one frame)
python -m tracking_pipeline.detect --video path/to/video.mp4 --frame 0 --threshold 0.5
```

Note: run as a module (`python -m tracking_pipeline…`) so package-relative imports resolve. The legacy `cd … && python pipeline.py` flow is no longer supported.

### Human Verification Mode (default)

RF-DETR detects persons in the first frame. A window shows detections numbered:

- **Press digit keys** (0-9) to select Athlete A, then Athlete B
- **Press 'm'** to manually draw bounding boxes (useful when athletes are intertwined)
- **Press 'q' or ESC** to cancel

If RF-DETR finds 0 persons, falls through to manual bounding box drawing automatically.

### Auto Mode (`--auto`)

Picks the 2 highest-confidence person detections. Falls back to manual if RF-DETR finds nothing.

## Output

All outputs go to `output/`:

| File | Description |
|------|-------------|
| `detections.json` | RF-DETR initial detection results |
| `verified_boxes.json` | User-verified athlete boxes |
| `tracking.json` | Per-frame tracking data (boxes, keypoints, state, IoU) |
| `tracked_output.mp4` | Annotated video with masks + boxes + state overlay |
| `debug_frames/` | Debug frame samples every 120 frames |

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input video |
| `--start_time` | start | Start time as MM:SS (e.g. `1:30`) |
| `--end_time` | end | End time as MM:SS (e.g. `3:00`) |
| `--detection_frame` | 0 | Frame offset for initial detection (relative to start) |
| `--threshold` | 0.5 | RF-DETR detection confidence threshold |
| `--rfdetr_size` | base | RF-DETR model: `base` or `large` |
| `--sam2_model` | base_plus | SAM2 model: `tiny`, `small`, `base_plus`, `large` |
| `--step_size` | 60 | SAM2 propagation chunk size (frames) |
| `--max_history` | 15 | SAM2 memory pruning window |
| `--detect_interval` | 5 | Run RF-DETR every N frames (0 = SAM2 only) |
| `--auto` | off | Skip verification, auto-select top 2 detections |
| `--cpu` | off | Force CPU inference |
| `--output_dir` | output | Output directory |

## Architecture

| File | Purpose |
|------|---------|
| `device.py` | MPS/CPU device detection, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `detect.py` | RF-DETR person detection (COCO class 0) |
| `select_boxes.py` | Human verify/correct bounding boxes via cv2 UI |
| `pose.py` | RTMPose keypoint estimation from bounding boxes |
| `sam2_manager.py` | Native SAM2VideoPredictor wrapper with memory pruning |
| `identity_manager.py` | DINOv2 + color histogram + multi-bin re-identification |
| `advanced_tracking.py` | Online MLP classifiers (DINOv2 joint features) |
| `state_machine.py` | Tracking state transitions (scramble, cut, fade handling) |
| `video_io.py` | Scene cut and fade detection |
| `hybrid_tracking.py` | Main hybrid tracking loop (orchestrates all managers) |
| `pipeline.py` | CLI orchestrator: detect → verify → track |

## Key Features

### Hybrid Tracking
SAM2 provides continuous mask propagation. RF-DETR runs periodically (every `--detect_interval` frames) for detection correction. When RF-DETR detects athletes, its boxes are fed back to SAM2 as prompts. When RF-DETR misses, SAM2 propagation carries forward.

### State Machine
Handles complex tracking scenarios:
- **TRACKING → SCRAMBLE**: When mask IoU > 0.7 (athletes intertwined)
- **SCRAMBLE → TRACKING**: When IoU < 0.6 (separated) — triggers identity verification
- **Scene cut → RE_ID_MODE**: Uses DINOv2 + Hungarian assignment to re-acquire athletes
- **Fade → BLACKOUT → RECOVERING**: Resets SAM2 state through fade transitions

### Online MLP Classifiers
Per-track binary classifiers (384→64→2) trained on DINOv2 joint-patch features. Cross-trained between athletes. Cost matrix: `(1-0.7)×IOU + 0.7×MLP_score - pose_penalty`.

### Memory Pruning
SAM2's conditioning frames grow unbounded. `prune_memory()` keeps only the last `--max_history` frames, preventing memory blowup on long videos.

### Identity Re-ID
DINOv2 vits14 embeddings (50%) + HSV color histograms (30%) + torso features (20%). Used for identity verification after scrambles and re-acquisition after scene cuts.

## Device Notes

- MPS (Apple Silicon) is used by default with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- SAM2 runs with `offload_video_to_cpu=True` and `offload_state_to_cpu=True` for memory efficiency
- RTMPose runs on CPU to save GPU memory
- Models downloaded on first run: SAM2 (~400MB), DINOv2 (~80MB), RF-DETR (~300MB)
