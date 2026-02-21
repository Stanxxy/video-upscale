# test_tracking - BJJ Athlete Detection & Tracking

Tests RF-DETR (detection) + SAM 2.1 (tracking) for recognizing and tracking BJJ athletes in video on Mac M4 Max.

## Pipeline

```
RF-DETR (detect persons in frame 0) → SAM 2.1 (track across all frames) → Annotated video + JSON
```

## Setup

```bash
# From project root, activate existing venv
source venv/bin/activate

# Install additional dependencies
pip install -r test_tracking/requirements.txt
```

## Usage

```bash
cd test_tracking

# Full pipeline (detect + track + visualize)
python pipeline.py --video /path/to/your_video.mp4 --max_frames 100

# Limit to fewer frames for quick test
python pipeline.py --video /path/to/your_video.mp4 --max_frames 50

# Force CPU if MPS has issues
python pipeline.py --video /path/to/your_video.mp4 --max_frames 100 --cpu

# Skip visualization (just get JSON)
python pipeline.py --video /path/to/your_video.mp4 --max_frames 100 --no_visualize

# Step by step:
python detect.py --video /path/to/your_video.mp4 --frame 0 --threshold 0.5
python track.py --video /path/to/your_video.mp4 --detections output/detections.json --max_frames 100
```

## Output

All outputs go to `output/`:
- `detections.json` — RF-DETR detection results for initial frame
- `tracking.json` — Per-frame tracking data (compatible with main pipeline format)
- `tracked_output.mp4` — Annotated video with bounding boxes + segmentation masks
- `tracking_summary.jpg` — Grid of sample frames showing tracking

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to input video |
| `--detection_frame` | 0 | Frame index for initial detection |
| `--threshold` | 0.5 | Detection confidence threshold |
| `--rfdetr_size` | base | RF-DETR model: `base` or `large` |
| `--sam2_model` | tiny | SAM2 model: `tiny`, `small`, `base_plus`, `large` |
| `--max_frames` | all | Limit number of frames to process |
| `--sampling_rate` | 1 | Process every Nth frame |
| `--cpu` | off | Force CPU inference |

## Device Notes

- MPS (Apple Silicon) is used by default with automatic CPU fallback for unsupported ops
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically
- SAM2 tiny model (~100MB) is downloaded on first run and cached by HuggingFace
- RF-DETR base model is downloaded on first run
