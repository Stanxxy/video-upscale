# video-upscale

A small Python pipeline to **crop the action** (using per-frame tracking boxes) and **enhance** those crops using:

- Real-ESRGAN-style super-resolution models (via `spandrel`)
- (Optional) Diffusion + ControlNet Tile upscaling (via `diffusers`)
- (Optional) Gemini-powered BJJ technique analysis to generate clip timestamps

## What it does

Given:

- An input video (default: `input_video.mp4`)
- A tracking JSON with per-frame athlete bounding boxes (default: `input_video_hybrid_tracked.json`)

It will:

- Compute a union box of all athletes for each processed frame
- Convert it into a padded square crop
- Enhance the crop and write JPEGs to an output directory
- If analysis is enabled, batch frames in a sliding window, call Gemini, and write `analysis_raw.json` + a deduplicated `analysis_final.json`

## Quickstart

Create a venv and install deps (exact versions are intentionally not pinned yet):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip

# Core pipeline deps
pip install numpy opencv-python pillow tqdm

# Enhancement (Real-ESRGAN / SwinIR / HAT via spandrel)
pip install torch spandrel

# Optional: diffusion mode
pip install diffusers transformers accelerate safetensors

# Optional: Gemini analysis
pip install google-genai
```

## Run

Baseline enhancement (Real-ESRGAN style, writes crops to `enhanced_crops/`):

```bash
python main.py --method esrgan --sampling 1 --output enhanced_crops
```

Use a different enhancement method:

```bash
python main.py --method swinir --model RealESRGAN_x4plus.pth
python main.py --method hat    --model RealESRGAN_x4plus.pth
python main.py --method diffusion --output enhanced_crops_diffusion --strength 0.5
```

Limit work for a quick smoke test:

```bash
python main.py --sampling 5 --max_frames 200
```

Enable analysis (single-agent):

```bash
python main.py --analyze --api_key "$GEMINI_API_KEY" --output enhanced_crops
```

Enable analysis (multi-agent + judge):

```bash
python main.py --analyze --multi_agent --api_key "$GEMINI_API_KEY" --output enhanced_crops
```

## Inputs

- **Video**: `--video` (default `input_video.mp4`)
- **Tracking JSON**: `--json` (default `input_video_hybrid_tracked.json`)

The tracking JSON is expected to look like:

```json
{
  "frames": [
    { "frame": 0, "athletes": [ { "box": [x1, y1, x2, y2] }, ... ] },
    ...
  ]
}
```

## Outputs

Inside `--output` (e.g. `enhanced_crops/`):

- `esrgan_frame_000123.jpg` / `swinir_frame_*.jpg` / `hat_frame_*.jpg` / `diff_frame_*.jpg`
- If `--analyze`:
  - `analysis_raw.json`: window-by-window Gemini outputs
  - `analysis_final.json`: deduplicated clip list (merged overlaps / near-adjacent)

## Notes

- **Device selection**: the upscaler and diffusion paths auto-pick `mps` (Apple Silicon), then `cuda`, then `cpu`.
- **Large files**: this repo intentionally `.gitignore`s `analysis_*`, `enhanced_*`, `input_video.mp4`, and `input_video_hybrid_tracked.json` since they’re data outputs/inputs.

