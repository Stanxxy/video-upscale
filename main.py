import argparse
from pipeline import process_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BJJ High-Res Combat-Crop Pipeline")
    parser.add_argument("--video", default="input_video.mp4", help="Path to input video")
    parser.add_argument("--json", default="input_video_hybrid_tracked.json", help="Path to tracking JSON")
    parser.add_argument("--output", default="enhanced_crops", help="Output directory for crops")
    parser.add_argument("--model", default="RealESRGAN_x4plus.pth", help="Path to Real-ESRGAN/SwinIR/HAT model")
    parser.add_argument("--method", default="esrgan", choices=["esrgan", "diffusion", "swinir", "hat"], help="Enhancement method")
    parser.add_argument("--sampling", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--target_size", type=int, default=1024, help="Target max dimension for output")
    parser.add_argument("--strength", type=float, default=0.5, help="Diffusion denoising strength")
    parser.add_argument("--analyze", action="store_true", help="Enable Gemini BJJ analysis")
    parser.add_argument("--multi_agent", action="store_true", help="Use Multi-Agent (3+1) system instead of single agent")
    parser.add_argument("--api_key", default=None, help="Google Gemini API Key")

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        json_path=args.json,
        output_dir=args.output,
        model_path=args.model,
        method=args.method,
        sampling_rate=args.sampling,
        max_frames=args.max_frames,
        target_size=args.target_size,
        diffusion_strength=args.strength,
        analyze=args.analyze,
        multi_agent=args.multi_agent,
        api_key=args.api_key,
    )
