import argparse
import os
import sys
import cv2
import time
from tiled_restorer import TiledRealESRGANRestorer

def main():
    parser = argparse.ArgumentParser(description="Test Tiled Real-ESRGAN Upscaling")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.png", help="Path to output image")
    parser.add_argument("--model", default="../RealESRGAN_x4plus.pth", help="Path to model file")
    parser.add_argument("--tile_size", type=int, default=400, help="Tile size (e.g. 400)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Tile padding")
    parser.add_argument("--fp32", action="store_true", help="Force FP32 (disable FP16)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
        
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return

    print(f"Initializing Tiled Restorer with model: {args.model}")
    print(f"Tile size: {args.tile_size}, Padding: {args.tile_pad}")
    
    restorer = TiledRealESRGANRestorer(
        model_path=args.model,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
        half=not args.fp32
    )
    
    print(f"Reading image: {args.input}")
    img = cv2.imread(args.input)
    if img is None:
        print("Error: Could not read image.")
        return

    print(f"Input resolution: {img.shape[1]}x{img.shape[0]}")
    
    start_time = time.time()
    print("Starting upscale...")
    
    # Run enhancement
    # Note: target_size is set to None to get full resolution x4 upscale
    # If you want to limit the output size, pass target_size parameter
    result = restorer.enhance(img)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Upscale complete in {duration:.2f} seconds.")
    print(f"Output resolution: {result.shape[1]}x{result.shape[0]}")
    
    cv2.imwrite(args.output, result)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
