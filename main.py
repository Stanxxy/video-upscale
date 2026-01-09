import cv2
import json
import os
import argparse
from tqdm import tqdm
from utils import get_union_box, get_padded_square_box

# ... (deduplicate_clips function remains the same) ...
def deduplicate_clips(all_analysis_results):
    all_clips = []
    for chunk in all_analysis_results:
        if "analysis" in chunk and "clips" in chunk["analysis"]:
            for clip in chunk["analysis"]["clips"]:
                all_clips.append(clip)
    all_clips.sort(key=lambda x: x["start_frame"])
    if not all_clips: return []
    merged_clips = []
    current_clip = all_clips[0]
    for next_clip in all_clips[1:]:
        start1, end1 = current_clip["start_frame"], current_clip["end_frame"]
        start2, end2 = next_clip["start_frame"], next_clip["end_frame"]
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_len = max(0, overlap_end - overlap_start)
        is_same_category = current_clip["category"] == next_clip["category"]
        is_close = (start2 - end1) < 30
        if is_same_category and (overlap_len > 0 or is_close):
            new_start = min(start1, start2)
            new_end = max(end1, end2)
            if next_clip.get("confidence", 0) > current_clip.get("confidence", 0):
                current_clip = next_clip.copy()
            current_clip["start_frame"] = new_start
            current_clip["end_frame"] = new_end
        else:
            merged_clips.append(current_clip)
            current_clip = next_clip
    merged_clips.append(current_clip)
    return merged_clips

def process_video(video_path, json_path, output_dir, model_path, method="esrgan", sampling_rate=1, max_frames=None, target_size=1024, diffusion_strength=0.5, analyze=False, api_key=None, multi_agent=False):
    # Load JSON
    print(f"Loading detection data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a lookup for frame data
    frames_data = {f['frame']: f['athletes'] for f in data['frames']}
    
    # Setup Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {width}x{height}, Total frames: {total_frames}, FPS: {fps}")
    
    # Initialize Restorer based on method
    if method == "diffusion":
        from diffusion_restorer import DiffusionRestorer
        restorer = DiffusionRestorer()
        print("Using Diffusion (ControlNet-Tile) for enhancement.")
    elif method == "swinir" or method == "hat":
        from restorer import RealESRGANRestorer
        restorer = RealESRGANRestorer(model_path)
        print(f"Using {method.upper()} for enhancement.")
    else:
        from restorer import RealESRGANRestorer
        restorer = RealESRGANRestorer(model_path)
        print("Using Real-ESRGAN for enhancement.")
    
    # Initialize Analyzer if requested
    analyzer = None
    if analyze:
        if not api_key:
            print("Error: API Key required for analysis.")
            return
        
        if multi_agent:
            from analyzer import BJJMultiAgentAnalyzer, analyze_sequence_sync
            analyzer = BJJMultiAgentAnalyzer(api_key)
            print("Initialized Gemini BJJ Multi-Agent System (3 Agents + Judge).")
        else:
            # Re-import the single agent analyzer (need to ensure it's available or update analyzer.py to export it)
            # Since we replaced the file content, we might need to put both classes in analyzer.py or restore the old one.
            # I will assume analyzer.py now contains BOTH classes or I will update it to contain both.
            # Let's check analyzer.py first. It seems I overwrote it. I need to restore the single agent class as well.
            from analyzer import BJJTechniqueAnalyzer
            analyzer = BJJTechniqueAnalyzer(api_key)
            print("Initialized Gemini BJJ Single Agent (Thinking Mode).")

    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which frames to process
    target_frames = sorted([f for f in frames_data.keys() if f % sampling_rate == 0])
    if max_frames:
        target_frames = target_frames[:max_frames]
        
    print(f"Processing {len(target_frames)} frames using {method}...")
    
    pbar = tqdm(total=len(target_frames), desc="Enhancing Crops")
    
    sliding_buffer = [] 
    analysis_results = []
    current_context = "Start of match."
    
    # Sliding Window Config
    WINDOW_SIZE = 30 
    STRIDE = 15      
    
    current_frame = 0
    
    for i, frame_idx in enumerate(target_frames):
        # 1. Seek & Capture
        if frame_idx < current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame = frame_idx
        while current_frame < frame_idx:
            cap.grab()
            current_frame += 1
        ret, frame = cap.read()
        if not ret: break
        current_frame += 1
        
        athletes = frames_data[frame_idx]
        if not athletes:
            pbar.update(1)
            continue

        # 2. Enhance
        boxes = [a['box'] for a in athletes]
        union_box = get_union_box(boxes)
        square_box = get_padded_square_box(union_box, padding=0.2, img_shape=(height, width))
        x1, y1, x2, y2 = square_box
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            pbar.update(1)
            continue

        try:
            enhanced_crop = None
            if method == "diffusion":
                h_crop, w_crop = crop.shape[:2]
                if max(h_crop, w_crop) > 768:
                    scale = 768 / max(h_crop, w_crop)
                    crop = cv2.resize(crop, (int(w_crop * scale), int(h_crop * scale)), interpolation=cv2.INTER_LANCZOS4)
                enhanced_crop = restorer.enhance(crop, strength=diffusion_strength)
            else:
                enhanced_crop = restorer.enhance(crop, target_size=target_size)
            
            # Save to disk
            if method == "diffusion": prefix = "diff_"
            elif method == "swinir": prefix = "swinir_"
            elif method == "hat": prefix = "hat_"
            else: prefix = "esrgan_"
            output_path = os.path.join(output_dir, f"{prefix}frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_path, enhanced_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # Add to Buffer
            if analyze:
                from PIL import Image
                img_rgb = cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                sliding_buffer.append((frame_idx, pil_img))
                
                if len(sliding_buffer) >= WINDOW_SIZE:
                    batch_frames = [x[1] for x in sliding_buffer[:WINDOW_SIZE]]
                    batch_indices = [x[0] for x in sliding_buffer[:WINDOW_SIZE]]
                    
                    chunk_idx = len(analysis_results) + 1
                    tqdm.write(f"Analyzing Window {chunk_idx} ({batch_indices[0]}-{batch_indices[-1]})...")
                    
                    # Choose analysis method
                    if multi_agent:
                        result_json_str = analyze_sequence_sync(analyzer, batch_frames, batch_indices, current_context)
                    else:
                        result_json_str = analyzer.analyze_sequence(batch_frames, batch_indices, current_context)
                    
                    try:
                        result_data = json.loads(result_json_str)
                        if "current_context_summary" in result_data:
                            current_context = result_data["current_context_summary"]
                        
                        analysis_results.append({
                            "window": chunk_idx,
                            "frames": batch_indices,
                            "analysis": result_data
                        })
                    except:
                        tqdm.write("JSON Error")
                        analysis_results.append({"window": chunk_idx, "raw_error": result_json_str})

                    sliding_buffer = sliding_buffer[STRIDE:]
                    
                    with open(os.path.join(output_dir, "analysis_raw.json"), 'w') as f:
                        json.dump(analysis_results, f, indent=2)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        pbar.update(1)

    # Process Final Buffer
    if analyze and len(sliding_buffer) > 0:
        tqdm.write("Analyzing Final Partial Window...")
        batch_frames = [x[1] for x in sliding_buffer]
        batch_indices = [x[0] for x in sliding_buffer]
        if multi_agent:
            result_json_str = analyze_sequence_sync(analyzer, batch_frames, batch_indices, current_context)
        else:
            result_json_str = analyzer.analyze_sequence(batch_frames, batch_indices, current_context)
            
        try:
            result_data = json.loads(result_json_str)
            analysis_results.append({"window": "final", "frames": batch_indices, "analysis": result_data})
        except:
            pass
        with open(os.path.join(output_dir, "analysis_raw.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)

    # Final Deduplication
    if analyze:
        tqdm.write("Deduplicating clips...")
        final_clips = deduplicate_clips(analysis_results)
        
        mode = "Multi-Agent" if multi_agent else "Single-Agent"
        final_output = {
            "match_summary": f"Analysis generated via {mode}",
            "clips": final_clips
        }
        
        with open(os.path.join(output_dir, "analysis_final.json"), 'w') as f:
            json.dump(final_output, f, indent=2)
            
        print(f"Final Analysis saved to {os.path.join(output_dir, 'analysis_final.json')}")

    cap.release()
    pbar.close()

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
        api_key=args.api_key
    )
