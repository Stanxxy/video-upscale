"""Finalize tracking output (video encode, cleanup)."""
import os
import subprocess
import time


def finalize_tracking(ctx):
    ctx.out.release()
    ctx.video_io.release()

    if ctx.user_cancelled or ctx.human_suspend:
        if ctx._json_file is not None:
            ctx._json_file.write("\n]}\n")
            ctx._json_file.close()
        ctx.sam2_mgr.cleanup()
        print(
            "Tracking cancelled by user."
            if ctx.user_cancelled
            else "Tracking suspended for human verification.",
        )
        return None

    if ctx._json_file is not None:
        ctx._json_file.write("\n]}\n")
        ctx._json_file.close()
        print(f"Tracking JSON: {ctx.json_path}")

    output_video = os.path.join(ctx.output_dir, "tracked_output.mp4")
    print(f"\nRe-encoding: {output_video}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", ctx.raw_output,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-loglevel", "error",
                output_video,
            ],
            check=True,
        )
        os.remove(ctx.raw_output)
    except Exception as e:
        print(f"FFmpeg failed ({e}). Raw video at: {ctx.raw_output}")
        output_video = ctx.raw_output

    ctx.sam2_mgr.cleanup()

    elapsed = time.time() - ctx.t_start
    n = ctx.frames_processed
    rate = n / elapsed if elapsed > 0 else 0
    print(f"\nDone in {elapsed:.1f}s ({n} frames, {rate:.1f} fps)")
    print(f"Output: {output_video}")

    return ctx.json_path
