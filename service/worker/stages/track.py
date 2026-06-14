"""STAGE 3: TRACKING (15-55%) and skip_upscale short-circuit."""
from __future__ import annotations

import json
import logging
import os
import time

from service.analysis_keyspaces_enums import JobState, PipelineStage
from service.checkpoints import (
    END_OF_TRACKING_SENTINEL,
    build_track_completed,
    build_tracking_started,
    build_upload_incremental,
)
from service.models import JobStatus, JobSuspendedError, ProcessingMode
from service.tracking_chain_merge import consolidate_tracking_json_with_job_chain

from service.worker.context import WorkerRunContext
from service.worker.callbacks.detection import _make_detection_cb
from service.worker.progress import (
    _make_worker_state,
    _pct_at_least,
    _schedule_background_coro,
    _track_completed_clip_worker_state,
    _tracking_progress_flags,
    _tracking_progress_pct_clip,
    _update_tracking_progress_with_partial,
)

logger = logging.getLogger("service.worker")


async def run_track_stage(ctx: WorkerRunContext) -> None:
    """Run sequential tracking; may complete the job when skip_upscale=True."""
    job_id = ctx.job_id
    request = ctx.request
    config = ctx.config
    job_store = ctx.job_store
    jobs_store = ctx.jobs_store
    work_dir = ctx.work_dir
    loop = ctx.loop
    progress_floor = ctx.progress_floor
    clip_start_frame = ctx.clip_start_frame
    clip_end_resolved = ctx.clip_end_resolved
    clip_total_frames = ctx.clip_total_frames
    tracking_start_frame = ctx.tracking_start_frame
    end_frame = ctx.end_frame
    video_path = ctx.video_path
    s3 = ctx.s3
    box_a = ctx.box_a
    box_b = ctx.box_b

    from service.segment_runner import compute_segment_ranges as _csr

    k_needed = len(_csr(clip_total_frames, config.segment_max_frames))
    use_parallel_upscale = (
        (k_needed > 1 or config.standard_segments > 1)
        and not request.resume_tracking_s3_key
        and not request.skip_upscale
    )

    logger.info("Job %s: stage tracking (15-55%%)", job_id)
    track_stage_pct = _pct_at_least(15.0, progress_floor)
    track_started_cf = min(
        max(tracking_start_frame - clip_start_frame + 1, 1),
        clip_total_frames,
    )
    await job_store.update_job(
        job_id, status=JobStatus.TRACKING,
        progress_percent=track_stage_pct,
        current_frame=track_started_cf,
        total_frames=clip_total_frames,
    )
    await jobs_store.update_progress(
        job_id, PipelineStage.TRACK, track_stage_pct,
        current_frame=track_started_cf,
        total_frames=clip_total_frames,
    )

    from service.tracking_runner import run_tracking_job

    tracking_output_dir = os.path.join(work_dir, "tracking")
    os.makedirs(tracking_output_dir, exist_ok=True)

    skip_tracking_runner = (
        request.resume_from_frame == END_OF_TRACKING_SENTINEL
        and bool(request.resume_tracking_s3_key)
    )

    partial_tracking_data = None
    if request.resume_tracking_s3_key and not skip_tracking_runner:
        logger.info(
            "Job %s: downloading partial tracking from %s",
            job_id,
            request.resume_tracking_s3_key,
        )
        partial_bucket = request.output_bucket or request.bucket
        partial_tracking_data = await loop.run_in_executor(
            None,
            s3.download_json,
            partial_bucket,
            request.resume_tracking_s3_key,
        )
        logger.info(
            "Job %s: partial tracking has %d frames",
            job_id,
            len(partial_tracking_data.get("frames", [])),
        )

    tracking_json_path: str | None = None
    if skip_tracking_runner:
        logger.info(
            "Job %s: skipping run_tracking_job — loading tracking JSON from %s",
            job_id,
            request.resume_tracking_s3_key,
        )
        load_bucket = request.output_bucket or request.bucket
        full_tracking = await loop.run_in_executor(
            None,
            s3.download_json,
            load_bucket,
            request.resume_tracking_s3_key,
        )
        tracking_json_path = os.path.join(tracking_output_dir, "tracking.json")
        with open(tracking_json_path, "w") as f:
            json.dump(full_tracking, f)
    else:
        last_ks_write = 0.0
        last_partial_upload = 0.0
        first_partial_upload = False

        def tracking_progress_cb(frames_done: int, total: int, global_idx: int):
            nonlocal last_ks_write, last_partial_upload, first_partial_upload
            clip_done, pct = _tracking_progress_pct_clip(
                global_idx, clip_start_frame, clip_total_frames, progress_floor,
            )
            resume_next_global = global_idx + 1
            now = time.monotonic()
            write_lifecycle, upload_partial = _tracking_progress_flags(
                now, last_ks_write, last_partial_upload,
            )
            if frames_done >= 1 and not first_partial_upload:
                upload_partial = True
                first_partial_upload = True
            if write_lifecycle:
                last_ks_write = now
            if upload_partial:
                last_partial_upload = now
            _schedule_background_coro(
                _update_tracking_progress_with_partial(
                    job_id, clip_done, clip_total_frames, pct,
                    job_store, jobs_store,
                    request, work_dir, s3,
                    resume_next_global=resume_next_global,
                    write_lifecycle=write_lifecycle,
                    upload_partial=upload_partial,
                ),
                loop,
                context="tracking progress write",
            )

        detection_cb = _make_detection_cb(
            job_id, loop, jobs_store, s3, config, request, work_dir,
            clip_start_frame=clip_start_frame,
            clip_total_frames=clip_total_frames,
            progress_floor=progress_floor,
        )
        import torch as _torch_s7
        is_cuda_s7 = _torch_s7.cuda.is_available()
        default_step = 120 if is_cuda_s7 else config.tracking_step_size
        eff_step_size = request.step_size or default_step
        eff_max_history = request.max_history or config.tracking_max_history
        eff_max_missing_frames = (
            request.max_missing_frames
            if request.max_missing_frames is not None
            else config.tracking_max_missing_frames
        )
        is_fast_mode = request.processing_mode == ProcessingMode.FAST
        if is_fast_mode:
            eff_sam2_model = "facebook/sam2.1-hiera-tiny"
            eff_prop_stride = config.fast_prop_stride
            eff_enable_pose = False
        else:
            eff_sam2_model = request.sam2_model
            eff_prop_stride = config.standard_prop_stride
            eff_enable_pose = True

        if is_fast_mode or eff_prop_stride > 1:
            eff_frame_stride = 1
        elif request.frame_stride > 0:
            eff_frame_stride = request.frame_stride
        else:
            import cv2 as _cv2
            cap_fps = _cv2.VideoCapture(video_path)
            src_fps = cap_fps.get(_cv2.CAP_PROP_FPS) or 30.0
            cap_fps.release()
            eff_frame_stride = max(1, round(src_fps / 10))
        logger.info(
            "Job %s tracking config: processing_mode=%s sam2=%s "
            "step_size=%s max_history=%s max_missing_frames=%s "
            "frame_stride=%s prop_stride=%s enable_pose=%s",
            job_id,
            request.processing_mode,
            eff_sam2_model,
            eff_step_size,
            eff_max_history,
            eff_max_missing_frames,
            eff_frame_stride,
            eff_prop_stride,
            eff_enable_pose,
        )

        await jobs_store.write_checkpoint(
            job_id, PipelineStage.TRACK, False,
            build_tracking_started(
                clip_start_frame=clip_start_frame,
                clip_end_frame=clip_end_resolved,
                worker_state=_make_worker_state(
                    progress_percent=track_stage_pct,
                    current_frame=track_started_cf,
                    total_frames=clip_total_frames,
                    stage_progress_fraction=(
                        track_started_cf / max(clip_total_frames, 1)
                    ),
                ),
            ),
        )

        tracking_json_path = await loop.run_in_executor(
            None,
            lambda: run_tracking_job(
                video_path,
                box_a,
                box_b,
                tracking_output_dir,
                sam2_model_id=eff_sam2_model,
                yolo_model=request.yolo_model,
                detection_threshold=request.detection_threshold,
                start_frame=tracking_start_frame,
                end_frame=end_frame,
                step_size=eff_step_size,
                max_history=eff_max_history,
                max_missing_frames=eff_max_missing_frames,
                frame_stride=eff_frame_stride,
                prop_stride=eff_prop_stride,
                enable_pose=eff_enable_pose,
                progress_cb=tracking_progress_cb,
                detection_cb=detection_cb,
                should_stop=lambda: job_store.is_cancelled(job_id),
                athlete_bindings=request.athlete_bindings,
                # Stream 0b: confirmed obj_id->player_id binding from the
                # correction. Seeds init_boxes so track_ids never flip on resume.
                player_mapping=request.player_mapping,
            ),
        )
    logger.info(
        "Job %s: tracking executor returned path=%s",
        job_id,
        tracking_json_path,
    )

    if tracking_json_path is None:
        raise JobSuspendedError("Awaiting mid-tracking detection correction")

    if partial_tracking_data:
        with open(tracking_json_path) as f:
            new_tracking = json.load(f)
        merged_frames = (
            partial_tracking_data.get("frames", []) + new_tracking.get("frames", [])
        )
        merged = {
            **new_tracking,
            "frames": merged_frames,
            "start_frame": partial_tracking_data.get(
                "start_frame", clip_start_frame,
            ),
        }
        with open(tracking_json_path, "w") as f:
            json.dump(merged, f)
        logger.info(
            "Job %s: merged %d partial + %d new = %d total frames",
            job_id,
            len(partial_tracking_data.get("frames", [])),
            len(new_tracking.get("frames", [])),
            len(merged_frames),
        )

    await consolidate_tracking_json_with_job_chain(
        jobs_store=jobs_store,
        s3=s3,
        bucket=request.output_bucket or request.bucket,
        leaf_job_id=job_id,
        local_tracking_json_path=tracking_json_path,
        clip_start_frame=clip_start_frame,
    )

    with open(tracking_json_path) as _f:
        track_data = json.load(_f)
    frame_count = len(track_data.get("frames", []))
    await jobs_store.write_checkpoint(
        job_id, PipelineStage.TRACK, False,
        build_track_completed(
            start_frame=clip_start_frame,
            frame_count=frame_count,
            worker_state=_track_completed_clip_worker_state(
                track_data, clip_start_frame, clip_total_frames,
                progress_percent=55.0,
            ),
        ),
    )

    if request.skip_upscale:
        await job_store.update_job(
            job_id, status=JobStatus.UPLOADING, progress_percent=80.0,
        )
        await jobs_store.update_progress(job_id, PipelineStage.UPLOAD, 80.0)

        output_bucket = request.output_bucket or request.bucket
        base_key = os.path.splitext(request.key)[0]
        tracking_result_key = f"{base_key}_tracked.json"
        tracked_video_key = f"{base_key}_tracked.mp4"

        s3.ensure_bucket(output_bucket)

        with open(tracking_json_path) as f:
            tracking_data = json.load(f)
        if request.resume_existing_upload_tracking_key != tracking_result_key:
            await loop.run_in_executor(
                None, s3.upload_json, tracking_data,
                output_bucket, tracking_result_key,
            )

        await jobs_store.write_checkpoint(
            job_id, PipelineStage.TRACK, False,
            build_track_completed(
                start_frame=clip_start_frame,
                frame_count=frame_count,
                tracking_s3_key=tracking_result_key,
                worker_state=_track_completed_clip_worker_state(
                    tracking_data, clip_start_frame, clip_total_frames,
                    progress_percent=55.0,
                ),
            ),
        )

        tracked_video = os.path.join(
            tracking_output_dir, "tracked_output.mp4",
        )
        if os.path.isfile(tracked_video):
            await loop.run_in_executor(
                None, s3.upload_file, tracked_video,
                output_bucket, tracked_video_key, "video/mp4",
            )

        await jobs_store.write_checkpoint(
            job_id, PipelineStage.UPLOAD, True,
            build_upload_incremental(
                tracking_s3_key=tracking_result_key,
                worker_state=_make_worker_state(
                    progress_percent=100.0, stage_progress_fraction=1.0,
                ),
            ),
        )

        await job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100.0,
            result_bucket=output_bucket,
            result_key=tracking_result_key,
        )
        await jobs_store.set_state(job_id, JobState.COMPLETED)
        logger.info(
            "Job %s completed (tracking only -> s3://%s/%s)",
            job_id, output_bucket, tracking_result_key,
        )
        ctx.pipeline_complete = True
        return

    ctx.tracking_json_path = tracking_json_path
    ctx.tracking_output_dir = tracking_output_dir
    ctx.frame_count = frame_count
    ctx.use_parallel_upscale = use_parallel_upscale
    ctx.k_needed = k_needed
