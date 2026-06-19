from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    """Load from environment variables or .env file. Use BJJ_ prefix (e.g. BJJ_GEMINI_API_KEY)."""

    # AWS / S3
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_endpoint_url: str = ""  # leave empty for AWS production; set only for local S3 emulators.
    sns_topic_arn: str = ""

    # Gemini (credential – set in .env, do not commit)
    gemini_api_key: str = ""
    # Per-request HTTP timeout for Gemini (ms). Prevents indefinite hangs on bad networks.
    gemini_request_timeout_ms: int = Field(
        default=600_000,
        description="Gemini HTTP client timeout per generate_content call (milliseconds).",
    )
    # Vision model for optional >2-candidate athlete hints (human never auto-applies).
    gemini_athlete_suggest_model: str = "gemini-3.1-flash-lite"

    # Model
    model_path: str = "RealESRGAN_x4plus.pth"

    # Detection — 24 hours to allow async corrections via REST
    detection_timeout: float = 86400.0

    # Tracking runtime defaults (used by service controller when request
    # does not explicitly provide overrides).
    tracking_step_size: int = 60
    tracking_max_history: int = 8
    tracking_max_missing_frames: int = 15

    # Service
    # G4 (anti-DoS / anti-OOM): the public-launch single-GB10 deployment runs ONE
    # analysis job on the GPU at a time. Default lowered from 2 → 1 because the
    # production target is a single GB10 where two concurrent pipelines starve each
    # other / OOM. The M1 memory data that justified 2/4 assumed a 128 GB Spark with
    # spare headroom — set BJJ_MAX_CONCURRENT_JOBS=2 (or 4) only on that target.
    # Override: BJJ_MAX_CONCURRENT_JOBS=1 (prod GB10, default) .. 4 (Spark max).
    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        description=(
            "G4: max analysis jobs allowed to hold the GPU concurrently. Default 1 "
            "for the single-GB10 public deployment. The Nth+1 request queues "
            "(up to max_queued_jobs) then 429s."
        ),
    )
    # G4: bounded queue depth. Requests beyond (max_concurrent_jobs) wait on the GPU
    # semaphore up to this many extra waiters; anything beyond is rejected with 429
    # immediately so callers fail fast instead of accumulating unbounded backlog.
    max_queued_jobs: int = Field(
        default=1,
        ge=0,
        description=(
            "G4: max NEW analysis jobs allowed to wait (queue) for the GPU beyond the "
            "running one. Total admitted = max_concurrent_jobs + max_queued_jobs; "
            "beyond that → HTTP 429. 0 = no queue (reject as soon as GPU is busy)."
        ),
    )
    # G7 (kill switch / panic button): when true, all GPU/Gemini analysis entrypoints
    # return HTTP 503 immediately without scheduling any work. Lets the founder stop
    # all GPU + Gemini spend without killing the uvicorn process. Toggle via
    # BJJ_ANALYSIS_DISABLED=true and restart (or it is read fresh per request via
    # the live ServiceConfig).
    analysis_disabled: bool = Field(
        default=False,
        description=(
            "G7 kill switch: BJJ_ANALYSIS_DISABLED=true makes new-analysis and resume "
            "entrypoints return 503 immediately (stops all GPU/Gemini spend)."
        ),
    )
    # G7 (daily cap): max number of NEW analysis jobs started per UTC calendar day.
    # Resume/correction jobs do NOT count (they finish in-flight work). In-memory
    # counter — resets on process restart (acceptable for the initial launch; a
    # durable per-day counter in Keyspaces is the follow-up).
    max_daily_analyses: int = Field(
        default=50,
        ge=0,
        description=(
            "G7 daily cap: max NEW analysis jobs started per UTC day. Over-cap → HTTP "
            "429. In-memory counter (resets on restart). 0 = block all new analyses."
        ),
    )
    # M3 S9: intra-job K-segment parallelism.
    # K=1 = sequential pipeline (safe default; no memory overhead change).
    # K=4 = target for Spark (128 GB unified); each segment gets its own SAM2 instance.
    # Override: BJJ_STANDARD_SEGMENTS=4 in the Spark deployment environment.
    standard_segments: int = 1
    service_port: int = 8000
    temp_dir: str = "/tmp/bjj-pipeline"
    # Stale-job recovery scans ``job_recovery_index`` partitions by calendar hour
    # (``heartbeat_bucket``). Include enough hours so overnight crashes remain visible
    # after restart (each reconcile tick runs one SELECT per distinct bucket).
    recovery_heartbeat_bucket_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="How many trailing UTC hour buckets to scan for stale RUNNING/INTERRUPTED recovery.",
    )
    # Safety bound: any candidate whose ``last_heartbeat_at`` is older than this
    # many hours is treated as garbage and skipped. Prevents reviving abandoned
    # jobs from a previous deployment when the bucket-scan window is widened.
    # Set to 0 to disable the upper bound (NOT recommended in production).
    recovery_max_heartbeat_age_hours: int = Field(
        default=24,
        ge=0,
        le=168,
        description=(
            "Upper-bound age (hours) on candidate heartbeats. Rows older than this "
            "are treated as garbage and skipped by both periodic and bootstrap "
            "recovery. 0 disables the bound."
        ),
    )
    # M2 S4: async Gemini fanout.
    # gemini_max_inflight=1 preserves the context chain (sequential) while still
    # overlapping upscale compute with the in-flight Gemini call.
    # Raise to 4-8 after regression harness validates accuracy >= 0.95.
    gemini_max_inflight: int = Field(
        default=1,
        ge=1,
        description="Max concurrent Gemini window tasks. 1=sequential chain (safe default).",
    )
    # M5: per-mode propagation stride knobs.
    # Fast mode: prop_stride=24 → 7956/24=332 SAM2 frames (~1.8 min tracking on GB10).
    # Standard mode: prop_stride=5 → 7956/5=1591 SAM2 frames (~15 min tracking on GB10).
    # Override via env: BJJ_FAST_PROP_STRIDE=24  BJJ_STANDARD_PROP_STRIDE=5
    fast_prop_stride: int = Field(
        default=12,
        ge=1,
        description="SAM2 propagation stride for fast mode. 12 → ~5 fps effective from 60 fps source.",
    )
    standard_prop_stride: int = Field(
        default=5,
        ge=1,
        description="SAM2 propagation stride for standard mode. 5 → 12 fps effective from 60 fps source.",
    )

    # Memory-bounded segment splitting (added alongside Mac dev support).
    # SAM2 on CUDA keeps all segment frames in GPU memory simultaneously.
    # Capping frames per segment bounds peak GPU memory regardless of video length.
    # 2000 frames ≈ 33s at 60fps → ~6 GB per segment (standard mode, prop_stride=5).
    # Override: BJJ_SEGMENT_MAX_FRAMES=1500 on 16GB targets to stay within budget.
    segment_max_frames: int = Field(
        default=2000,
        ge=60,
        description=(
            "Maximum video frames per SAM2 segment. K is computed dynamically as "
            "ceil(total_frames / segment_max_frames). Bounds peak memory regardless of "
            "video length. 2000 frames ≈ 33s at 60fps. Tune down on memory-constrained "
            "targets (e.g., BJJ_SEGMENT_MAX_FRAMES=1500 for 16GB)."
        ),
    )
    # Skip SAM2 pre-warm on Mac dev to speed up service restarts (~60s saved).
    # Leave false (default) on gx10 production where the first-job latency matters.
    disable_prewarm: bool = Field(
        default=False,
        description=(
            "Skip SAM2 model pre-warm at startup. Set BJJ_DISABLE_PREWARM=true on dev "
            "machines to speed up iteration. Leave false (default) on gx10 production."
        ),
    )

    # M2 S5: upscale target size and pre-scale cap.
    upscale_target_size: int = Field(
        default=768,
        ge=64,
        description="Long-edge target (px) for ESRGAN output. Was 1024; 768 saves ~3.5x FLOPs.",
    )

    # Upscale / second-pass loop: log at most once per interval while iterating frames.
    upscale_heartbeat_interval_sec: float = Field(
        default=30.0,
        ge=5.0,
        description="Minimum seconds between upscale-stage heartbeat log lines.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BJJ_",
        extra="ignore",
    )
