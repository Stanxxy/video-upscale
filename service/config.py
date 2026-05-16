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
    gemini_athlete_suggest_model: str = "gemini-3.1-flash-lite-preview"

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
    max_concurrent_jobs: int = 1
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
