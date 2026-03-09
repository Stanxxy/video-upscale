from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    """Load from environment variables or .env file. Use BJJ_ prefix (e.g. BJJ_GEMINI_API_KEY)."""

    # AWS / S3
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_endpoint_url: str = ""  # leave empty for real AWS; set for LocalStack etc.
    sns_topic_arn: str = ""

    # Gemini (credential – set in .env, do not commit)
    gemini_api_key: str = ""

    # Model
    model_path: str = "RealESRGAN_x4plus.pth"

    # VLLM (LM Studio / Qwen VL)
    vllm_base_url: str = "http://localhost:1234/v1"
    vllm_model: str = "qwen2.5-vl-7b-instruct"
    vllm_timeout_sec: float = 30.0

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BJJ_",
        extra="ignore",
    )
