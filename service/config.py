from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceConfig(BaseSettings):
    """Load from environment variables or .env file. Use BJJ_ prefix (e.g. BJJ_GEMINI_API_KEY)."""

    # AWS
    aws_region: str = "us-east-1"
    sns_topic_arn: str = ""

    # Gemini (credential – set in .env, do not commit)
    gemini_api_key: str = ""

    # Model
    model_path: str = "RealESRGAN_x4plus.pth"

    # Service
    max_concurrent_jobs: int = 1
    temp_dir: str = "/tmp/bjj-pipeline"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BJJ_",
        extra="ignore",
    )
