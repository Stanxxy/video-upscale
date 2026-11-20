"""Internal worker helpers."""
import logging

from service.config import ServiceConfig
from service.job_store import InMemoryJobStore
from service.s3 import S3Client

logger = logging.getLogger("service.worker")

# LocalStack's default account. Real AWS topic ARNs must never be published
# through the emulator endpoint.
_LOCALSTACK_DEFAULT_ACCOUNT_ID = "000000000000"


def _make_s3(config: ServiceConfig) -> S3Client:
    return S3Client(
        region=config.aws_region,
        endpoint_url=config.s3_endpoint_url or None,
        access_key_id=config.aws_access_key_id or None,
        secret_access_key=config.aws_secret_access_key or None,
    )


def _sns_account_id(topic_arn: str | None) -> str | None:
    if not topic_arn:
        return None
    parts = topic_arn.split(":")
    if len(parts) < 6 or parts[2] != "sns":
        return None
    return parts[4]


def _emulator_account_id(config: ServiceConfig) -> str:
    """LocalStack account implied by the configured credentials.

    Shared local infra isolates streams by AWS account: the namespace's access key
    is its 12-digit account id. Any other key (``test``, unset, or a real ``AKIA…``
    key) means the default emulator account. Real AWS account ARNs therefore never
    match, because real access keys are not 12-digit account ids.
    """
    key = config.aws_access_key_id
    if len(key) == 12 and key.isdigit():
        return key
    return _LOCALSTACK_DEFAULT_ACCOUNT_ID


def _sns_endpoint_url(
    config: ServiceConfig, topic_arn: str | None = None,
) -> str | None:
    endpoint = config.sns_endpoint_url or config.s3_endpoint_url or None
    if not endpoint:
        return None
    account_id = _sns_account_id(topic_arn)
    if account_id and account_id != _emulator_account_id(config):
        return None
    return endpoint


def _make_s3_for_bucket(config: ServiceConfig, bucket: str) -> S3Client:
    trial_buckets = {
        item.strip() for item in (config.trial_s3_buckets or "").split(",") if item.strip()
    }
    if bucket in trial_buckets and config.trial_s3_endpoint_url:
        return S3Client(
            region=config.aws_region,
            endpoint_url=config.trial_s3_endpoint_url,
            access_key_id=config.trial_aws_access_key_id or "test",
            secret_access_key=config.trial_aws_secret_access_key or "test",
        )
    return _make_s3(config)


def _is_cancelled(job_id: str, job_store: InMemoryJobStore) -> bool:
    if job_store.is_cancelled(job_id):
        logger.info("Job %s cancelled", job_id)
        return True
    return False
