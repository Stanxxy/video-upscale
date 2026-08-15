from service.config import ServiceConfig
from service.worker.helpers import _make_s3, _make_s3_for_bucket, _sns_endpoint_url


def test_trial_bucket_uses_localstack_endpoint() -> None:
    config = ServiceConfig(
        trial_s3_endpoint_url="http://127.0.0.1:4566",
        trial_s3_buckets="trial-video-bucket,other-trial",
        s3_endpoint_url="",
    )
    client = _make_s3_for_bucket(config, "trial-video-bucket")
    assert client._endpoint_url == "http://127.0.0.1:4566"


def test_production_bucket_keeps_real_aws_client() -> None:
    config = ServiceConfig(
        trial_s3_endpoint_url="http://127.0.0.1:4566",
        trial_s3_buckets="trial-video-bucket",
        s3_endpoint_url="",
    )
    trial = _make_s3_for_bucket(config, "trial-video-bucket")
    prod = _make_s3_for_bucket(config, "bjj-video-analysis")
    default = _make_s3(config)
    assert trial._endpoint_url == "http://127.0.0.1:4566"
    assert prod._endpoint_url is None
    assert default._endpoint_url is None


def test_sns_prefers_explicit_sns_endpoint_over_s3() -> None:
    config = ServiceConfig(
        sns_endpoint_url="http://127.0.0.1:4566",
        s3_endpoint_url="",
    )
    assert _sns_endpoint_url(config) == "http://127.0.0.1:4566"
    prod = ServiceConfig(sns_endpoint_url="", s3_endpoint_url="")
    assert _sns_endpoint_url(prod) is None


def test_real_aws_sns_arn_ignores_localstack_endpoint() -> None:
    config = ServiceConfig(
        sns_endpoint_url="http://127.0.0.1:4566",
        s3_endpoint_url="",
    )
    assert _sns_endpoint_url(
        config,
        "arn:aws:sns:us-east-1:026293068542:video_analysis_events",
    ) is None


def test_localstack_sns_arn_keeps_emulator_endpoint() -> None:
    config = ServiceConfig(
        sns_endpoint_url="http://127.0.0.1:4566",
        s3_endpoint_url="http://127.0.0.1:4566",
    )
    assert _sns_endpoint_url(
        config,
        "arn:aws:sns:us-east-1:000000000000:bjj-video-analysis-integration-events",
    ) == "http://127.0.0.1:4566"
