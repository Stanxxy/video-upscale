import boto3
import json
import logging
import os

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(
        self,
        region: str,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        self._endpoint_url = endpoint_url
        kwargs = {"region_name": region}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if access_key_id:
            kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            kwargs["aws_secret_access_key"] = secret_access_key
        self.client = boto3.client("s3", **kwargs)

    def ensure_bucket(self, bucket: str) -> None:
        """Create bucket if it doesn't exist (for LocalStack dev environments only)."""
        if not self._endpoint_url:
            return  # Real AWS bucket is pre-provisioned; skip auto-creation.
        try:
            self.client.head_bucket(Bucket=bucket)
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # 404 = bucket doesn't exist; 403 = exists but owned by another
            # account (common with LocalStack credential mismatches).
            if error_code in ("404", "NoSuchBucket"):
                logger.info("Creating bucket %s", bucket)
                try:
                    self.client.create_bucket(Bucket=bucket)
                except self.client.exceptions.ClientError as create_err:
                    create_code = create_err.response.get("Error", {}).get("Code", "")
                    if create_code == "BucketAlreadyExists":
                        logger.info("Bucket %s already exists, continuing", bucket)
                    else:
                        raise
            else:
                # 403 or other — bucket likely exists, continue optimistically
                logger.info("Bucket %s returned %s on head_bucket, assuming it exists", bucket, error_code)

    def download_file(self, bucket: str, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(bucket, key, local_path)
        return local_path

    def download_json(self, bucket: str, key: str) -> dict:
        """Download and parse a JSON object from S3."""
        resp = self.client.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())

    def upload_json(self, data: dict, bucket: str, key: str) -> str:
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"

    def upload_file(
        self,
        local_path: str,
        bucket: str,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a binary file (e.g. video) to S3."""
        self.client.upload_file(
            local_path, bucket, key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"s3://{bucket}/{key}"
