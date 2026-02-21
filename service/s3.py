import boto3
import json
import os


class S3Client:
    def __init__(self, region: str):
        self.client = boto3.client("s3", region_name=region)

    def download_file(self, bucket: str, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(bucket, key, local_path)
        return local_path

    def upload_json(self, data: dict, bucket: str, key: str) -> str:
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"
