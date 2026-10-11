"""File storage: S3 in the cloud, a local folder for docker-compose.
STORAGE=s3 (default) uses S3_BUCKET + AWS_REGION and the server's IAM role.
STORAGE=local uses LOCAL_STORAGE_DIR and serves files through the ingest API."""
import mimetypes
import os
import shutil
from pathlib import Path


class Storage:
    def __init__(self):
        self.mode = os.environ.get("STORAGE", "s3")
        if self.mode == "s3":
            import boto3
            from botocore.config import Config

            self.bucket = os.environ["S3_BUCKET"]
            region = os.environ.get("AWS_REGION", "ap-southeast-2")
            self.s3 = boto3.client(
                "s3", region_name=region,
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
            )
        elif self.mode == "local":
            self.root = Path(os.environ.get("LOCAL_STORAGE_DIR", "/data")).resolve()
            self.root.mkdir(parents=True, exist_ok=True)
            self.public = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")
        else:
            raise ValueError(f"unknown STORAGE={self.mode}")

    def local_path(self, key):
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("invalid key")
        return path

    def put_fileobj(self, fileobj, key):
        if self.mode == "s3":
            # A real content type lets browsers (Safari especially) play videos from presigned links.
            ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
            self.s3.upload_fileobj(fileobj, self.bucket, key, ExtraArgs={"ContentType": ctype})
        else:
            path = self.local_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as out:
                shutil.copyfileobj(fileobj, out)

    def put_file(self, src, key):
        with open(src, "rb") as f:
            self.put_fileobj(f, key)

    def get_file(self, key, dest):
        if self.mode == "s3":
            self.s3.download_file(self.bucket, key, str(dest))
        else:
            shutil.copyfile(self.local_path(key), dest)

    def exists(self, key):
        if self.mode == "s3":
            try:
                self.s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except self.s3.exceptions.ClientError:
                return False
        return self.local_path(key).exists()

    def url(self, key, expires_s=3600):
        """A link a browser can open: presigned for S3 (the bucket itself stays private)."""
        if self.mode == "s3":
            return self.s3.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s)
        return f"{self.public}/files/{key}"

    def download_prefix(self, prefix, dest, skip=()):
        """Copy every object under prefix into dest (used to fetch a model version).
        skip: file names to leave out (e.g. the unused 268 MB FP32 model)."""
        dest = Path(dest)
        if self.mode == "s3":
            paginator = self.s3.get_paginator("list_objects_v2")
            found = 0
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(prefix):].lstrip("/")
                    if not rel or rel in skip:
                        continue
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self.s3.download_file(self.bucket, obj["Key"], str(target))
                    found += 1
            if not found:
                raise FileNotFoundError(f"nothing under s3://{self.bucket}/{prefix}")
        else:
            shutil.copytree(self.local_path(prefix), dest, dirs_exist_ok=True)
