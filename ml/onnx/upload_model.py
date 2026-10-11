"""Upload an exported model folder to S3 as models/<version>/ (what the text worker downloads).
On the server the instance role supplies credentials; on Kaggle, set AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY from Kaggle Secrets first.
  python upload_model.py --dir artifacts --version v1"""
import argparse
import os
from pathlib import Path

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--version", required=True)
ap.add_argument("--bucket", default="sap-moderation-858082489486")
ap.add_argument("--region", default="ap-southeast-2")
ap.add_argument("--skip-fp32", action="store_true", help="don't upload the 268 MB FP32 model")
a = ap.parse_args()

s3 = boto3.client("s3", region_name=a.region)
root = Path(a.dir)
need = [root / "model.int8.onnx", root / "tokenizer" / "tokenizer.json"]
missing = [str(p) for p in need if not p.exists()]
if missing:
    raise SystemExit(f"missing {missing}; run onnx_bench.py first")
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    if a.skip_fp32 and path.name == "model.onnx":
        continue
    key = f"models/{a.version}/{path.relative_to(root).as_posix()}"
    s3.upload_file(str(path), a.bucket, key)
    print(f"uploaded s3://{a.bucket}/{key} ({os.path.getsize(path) / 1e6:.1f} MB)")
