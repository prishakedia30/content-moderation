"""Everything the Kaggle notebook needs, so its only cell is:

  !git clone -q https://github.com/prishakedia30/content-moderation.git && python content-moderation/ml/experiments/kaggle_run.py

Installs the extra packages, reads the AWS keys from Kaggle Secrets (only needed to upload model v2),
runs the experiment suite, the video benchmark, and writes the report."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = "/kaggle/working/results"


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


sh(sys.executable, "-m", "pip", "install", "-q", "onnx==1.17.0", "onnxruntime==1.20.1")
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    os.environ["AWS_ACCESS_KEY_ID"] = s.get_secret("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = s.get_secret("AWS_SECRET_ACCESS_KEY")
    upload = ["--upload-version", "v2"]
except Exception as e:   # no secrets attached: still run everything, just don't upload
    print("AWS secrets not available, model will not be uploaded:", e, flush=True)
    upload = []

run_all = [sys.executable, str(REPO / "ml" / "experiments" / "run_all.py"), "--data", "/kaggle/input", "--out", OUT]
sh(*run_all, *upload)
if sh(sys.executable, "-m", "pip", "install", "-q", "faster-whisper==1.1.0") == 0 and Path(OUT, "deploy").exists():
    sh(sys.executable, str(REPO / "ml" / "eval" / "video_bench.py"), "--model-dir", f"{OUT}/deploy", "--out", f"{OUT}/video")
sh(*run_all, "--only", "report")
print("finished; download results/results_bundle.zip from the Output tab", flush=True)
