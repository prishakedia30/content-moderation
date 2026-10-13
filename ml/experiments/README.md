# Experiment suite

One resumable script runs every evaluation in the report: model comparison, 3 seeds, the sweep,
ablations, split comparison, adversarial robustness (with and without the text-worker's normalizer),
identity-term bias and its mitigation, out-of-domain tests, the cost/capacity threshold policy,
the error sample, and the export of the deployed model (v2).

Kaggle notebook, GPU T4, Internet on, AWS secrets attached (only needed for the upload):

```python
!git clone -q https://github.com/prishakedia30/content-moderation.git
!pip install -q onnx==1.17.0 onnxruntime==1.20.1
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["AWS_ACCESS_KEY_ID"] = s.get_secret("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = s.get_secret("AWS_SECRET_ACCESS_KEY")
!python content-moderation/ml/experiments/run_all.py --data /kaggle/input --out /kaggle/working/results --upload-version v2
```

Results: `results/REPORT.md`, `results/fig/*.png`, and everything small in `results/results_bundle.zip`.
A stage that finished is skipped on a rerun; delete its `<stage>.json` to redo it.
Smoke test without a GPU: add `--quick`.
