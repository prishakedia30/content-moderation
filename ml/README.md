# ML lane: train, calibrate, export, evaluate

Everything heavy runs on a **free Kaggle GPU notebook**; the finished model goes straight to S3,
where the text worker picks it up. Nothing large passes through your laptop.

```
Kaggle (GPU)                                                  AWS
baseline.py ─► finetune.py ─► onnx_bench.py ─► eval_onnx.py ─► upload_model.py ─► s3://…/models/v1/
 TF-IDF bar    DistilBERT +    ONNX + INT8      FP32 vs INT8     (int8 + tokenizer     text-worker
               calibration +   export           accuracy         + thresholds)         downloads it
               thresholds
```

| Script | What it produces | For the report |
|---|---|---|
| `train/baseline.py` | `metrics.json` | Baseline P / R / F1 / ROC-AUC / PR-AUC |
| `train/finetune.py` | `model/` (+ `thresholds.yaml`), `metrics.json`, `pr_curve.png`, `roc_curve.png`, `reliability.png`, `errors.csv` | Main results, calibration (temperature, ECE before/after), chosen thresholds and what share of posts each band gets, error analysis |
| `onnx/onnx_bench.py` | `artifacts/` (ONNX FP32 + INT8 + tokenizer), latency table | Size and latency (run on the **server** for real latency; already done for v0) |
| `eval/eval_onnx.py` | `eval_test.md` | INT8 vs FP32 accuracy on the test set |
| `eval/eval_spans.py` | table | Video: model only vs word list only vs both |

## Kaggle steps (about 40 minutes total)

1. On kaggle.com, open the competition **Jigsaw Toxic Comment Classification Challenge** and click **Join / I understand and accept** (needed to use its data).
2. **Create → New Notebook**. In the right panel: **Add Input** → search the competition → add it. **Settings → Accelerator: GPU T4 x2** (or P100). **Internet: On** (needs a verified phone number).
3. Add your AWS keys: **Add-ons → Secrets** → add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (the `prisha-admin` keys from your CSV). Tick both for this notebook.
4. Paste and run these cells, one at a time:

```python
!git clone https://github.com/prishakedia30/content-moderation.git
%cd content-moderation/ml
!pip install -q onnx onnxruntime==1.20.1 boto3
!ls /kaggle/input/jigsaw-toxic-comment-classification-challenge
```

```python
DATA = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
!python train/baseline.py --data $DATA --out /kaggle/working/baseline
```

```python
!python train/finetune.py --data $DATA --out /kaggle/working/run1
```

```python
!MODEL=/kaggle/working/run1/model OUT=/kaggle/working/artifacts python onnx/onnx_bench.py
!python eval/eval_onnx.py --model-dir /kaggle/working/artifacts --data $DATA --limit 4000
```

```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["AWS_ACCESS_KEY_ID"] = s.get_secret("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = s.get_secret("AWS_SECRET_ACCESS_KEY")
!python onnx/upload_model.py --dir /kaggle/working/artifacts --version v1 --skip-fp32
```

5. Download the result files for the report from the notebook's **Output** panel:
   `baseline/metrics.json`, `run1/metrics.json`, `run1/*.png`, `run1/errors.csv`, `artifacts/eval_test.md`.
6. Switch the platform to the new model: in `deploy/helm/moderation/values.yaml` set `MODEL_VERSION: v1`,
   commit, push, wait for GitHub Actions, then `./scripts/deploy.sh` (see RUNBOOK step 10).

Notes:
- The latency numbers `onnx_bench.py` prints on Kaggle are from Kaggle's CPU. Report the
  **server** latency table (`onnx/results.md`). The v1 model has the same architecture as v0, so
  its speed is the same.
- If the first cell's `ls` shows `train.csv` instead of `train.csv.zip`, change `DATA` to match.
- Quick dry run before the full training: add `--subset 5000` to `finetune.py`.
- Out of Kaggle GPU hours? The platform already runs on the public DistilBERT toxicity model (v0).
  Report that honestly and keep the baseline + calibration results.

## Video evaluation

Label a few transcripts (`eval/clips/*.json`, each word `{"start","end","text","toxic"}`; one sample
is included), then on the server:

```bash
python eval/eval_spans.py --clips eval/clips --classify-url http://<server-ip>/classify
```
