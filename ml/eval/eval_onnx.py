"""Does INT8 quantisation cost accuracy? Score the held-out test set with the FP32 and INT8
ONNX models and compare F1 / AUC / ALLOW-REVIEW-BLOCK decisions.
Run on the server (same venv as onnx_bench.py), after onnx_bench.py exported your model:
  python eval_onnx.py --model-dir ~/onnx/artifacts --data train.csv --limit 4000"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import data  # noqa: E402
from metrics import classification, softmax_pos  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--limit", type=int, default=4000, help="test rows to score (0 = all ~16K)")
a = ap.parse_args()
md = Path(a.model_dir).expanduser()

cfg = {"t_low": 0.30, "t_high": 0.75, "temperature": 1.0}
if (md / "thresholds.yaml").exists():
    cfg.update(yaml.safe_load((md / "thresholds.yaml").read_text()))
_, _, test = data.split(data.load(a.data))
if a.limit:
    test = test.sample(a.limit, random_state=data.SEED)
y = test["label"].values
tok = AutoTokenizer.from_pretrained(md / "tokenizer")


def run(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    logits, t0 = [], time.time()
    for text in test["text"].astype(str):
        enc = tok(text, truncation=True, max_length=128, return_tensors="np")
        logits.append(sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                      "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0])
    return np.array(logits), time.time() - t0


def bands(p):
    return np.where(p >= cfg["t_high"], 2, np.where(p >= cfg["t_low"], 1, 0))


rows, probs = [], {}
for name, file in [("ONNX FP32", "model.onnx"), ("ONNX INT8", "model.int8.onnx")]:
    logits, secs = run(md / file)
    p = softmax_pos(logits, cfg["temperature"])
    probs[name] = p
    m = classification(y, p)
    rows.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {1000 * secs / len(y):.1f} |")

agree = (bands(probs["ONNX FP32"]) == bands(probs["ONNX INT8"])).mean()
lines = [f"Test rows: {len(y)} ({y.mean():.1%} toxic) | thresholds {cfg}", "",
         "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | ms/post |",
         "|---|---|---|---|---|---|---|", *rows, "",
         f"FP32 and INT8 give the same ALLOW/REVIEW/BLOCK decision on {agree:.1%} of test posts; "
         f"max probability difference {np.abs(probs['ONNX FP32'] - probs['ONNX INT8']).max():.4f}."]
print("\n".join(lines))
(md / "eval_test.md").write_text("\n".join(lines) + "\n")
