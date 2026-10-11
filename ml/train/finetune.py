"""Fine-tune distilbert-base-uncased for toxic / non-toxic, then calibrate and pick thresholds.
Built for a free Kaggle GPU (T4/P100): 1 epoch on ~128K comments takes roughly 10-20 minutes.

  python finetune.py --data /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip --out /kaggle/working/run1

Writes to --out:
  model/            fine-tuned weights + tokenizer + thresholds.yaml (give this folder to onnx_bench.py)
  metrics.json      test metrics, calibration, threshold policy
  *.png             PR curve, ROC curve, reliability diagram
  errors.csv        most confident mistakes, for the error-analysis section"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import data
from metrics import (choose_thresholds, classification, ece, fit_temperature, nll, policy,
                     softmax_pos)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="run")
ap.add_argument("--base", default="distilbert-base-uncased")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--subset", type=int, default=0, help="train on N rows only (quick test); 0 = all")
a = ap.parse_args()

torch.manual_seed(data.SEED)
out = Path(a.out)
(out / "model").mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

train, val, test = data.split(data.load(a.data))
if a.subset:
    train = train.sample(a.subset, random_state=data.SEED)
data.describe(train, val, test)

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForSequenceClassification.from_pretrained(
    a.base, num_labels=2, id2label={0: "non-toxic", 1: "toxic"}, label2id={"non-toxic": 0, "toxic": 1}
).to(device)


def loader(df, shuffle):
    rows = list(zip(df["text"].astype(str).tolist(), df["label"].astype(int).tolist()))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return DataLoader(rows, batch_size=a.bs if shuffle else a.bs * 2, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(df):
    model.eval()
    chunks = []
    for batch in tqdm(loader(df, False), desc="predict", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            chunks.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(chunks)


# Class weights: toxic comments are ~10% of the data, so errors on them cost more.
counts = np.bincount(train["label"], minlength=2)
weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

train_loader = loader(train, True)
steps = len(train_loader) * a.epochs
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

best_auc, best_state, t0 = -1.0, None, time.time()
for epoch in range(a.epochs):
    model.train()
    bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{a.epochs}")
    for step, batch in enumerate(bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            loss = loss_fn(model(**batch).logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 100 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}")
    val_auc = classification(val["label"].values, softmax_pos(predict(val)))["pr_auc"]
    print(f"epoch {epoch + 1}: val PR-AUC {val_auc:.4f}")
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
train_minutes = (time.time() - t0) / 60
model.load_state_dict(best_state)

# Calibration (on validation) and threshold choice (on calibrated validation probabilities).
val_logits, test_logits = predict(val), predict(test)
yv, yt = val["label"].values, test["label"].values
T = fit_temperature(val_logits, yv)
pv_raw, pv = softmax_pos(val_logits), softmax_pos(val_logits, T)
pt_raw, pt = softmax_pos(test_logits), softmax_pos(test_logits, T)
t_low, t_high = choose_thresholds(yv, pv)

res = {
    "model": f"{a.base} fine-tuned, {a.epochs} epoch(s), lr {a.lr}, batch {a.bs}, max_len {a.max_len}",
    "train_rows": len(train), "train_minutes": round(train_minutes, 1),
    "test": classification(yt, pt),
    "calibration": {"temperature": T,
                    "test_ece_before": ece(yt, pt_raw), "test_ece_after": ece(yt, pt),
                    "test_nll_before": nll(yt, pt_raw), "test_nll_after": nll(yt, pt)},
    "thresholds": {"t_low": t_low, "t_high": t_high},
    "test_policy": policy(yt, pt, t_low, t_high),
}
print(json.dumps(res, indent=2))
(out / "metrics.json").write_text(json.dumps(res, indent=2))

model.save_pretrained(out / "model")
tok.save_pretrained(out / "model")
(out / "model" / "thresholds.yaml").write_text(
    f"# chosen on the validation set by finetune.py\nt_low: {t_low}\nt_high: {t_high}\ntemperature: {T}\n")

# Error analysis: the model's most confident mistakes.
test = test.assign(prob=pt)
fp = test[(test.label == 0)].nlargest(25, "prob").assign(error="false positive")
fn = test[(test.label == 1)].nsmallest(25, "prob").assign(error="false negative")
import pandas as pd
pd.concat([fp, fn]).to_csv(out / "errors.csv", index=False)

# Plots for the report.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    prec, rec, _ = precision_recall_curve(yt, pt)
    plt.figure(figsize=(5, 4)); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve (AP {res['test']['pr_auc']:.3f})")
    plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(yt, pt)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC (AUC {res['test']['roc_auc']:.3f})"); plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    bins = np.linspace(0, 1, 11)
    plt.figure(figsize=(5, 4)); plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    for p, lab in [(pt_raw, "before"), (pt, f"after (T={T:.2f})")]:
        idx = np.digitize(p, bins[1:-1])
        xs = [p[idx == i].mean() for i in range(10) if (idx == i).any()]
        ys = [yt[idx == i].mean() for i in range(10) if (idx == i).any()]
        plt.plot(xs, ys, marker="o", label=lab)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed toxic rate")
    plt.title("Reliability diagram"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "reliability.png", dpi=150); plt.close()
except ImportError:
    print("matplotlib not installed; skipped plots")
print("saved to", out)
