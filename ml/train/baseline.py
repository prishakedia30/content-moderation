"""TF-IDF + logistic regression baseline: the bar DistilBERT has to beat.
  python ml/train/baseline.py --data train.csv --out results/baseline"""
import argparse
import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import data
from metrics import classification

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results/baseline")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

train, val, test = data.split(data.load(a.data))
data.describe(train, val, test)

t0 = time.time()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
Xtr = vec.fit_transform(train["text"])
clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear")
clf.fit(Xtr, train["label"])
train_s = time.time() - t0

res = {"model": "TF-IDF (1-2 grams) + logistic regression", "train_seconds": round(train_s, 1)}
for name, d in [("val", val), ("test", test)]:
    p = clf.predict_proba(vec.transform(d["text"]))[:, 1]
    res[name] = classification(d["label"].values, p)
print(json.dumps(res, indent=2))
(out / "metrics.json").write_text(json.dumps(res, indent=2))
