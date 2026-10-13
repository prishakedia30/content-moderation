"""The whole ML evaluation suite in one resumable script (Kaggle GPU, ~2-3 hours).

  python ml/experiments/run_all.py --data /kaggle/input/<jigsaw folder> --out /kaggle/working/results

Each stage writes <out>/<stage>.json and is skipped when that file exists, so a rerun continues where
the last one stopped. --only sweep,baselines runs just those stages; --quick shrinks everything
for a smoke test.

Stages
  setup        splits (identical to v1's), official competition test set
  baselines    TF-IDF word n-grams + LR, TF-IDF char n-grams + LR (binary and per category)
  v0           the public pre-trained model we started with (martin-ha/toxic-comment-model), no training
  sweep        lr {1e-5, 2e-5, 5e-5} x epochs {1, 2, 3} on a 40k training subset -> picks lr / epochs
  seeds        multi-label DistilBERT (the main model) with 3 seeds: mean +- std
  binary       binary DistilBERT (the v1 recipe) for comparison
  roberta      RoBERTa-base multi-label: the accuracy ceiling
  ablations    class weights on/off, max_len 64/128/256 (incl. long comments), calibration on/off
  splits       random stratified test set vs the official competition test set
  robustness   leetspeak / stretched / spaced / punctuation / look-alike / zero-width attacks,
               recall before and after the text-worker's normalizer
  bias         false-positive rate per identity group; counterfactual-augmentation mitigation
  ood          out-of-domain: Civil Comments, TweetEval-offensive
  comparison   one table: quality, size, CPU latency for every model
  policy       cost model and review-capacity thresholds
  errors       100 most confident mistakes for hand-labelling
  export       deploy model -> ONNX INT8 + thresholds.yaml (= model v2); optional S3 upload
  report       REPORT.md with every table and figure
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

STAGES = ["setup", "baselines", "v0", "sweep", "seeds", "binary", "roberta", "ablations", "splits",
          "robustness", "bias", "ood", "comparison", "policy", "errors", "export", "report"]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder (or train.csv) of the Jigsaw dataset")
ap.add_argument("--out", default="results")
ap.add_argument("--only", default="", help="comma-separated stages")
ap.add_argument("--skip", default="", help="comma-separated stages to skip")
ap.add_argument("--quick", action="store_true", help="tiny smoke test")
ap.add_argument("--distil", default="distilbert-base-uncased")
ap.add_argument("--large", default="roberta-base")
ap.add_argument("--v0", default="martin-ha/toxic-comment-model")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--upload-version", default="", help="e.g. v2: upload the exported model to S3")
ap.add_argument("--review-capacity", type=float, default=0.05)
a = ap.parse_args()

OUT = Path(a.out)
(OUT / "fig").mkdir(parents=True, exist_ok=True)
(OUT / "cache").mkdir(exist_ok=True)
LOG = open(OUT / "log.txt", "a")


def log(*msg):
    line = time.strftime("%H:%M:%S ") + " ".join(str(m) for m in msg)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cache_np(name, arr=None):
    p = OUT / "cache" / f"{name}.npy"
    if arr is not None:
        np.save(p, arr)
        return arr
    return np.load(p) if p.exists() else None


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    pyplot.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                            "axes.spines.right": False})
    return pyplot


INK, BLUE, GREEN, AMBER, RED, GREY = "#1f2a36", "#3346a8", "#2f7d5b", "#b7791f", "#b4263a", "#8a96a3"

# ---------------------------------------------------------------- data (always loaded)
data_path = Path(a.data)
train_csv = data_path if data_path.is_file() else C.find_file(data_path, "train.csv", "train.csv.zip")
if train_csv is None:
    sys.exit(f"no train.csv under {data_path}")
DF = C.load_jigsaw(train_csv)
if a.quick:
    DF = C.stratified_sample(DF, 3000).sort_values("id").reset_index(drop=True)
TRAIN, VAL, TEST = C.split(DF)
OFFICIAL = None if data_path.is_file() else C.load_official_test(data_path)
SUB = C.stratified_sample(TRAIN, 1000 if a.quick else 40000)
EPOCH_CAP = 1 if a.quick else 3
MAX_LEN = 128
log(f"train {len(TRAIN)}, val {len(VAL)}, test {len(TEST)}, sweep subset {len(SUB)}, "
    f"official test {0 if OFFICIAL is None else len(OFFICIAL)}, device {C.DEVICE}")
YV, YT = VAL["label"].values, TEST["label"].values
YV6, YT6 = VAL[C.LABELS].values, TEST[C.LABELS].values
LAT_TEXTS = TEST.sample(min(len(TEST), 200), random_state=1)["text"].astype(str).tolist()

MODELS = {}   # name -> (model, tok, kind) kept in memory between stages of the same run


def best_config():
    s = load("sweep")
    if s and "best" in s:
        return s["best"]["lr"], min(int(s["best"]["epochs"]), 2)
    return 2e-5, 1


def summarize(kind, val_logits, test_logits, T=None):
    """Metrics for one transformer run: binary any-toxic on the main score, plus per-category."""
    if T is None:
        T = C.fit_temperature_binary(val_logits, YV) if kind == "binary" else C.fit_temperatures_multi(val_logits, YV6)
    pt = C.probs(test_logits, kind, T)
    st = C.main_score(pt, kind)
    out = {"test": C.binary_metrics(YT, st), "temperature": T}
    if kind == "multi":
        out["per_class"] = C.per_class(YT6, pt)
        out["macro_pr_auc"] = float(np.nanmean([r["pr_auc"] for r in out["per_class"]]))
    return out


def get_model(name):
    """Re-load a trained model saved by an earlier stage (so stages can run in separate sessions)."""
    if name in MODELS:
        return MODELS[name]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = OUT / "models" / name
    if not d.exists():
        return None
    m = AutoModelForSequenceClassification.from_pretrained(d).to(C.DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(d)
    kind = "multi" if m.config.num_labels == 6 else "binary"
    MODELS[name] = (m, tok, kind)
    return MODELS[name]


def keep_model(name, model, tok, kind):
    d = OUT / "models" / name
    model.save_pretrained(d)
    tok.save_pretrained(d)
    MODELS[name] = (model, tok, kind)


def scorer(name, T=None):
    """texts -> calibrated main score, for a stored transformer model."""
    model, tok, kind = get_model(name)
    if T is None:
        T = load(f"model_{name}")["temperature"]

    def fn(texts, max_len=MAX_LEN):
        return C.main_score(C.probs(C.predict_logits(model, tok, list(texts), max_len), kind, T), kind)
    return fn


# ---------------------------------------------------------------- stages

def stage_setup():
    res = {"rows": {"train": len(TRAIN), "val": len(VAL), "test": len(TEST),
                    "official_test": 0 if OFFICIAL is None else len(OFFICIAL)},
           "toxic_rate": {"train": float(TRAIN.label.mean()), "test": float(TEST.label.mean())},
           "category_rate_train": {l: float(TRAIN[l].mean()) for l in C.LABELS},
           "category_count_train": {l: int(TRAIN[l].sum()) for l in C.LABELS}}
    save("setup", res)


def _tfidf(kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300_000,
                           sublinear_tf=True)


def stage_baselines():
    from sklearn.linear_model import LogisticRegression
    res = {}
    for kind in ["word", "char"]:
        t0 = time.time()
        vec = _tfidf(kind)
        Xtr = vec.fit_transform(TRAIN["text"].astype(str))
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear")
        clf.fit(Xtr, TRAIN["label"])
        Xv, Xt = vec.transform(VAL["text"].astype(str)), vec.transform(TEST["text"].astype(str))
        pv, pt = clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xt)[:, 1]
        # per category: one logistic regression per label
        per = np.zeros_like(YT6, dtype=float)
        for j, lab in enumerate(C.LABELS):
            cj = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear")
            cj.fit(Xtr, TRAIN[lab])
            per[:, j] = cj.predict_proba(Xt)[:, 1]
        name = f"tfidf_{kind}"
        path = OUT / "models" / f"{name}.pkl"
        path.parent.mkdir(exist_ok=True)
        pickle.dump((vec, clf), open(path, "wb"))
        lat = C.latency(lambda t: clf.predict_proba(vec.transform([t])), LAT_TEXTS)
        cache_np(f"{name}_val", pv), cache_np(f"{name}_test", pt)
        res[name] = {"test": C.binary_metrics(YT, pt), "val": C.binary_metrics(YV, pv),
                     "per_class": C.per_class(YT6, per),
                     "size_mb": round(path.stat().st_size / 1e6, 1), "p50_ms": lat[0], "p95_ms": lat[1],
                     "train_minutes": round((time.time() - t0) / 60, 1)}
        log(name, json.dumps(res[name]["test"]))
    save("baselines", res)


def stage_v0():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.v0)
    m = AutoModelForSequenceClassification.from_pretrained(a.v0).to(C.DEVICE).eval()
    lab = {i: str(l).lower() for i, l in m.config.id2label.items()}
    tox = next((i for i, l in lab.items() if "toxic" in l and "non" not in l), 1)

    def score(texts):
        lg = C.predict_logits(m, tok, list(texts), MAX_LEN)
        e = np.exp(lg - lg.max(1, keepdims=True))
        return e[:, tox] / e.sum(1)
    pt = score(TEST["text"])
    cache_np("v0_test", pt)
    paths = C.export_onnx(m, tok, OUT / "onnx" / "v0")
    fn, _ = C.onnx_scorer(paths["int8"], tok)
    lat = C.latency(fn, LAT_TEXTS)
    MODELS["v0"] = (m, tok, "v0")
    res = {"test": C.binary_metrics(YT, pt), "toxic_index": tox, "labels": lab,
           "size_mb_int8": round(paths["int8"].stat().st_size / 1e6, 1), "p50_ms": lat[0], "p95_ms": lat[1]}
    log("v0", json.dumps(res["test"]))
    save("v0", res)


def stage_sweep():
    lrs = [1e-5, 2e-5, 5e-5]
    rows = []
    for lr in lrs:
        _, _, hist, mins = C.train_model(a.distil, "multi", SUB, VAL, epochs=EPOCH_CAP, lr=lr, bs=a.bs,
                                         max_len=MAX_LEN, log=log)
        for h in hist:
            rows.append({"lr": lr, "epochs": h["epoch"], "val_pr_auc": h["val_pr_auc"]})
        log("sweep lr", lr, hist, f"{mins:.1f} min")
        save("sweep_partial", rows)
    best = max(rows, key=lambda r: r["val_pr_auc"])
    save("sweep", {"rows": rows, "best": best, "subset_rows": len(SUB),
                   "note": "epochs = evaluation after each epoch of a 3-epoch linear-decay schedule"})
    p = plt()
    fig, ax = p.subplots(figsize=(5.2, 3.6))
    for lr, col in zip(lrs, [GREY, BLUE, RED]):
        r = [x for x in rows if x["lr"] == lr]
        ax.plot([x["epochs"] for x in r], [x["val_pr_auc"] for x in r], marker="o", color=col, label=f"lr {lr:g}")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Validation PR-AUC"), ax.set_xticks(range(1, EPOCH_CAP + 1))
    ax.set_title(f"Learning-rate x epochs sweep ({len(SUB):,} training rows)")
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "sweep.png"), p.close(fig)


def _train_and_record(name, base, kind, seed=C.SEED, train_df=None, extra=None, keep=True, epochs=None, lr=None):
    blr, bep = best_config()
    lr, epochs = lr or blr, epochs or bep
    model, tok, hist, mins = C.train_model(base, kind, TRAIN if train_df is None else train_df, VAL,
                                           epochs=epochs, lr=lr, bs=a.bs, max_len=MAX_LEN, seed=seed,
                                           log=log, extra_train=extra)
    vl = C.predict_logits(model, tok, VAL["text"].tolist(), MAX_LEN)
    tl = C.predict_logits(model, tok, TEST["text"].tolist(), MAX_LEN)
    cache_np(f"{name}_val_logits", vl), cache_np(f"{name}_test_logits", tl)
    res = summarize(kind, vl, tl)
    res.update({"base": base, "kind": kind, "seed": seed, "lr": lr, "epochs": epochs,
                "history": hist, "train_minutes": round(mins, 1)})
    save(f"model_{name}", res)
    if keep:
        keep_model(name, model, tok, kind)
    log(name, json.dumps(res["test"]))
    return model, tok, res


def stage_seeds():
    runs = []
    for seed in [42, 43, 44]:
        name = "distil_multi" if seed == 42 else f"distil_multi_s{seed}"
        prev = load(f"model_{name}")
        if prev is None:
            _, _, prev = _train_and_record(name, a.distil, "multi", seed=seed, keep=seed == 42)
            if seed != 42:
                import torch
                MODELS.pop(name, None)
                torch.cuda.empty_cache()
        runs.append(prev)
    keys = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    agg = {k: {"mean": float(np.mean([r["test"][k] for r in runs])), "std": float(np.std([r["test"][k] for r in runs], ddof=1))}
           for k in keys}
    agg["macro_pr_auc"] = {"mean": float(np.mean([r["macro_pr_auc"] for r in runs])),
                           "std": float(np.std([r["macro_pr_auc"] for r in runs], ddof=1))}
    per = {lab: {"mean": float(np.mean([[c for c in r["per_class"] if c["label"] == lab][0]["pr_auc"] for r in runs])),
                 "std": float(np.std([[c for c in r["per_class"] if c["label"] == lab][0]["pr_auc"] for r in runs], ddof=1))}
           for lab in C.LABELS}
    save("seeds", {"seeds": [42, 43, 44], "aggregate": agg, "per_class_pr_auc": per,
                   "runs": [r["test"] for r in runs]})
    # per-category figure: PR-AUC vs prevalence (rare classes are the interesting part)
    base = load("baselines") or {}
    setup = load("setup")
    p = plt()
    fig, ax = p.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(C.LABELS))
    w = 0.27
    ax.bar(x - w, [per[l]["mean"] for l in C.LABELS], w, yerr=[per[l]["std"] for l in C.LABELS],
           color=BLUE, label="DistilBERT multi-label (3 seeds)")
    if "tfidf_char" in base:
        ax.bar(x, [c["pr_auc"] for c in base["tfidf_char"]["per_class"]], w, color=GREEN, label="Char TF-IDF + LR")
    if "tfidf_word" in base:
        ax.bar(x + w, [c["pr_auc"] for c in base["tfidf_word"]["per_class"]], w, color=GREY, label="Word TF-IDF + LR")
    ax.set_xticks(x, [f"{l.replace('_', ' ')}\n{setup['category_rate_train'][l]:.1%}" for l in C.LABELS], fontsize=8)
    ax.set_ylabel("Test PR-AUC"), ax.set_ylim(0, 1)
    ax.set_title("Per-category PR-AUC (label: share of training comments)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "per_class.png"), p.close(fig)


def stage_binary():
    _train_and_record("distil_binary", a.distil, "binary", keep=True)


def stage_roberta():
    _train_and_record("roberta_multi", a.large, "multi", keep=True)


def stage_ablations():
    import torch
    res = {"subset_rows": len(SUB)}
    lr, epochs = best_config()
    # class weighting and max_len, trained on the subset
    long_mask = None
    for tag, kw in [("weighted_len128", {}), ("plain_len128", {"weighted": False}),
                    ("weighted_len64", {"max_len": 64}), ("weighted_len256", {"max_len": 256})]:
        if load(f"abl_{tag}"):
            res[tag] = load(f"abl_{tag}")
            continue
        ml = kw.get("max_len", MAX_LEN)
        model, tok, hist, mins = C.train_model(a.distil, "multi", SUB, VAL, epochs=epochs, lr=lr, bs=a.bs,
                                               max_len=ml, weighted=kw.get("weighted", True), log=log)
        vl = C.predict_logits(model, tok, VAL["text"].tolist(), ml)
        tl = C.predict_logits(model, tok, TEST["text"].tolist(), ml)
        r = summarize("multi", vl, tl)
        if long_mask is None:
            lens = np.array([len(x) for x in tok(TEST["text"].astype(str).tolist(), truncation=False)["input_ids"]])
            long_mask = lens > 128
            res["long_comments"] = {"share_over_128_tokens": float(long_mask.mean()), "rows": int(long_mask.sum())}
        st = C.main_score(C.probs(tl, "multi", r["temperature"]), "multi")
        r["long_test_pr_auc"] = C.binary_metrics(YT[long_mask], st[long_mask])["pr_auc"]
        r["short_test_pr_auc"] = C.binary_metrics(YT[~long_mask], st[~long_mask])["pr_auc"]
        r["train_minutes"] = round(mins, 1)
        save(f"abl_{tag}", r)
        res[tag] = r
        log("ablation", tag, json.dumps(r["test"]))
        del model
        torch.cuda.empty_cache()
    # calibration on/off for the main model (full data)
    m = load("model_distil_multi")
    if m:
        vl, tl = cache_np("distil_multi_val_logits"), cache_np("distil_multi_test_logits")
        from metrics import choose_thresholds, policy
        cal = {}
        for tag, T in [("uncalibrated", [1.0] * 6), ("calibrated", m["temperature"])]:
            pv = C.main_score(C.probs(vl, "multi", T), "multi")
            pt = C.main_score(C.probs(tl, "multi", T), "multi")
            lo, hi = choose_thresholds(YV, pv)
            cal[tag] = {"temperatures": T, "test_ece": C.ece(YT, pt), "test_log_loss": C.bce(YT, pt),
                        "per_class_ece": {l: C.ece(YT6[:, j], C.probs(tl, "multi", T)[:, j]) for j, l in enumerate(C.LABELS)},
                        "thresholds": [lo, hi], "test_policy": policy(YT, pt, lo, hi)}
        res["calibration"] = cal
        p = plt()
        fig, ax = p.subplots(figsize=(4.4, 4))
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
        bins = np.linspace(0, 1, 11)
        for tag, col in [("uncalibrated", RED), ("calibrated", BLUE)]:
            pt = C.main_score(C.probs(tl, "multi", cal[tag]["temperatures"]), "multi")
            idx = np.digitize(pt, bins[1:-1])
            xs = [pt[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
            ys = [YT[idx == i].mean() for i in range(10) if (idx == i).sum() > 20]
            ax.plot(xs, ys, marker="o", color=col, label=f"{tag} (ECE {cal[tag]['test_ece']:.3f})")
        ax.set_xlabel("Predicted probability"), ax.set_ylabel("Observed toxic rate")
        ax.set_title("Reliability, main score"), ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(), fig.savefig(OUT / "fig" / "reliability.png"), p.close(fig)
    save("ablations", res)


def stage_splits():
    if OFFICIAL is None:
        save("splits", {"note": "official competition test set not found next to train.csv"})
        return
    res = {}
    for name in ["distil_multi", "distil_binary", "roberta_multi"]:
        if get_model(name) is None:
            continue
        fn = scorer(name)
        res[name] = {"random_split_test": load(f"model_{name}")["test"],
                     "official_test": C.binary_metrics(OFFICIAL["label"].values, fn(OFFICIAL["text"]))}
    for kind in ["word", "char"]:
        path = OUT / "models" / f"tfidf_{kind}.pkl"
        if path.exists():
            vec, clf = pickle.load(open(path, "rb"))
            res[f"tfidf_{kind}"] = {"random_split_test": load("baselines")[f"tfidf_{kind}"]["test"],
                                    "official_test": C.binary_metrics(OFFICIAL["label"].values,
                                                                      clf.predict_proba(vec.transform(OFFICIAL["text"].astype(str)))[:, 1])}
    res["note"] = ("Jigsaw has no timestamps or topic labels, so a time- or topic-based split is not possible. "
                   "The official competition test set is a separately sampled and separately labelled set, "
                   "used here as the second, harder split.")
    res["official_rows"], res["official_toxic_rate"] = len(OFFICIAL), float(OFFICIAL["label"].mean())
    save("splits", res)


def _threshold(name):
    """The model's own thresholds from the original 95/95 rule, on its validation scores
    (used by the robustness and bias analyses, so both use one fixed, documented operating point)."""
    from metrics import choose_thresholds
    vl = cache_np(f"{name}_val_logits")
    m = load(f"model_{name}")
    return choose_thresholds(YV, C.main_score(C.probs(vl, m["kind"], m["temperature"]), m["kind"]))


def stage_robustness():
    from normalize import Normalizer, vocab_from_tokenizer
    name = "distil_multi"
    model, tok, kind = get_model(name)
    fn = scorer(name)
    norm = Normalizer(vocab_from_tokenizer(tok))
    t_low, t_high = _threshold(name)

    tox = TEST[(TEST.label == 1) & TEST["text"].astype(str).map(C.has_attack_word)]
    tox = tox.sample(min(len(tox), 300 if a.quick else 3000), random_state=1)
    clean = TEST[TEST.label == 0].sample(min((TEST.label == 0).sum(), 500 if a.quick else 5000), random_state=1)

    def production(texts):
        """Exactly what the text-worker does: score raw and normalized text, keep the higher."""
        texts = list(texts)
        normed = [norm(t) for t in texts]
        raw = fn(texts)
        changed = [i for i, (t, n) in enumerate(zip(texts, normed)) if n != " ".join(t.split())]
        out = raw.copy()
        if changed:
            out[changed] = np.maximum(raw[changed], fn([normed[i] for i in changed]))
        return out, len(changed)

    char = None
    if (OUT / "models" / "tfidf_char.pkl").exists():
        vec, clf = pickle.load(open(OUT / "models" / "tfidf_char.pkl", "rb"))
        char = lambda texts: clf.predict_proba(vec.transform(list(texts)))[:, 1]  # noqa: E731
        char_t = 0.5

    rows = []
    for kind_ in ["none"] + C.PERTURBATIONS:
        texts = tox["text"].astype(str).tolist() if kind_ == "none" else \
            [C.perturb_text(t, kind_, seed=i) for i, t in enumerate(tox["text"].astype(str))]
        before = fn(texts)
        after, n_changed = production(texts)
        r = {"attack": kind_, "rows": len(texts),
             "caught_before": float((before >= t_low).mean()), "caught_after": float((after >= t_low).mean()),
             "blocked_before": float((before >= t_high).mean()), "blocked_after": float((after >= t_high).mean()),
             "normalizer_changed": n_changed / len(texts)}
        if char is not None:
            r["char_tfidf_recall@0.5"] = float((char(texts) >= char_t).mean())
        rows.append(r)
        log("robustness", json.dumps(r))
    s_raw = fn(clean["text"])
    s_prod, n_changed = production(clean["text"])
    clean_res = {"rows": len(clean), "normalizer_changed_share": n_changed / len(clean),
                 "flag_rate_before": float((s_raw >= t_low).mean()), "flag_rate_after": float((s_prod >= t_low).mean()),
                 "block_rate_before": float((s_raw >= t_high).mean()), "block_rate_after": float((s_prod >= t_high).mean())}
    examples = [{"attack": k, "text": C.perturb_text("You are a stupid idiot and a loser", k, seed=3)} for k in C.PERTURBATIONS]
    save("robustness", {"thresholds": [t_low, t_high], "toxic_rows": rows, "clean": clean_res, "examples": examples,
                        "definition": "caught = score >= t_low (sent to REVIEW or BLOCK); blocked = score >= t_high"})
    p = plt()
    fig, ax = p.subplots(figsize=(7, 3.6))
    labels = [r["attack"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [r["caught_before"] for r in rows], w, color=GREY, label="model alone")
    ax.bar(x + w / 2, [r["caught_after"] for r in rows], w, color=BLUE, label="with normalizer")
    ax.set_xticks(x, labels, fontsize=8, rotation=15), ax.set_ylim(0, 1.05)
    ax.set_ylabel("Toxic comments caught"), ax.set_title("Disguised abuse: recall before and after the normalizer")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "robustness.png"), p.close(fig)


def _bias_eval(fn, t_low):
    ev = C.identity_eval_set()
    ev["score"] = fn(ev["text"])
    rows = []
    for g, d in ev.groupby("group", sort=False):
        nt, tx = d[d.label == 0], d[d.label == 1]
        rows.append({"group": g, "fpr": float((nt.score >= t_low).mean()), "fpr@0.5": float((nt.score >= 0.5).mean()),
                     "mean_score_nontoxic": float(nt.score.mean()), "tpr": float((tx.score >= t_low).mean())})
    fprs = [r["fpr"] for r in rows if r["group"] != C.CONTROL_GROUP]
    means = [r["mean_score_nontoxic"] for r in rows if r["group"] != C.CONTROL_GROUP]
    summary = {"mean_fpr": float(np.mean(fprs)), "max_fpr": float(np.max(fprs)), "fpr_gap": float(np.max(fprs) - np.min(fprs)),
               "control_fpr": [r["fpr"] for r in rows if r["group"] == C.CONTROL_GROUP][0],
               "mean_nontoxic_score": float(np.mean(means)), "mean_tpr": float(np.mean([r["tpr"] for r in rows]))}
    return rows, summary


def stage_bias():
    t_low, _ = _threshold("distil_multi")
    before_rows, before = _bias_eval(scorer("distil_multi"), t_low)
    save("bias_partial", {"before": before_rows})
    if load("model_distil_multi_debiased") is None:
        aug, info = C.counterfactual_augment(TRAIN)
        save("bias_augmentation", info)
        _train_and_record("distil_multi_debiased", a.distil, "multi", extra=aug, keep=True)
    m = load("model_distil_multi_debiased")
    vl = cache_np("distil_multi_debiased_val_logits")
    from metrics import choose_thresholds
    t_low_d, _ = choose_thresholds(YV, C.main_score(C.probs(vl, "multi", m["temperature"]), "multi"))
    after_rows, after = _bias_eval(scorer("distil_multi_debiased"), t_low_d)
    res = {"t_low_before": t_low, "t_low_after": t_low_d, "before": before_rows, "after": after_rows,
           "summary_before": before, "summary_after": after, "augmentation": load("bias_augmentation"),
           "test_before": load("model_distil_multi")["test"], "test_after": m["test"],
           "definition": "fpr = share of clearly non-toxic template sentences scored >= t_low (sent to review or blocked)"}
    save("bias", res)
    p = plt()
    fig, ax = p.subplots(figsize=(8, 3.8))
    groups = [r["group"] for r in before_rows]
    x = np.arange(len(groups))
    w = 0.4
    ax.bar(x - w / 2, [r["fpr"] for r in before_rows], w, color=RED, label="before (main model)")
    ax.bar(x + w / 2, [r["fpr"] for r in after_rows], w, color=BLUE, label="after counterfactual augmentation")
    ax.set_xticks(x, groups, rotation=60, fontsize=8), ax.set_ylim(0, 1)
    ax.set_ylabel("False-positive rate"), ax.set_title("Neutral sentences that mention an identity, wrongly flagged")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "bias.png"), p.close(fig)


def stage_ood():
    sets = {}
    try:
        from datasets import load_dataset
        cc = load_dataset("google/civil_comments", split="test").to_pandas()
        cc = cc.sample(min(len(cc), 2000 if a.quick else 20000), random_state=1)
        sets["civil_comments"] = pd.DataFrame({"text": cc["text"], "label": (cc["toxicity"] >= 0.5).astype(int)})
    except Exception as e:
        log("civil comments unavailable:", e)
    try:
        from datasets import load_dataset
        te = load_dataset("cardiffnlp/tweet_eval", "offensive", split="test").to_pandas()
        sets["tweet_eval_offensive"] = pd.DataFrame({"text": te["text"], "label": te["label"].astype(int)})
    except Exception as e:
        log("tweet_eval unavailable:", e)
    res = {"datasets": {k: {"rows": len(v), "toxic_rate": float(v.label.mean())} for k, v in sets.items()}}
    scorers = {"distil_multi": scorer("distil_multi")}
    if get_model("roberta_multi"):
        scorers["roberta_multi"] = scorer("roberta_multi")
    if (OUT / "models" / "tfidf_char.pkl").exists():
        vec, clf = pickle.load(open(OUT / "models" / "tfidf_char.pkl", "rb"))
        scorers["tfidf_char"] = lambda t: clf.predict_proba(vec.transform(list(map(str, t))))[:, 1]
    if True:   # the zero-training reference model, for comparison
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if "v0" not in MODELS:
                MODELS["v0"] = (AutoModelForSequenceClassification.from_pretrained(a.v0).to(C.DEVICE).eval(),
                                AutoTokenizer.from_pretrained(a.v0), "v0")
            vm, vt, _ = MODELS["v0"]
            tox = load("v0")["toxic_index"] if load("v0") else 1

            def v0fn(t):
                lg = C.predict_logits(vm, vt, list(t), MAX_LEN)
                e = np.exp(lg - lg.max(1, keepdims=True))
                return e[:, int(tox)] / e.sum(1)
            scorers["v0"] = v0fn
        except Exception as e:
            log("v0 unavailable for ood:", e)
    rows = []
    for mname, fn in scorers.items():
        rows.append({"model": mname, "dataset": "jigsaw (in-domain test)", **C.binary_metrics(YT, fn(TEST["text"]))})
        for dname, d in sets.items():
            rows.append({"model": mname, "dataset": dname, **C.binary_metrics(d["label"].values, fn(d["text"]))})
        log("ood", mname, "done")
    res["rows"] = rows
    save("ood", res)
    p = plt()
    fig, ax = p.subplots(figsize=(6.4, 3.4))
    dnames = ["jigsaw (in-domain test)"] + list(sets)
    mnames = list(scorers)
    x = np.arange(len(dnames))
    w = 0.8 / len(mnames)
    for i, (mn, col) in enumerate(zip(mnames, [BLUE, RED, GREEN, GREY])):
        ax.bar(x + (i - (len(mnames) - 1) / 2) * w,
               [next(r["pr_auc"] for r in rows if r["model"] == mn and r["dataset"] == dn) for dn in dnames], w,
               color=col, label=mn.replace("_", " "))
    ax.set_xticks(x, [d.replace("_", " ") for d in dnames], fontsize=8), ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC"), ax.set_title("In-domain vs out-of-domain"), ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "ood.png"), p.close(fig)


def stage_comparison():
    """Quality + size + single-text CPU latency for every model, on the same machine and test set."""
    import torch
    rows = []
    base = load("baselines") or {}
    for k, label in [("tfidf_word", "TF-IDF word n-grams + LR"), ("tfidf_char", "TF-IDF char n-grams + LR")]:
        if k in base:
            b = base[k]
            rows.append({"model": label, "training": "ours", **b["test"], "size_mb": b["size_mb"],
                         "p50_ms": b["p50_ms"], "p95_ms": b["p95_ms"], "backend": "scikit-learn"})
    v0 = load("v0")
    if v0:
        rows.append({"model": "v0: martin-ha/toxic-comment-model", "training": "none (public model)", **v0["test"],
                     "size_mb": v0["size_mb_int8"], "p50_ms": v0["p50_ms"], "p95_ms": v0["p95_ms"], "backend": "ONNX INT8"})
    for name, label in [("distil_binary", "DistilBERT binary (v1 recipe)"),
                        ("distil_multi", "DistilBERT multi-label"),
                        ("distil_multi_debiased", "DistilBERT multi-label + debiasing"),
                        ("roberta_multi", "RoBERTa-base multi-label")]:
        m = load(f"model_{name}")
        if not m or get_model(name) is None:
            continue
        model, tok, kind = get_model(name)
        paths = C.export_onnx(model, tok, OUT / "onnx" / name)
        tfn, tm = C.torch_cpu_scorer(model, tok)
        p50, p95 = C.latency(tfn, LAT_TEXTS)
        rows.append({"model": label, "training": "ours", **m["test"], "size_mb": round(C.dir_mb(OUT / "models" / name), 1),
                     "p50_ms": p50, "p95_ms": p95, "backend": "PyTorch FP32"})
        del tm
        ofn, sess = C.onnx_scorer(paths["int8"], tok)
        p50, p95 = C.latency(ofn, LAT_TEXTS)
        # quality of the INT8 model itself (what production runs)
        tl = C.onnx_batch_logits(sess, tok, TEST["text"].tolist())
        q = C.main_score(C.probs(tl, kind, m["temperature"]), kind)
        rows.append({"model": label, "training": "ours", **C.binary_metrics(YT, q),
                     "size_mb": round(paths["int8"].stat().st_size / 1e6, 1), "p50_ms": p50, "p95_ms": p95,
                     "backend": "ONNX INT8"})
        log("comparison", label, rows[-2]["p50_ms"], rows[-1]["p50_ms"])
        torch.cuda.empty_cache()
    for r in rows:
        r["name"] = f"{r['model']} ({r['backend']})"
    save("comparison", {"rows": rows, "latency_note": "one text at a time, 2 CPU threads, tokenisation included, Kaggle CPU"})
    p = plt()
    fig, ax = p.subplots(figsize=(6.6, 4))
    for r in rows:
        col = BLUE if "DistilBERT" in r["model"] else RED if "RoBERTa" in r["model"] else AMBER if "v0" in r["model"] else GREEN
        mk = "o" if r["backend"] == "ONNX INT8" else "s" if "PyTorch" in r["backend"] else "^"
        ax.scatter(r["p50_ms"], r["pr_auc"], color=col, marker=mk, s=50, zorder=3)
        ax.annotate(r["name"].replace(" multi-label", "").replace("TF-IDF ", ""), (r["p50_ms"], r["pr_auc"]),
                    fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"), ax.set_xlabel("CPU latency per comment, p50 (ms, log scale)"), ax.set_ylabel("Test PR-AUC")
    ax.set_title("Quality vs speed")
    fig.tight_layout(), fig.savefig(OUT / "fig" / "comparison.png"), p.close(fig)


def _deploy_name():
    d, m = load("model_distil_multi_debiased"), load("model_distil_multi")
    if d and m and d["history"][-1]["val_pr_auc"] >= max(h["val_pr_auc"] for h in m["history"]) - 0.005:
        return "distil_multi_debiased"
    return "distil_multi"


def _policy_table(y, s, grid):
    """All (t_low, t_high) pairs: review rate, harmful content allowed, innocent posts blocked."""
    n, tox = len(y), y == 1
    rows = []
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            allow, block = s < lo, s >= hi
            review = ~allow & ~block
            rows.append((lo, hi, review.mean(), (allow & tox).sum() / max(tox.sum(), 1),
                         (block & ~tox).sum() / max((~tox).sum(), 1),
                         (allow & tox).sum() / n, (block & ~tox).sum() / n))
    return pd.DataFrame(rows, columns=["t_low", "t_high", "review_rate", "harmful_allowed", "innocent_blocked",
                                       "p_false_allow", "p_false_block"])


def stage_policy():
    from metrics import choose_thresholds, policy
    name = _deploy_name()
    m = load(f"model_{name}")
    sv = C.main_score(C.probs(cache_np(f"{name}_val_logits"), "multi", m["temperature"]), "multi")
    st = C.main_score(C.probs(cache_np(f"{name}_test_logits"), "multi", m["temperature"]), "multi")
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    tab = _policy_table(YV, sv, grid)

    # (a) the original rule: block precision >= 95%, at most 5% of toxic posts auto-allowed
    lo, hi = choose_thresholds(YV, sv)
    fixed = {"t_low": lo, "t_high": hi, "test": policy(YT, st, lo, hi)}

    # (b) review capacity: humans can check at most X% of traffic; allow as little harm as possible,
    #     keeping wrongly-blocked innocent posts under 1% of all innocent posts
    ok = tab[(tab.review_rate <= a.review_capacity) & (tab.innocent_blocked <= 0.01)]
    best = ok.sort_values(["harmful_allowed", "review_rate"]).iloc[0]
    cap = {"t_low": float(best.t_low), "t_high": float(best.t_high), "capacity": a.review_capacity,
           "test": policy(YT, st, float(best.t_low), float(best.t_high))}

    # (c) explicit cost model: cost per post = C_fn*P(false allow) + C_fp*P(false block) + C_r*P(review)
    sweep = []
    for c_fn in [5, 10, 20, 50, 100]:
        for c_fp in [2, 5, 10, 20, 50]:
            cost = c_fn * tab.p_false_allow + c_fp * tab.p_false_block + 1.0 * tab.review_rate
            r = tab.iloc[int(np.argmin(cost.values))]
            sweep.append({"C_fn": c_fn, "C_fp": c_fp, "C_r": 1, "t_low": float(r.t_low), "t_high": float(r.t_high),
                          "review_rate": float(r.review_rate), "harmful_allowed": float(r.harmful_allowed),
                          "innocent_blocked": float(r.innocent_blocked), "cost_per_1000_posts": float(cost.min() * 1000)})
    # frontier: least harm achievable for each review budget
    budgets = np.round(np.arange(0.0, 0.31, 0.01), 2)
    frontier = []
    for b in budgets:
        ok = tab[(tab.review_rate <= b + 1e-9) & (tab.innocent_blocked <= 0.01)]
        if len(ok):
            frontier.append({"review_budget": float(b), "harmful_allowed": float(ok.harmful_allowed.min())})
    save("policy", {"model": name, "fixed_95_95": fixed, "capacity_policy": cap, "cost_sweep": sweep,
                    "frontier": frontier})
    p = plt()
    fig, ax = p.subplots(figsize=(5.6, 3.8))
    ax.plot([f["review_budget"] for f in frontier], [f["harmful_allowed"] for f in frontier], color=INK, lw=1.6,
            label="best possible (validation)")
    ax.scatter([s["review_rate"] for s in sweep], [s["harmful_allowed"] for s in sweep], color=GREY, s=14,
               label="cost-model optima", zorder=3)
    ax.scatter([fixed["test"]["review_rate"]], [fixed["test"]["toxic_auto_allowed"]], color=AMBER, s=60, marker="D",
               label=f"95/95 rule ({fixed['t_low']:.2f}/{fixed['t_high']:.2f})", zorder=4)
    ax.scatter([cap["test"]["review_rate"]], [cap["test"]["toxic_auto_allowed"]], color=BLUE, s=70, marker="*",
               label=f"{a.review_capacity:.0%} review capacity ({cap['t_low']:.2f}/{cap['t_high']:.2f})", zorder=4)
    ax.set_xlabel("Share of posts sent to human review"), ax.set_ylabel("Share of toxic posts auto-allowed")
    ax.set_title("Review workload vs harm let through"), ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(), fig.savefig(OUT / "fig" / "policy.png"), p.close(fig)


def stage_errors():
    name = _deploy_name()
    m = load(f"model_{name}")
    st = C.main_score(C.probs(cache_np(f"{name}_test_logits"), "multi", m["temperature"]), "multi")
    t = TEST.assign(score=st)
    fp = t[t.label == 0].nlargest(50, "score").assign(error="false positive")
    fn = t[t.label == 1].nsmallest(50, "score").assign(error="false negative")
    out = pd.concat([fp, fn])[["id", "error", "score", "text"] + C.LABELS]
    out["category"] = ""
    out.to_csv(OUT / "errors_100.csv", index=False)
    save("errors", {"model": name, "file": "errors_100.csv",
                    "categories": ["sarcasm or irony", "quoted or reported toxicity", "profanity used positively",
                                   "label noise", "identity mention", "implicit or coded insult", "other"]})


def stage_export():
    import torch
    import yaml
    name = _deploy_name()
    model, tok, kind = get_model(name)
    m = load(f"model_{name}")
    pol = load("policy")
    vl = cache_np(f"{name}_val_logits")
    pv = C.probs(vl, kind, m["temperature"])
    cats = {}
    for j, lab in enumerate(C.LABELS):
        if lab == "toxic":
            continue   # the generic label is already the main score
        thr, f1 = C.best_f1_threshold(YV6[:, j], pv[:, j])
        cats[lab] = float(min(max(thr, 0.3), 0.95))   # best-F1 on validation, kept in a sane range
    lo, hi = (pol["capacity_policy"]["t_low"], pol["capacity_policy"]["t_high"]) if pol else \
        __import__("metrics").choose_thresholds(YV, C.main_score(pv, kind))
    d = OUT / "deploy"
    if d.exists():
        shutil.rmtree(d)
    paths = C.export_onnx(model, tok, d)
    paths["fp32"].rename(d / "model.onnx")
    cfg = {"t_low": float(lo), "t_high": float(hi), "temperature": 1.0, "labels": C.LABELS,
           "temperatures": [round(float(t), 3) for t in m["temperature"]], "categories": cats}
    (d / "thresholds.yaml").write_text(f"# {name}: chosen on the validation set by run_all.py\n" +
                                       yaml.safe_dump(cfg, sort_keys=False))
    # check the exported INT8 model against PyTorch and against the production classifier code
    from classifier import OnnxClassifier
    from decision import load_thresholds
    c = load_thresholds(d)
    prod = OnnxClassifier(d, "model.int8.onnx", c["temperature"], temperatures=c["temperatures"], labels=c["labels"])
    sample = TEST.sample(min(len(TEST), 1000), random_state=2)
    ref = C.main_score(C.probs(C.predict_logits(model, tok, sample["text"].tolist()), kind, m["temperature"]), kind)
    got = np.array([prod.score(t) for t in sample["text"].astype(str)])
    from decision import decide
    agree = float(np.mean([decide(x, lo, hi) == decide(y, lo, hi) for x, y in zip(ref, got)]))
    lat = C.latency(lambda t: prod.score(t), LAT_TEXTS)
    res = {"model": name, "thresholds": cfg, "int8_mb": round((d / "model.int8.onnx").stat().st_size / 1e6, 1),
           "decision_agreement_int8_vs_pytorch": agree, "max_score_diff": float(np.abs(ref - got).max()),
           "production_code_p50_ms": lat[0], "production_code_p95_ms": lat[1], "uploaded": None}
    if a.upload_version:
        try:
            import boto3
            bucket = os.environ.get("S3_BUCKET", "sap-moderation-858082489486")
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-southeast-2"))
            for f in d.rglob("*"):
                if f.is_file() and f.name != "model.onnx":
                    s3.upload_file(str(f), bucket, f"models/{a.upload_version}/{f.relative_to(d).as_posix()}")
            res["uploaded"] = f"s3://{bucket}/models/{a.upload_version}/"
        except Exception as e:
            res["uploaded"] = f"failed: {e}"
    save("export", res)
    log("export", json.dumps(res))
    torch.cuda.empty_cache()


def stage_report():
    import report
    report.write(OUT)


# ---------------------------------------------------------------- main
only = [s for s in a.only.split(",") if s]
skip = [s for s in a.skip.split(",") if s]
for st in STAGES:
    if (only and st not in only) or st in skip:
        continue
    if st != "report" and (OUT / f"{st}.json").exists():
        log(f"== {st}: done earlier, skipping")
        continue
    log(f"== {st}")
    t0 = time.time()
    try:
        globals()[f"stage_{st}"]()
        log(f"== {st} finished in {(time.time() - t0) / 60:.1f} min")
    except Exception:
        log(f"!! {st} failed:\n{traceback.format_exc()}")
log("all done")
