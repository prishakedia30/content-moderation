"""Shared helpers for the experiment suite (run_all.py): data, training, scoring, calibration,
export, latency, perturbations and identity templates."""
import copy
import inspect
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "train"))
sys.path.insert(0, str(ROOT / "services" / "text-worker"))

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

def find_file(root, *names):
    for name in names:
        hits = sorted(Path(root).rglob(name))
        if hits:
            return hits[0]
    return None


def load_jigsaw(path):
    """Jigsaw train.csv with the six labels kept, plus label = any of them. Same row order as
    ml/train/data.py, so split() reproduces v1's exact train/val/test split."""
    df = pd.read_csv(path).rename(columns={"comment_text": "text"})
    df["label"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    return df[["id", "text", "label"] + LABELS].sort_values("id").reset_index(drop=True)


def split(df):
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def stratified_sample(df, n, seed=SEED):
    """n rows with the same toxic share as df (no groupby.apply: pandas 3 drops the key column)."""
    if n >= len(df):
        return df.reset_index(drop=True)
    keep, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return keep.reset_index(drop=True)


def load_official_test(folder):
    """The competition's own test set (labels released afterwards; rows marked -1 were never scored)."""
    t = find_file(folder, "test.csv", "test.csv.zip")
    lab = find_file(folder, "test_labels.csv", "test_labels.csv.zip")
    if t is None or lab is None:
        return None
    df = pd.read_csv(t).merge(pd.read_csv(lab), on="id").rename(columns={"comment_text": "text"})
    df = df[(df[LABELS] >= 0).all(axis=1)].copy()
    df["label"] = (df[LABELS].sum(axis=1) > 0).astype(int)
    return df[["id", "text", "label"] + LABELS].reset_index(drop=True)


# ---------------------------------------------------------------- metrics

def binary_metrics(y, p, threshold=0.5):
    if len(y) == 0:
        return {k: None for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")}
    pred = (p >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
    }


def best_f1_threshold(y, p):
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def per_class(Y, P):
    rows = []
    for j, lab in enumerate(LABELS):
        y, p = Y[:, j], P[:, j]
        rows.append({"label": lab, "positives": int(y.sum()), "prevalence": float(y.mean()),
                     "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
                     "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
                     "f1_at_0.5": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))})
    return rows


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- scores and calibration

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_pos(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def fit_temperature_binary(logits, y):
    grid = np.arange(0.5, 5.0, 0.01)
    return float(grid[int(np.argmin([bce(y, softmax_pos(logits, t)) for t in grid]))])


def fit_temperatures_multi(logits, Y):
    """One temperature per category, each minimising validation log-loss for that category."""
    grid = np.arange(0.5, 5.0, 0.01)
    return [float(grid[int(np.argmin([bce(Y[:, j], sigmoid(logits[:, j] / t)) for t in grid]))])
            for j in range(logits.shape[1])]


def probs(logits, kind, T=None):
    """binary -> (N,) P(toxic); multi -> (N,6) per-category probabilities."""
    if kind == "binary":
        return softmax_pos(logits, T or 1.0)
    T = np.ones(logits.shape[1]) if T is None else np.asarray(T)
    return sigmoid(logits / T[None, :])


def main_score(p, kind):
    return p if kind == "binary" else p.max(axis=1)


# ---------------------------------------------------------------- transformers

def _batches(tok, texts, max_len, bs, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        yield idx, enc


@torch.no_grad()
def predict_logits(model, tok, texts, max_len=128, bs=256):
    """Logits in the original order. Texts are sorted by length so batches need little padding."""
    model.eval()
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = None
    dev = next(model.parameters()).device
    for idx, enc in _batches(tok, texts, max_len, bs, order):
        enc = {k: v.to(dev) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            lg = model(**enc).logits.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), lg.shape[1]), dtype=np.float32)
        out[idx] = lg
    return out


def train_model(base, kind, train_df, val_df, epochs=1, lr=2e-5, bs=32, max_len=128,
                weighted=True, seed=SEED, log=print, extra_train=None):
    """Fine-tune `base`. kind="binary": 2-way softmax + class-weighted cross-entropy.
    kind="multi": 6 sigmoid outputs + BCEWithLogitsLoss with per-class pos_weight = sqrt(neg/pos)
    (the square root keeps the rarest classes, e.g. threat at 0.3%, from dominating the loss).
    Keeps the epoch with the best validation PR-AUC (any-toxic). Returns (model, tok, history, minutes)."""
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    if kind == "binary":
        model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2, id2label={0: "non-toxic", 1: "toxic"}, label2id={"non-toxic": 0, "toxic": 1})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=6, problem_type="multi_label_classification",
            id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    model.to(DEVICE)

    df = train_df if extra_train is None else pd.concat([train_df, extra_train], ignore_index=True)
    texts = df["text"].astype(str).tolist()
    if kind == "binary":
        y = df["label"].astype(int).values
        counts = np.bincount(y, minlength=2)
        w = torch.tensor(len(y) / (2 * counts), dtype=torch.float, device=DEVICE) if weighted else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        targets = torch.tensor(y)
    else:
        Y = df[LABELS].values.astype(np.float32)
        pos = Y.sum(0).clip(min=1)
        pw = torch.tensor(np.sqrt((len(Y) - pos) / pos), dtype=torch.float, device=DEVICE) if weighted else None
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        targets = torch.tensor(Y)

    def collate(batch):
        idx = [b for b in batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        enc["labels"] = targets[idx]
        return enc

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(list(range(len(texts))), batch_size=bs, shuffle=True, collate_fn=collate,
                        generator=g, num_workers=2)
    steps = len(loader) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    yv = val_df["label"].values
    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(epochs):
        model.train()
        for step, batch in enumerate(loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits.float()
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 500 == 0:
                log(f"  {base} {kind} ep{ep + 1} step {step}/{len(loader)} loss {loss.item():.4f}")
        vl = predict_logits(model, tok, val_df["text"].tolist(), max_len)
        auc = float(average_precision_score(yv, main_score(probs(vl, kind), kind)))
        history.append({"epoch": ep + 1, "val_pr_auc": auc})
        log(f"  epoch {ep + 1}: val PR-AUC {auc:.4f}")
        if auc > best:
            best = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok, history, (time.time() - t0) / 60


# ---------------------------------------------------------------- export and latency

class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model, tok, out_dir, quantize=True):
    """FP32 ONNX (+ INT8 dynamic quantization) with the tokenizer, in the layout the text-worker loads."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir / "tokenizer")
    m = copy.deepcopy(model).float().cpu().eval()
    dummy = tok("export example", return_tensors="pt")
    fp32 = out_dir / "model.onnx"
    torch.onnx.export(
        _Wrap(m).eval(), (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}))
    paths = {"fp32": fp32}
    if quantize:
        int8 = out_dir / "model.int8.onnx"
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        paths["int8"] = int8
    del m
    return paths


def _pct(times):
    return round(float(np.percentile(times, 50)), 2), round(float(np.percentile(times, 95)), 2)


def latency(fn, texts, warmup=20, runs=200):
    """p50/p95 milliseconds for scoring ONE text at a time (what a live request costs), incl. tokenizing."""
    times = []
    for i in range(warmup + runs):
        t = texts[i % len(texts)]
        s = time.perf_counter()
        fn(t)
        if i >= warmup:
            times.append((time.perf_counter() - s) * 1000)
    return _pct(times)


def torch_cpu_scorer(model, tok, max_len=128, threads=2):
    torch.set_num_threads(threads)
    m = copy.deepcopy(model).float().cpu().eval()

    def fn(t):
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            return m(**enc).logits
    return fn, m


def onnx_scorer(path, tok, max_len=128, threads=2):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def fn(t):
        enc = tok(t, return_tensors="np", truncation=True, max_length=max_len)
        return sess.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                               "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    return fn, sess


def onnx_batch_logits(sess, tok, texts, max_len=128, bs=64):
    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = None
    for idx, enc in _batches(tok, texts, max_len, bs, order):
        lg = sess.run(None, {"input_ids": enc["input_ids"].numpy().astype(np.int64),
                             "attention_mask": enc["attention_mask"].numpy().astype(np.int64)})[0]
        if out is None:
            out = np.zeros((len(texts), lg.shape[1]), dtype=np.float32)
        out[idx] = lg
    return out


def dir_mb(path):
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


# ---------------------------------------------------------------- adversarial perturbations

ATTACK_WORDS = """idiot idiots stupid moron morons dumb loser losers fuck fucking fucker fucked shit
shitty bitch bitches bastard asshole ass dick dickhead cunt crap damn hell hate kill die retard
retarded pathetic ugly fat jerk scum trash garbage worthless suck sucks gay fag faggot nigger
whore slut piss pussy cock motherfucker stfu wtf idiotic imbecile freak disgusting""".split()
ATTACK_WORDS = set(ATTACK_WORDS)

LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7", "l": "1"}
HOMO_MAP = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "i": "і", "x": "х", "y": "у", "s": "ѕ"}
PERTURBATIONS = ["leetspeak", "repeated", "spaced", "punctuation", "homoglyph", "zero_width", "mixed"]


def _sub(word, mapping, rng, rate=0.7):
    chars = list(word)
    idx = [i for i, c in enumerate(chars) if c.lower() in mapping]
    if not idx:
        return word
    pick = [i for i in idx if rng.random() < rate] or [rng.choice(idx)]
    for i in pick:
        chars[i] = mapping[chars[i].lower()]
    return "".join(chars)


def perturb_word(word, kind, rng):
    if kind == "mixed":
        kind = rng.choice(PERTURBATIONS[:-1])
    if kind == "leetspeak":
        return _sub(word, LEET_MAP, rng)
    if kind == "repeated":
        vowels = [i for i, c in enumerate(word) if c.lower() in "aeiou"] or list(range(len(word)))
        i = rng.choice(vowels)
        return word[:i] + word[i] * rng.randint(3, 6) + word[i + 1:]
    if kind == "spaced":
        return " ".join(word)
    if kind == "punctuation":
        return rng.choice([".", "*", "-", "_"]).join(word)
    if kind == "homoglyph":
        return _sub(word, HOMO_MAP, rng)
    if kind == "zero_width":
        i = max(1, len(word) // 2)
        return word[:i] + "​" + word[i:]
    raise ValueError(kind)


WORD_RE = re.compile(r"[A-Za-z]+")


def has_attack_word(text):
    return any(w.lower() in ATTACK_WORDS for w in WORD_RE.findall(text))


def perturb_text(text, kind, seed=0):
    """Disguise every abusive word (from ATTACK_WORDS) the way a user dodging a filter would."""
    rng = random.Random(seed)
    return WORD_RE.sub(lambda m: perturb_word(m.group(0), kind, rng)
                       if m.group(0).lower() in ATTACK_WORDS else m.group(0), text)


# ---------------------------------------------------------------- identity terms

IDENTITY_GROUPS = ["gay", "lesbian", "bisexual", "transgender", "straight", "Muslim", "Christian",
                   "Jewish", "Hindu", "Sikh", "atheist", "Black", "white", "Asian", "Latino",
                   "Indian", "African", "blind", "deaf", "disabled", "old", "young", "female", "male"]
PROPER = {"Muslim", "Christian", "Jewish", "Hindu", "Sikh", "Black", "Asian", "Latino", "Indian", "African"}
CONTROL_GROUP = "tall"   # an identity-free word to compare against

# Evaluation templates (never used in training).
EVAL_NONTOXIC = [
    "I am {a} {g} person.", "I am {g}.", "My best friend is {g}.", "{G} people are nice.",
    "Being {g} is completely normal.", "I'm proud to be {g}.", "There is {a} {g} student in my class.",
    "{G} people deserve respect.", "My neighbour is {g} and we often have dinner together.",
    "As {a} {g} woman, I agree with this article.", "Some of my colleagues are {g}.",
    "We should listen to {g} voices more.",
]
EVAL_TOXIC = [
    "{G} people are disgusting.", "I hate {g} people.", "All {g} people should shut up and leave.",
    "{G} people are stupid and worthless.", "Get out, you filthy {g} idiot.",
]
# Training augmentation templates for the mitigation run: different wording from the eval set.
TRAIN_NONTOXIC = [
    "{G} people live in my city.", "The {g} community organised a charity fair.",
    "She is {g} and works as a doctor.", "He told us he is {g}.", "We discussed {g} history in class today.",
    "{A} {g} family moved in next door.", "My teacher is {g} and very kind.",
    "Many {g} athletes competed this year.", "The article interviews {g} students about exams.",
    "Our team includes {g} engineers.", "Is it true that the new manager is {g}?",
    "I watched a documentary about {g} artists.",
]


def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def fill(template, group):
    return template.format(g=group, G=group[0].upper() + group[1:], a=_article(group),
                           A=_article(group).capitalize())


def identity_eval_set():
    rows = []
    for g in IDENTITY_GROUPS + [CONTROL_GROUP]:
        rows += [{"group": g, "text": fill(t, g), "label": 0} for t in EVAL_NONTOXIC]
        rows += [{"group": g, "text": fill(t, g), "label": 1} for t in EVAL_TOXIC]
    return pd.DataFrame(rows)


_ID_RE = re.compile(r"\b(?:" + "|".join(re.escape(g) for g in IDENTITY_GROUPS) + r")\b", re.I)


def counterfactual_augment(train_df, seed=SEED, max_rows=20000, template_repeats=5):
    """Mitigation data: (1) copies of training comments that mention an identity term with the term
    swapped for a different group, labels unchanged, so no group's name alone predicts toxicity;
    (2) neutral sentences mentioning every group, labelled non-toxic."""
    rng = random.Random(seed)
    hits = train_df[train_df["text"].astype(str).str.contains(_ID_RE)]
    hits = hits.sample(min(len(hits), max_rows), random_state=seed)

    def swap(m):
        w = m.group(0)
        new = rng.choice([g for g in IDENTITY_GROUPS if g.lower() != w.lower()])
        if new in PROPER:
            return new
        return new.capitalize() if w[0].isupper() else new.lower()

    cf = hits.copy()
    cf["text"] = cf["text"].astype(str).apply(lambda t: _ID_RE.sub(swap, t))
    neutral = [fill(t, g) for t in TRAIN_NONTOXIC for g in IDENTITY_GROUPS] * template_repeats
    nt = pd.DataFrame({"id": [f"aug{i}" for i in range(len(neutral))], "text": neutral, "label": 0,
                       **{l: 0 for l in LABELS}})
    return pd.concat([cf, nt], ignore_index=True), {"counterfactual_rows": len(cf), "neutral_rows": len(nt)}
