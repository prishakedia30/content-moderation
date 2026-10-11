"""Shared evaluation helpers: classification metrics, calibration, threshold policy."""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def softmax_pos(logits, temperature=1.0):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def classification(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def nll(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=15):
    """Expected calibration error: average gap between confidence and accuracy per bin."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit_temperature(logits, y):
    """Temperature scaling: one scalar T that minimises validation NLL. Grid search is
    plenty for one parameter and needs no extra dependencies."""
    grid = np.arange(0.5, 5.0, 0.01)
    losses = [nll(y, softmax_pos(logits, t)) for t in grid]
    return float(grid[int(np.argmin(losses))])


def choose_thresholds(y, p, block_precision=0.95, allow_recall=0.95):
    """T_high: lowest threshold whose BLOCK decisions are >= 95% correct.
    T_low: highest threshold that still sends >= 95% of toxic posts to REVIEW or BLOCK,
    i.e. at most 5% of toxic posts are auto-allowed. Everything in between goes to a human."""
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    t_high = next((t for t in grid if (p >= t).any() and y[p >= t].mean() >= block_precision), 0.99)
    t_low = max((t for t in grid if y.sum() and y[p >= t].sum() / y.sum() >= allow_recall), default=0.01)
    if t_low > t_high:
        t_low = t_high
    return float(t_low), float(t_high)


def policy(y, p, t_low, t_high):
    """How the two-threshold policy behaves on a labelled set."""
    allow, block = p < t_low, p >= t_high
    review = ~allow & ~block
    return {
        "allow_rate": float(allow.mean()), "review_rate": float(review.mean()),
        "block_rate": float(block.mean()),
        "block_precision": float(y[block].mean()) if block.any() else None,
        "toxic_auto_allowed": float(allow[y == 1].mean()) if (y == 1).any() else None,
    }
