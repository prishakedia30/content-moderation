"""Turn a toxicity probability into ALLOW / REVIEW / BLOCK (two-threshold policy), and,
for multi-label models, flag the categories whose probability passes their own threshold."""
import os
from pathlib import Path

import yaml

DEFAULTS = {"t_low": 0.30, "t_high": 0.75, "temperature": 1.0}


def decide(score, t_low, t_high):
    if score >= t_high:
        return "BLOCK"
    if score >= t_low:
        return "REVIEW"
    return "ALLOW"


def flags(categories, cat_thresholds):
    """Categories at or above their own threshold, highest first: ["threat", "insult"].
    Only categories that have a threshold are flagged (the generic "toxic" usually has none)."""
    hit = [(p, k) for k, p in categories.items() if k in cat_thresholds and p >= cat_thresholds[k]]
    return [k for _, k in sorted(hit, reverse=True)]


def load_thresholds(model_dir=None):
    """Defaults, then thresholds.yaml shipped with the model (from calibration),
    then T_LOW / T_HIGH / TEMPERATURE env vars if someone sets them explicitly.
    Multi-label models also ship `labels`, per-label `temperatures` and `categories`
    (per-category flag thresholds)."""
    cfg = dict(DEFAULTS)
    cfg.update({"categories": {}, "temperatures": None, "labels": None})
    if model_dir:
        path = Path(model_dir) / "thresholds.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            cfg.update({k: float(v) for k, v in data.items() if k in DEFAULTS})
            if isinstance(data.get("categories"), dict):
                cfg["categories"] = {str(k): float(v) for k, v in data["categories"].items()}
            if isinstance(data.get("temperatures"), list):
                cfg["temperatures"] = [float(t) for t in data["temperatures"]]
            if isinstance(data.get("labels"), list):
                cfg["labels"] = [str(x) for x in data["labels"]]
    for key in DEFAULTS:
        if os.environ.get(key.upper()):
            cfg[key] = float(os.environ[key.upper()])
    if not 0 <= cfg["t_low"] <= cfg["t_high"] <= 1:
        raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {cfg}")
    return cfg
