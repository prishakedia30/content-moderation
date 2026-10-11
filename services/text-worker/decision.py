"""Turn a toxicity probability into ALLOW / REVIEW / BLOCK (two-threshold policy)."""
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


def load_thresholds(model_dir=None):
    """Defaults, then thresholds.yaml shipped with the model (from calibration),
    then T_LOW / T_HIGH / TEMPERATURE env vars if someone sets them explicitly."""
    cfg = dict(DEFAULTS)
    if model_dir:
        path = Path(model_dir) / "thresholds.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            cfg.update({k: float(v) for k, v in data.items() if k in DEFAULTS})
    for key in DEFAULTS:
        if os.environ.get(key.upper()):
            cfg[key] = float(os.environ[key.upper()])
    if not 0 <= cfg["t_low"] <= cfg["t_high"] <= 1:
        raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {cfg}")
    return cfg
