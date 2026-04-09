from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ALL_MODALITY_KEYS
from .utils import load_record, step_path


def reduce_attention_scores(input_dir, step, key, method="average"):
    record = load_record(str(step_path(input_dir, step)))
    vals = np.asarray(record[key][0], dtype=float)

    if method == "average":
        return float(np.mean(vals))
    if method == "median":
        return float(np.median(vals))
    if method == "max":
        return float(np.max(vals))
    if method == "min":
        return float(np.min(vals))
    if method == "sum":
        return float(np.sum(vals))
    if method == "sqrt_norm_sum":
        return float(np.sum(vals) / np.sqrt(vals.size))

    raise ValueError(f"Unknown reduction method: {method}")


def normalize_modality(values, method="robust_zscore"):
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return []

    if method == "robust_zscore":
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad > 1e-8:
            return ((arr - med) / (1.4826 * mad)).tolist()
        return np.zeros_like(arr).tolist()

    if method == "zscore":
        std = arr.std()
        if std > 1e-8:
            return ((arr - arr.mean()) / std).tolist()
        return np.zeros_like(arr).tolist()

    if method == "minmax":
        denom = arr.max() - arr.min()
        if denom > 1e-8:
            return ((arr - arr.min()) / denom).tolist()
        return np.zeros_like(arr).tolist()

    if method == "none":
        return arr.tolist()

    raise ValueError(f"Unknown normalization method: {method}")


def build_episode_series(ep, input_dir, key, method):
    steps = []
    values = []

    for step in range(ep.start_idx, ep.end_idx + 1):
        path = step_path(input_dir, step)
        if not path.exists():
            continue

        try:
            val = reduce_attention_scores(input_dir, step, key, method)
        except Exception:
            continue

        # Raw step index on x-axis
        steps.append(step)
        values.append(val)

    return steps, values


def build_all_series(ep, input_dir, methods, norm="robust_zscore"):
    all_series = {}

    for key in ALL_MODALITY_KEYS:
        steps, values = build_episode_series(
            ep=ep,
            input_dir=input_dir,
            key=key,
            method=methods.get(key, "average"),
        )
        all_series[key] = (steps, normalize_modality(values, norm))

    return all_series