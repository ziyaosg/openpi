from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_record, step_path


FULL_ATTN_KEY = "outputs/debug/raw_attn/full_attn"


def extract_patches(
    record: dict,
    key: str,
    data_source: str = "gradcam",
    full_attn_key: str = FULL_ATTN_KEY,
) -> np.ndarray:
    """Return a flat float32 array of patch/token values for one modality at one step.

    gradcam:  record[key][0] is the attribution array (batch-indexed list).
    raw_attn: key is a span key; we average full_attn over layers & heads,
              then slice out the modality's token range.
    """
    if data_source == "gradcam":
        return np.asarray(record[key][0], dtype=np.float32).reshape(-1)
    full_attn = np.asarray(record[full_attn_key])  # (layers, 1, heads, tokens)
    attn = full_attn.mean(axis=(0, 2))[0]           # (tokens,)
    span = record[key]
    start, end = int(span[0]), int(span[1])
    return attn[start:end].astype(np.float32)


def reduce_vals(vals, method: str) -> float:
    vals = np.asarray(vals)
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


def normalize_modality(values, method: str) -> list:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return []

    if method == "robust_zscore":
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return ((arr - med) / (1.4826 * mad)).tolist() if mad > 1e-8 else np.zeros_like(arr).tolist()

    if method == "zscore":
        std = arr.std()
        return ((arr - arr.mean()) / std).tolist() if std > 1e-8 else np.zeros_like(arr).tolist()

    if method == "minmax":
        denom = arr.max() - arr.min()
        return ((arr - arr.min()) / denom).tolist() if denom > 1e-8 else np.zeros_like(arr).tolist()

    if method == "none":
        return arr.tolist()

    raise ValueError(f"Unknown normalization method: {method}")


def compute_modality_preview_ranges(ep, input_dir, keys, data_source="gradcam") -> dict:
    """Compute per-key (min, max) ranges across an episode, with 5% padding."""
    ranges = {}

    for key in keys:
        global_min = float("inf")
        global_max = float("-inf")

        for step in range(ep.start_idx, ep.end_idx + 1):
            path = step_path(input_dir, step)
            if not path.exists():
                continue

            try:
                record = load_record(str(path))
                vals = extract_patches(record, key, data_source).astype(float)
            except Exception:
                continue

            if vals.size == 0:
                continue

            global_min = min(global_min, float(vals.min()))
            global_max = max(global_max, float(vals.max()))

        if global_min == float("inf") or global_max == float("-inf"):
            ranges[key] = (0.0, 1.0)
        else:
            if abs(global_max - global_min) < 1e-12:
                pad = 1e-3 if global_max == 0 else 0.05 * abs(global_max)
            else:
                pad = 0.05 * (global_max - global_min)
            ranges[key] = (global_min - pad, global_max + pad)

    return ranges
