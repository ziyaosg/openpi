from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .io import load_record, step_path


FULL_ATTN_KEY = "outputs/debug/attn/weights"

# ReLU gates — clips negative task/state attribution
# scores when reading from already-saved npy files.
ATTR_RELU_TASK  = False
ATTR_RELU_STATE = False

# ── Raw-attention head-selection settings (mirrors grid_visualization_raw_weights.py) ──
# Layer whose attention values are used for the final scores.
RAW_ATTN_TARGET_LAYER = 16
# Layer used only to rank heads by image-focus (early layers are more stable).
RAW_ATTN_HEAD_SELECTION_LAYER = 1
# Number of top image-focused heads to include.
RAW_ATTN_TOP_K_HEADS = 3


def _select_heads(
    layer_attn: np.ndarray,
    all_img_spans: List[Tuple[int, int]],
    task_span: Tuple[int, int] | None,
    state_span: Tuple[int, int] | None,
    top_k: int,
) -> np.ndarray:
    """Return indices of the top_k heads with the highest image/(task+state) budget ratio."""
    H = layer_attn.shape[0]

    img_budget = np.zeros(H, dtype=np.float32)
    for s0, s1 in all_img_spans:
        img_budget += layer_attn[:, s0:s1].sum(axis=1)

    text_budget = np.zeros(H, dtype=np.float32)
    if task_span is not None:
        text_budget += layer_attn[:, task_span[0]:task_span[1]].sum(axis=1)
    if state_span is not None:
        text_budget += layer_attn[:, state_span[0]:state_span[1]].sum(axis=1)

    ratio = img_budget / (text_budget + 1e-9)
    return np.argsort(ratio)[-min(top_k, H):]


def _extract_raw_attn_row(record: dict) -> np.ndarray:
    """Two-stage head selection on raw attention weights.

    1. Rank heads at RAW_ATTN_HEAD_SELECTION_LAYER by image/(task+state) budget.
    2. Take element-wise max across the top-K heads at RAW_ATTN_TARGET_LAYER.

    Returns a (S,) float32 array over all token positions.
    """
    from .keys import ATTN_KEYS, CAM_NAMES

    full_attn = np.asarray(record[FULL_ATTN_KEY])  # (L, 1, H, S)

    all_img_spans: List[Tuple[int, int]] = []
    for cam_name in CAM_NAMES:
        span_key = f"outputs/debug/spans/image/{cam_name}"
        if span_key in record:
            s = record[span_key]
            all_img_spans.append((int(s[0]), int(s[1])))

    task_span = state_span = None
    if ATTN_KEYS["task"] in record:
        s = record[ATTN_KEYS["task"]]
        task_span = (int(s[0]), int(s[1]))
    if ATTN_KEYS["state"] in record:
        s = record[ATTN_KEYS["state"]]
        state_span = (int(s[0]), int(s[1]))

    selector_attn = full_attn[RAW_ATTN_HEAD_SELECTION_LAYER, 0, :, :]  # (H, S)
    top_head_idx = _select_heads(selector_attn, all_img_spans, task_span, state_span, RAW_ATTN_TOP_K_HEADS)

    # Element-wise max across selected heads from the target layer
    attn_row = full_attn[RAW_ATTN_TARGET_LAYER, 0, :, :][top_head_idx, :].max(axis=0)  # (S,)
    return attn_row.astype(np.float32)


def extract_patches(
    record: dict,
    key: str,
    data_source: str = "gradcam",
    full_attn_key: str = FULL_ATTN_KEY,
) -> np.ndarray:
    """Return a flat float32 array of patch/token values for one modality at one step.

    gradcam:  record[key][0] is the attribution array (batch-indexed list).
    raw_attn: key is a span key; head selection via _extract_raw_attn_row(), then
              slice out the modality's token range.
    """
    if data_source == "gradcam":
        from .keys import GRADCAM_KEYS
        vals = np.asarray(record[key][0], dtype=np.float32).reshape(-1)
        if ATTR_RELU_TASK and key == GRADCAM_KEYS["task"]:
            vals = np.maximum(vals, 0.0)
        elif ATTR_RELU_STATE and key == GRADCAM_KEYS["state"]:
            vals = np.maximum(vals, 0.0)
        return vals
    attn_row = _extract_raw_attn_row(record)
    span = record[key]
    start, end = int(span[0]), int(span[1])
    return attn_row[start:end]





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
