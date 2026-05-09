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

# Layer used for both head selection and scoring.
RAW_ATTN_TARGET_LAYER = 16
RAW_ATTN_TOP_K_HEADS  = 3
# Head selection strategy: "ratio" (one-vs-rest) or "budget" (raw attention mass).
HEAD_SELECTION_MODE   = "budget"


def _select_heads_for_span(
    attn: np.ndarray,
    span: Tuple[int, int],
    all_spans: List[Tuple[int, int]],
    top_k: int,
) -> np.ndarray:
    """Return top_k heads for the given span using HEAD_SELECTION_MODE.

    "ratio":  score[h] = attn[h, span].sum() / attn[h, all_other_spans].sum()
              Ranks by relative preference — which heads attend to this span
              more than to everything else.
    "budget": score[h] = attn[h, span].sum()
              Ranks by absolute attention mass — which heads put the most
              attention on this span, compared directly across all heads.
    """
    s0, s1 = span
    target_budget = attn[:, s0:s1].sum(axis=1)   # (H,)
    if HEAD_SELECTION_MODE == "ratio":
        other_budget = np.zeros(attn.shape[0], dtype=np.float32)
        for sp in all_spans:
            if sp != span:
                other_budget += attn[:, sp[0]:sp[1]].sum(axis=1)
        score = target_budget / (other_budget + 1e-9)
    else:
        score = target_budget
    return np.argsort(score)[-min(top_k, attn.shape[0]):]


def _build_all_spans(record: dict) -> List[Tuple[int, int]]:
    """Collect all modality spans from a record."""
    from .keys import ATTN_KEYS, CAM_NAMES
    spans: List[Tuple[int, int]] = []
    for cam_name in CAM_NAMES:
        key = f"outputs/debug/spans/image/{cam_name}"
        if key in record:
            s = record[key]
            spans.append((int(s[0]), int(s[1])))
    if ATTN_KEYS["task"] in record:
        s = record[ATTN_KEYS["task"]]
        spans.append((int(s[0]), int(s[1])))
    if ATTN_KEYS["state"] in record:
        s = record[ATTN_KEYS["state"]]
        spans.append((int(s[0]), int(s[1])))
    return spans


def _extract_raw_attn_row(
    record: dict,
    target_span: Tuple[int, int],
    all_spans: List[Tuple[int, int]],
) -> np.ndarray:
    """Per-modality one-vs-rest head selection + max over heads at TARGET_LAYER.

    Returns float32 scores for target_span only.
    """
    full_attn  = np.asarray(record[FULL_ATTN_KEY])
    stored     = np.asarray(record["outputs/debug/attn/layers"]).tolist()
    attn       = full_attn[stored.index(RAW_ATTN_TARGET_LAYER), 0]   # (H, S)
    heads      = _select_heads_for_span(attn, target_span, all_spans, RAW_ATTN_TOP_K_HEADS)
    s0, s1     = target_span
    return attn[heads, s0:s1].max(axis=0).astype(np.float32)


FULL_V_KEY = "outputs/debug/attn/v"


def _extract_value_score_row(
    record: dict,
    mode: str,
    target_span: Tuple[int, int],
    all_spans: List[Tuple[int, int]],
) -> np.ndarray:
    """Per-modality one-vs-rest head selection + value scoring at TARGET_LAYER.

    Returns float32 scores for target_span only.
    """
    full_attn  = np.asarray(record[FULL_ATTN_KEY])
    full_v     = np.asarray(record[FULL_V_KEY])
    stored     = np.asarray(record["outputs/debug/attn/layers"]).tolist()
    tgt        = stored.index(RAW_ATTN_TARGET_LAYER)
    attn       = full_attn[tgt, 0]       # (H, S)
    v          = full_v[tgt, 0]          # (S, H, D)
    heads      = _select_heads_for_span(attn, target_span, all_spans, RAW_ATTN_TOP_K_HEADS)
    s0, s1     = target_span

    if mode == "v_norm":
        v_norms  = np.linalg.norm(v, axis=-1)          # (S, H)
        weighted = attn * v_norms.T                    # (H, S)
        return weighted[heads, s0:s1].max(axis=0).astype(np.float32)
    elif mode == "v_cosine":
        o      = np.einsum("hs,shd->hd", attn, v)
        o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)
        v_unit = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
        cos    = np.einsum("shd,hd->sh", v_unit, o_unit)          # (S, H)
        return cos[s0:s1, :][:, heads].max(axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Choose 'v_norm' or 'v_cosine'.")


def extract_patches(
    record: dict,
    key: str,
    data_source: str = "gradcam",
    full_attn_key: str = FULL_ATTN_KEY,
) -> np.ndarray:
    """Return a flat float32 array of patch/token values for one modality at one step.

    data_source options:
      "gradcam"      — GradCAM attribution scores
      "raw_alpha"    — raw gradient sum per patch/token
      "raw_alpha_norm" — L2-norm-based raw alpha
      "attn"         — raw softmax attention weights (head-selected)
      "v_norm"       — value-norm weighted attention
      "v_cosine"     — value direction cosine alignment
    """
    if data_source == "gradcam":
        from .keys import GRADCAM_KEYS
        vals = np.asarray(record[key][0], dtype=np.float32).reshape(-1)
        if ATTR_RELU_TASK and key == GRADCAM_KEYS["task"]:
            vals = np.maximum(vals, 0.0)
        elif ATTR_RELU_STATE and key == GRADCAM_KEYS["state"]:
            vals = np.maximum(vals, 0.0)
        return vals
    if data_source in ("raw_alpha", "raw_alpha_norm"):
        return np.asarray(record[key][0], dtype=np.float32).reshape(-1)
    span_raw     = record[key]
    target_span  = (int(span_raw[0]), int(span_raw[1]))
    all_spans    = _build_all_spans(record)
    if data_source in ("v_norm", "v_cosine"):
        return _extract_value_score_row(record, mode=data_source,
                                        target_span=target_span, all_spans=all_spans)
    # default: raw attention weights
    return _extract_raw_attn_row(record, target_span=target_span, all_spans=all_spans)





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
