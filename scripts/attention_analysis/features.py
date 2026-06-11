#!/usr/bin/env python3
"""
features.py — Consolidated per-step feature extraction and episode analysis.

One place for every feature that has been developed across conformal_analysis.py,
conformal_analysis_v2.py, analysis_figures.py, comprehensive_window_analysis.py,
and phase_aware_analysis.py.

Structure
---------
  Scalar utilities          gini, entropy, cosine, spatial_com, _topk_iou,
                            _com_dist_2d, _com_dist_1d, _emd_2d, _emd_1d,
                            _select_rw, _vc_scores

  Format helpers            detect_format, _span, _state_digit_slice

  Sub-extractors            extract_attn_routing     → attn-weight routing features
                            extract_heatmap_features → per-camera attribution scalars
                            extract_per_token        → per-action-token Gini (pi0fast)
                            extract_task_state       → task / state token features
                            extract_value_features   → value-vector features
                            extract_action_features  → action / motion features

  Main extractor            extract_features(rec, prev_rec, prev_centroids)
                              → (feats, centroids)

  Episode / group loading   load_episode_features, load_group, feature_names

  Matrix utilities          to_matrix, rolling_mean_matrix

  Window descriptors        window_descriptors(raw_mat, window_size)
                              → win_mean, win_std, win_slope,
                                d_mean, d_slope, d_slope_mag,
                                cen_slope, cen_slope_mag
                            Pass window_size=5 (fast) or window_size=15 (slow).

  Phase configs             PHASES_3  — early/mid/late
                            PHASES_4  — q1/q2/q3/q4

  Phase analysis            compute_descriptors(raw, window_size)
                            phase_aggregates(raw, phase_defs)
                            analyse_feature(fail_eps, succ_eps, feat,
                                            phase_defs, window_sizes)
                            analyse_all_features(ep_data, features, ...)

  AUROC / stats             auroc_at, effective_auroc,
                            compute_auroc_sequence,
                            compute_pvalue_sequence,
                            compute_cohens_d_sequence

  Candidate selection       select_cross_domain_candidates,
                            select_group_specific_candidates,
                            select_direction_change_features
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, wasserstein_distance
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# PHASE CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

# 3-phase: early / mid / late  (inclusive step ranges)
PHASES_3: Dict[str, Tuple[int, int]] = {
    "early": (0,  10),
    "mid":   (11, 30),
    "late":  (31, 59),
}

# 4-phase: quarterly
PHASES_4: Dict[str, Tuple[int, int]] = {
    "q1": (0,  14),
    "q2": (15, 29),
    "q3": (30, 44),
    "q4": (45, 59),
}

# Defaults used when callers don't pass phase_defs / window_sizes explicitly
DEFAULT_WINDOW_SIZES = (5, 15)   # small causal + large causal
DEFAULT_PHASE_DEFS   = PHASES_3

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PATCH_H = PATCH_W = 16    # spatial patch grid for all cameras
_RW_TOP_K   = 8           # top-K span-focused heads for raw-weights-max
_IOU_FRAC   = 0.10        # top fraction of positions for IoU overlap
AUROC_MIN_N = 5           # minimum class size to compute AUROC

NOISE_FEATURES: frozenset = frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _nan() -> float:
    return float("nan")


def gini(v: np.ndarray) -> float:
    """Gini coefficient of |v|.  0 = uniform, 1 = fully concentrated."""
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a)
    s = float(a.sum())
    if s < 1e-12 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy of the normalised absolute values of p."""
    p = np.abs(p.ravel()).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def _top_k_frac(v: np.ndarray, k: int) -> float:
    """Fraction of total |v| mass in the top-k elements.  NaN when v is None or too short."""
    if v is None:
        return _nan()
    a = np.abs(v.ravel()).astype(np.float64)
    s = a.sum()
    if s < 1e-12 or len(a) < k:
        return _nan()
    return float(np.partition(a, -k)[-k:].sum() / s)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays (ravelled)."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else _nan()


def spatial_com(h: np.ndarray) -> Tuple[float, float]:
    """Centre-of-mass (row, col) of a 2-D map.  Returns (0, 0) for zero maps."""
    h = np.abs(h).astype(np.float64)
    s = h.sum()
    if s < 1e-12:
        return 0.0, 0.0
    rows = np.arange(h.shape[0])[:, None]
    cols = np.arange(h.shape[1])[None, :]
    return float((h * rows).sum() / s), float((h * cols).sum() / s)


def _topk_iou(a: np.ndarray, b: np.ndarray, frac: float = _IOU_FRAC) -> float:
    """Overlap IoU of the top-frac positions of two attribution maps."""
    a, b = np.abs(a.ravel()), np.abs(b.ravel())
    k = max(1, int(round(len(a) * frac)))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else 0.0


def _com_dist_2d(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between spatial COMs of two 2-D (or flat) attribution maps."""
    def _to_2d(h):
        h = np.abs(h)
        return h.reshape(PATCH_H, PATCH_W) if h.ndim == 1 else h
    cy_a, cx_a = spatial_com(_to_2d(a))
    cy_b, cx_b = spatial_com(_to_2d(b))
    return float(np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2))


def _com_dist_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute difference of weighted-mean token positions (1-D sequences)."""
    a = np.abs(a.ravel()).astype(np.float64)
    b = np.abs(b.ravel()).astype(np.float64)
    pos = np.arange(len(a), dtype=np.float64)
    ca = float((a * pos).sum() / (a.sum() + 1e-12))
    cb = float((b * pos).sum() / (b.sum() + 1e-12))
    return abs(ca - cb)


def _emd_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of 1-D Wasserstein distances on x- and y-axis marginals (patch units)."""
    a = np.abs(a).astype(np.float64)
    b = np.abs(b).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    a2 = a.reshape(PATCH_H, PATCH_W) / sa
    b2 = b.reshape(PATCH_H, PATCH_W) / sb
    xs = np.arange(PATCH_W, dtype=np.float64)
    ys = np.arange(PATCH_H, dtype=np.float64)
    emd_x = wasserstein_distance(xs, xs, a2.sum(axis=0), b2.sum(axis=0))
    emd_y = wasserstein_distance(ys, ys, a2.sum(axis=1), b2.sum(axis=1))
    return float(0.5 * (emd_x + emd_y))


def _emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance over token positions."""
    a = np.abs(a.ravel()).astype(np.float64)
    b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    pos = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(pos, pos, a / sa, b / sb))


def _select_rw(attn_per_head: np.ndarray, sp: Tuple[int, int],
               top_k: int = _RW_TOP_K) -> np.ndarray:
    """Max attention over the top-K span-focused heads for span sp.

    attn_per_head: (H, S)  per-head raw attention weights (last layer)
    Returns: (sp[1]-sp[0],) vector
    """
    k = min(top_k, attn_per_head.shape[0])
    span_mass = attn_per_head[:, sp[0]:sp[1]].sum(axis=1)  # (H,)
    top_heads = np.argsort(span_mass)[-k:]
    return attn_per_head[top_heads, sp[0]:sp[1]].max(axis=0)


def _vc_scores(attn_hxs: np.ndarray, v_seq: np.ndarray,
               sp: Tuple[int, int], top_k: int = _RW_TOP_K) -> np.ndarray:
    """V-Cosine score per token in span sp.

    For each token s: score = max over top-K heads of cosine(v[s], head_output).
    v_seq: (S, 1, D); attn_hxs: (H, S).  Returns (sp[1]-sp[0],) float32.
    """
    k = min(top_k, attn_hxs.shape[0])
    span_mass = attn_hxs[:, sp[0]:sp[1]].sum(axis=1)  # (H,)
    top_heads = np.argsort(span_mass)[-k:]
    v0 = v_seq[:, 0, :]                                          # (S, D)
    o = attn_hxs @ v0                                            # (H, D)
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)
    v_unit = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-9)
    cos_all = v_unit @ o_unit.T                                  # (S, H)
    scores = cos_all[:, top_heads].max(axis=1).astype(np.float32)
    return scores[sp[0]:sp[1]]


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(rec: dict) -> str:
    """Return 'pi0fast' or 'pi05'."""
    return "pi0fast" if "outputs/debug/attn/layers" in rec else "pi05"


def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    v = rec.get(key)
    if v is None:
        return None
    v = np.asarray(v).ravel()
    return int(v[0]), int(v[1])


def _state_digit_slice(rec: dict) -> Tuple[int, Optional[int]]:
    """Return (k, end) so score_array[k:end] covers only digit tokens in 'State: {digits};\\n'.

    Returns (0, None) when piece metadata is absent (use the full array as-is).
    """
    pb_key = "outputs/debug/tokens/state/piece_begin"
    pe_key = "outputs/debug/tokens/state/piece_end"
    if pb_key not in rec or pe_key not in rec:
        return 0, None
    piece_begin = np.asarray(rec[pb_key])
    piece_end   = np.asarray(rec[pe_key])
    max_byte      = int(piece_end.max())
    CONTENT_START = 7             # len("State: ")
    CONTENT_END   = max_byte - 2  # excl. ";\n"
    k = int((piece_end   <= CONTENT_START).sum())
    s = int((piece_begin >= CONTENT_END).sum())
    return k, len(piece_begin) - s


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR A: ATTENTION ROUTING
# ─────────────────────────────────────────────────────────────────────────────

def extract_attn_routing(rec: dict, fmt: str) -> dict:
    """Modality attention fractions, routing entropy, and within-modality Gini.

    Public keys: frac_base, frac_wrist, frac_task, frac_state,
                 routing_entropy, attn_gini_{base,wrist,task,state},
                 attn_entropy_base
    Private keys (prefixed _): _attn_{base,wrist,task}, _rw_{...}, _vc_{...}
    """
    feats: dict = {}
    wkey = "outputs/debug/attn/weights"
    _pub_nan = ("frac_base", "frac_wrist", "frac_task", "frac_state",
                "routing_entropy", "attn_gini_base", "attn_entropy_base",
                "attn_gini_wrist", "attn_gini_task", "attn_gini_state")
    _priv_nan = ("_attn_base", "_attn_wrist", "_attn_task", "_attn_state",
                 "_rw_base", "_rw_wrist", "_rw_task", "_rw_state",
                 "_vc_base", "_vc_wrist", "_vc_task", "_vc_state")

    if wkey not in rec:
        for k in _pub_nan:
            feats[k] = _nan()
        for k in _priv_nan:
            feats[k] = None
        return feats

    w = np.asarray(rec[wkey])
    if fmt == "pi0fast":
        # (L, 1, H, S): last layer, squeeze batch, mean over heads
        mean_attn     = w[-1, 0].mean(axis=0)   # (S,)
        attn_per_head = w[-1, 0]                 # (H, S)
    else:
        # (L, H, T_action, S): last layer, slot 0 only, mean over heads
        mean_attn     = w[-1, :, 0, :].mean(axis=0)  # (S,)
        attn_per_head = w[-1, :, 0, :]               # (H, S)

    sp_base   = _span(rec, "outputs/debug/spans/image/base_0_rgb")
    sp_wrist  = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
    sp_rwrist = _span(rec, "outputs/debug/spans/image/right_wrist_0_rgb")
    sp_task   = _span(rec, "outputs/debug/spans/task")
    sp_state  = _span(rec, "outputs/debug/spans/state")

    # For pi0fast the right_wrist slot gets a zero image (image_mask=True); exclude its
    # mass from the denominator so fractions compare to pi05 (where image_mask=False).
    if fmt == "pi0fast" and sp_rwrist is not None:
        rwrist_mass = float(mean_attn[sp_rwrist[0]:sp_rwrist[1]].sum())
        total = float(mean_attn.sum()) - rwrist_mass + 1e-12
    else:
        total = float(mean_attn.sum()) + 1e-12

    def frac(sp):
        return float(mean_attn[sp[0]:sp[1]].sum() / total) if sp else _nan()

    feats["frac_base"]   = frac(sp_base)
    feats["frac_wrist"]  = frac(sp_wrist)
    feats["frac_task"]   = frac(sp_task)
    feats["frac_state"]  = frac(sp_state)

    fracs = [v for v in [feats["frac_base"], feats["frac_wrist"],
                          feats["frac_task"], feats["frac_state"]] if not np.isnan(v)]
    if fracs:
        p = np.clip(np.array(fracs), 0, None)
        p = p / (p.sum() + 1e-12)
        feats["routing_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        feats["routing_entropy"] = _nan()

    # For pi05 task span: truncate to real tokens to avoid sparse-vector Gini inflation.
    sp_task_gini = sp_task
    if sp_task and fmt == "pi05":
        tmk = "outputs/debug/tokens/task/token_mask"
        if tmk in rec:
            tm = np.asarray(rec[tmk]).astype(bool)
            nr = int(tm[:sp_task[1] - sp_task[0]].sum())
            if nr > 0:
                sp_task_gini = (sp_task[0], sp_task[0] + nr)

    for tag, sp in [("base", sp_base), ("wrist", sp_wrist),
                    ("task", sp_task_gini), ("state", sp_state)]:
        if sp:
            seg = mean_attn[sp[0]:sp[1]]
            feats[f"attn_gini_{tag}"] = gini(seg)
            if tag == "base":
                feats["attn_entropy_base"] = entropy(seg)
        else:
            feats[f"attn_gini_{tag}"] = _nan()
            if tag == "base":
                feats["attn_entropy_base"] = _nan()

    # Raw attn segments (private, used by heatmap extractor for cross-signal agreement).
    # pi05 task: truncate to real tokens.
    feats["_attn_base"]   = mean_attn[sp_base[0]:sp_base[1]]   if sp_base   else None
    feats["_attn_wrist"]  = mean_attn[sp_wrist[0]:sp_wrist[1]] if sp_wrist  else None
    feats["_attn_state"]  = mean_attn[sp_state[0]:sp_state[1]] if sp_state  else None
    if sp_task:
        if fmt == "pi05":
            tmk = "outputs/debug/tokens/task/token_mask"
            if tmk in rec:
                tm = np.asarray(rec[tmk]).astype(bool)
                span_len = sp_task[1] - sp_task[0]
                nr = int(tm[:span_len].sum())
                sp_task_real = (sp_task[0], sp_task[0] + nr) if nr > 0 else sp_task
            else:
                sp_task_real = sp_task
        else:
            sp_task_real = sp_task
        feats["_attn_task"] = mean_attn[sp_task_real[0]:sp_task_real[1]]
    else:
        sp_task_real = None
        feats["_attn_task"] = None

    # Raw-weights-max per modality (private).
    feats["_rw_base"]   = _select_rw(attn_per_head, sp_base)      if sp_base      else None
    feats["_rw_wrist"]  = _select_rw(attn_per_head, sp_wrist)     if sp_wrist     else None
    feats["_rw_task"]   = _select_rw(attn_per_head, sp_task_real) if sp_task_real else None
    feats["_rw_state"]  = _select_rw(attn_per_head, sp_state)     if sp_state     else None

    # V-Cosine scores (private). v shape: (L, 1, S, 1, D) for pi0fast; (L, S, 1, D) for pi05.
    vkey = "outputs/debug/attn/v"
    v_seq: Optional[np.ndarray] = None
    if vkey in rec:
        v_raw = np.asarray(rec[vkey])
        v_seq = v_raw[-1, 0, :, :, :] if fmt == "pi0fast" else v_raw[-1, :, :, :]
    for sp_vc, tag_vc in [(sp_base, "base"), (sp_wrist, "wrist"),
                           (sp_task_real, "task"), (sp_state, "state")]:
        feats[f"_vc_{tag_vc}"] = (
            _vc_scores(attn_per_head, v_seq, sp_vc) if (sp_vc is not None and v_seq is not None) else None
        )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR B: PER-CAMERA HEATMAP SCALARS
# ─────────────────────────────────────────────────────────────────────────────

def _load_heatmap_pi0fast(rec: dict, cam: str):
    """Return (gc_mean, ras_mean, rn_mean, gc_per_tok, ras_per_tok, rn_per_tok)."""
    gcs, ras_l, rns = [], [], []
    for i in range(5):
        kg  = f"outputs/debug/gradcam/action_tokens/{i}/{cam}"
        kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{cam}"
        krn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/{cam}"
        if kg not in rec:
            break
        gcs.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
        ras_l.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32))
        rns.append(np.abs(np.asarray(rec[krn])[0]).astype(np.float32))
    if not gcs:
        return None, None, None, [], [], []
    return (np.mean(np.stack(gcs), axis=0),
            np.mean(np.stack(ras_l), axis=0),
            np.mean(np.stack(rns), axis=0),
            gcs, ras_l, rns)


def _load_heatmap_pi05(rec: dict, cam: str):
    """Return (gc_mean, ras_mean, rn_mean, gc_per_slot, ras_per_slot, rn_per_slot).

    Tries per-slot keys action_tokens/{0-4}/{cam} (mean over slots 0-4).
    Falls back to single aggregate keys if per-slot data is absent.
    """
    gcs, ras_l, rns = [], [], []
    for i in range(5):
        kg  = f"outputs/debug/gradcam/action_tokens/{i}/{cam}"
        kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{cam}"
        krn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/{cam}"
        if kg not in rec:
            break
        gc_i = np.abs(np.squeeze(np.asarray(rec[kg]))).astype(np.float32)
        gcs.append(gc_i)
        ras_l.append(np.abs(np.squeeze(np.asarray(rec[kr]))).astype(np.float32)  if kr  in rec else np.zeros_like(gc_i))
        rns.append(np.abs(np.squeeze(np.asarray(rec[krn]))).astype(np.float32) if krn in rec else np.zeros_like(gc_i))
    if gcs:
        return (np.mean(np.stack(gcs),   axis=0),
                np.mean(np.stack(ras_l), axis=0),
                np.mean(np.stack(rns),   axis=0),
                gcs, ras_l, rns)
    # Fallback: single aggregate keys
    kg  = f"outputs/debug/gradcam/image/{cam}"
    kr  = f"outputs/debug/raw_alpha/summation/image/{cam}"
    krn = f"outputs/debug/raw_alpha/norm/image/{cam}"
    gc  = np.abs(np.asarray(rec[kg])).astype(np.float32)  if kg  in rec else None
    ras = np.abs(np.asarray(rec[kr])).astype(np.float32)  if kr  in rec else None
    rn  = np.abs(np.asarray(rec[krn])).astype(np.float32) if krn in rec else None
    return gc, ras, rn, [], [], []


def _heatmap_scalars(gc, ras, rn, attn_seg, label: str,
                     rw: Optional[np.ndarray] = None,
                     vc: Optional[np.ndarray] = None) -> dict:
    """All per-camera scalar features given (gc, ras, rn) heatmaps plus optional rw/vc."""
    f: dict = {}
    _nan_keys = (
        "gc_gini", "gc_entropy", "ras_gini", "ras_entropy", "rn_gini", "rn_entropy",
        "gc_max", "gc_mean", "gc_com_y", "gc_com_x",
        "agree_gc_ras", "agree_gc_rn", "agree_ras_rn",
        "agree_attn_gc", "agree_attn_ras",
        "rw_gini", "agree_gc_rw", "agree_ras_rw", "agree_rn_rw",
        "iou_gc_ras", "iou_gc_rn", "iou_ras_rn",
        "iou_gc_rw", "iou_ras_rw", "iou_rn_rw",
        "com_dist_gc_ras", "com_dist_gc_rn", "com_dist_ras_rn",
        "emd_gc_ras", "emd_gc_rn", "emd_ras_rn",
        "com_dist_gc_rw", "emd_gc_rw", "com_dist_ras_rw", "emd_ras_rw",
        "com_dist_rn_rw", "emd_rn_rw",
        "vc_gini",
        "agree_gc_vc", "agree_ras_vc", "agree_rn_vc", "agree_rw_vc",
        "iou_gc_vc", "iou_ras_vc", "iou_rn_vc", "iou_rw_vc",
        "com_dist_gc_vc", "emd_gc_vc", "com_dist_ras_vc", "emd_ras_vc",
        "com_dist_rn_vc", "emd_rn_vc", "com_dist_rw_vc", "emd_rw_vc",
        "attn_entropy",
    )
    if gc is None:
        for pfx in _nan_keys:
            f[f"{pfx}_{label}"] = _nan()
        return f

    f[f"gc_gini_{label}"]      = gini(gc)
    f[f"gc_entropy_{label}"]   = entropy(gc)
    f[f"ras_gini_{label}"]     = gini(ras)
    f[f"ras_entropy_{label}"]  = entropy(ras)
    f[f"rn_gini_{label}"]      = gini(rn)
    f[f"rn_entropy_{label}"]   = entropy(rn)
    f[f"gc_max_{label}"]       = float(gc.max())
    f[f"gc_mean_{label}"]      = float(gc.mean())
    cy, cx = spatial_com(gc)
    f[f"gc_com_y_{label}"]     = cy
    f[f"gc_com_x_{label}"]     = cx

    f[f"agree_gc_ras_{label}"] = cosine(gc, ras)
    f[f"agree_gc_rn_{label}"]  = cosine(gc, rn)
    f[f"agree_ras_rn_{label}"] = cosine(ras, rn)

    f[f"iou_gc_ras_{label}"]  = _topk_iou(gc, ras)
    f[f"iou_gc_rn_{label}"]   = _topk_iou(gc, rn)
    f[f"iou_ras_rn_{label}"]  = _topk_iou(ras, rn)

    f[f"com_dist_gc_ras_{label}"] = _com_dist_2d(gc, ras)
    f[f"com_dist_gc_rn_{label}"]  = _com_dist_2d(gc, rn)
    f[f"com_dist_ras_rn_{label}"] = _com_dist_2d(ras, rn)

    f[f"emd_gc_ras_{label}"]  = _emd_2d(gc, ras)
    f[f"emd_gc_rn_{label}"]   = _emd_2d(gc, rn)
    f[f"emd_ras_rn_{label}"]  = _emd_2d(ras, rn)

    if attn_seg is not None and attn_seg.size == gc.size:
        f[f"agree_attn_gc_{label}"]  = cosine(attn_seg, gc.ravel())
        f[f"agree_attn_ras_{label}"] = cosine(attn_seg, ras.ravel())
        f[f"attn_entropy_{label}"]   = entropy(attn_seg)
    else:
        f[f"agree_attn_gc_{label}"]  = _nan()
        f[f"agree_attn_ras_{label}"] = _nan()
        f[f"attn_entropy_{label}"]   = _nan()

    if rw is not None and rw.size == gc.size:
        rw_f = rw.ravel()
        f[f"rw_gini_{label}"]         = gini(rw_f)
        f[f"agree_gc_rw_{label}"]     = cosine(gc.ravel(), rw_f)
        f[f"agree_ras_rw_{label}"]    = cosine(ras.ravel(), rw_f)
        f[f"agree_rn_rw_{label}"]     = cosine(rn.ravel(), rw_f)
        f[f"iou_gc_rw_{label}"]       = _topk_iou(gc, rw_f)
        f[f"iou_ras_rw_{label}"]      = _topk_iou(ras, rw_f)
        f[f"iou_rn_rw_{label}"]       = _topk_iou(rn, rw_f)
        f[f"com_dist_gc_rw_{label}"]  = _com_dist_2d(gc, rw_f)
        f[f"emd_gc_rw_{label}"]       = _emd_2d(gc, rw_f)
        f[f"com_dist_ras_rw_{label}"] = _com_dist_2d(ras, rw_f)
        f[f"emd_ras_rw_{label}"]      = _emd_2d(ras, rw_f)
        f[f"com_dist_rn_rw_{label}"]  = _com_dist_2d(rn, rw_f)
        f[f"emd_rn_rw_{label}"]       = _emd_2d(rn, rw_f)
    else:
        for pfx in ("rw_gini", "agree_gc_rw", "agree_ras_rw", "agree_rn_rw",
                    "iou_gc_rw", "iou_ras_rw", "iou_rn_rw",
                    "com_dist_gc_rw", "emd_gc_rw", "com_dist_ras_rw", "emd_ras_rw",
                    "com_dist_rn_rw", "emd_rn_rw"):
            f[f"{pfx}_{label}"] = _nan()

    if vc is not None and vc.size == gc.size:
        vc_f = vc.ravel()
        f[f"vc_gini_{label}"]         = gini(vc_f)
        f[f"agree_gc_vc_{label}"]     = cosine(gc.ravel(), vc_f)
        f[f"agree_ras_vc_{label}"]    = cosine(ras.ravel(), vc_f)
        f[f"agree_rn_vc_{label}"]     = cosine(rn.ravel(), vc_f)
        f[f"iou_gc_vc_{label}"]       = _topk_iou(gc, vc_f)
        f[f"iou_ras_vc_{label}"]      = _topk_iou(ras, vc_f)
        f[f"iou_rn_vc_{label}"]       = _topk_iou(rn, vc_f)
        f[f"com_dist_gc_vc_{label}"]  = _com_dist_2d(gc, vc_f)
        f[f"emd_gc_vc_{label}"]       = _emd_2d(gc, vc_f)
        f[f"com_dist_ras_vc_{label}"] = _com_dist_2d(ras, vc_f)
        f[f"emd_ras_vc_{label}"]      = _emd_2d(ras, vc_f)
        f[f"com_dist_rn_vc_{label}"]  = _com_dist_2d(rn, vc_f)
        f[f"emd_rn_vc_{label}"]       = _emd_2d(rn, vc_f)
        if rw is not None and rw.size == gc.size:
            f[f"agree_rw_vc_{label}"]    = cosine(rw_f, vc_f)
            f[f"iou_rw_vc_{label}"]      = _topk_iou(rw_f, vc_f)
            f[f"com_dist_rw_vc_{label}"] = _com_dist_2d(rw_f, vc_f)
            f[f"emd_rw_vc_{label}"]      = _emd_2d(rw_f, vc_f)
        else:
            f[f"agree_rw_vc_{label}"]    = _nan()
            f[f"iou_rw_vc_{label}"]      = _nan()
            f[f"com_dist_rw_vc_{label}"] = _nan()
            f[f"emd_rw_vc_{label}"]      = _nan()
    else:
        for pfx in ("vc_gini",
                    "agree_gc_vc", "agree_ras_vc", "agree_rn_vc", "agree_rw_vc",
                    "iou_gc_vc", "iou_ras_vc", "iou_rn_vc", "iou_rw_vc",
                    "com_dist_gc_vc", "emd_gc_vc", "com_dist_ras_vc", "emd_ras_vc",
                    "com_dist_rn_vc", "emd_rn_vc", "com_dist_rw_vc", "emd_rw_vc"):
            f[f"{pfx}_{label}"] = _nan()

    return f


def extract_heatmap_features(rec: dict, fmt: str, attn_feats: dict) -> dict:
    """Per-camera heatmap scalars for base, wrist + cross-camera agreement.

    Returns public feature keys and a private _maps dict for per-token extractor.
    """
    feats: dict = {}
    cams = [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist")]
    maps: dict = {}

    for cam_key, cam_label in cams:
        if fmt == "pi0fast":
            gc, ras, rn, gcs_tok, ras_tok, rns_tok = _load_heatmap_pi0fast(rec, cam_key)
        else:
            gc, ras, rn, gcs_tok, ras_tok, rns_tok = _load_heatmap_pi05(rec, cam_key)
        attn_seg = attn_feats.get(f"_attn_{cam_label}")
        rw_seg   = attn_feats.get(f"_rw_{cam_label}")
        vc_seg   = attn_feats.get(f"_vc_{cam_label}")
        feats.update(_heatmap_scalars(gc, ras, rn, attn_seg, cam_label, rw=rw_seg, vc=vc_seg))
        maps[cam_label] = (gc, ras, rn, gcs_tok, ras_tok, rns_tok)

    # Cross-camera agreement (cosine, IoU, CoM-dist, EMD) for each signal type
    for sig, idx in [("gc", 0), ("ras", 1), ("rn", 2)]:
        for l1, l2 in [("base", "wrist")]:
            m1 = maps[l1][idx]
            m2 = maps[l2][idx]
            ok = m1 is not None and m2 is not None
            feats[f"agree_{l1}_{l2}_{sig}"]    = cosine(m1, m2)       if ok else _nan()
            feats[f"iou_{l1}_{l2}_{sig}"]      = _topk_iou(m1, m2)   if ok else _nan()
            feats[f"com_dist_{l1}_{l2}_{sig}"] = _com_dist_2d(m1, m2) if ok else _nan()
            feats[f"emd_{l1}_{l2}_{sig}"]      = _emd_2d(m1, m2)      if ok else _nan()

    feats["_maps"] = maps
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR C: PER-ACTION-TOKEN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_per_token(rec: dict, fmt: str, maps_data: dict) -> dict:
    """Per-action-token/slot Gini values (t0–t4) for cameras and task tokens.

    Stores individual values so callers can compute any aggregation (mean, max,
    weighted, first-token-only) without re-reading raw .npy files.
    Keys: gc_gini_{cam}_{t0..t4}, ras_gini_..., rn_gini_..., gc_gini_{cam}_std,
          horizon_agree_{gc,ras}_{cam}, gc_com_y/x_base_t0.
    Works for both pi0fast (action_tokens/{i}/...) and pi05 when per-slot keys
    are present; NaN for pi05 when only the single aggregate key was recorded.
    """
    feats: dict = {}
    N_TOK = 5

    # Cameras
    for cam_label in ("base", "wrist"):
        _, _, _, gcs, ras_l, rns = maps_data.get(cam_label, (None, None, None, [], [], []))

        if not gcs:
            for sig in ("gc", "ras", "rn"):
                for i in range(N_TOK):
                    feats[f"{sig}_gini_{cam_label}_t{i}"] = _nan()
                feats[f"{sig}_gini_{cam_label}_std"] = _nan()
            for sig in ("gc", "ras"):
                feats[f"horizon_agree_{sig}_{cam_label}"] = _nan()
            if cam_label == "base":
                feats["gc_com_y_base_t0"] = _nan()
                feats["gc_com_x_base_t0"] = _nan()
            continue

        gc_ginis  = [gini(gcs[i].ravel())  if i < len(gcs)  else _nan() for i in range(N_TOK)]
        ras_ginis = [gini(ras_l[i].ravel()) if i < len(ras_l) else _nan() for i in range(N_TOK)]
        rn_ginis  = [gini(rns[i].ravel())  if i < len(rns)  else _nan() for i in range(N_TOK)]

        for i in range(N_TOK):
            feats[f"gc_gini_{cam_label}_t{i}"]  = gc_ginis[i]
            feats[f"ras_gini_{cam_label}_t{i}"] = ras_ginis[i]
            feats[f"rn_gini_{cam_label}_t{i}"]  = rn_ginis[i]

        feats[f"gc_gini_{cam_label}_std"]  = float(np.nanstd(gc_ginis))
        feats[f"ras_gini_{cam_label}_std"] = float(np.nanstd(ras_ginis))
        feats[f"rn_gini_{cam_label}_std"]  = float(np.nanstd(rn_ginis))

        feats[f"horizon_agree_gc_{cam_label}"]  = cosine(gcs[0], gcs[-1])
        feats[f"horizon_agree_ras_{cam_label}"] = cosine(ras_l[0], ras_l[-1])

        if cam_label == "base":
            cy, cx = spatial_com(gcs[0])
            feats["gc_com_y_base_t0"] = cy
            feats["gc_com_x_base_t0"] = cx

    # Task per-token
    gc_task_list, ras_task_list, rn_task_list = [], [], []
    for i in range(N_TOK):
        kg  = f"outputs/debug/gradcam/action_tokens/{i}/task"
        kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/task"
        krn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
        if kg not in rec:
            break
        gc_task_list.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
        ras_task_list.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32))
        rn_task_list.append(np.abs(np.asarray(rec[krn])[0]).astype(np.float32))

    for i in range(N_TOK):
        feats[f"gc_gini_task_t{i}"]  = gini(gc_task_list[i])  if i < len(gc_task_list)  else _nan()
        feats[f"ras_gini_task_t{i}"] = gini(ras_task_list[i]) if i < len(ras_task_list) else _nan()
        feats[f"rn_gini_task_t{i}"]  = gini(rn_task_list[i])  if i < len(rn_task_list)  else _nan()

    feats["horizon_agree_gc_task"]  = (
        cosine(gc_task_list[0], gc_task_list[-1])   if len(gc_task_list)  >= 2 else _nan()
    )
    feats["horizon_agree_ras_task"] = (
        cosine(ras_task_list[0], ras_task_list[-1]) if len(ras_task_list) >= 2 else _nan()
    )

    # State per-action-token Gini
    gc_state_list2, ras_state_list2 = [], []
    for i in range(N_TOK):
        kg  = f"outputs/debug/gradcam/action_tokens/{i}/state"
        kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/state"
        if kg not in rec:
            break
        gc_state_list2.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
        ras_state_list2.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32) if kr in rec else np.zeros_like(gc_state_list2[-1]))

    gc_state_ginis  = [gini(gc_state_list2[i])  if i < len(gc_state_list2)  else _nan() for i in range(N_TOK)]
    ras_state_ginis = [gini(ras_state_list2[i]) if i < len(ras_state_list2) else _nan() for i in range(N_TOK)]

    for i in range(N_TOK):
        feats[f"gc_gini_state_t{i}"]  = gc_state_ginis[i]
        feats[f"ras_gini_state_t{i}"] = ras_state_ginis[i]

    feats["gc_gini_state_std"]  = float(np.nanstd(gc_state_ginis))
    feats["ras_gini_state_std"] = float(np.nanstd(ras_state_ginis))

    feats["horizon_agree_gc_state"]  = (
        cosine(gc_state_list2[0], gc_state_list2[-1])   if len(gc_state_list2)  >= 2 else _nan()
    )
    feats["horizon_agree_ras_state"] = (
        cosine(ras_state_list2[0], ras_state_list2[-1]) if len(ras_state_list2) >= 2 else _nan()
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR D: TASK & STATE TOKEN ATTRIBUTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_task_state(rec: dict, fmt: str, attn_feats: dict) -> dict:
    """Attribution features for task and state token sequences.

    Task keys: gc_gini_task, ras_gini_task, rn_gini_task,
               agree_{gc,ras,rn,rw,vc}_task, iou_*, com_dist_*, emd_* (all pairs)
    State keys: gc_gini_state, ras_gini_state, rw_gini_state, vc_gini_state,
                agree_*/com_dist_*/emd_* (available pairs only)
    Also includes inter-head agreement for task tokens (task_head_agr).
    """
    feats: dict = {}

    # ── Task tokens ──────────────────────────────────────────────────────────
    if fmt == "pi0fast":
        gc_task_list, ras_task_list, rn_task_list = [], [], []
        for i in range(5):
            kg  = f"outputs/debug/gradcam/action_tokens/{i}/task"
            kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/task"
            krn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
            if kg not in rec:
                break
            gc_task_list.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
            ras_task_list.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32))
            rn_task_list.append(np.abs(np.asarray(rec[krn])[0]).astype(np.float32))
        gc_task  = np.mean(np.stack(gc_task_list),  axis=0) if gc_task_list  else None
        ras_task = np.mean(np.stack(ras_task_list), axis=0) if ras_task_list else None
        rn_task  = np.mean(np.stack(rn_task_list),  axis=0) if rn_task_list  else None
        gc_task_full = ras_task_full = rn_task_full = None

        # State tokens (pi0fast only, digit-only slice)
        gc_state_list, ras_state_list = [], []
        for i in range(5):
            kg  = f"outputs/debug/gradcam/action_tokens/{i}/state"
            kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/state"
            if kg not in rec:
                break
            gc_state_list.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
            ras_state_list.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32))
        if gc_state_list:
            dk, dend = _state_digit_slice(rec)
            gc_state  = np.mean(np.stack(gc_state_list),  axis=0)[dk:dend]
            ras_state = np.mean(np.stack(ras_state_list), axis=0)[dk:dend]
        else:
            gc_state = ras_state = None
    else:
        # pi05: try per-slot keys first (slots 0-4, same pattern as pi0fast).
        # Fall back to single aggregate key if per-slot data is absent.
        gc_task_list, ras_task_list, rn_task_list = [], [], []
        for i in range(5):
            kg  = f"outputs/debug/gradcam/action_tokens/{i}/task"
            kr  = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/task"
            krn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
            if kg not in rec:
                break
            gc_task_list.append(np.abs(np.squeeze(np.asarray(rec[kg]))).astype(np.float32))
            ras_task_list.append(np.abs(np.squeeze(np.asarray(rec[kr]))).astype(np.float32))
            rn_task_list.append(np.abs(np.squeeze(np.asarray(rec[krn]))).astype(np.float32))

        if gc_task_list:
            gc_task  = np.mean(np.stack(gc_task_list),  axis=0)
            ras_task = np.mean(np.stack(ras_task_list), axis=0)
            rn_task  = np.mean(np.stack(rn_task_list),  axis=0)
            gc_task_full = ras_task_full = rn_task_full = None
        else:
            kg  = "outputs/debug/gradcam/task"
            kr  = "outputs/debug/raw_alpha/summation/task"
            krn = "outputs/debug/raw_alpha/norm/task"
            gc_task_full  = np.abs(np.asarray(rec[kg])).astype(np.float32)  if kg  in rec else None
            ras_task_full = np.abs(np.asarray(rec[kr])).astype(np.float32)  if kr  in rec else None
            rn_task_full  = np.abs(np.asarray(rec[krn])).astype(np.float32) if krn in rec else None

            if gc_task_full is not None:
                tmk = "outputs/debug/tokens/task/token_mask"
                if tmk in rec:
                    tm = np.asarray(rec[tmk]).astype(bool)
                    n_mask = min(len(tm), len(gc_task_full))
                    n_real = int(tm[:n_mask].sum())
                    gc_task  = gc_task_full[:n_real]  if n_real > 0 else gc_task_full
                    ras_task = ras_task_full[:n_real] if (ras_task_full is not None and n_real > 0) else ras_task_full
                    rn_task  = rn_task_full[:n_real]  if (rn_task_full  is not None and n_real > 0) else rn_task_full
                else:
                    gc_task, ras_task, rn_task = gc_task_full, ras_task_full, rn_task_full
            else:
                gc_task = ras_task = rn_task = None

        gc_state = ras_state = None

    attn_task = attn_feats.get("_attn_task")
    rw_task   = attn_feats.get("_rw_task")
    vc_task   = attn_feats.get("_vc_task")

    _TASK_NAN = (
        "gc_gini_task", "ras_gini_task", "rn_gini_task",
        "agree_gc_ras_task", "agree_gc_rn_task", "agree_ras_rn_task",
        "agree_attn_gc_task", "agree_attn_ras_task",
        "rw_gini_task", "agree_gc_rw_task", "agree_ras_rw_task", "agree_rn_rw_task",
        "iou_gc_ras_task", "iou_gc_rn_task",
        "com_dist_gc_ras_task", "com_dist_gc_rn_task",
        "emd_gc_ras_task", "emd_gc_rn_task",
        "com_dist_ras_rn_task", "emd_ras_rn_task",
        "com_dist_gc_rw_task", "emd_gc_rw_task",
        "com_dist_ras_rw_task", "emd_ras_rw_task",
        "com_dist_rn_rw_task", "emd_rn_rw_task",
        "vc_gini_task",
        "agree_gc_vc_task", "agree_ras_vc_task", "agree_rn_vc_task", "agree_rw_vc_task",
        "com_dist_gc_vc_task", "emd_gc_vc_task",
        "com_dist_ras_vc_task", "emd_ras_vc_task",
        "com_dist_rn_vc_task", "emd_rn_vc_task",
        "com_dist_rw_vc_task", "emd_rw_vc_task",
        "task_head_agr",
    )

    if gc_task is None:
        for k in _TASK_NAN:
            feats[k] = _nan()
    else:
        # For cosine agreements use full (padded) vectors in pi05 fallback path to match
        # shapes; when per-slot averages are used (gc_task_full is None), use gc_task directly.
        _gc_cmp  = gc_task_full  if (fmt == "pi05" and gc_task_full  is not None) else gc_task
        _ras_cmp = ras_task_full if (fmt == "pi05" and ras_task_full is not None) else ras_task
        _rn_cmp  = rn_task_full  if (fmt == "pi05" and rn_task_full  is not None) else rn_task

        feats["gc_gini_task"]  = gini(gc_task)
        feats["ras_gini_task"] = gini(ras_task)
        feats["rn_gini_task"]  = gini(rn_task) if rn_task is not None else _nan()

        feats["agree_gc_ras_task"]  = cosine(_gc_cmp, _ras_cmp)
        feats["agree_gc_rn_task"]   = cosine(_gc_cmp, _rn_cmp)  if _rn_cmp  is not None else _nan()
        feats["agree_ras_rn_task"]  = cosine(_ras_cmp, _rn_cmp) if _rn_cmp  is not None else _nan()

        if attn_task is not None and attn_task.size == (_gc_cmp if _gc_cmp is not None else gc_task).size:
            feats["agree_attn_gc_task"]  = cosine(attn_task, _gc_cmp)
            feats["agree_attn_ras_task"] = cosine(attn_task, _ras_cmp)
        else:
            feats["agree_attn_gc_task"]  = _nan()
            feats["agree_attn_ras_task"] = _nan()

        feats["iou_gc_ras_task"] = _topk_iou(gc_task, ras_task)
        feats["iou_gc_rn_task"]  = _topk_iou(gc_task, rn_task)  if rn_task  is not None else _nan()

        feats["com_dist_gc_ras_task"] = _com_dist_1d(gc_task, ras_task)
        feats["com_dist_gc_rn_task"]  = _com_dist_1d(gc_task, rn_task) if rn_task is not None else _nan()
        feats["com_dist_ras_rn_task"] = _com_dist_1d(ras_task, rn_task) if rn_task is not None else _nan()

        feats["emd_gc_ras_task"]  = _emd_1d(gc_task, ras_task)
        feats["emd_gc_rn_task"]   = _emd_1d(gc_task, rn_task)  if rn_task  is not None else _nan()
        feats["emd_ras_rn_task"]  = _emd_1d(ras_task, rn_task) if rn_task  is not None else _nan()

        if rw_task is not None:
            feats["rw_gini_task"]         = gini(rw_task)
            feats["agree_gc_rw_task"]     = cosine(gc_task, rw_task)
            feats["agree_ras_rw_task"]    = cosine(ras_task, rw_task)
            feats["agree_rn_rw_task"]     = cosine(rn_task, rw_task)  if rn_task  is not None else _nan()
            feats["iou_gc_rw_task"]       = _topk_iou(gc_task, rw_task)
            feats["iou_ras_rw_task"]      = _topk_iou(ras_task, rw_task)
            feats["iou_rn_rw_task"]       = _topk_iou(rn_task, rw_task) if rn_task is not None else _nan()
            feats["com_dist_gc_rw_task"]  = _com_dist_1d(gc_task, rw_task)
            feats["emd_gc_rw_task"]       = _emd_1d(gc_task, rw_task)
            feats["com_dist_ras_rw_task"] = _com_dist_1d(ras_task, rw_task)
            feats["emd_ras_rw_task"]      = _emd_1d(ras_task, rw_task)
            feats["com_dist_rn_rw_task"]  = _com_dist_1d(rn_task, rw_task) if rn_task is not None else _nan()
            feats["emd_rn_rw_task"]       = _emd_1d(rn_task, rw_task)      if rn_task is not None else _nan()
        else:
            for k in ("rw_gini_task", "agree_gc_rw_task", "agree_ras_rw_task", "agree_rn_rw_task",
                      "iou_gc_rw_task", "iou_ras_rw_task", "iou_rn_rw_task",
                      "com_dist_gc_rw_task", "emd_gc_rw_task",
                      "com_dist_ras_rw_task", "emd_ras_rw_task",
                      "com_dist_rn_rw_task", "emd_rn_rw_task"):
                feats[k] = _nan()

        if vc_task is not None:
            feats["vc_gini_task"]         = gini(vc_task)
            feats["agree_gc_vc_task"]     = cosine(gc_task, vc_task)
            feats["agree_ras_vc_task"]    = cosine(ras_task, vc_task)
            feats["agree_rn_vc_task"]     = cosine(rn_task, vc_task)  if rn_task  is not None else _nan()
            feats["agree_rw_vc_task"]     = cosine(rw_task, vc_task)  if rw_task  is not None else _nan()
            feats["iou_gc_vc_task"]       = _topk_iou(gc_task, vc_task)
            feats["iou_ras_vc_task"]      = _topk_iou(ras_task, vc_task)
            feats["iou_rn_vc_task"]       = _topk_iou(rn_task, vc_task)  if rn_task is not None else _nan()
            feats["iou_rw_vc_task"]       = _topk_iou(rw_task, vc_task)  if rw_task is not None else _nan()
            feats["com_dist_gc_vc_task"]  = _com_dist_1d(gc_task, vc_task)
            feats["emd_gc_vc_task"]       = _emd_1d(gc_task, vc_task)
            feats["com_dist_ras_vc_task"] = _com_dist_1d(ras_task, vc_task)
            feats["emd_ras_vc_task"]      = _emd_1d(ras_task, vc_task)
            feats["com_dist_rn_vc_task"]  = _com_dist_1d(rn_task, vc_task) if rn_task is not None else _nan()
            feats["emd_rn_vc_task"]       = _emd_1d(rn_task, vc_task)      if rn_task is not None else _nan()
            feats["com_dist_rw_vc_task"]  = _com_dist_1d(rw_task, vc_task) if rw_task is not None else _nan()
            feats["emd_rw_vc_task"]       = _emd_1d(rw_task, vc_task)      if rw_task is not None else _nan()
        else:
            for k in ("vc_gini_task",
                      "agree_gc_vc_task", "agree_ras_vc_task", "agree_rn_vc_task", "agree_rw_vc_task",
                      "iou_gc_vc_task", "iou_ras_vc_task", "iou_rn_vc_task", "iou_rw_vc_task",
                      "com_dist_gc_vc_task", "emd_gc_vc_task",
                      "com_dist_ras_vc_task", "emd_ras_vc_task",
                      "com_dist_rn_vc_task", "emd_rn_vc_task",
                      "com_dist_rw_vc_task", "emd_rw_vc_task"):
                feats[k] = _nan()

        # Inter-head cosine agreement for task-attending heads
        sp_task = _span(rec, "outputs/debug/spans/task")
        wkey = "outputs/debug/attn/weights"
        if sp_task and wkey in rec:
            w = np.asarray(rec[wkey])
            attn_ph = w[-1, 0] if fmt == "pi0fast" else w[-1, :, 0, :]  # (H, S) — slot 0 for pi05
            span_mass = attn_ph[:, sp_task[0]:sp_task[1]].sum(axis=1)
            top_heads = np.argsort(span_mass)[-8:]
            h_task = attn_ph[top_heads][:, sp_task[0]:sp_task[1]]
            if len(top_heads) > 1:
                h_n = h_task / (np.linalg.norm(h_task, axis=1, keepdims=True) + 1e-9)
                sim = h_n @ h_n.T
                iu = np.triu_indices(len(top_heads), k=1)
                feats["task_head_agr"] = float(sim[iu].mean())
            else:
                feats["task_head_agr"] = _nan()
        else:
            feats["task_head_agr"] = _nan()

    # ── State tokens ──────────────────────────────────────────────────────────
    # Both formats use exactly 4 attribution signals for state: gc, ras, rw, vc.
    # Neither records raw_alpha/norm for state tokens (confirmed by step_metrics_plot.py
    # for pi0fast and analysis_figures.py for pi05 — both use only gc, ra, rw, vc).
    dk, dend = _state_digit_slice(rec)

    # pi05: try per-slot state keys first (slots 0-4); fall back to single aggregate key.
    # pi0fast: gc_state/ras_state already set and digit-sliced in the task branch above.
    if fmt == "pi05":
        gc_state_sl, ras_state_sl = [], []
        for i in range(5):
            _k = f"outputs/debug/gradcam/action_tokens/{i}/state"
            _r = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/state"
            if _k not in rec:
                break
            gc_state_sl.append(np.abs(np.squeeze(np.asarray(rec[_k]))).astype(np.float32))
            ras_state_sl.append(np.abs(np.squeeze(np.asarray(rec[_r]))).astype(np.float32) if _r in rec else np.zeros_like(gc_state_sl[-1]))
        if gc_state_sl:
            gc_state  = np.mean(np.stack(gc_state_sl),  axis=0)[dk:dend]
            ras_state = np.mean(np.stack(ras_state_sl), axis=0)[dk:dend]
            rn_state_extra = None
        else:
            _k_gc_s  = "outputs/debug/gradcam/state"
            _k_ras_s = "outputs/debug/raw_alpha/summation/state"
            _k_rn_s  = "outputs/debug/raw_alpha/norm/state"
            gc_state  = (np.abs(np.asarray(rec[_k_gc_s])).astype(np.float32)[dk:dend]
                         if _k_gc_s  in rec else None)
            ras_state = (np.abs(np.asarray(rec[_k_ras_s])).astype(np.float32)[dk:dend]
                         if _k_ras_s in rec else None)
            rn_state_extra = (np.abs(np.asarray(rec[_k_rn_s])).astype(np.float32)[dk:dend]
                              if _k_rn_s in rec else None)
    else:
        rn_state_extra = None

    rw_state_raw = attn_feats.get("_rw_state")
    vc_state_raw = attn_feats.get("_vc_state")
    rw_state = rw_state_raw[dk:dend] if rw_state_raw is not None else None
    vc_state = vc_state_raw[dk:dend] if vc_state_raw is not None else None

    # Gini for each of the 5 signals (rn_state is pi05-only; NaN for pi0fast)
    feats["gc_gini_state"]  = gini(gc_state)        if gc_state        is not None else _nan()
    feats["ras_gini_state"] = gini(ras_state)       if ras_state       is not None else _nan()
    feats["rn_gini_state"]  = gini(rn_state_extra)  if rn_state_extra  is not None else _nan()
    feats["rw_gini_state"]  = gini(rw_state)        if rw_state        is not None else _nan()
    feats["vc_gini_state"]  = gini(vc_state)        if vc_state        is not None else _nan()

    # All pairwise agreements × 4 metrics (cosine, iou, com_dist, emd)
    # rn_state only available for pi05; pairs involving it are NaN for pi0fast.
    for (t1, v1), (t2, v2) in [
        (("gc",  gc_state),       ("ras", ras_state)),
        (("gc",  gc_state),       ("rn",  rn_state_extra)),
        (("gc",  gc_state),       ("rw",  rw_state)),
        (("gc",  gc_state),       ("vc",  vc_state)),
        (("ras", ras_state),      ("rn",  rn_state_extra)),
        (("ras", ras_state),      ("rw",  rw_state)),
        (("ras", ras_state),      ("vc",  vc_state)),
        (("rn",  rn_state_extra), ("rw",  rw_state)),
        (("rn",  rn_state_extra), ("vc",  vc_state)),
        (("rw",  rw_state),       ("vc",  vc_state)),
    ]:
        if v1 is not None and v2 is not None:
            feats[f"agree_{t1}_{t2}_state"]    = cosine(v1, v2)
            feats[f"iou_{t1}_{t2}_state"]      = _topk_iou(v1, v2)
            feats[f"com_dist_{t1}_{t2}_state"] = _com_dist_1d(v1, v2)
            feats[f"emd_{t1}_{t2}_state"]      = _emd_1d(v1, v2)
        else:
            feats[f"agree_{t1}_{t2}_state"]    = _nan()
            feats[f"iou_{t1}_{t2}_state"]      = _nan()
            feats[f"com_dist_{t1}_{t2}_state"] = _nan()
            feats[f"emd_{t1}_{t2}_state"]      = _nan()

    # Private: expose averaged attribution maps for extract_extra_features.
    feats["_gc_task"]        = gc_task
    feats["_ras_task"]       = ras_task
    feats["_rn_task"]        = rn_task
    feats["_gc_state"]       = gc_state
    feats["_ras_state"]      = ras_state
    feats["_rn_state_extra"] = rn_state_extra

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR E: VALUE VECTOR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_value_features(rec: dict, fmt: str,
                            prev_centroids: Optional[dict] = None) -> Tuple[dict, dict]:
    """Value-vector (attn/v) features.

    Returns (feats, centroids).  Pass centroids to next step for drift features.

    Keys: v_norm_mean_{base,wrist,task}, v_norm_gini_{...}, v_coherence_{...},
          v_align_{base_task, base_wrist, task_wrist},
          v_drift_{base, wrist, task}   (NaN at first step)
    """
    feats: dict = {}
    centroids: dict = {}
    vkey = "outputs/debug/attn/v"

    _NAN_KEYS = ("v_norm_mean_base", "v_norm_gini_base", "v_coherence_base",
                 "v_norm_mean_wrist", "v_norm_gini_wrist", "v_coherence_wrist",
                 "v_norm_mean_task",  "v_norm_gini_task",  "v_coherence_task",
                 "v_norm_mean_state", "v_norm_gini_state", "v_coherence_state",
                 "v_align_base_task", "v_align_base_wrist", "v_align_task_wrist",
                 "v_align_base_state", "v_align_wrist_state", "v_align_task_state",
                 "v_drift_base", "v_drift_wrist", "v_drift_task", "v_drift_state")

    if vkey not in rec:
        for k in _NAN_KEYS:
            feats[k] = _nan()
        return feats, centroids

    v_raw = np.asarray(rec[vkey]).astype(np.float32)
    if fmt == "pi0fast":
        V = v_raw[-1, 0, :, 0, :]  # (S, D)
    else:
        V = v_raw[-1, :, 0, :]     # (S, D)

    sp_base  = _span(rec, "outputs/debug/spans/image/base_0_rgb")
    sp_wrist = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
    sp_task  = _span(rec, "outputs/debug/spans/task")

    # Truncate pi05 task span to real tokens to avoid near-zero-vector distortion.
    if fmt == "pi05" and sp_task is not None:
        tmk = "outputs/debug/tokens/task/token_mask"
        if tmk in rec:
            tm = np.asarray(rec[tmk]).astype(bool)
            nr = int(tm[:sp_task[1] - sp_task[0]].sum())
            if nr > 0:
                sp_task = (sp_task[0], sp_task[0] + nr)

    def modality_stats(sp, tag):
        if sp is None:
            feats[f"v_norm_mean_{tag}"]  = _nan()
            feats[f"v_norm_gini_{tag}"]  = _nan()
            feats[f"v_coherence_{tag}"]  = _nan()
            centroids[f"v_centroid_{tag}"] = None
            return
        seg = V[sp[0]:sp[1]]                      # (N, D)
        norms = np.linalg.norm(seg, axis=1)
        feats[f"v_norm_mean_{tag}"] = float(norms.mean())
        feats[f"v_norm_gini_{tag}"] = gini(norms)
        centroid = seg.mean(axis=0)
        cnorm = np.linalg.norm(centroid)
        if cnorm > 1e-9:
            c_unit = centroid / cnorm
            dot = seg @ c_unit
            feats[f"v_coherence_{tag}"] = float((dot / (norms + 1e-9)).mean())
        else:
            feats[f"v_coherence_{tag}"] = _nan()
        centroids[f"v_centroid_{tag}"] = centroid

    sp_state_v = _span(rec, "outputs/debug/spans/state")

    modality_stats(sp_base,    "base")
    modality_stats(sp_wrist,   "wrist")
    modality_stats(sp_task,    "task")
    modality_stats(sp_state_v, "state")

    for t1, t2 in [("base", "task"), ("base", "wrist"), ("task", "wrist"),
                   ("base", "state"), ("wrist", "state"), ("task", "state")]:
        c1 = centroids.get(f"v_centroid_{t1}")
        c2 = centroids.get(f"v_centroid_{t2}")
        feats[f"v_align_{t1}_{t2}"] = cosine(c1, c2) if (c1 is not None and c2 is not None) else _nan()

    for tag in ("base", "wrist", "task", "state"):
        pc = prev_centroids.get(f"v_centroid_{tag}") if prev_centroids else None
        cc = centroids.get(f"v_centroid_{tag}")
        if pc is not None and cc is not None:
            feats[f"v_drift_{tag}"] = 1.0 - cosine(cc, pc)
        else:
            feats[f"v_drift_{tag}"] = _nan()

    return feats, centroids


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR F: ACTION & MOTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_action_features(rec: dict, prev_rec: Optional[dict] = None) -> dict:
    """Action oscillation, plan statistics, and joint motion features.

    Keys: action_trans_flip, action_rot_flip, action_grip_flip,
          action_magnitude, gripper_val,
          action_spread, action_plan_norm, action_horizon_align, action_dir_var,
          joint_vel, plan_consistency, action_grip_change, state_norm
    """
    feats: dict = {}
    actions = np.asarray(rec["outputs/actions"]).astype(np.float64)  # (10, 7)

    signs = np.sign(actions)
    flips = (signs[1:] != signs[:-1]).astype(float)
    feats["action_trans_flip"] = float(flips[:, 0:3].mean())
    feats["action_rot_flip"]   = float(flips[:, 3:6].mean())
    feats["action_grip_flip"]  = float(flips[:, 6].mean())
    feats["action_magnitude"]  = float(np.linalg.norm(actions[0]))
    feats["gripper_val"]       = float(actions[0, 6])

    step_norms = np.linalg.norm(actions, axis=1)  # (10,)
    feats["action_spread"]        = float(step_norms.std())
    feats["action_plan_norm"]     = float(step_norms.mean())
    feats["action_plan_entropy"]  = entropy(step_norms)
    feats["action_horizon_align"] = cosine(actions[0], actions[-1])
    feats["action_dir_var"]       = float(np.var(actions[:, 0:6], axis=0).mean())

    state_key = "outputs/state" if "outputs/state" in rec else "inputs/observation/state"
    state = np.asarray(rec.get(state_key, np.zeros(8))).astype(np.float64)[:8]
    feats["state_norm"] = float(np.linalg.norm(state))

    if prev_rec is not None:
        prev_sk = "outputs/state" if "outputs/state" in prev_rec else "inputs/observation/state"
        prev_state = np.asarray(prev_rec.get(prev_sk, state)).astype(np.float64)[:8]
        feats["joint_vel"] = float(np.linalg.norm(state - prev_state))

        prev_actions = np.asarray(prev_rec["outputs/actions"]).astype(np.float64)
        a0, p0 = actions[0], prev_actions[0]
        feats["plan_consistency"]   = float(np.dot(a0, p0) / (np.linalg.norm(a0) * np.linalg.norm(p0) + 1e-9))
        feats["action_grip_change"] = float(abs(actions[0, 6] - prev_actions[0, 6]))
    else:
        feats["joint_vel"]          = 0.0
        feats["plan_consistency"]   = 1.0
        feats["action_grip_change"] = 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# SUB-EXTRACTOR G: MASS RATIOS, MODALITY RATIOS, TEMPORAL, ENTROPY / TOP-K
# ─────────────────────────────────────────────────────────────────────────────

def extract_extra_features(
    maps_data: dict,
    attn_feats: dict,
    ts_feats: dict,
    rec: dict,
    prev_rec: Optional[dict],
    fmt: str,
) -> dict:
    """Mass ratios, image/task/state modality ratios, temporal agreement and CoM
    drift for all signals, and entropy/top-k concentration for task/state tokens.

    Keys produced
    -------------
    {gc,ras,rn,rw,vc}_mass_ratio_base_wrist
    {gc,ras,rn,rw,vc}_{image,task,state}_ratio  (image = base + wrist)
    {gc,ras,rn,rw,vc}_temporal_agree_{base,wrist}
    {ras,rn,rw,vc}_com_drift_{base,wrist}        (gc_com_drift_* lives in main extractor)
    {gc,ras,rn,rw,vc}_entropy_{task,state}
    {gc,ras,rw,vc}_top1_frac_{task,state}
    {gc,ras,rw,vc}_top3_frac_{task,state}
    """
    feats: dict = {}
    _EPS = 1e-12

    def _mass(v):
        return float(np.abs(v.ravel()).sum()) if v is not None else 0.0

    def _ratio(num_v, den_v):
        if num_v is None:
            return _nan()
        return _mass(num_v) / (_mass(den_v) + _EPS)

    # ── retrieve all current maps ────────────────────────────────────────────
    gc_base,  ras_base,  rn_base  = maps_data.get("base",  (None,) * 6)[:3]
    gc_wrist, ras_wrist, rn_wrist = maps_data.get("wrist", (None,) * 6)[:3]
    rw_base  = attn_feats.get("_rw_base");   rw_wrist = attn_feats.get("_rw_wrist")
    vc_base  = attn_feats.get("_vc_base");   vc_wrist = attn_feats.get("_vc_wrist")

    gc_task  = ts_feats.get("_gc_task");     ras_task = ts_feats.get("_ras_task")
    rn_task  = ts_feats.get("_rn_task");     rw_task  = attn_feats.get("_rw_task")
    vc_task  = attn_feats.get("_vc_task")

    gc_state  = ts_feats.get("_gc_state");   ras_state = ts_feats.get("_ras_state")
    rn_state  = ts_feats.get("_rn_state_extra")
    dk, dend  = _state_digit_slice(rec)
    rw_state_raw = attn_feats.get("_rw_state")
    vc_state_raw = attn_feats.get("_vc_state")
    rw_state = rw_state_raw[dk:dend] if rw_state_raw is not None else None
    vc_state = vc_state_raw[dk:dend] if vc_state_raw is not None else None

    # ── 1. base-vs-wrist mass ratios ─────────────────────────────────────────
    for sig, bv, wv in [
        ("gc",  gc_base,  gc_wrist),
        ("ras", ras_base, ras_wrist),
        ("rn",  rn_base,  rn_wrist),
        ("rw",  rw_base,  rw_wrist),
        ("vc",  vc_base,  vc_wrist),
    ]:
        feats[f"{sig}_mass_ratio_base_wrist"] = _ratio(bv, wv)

    # ── 2. image / task / state modality ratios ───────────────────────────────
    for sig, bv, wv, tv, sv in [
        ("gc",  gc_base,  gc_wrist,  gc_task,  gc_state),
        ("ras", ras_base, ras_wrist, ras_task, ras_state),
        ("rn",  rn_base,  rn_wrist,  rn_task,  rn_state),
        ("rw",  rw_base,  rw_wrist,  rw_task,  rw_state),
        ("vc",  vc_base,  vc_wrist,  vc_task,  vc_state),
    ]:
        img_m  = _mass(bv) + _mass(wv)
        task_m = _mass(tv)
        st_m   = _mass(sv)
        has_img = bv is not None or wv is not None
        feats[f"{sig}_image_task_ratio"]  = img_m  / (task_m + _EPS) if has_img else _nan()
        feats[f"{sig}_image_state_ratio"] = img_m  / (st_m   + _EPS) if has_img else _nan()
        feats[f"{sig}_task_state_ratio"]  = task_m / (st_m   + _EPS) if tv is not None else _nan()

    # ── 3. temporal agreement and CoM drift ──────────────────────────────────
    _cam_pairs = [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist")]

    prev_hm: dict = {}
    if prev_rec is not None:
        for cam_key, cam_label in _cam_pairs:
            if fmt == "pi0fast":
                prev_hm[cam_label] = _load_heatmap_pi0fast(prev_rec, cam_key)
            else:
                prev_hm[cam_label] = _load_heatmap_pi05(prev_rec, cam_key)

    for sig_idx, sig in enumerate(["gc", "ras", "rn"]):
        for _, cam_label in _cam_pairs:
            cur_map  = maps_data.get(cam_label, (None,) * 6)[sig_idx]
            prev_tup = prev_hm.get(cam_label)
            prev_map = prev_tup[sig_idx] if prev_tup is not None else None
            ok = cur_map is not None and prev_map is not None
            feats[f"{sig}_temporal_agree_{cam_label}"] = cosine(cur_map, prev_map)      if ok else _nan()
            if sig != "gc":  # gc_com_drift_{base,wrist} already emitted by main extractor
                feats[f"{sig}_com_drift_{cam_label}"] = _com_dist_2d(cur_map, prev_map) if ok else _nan()

    prev_ar: dict = extract_attn_routing(prev_rec, fmt) if prev_rec is not None else {}
    for sig in ("rw", "vc"):
        for _, cam_label in _cam_pairs:
            cur_seg  = attn_feats.get(f"_{sig}_{cam_label}")
            prev_seg = prev_ar.get(f"_{sig}_{cam_label}")
            ok = (cur_seg is not None and prev_seg is not None
                  and cur_seg.shape == prev_seg.shape)
            feats[f"{sig}_temporal_agree_{cam_label}"] = cosine(cur_seg, prev_seg)       if ok else _nan()
            feats[f"{sig}_com_drift_{cam_label}"]      = _com_dist_1d(cur_seg, prev_seg) if ok else _nan()

    # ── 4. entropy and top-k concentration for task / state ──────────────────
    for mod, gc_v, ras_v, rn_v, rw_v, vc_v in [
        ("task",  gc_task,  ras_task,  rn_task,  rw_task,  vc_task),
        ("state", gc_state, ras_state, rn_state, rw_state, vc_state),
    ]:
        for sig, v in [("gc", gc_v), ("ras", ras_v), ("rn", rn_v), ("rw", rw_v), ("vc", vc_v)]:
            feats[f"{sig}_entropy_{mod}"] = entropy(v) if v is not None else _nan()
        for sig, v in [("gc", gc_v), ("ras", ras_v), ("rw", rw_v), ("vc", vc_v)]:
            feats[f"{sig}_top1_frac_{mod}"] = _top_k_frac(v, 1)
            feats[f"{sig}_top3_frac_{mod}"] = _top_k_frac(v, 3)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(rec: dict,
                     prev_rec: Optional[dict] = None,
                     prev_centroids: Optional[dict] = None) -> Tuple[dict, dict]:
    """Extract all scalar features from one step record.

    Returns (feats, centroids).  Pass both to the next call:
      feats, centroids = extract_features(rec, prev_rec=prev_rec,
                                          prev_centroids=prev_centroids)

    Private keys (prefixed _) are stripped before returning.
    """
    fmt = detect_format(rec)
    feats: dict = {}

    # A. Attention routing (also produces private _attn_*, _rw_*, _vc_* intermediates)
    attn_feats = extract_attn_routing(rec, fmt)
    feats.update({k: v for k, v in attn_feats.items() if not k.startswith("_")})

    # B. Per-camera heatmap scalars + cross-camera agreement
    hm_feats = extract_heatmap_features(rec, fmt, attn_feats)
    maps_data = hm_feats.pop("_maps", {})
    feats.update(hm_feats)

    # C. Per-action-token/slot Gini values (NaN for pi05 when per-slot keys absent)
    feats.update(extract_per_token(rec, fmt, maps_data))

    # D. Task and state token attribution features
    ts_feats = extract_task_state(rec, fmt, attn_feats)
    feats.update({k: v for k, v in ts_feats.items() if not k.startswith("_")})

    # E. Value vector features
    v_feats, centroids = extract_value_features(rec, fmt, prev_centroids)
    feats.update(v_feats)

    # F. Action and motion features
    feats.update(extract_action_features(rec, prev_rec))

    # G. Extra features: mass ratios, modality ratios, temporal, entropy/top-k
    feats.update(extract_extra_features(maps_data, attn_feats, ts_feats, rec, prev_rec, fmt))

    # Attention mass scalars — recomputed from raw weights for both models so that:
    #   • all layers are averaged (matching pi05's recording formula)
    #   • rwrist is excluded from image_mass (redundant in both setups)
    # pi0fast: w shape (L, 1, H, S) → mean over (L, H), single decode step
    # pi05:    w shape (L, H, T_action, S) → mean over (L, H, T_action)
    wkey_am = "outputs/debug/attn/weights"
    if wkey_am in rec:
        _w_am = np.asarray(rec[wkey_am])
        if fmt == "pi0fast":
            def _span_mass_am(sp):
                if sp is None: return 0.0
                return float(_w_am[:, 0, :, sp[0]:sp[1]].sum(axis=-1).mean())
        else:
            def _span_mass_am(sp):
                if sp is None: return 0.0
                return float(_w_am[:, :, :, sp[0]:sp[1]].sum(axis=-1).mean())
        _sp_base_am  = _span(rec, "outputs/debug/spans/image/base_0_rgb")
        _sp_wrist_am = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
        _sp_task_am  = _span(rec, "outputs/debug/spans/task")
        _sp_state_am = _span(rec, "outputs/debug/spans/state")
        feats["attn_mass_image"] = _span_mass_am(_sp_base_am) + _span_mass_am(_sp_wrist_am)
        feats["attn_mass_task"]  = _span_mass_am(_sp_task_am)
        feats["attn_mass_state"] = _span_mass_am(_sp_state_am)
    else:
        feats["attn_mass_image"] = _nan()
        feats["attn_mass_task"]  = _nan()
        feats["attn_mass_state"] = _nan()

    # GradCAM CoM drift: base and wrist cameras
    for drift_cam, drift_label in [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist")]:
        gc_cam = maps_data.get(drift_label, (None,))[0]
        if gc_cam is not None and prev_rec is not None:
            if fmt == "pi0fast":
                prev_gc, _, _, _, _, _ = _load_heatmap_pi0fast(prev_rec, drift_cam)
            else:
                prev_gc, _, _, _, _, _ = _load_heatmap_pi05(prev_rec, drift_cam)
            if prev_gc is not None:
                cy1, cx1 = spatial_com(gc_cam)
                cy2, cx2 = spatial_com(prev_gc)
                feats[f"gc_com_drift_{drift_label}"] = float(np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2))
            else:
                feats[f"gc_com_drift_{drift_label}"] = _nan()
        else:
            feats[f"gc_com_drift_{drift_label}"] = 0.0

    return feats, centroids


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE & GROUP LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_features(data_dir: Path, ep: dict, max_steps: int) -> List[dict]:
    """Load and extract features for all steps in one episode."""
    records = []
    prev_rec = None
    prev_centroids: Optional[dict] = None
    for rel in range(max_steps):
        s = ep["start_idx"] + rel
        if s > ep["end_idx"]:
            break
        fpath = data_dir / f"step_{s}.npy"
        if not fpath.exists():
            continue
        try:
            rec = np.load(fpath, allow_pickle=True).item()
            feats, centroids = extract_features(rec, prev_rec=prev_rec,
                                                prev_centroids=prev_centroids)
            feats["rel_step"] = rel
            feats["abs_step"] = s
            records.append(feats)
            prev_rec = rec
            prev_centroids = centroids
        except Exception as e:
            print(f"    [warn] {fpath.name}: {e}")
    return records


def _atomic_save(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f, protocol=4)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_group(group_name: str,
               run_names: List[str],
               base_dir: Path,
               cache_dir: Path,
               max_steps: int = 60,
               use_cache: bool = True) -> Dict[str, List[List[dict]]]:
    """Load all episodes for a group, caching to disk.

    Returns {outcome: [[step_feats, ...], ...]}  where outcome ∈ {"success","failure"}.
    """
    cache_path = cache_dir / f"{group_name}.pkl"
    if use_cache and cache_path.exists():
        try:
            print(f"  Loading {group_name} from cache …")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [warn] cache corrupted ({e}), re-extracting …")

    ep_data: Dict[str, List[List[dict]]] = {"success": [], "failure": []}
    for run_name in run_names:
        data_dir = base_dir / run_name
        eps_path = data_dir / "client_output" / "episode_summaries.json"
        if not eps_path.exists():
            print(f"  [skip] {run_name}: no episode_summaries.json")
            continue
        with open(eps_path) as f:
            episodes = json.load(f)
        print(f"  {run_name}: {len(episodes)} episodes …")
        for ep in tqdm(episodes, desc=f"    {run_name[:40]}", ncols=80, leave=False):
            label = "success" if ep["success"] else "failure"
            recs = load_episode_features(data_dir, ep, max_steps)
            if recs:
                ep_data[label].append(recs)

    n_s = len(ep_data["success"]); n_f = len(ep_data["failure"])
    print(f"  {group_name}: {n_s} success, {n_f} failure episodes loaded.")
    _atomic_save(ep_data, cache_path)
    return ep_data


def feature_names(ep_data: Dict) -> List[str]:
    """Return sorted list of public feature names (exclude rel_step, abs_step)."""
    SKIP = {"rel_step", "abs_step"}
    for eps in ep_data.values():
        for ep in eps:
            if ep:
                return sorted(k for k in ep[0] if k not in SKIP)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# MATRIX UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def to_matrix(ep_list: List[List[dict]], key: str, max_steps: int) -> np.ndarray:
    """Build (N_episodes, max_steps) float64 matrix; NaN for missing steps."""
    out = np.full((len(ep_list), max_steps), np.nan, dtype=np.float64)
    for i, ep in enumerate(ep_list):
        for rec in ep:
            t = rec.get("rel_step", 0)
            if t < max_steps:
                v = rec.get(key)
                if v is not None:
                    try:
                        out[i, t] = float(v)
                    except (TypeError, ValueError):
                        pass
    return out


def rolling_mean_matrix(mat: np.ndarray, W: int) -> np.ndarray:
    """Causal rolling mean with window W.  Shape unchanged; leading steps use shorter window."""
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[1]):
        win = mat[:, max(0, t - W + 1): t + 1]
        out[:, t] = np.nanmean(win, axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

def _seg_slopes(seg: np.ndarray) -> np.ndarray:
    """Vectorised linear regression slope for each row of (N, W)."""
    N, W = seg.shape
    if W < 3:
        return np.full(N, np.nan)
    pos = np.arange(W, dtype=np.float64)
    mean_pos = pos.mean()
    dpos = pos - mean_pos
    denom = float(np.dot(dpos, dpos))
    if denom < 1e-10:
        return np.full(N, np.nan)

    slopes = np.full(N, np.nan)
    has_nan = np.isnan(seg).any(axis=1)

    ok = ~has_nan
    if ok.any():
        batch = seg[ok]
        dx = batch - batch.mean(axis=1, keepdims=True)
        slopes[ok] = (dx @ dpos) / denom

    for i in np.where(has_nan)[0]:
        row = seg[i]; valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        vp = pos[valid]; vx = row[valid]
        dvp = vp - vp.mean(); dvx = vx - vx.mean()
        dv = float(np.dot(dvp, dvp))
        if dv > 1e-10:
            slopes[i] = float(np.dot(dvp, dvx) / dv)

    return slopes


def window_descriptors(raw_mat: np.ndarray,
                       window_size: int = 5) -> Dict[str, np.ndarray]:
    """Compute sliding-window descriptors for a (N_episodes, T) raw feature matrix.

    Parameters
    ----------
    raw_mat     : (N, T) float64 array, NaN for missing steps
    window_size : causal window width W.  Use 5 for fast/online or 15 for slow/smooth.

    Returns dict of (N, T) arrays:
      win_mean      causal mean over [t-W+1, t]
      win_std       causal std  over [t-W+1, t]
      win_slope     causal linear slope over [t-W+1, t]
      d_mean        level shift: win_mean[t] - win_mean[t-W]
      d_slope       slope change: win_slope[t] - win_slope[t-W]
      d_slope_mag   |d_slope|
      cen_slope     centered slope over [t-H, t+H]  where H = (W-1)//2; NaN near edges
      cen_slope_mag |cen_slope|
    """
    N, T = raw_mat.shape
    W = window_size
    H = (W - 1) // 2   # centered half-width

    out: Dict[str, np.ndarray] = {
        k: np.full((N, T), np.nan, dtype=np.float64)
        for k in ("win_mean", "win_std", "win_slope",
                  "d_mean", "d_slope", "d_slope_mag",
                  "cen_slope", "cen_slope_mag")
    }

    for t in range(T):
        lo = max(0, t - W + 1)
        cur = raw_mat[:, lo: t + 1]      # (N, w)
        nv  = (~np.isnan(cur)).sum(axis=1)

        with np.errstate(all="ignore"):
            out["win_mean"][:, t] = np.where(nv >= 1, np.nanmean(cur, axis=1), np.nan)
            out["win_std"][:, t]  = np.where(nv >= 2, np.nanstd(cur, axis=1), np.nan)

        if cur.shape[1] >= 3:
            out["win_slope"][:, t] = _seg_slopes(cur)

        # Previous window for delta features
        prev_lo = max(0, lo - W)
        prev_hi = lo
        if prev_hi > prev_lo:
            prev = raw_mat[:, prev_lo:prev_hi]
            with np.errstate(all="ignore"):
                prev_mean = np.nanmean(prev, axis=1)
            out["d_mean"][:, t] = out["win_mean"][:, t] - prev_mean
            if prev.shape[1] >= 3:
                ps = _seg_slopes(prev)
                ds = out["win_slope"][:, t] - ps
                out["d_slope"][:, t]     = ds
                out["d_slope_mag"][:, t] = np.abs(ds)

        # Centered window
        if t >= H and t + H < T:
            cen = raw_mat[:, t - H: t + H + 1]
            if cen.shape[1] >= 3:
                cs = _seg_slopes(cen)
                out["cen_slope"][:, t]     = cs
                out["cen_slope_mag"][:, t] = np.abs(cs)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# AUROC / STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def auroc_at(sv: np.ndarray, fv: np.ndarray) -> float:
    """Mann-Whitney AUROC from success and failure value arrays.  NaN if insufficient data."""
    sv = sv[~np.isnan(sv)]; fv = fv[~np.isnan(fv)]
    if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
        return _nan()
    try:
        stat, _ = mannwhitneyu(fv, sv, alternative="two-sided")
        a = float(stat) / (len(sv) * len(fv))
        return max(a, 1.0 - a)
    except Exception:
        return _nan()


def effective_auroc(a: float) -> float:
    """Map AUROC to [0.5, 1] regardless of direction (distance from 0.5)."""
    return float(abs(a - 0.5) + 0.5) if not np.isnan(a) else _nan()


def compute_auroc_sequence(ep_data: Dict, feature: str, max_steps: int) -> np.ndarray:
    """AUROC at each step, using only episodes active at that step."""
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    return np.array([auroc_at(S[:, t], F[:, t]) for t in range(max_steps)])


def compute_pvalue_sequence(ep_data: Dict, feature: str, max_steps: int) -> np.ndarray:
    """Mann-Whitney two-sided p-value at each step."""
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    pvals = np.ones(max_steps)
    for t in range(max_steps):
        sv = S[:, t][~np.isnan(S[:, t])]
        fv = F[:, t][~np.isnan(F[:, t])]
        if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
            continue
        try:
            _, p = mannwhitneyu(sv, fv, alternative="two-sided")
            pvals[t] = float(p)
        except Exception:
            pass
    return pvals


def compute_cohens_d_sequence(ep_data: Dict, feature: str, max_steps: int) -> np.ndarray:
    """Cohen's d (|µS - µF| / pooled_std) at each step."""
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    ds = np.full(max_steps, np.nan)
    for t in range(max_steps):
        sv = S[:, t][~np.isnan(S[:, t])]
        fv = F[:, t][~np.isnan(F[:, t])]
        if len(sv) < 3 or len(fv) < 3:
            continue
        pooled = np.sqrt((np.var(sv, ddof=1) + np.var(fv, ddof=1)) / 2.0 + 1e-9)
        ds[t] = abs(float(sv.mean()) - float(fv.mean())) / pooled
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# PHASE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptors(raw: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
    """(N, T) descriptor matrices for a raw feature matrix.  See window_descriptors()."""
    return window_descriptors(raw, window_size=window_size)


def phase_aggregates(raw: np.ndarray,
                     phase_defs: Dict[str, Tuple[int, int]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-episode phase-level aggregates.

    Returns {phase_name: {"mean": (N,), "max": (N,)}}.
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for ph, (a, b) in phase_defs.items():
        seg = raw[:, a: b + 1]
        _max = np.where(np.isnan(seg), -np.inf, seg)
        ep_max = _max.max(axis=1)
        ep_max = np.where(np.isinf(ep_max), np.nan, ep_max)
        result[ph] = {
            "mean": np.nanmean(seg, axis=1),
            "max":  ep_max,
        }
    return result


def _get_raw(ep_list: List[List[dict]], feat: str, max_steps: int) -> np.ndarray:
    out = np.full((len(ep_list), max_steps), np.nan, dtype=np.float64)
    for i, ep in enumerate(ep_list):
        for t, step in enumerate(ep[:max_steps]):
            v = step.get(feat)
            if v is not None:
                try:
                    out[i, t] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def _dir_str(a: float) -> str:
    return "F>S" if (not np.isnan(a) and a > 0.5) else "F<S"


class FeaturePhaseResult:
    """All phase-aware statistics for one feature in one group."""
    __slots__ = ("phase_mean_auroc", "phase_max_auroc", "desc_phase_peak",
                 "delta_aurocs", "direction_change", "early_dir", "late_dir")

    def __init__(self):
        self.phase_mean_auroc: Dict[str, float] = {}
        self.phase_max_auroc:  Dict[str, float] = {}
        self.desc_phase_peak:  Dict[str, Dict[str, float]] = {}  # desc → {phase: auroc}
        self.delta_aurocs:     Dict[str, float] = {}
        self.direction_change: bool = False
        self.early_dir:        str  = "?"
        self.late_dir:         str  = "?"


def analyse_feature(fail_eps: List[List[dict]],
                    succ_eps: List[List[dict]],
                    feat: str,
                    phase_defs: Dict[str, Tuple[int, int]] = DEFAULT_PHASE_DEFS,
                    window_sizes: Tuple[int, ...] = DEFAULT_WINDOW_SIZES,
                    max_steps: int = 60) -> FeaturePhaseResult:
    """Compute all phase-aware statistics for a single feature.

    phase_defs   : e.g. PHASES_3 or PHASES_4
    window_sizes : window widths to compute descriptors for (e.g. (5, 15))
    """
    fail_raw = _get_raw(fail_eps, feat, max_steps)
    succ_raw = _get_raw(succ_eps, feat, max_steps)
    phase_names = list(phase_defs.keys())

    agg_f = phase_aggregates(fail_raw, phase_defs)
    agg_s = phase_aggregates(succ_raw, phase_defs)

    res = FeaturePhaseResult()
    for ph in phase_names:
        res.phase_mean_auroc[ph] = auroc_at(agg_s[ph]["mean"], agg_f[ph]["mean"])
        res.phase_max_auroc[ph]  = auroc_at(agg_s[ph]["max"],  agg_f[ph]["max"])

    # Per-step descriptors at each window size
    res.desc_phase_peak = {}
    for W in window_sizes:
        fail_desc = window_descriptors(fail_raw, W)
        succ_desc = window_descriptors(succ_raw, W)
        for desc_name, fail_d in fail_desc.items():
            succ_d = succ_desc[desc_name]
            key = f"{desc_name}_w{W}"
            res.desc_phase_peak[key] = {}
            for ph, (a, b) in phase_defs.items():
                best = _nan()
                for t in range(a, min(b + 1, max_steps)):
                    a_val = auroc_at(succ_d[:, t], fail_d[:, t])
                    if not np.isnan(a_val) and (np.isnan(best) or
                                                effective_auroc(a_val) > effective_auroc(best)):
                        best = a_val
                res.desc_phase_peak[key][ph] = best

    # Cross-phase deltas
    res.delta_aurocs = {}
    phase_pairs = []
    pn = phase_names
    for i in range(len(pn)):
        for j in range(i + 1, len(pn)):
            phase_pairs.append((f"{pn[j]}_minus_{pn[i]}", pn[j], pn[i]))
    for key, ph1, ph2 in phase_pairs:
        fd = agg_f[ph1]["mean"] - agg_f[ph2]["mean"]
        sd = agg_s[ph1]["mean"] - agg_s[ph2]["mean"]
        res.delta_aurocs[key] = auroc_at(sd, fd)

    # Direction change (only meaningful for 3+ phases with early/late defined)
    e_ph = phase_names[0]
    l_ph = phase_names[-1]
    e_a = res.phase_mean_auroc.get(e_ph, 0.5)
    l_a = res.phase_mean_auroc.get(l_ph, 0.5)
    res.early_dir = _dir_str(e_a)
    res.late_dir  = _dir_str(l_a)
    res.direction_change = (
        not np.isnan(e_a) and not np.isnan(l_a) and
        (e_a - 0.5) * (l_a - 0.5) < 0
    )
    return res


def analyse_all_features(ep_data: Dict,
                          features: List[str],
                          phase_defs: Dict[str, Tuple[int, int]] = DEFAULT_PHASE_DEFS,
                          window_sizes: Tuple[int, ...] = DEFAULT_WINDOW_SIZES,
                          max_steps: int = 60) -> Dict[str, FeaturePhaseResult]:
    """Compute FeaturePhaseResult for every feature in a group."""
    succ_eps = ep_data.get("success", [])
    fail_eps = ep_data.get("failure", [])
    results: Dict[str, FeaturePhaseResult] = {}
    for i, feat in enumerate(features):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(features)} features done")
        results[feat] = analyse_feature(fail_eps, succ_eps, feat,
                                        phase_defs=phase_defs,
                                        window_sizes=window_sizes,
                                        max_steps=max_steps)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_cross_domain_candidates(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    min_auroc: float = 0.65,
    min_groups: int = 2,
    dir_thresh: float = 0.60,
    phase_defs: Dict[str, Tuple[int, int]] = DEFAULT_PHASE_DEFS,
) -> List[Tuple[str, str, int, float]]:
    """Rule A: features discriminative in ≥ min_groups groups in some phase.

    Returns [(feature, best_phase, direction, mean_eff_auroc)] sorted by descending AUROC.
    """
    candidates = []
    phase_names = list(phase_defs.keys())

    for feat in features:
        for ph in phase_names:
            group_aurocs = []
            for feat_results in all_group_results.values():
                res = feat_results.get(feat)
                if res is None:
                    continue
                a = res.phase_mean_auroc.get(ph, _nan())
                if np.isnan(a):
                    best = 0.5
                    for dp in res.desc_phase_peak.values():
                        v = dp.get(ph, _nan())
                        if not np.isnan(v) and effective_auroc(v) > effective_auroc(best):
                            best = v
                    a = best
                group_aurocs.append(a)

            valid = [a for a in group_aurocs if not np.isnan(a)]
            if len(valid) < min_groups:
                continue
            n_discrim = sum(1 for a in valid if effective_auroc(a) >= min_auroc)
            if n_discrim < min_groups:
                continue

            dirs = [_dir_str(a) for a in valid if effective_auroc(a) >= dir_thresh]
            n_fgs = dirs.count("F>S"); n_fls = dirs.count("F<S")
            if max(n_fgs, n_fls) < min_groups:
                continue
            direction = +1 if n_fgs >= n_fls else -1
            mean_eff = float(np.mean([effective_auroc(a) for a in valid]))
            candidates.append((feat, ph, direction, mean_eff))

    seen: Dict[str, Tuple] = {}
    for c in candidates:
        feat = c[0]
        if feat not in seen or c[3] > seen[feat][3]:
            seen[feat] = c
    return sorted(seen.values(), key=lambda x: -x[3])


def select_group_specific_candidates(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    group: str,
    min_auroc: float = 0.70,
    top_n: int = 10,
    phase_defs: Dict[str, Tuple[int, int]] = DEFAULT_PHASE_DEFS,
) -> List[Tuple[str, str, int, float]]:
    """Rule B: top top_n features per group per phase with AUROC ≥ min_auroc.

    Returns [(feature, best_phase, direction, eff_auroc)].
    """
    res_map = all_group_results.get(group, {})
    phase_names = list(phase_defs.keys())
    per_phase: Dict[str, List] = {ph: [] for ph in phase_names}

    for feat in features:
        res = res_map.get(feat)
        if res is None:
            continue
        for ph in phase_names:
            a = res.phase_mean_auroc.get(ph, _nan())
            eff = effective_auroc(a)
            if eff >= min_auroc:
                per_phase[ph].append((feat, ph, _dir_str(a), eff))

    candidates = []
    for ph, entries in per_phase.items():
        entries.sort(key=lambda x: -x[3])
        for feat, phase, d_s, eff in entries[:top_n]:
            candidates.append((feat, phase, +1 if d_s == "F>S" else -1, eff))

    seen: Dict[str, Tuple] = {}
    for c in candidates:
        feat = c[0]
        if feat not in seen or c[3] > seen[feat][3]:
            seen[feat] = c
    return sorted(seen.values(), key=lambda x: -x[3])


def select_direction_change_features(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
) -> List[str]:
    """Rule C: features that change AUROC direction early→late in ≥ 1 group."""
    result = []
    for feat in features:
        for feat_results in all_group_results.values():
            res = feat_results.get(feat)
            if res and res.direction_change:
                result.append(feat)
                break
    return result
