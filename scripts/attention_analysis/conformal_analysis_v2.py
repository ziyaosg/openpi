#!/usr/bin/env python3
"""
conformal_analysis_v2.py — Comprehensive conformal early failure detection.

New features over v1:
  A. Per-action-token (pi0fast): token-0 dedicated features; token 0 vs 4 horizon
     agreement; std of Gini across 5 tokens (plan instability)
  B. Value vector features: per-modality L2-norm (mean, Gini), within-modality
     centroid coherence, cross-modality cosine alignment, temporal centroid drift
  C. Cross-camera agreement within same signal type (base↔wrist↔rwrist)
  D. Cross-signal agreement within camera (attn-weights ↔ gradcam ↔ ras ↔ rn)
  E. Right-wrist camera (was skipped in v1)
  F. Task modality: rn_gini_task + all cross-signal pairs
  G. Enhanced action features: plan spread/entropy, horizon alignment
  H. Integrated sliding-window AUROC (windows 1/3/5/10/20)
  I. Mahalanobis composite score and detection-time CDF

Output: /nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wasserstein_distance
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE       = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
OUTPUT_DIR = BASE / "analysis_conformal_v2"
CACHE_DIR  = OUTPUT_DIR / "feature_cache"
MAX_STEPS  = 60
FPR_TARGET = 0.05
AUROC_MIN_N = 5

RUN_GROUPS: Dict[str, List[str]] = {
    "pi05_libero10": [
        "policy_records_pi05_libero_20260511_021509",
        "policy_records_pi05_libero_20260512_182606",
        "policy_records_pi05_libero_20260512_201032",
    ],
    "pi0fast_libero10": [
        "policy_records_pi0_fast_libero_20260511_040629",
        "policy_records_pi0_fast_libero_20260513_021106",
    ],
    "pi0fast_liberoplus": [
        "policy_records_pi0_fast_libero_20260516_034939",
        "policy_records_pi0_fast_libero_20260516_034940",
    ],
}

# Features computed from right_wrist_0_rgb are NOISE for single-arm libero tasks.
# The model receives no right_wrist_image observation; the slot is filled with a
# repeated copy of wrist_image (gc cosine ≈ 0.905 between left and right wrist).
# Exclude these from any conformal prediction pipeline.
NOISE_FEATURES: frozenset = frozenset({
    "frac_rwrist", "attn_gini_rwrist",
    "gc_gini_rwrist", "gc_entropy_rwrist", "ras_gini_rwrist", "ras_entropy_rwrist",
    "rn_gini_rwrist", "rn_entropy_rwrist", "gc_max_rwrist", "gc_mean_rwrist",
    "gc_com_y_rwrist", "gc_com_x_rwrist",
    "agree_gc_ras_rwrist", "agree_gc_rn_rwrist", "agree_ras_rn_rwrist",
    "agree_attn_gc_rwrist", "agree_attn_ras_rwrist",
    "agree_base_rwrist_gc", "agree_base_rwrist_ras", "agree_base_rwrist_rn",
    "agree_wrist_rwrist_gc", "agree_wrist_rwrist_ras", "agree_wrist_rwrist_rn",
    "gc_gini_rwrist_t0", "ras_gini_rwrist_t0", "rn_gini_rwrist_t0",
    "gc_gini_rwrist_std", "horizon_agree_gc_rwrist", "horizon_agree_ras_rwrist",
})

# ─── Cross-format compatibility sets ───────────────────────────────────────────
# Goal: train conformal predictor on pi0fast/libero10, test on:
#   (a) pi0fast/liberoplus  — same model, different task distribution
#   (b) pi05/libero10       — different model, same tasks (current recordings)
#
# Pi05 state note: outputs/state IS recorded as (32,) with last 24 dims = 0,
# so effectively the same 8-dim joint state as pi0fast. state_norm and joint_vel
# ARE computable. The current pi05 recordings do NOT have outputs/debug/spans/state
# (state is not tokenized in the transformer for these runs); new pi05 runs
# with the fixed branch will have it.
#
# State score indexing note: the score arrays (gradcam, raw_alpha, raw_weights,
# v_cosine) for state cover ALL tokens in "State: {digits};\n", including the
# syntactic prefix ("State", ":") and suffix (";", "\n") tokens.  These tokens
# get high attention from the model as structural markers but carry no per-joint
# information.  _state_digit_slice() returns (k, end) to slice them out before
# computing Gini/cosine-agreement.
#
# For cross-format transfer (pi0fast→pi05), distribution mismatches:
#   - joint_vel: same 8-dim computation, but pi05 (91% success) makes 10× smaller
#     per-step movements than pi0fast (57% success). The signal direction is the
#     same (failures are faster), but absolute values don't overlap — calibration
#     thresholds from pi0fast would never fire on pi05.
#   - frac_task: 200 task tokens (pi05) vs 17 (pi0fast) → 0.50 vs 0.68 mean
#   - gc_gini_task: opposite regime — 0.77 pi05 vs 0.53 pi0fast
#   - v_drift_task: 3× larger for pi05 (longer task token centroid)
#   - state-modality features: NaN for these pi05 runs (no state span)
#   - per-token features: NaN for pi05 (no per-token GradCAM structure)
#
# CROSS_FORMAT_SAFE: use in the universal (all-group) conformal predictor.
# These features have overlapping value ranges across pi05 and pi0fast.
CROSS_FORMAT_SAFE: tuple = (
    # camera heatmap concentration — identical (16,16) format for both models
    "gc_gini_base",  "ras_gini_base",  "rn_gini_base",
    "gc_gini_wrist", "ras_gini_wrist", "rn_gini_wrist",
    "gc_entropy_base", "ras_entropy_base",
    # cross-signal agreement — gc and ras exist for base+wrist in both models
    "agree_gc_ras_base",  "agree_gc_rn_base",
    "agree_gc_ras_wrist",
    "agree_base_wrist_gc", "agree_base_wrist_rn",
    # value vectors base+wrist — both models have V∈R^{256} for 256 base/wrist tokens
    "v_align_base_wrist",  # nearly identical range across all groups — best transfer feature
    "v_drift_wrist",       # near-identical range (0.016 pi0fast vs 0.017 pi05)
    "v_coherence_base", "v_coherence_wrist",
    "v_norm_gini_base", "v_norm_gini_wrist",
    # action features (direction/entropy/alignment; scale-invariant ratios)
    "action_plan_entropy", "action_horizon_align",
    "action_rot_flip",     "plan_consistency",
    "action_spread",       "action_dir_var",
)

# CROSS_FORMAT_CAUTION: computable for both and directionally valid, but with
# distribution shift. Use with per-group z-score normalization.
CROSS_FORMAT_CAUTION: tuple = (
    # state_norm: same 8-dim computation but pi05 tighter (std 0.21 vs 0.92 pi0fast)
    "state_norm",
    # joint_vel: same 8-dim ||Δstate||₂ but pi05 makes 10× smaller per-step moves
    # (higher-quality policy). Works within each group, calibration doesn't transfer.
    "joint_vel",
    "v_drift_base",       # pi05 2× lower absolute values
    "action_magnitude",   "action_plan_norm",
    "attn_gini_base",     "attn_entropy_base",   # pi05 uses mean over 10 action tokens
    "frac_base",          # 3× smaller for pi0fast after rwrist fix; model behavior diff
    "gc_com_drift_base",
    # After token-mask fix, these are now in a compatible range:
    "frac_task",          # 0.677 pi0fast vs 0.739 pi05 after fix
    "gc_gini_task",       # 0.457 pi0fast vs 0.702 pi05 after fix (real tokens only)
    "ras_gini_task",      "rn_gini_task",
    "v_coherence_task",   # 0.766 pi0fast vs 0.645 pi05 after fix (real tokens only)
    "v_drift_task",       # temporal centroid change; scale-agnostic ratio
)

# FORMAT_SPECIFIC: do NOT use in the cross-format conformal predictor.
FORMAT_SPECIFIC: tuple = (
    # frac_wrist: pi05=0.24 vs pi0fast=0.054 — genuine 4.5× model behavior difference.
    # Pi05 relies heavily on wrist camera (pi0fast relies more on task tokens).
    "frac_wrist",
    # routing_entropy: depends on frac_wrist and frac_state which both differ
    "routing_entropy",
    # attn_gini_task: pi05 uses 1-KV-head GQA, different concentration profile
    "attn_gini_task",
    # agree_gc_ras_task: 0.622 pi05 vs 0.897 pi0fast — model attribution patterns differ fundamentally
    "agree_gc_ras_task",  "agree_gc_rn_task", "agree_attn_gc_task",
    "v_align_base_task",  "v_align_task_wrist",
    "v_norm_gini_task",   "v_norm_mean_task",
    # state-modality: NaN for these pi05 runs (state not tokenized in transformer).
    # NOTE: new pi05 runs with the fixed branch WILL have these — re-evaluate then.
    "frac_state", "attn_gini_state",
    "agree_gc_ras_state", "agree_attn_gc_state",
    # per-token: NaN for pi05 (single GradCAM heatmap per camera, not per action token)
    "gc_gini_base_t0", "ras_gini_base_t0", "rn_gini_base_t0",
    "gc_gini_wrist_t0", "ras_gini_wrist_t0", "rn_gini_wrist_t0",
    "gc_com_y_base_t0", "gc_com_x_base_t0",
    "horizon_agree_gc_base",  "horizon_agree_ras_base",
    "horizon_agree_gc_wrist", "horizon_agree_ras_wrist",
    "gc_gini_base_std", "ras_gini_base_std",
    "gc_gini_wrist_std", "ras_gini_wrist_std",
)

BLUE_M, BLUE_L = "#1f5fa6", "#a8c8e8"
RED_M,  RED_L  = "#b02020", "#f0a0a0"
LW_M, LW_I, AL = 2.2, 0.8, 0.35

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "legend.fontsize": 8, "legend.framealpha": 0.7,
})

# ─── Scalar utilities ───────────────────────────────────────────────────────────

def gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s < 1e-12 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def entropy(p: np.ndarray) -> float:
    p = np.abs(p.ravel()).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def spatial_com(h: np.ndarray) -> Tuple[float, float]:
    h = np.abs(h).astype(np.float64); s = h.sum()
    if s < 1e-12:
        return 0.0, 0.0
    rows = np.arange(h.shape[0])[:, None]
    cols = np.arange(h.shape[1])[None, :]
    return float((h * rows).sum() / s), float((h * cols).sum() / s)


def _nan() -> float:
    return float("nan")


# Camera heatmap spatial dimensions (same for pi0fast and pi05)
PATCH_H = PATCH_W = 16
_RW_TOP_K = 8      # span-focused heads for raw-weights-max computation
_IOU_FRAC = 0.10   # top fraction of positions for IoU


def _select_rw(attn_per_head: np.ndarray, sp: Tuple[int, int],
               top_k: int = _RW_TOP_K) -> np.ndarray:
    """
    Return max attention over the top-K span-focused heads for the given span.
    attn_per_head: (H, S)   per-head raw attention weights (last layer)
    Returns: (sp[1]-sp[0],) vector
    """
    k = min(top_k, attn_per_head.shape[0])
    span_mass = attn_per_head[:, sp[0]:sp[1]].sum(axis=1)  # (H,)
    top_heads = np.argsort(span_mass)[-k:]
    return attn_per_head[top_heads, sp[0]:sp[1]].max(axis=0)


def _topk_iou(a: np.ndarray, b: np.ndarray, frac: float = _IOU_FRAC) -> float:
    """Top-fraction overlap IoU between two attribution maps."""
    a, b = np.abs(a.ravel()), np.abs(b.ravel())
    k = max(1, int(round(len(a) * frac)))
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    union = top_a | top_b
    return float(len(top_a & top_b) / len(union)) if union else 0.0


def _com_dist_2d(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between 2-D spatial COMs; auto-reshapes flat vectors to PATCH_H×PATCH_W."""
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


def _emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein distance over token positions. Lower = more similar."""
    a = np.abs(a.ravel()).astype(np.float64)
    b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    pos = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(pos, pos, a / sa, b / sb))


def _emd_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of 1-D Wasserstein distances on x- and y-axis marginals (patch-grid units).
    Auto-reshapes flat vectors to PATCH_H×PATCH_W. Lower = more similar."""
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


def _vc_scores(attn_hxs: np.ndarray, v_seq: np.ndarray,
               sp: Tuple[int, int], top_k: int = _RW_TOP_K) -> np.ndarray:
    """V-Cosine score per token in span sp.

    For each token s: score = max over top_k heads of cosine(v[s], head_output_h).
    head_output_h = sum_s'(attn[h,s'] * v[s',0,:]).

    Inlined from grid_visualization_value_attn.score_v_cosine to avoid import
    dependency. Returns (sp[1]-sp[0],) array of float32.
    """
    k = min(top_k, attn_hxs.shape[0])
    span_mass = attn_hxs[:, sp[0]:sp[1]].sum(axis=1)   # (H,) budget per head
    top_heads  = np.argsort(span_mass)[-k:]              # top-K heads by budget
    v0      = v_seq[:, 0, :]                                                   # (S, D)
    o       = attn_hxs @ v0                                                    # (H, D)
    o_unit  = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)          # (H, D)
    v_unit  = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-9)        # (S, D)
    cos_all = v_unit @ o_unit.T                                                # (S, H)
    scores  = cos_all[:, top_heads].max(axis=1).astype(np.float32)            # (S,)
    return scores[sp[0]:sp[1]]


# ─── Format helpers ─────────────────────────────────────────────────────────────

def detect_format(rec: dict) -> str:
    return "pi0fast" if "outputs/debug/attn/layers" in rec else "pi05"


def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    v = rec.get(key)
    if v is None:
        return None
    v = np.asarray(v).ravel()
    return int(v[0]), int(v[1])


def _state_digit_slice(rec: dict) -> Tuple[int, Optional[int]]:
    """Return (k, end) such that score_array[k:end] covers only digit tokens.

    The state score arrays (gradcam, raw_alpha, raw_weights, v_cosine) span ALL
    tokens in the rendered "State: {digits};\\n" segment, including syntactic
    prefix tokens ("State", ":") and suffix tokens (";", "\\n").  These tokens
    score inconsistently with the actual state-digit information and distort
    Gini / cosine-agreement metrics.  This helper reads piece_begin/end to find
    the boundary:
      content_start = len("State: ") = 7   (first digit byte)
      content_end   = max_byte - len(";\\n") = max_byte - 2  (last digit byte excl.)
    Returns k (number of prefix-only pieces) and end (first suffix-only index) so
    score_array[k:end] is the digit-only slice.  Returns (0, None) if piece
    metadata is absent (use the full array as-is).
    """
    pb_key = "outputs/debug/tokens/state/piece_begin"
    pe_key = "outputs/debug/tokens/state/piece_end"
    if pb_key not in rec or pe_key not in rec:
        return 0, None
    piece_begin = np.asarray(rec[pb_key])
    piece_end   = np.asarray(rec[pe_key])
    max_byte      = int(piece_end.max())
    CONTENT_START = 7            # len("State: ")
    CONTENT_END   = max_byte - 2 # excl. ";\n"
    k   = int((piece_end   <= CONTENT_START).sum())   # prefix-only pieces
    s   = int((piece_begin >= CONTENT_END).sum())     # suffix-only pieces
    return k, len(piece_begin) - s


# ─── Sub-extractors ─────────────────────────────────────────────────────────────

def _extract_attn_routing(rec: dict, fmt: str) -> dict:
    """Modality attention fractions and within-modality concentration."""
    feats: dict = {}
    wkey = "outputs/debug/attn/weights"
    if wkey not in rec:
        for k in ("frac_base","frac_wrist","frac_rwrist","frac_task","frac_state",
                  "routing_entropy","attn_gini_base","attn_entropy_base",
                  "attn_gini_wrist","attn_gini_rwrist","attn_gini_task","attn_gini_state"):
            feats[k] = _nan()
        return feats

    w = np.asarray(rec[wkey])
    if fmt == "pi0fast":
        # Shape (L, 1, H, S): attention from the first decoded action token across H heads.
        mean_attn     = w[-1, 0].mean(axis=0)   # (S,)
        attn_per_head = w[-1, 0]                 # (H, S)
    else:
        # Shape (L, H, T_action, S): cross-attention from T_action=10 action tokens across H heads.
        mean_attn     = w[-1].mean(axis=0).mean(axis=0)  # (S,)  mean over H then T_action
        attn_per_head = w[-1].mean(axis=1)               # (H, S) mean over T_action, keep per-head

    sp_base   = _span(rec, "outputs/debug/spans/image/base_0_rgb")
    sp_wrist  = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
    sp_rwrist = _span(rec, "outputs/debug/spans/image/right_wrist_0_rgb")
    sp_task   = _span(rec, "outputs/debug/spans/task")
    sp_state  = _span(rec, "outputs/debug/spans/state")

    # For pi0fast the right_wrist slot receives a zero image but image_mask=True, so
    # the model accidentally attends to it (~14% of budget, pure noise).  Exclude it
    # from the denominator so fracs are comparable to pi05 (where image_mask=False
    # makes right_wrist correctly receive 0 attention and is already excluded).
    if fmt == "pi0fast" and sp_rwrist is not None:
        rwrist_mass = float(mean_attn[sp_rwrist[0]:sp_rwrist[1]].sum())
        total = mean_attn.sum() - rwrist_mass + 1e-12
    else:
        total = mean_attn.sum() + 1e-12

    def frac(sp):
        return float(mean_attn[sp[0]:sp[1]].sum() / total) if sp else _nan()

    feats["frac_base"]   = frac(sp_base)
    feats["frac_wrist"]  = frac(sp_wrist)
    feats["frac_rwrist"] = frac(sp_rwrist)   # will be >1 for pi0fast now, flagging it as noise
    feats["frac_task"]   = frac(sp_task)
    feats["frac_state"]  = frac(sp_state)

    fracs = [v for v in [feats["frac_base"], feats["frac_wrist"],
                          feats["frac_task"], feats["frac_state"]] if not np.isnan(v)]
    if fracs:
        p = np.array(fracs)
        p = np.clip(p, 0, None)  # fracs can technically exceed 1 before renorm; clip for entropy
        p = p / (p.sum() + 1e-12)
        feats["routing_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        feats["routing_entropy"] = _nan()

    # For pi05, truncate the task span to real tokens before computing Gini to avoid
    # sparse-vector inflation (same fix as gc/ras/rn task vectors below).
    sp_task_gini = sp_task
    if sp_task and fmt == "pi05":
        _tmk = "outputs/debug/tokens/task/token_mask"
        if _tmk in rec:
            _tm = np.asarray(rec[_tmk]).astype(bool)
            _nr = int(_tm[:sp_task[1] - sp_task[0]].sum())
            if _nr > 0:
                sp_task_gini = (sp_task[0], sp_task[0] + _nr)

    for tag, sp in [("base", sp_base), ("wrist", sp_wrist),
                    ("rwrist", sp_rwrist), ("task", sp_task_gini), ("state", sp_state)]:
        if sp:
            seg = mean_attn[sp[0]:sp[1]]
            feats[f"attn_gini_{tag}"] = gini(seg)
            if tag == "base":
                feats["attn_entropy_base"] = entropy(seg)
        else:
            feats[f"attn_gini_{tag}"] = _nan()
            if tag == "base":
                feats["attn_entropy_base"] = _nan()

    # Store raw attn segments for cross-signal agreement (returned in _extras).
    # For pi05 task: truncate to real tokens only (same as the Gini/cosine fix above).
    feats["_attn_base"]   = mean_attn[sp_base[0]:sp_base[1]]   if sp_base   else None
    feats["_attn_wrist"]  = mean_attn[sp_wrist[0]:sp_wrist[1]] if sp_wrist  else None
    feats["_attn_rwrist"] = mean_attn[sp_rwrist[0]:sp_rwrist[1]] if sp_rwrist else None
    if sp_task:
        if fmt == "pi05":
            task_mask_key = "outputs/debug/tokens/task/token_mask"
            if task_mask_key in rec:
                task_mask = np.asarray(rec[task_mask_key]).astype(bool)
                span_len  = sp_task[1] - sp_task[0]
                n_real    = int(task_mask[:span_len].sum())
                sp_task_real = (sp_task[0], sp_task[0] + n_real) if n_real > 0 else sp_task
            else:
                sp_task_real = sp_task
            feats["_attn_task"] = mean_attn[sp_task_real[0]:sp_task_real[1]]
        else:
            sp_task_real = sp_task
            feats["_attn_task"] = mean_attn[sp_task[0]:sp_task[1]]
    else:
        sp_task_real = None
        feats["_attn_task"] = None

    # Raw-weights-max per modality: max attention over top-K span-focused heads.
    feats["_rw_base"]   = _select_rw(attn_per_head, sp_base)      if sp_base      else None
    feats["_rw_wrist"]  = _select_rw(attn_per_head, sp_wrist)     if sp_wrist     else None
    feats["_rw_rwrist"] = _select_rw(attn_per_head, sp_rwrist)    if sp_rwrist    else None
    feats["_rw_task"]   = _select_rw(attn_per_head, sp_task_real) if sp_task_real else None
    feats["_rw_state"]  = _select_rw(attn_per_head, sp_state)     if sp_state     else None

    # V-Cosine scores per modality.
    # v_seq shape: (S, 1, D).  pi0fast stores (L, 1, S, 1, D); pi05 stores (L, S, 1, D).
    # We use the last stored layer (same as _extract_value_features) for consistency.
    vkey = "outputs/debug/attn/v"
    v_seq: Optional[np.ndarray] = None
    if vkey in rec:
        v_raw = np.asarray(rec[vkey])
        v_seq = v_raw[-1, 0, :, :, :] if fmt == "pi0fast" else v_raw[-1, :, :, :]
        # v_seq is now (S, 1, D) for both formats
    for sp_vc, tag_vc in [(sp_base, "base"), (sp_wrist, "wrist"),
                           (sp_rwrist, "rwrist"), (sp_task_real, "task"), (sp_state, "state")]:
        if sp_vc is not None and v_seq is not None:
            feats[f"_vc_{tag_vc}"] = _vc_scores(attn_per_head, v_seq, sp_vc)
        else:
            feats[f"_vc_{tag_vc}"] = None

    return feats


def _load_heatmap_pi0fast(rec: dict, cam: str) -> Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         List[np.ndarray],
                                                         List[np.ndarray],
                                                         List[np.ndarray]]:
    """Return (gc_mean, ras_mean, rn_mean, gc_per_tok, ras_per_tok, rn_per_tok)."""
    gcs, ras_l, rns = [], [], []
    for i in range(5):
        k_gc  = f"outputs/debug/gradcam/action_tokens/{i}/{cam}"
        k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{cam}"
        k_rn  = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/{cam}"
        if k_gc not in rec:
            break
        gcs.append(np.abs(np.asarray(rec[k_gc])[0]).astype(np.float32))
        ras_l.append(np.abs(np.asarray(rec[k_ras])[0]).astype(np.float32))
        rns.append(np.abs(np.asarray(rec[k_rn])[0]).astype(np.float32))
    if not gcs:
        return None, None, None, [], [], []
    gc_m  = np.mean(np.stack(gcs), axis=0)
    ras_m = np.mean(np.stack(ras_l), axis=0)
    rn_m  = np.mean(np.stack(rns), axis=0)
    return gc_m, ras_m, rn_m, gcs, ras_l, rns


def _load_heatmap_pi05(rec: dict, cam: str) -> Tuple[Optional[np.ndarray],
                                                       Optional[np.ndarray],
                                                       Optional[np.ndarray]]:
    k_gc  = f"outputs/debug/gradcam/image/{cam}"
    k_ras = f"outputs/debug/raw_alpha/summation/image/{cam}"
    k_rn  = f"outputs/debug/raw_alpha/norm/image/{cam}"
    gc  = np.abs(np.asarray(rec[k_gc]))  .astype(np.float32) if k_gc  in rec else None
    ras = np.abs(np.asarray(rec[k_ras])) .astype(np.float32) if k_ras in rec else None
    rn  = np.abs(np.asarray(rec[k_rn]))  .astype(np.float32) if k_rn  in rec else None
    return gc, ras, rn


def _heatmap_scalars(gc, ras, rn, attn_seg, label: str,
                     rw: Optional[np.ndarray] = None,
                     vc: Optional[np.ndarray] = None) -> dict:
    """Compute all per-camera scalar features given heatmaps, attn segment, rw, and vc maps."""
    f: dict = {}
    nan_keys = (
        "gc_gini","gc_entropy","ras_gini","ras_entropy","rn_gini","rn_entropy",
        "gc_max","gc_mean","gc_com_y","gc_com_x",
        "agree_gc_ras","agree_gc_rn","agree_ras_rn",
        "agree_attn_gc","agree_attn_ras",
        "rw_gini","agree_gc_rw","agree_ras_rw","agree_rn_rw",
        "iou_gc_ras","iou_gc_rn","iou_ras_rn",
        "com_dist_gc_ras","com_dist_gc_rn","com_dist_ras_rn",
        # EMD for existing gc/ras/rn pairs
        "emd_gc_ras","emd_gc_rn","emd_ras_rn",
        # rw COM-distance and EMD pairs
        "com_dist_gc_rw","emd_gc_rw",
        "com_dist_ras_rw","emd_ras_rw",
        "com_dist_rn_rw","emd_rn_rw",
        # vc Gini + cosine agreements + COM-distance and EMD pairs
        "vc_gini",
        "agree_gc_vc","agree_ras_vc","agree_rn_vc","agree_rw_vc",
        "com_dist_gc_vc","emd_gc_vc",
        "com_dist_ras_vc","emd_ras_vc",
        "com_dist_rn_vc","emd_rn_vc",
        "com_dist_rw_vc","emd_rw_vc",
    )
    if gc is None:
        for pfx in nan_keys:
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

    # Top-K IoU agreement (spatial pattern match, complementary to cosine)
    f[f"iou_gc_ras_{label}"]   = _topk_iou(gc, ras)
    f[f"iou_gc_rn_{label}"]    = _topk_iou(gc, rn)
    f[f"iou_ras_rn_{label}"]   = _topk_iou(ras, rn)

    # Centre-of-mass distance (2-D spatial shift between attribution focuses)
    f[f"com_dist_gc_ras_{label}"]  = _com_dist_2d(gc, ras)
    f[f"com_dist_gc_rn_{label}"]   = _com_dist_2d(gc, rn)
    f[f"com_dist_ras_rn_{label}"]  = _com_dist_2d(ras, rn)

    # EMD for existing gc/ras/rn pairs (x/y-marginal Wasserstein distance)
    f[f"emd_gc_ras_{label}"]  = _emd_2d(gc, ras)
    f[f"emd_gc_rn_{label}"]   = _emd_2d(gc, rn)
    f[f"emd_ras_rn_{label}"]  = _emd_2d(ras, rn)

    # Cross-signal: attn weights vs gradcam/ras on same camera patch tokens
    if attn_seg is not None:
        attn_map = attn_seg.reshape(gc.shape) if attn_seg.size == gc.size else None
        f[f"agree_attn_gc_{label}"]  = cosine(attn_seg, gc.ravel())  if attn_map is not None else _nan()
        f[f"agree_attn_ras_{label}"] = cosine(attn_seg, ras.ravel()) if attn_map is not None else _nan()
    else:
        f[f"agree_attn_gc_{label}"]  = _nan()
        f[f"agree_attn_ras_{label}"] = _nan()

    # Raw-weights-max: cosine agreements (existing) + new COM-dist and EMD pairs
    if rw is not None:
        rw_f = rw.ravel() if rw.ndim > 1 else rw
        f[f"rw_gini_{label}"]         = gini(rw_f)
        f[f"agree_gc_rw_{label}"]     = cosine(gc.ravel(), rw_f)
        f[f"agree_ras_rw_{label}"]    = cosine(ras.ravel(), rw_f)
        f[f"agree_rn_rw_{label}"]     = cosine(rn.ravel(), rw_f)
        f[f"com_dist_gc_rw_{label}"]  = _com_dist_2d(gc,  rw_f)
        f[f"emd_gc_rw_{label}"]       = _emd_2d(gc,  rw_f)
        f[f"com_dist_ras_rw_{label}"] = _com_dist_2d(ras, rw_f)
        f[f"emd_ras_rw_{label}"]      = _emd_2d(ras, rw_f)
        f[f"com_dist_rn_rw_{label}"]  = _com_dist_2d(rn,  rw_f)
        f[f"emd_rn_rw_{label}"]       = _emd_2d(rn,  rw_f)
    else:
        for pfx in ("rw_gini","agree_gc_rw","agree_ras_rw","agree_rn_rw",
                    "com_dist_gc_rw","emd_gc_rw","com_dist_ras_rw","emd_ras_rw",
                    "com_dist_rn_rw","emd_rn_rw"):
            f[f"{pfx}_{label}"] = _nan()

    # V-Cosine: Gini + cosine agreements + COM-dist and EMD against all other methods
    if vc is not None:
        vc_f = vc.ravel() if vc.ndim > 1 else vc
        f[f"vc_gini_{label}"]         = gini(vc_f)
        f[f"agree_gc_vc_{label}"]     = cosine(gc.ravel(),  vc_f)
        f[f"agree_ras_vc_{label}"]    = cosine(ras.ravel(), vc_f)
        f[f"agree_rn_vc_{label}"]     = cosine(rn.ravel(),  vc_f)
        f[f"com_dist_gc_vc_{label}"]  = _com_dist_2d(gc,  vc_f)
        f[f"emd_gc_vc_{label}"]       = _emd_2d(gc,  vc_f)
        f[f"com_dist_ras_vc_{label}"] = _com_dist_2d(ras, vc_f)
        f[f"emd_ras_vc_{label}"]      = _emd_2d(ras, vc_f)
        f[f"com_dist_rn_vc_{label}"]  = _com_dist_2d(rn,  vc_f)
        f[f"emd_rn_vc_{label}"]       = _emd_2d(rn,  vc_f)
        if rw is not None:
            f[f"agree_rw_vc_{label}"]    = cosine(rw_f, vc_f)
            f[f"com_dist_rw_vc_{label}"] = _com_dist_2d(rw_f, vc_f)
            f[f"emd_rw_vc_{label}"]      = _emd_2d(rw_f, vc_f)
        else:
            f[f"agree_rw_vc_{label}"]    = _nan()
            f[f"com_dist_rw_vc_{label}"] = _nan()
            f[f"emd_rw_vc_{label}"]      = _nan()
    else:
        for pfx in ("vc_gini",
                    "agree_gc_vc","agree_ras_vc","agree_rn_vc","agree_rw_vc",
                    "com_dist_gc_vc","emd_gc_vc","com_dist_ras_vc","emd_ras_vc",
                    "com_dist_rn_vc","emd_rn_vc","com_dist_rw_vc","emd_rw_vc"):
            f[f"{pfx}_{label}"] = _nan()

    return f


def _extract_heatmaps(rec: dict, fmt: str, attn_feats: dict) -> dict:
    """All per-camera heatmap features + cross-camera agreement."""
    feats: dict = {}
    cams = [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist"),
            ("right_wrist_0_rgb", "rwrist")]

    maps: dict = {}  # store raw maps for cross-camera agreement

    for cam_key, cam_label in cams:
        if fmt == "pi0fast":
            gc, ras, rn, gcs_per_tok, ras_per_tok, rns_per_tok = _load_heatmap_pi0fast(rec, cam_key)
        else:
            gc, ras, rn = _load_heatmap_pi05(rec, cam_key)
            gcs_per_tok = ras_per_tok = rns_per_tok = []

        attn_seg = attn_feats.get(f"_attn_{cam_label}")
        rw_seg   = attn_feats.get(f"_rw_{cam_label}")
        vc_seg   = attn_feats.get(f"_vc_{cam_label}")
        feats.update(_heatmap_scalars(gc, ras, rn, attn_seg, cam_label, rw=rw_seg, vc=vc_seg))
        maps[cam_label] = (gc, ras, rn, gcs_per_tok, ras_per_tok, rns_per_tok)

    # Cross-camera agreement within each signal type
    for sig, idx in [("gc", 0), ("ras", 1), ("rn", 2)]:
        pairs = [("base", "wrist"), ("base", "rwrist"), ("wrist", "rwrist")]
        for l1, l2 in pairs:
            m1 = maps[l1][idx]; m2 = maps[l2][idx]
            key = f"agree_{l1}_{l2}_{sig}"
            feats[key] = cosine(m1, m2) if (m1 is not None and m2 is not None) else _nan()

    feats["_maps"] = maps  # pass through for per-token features
    return feats


def _extract_per_token_pi0fast(maps_data: dict) -> dict:
    """
    Per-action-token features for pi0fast cameras.

    Stores individual token Gini values (t0–t4) for gc, ras, rn on each camera,
    enabling downstream comparison of token aggregation strategies:
      - per-token (t0, t1, t2, t3, t4)
      - mean (already in main heatmap features as gc_gini_{cam})
      - max-anomaly / min-gini (computed in analysis from per-token values)
      - weighted (near-term, far-term, learned — computed in analysis)
    """
    feats: dict = {}
    N_TOK = 5
    for cam_label in ("base", "wrist", "rwrist"):
        _, _, _, gcs, ras_l, rns = maps_data.get(cam_label, (None, None, None, [], [], []))

        # Nan-fill template for all per-token and aggregate keys
        _nan_sfxs = (
            [f"gc_gini_{cam_label}_t{i}"  for i in range(N_TOK)] +
            [f"ras_gini_{cam_label}_t{i}" for i in range(N_TOK)] +
            [f"rn_gini_{cam_label}_t{i}"  for i in range(N_TOK)] +
            [f"gc_gini_{cam_label}_std", f"ras_gini_{cam_label}_std",
             f"rn_gini_{cam_label}_std",
             f"horizon_agree_gc_{cam_label}", f"horizon_agree_ras_{cam_label}"]
        )
        if cam_label == "base":
            _nan_sfxs += ["gc_com_y_base_t0", "gc_com_x_base_t0"]

        if not gcs:
            for sfx in _nan_sfxs:
                feats[sfx] = _nan()
            continue

        # Individual token Gini values (t0–t4) — the primary new addition.
        # These let the analysis script compute any aggregation mode without
        # re-reading raw .npy files.
        gc_ginis  = [gini(gcs[i].ravel())  if i < len(gcs)  else float("nan") for i in range(N_TOK)]
        ras_ginis = [gini(ras_l[i].ravel()) if i < len(ras_l) else float("nan") for i in range(N_TOK)]
        rn_ginis  = [gini(rns[i].ravel())  if i < len(rns)  else float("nan") for i in range(N_TOK)]

        for i in range(N_TOK):
            feats[f"gc_gini_{cam_label}_t{i}"]  = gc_ginis[i]
            feats[f"ras_gini_{cam_label}_t{i}"] = ras_ginis[i]
            feats[f"rn_gini_{cam_label}_t{i}"]  = rn_ginis[i]

        # Token-0 spatial CoM (base cam only)
        if cam_label == "base":
            cy, cx = spatial_com(gcs[0])
            feats["gc_com_y_base_t0"] = cy
            feats["gc_com_x_base_t0"] = cx

        # Std of Gini across tokens (plan instability across horizon)
        feats[f"gc_gini_{cam_label}_std"]  = float(np.nanstd(gc_ginis))
        feats[f"ras_gini_{cam_label}_std"] = float(np.nanstd(ras_ginis))
        feats[f"rn_gini_{cam_label}_std"]  = float(np.nanstd(rn_ginis))

        # Horizon agreement: token 0 vs token 4
        feats[f"horizon_agree_gc_{cam_label}"]  = cosine(gcs[0], gcs[-1])
        feats[f"horizon_agree_ras_{cam_label}"] = cosine(ras_l[0], ras_l[-1])

    return feats


def _extract_task_features(rec: dict, fmt: str, attn_feats: dict) -> dict:
    """Task (and state) modality features for all 5 signal types."""
    feats: dict = {}

    if fmt == "pi0fast":
        gc_task_list, ras_task_list, rn_task_list = [], [], []
        for i in range(5):
            k_gc  = f"outputs/debug/gradcam/action_tokens/{i}/task"
            k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/task"
            k_rn  = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
            if k_gc not in rec:
                break
            gc_task_list.append(np.abs(np.asarray(rec[k_gc])[0]).astype(np.float32))
            ras_task_list.append(np.abs(np.asarray(rec[k_ras])[0]).astype(np.float32))
            rn_task_list.append(np.abs(np.asarray(rec[k_rn])[0]).astype(np.float32))
        gc_task  = np.mean(np.stack(gc_task_list),  axis=0) if gc_task_list  else None
        ras_task = np.mean(np.stack(ras_task_list), axis=0) if ras_task_list else None
        rn_task  = np.mean(np.stack(rn_task_list),  axis=0) if rn_task_list  else None

        # Per-token task Gini (t0–t4) — stored individually so the analysis
        # script can test different aggregation strategies (per-token, max,
        # weighted, learned) without re-reading raw .npy files.
        for i in range(5):
            if i < len(gc_task_list):
                feats[f"gc_gini_task_t{i}"]  = gini(gc_task_list[i])
                feats[f"ras_gini_task_t{i}"] = gini(ras_task_list[i])
                feats[f"rn_gini_task_t{i}"]  = gini(rn_task_list[i]) if i < len(rn_task_list) else _nan()
            else:
                feats[f"gc_gini_task_t{i}"]  = _nan()
                feats[f"ras_gini_task_t{i}"] = _nan()
                feats[f"rn_gini_task_t{i}"]  = _nan()

        # State tokens — average over action tokens, then slice to digit-only.
        # The score arrays cover all tokens in "State: {digits};\n" including
        # syntactic prefix ("State", ":") and suffix (";", "\n") tokens.
        # _state_digit_slice() finds (k, end) so array[k:end] is digit-only.
        gc_state_list, ras_state_list = [], []
        for i in range(5):
            k_gc  = f"outputs/debug/gradcam/action_tokens/{i}/state"
            k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/state"
            if k_gc not in rec:
                break
            gc_state_list.append(np.abs(np.asarray(rec[k_gc])[0]).astype(np.float32))
            ras_state_list.append(np.abs(np.asarray(rec[k_ras])[0]).astype(np.float32))
        if gc_state_list:
            dk, dend = _state_digit_slice(rec)
            gc_state  = np.mean(np.stack(gc_state_list),  axis=0)[dk:dend]
            ras_state = np.mean(np.stack(ras_state_list), axis=0)[dk:dend]
        else:
            gc_state = ras_state = None
    else:
        # pi05 format: single GradCAM vector per camera (not per action token).
        # task GradCAM is (200,) but only the first task_token_len tokens are real;
        # the rest are padding that was zeroed out by the model's token mask.
        # Load the full vector, then truncate to the real (masked) tokens so that
        # Gini/entropy are computed over the true distribution, not a sparse 200-dim one.
        k_gc  = "outputs/debug/gradcam/task"
        k_ras = "outputs/debug/raw_alpha/summation/task"
        k_rn  = "outputs/debug/raw_alpha/norm/task"
        gc_task_full  = np.abs(np.asarray(rec[k_gc]))  .astype(np.float32) if k_gc  in rec else None
        ras_task_full = np.abs(np.asarray(rec[k_ras])) .astype(np.float32) if k_ras in rec else None
        rn_task_full  = np.abs(np.asarray(rec[k_rn]))  .astype(np.float32) if k_rn  in rec else None
        gc_state = ras_state = None

        # Truncate to real tokens using token_mask (True for real, False for padding).
        task_mask_key = "outputs/debug/tokens/task/token_mask"
        if task_mask_key in rec and gc_task_full is not None:
            task_mask = np.asarray(rec[task_mask_key]).astype(bool)
            # token_mask may be longer than the vectors if it covers task+state tokens;
            # align to the gc_task_full length.
            n_mask = min(len(task_mask), len(gc_task_full))
            n_real = int(task_mask[:n_mask].sum())
            # Gini/entropy are computed over real tokens only to avoid sparse-vector inflation.
            # Cosine agreements use the full (zero-padded) vectors — zeros cancel in dot
            # product and norm, so cosine(full, full) == cosine(real, real).
            gc_task  = gc_task_full[:n_real]  if n_real > 0 else gc_task_full
            ras_task = ras_task_full[:n_real] if (ras_task_full is not None and n_real > 0) else ras_task_full
            rn_task  = rn_task_full[:n_real]  if (rn_task_full is not None and n_real > 0) else rn_task_full
        else:
            gc_task, ras_task, rn_task = gc_task_full, ras_task_full, rn_task_full
        # pi05 has no per-action-token maps, so per-token task Gini is unavailable
        for i in range(5):
            feats[f"gc_gini_task_t{i}"]  = _nan()
            feats[f"ras_gini_task_t{i}"] = _nan()
            feats[f"rn_gini_task_t{i}"]  = _nan()

    attn_task = attn_feats.get("_attn_task")
    rw_task   = attn_feats.get("_rw_task")    # raw-weights-max for task span
    vc_task   = attn_feats.get("_vc_task")    # V-Cosine scores for task span

    # All task-token feature names — used for the nan-fill branch below.
    _TASK_NAN_KEYS = (
        "gc_gini_task","ras_gini_task","rn_gini_task",
        # per-token Gini (the nan branch is only for gc_task is None, which
        # can't happen for pi0fast — but guard it for safety)
        *[f"gc_gini_task_t{i}"  for i in range(5)],
        *[f"ras_gini_task_t{i}" for i in range(5)],
        *[f"rn_gini_task_t{i}"  for i in range(5)],
        "agree_gc_ras_task","agree_gc_rn_task","agree_ras_rn_task",
        "agree_attn_gc_task","agree_attn_ras_task",
        "rw_gini_task","agree_gc_rw_task","agree_ras_rw_task","agree_rn_rw_task",
        "iou_gc_ras_task","iou_gc_rn_task",
        # existing COM-dist pairs
        "com_dist_gc_ras_task","com_dist_gc_rn_task",
        # new: EMD for existing pairs + missing ras-rn pair
        "emd_gc_ras_task","emd_gc_rn_task",
        "com_dist_ras_rn_task","emd_ras_rn_task",
        # rw COM-dist and EMD pairs
        "com_dist_gc_rw_task","emd_gc_rw_task",
        "com_dist_ras_rw_task","emd_ras_rw_task",
        "com_dist_rn_rw_task","emd_rn_rw_task",
        # vc Gini + cosine agreements + COM-dist and EMD pairs
        "vc_gini_task",
        "agree_gc_vc_task","agree_ras_vc_task","agree_rn_vc_task","agree_rw_vc_task",
        "com_dist_gc_vc_task","emd_gc_vc_task",
        "com_dist_ras_vc_task","emd_ras_vc_task",
        "com_dist_rn_vc_task","emd_rn_vc_task",
        "com_dist_rw_vc_task","emd_rw_vc_task",
    )

    if gc_task is not None:
        feats["gc_gini_task"]   = gini(gc_task)
        feats["ras_gini_task"]  = gini(ras_task)
        feats["rn_gini_task"]   = gini(rn_task) if rn_task is not None else _nan()
        # For cosine agreements, use the FULL vectors (including zeros for pi05) so that
        # both vectors have matching shape and the zeros cancel perfectly in the cosine.
        _gc_full  = gc_task_full  if fmt == "pi05" else gc_task
        _ras_full = ras_task_full if fmt == "pi05" else ras_task
        _rn_full  = rn_task_full  if fmt == "pi05" else rn_task
        feats["agree_gc_ras_task"]  = cosine(_gc_full, _ras_full)
        feats["agree_gc_rn_task"]   = cosine(_gc_full, _rn_full)  if _rn_full  is not None else _nan()
        feats["agree_ras_rn_task"]  = cosine(_ras_full, _rn_full) if _rn_full  is not None else _nan()
        if attn_task is not None and attn_task.size == (_gc_full if _gc_full is not None else gc_task).size:
            feats["agree_attn_gc_task"]  = cosine(attn_task, _gc_full)
            feats["agree_attn_ras_task"] = cosine(attn_task, _ras_full)
        else:
            feats["agree_attn_gc_task"]  = _nan()
            feats["agree_attn_ras_task"] = _nan()

        # IoU (top-10% token overlap)
        feats["iou_gc_ras_task"] = _topk_iou(gc_task, ras_task)
        feats["iou_gc_rn_task"]  = _topk_iou(gc_task, rn_task) if rn_task is not None else _nan()

        # Existing COM-dist pairs
        feats["com_dist_gc_ras_task"] = _com_dist_1d(gc_task, ras_task)
        feats["com_dist_gc_rn_task"]  = _com_dist_1d(gc_task, rn_task) if rn_task is not None else _nan()

        # New: EMD for existing pairs + ras-rn pair (COM+EMD)
        feats["emd_gc_ras_task"]      = _emd_1d(gc_task, ras_task)
        feats["emd_gc_rn_task"]       = _emd_1d(gc_task, rn_task)  if rn_task  is not None else _nan()
        feats["com_dist_ras_rn_task"] = _com_dist_1d(ras_task, rn_task) if rn_task is not None else _nan()
        feats["emd_ras_rn_task"]      = _emd_1d(ras_task, rn_task) if rn_task  is not None else _nan()

        # Raw-weights-max: cosine (existing) + new COM-dist and EMD pairs
        if rw_task is not None:
            feats["rw_gini_task"]         = gini(rw_task)
            feats["agree_gc_rw_task"]     = cosine(gc_task, rw_task)
            feats["agree_ras_rw_task"]    = cosine(ras_task, rw_task)
            feats["agree_rn_rw_task"]     = cosine(rn_task, rw_task) if rn_task is not None else _nan()
            feats["com_dist_gc_rw_task"]  = _com_dist_1d(gc_task,  rw_task)
            feats["emd_gc_rw_task"]       = _emd_1d(gc_task,  rw_task)
            feats["com_dist_ras_rw_task"] = _com_dist_1d(ras_task, rw_task)
            feats["emd_ras_rw_task"]      = _emd_1d(ras_task, rw_task)
            feats["com_dist_rn_rw_task"]  = _com_dist_1d(rn_task,  rw_task) if rn_task is not None else _nan()
            feats["emd_rn_rw_task"]       = _emd_1d(rn_task,  rw_task) if rn_task  is not None else _nan()
        else:
            for k in ("rw_gini_task","agree_gc_rw_task","agree_ras_rw_task","agree_rn_rw_task",
                      "com_dist_gc_rw_task","emd_gc_rw_task",
                      "com_dist_ras_rw_task","emd_ras_rw_task",
                      "com_dist_rn_rw_task","emd_rn_rw_task"):
                feats[k] = _nan()

        # V-Cosine: Gini + cosine agreements + COM-dist and EMD pairs
        if vc_task is not None:
            feats["vc_gini_task"]         = gini(vc_task)
            feats["agree_gc_vc_task"]     = cosine(gc_task,  vc_task)
            feats["agree_ras_vc_task"]    = cosine(ras_task, vc_task)
            feats["agree_rn_vc_task"]     = cosine(rn_task,  vc_task) if rn_task  is not None else _nan()
            feats["agree_rw_vc_task"]     = cosine(rw_task,  vc_task) if rw_task  is not None else _nan()
            feats["com_dist_gc_vc_task"]  = _com_dist_1d(gc_task,  vc_task)
            feats["emd_gc_vc_task"]       = _emd_1d(gc_task,  vc_task)
            feats["com_dist_ras_vc_task"] = _com_dist_1d(ras_task, vc_task)
            feats["emd_ras_vc_task"]      = _emd_1d(ras_task, vc_task)
            feats["com_dist_rn_vc_task"]  = _com_dist_1d(rn_task,  vc_task) if rn_task is not None else _nan()
            feats["emd_rn_vc_task"]       = _emd_1d(rn_task,  vc_task) if rn_task  is not None else _nan()
            feats["com_dist_rw_vc_task"]  = _com_dist_1d(rw_task,  vc_task) if rw_task is not None else _nan()
            feats["emd_rw_vc_task"]       = _emd_1d(rw_task,  vc_task) if rw_task  is not None else _nan()
        else:
            for k in ("vc_gini_task",
                      "agree_gc_vc_task","agree_ras_vc_task","agree_rn_vc_task","agree_rw_vc_task",
                      "com_dist_gc_vc_task","emd_gc_vc_task",
                      "com_dist_ras_vc_task","emd_ras_vc_task",
                      "com_dist_rn_vc_task","emd_rn_vc_task",
                      "com_dist_rw_vc_task","emd_rw_vc_task"):
                feats[k] = _nan()
    else:
        for k in _TASK_NAN_KEYS:
            feats[k] = _nan()

    # ── State token features ──────────────────────────────────────────────────
    # Digit-slice indices: score_array[dk:dend] = digit-only tokens, stripping
    # "State: " prefix and ";\n" suffix that are syntactic markers, not joint values.
    # _state_digit_slice returns (0, None) when metadata is absent → full array.
    dk, dend = _state_digit_slice(rec)

    # Slice raw-weights and V-Cosine to digit-only to match gc_state / ras_state length.
    rw_state_raw = attn_feats.get("_rw_state")
    vc_state_raw = attn_feats.get("_vc_state")
    rw_state = rw_state_raw[dk:dend] if rw_state_raw is not None else None
    vc_state = vc_state_raw[dk:dend] if vc_state_raw is not None else None

    _STATE_NAN_KEYS = (
        "gc_gini_state","ras_gini_state","rw_gini_state","vc_gini_state",
        "agree_gc_ras_state","agree_gc_rw_state","agree_gc_vc_state",
        "agree_ras_rw_state","agree_ras_vc_state","agree_rw_vc_state",
        "com_dist_gc_ras_state","emd_gc_ras_state",
        "com_dist_gc_rw_state","emd_gc_rw_state",
        "com_dist_gc_vc_state","emd_gc_vc_state",
        "com_dist_ras_rw_state","emd_ras_rw_state",
        "com_dist_ras_vc_state","emd_ras_vc_state",
        "com_dist_rw_vc_state","emd_rw_vc_state",
    )

    # rw_state and vc_state are available for both pi0fast and pi05.
    # gc_state/ras_state are only available for pi0fast (None for pi05).
    if rw_state is not None:
        feats["rw_gini_state"] = gini(rw_state)
    else:
        feats["rw_gini_state"] = _nan()

    if vc_state is not None:
        feats["vc_gini_state"] = gini(vc_state)
    else:
        feats["vc_gini_state"] = _nan()

    if rw_state is not None and vc_state is not None:
        feats["agree_rw_vc_state"]    = cosine(rw_state, vc_state)
        feats["com_dist_rw_vc_state"] = _com_dist_1d(rw_state, vc_state)
        feats["emd_rw_vc_state"]      = _emd_1d(rw_state, vc_state)
    else:
        feats["agree_rw_vc_state"]    = _nan()
        feats["com_dist_rw_vc_state"] = _nan()
        feats["emd_rw_vc_state"]      = _nan()

    if gc_state is not None:
        feats["gc_gini_state"]  = gini(gc_state)
        feats["ras_gini_state"] = gini(ras_state)
        feats["agree_gc_ras_state"]    = cosine(gc_state, ras_state)
        feats["com_dist_gc_ras_state"] = _com_dist_1d(gc_state, ras_state)
        feats["emd_gc_ras_state"]      = _emd_1d(gc_state, ras_state)
        if rw_state is not None:
            feats["agree_gc_rw_state"]    = cosine(gc_state, rw_state)
            feats["agree_ras_rw_state"]   = cosine(ras_state, rw_state)
            feats["com_dist_gc_rw_state"] = _com_dist_1d(gc_state, rw_state)
            feats["emd_gc_rw_state"]      = _emd_1d(gc_state, rw_state)
            feats["com_dist_ras_rw_state"]= _com_dist_1d(ras_state, rw_state)
            feats["emd_ras_rw_state"]     = _emd_1d(ras_state, rw_state)
        else:
            for k in ("agree_gc_rw_state","agree_ras_rw_state",
                      "com_dist_gc_rw_state","emd_gc_rw_state",
                      "com_dist_ras_rw_state","emd_ras_rw_state"):
                feats[k] = _nan()
        if vc_state is not None:
            feats["agree_gc_vc_state"]    = cosine(gc_state, vc_state)
            feats["agree_ras_vc_state"]   = cosine(ras_state, vc_state)
            feats["com_dist_gc_vc_state"] = _com_dist_1d(gc_state, vc_state)
            feats["emd_gc_vc_state"]      = _emd_1d(gc_state, vc_state)
            feats["com_dist_ras_vc_state"]= _com_dist_1d(ras_state, vc_state)
            feats["emd_ras_vc_state"]     = _emd_1d(ras_state, vc_state)
        else:
            for k in ("agree_gc_vc_state","agree_ras_vc_state",
                      "com_dist_gc_vc_state","emd_gc_vc_state",
                      "com_dist_ras_vc_state","emd_ras_vc_state"):
                feats[k] = _nan()
    else:
        for k in ("gc_gini_state","ras_gini_state",
                  "agree_gc_ras_state","com_dist_gc_ras_state","emd_gc_ras_state",
                  "agree_gc_rw_state","agree_ras_rw_state",
                  "com_dist_gc_rw_state","emd_gc_rw_state",
                  "com_dist_ras_rw_state","emd_ras_rw_state",
                  "agree_gc_vc_state","agree_ras_vc_state",
                  "com_dist_gc_vc_state","emd_gc_vc_state",
                  "com_dist_ras_vc_state","emd_ras_vc_state"):
            feats[k] = _nan()

    return feats


def _extract_value_features(rec: dict, fmt: str,
                              prev_centroids: Optional[dict]) -> Tuple[dict, dict]:
    """
    Value vector features from attn/v.

    Returns (feats_dict, centroids_dict).
    centroids_dict contains centroid arrays for passing to the next step.
    """
    feats: dict = {}
    centroids: dict = {}
    vkey = "outputs/debug/attn/v"

    nan_keys = ("v_norm_mean_base","v_norm_gini_base","v_coherence_base",
                "v_norm_mean_wrist","v_norm_gini_wrist","v_coherence_wrist",
                "v_norm_mean_task","v_norm_gini_task","v_coherence_task",
                "v_align_base_task","v_align_base_wrist","v_align_task_wrist",
                "v_drift_base","v_drift_wrist","v_drift_task")

    if vkey not in rec:
        for k in nan_keys:
            feats[k] = _nan()
        return feats, centroids

    v_raw = np.asarray(rec[vkey]).astype(np.float32)
    # Last layer, squeeze all singleton dims → (S, D)
    if fmt == "pi0fast":
        # (2, 1, S, 1, D) → (S, D)
        V = v_raw[-1, 0, :, 0, :]
    else:
        # (2, S, 1, D) → (S, D)
        V = v_raw[-1, :, 0, :]

    sp_base  = _span(rec, "outputs/debug/spans/image/base_0_rgb")
    sp_wrist = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
    sp_task  = _span(rec, "outputs/debug/spans/task")

    # For pi05 the task span covers the full max_token_len (e.g. 200 tokens) but only
    # the first task_token_len tokens are real; the rest are zero-padded.  Truncate the
    # span to the real tokens so that v_coherence/v_norm/v_align are not distorted by
    # the many near-zero V vectors of padding positions.
    if fmt == "pi05" and sp_task is not None:
        task_mask_key = "outputs/debug/tokens/task/token_mask"
        if task_mask_key in rec:
            task_mask = np.asarray(rec[task_mask_key]).astype(bool)
            span_len  = sp_task[1] - sp_task[0]
            n_real    = int(task_mask[:span_len].sum())
            if n_real > 0:
                sp_task = (sp_task[0], sp_task[0] + n_real)

    def modality_stats(sp, tag):
        if sp is None:
            feats[f"v_norm_mean_{tag}"]  = _nan()
            feats[f"v_norm_gini_{tag}"]  = _nan()
            feats[f"v_coherence_{tag}"]  = _nan()
            centroids[f"v_centroid_{tag}"] = None
            return
        seg = V[sp[0]:sp[1]]          # (N, D)
        norms = np.linalg.norm(seg, axis=1)            # (N,)
        feats[f"v_norm_mean_{tag}"] = float(norms.mean())
        feats[f"v_norm_gini_{tag}"] = gini(norms)
        centroid = seg.mean(axis=0)                     # (D,)
        cnorm = np.linalg.norm(centroid)
        if cnorm > 1e-9:
            c_unit = centroid / cnorm
            # Coherence: mean cosine of each row to centroid
            dot = seg @ c_unit                          # (N,)
            row_norms = norms + 1e-9
            feats[f"v_coherence_{tag}"] = float((dot / row_norms).mean())
        else:
            feats[f"v_coherence_{tag}"] = _nan()
        centroids[f"v_centroid_{tag}"] = centroid

    modality_stats(sp_base,  "base")
    modality_stats(sp_wrist, "wrist")
    modality_stats(sp_task,  "task")

    # Cross-modality alignment
    for t1, t2 in [("base","task"), ("base","wrist"), ("task","wrist")]:
        c1 = centroids.get(f"v_centroid_{t1}")
        c2 = centroids.get(f"v_centroid_{t2}")
        feats[f"v_align_{t1}_{t2}"] = cosine(c1, c2) if (c1 is not None and c2 is not None) else _nan()

    # Temporal centroid drift (1 - cosine similarity to prev)
    for tag in ("base", "wrist", "task"):
        pc = prev_centroids.get(f"v_centroid_{tag}") if prev_centroids else None
        cc = centroids.get(f"v_centroid_{tag}")
        if pc is not None and cc is not None:
            feats[f"v_drift_{tag}"] = 1.0 - cosine(cc, pc)
        else:
            feats[f"v_drift_{tag}"] = _nan()

    return feats, centroids


def _extract_action_features(rec: dict, prev_rec: Optional[dict]) -> dict:
    feats: dict = {}
    actions = np.asarray(rec["outputs/actions"]).astype(np.float64)  # (10, 7)

    signs = np.sign(actions)
    flips = (signs[1:] != signs[:-1]).astype(float)
    feats["action_trans_flip"] = float(flips[:, 0:3].mean())
    feats["action_rot_flip"]   = float(flips[:, 3:6].mean())
    feats["action_grip_flip"]  = float(flips[:, 6].mean())
    feats["action_magnitude"]  = float(np.linalg.norm(actions[0]))
    feats["gripper_val"]       = float(actions[0, 6])

    # Plan statistics over the 10-step horizon
    step_norms = np.linalg.norm(actions, axis=1)          # (10,)
    feats["action_spread"]         = float(step_norms.std())
    feats["action_plan_norm"]      = float(step_norms.mean())
    feats["action_plan_entropy"]   = entropy(step_norms)
    feats["action_horizon_align"]  = cosine(actions[0], actions[-1])

    # Action direction variance across the 10 steps
    feats["action_dir_var"] = float(np.var(actions[:, 0:6], axis=0).mean())

    # Temporal features (need previous step)
    # Use outputs/state if available; pi05 records it as (32,) with last 24 dims=0,
    # so we truncate to 8 dims for consistency with pi0fast's (8,) format.
    state_key = "outputs/state" if "outputs/state" in rec else "inputs/observation/state"
    state = np.asarray(rec.get(state_key, np.zeros(8))).astype(np.float64)[:8]

    if prev_rec is not None:
        prev_state_key = "outputs/state" if "outputs/state" in prev_rec else "inputs/observation/state"
        prev_state = np.asarray(prev_rec.get(prev_state_key, state)).astype(np.float64)[:8]
        feats["joint_vel"] = float(np.linalg.norm(state - prev_state))

        prev_actions = np.asarray(prev_rec["outputs/actions"]).astype(np.float64)
        a0, p0 = actions[0], prev_actions[0]
        na, np_ = np.linalg.norm(a0), np.linalg.norm(p0)
        feats["plan_consistency"] = float(np.dot(a0, p0) / (na * np_ + 1e-9))
        feats["action_grip_change"] = float(abs(actions[0, 6] - prev_actions[0, 6]))
    else:
        feats["joint_vel"]         = 0.0
        feats["plan_consistency"]  = 1.0
        feats["action_grip_change"]= 0.0

    feats["state_norm"] = float(np.linalg.norm(state))
    return feats


# ─── Main feature extractor ────────────────────────────────────────────────────

def extract_features(rec: dict,
                     prev_rec: Optional[dict] = None,
                     prev_centroids: Optional[dict] = None
                     ) -> Tuple[dict, dict]:
    """
    Extract all scalar features from one step record.

    Returns (feats, centroids):
      feats      — dict of scalar features (NaN for unavailable)
      centroids  — dict of value-vector centroids for passing to next step
    """
    fmt = detect_format(rec)
    feats: dict = {}

    # A. Attention routing
    attn_feats = _extract_attn_routing(rec, fmt)
    feats.update({k: v for k, v in attn_feats.items() if not k.startswith("_")})

    # B+C. Heatmaps (all cams) + cross-camera agreement
    hm_feats = _extract_heatmaps(rec, fmt, attn_feats)
    maps_data = hm_feats.pop("_maps", {})
    feats.update(hm_feats)

    # D. Task/state modality
    feats.update(_extract_task_features(rec, fmt, attn_feats))

    # E. Per-action-token (pi0fast only; NaN for pi05)
    if fmt == "pi0fast":
        feats.update(_extract_per_token_pi0fast(maps_data))
    else:
        for k in ("gc_gini_base_t0","ras_gini_base_t0","rn_gini_base_t0",
                  "gc_gini_wrist_t0","ras_gini_wrist_t0","rn_gini_wrist_t0",
                  "gc_gini_rwrist_t0","ras_gini_rwrist_t0","rn_gini_rwrist_t0",
                  "gc_com_y_base_t0","gc_com_x_base_t0",
                  "gc_gini_base_std","ras_gini_base_std",
                  "gc_gini_wrist_std","ras_gini_wrist_std",
                  "gc_gini_rwrist_std",
                  "horizon_agree_gc_base","horizon_agree_ras_base",
                  "horizon_agree_gc_wrist","horizon_agree_ras_wrist",
                  "horizon_agree_gc_rwrist"):
            feats[k] = _nan()

    # F. Value vector features
    v_feats, centroids = _extract_value_features(rec, fmt, prev_centroids)
    feats.update(v_feats)

    # G+H. Action and state features
    feats.update(_extract_action_features(rec, prev_rec))

    # Gradcam CoM drift (base cam)
    gc_base = maps_data.get("base", (None,))[0]
    if gc_base is not None and prev_rec is not None:
        if fmt == "pi0fast":
            prev_gc, _, _, _, _, _ = _load_heatmap_pi0fast(prev_rec, "base_0_rgb")
        else:
            prev_gc, _, _ = _load_heatmap_pi05(prev_rec, "base_0_rgb")
        if prev_gc is not None:
            cy1, cx1 = spatial_com(gc_base)
            cy2, cx2 = spatial_com(prev_gc)
            feats["gc_com_drift_base"] = float(np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2))
        else:
            feats["gc_com_drift_base"] = _nan()
    else:
        feats["gc_com_drift_base"] = 0.0

    return feats, centroids


# ─── Episode and group loading ─────────────────────────────────────────────────

def load_episode_features(data_dir: Path, ep: dict, max_steps: int) -> List[dict]:
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
    """Write pickle atomically via tmp → rename to avoid corruption."""
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
               max_steps: int = MAX_STEPS,
               use_cache: bool = True) -> Dict[str, List[List[dict]]]:
    cache_path = CACHE_DIR / f"{group_name}.pkl"
    if use_cache and cache_path.exists():
        try:
            print(f"  Loading {group_name} from cache …")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  [warn] cache corrupted ({e}), re-extracting …")

    ep_data: Dict[str, List[List[dict]]] = {"success": [], "failure": []}
    for run_name in run_names:
        data_dir = BASE / run_name
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


# ─── Analysis utilities ─────────────────────────────────────────────────────────

def feature_names(ep_data: Dict) -> List[str]:
    SKIP = {"rel_step", "abs_step"}
    for eps in ep_data.values():
        for ep in eps:
            if ep:
                return [k for k in ep[0] if k not in SKIP]
    return []


def to_matrix(ep_list: List[List[dict]], key: str, max_steps: int) -> np.ndarray:
    out = np.full((len(ep_list), max_steps), np.nan)
    for i, ep in enumerate(ep_list):
        for rec in ep:
            t = rec.get("rel_step", 0)
            if t < max_steps:
                out[i, t] = rec.get(key, np.nan)
    return out


def auroc_at(sv: np.ndarray, fv: np.ndarray) -> float:
    sv = sv[~np.isnan(sv)]; fv = fv[~np.isnan(fv)]
    if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
        return np.nan
    try:
        stat, _ = mannwhitneyu(fv, sv, alternative="two-sided")
        a = float(stat) / (len(sv) * len(fv))
        return max(a, 1.0 - a)
    except Exception:
        return np.nan


def compute_auroc_sequence(ep_data: Dict, feature: str, max_steps: int) -> np.ndarray:
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    return np.array([auroc_at(S[:, t], F[:, t]) for t in range(max_steps)])


def rolling_mean_matrix(mat: np.ndarray, W: int) -> np.ndarray:
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[1]):
        win = mat[:, max(0, t - W + 1): t + 1]
        out[:, t] = np.nanmean(win, axis=1)
    return out


def compute_pvalue_sequence(ep_data: Dict, feature: str, max_steps: int) -> np.ndarray:
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    pvals = np.ones(max_steps)
    for t in range(max_steps):
        sv = S[:, t][~np.isnan(S[:, t])]; fv = F[:, t][~np.isnan(F[:, t])]
        if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
            continue
        try:
            _, p = mannwhitneyu(sv, fv, alternative="two-sided")
            pvals[t] = float(p)
        except Exception:
            pass
    return pvals


# ─── Figure: AUROC heatmap + top curves ────────────────────────────────────────

def fig_auroc_overview(ep_data: Dict, group_name: str, max_steps: int = MAX_STEPS):
    feats = feature_names(ep_data)
    print(f"    Computing AUROC for {len(feats)} features …")
    auroc_mat = np.full((len(feats), max_steps), np.nan)
    for fi, feat in enumerate(tqdm(feats, desc="    AUROC", ncols=80, leave=False)):
        auroc_mat[fi] = compute_auroc_sequence(ep_data, feat, max_steps)

    mean_auroc = np.nanmean(auroc_mat, axis=1)
    order = np.argsort(mean_auroc)[::-1]
    s_feats = [feats[i] for i in order]
    s_mat   = auroc_mat[order]
    s_mean  = mean_auroc[order]

    top_n = min(50, len(s_feats))
    fig, ax = plt.subplots(figsize=(14, top_n * 0.37 + 1.5))
    im = ax.imshow(s_mat[:top_n], aspect="auto", vmin=0.5, vmax=1.0,
                   cmap="RdYlGn", origin="upper")
    ax.set_xticks(np.arange(0, max_steps, 5))
    ax.set_xticklabels(np.arange(0, max_steps, 5))
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels([f"{s_feats[i]}  ({s_mean[i]:.3f})" for i in range(top_n)], fontsize=6.5)
    ax.set_xlabel("Step within episode")
    ax.set_title(f"AUROC(step) heatmap — {group_name}\n"
                 f"(top-{top_n} by mean AUROC; green=separable, red=chance)",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig1_auroc_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")

    # Top-10 AUROC curves
    top10 = min(10, len(s_feats))
    colors = plt.cm.tab10(np.linspace(0, 1, top10))
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(max_steps)
    for i in range(top10):
        a = s_mat[i]; valid = ~np.isnan(a)
        ax.plot(steps[valid], a[valid], color=colors[i], lw=1.8,
                label=f"{s_feats[i]} ({s_mean[i]:.3f})")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(0.7, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 1.05)
    ax.set_title(f"Top-10 feature AUROC curves — {group_name}", fontweight="bold")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    fig.tight_layout()
    out2 = OUTPUT_DIR / group_name / "fig1b_auroc_top10.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out2}")

    return s_feats, s_mat, s_mean


# ─── Figure: Early detection ranking ───────────────────────────────────────────

def fig_early_ranking(s_feats, s_mat, group_name, horizons=(5, 10, 20, 30)):
    top_n = min(20, len(s_feats))
    n_h = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(n_h * 5.5, top_n * 0.43 + 1.5), sharey=True)
    if n_h == 1:
        axes = [axes]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_h))
    for ax, h, col in zip(axes, horizons, colors):
        vals = s_mat[:, h] if h < s_mat.shape[1] else np.full(len(s_feats), np.nan)
        order_h = np.argsort(vals)[::-1][:top_n]
        ax.barh(np.arange(top_n), vals[order_h], color=col, alpha=0.8)
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([s_feats[i] for i in order_h], fontsize=6.5)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.axvline(0.7, color="gray", ls=":", lw=0.8)
        ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC")
        ax.set_title(f"Step {h}", fontweight="bold"); ax.invert_yaxis()
    fig.suptitle(f"Best early-detection features — {group_name}", fontsize=11,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig2_early_ranking.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Mean ± CI trajectories ────────────────────────────────────────────

def fig_trajectories(ep_data, features, group_name, max_steps=MAX_STEPS, tag=""):
    n = len(features)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    steps = np.arange(max_steps)
    for ax, feat in zip(axes, features):
        pvals = compute_pvalue_sequence(ep_data, feat, max_steps)
        sig_mask = pvals < 0.05
        for label, cm, ci_col in [("success", BLUE_M, BLUE_L), ("failure", RED_M, RED_L)]:
            eps = [e for e in ep_data.get(label, []) if e]
            if not eps:
                continue
            mat = to_matrix(eps, feat, max_steps)
            n_active = np.sum(~np.isnan(mat), axis=0)
            mean = np.nanmean(mat, axis=0)
            se = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n_active, 1))
            ci95 = 1.96 * se
            for row in mat:
                valid = ~np.isnan(row)
                if valid.sum() < 2:
                    continue
                ax.plot(steps[valid], row[valid], color=ci_col, lw=LW_I, alpha=AL, zorder=1)
            valid = ~np.isnan(mean)
            ax.plot(steps[valid], mean[valid], color=cm, lw=LW_M, label=label, zorder=3)
            ax.fill_between(steps[valid], (mean - ci95)[valid], (mean + ci95)[valid],
                            color=cm, alpha=0.18, zorder=2)
        for t in range(max_steps - 1):
            if sig_mask[t]:
                ax.axvspan(t - 0.5, t + 0.5, color="gold", alpha=0.18, zorder=0)
        ax.set_ylabel(feat, fontsize=7)
        ax.set_title(feat, loc="left", pad=2, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Step within episode")
    tag_str = f"_{tag}" if tag else ""
    fig.suptitle(f"Mean ± 95%CI — {group_name}{tag_str}", fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / f"fig3_traj{tag_str}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Sliding-window AUROC ──────────────────────────────────────────────

def fig_window_auroc(ep_data, s_feats, group_name, max_steps=MAX_STEPS,
                     windows=(1, 3, 5, 10, 20)):
    top_feats = s_feats[:min(8, len(s_feats))]
    n_f = len(top_feats); n_w = len(windows)
    fig, axes = plt.subplots(n_f, 1, figsize=(13, 2.8 * n_f + 0.5), sharex=True)
    if n_f == 1:
        axes = [axes]
    steps = np.arange(max_steps)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_w))

    for ax, feat in zip(axes, top_feats):
        S_raw = to_matrix(ep_data["success"], feat, max_steps)
        F_raw = to_matrix(ep_data["failure"], feat, max_steps)
        for col, W in zip(colors, windows):
            S = rolling_mean_matrix(S_raw, W)
            F = rolling_mean_matrix(F_raw, W)
            aurocs = np.array([auroc_at(S[:, t], F[:, t]) for t in range(max_steps)])
            valid = ~np.isnan(aurocs)
            ax.plot(steps[valid], aurocs[valid], color=col, lw=1.8, label=f"W={W}")
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.axhline(0.7, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0.45, 1.05); ax.set_ylabel("AUROC")
        ax.set_title(feat, loc="left", fontsize=9, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Detection horizon T")
    fig.suptitle(f"Sliding-window AUROC — {group_name}\n"
                 f"(rolling mean of W steps → smoother separation)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig4_window_auroc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Detection-time CDF ────────────────────────────────────────────────

def fig_detection_cdf(ep_data, s_feats, group_name, max_steps=MAX_STEPS,
                      W=5, fpr=FPR_TARGET):
    top_feats = s_feats[:min(6, len(s_feats))]
    successes = ep_data.get("success", []); failures = ep_data.get("failure", [])
    if len(successes) < 10 or len(failures) < 5:
        return

    n_cal = int(len(successes) * 0.7)
    rng = np.random.default_rng(42)
    cal_eps = [successes[i] for i in rng.permutation(len(successes))[:n_cal]]
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(top_feats)))

    fig, (ax_cdf, ax_box) = plt.subplots(1, 2, figsize=(14, 5))
    all_detect = {}

    for feat, col in zip(top_feats, colors):
        S_cal = rolling_mean_matrix(to_matrix(cal_eps, feat, max_steps), W)
        F_mat = rolling_mean_matrix(to_matrix(failures, feat, max_steps), W)
        detect = np.full(len(failures), max_steps + 1, dtype=float)

        for t in range(1, max_steps):
            cv = S_cal[:, t]; cv = cv[~np.isnan(cv)]
            if len(cv) < 5:
                continue
            cm_val = cv.mean()
            thresh = np.quantile(np.abs(cv - cm_val), 1.0 - fpr)
            fv = F_mat[:, t]
            for fi, fs in enumerate(np.abs(fv - cm_val)):
                if not np.isnan(fs) and fs > thresh and detect[fi] > t:
                    detect[fi] = t

        all_detect[feat] = detect
        det = detect[detect <= max_steps]
        xs = np.sort(det); ys = np.arange(1, len(xs) + 1) / len(failures)
        ax_cdf.step(xs, ys, color=col, lw=2.0, where="post",
                    label=f"{feat} ({len(det)}/{len(failures)})")

    ax_cdf.axhline(1.0 - fpr, color="gray", ls=":", lw=1)
    ax_cdf.set_xlabel("Detection step T"); ax_cdf.set_ylabel("Fraction detected")
    ax_cdf.set_ylim(-0.02, 1.05)
    ax_cdf.set_title(f"Cumulative detection rate (W={W}, FPR≤{fpr*100:.0f}%)",
                     fontweight="bold")
    ax_cdf.legend(fontsize=7, loc="lower right")
    ax_cdf.spines[["top", "right"]].set_visible(False)

    data_box = [all_detect[f][all_detect[f] <= max_steps].tolist() for f in top_feats]
    bp = ax_box.boxplot(data_box, vert=True, patch_artist=True,
                        labels=[f[:18] for f in top_feats], showfliers=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax_box.set_ylabel("Detection step"); ax_box.set_title("First-detection step distribution",
                                                           fontweight="bold")
    ax_box.tick_params(axis="x", rotation=25, labelsize=7)
    ax_box.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Early detection analysis — {group_name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig5_detection_cdf.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Composite score (z-score sum) ─────────────────────────────────────

def fig_composite_score(ep_data, s_feats, group_name, max_steps=MAX_STEPS,
                        top_k=6, W=5, fpr=FPR_TARGET):
    top_feats = s_feats[:top_k]
    successes = ep_data.get("success", []); failures = ep_data.get("failure", [])
    if len(successes) < 10 or len(failures) < 5:
        return

    n_cal = int(len(successes) * 0.7)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(successes))
    cal_eps   = [successes[i] for i in perm[:n_cal]]
    probe_eps = [successes[i] for i in perm[n_cal:]] or cal_eps[:max(3, n_cal // 5)]

    S_m = {f: rolling_mean_matrix(to_matrix(cal_eps,   f, max_steps), W) for f in top_feats}
    P_m = {f: rolling_mean_matrix(to_matrix(probe_eps, f, max_steps), W) for f in top_feats}
    F_m = {f: rolling_mean_matrix(to_matrix(failures,  f, max_steps), W) for f in top_feats}

    tpr_comp = np.full(max_steps, np.nan)
    fpr_comp = np.full(max_steps, np.nan)
    best_tpr  = np.full(max_steps, np.nan)

    for t in range(1, max_steps):
        stats = {}
        for f in top_feats:
            cv = S_m[f][:, t]; cv = cv[~np.isnan(cv)]
            if len(cv) < 5:
                break
            stats[f] = (cv.mean(), cv.std() + 1e-9)
        else:
            def comp(mats_d):
                return np.nansum(
                    np.stack([np.abs(mats_d[f][:, t] - stats[f][0]) / stats[f][1]
                              for f in top_feats], axis=1), axis=1)

            cal_c = comp(S_m); probe_c = comp(P_m); fail_c = comp(F_m)
            cv = cal_c[~np.isnan(cal_c)]
            if len(cv) < 5:
                continue
            thresh = np.quantile(cv, 1.0 - fpr)
            n_fa = np.sum(~np.isnan(fail_c))
            if n_fa > 0:
                tpr_comp[t] = float(np.nansum(fail_c > thresh)) / n_fa
            pv = probe_c[~np.isnan(probe_c)]
            if len(pv) > 0:
                fpr_comp[t] = float((pv > thresh).sum()) / len(pv)

            # Best single-feature TPR for comparison
            f_tprs = {}
            for f in top_feats:
                cv_f = S_m[f][:, t]; cv_f = cv_f[~np.isnan(cv_f)]
                if len(cv_f) < 5:
                    continue
                cm_f = cv_f.mean(); cs_f = cv_f.std() + 1e-9
                thr_f = np.quantile(np.abs(cv_f - cm_f) / cs_f, 1.0 - fpr)
                fv = F_m[f][:, t]
                fscore = np.abs(fv - cm_f) / cs_f
                n_fa_f = np.sum(~np.isnan(fscore))
                if n_fa_f > 0:
                    f_tprs[f] = float(np.nansum(fscore > thr_f)) / n_fa_f
            if f_tprs:
                best_tpr[t] = max(f_tprs.values())
            continue

    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(max_steps)
    valid = ~np.isnan(tpr_comp)
    ax.plot(steps[valid], tpr_comp[valid], color=RED_M, lw=2.5,
            label=f"Composite top-{top_k}", zorder=3)
    valid_b = ~np.isnan(best_tpr)
    ax.plot(steps[valid_b], best_tpr[valid_b], color=BLUE_M, lw=2.0, ls="--",
            label="Best single feature", zorder=2)
    valid_f = ~np.isnan(fpr_comp)
    ax.plot(steps[valid_f], fpr_comp[valid_f], color="gray", lw=1.5, ls=":", label="FPR")
    ax.axhline(fpr, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Step T"); ax.set_ylabel("Rate"); ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Composite vs single-feature conformal score — {group_name}\n"
                 f"W={W}, FPR target={fpr*100:.0f}%", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig6_composite.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Distribution snapshots ────────────────────────────────────────────

def fig_distributions(ep_data, top_feats, group_name, max_steps=MAX_STEPS,
                       snapshot_steps=(5, 15, 25, 40)):
    from scipy.stats import gaussian_kde
    feats = top_feats[:min(6, len(top_feats))]
    n_f = len(feats); n_s = len(snapshot_steps)
    if n_f == 0:
        return
    fig, axes = plt.subplots(n_f, n_s, figsize=(n_s * 3.5, n_f * 3.0 + 0.5), sharex="row")
    for fi, feat in enumerate(feats):
        S = to_matrix(ep_data["success"], feat, max_steps)
        F = to_matrix(ep_data["failure"], feat, max_steps)
        all_v = np.concatenate([S.ravel(), F.ravel()])
        all_v = all_v[~np.isnan(all_v)]
        if len(all_v) == 0:
            continue
        xmin, xmax = np.percentile(all_v, 1), np.percentile(all_v, 99)
        if xmin >= xmax:
            xmin -= 0.1; xmax += 0.1
        for si, step in enumerate(snapshot_steps):
            ax = axes[fi][si]
            for vals, col, lab in [(S[:, step], BLUE_M, "success"), (F[:, step], RED_M, "failure")]:
                vals = vals[~np.isnan(vals)]
                if len(vals) < 3:
                    continue
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    xs = np.linspace(xmin, xmax, 200)
                    ax.plot(xs, kde(xs), color=col, lw=1.8, label=lab)
                    ax.fill_between(xs, kde(xs), alpha=0.18, color=col)
                except Exception:
                    ax.hist(vals, bins=15, color=col, alpha=0.4, density=True, label=lab)
            if fi == 0:
                ax.set_title(f"Step {step}", fontweight="bold")
            if si == 0:
                ax.set_ylabel(feat, fontsize=7)
            if fi == 0 and si == n_s - 1:
                ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Feature distributions — {group_name}", fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig7_distributions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"    → {out}")


# ─── Figure: Cross-group comparison ────────────────────────────────────────────

def fig_cross_group(group_data: dict, horizon: int = 20):
    if len(group_data) < 2:
        return
    all_feat_sets = [set(gd[0]) for gd in group_data.values()]
    common = sorted(set.intersection(*all_feat_sets))
    if not common:
        return

    group_names = list(group_data.keys())
    n_groups = len(group_names)

    feat_aurocs = {}
    for g, (feats, mat, mean) in group_data.items():
        feat_idx = {f: i for i, f in enumerate(feats)}
        feat_aurocs[g] = {f: mat[feat_idx[f], horizon] if horizon < mat.shape[1] else np.nan
                         for f in common}

    sort_key = {f: np.nanmean([feat_aurocs[g][f] for g in group_names]) for f in common}
    top_feats = sorted(common, key=lambda f: sort_key[f], reverse=True)[:25]
    n_feats = len(top_feats)

    fig, ax = plt.subplots(figsize=(10, n_feats * 0.45 + 2))
    x = np.arange(n_feats)
    w = 0.75 / n_groups
    colors_g = plt.cm.Set2(np.linspace(0, 1, n_groups))
    for gi, (g, col) in enumerate(zip(group_names, colors_g)):
        vals = [feat_aurocs[g][f] for f in top_feats]
        ax.barh(x + gi * w, vals, w, color=col, alpha=0.85, label=g)
    ax.set_yticks(x + w * (n_groups - 1) / 2)
    ax.set_yticklabels(top_feats, fontsize=7.5)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.axvline(0.7, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC")
    ax.set_title(f"Cross-group AUROC @ step {horizon}", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right"); ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUTPUT_DIR / f"fig_cross_group_h{horizon}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out}")


# ─── Feature groups for trajectory plots ───────────────────────────────────────

def group_features_for_plots(s_feats: List[str]) -> dict:
    def pick(keywords, n=8):
        return [f for f in s_feats if any(k in f for k in keywords)][:n]

    return {
        "routing":    pick(["frac_", "routing", "attn_gini", "attn_entropy"]),
        "gradcam":    pick(["gc_gini", "gc_entropy", "gc_max", "gc_com"]),
        "raw_alpha":  pick(["ras_gini", "rn_gini", "ras_entropy", "rn_entropy"]),
        "agree":      pick(["agree_"]),
        "token":      pick(["_t0", "_std", "horizon_agree"]),
        "value_vec":  pick(["v_norm", "v_gini", "v_coherence", "v_align", "v_drift"]),
        "action":     pick(["action_", "plan_", "gripper", "joint_", "state_norm"]),
    }


# ─── Text report ───────────────────────────────────────────────────────────────

def write_report(group_data: dict, max_steps: int = MAX_STEPS):
    horizons = [5, 10, 20, 30]
    lines = ["=" * 72, "CONFORMAL FEATURE RANKING REPORT (v2)", "=" * 72, ""]
    for group_name, (feats, mat, mean_auroc) in group_data.items():
        lines += [f"GROUP: {group_name}", "-" * 50]
        lines.append(f"{'Feature':<45}  mean  " + "  ".join(f"s{h}" for h in horizons))
        lines.append("-" * 72)
        for i in range(min(40, len(feats))):
            vals = [mat[i, h] if h < mat.shape[1] else np.nan for h in horizons]
            row = (f"{feats[i]:<45}  {mean_auroc[i]:.3f}  " +
                   "  ".join(f"{v:.3f}" if not np.isnan(v) else "  ---" for v in vals))
            lines.append(row)
        lines += ["", ""]
    out = OUTPUT_DIR / "feature_ranking_report_v2.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  → {out}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_group_data = {}

    for group_name, run_names in RUN_GROUPS.items():
        print(f"\n{'='*60}\nGROUP: {group_name}\n{'='*60}")

        ep_data = load_group(group_name, run_names, max_steps=MAX_STEPS)
        n_s = len(ep_data["success"]); n_f = len(ep_data["failure"])
        print(f"  Loaded: {n_s} success, {n_f} failure episodes")
        if n_s < 5 or n_f < 5:
            print("  [skip] not enough episodes"); continue

        print("  AUROC overview …")
        s_feats, auroc_mat, mean_auroc = fig_auroc_overview(ep_data, group_name)
        all_group_data[group_name] = (s_feats, auroc_mat, mean_auroc)
        print(f"  Top-5: {s_feats[:5]}")

        print("  Early detection ranking …")
        fig_early_ranking(s_feats, auroc_mat, group_name)

        print("  Trajectory plots …")
        feat_groups = group_features_for_plots(s_feats)
        for tag, feats_sub in feat_groups.items():
            if feats_sub:
                fig_trajectories(ep_data, feats_sub, group_name, tag=tag)

        print("  Distribution snapshots …")
        fig_distributions(ep_data, s_feats[:6], group_name)

        print("  Sliding-window AUROC …")
        fig_window_auroc(ep_data, s_feats, group_name)

        print("  Detection CDF …")
        fig_detection_cdf(ep_data, s_feats, group_name)

        print("  Composite score …")
        fig_composite_score(ep_data, s_feats, group_name)

    print(f"\n{'='*60}\nCross-group comparisons …")
    for h in [5, 10, 20, 30]:
        fig_cross_group(all_group_data, horizon=h)

    write_report(all_group_data)
    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
