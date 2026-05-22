#!/usr/bin/env python3
"""
conformal_feature_viz.py — Success-vs-failure trajectory visualization.

Two data sources:
  1. V2 cache (pi0fast_liberoplus group, ~1000 episodes) for the v2-ranked features.
     Groups 01-07 are plotted from this cache — no re-extraction needed.
  2. Fresh extraction from DATA_DIR for step_metrics-unique features that were
     not in v2 (raw-weights/V-cosine Ginis, method-pair IoU/CoM/EMD,
     task inter-head agreement). Only N_SAMPLE episodes are used here.

For each feature group, saves one PNG with per-feature subplots:
  - success (blue) vs failure (red)  mean ± 95% CI over steps
  - AUROC@[5,10,20,30] annotated in each panel
  - vertical dashed line at HUMAN_VIS_STEP (human can first see failure)

Usage: run from project root (openpi_liberoplus/)
  <python> scripts/attention_analysis/conformal_feature_viz.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wasserstein_distance
from tqdm import tqdm

from scripts.attention_visualization.grid_visualization_gradcam import to_2d_heatmap
from scripts.attention_visualization.grid_visualization_raw_weights import (
    select_heads_for_span, TOP_K_HEADS,
)
from scripts.attention_visualization.grid_visualization_value_attn import score_v_cosine
from scripts.attention_utils.keys import IMAGE_PATCH_GRID

# ─── Configuration ──────────────────────────────────────────────────────────────
# V2 cache — used for fixed feature groups (instant load, ~1000 episodes)
V2_CACHE_PATH = Path(
    "/nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/"
    "feature_cache/pi0fast_liberoplus.pkl"
)
# Single run for fresh extraction of new step_metrics features
DATA_DIR      = Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260516_034939")
EPISODE_JSON  = DATA_DIR / "client_output" / "episode_summaries.json"
OUTPUT_DIR    = Path("/nfs/roberts/scratch/pi_tkf6/zs377/conformal_feature_viz")
SUPP_CACHE    = OUTPUT_DIR / "supp_feature_cache.pkl"   # cache for new features only

N_SAMPLE      = 75   # success + failure episodes to use for new feature extraction
MAX_STEPS     = 103
PLOT_MAX_STEPS= 60
HUMAN_VIS_STEP= 25
AUROC_STEPS   = [5, 10, 20, 30]
AUROC_MIN_N   = 5
ACTION_TOKEN  = 0

PATCH_H, PATCH_W = IMAGE_PATCH_GRID  # (16, 16)
TOPK_PATCHES     = 26  # ~top 10% of 256

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"

RANK_THRESHOLD = 0.60   # min max-AUROC to appear in auto-ranked plot

# Features in v2 cache that are noise (right-wrist slot = repeated image)
V2_NOISE_PREFIX = ("frac_rwrist", "attn_gini_rwrist", "gc_gini_rwrist",
                   "ras_gini_rwrist", "rn_gini_rwrist", "gc_entropy_rwrist",
                   "ras_entropy_rwrist", "rn_entropy_rwrist",
                   "gc_max_rwrist", "gc_mean_rwrist", "gc_com_y_rwrist",
                   "gc_com_x_rwrist", "agree_gc_ras_rwrist", "agree_gc_rn_rwrist",
                   "agree_ras_rn_rwrist", "agree_attn_gc_rwrist", "agree_attn_ras_rwrist",
                   "agree_base_rwrist", "agree_wrist_rwrist",
                   "gc_gini_rwrist_t0", "ras_gini_rwrist_t0", "rn_gini_rwrist_t0",
                   "gc_gini_rwrist_std", "horizon_agree_gc_rwrist",
                   "horizon_agree_ras_rwrist")

# ─── Plot style ─────────────────────────────────────────────────────────────────
BLUE_M, BLUE_L = "#1f5fa6", "#a8c8e8"
RED_M,  RED_L  = "#b02020", "#f0a0a0"
LW_M = 2.2
FIG_W, ROW_H = 14, 2.6

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "legend.fontsize": 8, "legend.framealpha": 0.7,
})

# ─── Feature groups (sourced from v2 cache) ─────────────────────────────────────
# Each entry: (feature_key, y_axis_label, panel_title)
FEATURE_GROUPS: Dict[str, List[Tuple[str, str, str]]] = {
    "01_proprioceptive": [
        ("state_norm", "‖s‖₂",  "State norm  ‖s‖₂"),
        ("joint_vel",  "‖Δs‖₂", "Joint velocity  ‖state_t − state_{t−1}‖₂"),
    ],
    "02_action_planning": [
        ("action_magnitude",     "‖a₀‖₂",   "Action magnitude  (immediate step norm)"),
        ("action_plan_norm",     "mean norm","Plan norm  (mean ‖aₜ‖₂ over 10 steps)"),
        ("action_plan_entropy",  "entropy",  "Plan entropy  (entropy of 10-step norm dist.)"),
        ("action_horizon_align", "cosine",   "Horizon alignment  cos(a₀, a₉)"),
        ("action_spread",        "std",      "Plan spread  (std of 10-step norms)"),
        ("action_dir_var",       "var",      "Direction variance  (mean var across dims 0-5)"),
        ("action_rot_flip",      "flip rate","Rotation sign-flip rate within horizon"),
        ("plan_consistency",     "cosine",   "Plan consistency  cos(a₀_t , a₀_{t−1})"),
        ("gripper_val",          "gripper",  "Gripper value  (dim 6 of a₀)"),
    ],
    "03_value_vectors": [
        ("v_align_base_wrist",  "cosine",     "V align: base ↔ wrist"),
        ("v_align_base_task",   "cosine",     "V align: base ↔ task"),
        ("v_align_task_wrist",  "cosine",     "V align: task ↔ wrist"),
        ("v_coherence_wrist",   "mean cosine","V coherence: wrist cam"),
        ("v_coherence_base",    "mean cosine","V coherence: base cam"),
        ("v_coherence_task",    "mean cosine","V coherence: task tokens"),
        ("v_norm_gini_wrist",   "Gini",       "V norm Gini: wrist cam"),
        ("v_norm_gini_task",    "Gini",       "V norm Gini: task tokens"),
        ("v_norm_mean_wrist",   "mean",       "V norm mean: wrist cam"),
        ("v_drift_base",        "1−cosine",   "V drift: base cam  (centroid Δ from prev step)"),
        ("v_drift_wrist",       "1−cosine",   "V drift: wrist cam"),
    ],
    "04_attention_routing": [
        ("frac_base",        "fraction", "Attention fraction: base camera"),
        ("frac_wrist",       "fraction", "Attention fraction: wrist camera"),
        ("frac_task",        "fraction", "Attention fraction: task tokens"),
        ("frac_state",       "fraction", "Attention fraction: state tokens"),
        ("routing_entropy",  "entropy",  "Routing entropy  (over modality fracs)"),
        ("attn_gini_base",   "Gini",     "Attention Gini: base camera  (mean-attn)"),
        ("attn_entropy_base","entropy",  "Attention entropy: base camera"),
        ("attn_gini_task",   "Gini",     "Attention Gini: task tokens"),
        ("attn_gini_wrist",  "Gini",     "Attention Gini: wrist camera"),
    ],
    "05_heatmap_concentration": [
        ("gc_gini_base",   "Gini", "GradCAM Gini: base cam  (mean over 5 tokens)"),
        ("ras_gini_base",  "Gini", "Raw-α sum Gini: base cam"),
        ("rn_gini_base",   "Gini", "Raw-α norm Gini: base cam  ← ranked high in v2"),
        ("gc_gini_wrist",  "Gini", "GradCAM Gini: wrist cam"),
        ("ras_gini_wrist", "Gini", "Raw-α sum Gini: wrist cam"),
        ("rn_gini_wrist",  "Gini", "Raw-α norm Gini: wrist cam"),
        ("gc_gini_task",   "Gini", "GradCAM Gini: task tokens"),
        ("ras_gini_task",  "Gini", "Raw-α sum Gini: task tokens"),
        ("gc_com_y_base",  "row",  "GradCAM centre-of-mass Y: base cam"),
        ("gc_com_drift_base","patches","GradCAM CoM drift: base cam  (Δ from prev step)"),
    ],
    "06_cross_signal_agreement": [
        ("agree_gc_ras_base",   "cosine", "GradCAM ↔ raw-α sum: base cam"),
        ("agree_gc_rn_base",    "cosine", "GradCAM ↔ raw-α norm: base cam"),
        ("agree_attn_gc_base",  "cosine", "Attn-weights ↔ GradCAM: base cam"),
        ("agree_gc_ras_wrist",  "cosine", "GradCAM ↔ raw-α sum: wrist cam"),
        ("agree_attn_gc_wrist", "cosine", "Attn-weights ↔ GradCAM: wrist cam"),
        ("agree_base_wrist_gc", "cosine", "Base ↔ wrist GradCAM cosine"),
        ("agree_base_wrist_rn", "cosine", "Base ↔ wrist raw-α norm cosine"),
        ("agree_gc_ras_task",   "cosine", "GradCAM ↔ raw-α sum: task tokens"),
        ("agree_attn_gc_task",  "cosine", "Attn-weights ↔ GradCAM: task tokens"),
    ],
    "07_per_token": [
        ("rn_gini_base_t0",       "Gini",   "Raw-α norm Gini: base cam, token 0"),
        ("gc_gini_base_t0",       "Gini",   "GradCAM Gini: base cam, token 0"),
        ("ras_gini_base_t0",      "Gini",   "Raw-α sum Gini: base cam, token 0"),
        ("gc_gini_base_std",      "std",    "GradCAM Gini std across 5 tokens  (plan instability)"),
        ("horizon_agree_gc_base", "cosine", "GradCAM horizon: cos(token 0, token 4)  base cam"),
        ("horizon_agree_ras_base","cosine", "Raw-α horizon: cos(token 0, token 4)  base cam"),
        ("rn_gini_wrist_t0",      "Gini",   "Raw-α norm Gini: wrist cam, token 0"),
        ("horizon_agree_gc_wrist","cosine", "GradCAM horizon: cos(token 0, token 4)  wrist cam"),
    ],
    # ── Cross-format safe features ──────────────────────────────────────────────
    # Features whose value DISTRIBUTIONS are compatible across pi0fast AND pi05.
    #
    # Excluded from this group (overlap with note):
    #   - joint_vel: same 8-dim ||Δstate||₂ for both, but pi05 (91% success) moves
    #     10× more slowly than pi0fast (57% success) → calibration doesn't transfer
    #   - state_norm: computable but pi05 much tighter distribution (std 0.21 vs 0.92)
    #   - task features: 200 vs 17 task tokens → incompatible Gini/frac ranges
    #   - per-token features: NaN for pi05 (single GradCAM per camera, not per-token)
    #   - state-span features: NaN for these pi05 runs (state not yet tokenized in
    #     transformer; fix is on the pi05 branch, not merged at recording time)
    "00_crossformat_safe": [
        ("v_align_base_wrist",  "cosine",     "V align: base↔wrist  [0.926 pi0fast ↔ 0.940 pi05 — nearly identical]"),
        ("v_drift_wrist",       "1−cosine",   "V drift: wrist cam  [0.016 pi0fast ↔ 0.017 pi05 — best cross-format]"),
        ("v_coherence_wrist",   "mean cosine","V coherence: wrist cam"),
        ("v_coherence_base",    "mean cosine","V coherence: base cam"),
        ("v_norm_gini_wrist",   "Gini",       "V norm Gini: wrist cam"),
        ("v_norm_gini_base",    "Gini",       "V norm Gini: base cam"),
        ("rn_gini_base",        "Gini",       "Raw-α norm Gini: base cam  [0.680 pi0fast ↔ 0.735 pi05]"),
        ("gc_gini_base",        "Gini",       "GradCAM Gini: base cam  [0.260 pi0fast ↔ 0.315 pi05]"),
        ("ras_gini_base",       "Gini",       "Raw-α sum Gini: base cam"),
        ("rn_gini_wrist",       "Gini",       "Raw-α norm Gini: wrist cam"),
        ("gc_gini_wrist",       "Gini",       "GradCAM Gini: wrist cam"),
        ("agree_gc_ras_base",   "cosine",     "GradCAM↔raw-α sum: base cam"),
        ("agree_gc_rn_base",    "cosine",     "GradCAM↔raw-α norm: base cam"),
        ("agree_base_wrist_gc", "cosine",     "Base↔wrist GradCAM cosine"),
        ("agree_base_wrist_rn", "cosine",     "Base↔wrist raw-α norm cosine"),
        ("action_plan_entropy", "entropy",    "Plan entropy  (scale-invariant; works across policies)"),
        ("action_horizon_align","cosine",     "Horizon alignment  cos(a₀, a₉)"),
        ("action_rot_flip",     "flip rate",  "Rotation sign-flip rate within horizon"),
        ("plan_consistency",    "cosine",     "Plan consistency  cos(a₀_t, a₀_{t−1})"),
        ("action_spread",       "std",        "Plan spread  std(‖aₜ‖)"),
    ],
}

# ─── Scalar utilities ────────────────────────────────────────────────────────────

def _gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s < 1e-12 or n == 0: return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def _topk_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    top_a = set(np.argsort(a)[-TOPK_PATCHES:].tolist())
    top_b = set(np.argsort(b)[-TOPK_PATCHES:].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def _com_distance(a: np.ndarray, b: np.ndarray) -> float:
    def _com(h):
        h = np.abs(h.reshape(PATCH_H, PATCH_W)).astype(np.float64)
        s = h.sum() + 1e-12
        rows = np.arange(PATCH_H)[:, None]; cols = np.arange(PATCH_W)[None, :]
        return float((h * rows).sum() / s), float((h * cols).sum() / s)
    cy_a, cx_a = _com(a); cy_b, cx_b = _com(b)
    return float(np.sqrt((cy_a - cy_b)**2 + (cx_a - cx_b)**2))


def _emd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a.ravel()).astype(np.float64); b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12: return 0.0
    a /= sa; b /= sb
    a2 = a.reshape(PATCH_H, PATCH_W); b2 = b.reshape(PATCH_H, PATCH_W)
    xs = np.arange(PATCH_W); ys = np.arange(PATCH_H)
    ex = wasserstein_distance(xs, xs, a2.sum(0), b2.sum(0))
    ey = wasserstein_distance(ys, ys, a2.sum(1), b2.sum(1))
    return float(0.5 * (ex + ey))


def _nan() -> float: return float("nan")

def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    v = rec.get(key)
    if v is None: return None
    v = np.asarray(v).ravel()
    return int(v[0]), int(v[1])


# ─── Supplemental feature extractor (NEW features only) ─────────────────────────

def extract_new_features(rec: dict) -> dict:
    """Extract only the step_metrics-unique features not present in v2 cache."""
    feats: dict = {}
    if FULL_ATTN_KEY not in rec or FULL_V_KEY not in rec:
        return feats

    attn_tgt = np.asarray(rec[FULL_ATTN_KEY])[-1, 0]   # (H, S)
    v_tgt    = np.asarray(rec[FULL_V_KEY])[-1, 0]       # (S, 1, D)

    sp_base  = _span(rec, "outputs/debug/spans/image/base_0_rgb")
    sp_wrist = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
    sp_task  = _span(rec, "outputs/debug/spans/task")
    sp_state = _span(rec, "outputs/debug/spans/state")
    all_spans = [s for s in [sp_base, sp_wrist, sp_task, sp_state] if s is not None]

    base_heads  = select_heads_for_span(attn_tgt, sp_base,  all_spans, TOP_K_HEADS) if sp_base  else None
    wrist_heads = select_heads_for_span(attn_tgt, sp_wrist, all_spans, TOP_K_HEADS) if sp_wrist else None
    task_heads  = select_heads_for_span(attn_tgt, sp_task,  all_spans, TOP_K_HEADS) if sp_task  else None
    state_heads = select_heads_for_span(attn_tgt, sp_state, all_spans, TOP_K_HEADS) if sp_state else None

    # Raw-weights Gini per modality (head-selected max attn)
    for heads, sp, label in [(base_heads, sp_base, "base"), (wrist_heads, sp_wrist, "wrist"),
                              (task_heads, sp_task, "task"), (state_heads, sp_state, "state")]:
        if heads is not None and sp is not None:
            feats[f"gini_{label}_rw"] = _gini(attn_tgt[heads, sp[0]:sp[1]].max(axis=0))
        else:
            feats[f"gini_{label}_rw"] = _nan()

    # V-cosine Gini per modality (value-contribution saliency)
    for heads, sp, label in [(base_heads, sp_base, "base"), (wrist_heads, sp_wrist, "wrist"),
                              (task_heads, sp_task, "task"), (state_heads, sp_state, "state")]:
        if heads is not None and sp is not None:
            feats[f"gini_{label}_vc"] = _gini(score_v_cosine(attn_tgt, v_tgt, heads)[sp[0]:sp[1]])
        else:
            feats[f"gini_{label}_vc"] = _nan()

    # Task inter-head agreement (mean pairwise cosine of task-focused heads)
    if task_heads is not None and sp_task is not None and len(task_heads) > 1:
        h = attn_tgt[task_heads][:, sp_task[0]:sp_task[1]].astype(np.float64)
        h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        sim = h_n @ h_n.T
        iu  = np.triu_indices(len(task_heads), k=1)
        feats["task_head_agr"] = float(sim[iu].mean())
    else:
        feats["task_head_agr"] = _nan()

    # Method-pair comparisons for base and wrist cameras
    for cam, label, sp, heads in [
        ("base_0_rgb",       "base",  sp_base,  base_heads),
        ("left_wrist_0_rgb", "wrist", sp_wrist, wrist_heads),
    ]:
        if sp is None or heads is None:
            for pair in ("gc_ra", "gc_rn", "gc_rw", "gc_vc"):
                for m in ("cosine", "iou", "com_dist", "emd"):
                    feats[f"{label}_{pair}_{m}"] = _nan()
            continue

        # Load GradCAM and raw_alpha for method-pair comparison
        kg = f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/{cam}"
        kr = f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/{cam}"
        kn = f"outputs/debug/raw_alpha/norm/action_tokens/{ACTION_TOKEN}/{cam}"
        if kg not in rec:
            for pair in ("gc_ra","gc_rn","gc_rw","gc_vc"):
                for m in ("cosine","iou","com_dist","emd"):
                    feats[f"{label}_{pair}_{m}"] = _nan()
            continue

        gc_2d  = np.abs(to_2d_heatmap(np.asarray(rec[kg]),  cam)).astype(np.float64)
        ras_2d = np.abs(to_2d_heatmap(np.asarray(rec[kr]),  cam)).astype(np.float64)
        rn_2d  = np.abs(to_2d_heatmap(np.asarray(rec[kn]),  cam)).astype(np.float64) if kn in rec else ras_2d
        rw_2d  = attn_tgt[heads, sp[0]:sp[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
        vc_2d  = score_v_cosine(attn_tgt, v_tgt, heads)[sp[0]:sp[1]].reshape(PATCH_H, PATCH_W)

        for pair_name, h2 in [("gc_ra", ras_2d), ("gc_rn", rn_2d),
                                ("gc_rw", rw_2d), ("gc_vc", vc_2d)]:
            feats[f"{label}_{pair_name}_cosine"]   = _cosine(gc_2d, h2)
            feats[f"{label}_{pair_name}_iou"]      = _topk_iou(gc_2d, h2)
            feats[f"{label}_{pair_name}_com_dist"] = _com_distance(gc_2d, h2)
            feats[f"{label}_{pair_name}_emd"]      = _emd(gc_2d, h2)

    # Task/state token method-pair cosines (gc vs ra/rw/vc)
    for modality, sp_m, heads_m in [("task", sp_task, task_heads),
                                     ("state", sp_state, state_heads)]:
        km_gc = f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/{modality}"
        km_ra = f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/{modality}"
        if km_gc not in rec or heads_m is None or sp_m is None:
            for sfx in ("gc_ra_cosine","gc_rw_cosine","gc_vc_cosine","ra_rw_cosine"):
                feats[f"{modality}_{sfx}"] = _nan()
            continue
        gc_tok  = np.abs(np.asarray(rec[km_gc])[0]).astype(np.float64)
        ras_tok = np.abs(np.asarray(rec[km_ra])[0]).astype(np.float64)
        rw_tok  = attn_tgt[heads_m, sp_m[0]:sp_m[1]].max(axis=0).astype(np.float64)
        vc_tok  = score_v_cosine(attn_tgt, v_tgt, heads_m)[sp_m[0]:sp_m[1]].astype(np.float64)
        feats[f"{modality}_gc_ra_cosine"]  = _cosine(gc_tok, ras_tok)
        feats[f"{modality}_gc_rw_cosine"]  = _cosine(gc_tok, rw_tok)
        feats[f"{modality}_gc_vc_cosine"]  = _cosine(gc_tok, vc_tok)
        feats[f"{modality}_ra_rw_cosine"]  = _cosine(ras_tok, rw_tok)

    return feats


# ─── Data loading ────────────────────────────────────────────────────────────────

def load_v2_cache() -> Dict[str, List[List[dict]]]:
    """Load the existing v2 feature cache for fixed groups (fast)."""
    print(f"Loading v2 cache: {V2_CACHE_PATH}")
    with open(V2_CACHE_PATH, "rb") as f:
        raw = pickle.load(f)
    # Strip noise features (rwrist = repeated image, not real camera)
    ep_data: Dict[str, List[List[dict]]] = {}
    for label, eps in raw.items():
        cleaned = []
        for ep in eps:
            cleaned.append([
                {k: v for k, v in step.items()
                 if not any(k.startswith(pfx) for pfx in V2_NOISE_PREFIX)}
                for step in ep
            ])
        ep_data[label] = cleaned
    ns = len(ep_data.get("success", [])); nf = len(ep_data.get("failure", []))
    print(f"  {ns} success  {nf} failure  (v2 cache)")
    return ep_data


def load_supplemental(use_cache: bool = True) -> Dict[str, List[List[dict]]]:
    """Extract new step_metrics features from N_SAMPLE episodes in DATA_DIR."""
    if use_cache and SUPP_CACHE.exists():
        try:
            print(f"Loading supplemental cache: {SUPP_CACHE}")
            with open(SUPP_CACHE, "rb") as f:
                d = pickle.load(f)
            ns = len(d.get("success",[])); nf = len(d.get("failure",[]))
            print(f"  {ns} success  {nf} failure")
            return d
        except Exception as e:
            print(f"  [warn] supp cache broken ({e}), re-extracting …")

    with open(EPISODE_JSON) as f:
        all_eps = json.load(f)

    succ_eps = [ep for ep in all_eps if ep["success"]][:N_SAMPLE]
    fail_eps = [ep for ep in all_eps if not ep["success"]][:N_SAMPLE]
    print(f"Extracting new features: {len(succ_eps)} success + {len(fail_eps)} failure episodes …")

    ep_data: Dict[str, List[List[dict]]] = {"success": [], "failure": []}
    for label, eps_list in [("success", succ_eps), ("failure", fail_eps)]:
        for ep in tqdm(eps_list, desc=f"  {label}", ncols=80):
            steps = []
            for rel in range(MAX_STEPS):
                s = ep["start_idx"] + rel
                if s > ep["end_idx"]:
                    break
                fpath = DATA_DIR / f"step_{s}.npy"
                if not fpath.exists():
                    continue
                try:
                    rec   = np.load(fpath, allow_pickle=True).item()
                    feats = extract_new_features(rec)
                    feats["rel_step"] = rel
                    steps.append(feats)
                except Exception as exc:
                    print(f"    [warn] step {s}: {exc}")
            if steps:
                ep_data[label].append(steps)

    ns = len(ep_data["success"]); nf = len(ep_data["failure"])
    print(f"  {ns} success  {nf} failure  extracted.")
    SUPP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=SUPP_CACHE.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(ep_data, f, protocol=4)
    os.replace(tmp, SUPP_CACHE)
    return ep_data


# ─── AUROC computation ──────────────────────────────────────────────────────────

def to_matrix(ep_list: List[List[dict]], key: str) -> np.ndarray:
    out = np.full((len(ep_list), PLOT_MAX_STEPS), np.nan)
    for i, ep in enumerate(ep_list):
        for rec in ep:
            t = rec.get("rel_step", 0)
            if t < PLOT_MAX_STEPS:
                out[i, t] = rec.get(key, np.nan)
    return out


def auroc_at(s_mat: np.ndarray, f_mat: np.ndarray, step: int) -> float:
    sv = s_mat[:, step]; fv = f_mat[:, step]
    sv = sv[~np.isnan(sv)]; fv = fv[~np.isnan(fv)]
    if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
        return float("nan")
    stat, _ = mannwhitneyu(fv, sv, alternative="two-sided")
    a = float(stat) / (len(sv) * len(fv))
    return max(a, 1.0 - a)


def compute_aurocs(ep_data: Dict, key: str) -> Dict[int, float]:
    s_mat = to_matrix(ep_data["success"], key)
    f_mat = to_matrix(ep_data["failure"], key)
    return {step: auroc_at(s_mat, f_mat, step) for step in AUROC_STEPS}


# ─── Plotting ───────────────────────────────────────────────────────────────────

def plot_row(ax, ep_data: Dict, key: str, ylabel: str, title: str):
    s_mat = to_matrix(ep_data["success"], key)
    f_mat = to_matrix(ep_data["failure"], key)
    xs    = np.arange(PLOT_MAX_STEPS)
    for mat, cm, cl, lbl in [(s_mat, BLUE_M, BLUE_L, "success"),
                              (f_mat, RED_M,  RED_L,  "failure")]:
        mean  = np.nanmean(mat, axis=0)
        n     = np.sum(~np.isnan(mat), axis=0).astype(float)
        ci    = 1.96 * np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n, 1))
        valid = ~np.isnan(mean)
        ax.fill_between(xs[valid], (mean - ci)[valid], (mean + ci)[valid],
                        color=cl, alpha=0.3, zorder=2)
        ax.plot(xs[valid], mean[valid], color=cm, lw=LW_M, label=lbl, zorder=3)
    ax.axvline(HUMAN_VIS_STEP, color="#666", lw=1.0, ls=":", alpha=0.7,
               label=f"human vis. ~step {HUMAN_VIS_STEP}")

    auc_vals = {step: auroc_at(s_mat, f_mat, step) for step in AUROC_STEPS}
    auc_str  = "  ".join(
        f"@{s}:{v:.2f}" if not np.isnan(v) else f"@{s}:—"
        for s, v in auc_vals.items()
    )
    ax.text(0.99, 0.97, f"AUROC  {auc_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, loc="left", pad=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7, ncol=3)


def has_data(ep_data: Dict, key: str) -> bool:
    return any(
        not np.isnan(rec.get(key, float("nan")))
        for eps in ep_data.values() for ep in eps for rec in ep
    )


def save_group_fig(ep_data: Dict, group_name: str,
                   specs: List[Tuple[str, str, str]]):
    valid = [(k, yl, tt) for k, yl, tt in specs if has_data(ep_data, k)]
    if not valid:
        print(f"  [skip] {group_name}: all features are NaN")
        return
    n    = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(FIG_W, ROW_H * n), sharex=True)
    if n == 1:
        axes = [axes]
    label = group_name.replace("_", " ").title()
    ns = len(ep_data.get("success", [])); nf = len(ep_data.get("failure", []))
    fig.suptitle(f"{label}  (N={ns}S + {nf}F)", fontsize=12,
                 fontweight="bold", y=1.005)
    for ax, (key, yl, title) in zip(axes, valid):
        plot_row(ax, ep_data, key, yl, title)
    axes[-1].set_xlabel("Step within episode")
    fig.tight_layout()
    out = OUTPUT_DIR / f"fig_{group_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─── Step-metrics-unique feature ranking ────────────────────────────────────────

STEP_METRICS_UNIQUE: List[str] = [
    "gini_base_rw",  "gini_wrist_rw",  "gini_task_rw",  "gini_state_rw",
    "gini_base_vc",  "gini_wrist_vc",  "gini_task_vc",  "gini_state_vc",
    "task_head_agr",
    "base_gc_ra_cosine", "base_gc_ra_iou", "base_gc_ra_com_dist", "base_gc_ra_emd",
    "base_gc_rn_cosine", "base_gc_rn_iou", "base_gc_rn_com_dist", "base_gc_rn_emd",
    "base_gc_rw_cosine", "base_gc_rw_iou", "base_gc_rw_com_dist", "base_gc_rw_emd",
    "base_gc_vc_cosine", "base_gc_vc_iou", "base_gc_vc_com_dist", "base_gc_vc_emd",
    "wrist_gc_ra_cosine","wrist_gc_ra_iou","wrist_gc_ra_com_dist","wrist_gc_ra_emd",
    "wrist_gc_rw_cosine","wrist_gc_rw_iou","wrist_gc_rw_com_dist","wrist_gc_rw_emd",
    "wrist_gc_vc_cosine","wrist_gc_vc_iou","wrist_gc_vc_com_dist","wrist_gc_vc_emd",
    "task_gc_ra_cosine", "task_gc_rw_cosine",  "task_gc_vc_cosine",  "task_ra_rw_cosine",
    "state_gc_ra_cosine","state_gc_rw_cosine",  "state_gc_vc_cosine",
]


def rank_and_plot_new_features(supp_data: Dict):
    ranked = []
    for key in STEP_METRICS_UNIQUE:
        auc_by_step = compute_aurocs(supp_data, key)
        valid = {s: v for s, v in auc_by_step.items() if not np.isnan(v)}
        if not valid:
            continue
        max_auc = max(valid.values())
        ranked.append((key, max_auc, auc_by_step))
    ranked.sort(key=lambda x: -x[1])

    report = OUTPUT_DIR / "step_metrics_unique_ranking.txt"
    with open(report, "w") as rpt:
        rpt.write("Step-metrics-unique feature AUROC ranking\n")
        rpt.write(f"  (from {len(supp_data.get('success',[]))}S + "
                  f"{len(supp_data.get('failure',[]))}F episodes in DATA_DIR)\n\n")
        rpt.write(f"{'Feature':<40}  max  "
                  + "  ".join(f"@{s}" for s in AUROC_STEPS) + "\n")
        rpt.write("-" * 70 + "\n")
        for key, max_auc, auc_dict in ranked:
            vals = "  ".join(
                f"{auc_dict.get(s, float('nan')):.3f}"
                if not np.isnan(auc_dict.get(s, float("nan"))) else "  — "
                for s in AUROC_STEPS
            )
            rpt.write(f"{key:<40}  {max_auc:.3f}  {vals}\n")
    print(f"\n  Step-metrics ranking → {report}")
    print(f"  Top-10:")
    for key, mx, _ in ranked[:10]:
        print(f"    {key:<40}  {mx:.3f}")

    discriminative = [(k, mx, auc) for k, mx, auc in ranked if mx >= RANK_THRESHOLD]
    if not discriminative:
        print(f"\n  No new features above threshold {RANK_THRESHOLD}.")
        return
    print(f"\n  {len(discriminative)} features above threshold {RANK_THRESHOLD} → plotting.")
    specs = [
        (k, "value", f"{k}   (max AUROC {mx:.3f}   @{AUROC_STEPS})")
        for k, mx, _ in discriminative
    ]
    save_group_fig(supp_data, "08_step_metrics_unique", specs)


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}\n")

    # ── Fixed groups from v2 cache ────────────────────────────────────────────
    v2_data = load_v2_cache()
    print("\n=== Plotting fixed feature groups (v2 cache) ===")
    for group_name, specs in FEATURE_GROUPS.items():
        print(f"  {group_name} …")
        save_group_fig(v2_data, group_name, specs)

    # ── New step_metrics features via fresh extraction ────────────────────────
    print(f"\n=== Extracting new step_metrics features ({N_SAMPLE}S + {N_SAMPLE}F eps) ===")
    supp_data = load_supplemental(use_cache=True)
    rank_and_plot_new_features(supp_data)

    print("\nDone. All figures in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
