#!/usr/bin/env python3
"""
Fresh-perspective analysis for conformal prediction of policy failures.

Unlike step_metrics_plot.py (single-run, visual inspection), this script:
  - Pools all runs by (model, task_set) group
  - Computes AUROC(step) and Cohen's d(step) for every feature — directly
    answering "which signal first diverges between success and failure, and when"
  - Simulates online conformal prediction: calibrate on success episodes,
    flag an episode as likely failure at the earliest step where its
    nonconformity score exceeds the (1-α) quantile of the calibration set
  - Produces a cross-model comparison for the shared libero-10 benchmark

Works transparently with both recording formats:
  - pi05:    gradcam/image/{cam}, raw_alpha/{summation,norm}/image/{cam}
             attn/weights shape (L, H, A, S)
  - pi0fast: gradcam/action_tokens/{i}/{cam}, same for raw_alpha
             attn/weights shape (L, 1, H, S)  or (L, 1, H, S)

Output: /nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal/
"""
from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
OUTPUT_DIR = BASE / "analysis_conformal"

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

MAX_STEPS = 60        # analyse first 60 steps (success episodes end ~mean 50)
CACHE_DIR  = OUTPUT_DIR / "feature_cache"
FPR_TARGET = 0.05     # target FPR for conformal prediction simulation
AUROC_MIN_N = 5       # minimum class size to compute AUROC at a given step

# Plot style
BLUE_M, BLUE_L = "#1f5fa6", "#a8c8e8"
RED_M,  RED_L  = "#b02020", "#f0a0a0"
LW_M, LW_I, AL = 2.2, 0.8, 0.35

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "legend.fontsize": 8, "legend.framealpha": 0.7,
})

# ─────────────────────────────────────────────────────────────────────────────
# SCALAR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s < 1e-12 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def spatial_com(h: np.ndarray) -> Tuple[float, float]:
    """Centre-of-mass (row, col) of a 2-D map. Returns (0,0) for zero maps."""
    h = np.abs(h).astype(np.float64)
    s = h.sum()
    if s < 1e-12:
        return 0.0, 0.0
    rows = np.arange(h.shape[0])[:, None]
    cols = np.arange(h.shape[1])[None, :]
    return float((h * rows).sum() / s), float((h * cols).sum() / s)


def entropy(p: np.ndarray) -> float:
    p = np.abs(p.ravel()).astype(np.float64)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(rec: dict) -> str:
    return "pi0fast" if "outputs/debug/attn/layers" in rec else "pi05"


def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    if key not in rec:
        return None
    v = np.asarray(rec[key]).ravel()
    return int(v[0]), int(v[1])


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (per step, format-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _get_heatmaps_pi0fast(rec: dict, cam: str, n_action_tokens: int = 5):
    """Return (gc, ra_sum, ra_norm) aggregated over action tokens 0..n-1."""
    gcs, ras, rns = [], [], []
    for i in range(n_action_tokens):
        k_gc  = f"outputs/debug/gradcam/action_tokens/{i}/{cam}"
        k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{cam}"
        k_rn  = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/{cam}"
        if k_gc not in rec:
            break
        gcs.append(np.asarray(rec[k_gc])[0])
        ras.append(np.asarray(rec[k_ras])[0])
        rns.append(np.asarray(rec[k_rn])[0])
    if not gcs:
        return None, None, None
    gc  = np.mean(np.stack([np.abs(x) for x in gcs], axis=0), axis=0)
    ras = np.mean(np.stack([np.abs(x) for x in ras], axis=0), axis=0)
    rns = np.mean(np.stack([np.abs(x) for x in rns], axis=0), axis=0)
    return gc, ras, rns


def _get_heatmaps_pi05(rec: dict, cam: str):
    k_gc  = f"outputs/debug/gradcam/image/{cam}"
    k_ras = f"outputs/debug/raw_alpha/summation/image/{cam}"
    k_rn  = f"outputs/debug/raw_alpha/norm/image/{cam}"
    gc  = np.abs(np.asarray(rec[k_gc]))   if k_gc  in rec else None
    ras = np.abs(np.asarray(rec[k_ras]))  if k_ras in rec else None
    rn  = np.abs(np.asarray(rec[k_rn]))   if k_rn  in rec else None
    return gc, ras, rn


def _get_token_scores_pi0fast(rec: dict, modality: str, n_action_tokens: int = 5):
    """Return (gc, ra_sum) for task or state tokens."""
    gcs, ras = [], []
    for i in range(n_action_tokens):
        k_gc  = f"outputs/debug/gradcam/action_tokens/{i}/{modality}"
        k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{modality}"
        if k_gc not in rec:
            break
        gcs.append(np.asarray(rec[k_gc])[0])
        ras.append(np.asarray(rec[k_ras])[0])
    if not gcs:
        return None, None
    return (np.mean(np.stack([np.abs(x) for x in gcs], axis=0), axis=0),
            np.mean(np.stack([np.abs(x) for x in ras], axis=0), axis=0))


def _get_token_scores_pi05(rec: dict, modality: str):
    k_gc  = f"outputs/debug/gradcam/{modality}"
    k_ras = f"outputs/debug/raw_alpha/summation/{modality}"
    gc  = np.abs(np.asarray(rec[k_gc]))  if k_gc  in rec else None
    ras = np.abs(np.asarray(rec[k_ras])) if k_ras in rec else None
    return gc, ras


def extract_features(rec: dict, prev: Optional[dict] = None) -> dict:
    """
    Extract scalar features from one step record.
    prev: the previous step's record (for temporal diff features), or None.
    Returns dict of float values (NaN for unavailable features).
    """
    fmt = detect_format(rec)
    feats: dict = {}
    nan = float("nan")

    # ── Modality attention fractions from full attn/weights ───────────────────
    weights_key = "outputs/debug/attn/weights"
    if weights_key in rec:
        w = np.asarray(rec[weights_key])
        if fmt == "pi0fast":
            # (L, 1, H, S) → last layer, squeeze batch, mean over heads
            mean_attn = w[-1, 0].mean(axis=0)            # (S,)
        else:
            # pi05: (L, H, A, S) → last layer, mean over H and A
            mean_attn = w[-1].mean(axis=0).mean(axis=0)  # (S,)

        total = mean_attn.sum() + 1e-12
        sp_base  = _span(rec, "outputs/debug/spans/image/base_0_rgb")
        sp_wrist = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
        sp_rwrist= _span(rec, "outputs/debug/spans/image/right_wrist_0_rgb")
        sp_task  = _span(rec, "outputs/debug/spans/task")
        sp_state = _span(rec, "outputs/debug/spans/state")

        def frac(sp):
            return float(mean_attn[sp[0]:sp[1]].sum() / total) if sp else nan

        feats["frac_base"]   = frac(sp_base)
        feats["frac_wrist"]  = frac(sp_wrist)
        feats["frac_rwrist"] = frac(sp_rwrist)
        feats["frac_task"]   = frac(sp_task)
        feats["frac_state"]  = frac(sp_state)

        # Routing entropy (over the 3-4 main modalities)
        fracs = [v for v in [feats["frac_base"], feats["frac_wrist"],
                              feats["frac_task"], feats["frac_state"]]
                 if not np.isnan(v)]
        if fracs:
            p = np.array(fracs)
            feats["routing_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            feats["routing_entropy"] = nan

        # Attention concentration within each modality's tokens
        if sp_base:
            base_attn = mean_attn[sp_base[0]:sp_base[1]]
            feats["attn_gini_base"] = gini(base_attn)
            feats["attn_entropy_base"] = entropy(base_attn)
        else:
            feats["attn_gini_base"] = nan
            feats["attn_entropy_base"] = nan

        if sp_wrist:
            feats["attn_gini_wrist"] = gini(mean_attn[sp_wrist[0]:sp_wrist[1]])
        else:
            feats["attn_gini_wrist"] = nan

        if sp_task:
            feats["attn_gini_task"] = gini(mean_attn[sp_task[0]:sp_task[1]])
        else:
            feats["attn_gini_task"] = nan

        if sp_state:
            feats["attn_gini_state"] = gini(mean_attn[sp_state[0]:sp_state[1]])
        else:
            feats["attn_gini_state"] = nan
    else:
        for k in ["frac_base", "frac_wrist", "frac_rwrist", "frac_task", "frac_state",
                  "routing_entropy", "attn_gini_base", "attn_entropy_base",
                  "attn_gini_wrist", "attn_gini_task", "attn_gini_state"]:
            feats[k] = nan

    # ── GradCAM / raw-alpha heatmaps ──────────────────────────────────────────
    for cam_key, cam_label in [
        ("base_0_rgb", "base"),
        ("left_wrist_0_rgb", "wrist"),
    ]:
        if fmt == "pi0fast":
            gc, ras, rn = _get_heatmaps_pi0fast(rec, cam_key)
        else:
            gc, ras, rn = _get_heatmaps_pi05(rec, cam_key)

        if gc is not None:
            feats[f"gc_gini_{cam_label}"]   = gini(gc)
            feats[f"ras_gini_{cam_label}"]  = gini(ras)
            feats[f"rn_gini_{cam_label}"]   = gini(rn)
            feats[f"gc_max_{cam_label}"]    = float(gc.max())
            feats[f"gc_mean_{cam_label}"]   = float(gc.mean())
            feats[f"agree_gc_ras_{cam_label}"] = cosine(gc, ras)
            feats[f"agree_gc_rn_{cam_label}"]  = cosine(gc, rn)
            feats[f"agree_ras_rn_{cam_label}"] = cosine(ras, rn)
            cy, cx = spatial_com(gc)
            feats[f"gc_com_y_{cam_label}"] = cy
            feats[f"gc_com_x_{cam_label}"] = cx
        else:
            for sfx in ["gc_gini", "ras_gini", "rn_gini", "gc_max", "gc_mean",
                        "agree_gc_ras", "agree_gc_rn", "agree_ras_rn",
                        "gc_com_y", "gc_com_x"]:
                feats[f"{sfx}_{cam_label}"] = nan

    # ── Task token scores ─────────────────────────────────────────────────────
    if fmt == "pi0fast":
        gc_task, ras_task = _get_token_scores_pi0fast(rec, "task")
        gc_state, ras_state = _get_token_scores_pi0fast(rec, "state")
    else:
        gc_task, ras_task = _get_token_scores_pi05(rec, "task")
        gc_state, ras_state = None, None   # pi05 has no explicit state span

    if gc_task is not None:
        feats["gc_gini_task"]  = gini(gc_task)
        feats["ras_gini_task"] = gini(ras_task)
        feats["agree_gc_ras_task"] = cosine(gc_task, ras_task)
        feats["gc_max_task"]  = float(gc_task.max())
    else:
        feats["gc_gini_task"] = feats["ras_gini_task"] = nan
        feats["agree_gc_ras_task"] = feats["gc_max_task"] = nan

    if gc_state is not None:
        feats["gc_gini_state"]  = gini(gc_state)
        feats["ras_gini_state"] = gini(ras_state)
        feats["agree_gc_ras_state"] = cosine(gc_state, ras_state)
    else:
        feats["gc_gini_state"] = feats["ras_gini_state"] = nan
        feats["agree_gc_ras_state"] = nan

    # ── Action features ───────────────────────────────────────────────────────
    actions = np.asarray(rec["outputs/actions"]).astype(np.float64)  # (10, 7)
    signs = np.sign(actions)
    flips = (signs[1:] != signs[:-1]).astype(float)
    feats["action_trans_flip"] = float(flips[:, 0:3].mean())
    feats["action_rot_flip"]   = float(flips[:, 3:6].mean())
    feats["action_grip_flip"]  = float(flips[:, 6].mean())
    feats["action_magnitude"]  = float(np.linalg.norm(actions[0]))
    feats["gripper_val"]       = float(actions[0, 6])

    # ── Temporal diff features (need previous step) ───────────────────────────
    state_key = "outputs/state"
    if state_key not in rec:
        state_key = "inputs/observation/state"
    state = np.asarray(rec.get(state_key, np.zeros(8))).astype(np.float64)

    if prev is not None:
        prev_state_key = "outputs/state" if "outputs/state" in prev else "inputs/observation/state"
        prev_state = np.asarray(prev.get(prev_state_key, state)).astype(np.float64)
        feats["joint_vel"] = float(np.linalg.norm(state - prev_state))

        prev_actions = np.asarray(prev["outputs/actions"]).astype(np.float64)
        a0, p0 = actions[0], prev_actions[0]
        na, np_ = np.linalg.norm(a0), np.linalg.norm(p0)
        feats["plan_consistency"] = float(np.dot(a0, p0) / (na * np_ + 1e-9))

        # CoM drift since last step (base cam)
        if gc is not None and fmt == "pi0fast":
            prev_gc_base, _, _ = _get_heatmaps_pi0fast(prev, "base_0_rgb")
        elif gc is not None:
            prev_gc_base, _, _ = _get_heatmaps_pi05(prev, "base_0_rgb")
        else:
            prev_gc_base = None
        if prev_gc_base is not None:
            cy2, cx2 = spatial_com(prev_gc_base)
            feats["gc_com_drift_base"] = float(
                np.sqrt((feats["gc_com_y_base"] - cy2)**2 + (feats["gc_com_x_base"] - cx2)**2))
        else:
            feats["gc_com_drift_base"] = nan
    else:
        feats["joint_vel"] = 0.0
        feats["plan_consistency"] = 1.0
        feats["gc_com_drift_base"] = 0.0

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_episode_features(data_dir: Path, ep: dict, max_steps: int) -> List[dict]:
    records = []
    prev_rec = None
    for rel in range(max_steps):
        s = ep["start_idx"] + rel
        if s > ep["end_idx"]:
            break
        fpath = data_dir / f"step_{s}.npy"
        if not fpath.exists():
            continue
        try:
            rec = np.load(fpath, allow_pickle=True).item()
            feats = extract_features(rec, prev=prev_rec)
            feats["rel_step"] = rel
            feats["abs_step"] = s
            records.append(feats)
            prev_rec = rec
        except Exception as e:
            print(f"    [warn] {fpath.name}: {e}")
    return records


def load_group(
    group_name: str,
    run_names: List[str],
    max_steps: int = MAX_STEPS,
    use_cache: bool = True,
) -> Dict[str, List[List[dict]]]:
    """
    Load all episodes for a group. Returns {outcome: [[step_feats, ...], ...]}
    """
    cache_path = CACHE_DIR / f"{group_name}.pkl"
    if use_cache and cache_path.exists():
        print(f"  Loading {group_name} from cache ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    print(f"  {group_name}: {len(ep_data['success'])} success, "
          f"{len(ep_data['failure'])} failure episodes loaded.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(ep_data, f, protocol=4)
    return ep_data


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_matrix(ep_list: List[List[dict]], key: str, max_steps: int) -> np.ndarray:
    """Shape: (n_episodes, max_steps), NaN for missing steps."""
    out = np.full((len(ep_list), max_steps), np.nan)
    for i, ep in enumerate(ep_list):
        for rec in ep:
            t = rec.get("rel_step", 0)
            if t < max_steps:
                out[i, t] = rec.get(key, np.nan)
    return out


def compute_auroc_sequence(
    ep_data: Dict[str, List[List[dict]]],
    feature: str,
    max_steps: int,
) -> np.ndarray:
    """AUROC at each step t, using only episodes active at that step."""
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    aurocs = np.full(max_steps, np.nan)

    for t in range(max_steps):
        s_vals = S[:, t]
        f_vals = F[:, t]
        s_vals = s_vals[~np.isnan(s_vals)]
        f_vals = f_vals[~np.isnan(f_vals)]
        if len(s_vals) < AUROC_MIN_N or len(f_vals) < AUROC_MIN_N:
            continue
        # AUROC via Mann-Whitney U
        try:
            stat, _ = mannwhitneyu(f_vals, s_vals, alternative="two-sided")
            auroc = float(stat) / (len(s_vals) * len(f_vals))
            # Ensure AUROC >= 0.5 (flip direction if needed)
            if auroc < 0.5:
                auroc = 1.0 - auroc
            aurocs[t] = auroc
        except Exception:
            pass
    return aurocs


def compute_cohens_d_sequence(
    ep_data: Dict[str, List[List[dict]]],
    feature: str,
    max_steps: int,
) -> np.ndarray:
    """Cohen's d (|µS - µF| / pooled_std) at each step t."""
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    ds = np.full(max_steps, np.nan)

    for t in range(max_steps):
        sv = S[:, t]; sv = sv[~np.isnan(sv)]
        fv = F[:, t]; fv = fv[~np.isnan(fv)]
        if len(sv) < 3 or len(fv) < 3:
            continue
        pooled = np.sqrt((np.var(sv, ddof=1) + np.var(fv, ddof=1)) / 2.0 + 1e-9)
        ds[t] = abs(float(sv.mean()) - float(fv.mean())) / pooled
    return ds


def compute_pvalue_sequence(
    ep_data: Dict[str, List[List[dict]]],
    feature: str,
    max_steps: int,
) -> np.ndarray:
    S = to_matrix(ep_data["success"], feature, max_steps)
    F = to_matrix(ep_data["failure"], feature, max_steps)
    pvals = np.ones(max_steps)
    for t in range(max_steps):
        sv = S[:, t]; sv = sv[~np.isnan(sv)]
        fv = F[:, t]; fv = fv[~np.isnan(fv)]
        if len(sv) < AUROC_MIN_N or len(fv) < AUROC_MIN_N:
            continue
        try:
            _, p = mannwhitneyu(sv, fv, alternative="two-sided")
            pvals[t] = float(p)
        except Exception:
            pass
    return pvals


def feature_names(ep_data: Dict[str, List[List[dict]]]) -> List[str]:
    for eps in ep_data.values():
        for ep in eps:
            if ep:
                return [k for k in ep[0].keys() if k not in ("rel_step", "abs_step")]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: AUROC HEATMAP + TOP CURVES
# ─────────────────────────────────────────────────────────────────────────────

def fig_auroc_overview(
    ep_data: Dict[str, List[List[dict]]],
    group_name: str,
    max_steps: int = MAX_STEPS,
):
    feats = feature_names(ep_data)
    print(f"  Computing AUROC for {len(feats)} features …")

    auroc_matrix = np.full((len(feats), max_steps), np.nan)
    for fi, feat in enumerate(tqdm(feats, desc="  AUROC", ncols=80, leave=False)):
        auroc_matrix[fi] = compute_auroc_sequence(ep_data, feat, max_steps)

    # Sort features by mean AUROC (ignoring NaN)
    mean_auroc = np.nanmean(auroc_matrix, axis=1)
    order = np.argsort(mean_auroc)[::-1]
    sorted_feats  = [feats[i] for i in order]
    sorted_matrix = auroc_matrix[order]
    mean_sorted   = mean_auroc[order]

    # ── Panel 1: heatmap of top-40 features ──────────────────────────────────
    top_n = min(40, len(sorted_feats))
    fig, ax = plt.subplots(figsize=(14, top_n * 0.38 + 1.5))
    im = ax.imshow(
        sorted_matrix[:top_n],
        aspect="auto", vmin=0.5, vmax=1.0,
        cmap="RdYlGn", origin="upper",
    )
    ax.set_xticks(np.arange(0, max_steps, 5))
    ax.set_xticklabels(np.arange(0, max_steps, 5))
    ax.set_yticks(np.arange(top_n))
    ax.set_yticklabels(
        [f"{sorted_feats[i]}  ({mean_sorted[i]:.3f})" for i in range(top_n)],
        fontsize=7,
    )
    ax.set_xlabel("Step within episode")
    ax.set_title(
        f"AUROC(step) — {group_name}\n"
        f"(top-{top_n} features sorted by mean AUROC; green=separable, red=chance)",
        fontsize=10, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig1_auroc_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")

    # ── Panel 2: AUROC curves for top-10 features ─────────────────────────────
    top10 = min(10, len(sorted_feats))
    colors = plt.cm.tab10(np.linspace(0, 1, top10))
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(max_steps)
    for i in range(top10):
        a = sorted_matrix[i]
        valid = ~np.isnan(a)
        ax.plot(steps[valid], a[valid],
                color=colors[i], lw=1.8,
                label=f"{sorted_feats[i]} ({mean_sorted[i]:.3f})")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(0.7, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Step within episode")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.05)
    ax.set_title(f"Top-10 feature AUROC curves — {group_name}", fontweight="bold")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    fig.tight_layout()
    out2 = OUTPUT_DIR / group_name / "fig1b_auroc_top10.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out2}")

    return sorted_feats, sorted_matrix, mean_sorted


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: EARLY DETECTION RANKING
# ─────────────────────────────────────────────────────────────────────────────

def fig_early_detection_ranking(
    sorted_feats: List[str],
    sorted_matrix: np.ndarray,
    group_name: str,
    horizons: Tuple[int, ...] = (5, 10, 20, 30),
):
    """Bar chart: top-15 features by AUROC at various early-detection horizons."""
    top_n = min(15, len(sorted_feats))
    n_horiz = len(horizons)
    fig, axes = plt.subplots(1, n_horiz, figsize=(n_horiz * 5.5, top_n * 0.42 + 1.5),
                             sharey=True)
    if n_horiz == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_horiz))

    for ax, h, col in zip(axes, horizons, colors):
        # Best features at this horizon
        vals = sorted_matrix[:, h] if h < sorted_matrix.shape[1] else np.full(len(sorted_feats), np.nan)
        order_h = np.argsort(vals)[::-1][:top_n]
        ax.barh(
            np.arange(top_n),
            vals[order_h],
            color=col, alpha=0.8,
        )
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels([sorted_feats[i] for i in order_h], fontsize=7)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.axvline(0.7, color="gray", ls=":", lw=0.8)
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUROC")
        ax.set_title(f"Step {h}", fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle(
        f"Best early-detection features — {group_name}\n"
        f"(AUROC at fixed step horizon)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig2_early_ranking.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: MEAN ± CI TRAJECTORIES WITH p-VALUE BAND
# ─────────────────────────────────────────────────────────────────────────────

def fig_trajectories(
    ep_data: Dict[str, List[List[dict]]],
    features_to_plot: List[str],
    group_name: str,
    max_steps: int = MAX_STEPS,
    tag: str = "",
):
    n = len(features_to_plot)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]

    steps = np.arange(max_steps)

    for ax, feat in zip(axes, features_to_plot):
        pvals = compute_pvalue_sequence(ep_data, feat, max_steps)
        sig_mask = pvals < 0.05

        for label, cm, ci in [("success", BLUE_M, BLUE_L), ("failure", RED_M, RED_L)]:
            eps = [e for e in ep_data.get(label, []) if e]
            if not eps:
                continue
            mat = to_matrix(eps, feat, max_steps)
            n_active = np.sum(~np.isnan(mat), axis=0)
            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat, axis=0)
            se   = std / np.sqrt(np.maximum(n_active, 1))
            ci95 = 1.96 * se

            # Individual episode lines (thin, low alpha)
            for row in mat:
                valid = ~np.isnan(row)
                if valid.sum() < 2:
                    continue
                ax.plot(steps[valid], row[valid], color=ci, lw=LW_I, alpha=AL, zorder=1)

            # Mean line
            valid = ~np.isnan(mean)
            ax.plot(steps[valid], mean[valid], color=cm, lw=LW_M, label=label, zorder=3)
            ax.fill_between(steps[valid], (mean-ci95)[valid], (mean+ci95)[valid],
                            color=cm, alpha=0.18, zorder=2)

        # Significance stripe
        for t in range(max_steps - 1):
            if sig_mask[t]:
                ax.axvspan(t - 0.5, t + 0.5, color="gold", alpha=0.18, zorder=0)

        ax.set_ylabel(feat, fontsize=7)
        ax.set_title(feat, loc="left", pad=2, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Step within episode")
    sig_patch = plt.Line2D([0], [0], color="gold", lw=8, alpha=0.6, label="p<0.05 (MW)")
    axes[-1].legend(handles=[sig_patch] + axes[-1].get_legend_handles_labels()[0],
                    fontsize=7, loc="upper right")

    tag_str = f"_{tag}" if tag else ""
    title = f"Mean ± 95%CI trajectories — {group_name}{tag_str}"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / f"fig3_trajectories{tag_str}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: CONFORMAL PREDICTION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _running_mean(ep_records: List[dict], feat: str, max_steps: int) -> np.ndarray:
    """Running cumulative mean of a feature over steps 0..t for one episode."""
    vals = np.full(max_steps, np.nan)
    for rec in ep_records:
        t = rec.get("rel_step", 0)
        if t < max_steps:
            vals[t] = rec.get(feat, np.nan)
    # Forward-fill NaN and compute cumulative mean
    out = np.full(max_steps, np.nan)
    running_sum = 0.0
    count = 0
    for t in range(max_steps):
        if not np.isnan(vals[t]):
            running_sum += vals[t]
            count += 1
        if count > 0:
            out[t] = running_sum / count
    return out


def fig_conformal_simulation(
    ep_data: Dict[str, List[List[dict]]],
    features: List[str],
    group_name: str,
    max_steps: int = MAX_STEPS,
    fpr: float = FPR_TARGET,
    n_cal_frac: float = 0.6,
):
    """
    Leave-group-out conformal prediction simulation.

    For each candidate feature (or composite):
      1. Split success episodes into calibration (60%) and probe (40%)
      2. For each step T, compute running-mean nonconformity score
         score(ep, T) = |running_mean(feature, 0..T) - cal_mean|
      3. Threshold: (1 - fpr) quantile of scores over calibration success eps
      4. TPR = fraction of failure eps whose score exceeds threshold at step T
      5. FPR = fraction of probe success eps flagged (should be ≈ fpr)
    """
    successes = ep_data.get("success", [])
    failures  = ep_data.get("failure", [])
    if len(successes) < 10 or len(failures) < 5:
        print(f"  [skip conformal] not enough episodes in {group_name}")
        return

    n_cal = max(5, int(len(successes) * n_cal_frac))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(successes))
    cal_eps   = [successes[i] for i in perm[:n_cal]]
    probe_eps = [successes[i] for i in perm[n_cal:]]
    if len(probe_eps) < 3:
        probe_eps = cal_eps[:max(3, len(cal_eps) // 5)]

    top_feats = features[:min(8, len(features))]

    # Build composite score: z-score each feature on calibration set then average
    # At each step T, calibration mean and std are computed on running mean up to T
    print(f"  Conformal simulation on {len(top_feats)} features …")

    # Pre-compute running-mean trajectories
    def get_trajectories(eps_list, feat):
        return np.stack([_running_mean(ep, feat, max_steps) for ep in eps_list], axis=0)

    fig, axes = plt.subplots(len(top_feats), 2, figsize=(14, 3.0 * len(top_feats) + 0.5),
                             gridspec_kw={"width_ratios": [3, 1]})
    if len(top_feats) == 1:
        axes = [axes]

    steps = np.arange(max_steps)

    for row, feat in enumerate(tqdm(top_feats, desc="  conformal", ncols=80, leave=False)):
        ax_tpr = axes[row][0]
        ax_bar = axes[row][1]

        cal_traj = get_trajectories(cal_eps, feat)       # (n_cal, T)
        probe_traj = get_trajectories(probe_eps, feat)   # (n_probe, T)
        fail_traj = get_trajectories(failures, feat)     # (n_fail, T)

        tpr_seq = np.full(max_steps, np.nan)
        fpr_seq = np.full(max_steps, np.nan)
        detect_step = np.full(len(failures), max_steps + 1, dtype=float)

        for t in range(1, max_steps):
            cal_vals = cal_traj[:, t]
            cal_vals = cal_vals[~np.isnan(cal_vals)]
            if len(cal_vals) < 5:
                continue

            cal_mean = cal_vals.mean()
            # Nonconformity score: absolute deviation from calibration mean
            scores_cal   = np.abs(cal_vals - cal_mean)
            threshold    = np.quantile(scores_cal, 1.0 - fpr)

            fail_vals = fail_traj[:, t]
            n_fail_active = np.sum(~np.isnan(fail_vals))
            if n_fail_active < 1:
                continue

            fail_scores = np.abs(fail_vals - cal_mean)
            probe_vals  = probe_traj[:, t]
            probe_scores = np.abs(probe_vals[~np.isnan(probe_vals)] - cal_mean)

            tpr_seq[t] = float(np.nansum(fail_scores > threshold)) / n_fail_active
            if len(probe_scores) > 0:
                fpr_seq[t] = float((probe_scores > threshold).sum()) / len(probe_scores)

            # Record first detection step for each failure episode
            for fi, fs in enumerate(fail_scores):
                if not np.isnan(fs) and fs > threshold and detect_step[fi] > t:
                    detect_step[fi] = t

        # TPR and FPR curves
        valid = ~np.isnan(tpr_seq)
        ax_tpr.plot(steps[valid], tpr_seq[valid], color=RED_M, lw=2.0, label="TPR (failures detected)")
        valid_f = ~np.isnan(fpr_seq)
        ax_tpr.plot(steps[valid_f], fpr_seq[valid_f], color="gray", ls="--", lw=1.5, label=f"FPR (target={fpr})")
        ax_tpr.axhline(fpr, color="gray", ls=":", lw=1.0, alpha=0.6)
        ax_tpr.set_ylabel("Rate")
        ax_tpr.set_ylim(-0.02, 1.05)
        ax_tpr.legend(fontsize=7, loc="upper left")
        ax_tpr.set_title(f"{feat}", loc="left", fontsize=8, fontweight="bold")
        ax_tpr.spines[["top", "right"]].set_visible(False)

        # Distribution of first-detection step
        detected = detect_step[detect_step <= max_steps]
        undetected = (detect_step > max_steps).sum()
        if len(detected) > 0:
            ax_bar.hist(detected, bins=15, range=(0, max_steps),
                        color=RED_M, alpha=0.75, edgecolor="white")
            ax_bar.set_xlabel("First detection step")
            ax_bar.set_ylabel("# failures")
            ax_bar.set_title(f"Detected: {len(detected)}/{len(failures)}\n"
                             f"(undetected: {undetected})", fontsize=7)
        ax_bar.spines[["top", "right"]].set_visible(False)

    axes[-1][0].set_xlabel("Step within episode")
    fig.suptitle(
        f"Conformal prediction simulation — {group_name}\n"
        f"FPR target = {fpr*100:.0f}%  |  calibration = {n_cal} success episodes",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig4_conformal_simulation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: CROSS-GROUP AUROC COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def fig_cross_group_comparison(
    group_data: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    horizon: int = 20,
):
    """
    Compare AUROC at a fixed horizon across groups for features present in all.
    group_data: {group_name: (sorted_feats, auroc_matrix, mean_auroc)}
    """
    if len(group_data) < 2:
        return

    # Find common features
    all_feat_sets = [set(gd[0]) for gd in group_data.values()]
    common = sorted(set.intersection(*all_feat_sets))
    if not common:
        return

    group_names = list(group_data.keys())
    n_groups = len(group_names)
    n_feats  = min(20, len(common))

    # For each group, get AUROC at horizon for common features
    feat_aurocs = {}
    for g, (feats, mat, mean) in group_data.items():
        feat_idx = {f: i for i, f in enumerate(feats)}
        feat_aurocs[g] = {f: mat[feat_idx[f], horizon] if horizon < mat.shape[1] else np.nan
                         for f in common}

    # Sort by mean AUROC across groups
    sort_key = {f: np.nanmean([feat_aurocs[g][f] for g in group_names]) for f in common}
    top_feats = sorted(common, key=lambda f: sort_key[f], reverse=True)[:n_feats]

    fig, ax = plt.subplots(figsize=(10, n_feats * 0.45 + 2))
    x = np.arange(n_feats)
    width = 0.75 / n_groups
    colors_g = plt.cm.Set2(np.linspace(0, 1, n_groups))

    for gi, (g, col) in enumerate(zip(group_names, colors_g)):
        vals = [feat_aurocs[g][f] for f in top_feats]
        ax.barh(x + gi * width, vals, width, color=col, alpha=0.85, label=g)

    ax.set_yticks(x + width * (n_groups - 1) / 2)
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.axvline(0.7, color="gray", ls=":", lw=0.8)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC")
    ax.set_title(
        f"Cross-group AUROC @ step {horizon}\n"
        f"(top-{n_feats} features sorted by mean AUROC across groups)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUTPUT_DIR / f"fig5_cross_group_h{horizon}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: DISTRIBUTION SNAPSHOTS (KDE at fixed steps)
# ─────────────────────────────────────────────────────────────────────────────

def fig_distribution_snapshots(
    ep_data: Dict[str, List[List[dict]]],
    top_features: List[str],
    group_name: str,
    snapshot_steps: Tuple[int, ...] = (5, 15, 25, 40),
    max_steps: int = MAX_STEPS,
):
    """KDE plots at fixed steps to show how distributions diverge over time."""
    feats = top_features[:min(6, len(top_features))]
    n_f   = len(feats)
    n_s   = len(snapshot_steps)
    if n_f == 0:
        return

    fig, axes = plt.subplots(n_f, n_s, figsize=(n_s * 3.5, n_f * 3.0 + 0.5),
                             sharex="row")

    for fi, feat in enumerate(feats):
        S = to_matrix(ep_data["success"], feat, max_steps)
        F = to_matrix(ep_data["failure"], feat, max_steps)

        # Determine x-range from all non-NaN values
        all_vals = np.concatenate([S.ravel(), F.ravel()])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) == 0:
            continue
        xmin, xmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        if xmin >= xmax:
            xmin -= 0.1; xmax += 0.1

        for si, step in enumerate(snapshot_steps):
            ax = axes[fi][si]
            sv = S[:, step]; sv = sv[~np.isnan(sv)]
            fv = F[:, step]; fv = fv[~np.isnan(fv)]

            for vals, col, lab in [(sv, BLUE_M, "success"), (fv, RED_M, "failure")]:
                if len(vals) < 3:
                    continue
                from scipy.stats import gaussian_kde
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

    fig.suptitle(f"Feature distributions at fixed steps — {group_name}",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig6_distributions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_text_report(
    group_data: Dict[str, Tuple[List[str], np.ndarray, np.ndarray]],
    max_steps: int = MAX_STEPS,
):
    lines = [
        "=" * 72,
        "CONFORMAL PREDICTION FEATURE RANKING REPORT",
        "=" * 72,
        "",
    ]
    horizons = [5, 10, 20, 30]

    for group_name, (feats, mat, mean_auroc) in group_data.items():
        lines += [f"GROUP: {group_name}", "-" * 50]
        lines.append(f"{'Feature':<40}  mean  " + "  ".join(f"s{h}" for h in horizons))
        lines.append("-" * 72)
        for i in range(min(30, len(feats))):
            vals = [mat[i, h] if h < mat.shape[1] else np.nan for h in horizons]
            row = f"{feats[i]:<40}  {mean_auroc[i]:.3f}  " + \
                  "  ".join(f"{v:.3f}" if not np.isnan(v) else "  ---" for v in vals)
            lines.append(row)
        lines += ["", ""]

    out = OUTPUT_DIR / "feature_ranking_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_group_data = {}  # group_name → (sorted_feats, auroc_matrix, mean_auroc)

    for group_name, run_names in RUN_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"GROUP: {group_name}")
        print("="*60)

        ep_data = load_group(group_name, run_names, max_steps=MAX_STEPS)

        n_s = len(ep_data["success"])
        n_f = len(ep_data["failure"])
        print(f"  Loaded: {n_s} success, {n_f} failure episodes")
        if n_s < 5 or n_f < 5:
            print(f"  [skip] not enough episodes")
            continue

        print(f"  Computing AUROC curves …")
        sorted_feats, auroc_mat, mean_auroc = fig_auroc_overview(
            ep_data, group_name, max_steps=MAX_STEPS)

        all_group_data[group_name] = (sorted_feats, auroc_mat, mean_auroc)

        print(f"  Top-5 features: {sorted_feats[:5]}")

        print(f"  Early detection ranking …")
        fig_early_detection_ranking(sorted_feats, auroc_mat, group_name)

        print(f"  Trajectory plots for top-10 features …")
        # Group 1: attention routing
        routing_feats = [f for f in sorted_feats if any(
            s in f for s in ["frac_", "routing", "attn_gini"]) ][:8]
        # Group 2: gradcam / method agreement
        gradcam_feats = [f for f in sorted_feats if any(
            s in f for s in ["gc_gini", "ras_gini", "agree_", "gc_max"]) ][:8]
        # Group 3: action features
        action_feats  = [f for f in sorted_feats if any(
            s in f for s in ["action_", "plan_", "joint_", "gripper"]) ][:8]

        for feats_sub, tag in [(routing_feats, "routing"),
                               (gradcam_feats, "gradcam"),
                               (action_feats, "action")]:
            if feats_sub:
                fig_trajectories(ep_data, feats_sub, group_name,
                                 max_steps=MAX_STEPS, tag=tag)

        print(f"  Distribution snapshots for top-6 features …")
        fig_distribution_snapshots(ep_data, sorted_feats[:6], group_name)

        print(f"  Conformal prediction simulation …")
        fig_conformal_simulation(ep_data, sorted_feats[:8], group_name)

    print(f"\n{'='*60}")
    print("Cross-group comparison …")
    for h in [5, 10, 20, 30]:
        fig_cross_group_comparison(all_group_data, horizon=h)

    write_text_report(all_group_data)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
