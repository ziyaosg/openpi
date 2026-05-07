#!/usr/bin/env python3
"""
Per-step diagnostic plots comparing success vs failure episodes.

For each episode, per-step metrics are computed and plotted as individual
lines (low alpha) with a bold mean line, success in blue and failure in red.
All figures share the same x-axis (step within episode).

Figures produced:
  fig1_attention_concentration.png  — Gini per attribution method (4 rows)
  fig2_attention_routing.png        — modality attention fractions (4 rows)
  fig_inter_head_agreement.png      — task-token inter-head cosine similarity (1 row)
  fig3_action_oscillation.png       — sign-flip rates + plan consistency (4 rows)
  fig4_motion_consensus.png         — joint velocity (1 row)
  fig5_base_attribution.png         — cosine / IoU / CoM-dist / EMD, base cam (4 rows)
  fig6_wrist_attribution.png        — cosine / IoU / CoM-dist / EMD, left wrist cam (4 rows)
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from scripts.attention_visualization.grid_visualization_raw_weights import (
    select_heads, TARGET_LAYER, HEAD_SELECTION_LAYER, TOP_K_HEADS,
)
from scripts.attention_visualization.grid_visualization_value_attn import score_v_cosine
from scripts.attention_visualization.grid_visualization_gradcam import to_2d_heatmap
from scripts.attention_utils.keys import CAM_NAMES, IMAGE_PATCH_GRID

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("/home/ziyao/Documents/policy_records_20260423_180932")
EPISODE_JSON = DATA_DIR / "episode_summaries.json"
OUTPUT_DIR   = DATA_DIR / "step_metric_plots"

MIXED_TASK_IDS: set[int] = {2, 3, 5, 6, 9}
MAX_STEPS = 50

PATCH_H, PATCH_W = IMAGE_PATCH_GRID
CAM_BASE   = "base_0_rgb"
CAM_WRISTS = [c for c in CAM_NAMES if c != CAM_BASE]

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"
LAYERS_KEY    = "outputs/debug/attn/layers"

METHOD_NAMES = ["GradCAM", "Raw Alpha", "Raw Weights", "V-Cosine"]
METHOD_PAIRS = list(combinations(range(len(METHOD_NAMES)), 2))

# Plot style
COL_S_MEAN = "#1f5fa6"
COL_F_MEAN = "#b02020"
COL_S_IND  = "#a8c8e8"
COL_F_IND  = "#f0a0a0"
ALPHA_IND  = 0.55
LW_MEAN    = 2.2
LW_IND     = 1.2
FIG_W      = 12
ROW_H      = 2.6

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 8,
    "legend.framealpha": 0.7,
})

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    if key not in rec:
        return None
    return tuple(int(x) for x in np.asarray(rec[key]).tolist())


def _gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v)).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s == 0 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


TOPK_PATCHES = 26  # top ~10% of 256 patches

def _topk_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    top_a = set(np.argsort(a)[-TOPK_PATCHES:].tolist())
    top_b = set(np.argsort(b)[-TOPK_PATCHES:].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def _com_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between centres of mass in patch-grid units. Lower = more similar."""
    a = np.abs(a).astype(np.float64); b = np.abs(b).astype(np.float64)
    rows = np.arange(PATCH_H)[:, None]; cols = np.arange(PATCH_W)[None, :]
    sa, sb = a.sum() + 1e-12, b.sum() + 1e-12
    cy_a, cx_a = float((a * rows).sum() / sa), float((a * cols).sum() / sa)
    cy_b, cx_b = float((b * rows).sum() / sb), float((b * cols).sum() / sb)
    return float(np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2))


def _emd(a: np.ndarray, b: np.ndarray) -> float:
    """Mean of 1-D Wasserstein distances on x- and y-axis marginals (patch units).
    Lower = more similar."""
    a = np.abs(a.ravel()).astype(np.float64); b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    a /= sa; b /= sb
    a2, b2 = a.reshape(PATCH_H, PATCH_W), b.reshape(PATCH_H, PATCH_W)
    xs, ys = np.arange(PATCH_W), np.arange(PATCH_H)
    emd_x = wasserstein_distance(xs, xs, a2.sum(axis=0), b2.sum(axis=0))
    emd_y = wasserstein_distance(ys, ys, a2.sum(axis=1), b2.sum(axis=1))
    return float(0.5 * (emd_x + emd_y))


def _cam_agreement(heats: list) -> dict:
    """Compute mean of each metric across all method pairs for one camera."""
    cos, iou, com, emd = [], [], [], []
    for i, j in METHOD_PAIRS:
        cos.append(_cos_sim(heats[i], heats[j]))
        iou.append(_topk_iou(heats[i], heats[j]))
        com.append(_com_distance(heats[i], heats[j]))
        emd.append(_emd(heats[i], heats[j]))
    return {
        "cosine":   float(np.mean(cos)),
        "iou":      float(np.mean(iou)),
        "com_dist": float(np.mean(com)),
        "emd":      float(np.mean(emd)),
    }


def _com_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute difference of weighted-mean token positions. Lower = more similar."""
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
    a /= sa; b /= sb
    pos = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(pos, pos, a, b))


def _topk_iou_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a, b = a.ravel(), b.ravel()
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def _tok_agreement(scores: list) -> dict:
    """Compute mean of each 1-D metric across all method pairs for one token modality."""
    k = max(1, int(round(len(scores[0].ravel()) * 0.10)))
    cos, iou, com, emd = [], [], [], []
    for i, j in METHOD_PAIRS:
        cos.append(_cos_sim(scores[i], scores[j]))
        iou.append(_topk_iou_k(scores[i], scores[j], k))
        com.append(_com_distance_1d(scores[i], scores[j]))
        emd.append(_emd_1d(scores[i], scores[j]))
    return {
        "cosine":   float(np.mean(cos)),
        "iou":      float(np.mean(iou)),
        "com_dist": float(np.mean(com)),
        "emd":      float(np.mean(emd)),
    }


def _layer_indices(rec: dict) -> Tuple[int, int]:
    stored = np.asarray(rec[LAYERS_KEY]).tolist()
    return stored.index(HEAD_SELECTION_LAYER), stored.index(TARGET_LAYER)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (per step)
# ─────────────────────────────────────────────────────────────────────────────

def extract(rec: dict) -> dict:
    full_attn = np.asarray(rec[FULL_ATTN_KEY])     # (L, 1, H, S)
    full_v    = np.asarray(rec[FULL_V_KEY])         # (L, 1, S, K, D)
    actions   = np.asarray(rec["outputs/actions"])  # (10, 7)
    state     = np.asarray(rec["outputs/state"])    # (8,)

    sp_base   = _span(rec, f"outputs/debug/spans/image/{CAM_BASE}")
    sp_wrists = [s for s in (_span(rec, f"outputs/debug/spans/image/{c}") for c in CAM_WRISTS)
                 if s is not None]
    sp_task   = _span(rec, "outputs/debug/spans/task")
    sp_state  = _span(rec, "outputs/debug/spans/state")

    sel_idx, tgt_idx = _layer_indices(rec)
    attn_sel = full_attn[sel_idx, 0]  # (H, S)  head-selection layer
    attn_tgt = full_attn[tgt_idx, 0]  # (H, S)  target layer
    v_tgt    = full_v[tgt_idx, 0]     # (S, K, D)

    all_cam_spans = [sp_base] + sp_wrists
    top_heads = select_heads(attn_sel, all_cam_spans, sp_task, sp_state, TOP_K_HEADS)

    # ── Attention concentration (base cam, mean over all heads) ───────────────
    base_attn = attn_tgt[:, sp_base[0]:sp_base[1]].mean(axis=0)  # (N_patches,)
    gini_base = _gini(base_attn)
    p = base_attn / (base_attn.sum() + 1e-12)
    spatial_ent = float(-np.sum(p * np.log(p + 1e-12)))

    # ── Attention modality fractions ──────────────────────────────────────────
    def _frac(sp):
        return float(attn_tgt[:, sp[0]:sp[1]].sum(axis=1).mean()) if sp else 0.0
    base_frac  = _frac(sp_base)
    wrist_frac = sum(_frac(s) for s in sp_wrists)
    task_frac  = _frac(sp_task)
    state_frac = _frac(sp_state)

    # ── Task-token inter-head agreement (selected heads only) ─────────────────
    if sp_task and len(top_heads) > 1:
        h_task = attn_tgt[top_heads][:, sp_task[0]:sp_task[1]]  # (K, N_task)
        h_n = h_task / (np.linalg.norm(h_task, axis=1, keepdims=True) + 1e-9)
        sim = h_n @ h_n.T                                        # (K, K)
        iu = np.triu_indices(len(top_heads), k=1)
        task_head_agr = float(sim[iu].mean())
    else:
        task_head_agr = float("nan")

    # ── Action sign flips within predicted horizon ────────────────────────────
    signs = np.sign(actions)                          # (10, 7)
    flips = (signs[1:] != signs[:-1]).astype(float)  # (9, 7)
    trans_flip = float(flips[:, 0:3].mean())   # translation dims 0-2
    rot_flip   = float(flips[:, 3:6].mean())   # rotation dims 3-5
    grip_flip  = float(flips[:, 6].mean())     # gripper dim 6

    # ── Attribution heatmap method agreement (base cam) ───────────────────────
    raw_w = attn_tgt[top_heads].max(axis=0)                      # (S,)
    vcos  = score_v_cosine(attn_tgt, v_tgt, top_heads)           # (S,) already abs

    gc_h = np.abs(to_2d_heatmap(
        np.asarray(rec[f"outputs/debug/gradcam/image/{CAM_BASE}"]), CAM_BASE))
    ra_h = np.abs(to_2d_heatmap(
        np.asarray(rec[f"outputs/debug/raw_alpha/summation/image/{CAM_BASE}"]), CAM_BASE))
    rw_h = raw_w[sp_base[0]:sp_base[1]].reshape(PATCH_H, PATCH_W)
    vc_h = vcos[sp_base[0]:sp_base[1]].reshape(PATCH_H, PATCH_W)

    base_agr = _cam_agreement([gc_h, ra_h, rw_h, vc_h])

    # ── Left wrist cam — agreement + Gini ─────────────────────────────────────
    _nan4  = {k: float("nan") for k in ("cosine", "iou", "com_dist", "emd")}
    _nan_g = (float("nan"),) * 4
    if sp_wrists:
        sp_w, cam_w = sp_wrists[0], CAM_WRISTS[0]
        gc_hw = np.abs(to_2d_heatmap(
            np.asarray(rec[f"outputs/debug/gradcam/image/{cam_w}"]), cam_w))
        ra_hw = np.abs(to_2d_heatmap(
            np.asarray(rec[f"outputs/debug/raw_alpha/summation/image/{cam_w}"]), cam_w))
        rw_hw = raw_w[sp_w[0]:sp_w[1]].reshape(PATCH_H, PATCH_W)
        vc_hw = vcos[sp_w[0]:sp_w[1]].reshape(PATCH_H, PATCH_W)
        wrist_agr  = _cam_agreement([gc_hw, ra_hw, rw_hw, vc_hw])
        wrist_gini = (_gini(gc_hw.ravel()), _gini(ra_hw.ravel()),
                      _gini(rw_hw.ravel()), _gini(vc_hw.ravel()))
    else:
        wrist_agr  = _nan4
        wrist_gini = _nan_g

    # ── Task tokens — agreement + Gini ───────────────────────────────────────
    if sp_task:
        gc_task = np.abs(np.asarray(rec["outputs/debug/gradcam/task"])[0])
        ra_task = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/task"])[0])
        rw_task = raw_w[sp_task[0]:sp_task[1]]
        vc_task = vcos[sp_task[0]:sp_task[1]]
        task_agr  = _tok_agreement([gc_task, ra_task, rw_task, vc_task])
        task_gini = (_gini(gc_task), _gini(ra_task), _gini(rw_task), _gini(vc_task))
    else:
        task_agr  = _nan4
        task_gini = _nan_g

    # ── State tokens — agreement + Gini ──────────────────────────────────────
    if sp_state:
        gc_state = np.abs(np.asarray(rec["outputs/debug/gradcam/state"])[0])
        ra_state = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/state"])[0])
        rw_state = raw_w[sp_state[0]:sp_state[1]]
        vc_state = vcos[sp_state[0]:sp_state[1]]
        state_agr  = _tok_agreement([gc_state, ra_state, rw_state, vc_state])
        state_gini = (_gini(gc_state), _gini(ra_state), _gini(rw_state), _gini(vc_state))
    else:
        state_agr  = _nan4
        state_gini = _nan_g

    return {
        # Gini — base cam
        "gini_base_gc":      _gini(gc_h.ravel()),
        "gini_base_ra":      _gini(ra_h.ravel()),
        "gini_base_rw":      _gini(rw_h.ravel()),
        "gini_base_vc":      _gini(vc_h.ravel()),
        # Gini — wrist cam
        "gini_wrist_gc":     wrist_gini[0],
        "gini_wrist_ra":     wrist_gini[1],
        "gini_wrist_rw":     wrist_gini[2],
        "gini_wrist_vc":     wrist_gini[3],
        # Gini — task tokens
        "gini_task_gc":      task_gini[0],
        "gini_task_ra":      task_gini[1],
        "gini_task_rw":      task_gini[2],
        "gini_task_vc":      task_gini[3],
        # Gini — state tokens
        "gini_state_gc":     state_gini[0],
        "gini_state_ra":     state_gini[1],
        "gini_state_rw":     state_gini[2],
        "gini_state_vc":     state_gini[3],
        # modality fractions
        "base_frac":         base_frac,
        "wrist_frac":        wrist_frac,
        "task_frac":         task_frac,
        "state_frac":        state_frac,
        # inter-head agreement
        "task_head_agr":     task_head_agr,
        # action oscillation
        "trans_flip":        trans_flip,
        "rot_flip":          rot_flip,
        "grip_flip":         grip_flip,
        # attribution agreement — base cam
        "base_cosine":       base_agr["cosine"],
        "base_iou":          base_agr["iou"],
        "base_com_dist":     base_agr["com_dist"],
        "base_emd":          base_agr["emd"],
        # attribution agreement — wrist cam
        "wrist_cosine":      wrist_agr["cosine"],
        "wrist_iou":         wrist_agr["iou"],
        "wrist_com_dist":    wrist_agr["com_dist"],
        "wrist_emd":         wrist_agr["emd"],
        # attribution agreement — task tokens
        "task_cosine":       task_agr["cosine"],
        "task_iou":          task_agr["iou"],
        "task_com_dist":     task_agr["com_dist"],
        "task_emd":          task_agr["emd"],
        # attribution agreement — state tokens
        "state_cosine":      state_agr["cosine"],
        "state_iou":         state_agr["iou"],
        "state_com_dist":    state_agr["com_dist"],
        "state_emd":         state_agr["emd"],
        "_state":            state,
        "_action0":          actions[0].copy(),
    }


def load_episode(ep: dict) -> List[dict]:
    records      = []
    prev_state   = None
    prev_action0 = None

    for rel in range(MAX_STEPS):
        s = ep["start_idx"] + rel
        if s > ep["end_idx"]:
            break
        fpath = DATA_DIR / f"step_{s}.npy"
        if not fpath.exists():
            continue

        rec   = np.load(fpath, allow_pickle=True).item()
        feats = extract(rec)

        state   = feats.pop("_state")
        action0 = feats.pop("_action0")

        feats["joint_vel"] = (
            float(np.linalg.norm(state - prev_state)) if prev_state is not None else 0.0
        )
        if prev_action0 is not None:
            na = np.linalg.norm(action0); nb = np.linalg.norm(prev_action0)
            feats["plan_consistency"] = float(np.dot(action0, prev_action0) / (na * nb + 1e-9))
        else:
            feats["plan_consistency"] = 1.0


        feats["rel_step"] = rel
        records.append(feats)
        prev_state   = state
        prev_action0 = action0

    return records


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

EpData = Dict[str, List[List[dict]]]  # "success" / "failure" → list of episode records


def _to_matrix(ep_list: List[List[dict]], key: str) -> np.ndarray:
    max_len = max(len(ep) for ep in ep_list)
    out = np.full((len(ep_list), max_len), np.nan)
    for i, ep in enumerate(ep_list):
        vals = [r[key] for r in ep]
        out[i, :len(vals)] = vals
    return out


def _plot_row(ax, ep_data: EpData, key: str, ylabel: str, title: str):
    for label, cm, ci in [
        ("success", COL_S_MEAN, COL_S_IND),
        ("failure", COL_F_MEAN, COL_F_IND),
    ]:
        eps = ep_data.get(label, [])
        if not eps:
            continue
        arr = _to_matrix(eps, key)
        xs  = np.arange(arr.shape[1])
        for row in arr:
            valid = ~np.isnan(row)
            ax.plot(xs[valid], row[valid], color=ci, lw=LW_IND, alpha=ALPHA_IND)
        mean  = np.nanmean(arr, axis=0)
        valid = ~np.isnan(mean)
        ax.plot(xs[valid], mean[valid], color=cm, lw=LW_MEAN, label=label)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")


def _plot_variance_row(ax, ep_data: EpData, key: str, ylabel: str, title: str):
    """Plot per-step inter-episode variance of a metric for each outcome group."""
    for label, cm in [("success", COL_S_MEAN), ("failure", COL_F_MEAN)]:
        eps = ep_data.get(label, [])
        if not eps:
            continue
        arr = _to_matrix(eps, key)
        xs  = np.arange(arr.shape[1])
        var  = np.nanvar(arr, axis=0)
        valid = ~np.isnan(var)
        ax.plot(xs[valid], var[valid], color=cm, lw=LW_MEAN, label=label)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")


def _make_fig(n_rows: int, suptitle: str):
    fig, axes = plt.subplots(n_rows, 1, figsize=(FIG_W, ROW_H * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.array([axes])
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    return fig, axes


def _save(fig, axes, fname: str):
    axes[-1].set_xlabel("Step within episode")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _gini_fig(ep_data: EpData, modality: str, label: str, fname: str):
    specs = [
        (f"gini_{modality}_gc", "Gini", "GradCAM"),
        (f"gini_{modality}_ra", "Gini", "Raw Alpha"),
        (f"gini_{modality}_rw", "Gini", "Raw Weights"),
        (f"gini_{modality}_vc", "Gini", "V-Cosine"),
    ]
    fig, axes = _make_fig(4, f"Attention Concentration — Gini per Method ({label})")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, fname)


def fig1_attention_concentration(ep_data: EpData):
    _gini_fig(ep_data, "base",  "Base Camera",        "fig1a_gini_base.png")
    _gini_fig(ep_data, "wrist", "Left Wrist Camera",  "fig1b_gini_wrist.png")
    _gini_fig(ep_data, "task",  "Task Tokens",         "fig1c_gini_task.png")
    _gini_fig(ep_data, "state", "State Tokens",        "fig1d_gini_state.png")


def fig2_attention_routing(ep_data: EpData):
    specs = [
        ("base_frac",  "Attn fraction", "Base camera attention fraction"),
        ("wrist_frac", "Attn fraction", "Left wrist camera attention fraction"),
        ("task_frac",  "Attn fraction", "Task token attention fraction"),
        ("state_frac", "Attn fraction", "State token attention fraction"),
    ]
    fig, axes = _make_fig(len(specs), "Attention Routing — Modality Fractions (sum to ~1 minus other tokens)")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, "fig2_attention_routing.png")


def fig_inter_head_agreement(ep_data: EpData):
    fig, axes = _make_fig(1, "Task-Token Inter-Head Agreement (Heatmap-Selected Heads)")
    _plot_row(axes[0], ep_data, "task_head_agr", "Cosine sim",
              "Mean pairwise cosine similarity of task-token attention across top-k image heads")
    _save(fig, axes, "fig_inter_head_agreement.png")


def fig3_action_oscillation(ep_data: EpData):
    specs = [
        ("trans_flip",       "Flip rate",  "Translation sign-flip rate within horizon (dims 0–2)"),
        ("rot_flip",         "Flip rate",  "Rotation sign-flip rate within horizon (dims 3–5)"),
        ("grip_flip",        "Flip rate",  "Gripper sign-flip rate within horizon (dim 6)"),
        ("plan_consistency", "Cosine sim", "Inter-step plan consistency (cos sim of consecutive step-0 actions)"),
    ]
    fig, axes = _make_fig(len(specs), "Action Oscillation & Plan Stability")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, "fig3_action_oscillation.png")


def fig4_motion_consensus(ep_data: EpData):
    fig, axes = _make_fig(1, "Motion")
    _plot_row(axes[0], ep_data, "joint_vel", "L2 norm", "Joint velocity (‖state_t − state_{t−1}‖₂)")
    _save(fig, axes, "fig4_motion_consensus.png")


def _attribution_fig(ep_data: EpData, cam_prefix: str, cam_label: str, fname: str):
    """4-row attribution agreement figure for one camera."""
    specs = [
        (f"{cam_prefix}_cosine",   "Cosine sim",     "Cosine similarity  (↑ more similar)"),
        (f"{cam_prefix}_iou",      "Top-10% IoU",    f"Top-{TOPK_PATCHES} patch IoU  (↑ more similar)"),
        (f"{cam_prefix}_com_dist", "Dist (patches)", "Centre-of-mass distance  (↓ more similar)"),
        (f"{cam_prefix}_emd",      "EMD (patches)",  "Earth Mover's Distance — x/y marginals  (↓ more similar)"),
    ]
    fig, axes = _make_fig(len(specs), f"Attribution Method Agreement — {cam_label} (mean across all method pairs)")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, fname)


def _tok_attribution_fig(ep_data: EpData, prefix: str, label: str, fname: str):
    """4-row attribution agreement figure for one token modality (task or state)."""
    specs = [
        (f"{prefix}_cosine",   "Cosine sim",    "Cosine similarity  (↑ more similar)"),
        (f"{prefix}_iou",      "Top-10% IoU",   "Top-10% token IoU  (↑ more similar)"),
        (f"{prefix}_com_dist", "Dist (tokens)", "Centre-of-mass distance  (↓ more similar)"),
        (f"{prefix}_emd",      "EMD (tokens)",  "Earth Mover's Distance  (↓ more similar)"),
    ]
    fig, axes = _make_fig(len(specs), f"Attribution Method Agreement — {label} (mean across all method pairs)")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, fname)


def fig5_base_attribution(ep_data: EpData):
    _attribution_fig(ep_data, "base", "Base Camera", "fig5_base_attribution.png")


def fig6_wrist_attribution(ep_data: EpData):
    _attribution_fig(ep_data, "wrist", "Left Wrist Camera", "fig6_wrist_attribution.png")


def fig7_task_attribution(ep_data: EpData):
    _tok_attribution_fig(ep_data, "task", "Task Tokens", "fig7_task_attribution.png")


def fig8_state_attribution(ep_data: EpData):
    _tok_attribution_fig(ep_data, "state", "State Tokens", "fig8_state_attribution.png")




# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(EPISODE_JSON) as f:
        all_eps = json.load(f)

    mixed = [e for e in all_eps if e["task_id"] in MIXED_TASK_IDS]
    ep_data: EpData = {"success": [], "failure": []}

    print(f"Loading {len(mixed)} episodes …")
    for i, ep in enumerate(mixed, 1):
        label = "success" if ep["success"] else "failure"
        print(f"  [{i:2d}/{len(mixed)}] task {ep['task_id']} ep{ep['episode_num']:02d} ({label})",
              end=" … ", flush=True)
        records = load_episode(ep)
        ep_data[label].append(records)
        print(f"{len(records)} steps")

    ns = len(ep_data["success"]); nf = len(ep_data["failure"])
    print(f"\n{ns} success episodes, {nf} failure episodes.\n")

    print("Rendering figures …")
    fig1_attention_concentration(ep_data)
    fig2_attention_routing(ep_data)
    fig_inter_head_agreement(ep_data)
    fig3_action_oscillation(ep_data)
    fig4_motion_consensus(ep_data)
    fig5_base_attribution(ep_data)
    fig6_wrist_attribution(ep_data)
    fig7_task_attribution(ep_data)
    fig8_state_attribution(ep_data)
    print("Done.")


if __name__ == "__main__":
    main()
