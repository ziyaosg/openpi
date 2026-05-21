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
    select_heads_for_span, TARGET_LAYER, TOP_K_HEADS,
)
from scripts.attention_visualization.grid_visualization_value_attn import score_v_cosine
from scripts.attention_visualization.grid_visualization_gradcam import to_2d_heatmap
from scripts.attention_utils.keys import CAM_NAMES, IMAGE_PATCH_GRID

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260516_034939")
EPISODE_JSON = DATA_DIR / "client_output" / "episode_summaries.json"
OUTPUT_DIR   = Path("/nfs/roberts/scratch/pi_tkf6/zs377/visualization_pi0_fast_libero_20260516_034939/analysis")

FILTER_MIXED_TASKS = False   # only tasks with both success and failure episodes
ACTION_TOKEN = 0            # which action token index to use for gradcam/raw_alpha
MAX_STEPS = 60

PATCH_H, PATCH_W = IMAGE_PATCH_GRID
CAM_BASE   = "base_0_rgb"
CAM_LWRIST = "left_wrist_0_rgb"
# right_wrist_0_rgb is NOT a real camera for single-arm libero tasks: the model
# receives the same wrist_image in both wrist slots (cosine ≈ 0.905 confirmed).
# All right-wrist features are noise on a repeated signal — excluded intentionally.

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"
LAYERS_KEY    = "outputs/debug/attn/layers"

METHOD_NAMES  = ["GradCAM", "Raw Alpha", "Raw Weights", "V-Cosine"]
METHOD_SHORT  = ["gc",      "ra",        "rw",           "vc"]
METHOD_PAIRS  = list(combinations(range(len(METHOD_NAMES)), 2))
PAIR_NAMES    = [f"{METHOD_SHORT[i]}_{METHOD_SHORT[j]}" for i, j in METHOD_PAIRS]
PAIR_LABELS   = [f"{METHOD_NAMES[i]} vs {METHOD_NAMES[j]}" for i, j in METHOD_PAIRS]

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


def _state_digit_slice(rec: dict) -> Tuple[int, Optional[int]]:
    """(k, end): score_array[k:end] covers only state digit tokens.

    The state score arrays span "State: {digits};\\n" including syntactic
    prefix ("State", ":") and suffix (";", "\\n") tokens that receive high
    model attention as structural markers but carry no per-joint information.
    Returns (0, None) when piece metadata is absent (use full array).
    """
    pb = rec.get("outputs/debug/tokens/state/piece_begin")
    pe = rec.get("outputs/debug/tokens/state/piece_end")
    if pb is None or pe is None:
        return 0, None
    piece_begin = np.asarray(pb)
    piece_end   = np.asarray(pe)
    max_byte      = int(piece_end.max())
    CONTENT_START = 7            # len("State: ")
    CONTENT_END   = max_byte - 2  # excl. ";\n"
    k = int((piece_end   <= CONTENT_START).sum())
    s = int((piece_begin >= CONTENT_END).sum())
    return k, len(piece_begin) - s


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
    """Compute each metric for every method pair separately for one camera.
    Returns {pair_name: {metric: value}}."""
    result = {}
    for (i, j), pname in zip(METHOD_PAIRS, PAIR_NAMES):
        result[pname] = {
            "cosine":   _cos_sim(heats[i], heats[j]),
            "iou":      _topk_iou(heats[i], heats[j]),
            "com_dist": _com_distance(heats[i], heats[j]),
            "emd":      _emd(heats[i], heats[j]),
        }
    return result


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
    """Compute each 1-D metric for every method pair separately for one token modality.
    Returns {pair_name: {metric: value}}."""
    k = max(1, int(round(len(scores[0].ravel()) * 0.10)))
    result = {}
    for (i, j), pname in zip(METHOD_PAIRS, PAIR_NAMES):
        result[pname] = {
            "cosine":   _cos_sim(scores[i], scores[j]),
            "iou":      _topk_iou_k(scores[i], scores[j], k),
            "com_dist": _com_distance_1d(scores[i], scores[j]),
            "emd":      _emd_1d(scores[i], scores[j]),
        }
    return result


def _layer_indices(rec: dict) -> int:
    stored = np.asarray(rec[LAYERS_KEY]).tolist()
    return stored.index(TARGET_LAYER)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (per step)
# ─────────────────────────────────────────────────────────────────────────────

def extract(rec: dict) -> dict:
    full_attn = np.asarray(rec[FULL_ATTN_KEY])     # (L, 1, H, S)
    full_v    = np.asarray(rec[FULL_V_KEY])         # (L, 1, S, K, D)
    actions   = np.asarray(rec["outputs/actions"])  # (10, 7)
    state     = np.asarray(rec["outputs/state"])    # (8,)

    sp_base   = _span(rec, f"outputs/debug/spans/image/{CAM_BASE}")
    sp_lwrist = _span(rec, f"outputs/debug/spans/image/{CAM_LWRIST}")
    sp_task   = _span(rec, "outputs/debug/spans/task")
    sp_state  = _span(rec, "outputs/debug/spans/state")
    # sp_rwrist deliberately excluded: right_wrist slot = repeated wrist_image, not real camera

    tgt_idx  = _layer_indices(rec)
    attn_tgt = full_attn[tgt_idx, 0]  # (H, S)
    v_tgt    = full_v[tgt_idx, 0]     # (S, K, D)

    # ── Per-modality one-vs-rest head selection at TARGET_LAYER ───────────────
    all_spans: List[Tuple[int, int]] = [s for s in
        [sp_base, sp_lwrist, sp_task, sp_state] if s is not None]

    base_heads   = select_heads_for_span(attn_tgt, sp_base,   all_spans, TOP_K_HEADS)
    lwrist_heads = select_heads_for_span(attn_tgt, sp_lwrist, all_spans, TOP_K_HEADS) if sp_lwrist else None
    task_heads   = select_heads_for_span(attn_tgt, sp_task,   all_spans, TOP_K_HEADS) if sp_task   else None
    state_heads  = select_heads_for_span(attn_tgt, sp_state,  all_spans, TOP_K_HEADS) if sp_state  else None

    # ── Attention concentration (base cam, mean over all heads) ───────────────
    base_attn = attn_tgt[:, sp_base[0]:sp_base[1]].mean(axis=0)  # (N_patches,)
    gini_base = _gini(base_attn)
    p = base_attn / (base_attn.sum() + 1e-12)
    spatial_ent = float(-np.sum(p * np.log(p + 1e-12)))

    # ── Attention modality fractions ──────────────────────────────────────────
    def _frac(sp):
        return float(attn_tgt[:, sp[0]:sp[1]].sum(axis=1).mean()) if sp else 0.0
    base_frac   = _frac(sp_base)
    lwrist_frac = _frac(sp_lwrist)
    task_frac   = _frac(sp_task)
    state_frac  = _frac(sp_state)

    # ── Task-token inter-head agreement (task-focused heads) ──────────────────
    if sp_task and task_heads is not None and len(task_heads) > 1:
        h_task = attn_tgt[task_heads][:, sp_task[0]:sp_task[1]]  # (K, N_task)
        h_n = h_task / (np.linalg.norm(h_task, axis=1, keepdims=True) + 1e-9)
        sim = h_n @ h_n.T                                         # (K, K)
        iu = np.triu_indices(len(task_heads), k=1)
        task_head_agr = float(sim[iu].mean())
    else:
        task_head_agr = float("nan")

    # ── Action sign flips within predicted horizon ────────────────────────────
    signs = np.sign(actions)                          # (10, 7)
    flips = (signs[1:] != signs[:-1]).astype(float)  # (9, 7)
    trans_flip = float(flips[:, 0:3].mean())   # translation dims 0-2
    rot_flip   = float(flips[:, 3:6].mean())   # rotation dims 3-5
    grip_flip  = float(flips[:, 6].mean())     # gripper dim 6

    # ── Shared nan placeholders ───────────────────────────────────────────────
    _nan_metrics = {"cosine": float("nan"), "iou": float("nan"), "com_dist": float("nan"), "emd": float("nan")}
    _nan4  = {pname: _nan_metrics for pname in PAIR_NAMES}
    _nan_g = (float("nan"),) * 4

    # ── Attribution agreement — base cam ──────────────────────────────────────
    gc_base = np.abs(to_2d_heatmap(
        np.asarray(rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/{CAM_BASE}"]), CAM_BASE))
    ra_base = np.abs(to_2d_heatmap(
        np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/{CAM_BASE}"]), CAM_BASE))
    rw_base = attn_tgt[base_heads, sp_base[0]:sp_base[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
    vc_base = score_v_cosine(attn_tgt, v_tgt, base_heads)[sp_base[0]:sp_base[1]].reshape(PATCH_H, PATCH_W)
    base_agr = _cam_agreement([gc_base, ra_base, rw_base, vc_base])
    base_gini = (_gini(gc_base.ravel()), _gini(ra_base.ravel()),
                 _gini(rw_base.ravel()), _gini(vc_base.ravel()))

    # ── Attribution agreement — left wrist cam ────────────────────────────────
    gc_lwrist = ra_lwrist = rw_lwrist = vc_lwrist = None
    if sp_lwrist and lwrist_heads is not None:
        gc_lwrist = np.abs(to_2d_heatmap(
            np.asarray(rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/{CAM_LWRIST}"]), CAM_LWRIST))
        ra_lwrist = np.abs(to_2d_heatmap(
            np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/{CAM_LWRIST}"]), CAM_LWRIST))
        rw_lwrist = attn_tgt[lwrist_heads, sp_lwrist[0]:sp_lwrist[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
        vc_lwrist = score_v_cosine(attn_tgt, v_tgt, lwrist_heads)[sp_lwrist[0]:sp_lwrist[1]].reshape(PATCH_H, PATCH_W)
        lwrist_agr  = _cam_agreement([gc_lwrist, ra_lwrist, rw_lwrist, vc_lwrist])
        lwrist_gini = (_gini(gc_lwrist.ravel()), _gini(ra_lwrist.ravel()),
                       _gini(rw_lwrist.ravel()), _gini(vc_lwrist.ravel()))
    else:
        lwrist_agr  = _nan4
        lwrist_gini = _nan_g

    # ── Attribution agreement — task tokens ───────────────────────────────────
    if sp_task and task_heads is not None:
        gc_task = np.abs(np.asarray(rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/task"])[0])
        ra_task = np.abs(np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/task"])[0])
        rw_task = attn_tgt[task_heads, sp_task[0]:sp_task[1]].max(axis=0)
        vc_task = score_v_cosine(attn_tgt, v_tgt, task_heads)[sp_task[0]:sp_task[1]]
        task_agr  = _tok_agreement([gc_task, ra_task, rw_task, vc_task])
        task_gini = (_gini(gc_task), _gini(ra_task), _gini(rw_task), _gini(vc_task))
    else:
        task_agr  = _nan4
        task_gini = _nan_g

    # ── Attribution agreement — state tokens (digit-only) ────────────────────
    # Score arrays cover the full "State: {digits};\n" segment; slice to digits
    # only to avoid inflating Gini with high-attention syntactic prefix/suffix.
    if sp_state and state_heads is not None:
        dk, dend = _state_digit_slice(rec)
        gc_state_full = np.abs(np.asarray(rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN}/state"])[0])
        ra_state_full = np.abs(np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN}/state"])[0])
        rw_state_full = attn_tgt[state_heads, sp_state[0]:sp_state[1]].max(axis=0)
        vc_state_full = score_v_cosine(attn_tgt, v_tgt, state_heads)[sp_state[0]:sp_state[1]]
        gc_state = gc_state_full[dk:dend]
        ra_state = ra_state_full[dk:dend]
        rw_state = rw_state_full[dk:dend]
        vc_state = vc_state_full[dk:dend]
        state_agr  = _tok_agreement([gc_state, ra_state, rw_state, vc_state])
        state_gini = (_gini(gc_state), _gini(ra_state), _gini(rw_state), _gini(vc_state))
    else:
        state_agr  = _nan4
        state_gini = _nan_g

    return {
        # Gini — base cam
        "gini_base_gc":      base_gini[0],
        "gini_base_ra":      base_gini[1],
        "gini_base_rw":      base_gini[2],
        "gini_base_vc":      base_gini[3],
        # Gini — left wrist cam (the only real wrist camera for single-arm tasks)
        "gini_lwrist_gc":    lwrist_gini[0],
        "gini_lwrist_ra":    lwrist_gini[1],
        "gini_lwrist_rw":    lwrist_gini[2],
        "gini_lwrist_vc":    lwrist_gini[3],
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
        "lwrist_frac":       lwrist_frac,
        "task_frac":         task_frac,
        "state_frac":        state_frac,
        # inter-head agreement
        "task_head_agr":     task_head_agr,
        # action oscillation
        "trans_flip":        trans_flip,
        "rot_flip":          rot_flip,
        "grip_flip":         grip_flip,
        # attribution agreement — per pair, per modality
        **{f"base_{pname}_{m}":   base_agr[pname][m]   for pname in PAIR_NAMES for m in ("cosine", "iou", "com_dist", "emd")},
        **{f"lwrist_{pname}_{m}": lwrist_agr[pname][m] for pname in PAIR_NAMES for m in ("cosine", "iou", "com_dist", "emd")},
        **{f"task_{pname}_{m}":   task_agr[pname][m]   for pname in PAIR_NAMES for m in ("cosine", "iou", "com_dist", "emd")},
        **{f"state_{pname}_{m}":  state_agr[pname][m]  for pname in PAIR_NAMES for m in ("cosine", "iou", "com_dist", "emd")},
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

        try:
            rec   = np.load(fpath, allow_pickle=True).item()
            feats = extract(rec)
        except Exception as e:
            print(f"    [warn] step {s}: {e}")
            continue

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
        eps = [e for e in ep_data.get(label, []) if e]
        if not eps:
            continue
        arr = _to_matrix(eps, key)
        xs  = np.arange(arr.shape[1])
        for row in arr:
            valid = ~np.isnan(row)
            ax.plot(xs[valid], row[valid], color=ci, lw=LW_IND, alpha=ALPHA_IND, zorder=2)
        mean  = np.nanmean(arr, axis=0)
        valid = ~np.isnan(mean)
        ax.plot(xs[valid], mean[valid], color=cm, lw=LW_MEAN, label=label, zorder=3)

    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right")


def _plot_variance_row(ax, ep_data: EpData, key: str, ylabel: str, title: str):
    """Plot per-step inter-episode variance of a metric for each outcome group."""
    for label, cm in [("success", COL_S_MEAN), ("failure", COL_F_MEAN)]:
        eps = [e for e in ep_data.get(label, []) if e]
        if not eps:
            continue
        arr = _to_matrix(eps, key)
        xs  = np.arange(arr.shape[1])
        var  = np.nanvar(arr, axis=0)
        valid = ~np.isnan(var)
        ax.plot(xs[valid], var[valid], color=cm, lw=LW_MEAN, label=label, zorder=3)

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
    _gini_fig(ep_data, "base",   "Base Camera",       "fig1a_gini_base.png")
    _gini_fig(ep_data, "lwrist", "Left Wrist Camera", "fig1b_gini_lwrist.png")
    _gini_fig(ep_data, "task",   "Task Tokens",       "fig1c_gini_task.png")
    _gini_fig(ep_data, "state",  "State Tokens",      "fig1d_gini_state.png")


def fig2_attention_routing(ep_data: EpData):
    specs = [
        ("base_frac",   "Attn fraction", "Base camera attention fraction"),
        ("lwrist_frac", "Attn fraction", "Wrist camera attention fraction"),
        ("task_frac",   "Attn fraction", "Task token attention fraction"),
        ("state_frac",  "Attn fraction", "State token attention fraction"),
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


def _attribution_fig(ep_data: EpData, cam_prefix: str, cam_label: str, fname_prefix: str):
    """One 4-row figure per method pair for one camera."""
    for pname, plabel in zip(PAIR_NAMES, PAIR_LABELS):
        specs = [
            (f"{cam_prefix}_{pname}_cosine",   "Cosine sim",     "Cosine similarity  (↑ more similar)"),
            (f"{cam_prefix}_{pname}_iou",      "Top-10% IoU",    f"Top-{TOPK_PATCHES} patch IoU  (↑ more similar)"),
            (f"{cam_prefix}_{pname}_com_dist", "Dist (patches)", "Centre-of-mass distance  (↓ more similar)"),
            (f"{cam_prefix}_{pname}_emd",      "EMD (patches)",  "Earth Mover's Distance — x/y marginals  (↓ more similar)"),
        ]
        fig, axes = _make_fig(len(specs), f"Attribution Agreement — {cam_label} ({plabel})")
        for ax, (key, yl, title) in zip(axes, specs):
            _plot_row(ax, ep_data, key, yl, title)
        _save(fig, axes, f"{fname_prefix}_{pname}.png")


def _tok_attribution_fig(ep_data: EpData, prefix: str, label: str, fname_prefix: str):
    """One 4-row figure per method pair for one token modality (task or state)."""
    for pname, plabel in zip(PAIR_NAMES, PAIR_LABELS):
        specs = [
            (f"{prefix}_{pname}_cosine",   "Cosine sim",    "Cosine similarity  (↑ more similar)"),
            (f"{prefix}_{pname}_iou",      "Top-10% IoU",   "Top-10% token IoU  (↑ more similar)"),
            (f"{prefix}_{pname}_com_dist", "Dist (tokens)", "Centre-of-mass distance  (↓ more similar)"),
            (f"{prefix}_{pname}_emd",      "EMD (tokens)",  "Earth Mover's Distance  (↓ more similar)"),
        ]
        fig, axes = _make_fig(len(specs), f"Attribution Agreement — {label} ({plabel})")
        for ax, (key, yl, title) in zip(axes, specs):
            _plot_row(ax, ep_data, key, yl, title)
        _save(fig, axes, f"{fname_prefix}_{pname}.png")


def fig5_base_attribution(ep_data: EpData):
    _attribution_fig(ep_data, "base", "Base Camera", "fig5_base_attribution")


def fig6_lwrist_attribution(ep_data: EpData):
    _attribution_fig(ep_data, "lwrist", "Left Wrist Camera", "fig6_lwrist_attribution")


def fig7_task_attribution(ep_data: EpData):
    _tok_attribution_fig(ep_data, "task", "Task Tokens", "fig7_task_attribution")


def fig8_state_attribution(ep_data: EpData):
    _tok_attribution_fig(ep_data, "state", "State Tokens", "fig8_state_attribution")






# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(EPISODE_JSON) as f:
        all_eps = json.load(f)

    if FILTER_MIXED_TASKS:
        from collections import defaultdict
        outcomes: dict = defaultdict(set)
        for ep in all_eps:
            outcomes[ep["task_id"]].add(ep["success"])
        mixed_ids = {tid for tid, outs in outcomes.items() if len(outs) == 2}
        episodes = [ep for ep in all_eps if ep["task_id"] in mixed_ids]
        print(f"Mixed-outcome tasks: {sorted(mixed_ids)}")
        print(f"Episodes to process: {len(episodes)} / {len(all_eps)}")
    else:
        episodes = all_eps

    ep_data: EpData = {"success": [], "failure": []}

    print(f"Loading {len(episodes)} episodes …")
    for i, ep in enumerate(episodes, 1):
        label = "success" if ep["success"] else "failure"
        print(f"  [{i:3d}/{len(episodes)}] task {ep['task_id']} ep{ep['episode_num']:03d} ({label})",
              end=" … ", flush=True)
        records = load_episode(ep)
        if records:
            ep_data[label].append(records)
        else:
            print(f"0 steps — skipped (no usable step files found)")
            continue
        print(f"{len(records)} steps")

    ns = len(ep_data["success"]); nf = len(ep_data["failure"])
    print(f"\n{ns} success episodes, {nf} failure episodes.\n")

    print("Rendering figures …")
    fig1_attention_concentration(ep_data)   # base / lwrist / task / state gini
    fig2_attention_routing(ep_data)         # modality fractions
    fig_inter_head_agreement(ep_data)
    fig3_action_oscillation(ep_data)
    fig4_motion_consensus(ep_data)
    fig5_base_attribution(ep_data)          # base cam method-pair agreement
    fig6_lwrist_attribution(ep_data)        # wrist cam method-pair agreement
    fig7_task_attribution(ep_data)          # task token method-pair agreement
    fig8_state_attribution(ep_data)         # state token method-pair agreement
    print("Done.")


if __name__ == "__main__":
    main()
