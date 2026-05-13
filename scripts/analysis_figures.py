#!/usr/bin/env python3
"""Per-step analysis figures for pi05/libero policy records.

Computes step-level metrics for each episode, then plots success vs failure
comparisons. Figures produced:

  fig1a_gini_base.png       — Gini coefficient per attribution method (base camera)
  fig1b_gini_wrist.png      — Gini (left wrist camera)
  fig1c_gini_task.png       — Gini (task tokens)
  fig2_attention_routing.png — modality attention fractions over steps
  fig3_inter_head_agreement.png — task-token inter-head cosine similarity
  fig4_action_oscillation.png   — action sign-flip rates + plan consistency
  fig5_motion.png           — joint-state L2 velocity
  fig6a_base_attribution.png — attribution method agreement, base camera
  fig6b_wrist_attribution.png — attribution method agreement, wrist camera
  fig6c_task_attribution.png  — attribution method agreement, task tokens

Run:
  python3 scripts/pi05_vis/analysis_figures.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from .common import (
    CAM_NAMES, PATCH_H, PATCH_W,
    FULL_ATTN_KEY, FULL_V_KEY, LAYER_IDX, TOP_K_HEADS,
    get_attn_hs, get_v_skd, get_span, get_all_spans, select_heads,
    score_v_cosine, as_u8_rgb,
)

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR    = Path("/data/ziyao/policy_records_pi05_libero_20260512_182606")
OUTPUT_DIR   = Path("/data/ziyao/visualization_pi05_libero_20260512_182606/analysis")
EPISODE_JSON = INPUT_DIR / "client_output" / "episode_summaries.json"

MAX_STEPS = 60   # cap per episode to keep runtime manageable

# Only episodes from tasks that appear in both outcomes (success + failure).
# Set False when running with num_trials_per_task=1 (no task has both outcomes).
FILTER_MIXED_TASKS = True

METHOD_NAMES = ["GradCAM", "Raw Alpha", "Raw Weights", "V-Cosine"]
METHOD_SHORT = ["gc", "ra", "rw", "vc"]
METHOD_PAIRS = list(combinations(range(len(METHOD_NAMES)), 2))
PAIR_NAMES   = [f"{METHOD_SHORT[i]}_{METHOD_SHORT[j]}" for i, j in METHOD_PAIRS]
PAIR_LABELS  = [f"{METHOD_NAMES[i]} vs {METHOD_NAMES[j]}" for i, j in METHOD_PAIRS]

TOPK_PATCHES = 26    # top ~10% of 256 patches

# Plot styling
COL_S_MEAN = "#1f5fa6"
COL_F_MEAN = "#b02020"
COL_S_IND  = "#a8c8e8"
COL_F_IND  = "#f0a0a0"
ALPHA_IND  = 0.5
LW_MEAN    = 2.2
LW_IND     = 1.0
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
})

# ─── Statistical utilities ────────────────────────────────────────────────────

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


def _topk_iou(a: np.ndarray, b: np.ndarray, k: int = TOPK_PATCHES) -> float:
    a, b = a.ravel(), b.ravel()
    ta = set(np.argsort(a)[-k:].tolist())
    tb = set(np.argsort(b)[-k:].tolist())
    return len(ta & tb) / len(ta | tb)


def _com_dist_2d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a).astype(np.float64); b = np.abs(b).astype(np.float64)
    rows = np.arange(PATCH_H)[:, None]; cols = np.arange(PATCH_W)[None, :]
    sa, sb = a.sum() + 1e-12, b.sum() + 1e-12
    cy_a = float((a * rows).sum() / sa); cx_a = float((a * cols).sum() / sa)
    cy_b = float((b * rows).sum() / sb); cx_b = float((b * cols).sum() / sb)
    return float(np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2))


def _emd_2d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a.ravel()).astype(np.float64); b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    a /= sa; b /= sb
    a2, b2 = a.reshape(PATCH_H, PATCH_W), b.reshape(PATCH_H, PATCH_W)
    xs, ys = np.arange(PATCH_W), np.arange(PATCH_H)
    emd_x = wasserstein_distance(xs, xs, a2.sum(0), b2.sum(0))
    emd_y = wasserstein_distance(ys, ys, a2.sum(1), b2.sum(1))
    return float(0.5 * (emd_x + emd_y))


def _cam_agreement(heats: list) -> dict:
    result = {}
    for (i, j), pname in zip(METHOD_PAIRS, PAIR_NAMES):
        result[pname] = {
            "cosine":   _cos_sim(heats[i], heats[j]),
            "iou":      _topk_iou(heats[i], heats[j]),
            "com_dist": _com_dist_2d(heats[i], heats[j]),
            "emd":      _emd_2d(heats[i], heats[j]),
        }
    return result


def _com_dist_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a.ravel()).astype(np.float64)
    b = np.abs(b.ravel()).astype(np.float64)
    pos = np.arange(len(a), dtype=np.float64)
    ca = float((a * pos).sum() / (a.sum() + 1e-12))
    cb = float((b * pos).sum() / (b.sum() + 1e-12))
    return abs(ca - cb)


def _emd_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.abs(a.ravel()).astype(np.float64); b = np.abs(b.ravel()).astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    a /= sa; b /= sb
    pos = np.arange(len(a), dtype=np.float64)
    return float(wasserstein_distance(pos, pos, a, b))


def _tok_agreement(scores: list) -> dict:
    k = max(1, int(round(len(scores[0].ravel()) * 0.10)))
    result = {}
    for (i, j), pname in zip(METHOD_PAIRS, PAIR_NAMES):
        result[pname] = {
            "cosine":   _cos_sim(scores[i], scores[j]),
            "iou":      _topk_iou(scores[i], scores[j], k),
            "com_dist": _com_dist_1d(scores[i], scores[j]),
            "emd":      _emd_1d(scores[i], scores[j]),
        }
    return result


# ─── Feature extraction (per step) ───────────────────────────────────────────

_NAN4 = {pn: {"cosine": np.nan, "iou": np.nan, "com_dist": np.nan, "emd": np.nan}
          for pn in PAIR_NAMES}
_NAN_GINI4 = (np.nan, np.nan, np.nan, np.nan)

CAM_BASE  = "base_0_rgb"
CAM_WRIST = "left_wrist_0_rgb"


def extract(rec: dict, action_token: int = 0) -> dict:
    attn = get_attn_hs(rec, action_token=action_token)  # (H, S)
    v    = get_v_skd(rec)                               # (S, K, D)

    sp_base  = get_span(rec, f"outputs/debug/spans/image/{CAM_BASE}")
    sp_wrist = get_span(rec, f"outputs/debug/spans/image/{CAM_WRIST}")
    sp_task  = get_span(rec, "outputs/debug/spans/task")
    sp_state = get_span(rec, "outputs/debug/spans/state")

    all_spans = [s for s in [sp_base, sp_wrist, sp_task, sp_state] if s is not None]
    if len(all_spans) < 2:
        raise ValueError("Insufficient spans in record")

    base_heads  = select_heads(attn, sp_base,  all_spans, TOP_K_HEADS) if sp_base  else None
    wrist_heads = select_heads(attn, sp_wrist, all_spans, TOP_K_HEADS) if sp_wrist else None
    task_heads  = select_heads(attn, sp_task,  all_spans, TOP_K_HEADS) if sp_task  else None
    state_heads = select_heads(attn, sp_state, all_spans, TOP_K_HEADS) if sp_state else None

    # ── Gini: base camera ──────────────────────────────────────────────────────
    gc_base = np.abs(np.asarray(rec["outputs/debug/gradcam/image/base_0_rgb"], dtype=np.float32))
    ra_base = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/image/base_0_rgb"], dtype=np.float32))
    if sp_base and base_heads is not None:
        rw_base = attn[base_heads, sp_base[0]:sp_base[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
        vc_base = score_v_cosine(attn, v, base_heads)[sp_base[0]:sp_base[1]].reshape(PATCH_H, PATCH_W)
    else:
        rw_base = vc_base = np.zeros((PATCH_H, PATCH_W))
    gini_base = (_gini(gc_base.ravel()), _gini(ra_base.ravel()),
                 _gini(rw_base.ravel()), _gini(vc_base.ravel()))

    # ── Gini: wrist camera ─────────────────────────────────────────────────────
    if sp_wrist and wrist_heads is not None:
        gc_wrist = np.abs(np.asarray(rec["outputs/debug/gradcam/image/left_wrist_0_rgb"], dtype=np.float32))
        ra_wrist = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/image/left_wrist_0_rgb"], dtype=np.float32))
        rw_wrist = attn[wrist_heads, sp_wrist[0]:sp_wrist[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
        vc_wrist = score_v_cosine(attn, v, wrist_heads)[sp_wrist[0]:sp_wrist[1]].reshape(PATCH_H, PATCH_W)
        gini_wrist = (_gini(gc_wrist.ravel()), _gini(ra_wrist.ravel()),
                      _gini(rw_wrist.ravel()), _gini(vc_wrist.ravel()))
    else:
        gc_wrist = ra_wrist = rw_wrist = vc_wrist = None
        gini_wrist = _NAN_GINI4

    # ── Gini: task tokens ──────────────────────────────────────────────────────
    if sp_task and task_heads is not None:
        gc_task = np.abs(np.asarray(rec["outputs/debug/gradcam/task"], dtype=np.float32))
        ra_task = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/task"], dtype=np.float32))
        rw_task = attn[task_heads, sp_task[0]:sp_task[1]].max(axis=0)
        vc_task = score_v_cosine(attn, v, task_heads)[sp_task[0]:sp_task[1]]
        gini_task = (_gini(gc_task), _gini(ra_task), _gini(rw_task), _gini(vc_task))
    else:
        gc_task = ra_task = rw_task = vc_task = None
        gini_task = _NAN_GINI4

    # ── Gini: state tokens ─────────────────────────────────────────────────────
    if sp_state and state_heads is not None:
        gc_state = np.abs(np.asarray(rec["outputs/debug/gradcam/state"], dtype=np.float32))
        ra_state = np.abs(np.asarray(rec["outputs/debug/raw_alpha/summation/state"], dtype=np.float32))
        rw_state = attn[state_heads, sp_state[0]:sp_state[1]].max(axis=0)
        vc_state = score_v_cosine(attn, v, state_heads)[sp_state[0]:sp_state[1]]
        gini_state = (_gini(gc_state), _gini(ra_state), _gini(rw_state), _gini(vc_state))
    else:
        gc_state = ra_state = rw_state = vc_state = None
        gini_state = _NAN_GINI4

    # ── Attention modality fractions ───────────────────────────────────────────
    def _frac(sp):
        return float(attn[:, sp[0]:sp[1]].sum(axis=1).mean()) if sp else 0.0

    base_frac  = _frac(sp_base)
    wrist_frac = _frac(sp_wrist)
    task_frac  = _frac(sp_task)
    state_frac = _frac(sp_state)

    # ── Task-token inter-head agreement ────────────────────────────────────────
    if sp_task and task_heads is not None and len(task_heads) > 1:
        h_task = attn[task_heads][:, sp_task[0]:sp_task[1]]
        h_n = h_task / (np.linalg.norm(h_task, axis=1, keepdims=True) + 1e-9)
        sim = h_n @ h_n.T
        iu = np.triu_indices(len(task_heads), k=1)
        task_head_agr = float(sim[iu].mean())
    else:
        task_head_agr = np.nan

    # ── Action oscillation ─────────────────────────────────────────────────────
    actions = np.asarray(rec["outputs/actions"])   # (10, 7)
    signs   = np.sign(actions)
    flips   = (signs[1:] != signs[:-1]).astype(float)
    trans_flip = float(flips[:, 0:3].mean())
    rot_flip   = float(flips[:, 3:6].mean())
    grip_flip  = float(flips[:, 6].mean())

    # ── Attribution agreement — base camera ────────────────────────────────────
    if gc_wrist is not None:
        base_agr  = _cam_agreement([gc_base, ra_base, rw_base, vc_base])
        wrist_agr = _cam_agreement([gc_wrist, ra_wrist, rw_wrist, vc_wrist])
    else:
        base_agr  = _cam_agreement([gc_base, ra_base, rw_base, vc_base])
        wrist_agr = _NAN4

    # ── Attribution agreement — task tokens ────────────────────────────────────
    if gc_task is not None:
        task_agr = _tok_agreement([gc_task, ra_task, rw_task, vc_task])
    else:
        task_agr = _NAN4

    # ── Attribution agreement — state tokens ───────────────────────────────────
    if gc_state is not None:
        state_agr = _tok_agreement([gc_state, ra_state, rw_state, vc_state])
    else:
        state_agr = _NAN4

    state = np.asarray(rec.get("outputs/state", np.zeros(32)), dtype=np.float64)
    action0 = actions[0].copy()

    return {
        "gini_base_gc":  gini_base[0],  "gini_base_ra":  gini_base[1],
        "gini_base_rw":  gini_base[2],  "gini_base_vc":  gini_base[3],
        "gini_wrist_gc": gini_wrist[0], "gini_wrist_ra": gini_wrist[1],
        "gini_wrist_rw": gini_wrist[2], "gini_wrist_vc": gini_wrist[3],
        "gini_task_gc":  gini_task[0],  "gini_task_ra":  gini_task[1],
        "gini_task_rw":  gini_task[2],  "gini_task_vc":  gini_task[3],
        "gini_state_gc": gini_state[0], "gini_state_ra": gini_state[1],
        "gini_state_rw": gini_state[2], "gini_state_vc": gini_state[3],
        "base_frac": base_frac, "wrist_frac": wrist_frac,
        "task_frac": task_frac, "state_frac": state_frac,
        "task_head_agr": task_head_agr,
        "trans_flip": trans_flip, "rot_flip": rot_flip, "grip_flip": grip_flip,
        **{f"base_{pn}_{m}":  base_agr[pn][m]  for pn in PAIR_NAMES for m in ("cosine","iou","com_dist","emd")},
        **{f"wrist_{pn}_{m}": wrist_agr[pn][m] for pn in PAIR_NAMES for m in ("cosine","iou","com_dist","emd")},
        **{f"task_{pn}_{m}":  task_agr[pn][m]  for pn in PAIR_NAMES for m in ("cosine","iou","com_dist","emd")},
        **{f"state_{pn}_{m}": state_agr[pn][m] for pn in PAIR_NAMES for m in ("cosine","iou","com_dist","emd")},
        "_state":   state,
        "_action0": action0,
    }


def load_episode(ep: dict) -> List[dict]:
    records = []
    prev_state   = None
    prev_action0 = None

    for rel in range(MAX_STEPS):
        s = ep["start_idx"] + rel
        if s > ep["end_idx"]:
            break
        fpath = INPUT_DIR / f"step_{s}.npy"
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
            feats["plan_consistency"] = float(
                np.dot(action0, prev_action0) / (na * nb + 1e-9))
        else:
            feats["plan_consistency"] = 1.0

        feats["rel_step"] = rel
        records.append(feats)
        prev_state   = state
        prev_action0 = action0

    return records


# ─── Plot helpers ─────────────────────────────────────────────────────────────

EpData = Dict[str, List[List[dict]]]


def _to_matrix(ep_list: List[List[dict]], key: str) -> np.ndarray:
    max_len = max(len(ep) for ep in ep_list)
    out = np.full((len(ep_list), max_len), np.nan)
    for i, ep in enumerate(ep_list):
        vals = [r[key] for r in ep]
        out[i, :len(vals)] = vals
    return out


def _plot_row(ax, ep_data: EpData, key: str, ylabel: str, title: str):
    for label, cm, ci in [("success", COL_S_MEAN, COL_S_IND), ("failure", COL_F_MEAN, COL_F_IND)]:
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


# ─── Individual figure generators ────────────────────────────────────────────

def _gini_fig(ep_data: EpData, modality: str, label: str, fname: str):
    specs = [
        (f"gini_{modality}_gc", "Gini", "GradCAM"),
        (f"gini_{modality}_ra", "Gini", "Raw Alpha"),
        (f"gini_{modality}_rw", "Gini", "Raw Weights"),
        (f"gini_{modality}_vc", "Gini", "V-Cosine"),
    ]
    fig, axes = _make_fig(4, f"Attention Concentration — Gini ({label})")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, fname)


def fig1_gini(ep_data: EpData):
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
    fig, axes = _make_fig(len(specs), "Attention Routing — Modality Fractions")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, "fig2_attention_routing.png")


def fig3_inter_head_agreement(ep_data: EpData):
    fig, axes = _make_fig(1, "Task-Token Inter-Head Agreement")
    _plot_row(axes[0], ep_data, "task_head_agr", "Cosine sim",
              "Mean pairwise cosine similarity of task-token attention across top-k heads")
    _save(fig, axes, "fig3_inter_head_agreement.png")


def fig4_action_oscillation(ep_data: EpData):
    specs = [
        ("trans_flip",       "Flip rate",  "Translation sign-flip rate within horizon (dims 0–2)"),
        ("rot_flip",         "Flip rate",  "Rotation sign-flip rate within horizon (dims 3–5)"),
        ("grip_flip",        "Flip rate",  "Gripper sign-flip rate (dim 6)"),
        ("plan_consistency", "Cosine sim", "Inter-step plan consistency (cos of consecutive step-0 actions)"),
    ]
    fig, axes = _make_fig(len(specs), "Action Oscillation & Plan Stability")
    for ax, (key, yl, title) in zip(axes, specs):
        _plot_row(ax, ep_data, key, yl, title)
    _save(fig, axes, "fig4_action_oscillation.png")


def fig5_motion(ep_data: EpData):
    fig, axes = _make_fig(1, "Motion")
    _plot_row(axes[0], ep_data, "joint_vel", "L2 norm",
              "Joint velocity (‖state_t − state_{t−1}‖₂)")
    _save(fig, axes, "fig5_motion.png")


def _attribution_fig(ep_data: EpData, prefix: str, label: str, is_2d: bool, fname_prefix: str):
    m_keys = ("cosine", "iou", "com_dist", "emd")
    m_labels = (
        "Cosine similarity  (↑ more similar)",
        f"Top-{TOPK_PATCHES if is_2d else '10%'} IoU  (↑ more similar)",
        "Centre-of-mass distance  (↓ more similar)",
        "Earth Mover's Distance  (↓ more similar)",
    )
    for pname, plabel in zip(PAIR_NAMES, PAIR_LABELS):
        specs = [(f"{prefix}_{pname}_{m}", m_labels[i].split("  ")[0], ml)
                 for i, (m, ml) in enumerate(zip(m_keys, m_labels))]
        fig, axes = _make_fig(4, f"Attribution Agreement — {label} ({plabel})")
        for ax, (key, yl, title) in zip(axes, specs):
            _plot_row(ax, ep_data, key, yl, title)
        _save(fig, axes, f"{fname_prefix}_{pname}.png")


def fig6_attribution(ep_data: EpData):
    _attribution_fig(ep_data, "base",  "Base Camera",        is_2d=True,  fname_prefix="fig6a_base_attribution")
    _attribution_fig(ep_data, "wrist", "Left Wrist Camera",  is_2d=True,  fname_prefix="fig6b_wrist_attribution")
    _attribution_fig(ep_data, "task",  "Task Tokens",         is_2d=False, fname_prefix="fig6c_task_attribution")
    _attribution_fig(ep_data, "state", "State Tokens",        is_2d=False, fname_prefix="fig6d_state_attribution")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(EPISODE_JSON) as f:
        all_eps = json.load(f)

    if FILTER_MIXED_TASKS:
        # Find task_ids that have at least one success and one failure
        from collections import defaultdict
        outcomes: Dict[int, set] = defaultdict(set)
        for ep in all_eps:
            outcomes[ep["task_id"]].add(ep["success"])
        mixed_ids = {tid for tid, outs in outcomes.items() if len(outs) == 2}
        episodes = [ep for ep in all_eps if ep["task_id"] in mixed_ids]
        print(f"Mixed-outcome tasks: {sorted(mixed_ids)}")
        print(f"Episodes to process: {len(episodes)} / {len(all_eps)}")
    else:
        episodes = all_eps

    ep_data: EpData = {"success": [], "failure": []}

    for i, ep in enumerate(episodes, 1):
        label = "success" if ep["success"] else "failure"
        print(f"  [{i:3d}/{len(episodes)}] task {ep['task_id']} ep{ep['episode_num']:03d} ({label}) …",
              end=" ", flush=True)
        records = load_episode(ep)
        ep_data[label].append(records)
        print(f"{len(records)} steps")

    ns = len(ep_data["success"]); nf = len(ep_data["failure"])
    print(f"\n{ns} success, {nf} failure episodes. Rendering figures …\n")

    fig1_gini(ep_data)
    fig2_attention_routing(ep_data)
    fig3_inter_head_agreement(ep_data)
    fig4_action_oscillation(ep_data)
    fig5_motion(ep_data)
    fig6_attribution(ep_data)

    print("Done.")


if __name__ == "__main__":
    main()
