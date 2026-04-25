#!/usr/bin/env python3
"""
Failure-predictor visualization: all analysis plots for success-vs-failure
attention and action feature comparison.

Run:
    python -m scripts.attention_analysis.failure_plots
  or
    python scripts/attention_analysis/failure_plots.py

Saves one PNG per figure to OUTPUT_DIR.

Plots produced:
  1. phase1_early_signals.png       — Gini / spatial_ent / task_head_agreement [P1e NEW]
  2. phase2_modality_crossing.png   — 3×2: all P2 features incl. gripper_open [P2e NEW★★★]
                                       and joint_vel_entropy [P2j NEW★★★]
  3. phase3_explosion.png           — State vel + action std spaghetti (steps 0-50)
  4. spatial_attention_maps.png     — Mean 16×16 base-camera attention at steps 0/5/10/15
  5. episode_heatmap.png            — Episodes × steps: img_attn, gini, grip_open, jv_ent
  6. effect_size_over_time.png      — Cohen's d for all features incl. new ones
  7. early_scatter.png              — 4-panel scatter: best feature pairs per phase
  8. auc_vs_steps.png               — Cumulative AUC incl. new features
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats as scipy_stats

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR   = Path("/home/ziyao/Documents/policy_records_20260423_180932")
OUTPUT_DIR = DATA_DIR / "failure_analysis_plots"
EPISODE_JSON = DATA_DIR / "episode_summaries.json"

# _20260423_: task 6 added (7S/3F); _20260421_: task 6 had only 1 failure (excluded).
MIXED_TASK_IDS: set[int] = {2, 3, 5, 6, 9}

TASK_SHORT = {2: "stove+moka", 3: "bowl+drawer", 5: "book+caddy",
              6: "mug+choc", 9: "mug+microwave"}
TASK_LONG  = {
    2: "turn on stove and put moka pot on it",
    3: "put black bowl in bottom drawer and close",
    5: "pick up book and place in back of caddy",
    6: "put white mug on plate and choc pudding right of plate",
    9: "put yellow/white mug in microwave and close",
}

# colours
C_SUCCESS = "#2196F3"   # blue
C_FAILURE = "#F44336"   # red
C_S_LIGHT = "#90CAF9"
C_F_LIGHT = "#FFCDD2"

PATCH_GRID = (16, 16)   # (H_patches, W_patches) for base camera

# ──────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION  (identical logic to failure_stats.py)
# ══════════════════════════════════════════════════════════════

def _load_spans(rec: dict):
    def _sp(k): return tuple(int(x) for x in np.asarray(rec[k]).tolist())
    return (
        _sp("outputs/debug/spans/image/base_0_rgb"),
        _sp("outputs/debug/spans/image/left_wrist_0_rgb"),
        _sp("outputs/debug/spans/image/right_wrist_0_rgb"),
        _sp("outputs/debug/spans/task"),
        _sp("outputs/debug/spans/state"),
    )


def _gini(values: np.ndarray) -> float:
    a = np.sort(np.abs(values)).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s == 0 or n == 0: return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum()); n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos; fpr = fp / n_neg
    return float(np.trapz(tpr, fpr))


def extract_step_features(rec: dict) -> dict:
    attn_w   = np.asarray(rec["outputs/debug/attn/weights"])
    attn_l16 = attn_w[1, 0]   # (H=8, S=819)
    full_v   = np.asarray(rec["outputs/debug/attn/v"])
    actions  = np.asarray(rec["outputs/actions"])  # (10, 7)
    state    = np.asarray(rec["outputs/state"])
    gmap_b   = np.asarray(rec["outputs/debug/gradcam/image/base_0_rgb"]).reshape(-1)

    sp_base, sp_w1, sp_w2, sp_task, sp_state = _load_spans(rec)

    base_patch = attn_l16[:, sp_base[0]:sp_base[1]].mean(axis=0)  # (256,)
    v0 = full_v[1, 0, :, 0, :]   # (S, D)  K=1 GQA

    # ── Phase 1 ──────────────────────────────────────────────
    gini_base   = _gini(base_patch)
    p_base      = base_patch / (base_patch.sum() + 1e-12)
    spatial_ent = float(-np.sum(p_base * np.log(p_base + 1e-12)))
    action_jerk = float(np.abs(np.diff(actions, axis=0)).mean())

    # [P1e NEW] Task-token inter-head cosine similarity
    task_ph = attn_l16[:, sp_task[0]:sp_task[1]]
    t_normed = task_ph / (np.linalg.norm(task_ph, axis=1, keepdims=True) + 1e-9)
    sim_task = t_normed @ t_normed.T
    iu = np.triu_indices(attn_l16.shape[0], k=1)
    task_head_agreement = float(sim_task[iu].mean())

    # ── Phase 2 ──────────────────────────────────────────────
    S = attn_l16.shape[1]
    img_attn_frac = float(
        (attn_l16[:, sp_base[0]:sp_base[1]].sum(axis=1)
         + attn_l16[:, sp_w1[0]:sp_w1[1]].sum(axis=1)
         + attn_l16[:, sp_w2[0]:sp_w2[1]].sum(axis=1)).mean() / S
    )
    task_attn_frac  = float(attn_l16[:, sp_task[0]:sp_task[1]].sum(axis=1).mean() / S)
    state_attn_frac = float(attn_l16[:, sp_state[0]:sp_state[1]].sum(axis=1).mean() / S)
    action_std      = float(actions.std())
    v_norms         = np.linalg.norm(v0, axis=-1)
    vnorm_score     = (attn_l16 * v_norms[np.newaxis, :]).max(axis=0)
    vnorm_base      = float(vnorm_score[sp_base[0]:sp_base[1]].mean())

    # [P2e NEW] Gripper open fraction
    gripper_open_frac = float((actions[:, 6] > 0).mean())

    # [P2f NEW] Action sign-flip rate
    signs = np.sign(actions)
    sign_flip_rate = float((signs[1:] != signs[:-1]).mean())

    # [P2g NEW] Head attention entropy variance
    p_h = np.clip(attn_l16, 1e-9, 1.0)
    per_head_ent = -(p_h * np.log(p_h)).sum(axis=1)
    head_ent_var = float(per_head_ent.var())

    # [P2h NEW] Value-projection alignment on base patches
    o = attn_l16 @ v0
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)
    v_base = v0[sp_base[0]:sp_base[1]]
    v_base_unit = v_base / (np.linalg.norm(v_base, axis=-1, keepdims=True) + 1e-9)
    v_proj_align = float(np.abs(v_base_unit @ o_unit.T).mean())

    # [P2i NEW] GradCAM energy on base camera
    gradcam_base_energy = float(np.abs(gmap_b).mean())

    return {
        # Phase 1
        "gini_base":           gini_base,
        "spatial_ent":         spatial_ent,
        "action_jerk":         action_jerk,
        "task_head_agreement": task_head_agreement,
        # Phase 2
        "img_attn_frac":       img_attn_frac,
        "task_attn_frac":      task_attn_frac,
        "state_attn_frac":     state_attn_frac,
        "action_std":          action_std,
        "vnorm_base":          vnorm_base,
        "gripper_open_frac":   gripper_open_frac,
        "sign_flip_rate":      sign_flip_rate,
        "head_ent_var":        head_ent_var,
        "v_proj_align":        v_proj_align,
        "gradcam_base_energy": gradcam_base_energy,
        # Pass-through
        "base_patch_mean": base_patch,
        "state": state,
    }


def load_all_features(episodes: list, max_steps: int = 50) -> List[dict]:
    result = []
    for ep in episodes:
        label = "success" if ep["success"] else "failure"
        per_step: dict[str, list] = defaultdict(list)
        base_patches  = []
        prev_state    = None
        prev_action0  = None
        prev_peak     = None

        for rel in range(max_steps):
            s = ep["start_idx"] + rel
            if s > ep["end_idx"]:
                break
            fpath = DATA_DIR / f"step_{s}.npy"
            if not fpath.exists():
                continue
            rec = np.load(fpath, allow_pickle=True).item()
            feats = extract_step_features(rec)
            base_patches.append(feats.pop("base_patch_mean"))
            state = feats.pop("state")

            # stateful: state velocity
            feats["state_vel"] = (
                float(np.linalg.norm(state - prev_state)) if prev_state is not None else 0.0
            )
            prev_state = state

            # [P2j NEW] joint velocity entropy
            if prev_state is not None:
                jv = np.abs(state - prev_state) + 1e-9
                feats["joint_vel_entropy"] = float(-(jv/jv.sum() * np.log(jv/jv.sum())).sum())
            else:
                feats["joint_vel_entropy"] = 0.0

            # [P2k NEW] inter-step plan consistency
            cur_a0 = np.asarray(rec["outputs/actions"])[0]
            if prev_action0 is not None:
                feats["plan_consistency"] = float(
                    np.dot(cur_a0, prev_action0) /
                    (np.linalg.norm(cur_a0) * np.linalg.norm(prev_action0) + 1e-9)
                )
            else:
                feats["plan_consistency"] = 1.0
            prev_action0 = cur_a0

            # [P2l NEW] attention peak drift
            sp_b = tuple(int(x) for x in np.asarray(
                rec["outputs/debug/spans/image/base_0_rgb"]).tolist())
            base_map = np.asarray(rec["outputs/debug/attn/weights"])[1, 0, :,
                           sp_b[0]:sp_b[1]].mean(axis=0).reshape(16, 16)
            peak = np.array(np.unravel_index(base_map.argmax(), base_map.shape), dtype=float)
            feats["attn_peak_drift"] = (
                float(np.linalg.norm(peak - prev_peak)) if prev_peak is not None else 0.0
            )
            prev_peak = peak

            for k, v in feats.items():
                per_step[k].append(v)

        result.append({
            "label": label, "task_id": ep["task_id"],
            "episode_num": ep["episode_num"], "success": ep["success"],
            "per_step": dict(per_step),
            "base_patches": base_patches,
        })
    return result


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _step_means_ci(ep_list: list, feat: str, n_steps: int = 50):
    """Returns (steps, mean_array, ci_lo, ci_hi) from episodes in ep_list."""
    per_t: list[list] = [[] for _ in range(n_steps)]
    for ep in ep_list:
        vals = ep["per_step"].get(feat, [])
        for t, v in enumerate(vals[:n_steps]):
            per_t[t].append(v)
    steps, means, lo, hi = [], [], [], []
    for t, vs in enumerate(per_t):
        if len(vs) >= 2:
            arr = np.array(vs)
            m = arr.mean()
            se = arr.std(ddof=1) / np.sqrt(len(arr))
            steps.append(t)
            means.append(m)
            lo.append(m - 1.96 * se)
            hi.append(m + 1.96 * se)
    return np.array(steps), np.array(means), np.array(lo), np.array(hi)


def _cohens_d(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2: return 0.0
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


# ══════════════════════════════════════════════════════════════
# PLOT 1 — Phase 1: Early signals (steps 0–10)
# ══════════════════════════════════════════════════════════════

def plot_phase1_early(eps_s: list, eps_f: list):
    """
    3×2 grid:
      Row 0: gini_base time series (both tasks and combined)
      Row 1: spatial_entropy time series
      Row 2: action_jerk time series
    For each: left = combined; right = per-episode scatter (mean steps 0-9)
    LaTeX labels:
      gini_base    G = 2∑i·ᾱᵢ / (n∑ᾱᵢ) − (n+1)/n
      spatial_ent  H = −∑p̃ log p̃
      action_jerk  J = mean|Δa|
    """
    N_STEPS = 10
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle("Phase 1 — Early Latent Divergence (Steps 0–9)\n"
                 "Features significant or trending before any visual failure",
                 fontsize=13, fontweight="bold")

    feature_meta = [
        ("gini_base",
         r"$G = \frac{2\sum_i i\bar\alpha_i}{n\sum_i \bar\alpha_i} - \frac{n+1}{n}$",
         "[P1a] Gini — base camera attention\n(↑ failure = concentrated/fixating)"),
        ("spatial_ent",
         r"$H = -\sum_p \tilde\alpha_p \log \tilde\alpha_p$",
         "[P1b] Spatial entropy — base camera attention\n(↓ failure = concentrated)"),
        ("task_head_agreement",
         r"$\bar c_{task} = \frac{2}{H(H-1)}\sum_{h<h'}\cos(\vec a_h|_{task},\vec a_{h'}|_{task})$",
         "[P1e NEW ★★] Task-token inter-head agreement\n(↓ failure = heads disagree on task words)"),
    ]

    for row, (feat, formula, ylabel) in enumerate(feature_meta):
        ax_ts  = axes[row, 0]
        ax_sc  = axes[row, 1]

        # Time series
        ts_s, ms, ls, hs = _step_means_ci(eps_s, feat, N_STEPS)
        ts_f, mf, lf, hf = _step_means_ci(eps_f, feat, N_STEPS)

        ax_ts.fill_between(ts_s, ls, hs, color=C_S_LIGHT, alpha=0.6)
        ax_ts.fill_between(ts_f, lf, hf, color=C_F_LIGHT, alpha=0.6)
        ax_ts.plot(ts_s, ms, color=C_SUCCESS, lw=2, label="Success")
        ax_ts.plot(ts_f, mf, color=C_FAILURE, lw=2, label="Failure")
        ax_ts.set_xlabel("Step (relative to episode start)")
        ax_ts.set_ylabel(formula, fontsize=9)
        ax_ts.set_title(f"{ylabel}", fontsize=9)
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)

        # p-value annotation per step
        for t in range(N_STEPS):
            sv = [ep["per_step"][feat][t] for ep in eps_s if t < len(ep["per_step"].get(feat, []))]
            fv = [ep["per_step"][feat][t] for ep in eps_f if t < len(ep["per_step"].get(feat, []))]
            if len(sv) > 1 and len(fv) > 1:
                _, p = scipy_stats.ttest_ind(sv, fv, equal_var=False)
                if p < 0.10:
                    y_ann = max(np.mean(sv), np.mean(fv))
                    ax_ts.annotate(f"p={p:.2f}", xy=(t, y_ann), fontsize=6,
                                   color="purple", ha="center", va="bottom")

        # Per-episode scatter: mean over steps 0-9
        jitter = 0.15
        for ep in eps_s:
            vals = ep["per_step"].get(feat, [])[:N_STEPS]
            if vals:
                x = ep["task_id"] + np.random.uniform(-jitter, jitter)
                ax_sc.scatter(x, np.mean(vals), color=C_SUCCESS, alpha=0.7, s=45,
                              marker="o", zorder=3)
                ax_sc.annotate(f"e{ep['episode_num']:02d}",
                               xy=(x, np.mean(vals)), fontsize=5,
                               ha="center", va="bottom", color=C_SUCCESS)
        for ep in eps_f:
            vals = ep["per_step"].get(feat, [])[:N_STEPS]
            if vals:
                x = ep["task_id"] + np.random.uniform(-jitter, jitter)
                ax_sc.scatter(x, np.mean(vals), color=C_FAILURE, alpha=0.7, s=45,
                              marker="X", zorder=3)
                ax_sc.annotate(f"e{ep['episode_num']:02d}",
                               xy=(x, np.mean(vals)), fontsize=5,
                               ha="center", va="top", color=C_FAILURE)

        ax_sc.set_xticks(sorted(MIXED_TASK_IDS))
        ax_sc.set_xticklabels([TASK_SHORT[t] for t in sorted(MIXED_TASK_IDS)], fontsize=8)
        ax_sc.set_xlabel("Task")
        ax_sc.set_ylabel(formula, fontsize=9)
        ax_sc.set_title(f"Per-episode mean (steps 0–9)", fontsize=9)
        ax_sc.grid(True, alpha=0.3)
        s_patch = mpatches.Patch(color=C_SUCCESS, label="Success")
        f_patch = mpatches.Patch(color=C_FAILURE, label="Failure", alpha=0.7)
        ax_sc.legend(handles=[s_patch, f_patch], fontsize=8)

    fig.tight_layout()
    _save(fig, "phase1_early_signals.png")


# ══════════════════════════════════════════════════════════════
# PLOT 2 — Phase 2: Modality attention crossing (steps 0–25)
# ══════════════════════════════════════════════════════════════

def plot_phase2_crossing(eps_s: list, eps_f: list):
    """3×2 grid covering all Phase 2 features, steps 0–25."""
    N_STEPS = 26
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("Phase 2 — Emerging Divergence (Steps 0–25)\n"
                 r"Existing: $f_{img}$ crosses, $v_t$ diverges.  "
                 r"NEW ★★★: gripper open fraction, joint velocity entropy",
                 fontsize=11, fontweight="bold")

    def _ts_panel(ax, feat, formula, title, annot_steps=(10, 15, 20, 25)):
        ts_s, ms, ls, hs = _step_means_ci(eps_s, feat, N_STEPS)
        ts_f, mf, lf, hf = _step_means_ci(eps_f, feat, N_STEPS)
        ax.fill_between(ts_s, ls, hs, color=C_S_LIGHT, alpha=0.6)
        ax.fill_between(ts_f, lf, hf, color=C_F_LIGHT, alpha=0.6)
        ax.plot(ts_s, ms, color=C_SUCCESS, lw=2.5, label="Success")
        ax.plot(ts_f, mf, color=C_FAILURE, lw=2.5, label="Failure")
        if feat == "img_attn_frac" and len(ms) > 5 and len(mf) > 5:
            diff = ms - mf[:len(ms)]
            t_cross, started_neg = None, diff[0] < 0
            for t_c in range(1, len(diff) - 2):
                if started_neg and diff[t_c:t_c+3].mean() > 0:
                    t_cross = t_c; break
            if t_cross is not None:
                ax.axvline(t_cross, color="purple", lw=1.5, ls="--", alpha=0.7,
                           label=f"Crossing ≈ step {t_cross}")
        for t in annot_steps:
            if t >= N_STEPS: continue
            sv = [ep["per_step"].get(feat, [])[t] for ep in eps_s if t < len(ep["per_step"].get(feat, []))]
            fv = [ep["per_step"].get(feat, [])[t] for ep in eps_f if t < len(ep["per_step"].get(feat, []))]
            if len(sv) > 1 and len(fv) > 1:
                _, p = scipy_stats.ttest_ind(sv, fv, equal_var=False)
                star = "★★★" if p<0.001 else ("★★" if p<0.01 else ("★" if p<0.05 else ("·" if p<0.10 else "")))
                if star:
                    y_ann = max(np.mean(sv), np.mean(fv))
                    ax.annotate(f"t={t}\np={p:.3f}{star}", xy=(t, y_ann), fontsize=5.5,
                                color="purple", ha="center", va="bottom")
        ax.set_xlabel("Step"); ax.set_ylabel(formula, fontsize=8)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    panels = [
        ("img_attn_frac",
         r"$f_{img}=\frac{1}{HS}\sum_h\sum_p A^{(16)}_{h,p}$",
         "[P2a] All-image attention fraction\n(crosses direction: success > failure after ~step 8)"),
        ("task_attn_frac",
         r"$f_{task}=\frac{1}{HS}\sum_h\sum_j A^{(16)}_{h,j}$",
         "[P2b] Task-token attention fraction\n(↑ failure = re-reading instruction)"),
        ("gripper_open_frac",
         r"$f_{grip}=\frac{1}{T}\sum_t \mathbf{1}[a_{t,6}>0]$",
         "[P2e NEW ★★★] Gripper open fraction\nSuccess=0.83 vs Failure=0.59  (p=0.0001)"),
        ("joint_vel_entropy",
         r"$H_{jv}=-\sum_k \hat q_k\log\hat q_k,\;\hat q_k=|\dot q_k|/\sum|\dot q|$",
         "[P2j NEW ★★★] Joint velocity entropy\nSuccess=1.49 vs Failure=1.33  (p=0.0001)"),
        ("sign_flip_rate",
         r"$\phi_{flip}=\frac{1}{(T-1)D}\sum_{t,d}\mathbf{1}[\mathrm{sgn}(a_{t,d})\ne\mathrm{sgn}(a_{t-1,d})]$",
         "[P2f NEW ★★] Action sign-flip rate\nFailure=0.036 > Success=0.028  (p=0.006)"),
        ("state_vel",
         r"$v_t=\|s_t-s_{t-1}\|_2$",
         "[P2d] State velocity\n(first significant at step ~20)"),
    ]
    for ax, (feat, formula, title) in zip(axes.flat, panels):
        _ts_panel(ax, feat, formula, title)

    fig.tight_layout()
    _save(fig, "phase2_modality_crossing.png")


# ══════════════════════════════════════════════════════════════
# PLOT 3 — Phase 3: Explosion (steps 0–50)
# ══════════════════════════════════════════════════════════════

def plot_phase3_explosion(eps_s: list, eps_f: list):
    """
    2×1 panels:
      Top: state_vel over 0-50 with spaghetti individual traces
      Bottom: action_std over 0-50 with spaghetti traces
    """
    N_STEPS = 50
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("Phase 3 — Visible Failure Loop (Steps 0–50)\n"
                 r"$v_t$ and $\sigma_a$ explode after step 25 in failures",
                 fontsize=12, fontweight="bold")

    for ax, feat, formula, title in [
        (axes[0], "state_vel",
         r"$v_t = \|s_t - s_{t-1}\|_2$",
         "State velocity  (robot thrashing — 12× higher by step 45)"),
        (axes[1], "action_std",
         r"$\sigma_a = \text{std}(\{a_{t,d}\})$  over 10-step horizon × 7 dims",
         "Action std  (2.5× higher by step 45, p=0.0008)"),
    ]:
        # Individual episode traces (spaghetti)
        for ep in eps_s:
            vals = ep["per_step"].get(feat, [])
            if vals:
                t_arr = np.arange(len(vals))
                ax.plot(t_arr, vals, color=C_SUCCESS, alpha=0.18, lw=0.9)
        for ep in eps_f:
            vals = ep["per_step"].get(feat, [])
            if vals:
                t_arr = np.arange(len(vals))
                ax.plot(t_arr, vals, color=C_FAILURE, alpha=0.18, lw=0.9)

        # Group means + CI on top
        ts_s, ms, ls, hs = _step_means_ci(eps_s, feat, N_STEPS)
        ts_f, mf, lf, hf = _step_means_ci(eps_f, feat, N_STEPS)
        ax.fill_between(ts_s, ls, hs, color=C_SUCCESS, alpha=0.35)
        ax.fill_between(ts_f, lf, hf, color=C_FAILURE, alpha=0.35)
        ax.plot(ts_s, ms, color=C_SUCCESS, lw=3, label="Success (mean ± 95% CI)")
        ax.plot(ts_f, mf, color=C_FAILURE, lw=3, label="Failure (mean ± 95% CI)")
        ax.axvline(25, color="orange", lw=1.5, ls="--", alpha=0.8, label="Phase 3 start (step 25)")
        ax.set_ylabel(formula, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Step (relative to episode start)")
    fig.tight_layout()
    _save(fig, "phase3_explosion.png")


# ══════════════════════════════════════════════════════════════
# PLOT 4 — Spatial attention maps (16×16 per step)
# ══════════════════════════════════════════════════════════════

def plot_spatial_attention_maps(eps_s: list, eps_f: list):
    """
    For each task in MIXED_TASK_IDS, and for steps 0, 5, 10, 15:
    show mean base-camera attention map for success vs failure.
    Layout: tasks × steps × (S | F)
    """
    STEPS_TO_SHOW = [0, 5, 10, 15]
    tasks = sorted(MIXED_TASK_IDS)
    n_tasks = len(tasks)
    n_steps_shown = len(STEPS_TO_SHOW)

    # rows: tasks; cols: steps * 2 (S/F pairs)
    n_cols = n_steps_shown * 2
    fig, axes = plt.subplots(n_tasks, n_cols,
                              figsize=(n_cols * 1.9 + 1, n_tasks * 2.4 + 1.5))
    fig.suptitle("Mean Base-Camera Attention Maps (16×16 patches)\n"
                 "Success vs Failure by task and step",
                 fontsize=12, fontweight="bold")

    for row, task_id in enumerate(tasks):
        t_eps_s = [e for e in eps_s if e["task_id"] == task_id]
        t_eps_f = [e for e in eps_f if e["task_id"] == task_id]

        for col_pair, step in enumerate(STEPS_TO_SHOW):
            # Gather maps
            maps_s = [e["base_patches"][step] for e in t_eps_s if step < len(e["base_patches"])]
            maps_f = [e["base_patches"][step] for e in t_eps_f if step < len(e["base_patches"])]

            for side, (maps, color, label) in enumerate([
                (maps_s, C_SUCCESS, "S"),
                (maps_f, C_FAILURE, "F"),
            ]):
                col = col_pair * 2 + side
                ax = axes[row, col]
                if maps:
                    mean_map = np.mean(maps, axis=0).reshape(PATCH_GRID)
                    # Use individual colourmap per panel so differences are visible
                    ax.imshow(mean_map, cmap="hot", interpolation="bilinear", aspect="equal")
                    n_ep = len(maps)
                    ax.set_title(f"{label} (n={n_ep})\nstep {step}", fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7)
                    ax.set_facecolor("#eee")
                ax.axis("off")
                # Thin coloured border
                for spine in ax.spines.values():
                    spine.set_edgecolor(C_SUCCESS if label == "S" else C_FAILURE)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

        # Row label
        axes[row, 0].set_ylabel(TASK_SHORT[task_id], fontsize=8, rotation=90, labelpad=4)

    # Column headers
    for col_pair, step in enumerate(STEPS_TO_SHOW):
        for side, lbl in enumerate(["SUCCESS", "FAILURE"]):
            col = col_pair * 2 + side
            axes[0, col].set_title(f"{lbl}\nstep {step}", fontsize=7.5,
                                   color=C_SUCCESS if lbl == "SUCCESS" else C_FAILURE,
                                   fontweight="bold")

    fig.tight_layout()
    _save(fig, "spatial_attention_maps.png")


# ══════════════════════════════════════════════════════════════
# PLOT 5 — Episode × step heatmaps
# ══════════════════════════════════════════════════════════════

def plot_episode_heatmap(eps_all: list):
    """
    Four heatmaps: img_attn_frac, gini_base, gripper_open_frac, joint_vel_entropy.
    Rows = episodes (success first, then failure); columns = steps 0–49.
    """
    N_STEPS = 50
    feats = [
        ("img_attn_frac",      r"$f_{img}$ — image attn fraction",     "Blues"),
        ("gini_base",          r"$G$ — Gini base-cam attn",             "Reds"),
        ("gripper_open_frac",  r"$f_{grip}$ — gripper open fraction [NEW★★★]", "Greens"),
        ("joint_vel_entropy",  r"$H_{jv}$ — joint velocity entropy [NEW★★★]", "Purples"),
    ]
    # Sort episodes: success first, then failure; within each group by task then episode_num
    ordered = sorted(eps_all,
                     key=lambda e: (not e["success"], e["task_id"], e["episode_num"]))

    n_ep = len(ordered)
    yticks = []
    yticklabels = []

    fig, axes = plt.subplots(1, 4, figsize=(28, max(8, n_ep * 0.28 + 2)))
    fig.suptitle("Episode × Step Feature Heatmaps\n"
                 "(episodes sorted: SUCCESS first; label = t<task>e<ep><S|F>)",
                 fontsize=12, fontweight="bold")

    for ax, (feat, title, cmap) in zip(axes, feats):
        matrix = np.full((n_ep, N_STEPS), np.nan)
        for i, ep in enumerate(ordered):
            vals = ep["per_step"].get(feat, [])
            for t, v in enumerate(vals[:N_STEPS]):
                matrix[i, t] = v

        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                       interpolation="nearest",
                       vmin=np.nanpercentile(matrix, 5),
                       vmax=np.nanpercentile(matrix, 95))
        ax.set_xlabel("Step")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        # Draw separator between success and failure
        n_success = sum(1 for e in ordered if e["success"])
        ax.axhline(n_success - 0.5, color="black", lw=2, ls="--")
        ax.text(N_STEPS + 0.5, n_success / 2,
                "SUCCESS", va="center", ha="left", color=C_SUCCESS,
                fontsize=7, fontweight="bold")
        ax.text(N_STEPS + 0.5, n_success + (n_ep - n_success) / 2,
                "FAILURE", va="center", ha="left", color=C_FAILURE,
                fontsize=7, fontweight="bold")

    # Y-axis labels (shared, applied to left plot)
    ytick_pos, ytick_labels = [], []
    for i, ep in enumerate(ordered):
        lbl = f"t{ep['task_id']}e{ep['episode_num']:02d}{'S' if ep['success'] else 'F'}"
        ytick_pos.append(i)
        ytick_labels.append(lbl)
    axes[0].set_yticks(ytick_pos)
    axes[0].set_yticklabels(ytick_labels, fontsize=5.5)
    axes[1].set_yticks([])

    fig.tight_layout()
    _save(fig, "episode_heatmap.png")


# ══════════════════════════════════════════════════════════════
# PLOT 6 — Cohen's d per feature over steps
# ══════════════════════════════════════════════════════════════

def plot_effect_size_over_time(eps_s: list, eps_f: list):
    """
    For each feature, compute Cohen's d(S, F) at every step.
    Plot all features in one chart.
    Positive d → success > failure; negative → failure > success.
    """
    N_STEPS = 50
    features = {
        "gini_base":           r"$G$ [P1a]",
        "task_head_agreement": r"$\bar c_{task}$ [P1e NEW]",
        "gripper_open_frac":   r"$f_{grip}$ [P2e NEW★★★]",
        "joint_vel_entropy":   r"$H_{jv}$ [P2j NEW★★★]",
        "sign_flip_rate":      r"$\phi_{flip}$ [P2f NEW★★]",
        "img_attn_frac":       r"$f_{img}$ [P2a]",
        "task_attn_frac":      r"$f_{task}$ [P2b]",
        "state_vel":           r"$v_t$ [P2d]",
        "action_std":          r"$\sigma_a$ [P3a]",
    }
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(features)))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axhline(-0.5, color="gray", lw=0.5, ls=":")
    ax.axvline(10, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(25, color="gray", lw=0.8, ls="--", alpha=0.5)

    for (feat, label), color in zip(features.items(), cmap_colors):
        d_vals, steps = [], []
        for t in range(N_STEPS):
            sv = [ep["per_step"].get(feat, [])[t]
                  for ep in eps_s if t < len(ep["per_step"].get(feat, []))]
            fv = [ep["per_step"].get(feat, [])[t]
                  for ep in eps_f if t < len(ep["per_step"].get(feat, []))]
            if len(sv) >= 3 and len(fv) >= 3:
                d_vals.append(_cohens_d(sv, fv))
                steps.append(t)
        if steps:
            ax.plot(steps, d_vals, color=color, lw=2, label=label)
            ax.scatter(steps[0], d_vals[0], color=color, s=30, zorder=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cohen's d  (success − failure)\npositive = success higher")
    ax.set_title("Effect Size (Cohen's d) per Feature over Steps 0–49\n"
                 "Phase boundaries at step 10 and 25", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.25)
    # Phase labels placed at 10% from the bottom of the actual data range
    ylo, yhi = ax.get_ylim()
    y_lbl = ylo + 0.10 * (yhi - ylo)
    ax.text(5,    y_lbl, "Phase 1", ha="center", fontsize=8, color="gray")
    ax.text(17.5, y_lbl, "Phase 2", ha="center", fontsize=8, color="gray")
    ax.text(37.5, y_lbl, "Phase 3", ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    _save(fig, "effect_size_over_time.png")


# ══════════════════════════════════════════════════════════════
# PLOT 7 — Per-episode scatter at early steps
# ══════════════════════════════════════════════════════════════

def plot_early_scatter(eps_s: list, eps_f: list):
    """2×2 scatter grid: best early feature pairs, each episode labelled."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Per-Episode Feature Scatter — Early Detection\n"
                 "(each point = 1 episode; label = t<task>e<ep>)",
                 fontsize=12, fontweight="bold")

    def _scatter_panel(ax, x_key, x_slice, y_key, y_slice, xlabel, ylabel, title):
        for eps, color, marker, group in [
            (eps_s, C_SUCCESS, "o", "Success"),
            (eps_f, C_FAILURE, "X", "Failure"),
        ]:
            xs, ys, labels = [], [], []
            for ep in eps:
                xv = ep["per_step"].get(x_key, [])[x_slice[0]:x_slice[1]]
                yv = ep["per_step"].get(y_key, [])[y_slice[0]:y_slice[1]]
                if xv and yv:
                    xs.append(np.mean(xv)); ys.append(np.mean(yv))
                    labels.append(f"t{ep['task_id']}e{ep['episode_num']:02d}")
            ax.scatter(xs, ys, color=color, marker=marker, s=70, alpha=0.8, label=group, zorder=3)
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y), fontsize=5, ha="center",
                            va="bottom" if color == C_SUCCESS else "top", color=color)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # A: task_head_agreement (0-9) vs gini_base (0-9)   — pure Phase 1
    _scatter_panel(axes[0, 0],
        "task_head_agreement", (0, 10), "gini_base", (0, 10),
        r"$\bar c_{task}$ (task head agreement, steps 0–9)  [↑ success]",
        r"$G$ (Gini base-cam, steps 0–9)  [↑ failure]",
        "[P1e + P1a] Earliest predictors — steps 0–9")

    # B: gripper_open_frac (10-20) vs joint_vel_entropy (10-20)  — new ★★★ pair
    _scatter_panel(axes[0, 1],
        "gripper_open_frac", (10, 20), "joint_vel_entropy", (10, 20),
        r"$f_{grip}$ (gripper open frac, steps 10–20)  [↑ success]",
        r"$H_{jv}$ (joint vel entropy, steps 10–20)  [↑ success]",
        "[P2e + P2j NEW ★★★] Strongest mid-phase predictors — steps 10–20")

    # C: sign_flip_rate (10-25) vs plan_consistency (10-25)
    _scatter_panel(axes[1, 0],
        "sign_flip_rate", (10, 25), "plan_consistency", (10, 25),
        r"$\phi_{flip}$ (sign-flip rate, steps 10–25)  [↑ failure]",
        r"$c_t$ (plan consistency, steps 10–25)  [↑ success]",
        "[P2f + P2k NEW] Action plan stability — steps 10–25")

    # D: state_vel (15-25) vs img_attn_frac (10-25)
    _scatter_panel(axes[1, 1],
        "state_vel", (15, 25), "img_attn_frac", (10, 25),
        r"$v_t$ (state vel, steps 15–25)  [↑ failure]",
        r"$f_{img}$ (image attn frac, steps 10–25)  [↑ success]",
        "[P2d + P2a] Classical visual grounding + motion speed")

    fig.tight_layout()
    _save(fig, "early_scatter.png")


# ══════════════════════════════════════════════════════════════
# PLOT 8 — AUC vs step window
# ══════════════════════════════════════════════════════════════

def plot_auc_vs_steps(eps_all: list):
    """
    At each step window [0, t], compute the leave-one-episode-out AUC
    for each feature (and for a combined logistic-style linear score).

    Feature score = cumulative mean of that feature over steps 0..t.
    AUC is computed without sklearn: manual trapezoidal ROC.
    """
    N_STEPS = 50
    features = {
        # Phase 1
        "gini_base":           r"$G$ (Gini) P1a",
        "task_head_agreement": r"$\bar c_{task}$ (head agr) P1e NEW",
        # Phase 2 — new ★★★
        "gripper_open_frac":   r"$f_{grip}$ (grip open) P2e NEW★★★",
        "joint_vel_entropy":   r"$H_{jv}$ (joint vel ent) P2j NEW★★★",
        "sign_flip_rate":      r"$\phi_{flip}$ (flip rate) P2f NEW★★",
        # Phase 2 — existing
        "img_attn_frac":       r"$f_{img}$ (neg) P2a",
        "task_attn_frac":      r"$f_{task}$ P2b",
        "state_vel":           r"$v_t$ P2d",
        # Phase 3
        "action_std":          r"$\sigma_a$ P3a",
    }
    NEGATE = {"img_attn_frac", "spatial_ent", "task_head_agreement",
              "gripper_open_frac", "joint_vel_entropy"}

    # 1 = failure, 0 = success
    y_true = np.array([0 if e["success"] else 1 for e in eps_all])

    cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(features) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0.5, color="gray", lw=1, ls="--", label="Random (AUC=0.5)")
    ax.axvline(10, color="lightgray", lw=1, ls="--")
    ax.axvline(25, color="lightgray", lw=1, ls="--")
    ax.text(5,  0.51, "Ph.1", ha="center", fontsize=8, color="gray")
    ax.text(17, 0.51, "Ph.2", ha="center", fontsize=8, color="gray")
    ax.text(37, 0.51, "Ph.3", ha="center", fontsize=8, color="gray")

    for (feat, label), color in zip(features.items(), cmap_colors):
        auc_vals, steps = [], []
        for t in range(2, N_STEPS):
            scores = []
            for ep in eps_all:
                vals = ep["per_step"].get(feat, [])[:t]
                scores.append(np.mean(vals) if vals else 0.0)
            scores = np.array(scores)
            if feat in NEGATE:
                scores = -scores
            if len(np.unique(y_true)) == 2:
                auc = _roc_auc(y_true, scores)
                # Take max(auc, 1-auc) — direction-agnostic
                auc = max(auc, 1 - auc)
                auc_vals.append(auc)
                steps.append(t)
        if steps:
            ax.plot(steps, auc_vals, color=color, lw=2, label=label)

    # Combined score: normalise each feature, sum (sign-adjusted)
    signs = {f: -1 if f in NEGATE else 1 for f in features}
    auc_combined, steps_combined = [], []
    for t in range(2, N_STEPS):
        feat_arrays = {}
        for feat in features:
            arr = np.array([
                np.mean(ep["per_step"].get(feat, [])[:t]) if ep["per_step"].get(feat, []) else 0.0
                for ep in eps_all
            ])
            feat_arrays[feat] = arr
        # Z-score each, sum
        combined = np.zeros(len(eps_all))
        for feat in features:
            arr = feat_arrays[feat]
            std = arr.std() + 1e-9
            combined += signs[feat] * (arr - arr.mean()) / std
        if len(np.unique(y_true)) == 2:
            auc = _roc_auc(y_true, combined)
            auc = max(auc, 1 - auc)
            auc_combined.append(auc)
            steps_combined.append(t)
    if steps_combined:
        ax.plot(steps_combined, auc_combined, color="black", lw=3, ls="-",
                label="Combined (z-scored sum)", zorder=5)

    ax.set_xlabel("Step window [0, t]")
    ax.set_ylabel("AUC (direction-agnostic, all episodes)")
    ax.set_title("Failure Predictor AUC vs. Step Window\n"
                 "Combined score uses z-scored sum of all features", fontsize=11, fontweight="bold")
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "auc_vs_steps.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    with open(EPISODE_JSON) as f:
        all_episodes = json.load(f)

    mixed_eps_meta = [e for e in all_episodes if e["task_id"] in MIXED_TASK_IDS]
    print(f"  Loading features for {len(mixed_eps_meta)} episodes …", flush=True)
    eps_all = load_all_features(mixed_eps_meta)
    eps_s = [e for e in eps_all if e["success"]]
    eps_f = [e for e in eps_all if not e["success"]]
    print(f"  {len(eps_s)} success, {len(eps_f)} failure episodes loaded.")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()

    print("  1/8  Phase 1 early signals …")
    plot_phase1_early(eps_s, eps_f)

    print("  2/8  Phase 2 modality crossing …")
    plot_phase2_crossing(eps_s, eps_f)

    print("  3/8  Phase 3 explosion …")
    plot_phase3_explosion(eps_s, eps_f)

    print("  4/8  Spatial attention maps …")
    plot_spatial_attention_maps(eps_s, eps_f)

    print("  5/8  Episode heatmaps …")
    plot_episode_heatmap(eps_all)

    print("  6/8  Effect size over time …")
    plot_effect_size_over_time(eps_s, eps_f)

    print("  7/8  Early feature scatter …")
    plot_early_scatter(eps_s, eps_f)

    print("  8/8  AUC vs steps …")
    plot_auc_vs_steps(eps_all)

    print()
    print("  Done. All plots saved to:")
    print(f"    {OUTPUT_DIR}/")
    print()
    print("  Files:")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    {p.name}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
