#!/usr/bin/env python3
"""
Gini coefficient over inference steps: mean trajectory + variance comparison
between success and failure episodes.

Run:
    python -m scripts.attention_analysis.gini_variance
  or
    python scripts/attention_analysis/gini_variance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR     = Path("/home/ziyao/Documents/policy_records_20260423_180932")
EPISODE_JSON = DATA_DIR / "episode_summaries.json"
OUTPUT_DIR   = DATA_DIR / "failure_analysis_plots"

MIXED_TASK_IDS = {2, 3, 5, 6, 9}
MAX_STEPS      = 50

C_SUCCESS = "#2196F3"
C_FAILURE = "#F44336"
# ──────────────────────────────────────────────────────────────


def _gini(values: np.ndarray) -> float:
    a = np.sort(np.abs(values)).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s == 0 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def load_gini_trajectories(episodes: list) -> tuple[list, list]:
    """Returns (success_trajs, failure_trajs), each a list of 1-D numpy arrays."""
    success, failure = [], []
    for ep in episodes:
        ginis = []
        for rel in range(MAX_STEPS):
            s = ep["start_idx"] + rel
            if s > ep["end_idx"]:
                break
            fpath = DATA_DIR / f"step_{s}.npy"
            if not fpath.exists():
                continue
            rec = np.load(fpath, allow_pickle=True).item()

            attn = np.asarray(rec["outputs/debug/attn/weights"])[1, 0]  # (H, S)
            sp_b = tuple(int(x) for x in np.asarray(
                rec["outputs/debug/spans/image/base_0_rgb"]).tolist())
            base_patch = attn[:, sp_b[0]:sp_b[1]].mean(axis=0)  # (256,)
            ginis.append(_gini(base_patch))

        arr = np.array(ginis)
        (success if ep["success"] else failure).append(arr)
    return success, failure


def _pad_to_matrix(trajs: list[np.ndarray], n_steps: int) -> np.ndarray:
    """Stack trajectories into (N_episodes, n_steps), padding short ones with NaN."""
    mat = np.full((len(trajs), n_steps), np.nan)
    for i, t in enumerate(trajs):
        mat[i, :len(t)] = t
    return mat


def main():
    with open(EPISODE_JSON) as f:
        all_eps = json.load(f)
    mixed = [e for e in all_eps if e["task_id"] in MIXED_TASK_IDS]
    n_s = sum(1 for e in mixed if e["success"])
    n_f = sum(1 for e in mixed if not e["success"])

    print(f"Loading Gini for {len(mixed)} episodes ({n_s} success, {n_f} failure)…",
          flush=True)
    s_trajs, f_trajs = load_gini_trajectories(mixed)
    print("Done.")

    s_mat = _pad_to_matrix(s_trajs, MAX_STEPS)   # (N_s, 50)
    f_mat = _pad_to_matrix(f_trajs, MAX_STEPS)   # (N_f, 50)

    # ── per-step statistics ──────────────────────────────────────────────
    s_mean = np.nanmean(s_mat, axis=0)
    f_mean = np.nanmean(f_mat, axis=0)
    s_std  = np.nanstd(s_mat,  axis=0, ddof=1)
    f_std  = np.nanstd(f_mat,  axis=0, ddof=1)
    s_var  = np.nanvar(s_mat,  axis=0, ddof=1)
    f_var  = np.nanvar(f_mat,  axis=0, ddof=1)

    steps = np.arange(MAX_STEPS)

    # p-value per step (Welch t-test on Gini values)
    p_vals = []
    for t in range(MAX_STEPS):
        sv = s_mat[:, t][~np.isnan(s_mat[:, t])]
        fv = f_mat[:, t][~np.isnan(f_mat[:, t])]
        if len(sv) > 1 and len(fv) > 1:
            _, p = scipy_stats.ttest_ind(sv, fv, equal_var=False)
        else:
            p = float("nan")
        p_vals.append(p)
    p_vals = np.array(p_vals)

    # ── print numerical summary ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  GINI COEFFICIENT — SUCCESS vs FAILURE")
    print(f"  G = 2∑(i·ᾱᵢ)/(n·∑ᾱᵢ) − (n+1)/n   "
          f"(sorted ascending, mean over {8} heads)")
    print("=" * 70)
    print()

    # episode-level summary variance (mean Gini over all steps each episode)
    s_ep_means = [np.nanmean(t) for t in s_trajs]
    f_ep_means = [np.nanmean(t) for t in f_trajs]
    s_ep_var   = [np.nanvar(t, ddof=1) for t in s_trajs]   # within-episode variance
    f_ep_var   = [np.nanvar(t, ddof=1) for t in f_trajs]

    print("  EPISODE-LEVEL SUMMARY")
    print(f"  {'':20}  {'N':>4}  {'mean G':>8}  {'std G':>8}  "
          f"{'within-ep var':>14}")
    print("  " + "─" * 60)
    for label, ep_means, ep_vars in [
        ("SUCCESS", s_ep_means, s_ep_var),
        ("FAILURE", f_ep_means, f_ep_var),
    ]:
        print(f"  {label:20}  {len(ep_means):>4}  "
              f"{np.mean(ep_means):>8.4f}  "
              f"{np.std(ep_means, ddof=1):>8.4f}  "
              f"{np.mean(ep_vars):>14.6f}")
    _, p_ep = scipy_stats.ttest_ind(s_ep_means, f_ep_means, equal_var=False)
    print(f"  {'Welch t-test (mean G)':20}  {'':>4}  {'':>8}  {'':>8}  "
          f"p={p_ep:.4f}{'  ★' if p_ep<0.05 else ('  ·' if p_ep<0.10 else '')}")
    _, p_var = scipy_stats.ttest_ind(s_ep_var, f_ep_var, equal_var=False)
    print(f"  {'t-test (within-ep var)':20}  {'':>4}  {'':>8}  {'':>8}  "
          f"p={p_var:.4f}{'  ★' if p_var<0.05 else ('  ·' if p_var<0.10 else '')}")

    print()
    print("  PER-STEP BREAKDOWN  (every 5 steps)")
    print(f"  {'step':>5}  {'S mean':>8}  {'S std':>7}  {'S var':>8}  "
          f"│  {'F mean':>8}  {'F std':>7}  {'F var':>8}  │  p-val")
    print("  " + "─" * 72)
    for t in range(0, MAX_STEPS, 5):
        sv = s_mat[:, t][~np.isnan(s_mat[:, t])]
        fv = f_mat[:, t][~np.isnan(f_mat[:, t])]
        if len(sv) < 2 or len(fv) < 2:
            continue
        p = p_vals[t]
        star = "  ★★★" if p<0.001 else ("  ★★" if p<0.01 else ("  ★" if p<0.05 else ("  ·" if p<0.10 else "")))
        print(f"  {t:>5}  {sv.mean():>8.4f}  {sv.std(ddof=1):>7.4f}  "
              f"{sv.var(ddof=1):>8.6f}  "
              f"│  {fv.mean():>8.4f}  {fv.std(ddof=1):>7.4f}  "
              f"{fv.var(ddof=1):>8.6f}  │  {p:.4f}{star}")
    print()

    # ── plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        r"Gini Coefficient of Base-Camera Attention  —  Success vs Failure"
        "\n"
        r"$G = \frac{2\sum_i i\cdot\bar\alpha_i}{n\sum_i\bar\alpha_i} - \frac{n+1}{n}$"
        r"    ($\bar\alpha_i$ = mean-over-heads patch attention, sorted ascending)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel A: mean ± 1 SD trajectory ─────────────────────────────────
    ax = axes[0, 0]
    n_s_at = (~np.isnan(s_mat)).sum(axis=0)
    n_f_at = (~np.isnan(f_mat)).sum(axis=0)
    valid = (n_s_at >= 2) & (n_f_at >= 2)

    ax.fill_between(steps[valid],
                    (s_mean - s_std)[valid], (s_mean + s_std)[valid],
                    color=C_SUCCESS, alpha=0.25)
    ax.fill_between(steps[valid],
                    (f_mean - f_std)[valid], (f_mean + f_std)[valid],
                    color=C_FAILURE, alpha=0.25)
    ax.plot(steps[valid], s_mean[valid], color=C_SUCCESS, lw=2.5, label=f"Success (n={n_s})")
    ax.plot(steps[valid], f_mean[valid], color=C_FAILURE, lw=2.5, label=f"Failure (n={n_f})")

    # annotate significant steps
    for t in steps[valid]:
        p = p_vals[t]
        if p < 0.05:
            y_top = max(s_mean[t] + s_std[t], f_mean[t] + f_std[t])
            ax.annotate(f"p={p:.2f}★", xy=(t, y_top), fontsize=5.5,
                        color="purple", ha="center", va="bottom")

    ax.axvline(10, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(25, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.text(5,  ax.get_ylim()[0], "Ph.1", ha="center", fontsize=7, color="gray")
    ax.text(17, ax.get_ylim()[0], "Ph.2", ha="center", fontsize=7, color="gray")
    ax.text(37, ax.get_ylim()[0], "Ph.3", ha="center", fontsize=7, color="gray")
    ax.set_xlabel("Inference step"); ax.set_ylabel(r"$G$ (Gini)")
    ax.set_title("Mean ± 1 SD trajectory", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel B: per-step variance ────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(steps[valid], s_var[valid], color=C_SUCCESS, lw=2.5, label=f"Success (n={n_s})")
    ax.plot(steps[valid], f_var[valid], color=C_FAILURE, lw=2.5, label=f"Failure (n={n_f})")
    ax.fill_between(steps[valid], 0, s_var[valid], color=C_SUCCESS, alpha=0.18)
    ax.fill_between(steps[valid], 0, f_var[valid], color=C_FAILURE, alpha=0.18)
    ax.axvline(10, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(25, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Inference step"); ax.set_ylabel(r"Var($G$)")
    ax.set_title("Per-step variance across episodes", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel C: spaghetti — individual episode traces ───────────────────
    ax = axes[1, 0]
    for traj in s_trajs:
        ax.plot(np.arange(len(traj)), traj, color=C_SUCCESS, alpha=0.20, lw=0.9)
    for traj in f_trajs:
        ax.plot(np.arange(len(traj)), traj, color=C_FAILURE, alpha=0.20, lw=0.9)
    # bold means on top
    ax.plot(steps[valid], s_mean[valid], color=C_SUCCESS, lw=2.5, label="Success mean")
    ax.plot(steps[valid], f_mean[valid], color=C_FAILURE, lw=2.5, label="Failure mean")
    ax.axvline(10, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(25, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Inference step"); ax.set_ylabel(r"$G$ (Gini)")
    ax.set_title("Individual episode traces + group means", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Panel D: within-episode variance distribution (box + strip) ──────
    ax = axes[1, 1]
    positions = [1, 2]
    data      = [s_ep_var, f_ep_var]
    colors    = [C_SUCCESS, C_FAILURE]
    labels    = [f"Success\n(n={n_s})", f"Failure\n(n={n_f})"]

    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for pos, vals, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(pos + jitter, vals, color=color, s=30, alpha=0.7, zorder=3)

    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Within-episode Var($G$)")
    ax.set_title("Within-episode Gini variance distribution\n"
                 f"(Welch t-test p={p_var:.4f}"
                 f"{'  ★' if p_var<0.05 else ('  ·' if p_var<0.10 else '')})",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "gini_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
