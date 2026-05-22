#!/usr/bin/env python3
"""
Supplemental window-based conformal analysis.

Loads cached feature data from conformal_analysis.py and adds:
1. Sliding-window AUROC: at horizon T, best AUROC using window mean of last W steps
2. Multivariate conformal score: combine top-K features via Mahalanobis distance
3. Detection-time distribution: for each failure episode, at what step T would
   a calibrated single-feature alarm first fire?
4. Composite score ROC: combine gini_base + plan_consistency + action_magnitude

Run AFTER conformal_analysis.py has cached its feature data.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

BASE       = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
CACHE_DIR  = BASE / "analysis_conformal" / "feature_cache"
OUTPUT_DIR = BASE / "analysis_conformal"
MAX_STEPS  = 60
FPR        = 0.05

BLUE_M = "#1f5fa6"; RED_M = "#b02020"

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def to_matrix(ep_list: List[List[dict]], key: str, max_steps: int) -> np.ndarray:
    out = np.full((len(ep_list), max_steps), np.nan)
    for i, ep in enumerate(ep_list):
        for rec in ep:
            t = rec.get("rel_step", 0)
            if t < max_steps:
                out[i, t] = rec.get(key, np.nan)
    return out


def rolling_mean_matrix(mat: np.ndarray, W: int) -> np.ndarray:
    """Rolling mean over last W steps. Shape stays (N, T)."""
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[1]):
        window = mat[:, max(0, t - W + 1): t + 1]
        out[:, t] = np.nanmean(window, axis=1)
    return out


def auroc_at_step(S: np.ndarray, F: np.ndarray, t: int, min_n: int = 5) -> float:
    sv = S[:, t]; sv = sv[~np.isnan(sv)]
    fv = F[:, t]; fv = fv[~np.isnan(fv)]
    if len(sv) < min_n or len(fv) < min_n:
        return np.nan
    try:
        stat, _ = mannwhitneyu(fv, sv, alternative="two-sided")
        a = float(stat) / (len(sv) * len(fv))
        return max(a, 1.0 - a)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 1. WINDOW AUROC
# ─────────────────────────────────────────────────────────────────────────────

def fig_window_auroc(
    ep_data: Dict,
    group_name: str,
    features: List[str],
    windows: Tuple[int, ...] = (1, 3, 5, 10, 20),
):
    """
    For the top features, show AUROC(T) using a rolling window of size W.
    Demonstrates how aggregating over more steps improves early detection.
    """
    top_feats = features[:min(6, len(features))]
    n_f = len(top_feats)
    n_w = len(windows)

    fig, axes = plt.subplots(n_f, 1, figsize=(13, 2.8 * n_f + 0.5), sharex=True)
    if n_f == 1:
        axes = [axes]

    steps = np.arange(MAX_STEPS)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_w))

    for ax, feat in zip(axes, top_feats):
        S_raw = to_matrix(ep_data["success"], feat, MAX_STEPS)
        F_raw = to_matrix(ep_data["failure"], feat, MAX_STEPS)

        for col, W in zip(colors, windows):
            S = rolling_mean_matrix(S_raw, W)
            F = rolling_mean_matrix(F_raw, W)
            aurocs = np.array([auroc_at_step(S, F, t) for t in range(MAX_STEPS)])
            valid = ~np.isnan(aurocs)
            ax.plot(steps[valid], aurocs[valid], color=col, lw=1.8, label=f"W={W}")

        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.axhline(0.7, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0.45, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title(feat, loc="left", fontsize=9, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Step T (detection horizon)")
    fig.suptitle(
        f"Window AUROC — {group_name}\n"
        f"(how rolling window mean of W steps improves AUROC over single step)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig_window_auroc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTION TIME HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

def fig_detection_time(
    ep_data: Dict,
    features: List[str],
    group_name: str,
    W: int = 5,
    fpr: float = FPR,
):
    """
    For top-5 features, show: at what step does each failure episode
    first get flagged? Plotted as CDF (fraction detected by step T).
    """
    top_feats = features[:min(5, len(features))]
    successes = ep_data.get("success", [])
    failures  = ep_data.get("failure", [])
    if len(successes) < 10 or len(failures) < 5:
        return

    n_cal = int(len(successes) * 0.7)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(successes))
    cal_eps = [successes[i] for i in perm[:n_cal]]

    steps = np.arange(MAX_STEPS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_cdf = axes[0]; ax_box = axes[1]

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(top_feats)))

    all_detect_times = {}
    for feat, col in zip(top_feats, colors):
        S_cal = rolling_mean_matrix(to_matrix(cal_eps, feat, MAX_STEPS), W)
        F_mat = rolling_mean_matrix(to_matrix(failures, feat, MAX_STEPS), W)

        detect_step = np.full(len(failures), MAX_STEPS + 1, dtype=float)

        for t in range(1, MAX_STEPS):
            cal_v = S_cal[:, t]; cal_v = cal_v[~np.isnan(cal_v)]
            if len(cal_v) < 5:
                continue
            cal_m = cal_v.mean()
            scores_cal = np.abs(cal_v - cal_m)
            thresh = np.quantile(scores_cal, 1.0 - fpr)

            fail_v = F_mat[:, t]
            fail_s = np.abs(fail_v - cal_m)
            for fi, fs in enumerate(fail_s):
                if not np.isnan(fs) and fs > thresh and detect_step[fi] > t:
                    detect_step[fi] = t

        all_detect_times[feat] = detect_step

        detected = detect_step[detect_step <= MAX_STEPS]
        n_fail = len(failures)
        xs = np.sort(detected)
        ys = np.arange(1, len(xs) + 1) / n_fail
        ax_cdf.step(xs, ys, color=col, lw=2.0, where="post",
                    label=f"{feat} (detected {len(detected)}/{n_fail})")

    ax_cdf.axhline(1.0 - fpr, color="gray", ls=":", lw=1, label=f"Expected FPR={fpr*100:.0f}%")
    ax_cdf.set_xlabel("Detection step T")
    ax_cdf.set_ylabel("Fraction of failures detected")
    ax_cdf.set_ylim(-0.02, 1.05)
    ax_cdf.set_title(f"Cumulative detection rate (W={W}, FPR≤{fpr*100:.0f}%)", fontweight="bold")
    ax_cdf.legend(fontsize=7, loc="lower right")
    ax_cdf.spines[["top", "right"]].set_visible(False)

    # Box plot of detection times
    data_box = [all_detect_times[f][all_detect_times[f] <= MAX_STEPS].tolist()
                for f in top_feats]
    labels_box = [f[:20] for f in top_feats]
    bp = ax_box.boxplot(data_box, vert=True, patch_artist=True,
                        labels=labels_box, showfliers=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax_box.set_ylabel("Detection step")
    ax_box.set_title("Distribution of first-detection steps", fontweight="bold")
    ax_box.tick_params(axis="x", rotation=25, labelsize=7)
    ax_box.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Early detection analysis — {group_name}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig_detection_time.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTIVARIATE COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def fig_composite_score(
    ep_data: Dict,
    features: List[str],
    group_name: str,
    top_k: int = 5,
    W: int = 5,
    fpr: float = FPR,
):
    """
    Composite nonconformity score: standardized sum of top-K features.
    At each step T, z-score each feature against calibration mean/std,
    then sum z-scores. Shows TPR vs FPR at each step.
    """
    top_feats = features[:top_k]
    successes = ep_data.get("success", [])
    failures  = ep_data.get("failure", [])
    if len(successes) < 10 or len(failures) < 5:
        return

    n_cal = int(len(successes) * 0.7)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(successes))
    cal_eps   = [successes[i] for i in perm[:n_cal]]
    probe_eps = [successes[i] for i in perm[n_cal:]]
    if len(probe_eps) < 3:
        probe_eps = cal_eps[:max(3, n_cal // 5)]

    # Build rolling-mean matrices per feature
    S_mats = {f: rolling_mean_matrix(to_matrix(cal_eps,   f, MAX_STEPS), W) for f in top_feats}
    P_mats = {f: rolling_mean_matrix(to_matrix(probe_eps, f, MAX_STEPS), W) for f in top_feats}
    F_mats = {f: rolling_mean_matrix(to_matrix(failures,  f, MAX_STEPS), W) for f in top_feats}

    tpr_comp = np.full(MAX_STEPS, np.nan)
    fpr_comp = np.full(MAX_STEPS, np.nan)

    # Also track best single-feature TPR for comparison
    best_single_tpr = np.full(MAX_STEPS, np.nan)
    best_feat_at_t  = [""] * MAX_STEPS

    for t in range(1, MAX_STEPS):
        # Per-feature z-scores using calibration stats
        cal_means, cal_stds = {}, {}
        for f in top_feats:
            cv = S_mats[f][:, t]; cv = cv[~np.isnan(cv)]
            if len(cv) < 5:
                break
            cal_means[f] = cv.mean()
            cal_stds[f]  = cv.std() + 1e-9
        else:
            # Composite score for calibration
            def composite(mats_dict):
                z_scores = []
                for f in top_feats:
                    v = mats_dict[f][:, t]
                    z = np.abs(v - cal_means[f]) / cal_stds[f]
                    z_scores.append(z)
                return np.nansum(np.stack(z_scores, axis=1), axis=1)

            cal_comp   = composite(S_mats)
            probe_comp = composite(P_mats)
            fail_comp  = composite(F_mats)

            cal_valid = cal_comp[~np.isnan(cal_comp)]
            if len(cal_valid) < 5:
                continue
            thresh = np.quantile(cal_valid, 1.0 - fpr)

            n_fail_active = np.sum(~np.isnan(fail_comp))
            if n_fail_active < 1:
                continue
            tpr_comp[t] = float(np.nansum(fail_comp > thresh)) / n_fail_active

            probe_valid = probe_comp[~np.isnan(probe_comp)]
            if len(probe_valid) > 0:
                fpr_comp[t] = float((probe_valid > thresh).sum()) / len(probe_valid)

            # Best single-feature TPR
            feat_tprs = {}
            for f in top_feats:
                cv = S_mats[f][:, t]; cv = cv[~np.isnan(cv)]
                if len(cv) < 5:
                    continue
                cm = cv.mean(); cstd = cv.std() + 1e-9
                cs = np.abs(cv - cm) / cstd
                thresh_f = np.quantile(cs, 1.0 - fpr)
                fv = F_mats[f][:, t]
                fscore = np.abs(fv - cm) / cstd
                n_f_active = np.sum(~np.isnan(fscore))
                if n_f_active > 0:
                    feat_tprs[f] = float(np.nansum(fscore > thresh_f)) / n_f_active

            if feat_tprs:
                bf = max(feat_tprs, key=feat_tprs.get)
                best_single_tpr[t] = feat_tprs[bf]
                best_feat_at_t[t]  = bf
            continue

    fig, ax = plt.subplots(figsize=(12, 5))
    steps = np.arange(MAX_STEPS)

    valid = ~np.isnan(tpr_comp)
    ax.plot(steps[valid], tpr_comp[valid], color=RED_M, lw=2.5, label=f"Composite (top-{top_k})", zorder=3)
    valid_s = ~np.isnan(best_single_tpr)
    ax.plot(steps[valid_s], best_single_tpr[valid_s], color=BLUE_M, lw=2.0, ls="--",
            label="Best single feature", zorder=2)
    valid_f = ~np.isnan(fpr_comp)
    ax.plot(steps[valid_f], fpr_comp[valid_f], color="gray", lw=1.5, ls=":", label=f"FPR (target={fpr})")
    ax.axhline(fpr, color="gray", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Step T (detection horizon)")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"Composite vs single-feature conformal score — {group_name}\n"
        f"W={W} steps window, FPR target={fpr*100:.0f}%",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUTPUT_DIR / group_name / "fig_composite_score.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for cache_file in sorted(CACHE_DIR.glob("*.pkl")):
        group_name = cache_file.stem
        print(f"\n=== {group_name} ===")
        with open(cache_file, "rb") as f:
            ep_data = pickle.load(f)

        ns = len(ep_data.get("success", []))
        nf = len(ep_data.get("failure", []))
        print(f"  {ns} success, {nf} failure episodes")

        if ns < 5 or nf < 5:
            print("  [skip] not enough episodes")
            continue

        # Get sorted feature list from main analysis
        main_auroc_txt = OUTPUT_DIR / "feature_ranking_report.txt"
        if main_auroc_txt.exists():
            # Parse the ranking report for this group
            lines = main_auroc_txt.read_text().split("\n")
            feats = []
            in_group = False
            for line in lines:
                if f"GROUP: {group_name}" in line:
                    in_group = True
                elif in_group and "GROUP:" in line:
                    break
                elif in_group and line and not line.startswith("-") and not line.startswith("=") and not line.startswith("F"):
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0][0].isalpha():
                        feats.append(parts[0])
            if not feats:
                # Fallback: compute AUROC here
                from scripts.attention_analysis.conformal_analysis import (
                    compute_auroc_sequence, feature_names
                )
                all_f = feature_names(ep_data)
                all_a = {}
                for f in all_f:
                    a = compute_auroc_sequence(ep_data, f, MAX_STEPS)
                    all_a[f] = np.nanmean(a) if not np.all(np.isnan(a)) else 0.5
                feats = sorted(all_a, key=all_a.get, reverse=True)
        else:
            # Compute features ranking directly
            import sys
            sys.path.insert(0, str(Path(__file__).parents[2]))
            from scripts.attention_analysis.conformal_analysis import (
                compute_auroc_sequence, feature_names
            )
            all_f = feature_names(ep_data)
            all_a = {}
            for f in all_f:
                a = compute_auroc_sequence(ep_data, f, MAX_STEPS)
                all_a[f] = np.nanmean(a) if not np.all(np.isnan(a)) else 0.5
            feats = sorted(all_a, key=all_a.get, reverse=True)

        print(f"  Running supplemental analyses for {len(feats)} features …")

        print("  Window AUROC …")
        fig_window_auroc(ep_data, group_name, feats)

        print("  Detection time histogram …")
        fig_detection_time(ep_data, feats, group_name)

        print("  Composite score …")
        fig_composite_score(ep_data, feats, group_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
