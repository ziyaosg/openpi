from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_names import slugify, patch_dist_filename


def calculate_changes_in_slope(
    *,
    all_series,
    short_label,
    percentage: float = 1.0,
    eps: float = 1e-8,
):
    results = {}

    for modality_key, (_, values) in all_series.items():
        arr = np.asarray(values, dtype=float)

        if arr.size < 3:
            results[short_label(modality_key)] = 0
            continue

        cutoff = max(2, int(len(arr) * percentage))
        arr = arr[:cutoff]

        slopes = np.diff(arr)

        # Vectorized sign computation: ignore near-zero slopes (treat as flat)
        raw_signs = np.sign(slopes)
        signs = raw_signs[np.abs(slopes) > eps]

        if len(signs) < 2:
            results[short_label(modality_key)] = 0
            continue

        num_changes = int(np.sum(np.diff(signs) != 0))

        results[short_label(modality_key)] = num_changes

    return results


def plot_slope_change_scatter(
    *,
    results_per_episode,
    output_dir,
):
    xs, ys, colors = [], [], []

    for i, (ep, slope_dict) in enumerate(results_per_episode):
        xs.append(i)
        ys.append(sum(slope_dict.values()))
        colors.append("green" if ep.success else "red")

    if not xs:
        print("No data for slope change scatter plot")
        return

    plt.figure(figsize=(10, 5))
    plt.scatter(xs, ys, c=colors)
    plt.xlabel("Episode Index")
    plt.ylabel("Total Slope Sign Changes")
    plt.title("Slope Sign Changes per Episode (Green=Success, Red=Failure)")
    plt.tight_layout()

    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slope_change_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved slope change scatter: {out_path}")


def plot_slope_change_per_modality(
    *,
    results_per_episode,
    output_dir,
    percentage: float = 1.0,
):
    if not results_per_episode:
        print("No data to plot")
        return

    if not (0 < percentage <= 1.0):
        raise ValueError("percentage must be in (0, 1]")

    modalities = list(results_per_episode[0][1].keys())
    n = len(modalities)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for idx, modality in enumerate(modalities):
        xs, ys, colors = [], [], []
        for i, (ep, slope_dict) in enumerate(results_per_episode):
            xs.append(i)
            ys.append(slope_dict[modality])
            colors.append("green" if ep.success else "red")

        axes[idx].scatter(xs, ys, c=colors, alpha=0.7)
        axes[idx].set_title(modality)
        axes[idx].set_ylabel("Sign changes")

    axes[-1].set_xlabel("Episode Index")

    pct_str = f"{int(percentage * 100)}%"
    fig.suptitle(f"Slope Sign Changes per Modality (First {pct_str} of Episode)", y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"slope_changes_per_modality_{int(percentage*100)}pct.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-modality slope scatter: {out_path}")


def plot_patch_distribution_per_episode(
    *,
    ep,
    raw_series,
    output_dir,
    short_label,
    ylim_per_key=None,
    n_time_bins=5,
):
    """
    For a single episode, plot a violin for each modality showing the distribution
    of raw patch values (all 256 patches per step, not the reduced scalar) binned
    by normalized episode progress.

    x-axis: time bin (normalized 0→1 in n_time_bins steps)
    y-axis: raw GradCAM / score value
    Each violin: all patch values from all steps in that time bin.

    This directly answers: is attention concentrated in few patches or widespread?
    A narrow violin spiked near zero with outliers = sparse. A wide violin = widespread.
    """
    modality_keys = list(raw_series.keys())
    n = len(modality_keys)
    bin_edges = np.linspace(0.0, 1.0, n_time_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, k in enumerate(modality_keys):
        bins = [[] for _ in range(n_time_bins)]
        for t, patches in raw_series[k]:
            b = min(int(t * n_time_bins), n_time_bins - 1)
            bins[b].append(patches)

        data = [np.concatenate(b) if b else np.empty(0) for b in bins]
        nonempty = [i for i, d in enumerate(data) if d.size > 1]

        if nonempty:
            axes[idx].violinplot(
                [data[i] for i in nonempty],
                positions=[bin_centers[i] for i in nonempty],
                widths=0.8 / n_time_bins,
                showmedians=True,
            )

        axes[idx].set_xticks(bin_centers)
        axes[idx].set_xticklabels(bin_labels, rotation=30, ha="right")
        axes[idx].set_xlabel("Normalized Episode Progress")
        axes[idx].set_ylabel("Raw patch value")
        axes[idx].set_title(short_label(k))
        if ylim_per_key and k in ylim_per_key:
            axes[idx].set_ylim(ylim_per_key[k])

    fig.suptitle(
        f"Raw Patch Distribution over Episode Progress\n"
        f"Episode {ep.episode_num} | success={ep.success}"
    )
    plt.tight_layout()

    filename = patch_dist_filename(ep)

    task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
    ep_folder = f"ep{ep.episode_num}_{'true' if ep.success else 'false'}"
    task_episode_dir = Path(output_dir) / task_folder / ep_folder
    task_episode_dir.mkdir(parents=True, exist_ok=True)

    grouped_dir = Path(output_dir) / ("successes" if ep.success else "failures") / "patch_dist"
    grouped_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(task_episode_dir / filename, dpi=150, bbox_inches="tight")
    plt.savefig(grouped_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved patch distribution: {task_episode_dir / filename}")
    print(f"Also saved patch distribution: {grouped_dir / filename}")


def plot_patch_distribution_aggregated(
    *,
    raw_patches_per_episode,
    output_dir,
    short_label,
    n_bins=80,
):
    """
    One figure with one subplot per modality. Each subplot shows a histogram of
    ALL raw patch values pooled across every step of every episode.

    This gives the global sparsity picture: if the distribution is heavily
    right-skewed (spike near zero, long tail) then only a few patches per step
    have high GradCAM values. If it is more uniform, attention is widespread.
    """
    modality_keys = list(raw_patches_per_episode[0][1].keys())
    pooled = {k: [] for k in modality_keys}

    for _, raw_series in raw_patches_per_episode:
        for k, series in raw_series.items():
            for _, patches in series:
                pooled[k].append(patches)

    n = len(modality_keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]

    for idx, k in enumerate(modality_keys):
        vals = np.concatenate(pooled[k]) if pooled[k] else np.empty(0)
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        axes[idx].hist(vals, bins=n_bins, range=(lo, hi), edgecolor="none")
        axes[idx].set_xlim(lo, hi)
        axes[idx].set_title(short_label(k))
        axes[idx].set_xlabel("Raw patch value")
        axes[idx].set_ylabel("Count (patches × steps × episodes)")

    fig.suptitle("Raw Patch Value Distribution — Aggregated over All Steps and Episodes")
    plt.tight_layout()

    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patch_distribution_aggregated.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated patch distribution: {out_path}")
