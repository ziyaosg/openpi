from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..attention_utils.io import load_episode_infos, load_episode_records
from ..attention_utils.keys import get_modality_keys
from ..attention_utils.names import (
    attention_plot_filename,
    norm_code,
    patch_dist_filename,
    reduction_code,
    short_label,
)
from ..attention_utils.series import FULL_ATTN_KEY, extract_patches, normalize_modality, reduce_vals

# ============================================================
# PATHS
# ============================================================
INPUT_DIR = "/home/ziyao/Documents/policy_records_20260407_155916"
OUTPUT_DIR = "/home/ziyao/Documents/policy_records_20260407_155916/analysis_gradcam_avg_&_zscore"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260407_155916/episode_summaries.json"

# ============================================================
# DATA SOURCE
# "gradcam"  -- use GradCAM attribution maps (outputs/debug/attr/...)
# "raw_attn" -- use raw attention weights    (outputs/debug/raw_attn/...)
# ============================================================
DATA_SOURCE = "gradcam"

# ============================================================
# REDUCTION METHOD
# How to collapse the 2D heatmap (or 1D score array for task/state)
# down to a single scalar per step.
# Options: "average", "median", "max", "min", "sum", "sqrt_norm_sum"
# ============================================================
REDUCTION_METHOD_IMAGE = "average"
REDUCTION_METHOD_TASK  = "average"
REDUCTION_METHOD_STATE = "average"

# ============================================================
# NORMALIZATION METHOD
# How to normalize the per-step scalar series within each episode
# so modalities are on a comparable scale.
# Options: "zscore"         -- center on mean, scale by std
#          "robust_zscore"  -- center on median, scale by MAD
#          "minmax"         -- scale to [0, 1]
#          "none"           -- no normalization, raw values
# ============================================================
NORM_METHOD = "zscore"

# Derived from the variables above — do not edit directly.
_MODALITY_KEYS = list(get_modality_keys(DATA_SOURCE).values())
_REDUCTIONS = [REDUCTION_METHOD_IMAGE, REDUCTION_METHOD_IMAGE, REDUCTION_METHOD_TASK, REDUCTION_METHOD_STATE]
REDUCTION_PER_KEY = dict(zip(_MODALITY_KEYS, _REDUCTIONS))



def build_batch_series(records) -> dict:
    """Build normalized scalar series for all modalities from pre-loaded records."""
    series = {k: ([], []) for k in REDUCTION_PER_KEY}

    for t, record in records:
        for k, method in REDUCTION_PER_KEY.items():
            try:
                val = reduce_vals(extract_patches(record, k, DATA_SOURCE, FULL_ATTN_KEY), method)
            except Exception as e:
                print(f"[WARN] key {k}: {e}")
                continue
            series[k][0].append(t)
            series[k][1].append(val)

    all_series = {}
    for k, (steps, values) in series.items():
        all_series[k] = (steps, normalize_modality(values, NORM_METHOD))
    return all_series


def build_raw_patch_series(records) -> dict:
    """Build raw patch arrays for all modalities from pre-loaded records."""
    keys = list(REDUCTION_PER_KEY)
    raw = {k: [] for k in keys}

    for t, record in records:
        for k in keys:
            try:
                patches = extract_patches(record, k, DATA_SOURCE, FULL_ATTN_KEY)
            except Exception as e:
                print(f"[WARN] key {k}: {e}")
                continue
            raw[k].append((t, patches))
    return raw


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


def _ylabel():
    base = {
        "zscore": "z-score",
        "robust_zscore": "robust z-score",
        "minmax": "min-max [0,1]",
        "none": "raw attention",
    }.get(NORM_METHOD, NORM_METHOD)
    return f"Attention ({base})"


def plot_episode_all_modalities(ep, all_series):
    if not any(len(steps) > 0 for steps, _ in all_series.values()):
        return

    grouped_dir = Path(OUTPUT_DIR) / ("attention_success" if ep.success else "attention_failure")
    grouped_dir.mkdir(parents=True, exist_ok=True)

    norm_str = norm_code(NORM_METHOD)
    filename = attention_plot_filename(ep, norm_str, reduction_code(REDUCTION_PER_KEY))

    plt.figure(figsize=(12, 5))
    for k, (s, v) in all_series.items():
        if s:
            plt.plot(s, v, label=short_label(k))

    plt.xlabel("Normalized Episode Progress")
    plt.ylabel(_ylabel())
    plt.title(
        f"Task {ep.task_id}: {ep.task} | "
        f"Episode {ep.episode_num} | success={ep.success}\n"
        f"Steps {ep.start_idx} to {ep.end_idx}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(grouped_dir / filename, dpi=150)
    print(f"Saved attention plot: {grouped_dir / filename}")
    plt.close()


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

    out_dir = Path(output_dir) / "slope_sign_changes"
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

    out_dir = Path(output_dir) / "slope_sign_changes"
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

    grouped_dir = Path(output_dir) / ("patch_dist_success" if ep.success else "patch_dist_failure")
    grouped_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(grouped_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved patch distribution: {grouped_dir / filename}")



def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)

    all_series_per_episode = []
    raw_patches_per_episode = []

    for ep in episodes:
        print(f"Processing Episode {ep.episode_num} | success={ep.success}")
        records = load_episode_records(ep, INPUT_DIR)
        all_series = build_batch_series(records)
        raw = build_raw_patch_series(records)

        all_series_per_episode.append((ep, all_series))
        raw_patches_per_episode.append((ep, raw))

        plot_episode_all_modalities(ep, all_series)

    # Compute global y-limits per modality (1st–99th percentile across all episodes)
    # so that the same modality subplot uses the same scale in every episode's plot.
    modality_keys = list(raw_patches_per_episode[0][1].keys())
    ylim_per_key = {}
    for k in modality_keys:
        all_patches = np.concatenate([
            patches
            for _, raw in raw_patches_per_episode
            for _, patches in raw[k]
        ])
        ylim_per_key[k] = (float(np.percentile(all_patches, 1)), float(np.percentile(all_patches, 99)))

    for ep, raw in raw_patches_per_episode:
        plot_patch_distribution_per_episode(
            ep=ep,
            raw_series=raw,
            output_dir=OUTPUT_DIR,
            short_label=short_label,
            ylim_per_key=ylim_per_key,
        )

    print("Done plotting individual episodes.")

    # 1–10% in 1% steps, then 20–100% in 10% steps
    percentages = [p / 100.0 for p in range(1, 11)] + [p / 100.0 for p in range(20, 110, 10)]

    for percentage in percentages:
        results_per_episode = [
            (ep, calculate_changes_in_slope(
                all_series=all_series,
                short_label=short_label,
                percentage=percentage,
            ))
            for ep, all_series in all_series_per_episode
        ]
        print(f"Plotting for first {int(percentage * 100)}% of episode")
        plot_slope_change_per_modality(
            results_per_episode=results_per_episode,
            output_dir=OUTPUT_DIR,
            percentage=percentage,
        )

    plot_slope_change_scatter(
        results_per_episode=results_per_episode,
        output_dir=OUTPUT_DIR,
    )

if __name__ == "__main__":
    main()
    