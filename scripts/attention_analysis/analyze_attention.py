from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calculate_extrema(
    *,
    all_series,
    short_label,
    percentage: float = 1.0,
    eps: float = 1e-8,
):
    """
    Count extrema via sign flips in slope, ignoring flat regions.
    """
    results = {}

    for modality_key, (_, values) in all_series.items():
        arr = np.asarray(values, dtype=float)

        if arr.size < 3:
            results[short_label(modality_key)] = 0
            continue

        cutoff = max(2, int(len(arr) * percentage))
        arr = arr[:cutoff]

        slopes = np.diff(arr)

        signs = []
        for slope in slopes:
            if slope > eps:
                signs.append(1)
            elif slope < -eps:
                signs.append(-1)
            # ignore flat regions

        if len(signs) < 2:
            results[short_label(modality_key)] = 0
            continue

        num_extrema = sum(
            1 for a, b in zip(signs[:-1], signs[1:]) if a != b
        )

        results[short_label(modality_key)] = num_extrema

    return results

def plot_extrema_scatter(
    *,
    results_per_episode,
    output_dir,
):
    """
    Plot one dot per episode showing total extrema.
    Green = success, Red = failure
    """
    xs = []
    ys = []
    colors = []

    for i, (ep, extrema_dict) in enumerate(results_per_episode):
        total_extrema = sum(extrema_dict.values())
        xs.append(i)
        ys.append(total_extrema)
        colors.append("green" if ep.success else "red")

    if not xs:
        print("No data for extrema scatter plot")
        return

    plt.figure(figsize=(10, 5))
    plt.scatter(xs, ys, c=colors)

    plt.xlabel("Episode Index")
    plt.ylabel("Total Extrema")
    plt.title("Extrema per Episode (Green=Success, Red=Failure)")

    plt.tight_layout()

    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "extrema_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved extrema scatter: {out_path}")


def plot_extrema_per_modality(
    *,
    results_per_episode,
    output_dir,
    percentage: float = 1.0,
):
    """
    Plot extrema per modality for each episode.
    """
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

        for i, (ep, extrema_dict) in enumerate(results_per_episode):
            xs.append(i)
            ys.append(extrema_dict[modality])
            colors.append("green" if ep.success else "red")

        axes[idx].scatter(xs, ys, c=colors, alpha=0.7)
        axes[idx].set_title(modality)
        axes[idx].set_ylabel("Extrema")

    axes[-1].set_xlabel("Episode Index")

    pct_str = f"{int(percentage * 100)}%"
    fig.suptitle(f"Extrema per Modality (First {pct_str} of Episode)", y=1.02)

    plt.tight_layout()

    out_dir = Path(output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"extrema_per_modality_{int(percentage*100)}pct.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved per-modality extrema scatter: {out_path}")

