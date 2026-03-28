from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot_names import slope_correlation_filename


def _analysis_output_dir(output_dir: str, success: bool, kind: str) -> Path:
    analysis_dir = Path(output_dir) / "analysis" / kind
    subdir = "successes" if success else "failures"
    out_dir = analysis_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_slope_correlation_matrix(
    *,
    ep,
    all_series,
    series_normalization,
    output_dir,
    short_label,
    norm_code,
):
    valid_lengths = [len(v) for _, v in all_series.values() if len(v) > 1]
    if not valid_lengths:
        return

    min_len = min(valid_lengths)
    if min_len < 2:
        return

    slope_series = []
    labels = []

    for k, (_, values) in all_series.items():
        if len(values) >= min_len:
            slopes = np.diff(np.asarray(values[:min_len]))
            if len(slopes) > 0:
                slope_series.append(slopes)
                labels.append(short_label(k))

    if len(slope_series) < 2:
        return

    corr = np.corrcoef(np.array(slope_series))

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(f"Slope Correlation Matrix\nEpisode {ep.episode_num} | success={ep.success}")
    plt.colorbar(im).set_label("Slope correlation")
    plt.tight_layout()

    out_dir = _analysis_output_dir(output_dir, ep.success, kind="slope_corr")
    filename = slope_correlation_filename(ep, norm_code(series_normalization))

    plt.savefig(out_dir / filename, dpi=150)
    plt.close()


def calculate_changes_in_slope(
    *,
    all_series,
    short_label,
    eps: float = 1e-8,
):
    """
    Count how many times the slope changes sign for each modality.

    A slope is computed as the first difference between consecutive values.
    Near-zero slopes (abs(slope) <= eps) are ignored.
    """
    results = {}

    for modality_key, (_, values) in all_series.items():
        arr = np.asarray(values, dtype=float)

        # Need at least 3 points to have 2 slopes and therefore 1 possible sign change
        if arr.size < 3:
            results[short_label(modality_key)] = 0
            continue

        slopes = np.diff(arr)

        signs = []
        for slope in slopes:
            if slope > eps:
                signs.append(1)
            elif slope < -eps:
                signs.append(-1)

        if len(signs) < 2:
            results[short_label(modality_key)] = 0
            continue

        num_changes = sum(
            1
            for prev_sign, curr_sign in zip(signs[:-1], signs[1:])
            if prev_sign != curr_sign
        )
        results[short_label(modality_key)] = num_changes

    return results

def plot_slope_change_scatter(
    *,
    results_per_episode,
    output_dir,
):
    """
    Plot one dot per episode showing total slope sign changes.
    Green = success, Red = failure
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    xs = []
    ys = []
    colors = []

    for i, (ep, slope_dict) in enumerate(results_per_episode):
        total_changes = sum(slope_dict.values())

        xs.append(i)  # episode index
        ys.append(total_changes)

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