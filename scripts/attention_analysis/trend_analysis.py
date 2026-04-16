"""
Group-averaged attention trend analysis: |zscore| over normalized episode time.

For each modality, plots the mean ± 1 SEM of |zscore(t)| across all success episodes
(green) and all failure episodes (red), binned into N_TIME_BINS steps from 0→1.

Vertical dashed lines mark the mean peak position for each group, making the
early-peak (failure) vs late-peak (success) divergence directly visible.

Run:
    python -m scripts.attention_analysis.trend_analysis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..attention_utils.io import load_episode_infos, load_record, step_path
from ..attention_utils.keys import get_modality_keys
from ..attention_utils.series import extract_patches, normalize_modality, reduce_vals

# ============================================================
# PATHS
# ============================================================
INPUT_DIR  = Path("/home/ziyao/Documents/policy_records_20260407_155916")
OUTPUT_DIR = Path("/home/ziyao/Documents/policy_records_20260407_155916/analysis_gradcam_avg_&_zscore")
EPISODE_SUMMARIES_JSON = INPUT_DIR / "episode_summaries.json"

# ============================================================
# DATA SOURCE & MODALITY KEYS
# ============================================================
DATA_SOURCE = "gradcam"   # "gradcam" or "raw_attn"

MODALITY_KEYS = get_modality_keys(DATA_SOURCE)

# ============================================================
# ANALYSIS SETTINGS
# ============================================================
REDUCTION_METHOD  = "average"
NORM_METHOD       = "zscore"
N_TIME_BINS_LIST  = [20, 50, 100]   # each value produces a separate output PNG
MIN_EPISODE_STEPS = 6               # episodes shorter than this are skipped


def _load_episode_scalar_series(ep) -> dict[str, list[float]]:
    """Reduced scalar per step for every modality, for one episode."""
    series: dict[str, list[float]] = {name: [] for name in MODALITY_KEYS}
    for step in range(ep.start_idx, ep.end_idx + 1):
        p = step_path(INPUT_DIR, step)
        if not p.exists():
            continue
        try:
            rec = load_record(str(p))
        except Exception:
            continue
        for name, key in MODALITY_KEYS.items():
            try:
                v = reduce_vals(extract_patches(rec, key, DATA_SOURCE), REDUCTION_METHOD)
                series[name].append(v)
            except Exception:
                pass
    return series


def _bin_zscore_magnitude(norm_series: list[float], n_time_bins: int, max_steps: int) -> np.ndarray:
    """
    Compute |zscore| binned into n_time_bins equal-width windows over [0, max_steps].

    Returns array of shape (n_time_bins,) with the mean |value| per bin.
    NaN for bins with no data.
    """
    arr = np.asarray(norm_series, dtype=float)
    t = np.arange(len(arr), dtype=float)
    edges = np.linspace(0.0, max_steps, n_time_bins + 1)

    bins = np.full(n_time_bins, np.nan)
    for i in range(n_time_bins):
        mask = (t >= edges[i]) & (t < edges[i + 1])
        if i == n_time_bins - 1:
            mask = (t >= edges[i]) & (t <= edges[i + 1])
        vals = np.abs(arr[mask])
        if vals.size > 0:
            bins[i] = vals.mean()
    return bins


def build_trend_data(episodes, n_time_bins: int, max_steps: int) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns, per modality, per group ("success" / "failure"):
        array of shape (n_episodes, n_time_bins) with binned |zscore|.
    """
    result: dict[str, dict[str, list]] = {
        name: {"success": [], "failure": []} for name in MODALITY_KEYS
    }

    for ep in episodes:
        raw_series = _load_episode_scalar_series(ep)
        group = "success" if ep.success else "failure"

        for name, scalars in raw_series.items():
            if len(scalars) < MIN_EPISODE_STEPS:
                continue
            norm = normalize_modality(scalars, NORM_METHOD)
            binned = _bin_zscore_magnitude(norm, n_time_bins, max_steps)
            result[name][group].append(binned)

    return {
        name: {
            grp: np.array(rows) for grp, rows in groups.items()
        }
        for name, groups in result.items()
    }


def _mean_peak_position(matrix: np.ndarray, n_time_bins: int, max_steps: int) -> float:
    """Mean inference step of the peak |magnitude| bin across episodes."""
    bin_centers = np.linspace(0.0, max_steps, n_time_bins)
    peaks = []
    for row in matrix:
        if (~np.isnan(row)).any():
            peaks.append(bin_centers[np.nanargmax(row)])
    return float(np.mean(peaks)) if peaks else float("nan")


def plot_trend_analysis(trend_data: dict, output_dir: Path, n_time_bins: int, max_steps: int):
    modalities = list(MODALITY_KEYS.keys())
    n = len(modalities)
    bin_centers = np.linspace(0.0, max_steps, n_time_bins)

    fig_width = max(6 * n, n_time_bins * 0.15 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4), sharey=False)
    if n == 1:
        axes = [axes]

    group_style = {
        "success": {"color": "#2ca02c", "label": "success"},
        "failure": {"color": "#d62728", "label": "failure"},
    }

    for ax, name in zip(axes, modalities):
        data = trend_data[name]

        for grp, style in group_style.items():
            matrix = data[grp]  # (n_episodes, n_time_bins)
            if matrix.size == 0:
                continue

            mean = np.nanmean(matrix, axis=0)
            sem  = np.nanstd(matrix, axis=0) / np.sqrt((~np.isnan(matrix)).sum(axis=0).clip(1))

            color = style["color"]
            n_ep  = matrix.shape[0]
            label = f"{style['label']} (n={n_ep})"

            ax.plot(bin_centers, mean, color=color, linewidth=2, label=label)
            ax.fill_between(bin_centers, mean - sem, mean + sem, color=color, alpha=0.25)

            peak = _mean_peak_position(matrix, n_time_bins, max_steps)
            if not np.isnan(peak):
                ax.axvline(peak, color=color, linewidth=1.2, linestyle="--", alpha=0.7)

        ax.set_title(name)
        ax.set_xlabel("Inference step")
        ax.set_ylabel(f"|{NORM_METHOD}| of attention")
        ax.legend(fontsize=8)
        ax.set_xlim(0.0, max_steps)

    source_label = "Raw attention weights" if DATA_SOURCE == "raw_attn" else "GradCAM"
    fig.suptitle(
        f"Attention trend analysis — {source_label}\n"
        f"(mean ± 1 SEM across episodes; dashed = mean peak position)",
        y=1.02,
    )
    plt.tight_layout()

    out_dir = output_dir / "trend_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{n_time_bins}bins.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    print(f"Loaded {len(episodes)} episodes "
          f"({sum(e.success for e in episodes)} success, "
          f"{sum(not e.success for e in episodes)} failure)")

    max_steps = max(ep.end_idx - ep.start_idx + 1 for ep in episodes)
    print(f"Max episode length: {max_steps} steps (x-axis range)")

    for n_time_bins in N_TIME_BINS_LIST:
        print(f"Building trend data ({n_time_bins} bins)...")
        trend_data = build_trend_data(episodes, n_time_bins, max_steps)

        for name, groups in trend_data.items():
            for grp, mat in groups.items():
                print(f"  {name}/{grp}: {mat.shape[0]} episodes")

        plot_trend_analysis(trend_data, OUTPUT_DIR, n_time_bins, max_steps)


if __name__ == "__main__":
    main()
