from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..attention_utils.io import load_episode_infos, load_episode_records
from ..attention_utils.keys import get_modality_keys
from ..attention_utils.names import (
    attention_plot_filename,
    norm_code,
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


def build_variance_series(raw_series: dict) -> dict:
    """Per-step patch variance derived from a pre-built raw_series.

    For each modality key, computes np.var across patch/token values at each
    step, giving {key: (ts, variances)}.
    """
    return {
        key: (
            [t for t, _ in steps_patches],
            [float(np.var(patches)) for _, patches in steps_patches],
        )
        for key, steps_patches in raw_series.items()
    }


def plot_episode_zscore_with_variance(ep, all_series, var_series, max_steps: int):
    modality_keys = list(REDUCTION_PER_KEY.keys())
    n_mod = len(modality_keys)

    fig_width = max(12, max_steps * 0.12)

    height_ratios = [2.3] + [1.0] * n_mod
    fig, axes = plt.subplots(
        1 + n_mod, 1,
        figsize=(fig_width, 4 + 2 * n_mod),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.15},
        sharex=True,
    )
    ax_main = axes[0]
    ax_vars = axes[1:]

    # ── Zscore line plot ──────────────────────────────────────────────────
    colors = {}
    for k, (s, v) in all_series.items():
        if s:
            (line,) = ax_main.plot(s, v, label=short_label(k))
            colors[k] = line.get_color()
    ax_main.set_ylabel(_ylabel())
    ax_main.set_xlim(-0.5, max_steps - 0.5)
    ax_main.legend()
    ax_main.tick_params(labelbottom=False)

    # ── Per-modality variance bars ────────────────────────────────────────
    for ax, k in zip(ax_vars, modality_keys):
        ts, variances = var_series[k]
        if ts:
            ax.bar(ts, variances, width=0.8, color=colors.get(k, "steelblue"), alpha=0.75)
        ax.set_ylabel(short_label(k), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(labelbottom=False)

    ax_vars[-1].tick_params(labelbottom=True)
    ax_vars[-1].set_xlabel("Inference Step")
    fig.text(0.02, 0.28, "Patch variance (raw)", va="center", rotation="vertical", fontsize=9)

    plt.tight_layout()

    grouped_dir = Path(OUTPUT_DIR) / ("attention_success" if ep.success else "attention_failure")
    grouped_dir.mkdir(parents=True, exist_ok=True)

    norm_str = norm_code(NORM_METHOD)
    filename = attention_plot_filename(ep, norm_str, reduction_code(REDUCTION_PER_KEY))
    plt.savefig(grouped_dir / filename, dpi=150, bbox_inches="tight")
    print(f"Saved zscore+variance plot: {grouped_dir / filename}")
    plt.close()


def calculate_changes_in_slope(
    *,
    all_series,
    short_label,
    n_steps: int,
    eps: float = 1e-8,
):
    results = {}

    for modality_key, (_, values) in all_series.items():
        arr = np.asarray(values, dtype=float)

        if arr.size < 3:
            results[short_label(modality_key)] = 0
            continue

        cutoff = min(n_steps, len(arr))
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
    n_steps: int,
):
    if not results_per_episode:
        print("No data to plot")
        return

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

    fig.suptitle(f"Slope Sign Changes per Modality (First {n_steps} Steps)", y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir) / "slope_sign_changes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"slope_changes_per_modality_{n_steps}steps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved per-modality slope scatter: {out_path}")





def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    max_steps = max(ep.end_idx - ep.start_idx + 1 for ep in episodes)
    print(f"Max episode length: {max_steps} steps")

    all_series_per_episode = []

    for ep in episodes:
        print(f"Processing Episode {ep.episode_num} | success={ep.success}")
        records = load_episode_records(ep, INPUT_DIR)
        all_series = build_batch_series(records)
        raw = build_raw_patch_series(records)
        var_series = build_variance_series(raw)

        all_series_per_episode.append((ep, all_series))

        plot_episode_zscore_with_variance(ep, all_series, var_series, max_steps)

    print("Done plotting individual episodes.")

    # 1–10 individual, then 20, 30, ... up to max_steps
    step_counts = list(range(1, 11)) + list(range(20, max_steps + 1, 10))
    if step_counts[-1] != max_steps:
        step_counts.append(max_steps)

    for n_steps in step_counts:
        results_per_episode = [
            (ep, calculate_changes_in_slope(
                all_series=all_series,
                short_label=short_label,
                n_steps=n_steps,
            ))
            for ep, all_series in all_series_per_episode
        ]
        print(f"Plotting for first {n_steps} steps")
        plot_slope_change_per_modality(
            results_per_episode=results_per_episode,
            output_dir=OUTPUT_DIR,
            n_steps=n_steps,
        )

    plot_slope_change_scatter(
        results_per_episode=results_per_episode,
        output_dir=OUTPUT_DIR,
    )

if __name__ == "__main__":
    main()
    