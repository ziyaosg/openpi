"""
Group-averaged attention trend analysis: |zscore| over actual inference steps.

For each modality, plots the mean ± 1 SEM of |zscore(t)| across all success episodes
(green) and all failure episodes (red), aligned on a shared step axis 0..max_steps.

Three vertical dashed lines are drawn:
  - success (green --):  mean peak step across success episodes
  - failure (red --):    mean peak step across failure episodes (full length)
  - failure* (red :):    mean peak step across failure episodes, limited to the
                         same step range covered by success (max_success_steps)

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


def build_trend_data(episodes, max_steps: int) -> dict[str, dict[str, np.ndarray]]:
    """
    Returns, per modality, per group ("success" / "failure"):
        array of shape (n_episodes, max_steps) with |zscore| at each step.
    Steps beyond an episode's actual length are NaN.
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
            norm = np.abs(normalize_modality(scalars, NORM_METHOD))
            padded = np.full(max_steps, np.nan)
            padded[:len(norm)] = norm
            result[name][group].append(padded)

    return {
        name: {grp: np.array(rows) for grp, rows in groups.items()}
        for name, groups in result.items()
    }


def _mean_peak_step(matrix: np.ndarray, step_limit: int | None = None) -> float:
    """Mean step index of the peak |magnitude| across episodes.

    Args:
        matrix:     shape (n_episodes, max_steps)
        step_limit: if given, only consider steps 0..step_limit-1
    """
    peaks = []
    for row in matrix:
        r = row[:step_limit] if step_limit is not None else row
        if (~np.isnan(r)).any():
            peaks.append(int(np.nanargmax(r)))
    return float(np.mean(peaks)) if peaks else float("nan")


def plot_trend_analysis(
    trend_data: dict,
    output_dir: Path,
    max_steps: int,
    max_success_steps: int,
):
    modalities = list(MODALITY_KEYS.keys())
    n = len(modalities)
    step_indices = np.arange(max_steps)

    fig_width = max(6 * n, max_steps * 0.08 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, modalities):
        data = trend_data[name]

        # ── success ──────────────────────────────────────────────────────
        success_matrix = data["success"]
        if success_matrix.size > 0:
            mean = np.nanmean(success_matrix, axis=0)
            sem  = np.nanstd(success_matrix, axis=0) / np.sqrt(
                (~np.isnan(success_matrix)).sum(axis=0).clip(1)
            )
            n_ep = success_matrix.shape[0]
            ax.plot(step_indices, mean, color="#2ca02c", linewidth=2,
                    label=f"success (n={n_ep})")
            ax.fill_between(step_indices, mean - sem, mean + sem,
                            color="#2ca02c", alpha=0.25)

            peak = _mean_peak_step(success_matrix)
            if not np.isnan(peak):
                ax.axvline(peak, color="#2ca02c", linewidth=1.2,
                           linestyle="--", alpha=0.8,
                           label=f"success peak ({peak:.0f})")

        # ── failure ───────────────────────────────────────────────────────
        failure_matrix = data["failure"]
        if failure_matrix.size > 0:
            mean = np.nanmean(failure_matrix, axis=0)
            sem  = np.nanstd(failure_matrix, axis=0) / np.sqrt(
                (~np.isnan(failure_matrix)).sum(axis=0).clip(1)
            )
            n_ep = failure_matrix.shape[0]
            ax.plot(step_indices, mean, color="#d62728", linewidth=2,
                    label=f"failure (n={n_ep})")
            ax.fill_between(step_indices, mean - sem, mean + sem,
                            color="#d62728", alpha=0.25)

            # full-length peak
            peak_full = _mean_peak_step(failure_matrix)
            if not np.isnan(peak_full):
                ax.axvline(peak_full, color="#d62728", linewidth=1.2,
                           linestyle="--", alpha=0.8,
                           label=f"failure peak ({peak_full:.0f})")

            # peak limited to the same range as success
            if max_success_steps < max_steps:
                peak_trunc = _mean_peak_step(failure_matrix,
                                             step_limit=max_success_steps)
                if not np.isnan(peak_trunc):
                    ax.axvline(peak_trunc, color="#d62728", linewidth=1.2,
                               linestyle=":", alpha=0.8,
                               label=f"failure peak (first {max_success_steps} steps): {peak_trunc:.0f}")

        ax.set_title(name)
        ax.set_xlabel("Inference step")
        ax.set_ylabel(f"|{NORM_METHOD}| of attention")
        ax.legend(fontsize=7)
        ax.set_xlim(0, max_steps)

    source_label = "Raw attention weights" if DATA_SOURCE == "raw_attn" else "GradCAM"
    fig.suptitle(
        f"Attention trend analysis — {source_label}\n"
        f"(mean ± 1 SEM across episodes; dashed = mean peak step; "
        f"dotted = failure peak over first {max_success_steps} steps)",
        y=1.02,
    )
    plt.tight_layout()

    out_dir = output_dir / "trend_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    print(f"Loaded {len(episodes)} episodes "
          f"({sum(e.success for e in episodes)} success, "
          f"{sum(not e.success for e in episodes)} failure)")

    max_steps = max(ep.end_idx - ep.start_idx + 1 for ep in episodes)
    max_success_steps = max(
        (ep.end_idx - ep.start_idx + 1 for ep in episodes if ep.success),
        default=max_steps,
    )
    print(f"Max episode length: {max_steps} steps | "
          f"Max success episode length: {max_success_steps} steps")

    print("Building trend data...")
    trend_data = build_trend_data(episodes, max_steps)

    for name, groups in trend_data.items():
        for grp, mat in groups.items():
            print(f"  {name}/{grp}: {mat.shape[0]} episodes")

    plot_trend_analysis(trend_data, OUTPUT_DIR, max_steps, max_success_steps)


if __name__ == "__main__":
    main()
