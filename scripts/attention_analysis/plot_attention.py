from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .analyze_attention import (
    calculate_changes_in_slope,
    plot_slope_change_scatter,
    plot_slope_change_per_modality,
    plot_patch_distribution_per_episode,
    plot_patch_distribution_aggregated,
)
from .plot_names import (
    slugify,
    short_label,
    norm_code,
    reduction_code,
    attention_plot_filename,
)

# ============================================================
# PATHS
# ============================================================
INPUT_DIR = "/home/ziyao/Documents/policy_records_20260324_141715"
OUTPUT_DIR = "/home/ziyao/Documents/policy_records_20260324_141715/attention_analysis_mean_&_zscore"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260324_141715/episode_summaries.json"

# ============================================================
# MODALITY KEYS  (must match keys saved in the .npy files)
# ============================================================
IMG1_KEY  = "outputs/debug/attr/image/right_wrist_0_rgb"
IMG2_KEY  = "outputs/debug/attr/image/left_wrist_0_rgb"
TASK_KEY  = "outputs/debug/attr/task/scores"
STATE_KEY = "outputs/debug/attr/state/scores"

# ============================================================
# REDUCTION METHOD
# How to collapse the 2D heatmap (or 1D score array for task/state)
# down to a single scalar per step.
# Options: "average", "median", "max", "min", "sum", "sqrt_norm_sum"
# Note: use "average" not "mean"
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

# Derived from the three variables above — do not edit directly.
# Used wherever both the modality key and its reduction method are needed together.
REDUCTION_PER_KEY = {
    IMG1_KEY:  REDUCTION_METHOD_IMAGE,
    IMG2_KEY:  REDUCTION_METHOD_IMAGE,
    TASK_KEY:  REDUCTION_METHOD_TASK,
    STATE_KEY: REDUCTION_METHOD_STATE,
}


@dataclass(frozen=True)
class EpisodeInfo:
    task_id: int
    task: str
    episode_num: int
    success: bool
    start_idx: int
    end_idx: int


def load_episode_infos(path: str) -> List[EpisodeInfo]:
    with open(path, "r") as f:
        data = json.load(f)
    return [EpisodeInfo(**d) for d in data]


def _load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def _reduce(vals, method: str) -> float:
    vals = np.asarray(vals)
    if method == "average":
        return float(np.mean(vals))
    if method == "median":
        return float(np.median(vals))
    if method == "max":
        return float(np.max(vals))
    if method == "min":
        return float(np.min(vals))
    if method == "sum":
        return float(np.sum(vals))
    if method == "sqrt_norm_sum":
        return float(np.sum(vals) / np.sqrt(vals.size))
    raise ValueError(f"Unknown reduction method: {method}")


def _load_episode_records(ep) -> list:
    """Load each step's npy file exactly once. Returns list of (normalized_time, record) pairs."""
    length = ep.end_idx - ep.start_idx
    records = []
    for step in range(ep.start_idx, ep.end_idx + 1):
        path = os.path.join(INPUT_DIR, f"step_{step}.npy")
        if not os.path.exists(path):
            continue
        try:
            record = _load_record(path)
        except Exception as e:
            print(f"[WARN] step {step}: {e}")
            continue
        t = (step - ep.start_idx) / length if length else 0.0
        records.append((t, record))
    return records


def normalize_modality(values, method):
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return []

    if method == "robust_zscore":
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return ((arr - med) / (1.4826 * mad)).tolist() if mad > 1e-8 else np.zeros_like(arr).tolist()

    if method == "zscore":
        std = arr.std()
        return ((arr - arr.mean()) / std).tolist() if std > 1e-8 else np.zeros_like(arr).tolist()

    if method == "minmax":
        denom = arr.max() - arr.min()
        return ((arr - arr.min()) / denom).tolist() if denom > 1e-8 else np.zeros_like(arr).tolist()

    if method == "none":
        return arr.tolist()

    raise ValueError(f"Unknown normalization method: {method}")


def build_all_series(records) -> dict:
    """Build normalized scalar series for all modalities from pre-loaded records."""
    series = {k: ([], []) for k in REDUCTION_PER_KEY}

    for t, record in records:
        for k, method in REDUCTION_PER_KEY.items():
            try:
                val = _reduce(record[k][0], method)
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
                patches = np.asarray(record[k][0]).reshape(-1).astype(np.float32)
            except Exception as e:
                print(f"[WARN] key {k}: {e}")
                continue
            raw[k].append((t, patches))
    return raw


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

    task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
    ep_folder = f"ep{ep.episode_num}_{'true' if ep.success else 'false'}"

    task_episode_dir = Path(OUTPUT_DIR) / task_folder / ep_folder
    task_episode_dir.mkdir(parents=True, exist_ok=True)

    grouped_dir = Path(OUTPUT_DIR) / ("successes" if ep.success else "failures") / "attn"
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

    plt.savefig(task_episode_dir / filename, dpi=150)
    plt.savefig(grouped_dir / filename, dpi=150)
    print(f"Saved attention plot: {task_episode_dir / filename}")
    print(f"Also saved attention plot: {grouped_dir / filename}")
    plt.close()


def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)

    all_series_per_episode = []
    raw_patches_per_episode = []

    for ep in episodes:
        print(f"Processing Episode {ep.episode_num} | success={ep.success}")
        records = _load_episode_records(ep)
        all_series = build_all_series(records)
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

    plot_patch_distribution_aggregated(
        raw_patches_per_episode=raw_patches_per_episode,
        output_dir=OUTPUT_DIR,
        short_label=short_label,
    )


if __name__ == "__main__":
    main()
