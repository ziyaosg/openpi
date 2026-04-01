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
    plot_slope_correlation_matrix,
    plot_slope_change_scatter,
    plot_slope_change_per_modality,
)
from .plot_names import (
    slugify,
    short_label,
    norm_code,
    reduction_code,
    attention_plot_filename,
)

INPUT_DIR = "/home/annsong/Documents/policy_records_20260324_141715"
OUTPUT_DIR = "/home/annsong/Documents/policy_records_20260324_141715/plots"
EPISODE_SUMMARIES_JSON = "/home/annsong/Documents/policy_records_20260324_141715/episode_summaries.json"

IMG1_KEY = "outputs/debug/attr/image/right_wrist_0_rgb"
IMG2_KEY = "outputs/debug/attr/image/left_wrist_0_rgb"
TASK_KEY = "outputs/debug/attr/task/scores"
STATE_KEY = "outputs/debug/attr/state/scores"


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


def load_record(npy_path: str) -> dict:
    # print(f"Loading record from {npy_path}")
    p = np.load(npy_path, allow_pickle=True)
    # print(f"Loaded record from {npy_path}, type: {type(p)}, shape: {p.shape}, dtype: {p.dtype}")
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def reduce_attention_scores(step, key, method="average"):
    record = load_record(os.path.join(INPUT_DIR, f"step_{step}.npy"))
    vals = record[key][0]
    if method == "average":
        return float(np.mean(vals))
    if method == "max":
        return float(np.max(vals))
    if method == "min":
        return float(np.min(vals))
    if method == "sum":
        return float(np.sum(vals))
    if method == "sqrt_norm_sum":
        return float(np.sum(vals) / np.sqrt(vals.size))

    raise ValueError(method)


def normalize_modality(values, method="robust_zscore"):
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


def build_episode_series(ep, key, method):
    steps, values = [], []
    length = ep.end_idx - ep.start_idx

    for step in range(ep.start_idx, ep.end_idx + 1):
        path = os.path.join(INPUT_DIR, f"step_{step}.npy")
        if not os.path.exists(path):
            continue
        try:
            val = reduce_attention_scores(step, key, method)
        except Exception:
            continue
        x = (step - ep.start_idx) / length if length else 0.0
        steps.append(x)
        values.append(val)

    return steps, values


def build_all_series(ep, methods, norm="robust_zscore"):
    keys = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]
    all_series = {}

    for k in keys:
        s, v = build_episode_series(ep, k, methods.get(k, "average"))
        all_series[k] = (s, normalize_modality(v, norm))
    
    return all_series


def plot_episode_all_modalities(ep, all_series, methods, norm="robust_zscore"):
    keys = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]

    if not any(len(steps) > 0 for steps, _ in all_series.values()):
        return

    task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
    ep_folder = f"ep{ep.episode_num}_{'true' if ep.success else 'false'}"

    task_episode_dir = Path(OUTPUT_DIR) / task_folder / ep_folder
    task_episode_dir.mkdir(parents=True, exist_ok=True)

    grouped_dir = Path(OUTPUT_DIR) / ("successes" if ep.success else "failures")
    grouped_dir.mkdir(parents=True, exist_ok=True)

    filename = attention_plot_filename(ep, norm_code(norm), reduction_code(methods))

    plt.figure(figsize=(12, 5))

    for k in keys:
        s, v = all_series[k]
        if s:
            plt.plot(s, v, label=short_label(k))

    plt.xlabel("Normalized Episode Progress")

    if norm == "zscore":
        ylabel = "Attention (z-score within episode)"
    elif norm == "robust_zscore":
        ylabel = "Attention (robust z-score within episode)"
    elif norm == "minmax":
        ylabel = "Attention (min-max within episode)"
    else:
        ylabel = "Attention"

    plt.ylabel(ylabel)
    plt.title(
        f"Task {ep.task_id}: {ep.task} | "
        f"Episode {ep.episode_num} | success={ep.success}\n"
        f"Steps {ep.start_idx} to {ep.end_idx}"
    )
    plt.legend()
    plt.tight_layout()

    task_episode_path = task_episode_dir / filename
    grouped_path = grouped_dir / filename

    plt.savefig(task_episode_path, dpi=150)
    plt.savefig(grouped_path, dpi=150)
    print(f"Saved attention plot: {task_episode_path}")
    print(f"Also saved attention plot: {grouped_path}")
    plt.close()

def main():
    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)

    methods = {
        IMG1_KEY: "average",
        IMG2_KEY: "average",
        TASK_KEY: "average",
        STATE_KEY: "average",
    }

    norm = "robust_zscore"

    all_series_per_episode = []

    for ep in episodes:
        print(f"Processing Episode {ep.episode_num} | success={ep.success}")
        all_series = build_all_series(ep, methods, norm)
        all_series_per_episode.append((ep, all_series))

        plot_episode_all_modalities(ep, all_series, methods, norm)

    print("Done plotting individual episodes.")

    for pct in range(1, 11, 1):
        percentage = pct / 100.0

        results_per_episode = []

        for ep, all_series in all_series_per_episode:
            slope_dict = calculate_changes_in_slope(
                all_series=all_series,
                short_label=short_label,
                percentage=percentage,
            )

            results_per_episode.append((ep, slope_dict))

        print(f"Plotting for first {pct}% of episodes")

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