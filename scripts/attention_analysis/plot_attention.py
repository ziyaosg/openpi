from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .analyze_attention import (
    calculate_extrema,
    plot_extrema_scatter,
    plot_extrema_per_modality,
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

from pathlib import Path
import numpy as np
import os


def save_local_extrema_steps_from_records(
    *,
    input_dir,
    key,
    output_dir,
    name=None,
    eps: float = 1e-8,
):
    """
    For files like step_0.npy, step_1.npy, ..., load one scalar per step from record[key]
    and save the step numbers where that scalar is a local min / local max.

    Assumes load_record(path) exists.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = sorted(
        input_dir.glob("step_*.npy"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if not step_files:
        raise ValueError(f"No step_*.npy files found in {input_dir}")

    steps = []
    values = []

    for path in step_files:
        step = int(path.stem.split("_")[1])
        record = load_record(str(path))
        val = record[key]

        # Reduce to scalar if needed
        arr = np.asarray(val, dtype=float)
        scalar = float(arr.mean())

        steps.append(step)
        values.append(scalar)

    values = np.asarray(values, dtype=float)

    local_mins = []
    local_maxs = []

    for i in range(1, len(values) - 1):
        prev_v = values[i - 1]
        curr_v = values[i]
        next_v = values[i + 1]

        if (curr_v < prev_v - eps) and (curr_v < next_v - eps):
            local_mins.append(steps[i])

        if (curr_v > prev_v + eps) and (curr_v > next_v + eps):
            local_maxs.append(steps[i])

    prefix = name or key.replace("/", "_")

    min_path = output_dir / f"{prefix}_local_mins.txt"
    max_path = output_dir / f"{prefix}_local_maxs.txt"

    with open(min_path, "w") as f:
        for step in local_mins:
            f.write(f"{step}\n")

    with open(max_path, "w") as f:
        for step in local_maxs:
            f.write(f"{step}\n")

    print(f"Saved local mins to: {min_path}")
    print(f"Saved local maxs to: {max_path}")

    return steps, values, local_mins, local_maxs

def plot_individual_image_patch_attention(step: int, key: str):
    record = load_record(os.path.join(INPUT_DIR, f"step_{step}.npy"))
    vals = record[key][0]  # (16, 16)

    if vals.shape != (16, 16):
        raise ValueError(f"Expected (16,16), got {vals.shape}")

    flat_vals = vals.flatten()

    labels = [(i, j) for i in range(16) for j in range(16)]
    x = np.arange(256)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.bar(x, flat_vals)

    plt.ylabel("Raw Attention")
    plt.xlabel("Patch (row, col)")

    tick_positions = x[::16]  # one per row
    tick_labels = [str(labels[i]) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(f"Image Patch Attention | step={step}")

    plt.tight_layout()
    plt.savefig(f"image_patch_attention_step_{step}.png", dpi=150)
    plt.close()


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
        # Uncomment out the line below if you want normalized episode progress
        steps.append(x)
        # steps.append(step)
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

def get_raw_modality_vector(step, key):
    """
    Returns a 1D vector for one modality at one step.
    - image: (16, 16) -> (256,)
    - task/state: (T,) stays (T,)
    """
    record = load_record(os.path.join(INPUT_DIR, f"step_{step}.npy"))
    vals = np.asarray(record[key], dtype=float)
    vals = np.squeeze(vals)

    if vals.ndim == 2:
        if vals.shape != (16, 16):
            raise ValueError(f"{key}: expected (16,16), got {vals.shape}")
        return vals.reshape(-1)  # 256 patches

    if vals.ndim == 1:
        return vals

    raise ValueError(f"{key}: expected 1D or 2D array, got shape {vals.shape}")


def build_episode_multiline_series(ep, key):
    steps = []
    vectors = []

    for step in range(ep.start_idx, ep.end_idx + 1):
        path = os.path.join(INPUT_DIR, f"step_{step}.npy")
        if not os.path.exists(path):
            continue

        try:
            vec = get_raw_modality_vector(step, key)
        except Exception as e:
            print(f"Skipping step {step} for {key}: {e}")
            continue

        steps.append(step)
        vectors.append(vec)

    if not vectors:
        return [], np.empty((0, 0))

    max_dim = max(len(v) for v in vectors)

    padded = np.full((len(vectors), max_dim), np.nan)

    for i, v in enumerate(vectors):
        padded[i, :len(v)] = v

    return steps, padded.T  # (num_lines, num_steps)


def plot_episode_modalities_all_lines(ep, keys=None, alpha=0.15, linewidth=0.8):
    """
    For one episode:
    - image modalities: 256 lines each
    - task/state modalities: one line per token
    - modalities stacked vertically
    """
    if keys is None:
        keys = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]

    series = {}
    for k in keys:
        steps, values = build_episode_multiline_series(ep, k)
        series[k] = (steps, values)

    if not any(len(steps) > 0 for steps, _ in series.values()):
        return

    task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
    ep_folder = f"ep{ep.episode_num}_{'true' if ep.success else 'false'}"

    task_episode_dir = Path(OUTPUT_DIR) / task_folder / ep_folder
    task_episode_dir.mkdir(parents=True, exist_ok=True)

    grouped_dir = Path(OUTPUT_DIR) / ("successes" if ep.success else "failures")
    grouped_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(keys),
        ncols=1,
        figsize=(14, 3.5 * len(keys)),
        sharex=True,
        constrained_layout=True,
    )

    if len(keys) == 1:
        axes = [axes]

    for ax, k in zip(axes, keys):
        steps, values = series[k]

        if len(steps) == 0 or values.size == 0:
            ax.set_title(f"{short_label(k)} (no data)")
            ax.set_ylabel("Attention")
            continue

        # values shape: (num_lines, num_steps)
        for i in range(values.shape[0]):
            ax.plot(
                steps,
                values[i],
                color="#1f77b4",
                alpha=alpha,
                linewidth=linewidth,
            )
        # mean_line = np.nanmean(values, axis=0)
        # ax.plot(steps, mean_line, color="red", linewidth=2, label="mean")
        # ax.legend()   

        ax.set_title(f"{short_label(k)} ({values.shape[0]} lines)")
        ax.set_ylabel("Attention")

    axes[-1].set_xlabel("Step")

    fig.suptitle(
        f"Task {ep.task_id}: {ep.task} | "
        f"Episode {ep.episode_num} | success={ep.success}\n"
        f"Steps {ep.start_idx} to {ep.end_idx}",
        y=1.02,
    )

    filename = f"raw_attention_all_lines_t{ep.task_id}_ep{ep.episode_num}.png"
    task_episode_path = task_episode_dir / filename
    grouped_path = grouped_dir / filename

    plt.savefig(task_episode_path, dpi=150, bbox_inches="tight")
    plt.savefig(grouped_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {task_episode_path}")
    print(f"Also saved: {grouped_path}")

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
        plot_episode_modalities_all_lines(ep)

    print("Done plotting individual episodes.")
    # print("Done plotting individual episodes.")

    # for pct in range(1, 11, 1):
    #     percentage = pct / 100.0

    #     results_per_episode = []

    #     for ep, all_series in all_series_per_episode:
    #         extrema_dict = calculate_extrema(
    #         all_series=all_series,
    #         short_label=short_label,
    #         percentage=percentage,
    #     )

    #     results_per_episode.append((ep, extrema_dict))

    #     print(f"Plotting for first {pct}% of episodes")

    #     plot_extrema_per_modality(
    #         results_per_episode=results_per_episode,
    #         output_dir=OUTPUT_DIR,
    #         percentage=percentage,
    #     )

    # plot_extrema_scatter(
    #     results_per_episode=results_per_episode,
    #     output_dir=OUTPUT_DIR,
    # )


if __name__ == "__main__":
    main()