from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .constants import ALL_MODALITY_KEYS, OUTPUT_DIR
from ..plot_names import (
    attention_plot_filename,
    norm_code,
    reduction_code,
    short_label,
    slugify,
)


def plot_episode_all_modalities_static(ep, all_series, methods, norm="robust_zscore"):
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

    for key in ALL_MODALITY_KEYS:
        steps, values = all_series[key]
        if steps:
            plt.plot(steps, values, label=short_label(key))

    plt.xlabel("Step Index")

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