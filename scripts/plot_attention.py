from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

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

    infos: List[EpisodeInfo] = []
    for d in data:
        infos.append(
            EpisodeInfo(
                task_id=int(d["task_id"]),
                task=str(d["task"]),
                episode_num=int(d["episode_num"]),
                success=bool(d["success"]),
                start_idx=int(d["start_idx"]),
                end_idx=int(d["end_idx"]),
            )
        )
    return infos


def slugify(s: str, max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def reduce_attention_scores(step: int, modality_key: str, method: str = "average") -> float:
    record = load_record(os.path.join(INPUT_DIR, f"step_{step}.npy"))
    if modality_key not in record:
        raise KeyError(f"{modality_key} not found in record for step {step}")

    attention_scores = record[modality_key][0]
    if method == "average":
        return float(np.mean(attention_scores))
    elif method == "max":
        return float(np.max(attention_scores))
    elif method == "min":
        return float(np.min(attention_scores))
    elif method == "sum":
        return float(np.sum(attention_scores))
    elif method == "sqrt_norm_sum":
        return float(np.sum(attention_scores) / np.sqrt(attention_scores.size))
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_modality(values: list[float], method: str = "zscore") -> list[float]:
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return []

    if method == "none":
        return arr.tolist()

    if method == "zscore":
        std = arr.std()
        if std < 1e-8:
            return np.zeros_like(arr).tolist()
        return ((arr - arr.mean()) / std).tolist()

    if method == "robust_zscore":
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-8:
            return np.zeros_like(arr).tolist()
        return ((arr - median) / (1.4826 * mad)).tolist()

    if method == "minmax":
        denom = arr.max() - arr.min()
        if denom < 1e-8:
            return np.zeros_like(arr).tolist()
        return ((arr - arr.min()) / denom).tolist()

    raise ValueError(f"Unknown normalization method: {method}")


def short_label(modality_key: str) -> str:
    mapping = {
        IMG1_KEY: "right_wrist_rgb",
        IMG2_KEY: "left_wrist_rgb",
        TASK_KEY: "task",
        STATE_KEY: "state",
    }
    return mapping.get(modality_key, modality_key)


def norm_code(method: str) -> str:
    return {
        "zscore": "z",
        "robust_zscore": "rz",
        "minmax": "mm",
        "none": "raw",
    }.get(method, slugify(method))


def method_code(method: str) -> str:
    return {
        "average": "avg",
        "sqrt_norm_sum": "sqrt",
        "sum": "sum",
        "max": "max",
        "min": "min",
    }.get(method, slugify(method))


def reduction_code(methods: dict[str, str]) -> str:
    unique = set(methods.values())
    if len(unique) == 1:
        return method_code(next(iter(unique)))
    return "mixed"


def build_episode_series(
    ep: EpisodeInfo,
    modality_key: str,
    reduction_method: str,
) -> tuple[list[float], list[float]]:
    """
    For each step in the episode, load the attention scores for the given modality,
    apply the reduction method, and build a series of (normalized_step, value) pairs.
    """
    steps: list[float] = []
    values: list[float] = []

    episode_length = ep.end_idx - ep.start_idx

    for step in range(ep.start_idx, ep.end_idx + 1):
        npy_path = os.path.join(INPUT_DIR, f"step_{step}.npy")
        if not os.path.exists(npy_path):
            print(f"Missing file for step {step}, skipping")
            continue

        try:
            value = reduce_attention_scores(step, modality_key, reduction_method)
        except Exception as e:
            print(f"Skipping step {step} for {modality_key}: {e}")
            continue

        if episode_length == 0:
            x = 0.0
        else:
            x = (step - ep.start_idx) / episode_length

        steps.append(x)
        values.append(value)

    return steps, values


def plot_episode_all_modalities(
    ep: EpisodeInfo,
    default_reduction_method: str = "average",
    modality_reduction_methods: dict[str, str] | None = None,
    series_normalization: str = "robust_zscore",
) -> None:
    modality_keys = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]
    modality_reduction_methods = modality_reduction_methods or {}

    all_series: dict[str, tuple[list[float], list[float]]] = {}
    used_reduction_methods: dict[str, str] = {}

    for modality_key in modality_keys:
        reduction_method = modality_reduction_methods.get(modality_key, default_reduction_method)
        used_reduction_methods[modality_key] = reduction_method

        steps, raw_values = build_episode_series(
            ep=ep,
            modality_key=modality_key,
            reduction_method=reduction_method,
        )

        normalized_values = normalize_modality(raw_values, method=series_normalization)
        all_series[modality_key] = (steps, normalized_values)

    if not any(len(steps) > 0 for steps, _ in all_series.values()):
        print(f"No valid data for episode {ep.episode_num}")
        return

    task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
    ep_folder = f"ep{ep.episode_num}_{'success' if ep.success else 'failure'}"
    plot_dir = Path(OUTPUT_DIR) / task_folder / ep_folder
    plot_dir.mkdir(parents=True, exist_ok=True)

    success_dir = Path(OUTPUT_DIR) / "successes"
    failure_dir = Path(OUTPUT_DIR) / "failures"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    norm_str = norm_code(series_normalization)
    red_str = reduction_code(used_reduction_methods)

    base_name = f"t{ep.task_id}_ep{ep.episode_num}"
    filename = f"{base_name}_attn_{norm_str}_{red_str}.png"
    out_png = plot_dir / filename

    plt.figure(figsize=(12, 5))

    for modality_key in modality_keys:
        steps, values = all_series[modality_key]
        if steps:
            label = f"{short_label(modality_key)} ({used_reduction_methods[modality_key]})"
            plt.plot(steps, values, label=label)

    plt.xlabel("Normalized Episode Progress")

    if series_normalization == "zscore":
        ylabel = "Attention (z-score within episode)"
    elif series_normalization == "robust_zscore":
        ylabel = "Attention (robust z-score within episode)"
    elif series_normalization == "minmax":
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

    plt.savefig(out_png, dpi=150)

    extra_path = (success_dir if ep.success else failure_dir) / filename
    plt.savefig(extra_path, dpi=150)

    plt.close()

    print(f"Saved: {out_png}")
    print(f"Also saved: {extra_path}")


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)

    modality_reduction_methods = {
        IMG1_KEY: "average",
        IMG2_KEY: "average",
        TASK_KEY: "average",
        STATE_KEY: "average",
    }

    for ep in episodes:
        plot_episode_all_modalities(
            ep,
            default_reduction_method="average",
            modality_reduction_methods=modality_reduction_methods,
            series_normalization="robust_zscore",
        )


if __name__ == "__main__":
    main()