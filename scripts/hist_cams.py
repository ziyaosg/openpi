#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for saving files
matplotlib.use("Agg")

# ------ CONFIG ------
CAMERAS = ["base_0_rgb", "left_wrist_0_rgb"]
CAM_KEYS = {c: f"cam/{c}" for c in CAMERAS}

# --------------------
# Helper functions
# --------------------
def sanitize(name: str, maxlen: int = 140) -> str:
    name = name.strip().lower().replace(" ", "_")
    import re
    name = re.sub(r"[^a-z0-9._-]+", "", name)
    return name[:maxlen]


def load_episodes(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict) and "EPISODES" in data:
        episodes = data["EPISODES"]
    else:
        raise ValueError(f"Unrecognized episodes JSON format in {path}")
    for i, ep in enumerate(episodes):
        if "start_idx" not in ep or "end_idx" not in ep:
            raise ValueError(f"Episode {i} missing start_idx/end_idx")
    return episodes


def get_episode_index(idx: int, episodes: List[Dict]) -> Optional[int]:
    for i, ep in enumerate(episodes):
        if ep["start_idx"] <= idx <= ep["end_idx"]:
            return i
    return None


def collect_cam_values(raw_dir: str, episodes: List[Dict]):
    npz_paths = sorted(glob.glob(os.path.join(raw_dir, "step_*.npz")))
    n_steps = len(npz_paths)

    # global values
    values_global: Dict[str, List[np.ndarray]] = {c: [] for c in CAMERAS}
    # per-task values
    values_task: List[Dict[str, List[np.ndarray]]] = [
        {c: [] for c in CAMERAS} for _ in episodes
    ]
    # per-image values
    values_image: List[Dict[str, np.ndarray]] = [
        {c: np.array([], dtype=np.float32) for c in CAMERAS} for _ in range(n_steps)
    ]

    for idx, npz_path in enumerate(npz_paths):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Failed to load {os.path.basename(npz_path)}: {e}")
            continue

        ep_idx = get_episode_index(idx, episodes)

        for cam in CAMERAS:
            key = CAM_KEYS[cam]
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            flat = arr.reshape(-1)

            # global
            values_global[cam].append(flat)

            # per-task
            if ep_idx is not None:
                values_task[ep_idx][cam].append(flat)

            # per-image
            values_image[idx][cam] = flat

        data.close()

    # concatenate global
    for cam in CAMERAS:
        if values_global[cam]:
            values_global[cam] = np.concatenate(values_global[cam])
        else:
            values_global[cam] = np.array([], dtype=np.float32)

    return npz_paths, values_global, values_task, values_image


def compute_overall_min_max(values_global: Dict[str, np.ndarray]) -> Tuple[float, float]:
    mins = []
    maxs = []
    for cam in CAMERAS:
        v = values_global[cam]
        if v.size == 0:
            continue
        mins.append(v.min())
        maxs.append(v.max())
    if not mins:
        raise RuntimeError("No CAM values found to compute global min/max.")
    overall_min = float(np.min(mins))
    overall_max = float(np.max(maxs))
    # avoid zero-width range
    if overall_min == overall_max:
        overall_max = overall_min + 1e-6
    return overall_min, overall_max


def compute_ymax_for_category(
    all_groups: List[Dict[str, np.ndarray]],
    bin_edges: np.ndarray,
) -> float:
    """Get max count across all histograms in a category."""
    ymax = 0.0
    for group in all_groups:
        for cam in CAMERAS:
            vals = group[cam]
            if vals.size == 0:
                continue
            counts, _ = np.histogram(vals, bins=bin_edges)
            local_max = float(counts.max()) if counts.size > 0 else 0.0
            if local_max > ymax:
                ymax = local_max
    # avoid degenerate
    if ymax <= 0:
        ymax = 1.0
    return ymax


def plot_histogram_pair(
    cam_values: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    title: str,
    save_path: str,
    n_bins_label: int,
    total_cells_label: Dict[str, int],
    ylim: Tuple[float, float],
):
    """
    cam_values: dict cam -> 1D array of values
    bin_edges: shared bin edges for both cams
    ylim: (ymin, ymax) common for category
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bin_width = bin_edges[1] - bin_edges[0]
    y_min, y_max = ylim

    for ax, cam in zip(axes, CAMERAS):
        vals = cam_values[cam]
        if vals.size > 0:
            counts, edges = np.histogram(vals, bins=bin_edges)
        else:
            counts = np.zeros(len(bin_edges) - 1, dtype=int)
            edges = bin_edges

        centers = 0.5 * (edges[:-1] + edges[1:])
        bars = ax.bar(
            centers,
            counts,
            width=bin_width * 0.9,
            align="center",
            alpha=0.7,
            edgecolor="black",
        )

        # Annotate each bar with count (vertical, slightly above bar)
        for x, c in zip(centers, counts):
            if c <= 0:
                continue
            y_offset = (y_max - y_min) * 0.01
            ax.text(
                x,
                c + y_offset,
                f"{int(c)}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                rotation_mode="anchor",
            )

        ax.set_title(f"{cam}", fontsize=11)
        ax.set_xlabel("CAM activation", fontsize=10)
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(y_min, y_max)

        # Text box: bins, width, range, total cells
        total_cells = total_cells_label.get(cam, 0)
        txt = (
            f"bins = {n_bins_label}\n"
            f"bin width ≈ {bin_width:.6f}\n"
            f"range = [{bin_edges[0]:.6f}, {bin_edges[-1]:.6f}]\n"
            f"total cells = {total_cells}"
        )
        ax.text(
            0.99,
            0.99,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build histograms of raw CAM activations (global, per-task, per-image)."
    )
    ap.add_argument(
        "--raw-dir",
        required=True,
        help="Directory with raw step_*.npz (e.g. .../29999/raw)",
    )
    ap.add_argument(
        "--episodes-json",
        required=True,
        help="JSON with episode/task splits and start_idx/end_idx.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output root directory for histograms.",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=70,
        help="Number of bins (shared across all plots).",
    )
    args = ap.parse_args()

    raw_dir = args.raw_dir
    out_root = args.out_dir
    episodes = load_episodes(args.episodes_json)

    os.makedirs(out_root, exist_ok=True)
    dir_global = os.path.join(out_root, "global")
    dir_task = os.path.join(out_root, "per_task")
    dir_image = os.path.join(out_root, "per_image")
    os.makedirs(dir_global, exist_ok=True)
    os.makedirs(dir_task, exist_ok=True)
    os.makedirs(dir_image, exist_ok=True)

    # 1) Collect all raw CAM values
    npz_paths, values_global, values_task, values_image = collect_cam_values(
        raw_dir, episodes
    )

    # 2) Compute a single global min/max from all values (across both cams)
    overall_min, overall_max = compute_overall_min_max(values_global)
    print(f"[INFO] Global raw CAM range: min={overall_min:.6f}, max={overall_max:.6f}")

    # Shared bin edges for all categories
    bin_edges = np.linspace(overall_min, overall_max, args.bins + 1)

    # --------------------------------------------------
    # CATEGORY 1: GLOBAL (whole run, both cameras)
    # --------------------------------------------------
    global_group = {
        cam: values_global[cam] for cam in CAMERAS
    }
    ymax_global = compute_ymax_for_category([global_group], bin_edges)
    ylim_global = (0.0, ymax_global * 1.10)

    total_cells_global = {
        cam: int(global_group[cam].size) for cam in CAMERAS
    }

    global_title = "Global CAM activations (entire run)"
    global_save = os.path.join(dir_global, "global_base_vs_wrist.png")
    plot_histogram_pair(
        global_group,
        bin_edges,
        global_title,
        global_save,
        n_bins_label=args.bins,
        total_cells_label=total_cells_global,
        ylim=ylim_global,
    )

    # --------------------------------------------------
    # CATEGORY 2: PER TASK
    # --------------------------------------------------
    # Build per-task groups
    per_task_groups: List[Dict[str, np.ndarray]] = []
    for i, ep in enumerate(episodes):
        task_group = {}
        for cam in CAMERAS:
            if values_task[i][cam]:
                task_group[cam] = np.concatenate(values_task[i][cam])
            else:
                task_group[cam] = np.array([], dtype=np.float32)
        per_task_groups.append(task_group)

    # ymin/ymax shared across all tasks
    ymax_task = compute_ymax_for_category(per_task_groups, bin_edges)
    ylim_task = (0.0, ymax_task * 1.10)

    for i, ep in enumerate(episodes):
        cam_values = per_task_groups[i]
        # Skip completely empty tasks (just in case)
        if all(v.size == 0 for v in cam_values.values()):
            continue

        raw_min = min(
            (v.min() for v in cam_values.values() if v.size > 0),
            default=0.0,
        )
        raw_max = max(
            (v.max() for v in cam_values.values() if v.size > 0),
            default=0.0,
        )
        print(
            f"TASK {ep['task_id']} raw min={raw_min:.6f} max={raw_max:.6f}"
        )
        for cam in CAMERAS:
            v = cam_values[cam]
            if v.size > 0:
                print(
                    f"  - {cam}: min={v.min():.6f} max={v.max():.6f}"
                )

        task_name = sanitize(ep["task"])
        title = f"Task {ep['task_id']}: {ep['task']}"
        save_path = os.path.join(
            dir_task,
            f"task_{ep['task_id']}_{task_name}.png",
        )

        total_cells_task = {
            cam: int(cam_values[cam].size) for cam in CAMERAS
        }

        plot_histogram_pair(
            cam_values,
            bin_edges,
            title,
            save_path,
            n_bins_label=args.bins,
            total_cells_label=total_cells_task,
            ylim=ylim_task,
        )

    # --------------------------------------------------
    # CATEGORY 3: PER IMAGE (per step)
    # --------------------------------------------------
    # Build list of per-image groups, one per step
    per_image_groups: List[Dict[str, np.ndarray]] = []
    for idx in range(len(npz_paths)):
        group = {}
        for cam in CAMERAS:
            group[cam] = values_image[idx][cam]
        per_image_groups.append(group)

    ymax_image = compute_ymax_for_category(per_image_groups, bin_edges)
    ylim_image = (0.0, ymax_image * 1.10)

    for idx, npz_path in enumerate(npz_paths):
        cam_values = per_image_groups[idx]
        if all(v.size == 0 for v in cam_values.values()):
            continue

        # Extract step id from filename: step_XXXXXXXXXXXX.npz
        stem = os.path.splitext(os.path.basename(npz_path))[0]  # e.g. "step_1763598408087"
        fname = f"pair_{idx:05d}_{stem}.png"
        save_path = os.path.join(dir_image, fname)

        total_cells_img = {
            cam: int(cam_values[cam].size) for cam in CAMERAS
        }

        title = f"Per-image CAM activations: {stem}"
        plot_histogram_pair(
            cam_values,
            bin_edges,
            title,
            save_path,
            n_bins_label=args.bins,
            total_cells_label=total_cells_img,
            ylim=ylim_image,
        )


if __name__ == "__main__":
    main()
