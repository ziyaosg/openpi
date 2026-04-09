from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class EpisodeInfo:
    task_id: int
    task: str
    episode_num: int
    success: bool
    start_idx: int
    end_idx: int


def load_episode_infos(path: str | Path) -> List[EpisodeInfo]:
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)
    return [EpisodeInfo(**d) for d in data]


@lru_cache(maxsize=4096)
def load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def step_path(input_dir: str | Path, step: int) -> Path:
    return Path(input_dir) / f"step_{step}.npy"

def compute_modality_preview_ranges(ep, input_dir, keys):
    """Compute per-key (min, max) ranges across an episode, with 5 % padding."""
    ranges = {}

    for key in keys:
        global_min = float("inf")
        global_max = float("-inf")

        for step in range(ep.start_idx, ep.end_idx + 1):
            path = step_path(input_dir, step)
            if not path.exists():
                continue

            try:
                record = load_record(str(path))
                vals = np.asarray(record[key][0], dtype=float).flatten()
            except Exception:
                continue

            if vals.size == 0:
                continue

            global_min = min(global_min, float(vals.min()))
            global_max = max(global_max, float(vals.max()))

        if global_min == float("inf") or global_max == float("-inf"):
            ranges[key] = (0.0, 1.0)
        else:
            if abs(global_max - global_min) < 1e-12:
                pad = 1e-3 if global_max == 0 else 0.05 * abs(global_max)
            else:
                pad = 0.05 * (global_max - global_min)
            ranges[key] = (global_min - pad, global_max + pad)

    return ranges