from __future__ import annotations

import json
import os
import re
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


def step_index(path: str) -> int:
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def episode_for_step(step: int, episodes: List[EpisodeInfo]) -> EpisodeInfo:
    for ep in episodes:
        if ep.start_idx <= step <= ep.end_idx:
            return ep
    raise KeyError(f"Step {step} not covered by any episode")


def load_episode_records(ep, input_dir) -> list:
    """Load each step's npy record for an episode. Returns list of (normalized_time, record) pairs."""
    input_dir = Path(input_dir)
    length = ep.end_idx - ep.start_idx
    records = []
    for step in range(ep.start_idx, ep.end_idx + 1):
        path = step_path(input_dir, step)
        if not path.exists():
            continue
        try:
            record = load_record(str(path))
        except Exception as e:
            print(f"[WARN] step {step}: {e}")
            continue
        t = (step - ep.start_idx) / length if length else 0.0
        records.append((t, record))
    return records
