from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ALL_MODALITY_KEYS
from ..utils import load_record, step_path, reduce_vals, normalize_modality


def reduce_attention_scores(input_dir, step, key, method="average"):
    record = load_record(str(step_path(input_dir, step)))
    vals = np.asarray(record[key][0], dtype=float)
    return reduce_vals(vals, method)


def build_episode_series(ep, input_dir, key, method):
    steps = []
    values = []

    for step in range(ep.start_idx, ep.end_idx + 1):
        path = step_path(input_dir, step)
        if not path.exists():
            continue

        try:
            val = reduce_attention_scores(input_dir, step, key, method)
        except Exception:
            continue

        # Raw step index on x-axis
        steps.append(step)
        values.append(val)

    return steps, values


def build_all_series(ep, input_dir, methods, norm="robust_zscore"):
    all_series = {}

    for key in ALL_MODALITY_KEYS:
        steps, values = build_episode_series(
            ep=ep,
            input_dir=input_dir,
            key=key,
            method=methods.get(key, "average"),
        )
        all_series[key] = (steps, normalize_modality(values, norm))

    return all_series