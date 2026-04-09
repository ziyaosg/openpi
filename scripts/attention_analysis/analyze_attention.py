from __future__ import annotations

import numpy as np


def calculate_changes_in_slope(
    *,
    all_series,
    short_label,
    percentage: float = 1.0,
    eps: float = 1e-8,
):
    results = {}

    for modality_key, (_, values) in all_series.items():
        arr = np.asarray(values, dtype=float)

        if arr.size < 3:
            results[short_label(modality_key)] = 0
            continue

        cutoff = max(2, int(len(arr) * percentage))
        arr = arr[:cutoff]

        slopes = np.diff(arr)

        # Vectorized sign computation: ignore near-zero slopes (treat as flat)
        raw_signs = np.sign(slopes)
        signs = raw_signs[np.abs(slopes) > eps]

        if len(signs) < 2:
            results[short_label(modality_key)] = 0
            continue

        num_changes = int(np.sum(np.diff(signs) != 0))

        results[short_label(modality_key)] = num_changes

    return results
