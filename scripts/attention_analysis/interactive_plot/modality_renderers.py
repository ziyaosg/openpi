from __future__ import annotations

import numpy as np

from .config import IMAGE_MODALITY_KEYS, DATA_SOURCE
from ...attention_utils.io import load_record, step_path
from ...attention_utils.names import short_label
from ...attention_utils.series import extract_patches


def render_raw_attention_bar(ax, input_dir, step: int, key: str, y_range=None):
    record = load_record(str(step_path(input_dir, step)))
    flat_vals = extract_patches(record, key, DATA_SOURCE).astype(float)
    x = np.arange(len(flat_vals))

    if key in IMAGE_MODALITY_KEYS:
        tick_positions = x[::16]
        tick_labels = [str(i) for i in tick_positions]
    else:
        tick_positions = None
        tick_labels = None

    ax.clear()
    ax.bar(x, flat_vals)
    ax.set_title(f"{short_label(key)} | step={step}")
    ax.set_xlabel("Patch Index / Token Index")
    ax.set_ylabel("Raw Attention")

    if tick_positions is not None:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

    if y_range is not None:
        ax.set_ylim(*y_range)
