from __future__ import annotations

import numpy as np

from .config import IMAGE_MODALITY_KEYS
from .utils import load_record, step_path
from ..plot_names import short_label


def render_raw_attention_bar(ax, input_dir, step: int, key: str, y_range=None):
    record = load_record(str(step_path(input_dir, step)))
    vals = np.asarray(record[key][0], dtype=float)

    if key in IMAGE_MODALITY_KEYS:
        if vals.shape != (16, 16):
            raise ValueError(f"Expected shape (16, 16) for image modality {key}, got {vals.shape}")
        flat_vals = vals.flatten()
        x = np.arange(len(flat_vals))
        tick_positions = x[::16]
        tick_labels = [str(i) for i in tick_positions]
    else:
        flat_vals = vals.flatten()
        x = np.arange(len(flat_vals))
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
