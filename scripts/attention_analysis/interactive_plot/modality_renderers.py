from __future__ import annotations

import numpy as np

from .utils import load_record, step_path
from ..plot_names import short_label


def render_raw_attention_bar(ax, input_dir, step: int, key: str, y_range=None):
    record = load_record(str(step_path(input_dir, step)))
    vals = np.asarray(record[key][0], dtype=float)
    flat_vals = vals.flatten()
    x = np.arange(len(flat_vals))

    ax.clear()
    ax.bar(x, flat_vals)

    ax.set_title(f"{short_label(key)} | step={step}")
    ax.set_xlabel("Patch Index / Token Index")
    ax.set_ylabel("Raw Attention")

    if y_range is not None:
        ax.set_ylim(*y_range)

    if vals.shape == (16, 16):
        tick_positions = x[::16]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i) for i in tick_positions], rotation=45)


def render_modality_preview(ax, input_dir, step: int, key: str, y_range=None):
    render_raw_attention_bar(
        ax=ax,
        input_dir=input_dir,
        step=step,
        key=key,
        y_range=y_range,
    )