from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .config import ALL_MODALITY_KEYS, OUTPUT_DIR
from .utils import compute_modality_preview_ranges
from .modality_renderers import render_raw_attention_bar
from ..plot_names import short_label


def _ylabel_for_norm(norm: str) -> str:
    if norm == "zscore":
        return "Attention (z-score within episode)"
    if norm == "robust_zscore":
        return "Attention (robust z-score within episode)"
    if norm == "minmax":
        return "Attention (min-max within episode)"
    return "Attention"


def plot_episode_all_modalities_interactive(ep, input_dir, all_series, methods, norm="robust_zscore"):
    if not any(len(steps) > 0 for steps, _ in all_series.values()):
        return

    preview_ranges = compute_modality_preview_ranges(
        ep=ep,
        input_dir=input_dir,
        keys=ALL_MODALITY_KEYS,
    )

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.4])

    ax_main = fig.add_subplot(gs[0])
    ax_preview = fig.add_subplot(gs[1])

    line_meta = {}

    for key in ALL_MODALITY_KEYS:
        steps, values = all_series[key]
        if not steps:
            continue

        line, = ax_main.plot(
            steps,
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=short_label(key),
        )

        line_meta[line] = {
            "key": key,
            "steps": np.asarray(steps, dtype=float),
            "values": np.asarray(values, dtype=float),
        }

    ax_main.set_xlabel("Step Index")
    ax_main.set_ylabel(_ylabel_for_norm(norm))
    ax_main.set_title(
        f"Task {ep.task_id}: {ep.task} | "
        f"Episode {ep.episode_num} | success={ep.success}\n"
        f"Steps {ep.start_idx} to {ep.end_idx}"
    )
    ax_main.legend()

    annotation = ax_main.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->"),
    )
    annotation.set_visible(False)

    ax_preview.set_title("Hover over a point to see raw attention by patch/token")
    ax_preview.set_xlabel("Patch Index / Token Index")
    ax_preview.set_ylabel("Raw Attention")

    state = {"line": None, "idx": None, "step": None, "key": None}

    def nearest_hit(event):
        if event.inaxes != ax_main or event.xdata is None or event.ydata is None:
            return None

        best = None
        best_dist = float("inf")

        for line, meta in line_meta.items():
            xs = meta["steps"]
            ys = meta["values"]

            dx = xs - event.xdata
            dy = ys - event.ydata
            d2 = dx * dx + dy * dy

            idx = int(np.argmin(d2))
            dist = float(d2[idx])

            if dist < best_dist:
                best_dist = dist
                best = (line, idx, xs[idx], ys[idx], meta["key"])

        return best, best_dist

    def on_move(event):
        result = nearest_hit(event)
        if result is None:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        hit, dist = result
        line, idx, x, y, key = hit

        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()
        xspan = xlim[1] - xlim[0]
        yspan = ylim[1] - ylim[0]
        threshold = (0.03 * xspan) ** 2 + (0.05 * yspan) ** 2

        if dist > threshold:
            if annotation.get_visible():
                annotation.set_visible(False)
                fig.canvas.draw_idle()
            return

        annotation.xy = (x, y)
        annotation.set_text(
            f"modality={short_label(key)}\n"
            f"step={int(round(x))}\n"
            f"value={y:.3f}"
        )
        annotation.set_visible(True)

        if state["line"] is not line or state["idx"] != idx:
            render_raw_attention_bar(
                ax=ax_preview,
                input_dir=input_dir,
                step=int(round(x)),
                key=key,
                y_range=preview_ranges.get(key),
            )
            state["line"] = line
            state["idx"] = idx
            state["step"] = int(round(x))
            state["key"] = key

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key != "s":
            return
        if state["step"] is None:
            print("Nothing hovered yet — move over a point first, then press s.")
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = (
            f"t{ep.task_id}_ep{ep.episode_num}"
            f"_step{state['step']}_{short_label(state['key'])}.png"
        )
        path = OUTPUT_DIR / fname
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()