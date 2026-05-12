#!/usr/bin/env python3
"""Raw-alpha (gradient summation) heatmaps for pi05/libero policy records.

Output layout per step:
  top row:    raw-alpha overlay on each camera image
  bottom row: original images
  right panel: task-token raw-alpha scores

Run:
  python3 scripts/pi05_vis/heatmap_raw_alpha.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    CAM_NAMES, CAM_IMAGE_KEYS,
    load_episode_infos, episode_for_step, load_record, step_index, slugify,
    as_u8_rgb, resize_heatmap, overlay_heatmap, make_grid,
    clip_gamma_normalize_joint, make_task_text_panel,
)

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR  = Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260511_021509")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/zs377/visualization_policy_records_pi05_libero_20260511_021509/heatmap_raw_alpha")
EPISODE_JSON = INPUT_DIR / "client_output" / "episode_summaries.json"

# "summation" or "norm" — controls which raw-alpha variant to use
VARIANT = "summation"

APPLY_ABS       = True
CLIP_PERCENTILE = 95
GAMMA           = 0.4
ALPHA           = 0.30
BG              = (255, 255, 255)
GAP_X           = 16
GAP_Y           = 16


# ─── Per-step processing ──────────────────────────────────────────────────────

def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    heats, imgs = [], []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        attr_key = f"outputs/debug/raw_alpha/{VARIANT}/image/{cam_name}"
        if attr_key not in rec:
            raise KeyError(f"Missing {attr_key}")
        heat = np.asarray(rec[attr_key], dtype=np.float32)     # (16, 16)
        if APPLY_ABS:
            heat = np.abs(heat)
        heats.append(heat)
        imgs.append(as_u8_rgb(rec[img_key]))

    heats01 = clip_gamma_normalize_joint(heats, CLIP_PERCENTILE, GAMMA)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]))
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_key = f"outputs/debug/raw_alpha/{VARIANT}/task"
    task_scores = np.asarray(rec[task_key], dtype=np.float32)   # (200,)
    text_panel = make_task_text_panel(rec, task_scores, grid.height)

    combined = Image.new("RGB", (grid.width + GAP_X + text_panel.width, grid.height), BG)
    combined.paste(grid, (0, 0))
    combined.paste(text_panel, (grid.width + GAP_X, 0))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_png)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_infos(EPISODE_JSON)
    files = sorted(glob.glob(str(INPUT_DIR / "step_*.npy")), key=step_index)
    if not files:
        raise FileNotFoundError(f"No step_*.npy in {INPUT_DIR}")

    for f in files:
        s  = step_index(f)
        ep = episode_for_step(s, episodes)
        out = str(
            OUTPUT_DIR
            / f"t{ep.task_id}_{slugify(ep.task)}"
            / f"ep{ep.episode_num}_{str(ep.success).lower()}"
            / f"step_{s:06d}.png"
        )
        try:
            process_one(f, out)
            print(f"[OK] {out}")
        except Exception as e:
            print(f"[SKIP] step {s}: {e}")


if __name__ == "__main__":
    main()
