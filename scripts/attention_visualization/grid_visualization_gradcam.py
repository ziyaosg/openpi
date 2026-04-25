#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .grid_utils import (
    EpisodeInfo,
    load_episode_infos,
    episode_for_step,
    slugify,
    load_record,
    step_index,
    as_u8_rgb,
    resize_heatmap,
    overlay_heatmap,
    make_grid,
)
from .text_utils import render_text_panel_as_image
from ..attention_utils.keys import CAM_IMAGE_KEYS, CAM_NAMES

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260423_180932"
OUTPUT_DIR             = "/home/ziyao/Documents/policy_records_20260423_180932/heatmap_gradcam"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260423_180932/episode_summaries.json"

APPLY_RELU = True
ALPHA      = 0.30
SMOOTHING  = "BILINEAR"
BG         = (255, 255, 255)
GAP_X      = 16
GAP_Y      = 16


# ============================================================
# GRADCAM-SPECIFIC FUNCTIONS
# ============================================================
def to_2d_heatmap(arr: np.ndarray, key: str) -> np.ndarray:
    # attribution arrays are stored with a leading batch dim (1, H, W); strip it
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{key} must be numpy array, got {type(arr)}")
    if arr.ndim == 2:
        h = arr
    elif arr.ndim == 3:
        h = arr[0]
    else:
        raise ValueError(f"{key} must be 2D or 3D (batched), got shape {arr.shape}")
    if h.ndim != 2:
        raise ValueError(f"{key} after unbatch must be 2D, got shape {h.shape}")
    return h.astype(np.float32)


def normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    # normalize both cameras to a shared [0,1] scale so their heatmap
    # intensities are directly comparable in the side-by-side grid
    flat  = np.concatenate([h.reshape(-1) for h in heats])
    mn, mx = float(flat.min()), float(flat.max())
    denom = (mx - mn) if mx > mn else 1.0
    return [(h - mn) / denom for h in heats]



# ============================================================
# PER-STEP PROCESSING
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    heats, imgs = [], []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        attr_key = f"outputs/debug/gradcam/image/{cam_name}"
        if attr_key not in rec:
            raise KeyError(f"Missing {attr_key} in {npy_path}")
        if img_key not in rec:
            raise KeyError(f"Missing {img_key} in {npy_path}")
        heat = to_2d_heatmap(rec[attr_key], attr_key)
        if APPLY_RELU:
            # GradCAM scores can be negative (gradient pointing away from class);
            # keeping only positive values highlights what the model attends to,
            # not what it suppresses
            heat = np.maximum(heat, 0.0)
        heats.append(heat)
        imgs.append(as_u8_rgb(rec[img_key]))

    if np.array_equal(imgs[0], imgs[1]):
        print(f"[WARN] Underlying images identical in {npy_path}. Keys: {CAM_IMAGE_KEYS}")

    heats01 = normalize_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]), smoothing=SMOOTHING)
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid       = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)
    text_panel = render_text_panel_as_image(rec, grid.height)

    combined = Image.new("RGB", (grid.width + GAP_X + text_panel.width, grid.height), BG)
    combined.paste(grid,       (0, 0))
    combined.paste(text_panel, (grid.width + GAP_X, 0))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_png)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)
    files    = sorted(glob.glob(str(in_dir / "step_*.npy")), key=step_index)
    if not files:
        raise FileNotFoundError(f"No step_*.npy found in {INPUT_DIR}")

    for f in files:
        s   = step_index(f)
        ep  = episode_for_step(s, episodes)
        out = str(out_dir / f"t{ep.task_id}_{slugify(ep.task)}" / f"ep{ep.episode_num}_{str(ep.success).lower()}" / f"step_{s:06d}.png")
        process_one(f, out)
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
