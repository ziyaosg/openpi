#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .grid_utils import (
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
from .text_utils import render_text_panel_from_token_scores
from ..attention_utils.keys import CAM_IMAGE_KEYS, CAM_NAMES

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260421_133206"
OUTPUT_DIR             = "/home/ziyao/Documents/policy_records_20260421_133206/heatmap_raw_alpha"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260421_133206/episode_summaries.json"

# Raw alpha is the sum of gradients over the embedding dim per patch/token.
# Values can be positive or negative; taking abs before normalizing treats
# both as "salient" regardless of sign.
APPLY_ABS  = True
ALPHA      = 0.30
SMOOTHING  = "BILINEAR"
BG         = (255, 255, 255)
GAP_X      = 16
GAP_Y      = 16

# Raw-alpha distributions have heavy tails: one or two patches can have
# abs values 10–60× the 95th percentile, compressing everything else to
# near-zero after plain min-max.  Clipping at CLIP_PERCENTILE before
# normalizing keeps the joint two-column scale intact while letting the
# other 95 % of patches use the full colour range.
# GAMMA < 1 then pulls mid-range patches into the visible colour band
# (same role as in grid_visualization_raw_weights).
CLIP_PERCENTILE = 95
GAMMA           = 0.4


# ============================================================
# HELPERS
# ============================================================
def to_2d_heatmap(arr: np.ndarray, key: str) -> np.ndarray:
    if arr.ndim == 2:
        h = arr
    elif arr.ndim == 3:
        h = arr[0]
    else:
        raise ValueError(f"{key} must be 2D or 3D (batched), got shape {arr.shape}")
    return h.astype(np.float32)


def clip_normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    """Joint normalization with percentile clipping and gamma stretch.

    1. Pool all abs values across both cameras jointly so the two columns
       stay on the same scale.
    2. Clip at CLIP_PERCENTILE — the top 5 % of patches saturate at 1.0
       but the rest spread across the full colour range.
    3. Gamma < 1 pulls mid-range patches up into the visible colour band.
    """
    flat     = np.concatenate([h.reshape(-1) for h in heats])
    clip_val = float(np.percentile(flat, CLIP_PERCENTILE))
    scale    = clip_val if clip_val > 0 else 1.0
    return [(np.clip(h / scale, 0.0, 1.0) ** GAMMA) for h in heats]


# ============================================================
# PER-STEP PROCESSING
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    heats, imgs = [], []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        attr_key = f"outputs/debug/raw_alpha/summation/image/{cam_name}"
        for k in (attr_key, img_key):
            if k not in rec:
                raise KeyError(f"Missing {k} in {npy_path}")
        heat = to_2d_heatmap(np.asarray(rec[attr_key]), attr_key)
        if APPLY_ABS:
            heat = np.abs(heat)
        heats.append(heat)
        imgs.append(as_u8_rgb(rec[img_key]))

    if np.array_equal(imgs[0], imgs[1]):
        print(f"[WARN] Underlying images identical in {npy_path}. Keys: {CAM_IMAGE_KEYS}")

    heats01 = clip_normalize_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]), smoothing=SMOOTHING)
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_key  = "outputs/debug/raw_alpha/summation/task"
    state_key = "outputs/debug/raw_alpha/summation/state"
    for k in (task_key, state_key):
        if k not in rec:
            raise KeyError(f"Missing {k} in {npy_path} — cannot build text panel")

    task_scores  = np.asarray(rec[task_key][0],  dtype=np.float32)
    state_scores = np.asarray(rec[state_key][0], dtype=np.float32)
    text_panel = render_text_panel_from_token_scores(rec, task_scores, state_scores, grid.height)

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
