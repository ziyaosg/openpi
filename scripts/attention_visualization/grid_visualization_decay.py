#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")

from .grid_utils import (
    load_episode_infos,
    episode_for_step,
    slugify,
    load_record,
    step_index,
    as_u8_rgb,
    resize_heatmap,
    overlay_heatmap,
)

from ..attention_utils.keys import CAM_IMAGE_KEYS, CAM_NAMES
from .grid_visualization_gradcam import to_2d_heatmap


# ============================================================
# CONFIG
# ============================================================
INPUT_DIR  = "/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/policy_records_20260426_100003"
OUTPUT_DIR = "/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/policy_records_20260426_100003/decay_overlay"
EPISODE_JSON = f"{INPUT_DIR}/episode_summaries.json"

IMG_SIZE = 420
ALPHA = 0.5
SMOOTHING = "BILINEAR"

NUM_ACTION_TOKENS = 5
ACTION_DECAY = 0.7

FALLBACK_SHAPE = (14, 14)

ROW_GAP = 30
LABEL_W = 260


# ============================================================
# FONT
# ============================================================
def load_font(size=28):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

FONT = load_font()


# ============================================================
# LOAD ACTION TOKENS
# ============================================================
def extract_tokens(rec: dict, branch: str):
    base = f"outputs/debug/{branch}/action_tokens"

    tokens = []
    for t in range(NUM_ACTION_TOKENS):
        per_cam = []
        for cam in CAM_NAMES:
            key = f"{base}/{t}/{cam}"
            try:
                if key in rec:
                    per_cam.append(to_2d_heatmap(rec[key], cam))
                else:
                    per_cam.append(None)
            except Exception:
                per_cam.append(None)
        tokens.append(per_cam)

    return tokens


# ============================================================
# NORMALIZE
# ============================================================
def norm(h):
    if h is None:
        return None
    h = np.asarray(h, dtype=np.float32)
    if h.max() > 0:
        h = h / (h.max() + 1e-8)
    return h


# ============================================================
# DECAY FUSION
# ============================================================
def decay_fuse(tokens):
    weights = np.array([ACTION_DECAY ** i for i in range(len(tokens))], dtype=np.float32)
    weights /= weights.sum() + 1e-8

    fused = []
    for cam_idx in range(len(CAM_NAMES)):
        acc = None
        for t_idx, w in enumerate(weights):
            h = tokens[t_idx][cam_idx]
            if h is None:
                continue
            h = np.asarray(h, dtype=np.float32)
            acc = w * h if acc is None else acc + w * h

        if acc is None:
            acc = np.zeros(FALLBACK_SHAPE, dtype=np.float32)

        if acc.max() > 0:
            acc /= acc.max()

        fused.append(acc)

    return fused


# ============================================================
# TOKEN ROWS (5 DISTINCT HEATMAPS)
# ============================================================
def token_rows(tokens):
    """
    returns:
        list[token][cam] heatmaps
    """
    out = []

    for t in range(NUM_ACTION_TOKENS):
        per_cam = []
        for cam_idx in range(len(CAM_NAMES)):
            h = norm(tokens[t][cam_idx])
            if h is None:
                h = np.zeros(FALLBACK_SHAPE, dtype=np.float32)
            per_cam.append(h)
        out.append(per_cam)

    return out


# ============================================================
# OVERLAY
# ============================================================
def overlay_all(imgs, heats):
    out = []
    for img, h in zip(imgs, heats):
        out.append(
            overlay_heatmap(
                img,
                resize_heatmap(h, (IMG_SIZE, IMG_SIZE), SMOOTHING),
                ALPHA,
            )
        )
    return out


# ============================================================
# LABEL
# ============================================================
def draw_label(draw, text, x, y, h):
    bbox = draw.textbbox((0, 0), text, font=FONT)
    th = bbox[3] - bbox[1]
    draw.text((x, y + (h - th) // 2), text, fill=(20, 20, 20), font=FONT)


# ============================================================
# PROCESS ONE STEP
# ============================================================
def process_one(npy_path: str, out_png: str):
    rec = load_record(npy_path)

    imgs = [
        np.array(
            Image.fromarray(as_u8_rgb(rec[k])).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
        )
        for k in CAM_IMAGE_KEYS
    ]

    tokens = extract_tokens(rec, "gradcam")

    decayed = decay_fuse(tokens)
    token_grid = token_rows(tokens)

    n_cols = len(CAM_IMAGE_KEYS)
    row_h = IMG_SIZE

    canvas_w = LABEL_W + n_cols * IMG_SIZE
    canvas_h = (1 + NUM_ACTION_TOKENS) * (row_h + ROW_GAP)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # ========================================================
    # ROW 1: DECAYED
    # ========================================================
    y = 0
    draw_label(draw, "Decayed (5-token fusion)", 10, y, row_h)

    vis = overlay_all(imgs, decayed)

    for c, img in enumerate(vis):
        x = LABEL_W + c * IMG_SIZE
        canvas.paste(Image.fromarray(img), (x, y))

    # ========================================================
    # ROWS 2–6: ONE ROW PER ACTION TOKEN
    # ========================================================
    for t in range(NUM_ACTION_TOKENS):
        y = (t + 1) * (row_h + ROW_GAP)
        draw_label(draw, f"Action Token t{t}", 10, y, row_h)

        for cam_idx in range(n_cols):
            h = token_grid[t][cam_idx]

            vis = overlay_heatmap(
                imgs[cam_idx],
                resize_heatmap(h, (IMG_SIZE, IMG_SIZE), SMOOTHING),
                ALPHA,
            )

            x = LABEL_W + cam_idx * IMG_SIZE
            canvas.paste(Image.fromarray(vis), (x, y))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

    print(f"[OK] saved {out_png}")


# ============================================================
# MAIN
# ============================================================
def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_infos(EPISODE_JSON)
    files = sorted(glob.glob(str(in_dir / "step_*.npy")), key=step_index)

    for f in files:
        step = step_index(f)
        ep = episode_for_step(step, episodes)

        outcome = "success" if ep.success else "failure"
        out = out_dir / slugify(ep.task) / outcome / f"ep_{ep.episode_num:03d}" / f"step_{step:06d}.png"
        process_one(f, str(out))


if __name__ == "__main__":
    main()