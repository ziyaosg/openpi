#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

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
from .text_utils import render_text_panel_from_token_scores
from ..attention_utils.keys import CAM_IMAGE_KEYS, CAM_NAMES, IMAGE_PATCH_GRID
from ..attention_utils.series import FULL_ATTN_KEY

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260407_155916"
OUTPUT_DIR             = "/home/ziyao/Documents/policy_records_20260407_155916/heatmap_raw_weights"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260407_155916/episode_summaries.json"

# Layer whose attention is used for the final heatmap.
TARGET_LAYER = 16

# Layer used only for head selection.  Layer 1 puts ~55 % of its budget into
# image tokens and its head ranking is stable across steps.
HEAD_SELECTION_LAYER = 1

# How many top image-focused heads to include.
TOP_K_HEADS = 3


ALPHA      = 0.30
SMOOTHING  = "BILINEAR"
BG         = (255, 255, 255)
GAP_X      = 16
GAP_Y      = 16
PERCENTILE_CLIP      = 95   # clip hot outliers before normalizing
GAMMA                = 0.4  # power stretch to brighten mid-range patches

# ============================================================
# RAW-WEIGHTS-SPECIFIC FUNCTIONS
# ============================================================
def select_heads(
    layer_attn: np.ndarray,
    all_img_spans: List[Tuple[int, int]],
    task_span: Tuple[int, int] | None,
    state_span: Tuple[int, int] | None,
    top_k: int,
) -> np.ndarray:
    """
    Rank attention heads by image/(task+state) budget ratio and return the
    indices of the top_k most image-focused heads.
    """
    H = layer_attn.shape[0]

    img_budget = np.zeros(H, dtype=np.float32)
    for s0, s1 in all_img_spans:
        img_budget += layer_attn[:, s0:s1].sum(axis=1)

    text_budget = np.zeros(H, dtype=np.float32)
    if task_span is not None:
        text_budget += layer_attn[:, task_span[0]:task_span[1]].sum(axis=1)
    if state_span is not None:
        text_budget += layer_attn[:, state_span[0]:state_span[1]].sum(axis=1)

    ratio = img_budget / (text_budget + 1e-9)
    return np.argsort(ratio)[-min(top_k, H):]


def log_normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    # Step 1: divide by total image-token attention across all cameras.
    # Raw softmax budgets are dominated by text tokens (50-90%), so image scores
    # are tiny in absolute terms. Rescaling by the image budget brings them into
    # a comparable range without being swamped by the text portion.
    total = sum(float(h.sum()) for h in heats)
    denom = total if total > 0 else 1.0
    # Step 2: log1p compress — squashes the heavy tail of a few hot patches so
    # the rest of the patches are visible rather than appearing uniformly dark.
    compressed = [np.log1p(np.maximum(h.astype(np.float32) / denom, 0.0)) for h in heats]

    flat     = np.concatenate([h.reshape(-1) for h in compressed])
    mn       = float(flat.min())
    # Step 3: clip at the 95th percentile before min-max so that 1-2 extreme
    # patches don't monopolise the colour scale (without this, ~250/256 patches
    # map to near-zero and look invisible)
    clip_val = float(np.percentile(flat, PERCENTILE_CLIP))
    scale    = (clip_val - mn) if clip_val > mn else 1.0

    # Steps 4+5: joint min-max across cameras (keeps intensities comparable),
    # then gamma < 1 pulls mid-range patches up into the visible colour range
    return [(np.clip((h - mn) / scale, 0.0, 1.0) ** GAMMA) for h in compressed]



# ============================================================
# PER-STEP PROCESSING
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    if FULL_ATTN_KEY not in rec:
        raise KeyError(f"Missing {FULL_ATTN_KEY} in {npy_path}")

    full_attn = np.asarray(rec[FULL_ATTN_KEY])
    if full_attn.ndim != 4:
        raise ValueError(f"{full_attn_key} expected 4D (L,B,H,S), got {full_attn.shape}")

    cam_spans = []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        span_key = f"outputs/debug/spans/image/{cam_name}"
        for k in (span_key, img_key):
            if k not in rec:
                raise KeyError(f"Missing {k} in {npy_path}")
        cam_spans.append(tuple(int(x) for x in np.asarray(rec[span_key]).tolist()))

    task_span = state_span = None
    if "outputs/debug/spans/task" in rec:
        task_span  = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/task"]).tolist())
    if "outputs/debug/spans/state" in rec:
        state_span = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/state"]).tolist())

    # Two-stage head selection: pick heads using HEAD_SELECTION_LAYER (early layer
    # where image/text head specialisation is stable across steps), then read the
    # actual attention values from TARGET_LAYER (late layer with sharper spatial
    # concentration). This gives stability in which heads we use combined with
    # spatial sharpness in the scores we read.
    selector_attn = full_attn[HEAD_SELECTION_LAYER, 0, :, :]  # (H, S)
    top_head_idx  = select_heads(selector_attn, cam_spans, task_span, state_span, TOP_K_HEADS)
    # take the element-wise max across the selected heads so each token position
    # gets the strongest signal from any of those heads
    attn_row      = full_attn[TARGET_LAYER, 0, :, :][top_head_idx, :].max(axis=0)  # (S,)

    print(f"  heads selected via layer {HEAD_SELECTION_LAYER}, applied to layer {TARGET_LAYER}: {sorted(int(h) for h in top_head_idx)}")

    Hp, Wp = IMAGE_PATCH_GRID
    heats, imgs = [], []
    for span, (cam_name, img_key) in zip(cam_spans, zip(CAM_NAMES, CAM_IMAGE_KEYS)):
        cam_scores = attn_row[span[0]:span[1]]
        if cam_scores.size != Hp * Wp:
            raise ValueError(f"{cam_name}: span length {cam_scores.size} != grid {Hp}x{Wp}={Hp*Wp}")
        heats.append(cam_scores.reshape(Hp, Wp))
        imgs.append(as_u8_rgb(rec[img_key]))

    if np.array_equal(imgs[0], imgs[1]):
        print(f"[WARN] Underlying images identical in {npy_path}. Keys: {CAM_IMAGE_KEYS}")

    heats01 = log_normalize_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]), smoothing=SMOOTHING)
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_token_scores  = attn_row[task_span[0]:task_span[1]]   if task_span  is not None else None
    state_token_scores = attn_row[state_span[0]:state_span[1]] if state_span is not None else None
    if task_token_scores is None or state_token_scores is None:
        raise KeyError("task/state spans missing from record — cannot build text panel")
    text_panel = render_text_panel_from_token_scores(rec, task_token_scores, state_token_scores, grid.height)

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
