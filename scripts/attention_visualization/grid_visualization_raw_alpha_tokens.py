#!/usr/bin/env python3
"""Raw Alpha attribution per decoded action token.

Layout (6 rows × 4 modality columns):
  Columns:  [label] | Base Camera | Wrist Camera | Task | State
  Row 0:   Original — raw images, blank token columns
  Row 1-5: Raw Alpha for action token 0, 1, 2, 3, 4
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
)
from .text_utils import get_token_labels, render_token_strip_as_image
from ..attention_utils.keys import CAM_NAMES, CAM_IMAGE_KEYS
from .grid_visualization_gradcam import to_2d_heatmap
from .grid_visualization_raw_alpha import clip_normalize_joint, CLIP_PERCENTILE, GAMMA

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260516_034939"
OUTPUT_DIR             = "/nfs/roberts/scratch/pi_tkf6/zs377/visualization_pi0_fast_libero_20260516_034939/heatmap_raw_alpha_tokens"
EPISODE_SUMMARIES_JSON = "/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260516_034939/client_output/episode_summaries.json"

MAX_TOTAL_EPISODES = 60   # set to None for unlimited

NUM_ACTION_TOKENS = 5
APPLY_ABS         = True
ALPHA             = 0.30
SMOOTHING         = "BILINEAR"
BG                = (255, 255, 255)
GAP_X             = 16
GAP_Y             = 16
LABEL_W           = 180
HEADER_H          = 42
FONT_SIZE         = 18
TOKEN_FONT_SIZE   = 38

IMG_DISPLAY  = 420
TASK_TILE_W  = 110
STATE_TILE_W = 42

ROW_LABELS = ["Original"] + [f"Token {i}" for i in range(NUM_ACTION_TOKENS)]
COL_LABELS = ["Base Camera", "Wrist Camera", "Task", "State"]


# ============================================================
# FONT
# ============================================================
def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

_FONT       = _load_font(FONT_SIZE)
_TOKEN_FONT = _load_font(TOKEN_FONT_SIZE)


def _draw_centered(draw: ImageDraw.ImageDraw, text: str,
                   x0: int, y0: int, w: int, h: int) -> None:
    bbox = draw.textbbox((0, 0), text, font=_FONT)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (w - tw) // 2, y0 + (h - th) // 2), text, fill=(30, 30, 30), font=_FONT)


def _clip_normalize_1d(scores: np.ndarray) -> np.ndarray:
    a        = np.abs(scores).astype(np.float32)
    clip_val = float(np.percentile(a, CLIP_PERCENTILE)) if a.size > 1 else float(a.max())
    scale    = clip_val if clip_val > 0 else 1.0
    return (np.clip(a / scale, 0.0, 1.0) ** GAMMA)


# ============================================================
# PER-STEP PROCESSING
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec  = load_record(npy_path)
    imgs = [
        np.array(Image.fromarray(as_u8_rgb(rec[k])).resize(
            (IMG_DISPLAY, IMG_DISPLAY), Image.BILINEAR))
        for k in CAM_IMAGE_KEYS
    ]
    H = W = row_h = IMG_DISPLAY

    task_labels,  task_idx  = get_token_labels(rec, "task",  strip_prefix="Task: ",  strip_suffix=", ")
    state_labels, state_idx = get_token_labels(rec, "state", strip_prefix="State: ", strip_suffix=";\n")
    task_col_w  = len(task_labels)  * TASK_TILE_W
    state_col_w = len(state_labels) * STATE_TILE_W

    image_heats_per_tok: List[List[np.ndarray]] = []
    task_strips:  List[Image.Image] = []
    state_strips: List[Image.Image] = []

    for tok_idx in range(NUM_ACTION_TOKENS):
        heats = []
        for cam in CAM_NAMES:
            h = to_2d_heatmap(
                np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{tok_idx}/{cam}"]), cam
            )
            heats.append(np.abs(h) if APPLY_ABS else h)
        image_heats_per_tok.append(clip_normalize_joint(heats))

        task_scores  = rec[f"outputs/debug/raw_alpha/summation/action_tokens/{tok_idx}/task"][0].astype(np.float32)
        state_scores = rec[f"outputs/debug/raw_alpha/summation/action_tokens/{tok_idx}/state"][0].astype(np.float32)

        # state_idx are absolute piece indices; adjust for digit-only arrays
        _n_state_pieces = len(rec["outputs/debug/tokens/state/piece_id"])
        _state_k = state_idx[0] if state_idx else 0
        if len(state_scores) < _n_state_pieces:
            _state_sel = state_scores[np.array(state_idx) - _state_k]
        else:
            _state_sel = state_scores[state_idx]

        task_strips.append(render_token_strip_as_image(
            _clip_normalize_1d(task_scores[task_idx]), task_labels, row_h, TASK_TILE_W, _TOKEN_FONT, bg=BG
        ))
        state_strips.append(render_token_strip_as_image(
            _clip_normalize_1d(_state_sel), state_labels, row_h, STATE_TILE_W, _TOKEN_FONT, bg=BG
        ))

    def _overlays(heats: List[np.ndarray]) -> List[np.ndarray]:
        return [
            overlay_heatmap(img, resize_heatmap(h01, (H, W), SMOOTHING), ALPHA)
            for img, h01 in zip(imgs, heats)
        ]

    cell_rows: List[List[np.ndarray]] = [list(imgs)] + [_overlays(h) for h in image_heats_per_tok]

    # Assemble canvas
    n_rows = len(ROW_LABELS)
    n_cams = len(CAM_IMAGE_KEYS)

    x_cam   = [GAP_X + LABEL_W + c * (GAP_X + W) + GAP_X for c in range(n_cams)]
    x_task  = GAP_X + LABEL_W + n_cams * (GAP_X + W) + GAP_X
    x_state = x_task + task_col_w + GAP_X

    canvas_w = x_state + state_col_w + GAP_X
    canvas_h = GAP_Y + HEADER_H + n_rows * (GAP_Y + row_h) + GAP_Y
    canvas   = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw     = ImageDraw.Draw(canvas)

    for lbl, x0, cw in zip(COL_LABELS, x_cam + [x_task, x_state], [W, W, task_col_w, state_col_w]):
        _draw_centered(draw, lbl, x0, GAP_Y, cw, HEADER_H)

    for r, (row_lbl, cells) in enumerate(zip(ROW_LABELS, cell_rows)):
        y0 = GAP_Y + HEADER_H + r * (GAP_Y + row_h) + GAP_Y
        _draw_centered(draw, row_lbl, GAP_X, y0, LABEL_W, row_h)

        for c, arr in enumerate(cells):
            canvas.paste(Image.fromarray(arr, "RGB"), (x_cam[c], y0))

        if r > 0:
            canvas.paste(task_strips[r - 1],  (x_task,  y0))
            canvas.paste(state_strips[r - 1], (x_state, y0))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


# ============================================================
# EPISODE SELECTION
# ============================================================
def _select_episodes(episodes: List[EpisodeInfo], max_total: int | None) -> set | None:
    """Return the set of allowed episode_nums, or None for unlimited.

    Strategy: aim for an even success/failure split.  Failures are ranked by
    proximity to the chosen successes (consecutive episodes run similar tasks).
    If either side runs out, the deficit is filled from the other side, again
    ranked by proximity to whatever has already been chosen.
    """
    if max_total is None:
        return None
    successes = [ep for ep in episodes if ep.success]
    failures  = [ep for ep in episodes if not ep.success]

    target_suc  = max_total // 2
    target_fail = max_total - target_suc

    # Even-split initial selection
    chosen_suc = successes[:target_suc]
    if chosen_suc:
        suc_nums    = [ep.episode_num for ep in chosen_suc]
        ranked_fail = sorted(failures, key=lambda ep: min(abs(ep.episode_num - s) for s in suc_nums))
    else:
        ranked_fail = list(failures)
    chosen_fail = ranked_fail[:target_fail]

    chosen = list(chosen_suc) + list(chosen_fail)

    # Fill any shortfall (whichever side ran out) by proximity to already chosen
    need = max_total - len(chosen)
    if need > 0:
        chosen_nums = {ep.episode_num for ep in chosen}
        pool        = [ep for ep in episodes if ep.episode_num not in chosen_nums]
        if chosen:
            ref  = [ep.episode_num for ep in chosen]
            pool = sorted(pool, key=lambda ep: min(abs(ep.episode_num - r) for r in ref))
        chosen += pool[:need]

    n_suc  = sum(1 for ep in chosen if ep.success)
    n_fail = len(chosen) - n_suc
    print(f"[info] selected {len(chosen)} episodes: {n_suc} success, {n_fail} failure "
          f"(out of {len(successes)} / {len(failures)} total).")
    return {ep.episode_num for ep in chosen}


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

    allowed_episode_nums = _select_episodes(episodes, MAX_TOTAL_EPISODES)

    skipped = 0
    for f in files:
        s   = step_index(f)
        ep  = episode_for_step(s, episodes)

        if allowed_episode_nums is not None and ep.episode_num not in allowed_episode_nums:
            skipped += 1
            continue

        out = str(
            out_dir
            / f"t{ep.task_id}_{slugify(ep.task)}"
            / f"ep{ep.episode_num}_{str(ep.success).lower()}"
            / f"step_{s:06d}.png"
        )
        if Path(out).exists():
            skipped += 1
            continue
        process_one(f, out)
        print(f"[OK] {out}")

    if skipped:
        print(f"[skip] {skipped} already-existing files skipped.")


if __name__ == "__main__":
    main()
