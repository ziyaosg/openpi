#!/usr/bin/env python3
"""Combined 4-method attribution heatmap per step.

Layout (5 data rows × 4 modality columns):
  Columns:  [label] | Base Camera | Wrist Camera | Task | State
  Row 0:   Original — raw images, blank token columns
  Row 1:   GradCAM
  Row 2:   Raw Alpha  (gradient summation, abs + percentile-clip + gamma)
  Row 3:   Raw Weights  (softmax attention, top-k image-focused heads)
  Row 4:   V-Cosine   (|cos(value, head output)|, top-k heads)

Head selection for Raw Weights and V-Cosine is done per modality (each span)
at TARGET_LAYER (layer 16).

Output: one PNG per step, e.g. heatmap_combined/t0_task/ep0_true/step_000000.png
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

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
from ..attention_utils.keys import CAM_NAMES, CAM_IMAGE_KEYS, IMAGE_PATCH_GRID
from .grid_visualization_gradcam import to_2d_heatmap, normalize_joint
from .grid_visualization_raw_alpha import clip_normalize_joint, CLIP_PERCENTILE, GAMMA
from .grid_visualization_raw_weights import (
    select_heads_for_span,
    log_normalize_joint,
    TARGET_LAYER,
    TOP_K_HEADS,
)
from .grid_visualization_value_attn import score_v_cosine

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260511_040629"
OUTPUT_DIR             = "/nfs/roberts/scratch/pi_tkf6/zs377/visualization_pi0_fast_libero_20260511_040629/heatmap_combined"
EPISODE_SUMMARIES_JSON = "/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi0_fast_libero_20260511_040629/client_output/episode_summaries.json"

MAX_EPISODES_PER_TASK = 6   # set to None for unlimited

# Which decoded action token to visualize (0-indexed).
ACTION_TOKEN_IDX = 0

ALPHA     = 0.30
SMOOTHING = "BILINEAR"
BG        = (255, 255, 255)
GAP_X     = 16
GAP_Y     = 16
LABEL_W   = 180
HEADER_H  = 42
FONT_SIZE       = 18
TOKEN_FONT_SIZE = 38

IMG_DISPLAY = 420

TASK_TILE_W  = 110
STATE_TILE_W = 42

APPLY_RELU_GRADCAM  = True
APPLY_ABS_RAW_ALPHA = True

ROW_LABELS = ["Original", "GradCAM", "Raw Alpha", "Raw Weights", "V-Cosine"]
COL_LABELS = ["Base Camera", "Wrist Camera", "Task", "State"]

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"
LAYERS_KEY    = "outputs/debug/attn/layers"


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


# ============================================================
# SPAN HELPERS
# ============================================================
def _cam_spans(rec: dict) -> List[Tuple[int, int]]:
    return [
        tuple(int(x) for x in np.asarray(rec[f"outputs/debug/spans/image/{cam}"]).tolist())
        for cam in CAM_NAMES
    ]


def _text_spans(rec: dict) -> Tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
    task = state = None
    if "outputs/debug/spans/task" in rec:
        task  = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/task"]).tolist())
    if "outputs/debug/spans/state" in rec:
        state = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/state"]).tolist())
    return task, state


def _layer_index(rec: dict) -> int:
    stored = np.asarray(rec[LAYERS_KEY]).tolist()
    return stored.index(TARGET_LAYER)


def _clip_normalize_1d(scores: np.ndarray) -> np.ndarray:
    a        = np.abs(scores).astype(np.float32)
    clip_val = float(np.percentile(a, CLIP_PERCENTILE)) if a.size > 1 else float(a.max())
    scale    = clip_val if clip_val > 0 else 1.0
    return (np.clip(a / scale, 0.0, 1.0) ** GAMMA)


# ============================================================
# GRADIENT-BASED PATCH HEATMAPS  (indexed by ACTION_TOKEN_IDX)
# ============================================================
def _heatmaps_gradcam(rec: dict) -> List[np.ndarray]:
    heats = []
    for cam in CAM_NAMES:
        h = to_2d_heatmap(rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN_IDX}/{cam}"], cam)
        heats.append(np.maximum(h, 0.0) if APPLY_RELU_GRADCAM else h)
    return normalize_joint(heats)


def _heatmaps_raw_alpha(rec: dict) -> List[np.ndarray]:
    heats = []
    for cam in CAM_NAMES:
        h = to_2d_heatmap(
            np.asarray(rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN_IDX}/{cam}"]), cam
        )
        heats.append(np.abs(h) if APPLY_ABS_RAW_ALPHA else h)
    return clip_normalize_joint(heats)


# ============================================================
# PER-STEP COMBINED IMAGE
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec  = load_record(npy_path)
    imgs = [
        np.array(Image.fromarray(as_u8_rgb(rec[k])).resize(
            (IMG_DISPLAY, IMG_DISPLAY), Image.BILINEAR))
        for k in CAM_IMAGE_KEYS
    ]
    H = W = row_h = IMG_DISPLAY
    Hp, Wp = IMAGE_PATCH_GRID

    # Token labels — content only, prefix/suffix stripped
    task_labels,  task_idx  = get_token_labels(rec, "task",  strip_prefix="Task: ",  strip_suffix=", ")
    state_labels, state_idx = get_token_labels(rec, "state", strip_prefix="State: ", strip_suffix=";\n")
    task_col_w  = len(task_labels)  * TASK_TILE_W
    state_col_w = len(state_labels) * STATE_TILE_W

    # Shared attention precomputation
    cam_spans         = _cam_spans(rec)
    task_sp, state_sp = _text_spans(rec)
    tgt_idx           = _layer_index(rec)
    full_attn = np.asarray(rec[FULL_ATTN_KEY])   # (L, B, H, S)
    full_v    = np.asarray(rec[FULL_V_KEY])       # (L, B, S, K, D)
    attn      = full_attn[tgt_idx, 0]             # (H, S)
    v         = full_v[tgt_idx, 0]                # (S, K, D)

    # Per-modality one-vs-rest head selection at TARGET_LAYER
    all_spans: List[Tuple[int, int]] = list(cam_spans)
    if task_sp:  all_spans.append(task_sp)
    if state_sp: all_spans.append(state_sp)

    cam_heads   = [select_heads_for_span(attn, sp, all_spans, TOP_K_HEADS) for sp in cam_spans]
    task_heads  = select_heads_for_span(attn, task_sp,  all_spans, TOP_K_HEADS) if task_sp  else None
    state_heads = select_heads_for_span(attn, state_sp, all_spans, TOP_K_HEADS) if state_sp else None

    # Raw-weights and v-cosine scores per modality
    rw_cams   = [attn[h, s0:s1].max(axis=0).reshape(Hp, Wp) for h, (s0, s1) in zip(cam_heads, cam_spans)]
    vcos_cams = [score_v_cosine(attn, v, h)[s0:s1].reshape(Hp, Wp) for h, (s0, s1) in zip(cam_heads, cam_spans)]

    _z = np.zeros(0, np.float32)
    rw_task    = attn[task_heads,  task_sp[0]:task_sp[1]].max(axis=0)            if task_sp  and task_heads  is not None else _z
    rw_state   = attn[state_heads, state_sp[0]:state_sp[1]].max(axis=0)          if state_sp and state_heads is not None else _z
    vcos_task  = score_v_cosine(attn, v, task_heads)[task_sp[0]:task_sp[1]]      if task_sp  and task_heads  is not None else _z
    vcos_state = score_v_cosine(attn, v, state_heads)[state_sp[0]:state_sp[1]]   if state_sp and state_heads is not None else _z

    # Per-method image heatmaps
    image_heats: List[List[np.ndarray]] = [
        _heatmaps_gradcam(rec),
        _heatmaps_raw_alpha(rec),
        log_normalize_joint(rw_cams),
        normalize_joint(vcos_cams),
    ]

    # Per-method token scores (content tokens only)
    task_scores_per_method: List[np.ndarray] = [
        ts[task_idx] for ts in [
            rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN_IDX}/task"][0].astype(np.float32),
            rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN_IDX}/task"][0].astype(np.float32),
            rw_task,
            vcos_task,
        ]
    ]
    state_scores_per_method: List[np.ndarray] = [
        ss[state_idx] for ss in [
            rec[f"outputs/debug/gradcam/action_tokens/{ACTION_TOKEN_IDX}/state"][0].astype(np.float32),
            rec[f"outputs/debug/raw_alpha/summation/action_tokens/{ACTION_TOKEN_IDX}/state"][0].astype(np.float32),
            rw_state,
            vcos_state,
        ]
    ]

    task_strips = [
        render_token_strip_as_image(_clip_normalize_1d(ts), task_labels,  row_h, TASK_TILE_W,  _TOKEN_FONT, bg=BG)
        for ts in task_scores_per_method
    ]
    state_strips = [
        render_token_strip_as_image(_clip_normalize_1d(ss), state_labels, row_h, STATE_TILE_W, _TOKEN_FONT, bg=BG)
        for ss in state_scores_per_method
    ]

    def _overlays(heats: List[np.ndarray]) -> List[np.ndarray]:
        return [
            overlay_heatmap(img, resize_heatmap(h01, (H, W), SMOOTHING), ALPHA)
            for img, h01 in zip(imgs, heats)
        ]

    cell_rows: List[List[np.ndarray]] = [list(imgs)] + [_overlays(h) for h in image_heats]

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

    col_labels = [f"Base Cam (tok {ACTION_TOKEN_IDX})", f"Wrist Cam (tok {ACTION_TOKEN_IDX})", "Task", "State"]
    for lbl, x0, cw in zip(col_labels, x_cam + [x_task, x_state], [W, W, task_col_w, state_col_w]):
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

    # Pre-scan existing episode dirs so subsequent runs pick up from where they left off.
    # Keys: task_id → set of episode_nums that already have (or are being) visualized.
    allowed_episodes: dict = defaultdict(set)
    if out_dir.exists():
        for task_dir in out_dir.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith("t"):
                continue
            try:
                task_id = int(task_dir.name.split("_")[0][1:])
            except (ValueError, IndexError):
                continue
            for ep_dir in task_dir.iterdir():
                if ep_dir.is_dir() and ep_dir.name.startswith("ep"):
                    try:
                        ep_num = int(ep_dir.name.split("_")[0][2:])
                        allowed_episodes[task_id].add(ep_num)
                    except (ValueError, IndexError):
                        pass

    skipped = 0
    for f in files:
        s   = step_index(f)
        ep  = episode_for_step(s, episodes)

        if MAX_EPISODES_PER_TASK is not None:
            if ep.episode_num not in allowed_episodes[ep.task_id]:
                if len(allowed_episodes[ep.task_id]) >= MAX_EPISODES_PER_TASK:
                    skipped += 1
                    continue
                allowed_episodes[ep.task_id].add(ep.episode_num)

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
