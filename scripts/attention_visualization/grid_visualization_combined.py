#!/usr/bin/env python3
"""Combined 5-method attribution heatmap per step.

Layout (6 data rows × 4 modality columns):
  Columns:  [label] | Base Camera | Wrist Camera | Task | State
  Row 0:   Original — raw images, blank token columns
  Row 1:   GradCAM
  Row 2:   Raw Alpha  (gradient summation, abs + percentile-clip + gamma)
  Row 3:   Raw Weights  (softmax attention, top-k image-focused heads)
  Row 4:   V-Norm  (attn × ‖value‖, top-k heads)
  Row 5:   V-Cosine  (|cos(value, head output)|, top-k heads)

Camera columns are W×W image cells (vertically centred in each row).
Task and State columns show tokens left-to-right as coloured tiles; their
width is N_tokens × TILE_W, always wider than the image columns.
"""
from __future__ import annotations

import glob
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
    select_heads,
    log_normalize_joint,
    TARGET_LAYER,
    HEAD_SELECTION_LAYER,
    TOP_K_HEADS,
)
from .grid_visualization_value_attn import score_v_norm, score_v_cosine

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260421_133206"
OUTPUT_DIR             = "/home/ziyao/Documents/policy_records_20260421_133206/heatmap_combined"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260421_133206/episode_summaries.json"

ALPHA     = 0.30
SMOOTHING = "BILINEAR"
BG        = (255, 255, 255)
GAP_X     = 12
GAP_Y     = 12
LABEL_W   = 130
HEADER_H  = 30
FONT_SIZE       = 14   # row labels and column headers
TOKEN_FONT_SIZE = 28   # token strip labels (rotated inside each tile)

# Images are upscaled to this size for display; token strips use the same height
# so every cell in a row is exactly IMG_DISPLAY × IMG_DISPLAY (or × col_w for tokens).
IMG_DISPLAY = 300

# Pixel width allocated per token tile in the horizontal strips.
# Column width = N_content_tokens × TILE_W, always wider than the image columns.
TASK_TILE_W  = 80   # task tokens are English subwords (3–8 chars)
STATE_TILE_W = 56   # state tokens are single-char numerals / spaces

APPLY_RELU_GRADCAM  = True
APPLY_ABS_RAW_ALPHA = True

ROW_LABELS = ["Original", "GradCAM", "Raw Alpha", "Raw Weights", "V-Norm", "V-Cosine"]
COL_LABELS = ["Base Camera", "Wrist Camera", "Task", "State"]

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"
LAYERS_KEY    = "outputs/debug/attn/layers"


# ============================================================
# FONT  (row labels and column headers)
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


def _layer_indices(rec: dict) -> Tuple[int, int]:
    stored = np.asarray(rec[LAYERS_KEY]).tolist()
    return stored.index(HEAD_SELECTION_LAYER), stored.index(TARGET_LAYER)


# ============================================================
# 1-D TOKEN SCORE NORMALIZATION
# Same CLIP_PERCENTILE and GAMMA as clip_normalize_joint so image patches
# and token strips share an identical perceptual scale.
# ============================================================
def _clip_normalize_1d(scores: np.ndarray) -> np.ndarray:
    a        = np.abs(scores).astype(np.float32)
    clip_val = float(np.percentile(a, CLIP_PERCENTILE)) if a.size > 1 else float(a.max())
    scale    = clip_val if clip_val > 0 else 1.0
    return (np.clip(a / scale, 0.0, 1.0) ** GAMMA)


# ============================================================
# GRADIENT-BASED PATCH HEATMAPS
# ============================================================
def _heatmaps_gradcam(rec: dict) -> List[np.ndarray]:
    heats = []
    for cam in CAM_NAMES:
        h = to_2d_heatmap(rec[f"outputs/debug/gradcam/image/{cam}"], cam)
        heats.append(np.maximum(h, 0.0) if APPLY_RELU_GRADCAM else h)
    return normalize_joint(heats)


def _heatmaps_raw_alpha(rec: dict) -> List[np.ndarray]:
    heats = []
    for cam in CAM_NAMES:
        h = to_2d_heatmap(np.asarray(rec[f"outputs/debug/raw_alpha/summation/image/{cam}"]), cam)
        heats.append(np.abs(h) if APPLY_ABS_RAW_ALPHA else h)
    return clip_normalize_joint(heats)


# ============================================================
# PER-STEP COMBINED IMAGE
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec  = load_record(npy_path)
    # Upscale camera images to IMG_DISPLAY for display; token strips use the same height.
    imgs = [
        np.array(Image.fromarray(as_u8_rgb(rec[k])).resize(
            (IMG_DISPLAY, IMG_DISPLAY), Image.BILINEAR))
        for k in CAM_IMAGE_KEYS
    ]
    H = W = row_h = IMG_DISPLAY
    Hp, Wp = IMAGE_PATCH_GRID

    # ── Token labels — content only, prefix/suffix stripped ──────────────────
    task_labels,  task_idx  = get_token_labels(rec, "task",  strip_prefix="Task: ",  strip_suffix=", ")
    state_labels, state_idx = get_token_labels(rec, "state", strip_prefix="State: ", strip_suffix=";\n")
    task_col_w  = len(task_labels)  * TASK_TILE_W
    state_col_w = len(state_labels) * STATE_TILE_W

    # ── Shared attention precomputation ──────────────────────────────────────
    cam_spans         = _cam_spans(rec)
    task_sp, state_sp = _text_spans(rec)
    sel_idx, tgt_idx  = _layer_indices(rec)
    full_attn = np.asarray(rec[FULL_ATTN_KEY])   # (L, B, H, S)
    full_v    = np.asarray(rec[FULL_V_KEY])       # (L, B, S, K, D)  K=1 for GQA
    top_heads = select_heads(full_attn[sel_idx, 0], cam_spans, task_sp, state_sp, TOP_K_HEADS)
    attn      = full_attn[tgt_idx, 0]             # (H, S)
    v         = full_v[tgt_idx, 0]               # (S, K, D)

    raw_w_full   = attn[top_heads].max(axis=0)         # (S,)
    vnorm_full   = score_v_norm(attn, v, top_heads)    # (S,)
    vcosine_full = score_v_cosine(attn, v, top_heads)  # (S,)

    def _patch(s: np.ndarray) -> List[np.ndarray]:
        return [s[s0:s1].reshape(Hp, Wp) for s0, s1 in cam_spans]

    def _task_slice(s: np.ndarray) -> np.ndarray:
        return s[task_sp[0]:task_sp[1]].astype(np.float32) if task_sp else np.zeros(0, np.float32)

    def _state_slice(s: np.ndarray) -> np.ndarray:
        return s[state_sp[0]:state_sp[1]].astype(np.float32) if state_sp else np.zeros(0, np.float32)

    # ── Per-method image heatmaps ─────────────────────────────────────────────
    image_heats: List[List[np.ndarray]] = [
        _heatmaps_gradcam(rec),
        _heatmaps_raw_alpha(rec),
        log_normalize_joint(_patch(raw_w_full)),
        log_normalize_joint(_patch(vnorm_full)),
        normalize_joint(_patch(vcosine_full)),
    ]

    # ── Per-method token scores (content tokens only, indexed by task/state_idx) ─
    task_scores_per_method: List[np.ndarray] = [
        ts[task_idx] for ts in [
            rec["outputs/debug/gradcam/task"][0].astype(np.float32),
            rec["outputs/debug/raw_alpha/summation/task"][0].astype(np.float32),
            _task_slice(raw_w_full),
            _task_slice(vnorm_full),
            _task_slice(vcosine_full),
        ]
    ]
    state_scores_per_method: List[np.ndarray] = [
        ss[state_idx] for ss in [
            rec["outputs/debug/gradcam/state"][0].astype(np.float32),
            rec["outputs/debug/raw_alpha/summation/state"][0].astype(np.float32),
            _state_slice(raw_w_full),
            _state_slice(vnorm_full),
            _state_slice(vcosine_full),
        ]
    ]

    # ── Token strip images ────────────────────────────────────────────────────
    task_strips = [
        render_token_strip_as_image(_clip_normalize_1d(ts), task_labels,  row_h, TASK_TILE_W,  _TOKEN_FONT, bg=BG)
        for ts in task_scores_per_method
    ]
    state_strips = [
        render_token_strip_as_image(_clip_normalize_1d(ss), state_labels, row_h, STATE_TILE_W, _TOKEN_FONT, bg=BG)
        for ss in state_scores_per_method
    ]

    # ── Overlay image rows ────────────────────────────────────────────────────
    def _overlays(heats: List[np.ndarray]) -> List[np.ndarray]:
        return [
            overlay_heatmap(img, resize_heatmap(h01, (H, W), SMOOTHING), ALPHA)
            for img, h01 in zip(imgs, heats)
        ]

    cell_rows: List[List[np.ndarray]] = [list(imgs)] + [_overlays(h) for h in image_heats]

    # ── Assemble canvas ───────────────────────────────────────────────────────
    n_rows = len(ROW_LABELS)
    n_cams = len(CAM_IMAGE_KEYS)

    x_cam   = [GAP_X + LABEL_W + c * (GAP_X + W) + GAP_X for c in range(n_cams)]
    x_task  = GAP_X + LABEL_W + n_cams * (GAP_X + W) + GAP_X
    x_state = x_task + task_col_w + GAP_X

    canvas_w = x_state + state_col_w + GAP_X
    canvas_h = GAP_Y + HEADER_H + n_rows * (GAP_Y + row_h) + GAP_Y
    canvas   = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw     = ImageDraw.Draw(canvas)

    # Column headers
    for lbl, x0, cw in zip(COL_LABELS, x_cam + [x_task, x_state], [W, W, task_col_w, state_col_w]):
        _draw_centered(draw, lbl, x0, GAP_Y, cw, HEADER_H)

    # Data rows
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

    for f in files:
        s   = step_index(f)
        ep  = episode_for_step(s, episodes)
        out = str(
            out_dir
            / f"t{ep.task_id}_{slugify(ep.task)}"
            / f"ep{ep.episode_num}_{str(ep.success).lower()}"
            / f"step_{s:06d}.png"
        )
        process_one(f, out)
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
