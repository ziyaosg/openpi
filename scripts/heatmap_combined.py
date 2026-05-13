#!/usr/bin/env python3
"""Combined 4-method attribution heatmap per step.

Layout (5 data rows × 3 columns):
  Columns: [label] | Base Camera | Left Wrist | Task Tokens
  Row 0:  Original — raw images, blank task panel
  Row 1:  GradCAM
  Row 2:  Raw Alpha
  Row 3:  Raw Weights
  Row 4:  V-Cosine
"""
from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import (
    CAM_NAMES, CAM_IMAGE_KEYS, PATCH_H, PATCH_W,
    TOP_K_HEADS,
    load_episode_infos, episode_for_step, load_record, step_index, slugify,
    as_u8_rgb, resize_heatmap, overlay_heatmap,
    normalize_joint_minmax, clip_gamma_normalize_joint, log_clip_gamma_normalize_joint,
    get_attn_hs, get_v_skd, get_span, get_all_spans, select_heads,
    score_v_cosine, get_token_labels, render_token_strip_as_image,
)

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR    = Path("/data/ziyao/policy_records_pi05_libero_20260512_182606")
OUTPUT_DIR   = Path("/data/ziyao/visualization_pi05_libero_20260512_182606/heatmap_combined")
EPISODE_JSON = INPUT_DIR / "client_output" / "episode_summaries.json"

MAX_EPISODES_PER_TASK = 10   # set to None for unlimited

ACTION_TOKEN    = 0
VARIANT         = "summation"
APPLY_RELU      = True
APPLY_ABS       = True
CLIP_PERCENTILE = 95
GAMMA           = 0.4

ALPHA       = 0.30
BG          = (255, 255, 255)
GAP_X       = 16
GAP_Y       = 16
LABEL_W     = 180
HEADER_H    = 42
FONT_SIZE   = 18
IMG_DISPLAY = 420

SMOOTHING = "BILINEAR"

TASK_TILE_W     = 110
STATE_TILE_W    = 42
TOKEN_FONT_SIZE = 38

ROW_LABELS = ["Original", "GradCAM", "Raw Alpha", "Raw Weights", "V-Cosine"]
COL_LABELS = ["Base Camera", "Left Wrist", "Task Tokens", "State Tokens"]


# ─── Font ─────────────────────────────────────────────────────────────────────

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



# ─── Normalization ────────────────────────────────────────────────────────────

def _clip_normalize_1d(scores: np.ndarray) -> np.ndarray:
    a = np.abs(scores).astype(np.float32)
    if a.size == 0:
        return a
    clip_val = float(np.percentile(a, CLIP_PERCENTILE)) if a.size > 1 else float(a.max())
    scale    = clip_val if clip_val > 0 else 1.0
    return (np.clip(a / scale, 0.0, 1.0) ** GAMMA)


def _normalize_cosine_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    flat = np.concatenate([h.reshape(-1) for h in heats])
    mn, mx = float(flat.min()), float(flat.max())
    scale = (mx - mn) if mx > mn else 1.0
    return [(h.astype(np.float32) - mn) / scale for h in heats]


# ─── Per-step processing ──────────────────────────────────────────────────────

def process_one(npy_path: str, out_png: str) -> None:
    rec   = load_record(npy_path)
    row_h = IMG_DISPLAY

    # Token labels and content indices for task and state modalities
    task_labels,  task_idx  = get_token_labels(rec, "task",  strip_prefix="Task: ",  strip_suffix=", ")
    state_labels, state_idx = get_token_labels(rec, "state", strip_prefix="State: ", strip_suffix=";\n")

    # Camera images upscaled to IMG_DISPLAY × IMG_DISPLAY (base + left wrist only)
    imgs = [
        np.array(Image.fromarray(as_u8_rgb(rec[k])).resize(
            (IMG_DISPLAY, IMG_DISPLAY), Image.BILINEAR))
        for k in CAM_IMAGE_KEYS[:2]
    ]

    # Shared attention data
    attn      = get_attn_hs(rec, action_token=ACTION_TOKEN)  # (H, S)
    v         = get_v_skd(rec)                                # (S, K, D)
    all_spans = get_all_spans(rec)
    cam_spans = [get_span(rec, f"outputs/debug/spans/image/{cn}") for cn in CAM_NAMES[:2]]
    cam_heads = [select_heads(attn, sp, all_spans, TOP_K_HEADS) for sp in cam_spans]

    # ── Image heatmaps per method ─────────────────────────────────────────────

    gc_raw = [np.maximum(np.asarray(rec[f"outputs/debug/gradcam/image/{cn}"], dtype=np.float32), 0.0)
              if APPLY_RELU else np.asarray(rec[f"outputs/debug/gradcam/image/{cn}"], dtype=np.float32)
              for cn in CAM_NAMES[:2]]
    gc_heats01 = normalize_joint_minmax(gc_raw)

    ra_raw = [np.abs(np.asarray(rec[f"outputs/debug/raw_alpha/{VARIANT}/image/{cn}"], dtype=np.float32))
              if APPLY_ABS else np.asarray(rec[f"outputs/debug/raw_alpha/{VARIANT}/image/{cn}"], dtype=np.float32)
              for cn in CAM_NAMES[:2]]
    ra_heats01 = clip_gamma_normalize_joint(ra_raw, CLIP_PERCENTILE, GAMMA)

    rw_raw = [attn[heads, sp[0]:sp[1]].max(axis=0).reshape(PATCH_H, PATCH_W)
              for heads, sp in zip(cam_heads, cam_spans)]
    rw_heats01 = log_clip_gamma_normalize_joint(rw_raw, CLIP_PERCENTILE, GAMMA)

    vc_raw = [score_v_cosine(attn, v, heads)[sp[0]:sp[1]].reshape(PATCH_H, PATCH_W)
              for heads, sp in zip(cam_heads, cam_spans)]
    vc_heats01 = _normalize_cosine_joint(vc_raw)

    heats_per_method = [gc_heats01, ra_heats01, rw_heats01, vc_heats01]

    # ── Task scores per method ────────────────────────────────────────────────

    task_span = get_span(rec, "outputs/debug/spans/task")
    _z200     = np.zeros(200, dtype=np.float32)

    def _rw_task() -> np.ndarray:
        if task_span is None:
            return _z200
        th  = select_heads(attn, task_span, all_spans, TOP_K_HEADS)
        raw = attn[th, task_span[0]:task_span[1]].max(axis=0)
        out = np.zeros(200, dtype=np.float32)
        n   = min(raw.size, int(np.asarray(rec["outputs/debug/tokens/task/token_mask"]).sum()))
        out[:n] = raw[:n]
        return out

    def _vc_task() -> np.ndarray:
        if task_span is None:
            return _z200
        th  = select_heads(attn, task_span, all_spans, TOP_K_HEADS)
        raw = score_v_cosine(attn, v, th)[task_span[0]:task_span[1]]
        out = np.zeros(200, dtype=np.float32)
        n   = min(raw.size, int(np.asarray(rec["outputs/debug/tokens/task/token_mask"]).sum()))
        out[:n] = raw[:n]
        return out

    _n_task = int(np.asarray(rec["outputs/debug/tokens/task/token_mask"]).sum())

    def _pad_task(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(200, dtype=np.float32)
        n = min(arr.size, _n_task)
        out[:n] = arr[:n]
        return out

    task_scores_per_method = [
        _pad_task(np.asarray(rec["outputs/debug/gradcam/task"], dtype=np.float32)),
        _pad_task(np.asarray(rec[f"outputs/debug/raw_alpha/{VARIANT}/task"], dtype=np.float32)),
        _rw_task(),
        _vc_task(),
    ]

    # ── State scores per method ───────────────────────────────────────────────
    state_span = get_span(rec, "outputs/debug/spans/state")
    if state_span:
        state_heads = select_heads(attn, state_span, all_spans, TOP_K_HEADS)
        gc_state = np.abs(np.asarray(rec["outputs/debug/gradcam/state"], dtype=np.float32))
        ra_state = np.abs(np.asarray(rec[f"outputs/debug/raw_alpha/{VARIANT}/state"], dtype=np.float32))
        rw_state = attn[state_heads, state_span[0]:state_span[1]].max(axis=0)
        vc_state = score_v_cosine(attn, v, state_heads)[state_span[0]:state_span[1]]
        state_scores_per_method = [gc_state, ra_state, rw_state, vc_state]
    else:
        state_scores_per_method = [np.zeros(1, dtype=np.float32)] * 4

    # ── Task token strips (rows 1–4) ─────────────────────────────────────────
    task_n = min(len(ts) for ts in task_scores_per_method)
    _valid_task = task_idx < task_n
    task_idx_v    = task_idx[_valid_task]
    task_labels_v = [task_labels[j] for j, ok in enumerate(_valid_task) if ok]
    task_col_w    = len(task_labels_v) * TASK_TILE_W
    task_strips = [
        render_token_strip_as_image(
            _clip_normalize_1d(ts[task_idx_v]), task_labels_v, row_h, TASK_TILE_W, _TOKEN_FONT, bg=BG
        )
        for ts in task_scores_per_method
    ]

    # ── State token strips (rows 1–4) ────────────────────────────────────────
    state_n = min(len(ss) for ss in state_scores_per_method)
    _valid_state = state_idx < state_n
    state_idx_v    = state_idx[_valid_state]
    state_labels_v = [state_labels[j] for j, ok in enumerate(_valid_state) if ok]
    state_col_w    = len(state_labels_v) * STATE_TILE_W
    state_strips = [
        render_token_strip_as_image(
            _clip_normalize_1d(ss[state_idx_v]), state_labels_v, row_h, STATE_TILE_W, _TOKEN_FONT, bg=BG
        )
        for ss in state_scores_per_method
    ]

    # ── Overlay images per method ─────────────────────────────────────────────
    def _overlays(heats01: List[np.ndarray]) -> List[np.ndarray]:
        return [
            overlay_heatmap(img, resize_heatmap(h, (IMG_DISPLAY, IMG_DISPLAY), SMOOTHING), ALPHA)
            for img, h in zip(imgs, heats01)
        ]

    cell_rows: List[List[np.ndarray]] = [list(imgs)] + [_overlays(h) for h in heats_per_method]

    # ── Canvas layout ─────────────────────────────────────────────────────────
    n_rows = len(ROW_LABELS)
    n_cams = 2  # base + left wrist only; right wrist discarded

    x_cams  = [GAP_X + LABEL_W + GAP_X + c * (IMG_DISPLAY + GAP_X) for c in range(n_cams)]
    x_task  = x_cams[-1] + IMG_DISPLAY + GAP_X
    x_state = x_task + task_col_w + GAP_X

    canvas_w = x_state + state_col_w + GAP_X
    canvas_h = GAP_Y + HEADER_H + n_rows * (GAP_Y + row_h) + GAP_Y
    canvas   = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw     = ImageDraw.Draw(canvas)

    # Column headers
    for lbl, x0, cw in zip(COL_LABELS, x_cams + [x_task, x_state],
                            [IMG_DISPLAY] * n_cams + [task_col_w, state_col_w]):
        _draw_centered(draw, lbl, x0, GAP_Y, cw, HEADER_H)

    # Data rows
    for r, (row_lbl, cells) in enumerate(zip(ROW_LABELS, cell_rows)):
        y0 = GAP_Y + HEADER_H + r * (GAP_Y + row_h) + GAP_Y
        _draw_centered(draw, row_lbl, GAP_X, y0, LABEL_W, row_h)
        for c, arr in enumerate(cells):
            canvas.paste(Image.fromarray(arr, "RGB"), (x_cams[c], y0))
        if r > 0:
            canvas.paste(task_strips[r - 1],  (x_task,  y0))
            canvas.paste(state_strips[r - 1], (x_state, y0))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_infos(EPISODE_JSON)
    files    = sorted(glob.glob(str(INPUT_DIR / "step_*.npy")), key=step_index)
    if not files:
        raise FileNotFoundError(f"No step_*.npy in {INPUT_DIR}")

    # Pre-scan existing episode dirs so subsequent runs pick up from where they left off.
    allowed_episodes: dict = defaultdict(set)
    if OUTPUT_DIR.exists():
        for task_dir in OUTPUT_DIR.iterdir():
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
        s  = step_index(f)
        ep = episode_for_step(s, episodes)

        if MAX_EPISODES_PER_TASK is not None:
            if ep.episode_num not in allowed_episodes[ep.task_id]:
                if len(allowed_episodes[ep.task_id]) >= MAX_EPISODES_PER_TASK:
                    skipped += 1
                    continue
                allowed_episodes[ep.task_id].add(ep.episode_num)

        out = str(
            OUTPUT_DIR
            / f"t{ep.task_id}_{slugify(ep.task)}"
            / f"ep{ep.episode_num}_{str(ep.success).lower()}"
            / f"step_{s:06d}.png"
        )
        if Path(out).exists():
            skipped += 1
            continue
        try:
            process_one(f, out)
            print(f"[OK] {out}")
        except Exception as e:
            print(f"[SKIP] step {s}: {e}")

    if skipped:
        print(f"[skip] {skipped} already-existing files skipped.")


if __name__ == "__main__":
    main()
