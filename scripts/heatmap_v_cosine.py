#!/usr/bin/env python3
"""V-cosine heatmaps for pi05/libero policy records.

Score: cos(v[s,0,:], head_output_h), max over top-k heads per modality.
High scores mean the value at position s points in the same direction as
what the selected head actually outputs — a signal of functional alignment.

Output layout per step:
  top row:    v-cosine overlay on each camera image
  bottom row: original images
  right panel: task-token v-cosine scores

Run:
  python3 scripts/pi05_vis/heatmap_v_cosine.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    CAM_NAMES, CAM_IMAGE_KEYS, PATCH_H, PATCH_W,
    FULL_ATTN_KEY, FULL_V_KEY, LAYER_IDX, TOP_K_HEADS,
    load_episode_infos, episode_for_step, load_record, step_index, slugify,
    as_u8_rgb, resize_heatmap, overlay_heatmap, make_grid,
    get_attn_hs, get_v_skd, get_span, get_all_spans, select_heads,
    score_v_cosine, make_task_text_panel,
)

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR  = Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260511_021509")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/zs377/visualization_policy_records_pi05_libero_20260511_021509/heatmap_v_cosine")
EPISODE_JSON = INPUT_DIR / "client_output" / "episode_summaries.json"

# Which action token to visualize (0–9)
ACTION_TOKEN = 0

ALPHA = 0.30
BG    = (255, 255, 255)
GAP_X = 16
GAP_Y = 16


def _normalize_cosine_joint(heats):
    """Joint min-max for cosine scores in [-1, 1] → [0, 1]."""
    import numpy as np
    flat = np.concatenate([h.reshape(-1) for h in heats])
    mn, mx = float(flat.min()), float(flat.max())
    scale = (mx - mn) if mx > mn else 1.0
    return [(h.astype(np.float32) - mn) / scale for h in heats]


# ─── Per-step processing ──────────────────────────────────────────────────────

def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    attn = get_attn_hs(rec, action_token=ACTION_TOKEN)  # (H, S)
    v    = get_v_skd(rec)                                # (S, K, D)
    all_spans = get_all_spans(rec)

    heats, imgs = [], []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        span = get_span(rec, f"outputs/debug/spans/image/{cam_name}")
        if span is None:
            raise KeyError(f"Missing span for {cam_name}")
        heads      = select_heads(attn, span, all_spans, TOP_K_HEADS)
        cam_scores = score_v_cosine(attn, v, heads)[span[0]:span[1]]  # (N_patches,)
        if cam_scores.size != PATCH_H * PATCH_W:
            raise ValueError(f"{cam_name}: span {cam_scores.size} != {PATCH_H}×{PATCH_W}")
        heats.append(cam_scores.reshape(PATCH_H, PATCH_W))
        imgs.append(as_u8_rgb(rec[img_key]))

    heats01 = _normalize_cosine_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]))
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_span = get_span(rec, "outputs/debug/spans/task")
    if task_span is not None:
        task_heads  = select_heads(attn, task_span, all_spans, TOP_K_HEADS)
        task_scores_full = score_v_cosine(attn, v, task_heads)  # (S,)
        task_token_scores = task_scores_full[task_span[0]:task_span[1]]
        full_task = np.zeros(200, dtype=np.float32)
        tok_mask  = np.asarray(rec["outputs/debug/tokens/task/token_mask"])
        real_count = int(tok_mask.sum())
        n_fill = min(task_token_scores.size, real_count)
        full_task[:n_fill] = task_token_scores[:n_fill]
        text_panel = make_task_text_panel(rec, full_task, grid.height)
    else:
        text_panel = None

    if text_panel is not None:
        combined = Image.new("RGB", (grid.width + GAP_X + text_panel.width, grid.height), BG)
        combined.paste(grid, (0, 0))
        combined.paste(text_panel, (grid.width + GAP_X, 0))
    else:
        combined = grid

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
