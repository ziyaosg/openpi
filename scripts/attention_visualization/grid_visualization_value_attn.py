#!/usr/bin/env python3
"""Value-weighted attention heatmaps.

Two modes controlled by MODE:

  "v_norm"   — value-norm weighted attention:
                score[s] = attn[h,s] * ||v[s,h,:]||_2
                High score means the head both attends to position s AND
                has a large-magnitude value there. Visualises WHERE the
                head is doing significant work.

  "v_cosine" — value direction alignment:
                o_h = sum_s(attn[h,s] * v[s,h,:])        (head output)
                score[s] = |cosine(v[s,h,:], o_h)|
                High score means the value at position s points in the
                same direction as what the head actually outputs.
                Visualises WHICH positions drive the head's output direction.

In both cases head selection follows the two-stage strategy from
grid_visualization_raw_weights.py: rank heads at HEAD_SELECTION_LAYER by
image/(task+state) budget ratio, then aggregate the selected heads at
TARGET_LAYER via element-wise max.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

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
from ..attention_utils.keys import CAM_IMAGE_KEYS, CAM_NAMES, IMAGE_PATCH_GRID

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260421_133206"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260421_133206/episode_summaries.json"

# "v_norm" or "v_cosine"
MODE = "v_norm"

OUTPUT_DIR = str(Path(INPUT_DIR) / f"heatmap_{MODE}")

# Layer whose values/attention are used for the final scores.
TARGET_LAYER = 16
# Layer used only for head selection (stable image-focused head ranking).
HEAD_SELECTION_LAYER = 1
# Number of top image-focused heads to include.
TOP_K_HEADS = 3

ALPHA     = 0.30
SMOOTHING = "BILINEAR"
BG        = (255, 255, 255)
GAP_X     = 16
GAP_Y     = 16
# Percentile clip before min-max (suppresses extreme outlier patches).
PERCENTILE_CLIP = 95
# Gamma < 1 pulls mid-range values into the visible colour range.
GAMMA = 0.4

FULL_ATTN_KEY = "outputs/debug/attn/weights"
FULL_V_KEY    = "outputs/debug/attn/v"
LAYERS_KEY    = "outputs/debug/attn/layers"


# ============================================================
# HEAD SELECTION  (same logic as grid_visualization_raw_weights)
# ============================================================
def _select_heads(
    layer_attn: np.ndarray,
    all_img_spans: List[Tuple[int, int]],
    task_span: Tuple[int, int] | None,
    state_span: Tuple[int, int] | None,
    top_k: int,
) -> np.ndarray:
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


# ============================================================
# SCORE FUNCTIONS
# ============================================================
def score_v_norm(
    attn: np.ndarray,   # (H, S)
    v: np.ndarray,      # (S, K, D)  K=1 for GQA
    top_heads: np.ndarray,
) -> np.ndarray:
    """attn[h,s] * ||v[s,0,:]||_2, then max over selected heads → (S,)."""
    v_norms = np.linalg.norm(v[:, 0, :], axis=-1)   # (S,)
    weighted = attn * v_norms[np.newaxis, :]          # (H, S)
    return weighted[top_heads, :].max(axis=0).astype(np.float32)


def score_v_cosine(
    attn: np.ndarray,   # (H, S)
    v: np.ndarray,      # (S, K, D)  K=1 for GQA
    top_heads: np.ndarray,
) -> np.ndarray:
    """|cosine(v[s,0,:], o_h)| where o_h = attn[h,:] @ v[:,0,:],
    then max over selected heads → (S,)."""
    v0 = v[:, 0, :]                                                          # (S, D)
    o = attn @ v0                                                            # (H, D)
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)         # (H, D)
    v_unit = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-9)       # (S, D)
    cos = v_unit @ o_unit.T                                                  # (S, H)
    return np.abs(cos)[:, top_heads].max(axis=1).astype(np.float32)


# ============================================================
# NORMALISATION
# ============================================================
def _normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    """Log-compress, clip at PERCENTILE_CLIP, joint min-max, gamma stretch."""
    total = sum(float(h.sum()) for h in heats)
    denom = total if total > 0 else 1.0
    compressed = [np.log1p(np.maximum(h / denom, 0.0)) for h in heats]
    flat = np.concatenate([h.reshape(-1) for h in compressed])
    mn = float(flat.min())
    clip_val = float(np.percentile(flat, PERCENTILE_CLIP))
    scale = (clip_val - mn) if clip_val > mn else 1.0
    return [(np.clip((h - mn) / scale, 0.0, 1.0) ** GAMMA) for h in compressed]


def _normalize_cosine_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    """Joint min-max for cosine scores which are already in [0, 1]."""
    flat = np.concatenate([h.reshape(-1) for h in heats])
    mn, mx = float(flat.min()), float(flat.max())
    scale = (mx - mn) if mx > mn else 1.0
    return [(h - mn) / scale for h in heats]


# ============================================================
# PER-STEP PROCESSING
# ============================================================
def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    for key in (FULL_ATTN_KEY, FULL_V_KEY, LAYERS_KEY):
        if key not in rec:
            raise KeyError(f"Missing {key!r} in {npy_path}")

    full_attn = np.asarray(rec[FULL_ATTN_KEY])   # (2, B, H, S)
    full_v    = np.asarray(rec[FULL_V_KEY])       # (2, B, S, H, D)

    stored_layers = np.asarray(rec[LAYERS_KEY]).tolist()
    sel_idx    = stored_layers.index(HEAD_SELECTION_LAYER)
    target_idx = stored_layers.index(TARGET_LAYER)

    cam_spans = []
    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        for k in (f"outputs/debug/spans/image/{cam_name}", img_key):
            if k not in rec:
                raise KeyError(f"Missing {k!r} in {npy_path}")
        cam_spans.append(tuple(int(x) for x in np.asarray(rec[f"outputs/debug/spans/image/{cam_name}"]).tolist()))

    task_span = state_span = None
    if "outputs/debug/spans/task" in rec:
        task_span  = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/task"]).tolist())
    if "outputs/debug/spans/state" in rec:
        state_span = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/state"]).tolist())

    # Two-stage head selection.
    selector_attn = full_attn[sel_idx, 0, :, :]     # (H, S)
    top_heads = _select_heads(selector_attn, cam_spans, task_span, state_span, TOP_K_HEADS)
    print(f"  heads (layer {HEAD_SELECTION_LAYER} → layer {TARGET_LAYER}): {sorted(int(h) for h in top_heads)}")

    attn   = full_attn[target_idx, 0, :, :]          # (H, S)
    v      = full_v[target_idx, 0, :, :, :]          # (S, H, D)

    if MODE == "v_norm":
        score_fn = score_v_norm
    elif MODE == "v_cosine":
        score_fn = score_v_cosine
    else:
        raise ValueError(f"Unknown MODE {MODE!r}. Choose 'v_norm' or 'v_cosine'.")

    full_score = score_fn(attn, v, top_heads)    # (S,)

    Hp, Wp = IMAGE_PATCH_GRID
    heats, imgs = [], []
    for span, (cam_name, img_key) in zip(cam_spans, zip(CAM_NAMES, CAM_IMAGE_KEYS)):
        cam_scores = full_score[span[0]:span[1]]
        if cam_scores.size != Hp * Wp:
            raise ValueError(f"{cam_name}: span length {cam_scores.size} != {Hp}×{Wp}={Hp*Wp}")
        heats.append(cam_scores.reshape(Hp, Wp))
        imgs.append(as_u8_rgb(rec[img_key]))

    if np.array_equal(imgs[0], imgs[1]):
        print(f"[WARN] Images identical in {npy_path}")

    if MODE == "v_norm":
        heats01 = _normalize_joint(heats)
    else:
        heats01 = _normalize_cosine_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]), smoothing=SMOOTHING)
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_scores  = full_score[task_span[0]:task_span[1]]   if task_span  is not None else None
    state_scores = full_score[state_span[0]:state_span[1]] if state_span is not None else None
    if task_scores is None or state_scores is None:
        raise KeyError("task/state spans missing — cannot build text panel")
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
