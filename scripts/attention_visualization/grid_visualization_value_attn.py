#!/usr/bin/env python3
"""Value-weighted attention heatmaps.

Three modes controlled by MODE:

  "v_norm"     — value-norm weighted attention:
                  score[s] = attn[h,s] * ||v[s,0,:]||_2
                  High score means the head both attends to position s AND
                  has a large-magnitude value there.

  "v_cosine"   — value direction alignment:
                  o_h = sum_s(attn[h,s] * v[s,0,:])       (head output)
                  score[s] = |cosine(v[s,0,:], o_h)|
                  High score means the value at position s points in the
                  same direction as what the head actually outputs.

  "v_combined" — projection of token contribution onto output direction:
                  score[s] = |attn[h,s] * <v[s,0,:], o_unit_h>|
                           = attn[h,s] * ||v[s,0,:]|| * |cos(v[s,0,:], o_h)|
                  Only positions that are both strongly attended AND
                  directionally aligned score high. Combines the signals
                  from v_norm and v_cosine.

v is stored with K KV-heads (K=1 for gemma_2b GQA); v[s,0,:] is the single
shared value vector per position. Head selection follows the two-stage
strategy from grid_visualization_raw_weights.py.
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
INPUT_DIR              = "/home/ziyao/Documents/policy_records_20260423_180932"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/policy_records_20260423_180932/episode_summaries.json"

# "v_norm", "v_cosine", or "v_combined"
MODE = "v_norm"

OUTPUT_DIR = str(Path(INPUT_DIR) / f"heatmap_{MODE}")

# Layer whose values/attention are used for both head selection and scoring.
TARGET_LAYER = 16
# Number of top heads to include per modality.
TOP_K_HEADS = 3
# Head selection strategy: "ratio" (one-vs-rest) or "budget" (raw attention mass).
HEAD_SELECTION_MODE = "budget"

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
def _select_heads_for_span(
    attn: np.ndarray,
    span: Tuple[int, int],
    all_spans: List[Tuple[int, int]],
    top_k: int,
) -> np.ndarray:
    """Return top_k heads for the given span using HEAD_SELECTION_MODE.

    "ratio":  score[h] = attn[h, span].sum() / attn[h, all_other_spans].sum()
              Ranks by relative preference — which heads attend to this span
              more than to everything else.
    "budget": score[h] = attn[h, span].sum()
              Ranks by absolute attention mass — which heads put the most
              attention on this span, compared directly across all heads.
    """
    s0, s1 = span
    target_budget = attn[:, s0:s1].sum(axis=1)   # (H,)
    if HEAD_SELECTION_MODE == "ratio":
        other_budget = np.zeros(attn.shape[0], dtype=np.float32)
        for sp in all_spans:
            if sp != span:
                other_budget += attn[:, sp[0]:sp[1]].sum(axis=1)
        score = target_budget / (other_budget + 1e-9)
    else:
        score = target_budget
    return np.argsort(score)[-min(top_k, attn.shape[0]):]


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
    """cosine(v[s,0,:], o_h) where o_h = attn[h,:] @ v[:,0,:],
    then max over selected heads → (S,). Values in [-1, 1]; only
    positively-aligned tokens score high."""
    v0 = v[:, 0, :]                                                          # (S, D)
    o = attn @ v0                                                            # (H, D)
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)         # (H, D)
    v_unit = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-9)       # (S, D)
    cos = v_unit @ o_unit.T                                                  # (S, H)
    return cos[:, top_heads].max(axis=1).astype(np.float32)


def score_v_combined(
    attn: np.ndarray,   # (H, S)
    v: np.ndarray,      # (S, K, D)  K=1 for GQA
    top_heads: np.ndarray,
) -> np.ndarray:
    """|attn[h,s] * <v[s,0,:], o_unit_h>|, then max over selected heads → (S,).
    Equivalent to attn[s] * ||v[s]|| * |cos(v[s], o_h)|."""
    v0 = v[:, 0, :]                                                          # (S, D)
    o = attn @ v0                                                            # (H, D)
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)         # (H, D)
    proj = attn.T * (v0 @ o_unit.T)                                          # (S, H)
    return np.abs(proj)[:, top_heads].max(axis=1).astype(np.float32)


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
    """Joint min-max for cosine scores in [-1, 1]."""
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

    # Head selection and scoring both use TARGET_LAYER.
    attn = full_attn[target_idx, 0, :, :]   # (H, S)
    v    = full_v[target_idx, 0, :, :, :]   # (S, K, D)

    if MODE == "v_norm":
        score_fn = score_v_norm
    elif MODE == "v_cosine":
        score_fn = score_v_cosine
    elif MODE == "v_combined":
        score_fn = score_v_combined
    else:
        raise ValueError(f"Unknown MODE {MODE!r}. Choose 'v_norm', 'v_cosine', or 'v_combined'.")

    # Per-modality one-vs-rest head selection and scoring, both at TARGET_LAYER.
    if task_span is None or state_span is None:
        raise KeyError("task/state spans missing — cannot build text panel")
    all_spans = list(cam_spans) + [task_span, state_span]

    Hp, Wp = IMAGE_PATCH_GRID
    heats, imgs = [], []
    for span, (cam_name, img_key) in zip(cam_spans, zip(CAM_NAMES, CAM_IMAGE_KEYS)):
        heads      = _select_heads_for_span(attn, span, all_spans, TOP_K_HEADS)
        cam_scores = score_fn(attn, v, heads)[span[0]:span[1]]
        print(f"  {cam_name} heads (layer {TARGET_LAYER}): {sorted(int(h) for h in heads)}")
        if cam_scores.size != Hp * Wp:
            raise ValueError(f"{cam_name}: span length {cam_scores.size} != {Hp}×{Wp}={Hp*Wp}")
        heats.append(cam_scores.reshape(Hp, Wp))
        imgs.append(as_u8_rgb(rec[img_key]))

    if np.array_equal(imgs[0], imgs[1]):
        print(f"[WARN] Images identical in {npy_path}")

    if MODE == "v_cosine":
        heats01 = _normalize_cosine_joint(heats)
    else:
        heats01 = _normalize_joint(heats)   # v_norm and v_combined: log1p + percentile clip + gamma

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]), smoothing=SMOOTHING)
        overlays.append(overlay_heatmap(img, h_up, alpha=ALPHA))
        originals.append(img)

    grid = make_grid(overlays, originals, bg=BG, gap_x=GAP_X, gap_y=GAP_Y)

    task_heads   = _select_heads_for_span(attn, task_span,  all_spans, TOP_K_HEADS)
    state_heads  = _select_heads_for_span(attn, state_span, all_spans, TOP_K_HEADS)
    task_scores  = score_fn(attn, v, task_heads)[task_span[0]:task_span[1]]
    state_scores = score_fn(attn, v, state_heads)[state_span[0]:state_span[1]]
    print(f"  task  heads (layer {TARGET_LAYER}): {sorted(int(h) for h in task_heads)}")
    print(f"  state heads (layer {TARGET_LAYER}): {sorted(int(h) for h in state_heads)}")
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
