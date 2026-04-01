#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import sentencepiece

import openpi.shared.download as download


# ============================================================
# PATHS
# ============================================================
INPUT_DIR = "/home/ziyao/Documents/policy_records_20260304_183031"
OUTPUT_DIR = "/home/ziyao/Documents/policy_records_20260304_183031/raw_attention_grids_folders"
EPISODE_SUMMARIES_JSON = "/home/ziyao/Documents/Projects/openpi/libero_videos/episode_summaries.json"


# ============================================================
# RAW ATTENTION KEYS
# ============================================================
# These must match the debug structure you added in pi0_fast.py
CAM_ATTR_KEYS = [
    "outputs/debug/raw_attn/image/right_wrist_0_rgb/scores",
    "outputs/debug/raw_attn/image/left_wrist_0_rgb/scores",
]

CAM_GRID_KEYS = [
    "outputs/debug/raw_attn/image/right_wrist_0_rgb/grid",
    "outputs/debug/raw_attn/image/left_wrist_0_rgb/grid",
]

CAM_IMAGE_KEYS = [
    "inputs/observation/image",
    "inputs/observation/wrist_image",
]

TASK_SCORE_KEY = "outputs/debug/raw_attn/task/scores"
TASK_TOKEN_KEY = "outputs/debug/raw_attn/task/token_ids"

STATE_SCORE_KEY = "outputs/debug/raw_attn/state/scores"
STATE_TOKEN_KEY = "outputs/debug/raw_attn/state/token_ids"

TARGET_LAYER_KEY = "outputs/debug/raw_attn/target_layer"


# ============================================================
# DISPLAY SETTINGS
# ============================================================
APPLY_RELU = False  # raw attention is already nonnegative
ALPHA = 0.30
SMOOTHING = "BILINEAR"

BG = (255, 255, 255)
GAP_X = 16
GAP_Y = 16

MAX_TEXT_TOKENS = 64
BAR_H = 320


# ============================================================
# TOKENIZER
# ============================================================
def load_tokenizer():
    path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model",
        gs={"token": "anon"},
    )
    with path.open("rb") as f:
        sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return sp


SP = load_tokenizer()


# ============================================================
# Episode indexing helpers
# ============================================================
@dataclass(frozen=True)
class EpisodeInfo:
    task_id: int
    task: str
    episode_num: int
    success: bool
    start_idx: int
    end_idx: int


def load_episode_infos(path: str) -> List[EpisodeInfo]:
    with open(path, "r") as f:
        data = json.load(f)
    infos: List[EpisodeInfo] = []
    for d in data:
        infos.append(
            EpisodeInfo(
                task_id=int(d["task_id"]),
                task=str(d["task"]),
                episode_num=int(d["episode_num"]),
                success=bool(d["success"]),
                start_idx=int(d["start_idx"]),
                end_idx=int(d["end_idx"]),
            )
        )
    return infos


def episode_for_step(step: int, episodes: List[EpisodeInfo]) -> EpisodeInfo:
    for ep in episodes:
        if ep.start_idx <= step <= ep.end_idx:
            return ep
    raise KeyError(f"Step {step} not covered by any episode range in {EPISODE_SUMMARIES_JSON}")


def slugify(s: str, max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


# ============================================================
# Core loading / parsing
# ============================================================
def step_index(path: str) -> int:
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def as_u8_rgb(img) -> np.ndarray:
    """
    Accept:
      - HWC uint8/float
      - CHW uint8/float
      - batched versions with batch size 1
    Return:
      - HWC uint8 RGB
    """
    arr = np.array(img)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Bad image array shape: {arr.shape}")

    arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.min() >= -1.0 and arr.max() <= 1.0:
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0
            arr = np.clip(arr, 0.0, 1.0)
            arr = (255.0 * arr).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def to_1d_scores(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Raw attention scores are expected to be:
      - (N,)
      - (1, N)
      - (B, N) and we use item 0
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{key} must be numpy array, got {type(arr)}")

    if arr.ndim == 1:
        x = arr
    elif arr.ndim == 2:
        x = arr[0]
    else:
        raise ValueError(f"{key} must be 1D or 2D, got shape {arr.shape}")

    return x.astype(np.float32)


def normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    flat = np.concatenate([h.reshape(-1) for h in heats], axis=0)
    mn = float(flat.min())
    mx = float(flat.max())
    denom = (mx - mn) if mx > mn else 1.0
    return [(h - mn) / denom for h in heats]


def resize_heatmap(heat01: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    im = Image.fromarray((np.clip(heat01, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

    if SMOOTHING == "BILINEAR":
        im2 = im.resize((w, h), resample=Image.BILINEAR)
    elif SMOOTHING == "NEAREST":
        im2 = im.resize((w, h), resample=Image.NEAREST)
    else:
        raise ValueError("SMOOTHING must be BILINEAR or NEAREST")

    return np.array(im2, dtype=np.float32) / 255.0


def jet_colormap(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap(img_u8: np.ndarray, heat01_up: np.ndarray) -> np.ndarray:
    heat_rgb = jet_colormap(heat01_up).astype(np.float32)
    img_f = img_u8.astype(np.float32)
    out = img_f * (1.0 - ALPHA) + heat_rgb * ALPHA
    return np.clip(out, 0, 255).astype(np.uint8)


def scores_to_patch_grid(scores_1d: np.ndarray, grid_hw: Tuple[int, int], key: str) -> np.ndarray:
    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    expected = hp * wp
    if scores_1d.size != expected:
        raise ValueError(
            f"{key} expected {expected} patch scores for grid {grid_hw}, got {scores_1d.size}"
        )
    return scores_1d.reshape(hp, wp)


def decode_token_ids(token_ids: np.ndarray) -> List[str]:
    ids = np.asarray(token_ids)
    if ids.ndim == 2:
        ids = ids[0]
    toks = []
    for tid in ids.tolist():
        piece = SP.id_to_piece(int(tid))
        toks.append(piece.replace("▁", ""))
    return toks


# ============================================================
# Bar chart rendering
# ============================================================
def render_bar_chart(scores: np.ndarray, token_ids: np.ndarray, title: str) -> Image.Image:
    import matplotlib.pyplot as plt

    vals = np.asarray(scores)
    if vals.ndim == 2:
        vals = vals[0]

    labels = decode_token_ids(token_ids)

    n = min(len(vals), len(labels), MAX_TEXT_TOKENS)
    vals = vals[:n]
    labels = labels[:n]

    # Normalize locally for readability
    vals = vals.astype(np.float32)
    if vals.size > 0:
        mn, mx = float(vals.min()), float(vals.max())
        denom = (mx - mn) if mx > mn else 1.0
        vals = (vals - mn) / denom

    fig_w = max(10, 0.45 * n)
    fig_h = 4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.bar(np.arange(n), vals)
    ax.set_title(title)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90)
    fig.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)

    return Image.fromarray(img, mode="RGB")


# ============================================================
# Grid layout
# ============================================================
def make_grid(overlays: List[np.ndarray], originals: List[np.ndarray]) -> Image.Image:
    """
    Same layout as your Grad-CAM script:
      - Row 0: overlays
      - Row 1: originals
    """
    assert len(overlays) == len(originals)
    C = len(overlays)

    heights = [im.shape[0] for im in originals]
    widths = [im.shape[1] for im in originals]
    cell_h = max(heights)
    cell_w = max(widths)

    grid_w = C * cell_w + (C + 1) * GAP_X
    grid_h = 2 * cell_h + 3 * GAP_Y

    canvas = Image.new("RGB", (grid_w, grid_h), BG)

    for c in range(C):
        x0 = GAP_X + c * (cell_w + GAP_X)

        # top row: overlay
        y0 = GAP_Y
        ov = Image.fromarray(overlays[c], mode="RGB")
        if overlays[c].shape[:2] != (cell_h, cell_w):
            tmp = Image.new("RGB", (cell_w, cell_h), BG)
            tmp.paste(ov, (0, 0))
            ov = tmp
        canvas.paste(ov, (x0, y0))

        # bottom row: original
        y1 = 2 * GAP_Y + cell_h
        org = Image.fromarray(originals[c], mode="RGB")
        if originals[c].shape[:2] != (cell_h, cell_w):
            tmp = Image.new("RGB", (cell_w, cell_h), BG)
            tmp.paste(org, (0, 0))
            org = tmp
        canvas.paste(org, (x0, y1))

    return canvas


# ============================================================
# Per-step processing
# ============================================================
def process_one(npy_path: str, out_png: str, out_task_png: str, out_state_png: str) -> None:
    rec = load_record(npy_path)

    if TARGET_LAYER_KEY not in rec:
        raise KeyError(f"Missing {TARGET_LAYER_KEY} in {npy_path}")

    heats, imgs = [], []

    for attr_key, grid_key, img_key in zip(CAM_ATTR_KEYS, CAM_GRID_KEYS, CAM_IMAGE_KEYS):
        if attr_key not in rec:
            raise KeyError(f"Missing {attr_key} in {npy_path}")
        if grid_key not in rec:
            raise KeyError(f"Missing {grid_key} in {npy_path}")
        if img_key not in rec:
            raise KeyError(f"Missing {img_key} in {npy_path}")

        scores_1d = to_1d_scores(rec[attr_key], attr_key)
        if APPLY_RELU:
            scores_1d = np.maximum(scores_1d, 0.0)

        grid_hw = tuple(np.asarray(rec[grid_key]).tolist())
        heat = scores_to_patch_grid(scores_1d, grid_hw, attr_key)

        img = as_u8_rgb(rec[img_key])

        heats.append(heat)
        imgs.append(img)

    if np.array_equal(imgs[0], imgs[1]):
        print(
            f"[WARN] Underlying images identical in {npy_path}. "
            f"Keys used: {CAM_IMAGE_KEYS[0]} and {CAM_IMAGE_KEYS[1]}"
        )

    heats01 = normalize_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]))
        overlays.append(overlay_heatmap(img, h_up))
        originals.append(img)

    grid = make_grid(overlays, originals)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)

    # task chart
    if TASK_SCORE_KEY in rec and TASK_TOKEN_KEY in rec:
        task_img = render_bar_chart(
            rec[TASK_SCORE_KEY],
            rec[TASK_TOKEN_KEY],
            title=f"Raw Attention to Task Tokens (layer {int(rec[TARGET_LAYER_KEY])})",
        )
        task_img.save(out_task_png)

    # state chart
    if STATE_SCORE_KEY in rec and STATE_TOKEN_KEY in rec:
        state_img = render_bar_chart(
            rec[STATE_SCORE_KEY],
            rec[STATE_TOKEN_KEY],
            title=f"Raw Attention to State Tokens (layer {int(rec[TARGET_LAYER_KEY])})",
        )
        state_img.save(out_state_png)


# ============================================================
# Main
# ============================================================
def main() -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_infos(EPISODE_SUMMARIES_JSON)

    files = sorted(glob.glob(str(in_dir / "step_*.npy")), key=step_index)
    if not files:
        raise FileNotFoundError(f"No step_*.npy found in {INPUT_DIR}")

    for f in files:
        s = step_index(f)

        ep = episode_for_step(s, episodes)
        task_folder = f"t{ep.task_id}_{slugify(ep.task)}"
        ep_folder = f"ep{ep.episode_num}_{str(ep.success).lower()}"

        # Main 2x2 image grid
        out_png = str(out_dir / task_folder / ep_folder / f"step_{s:06d}.png")

        # Separate token charts in the same episode folder
        out_task_png = str(out_dir / task_folder / ep_folder / f"step_{s:06d}_task.png")
        out_state_png = str(out_dir / task_folder / ep_folder / f"step_{s:06d}_state.png")

        process_one(f, out_png, out_task_png, out_state_png)
        print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
