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


# ============================================================
# PATHS
# ============================================================
INPUT_DIR = "/data/ann/policy_records_20260402_123333"
OUTPUT_DIR = "/data/ann/policy_records_20260402_123333/raw_attention_grids_folders"
EPISODE_SUMMARIES_JSON = "/data/ann/policy_records_20260402_123333/episode_summaries.json"


# ============================================================
# SETTINGS
# ============================================================
# Layer index into the 18-layer network.
# Layer 15 (0-indexed) = 83% through the network — late enough for semantic
# features, same as the original RAW_ATTN_TARGET_LAYER for comparability.
TARGET_LAYER = 16

# Camera names must match the span/grid keys saved in pi0_fast.py
CAM_NAMES = ["right_wrist_0_rgb", "left_wrist_0_rgb"]

CAM_IMAGE_KEYS = [
    "inputs/observation/image",
    "inputs/observation/wrist_image",
]

ALPHA = 0.30
SMOOTHING = "BILINEAR"

BG = (255, 255, 255)
GAP_X = 16
GAP_Y = 16


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


PERCENTILE_CLIP = 95   # clip hot outliers before normalizing
GAMMA = 0.4            # power stretch to brighten mid-range patches


def log_normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pipeline:
      1. Within-image-budget rescale: divide by total attention across all image patches
         so task/state absorption (50-90% of softmax budget) doesn't suppress image scores.
      2. log1p compress.
      3. Clip to PERCENTILE_CLIP (default 95th) so 1-2 extreme patches don't monopolize
         the colorscale — without this, p50 ≈ 0.0004 and only ~5/256 patches are visible.
      4. Joint min-max across cameras.
      5. Gamma stretch (x^GAMMA, default 0.4) to pull mid-range patches up into visible range.
    """
    total_image_attn = sum(float(h.sum()) for h in heats)
    denom = total_image_attn if total_image_attn > 0 else 1.0
    rescaled = [h.astype(np.float32) / denom for h in heats]

    compressed = [np.log1p(np.maximum(h, 0.0)) for h in rescaled]
    flat = np.concatenate([h.reshape(-1) for h in compressed], axis=0)
    mn = float(flat.min())
    clip_val = float(np.percentile(flat, PERCENTILE_CLIP))
    scale = (clip_val - mn) if clip_val > mn else 1.0

    result = []
    for h in compressed:
        h01 = np.clip((h - mn) / scale, 0.0, 1.0)
        result.append(h01 ** GAMMA)
    return result


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


# ============================================================
# Grid layout
# ============================================================
def make_grid(overlays: List[np.ndarray], originals: List[np.ndarray]) -> Image.Image:
    """
    Row 0: overlays
    Row 1: originals
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

        y0 = GAP_Y
        ov = Image.fromarray(overlays[c], mode="RGB")
        if overlays[c].shape[:2] != (cell_h, cell_w):
            tmp = Image.new("RGB", (cell_w, cell_h), BG)
            tmp.paste(ov, (0, 0))
            ov = tmp
        canvas.paste(ov, (x0, y0))

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
def process_one(npy_path: str, out_png: str) -> None:
    rec = load_record(npy_path)

    full_attn_key = "outputs/debug/raw_attn/full_attn"
    if full_attn_key not in rec:
        raise KeyError(f"Missing {full_attn_key} in {npy_path}")

    # full_attn shape: (num_layers, B, num_heads, S_prefix)
    full_attn = np.asarray(rec[full_attn_key])
    if full_attn.ndim != 4:
        raise ValueError(f"{full_attn_key} expected 4D (L,B,H,S), got {full_attn.shape}")

    # Select layer and max over heads -> (S_prefix,)
    # Max preserves the strongest spatial signal from image-attending heads without
    # diluting it with task/state-focused heads that are near-uniform over image patches.
    attn_row = full_attn[TARGET_LAYER, 0, :, :].max(axis=0)  # (S_prefix,)

    heats, imgs = [], []

    for cam_name, img_key in zip(CAM_NAMES, CAM_IMAGE_KEYS):
        span_key = f"outputs/debug/raw_attn/spans/image/{cam_name}"
        grid_key = f"outputs/debug/meta/image_grids/{cam_name}"

        if span_key not in rec:
            raise KeyError(f"Missing {span_key} in {npy_path}")
        if grid_key not in rec:
            raise KeyError(f"Missing {grid_key} in {npy_path}")
        if img_key not in rec:
            raise KeyError(f"Missing {img_key} in {npy_path}")

        span = tuple(int(x) for x in np.asarray(rec[span_key]).tolist())  # (start, end)
        grid_hw = tuple(int(x) for x in np.asarray(rec[grid_key]).tolist())  # (Hp, Wp)

        cam_scores = attn_row[span[0]:span[1]]  # (Hp*Wp,)
        Hp, Wp = grid_hw
        if cam_scores.size != Hp * Wp:
            raise ValueError(
                f"{cam_name}: span length {cam_scores.size} != grid {Hp}x{Wp}={Hp*Wp}"
            )

        heat = cam_scores.reshape(Hp, Wp)
        img = as_u8_rgb(rec[img_key])

        heats.append(heat)
        imgs.append(img)

    if np.array_equal(imgs[0], imgs[1]):
        print(
            f"[WARN] Underlying images identical in {npy_path}. "
            f"Keys: {CAM_IMAGE_KEYS[0]} and {CAM_IMAGE_KEYS[1]}"
        )

    # Joint log normalization across cameras (log1p + shared min-max)
    heats01 = log_normalize_joint(heats)

    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]))
        overlays.append(overlay_heatmap(img, h_up))
        originals.append(img)

    grid = make_grid(overlays, originals)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)


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

        out_png = str(out_dir / task_folder / ep_folder / f"step_{s:06d}.png")

        process_one(f, out_png)
        print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
