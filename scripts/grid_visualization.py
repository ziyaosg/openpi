#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


INPUT_DIR = "/home/ziyao/policy_records_20260225_211012"
OUTPUT_DIR = "/home/ziyao/policy_records_20260225_211012/gradcam_grids"

CAM_ATTR_KEYS = [
    "outputs/debug/attr/image/right_wrist_0_rgb",
    "outputs/debug/attr/image/left_wrist_0_rgb",
]

CAM_IMAGE_KEYS = [
    "inputs/observation/image",  
    "inputs/observation/wrist_image",        
]

APPLY_RELU = True

# Alpha-blending factor when overlaying heatmap on top of RGB
ALPHA = 0.30

SMOOTHING = "BILINEAR"

# Grid
BG = (255, 255, 255)  # white background
GAP_X = 16            # horizontal padding between panels
GAP_Y = 16            # vertical padding between rows


def load_record(npy_path: str) -> dict:
    """
    Load one step_*.npy file.
    Some npy files are saved as a 0-d object ndarray containing a dict, so unwrap that.
    """
    p = np.load(npy_path, allow_pickle=True)
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def as_u8_rgb(img) -> np.ndarray:
    """
    Ensure an input image is uint8 RGB with shape (H, W, 3).
    We accept (H, W, 3) or (H, W, 4) and drop alpha if present.
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] in (3, 4)):
        raise ValueError(
            f"Bad image array shape: {None if not isinstance(img, np.ndarray) else img.shape}"
        )
    img = img[..., :3]  # drop alpha if it exists
    if img.dtype != np.uint8:
        # Some pipelines may store float images; clip to [0,255] and cast.
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def to_2d_heatmap(arr: np.ndarray, key: str) -> np.ndarray:
    """
    Your attribution can be:
      - (H, W)            : already 2D
      - (1, H, W)         : batch dimension = 1
      - (B, H, W)         : general batch, we take index 0 by default

    Returns:
      (H, W) float32 heatmap.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{key} must be numpy array, got {type(arr)}")

    if arr.ndim == 2:
        h = arr
    elif arr.ndim == 3:
        # Use the first element of the batch.
        # If you ever log multiple items per step and want the average:
        #   h = arr.mean(axis=0)
        h = arr[0]
    else:
        raise ValueError(
            f"{key} must be 2D or 3D (batched) heatmap, got shape {arr.shape}"
        )

    if h.ndim != 2:
        raise ValueError(f"{key} after unbatch must be 2D, got shape {h.shape}")

    return h.astype(np.float32)


def normalize_joint(heats: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize across two camera angles using a shared min/max
    """
    flat = np.concatenate([h.reshape(-1) for h in heats], axis=0)
    mn = float(flat.min())
    mx = float(flat.max())
    denom = (mx - mn) if mx > mn else 1.0
    return [(h - mn) / denom for h in heats]


def resize_heatmap(heat01: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Upsample a patch-grid heatmap (e.g., 16x16) to the original image resolution (H, W).

    Input:
      heat01: (Hg, Wg) in [0,1]
      out_hw: (H, W) of the target image

    Output:
      (H, W) float32 in [0,1]
    """
    h, w = out_hw
    im = Image.fromarray(
        (np.clip(heat01, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    )
    # Smooth upsampling
    if SMOOTHING == "BILINEAR":
        im2 =  im.resize((w, h), resample=Image.BILINEAR)
    elif SMOOTHING == "NEAREST":
        im2 =  im.resize((w, h), resample=Image.NEAREST)
    else:
        raise Exception("Please specify a valid smoothing method. BILINEAR or NEAREST.")
    return (np.array(im2, dtype=np.float32) / 255.0)


def jet_colormap(x01: np.ndarray) -> np.ndarray:
    """
    Lightweight Jet-like colormap without matplotlib.

    Input:
      x01: (H, W) in [0,1]
    Output:
      (H, W, 3) uint8 RGB
    """
    x = np.clip(x01, 0.0, 1.0)
    # Piecewise-linear "jet" approximation
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def overlay_heatmap(img_u8: np.ndarray, heat01_up: np.ndarray) -> np.ndarray:
    """
    Alpha-blend the heatmap (colorized) over the RGB image.
    """
    heat_rgb = jet_colormap(heat01_up).astype(np.float32)
    img_f = img_u8.astype(np.float32)
    out = img_f * (1.0 - ALPHA) + heat_rgb * ALPHA
    return np.clip(out, 0, 255).astype(np.uint8)


def make_grid(overlays: List[np.ndarray], originals: List[np.ndarray]) -> Image.Image:
    """
    Make a 2 x 2 grid:
      - Row 0: overlays
      - Row 1: originals
      - Each column corresponds to the same "camera"/attr key.
    """
    assert len(overlays) == len(originals)
    C = len(overlays)

    # Make all cells the same size (pad smaller ones if needed)
    heights = [im.shape[0] for im in originals]
    widths = [im.shape[1] for im in originals]
    cell_h = max(heights)
    cell_w = max(widths)

    grid_w = C * cell_w + (C + 1) * GAP_X
    grid_h = 2 * cell_h + 3 * GAP_Y

    canvas = Image.new("RGB", (grid_w, grid_h), BG)

    for c in range(C):
        x0 = GAP_X + c * (cell_w + GAP_X)

        # ---------- Top row: overlay ----------
        y0 = GAP_Y
        ov = Image.fromarray(overlays[c], mode="RGB")
        if overlays[c].shape[:2] != (cell_h, cell_w):
            tmp = Image.new("RGB", (cell_w, cell_h), BG)
            tmp.paste(ov, (0, 0))
            ov = tmp
        canvas.paste(ov, (x0, y0))

        # ---------- Bottom row: original ----------
        y1 = 2 * GAP_Y + cell_h
        org = Image.fromarray(originals[c], mode="RGB")
        if originals[c].shape[:2] != (cell_h, cell_w):
            tmp = Image.new("RGB", (cell_w, cell_h), BG)
            tmp.paste(org, (0, 0))
            org = tmp
        canvas.paste(org, (x0, y1))

    return canvas


def step_index(path: str) -> int:
    """
    Extract step number from a filename like 'step_0.npy'.
    Used to sort and to name output PNGs.
    """
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def process_one(npy_path: str, out_png: str) -> None:
    """
    Load one record, produce one 2x2 (2 rows x 2 cols) visualization, save as PNG.
    """
    rec = load_record(npy_path)

    heats, imgs = [], []

    # Pair each attribution map with its chosen underlying RGB image.
    for attr_key, img_key in zip(CAM_ATTR_KEYS, CAM_IMAGE_KEYS):
        if attr_key not in rec:
            raise KeyError(f"Missing {attr_key} in {npy_path}")
        if img_key not in rec:
            raise KeyError(f"Missing {img_key} in {npy_path}")

        # Convert heatmap to 2D and (optionally) keep only positive evidence
        heat = to_2d_heatmap(rec[attr_key], attr_key)
        if APPLY_RELU:
            heat = np.maximum(heat, 0.0)

        # Convert RGB image to uint8 (H, W, 3)
        img = as_u8_rgb(rec[img_key])

        heats.append(heat)
        imgs.append(img)

    # If the two stored images are literally identical, something is odd upstream.
    if np.array_equal(imgs[0], imgs[1]):
        print(
            f"[WARN] Underlying images identical in {npy_path}. "
            f"Keys used: {CAM_IMAGE_KEYS[0]} and {CAM_IMAGE_KEYS[1]}"
        )

    # Normalize the two heatmaps *together* so intensities are comparable.
    heats01 = normalize_joint(heats)

    # Build per-column overlay images
    overlays, originals = [], []
    for img, h01 in zip(imgs, heats01):
        # Upsample from patch grid (e.g., 16x16) to the image resolution
        h_up = resize_heatmap(h01, out_hw=(img.shape[0], img.shape[1]))
        overlays.append(overlay_heatmap(img, h_up))
        originals.append(img)

    # Compose 2x2 grid and write
    grid = make_grid(overlays, originals)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)


def main() -> None:
    """
    Iterate over all step_*.npy files and write a corresponding PNG for each.
    """
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_dir / "step_*.npy")), key=step_index)
    if not files:
        raise FileNotFoundError(f"No step_*.npy found in {INPUT_DIR}")

    for f in files:
        s = step_index(f)
        out_png = str(out_dir / f"step_{s:06d}.png")
        process_one(f, out_png)
        print(f"[OK] {out_png}")


if __name__ == "__main__":
    main()
