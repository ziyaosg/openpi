from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

# Make scripts/ importable so we can use attention_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_utils.io import (  # noqa: E402
    EpisodeInfo,
    episode_for_step,
    load_episode_infos,
    load_record,
    step_index,
)
from attention_utils.names import slugify  # noqa: E402


def as_u8_rgb(img) -> np.ndarray:
    arr = np.array(img)
    # drop batch dim if present (B=1)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # channel-first (C, H, W) → channel-last (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Bad image array shape: {arr.shape}")
    arr = arr[..., :3]  # drop alpha if present
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.min() >= -1.0 and arr.max() <= 1.0:
            # float image in [-1,1] or [0,1] — scale to [0,255]
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0  # [-1, 1] → [0, 1]
            arr = np.clip(arr, 0.0, 1.0)
            arr = (255.0 * arr).astype(np.uint8)
        else:
            # assume already in [0,255] range stored as float
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def jet_colormap(x01: np.ndarray) -> np.ndarray:
    """Jet-like colormap. Input: (...) in [0,1]. Output: (..., 3) uint8.

    Each channel is a piecewise-linear tent function shifted along the [0,1] axis:
      blue peaks at x=0.25, green at x=0.5, red at x=0.75.
    The formula `1.5 - |4x - k|` gives a triangle with peak 1.5 at x=k/4,
    clipped to [0,1] so only the central portion is visible.
    """
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def jet_color(x: float) -> tuple:
    """Scalar version of jet_colormap; returns an RGBA tuple for matplotlib patches."""
    x = float(np.clip(x, 0.0, 1.0))
    r = float(np.clip(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0))
    g = float(np.clip(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0))
    b = float(np.clip(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0))
    return (r, g, b, 1.0)


def resize_heatmap(heat01: np.ndarray, out_hw: Tuple[int, int], smoothing: str = "BILINEAR") -> np.ndarray:
    # Upsample a small patch-grid heatmap (e.g. 16×16) to full image resolution.
    # We go through PIL because it gives clean bilinear filtering without importing scipy.
    # The round-trip float→uint8→float loses 1/255 precision, which is negligible here.
    h, w = out_hw
    im = Image.fromarray((np.clip(heat01, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resample = Image.BILINEAR if smoothing == "BILINEAR" else Image.NEAREST
    return np.array(im.resize((w, h), resample=resample), dtype=np.float32) / 255.0


def overlay_heatmap(img_u8: np.ndarray, heat01_up: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    heat_rgb = jet_colormap(heat01_up).astype(np.float32)
    out = img_u8.astype(np.float32) * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_grid(
    overlays: List[np.ndarray],
    originals: List[np.ndarray],
    bg: Tuple[int, int, int] = (255, 255, 255),
    gap_x: int = 16,
    gap_y: int = 16,
) -> Image.Image:
    """2 rows × C cols grid: row 0 = overlays, row 1 = originals."""
    C = len(overlays)
    assert C == len(originals)

    cell_h = max(im.shape[0] for im in originals)
    cell_w = max(im.shape[1] for im in originals)

    canvas = Image.new("RGB", (C * cell_w + (C + 1) * gap_x, 2 * cell_h + 3 * gap_y), bg)

    for c in range(C):
        x0 = gap_x + c * (cell_w + gap_x)
        for row, arr in enumerate([overlays[c], originals[c]]):
            y0 = gap_y + row * (cell_h + gap_y)
            cell_img = Image.fromarray(arr, mode="RGB")
            if arr.shape[:2] != (cell_h, cell_w):
                tmp = Image.new("RGB", (cell_w, cell_h), bg)
                tmp.paste(cell_img, (0, 0))
                cell_img = tmp
            canvas.paste(cell_img, (x0, y0))

    return canvas
