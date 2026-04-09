from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sentencepiece
import openpi.shared.download as download

# ============================================================
# ATTRIBUTION KEYS
# ============================================================
TASK_SCORE_KEY       = "outputs/debug/attr/task/scores"
TASK_PIECE_BEGIN_KEY = "outputs/debug/attr/task/task_piece_begin"
TASK_PIECE_END_KEY   = "outputs/debug/attr/task/task_piece_end"
TASK_PIECE_ID_KEY    = "outputs/debug/attr/task/task_piece_id"

STATE_SCORE_KEY       = "outputs/debug/attr/state/scores"
STATE_PIECE_BEGIN_KEY = "outputs/debug/attr/state/state_piece_begin"
STATE_PIECE_END_KEY   = "outputs/debug/attr/state/state_piece_end"
STATE_PIECE_ID_KEY    = "outputs/debug/attr/state/state_piece_id"

# Controls how wide the text panel is relative to the grid height.
# Increase to widen the panel, decrease to narrow it.
CHAR_WIDTH = 0.08   # inches per character tile in the matplotlib figure


# ============================================================
# SHARED EPISODE UTILITIES
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
    return [
        EpisodeInfo(
            task_id=int(d["task_id"]),
            task=str(d["task"]),
            episode_num=int(d["episode_num"]),
            success=bool(d["success"]),
            start_idx=int(d["start_idx"]),
            end_idx=int(d["end_idx"]),
        )
        for d in data
    ]


def episode_for_step(step: int, episodes: List[EpisodeInfo]) -> EpisodeInfo:
    for ep in episodes:
        if ep.start_idx <= step <= ep.end_idx:
            return ep
    raise KeyError(f"Step {step} not covered by any episode")


def slugify(s: str, max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len].rstrip("_") if len(s) > max_len else s


def load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    # numpy saves a plain dict as a 0-d object array; .item() unwraps it
    if isinstance(p, np.ndarray) and p.shape == () and p.dtype == object:
        p = p.item()
    if not isinstance(p, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(p)}")
    return p


def step_index(path: str) -> int:
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


# ============================================================
# SHARED IMAGE UTILITIES
# ============================================================
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


# ============================================================
# TEXT PANEL  (sentencepiece tokenizer, loaded once at import)
# ============================================================
_tokenizer_path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model",
    gs={"token": "anon"},
)
with _tokenizer_path.open("rb") as _f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=_f.read())


def _chars_and_norm(
    token_scores: np.ndarray,
    begin: np.ndarray,
    end: np.ndarray,
    piece_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand per-token scores to per-character arrays using piece spans.

    The tokenizer produces subword pieces (e.g. "basket" → ["bask", "et"]).
    Each piece maps to a character range [begin[i], end[i]) in the decoded string.
    We replicate the token's score once per character so the heatmap tiles align
    with individual characters rather than subword pieces.
    Scores are abs-valued then normalized to [0,1] within this modality.
    """
    decoded = sp.decode(piece_ids)
    chars, scores = [], []
    for i, (b, e) in enumerate(zip(begin, end)):
        for j in range(int(b), int(e)):
            chars.append(decoded[j])
            scores.append(float(token_scores[i]))
    values = np.abs(np.array(scores, dtype=np.float32))
    maxv   = float(values.max()) if values.size > 0 else 0.0
    norm   = values / maxv if maxv > 0 else np.zeros_like(values)
    return np.array(chars), norm


def _render_heatmap(ax, chars: np.ndarray, norm: np.ndarray, title: str, fontsize: int = 7) -> None:
    for i, (c, v) in enumerate(zip(chars, norm)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=jet_color(v)))
        ax.text(i + 0.5, 0.5, c, ha="center", va="center", fontsize=fontsize, color="black")
    ax.set_xlim(0, len(chars))
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=fontsize + 1, pad=2)
    ax.axis("off")


def _build_text_panel_image(
    task_chars: np.ndarray,
    task_norm: np.ndarray,
    state_chars: np.ndarray,
    state_norm: np.ndarray,
    target_height_px: int,
) -> Image.Image:
    dpi      = 150
    # figure width scales with the longer of the two text sequences so every
    # character gets roughly CHAR_WIDTH inches regardless of text length
    fig_w_in = max(max(len(task_chars), len(state_chars), 1) * CHAR_WIDTH, 2.0)
    fig_h_in = 3.5

    fig, (ax_task, ax_state) = plt.subplots(
        2, 1, figsize=(fig_w_in, fig_h_in), dpi=dpi,
        gridspec_kw={"hspace": 0.4},
    )
    _render_heatmap(ax_task,  task_chars,  task_norm,  "Task")
    _render_heatmap(ax_state, state_chars, state_norm, "State")

    # render to an in-memory PNG, then open as PIL — avoids writing to disk
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)

    # scale to target_height_px while preserving aspect ratio so the panel
    # lines up with the image grid height when stitched side-by-side
    pil   = Image.open(buf).convert("RGB")
    ratio = target_height_px / pil.height
    return pil.resize((max(int(pil.width * ratio), 1), target_height_px), Image.BILINEAR)


def render_text_panel_as_image(obj: dict, target_height_px: int) -> Image.Image:
    """
    Text panel from pre-computed attribution scores (used by gradcam script).
    Width is determined by character count × CHAR_WIDTH.
    """
    task_chars,  task_norm  = _chars_and_norm(
        obj[TASK_SCORE_KEY][0],  obj[TASK_PIECE_BEGIN_KEY],  obj[TASK_PIECE_END_KEY],  obj[TASK_PIECE_ID_KEY].tolist()
    )
    state_chars, state_norm = _chars_and_norm(
        obj[STATE_SCORE_KEY][0], obj[STATE_PIECE_BEGIN_KEY], obj[STATE_PIECE_END_KEY], obj[STATE_PIECE_ID_KEY].tolist()
    )
    return _build_text_panel_image(task_chars, task_norm, state_chars, state_norm, target_height_px)


def render_text_panel_from_token_scores(
    obj: dict,
    task_token_scores: np.ndarray,
    state_token_scores: np.ndarray,
    target_height_px: int,
) -> Image.Image:
    """
    Text panel from raw per-token attention scores (used by raw_weights script).
    task_token_scores / state_token_scores are 1-D arrays with one value per token.
    """
    task_chars,  task_norm  = _chars_and_norm(
        task_token_scores,  obj[TASK_PIECE_BEGIN_KEY],  obj[TASK_PIECE_END_KEY],  obj[TASK_PIECE_ID_KEY].tolist()
    )
    state_chars, state_norm = _chars_and_norm(
        state_token_scores, obj[STATE_PIECE_BEGIN_KEY], obj[STATE_PIECE_END_KEY], obj[STATE_PIECE_ID_KEY].tolist()
    )
    return _build_text_panel_image(task_chars, task_norm, state_chars, state_norm, target_height_px)
