"""
Common utilities for pi05 attention visualization.

Data format for a pi05/libero run (3 cameras, max_token_len=200, action_horizon=10):
  attn weights: (L, H, action_tokens, S) = (2, 8, 10, key_end)
  V:            (L, key_end, K, D)       = (2, key_end, 1, 256)  [K=1 for GQA]
  GradCAM img:  (16, 16) float32
  GradCAM task: (200,) float32 (covers full tokenized_prompt including state)
  GradCAM state:(state_token_len,) float32 (subset of task span)
  Raw alpha img:(16, 16) float32
  Raw alpha task:(200,) float32
  Cameras:      base_0_rgb (0:256), left_wrist_0_rgb (256:512), right_wrist_0_rgb (512:768)
  Task tokens:  768 : 768+task_token_len   (language prefix up to "State: ")
  State tokens: 768+task_token_len : 768+task_token_len+state_token_len
  key_end = prefix_len (= 768+200 = 968) for pi05; prefix_len+1 for pi0.
  Attention mass per modality in outputs/debug/attention_mass/{image,task,state}_mass.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

CAM_NAMES = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
# right_wrist reuses wrist_image (libero has 2 cameras; 3rd slot duplicated)
CAM_IMAGE_KEYS = [
    "inputs/observation/image",
    "inputs/observation/wrist_image",
    "inputs/observation/wrist_image",
]

PATCH_H, PATCH_W = 16, 16

FULL_ATTN_KEY = "outputs/debug/attn/weights"   # (L, H, action_tokens, S)
FULL_V_KEY    = "outputs/debug/attn/v"          # (L, S, K, D)

# Two layers are stored (indices 0 and 1); use the last (deepest) one.
LAYER_IDX = 1
TOP_K_HEADS = 3
HEAD_SELECTION_MODE = "ratio"  # "ratio" or "budget"

_SP_PATH = Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model"

# ─── Episode metadata ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeInfo:
    task_id: int
    task: str
    episode_num: int
    success: bool
    start_idx: int
    end_idx: int


def load_episode_infos(path: str | Path) -> List[EpisodeInfo]:
    with open(path) as f:
        data = json.load(f)
    return [EpisodeInfo(**d) for d in data]


def episode_for_step(step: int, episodes: List[EpisodeInfo]) -> EpisodeInfo:
    for ep in episodes:
        if ep.start_idx <= step <= ep.end_idx:
            return ep
    raise KeyError(f"Step {step} not in any episode")


def slugify(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len]


# ─── Data loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def load_record(npy_path: str) -> dict:
    p = np.load(npy_path, allow_pickle=True)
    if p.shape == () and p.dtype == object:
        p = p.item()
    return p


def step_index(path: str) -> int:
    m = re.search(r"step_(\d+)\.npy$", path)
    return int(m.group(1)) if m else -1


# ─── Attention extraction ─────────────────────────────────────────────────────

def get_attn_hs(record: dict, action_token: int = 0) -> np.ndarray:
    """Return (H, S) attention for the target layer and given action token."""
    # stored shape: (L, H, action_tokens, S)
    full_attn = np.asarray(record[FULL_ATTN_KEY])
    return full_attn[LAYER_IDX, :, action_token, :].astype(np.float32)


def get_v_skd(record: dict) -> np.ndarray:
    """Return (S, K, D) value matrix for the target layer."""
    # stored shape: (L, S, K, D)
    full_v = np.asarray(record[FULL_V_KEY])
    return full_v[LAYER_IDX].astype(np.float32)


def get_span(record: dict, key: str) -> Optional[Tuple[int, int]]:
    if key not in record:
        return None
    return tuple(int(x) for x in np.asarray(record[key]).tolist())


def get_all_spans(record: dict) -> List[Tuple[int, int]]:
    spans = []
    for cam_name in CAM_NAMES:
        s = get_span(record, f"outputs/debug/spans/image/{cam_name}")
        if s:
            spans.append(s)
    s = get_span(record, "outputs/debug/spans/task")
    if s:
        spans.append(s)
    s = get_span(record, "outputs/debug/spans/state")
    if s:
        spans.append(s)
    return spans


# ─── Head selection ───────────────────────────────────────────────────────────

def select_heads(
    attn: np.ndarray,           # (H, S)
    span: Tuple[int, int],
    all_spans: List[Tuple[int, int]],
    top_k: int = TOP_K_HEADS,
) -> np.ndarray:
    """One-vs-rest (ratio) or raw-budget head selection."""
    s0, s1 = span
    target_budget = attn[:, s0:s1].sum(axis=1)     # (H,)
    if HEAD_SELECTION_MODE == "ratio":
        other = np.zeros(attn.shape[0], dtype=np.float32)
        for sp in all_spans:
            if sp != span:
                other += attn[:, sp[0]:sp[1]].sum(axis=1)
        score = target_budget / (other + 1e-9)
    else:
        score = target_budget
    return np.argsort(score)[-min(top_k, attn.shape[0]):]


# ─── V-cosine scoring ─────────────────────────────────────────────────────────

def score_v_cosine(
    attn: np.ndarray,       # (H, S)
    v: np.ndarray,          # (S, K, D)
    top_heads: np.ndarray,
) -> np.ndarray:
    """cos(v[s,0,:], head_output_h), max over selected heads → (S,)."""
    v0 = v[:, 0, :]                                                     # (S, D)
    o = attn @ v0                                                       # (H, D)
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)    # (H, D)
    v_unit = v0 / (np.linalg.norm(v0, axis=-1, keepdims=True) + 1e-9)  # (S, D)
    cos = v_unit @ o_unit.T                                             # (S, H)
    return cos[:, top_heads].max(axis=1).astype(np.float32)


# ─── Image utilities ──────────────────────────────────────────────────────────

def as_u8_rgb(img) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
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


def jet_colormap(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def jet_color(x: float) -> Tuple[float, float, float, float]:
    x = float(np.clip(x, 0.0, 1.0))
    r = float(np.clip(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0))
    g = float(np.clip(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0))
    b = float(np.clip(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0))
    return (r, g, b, 1.0)


def resize_heatmap(
    heat01: np.ndarray,
    out_hw: Tuple[int, int],
    smoothing: str = "BILINEAR",
) -> np.ndarray:
    h, w = out_hw
    im = Image.fromarray((np.clip(heat01, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    resample = Image.BILINEAR if smoothing == "BILINEAR" else Image.NEAREST
    return np.array(im.resize((w, h), resample=resample), dtype=np.float32) / 255.0


def overlay_heatmap(
    img_u8: np.ndarray,
    heat01_up: np.ndarray,
    alpha: float = 0.30,
) -> np.ndarray:
    heat_rgb = jet_colormap(heat01_up).astype(np.float32)
    out = img_u8.astype(np.float32) * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_grid(
    overlays: List[np.ndarray],
    originals: List[np.ndarray],
    labels: Optional[List[str]] = None,
    bg: Tuple[int, int, int] = (255, 255, 255),
    gap_x: int = 16,
    gap_y: int = 16,
) -> Image.Image:
    """2-row grid: top row = heatmap overlays, bottom row = originals."""
    C = len(overlays)
    cell_h = max(im.shape[0] for im in originals)
    cell_w = max(im.shape[1] for im in originals)
    canvas_w = C * cell_w + (C + 1) * gap_x
    canvas_h = 2 * cell_h + 3 * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
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


# ─── Normalization helpers ────────────────────────────────────────────────────

def normalize_joint_minmax(heats: List[np.ndarray]) -> List[np.ndarray]:
    """Joint min-max across all heats → [0, 1]."""
    flat = np.concatenate([h.reshape(-1) for h in heats])
    mn, mx = float(flat.min()), float(flat.max())
    denom = (mx - mn) if mx > mn else 1.0
    return [(h.astype(np.float32) - mn) / denom for h in heats]


def clip_gamma_normalize_joint(
    heats: List[np.ndarray],
    clip_percentile: float = 95,
    gamma: float = 0.4,
) -> List[np.ndarray]:
    """Percentile-clip + gamma stretch, joint across all heats."""
    flat = np.concatenate([h.reshape(-1) for h in heats])
    clip_val = float(np.percentile(flat, clip_percentile))
    scale = clip_val if clip_val > 0 else 1.0
    return [(np.clip(h.astype(np.float32) / scale, 0.0, 1.0) ** gamma) for h in heats]


def log_clip_gamma_normalize_joint(
    heats: List[np.ndarray],
    clip_percentile: float = 95,
    gamma: float = 0.4,
) -> List[np.ndarray]:
    """Log1p compress → percentile-clip → joint min-max → gamma stretch."""
    total = sum(float(h.sum()) for h in heats)
    denom = total if total > 0 else 1.0
    compressed = [np.log1p(np.maximum(h.astype(np.float32) / denom, 0.0)) for h in heats]
    flat = np.concatenate([c.reshape(-1) for c in compressed])
    mn = float(flat.min())
    clip_val = float(np.percentile(flat, clip_percentile))
    scale = (clip_val - mn) if clip_val > mn else 1.0
    return [(np.clip((c - mn) / scale, 0.0, 1.0) ** gamma) for c in compressed]


# ─── Text panel ───────────────────────────────────────────────────────────────

_sp = None


def _get_sp():
    global _sp
    if _sp is None:
        import sentencepiece
        _sp = sentencepiece.SentencePieceProcessor()
        _sp.Load(str(_SP_PATH))
    return _sp


def make_task_text_panel(
    record: dict,
    task_scores: np.ndarray,        # raw scores per task-token position (200,)
    target_height_px: int,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Render a task-token heatmap panel sized to target_height_px."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    token_ids = np.asarray(record["outputs/debug/tokens/task/token_ids"])   # (200,)
    token_mask = np.asarray(record["outputs/debug/tokens/task/token_mask"])  # (200,)

    sp = _get_sp()
    real_ids = token_ids[token_mask].tolist()
    real_scores = task_scores[token_mask].astype(np.float32)

    labels = [sp.id_to_piece(tid).replace("▁", " ").strip() for tid in real_ids]

    abs_scores = np.abs(real_scores)
    mx = abs_scores.max()
    norm_scores = abs_scores / mx if mx > 1e-9 else np.zeros_like(abs_scores)

    n = max(len(real_ids), 1)
    dpi = 150
    fig_w = max(n * 0.22, 3.0)
    fig_h = 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    for i, (label, score) in enumerate(zip(labels, norm_scores)):
        r, g, b, _ = jet_color(float(score))
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=(r, g, b)))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        tc = "black" if lum > 0.45 else "white"
        ax.text(i + 0.5, 0.5, label, ha="center", va="center",
                fontsize=6, rotation=90, color=tc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_title("Task tokens", fontsize=8, pad=2)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)

    pil = Image.open(buf).convert("RGB")
    ratio = target_height_px / pil.height
    return pil.resize((max(int(pil.width * ratio), 1), target_height_px), Image.BILINEAR)
