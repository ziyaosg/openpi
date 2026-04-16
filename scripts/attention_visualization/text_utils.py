from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece
from PIL import Image

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))  # src/ → openpi

import openpi.shared.download as download  # noqa: E402
from ..attention_utils.keys import GRADCAM_KEYS  # noqa: E402
from .grid_utils import jet_color  # noqa: E402

# ============================================================
# RECORD KEYS FOR TEXT PANEL DATA
# ============================================================
TASK_SCORE_KEY       = GRADCAM_KEYS["task"]
TASK_PIECE_BEGIN_KEY = "outputs/debug/attr/task/task_piece_begin"
TASK_PIECE_END_KEY   = "outputs/debug/attr/task/task_piece_end"
TASK_PIECE_ID_KEY    = "outputs/debug/attr/task/task_piece_id"

STATE_SCORE_KEY       = GRADCAM_KEYS["state"]
STATE_PIECE_BEGIN_KEY = "outputs/debug/attr/state/state_piece_begin"
STATE_PIECE_END_KEY   = "outputs/debug/attr/state/state_piece_end"
STATE_PIECE_ID_KEY    = "outputs/debug/attr/state/state_piece_id"

# Controls how wide the text panel is relative to the grid height.
# Increase to widen the panel, decrease to narrow it.
CHAR_WIDTH = 0.08   # inches per character tile in the matplotlib figure

# ============================================================
# TOKENIZER  (loaded once at import)
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
    *,
    strip_prefix: str = "",
    strip_suffix: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand per-token scores to per-character arrays using piece spans.

    The tokenizer produces subword pieces (e.g. "basket" → ["bask", "et"]).
    Each piece maps to a character range [begin[i], end[i]) in the decoded string.
    We replicate the token's score once per character so the heatmap tiles align
    with individual characters rather than subword pieces.

    strip_prefix/strip_suffix: literal strings at the start/end of the decoded
    segment (e.g. "Task: " / ", ") to exclude from both visualization and
    normalization so that punctuation doesn't affect the color scale.

    Scores are abs-valued then normalized to [0,1] within the content region.
    """
    decoded = sp.decode(piece_ids)
    content_start = len(strip_prefix)
    content_end   = len(decoded) - len(strip_suffix) if strip_suffix else len(decoded)

    chars, scores = [], []
    for i, (b, e) in enumerate(zip(begin, end)):
        for j in range(int(b), int(e)):
            if content_start <= j < content_end:
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
    """Text panel from pre-computed attribution scores (used by gradcam script)."""
    task_chars,  task_norm  = _chars_and_norm(
        obj[TASK_SCORE_KEY][0],  obj[TASK_PIECE_BEGIN_KEY],  obj[TASK_PIECE_END_KEY],  obj[TASK_PIECE_ID_KEY].tolist(),
        strip_prefix="Task: ", strip_suffix=", ",
    )
    state_chars, state_norm = _chars_and_norm(
        obj[STATE_SCORE_KEY][0], obj[STATE_PIECE_BEGIN_KEY], obj[STATE_PIECE_END_KEY], obj[STATE_PIECE_ID_KEY].tolist(),
        strip_prefix="State: ", strip_suffix=";\n",
    )
    return _build_text_panel_image(task_chars, task_norm, state_chars, state_norm, target_height_px)


def render_text_panel_from_token_scores(
    obj: dict,
    task_token_scores: np.ndarray,
    state_token_scores: np.ndarray,
    target_height_px: int,
) -> Image.Image:
    """Text panel from raw per-token attention scores (used by raw_weights script)."""
    task_chars,  task_norm  = _chars_and_norm(
        task_token_scores,  obj[TASK_PIECE_BEGIN_KEY],  obj[TASK_PIECE_END_KEY],  obj[TASK_PIECE_ID_KEY].tolist(),
        strip_prefix="Task: ", strip_suffix=", ",
    )
    state_chars, state_norm = _chars_and_norm(
        state_token_scores, obj[STATE_PIECE_BEGIN_KEY], obj[STATE_PIECE_END_KEY], obj[STATE_PIECE_ID_KEY].tolist(),
        strip_prefix="State: ", strip_suffix=";\n",
    )
    return _build_text_panel_image(task_chars, task_norm, state_chars, state_norm, target_height_px)
