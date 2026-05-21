from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))  # src/ → openpi

import openpi.shared.download as download  # noqa: E402
from ..attention_utils.keys import GRADCAM_KEYS  # noqa: E402
from .grid_utils import jet_color  # noqa: E402

# ============================================================
# RECORD KEYS FOR TEXT PANEL DATA
# ============================================================
TASK_SCORE_KEY       = GRADCAM_KEYS["task"]
TASK_PIECE_BEGIN_KEY = "outputs/debug/tokens/task/piece_begin"
TASK_PIECE_END_KEY   = "outputs/debug/tokens/task/piece_end"
TASK_PIECE_ID_KEY    = "outputs/debug/tokens/task/piece_id"

STATE_SCORE_KEY       = GRADCAM_KEYS["state"]
STATE_PIECE_BEGIN_KEY = "outputs/debug/tokens/state/piece_begin"
STATE_PIECE_END_KEY   = "outputs/debug/tokens/state/piece_end"
STATE_PIECE_ID_KEY    = "outputs/debug/tokens/state/piece_id"

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

    token_scores may cover ALL pieces (len == len(begin)) or only the content
    (digit) pieces (len < len(begin)).  When digit-only, an offset k is computed
    from the strip_prefix length so that score[i - k] is used for content piece i.

    Scores are abs-valued then normalized to [0,1] within the content region.
    """
    decoded = sp.decode(piece_ids)
    content_start = len(strip_prefix)
    content_end   = len(decoded) - len(strip_suffix) if strip_suffix else len(decoded)

    n_pieces = len(begin)
    n_scores = len(token_scores)

    # Determine piece offset k: if scores are digit-only, index as score[i - k].
    # k = number of prefix-only pieces (entirely before content_start).
    if n_scores < n_pieces:
        k = int(np.sum(np.asarray(end) <= content_start))
    else:
        k = 0  # scores cover all pieces; use i directly

    chars, scores = [], []
    for i, (b, e) in enumerate(zip(begin, end)):
        for j in range(int(b), int(e)):
            if content_start <= j < content_end:
                chars.append(decoded[j])
                idx = i - k
                if 0 <= idx < n_scores:
                    scores.append(float(token_scores[idx]))
                else:
                    scores.append(0.0)
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




def get_token_labels(
    rec: dict,
    modality: str,
    strip_prefix: str = "",
    strip_suffix: str = "",
) -> Tuple[List[str], List[int]]:
    """Content token labels and their original indices for 'task' or 'state'.

    Tokens that fall entirely within strip_prefix or strip_suffix are excluded.
    Boundary tokens are clipped to the content region so the leading/trailing
    whitespace that belongs to the prefix or suffix is not shown.
    Pure-whitespace tokens (e.g. separators between state values) get an empty
    label so they show as a coloured tile with no text — no '□' placeholder.

    Returns
    -------
    labels  : one string per content token (empty string for whitespace tokens)
    indices : original 0-based token indices into the FULL piece array.
              When the score array covers ALL tokens (len == n_pieces), use
              score[indices[j]] directly.  When the score array is digit-only
              (len < n_pieces), use score[indices[j] - k] where k is the number
              of prefix-only pieces — equivalently, k = indices[0].
    """
    piece_ids   = rec[f"outputs/debug/tokens/{modality}/piece_id"]
    piece_begin = rec[f"outputs/debug/tokens/{modality}/piece_begin"]
    piece_end   = rec[f"outputs/debug/tokens/{modality}/piece_end"]

    decoded       = sp.decode(piece_ids.tolist())
    content_start = len(strip_prefix)
    content_end   = len(decoded) - len(strip_suffix) if strip_suffix else len(decoded)

    labels: List[str]  = []
    indices: List[int] = []
    for i, (b, e) in enumerate(zip(piece_begin, piece_end)):
        b, e = int(b), int(e)
        if e <= content_start or b >= content_end:
            continue                          # entirely in prefix / suffix
        text = decoded[max(b, content_start) : min(e, content_end)]
        labels.append(text.strip())           # "" for space tokens — no text on tile
        indices.append(i)

    return labels, indices


def render_token_strip_as_image(
    norm_scores: np.ndarray,
    labels: List[str],
    strip_h: int,
    tile_w: int,
    font: ImageFont.ImageFont,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Horizontal 1-D heatmap: tokens arranged left to right.

    norm_scores — (N,) float32 in [0, 1]; maps directly to jet colour.
    labels      — one short string per token (from get_token_labels).
    strip_h     — height of the returned image (= row height).
    tile_w      — pixel width allocated to each token tile.

    Each token occupies a tile_w × strip_h coloured rectangle.
    Text is rotated 90° counterclockwise (reads bottom-to-top) so the word
    length runs along the tall axis of the tile, allowing large readable fonts.
    Characters are dropped from the right only if the rotated label is taller
    than strip_h - 4 px.  Text colour (black / white) is chosen for contrast.
    """
    N       = len(norm_scores)
    strip_w = N * tile_w
    # Use RGBA so rotated text can be alpha-composited without a coloured halo.
    img  = Image.new("RGBA", (strip_w, strip_h), (*bg, 255))
    draw = ImageDraw.Draw(img)

    for i, (score, label) in enumerate(zip(norm_scores, labels)):
        x0 = i * tile_w
        x1 = (i + 1) * tile_w - 1

        r, g, b, _ = jet_color(float(score))
        fill_rgb = (int(r * 255), int(g * 255), int(b * 255))
        draw.rectangle([x0, 0, x1, strip_h - 1], fill=(*fill_rgb, 255))

        if not label:
            continue

        # Measure and truncate so the rotated label height fits in strip_h.
        display = label
        bbox    = draw.textbbox((0, 0), display, font=font)
        tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
        while len(display) > 1 and tw > strip_h - 4:
            display = display[:-1]
            bbox    = draw.textbbox((0, 0), display, font=font)
            tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]

        lum        = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0, 255) if lum > 0.45 else (255, 255, 255, 255)

        # Render onto a transparent layer, rotate 90° CCW (bottom-to-top).
        # Draw at (-bbox[0], -bbox[1]) so the glyph is flush with the layer edges,
        # avoiding clipping when bbox offsets are non-zero.
        text_layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        ImageDraw.Draw(text_layer).text((-bbox[0], -bbox[1]), display, fill=text_color, font=font)
        rotated = text_layer.rotate(90, expand=True)  # size becomes (th, tw)

        # Centre rotated text in the tile (rotated.width=th, rotated.height=tw).
        rx = x0 + max((tile_w - rotated.width)  // 2, 0)
        ry =      max((strip_h - rotated.height) // 2, 0)
        img.alpha_composite(rotated, (rx, ry))

    return img.convert("RGB")


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
