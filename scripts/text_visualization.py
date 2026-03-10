import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
from pathlib import Path
from typing import List, Tuple
from numpy.ma import ids
import sentencepiece
import openpi.shared.download as download

INPUT_DIR = "/data/ann/policy_records_20260309_215220"
OUTPUT_DIR = "/data/ann/policy_records_20260309_215220/task_heatmaps"

TASK_KEYS = [
    "outputs/debug/attr/task/scores",
    "outputs/debug/attr/task/task_piece_begin",
    "outputs/debug/attr/task/task_piece_end",
    "outputs/debug/attr/task/task_piece_id"
]

# Load tokenizer
path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model",
    gs={"token": "anon"},
)
with path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

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

def step_index(path: str) -> int:
    """
    Extract step number from a filename like 'step_0.npy'.
    Used to sort and to name output PNGs.
    """
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def process_one(npy_path: str, out_png: str) -> None:
    """
    Load one record, produce one task heatmap, save as PNG.
    """

    obj = load_record(npy_path)

    task_scores = obj[TASK_KEYS[0]]
    task_piece_begin = obj[TASK_KEYS[1]]
    task_piece_end = obj[TASK_KEYS[2]]
    task_piece_id = obj[TASK_KEYS[3]]

    # print("task_scores:", task_scores.shape, task_scores)
    # print("task_piece_begin:", task_piece_begin.shape, task_piece_begin)
    # print("task_piece_end:", task_piece_end.shape, task_piece_end)
    # print("task_piece_id:", task_piece_id.shape, task_piece_id)

    ids = task_piece_id.tolist()
    scores = task_scores.tolist()[0]

    # Decode to text
    decoded = sp.decode(ids)

    # print("Decoded text:")
    # print(decoded)

    task_characters = []
    task_characters_scores = []

    for i, (b, e) in enumerate(zip(task_piece_begin, task_piece_end)):
        # print(int(b), int(e))
        for j in range(int(b), int(e)):
            task_characters.append(decoded[j])
            task_characters_scores.append(scores[i])

    # print(task_characters)
    # print(task_characters_scores)

    chars = np.array(task_characters)
    scores = np.array(task_characters_scores)

    values = np.abs(scores)
    norm = values / values.max()

    fig, ax = plt.subplots(figsize=(len(chars)*0.3, 2))

    cmap = plt.cm.YlOrRd  # lighter color palette

    for i, (c, v) in enumerate(zip(chars, norm)):
        # compress color range so it stays light
        v = 0.2 + 0.6 * v
        color = cmap(v)

        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, 0.5, c,
                ha="center",
                va="center",
                fontsize=10,
                color="black")

    ax.set_xlim(0, len(chars))
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

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