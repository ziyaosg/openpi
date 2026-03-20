#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import sentencepiece
import openpi.shared.download as download

INPUT_DIR = "/data/ann/policy_records_20260317_194622"
OUTPUT_DIR = "/data/ann/policy_records_20260317_194622/task_heatmaps"
EPISODE_SUMMARIES_JSON = "/home/annsong/Desktop/openpi/libero_videos/episode_summaries.json"

TASK_KEYS = [
    "outputs/debug/attr/task/scores",
    "outputs/debug/attr/task/task_piece_begin",
    "outputs/debug/attr/task/task_piece_end",
    "outputs/debug/attr/task/task_piece_id",
]


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
    """
    Find the episode whose [start_idx, end_idx] contains step.
    Assumes end_idx is inclusive.
    """
    for ep in episodes:
        if ep.start_idx <= step <= ep.end_idx:
            return ep
    raise KeyError(f"Step {step} not covered by any episode range in {EPISODE_SUMMARIES_JSON}")


def slugify(s: str, max_len: int = 120) -> str:
    """
    Folder-safe slug:
      - lowercase
      - replace non-alnum with underscores
      - collapse repeats
      - trim length
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model",
    gs={"token": "anon"},
)
with path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())


def jet_color(x: float):
    """
    Match the lightweight Jet-like colormap from the image heatmap script.
    Input x should be in [0, 1].
    Returns an (r, g, b, a) tuple for matplotlib.
    """
    x = float(np.clip(x, 0.0, 1.0))
    r = np.clip(1.5 - abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - abs(4.0 * x - 1.0), 0.0, 1.0)
    return (r, g, b, 1.0)


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

    token_ids = task_piece_id.tolist()
    piece_scores = task_scores.tolist()[0]

    # Decode to text
    decoded = sp.decode(token_ids)

    task_characters = []
    task_character_scores = []

    for i, (b, e) in enumerate(zip(task_piece_begin, task_piece_end)):
        for j in range(int(b), int(e)):
            task_characters.append(decoded[j])
            task_character_scores.append(piece_scores[i])

    chars = np.array(task_characters)
    scores = np.array(task_character_scores, dtype=np.float32)

    values = np.abs(scores)
    maxv = float(values.max()) if values.size > 0 else 0.0
    norm = values / maxv if maxv > 0 else np.zeros_like(values)

    fig, ax = plt.subplots(figsize=(max(len(chars) * 0.3, 1), 2))

    for i, (c, v) in enumerate(zip(chars, norm)):
        color = jet_color(v)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(
            i + 0.5,
            0.5,
            c,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    ax.set_xlim(0, len(chars))
    ax.set_ylim(0, 1)
    ax.axis("off")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Iterate over all step_*.npy files and write a corresponding PNG for each,
    grouped into nested folders by task and episode.
    """
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