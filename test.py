import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download

# Load tokenizer
path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model",
    gs={"token": "anon"},
)
with path.open("rb") as f:
    sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())

prompt = "Pick_up the red block"
cleaned_text = prompt.lower().strip().replace("_", " ")
task_segment = f"Task: {cleaned_text}, "

# Get immutable proto WITHOUT add_bos
proto = sp.encode(task_segment, out_type="immutable_proto")

# Build task_pieces from proto
task_pieces = [
    {
        "id": int(p.id),
        "piece": p.piece,
        "surface": p.surface,
        "begin": int(p.begin),
        "end": int(p.end),
    }
    for p in proto.pieces
]

# Manually prepend BOS metadata
bos_piece = {
    "id": int(sp.bos_id()),
    "piece": "<s>",
    "surface": "",
    "begin": 0,
    "end": 0,
}

task_pieces = [bos_piece] + task_pieces

# Print token pieces
for p in task_pieces:
    print(p["id"], p["piece"], repr(p["surface"]), p["begin"], p["end"])

task_tokens = [p["id"] for p in task_pieces]
print(task_tokens)

print("task_segment:", repr(task_segment))
print()

for i, p in enumerate(task_pieces):
    print(
        f"{i:2d} | id={p['id']:<6d} | piece={p['piece']:<12} | "
        f"surface={p['surface']!r:<15} | span=({p['begin']},{p['end']})"
    )

# Fake scores just to validate projection
token_scores = np.random.randn(len(task_pieces))

char_scores = np.zeros(len(task_segment), dtype=np.float32)

for p, score in zip(task_pieces, token_scores):
    b, e = p["begin"], p["end"]
    if e > b:
        char_scores[b:e] += score

print("\nCharacter scores:")
for i, (ch, sc) in enumerate(zip(task_segment, char_scores)):
    print(f"{i:2d} {ch!r} {sc:+.4f}")