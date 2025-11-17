# src/openpi/models/utils/decode.py
import numpy as np

def group_tokens_truncate(tokens, group=7):
    """Drop leftover IDs so reshape never crashes."""
    t = np.asarray(tokens).ravel()
    n = t.size // group
    return t[: n * group].reshape(n, group)

def group_tokens_pad(tokens, group=7, pad_id=0):
    """Pad with pad_id so reshape never crashes."""
    t = np.asarray(tokens).ravel()
    rem = (-t.size) % group  # how many to pad
    if rem:
        t = np.concatenate([t, np.full(rem, pad_id, dtype=t.dtype)])
    return t.reshape(-1, group)
