#!/usr/bin/env python3
"""
patch_state_features.py — Recompute gc_gini_state, ras_gini_state, agree_gc_ras_state
in the v2 feature cache using the corrected digit-only state indexing.

Only pi0fast groups are affected (pi05 has no state tokens in current recordings).
The patch loads each cached episode, locates the original .npy file by abs_step,
recomputes the 3 state features, and saves the updated cache atomically.

Usage:
  python scripts/attention_analysis/patch_state_features.py
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

BASE       = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
CACHE_DIR  = BASE / "analysis_conformal_v2" / "feature_cache"

# Only pi0fast groups have state tokens; pi05 leaves state features as NaN
PATCH_GROUPS: Dict[str, List[str]] = {
    "pi0fast_libero10": [
        "policy_records_pi0_fast_libero_20260511_040629",
        "policy_records_pi0_fast_libero_20260513_021106",
    ],
    "pi0fast_liberoplus": [
        "policy_records_pi0_fast_libero_20260516_034939",
        "policy_records_pi0_fast_libero_20260516_034940",
    ],
}


# ─── Digit-only slice helper ──────────────────────────────────────────────────

def _state_digit_slice(rec: dict) -> Tuple[int, Optional[int]]:
    pb = rec.get("outputs/debug/tokens/state/piece_begin")
    pe = rec.get("outputs/debug/tokens/state/piece_end")
    if pb is None or pe is None:
        return 0, None
    piece_begin = np.asarray(pb)
    piece_end   = np.asarray(pe)
    max_byte      = int(piece_end.max())
    CONTENT_START = 7
    CONTENT_END   = max_byte - 2
    k = int((piece_end   <= CONTENT_START).sum())
    s = int((piece_begin >= CONTENT_END).sum())
    return k, len(piece_begin) - s


def _gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v)).astype(np.float64)
    n = len(a); s = float(a.sum())
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n) if s > 0 and n > 0 else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def recompute_state_features(npy_path: Path) -> Optional[Dict[str, float]]:
    """Return corrected state features for one step, or None if not applicable."""
    try:
        rec = np.load(npy_path, allow_pickle=True).item()
    except Exception:
        return None

    k_gc  = "outputs/debug/gradcam/action_tokens/0/state"
    k_ras = "outputs/debug/raw_alpha/summation/action_tokens/0/state"
    if k_gc not in rec or k_ras not in rec:
        return None

    dk, dend = _state_digit_slice(rec)
    gc_state_list, ras_state_list = [], []
    for i in range(5):
        kg = f"outputs/debug/gradcam/action_tokens/{i}/state"
        kr = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/state"
        if kg not in rec:
            break
        gc_state_list.append(np.abs(np.asarray(rec[kg])[0]).astype(np.float32))
        ras_state_list.append(np.abs(np.asarray(rec[kr])[0]).astype(np.float32))

    if not gc_state_list:
        return None

    gc  = np.mean(np.stack(gc_state_list),  axis=0)[dk:dend]
    ras = np.mean(np.stack(ras_state_list), axis=0)[dk:dend]

    return {
        "gc_gini_state":      _gini(gc),
        "ras_gini_state":     _gini(ras),
        "agree_gc_ras_state": _cosine(gc, ras),
    }


def _atomic_save(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(data, f, protocol=4)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def patch_group(group_name: str, run_names: List[str]) -> None:
    cache_path = CACHE_DIR / f"{group_name}.pkl"
    if not cache_path.exists():
        print(f"  [skip] {group_name}: cache not found")
        return

    print(f"\nPatching {group_name} ...")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    # Build abs_step → npy_path index across all runs
    step_to_path: Dict[int, Path] = {}
    for run_name in run_names:
        data_dir = BASE / run_name
        for npy in data_dir.glob("step_*.npy"):
            try:
                s = int(npy.stem.split("_")[1])
                step_to_path[s] = npy
            except ValueError:
                pass
    print(f"  indexed {len(step_to_path)} step files across {len(run_names)} runs")

    n_updated = 0
    n_missing = 0
    for label in ("success", "failure"):
        for ep in tqdm(data[label], desc=f"  {label}", ncols=80):
            for step_dict in ep:
                abs_step = step_dict.get("abs_step")
                if abs_step is None:
                    continue
                npy_path = step_to_path.get(int(abs_step))
                if npy_path is None:
                    n_missing += 1
                    continue
                corrected = recompute_state_features(npy_path)
                if corrected is None:
                    continue
                step_dict.update(corrected)
                n_updated += 1

    print(f"  updated {n_updated} step dicts, {n_missing} abs_steps not found in index")
    _atomic_save(data, cache_path)
    print(f"  saved → {cache_path}")


def main() -> None:
    for group_name, run_names in PATCH_GROUPS.items():
        patch_group(group_name, run_names)
    print("\nDone.")


if __name__ == "__main__":
    main()
