#!/usr/bin/env python3
"""
load_features.py — Extract and cache features for all policy-record groups.

Usage:
    python scripts/attention_analysis/load_features.py [--force]

    --force   re-extract even if a cache file already exists (use_cache=False)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.attention_analysis.features import load_group

# ── Storage roots ─────────────────────────────────────────────────────────────
SCRATCH  = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
PROJECT  = Path("/nfs/roberts/project/pi_tkf6/zs377")
CACHE_DIR = PROJECT / "feature_cache"

# ── Group definitions ─────────────────────────────────────────────────────────
# Each entry: (group_name, base_dir, [run_names])
# Note: pi0fast_liberoplus dirs on disk use "pi0fast_liberoplus" (no extra underscore),
#       not the "pi0_fast_libero" pattern used in the other libero10 groups.
GROUPS = [
    (
        "pi05_libero10",
        SCRATCH,
        [
            "policy_records_pi05_libero_20260512_182606",
            "policy_records_pi05_libero_20260512_201032",
        ],
    ),
    (
        "pi0fast_libero10",
        SCRATCH,
        [
            "policy_records_pi0_fast_libero_20260511_040629",
            "policy_records_pi0_fast_libero_20260513_021106",
        ],
    ),
    (
        "pi0fast_liberoplus",
        SCRATCH,
        [
            "policy_records_pi0fast_liberoplus_20260516_034939",
            "policy_records_pi0fast_liberoplus_20260516_034940",
        ],
    ),
    (
        "pi05_liberoplus",
        PROJECT,
        [
            "policy_records_pi05_liberoplus_20260522_032743",
            "policy_records_pi05_liberoplus_20260522_031717",
            "policy_records_pi05_liberoplus_20260524_163155",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="ignore existing cache and re-extract")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    use_cache = not args.force

    all_data: dict = {}
    t0_total = time.time()

    for group_name, base_dir, run_names in GROUPS:
        print(f"\n{'='*60}")
        print(f"Group: {group_name}  (base: {base_dir})")
        t0 = time.time()
        ep_data = load_group(
            group_name=group_name,
            run_names=run_names,
            base_dir=base_dir,
            cache_dir=CACHE_DIR,
            max_steps=60,
            use_cache=use_cache,
        )
        elapsed = time.time() - t0
        n_s = len(ep_data.get("success", []))
        n_f = len(ep_data.get("failure", []))
        print(f"  done in {elapsed:.1f}s — {n_s} success / {n_f} failure episodes")
        all_data[group_name] = ep_data

    total = time.time() - t0_total
    print(f"\n{'='*60}")
    print(f"All groups loaded in {total:.1f}s. Cache: {CACHE_DIR}")
    print("\nSummary:")
    for gname, ep_data in all_data.items():
        n_s = len(ep_data.get("success", []))
        n_f = len(ep_data.get("failure", []))
        # feature count from first available episode step
        n_feat = 0
        for eps in ep_data.values():
            for ep in eps:
                if ep:
                    n_feat = len([k for k in ep[0] if k not in ("rel_step", "abs_step")])
                    break
            if n_feat:
                break
        print(f"  {gname:25s}: {n_s:3d} success  {n_f:3d} failure  {n_feat} features/step")


if __name__ == "__main__":
    main()
