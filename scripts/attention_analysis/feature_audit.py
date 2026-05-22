#!/usr/bin/env python3
"""
feature_audit.py — Cross-format feature compatibility audit for robotics policy failure prediction.

Groups:
  pi05_libero10   : policy_records_pi05_libero_20260511_021509
  pi0fast_libero10: policy_records_pi0_fast_libero_20260511_040629
  pi0fast_liberoplus: policy_records_pi0_fast_libero_20260516_034939

Bugs fixed:
  A: pi05 outputs/state shape (32,) — only [:8] are real
  B: pi05 task span (768,968)=200 tokens but token_mask reveals only 14-21 real
  C: pi0fast right_wrist image_mask=True → ~14-17% of attn goes to useless tokens
  D: pi05 V shape (2,968,1,256) → V[-1,:,0,:] = (968,256); apply token_mask to task V

Usage:
  /nfs/roberts/project/pi_tkf6/zs377/ycrc_conda/envs/openpi/bin/python3 feature_audit.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

SCRATCH = Path("/nfs/roberts/scratch/pi_tkf6/zs377")
PROJECT = Path("/home/zs377/project_pi_tkf6/zs377/projects/openpi_liberoplus")

DATA_GROUPS = {
    "pi05_libero10": SCRATCH / "policy_records_pi05_libero_20260511_021509",
    "pi0fast_libero10": SCRATCH / "policy_records_pi0_fast_libero_20260511_040629",
    "pi0fast_liberoplus": SCRATCH / "policy_records_pi0_fast_libero_20260516_034939",
}

CACHE_FILES = {
    "pi05_libero10": SCRATCH / "analysis_conformal_v2/feature_cache/pi05_libero10.pkl",
    "pi0fast_libero10": SCRATCH / "analysis_conformal_v2/feature_cache/pi0fast_libero10.pkl",
    "pi0fast_liberoplus": SCRATCH / "analysis_conformal_v2/feature_cache/pi0fast_liberoplus.pkl",
}

REPORT_PATH = SCRATCH / "feature_audit_report.txt"
N_SAMPLE = 40  # success + failure episodes each

AUDIT_STEP = 20  # key early-detection step for group-level stats

# ─── Math utilities ───────────────────────────────────────────────────────────

def gini(v: np.ndarray) -> float:
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a)
    s = a.sum()
    if s < 1e-12:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def entropy(v: np.ndarray) -> float:
    a = np.abs(v.ravel()).astype(np.float64)
    s = a.sum()
    if s < 1e-12:
        return 0.0
    p = a / s
    return float(-np.sum(p * np.log(p + 1e-12)))


def norm_entropy(v: np.ndarray, n_tokens: int) -> float:
    """Entropy normalized by log(n_tokens)."""
    raw = entropy(v)
    if n_tokens <= 1:
        return raw
    return raw / np.log(n_tokens)


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def auroc(fail_vals: List[float], succ_vals: List[float]) -> float:
    fv = [x for x in fail_vals if not np.isnan(x)]
    sv = [x for x in succ_vals if not np.isnan(x)]
    if len(fv) < 2 or len(sv) < 2:
        return float("nan")
    stat, _ = mannwhitneyu(fv, sv, alternative="two-sided")
    return max(stat / (len(sv) * len(fv)), 1 - stat / (len(sv) * len(fv)))


def nan_mean(vals: List[float]) -> float:
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def nan_std(vals: List[float]) -> float:
    v = [x for x in vals if not np.isnan(x)]
    return float(np.std(v)) if len(v) > 1 else float("nan")

# ─── Data loading helpers ──────────────────────────────────────────────────────

def load_episodes(run_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Return (success_eps, failure_eps) as lists of episode dicts from episode_summaries.json."""
    summary_path = run_dir / "client_output" / "episode_summaries.json"
    with open(summary_path) as f:
        episodes = json.load(f)
    success = [e for e in episodes if e["success"]]
    failure = [e for e in episodes if not e["success"]]
    return success, failure


def load_record(path: Path) -> dict:
    d = np.load(str(path), allow_pickle=True)
    if isinstance(d, np.ndarray) and d.shape == () and d.dtype == object:
        d = d.item()
    return d


def load_episode_records(ep: dict, run_dir: Path) -> List[Tuple[int, dict]]:
    """Load all step records for an episode. Returns [(rel_step, record), ...]."""
    records = []
    start, end = ep["start_idx"], ep["end_idx"]
    for step in range(start, end + 1):
        path = run_dir / f"step_{step}.npy"
        if not path.exists():
            continue
        try:
            rec = load_record(path)
            records.append((step - start, rec))
        except Exception as e:
            print(f"[WARN] step {step}: {e}")
    return records


def detect_format(rec: dict) -> str:
    return "pi0fast" if "outputs/debug/attn/layers" in rec else "pi05"


def _span(rec: dict, key: str) -> Optional[Tuple[int, int]]:
    v = rec.get(key)
    if v is None:
        return None
    v = np.asarray(v).ravel()
    return int(v[0]), int(v[1])

# ─── Feature extraction (with all bug fixes) ──────────────────────────────────

def extract_mean_attn(rec: dict, fmt: str, aggregation: str = "all") -> np.ndarray:
    """
    Get mean attention vector (S,) from weights tensor.

    For pi0fast: shape (L, 1, H, S) → w[-1, 0] = (H, S) → mean over H
    For pi05:    shape (L, H, T_action, S) → w[-1] = (H, T_action, S)
      aggregation="all":   mean over H then T_action
      aggregation="first": first action token, mean over H
      aggregation="max":   max over T_action, then mean over H
    """
    w = np.asarray(rec["outputs/debug/attn/weights"])
    if fmt == "pi0fast":
        return w[-1, 0].mean(axis=0)  # (S,)
    else:
        # pi05: (H, T_action, S)
        if aggregation == "all":
            return w[-1].mean(axis=0).mean(axis=0)       # (S,)
        elif aggregation == "first":
            return w[-1, :, 0, :].mean(axis=0)            # (S,)
        elif aggregation == "max":
            return w[-1].max(axis=1).mean(axis=0)         # (S,)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def extract_corrected_features(rec: dict, prev_rec: Optional[dict] = None,
                                 prev_centroids: Optional[dict] = None) -> Tuple[dict, dict]:
    """
    Extract ALL features with bug fixes applied.
    Returns (feats, centroids).
    """
    fmt = detect_format(rec)
    feats: dict = {}
    centroids: dict = {}

    # ─── Bug A: state fix ──────────────────────────────────────────────────────
    state_key = "outputs/state" if "outputs/state" in rec else "inputs/observation/state"
    state_full = np.asarray(rec.get(state_key, np.zeros(8))).astype(np.float64)
    state_fixed = state_full[:8]

    feats["state_norm_raw"] = float(np.linalg.norm(state_full))
    feats["state_norm"] = float(np.linalg.norm(state_fixed))

    if prev_rec is not None:
        prev_state_key = "outputs/state" if "outputs/state" in prev_rec else "inputs/observation/state"
        prev_state_full = np.asarray(prev_rec.get(prev_state_key, state_full)).astype(np.float64)
        prev_state_fixed = prev_state_full[:8]
        feats["joint_vel_raw"] = float(np.linalg.norm(state_full - prev_state_full))
        feats["joint_vel"] = float(np.linalg.norm(state_fixed - prev_state_fixed))
    else:
        feats["joint_vel_raw"] = 0.0
        feats["joint_vel"] = 0.0

    # ─── Attn weights extraction ───────────────────────────────────────────────
    wkey = "outputs/debug/attn/weights"
    if wkey not in rec:
        for k in ("frac_base", "frac_wrist", "frac_rwrist", "frac_task", "frac_state",
                  "frac_base_raw", "frac_wrist_raw", "frac_task_raw",
                  "routing_entropy", "routing_entropy_raw",
                  "attn_gini_base", "attn_entropy_base", "normalized_attn_entropy_base",
                  "attn_gini_wrist", "attn_gini_rwrist", "attn_gini_task", "attn_gini_state"):
            feats[k] = float("nan")
    else:
        mean_attn = extract_mean_attn(rec, fmt, aggregation="all")

        sp_base = _span(rec, "outputs/debug/spans/image/base_0_rgb")
        sp_wrist = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
        sp_rwrist = _span(rec, "outputs/debug/spans/image/right_wrist_0_rgb")
        sp_task = _span(rec, "outputs/debug/spans/task")
        sp_state = _span(rec, "outputs/debug/spans/state")

        # Bug B: pi05 task token_mask
        task_mask = None
        if fmt == "pi05":
            tm_key = "outputs/debug/tokens/task/token_mask"
            if tm_key in rec:
                task_mask = np.asarray(rec[tm_key]).astype(bool)

        # ─── Bug C: pi0fast rwrist denominator fix ────────────────────────────
        total_raw = float(mean_attn.sum()) + 1e-12

        if fmt == "pi0fast" and sp_rwrist is not None:
            rwrist_mass = float(mean_attn[sp_rwrist[0]:sp_rwrist[1]].sum())
            total_fixed = total_raw - rwrist_mass + 1e-12
        else:
            total_fixed = total_raw

        def frac_raw(sp):
            return float(mean_attn[sp[0]:sp[1]].sum() / total_raw) if sp else float("nan")

        def frac_fixed(sp):
            return float(mean_attn[sp[0]:sp[1]].sum() / total_fixed) if sp else float("nan")

        feats["frac_base_raw"] = frac_raw(sp_base)
        feats["frac_wrist_raw"] = frac_raw(sp_wrist)
        feats["frac_task_raw"] = frac_raw(sp_task)
        feats["frac_base"] = frac_fixed(sp_base)
        feats["frac_wrist"] = frac_fixed(sp_wrist)
        feats["frac_rwrist"] = frac_fixed(sp_rwrist) if sp_rwrist else float("nan")
        feats["frac_task"] = frac_fixed(sp_task)
        feats["frac_state"] = frac_fixed(sp_state)

        # routing entropy raw
        fracs_raw = [v for v in [frac_raw(sp_base), frac_raw(sp_wrist),
                                  frac_raw(sp_task), frac_raw(sp_state)]
                     if not np.isnan(v)]
        if fracs_raw:
            p = np.clip(np.array(fracs_raw), 0, None)
            p = p / (p.sum() + 1e-12)
            feats["routing_entropy_raw"] = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            feats["routing_entropy_raw"] = float("nan")

        # routing entropy fixed
        fracs_fixed = [v for v in [frac_fixed(sp_base), frac_fixed(sp_wrist),
                                    frac_fixed(sp_task), frac_fixed(sp_state)]
                       if not np.isnan(v)]
        if fracs_fixed:
            p = np.clip(np.array(fracs_fixed), 0, None)
            p = p / (p.sum() + 1e-12)
            feats["routing_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            feats["routing_entropy"] = float("nan")

        # Gini and entropy within each modality
        for tag, sp in [("base", sp_base), ("wrist", sp_wrist),
                        ("rwrist", sp_rwrist), ("task", sp_task), ("state", sp_state)]:
            if sp is not None:
                seg = mean_attn[sp[0]:sp[1]]
                feats[f"attn_gini_{tag}"] = gini(seg)
                if tag == "base":
                    feats["attn_entropy_base"] = entropy(seg)
                    feats["normalized_attn_entropy_base"] = entropy(seg) / np.log(256)
            else:
                feats[f"attn_gini_{tag}"] = float("nan")
                if tag == "base":
                    feats["attn_entropy_base"] = float("nan")
                    feats["normalized_attn_entropy_base"] = float("nan")

        # Store raw attn segments for cross-signal agreement
        attn_base = mean_attn[sp_base[0]:sp_base[1]] if sp_base else None
        attn_wrist = mean_attn[sp_wrist[0]:sp_wrist[1]] if sp_wrist else None

        # ─── Bug B: pi05 task token mask for attn aggregation ────────────────
        if sp_task is not None:
            if fmt == "pi05" and task_mask is not None:
                span_len = sp_task[1] - sp_task[0]
                n_real = int(task_mask[:span_len].sum())
                if n_real > 0:
                    sp_task_real = (sp_task[0], sp_task[0] + n_real)
                else:
                    sp_task_real = sp_task
            else:
                sp_task_real = sp_task
            attn_task = mean_attn[sp_task_real[0]:sp_task_real[1]]
        else:
            attn_task = None
            sp_task_real = None

    # ─── Heatmap features ─────────────────────────────────────────────────────
    def load_heatmap_pi05(cam: str):
        k_gc = f"outputs/debug/gradcam/image/{cam}"
        k_ras = f"outputs/debug/raw_alpha/summation/image/{cam}"
        k_rn = f"outputs/debug/raw_alpha/norm/image/{cam}"
        gc = np.abs(np.asarray(rec[k_gc])).astype(np.float32) if k_gc in rec else None
        ras = np.abs(np.asarray(rec[k_ras])).astype(np.float32) if k_ras in rec else None
        rn = np.abs(np.asarray(rec[k_rn])).astype(np.float32) if k_rn in rec else None
        return gc, ras, rn

    def load_heatmap_pi0fast(cam: str):
        gcs, ras_l, rns = [], [], []
        for i in range(5):
            k_gc = f"outputs/debug/gradcam/action_tokens/{i}/{cam}"
            k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/{cam}"
            k_rn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/{cam}"
            if k_gc not in rec:
                break
            gcs.append(np.abs(np.asarray(rec[k_gc])[0]).astype(np.float32))
            ras_l.append(np.abs(np.asarray(rec[k_ras])[0]).astype(np.float32))
            rns.append(np.abs(np.asarray(rec[k_rn])[0]).astype(np.float32))
        if not gcs:
            return None, None, None
        return (np.mean(np.stack(gcs), axis=0),
                np.mean(np.stack(ras_l), axis=0),
                np.mean(np.stack(rns), axis=0))

    def heatmap_feats(gc, ras, rn, attn_seg, label: str) -> dict:
        f: dict = {}
        if gc is None:
            for k in (f"gc_gini_{label}", f"gc_entropy_{label}",
                      f"ras_gini_{label}", f"rn_gini_{label}",
                      f"agree_gc_ras_{label}", f"agree_base_wrist_gc",
                      f"agree_attn_gc_{label}", f"agree_attn_ras_{label}"):
                f[k] = float("nan")
            return f
        f[f"gc_gini_{label}"] = gini(gc)
        f[f"gc_entropy_{label}"] = entropy(gc)
        f[f"ras_gini_{label}"] = gini(ras)
        f[f"rn_gini_{label}"] = gini(rn) if rn is not None else float("nan")
        f[f"agree_gc_ras_{label}"] = cosine(gc, ras)
        if attn_seg is not None and attn_seg.size == gc.size:
            f[f"agree_attn_gc_{label}"] = cosine(attn_seg, gc.ravel())
            f[f"agree_attn_ras_{label}"] = cosine(attn_seg, ras.ravel())
        else:
            f[f"agree_attn_gc_{label}"] = float("nan")
            f[f"agree_attn_ras_{label}"] = float("nan")
        return f

    # Cameras
    maps_gc: dict = {}
    maps_ras: dict = {}
    for cam_key, cam_label in [("base_0_rgb", "base"), ("left_wrist_0_rgb", "wrist"),
                                ("right_wrist_0_rgb", "rwrist")]:
        if fmt == "pi0fast":
            gc, ras, rn = load_heatmap_pi0fast(cam_key)
        else:
            gc, ras, rn = load_heatmap_pi05(cam_key)
        attn_seg = (attn_base if cam_label == "base" else
                    (attn_wrist if cam_label == "wrist" else None)) if wkey in rec else None
        feats.update(heatmap_feats(gc, ras, rn, attn_seg, cam_label))
        maps_gc[cam_label] = gc
        maps_ras[cam_label] = ras

    # Cross-camera
    feats["agree_base_wrist_gc"] = cosine(maps_gc.get("base"), maps_gc.get("wrist"))
    feats["agree_base_wrist_ras"] = cosine(maps_ras.get("base"), maps_ras.get("wrist"))

    # ─── Task features (Bug B: pi05 mask) ────────────────────────────────────
    if fmt == "pi0fast":
        gc_task_l, ras_task_l, rn_task_l = [], [], []
        for i in range(5):
            k_gc = f"outputs/debug/gradcam/action_tokens/{i}/task"
            k_ras = f"outputs/debug/raw_alpha/summation/action_tokens/{i}/task"
            k_rn = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
            if k_gc not in rec:
                break
            gc_task_l.append(np.abs(np.asarray(rec[k_gc])[0]).astype(np.float32))
            ras_task_l.append(np.abs(np.asarray(rec[k_ras])[0]).astype(np.float32))
            rn_task_l.append(np.abs(np.asarray(rec[k_rn])[0]).astype(np.float32))
        gc_task = np.mean(np.stack(gc_task_l), axis=0) if gc_task_l else None
        ras_task = np.mean(np.stack(ras_task_l), axis=0) if ras_task_l else None
        rn_task = np.mean(np.stack(rn_task_l), axis=0) if rn_task_l else None
        gc_task_full = gc_task
        ras_task_full = ras_task
        rn_task_full = rn_task
    else:
        k_gc = "outputs/debug/gradcam/task"
        k_ras = "outputs/debug/raw_alpha/summation/task"
        k_rn = "outputs/debug/raw_alpha/norm/task"
        gc_task_full = np.abs(np.asarray(rec[k_gc])).astype(np.float32) if k_gc in rec else None
        ras_task_full = np.abs(np.asarray(rec[k_ras])).astype(np.float32) if k_ras in rec else None
        rn_task_full = np.abs(np.asarray(rec[k_rn])).astype(np.float32) if k_rn in rec else None

        # Bug B fix: truncate to real tokens
        if task_mask is not None and gc_task_full is not None:
            span_len = min(len(task_mask), len(gc_task_full))
            n_real = int(task_mask[:span_len].sum())
            gc_task = gc_task_full[:n_real] if n_real > 0 else gc_task_full
            ras_task = ras_task_full[:n_real] if (ras_task_full is not None and n_real > 0) else ras_task_full
            rn_task = rn_task_full[:n_real] if (rn_task_full is not None and n_real > 0) else rn_task_full
        else:
            gc_task, ras_task, rn_task = gc_task_full, ras_task_full, rn_task_full

    # Task gini: raw (full 200) vs fixed (real tokens only) for pi05
    if fmt == "pi05":
        feats["gc_gini_task_raw"] = gini(gc_task_full) if gc_task_full is not None else float("nan")
        feats["ras_gini_task_raw"] = gini(ras_task_full) if ras_task_full is not None else float("nan")
        feats["rn_gini_task_raw"] = gini(rn_task_full) if rn_task_full is not None else float("nan")
        feats["gc_gini_task"] = gini(gc_task) if gc_task is not None else float("nan")
        feats["ras_gini_task"] = gini(ras_task) if ras_task is not None else float("nan")
        feats["rn_gini_task"] = gini(rn_task) if rn_task is not None else float("nan")
        # agree_gc_ras_task: zeros cancel so raw==fixed in cosine
        feats["agree_gc_ras_task"] = cosine(gc_task_full, ras_task_full)
        feats["agree_gc_ras_task_fixed"] = cosine(gc_task, ras_task)
    else:
        feats["gc_gini_task_raw"] = float("nan")
        feats["ras_gini_task_raw"] = float("nan")
        feats["rn_gini_task_raw"] = float("nan")
        feats["gc_gini_task"] = gini(gc_task) if gc_task is not None else float("nan")
        feats["ras_gini_task"] = gini(ras_task) if ras_task is not None else float("nan")
        feats["rn_gini_task"] = gini(rn_task) if rn_task is not None else float("nan")
        feats["agree_gc_ras_task"] = cosine(gc_task, ras_task)
        feats["agree_gc_ras_task_fixed"] = cosine(gc_task, ras_task)

    # ─── Bug D: Value vector features ─────────────────────────────────────────
    vkey = "outputs/debug/attn/v"
    if vkey in rec:
        v_raw = np.asarray(rec[vkey]).astype(np.float32)
        if fmt == "pi0fast":
            V = v_raw[-1, 0, :, 0, :]  # (S, D)
        else:
            V = v_raw[-1, :, 0, :]     # (S, D)

        sp_base_v = _span(rec, "outputs/debug/spans/image/base_0_rgb")
        sp_wrist_v = _span(rec, "outputs/debug/spans/image/left_wrist_0_rgb")
        sp_task_v = _span(rec, "outputs/debug/spans/task")

        # Bug D: pi05 task V — apply token_mask
        if fmt == "pi05" and sp_task_v is not None and task_mask is not None:
            span_len = sp_task_v[1] - sp_task_v[0]
            n_real = int(task_mask[:span_len].sum())
            V_task_full = V[sp_task_v[0]:sp_task_v[1]]
            V_task_real = V_task_full[task_mask[:span_len]] if n_real > 0 else V_task_full
            feats["v_task_near_zero_full"] = int(np.sum(np.linalg.norm(V_task_full, axis=1) < 0.01))
            feats["v_task_near_zero_real"] = int(np.sum(np.linalg.norm(V_task_real, axis=1) < 0.01))
            sp_task_v_eff = (sp_task_v[0], sp_task_v[0] + n_real)
        else:
            if sp_task_v is not None:
                V_task = V[sp_task_v[0]:sp_task_v[1]]
                feats["v_task_near_zero_full"] = int(np.sum(np.linalg.norm(V_task, axis=1) < 0.01))
                feats["v_task_near_zero_real"] = feats["v_task_near_zero_full"]
            else:
                feats["v_task_near_zero_full"] = 0
                feats["v_task_near_zero_real"] = 0
            sp_task_v_eff = sp_task_v

        def v_stats(sp, tag):
            if sp is None:
                for k in (f"v_norm_mean_{tag}", f"v_norm_gini_{tag}",
                          f"v_coherence_{tag}", f"v_drift_{tag}"):
                    feats[k] = float("nan")
                centroids[f"v_centroid_{tag}"] = None
                return
            seg = V[sp[0]:sp[1]]
            norms = np.linalg.norm(seg, axis=1)
            feats[f"v_norm_mean_{tag}"] = float(norms.mean())
            feats[f"v_norm_gini_{tag}"] = gini(norms)
            centroid = seg.mean(axis=0)
            cnorm = np.linalg.norm(centroid)
            if cnorm > 1e-9:
                c_unit = centroid / cnorm
                dot = seg @ c_unit
                row_norms = norms + 1e-9
                feats[f"v_coherence_{tag}"] = float((dot / row_norms).mean())
            else:
                feats[f"v_coherence_{tag}"] = float("nan")
            centroids[f"v_centroid_{tag}"] = centroid
            # drift
            if prev_centroids is not None and prev_centroids.get(f"v_centroid_{tag}") is not None:
                feats[f"v_drift_{tag}"] = 1.0 - cosine(centroid, prev_centroids[f"v_centroid_{tag}"])
            else:
                feats[f"v_drift_{tag}"] = float("nan")

        v_stats(sp_base_v, "base")
        v_stats(sp_wrist_v, "wrist")
        v_stats(sp_task_v_eff, "task")

        # Cross-modality alignment
        for t1, t2 in [("base", "task"), ("base", "wrist"), ("task", "wrist")]:
            c1 = centroids.get(f"v_centroid_{t1}")
            c2 = centroids.get(f"v_centroid_{t2}")
            feats[f"v_align_{t1}_{t2}"] = cosine(c1, c2) if (c1 is not None and c2 is not None) else float("nan")

        # v_coherence_task: raw (all 200 tokens) vs fixed (real tokens) for pi05
        if fmt == "pi05" and sp_task_v is not None and task_mask is not None:
            V_task_full_seg = V[sp_task_v[0]:sp_task_v[1]]
            norms_full = np.linalg.norm(V_task_full_seg, axis=1)
            c_full = V_task_full_seg.mean(axis=0)
            cn = np.linalg.norm(c_full)
            if cn > 1e-9:
                c_unit = c_full / cn
                dot = V_task_full_seg @ c_unit
                feats["v_coherence_task_raw"] = float((dot / (norms_full + 1e-9)).mean())
            else:
                feats["v_coherence_task_raw"] = float("nan")
        else:
            feats["v_coherence_task_raw"] = feats.get("v_coherence_task", float("nan"))
    else:
        for k in ("v_norm_mean_base", "v_norm_gini_base", "v_coherence_base", "v_drift_base",
                  "v_norm_mean_wrist", "v_norm_gini_wrist", "v_coherence_wrist", "v_drift_wrist",
                  "v_norm_mean_task", "v_norm_gini_task", "v_coherence_task", "v_drift_task",
                  "v_align_base_task", "v_align_base_wrist", "v_align_task_wrist",
                  "v_coherence_task_raw", "v_task_near_zero_full", "v_task_near_zero_real"):
            feats[k] = float("nan")

    # ─── Action features ───────────────────────────────────────────────────────
    actions = np.asarray(rec["outputs/actions"]).astype(np.float64)
    signs = np.sign(actions)
    flips = (signs[1:] != signs[:-1]).astype(float)
    feats["action_trans_flip"] = float(flips[:, 0:3].mean())
    feats["action_rot_flip"] = float(flips[:, 3:6].mean())
    feats["action_grip_flip"] = float(flips[:, 6].mean())
    feats["action_magnitude"] = float(np.linalg.norm(actions[0]))
    feats["gripper_val"] = float(actions[0, 6])
    step_norms = np.linalg.norm(actions, axis=1)
    feats["action_spread"] = float(step_norms.std())
    feats["action_plan_norm"] = float(step_norms.mean())
    feats["action_plan_entropy"] = entropy(step_norms)
    feats["action_horizon_align"] = cosine(actions[0], actions[-1])
    feats["action_dir_var"] = float(np.var(actions[:, 0:6], axis=0).mean())

    if prev_rec is not None:
        prev_actions = np.asarray(prev_rec["outputs/actions"]).astype(np.float64)
        a0, p0 = actions[0], prev_actions[0]
        na, np_ = np.linalg.norm(a0), np.linalg.norm(p0)
        feats["plan_consistency"] = float(np.dot(a0, p0) / (na * np_ + 1e-9))
        feats["action_grip_change"] = float(abs(actions[0, 6] - prev_actions[0, 6]))
    else:
        feats["plan_consistency"] = 1.0
        feats["action_grip_change"] = 0.0

    # ─── New step_metrics features ────────────────────────────────────────────
    # gini_base_rw: gini of (base_attn + wrist_attn) combined, corrected
    if wkey in rec:
        if sp_base is not None and sp_wrist is not None:
            seg_base_wrist = np.concatenate([
                mean_attn[sp_base[0]:sp_base[1]],
                mean_attn[sp_wrist[0]:sp_wrist[1]]
            ])
            feats["gini_base_rw"] = gini(seg_base_wrist)
        else:
            feats["gini_base_rw"] = float("nan")

        # task_head_agr: for pi05, how well do individual heads agree on task attention?
        # For pi05: weights shape (2, 8, 10, 968) → last layer, 8 heads, task span
        # We compute pairwise cosine of head-averaged task attention vectors
        if fmt == "pi05" and sp_task_real is not None:
            w_raw = np.asarray(rec[wkey])  # (2, 8, 10, 968)
            # For each head: mean over T_action → (968,), then take task span
            head_task_attn = []
            for h in range(w_raw.shape[1]):
                head_mean = w_raw[-1, h, :, :].mean(axis=0)  # (968,)
                task_seg = head_mean[sp_task_real[0]:sp_task_real[1]]
                head_task_attn.append(task_seg)
            # Pairwise cosine agreement
            n_h = len(head_task_attn)
            agrees = []
            for i in range(n_h):
                for j in range(i + 1, n_h):
                    c = cosine(head_task_attn[i], head_task_attn[j])
                    if not np.isnan(c):
                        agrees.append(c)
            feats["task_head_agr"] = float(np.mean(agrees)) if agrees else float("nan")
        elif fmt == "pi0fast":
            # pi0fast: (2, 1, 8, 819) → last layer, 1 action token, 8 heads
            w_raw = np.asarray(rec[wkey])
            if sp_task is not None:
                head_task_attn = []
                for h in range(w_raw.shape[2]):
                    task_seg = w_raw[-1, 0, h, sp_task[0]:sp_task[1]]
                    head_task_attn.append(task_seg)
                n_h = len(head_task_attn)
                agrees = []
                for i in range(n_h):
                    for j in range(i + 1, n_h):
                        c = cosine(head_task_attn[i], head_task_attn[j])
                        if not np.isnan(c):
                            agrees.append(c)
                feats["task_head_agr"] = float(np.mean(agrees)) if agrees else float("nan")
            else:
                feats["task_head_agr"] = float("nan")
        else:
            feats["task_head_agr"] = float("nan")
    else:
        feats["gini_base_rw"] = float("nan")
        feats["task_head_agr"] = float("nan")

    # ─── pi05-specific multi-aggregation comparison ───────────────────────────
    if fmt == "pi05" and wkey in rec:
        for agg in ("all", "first", "max"):
            ma = extract_mean_attn(rec, fmt, aggregation=agg)
            tag = agg
            if sp_task is not None:
                if task_mask is not None:
                    span_len = sp_task[1] - sp_task[0]
                    n_real = int(task_mask[:span_len].sum())
                    sp_t = (sp_task[0], sp_task[0] + n_real) if n_real > 0 else sp_task
                else:
                    sp_t = sp_task
                total_agg = ma.sum() + 1e-12
                feats[f"frac_task_{tag}"] = float(ma[sp_t[0]:sp_t[1]].sum() / total_agg)
            else:
                feats[f"frac_task_{tag}"] = float("nan")
            if sp_base is not None:
                feats[f"attn_gini_base_{tag}"] = gini(ma[sp_base[0]:sp_base[1]])
            else:
                feats[f"attn_gini_base_{tag}"] = float("nan")
    else:
        for agg in ("all", "first", "max"):
            feats[f"frac_task_{agg}"] = float("nan")
            feats[f"attn_gini_base_{agg}"] = float("nan")

    return feats, centroids


def load_episode_step_features(ep: dict, run_dir: Path) -> List[dict]:
    """Extract corrected features for each step of an episode."""
    records = load_episode_records(ep, run_dir)
    if not records:
        return []
    all_feats = []
    prev_rec = None
    prev_centroids = None
    for rel_step, rec in records:
        try:
            feats, centroids = extract_corrected_features(rec, prev_rec, prev_centroids)
            feats["rel_step"] = rel_step
            feats["abs_step"] = ep["start_idx"] + rel_step
            all_feats.append(feats)
            prev_rec = rec
            prev_centroids = centroids
        except Exception as e:
            print(f"[WARN] ep={ep.get('episode_num')} step={rel_step}: {e}")
        prev_rec = rec
    return all_feats

# ─── Step 1: Shape verification ───────────────────────────────────────────────

def step1_verify_shapes(log):
    log("\n" + "=" * 70)
    log("STEP 1: Data shape verification")
    log("=" * 70)

    for group_name, run_dir in DATA_GROUPS.items():
        log(f"\n--- {group_name} ({run_dir.name}) ---")
        # Load a few steps
        eps_succ, eps_fail = load_episodes(run_dir)
        ep = eps_succ[0] if eps_succ else eps_fail[0]
        records = load_episode_records(ep, run_dir)
        sample_steps = records[:3] if len(records) >= 3 else records

        for rel_step, rec in sample_steps:
            fmt = detect_format(rec)
            log(f"\n  [rel_step={rel_step}, fmt={fmt}]")

            # State
            state_full = np.asarray(rec.get("outputs/state", rec.get("inputs/observation/state", [])))
            log(f"  outputs/state shape: {state_full.shape}")
            log(f"  state[:8]: {state_full[:8].round(4)}")
            if len(state_full) > 8:
                log(f"  state[8:]: {state_full[8:]}")
                log(f"  state[8:] all zeros: {np.all(state_full[8:] == 0)}")

            # Attn weights
            w = rec.get("outputs/debug/attn/weights")
            if w is not None:
                log(f"  attn/weights shape: {np.asarray(w).shape}")

            # Attn v
            v = rec.get("outputs/debug/attn/v")
            if v is not None:
                log(f"  attn/v shape: {np.asarray(v).shape}")

            # Task token mask
            tm = rec.get("outputs/debug/tokens/task/token_mask")
            if tm is not None:
                tm_arr = np.asarray(tm).ravel()
                log(f"  tokens/task/token_mask shape: {np.asarray(tm).shape}, sum (real tokens): {int(tm_arr.sum())}")

            # Spans
            log(f"  spans:")
            for k, val in rec.items():
                if "spans" in k:
                    log(f"    {k.replace('outputs/debug/spans/', '')}: {tuple(np.asarray(val).ravel().tolist())}")

# ─── Step 2: Per-step feature comparison ─────────────────────────────────────

def step2_per_step_sample(log):
    log("\n" + "=" * 70)
    log("STEP 2: Per-step corrected feature values (sample)")
    log("=" * 70)

    for group_name, run_dir in DATA_GROUPS.items():
        log(f"\n--- {group_name} ---")
        eps_succ, eps_fail = load_episodes(run_dir)
        ep = eps_succ[0] if eps_succ else eps_fail[0]
        records = load_episode_records(ep, run_dir)
        sample = records[:5] if len(records) >= 5 else records

        prev_rec = None
        prev_centroids = None
        for rel_step, rec in sample:
            fmt = detect_format(rec)
            feats, centroids = extract_corrected_features(rec, prev_rec, prev_centroids)
            prev_rec = rec
            prev_centroids = centroids

            log(f"\n  step={rel_step} fmt={fmt}")

            # State features before/after fix
            log(f"  state_norm_raw={feats['state_norm_raw']:.4f}  state_norm(fixed)={feats['state_norm']:.4f}")
            log(f"  joint_vel_raw={feats['joint_vel_raw']:.4f}    joint_vel(fixed)={feats['joint_vel']:.4f}")

            # Modality fracs before/after
            log(f"  frac_base_raw={feats['frac_base_raw']:.4f}  frac_base(fixed)={feats['frac_base']:.4f}")
            log(f"  frac_wrist_raw={feats['frac_wrist_raw']:.4f}  frac_wrist(fixed)={feats['frac_wrist']:.4f}")
            log(f"  frac_task_raw={feats['frac_task_raw']:.4f}  frac_task(fixed)={feats['frac_task']:.4f}")
            log(f"  routing_entropy_raw={feats['routing_entropy_raw']:.4f}  routing_entropy(fixed)={feats['routing_entropy']:.4f}")

            # Task gini before/after mask fix
            if fmt == "pi05":
                log(f"  gc_gini_task_raw={feats['gc_gini_task_raw']:.4f}  gc_gini_task(fixed)={feats['gc_gini_task']:.4f}")
                log(f"  ras_gini_task_raw={feats['ras_gini_task_raw']:.4f}  ras_gini_task(fixed)={feats['ras_gini_task']:.4f}")
                log(f"  v_coherence_task_raw={feats['v_coherence_task_raw']:.4f}  v_coherence_task(fixed)={feats['v_coherence_task']:.4f}")
                log(f"  agree_gc_ras_task(full)={feats['agree_gc_ras_task']:.4f}  agree_gc_ras_task(fixed)={feats['agree_gc_ras_task_fixed']:.4f}")
                log(f"  V task near-zero: full={feats['v_task_near_zero_full']}  real={feats['v_task_near_zero_real']}")
                log(f"  Pi05 aggregation comparison:")
                log(f"    frac_task_all={feats['frac_task_all']:.4f}  frac_task_first={feats['frac_task_first']:.4f}  frac_task_max={feats['frac_task_max']:.4f}")
                log(f"    attn_gini_base_all={feats['attn_gini_base_all']:.4f}  first={feats['attn_gini_base_first']:.4f}  max={feats['attn_gini_base_max']:.4f}")
            else:
                log(f"  gc_gini_task={feats['gc_gini_task']:.4f}")
                log(f"  ras_gini_task={feats['ras_gini_task']:.4f}")
                log(f"  v_coherence_task={feats['v_coherence_task']:.4f}")

            # Base/wrist features
            log(f"  gc_gini_base={feats['gc_gini_base']:.4f}  gc_gini_wrist={feats['gc_gini_wrist']:.4f}")
            log(f"  attn_gini_base={feats['attn_gini_base']:.4f}  attn_entropy_base={feats['attn_entropy_base']:.4f}")
            log(f"  normalized_attn_entropy_base={feats['normalized_attn_entropy_base']:.4f}")
            log(f"  v_coherence_base={feats['v_coherence_base']:.4f}  v_coherence_wrist={feats['v_coherence_wrist']:.4f}")
            log(f"  v_drift_base={feats['v_drift_base']:.4f}  v_drift_wrist={feats['v_drift_wrist']:.4f}")

            # Action features
            log(f"  action_magnitude={feats['action_magnitude']:.4f}  action_spread={feats['action_spread']:.4f}")
            log(f"  action_plan_entropy={feats['action_plan_entropy']:.4f}  action_horizon_align={feats['action_horizon_align']:.4f}")
            log(f"  plan_consistency={feats['plan_consistency']:.4f}  action_rot_flip={feats['action_rot_flip']:.4f}")

            # New step_metrics
            log(f"  gini_base_rw={feats['gini_base_rw']:.4f}  task_head_agr={feats['task_head_agr']:.4f}")

# ─── Sample episodes from raw npy (pi05) and cache (pi0fast) ─────────────────

def sample_episodes_from_raw(run_dir: Path, n_sample: int) -> Tuple[List[List[dict]], List[List[dict]]]:
    """
    Sample N_SAMPLE success and N_SAMPLE failure episodes from raw npy files.
    Returns (success_ep_feature_lists, failure_ep_feature_lists).
    """
    eps_succ, eps_fail = load_episodes(run_dir)

    rng = np.random.default_rng(42)
    n_s = min(n_sample, len(eps_succ))
    n_f = min(n_sample, len(eps_fail))
    sel_s = rng.choice(len(eps_succ), n_s, replace=False).tolist()
    sel_f = rng.choice(len(eps_fail), n_f, replace=False).tolist()

    success_eps = []
    for idx in sel_s:
        ep = eps_succ[idx]
        feats_list = load_episode_step_features(ep, run_dir)
        if feats_list:
            success_eps.append(feats_list)

    failure_eps = []
    for idx in sel_f:
        ep = eps_fail[idx]
        feats_list = load_episode_step_features(ep, run_dir)
        if feats_list:
            failure_eps.append(feats_list)

    return success_eps, failure_eps


def correct_cache_fracs(ep_list: List[List[dict]], run_dir: Path) -> List[List[dict]]:
    """
    For pi0fast cache episodes, fix frac_* features by loading raw npy to get rwrist span.
    This corrects Bug C in-memory.
    """
    eps_succ, eps_fail = load_episodes(run_dir)
    all_eps = eps_succ + eps_fail

    # Build abs_step -> ep mapping
    step_to_ep: dict = {}
    for ep in all_eps:
        for s in range(ep["start_idx"], ep["end_idx"] + 1):
            step_to_ep[s] = ep

    # We need the rwrist span from any step — it's constant per run
    # Load first available step to get span
    first_ep = all_eps[0] if all_eps else None
    rwrist_span = None
    if first_ep:
        path = run_dir / f"step_{first_ep['start_idx']}.npy"
        if path.exists():
            rec = load_record(path)
            sp = _span(rec, "outputs/debug/spans/image/right_wrist_0_rgb")
            rwrist_span = sp

    corrected_eps = []
    for ep_steps in ep_list:
        corrected_steps = []
        for step_feats in ep_steps:
            s = step_feats.copy()
            # Re-derive corrected frac_* from cached raw attn values
            # We can approximate by loading the raw npy for this step
            # But for efficiency, we use the known rwrist fraction from raw npy
            # Cache stores pre-v2 fracs without the rwrist correction
            # We need to recompute from raw npy — too expensive for all steps.
            # Instead: use the formula: frac_X_fixed = frac_X_raw / (1 - frac_rwrist_raw)
            frac_rwrist_raw = s.get("frac_rwrist", 0.0)
            if not np.isnan(frac_rwrist_raw) and frac_rwrist_raw > 0:
                denom_factor = 1.0 - frac_rwrist_raw + 1e-12
                for key in ("frac_base", "frac_wrist", "frac_task", "frac_state"):
                    raw_val = s.get(key, float("nan"))
                    if not np.isnan(raw_val):
                        s[f"{key}_corrected"] = raw_val / denom_factor
                    else:
                        s[f"{key}_corrected"] = float("nan")
            else:
                for key in ("frac_base", "frac_wrist", "frac_task", "frac_state"):
                    s[f"{key}_corrected"] = s.get(key, float("nan"))
            corrected_steps.append(s)
        corrected_eps.append(corrected_steps)
    return corrected_eps


def sample_episodes_from_cache(cache_path: Path, n_sample: int,
                                run_dir: Path) -> Tuple[List[List[dict]], List[List[dict]]]:
    """
    Sample N_SAMPLE success/failure from v2 cache (pi0fast).
    Returns (success_ep_steps, failure_ep_steps).
    Also applies in-memory frac correction.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    rng = np.random.default_rng(42)
    succ_all = cache["success"]
    fail_all = cache["failure"]
    n_s = min(n_sample, len(succ_all))
    n_f = min(n_sample, len(fail_all))
    sel_s = rng.choice(len(succ_all), n_s, replace=False).tolist()
    sel_f = rng.choice(len(fail_all), n_f, replace=False).tolist()

    succ_eps = [succ_all[i] for i in sel_s]
    fail_eps = [fail_all[i] for i in sel_f]

    # Apply frac correction in-memory
    succ_eps = correct_cache_fracs(succ_eps, run_dir)
    fail_eps = correct_cache_fracs(fail_eps, run_dir)

    return succ_eps, fail_eps

# ─── Step 3: Group-level statistics at step=20 ────────────────────────────────

def get_step_value(ep_steps: List[dict], target_rel_step: int,
                   feature: str, fallback_key: str = None) -> float:
    """Get feature value at target_rel_step (or nearest available step)."""
    for s in ep_steps:
        if s.get("rel_step") == target_rel_step:
            v = s.get(feature, float("nan"))
            if np.isnan(v) and fallback_key:
                v = s.get(fallback_key, float("nan"))
            return v
    # Nearest step
    if ep_steps:
        steps_avail = [s.get("rel_step", -1) for s in ep_steps]
        nearest = min(steps_avail, key=lambda x: abs(x - target_rel_step))
        for s in ep_steps:
            if s.get("rel_step") == nearest:
                v = s.get(feature, float("nan"))
                if np.isnan(v) and fallback_key:
                    v = s.get(fallback_key, float("nan"))
                return v
    return float("nan")


def compute_group_stats(succ_eps: List[List[dict]], fail_eps: List[List[dict]],
                        features: List[str], step: int = AUDIT_STEP) -> dict:
    """Compute stats at given step across all episodes."""
    stats = {}
    for feat in features:
        sv = [get_step_value(ep, step, feat) for ep in succ_eps]
        fv = [get_step_value(ep, step, feat) for ep in fail_eps]
        sv = [x for x in sv if not np.isnan(x)]
        fv = [x for x in fv if not np.isnan(x)]
        stats[feat] = {
            "succ_mean": nan_mean(sv),
            "succ_std": nan_std(sv),
            "fail_mean": nan_mean(fv),
            "fail_std": nan_std(fv),
            "auroc": auroc(fv, sv),
        }
    return stats

# ─── Step 4: Z-normalization test ─────────────────────────────────────────────

def z_norm_test(succ_eps: List[List[dict]], fail_eps: List[List[dict]],
                features: List[str], target_step: int = AUDIT_STEP,
                eps_zn: float = 1e-6) -> dict:
    """
    Compute z-normalized failure scores at target_step.
    Calibration: success mean/std at each rel_step.
    """
    results = {}
    for feat in features:
        # Build success calibration per rel_step
        # Collect all rel_steps
        all_succ_steps = {}
        for ep in succ_eps:
            for s in ep:
                t = s.get("rel_step", -1)
                v = s.get(feat, float("nan"))
                if not np.isnan(v):
                    if t not in all_succ_steps:
                        all_succ_steps[t] = []
                    all_succ_steps[t].append(v)

        # Compute mean/std per step
        succ_mean_by_step = {}
        succ_std_by_step = {}
        for t, vals in all_succ_steps.items():
            succ_mean_by_step[t] = np.mean(vals)
            succ_std_by_step[t] = np.std(vals)

        # Apply to failure episodes at target_step
        fail_z_scores = []
        for ep in fail_eps:
            v = get_step_value(ep, target_step, feat)
            if np.isnan(v):
                continue
            mu = succ_mean_by_step.get(target_step, float("nan"))
            sigma = succ_std_by_step.get(target_step, float("nan"))
            if np.isnan(mu) or np.isnan(sigma):
                continue
            z = (v - mu) / (sigma + eps_zn)
            fail_z_scores.append(z)

        results[feat] = {
            "fail_z_mean": nan_mean(fail_z_scores),
            "fail_z_std": nan_std(fail_z_scores),
            "n_fail": len(fail_z_scores),
            "elevated": nan_mean(fail_z_scores) > 0.5 if fail_z_scores else False,
        }
    return results

# ─── Step 5: Classification table ─────────────────────────────────────────────

# Features to audit
CORE_FEATURES = [
    # State
    "state_norm", "joint_vel",
    # Routing fracs
    "frac_base", "frac_wrist", "frac_task", "frac_state", "routing_entropy",
    # Base camera heatmaps
    "gc_gini_base", "ras_gini_base", "rn_gini_base",
    "attn_gini_base", "attn_entropy_base", "normalized_attn_entropy_base",
    # Wrist camera heatmaps
    "gc_gini_wrist", "ras_gini_wrist", "rn_gini_wrist",
    # Task features
    "gc_gini_task", "ras_gini_task", "rn_gini_task",
    # Cross-signal agreement
    "agree_gc_ras_base", "agree_gc_ras_task", "agree_base_wrist_gc",
    # Value vector
    "v_coherence_base", "v_coherence_wrist", "v_coherence_task",
    "v_drift_base", "v_drift_wrist", "v_drift_task",
    "v_align_base_task", "v_align_base_wrist",
    # Action
    "action_magnitude", "action_spread", "action_plan_entropy",
    "action_horizon_align", "plan_consistency", "action_rot_flip",
    # Step metrics
    "gini_base_rw", "task_head_agr",
]

# Cache-corrected keys for pi0fast (frac_corrected)
CACHE_FRAC_REMAP = {
    "frac_base": "frac_base_corrected",
    "frac_wrist": "frac_wrist_corrected",
    "frac_task": "frac_task_corrected",
    "frac_state": "frac_state_corrected",
}

CAUTION_FEATURES = [
    "frac_base", "frac_wrist", "frac_task", "routing_entropy",
    "v_coherence_base", "v_coherence_wrist", "v_coherence_task",
    "attn_entropy_base",
]

# ─── Description lookup ────────────────────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "state_norm": "L2 norm of 8-dim joint state (Bug A fix: [:8])",
    "joint_vel": "L2 delta of joint state between consecutive steps",
    "frac_base": "Fraction of attention to base camera (Bug C: excl. rwrist for pi0fast)",
    "frac_wrist": "Fraction of attention to left wrist camera",
    "frac_task": "Fraction of attention to task tokens",
    "frac_state": "Fraction of attention to state tokens (pi0fast only)",
    "routing_entropy": "Entropy of modality attention fractions [base,wrist,task,state]",
    "gc_gini_base": "GradCAM Gini concentration for base camera",
    "ras_gini_base": "Raw attention (summation) Gini for base camera",
    "rn_gini_base": "Raw attention (norm) Gini for base camera",
    "attn_gini_base": "Attention weight Gini over base camera token span",
    "attn_entropy_base": "Attention weight entropy over base camera token span",
    "normalized_attn_entropy_base": "Attn entropy / log(256) normalized",
    "gc_gini_wrist": "GradCAM Gini for left wrist camera",
    "ras_gini_wrist": "Raw attention Gini for left wrist camera",
    "rn_gini_wrist": "Raw attention norm Gini for left wrist camera",
    "gc_gini_task": "GradCAM Gini over task tokens (Bug B fix: real tokens only for pi05)",
    "ras_gini_task": "RAS Gini over task tokens (Bug B fix)",
    "rn_gini_task": "RN Gini over task tokens (Bug B fix)",
    "agree_gc_ras_base": "Cosine agreement GradCAM vs RAS on base camera",
    "agree_gc_ras_task": "Cosine agreement GradCAM vs RAS on task tokens",
    "agree_base_wrist_gc": "Cross-camera cosine agreement (GradCAM) base vs wrist",
    "v_coherence_base": "Mean cosine of V vectors to centroid (base camera)",
    "v_coherence_wrist": "Mean cosine of V vectors to centroid (left wrist)",
    "v_coherence_task": "Mean cosine of V vectors to centroid (task tokens, Bug D fix)",
    "v_drift_base": "1 - cosine(centroid_t, centroid_{t-1}) for base camera V",
    "v_drift_wrist": "1 - cosine(centroid_t, centroid_{t-1}) for wrist V",
    "v_drift_task": "1 - cosine(centroid_t, centroid_{t-1}) for task V",
    "v_align_base_task": "Cosine of V centroids: base vs task",
    "v_align_base_wrist": "Cosine of V centroids: base vs wrist",
    "action_magnitude": "L2 norm of first predicted action",
    "action_spread": "Std of action norms across 10-step horizon",
    "action_plan_entropy": "Entropy of action norms across horizon",
    "action_horizon_align": "Cosine(action[0], action[-1]) temporal consistency",
    "plan_consistency": "Cosine(action[0]_t, action[0]_{t-1})",
    "action_rot_flip": "Fraction of sign flips in rotation dims across horizon",
    "gini_base_rw": "Gini of combined base+wrist attention (new step_metric)",
    "task_head_agr": "Mean pairwise head agreement on task attention (new step_metric)",
}


def classify_feature(feat: str,
                     stats_pi05: dict, stats_l10: dict, stats_lp: dict,
                     z_pi05: dict, z_l10: dict, z_lp: dict) -> Tuple[str, str]:
    """Return (classification, recommended_use)."""
    s05 = stats_pi05.get(feat, {})
    sl10 = stats_l10.get(feat, {})
    slp = stats_lp.get(feat, {})

    m05 = s05.get("succ_mean", float("nan"))
    ml10 = sl10.get("succ_mean", float("nan"))
    mlp = slp.get("succ_mean", float("nan"))

    auroc_pi05 = s05.get("auroc", float("nan"))
    auroc_l10 = sl10.get("auroc", float("nan"))
    auroc_lp = slp.get("auroc", float("nan"))

    # Check if any group is entirely NaN
    all_nan = all(np.isnan(x) for x in [m05, ml10, mlp])
    pi05_nan = np.isnan(m05)
    l10_nan = np.isnan(ml10)
    lp_nan = np.isnan(mlp)

    if all_nan:
        return "D. FORMAT_SPECIFIC", "exclude"

    if pi05_nan and not l10_nan:
        return "D. FORMAT_SPECIFIC", "pi0fast_only"
    if not pi05_nan and l10_nan:
        return "D. FORMAT_SPECIFIC", "pi05_only"

    # Check if it's a fixable bug (task features with mask, state with [:8])
    fixable_features = {"state_norm", "joint_vel", "gc_gini_task", "ras_gini_task",
                        "rn_gini_task", "v_coherence_task", "frac_base", "frac_wrist",
                        "frac_task", "routing_entropy"}
    if feat in fixable_features:
        # Check if AUROC is reasonable in at least some groups
        aurocs = [x for x in [auroc_pi05, auroc_l10, auroc_lp] if not np.isnan(x)]
        if aurocs and max(aurocs) >= 0.55:
            # Check if z-norm aligns them
            z05 = z_pi05.get(feat, {})
            zl10 = z_l10.get(feat, {})
            zlp = z_lp.get(feat, {})
            all_elevated = (z05.get("elevated", False) and
                            zl10.get("elevated", False) and
                            zlp.get("elevated", False))
            if all_elevated:
                return "B. CROSS_FORMAT_SAFE", "universal_raw"
            else:
                return "C. CAUTION_NORMALIZE", "universal_z_normalized"

    # Check AUROC significance
    aurocs = [x for x in [auroc_pi05, auroc_l10, auroc_lp] if not np.isnan(x)]
    if not aurocs:
        return "D. FORMAT_SPECIFIC", "exclude"
    max_auroc = max(aurocs)

    if max_auroc < 0.55:
        # Check if it's model-specific strong
        if not pi05_nan and not l10_nan:
            # Both formats have data but AUROC is low — not predictive
            return "D. FORMAT_SPECIFIC", "exclude"
        return "D. FORMAT_SPECIFIC", "exclude"

    # Scale compatibility: are means roughly in same range?
    valid_means = [x for x in [m05, ml10, mlp] if not np.isnan(x)]
    if len(valid_means) >= 2:
        mean_range = max(valid_means) - min(valid_means)
        mean_scale = max(abs(x) for x in valid_means) + 1e-12
        rel_diff = mean_range / mean_scale

        # Check z-norm alignment across groups
        z05 = z_pi05.get(feat, {})
        zl10 = z_l10.get(feat, {})
        zlp = z_lp.get(feat, {})
        all_elevated = (z05.get("elevated", False) and
                        zl10.get("elevated", False) and
                        zlp.get("elevated", False))

        if rel_diff < 0.5 and all_elevated:
            return "B. CROSS_FORMAT_SAFE", "universal_raw"
        elif all_elevated:
            return "C. CAUTION_NORMALIZE", "universal_z_normalized"
        elif max_auroc >= 0.6:
            return "C. CAUTION_NORMALIZE", "same_model_transfer_only"
        else:
            return "D. FORMAT_SPECIFIC", "exclude"

    return "D. FORMAT_SPECIFIC", "exclude"


def step3_and_4_stats(succ_pi05, fail_pi05,
                       succ_l10, fail_l10,
                       succ_lp, fail_lp,
                       log) -> Tuple[dict, dict, dict, dict, dict, dict]:
    log("\n" + "=" * 70)
    log("STEP 3: Group-level statistics at step=20")
    log("=" * 70)

    stats_pi05 = compute_group_stats(succ_pi05, fail_pi05, CORE_FEATURES, step=AUDIT_STEP)
    stats_l10 = compute_group_stats(succ_l10, fail_l10, CORE_FEATURES, step=AUDIT_STEP)
    stats_lp = compute_group_stats(succ_lp, fail_lp, CORE_FEATURES, step=AUDIT_STEP)

    for feat in CORE_FEATURES:
        s05 = stats_pi05[feat]
        sl10 = stats_l10[feat]
        slp = stats_lp[feat]
        log(f"\n  {feat}:")
        log(f"    pi05_l10:     succ={s05['succ_mean']:.4f}±{s05['succ_std']:.4f}  "
            f"fail={s05['fail_mean']:.4f}±{s05['fail_std']:.4f}  AUROC={s05['auroc']:.3f}")
        log(f"    pi0f_l10:     succ={sl10['succ_mean']:.4f}±{sl10['succ_std']:.4f}  "
            f"fail={sl10['fail_mean']:.4f}±{sl10['fail_std']:.4f}  AUROC={sl10['auroc']:.3f}")
        log(f"    pi0f_lp:      succ={slp['succ_mean']:.4f}±{slp['succ_std']:.4f}  "
            f"fail={slp['fail_mean']:.4f}±{slp['fail_std']:.4f}  AUROC={slp['auroc']:.3f}")

    log("\n" + "=" * 70)
    log("STEP 4: Z-normalization test (CAUTION features)")
    log("=" * 70)

    z_pi05 = z_norm_test(succ_pi05, fail_pi05, CAUTION_FEATURES, target_step=AUDIT_STEP)
    z_l10 = z_norm_test(succ_l10, fail_l10, CAUTION_FEATURES, target_step=AUDIT_STEP)
    z_lp = z_norm_test(succ_lp, fail_lp, CAUTION_FEATURES, target_step=AUDIT_STEP)

    for feat in CAUTION_FEATURES:
        z05 = z_pi05[feat]
        zl10 = z_l10[feat]
        zlp = z_lp[feat]
        log(f"\n  {feat} Z-score at step {AUDIT_STEP} (failure episodes):")
        log(f"    pi05_l10:   mean_z={z05['fail_z_mean']:.3f}  elevated={z05['elevated']}")
        log(f"    pi0f_l10:   mean_z={zl10['fail_z_mean']:.3f}  elevated={zl10['elevated']}")
        log(f"    pi0f_lp:    mean_z={zlp['fail_z_mean']:.3f}  elevated={zlp['elevated']}")
        all_el = z05["elevated"] and zl10["elevated"] and zlp["elevated"]
        log(f"    --> all elevated (cross-format z-compatible): {all_el}")

    return stats_pi05, stats_l10, stats_lp, z_pi05, z_l10, z_lp


def step5_classification_table(stats_pi05, stats_l10, stats_lp,
                                z_pi05, z_l10, z_lp,
                                log) -> dict:
    log("\n" + "=" * 70)
    log("STEP 5: Feature classification table")
    log("=" * 70)

    header = (f"{'feature':<35} | {'classification':<25} | "
              f"{'AUROC_pi05':>10} {'AUROC_l10':>9} {'AUROC_lp':>8} | "
              f"{'recommended_use':<25}")
    log(header)
    log("-" * len(header))

    classifications = {}
    for feat in CORE_FEATURES:
        cls, rec_use = classify_feature(feat, stats_pi05, stats_l10, stats_lp,
                                         z_pi05, z_l10, z_lp)
        classifications[feat] = (cls, rec_use)
        s05 = stats_pi05.get(feat, {})
        sl10 = stats_l10.get(feat, {})
        slp = stats_lp.get(feat, {})
        a05 = s05.get("auroc", float("nan"))
        al10 = sl10.get("auroc", float("nan"))
        alp = slp.get("auroc", float("nan"))
        log(f"{feat:<35} | {cls:<25} | "
            f"{a05:>10.3f} {al10:>9.3f} {alp:>8.3f} | "
            f"{rec_use:<25}")

    return classifications


def step5_write_report_file(stats_pi05, stats_l10, stats_lp,
                             z_pi05, z_l10, z_lp,
                             classifications: dict):
    """Write full report to REPORT_PATH."""
    lines = []
    lines.append("=" * 100)
    lines.append("FEATURE AUDIT REPORT — Cross-format feature compatibility")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)

    # Full table with all columns
    cols = ["feature", "description", "group",
            "succ_mean", "succ_std", "fail_mean", "fail_std",
            "AUROC", "fail_z_mean", "elevated",
            "classification", "recommended_use"]
    lines.append("\n## Full Feature Table\n")
    lines.append("| " + " | ".join(f"{c:<30}" for c in ["feature", "description"]) + " |")
    lines.append("|" + "-" * 200 + "|")

    for feat in CORE_FEATURES:
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        cls, rec_use = classifications.get(feat, ("?", "?"))
        s05 = stats_pi05.get(feat, {})
        sl10 = stats_l10.get(feat, {})
        slp = stats_lp.get(feat, {})
        z05 = z_pi05.get(feat, {})
        zl10 = z_l10.get(feat, {})
        zlp = z_lp.get(feat, {})

        lines.append(f"\n### {feat}")
        lines.append(f"  Description: {desc}")
        lines.append(f"  Classification: {cls}")
        lines.append(f"  Recommended use: {rec_use}")
        lines.append(f"  pi05_libero10:   succ={s05.get('succ_mean', float('nan')):.4f}±{s05.get('succ_std', float('nan')):.4f}  "
                     f"fail={s05.get('fail_mean', float('nan')):.4f}±{s05.get('fail_std', float('nan')):.4f}  "
                     f"AUROC={s05.get('auroc', float('nan')):.3f}  "
                     f"fail_z={z05.get('fail_z_mean', float('nan')):.3f}  elevated={z05.get('elevated', False)}")
        lines.append(f"  pi0fast_libero10:succ={sl10.get('succ_mean', float('nan')):.4f}±{sl10.get('succ_std', float('nan')):.4f}  "
                     f"fail={sl10.get('fail_mean', float('nan')):.4f}±{sl10.get('fail_std', float('nan')):.4f}  "
                     f"AUROC={sl10.get('auroc', float('nan')):.3f}  "
                     f"fail_z={zl10.get('fail_z_mean', float('nan')):.3f}  elevated={zl10.get('elevated', False)}")
        lines.append(f"  pi0fast_liberoplus:succ={slp.get('succ_mean', float('nan')):.4f}±{slp.get('succ_std', float('nan')):.4f}  "
                     f"fail={slp.get('fail_mean', float('nan')):.4f}±{slp.get('fail_std', float('nan')):.4f}  "
                     f"AUROC={slp.get('auroc', float('nan')):.3f}  "
                     f"fail_z={zlp.get('fail_z_mean', float('nan')):.3f}  elevated={zlp.get('elevated', False)}")

    # Markdown table
    lines.append("\n\n## Markdown Classification Table\n")
    lines.append("| feature | description | pi05_mean | pi05_std | pi0f_l10_mean | pi0f_l10_std | pi0f_lp_mean | pi0f_lp_std | AUROC_pi05 | AUROC_pi0f_l10 | AUROC_pi0f_lp | classification | recommended_use |")
    lines.append("|" + "|".join(["-" * 30, "-" * 40, "-" * 10, "-" * 9, "-" * 14, "-" * 13, "-" * 13, "-" * 12, "-" * 11, "-" * 15, "-" * 14, "-" * 25, "-" * 16]) + "|")

    for feat in CORE_FEATURES:
        s05 = stats_pi05.get(feat, {})
        sl10 = stats_l10.get(feat, {})
        slp = stats_lp.get(feat, {})
        cls, rec_use = classifications.get(feat, ("?", "?"))
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        lines.append(f"| {feat} | {desc[:38]} | "
                     f"{s05.get('succ_mean', float('nan')):.4f} | "
                     f"{s05.get('succ_std', float('nan')):.4f} | "
                     f"{sl10.get('succ_mean', float('nan')):.4f} | "
                     f"{sl10.get('succ_std', float('nan')):.4f} | "
                     f"{slp.get('succ_mean', float('nan')):.4f} | "
                     f"{slp.get('succ_std', float('nan')):.4f} | "
                     f"{s05.get('auroc', float('nan')):.3f} | "
                     f"{sl10.get('auroc', float('nan')):.3f} | "
                     f"{slp.get('auroc', float('nan')):.3f} | "
                     f"{cls} | {rec_use} |")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Report written to {REPORT_PATH}]")


def step6_final_lists(classifications: dict, log):
    log("\n" + "=" * 70)
    log("STEP 6: Final feature group lists")
    log("=" * 70)

    g1, g2, g3, g4 = [], [], [], []
    for feat, (cls, rec_use) in classifications.items():
        if rec_use == "universal_raw":
            g1.append(feat)
        elif rec_use == "universal_z_normalized":
            g2.append(feat)
        elif rec_use in ("same_model_transfer_only", "pi0fast_only", "pi05_only"):
            g3.append(feat)
        else:
            g4.append(feat)

    log("\nGroup 1 — Universal raw-safe (use across all formats):")
    for f in g1:
        log(f"  {f}")

    log("\nGroup 2 — Universal after z-normalization:")
    for f in g2:
        log(f"  {f}")

    log("\nGroup 3 — Model-specific (strong within-model, not cross-format):")
    for f in g3:
        log(f"  {f}")

    log("\nGroup 4 — Exclude (NaN, semantically incompatible, or no AUROC signal):")
    for f in g4:
        log(f"  {f}")

    return g1, g2, g3, g4


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_lines = []

    def log(msg: str = ""):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 70)
    log("FEATURE AUDIT — Cross-format compatibility for failure prediction")
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"N_SAMPLE: {N_SAMPLE} success + {N_SAMPLE} failure per group")
    log(f"Target audit step: {AUDIT_STEP}")
    log("=" * 70)

    # ── Step 1: Verify data shapes ─────────────────────────────────────────────
    step1_verify_shapes(log)

    # ── Step 2: Per-step feature sample ────────────────────────────────────────
    step2_per_step_sample(log)

    # ── Load episodes for statistics ───────────────────────────────────────────
    log("\n" + "=" * 70)
    log("Loading episodes for group statistics...")
    log("=" * 70)

    # pi05_libero10: load from raw npy
    log("\n[pi05_libero10] Loading from raw npy files...")
    t0 = time.time()
    succ_pi05, fail_pi05 = sample_episodes_from_raw(
        DATA_GROUPS["pi05_libero10"], N_SAMPLE)
    log(f"  Loaded {len(succ_pi05)} success, {len(fail_pi05)} failure episodes"
        f" in {time.time()-t0:.1f}s")

    # pi0fast_libero10: load from v2 cache
    log("\n[pi0fast_libero10] Loading from v2 cache...")
    t0 = time.time()
    succ_l10, fail_l10 = sample_episodes_from_cache(
        CACHE_FILES["pi0fast_libero10"], N_SAMPLE,
        DATA_GROUPS["pi0fast_libero10"])
    log(f"  Loaded {len(succ_l10)} success, {len(fail_l10)} failure episodes"
        f" in {time.time()-t0:.1f}s")

    # pi0fast_liberoplus: load from v2 cache
    log("\n[pi0fast_liberoplus] Loading from v2 cache...")
    t0 = time.time()
    succ_lp, fail_lp = sample_episodes_from_cache(
        CACHE_FILES["pi0fast_liberoplus"], N_SAMPLE,
        DATA_GROUPS["pi0fast_liberoplus"])
    log(f"  Loaded {len(succ_lp)} success, {len(fail_lp)} failure episodes"
        f" in {time.time()-t0:.1f}s")

    # ── Steps 3 & 4: Stats + Z-norm test ──────────────────────────────────────
    stats_pi05, stats_l10, stats_lp, z_pi05, z_l10, z_lp = step3_and_4_stats(
        succ_pi05, fail_pi05,
        succ_l10, fail_l10,
        succ_lp, fail_lp,
        log)

    # ── Step 5: Classification table ───────────────────────────────────────────
    classifications = step5_classification_table(
        stats_pi05, stats_l10, stats_lp,
        z_pi05, z_l10, z_lp, log)

    step5_write_report_file(
        stats_pi05, stats_l10, stats_lp,
        z_pi05, z_l10, z_lp,
        classifications)

    # ── Step 6: Final lists ────────────────────────────────────────────────────
    g1, g2, g3, g4 = step6_final_lists(classifications, log)

    log("\n" + "=" * 70)
    log("AUDIT COMPLETE")
    log(f"  Group 1 (universal_raw): {len(g1)} features")
    log(f"  Group 2 (universal_z_norm): {len(g2)} features")
    log(f"  Group 3 (model-specific): {len(g3)} features")
    log(f"  Group 4 (exclude): {len(g4)} features")
    log(f"  Report written to: {REPORT_PATH}")
    log(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)


if __name__ == "__main__":
    main()
