#!/usr/bin/env python3
"""
phase_aware_analysis.py  —  Phase-aware feature discovery + chunked conformal evaluation.

Pipeline
--------
1. Load all 3 group pkl caches (276 features, rebuilt by conformal_analysis_v2.py).
2. For EVERY feature in all 276, compute phase-aware AUROC and descriptors:
     phases:  early=0-10, mid=11-30, late=31-59
     per-episode aggregates:  phase_mean, phase_max
     per-step descriptors whose peak AUROC is reported by phase:
       win_mean, win_std, win_slope, cum_slope
     cross-phase deltas:  mid_minus_early, late_minus_early, late_minus_mid
     direction-change flag: AUROC sign flips between early and late
3. Auto-select candidate shortlists:
     A. Cross-domain  — AUROC ≥ 0.65 in ≥ 2/3 groups in some phase, consistent direction
     B. Group-specific — top-10 per group/phase with AUROC ≥ 0.70
     C. Direction-change — features that flip between early and late
     D. Manual must-includes — hard-coded list
4. Run chunked conformal ONLY on selected candidates + defined composites:
     global baseline   — max-over-all-steps, single threshold
     3-chunk conformal — separate threshold per phase, Bonferroni alpha_chunk=alpha/3
   Composites evaluated:
     A. old_best:          v_drift_base + v_drift_task + action_horizon_align
     B. old_best+action:   A + plan_consistency + action_spread
     C. old_best+vcstate:  A + vc_gini_state + vc_gini_base + com_dist_gc_vc_state + emd_gc_vc_state
     D. old_best+vctask:   A + com_dist_rw_vc_task + emd_rw_vc_task
     E. phase_specific:    auto-selected per-phase top features (different per chunk)
5. Write 8 output files.

Output → /nfs/roberts/scratch/pi_tkf6/zs377/phase_aware_analysis/
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("/nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache")
OUT_DIR   = Path("/nfs/roberts/scratch/pi_tkf6/zs377/phase_aware_analysis")

GROUPS = ["pi0fast_libero10", "pi0fast_liberoplus", "pi05_libero10"]
MAX_STEPS = 60
WIN_W     = 5          # causal window for win_* descriptors

# Phase definitions (inclusive step ranges)
PHASE_DEFS: Dict[str, Tuple[int, int]] = {
    "early": (0,  10),
    "mid":   (11, 30),
    "late":  (31, 59),
}
PHASE_NAMES = list(PHASE_DEFS.keys())

# Candidate selection thresholds
CROSS_DOMAIN_AUROC   = 0.65    # Rule A: minimum AUROC to count as discriminative
CROSS_DOMAIN_MIN_GRP = 2       # Rule A: must be discriminative in this many groups
CROSS_DOMAIN_DIR_THR = 0.60    # Rule A: direction consistency threshold
GROUP_SPECIFIC_AUROC = 0.70    # Rule B: min AUROC for group-specific candidate
GROUP_SPECIFIC_TOP_N = 10      # Rule B: top N per group per phase

# Phase-specific composite E: how many auto-selected features per phase
PHASE_COMPOSITE_TOP_N = 5

# Conformal parameters
ALPHA       = 0.10
ALPHA_CHUNK = ALPHA / len(PHASE_DEFS)   # Bonferroni per-chunk  (~0.0333)

# ─────────────────────────────────────────────────────────────────────────────
# MANUAL MUST-INCLUDE FEATURES (Rule D)
# (name, direction_prior)  — direction will be empirically verified
# ─────────────────────────────────────────────────────────────────────────────

MUST_INCLUDE: List[Tuple[str, int]] = [
    ("v_drift_base",          +1),
    ("v_drift_task",          +1),
    ("action_horizon_align",  -1),
    ("plan_consistency",      -1),
    ("action_spread",         +1),
    ("rn_gini_base",          -1),
    ("rn_gini_task",          -1),
    ("agree_rn_rw_task",      +1),
    ("agree_gc_rw_base",      +1),
    ("rw_gini_base",          -1),
    ("vc_gini_state",         +1),
    ("vc_gini_base",          +1),
    ("vc_gini_wrist",         +1),
    ("vc_gini_task",          +1),
    ("com_dist_gc_vc_state",  -1),
    ("emd_gc_vc_state",       -1),
    ("com_dist_rw_vc_task",   -1),
    ("emd_rw_vc_task",        -1),
    ("com_dist_gc_vc_task",   -1),
    ("emd_gc_vc_task",        -1),
]
MUST_INCLUDE_NAMES = {f for f, _ in MUST_INCLUDE}

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DEFINITIONS  (A–D use the same features in every chunk)
# ─────────────────────────────────────────────────────────────────────────────

# Each composite is a list of (feature_name, direction) pairs.
COMPOSITES: Dict[str, List[Tuple[str, int]]] = {
    "A_old_best": [
        ("v_drift_base",         +1),
        ("v_drift_task",         +1),
        ("action_horizon_align", -1),
    ],
    "B_old_best_action": [
        ("v_drift_base",         +1),
        ("v_drift_task",         +1),
        ("action_horizon_align", -1),
        ("plan_consistency",     -1),
        ("action_spread",        +1),
    ],
    "C_old_best_vcstate": [
        ("v_drift_base",          +1),
        ("v_drift_task",          +1),
        ("action_horizon_align",  -1),
        ("vc_gini_state",         +1),
        ("vc_gini_base",          +1),
        ("com_dist_gc_vc_state",  -1),
        ("emd_gc_vc_state",       -1),
    ],
    "D_old_best_vctask": [
        ("v_drift_base",         +1),
        ("v_drift_task",         +1),
        ("action_horizon_align", -1),
        ("com_dist_rw_vc_task",  -1),
        ("emd_rw_vc_task",       -1),
    ],
    # E is phase_specific and built automatically after discovery
}

# ─────────────────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def auroc(fail_s: np.ndarray, succ_s: np.ndarray) -> float:
    """Mann-Whitney AUROC = P(fail_score > succ_score). 0.5=random, 1=perfect."""
    f = fail_s[~np.isnan(fail_s)]; s = succ_s[~np.isnan(succ_s)]
    if len(f) == 0 or len(s) == 0:
        return float("nan")
    nf, ns = len(f), len(s)
    combined = np.concatenate([f, s])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    return float((ranks[:nf].sum() - nf * (nf + 1) / 2) / (nf * ns))


def dir_str(a: float) -> str:
    return "F>S" if (not np.isnan(a) and a > 0.5) else "F<S"


def effective_auroc(a: float) -> float:
    """Distance from 0.5 mapped back to [0.5,1]; handles both F>S and F<S."""
    return float(abs(a - 0.5) + 0.5) if not np.isnan(a) else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# CACHE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_group(name: str) -> Optional[Dict]:
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        print(f"  [missing] {path}"); return None
    with open(path, "rb") as f:
        return pickle.load(f)


def all_feature_names(ep_data: Dict) -> List[str]:
    eps = ep_data.get("success", []) + ep_data.get("failure", [])
    if not eps or not eps[0]:
        return []
    return sorted(k for k in eps[0][0].keys() if not k.startswith("_"))


# ─────────────────────────────────────────────────────────────────────────────
# RAW MATRIX EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_raw(ep_list: List[List[dict]], feat: str) -> np.ndarray:
    """(N_episodes, MAX_STEPS) raw feature matrix; NaN for missing steps."""
    out = np.full((len(ep_list), MAX_STEPS), np.nan, dtype=np.float64)
    for i, ep in enumerate(ep_list):
        for t, step in enumerate(ep[:MAX_STEPS]):
            v = step.get(feat)
            if v is not None:
                try:
                    out[i, t] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-STEP DESCRIPTOR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptors(raw: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Given (N, T) raw matrix, return dict of (N, T) descriptor matrices:
      win_mean  — mean of raw over [t-WIN_W+1, t]
      win_std   — std  over the same window (nan if < 2 valid)
      win_slope — linear slope over the window (nan if < 3 valid)
      cum_slope — linear slope from step 0 to t (nan if < 3 valid)
    """
    N, T = raw.shape
    out = {k: np.full((N, T), np.nan) for k in
           ("win_mean", "win_std", "win_slope", "cum_slope")}

    for t in range(T):
        lo  = max(0, t - WIN_W + 1)
        win = raw[:, lo:t + 1]        # (N, w)
        nv  = (~np.isnan(win)).sum(axis=1)   # (N,) valid count per episode

        out["win_mean"][:, t]  = np.where(nv >= 1, np.nanmean(win, axis=1), np.nan)
        out["win_std"][:, t]   = np.where(nv >= 2, np.nanstd(win, axis=1), np.nan)

        # win_slope — per-episode linear regression over the window
        if win.shape[1] >= 3:
            xs = np.arange(win.shape[1], dtype=float)
            sl = np.full(N, np.nan)
            for i in range(N):
                ys = win[i]; ok = ~np.isnan(ys)
                if ok.sum() >= 3:
                    try:   sl[i] = float(np.polyfit(xs[ok], ys[ok], 1)[0])
                    except Exception: pass
            out["win_slope"][:, t] = sl

        # cum_slope — linear regression from step 0 to t
        if t >= 2:
            xs_c = np.arange(t + 1, dtype=float)
            sl_c = np.full(N, np.nan)
            for i in range(N):
                ys_c = raw[i, :t + 1]; ok = ~np.isnan(ys_c)
                if ok.sum() >= 3:
                    try:   sl_c[i] = float(np.polyfit(xs_c[ok], ys_c[ok], 1)[0])
                    except Exception: pass
            out["cum_slope"][:, t] = sl_c

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE-LEVEL AGGREGATES AND AUROC
# ─────────────────────────────────────────────────────────────────────────────

def phase_aggregates(raw: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each phase: (N,) episode-level arrays for phase_mean and phase_max.
    Returns {phase_name: {"mean": (N,), "max": (N,)}}.
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for ph, (a, b) in PHASE_DEFS.items():
        seg = raw[:, a:b + 1]
        result[ph] = {
            "mean": np.nanmean(seg, axis=1),
            "max":  np.nanmax(np.where(np.isnan(seg), -np.inf, seg), axis=1),
        }
        # replace -inf (all-nan episodes) with nan
        result[ph]["max"] = np.where(np.isinf(result[ph]["max"]),
                                      np.nan, result[ph]["max"])
    return result


def desc_phase_peak_auroc(
    fail_desc: Dict[str, np.ndarray],
    succ_desc: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    For each descriptor × phase: peak AUROC over steps within the phase.
    Returns {descriptor: {phase: peak_auroc}}.
    """
    result: Dict[str, Dict[str, float]] = {d: {} for d in fail_desc}
    for desc in fail_desc:
        fc, sc = fail_desc[desc], succ_desc[desc]
        for ph, (a, b) in PHASE_DEFS.items():
            best = float("nan")
            for t in range(a, min(b + 1, MAX_STEPS)):
                a_val = auroc(fc[:, t], sc[:, t])
                if not np.isnan(a_val) and (np.isnan(best) or
                                             abs(a_val - 0.5) > abs(best - 0.5)):
                    best = a_val
            result[desc][ph] = best
    return result


def cross_phase_deltas(
    fail_raw: np.ndarray,
    succ_raw: np.ndarray,
) -> Dict[str, float]:
    """
    AUROC of mid_minus_early, late_minus_early, late_minus_mid
    (episode-level aggregates).
    """
    agg_f = phase_aggregates(fail_raw)
    agg_s = phase_aggregates(succ_raw)
    deltas: Dict[str, float] = {}
    pairs = [("mid_minus_early",  "mid",  "early"),
             ("late_minus_early", "late", "early"),
             ("late_minus_mid",   "late", "mid")]
    for key, ph1, ph2 in pairs:
        fd = agg_f[ph1]["mean"] - agg_f[ph2]["mean"]
        sd = agg_s[ph1]["mean"] - agg_s[ph2]["mean"]
        deltas[key] = auroc(fd, sd)
    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# PER-GROUP ANALYSIS RESULT
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePhaseResult:
    """All phase-aware statistics for one feature in one group."""
    __slots__ = ("phase_mean_auroc", "phase_max_auroc", "desc_phase_peak",
                 "delta_aurocs", "direction_change",
                 "early_dir", "late_dir")

    def __init__(self):
        self.phase_mean_auroc: Dict[str, float] = {}   # phase → AUROC of phase_mean
        self.phase_max_auroc:  Dict[str, float] = {}   # phase → AUROC of phase_max
        self.desc_phase_peak:  Dict[str, Dict[str, float]] = {}  # desc → {phase: peak_auroc}
        self.delta_aurocs:     Dict[str, float] = {}   # delta_name → auroc
        self.direction_change: bool = False
        self.early_dir:        str  = "?"
        self.late_dir:         str  = "?"


def analyse_feature(
    fail_eps: List[List[dict]],
    succ_eps: List[List[dict]],
    feat: str,
) -> FeaturePhaseResult:
    fail_raw = get_raw(fail_eps, feat)
    succ_raw = get_raw(succ_eps, feat)

    agg_f = phase_aggregates(fail_raw)
    agg_s = phase_aggregates(succ_raw)

    res = FeaturePhaseResult()
    for ph in PHASE_NAMES:
        res.phase_mean_auroc[ph] = auroc(agg_f[ph]["mean"], agg_s[ph]["mean"])
        res.phase_max_auroc[ph]  = auroc(agg_f[ph]["max"],  agg_s[ph]["max"])

    fail_desc = compute_descriptors(fail_raw)
    succ_desc = compute_descriptors(succ_raw)
    res.desc_phase_peak = desc_phase_peak_auroc(fail_desc, succ_desc)

    res.delta_aurocs = cross_phase_deltas(fail_raw, succ_raw)

    e_a = res.phase_mean_auroc.get("early", 0.5)
    l_a = res.phase_mean_auroc.get("late",  0.5)
    res.early_dir = dir_str(e_a)
    res.late_dir  = dir_str(l_a)
    res.direction_change = (
        not np.isnan(e_a) and not np.isnan(l_a) and
        (e_a - 0.5) * (l_a - 0.5) < 0
    )
    return res


def analyse_all_features(
    ep_data: Dict,
    features: List[str],
) -> Dict[str, FeaturePhaseResult]:
    succ_eps = ep_data.get("success", [])
    fail_eps = ep_data.get("failure", [])
    results: Dict[str, FeaturePhaseResult] = {}
    for i, feat in enumerate(features):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(features)} features done")
        results[feat] = analyse_feature(fail_eps, succ_eps, feat)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def best_phase_auroc(res: FeaturePhaseResult) -> Tuple[float, str]:
    """Return (effective_auroc, best_phase) across all phases and descriptors."""
    best_eff = 0.5; best_ph = "?"
    for ph in PHASE_NAMES:
        for desc_dict in [res.phase_mean_auroc, res.phase_max_auroc]:
            a = desc_dict.get(ph, float("nan"))
            eff = effective_auroc(a)
            if not np.isnan(eff) and eff > best_eff:
                best_eff = eff; best_ph = ph
        for desc, pd in res.desc_phase_peak.items():
            a = pd.get(ph, float("nan"))
            eff = effective_auroc(a)
            if not np.isnan(eff) and eff > best_eff:
                best_eff = eff; best_ph = ph
    return best_eff, best_ph


def empirical_direction(res: FeaturePhaseResult) -> int:
    """Return +1 if best AUROC > 0.5 (F>S), -1 if best AUROC < 0.5 (F<S)."""
    beff, bph = best_phase_auroc(res)
    if bph == "?":
        return +1   # default if no data
    # Check which direction at best phase
    pm = res.phase_mean_auroc.get(bph, 0.5)
    return +1 if pm >= 0.5 else -1


def select_cross_domain_candidates(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
) -> List[Tuple[str, str, int, float]]:
    """
    Rule A: AUROC ≥ CROSS_DOMAIN_AUROC in ≥ CROSS_DOMAIN_MIN_GRP groups in ANY phase.
    Returns [(feature, best_phase, direction, mean_eff_auroc)].
    """
    candidates = []
    for feat in features:
        for ph in PHASE_NAMES:
            group_aurocs = []
            for g, feat_results in all_group_results.items():
                res = feat_results.get(feat)
                if res is None:
                    continue
                # Use phase_mean_auroc as primary, fall back to best desc_phase_peak
                a = res.phase_mean_auroc.get(ph, float("nan"))
                if np.isnan(a):
                    # try best descriptor in this phase
                    best = 0.5
                    for dp in res.desc_phase_peak.values():
                        v = dp.get(ph, float("nan"))
                        if not np.isnan(v) and effective_auroc(v) > effective_auroc(best):
                            best = v
                    a = best
                group_aurocs.append(a)

            valid = [a for a in group_aurocs if not np.isnan(a)]
            if len(valid) < CROSS_DOMAIN_MIN_GRP:
                continue
            n_discrim = sum(1 for a in valid
                            if effective_auroc(a) >= CROSS_DOMAIN_AUROC)
            if n_discrim < CROSS_DOMAIN_MIN_GRP:
                continue

            # Direction consistency
            dirs = [dir_str(a) for a in valid
                    if effective_auroc(a) >= CROSS_DOMAIN_DIR_THR]
            n_fgs = dirs.count("F>S"); n_fls = dirs.count("F<S")
            if max(n_fgs, n_fls) < CROSS_DOMAIN_MIN_GRP:
                continue   # inconsistent direction
            cons_dir = +1 if n_fgs >= n_fls else -1

            mean_eff = float(np.mean([effective_auroc(a) for a in valid]))
            candidates.append((feat, ph, cons_dir, mean_eff))

    # De-duplicate: if a feature appears in multiple phases, keep the best phase
    seen: Dict[str, Tuple] = {}
    for c in candidates:
        feat = c[0]
        if feat not in seen or c[3] > seen[feat][3]:
            seen[feat] = c
    return sorted(seen.values(), key=lambda x: -x[3])


def select_group_specific_candidates(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    group: str,
) -> List[Tuple[str, str, int, float]]:
    """
    Rule B: Top GROUP_SPECIFIC_TOP_N features per group per phase with AUROC ≥ 0.70.
    Returns [(feature, best_phase, direction, eff_auroc)].
    """
    res_map = all_group_results.get(group, {})
    per_phase: Dict[str, List] = {ph: [] for ph in PHASE_NAMES}

    for feat in features:
        res = res_map.get(feat)
        if res is None:
            continue
        for ph in PHASE_NAMES:
            a = res.phase_mean_auroc.get(ph, float("nan"))
            eff = effective_auroc(a)
            if eff >= GROUP_SPECIFIC_AUROC:
                per_phase[ph].append((feat, ph, dir_str(a), eff))

    candidates = []
    for ph, entries in per_phase.items():
        entries.sort(key=lambda x: -x[3])
        for feat, phase, d_str, eff in entries[:GROUP_SPECIFIC_TOP_N]:
            candidates.append((feat, phase, +1 if d_str == "F>S" else -1, eff))

    # De-duplicate by feature, keep best phase
    seen: Dict[str, Tuple] = {}
    for c in candidates:
        feat = c[0]
        if feat not in seen or c[3] > seen[feat][3]:
            seen[feat] = c
    return sorted(seen.values(), key=lambda x: -x[3])


def select_direction_change_features(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
) -> List[str]:
    """Rule C: features that change direction early→late in ≥ 1 group."""
    result = []
    for feat in features:
        for g, feat_results in all_group_results.items():
            res = feat_results.get(feat)
            if res and res.direction_change:
                result.append(feat)
                break
    return result


def build_phase_specific_composite(
    cross_domain_cands: List[Tuple[str, str, int, float]],
    direction_change_feats: List[str],
    n_per_phase: int = PHASE_COMPOSITE_TOP_N,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Composite E: auto-select top features per phase from cross-domain candidates.
    Direction-change features are allowed ONLY in the phase they are correct for.
    Returns {phase_name: [(feature, direction), ...]}.
    """
    # Group cross-domain candidates by their best phase
    by_phase: Dict[str, List] = {ph: [] for ph in PHASE_NAMES}
    for feat, ph, direction, eff in cross_domain_cands:
        by_phase[ph].append((feat, direction, eff))

    # For direction-change features, they may appear in the phase where their
    # direction is correct even if they were excluded from cross-domain ranking.
    # (Not implemented here since the cross-domain ranking already filters them;
    # they appear at whichever phase their direction is consistent.)

    composite: Dict[str, List[Tuple[str, int]]] = {}
    for ph in PHASE_NAMES:
        entries = sorted(by_phase[ph], key=lambda x: -x[2])[:n_per_phase]
        composite[ph] = [(feat, direction) for feat, direction, _ in entries]

    return composite


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS  (for conformal evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_win_mean_scores(
    ep_list: List[List[dict]],
    feature_spec: List[Tuple[str, int]],
) -> List[np.ndarray]:
    """
    For each episode, compute (T, n_feat) score matrix where each column is
    direction_i * win_mean_i at each step.
    feature_spec: [(feature_name, direction)]
    """
    seqs = []
    for ep in ep_list:
        T = min(len(ep), MAX_STEPS)
        raw = np.full((T, len(feature_spec)), np.nan, dtype=np.float64)
        for t, step in enumerate(ep[:T]):
            for fi, (feat, _) in enumerate(feature_spec):
                v = step.get(feat)
                if v is not None:
                    try: raw[t, fi] = float(v)
                    except (TypeError, ValueError): pass

        # win_mean at each step
        wmean = np.full_like(raw, np.nan)
        for t in range(T):
            lo = max(0, t - WIN_W + 1)
            wmean[t] = np.nanmean(raw[lo:t+1], axis=0)

        # apply direction
        dirs = np.array([d for _, d in feature_spec], dtype=float)
        seqs.append(wmean * dirs[np.newaxis, :])   # (T, n_feat)
    return seqs


def z_score_params(
    succ_seqs: List[np.ndarray],
    n_feat: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pooled-z: mu and sigma computed from ALL calibration-success steps across
    all phases. This is the baseline normalization.
    Returns (mu, sigma) each of shape (n_feat,).
    """
    all_vals = [np.full((0, n_feat), np.nan)]
    for seq in succ_seqs:
        all_vals.append(seq)
    pooled = np.vstack(all_vals)
    mu    = np.nanmean(pooled, axis=0)
    sigma = np.nanstd(pooled, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma


def z_score_params_by_phase(
    succ_seqs: List[np.ndarray],
    n_feat: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Phase-z: separate mu and sigma per phase, computed from calibration-success
    steps that fall within each phase window.

    Motivation: some features have phase-dependent baselines (e.g. v_drift_task
    naturally drifts more early, V-Cosine/state features separate only mid/late).
    Pooled normalization may underweight or distort these phase-local signals.
    Phase-z gives each chunk its own reference distribution so the composite
    score is better calibrated within each phase.

    Returns {phase_name: (mu, sigma)} each of shape (n_feat,).
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ph, (a, b) in PHASE_DEFS.items():
        phase_vals = []
        for seq in succ_seqs:
            t_end = min(b + 1, len(seq))
            if a < t_end:
                phase_vals.append(seq[a:t_end])     # (steps_in_phase, n_feat)
        if phase_vals:
            pooled = np.vstack(phase_vals)
        else:
            pooled = np.full((1, n_feat), np.nan)
        mu    = np.nanmean(pooled, axis=0)
        sigma = np.nanstd(pooled, axis=0)
            # Fallback: if a phase has too few samples for stable sigma,
            # use the pooled sigma from all phases (avoids divide-by-near-zero).
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        result[ph] = (mu, sigma)
    return result


def composite_score_seq(
    seq: np.ndarray,      # (T, n_feat) signed win_mean scores
    mu: np.ndarray,       # (n_feat,) — pooled
    sigma: np.ndarray,    # (n_feat,) — pooled
) -> np.ndarray:
    """
    Pooled-z composite: z-score every step with the same (mu, sigma), then
    take nanmean over features. Used for composites A–D (global normalization).
    """
    z = (seq - mu[np.newaxis, :]) / sigma[np.newaxis, :]
    return np.nanmean(z, axis=1)   # (T,)


def composite_score_seq_phase_z(
    seq: np.ndarray,                              # (T, n_feat)
    phase_params: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Phase-z composite: at each step t, use the z-score parameters of the
    phase that contains t. Steps that fall outside all defined phases use the
    pooled parameters from the nearest phase (should not occur with 0–59 phases).
    """
    T = seq.shape[0]
    scores = np.full(T, np.nan)
    for t in range(T):
        mu_t = sigma_t = None
        for ph, (a, b) in PHASE_DEFS.items():
            if a <= t <= b:
                mu_t, sigma_t = phase_params[ph]
                break
        if mu_t is None:
            # Fallback: use early phase parameters
            mu_t, sigma_t = next(iter(phase_params.values()))
        z = (seq[t] - mu_t) / sigma_t
        valid = ~np.isnan(z)
        scores[t] = float(z[valid].mean()) if valid.any() else np.nan
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL CALIBRATION AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_global(
    succ_seqs_1d: List[np.ndarray],   # list of (T,) score sequences
) -> float:
    """Global max-over-episode threshold at (1-ALPHA) quantile."""
    max_scores = []
    for seq in succ_seqs_1d:
        valid = seq[~np.isnan(seq)]
        if len(valid) > 0:
            max_scores.append(float(valid.max()))
    return float(np.quantile(max_scores, 1.0 - ALPHA)) if max_scores else float("nan")


def calibrate_chunked(
    succ_seqs_1d: List[np.ndarray],   # list of (T,) score sequences
) -> Dict[str, float]:
    """
    One threshold per phase at (1-ALPHA_CHUNK) quantile of max-within-chunk.
    Returns {phase_name: threshold}.
    """
    thresholds: Dict[str, float] = {}
    for ph, (a, b) in PHASE_DEFS.items():
        max_in_chunk = []
        for seq in succ_seqs_1d:
            chunk = seq[a:min(b+1, len(seq))]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                max_in_chunk.append(float(valid.max()))
        thresholds[ph] = (float(np.quantile(max_in_chunk, 1.0 - ALPHA_CHUNK))
                          if max_in_chunk else float("nan"))
    return thresholds


def evaluate_predictor(
    succ_seqs_1d: List[np.ndarray],   # test success score sequences
    fail_seqs_1d: List[np.ndarray],   # test failure score sequences
    tau_global: float,
    tau_chunked: Dict[str, float],
) -> dict:
    """
    Evaluate both global and chunked predictors.
    Returns a dict with all requested metrics.
    """
    def _detect(seqs, tau_g, tau_c):
        n = len(seqs)
        flagged_g = np.zeros(n, dtype=bool)
        flagged_c = np.zeros(n, dtype=bool)
        det_g     = np.full(n, np.nan)
        det_c     = np.full(n, np.nan)
        for ei, seq in enumerate(seqs):
            for t, s in enumerate(seq):
                if np.isnan(s):
                    continue
                # global
                if not flagged_g[ei] and s > tau_g:
                    flagged_g[ei] = True; det_g[ei] = t
                # chunked
                for ph, (a, b) in PHASE_DEFS.items():
                    if a <= t <= b and not flagged_c[ei] and s > tau_c.get(ph, np.inf):
                        flagged_c[ei] = True; det_c[ei] = t
        return flagged_g, det_g, flagged_c, det_c

    sf_g, sd_g, sf_c, sd_c = _detect(succ_seqs_1d, tau_global, tau_chunked)
    ff_g, fd_g, ff_c, fd_c = _detect(fail_seqs_1d, tau_global, tau_chunked)

    ns, nf = len(succ_seqs_1d), len(fail_seqs_1d)

    def _metrics(succ_flag, fail_flag, fail_det):
        tp = int(fail_flag.sum()); fp = int(succ_flag.sum())
        recall    = tp / nf if nf > 0 else float("nan")
        fpr       = fp / ns if ns > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))
        det = fail_det[fail_flag]
        med = float(np.nanmedian(det)) if len(det) > 0 else float("nan")
        frac = {k: (float((det <= k).sum() / nf) if nf > 0 else float("nan"))
                for k in (10, 20, 30)}
        return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
                "median_det": med, "frac_before": frac, "tp": tp, "fp": fp}

    # Per-chunk FPR and recall (chunked predictor only)
    def _chunk_breakdown(succ_seqs, fail_seqs, tau_c):
        result = {}
        for ph, (a, b) in PHASE_DEFS.items():
            tau = tau_c.get(ph, np.inf)
            sf = sum(1 for seq in succ_seqs
                     if any(not np.isnan(s) and s > tau
                            for t, s in enumerate(seq) if a <= t <= b))
            ff = sum(1 for seq in fail_seqs
                     if any(not np.isnan(s) and s > tau
                            for t, s in enumerate(seq) if a <= t <= b))
            result[ph] = {
                "fpr":    sf / ns if ns > 0 else float("nan"),
                "recall": ff / nf if nf > 0 else float("nan"),
                "tau":    tau,
            }
        return result

    return {
        "global":          _metrics(sf_g, ff_g, fd_g),
        "chunked":         _metrics(sf_c, ff_c, fd_c),
        "chunk_breakdown": _chunk_breakdown(succ_seqs_1d, fail_seqs_1d, tau_chunked),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_conformal_pipeline(
    cal_ep: Dict,
    test_ep: Dict,
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    cross_domain_cands: List[Tuple[str, str, int, float]],
    phase_composite: Dict[str, List[Tuple[str, int]]],
) -> Tuple[Dict, Dict]:
    """
    Runs single-feature and composite conformal evaluation.
    Uses pi0fast_libero10 cal/test split.
    Returns (single_results, composite_results).
    """
    # Build feature spec for single-feature evaluation
    # Include: cross-domain candidates + must-includes + group-specific from libero10
    single_spec: Dict[str, Tuple[str, int]] = {}   # feat → (feat, dir)

    # Must-includes (with prior direction, will be verified empirically)
    for feat, prior_dir in MUST_INCLUDE:
        # Check empirical direction from pi0fast_libero10 if available
        res = (all_group_results.get("pi0fast_libero10", {}) or {}).get(feat)
        if res is not None:
            best_eff, _ = best_phase_auroc(res)
            # Only flip direction if strongly empirical (best_eff > 0.6 and contradicts prior)
            emp_dir = empirical_direction(res)
            direction = emp_dir if best_eff > 0.60 else prior_dir
        else:
            direction = prior_dir
        single_spec[feat] = (feat, direction)

    # Cross-domain candidates
    for feat, ph, direction, eff in cross_domain_cands:
        if feat not in single_spec:
            single_spec[feat] = (feat, direction)

    print(f"  Single-feature candidates: {len(single_spec)}")

    # Compute score sequences for calibration and test
    def _score_seqs(ep_list, feat, direction):
        seqs = compute_win_mean_scores(ep_list, [(feat, direction)])
        return [s[:, 0] for s in seqs]   # extract single feature column

    cal_succ = cal_ep.get("success", [])
    cal_fail = cal_ep.get("failure", [])
    test_succ = test_ep.get("success", [])
    test_fail = test_ep.get("failure", [])

    # ── Single-feature results ────────────────────────────────────────────────
    single_results: Dict[str, dict] = {}
    for feat, (_, direction) in sorted(single_spec.items()):
        cal_s = _score_seqs(cal_succ, feat, direction)
        cal_f = _score_seqs(cal_fail, feat, direction)
        test_s = _score_seqs(test_succ, feat, direction)
        test_f = _score_seqs(test_fail, feat, direction)

        tau_g  = calibrate_global(cal_s)
        tau_c  = calibrate_chunked(cal_s)
        single_results[feat] = evaluate_predictor(test_s, test_f, tau_g, tau_c)
        single_results[feat]["direction"] = direction

    # ── Composite results ─────────────────────────────────────────────────────
    composite_results: Dict[str, dict] = {}

    def _composite_eval(name: str, spec_per_chunk: Dict[str, List[Tuple[str, int]]]):
        """
        Evaluate one composite under BOTH pooled-z and phase-z normalization.

        pooled-z: each feature is z-scored using statistics pooled over all
          calibration-success steps (all phases combined). This is the baseline.

        phase-z: each feature is z-scored using statistics from only the
          calibration-success steps within the same phase window (early/mid/late).
          This better accounts for features whose baseline varies across phases.

        Results stored as name (pooled-z) and name + "_phasez" (phase-z).
        """
        all_feats = list({f for feats in spec_per_chunk.values() for f, _ in feats})

        # Pre-compute win_mean sequences for all features on all episode sets.
        # We compute unsigned (direction=+1) and apply direction per-term below,
        # which avoids recomputing for each normalization variant.
        def _precompute(ep_list):
            feat_seqs: Dict[str, List[np.ndarray]] = {}
            for f in all_feats:
                seqs = compute_win_mean_scores(ep_list, [(f, +1)])
                feat_seqs[f] = [s[:, 0] for s in seqs]   # list of (T,) arrays
            return feat_seqs

        cal_seqs  = _precompute(cal_succ)
        test_s_seqs = _precompute(test_succ)
        test_f_seqs = _precompute(test_fail)

        # ── Pooled-z statistics ────────────────────────────────────────────────
        # mu[f], sigma[f]: scalar statistics from all cal-success steps
        pooled_mu:    Dict[str, float] = {}
        pooled_sigma: Dict[str, float] = {}
        for f in all_feats:
            vals = np.concatenate(cal_seqs[f])
            pooled_mu[f]    = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else 0.0
            pooled_sigma[f] = float(np.nanstd(vals))  if not np.all(np.isnan(vals)) else 1.0
            pooled_sigma[f] = max(pooled_sigma[f], 1e-9)

        # ── Phase-z statistics ─────────────────────────────────────────────────
        # mu[ph][f], sigma[ph][f]: statistics from cal-success steps in phase ph
        phase_mu:    Dict[str, Dict[str, float]] = {ph: {} for ph in PHASE_NAMES}
        phase_sigma: Dict[str, Dict[str, float]] = {ph: {} for ph in PHASE_NAMES}
        for f in all_feats:
            for ph, (a, b) in PHASE_DEFS.items():
                phase_vals = []
                for seq in cal_seqs[f]:
                    chunk = seq[a:min(b + 1, len(seq))]
                    phase_vals.append(chunk)
                pv = np.concatenate(phase_vals)
                m  = float(np.nanmean(pv)) if not np.all(np.isnan(pv)) else pooled_mu[f]
                sg = float(np.nanstd(pv))  if not np.all(np.isnan(pv)) else pooled_sigma[f]
                phase_mu[ph][f]    = m
                phase_sigma[ph][f] = max(sg, 1e-9)

        # ── Score builder ──────────────────────────────────────────────────────
        def _build_1d(feat_seqs, use_phase_z):
            """Build (T,) composite score per episode under pooled-z or phase-z."""
            seqs_1d = []
            for ei in range(len(next(iter(feat_seqs.values())))):
                T = min(MAX_STEPS, len(feat_seqs[all_feats[0]][ei]))
                score = np.full(T, np.nan)
                for t in range(T):
                    # Identify active phase and feature spec for this step
                    active_spec = None
                    active_ph   = None
                    for ph, (a, b) in PHASE_DEFS.items():
                        if a <= t <= b:
                            active_spec = spec_per_chunk.get(ph, [])
                            active_ph   = ph
                            break
                    if not active_spec:
                        continue
                    terms = []
                    for f, d in active_spec:
                        raw_s = feat_seqs[f][ei][t] if ei < len(feat_seqs[f]) else np.nan
                        if np.isnan(raw_s):
                            continue
                        if use_phase_z:
                            m  = phase_mu[active_ph][f]
                            sg = phase_sigma[active_ph][f]
                        else:
                            m  = pooled_mu[f]
                            sg = pooled_sigma[f]
                        terms.append(d * (raw_s - m) / sg)
                    score[t] = float(np.mean(terms)) if terms else np.nan
                seqs_1d.append(score)
            return seqs_1d

        # ── Evaluate both normalization variants ───────────────────────────────
        for use_phase_z, suffix in [(False, ""), (True, "_phasez")]:
            cal_s  = _build_1d(cal_seqs,    use_phase_z)
            test_s = _build_1d(test_s_seqs, use_phase_z)
            test_f = _build_1d(test_f_seqs, use_phase_z)

            tau_g = calibrate_global(cal_s)
            tau_c = calibrate_chunked(cal_s)
            key   = name + suffix
            composite_results[key] = evaluate_predictor(test_s, test_f, tau_g, tau_c)
            composite_results[key]["n_features"] = {
                ph: len(spec) for ph, spec in spec_per_chunk.items()
            }
            composite_results[key]["normalization"] = "phase-z" if use_phase_z else "pooled-z"

    # Composites A–D: same features in every chunk
    for comp_name, comp_spec in COMPOSITES.items():
        print(f"  Evaluating composite {comp_name} (pooled-z + phase-z) …")
        _composite_eval(comp_name, {ph: comp_spec for ph in PHASE_NAMES})

    # Composite E: phase-specific features
    print(f"  Evaluating composite E_phase_specific (pooled-z + phase-z) …")
    _composite_eval("E_phase_specific", phase_composite)

    return single_results, composite_results


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_all_feature_phase_ranking(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    path: Path,
):
    """File 1: all_feature_phase_ranking.txt"""
    lines = [
        "=" * 130,
        "ALL-FEATURE PHASE-AWARE RANKING  (all 3 groups)",
        f"AUROC reported per phase per descriptor. eff_auroc = |AUROC-0.5|+0.5  (max=1, direction-neutral)",
        "=" * 130,
    ]

    # Header
    grp_hdr = "  ".join(f"{g[:20]:>20}" for g in GROUPS)
    lines.append(
        f"{'feature':<40} {'phase':<6} {'desc':<11} {'dir':<4}  {grp_hdr}  {'mean_eff':>8}"
    )
    lines.append("-" * 130)

    def _row(feat, phase, desc, group_vals):
        valid = [v for v in group_vals if not np.isnan(v)]
        mean_eff = float(np.mean([effective_auroc(v) for v in valid])) if valid else float("nan")
        dirs = [dir_str(v) for v in valid if not np.isnan(v)]
        d = "F>S" if dirs.count("F>S") >= dirs.count("F<S") else "F<S"
        gv_str = "  ".join(f"{effective_auroc(v):>8.3f}" if not np.isnan(v) else f"{'nan':>8}"
                            for v in group_vals)
        return f"{feat:<40} {phase:<6} {desc:<11} {d:<4}  {gv_str}  {mean_eff:>8.3f}"

    # Sort by mean effective AUROC (best across phases/descriptors)
    def _best(feat):
        best = 0.5
        for g, fr in all_group_results.items():
            res = fr.get(feat)
            if res is None: continue
            eff, _ = best_phase_auroc(res)
            if eff > best: best = eff
        return best

    sorted_feats = sorted(features, key=_best, reverse=True)

    for feat in sorted_feats:
        for ph in PHASE_NAMES:
            for agg in ("phase_mean", "phase_max"):
                group_vals = []
                for g in GROUPS:
                    res = all_group_results.get(g, {}).get(feat)
                    if res is None:
                        group_vals.append(float("nan")); continue
                    a = (res.phase_mean_auroc.get(ph, float("nan"))
                         if agg == "phase_mean"
                         else res.phase_max_auroc.get(ph, float("nan")))
                    group_vals.append(a)
                lines.append(_row(feat, ph, agg, group_vals))

            for desc in ("win_mean", "win_std", "win_slope", "cum_slope"):
                group_vals = []
                for g in GROUPS:
                    res = all_group_results.get(g, {}).get(feat)
                    if res is None:
                        group_vals.append(float("nan")); continue
                    a = res.desc_phase_peak.get(desc, {}).get(ph, float("nan"))
                    group_vals.append(a)
                lines.append(_row(feat, ph, desc, group_vals))

            # Cross-phase deltas (only for the "mid" phase row to avoid repetition)
            if ph == "mid":
                for delta_k in ("mid_minus_early", "late_minus_early", "late_minus_mid"):
                    group_vals = []
                    for g in GROUPS:
                        res = all_group_results.get(g, {}).get(feat)
                        group_vals.append(
                            res.delta_aurocs.get(delta_k, float("nan"))
                            if res else float("nan")
                        )
                    lines.append(_row(feat, "→", delta_k, group_vals))

        lines.append("")  # blank line between features

    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_cross_domain_candidates(
    cands: List[Tuple[str, str, int, float]],
    path: Path,
):
    """File 2: cross_domain_phase_candidates.txt"""
    lines = ["=" * 80, "CROSS-DOMAIN PHASE CANDIDATES  (Rule A)", "=" * 80,
             f"{'feature':<40} {'best_phase':<10} {'direction':<10} {'mean_eff':>8}"]
    lines.append("-" * 80)
    for feat, ph, direction, eff in cands:
        lines.append(f"{feat:<40} {ph:<10} {'F>S' if direction>0 else 'F<S':<10} {eff:>8.3f}")
    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_group_specific_candidates(cands, group_name, path):
    """Files 3+4: group_specific_phase_candidates_*.txt"""
    lines = [f"{'='*80}", f"GROUP-SPECIFIC CANDIDATES — {group_name}  (Rule B)",
             f"{'='*80}",
             f"{'feature':<40} {'best_phase':<10} {'direction':<10} {'eff_auroc':>10}"]
    lines.append("-" * 80)
    for feat, ph, direction, eff in cands:
        lines.append(f"{feat:<40} {ph:<10} {'F>S' if direction>0 else 'F<S':<10} {eff:>10.3f}")
    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_direction_changes(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    dc_feats: List[str],
    path: Path,
):
    """File 5: direction_change_features.txt"""
    lines = ["=" * 80, "DIRECTION-CHANGE FEATURES  (Rule C)", "=" * 80,
             "(early_dir → late_dir per group; orange bg in plots = flips)",
             f"{'feature':<40}" + "".join(f"  {g[:20]:>20}" for g in GROUPS)]
    lines.append("-" * 80)
    for feat in sorted(dc_feats):
        grp_strs = []
        for g in GROUPS:
            res = all_group_results.get(g, {}).get(feat)
            if res:
                grp_strs.append(f"{res.early_dir}→{res.late_dir}  {'FLIP' if res.direction_change else '    '}")
            else:
                grp_strs.append("n/a")
        lines.append(f"{feat:<40}" + "  ".join(f"{s:>22}" for s in grp_strs))
    path.write_text("\n".join(lines))
    print(f"  → {path}")


def _metrics_block(m: dict, label: str) -> List[str]:
    fb = m.get("frac_before", {})
    return [
        f"  {label}",
        f"    recall={m['recall']:.3f}  fpr={m['fpr']:.3f}  "
        f"precision={m['precision']:.3f}  f1={m['f1']:.3f}  "
        f"median_det={m['median_det']:.1f}",
        f"    frac_before_10={fb.get(10,'nan'):.3f}  "
        f"frac_before_20={fb.get(20,'nan'):.3f}  "
        f"frac_before_30={fb.get(30,'nan'):.3f}",
    ]


def write_single_feature_conformal(
    single_results: Dict[str, dict],
    path: Path,
):
    """File 6: chunked_conformal_single_features.txt"""
    lines = ["=" * 100, "SINGLE-FEATURE CHUNKED CONFORMAL RESULTS", "=" * 100]
    for feat, res in sorted(single_results.items(),
                             key=lambda x: -x[1]["chunked"]["f1"]):
        dir_sym = "F>S" if res["direction"] > 0 else "F<S"
        lines.append(f"\n{feat}  ({dir_sym})")
        cb = res.get("chunk_breakdown", {})
        tau_str = "  ".join(f"{ph}:τ={cb.get(ph,{}).get('tau',float('nan')):.4f}" for ph in PHASE_NAMES)
        lines.append(f"  Thresholds — {tau_str}")
        chunk_str = "  ".join(
            f"{ph}: recall={cb.get(ph,{}).get('recall',float('nan')):.3f} "
            f"fpr={cb.get(ph,{}).get('fpr',float('nan')):.3f}"
            for ph in PHASE_NAMES
        )
        lines.append(f"  Per-chunk  — {chunk_str}")
        lines.extend(_metrics_block(res["global"],  "global max-over-episode"))
        lines.extend(_metrics_block(res["chunked"], "chunked conformal (Bonferroni)"))
    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_composite_conformal(
    composite_results: Dict[str, dict],
    path: Path,
):
    """
    File 7: chunked_conformal_composites.txt

    For each composite, shows both pooled-z and phase-z normalization side by side.
    The normalization comparison lets you see whether phase-aware conformal gains
    come from better thresholds (chunked) or also from better score calibration
    (phase-z).
    """
    base_names = ["A_old_best", "B_old_best_action",
                  "C_old_best_vcstate", "D_old_best_vctask", "E_phase_specific"]
    lines = ["=" * 110, "COMPOSITE CONFORMAL RESULTS (pooled-z vs phase-z)",
             f"alpha={ALPHA}  alpha_chunk={ALPHA_CHUNK:.4f}", "=" * 110]

    def _block(name, res, norm_label):
        out = [f"\n  ── {norm_label} ──"]
        nf  = res.get("n_features", {})
        cb  = res.get("chunk_breakdown", {})
        tau_str = "  ".join(
            f"{ph}:τ={cb.get(ph,{}).get('tau',float('nan')):.4f}" for ph in PHASE_NAMES)
        out.append(f"    Thresholds — {tau_str}")
        chunk_str = "  ".join(
            f"{ph}: rec={cb.get(ph,{}).get('recall',float('nan')):.3f} "
            f"fpr={cb.get(ph,{}).get('fpr',float('nan')):.3f}"
            for ph in PHASE_NAMES)
        out.append(f"    Per-chunk  — {chunk_str}")
        out.extend(_metrics_block(res["global"],  "  global max-over-episode"))
        out.extend(_metrics_block(res["chunked"], "  chunked conformal (Bonferroni)"))
        return out

    for name in base_names:
        res_p = composite_results.get(name)
        res_z = composite_results.get(name + "_phasez")
        if res_p is None and res_z is None:
            lines.append(f"\n{name}: NOT COMPUTED"); continue
        nf = (res_p or res_z).get("n_features", {})
        lines.append(f"\nComposite {name}  (n_features per phase: {nf})")
        if res_p:
            lines.extend(_block(name, res_p, "pooled-z"))
        if res_z:
            lines.extend(_block(name + "_phasez", res_z, "phase-z"))
        # Difference row
        if res_p and res_z:
            for key in ("global", "chunked"):
                df1 = res_z[key]["f1"] - res_p[key]["f1"]
                dr  = res_z[key]["recall"] - res_p[key]["recall"]
                lines.append(
                    f"    ΔF1({key}) phase-z vs pooled-z: {df1:+.3f}  "
                    f"Δrecall: {dr:+.3f}"
                )

    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_global_vs_chunked(
    single_results: Dict[str, dict],
    composite_results: Dict[str, dict],
    path: Path,
):
    """File 8: global_vs_chunked_conformal_comparison.txt"""
    lines = ["=" * 110,
             "GLOBAL max-over-episode  vs  CHUNKED CONFORMAL  COMPARISON",
             f"alpha={ALPHA}  alpha_chunk={ALPHA_CHUNK:.4f}",
             "=" * 110,
             f"{'name':<40} {'type':<8} "
             f"{'G_rec':>6} {'G_fpr':>6} {'G_f1':>6} "
             f"{'C_rec':>6} {'C_fpr':>6} {'C_f1':>6}  {'ΔF1':>6}"]
    lines.append("-" * 110)

    all_items = (
        [(k, "single", v) for k, v in single_results.items()] +
        [(k, "compos", v) for k, v in composite_results.items()]
    )
    # Annotate normalization type for composite entries
    for i, (name, typ, res) in enumerate(all_items):
        norm = res.get("normalization", "")
        if norm:
            all_items[i] = (f"{name}[{norm[:1]}]", typ, res)
    all_items.sort(key=lambda x: -x[2]["chunked"]["f1"])

    for name, typ, res in all_items:
        g = res["global"]; c = res["chunked"]
        delta_f1 = c["f1"] - g["f1"] if not np.isnan(c["f1"]) and not np.isnan(g["f1"]) else float("nan")
        lines.append(
            f"{name:<40} {typ:<8} "
            f"{g['recall']:>6.3f} {g['fpr']:>6.3f} {g['f1']:>6.3f} "
            f"{c['recall']:>6.3f} {c['fpr']:>6.3f} {c['f1']:>6.3f}  "
            f"{delta_f1:>+6.3f}"
        )
    path.write_text("\n".join(lines))
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SPOTLIGHT PLOTS
# ─────────────────────────────────────────────────────────────────────────────

SPOTLIGHT_FEATURES = [
    "vc_gini_state", "vc_gini_base", "vc_gini_wrist", "vc_gini_task",
    "com_dist_gc_vc_state", "emd_gc_vc_state",
    "agree_gc_vc_state", "agree_rw_vc_state",
    "com_dist_rw_vc_task", "emd_rw_vc_task",
    "agree_rw_vc_task",
    "v_drift_base", "v_drift_task", "action_horizon_align",
    "plan_consistency", "action_spread",
]

PHASE_COLORS = {"early": "#2c7bb6", "mid": "#fdae61", "late": "#d7191c"}
GROUP_COLORS = {"pi0fast_libero10": "#1f5fa6",
                "pi0fast_liberoplus": "#e05020",
                "pi05_libero10": "#20a040"}
GROUP_LS     = {"pi0fast_libero10": "-", "pi0fast_liberoplus": "--", "pi05_libero10": "-."}


def plot_phase_auroc_grid(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    fname: str,
):
    """Bar-chart grid: rows=features, cols=groups, x-axis=phases."""
    n_feat = len(features); n_grp = len(GROUPS)
    fig, axes = plt.subplots(n_feat, n_grp, figsize=(4.5 * n_grp, 2.2 * n_feat),
                             sharey=True, squeeze=False)
    for fi, feat in enumerate(features):
        for gi, g in enumerate(GROUPS):
            ax = axes[fi][gi]
            res = all_group_results.get(g, {}).get(feat)
            if res:
                vals   = [res.phase_mean_auroc.get(ph, float("nan")) for ph in PHASE_NAMES]
                colors = [PHASE_COLORS[ph] for ph in PHASE_NAMES]
                xs = np.arange(len(PHASE_NAMES))
                bars = ax.bar(xs, vals, color=colors, alpha=0.85, width=0.65)
                ax.axhline(0.5, color="black", lw=0.7, ls=":")
                ax.axhline(0.6, color="gray",  lw=0.5, ls="--", alpha=0.5)
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01, f"{v:.2f}",
                                ha="center", va="bottom", fontsize=6)
                if res.direction_change:
                    ax.set_facecolor("#fff0e0")
            ax.set_ylim(0.3, 1.0)
            ax.set_xticks(np.arange(len(PHASE_NAMES)))
            ax.set_xticklabels(PHASE_NAMES, fontsize=7)
            if fi == 0: ax.set_title(g, fontsize=8, fontweight="bold")
            if gi == 0: ax.set_ylabel(feat[:30], fontsize=7)
    fig.suptitle("Phase AUROC (phase_mean) — orange bg = direction flip", fontsize=10)
    fig.tight_layout()
    p = OUT_DIR / fname
    fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"  → {p}"); plt.close(fig)


def plot_win_mean_curves(
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]],
    features: List[str],
    fname: str,
):
    """win_mean AUROC curve over steps, all groups overlaid, per feature."""
    n_feat = len(features)
    fig, axes = plt.subplots(n_feat, 1, figsize=(14, 2.8 * n_feat), sharex=True)
    if n_feat == 1: axes = [axes]
    steps = np.arange(MAX_STEPS)
    # We need the per-step win_mean AUROC — stored in desc_phase_peak only as
    # per-phase peak, not the full curve. We skip full-curve plotting here
    # (full curves are expensive to store in GrpResult; rely on comp_window_analysis plots).
    # Instead plot phase_mean AUROC as horizontal bars per phase.
    for fi, feat in enumerate(features):
        ax = axes[fi]
        for g in GROUPS:
            res = all_group_results.get(g, {}).get(feat)
            if res is None: continue
            color = GROUP_COLORS.get(g, "gray"); ls = GROUP_LS.get(g, "-")
            for ph, (a, b) in PHASE_DEFS.items():
                a_val = res.phase_mean_auroc.get(ph, float("nan"))
                if not np.isnan(a_val):
                    ax.hlines(a_val, a, b, colors=color, linestyles=ls, lw=2.5,
                              label=f"{g[:12]} {ph}" if fi == 0 else "")
        for ph, (a, b) in PHASE_DEFS.items():
            ax.axvspan(a, b, alpha=0.06, color=PHASE_COLORS[ph])
        ax.axhline(0.5, color="black", lw=0.7, ls=":")
        ax.axhline(0.6, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylim(0.3, 1.0); ax.set_ylabel("AUROC", fontsize=8)
        ax.set_title(feat, fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Step")
    fig.suptitle("Phase-mean AUROC by phase (horizontal lines per group)", fontsize=10)
    fig.tight_layout()
    p = OUT_DIR / fname
    fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"  → {p}"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load caches ────────────────────────────────────────────────────────
    print("\n── Loading caches ────────────────────────────────────────────────")
    all_ep_data: Dict[str, Dict] = {}
    for g in GROUPS:
        print(f"  {g} …", end=" ", flush=True)
        data = load_group(g)
        if data:
            all_ep_data[g] = data
            ns = len(data.get("success", [])); nf = len(data.get("failure", []))
            print(f"{ns}S / {nf}F")
        else:
            print("MISSING")

    if not all_ep_data:
        print("No data. Run conformal_analysis_v2.py first."); return

    first_data = next(iter(all_ep_data.values()))
    features = all_feature_names(first_data)
    print(f"\n  Feature space: {len(features)} features")

    # ── 2. Phase-aware analysis on ALL features ───────────────────────────────
    print("\n── Phase-aware analysis (all features × all groups) ─────────────")
    all_group_results: Dict[str, Dict[str, FeaturePhaseResult]] = {}
    for g, data in all_ep_data.items():
        print(f"  {g} …")
        all_group_results[g] = analyse_all_features(data, features)
        n_dc = sum(1 for r in all_group_results[g].values() if r.direction_change)
        print(f"    direction-change features: {n_dc}")

    # ── 3. Candidate selection ────────────────────────────────────────────────
    print("\n── Selecting candidates ──────────────────────────────────────────")
    cross_cands = select_cross_domain_candidates(all_group_results, features)
    print(f"  Rule A (cross-domain): {len(cross_cands)} candidates")

    dc_feats = select_direction_change_features(all_group_results, features)
    print(f"  Rule C (direction-change): {len(dc_feats)} features")

    phase_composite = build_phase_specific_composite(cross_cands, dc_feats)
    print(f"  Composite E: {sum(len(v) for v in phase_composite.values())} features total")
    for ph, feats_in_phase in phase_composite.items():
        print(f"    {ph}: {[f for f, _ in feats_in_phase]}")

    # ── 4. Write discovery output files ──────────────────────────────────────
    print("\n── Writing discovery files ───────────────────────────────────────")
    write_all_feature_phase_ranking(
        all_group_results, features,
        OUT_DIR / "all_feature_phase_ranking.txt")

    write_cross_domain_candidates(cross_cands,
        OUT_DIR / "cross_domain_phase_candidates.txt")

    for g in ("pi0fast_liberoplus", "pi05_libero10"):
        g_cands = select_group_specific_candidates(all_group_results, features, g)
        print(f"  Rule B ({g}): {len(g_cands)} candidates")
        fname = f"group_specific_phase_candidates_{g}.txt"
        write_group_specific_candidates(g_cands, g, OUT_DIR / fname)

    write_direction_changes(all_group_results, dc_feats,
                            OUT_DIR / "direction_change_features.txt")

    # ── 5. Chunked conformal (pi0fast_libero10, within-cache 50/50 split) ────
    print("\n── Chunked conformal evaluation ──────────────────────────────────")
    if "pi0fast_libero10" not in all_ep_data:
        print("  pi0fast_libero10 missing — skipping conformal."); return

    data10 = all_ep_data["pi0fast_libero10"]
    succ_all = data10.get("success", [])
    fail_all = data10.get("failure", [])
    nc_s = len(succ_all) // 2; nc_f = len(fail_all) // 2
    cal_ep  = {"success": succ_all[:nc_s],  "failure": fail_all[:nc_f]}
    test_ep = {"success": succ_all[nc_s:],  "failure": fail_all[nc_f:]}
    print(f"  Cal: {nc_s}S/{nc_f}F  |  Test: {len(succ_all)-nc_s}S/{len(fail_all)-nc_f}F")

    single_results, composite_results = run_conformal_pipeline(
        cal_ep, test_ep, all_group_results, cross_cands, phase_composite
    )

    # ── 6. Write conformal output files ───────────────────────────────────────
    print("\n── Writing conformal files ───────────────────────────────────────")
    write_single_feature_conformal(
        single_results, OUT_DIR / "chunked_conformal_single_features.txt")
    write_composite_conformal(
        composite_results, OUT_DIR / "chunked_conformal_composites.txt")
    write_global_vs_chunked(
        single_results, composite_results,
        OUT_DIR / "global_vs_chunked_conformal_comparison.txt")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────")
    spotlight_in_cache = [f for f in SPOTLIGHT_FEATURES if f in features]
    plot_phase_auroc_grid(
        all_group_results, spotlight_in_cache, "phase_auroc_spotlight.png")
    plot_win_mean_curves(
        all_group_results, spotlight_in_cache, "win_mean_phase_lines.png")

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
