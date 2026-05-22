#!/usr/bin/env python3
"""
comprehensive_window_analysis.py — Continuous step-by-step sliding window analysis.

Evaluates every feature at EVERY step 0..MAX_STEP (not just 5 checkpoints) using:

  Causal  [t-X+1, t]   X=5   fully usable in real-time conformal prediction
    win_mean      — local level
    win_std       — local volatility
    win_slope     — local trend direction (sign matters)
    d_mean        — level shift from previous window  (acceleration of level)
    d_slope       — slope change from previous window (curvature/acceleration)
    d_slope_mag   — |d_slope|  (magnitude of curvature, direction-agnostic)

  Centered [t-Y, t+Y]  Y=4   retrospective only (requires future data)
    cen_slope     — instantaneous velocity (best offline slope estimate)
    cen_slope_mag — |cen_slope|

Key questions answered:
  1. At what step does each feature FIRST reliably separate success from failure?
  2. Which features are cross-domain consistent at the earliest steps?
  3. Which features are group-specific (pi0fast_liberoplus vs pi05_libero10)?
  4. What is P/R/F1 at the detection point for conformal threshold design?

NOTE on missing features: the v2 cache has 3 attribution methods (gc/ras/rn).
Features using raw-weights-max (rw) and V-cosine (vc) — plus IoU/EMD/COM-dist
agreement metrics from step_metrics_plot.py — are not yet in the cache.
See conformal_analysis_v2.py to add them.

Outputs → /nfs/roberts/scratch/pi_tkf6/zs377/comprehensive_window/
  combined_ranking.txt              earliest cross-domain detection per feature
  group_ranking_pi0fast_liberoplus.txt  group-specific top-40
  group_ranking_pi05_libero10.txt       group-specific top-40
  cross_domain_timeline.txt         n features consistent at each step
  prf1_at_detection.txt             P/R/F1 for top-30 features at detection step
  fig_auroc_<feature>.png           AUROC curves for top-15 cross-domain features
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# ─── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR  = Path("/nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/zs377/comprehensive_window")

N_SAMPLE     = 75
MAX_STEP     = 59          # evaluate at every step 0..59 (cache has up to 60 steps)
CAUSAL_X     = 5           # causal window width: [t-X+1, t]
CENTERED_Y   = 4           # centered half-width: [t-Y, t+Y]; valid t in [Y, MAX_STEP-Y]
AUROC_THRESH = 0.60
CONSIST_MIN  = 2           # groups needed for "cross-domain consistent"
MIN_CONSEC   = 3           # consecutive steps above AUROC_THRESH to call "detected"
TARGET_RECALL = 0.80
SEED         = 42
N_TOP_PRF1   = 30          # compute P/R/F1 for this many top features
N_TOP_PLOT   = 15          # plot AUROC curves for this many features

GROUPS: Dict[str, Path] = {
    "pi0fast_libero10":   CACHE_DIR / "pi0fast_libero10.pkl",
    "pi0fast_liberoplus": CACHE_DIR / "pi0fast_liberoplus.pkl",
    "pi05_libero10":      CACHE_DIR / "pi05_libero10.pkl",
}

# Right-wrist is a duplicated signal (cosine≈0.905 confirmed for single-arm libero).
NOISE_FEATURES: frozenset = frozenset({
    "frac_rwrist", "attn_gini_rwrist",
    "gc_gini_rwrist", "gc_entropy_rwrist", "ras_gini_rwrist", "ras_entropy_rwrist",
    "rn_gini_rwrist", "rn_entropy_rwrist", "gc_max_rwrist", "gc_mean_rwrist",
    "gc_com_y_rwrist", "gc_com_x_rwrist",
    "agree_gc_ras_rwrist", "agree_gc_rn_rwrist", "agree_ras_rn_rwrist",
    "agree_attn_gc_rwrist", "agree_attn_ras_rwrist",
    "agree_base_rwrist_gc", "agree_base_rwrist_ras", "agree_base_rwrist_rn",
    "agree_wrist_rwrist_gc", "agree_wrist_rwrist_ras", "agree_wrist_rwrist_rn",
    "gc_gini_rwrist_t0", "ras_gini_rwrist_t0", "rn_gini_rwrist_t0",
    "gc_gini_rwrist_std", "ras_gini_rwrist_std",
    "horizon_agree_gc_rwrist", "horizon_agree_ras_rwrist",
})
SKIP_FEATURES: frozenset = frozenset({
    "abs_step", "rel_step", "action_plan_entropy",
})

CAUSAL_DESCS  = ["win_mean", "win_std", "win_slope", "d_mean", "d_slope", "d_slope_mag"]
CENTERED_DESCS = ["cen_slope", "cen_slope_mag"]
ALL_DESCS     = CAUSAL_DESCS + CENTERED_DESCS

# Plot style
BLUE, RED = "#1f5fa6", "#b02020"
BLUE_L, RED_L = "#a8c8e8", "#f0a0a0"
GROUP_COLORS = {"pi0fast_libero10": "#1f5fa6", "pi0fast_liberoplus": "#2ca02c", "pi05_libero10": "#d62728"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "legend.fontsize": 8,
})

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_group(path: Path, n_sample: int, seed: int) -> Tuple[List, List]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    rng = np.random.default_rng(seed)
    def _sample(eps):
        n = min(n_sample, len(eps))
        return [eps[i] for i in rng.choice(len(eps), size=n, replace=False)]
    return _sample(data["success"]), _sample(data["failure"])


def build_ts_matrix(episodes: List, feature: str, max_step: int) -> np.ndarray:
    """Return (N_episodes, max_step+1) float64 matrix; NaN where step is missing."""
    T = max_step + 1
    mat = np.full((len(episodes), T), np.nan, dtype=np.float64)
    for i, ep in enumerate(episodes):
        for t in range(min(len(ep), T)):
            v = ep[t].get(feature, np.nan)
            if v is None:
                continue
            try:
                mat[i, t] = float(v)
            except (TypeError, ValueError):
                pass
    return mat


# ─── Vectorized descriptor computation ───────────────────────────────────────

def _seg_slopes(seg: np.ndarray) -> np.ndarray:
    """
    Linear regression slope for each row of seg (N, W).
    Uses the closed-form formula; fast path skips NaN rows.
    """
    N, W = seg.shape
    if W < 3:
        return np.full(N, np.nan)
    pos = np.arange(W, dtype=np.float64)
    mean_pos = pos.mean()
    dpos = pos - mean_pos
    denom = float(np.dot(dpos, dpos))
    if denom < 1e-10:
        return np.full(N, np.nan)

    has_nan = np.isnan(seg).any(axis=1)
    slopes = np.full(N, np.nan)

    ok = ~has_nan
    if ok.any():
        batch = seg[ok]
        dx = batch - batch.mean(axis=1, keepdims=True)
        slopes[ok] = (dx @ dpos) / denom

    for i in np.where(has_nan)[0]:
        row = seg[i]
        valid = ~np.isnan(row)
        if valid.sum() < 3:
            continue
        vp = pos[valid]; vx = row[valid]
        dvp = vp - vp.mean(); dvx = vx - vx.mean()
        dv = float(np.dot(dvp, dvp))
        if dv > 1e-10:
            slopes[i] = float(np.dot(dvp, dvx) / dv)

    return slopes


def desc_at_step(ts_s: np.ndarray, ts_f: np.ndarray, t: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute all descriptors for success and failure matrices at step t.
    Returns dict[desc_name] → (sv, fv) where sv/fv are (N,) arrays.
    """
    T = ts_s.shape[1]
    r: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ── Causal window ────────────────────────────────────────────────────────
    cs = max(0, t - CAUSAL_X + 1)
    cur_s = ts_s[:, cs : t + 1]
    cur_f = ts_f[:, cs : t + 1]

    with np.errstate(all='ignore'):
        win_mean_s = np.nanmean(cur_s, axis=1)
        win_mean_f = np.nanmean(cur_f, axis=1)
        win_std_s  = np.nanstd(cur_s,  axis=1)
        win_std_f  = np.nanstd(cur_f,  axis=1)

    win_slope_s = _seg_slopes(cur_s)
    win_slope_f = _seg_slopes(cur_f)

    r["win_mean"]  = (win_mean_s, win_mean_f)
    r["win_std"]   = (win_std_s, win_std_f)
    r["win_slope"] = (win_slope_s, win_slope_f)

    # ── Previous causal window for delta features ────────────────────────────
    ps_end = cs
    ps_start = max(0, cs - CAUSAL_X)
    if ps_end > ps_start:
        prev_s = ts_s[:, ps_start : ps_end]
        prev_f = ts_f[:, ps_start : ps_end]
        with np.errstate(all='ignore'):
            prev_mean_s = np.nanmean(prev_s, axis=1)
            prev_mean_f = np.nanmean(prev_f, axis=1)
        prev_slope_s = _seg_slopes(prev_s)
        prev_slope_f = _seg_slopes(prev_f)

        d_mean_s = win_mean_s - prev_mean_s
        d_mean_f = win_mean_f - prev_mean_f

        d_slope_s = win_slope_s - prev_slope_s
        d_slope_f = win_slope_f - prev_slope_f
    else:
        nan_s = np.full(len(ts_s), np.nan)
        nan_f = np.full(len(ts_f), np.nan)
        d_mean_s = d_mean_f = nan_s.copy()
        d_slope_s = d_slope_f = nan_f.copy()

    r["d_mean"]      = (d_mean_s, d_mean_f)
    r["d_slope"]     = (d_slope_s, d_slope_f)
    r["d_slope_mag"] = (np.abs(d_slope_s), np.abs(d_slope_f))

    # ── Centered window (retrospective) ─────────────────────────────────────
    if t >= CENTERED_Y and t + CENTERED_Y < T:
        cen_s = ts_s[:, t - CENTERED_Y : t + CENTERED_Y + 1]
        cen_f = ts_f[:, t - CENTERED_Y : t + CENTERED_Y + 1]
        cs_sl = _seg_slopes(cen_s)
        cf_sl = _seg_slopes(cen_f)
        r["cen_slope"]     = (cs_sl, cf_sl)
        r["cen_slope_mag"] = (np.abs(cs_sl), np.abs(cf_sl))
    else:
        nan_s = np.full(len(ts_s), np.nan)
        nan_f = np.full(len(ts_f), np.nan)
        r["cen_slope"]     = (nan_s, nan_f.copy())
        r["cen_slope_mag"] = (nan_s.copy(), nan_f.copy())

    return r


# ─── AUROC helpers ────────────────────────────────────────────────────────────

def auroc_pair(sv: np.ndarray, fv: np.ndarray) -> Tuple[float, float]:
    """Return (auroc_abs, auroc_signed). Positive sign = failures score higher."""
    s = sv[~np.isnan(sv)]; f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan, np.nan
    try:
        stat, _ = mannwhitneyu(s, f, alternative="two-sided")
        u = float(stat) / (len(s) * len(f))
        return max(u, 1.0 - u), u - 0.5
    except Exception:
        return np.nan, np.nan


def prf1_scan(sv: np.ndarray, fv: np.ndarray, direction: str) -> Tuple[float, float, float]:
    s = sv[~np.isnan(sv)]; f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan, np.nan, np.nan
    scores = np.concatenate([s, f])
    labels = np.array([0] * len(s) + [1] * len(f))
    thresholds = np.unique(scores)
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for thresh in thresholds:
        preds = (scores >= thresh) if direction == "+" else (scores <= thresh)
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        fn = int(np.sum(~preds & (labels == 1)))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r
    return best_p, best_r, best_f1


def prf1_at_recall(
    sv: np.ndarray, fv: np.ndarray, direction: str, target: float = TARGET_RECALL
) -> Tuple[float, float, float]:
    s = sv[~np.isnan(sv)]; f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan, np.nan, np.nan
    scores = np.concatenate([s, f])
    labels = np.array([0] * len(s) + [1] * len(f))
    thresholds = np.unique(scores)
    if direction == "+":
        thresholds = thresholds[::-1]
    best_p, best_r, best_f1 = np.nan, np.nan, np.nan
    fb_r, fb_p, fb_f1 = 0.0, 0.0, 0.0
    for thresh in thresholds:
        preds = (scores >= thresh) if direction == "+" else (scores <= thresh)
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        fn = int(np.sum(~preds & (labels == 1)))
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if r > fb_r:
            fb_r, fb_p, fb_f1 = r, p, f1
        if r >= target:
            best_p, best_r, best_f1 = p, r, f1
            break
    if np.isnan(best_r):
        best_p, best_r, best_f1 = fb_p, fb_r, fb_f1
    return best_p, best_r, best_f1


# ─── Main analysis pass ───────────────────────────────────────────────────────

# GrpResults[(feat, desc)] → (auroc_arr, sign_arr) each shape (MAX_STEP+1,)
GrpResults = Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]


def analyse_group(eps_s: List, eps_f: List, features: List[str]) -> GrpResults:
    """Compute AUROC arrays at every step for all (feature, descriptor) pairs."""
    results: GrpResults = {}

    for feat in features:
        ts_s = build_ts_matrix(eps_s, feat, MAX_STEP)  # (N_s, T)
        ts_f = build_ts_matrix(eps_f, feat, MAX_STEP)  # (N_f, T)

        # Collect per-descriptor arrays
        auroc_arrays: Dict[str, np.ndarray] = {d: np.full(MAX_STEP + 1, np.nan) for d in ALL_DESCS}
        sign_arrays:  Dict[str, np.ndarray] = {d: np.full(MAX_STEP + 1, np.nan) for d in ALL_DESCS}

        for t in range(MAX_STEP + 1):
            dvs = desc_at_step(ts_s, ts_f, t)
            for desc in ALL_DESCS:
                sv, fv = dvs[desc]
                a, s = auroc_pair(sv, fv)
                auroc_arrays[desc][t] = a
                sign_arrays[desc][t]  = s

        for desc in ALL_DESCS:
            results[(feat, desc)] = (auroc_arrays[desc], sign_arrays[desc])

    return results


# ─── Detection helpers ────────────────────────────────────────────────────────

def first_detection(auroc_arr: np.ndarray, thresh: float = AUROC_THRESH,
                    min_consec: int = MIN_CONSEC) -> int:
    """
    First step where AUROC >= thresh for min_consec consecutive steps.
    Returns -1 if never detected.
    """
    consec = 0; start = -1
    for t, a in enumerate(auroc_arr):
        if not np.isnan(a) and a >= thresh:
            if consec == 0:
                start = t
            consec += 1
            if consec >= min_consec:
                return start
        else:
            consec = 0; start = -1
    return -1


def peak_info(auroc_arr: np.ndarray) -> Tuple[float, int]:
    """(peak_auroc, peak_step). Returns (nan, -1) if all nan."""
    valid = ~np.isnan(auroc_arr)
    if not valid.any():
        return np.nan, -1
    idx = int(np.nanargmax(auroc_arr))
    return float(auroc_arr[idx]), idx


# ─── Output writers ───────────────────────────────────────────────────────────

def build_combined_ranking(
    all_results: Dict[str, GrpResults], features: List[str]
) -> List[Tuple]:
    """
    For each (feat, desc): find earliest step with AUROC >= threshold in >= CONSIST_MIN groups
    (same direction). Returns list of (feat, desc, earliest_t, n_groups, mean_peak, dir) sorted
    by (n_groups desc, earliest_t asc, mean_peak desc).
    """
    group_names = list(all_results.keys())
    records = []

    for feat in features:
        for desc in ALL_DESCS:
            first_ts, peaks, signs_at_peak = [], [], []
            for g in group_names:
                arr, sarr = all_results[g].get((feat, desc), (np.full(MAX_STEP+1, np.nan),)*2)
                fd = first_detection(arr)
                pk, pi = peak_info(arr)
                first_ts.append(fd)
                peaks.append(pk)
                signs_at_peak.append(float(sarr[pi]) if pi >= 0 and not np.isnan(sarr[pi]) else 0.0)

            detected = [ft for ft in first_ts if ft >= 0]
            if len(detected) < CONSIST_MIN:
                continue

            # Direction consistency among detected groups
            detected_signs = [signs_at_peak[i] for i, ft in enumerate(first_ts) if ft >= 0]
            dirs = [int(np.sign(s)) for s in detected_signs if s != 0]
            same_dir = len(set(dirs)) <= 1 if dirs else False
            if not same_dir:
                continue

            dir_str = "F>S" if (dirs and dirs[0] > 0) else "F<S"
            earliest = min(detected)
            n_groups = len(detected)
            mean_peak = float(np.nanmean(peaks))

            records.append((feat, desc, earliest, n_groups, mean_peak, dir_str,
                            first_ts, peaks))

    records.sort(key=lambda r: (-r[3], r[2], -r[4]))
    return records


def write_combined_ranking(records: List[Tuple], group_names: List[str], path: Path) -> None:
    lines = [
        "Combined cross-domain ranking — earliest detection across groups",
        f"Detection = AUROC >= {AUROC_THRESH} for {MIN_CONSEC}+ consecutive steps",
        f"CONSIST_MIN = {CONSIST_MIN}/3 groups required, same direction",
        f"Total entries: {len(records)}",
        "",
        (f"{'feature':<35} {'desc':<15} {'first_t':>7}  {'n_grp':>5}  "
         f"{'mean_pk':>7}  {'dir':>4}  " +
         "  ".join(f"{g[:12]:>12}" for g in group_names) + "  (first_t)"),
        "-" * 140,
    ]
    for feat, desc, earliest, n_grp, mpk, dir_str, first_ts, peaks in records:
        ft_str = "  ".join(f"{t:>12}" if t >= 0 else f"{'none':>12}" for t in first_ts)
        lines.append(
            f"{feat:<35} {desc:<15} {earliest:>7}  {n_grp:>5}  "
            f"{mpk:>7.3f}  {dir_str:>4}  {ft_str}"
        )
    path.write_text("\n".join(lines))
    print(f"  wrote {path.name} ({len(records)} entries)")


def write_group_ranking(
    all_results: Dict[str, GrpResults], features: List[str],
    group_name: str, path: Path, n_top: int = 40
) -> None:
    """Rank features by earliest first-detection step for a specific group."""
    records = []
    for feat in features:
        best_fd, best_pk, best_desc = MAX_STEP + 1, 0.0, ""
        for desc in ALL_DESCS:
            arr, sarr = all_results[group_name].get((feat, desc), (np.full(MAX_STEP+1, np.nan),)*2)
            fd = first_detection(arr)
            pk, pi = peak_info(arr)
            if fd >= 0 and (fd < best_fd or (fd == best_fd and pk > best_pk)):
                best_fd, best_pk, best_desc = fd, pk, desc
            # Also record direction at peak
            if pi >= 0 and not np.isnan(sarr[pi]):
                dir_sign = sarr[pi]
        if best_fd <= MAX_STEP:
            arr, sarr = all_results[group_name].get((feat, best_desc), (np.full(MAX_STEP+1, np.nan),)*2)
            _, pi = peak_info(arr)
            dir_str = "F>S" if (pi >= 0 and not np.isnan(sarr[pi]) and sarr[pi] > 0) else "F<S"
            records.append((feat, best_desc, best_fd, best_pk, dir_str))

    records.sort(key=lambda r: (r[2], -r[3]))

    lines = [
        f"Feature ranking — {group_name}",
        f"Earliest first-detection step per feature (best descriptor shown)",
        "",
        f"{'feature':<35} {'desc':<15} {'first_t':>7}  {'peak_auroc':>10}  {'dir':>4}",
        "-" * 85,
    ]
    for feat, desc, fd, pk, dir_str in records[:n_top]:
        lines.append(f"{feat:<35} {desc:<15} {fd:>7}  {pk:>10.3f}  {dir_str:>4}")

    path.write_text("\n".join(lines))
    print(f"  wrote {path.name}")


def write_cross_domain_timeline(
    all_results: Dict[str, GrpResults], features: List[str], path: Path
) -> None:
    """At each step t, count features that are cross-domain consistent."""
    group_names = list(all_results.keys())
    lines = [
        "Cross-domain consistency timeline",
        f"AUROC >= {AUROC_THRESH} at step t in >= {CONSIST_MIN}/3 groups, same direction",
        "",
        f"{'t':>3}  {'3/3':>5}  {'2/3':>5}  {'total':>6}  top (3/3) features at this step",
        "-" * 90,
    ]

    for t in range(MAX_STEP + 1):
        three_list, two_list = [], []
        for feat in features:
            for desc in ALL_DESCS:
                aucs, signs = [], []
                for g in group_names:
                    arr, sarr = all_results[g].get((feat, desc), (np.full(MAX_STEP+1, np.nan),)*2)
                    aucs.append(float(arr[t]) if t < len(arr) else np.nan)
                    signs.append(int(np.sign(sarr[t])) if (t < len(sarr) and not np.isnan(sarr[t])) else 0)
                above = sum(1 for a in aucs if not np.isnan(a) and a >= AUROC_THRESH)
                nz = [s for s in signs if s != 0]
                same = len(set(nz)) <= 1 if nz else False
                tag = f"{feat}({desc})"
                if above >= 3 and same:
                    three_list.append(tag)
                elif above >= 2 and same:
                    two_list.append(tag)

        top_str = " | ".join(three_list[:4]) if three_list else "—"
        lines.append(
            f"{t:>3}  {len(three_list):>5}  {len(two_list):>5}  "
            f"{len(three_list)+len(two_list):>6}  {top_str}"
        )

    path.write_text("\n".join(lines))
    print(f"  wrote {path.name}")


def write_prf1(
    all_results: Dict[str, GrpResults], group_episodes: Dict[str, Tuple],
    features: List[str], combined_records: List[Tuple], path: Path
) -> None:
    """Compute P/R/F1 at first-detection step for the top N_TOP_PRF1 cross-domain features."""
    group_names = list(all_results.keys())
    lines = [
        "P/R/F1 at first-detection step — top cross-domain features",
        f"F1@mx  = threshold maximising F1",
        f"F1@r80 = threshold achieving recall >= {TARGET_RECALL:.2f} (conformal alpha=0.20)",
        "",
    ]

    for feat, desc, earliest, n_grp, mpk, dir_str, first_ts, _ in combined_records[:N_TOP_PRF1]:
        lines.append(f"\n{'='*70}")
        lines.append(f"{feat}  [{desc}]  earliest={earliest}  n_groups={n_grp}  {dir_str}")
        lines.append(
            f"  {'group':<25} {'t_eval':>6}  {'AUROC':>6}  "
            f"{'P@mx':>6} {'R@mx':>6} {'F1@mx':>6}  "
            f"{'P@r80':>6} {'R@r80':>6} {'F1@r80':>6}"
        )

        for g, ft in zip(group_names, first_ts):
            t_eval = ft if ft >= 0 else earliest
            eps_s, eps_f = group_episodes[g]

            ts_s = build_ts_matrix(eps_s, feat, MAX_STEP)
            ts_f = build_ts_matrix(eps_f, feat, MAX_STEP)
            dvs = desc_at_step(ts_s, ts_f, t_eval)
            sv, fv = dvs[desc]

            a_abs, a_sgn = auroc_pair(sv, fv)
            direction = "+" if (not np.isnan(a_sgn) and a_sgn >= 0) else "-"

            p_mf, r_mf, f1_mf    = prf1_scan(sv, fv, direction)
            p_r80, r_r80, f1_r80 = prf1_at_recall(sv, fv, direction)

            def _f(v):
                return f"{v:.3f}" if not np.isnan(v) else " nan"

            lines.append(
                f"  {g:<25} {t_eval:>6}  {_f(a_abs):>6}  "
                f"{_f(p_mf):>6} {_f(r_mf):>6} {_f(f1_mf):>6}  "
                f"{_f(p_r80):>6} {_f(r_r80):>6} {_f(f1_r80):>6}"
            )

    path.write_text("\n".join(lines))
    print(f"  wrote {path.name}")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_auroc_curves(
    all_results: Dict[str, GrpResults], group_episodes: Dict[str, Tuple],
    combined_records: List[Tuple], out_dir: Path
) -> None:
    """
    For each of the top N_TOP_PLOT cross-domain features: one figure with
    2 rows — top: mean±CI trajectories; bottom: AUROC curves for all groups.
    """
    group_names = list(all_results.keys())
    steps = np.arange(MAX_STEP + 1)
    thresh_line = np.full(MAX_STEP + 1, AUROC_THRESH)

    seen_feats = set()
    top_entries = []
    for r in combined_records:
        if r[0] not in seen_feats:
            top_entries.append(r)
            seen_feats.add(r[0])
        if len(top_entries) >= N_TOP_PLOT:
            break

    for feat, desc, earliest, n_grp, mpk, dir_str, first_ts, _ in top_entries:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax_traj, ax_auroc = axes

        # ── Row 1: mean trajectories ─────────────────────────────────────────
        for g, col in GROUP_COLORS.items():
            eps_s, eps_f = group_episodes[g]
            ts_s = build_ts_matrix(eps_s, feat, MAX_STEP)
            ts_f = build_ts_matrix(eps_f, feat, MAX_STEP)

            def _ci(mat):
                mu = np.nanmean(mat, axis=0)
                n = np.sum(~np.isnan(mat), axis=0).clip(1)
                sem = np.nanstd(mat, axis=0) / np.sqrt(n)
                return mu, mu - 1.96 * sem, mu + 1.96 * sem

            mu_s, lo_s, hi_s = _ci(ts_s)
            mu_f, lo_f, hi_f = _ci(ts_f)

            col_s = col + "88"; col_f = col + "cc"
            ax_traj.fill_between(steps, lo_s, hi_s, color=col, alpha=0.08)
            ax_traj.fill_between(steps, lo_f, hi_f, color=col, alpha=0.18)
            ax_traj.plot(steps, mu_s, color=col, lw=1.5, ls="--", label=f"{g[:12]} S", alpha=0.8)
            ax_traj.plot(steps, mu_f, color=col, lw=2.0, ls="-",  label=f"{g[:12]} F")

        ax_traj.set_ylabel(feat, fontsize=8)
        ax_traj.set_title(f"{feat}  [{desc}]  {dir_str}  earliest={earliest}", loc="left")
        ax_traj.legend(fontsize=7, ncol=3, loc="upper right")

        # ── Row 2: AUROC curves ──────────────────────────────────────────────
        ax_auroc.plot(steps, thresh_line, color="gray", lw=0.8, ls=":", alpha=0.7, label=f"thresh={AUROC_THRESH}")

        for g, col in GROUP_COLORS.items():
            arr, _ = all_results[g].get((feat, desc), (np.full(MAX_STEP+1, np.nan),)*2)
            ax_auroc.plot(steps, arr, color=col, lw=1.8, label=g)

        if earliest >= 0:
            ax_auroc.axvline(earliest, color="red", lw=1.0, ls="--", alpha=0.6, label=f"first_t={earliest}")

        ax_auroc.set_ylim(0.45, 1.0)
        ax_auroc.set_ylabel("AUROC")
        ax_auroc.set_xlabel("Step")
        ax_auroc.legend(fontsize=7, loc="upper left")

        fig.suptitle(f"Continuous AUROC — {feat}  [{desc}]", fontsize=10, fontweight="bold")
        fig.tight_layout()
        safe = feat.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"fig_auroc_{safe}.png", dpi=120)
        plt.close(fig)
        print(f"    fig_auroc_{safe}.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading group caches ...")
    group_episodes: Dict[str, Tuple[List, List]] = {}
    for gname, gpath in GROUPS.items():
        print(f"  {gname} ...", end=" ", flush=True)
        eps_s, eps_f = load_group(gpath, N_SAMPLE, SEED)
        group_episodes[gname] = (eps_s, eps_f)
        print(f"S={len(eps_s)}, F={len(eps_f)}")

    # Feature universe: all cached features minus right-wrist noise and trivial features
    first_ep = group_episodes["pi0fast_liberoplus"][0][0]
    all_keys = list(first_ep[0].keys())
    features = [k for k in all_keys if k not in NOISE_FEATURES and k not in SKIP_FEATURES]
    print(f"\n{len(features)} features  ×  {len(ALL_DESCS)} descriptors  ×  {MAX_STEP+1} steps  ×  {len(GROUPS)} groups")
    print(f"Causal window X={CAUSAL_X}  |  Centered window Y={CENTERED_Y} (retrospective)")
    print(f"Detection threshold AUROC>={AUROC_THRESH} for {MIN_CONSEC}+ consecutive steps\n")

    # Pass 1 — continuous AUROC arrays for every (feature, descriptor, step)
    all_results: Dict[str, GrpResults] = {}
    for gname, (eps_s, eps_f) in group_episodes.items():
        print(f"Analysing {gname} ...", flush=True)
        all_results[gname] = analyse_group(eps_s, eps_f, features)
        n = len(all_results[gname])
        print(f"  {n} (feature, descriptor) pairs computed")

    # Build combined ranking
    print("\nBuilding combined cross-domain ranking ...")
    combined = build_combined_ranking(all_results, features)
    print(f"  {len(combined)} cross-domain consistent (feat, desc) entries")

    # Write output tables
    print("\nWriting outputs ...")
    group_names = list(all_results.keys())
    write_combined_ranking(combined, group_names, OUTPUT_DIR / "combined_ranking.txt")
    for gname in group_names:
        safe_g = gname.replace("+", "_")
        write_group_ranking(all_results, features, gname,
                            OUTPUT_DIR / f"group_ranking_{gname}.txt")
    write_cross_domain_timeline(all_results, features, OUTPUT_DIR / "cross_domain_timeline.txt")

    # Pass 2 — P/R/F1 for top N_TOP_PRF1 features
    print(f"\nComputing P/R/F1 for top {N_TOP_PRF1} cross-domain features ...")
    write_prf1(all_results, group_episodes, features, combined,
               OUTPUT_DIR / "prf1_at_detection.txt")

    # Plots
    print(f"\nPlotting AUROC curves for top {N_TOP_PLOT} features ...")
    plot_auroc_curves(all_results, group_episodes, combined, OUTPUT_DIR)

    # Quick summary to stdout
    print("\n=== Combined ranking (top 20) ===")
    print(f"{'feature':<35} {'desc':<15} {'first_t':>7}  {'n_grp':>5}  {'mean_pk':>7}  dir")
    print("-" * 85)
    for feat, desc, et, ng, mpk, dir_str, fts, _ in combined[:20]:
        print(f"{feat:<35} {desc:<15} {et:>7}  {ng:>5}  {mpk:>7.3f}  {dir_str}")

    # Cross-domain timeline summary
    print("\n=== Cross-domain timeline (steps where n_3of3 > 0) ===")
    ct_text = (OUTPUT_DIR / "cross_domain_timeline.txt").read_text().split("\n")
    for line in ct_text[4:]:  # skip header
        if line.startswith("  ") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                n3 = int(parts[1])
                if n3 > 0:
                    print(line)
            except ValueError:
                pass

    print(f"\nDone. All outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
