#!/usr/bin/env python3
"""
sliding_window_analysis.py — Sliding-window trend analysis for conformal failure prediction.

Complements trend_analysis.py by replacing cumulative descriptors with fixed-size
sliding windows.  At each evaluation step t two W-step windows are computed:

  current window : ts[t-W+1 .. t]
  previous window: ts[t-2W+1 .. t-W]

Windowed descriptors capture the *local* trend rather than cumulative trend.
d_slope = slope(current) - slope(previous) is local acceleration / deceleration.

Additional metrics beyond AUROC:
  * P/R/F1 at the threshold that maximises F1   (metric: f1_maxf1)
  * P/R/F1 at the threshold achieving recall >= TARGET_RECALL = 0.80
    — directly relevant to conformal prediction with alpha = 0.20

Usage (from project root):
  python scripts/attention_analysis/sliding_window_analysis.py

Outputs → /nfs/roberts/scratch/pi_tkf6/zs377/sliding_window_analysis/
  sw_full_auroc.txt        full AUROC table for all (feat, win_desc, t, W)
  sw_consistency.txt       features cross-domain consistent (AUROC + F1)
  sw_top_signatures.txt    top 30 features ranked by cross-domain score
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.stats import mannwhitneyu

# ─── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR  = Path("/nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/zs377/sliding_window_analysis")

N_SAMPLE      = 75
MAX_STEP      = 25
TARGET_STEPS  = [5, 10, 15, 20, 25]
WIN_SIZES     = [5, 8]   # W=5: non-overlapping at 5-step eval intervals; W=8: more robust slope
AUROC_THRESH  = 0.60
CONSIST_MIN   = 2        # min groups for "cross-domain consistent"
TARGET_RECALL = 0.80     # fixed operating point for conformal alpha = 0.20
SEED          = 42

GROUPS: Dict[str, Path] = {
    "pi0fast_libero10":   CACHE_DIR / "pi0fast_libero10.pkl",
    "pi0fast_liberoplus": CACHE_DIR / "pi0fast_liberoplus.pkl",
    "pi05_libero10":      CACHE_DIR / "pi05_libero10.pkl",
}

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
    "abs_step", "rel_step",
    "action_plan_entropy",
})

WIN_DESCRIPTORS = [
    "win_mean",      # mean of ts within current window
    "win_std",       # std of ts within current window
    "win_slope",     # linear regression slope within current window
    "win_range",     # max - min within current window
    "d_mean",        # win_mean(current) - win_mean(previous)
    "d_slope",       # win_slope(current) - win_slope(previous)  [acceleration]
    "d_slope_mag",   # |d_slope|  [magnitude of acceleration, direction-agnostic]
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_group(path: Path, n_sample: int, seed: int) -> Tuple[List, List]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    rng = np.random.default_rng(seed)

    def _sample(eps):
        n = min(n_sample, len(eps))
        idx = rng.choice(len(eps), size=n, replace=False)
        return [eps[i] for i in idx]

    return _sample(data["success"]), _sample(data["failure"])


def get_time_series(episode: List[dict], feature: str, max_step: int) -> np.ndarray:
    ts = []
    for step_idx in range(max_step + 1):
        if step_idx < len(episode):
            v = episode[step_idx].get(feature, np.nan)
            if v is None:
                v = np.nan
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = np.nan
        else:
            v = np.nan
        ts.append(v)
    return np.asarray(ts, dtype=np.float64)


# ─── Windowed descriptors ─────────────────────────────────────────────────────

def _win_slope(seg: np.ndarray) -> float:
    ok = ~np.isnan(seg)
    if ok.sum() < 3:
        return np.nan
    idx = np.where(ok)[0].astype(float)
    try:
        return float(np.polyfit(idx, seg[ok], 1)[0])
    except Exception:
        return np.nan


def windowed_descriptors(ts: np.ndarray, t: int, W: int) -> Dict[str, float]:
    """
    Compute local windowed descriptors for time series ts at evaluation step t.

    Current window:  ts[max(0, t-W+1) : t+1]
    Previous window: ts[max(0, t-2W+1) : max(0, t-W+1)]

    d_slope requires both windows to have >= 3 valid points.
    """
    cur_start = max(0, t - W + 1)
    cur = ts[cur_start : t + 1]

    prev_end   = max(0, t - W + 1)
    prev_start = max(0, t - 2 * W + 1)
    prev = ts[prev_start : prev_end] if prev_end > prev_start else np.array([])

    cur_ok  = cur[~np.isnan(cur)]
    prev_ok = prev[~np.isnan(prev)] if len(prev) > 0 else np.array([])

    result: Dict[str, float] = {}
    result["win_mean"]  = float(np.nanmean(cur))  if len(cur_ok) >= 1 else np.nan
    result["win_std"]   = float(np.nanstd(cur))   if len(cur_ok) >= 2 else np.nan
    result["win_slope"] = _win_slope(cur)
    result["win_range"] = (float(np.nanmax(cur) - np.nanmin(cur))
                           if len(cur_ok) >= 2 else np.nan)

    prev_mean  = float(np.nanmean(prev)) if len(prev_ok) >= 1 else np.nan
    prev_slope = _win_slope(prev)        if len(prev_ok) >= 3 else np.nan

    result["d_mean"] = (result["win_mean"] - prev_mean
                        if not np.isnan(result["win_mean"]) and not np.isnan(prev_mean)
                        else np.nan)

    if not np.isnan(result["win_slope"]) and not np.isnan(prev_slope):
        d = result["win_slope"] - prev_slope
        result["d_slope"]     = d
        result["d_slope_mag"] = abs(d)
    else:
        result["d_slope"]     = np.nan
        result["d_slope_mag"] = np.nan

    return result


# ─── AUROC and P/R/F1 helpers ─────────────────────────────────────────────────

def auroc_abs(sv: np.ndarray, fv: np.ndarray) -> float:
    s = sv[~np.isnan(sv)]
    f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan
    try:
        stat, _ = mannwhitneyu(s, f, alternative="two-sided")
        u = float(stat) / (len(s) * len(f))
        return max(u, 1.0 - u)
    except Exception:
        return np.nan


def auroc_signed(sv: np.ndarray, fv: np.ndarray) -> float:
    """Positive = failures score higher."""
    s = sv[~np.isnan(sv)]
    f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan
    try:
        stat, _ = mannwhitneyu(s, f, alternative="two-sided")
        return float(stat) / (len(s) * len(f)) - 0.5
    except Exception:
        return np.nan


def prf1_scan(sv: np.ndarray, fv: np.ndarray, direction: str) -> Tuple[float, float, float]:
    """
    Scan all unique score thresholds and return (P, R, F1) at the one that
    maximises F1.  direction '+' → failures score higher (threshold: score >= thresh).
    """
    s = sv[~np.isnan(sv)]
    f = fv[~np.isnan(fv)]
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
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r

    return best_p, best_r, best_f1


def prf1_at_recall(
    sv: np.ndarray, fv: np.ndarray, direction: str, target_recall: float
) -> Tuple[float, float, float]:
    """
    Return (P, R, F1) at the most conservative threshold that still achieves
    recall >= target_recall.  This maximises precision for the recall target.

    For direction '+': sweep thresholds from highest to lowest (recall rises as
    threshold falls).  For direction '-': sweep lowest to highest.
    Stop at the first threshold where recall >= target_recall.
    """
    s = sv[~np.isnan(sv)]
    f = fv[~np.isnan(fv)]
    if len(s) < 3 or len(f) < 3:
        return np.nan, np.nan, np.nan

    scores = np.concatenate([s, f])
    labels = np.array([0] * len(s) + [1] * len(f))

    thresholds = np.unique(scores)
    if direction == "+":
        thresholds = thresholds[::-1]  # high → low: recall increases as we iterate

    best_p, best_r, best_f1 = np.nan, np.nan, np.nan
    fallback_r, fallback_p, fallback_f1 = 0.0, 0.0, 0.0

    for thresh in thresholds:
        preds = (scores >= thresh) if direction == "+" else (scores <= thresh)
        tp = int(np.sum(preds & (labels == 1)))
        fp = int(np.sum(preds & (labels == 0)))
        fn = int(np.sum(~preds & (labels == 1)))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if r > fallback_r:
            fallback_r, fallback_p, fallback_f1 = r, p, f1
        if r >= target_recall:
            best_p, best_r, best_f1 = p, r, f1
            break  # most conservative threshold achieving target → best precision

    if np.isnan(best_r):
        best_p, best_r, best_f1 = fallback_p, fallback_r, fallback_f1

    return best_p, best_r, best_f1


# ─── Group analysis ───────────────────────────────────────────────────────────

# result value: (auroc_abs, auroc_signed, p_mf, r_mf, f1_mf, p_r80, r_r80, f1_r80)
ResultKey = Tuple[str, str, int, int]   # (feature, win_desc, t, W)
ResultVal = Tuple[float, ...]


def analyse_group(
    eps_s: List, eps_f: List, features: List[str]
) -> Dict[ResultKey, ResultVal]:
    results: Dict[ResultKey, ResultVal] = {}

    for feat in features:
        ts_s = [get_time_series(ep, feat, MAX_STEP) for ep in eps_s]
        ts_f = [get_time_series(ep, feat, MAX_STEP) for ep in eps_f]

        for W in WIN_SIZES:
            for t in TARGET_STEPS:
                desc_s: Dict[str, List[float]] = {d: [] for d in WIN_DESCRIPTORS}
                desc_f: Dict[str, List[float]] = {d: [] for d in WIN_DESCRIPTORS}

                for ts in ts_s:
                    d = windowed_descriptors(ts, t, W)
                    for k in WIN_DESCRIPTORS:
                        desc_s[k].append(d.get(k, np.nan))
                for ts in ts_f:
                    d = windowed_descriptors(ts, t, W)
                    for k in WIN_DESCRIPTORS:
                        desc_f[k].append(d.get(k, np.nan))

                for wdesc in WIN_DESCRIPTORS:
                    sv = np.asarray(desc_s[wdesc], dtype=np.float64)
                    fv = np.asarray(desc_f[wdesc], dtype=np.float64)

                    a_abs = auroc_abs(sv, fv)
                    a_sgn = auroc_signed(sv, fv)
                    direction = "+" if (not np.isnan(a_sgn) and a_sgn >= 0) else "-"

                    p_mf,  r_mf,  f1_mf  = prf1_scan(sv, fv, direction)
                    p_r80, r_r80, f1_r80  = prf1_at_recall(sv, fv, direction, TARGET_RECALL)

                    results[(feat, wdesc, t, W)] = (
                        a_abs, a_sgn, p_mf, r_mf, f1_mf, p_r80, r_r80, f1_r80
                    )

    return results


# ─── Output tables ────────────────────────────────────────────────────────────

def write_full_auroc_table(
    all_results: Dict[str, Dict], features: List[str], path: Path
) -> None:
    group_names = list(all_results.keys())
    lines = [
        "Sliding window AUROC table",
        f"WIN_SIZES={WIN_SIZES}  TARGET_STEPS={TARGET_STEPS}  N_SAMPLE={N_SAMPLE}",
        "",
        (f"{'feature':<35} {'win_desc':<15} {'t':>3} {'W':>3}  " +
         "  ".join(f"{g[:16]:>16}" for g in group_names)),
        "-" * 100,
    ]

    for feat in features:
        for W in WIN_SIZES:
            for wdesc in WIN_DESCRIPTORS:
                for t in TARGET_STEPS:
                    row = f"{feat:<35} {wdesc:<15} {t:>3} {W:>3}  "
                    for g in group_names:
                        entry = all_results[g].get((feat, wdesc, t, W))
                        if entry is not None and not np.isnan(entry[0]):
                            sgn = "+" if entry[1] > 0 else "-"
                            row += f"  {entry[0]:.3f}({sgn})       "
                        else:
                            row += f"  nan            "
                    lines.append(row)
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"  wrote {path.name} ({len(lines)} lines)")


def write_consistency_table(
    all_results: Dict[str, Dict], features: List[str], path: Path
) -> None:
    """
    Emit consistent entries: AUROC >= AUROC_THRESH in >= CONSIST_MIN groups AND
    same direction across groups that pass the threshold.
    Columns include per-group AUROC, mean F1@maxF1, and mean F1@recall>=0.80.
    """
    group_names = list(all_results.keys())
    records = []

    for feat in features:
        for W in WIN_SIZES:
            for wdesc in WIN_DESCRIPTORS:
                for t in TARGET_STEPS:
                    aucs, signs, f1s_mf, f1s_r80 = [], [], [], []
                    for g in group_names:
                        entry = all_results[g].get((feat, wdesc, t, W))
                        if entry is not None:
                            aucs.append(entry[0])
                            signs.append(0 if np.isnan(entry[1]) else int(np.sign(entry[1])))
                            f1s_mf.append(entry[4])
                            f1s_r80.append(entry[7])
                        else:
                            aucs.append(np.nan)
                            signs.append(0)
                            f1s_mf.append(np.nan)
                            f1s_r80.append(np.nan)

                    aucs_arr = np.asarray(aucs)
                    above = int(np.sum(aucs_arr >= AUROC_THRESH))
                    nz = [s for s in signs if s != 0]
                    same_dir = (len(set(nz)) <= 1) if nz else False

                    if above < CONSIST_MIN or not same_dir:
                        continue

                    max_a   = float(np.nanmax(aucs_arr))
                    mean_a  = float(np.nanmean(aucs_arr))
                    mf1_mf  = float(np.nanmean(f1s_mf))
                    mf1_r80 = float(np.nanmean(f1s_r80))
                    dir_str = "F>S" if (nz and nz[0] > 0) else "F<S"

                    records.append((feat, wdesc, t, W, above, dir_str,
                                    max_a, mean_a, mf1_mf, mf1_r80,
                                    aucs, f1s_mf, f1s_r80))

    # primary sort: n_groups descending, secondary: mean_auroc
    records.sort(key=lambda r: (-r[4], -r[7]))

    g_cols = "  ".join(f"{g[:12]:>12}" for g in group_names)
    header = (
        f"{'feature':<35} {'win_desc':<15} {'t':>3} {'W':>3}  "
        f"{'n>=thr':>6}  {'dir':>4}  {'maxA':>6}  {'meanA':>6}  "
        f"{'F1@mx':>6}  {'F1@r80':>6}  {g_cols}"
    )
    lines = [
        f"Sliding window cross-domain consistent features",
        f"AUROC >= {AUROC_THRESH} in >= {CONSIST_MIN}/3 groups, same direction",
        f"F1@mx  = F1 at max-F1 threshold (mean across groups)",
        f"F1@r80 = F1 at recall >= {TARGET_RECALL:.2f} threshold (conformal alpha=0.20)",
        f"Total consistent entries: {len(records)}",
        "",
        header,
        "-" * len(header),
    ]

    for (feat, wdesc, t, W, above, dir_str, max_a, mean_a, f1_mf, f1_r80,
         aucs, f1s_mf, f1s_r80) in records:
        auc_str = "  ".join(f"{a:.3f}" if not np.isnan(a) else "  nan" for a in aucs)
        lines.append(
            f"{feat:<35} {wdesc:<15} {t:>3} {W:>3}  "
            f"{above:>6}  {dir_str:>4}  {max_a:>6.3f}  {mean_a:>6.3f}  "
            f"{f1_mf:>6.3f}  {f1_r80:>6.3f}  {auc_str}"
        )

    path.write_text("\n".join(lines))
    print(f"  wrote {path.name} ({len(records)} consistent entries)")


def write_top_signatures(
    all_results: Dict[str, Dict], features: List[str], path: Path, n_top: int = 30
) -> None:
    """
    For each feature, find its best (wdesc, t, W) combination by mean AUROC
    across groups and report alongside F1 metrics.
    """
    group_names = list(all_results.keys())
    summary = []

    for feat in features:
        best = dict(wdesc="?", t=None, W=5, max_a=0.0, mean_a=0.0,
                    n_above=0, dir="?", f1_mf=0.0, f1_r80=0.0)
        for W in WIN_SIZES:
            for wdesc in WIN_DESCRIPTORS:
                for t in TARGET_STEPS:
                    entries = [all_results[g].get((feat, wdesc, t, W)) for g in group_names]
                    aucs  = [e[0] if e else np.nan for e in entries]
                    signs = [e[1] if e else np.nan for e in entries]
                    f1s   = [e[4] if e else np.nan for e in entries]
                    f1rs  = [e[7] if e else np.nan for e in entries]

                    arr   = np.asarray(aucs)
                    above = int(np.sum(arr >= AUROC_THRESH))
                    max_a  = float(np.nanmax(arr))  if not np.all(np.isnan(arr)) else 0.0
                    mean_a = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
                    f1_mf  = float(np.nanmean(f1s)) if not np.all(np.isnan(f1s)) else 0.0
                    f1_r80 = float(np.nanmean(f1rs))if not np.all(np.isnan(f1rs))else 0.0
                    nz = [s for s in signs if not np.isnan(s) and s != 0]
                    dir_str = "F>S" if (nz and np.sign(nz[0]) > 0) else "F<S"

                    if mean_a > best["mean_a"]:
                        best = dict(wdesc=wdesc, t=t, W=W, max_a=max_a,
                                    mean_a=mean_a, n_above=above, dir=dir_str,
                                    f1_mf=f1_mf, f1_r80=f1_r80)
        summary.append((feat, best))

    summary.sort(key=lambda x: (-x[1]["n_above"], -x[1]["mean_a"]))

    lines = ["Top sliding window signatures (best win_desc/t/W per feature)", ""]
    lines.append(
        f"{'feature':<35} {'win_desc':<15} {'t*':>4} {'W*':>3}  "
        f"{'n>=thr':>6}  {'maxA':>6}  {'meanA':>6}  {'F1@mx':>6}  {'F1@r80':>6}  {'dir':>4}"
    )
    lines.append("-" * 110)
    for feat, info in summary[:n_top]:
        t_str = str(info["t"]) if info["t"] is not None else "n/a"
        lines.append(
            f"{feat:<35} {info['wdesc']:<15} {t_str:>4} {info['W']:>3}  "
            f"{info['n_above']:>6}  {info['max_a']:>6.3f}  {info['mean_a']:>6.3f}  "
            f"{info['f1_mf']:>6.3f}  {info['f1_r80']:>6.3f}  {info['dir']:>4}"
        )
    path.write_text("\n".join(lines))
    print(f"  wrote {path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading group caches ...")
    group_episodes: Dict[str, Tuple[List, List]] = {}
    for gname, gpath in GROUPS.items():
        print(f"  {gname} ... ", end="", flush=True)
        eps_s, eps_f = load_group(gpath, N_SAMPLE, SEED)
        group_episodes[gname] = (eps_s, eps_f)
        print(f"S={len(eps_s)}, F={len(eps_f)}")

    first_ep = group_episodes["pi0fast_liberoplus"][0][0]
    all_keys = list(first_ep[0].keys())
    features = [
        k for k in all_keys
        if k not in NOISE_FEATURES and k not in SKIP_FEATURES
    ]
    n_combos = len(features) * len(WIN_DESCRIPTORS) * len(TARGET_STEPS) * len(WIN_SIZES)
    print(f"\n{len(features)} features × {len(WIN_DESCRIPTORS)} win_descs "
          f"× {len(TARGET_STEPS)} steps × {len(WIN_SIZES)} window sizes "
          f"= {n_combos} combinations per group")

    all_results: Dict[str, Dict] = {}
    for gname, (eps_s, eps_f) in group_episodes.items():
        print(f"\nAnalysing {gname} ...", flush=True)
        all_results[gname] = analyse_group(eps_s, eps_f, features)
        print(f"  {len(all_results[gname])} (feat, win_desc, t, W) entries")

    print("\nWriting output tables ...")
    write_full_auroc_table(all_results, features, OUTPUT_DIR / "sw_full_auroc.txt")
    write_consistency_table(all_results, features, OUTPUT_DIR / "sw_consistency.txt")
    write_top_signatures(all_results, features, OUTPUT_DIR / "sw_top_signatures.txt")

    ct = (OUTPUT_DIR / "sw_consistency.txt").read_text()
    print("\n--- sw_consistency.txt (first 60 lines) ---")
    print("\n".join(ct.split("\n")[:60]))

    ts = (OUTPUT_DIR / "sw_top_signatures.txt").read_text()
    print("\n--- sw_top_signatures.txt ---")
    print(ts)

    print(f"\nDone. All outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
