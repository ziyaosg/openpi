#!/usr/bin/env python3
"""
token_aggregation_analysis.py

Empirically tests whether averaging over all 5 decoded action tokens is the best
aggregation strategy for pi0fast GradCAM/RAS/RN Gini features, or whether
per-token, max-anomaly, or weighted aggregations do better.

Reads per-token Gini values (t0–t4) from the pkl feature caches and computes:
  - Per-step and phase-specific AUROC for each aggregation mode
  - Conformal (global + chunked) performance for each mode
  - A comparison table: which mode beats the naive mean, and by how much?

Aggregation modes tested per feature group:
  tok0 … tok4   : individual decoded action token (near-term to far-term)
  mean          : uniform average (≡ current cache feature gc_gini_base, etc.)
  max_anom      : token with highest failure score (min Gini for F<S features)
  near_wt       : weighted [0.40, 0.25, 0.15, 0.10, 0.10]
  far_wt        : weighted [0.10, 0.10, 0.15, 0.25, 0.40]
  learned       : AUROC-weighted on calibration data (per-feature weight)

Hypothesis being tested:
  token-0 → immediate failure signal (near-term action plan)
  token-4 → plan-horizon inconsistency (far-term action plan)
  max     → earliest single-token alarm, useful for conformal detection
  near_wt → emphasises the token most relevant to the current step
  mean    → stable but may dilute token-specific early signals

Output → /nfs/roberts/scratch/pi_tkf6/zs377/token_aggregation_analysis/
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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR  = Path("/nfs/roberts/scratch/pi_tkf6/zs377/analysis_conformal_v2/feature_cache")
OUT_DIR    = Path("/nfs/roberts/scratch/pi_tkf6/zs377/token_aggregation_analysis")

GROUPS     = ["pi0fast_libero10", "pi0fast_liberoplus"]  # pi05 has no per-token maps
MAX_STEPS  = 60
WIN_W      = 5

PHASE_DEFS: Dict[str, Tuple[int, int]] = {
    "early": (0, 10), "mid": (11, 30), "late": (31, 59),
}

ALPHA       = 0.10
ALPHA_CHUNK = ALPHA / len(PHASE_DEFS)   # Bonferroni ≈ 0.0333

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS  (set of 5 per-token cache keys + meta-info)
# ─────────────────────────────────────────────────────────────────────────────

# (group_name, base_feat_name, direction_prior, mean_key)
# base_feat_name is the stem; per-token keys are {stem}_t{i}
# mean_key is the averaged feature already in the cache

FEATURE_GROUPS = [
    # Camera: base
    ("gc_gini_base",   "gc_gini_base",   -1, "gc_gini_base"),
    ("ras_gini_base",  "ras_gini_base",  -1, "ras_gini_base"),
    ("rn_gini_base",   "rn_gini_base",   -1, "rn_gini_base"),
    # Camera: wrist
    ("gc_gini_wrist",  "gc_gini_wrist",  -1, "gc_gini_wrist"),
    ("ras_gini_wrist", "ras_gini_wrist", -1, "ras_gini_wrist"),
    ("rn_gini_wrist",  "rn_gini_wrist",  -1, "rn_gini_wrist"),
    # Task tokens
    ("gc_gini_task",   "gc_gini_task",   -1, "gc_gini_task"),
    ("ras_gini_task",  "ras_gini_task",  -1, "ras_gini_task"),
    ("rn_gini_task",   "rn_gini_task",   -1, "rn_gini_task"),
]

# Token weights for near-term and far-term schemes
W_NEAR = np.array([0.40, 0.25, 0.15, 0.10, 0.10])
W_FAR  = np.array([0.10, 0.10, 0.15, 0.25, 0.40])
W_UNI  = np.ones(5) / 5                           # uniform = same as mean

WEIGHT_SCHEMES = {
    "mean":    W_UNI,
    "near_wt": W_NEAR,
    "far_wt":  W_FAR,
}

N_TOK = 5

# ─────────────────────────────────────────────────────────────────────────────
# MATH
# ─────────────────────────────────────────────────────────────────────────────

def auroc(fail_s: np.ndarray, succ_s: np.ndarray) -> float:
    f = fail_s[~np.isnan(fail_s)]; s = succ_s[~np.isnan(succ_s)]
    if len(f) == 0 or len(s) == 0:
        return float("nan")
    nf, ns = len(f), len(s)
    combined = np.concatenate([f, s])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    return float((ranks[:nf].sum() - nf * (nf + 1) / 2) / (nf * ns))


def effective_auroc(a: float) -> float:
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

# ─────────────────────────────────────────────────────────────────────────────
# PER-TOKEN AGGREGATION FROM CACHE
# ─────────────────────────────────────────────────────────────────────────────

def get_token_matrix(
    ep_list: List[List[dict]],
    stem: str,
) -> np.ndarray:
    """
    Returns (N_episodes, MAX_STEPS, N_TOK) array of per-token raw feature values.
    stem: base name, e.g. 'gc_gini_base'. Keys expected: gc_gini_base_t0 … _t4.
    """
    out = np.full((len(ep_list), MAX_STEPS, N_TOK), np.nan, dtype=np.float64)
    tok_keys = [f"{stem}_t{i}" for i in range(N_TOK)]
    for ep_i, ep in enumerate(ep_list):
        for t, step in enumerate(ep[:MAX_STEPS]):
            for ti, k in enumerate(tok_keys):
                v = step.get(k)
                if v is not None:
                    try: out[ep_i, t, ti] = float(v)
                    except (TypeError, ValueError): pass
    return out


def get_mean_from_cache(
    ep_list: List[List[dict]],
    mean_key: str,
) -> np.ndarray:
    """(N_episodes, MAX_STEPS) raw mean feature (already averaged over tokens in cache)."""
    out = np.full((len(ep_list), MAX_STEPS), np.nan, dtype=np.float64)
    for ep_i, ep in enumerate(ep_list):
        for t, step in enumerate(ep[:MAX_STEPS]):
            v = step.get(mean_key)
            if v is not None:
                try: out[ep_i, t] = float(v)
                except (TypeError, ValueError): pass
    return out


def aggregate_tokens(
    tok_mat: np.ndarray,   # (N, T, 5)
    mode: str,
    direction: int,
    learned_weights: Optional[np.ndarray] = None,   # (5,) per-token weights
) -> np.ndarray:
    """
    (N, T) aggregated score for a given mode.
    direction: +1 (F>S, high=failure) or -1 (F<S, negate to high=failure).

    Modes:
      tok{i}    : raw Gini of token i
      mean      : uniform average
      near_wt   : near-term weighted
      far_wt    : far-term weighted
      max_anom  : max of direction*raw over tokens (most anomalous token)
      learned   : weighted by learned_weights
    """
    if mode.startswith("tok") and mode[3:].isdigit():
        i = int(mode[3:])
        return direction * tok_mat[:, :, i]

    if mode in WEIGHT_SCHEMES:
        w = WEIGHT_SCHEMES[mode]
        # Weighted mean, ignoring NaN tokens
        valid_mask = ~np.isnan(tok_mat)   # (N, T, 5)
        weighted = np.where(valid_mask, tok_mat * w[np.newaxis, np.newaxis, :], 0.0)
        weight_sum = np.where(valid_mask, w[np.newaxis, np.newaxis, :], 0.0).sum(axis=2)
        agg = np.where(weight_sum > 0, weighted.sum(axis=2) / weight_sum, np.nan)
        return direction * agg

    if mode == "max_anom":
        # Most anomalous token: max of (direction * token_value)
        # For F<S features (direction=-1): min raw value → most concentrated
        # For F>S features (direction=+1): max raw value
        return np.nanmax(direction * tok_mat, axis=2)

    if mode == "learned" and learned_weights is not None:
        w = learned_weights
        valid_mask = ~np.isnan(tok_mat)
        weighted = np.where(valid_mask, tok_mat * w[np.newaxis, np.newaxis, :], 0.0)
        weight_sum = np.where(valid_mask, w[np.newaxis, np.newaxis, :], 0.0).sum(axis=2)
        agg = np.where(weight_sum > 0, weighted.sum(axis=2) / weight_sum, np.nan)
        return direction * agg

    raise ValueError(f"Unknown aggregation mode: {mode}")


def win_mean_of(raw_scores: np.ndarray) -> np.ndarray:
    """(N, T) → (N, T) win_mean with causal window of WIN_W."""
    N, T = raw_scores.shape
    out = np.full_like(raw_scores, np.nan)
    for t in range(T):
        lo = max(0, t - WIN_W + 1)
        out[:, t] = np.nanmean(raw_scores[:, lo:t+1], axis=1)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# LEARNED WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def learn_token_weights(
    fail_tok: np.ndarray,   # (N_fail, T, 5) — all steps used for calibration
    succ_tok: np.ndarray,   # (N_succ, T, 5)
    direction: int,
) -> np.ndarray:
    """
    Calibration-only AUROC-based token weighting.
    For each token i, compute the peak per-step AUROC of (direction * token_i).
    Weight token i proportional to its peak AUROC minus 0.5 (clamped ≥ 0).
    Normalise weights to sum to 1.

    This is a simple, calibration-only approach — no tuning on test data.
    """
    weights = np.zeros(N_TOK, dtype=float)
    for i in range(N_TOK):
        best = 0.5
        for t in range(MAX_STEPS):
            a = auroc(direction * fail_tok[:, t, i], direction * succ_tok[:, t, i])
            if not np.isnan(a) and effective_auroc(a) > effective_auroc(best):
                best = a
        weights[i] = max(0.0, effective_auroc(best) - 0.5)
    s = weights.sum()
    return weights / s if s > 1e-9 else np.ones(N_TOK) / N_TOK

# ─────────────────────────────────────────────────────────────────────────────
# AUROC EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def per_step_auroc(
    fail_scores: np.ndarray,   # (N_fail, T)
    succ_scores: np.ndarray,   # (N_succ, T)
) -> np.ndarray:
    """(T,) per-step AUROC curve."""
    T = fail_scores.shape[1]
    curve = np.full(T, np.nan)
    for t in range(T):
        curve[t] = auroc(fail_scores[:, t], succ_scores[:, t])
    return curve


def phase_auroc_from_scores(
    fail_scores: np.ndarray,
    succ_scores: np.ndarray,
) -> Dict[str, float]:
    """AUROC of episode-level phase_mean for each phase."""
    result: Dict[str, float] = {}
    for ph, (a, b) in PHASE_DEFS.items():
        f_agg = np.nanmean(fail_scores[:, a:b+1], axis=1)
        s_agg = np.nanmean(succ_scores[:, a:b+1], axis=1)
        result[ph] = auroc(f_agg, s_agg)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_conformal(
    cal_succ_scores: np.ndarray,    # (N_cal_succ, T)
    test_succ_scores: np.ndarray,
    test_fail_scores: np.ndarray,
) -> Dict[str, dict]:
    """Returns {"global": {...}, "chunked": {...}}."""

    def _max_scores(scores, a, b):
        chunk = scores[:, a:min(b+1, scores.shape[1])]
        valid_max = np.full(len(scores), np.nan)
        for i in range(len(scores)):
            v = chunk[i][~np.isnan(chunk[i])]
            if len(v) > 0:
                valid_max[i] = float(v.max())
        return valid_max

    # Global threshold
    cal_max = _max_scores(cal_succ_scores, 0, MAX_STEPS - 1)
    tau_g   = float(np.nanquantile(cal_max, 1.0 - ALPHA))

    # Chunked thresholds
    tau_c: Dict[str, float] = {}
    for ph, (a, b) in PHASE_DEFS.items():
        chunk_max = _max_scores(cal_succ_scores, a, b)
        tau_c[ph] = float(np.nanquantile(chunk_max, 1.0 - ALPHA_CHUNK))

    ns = len(test_succ_scores); nf = len(test_fail_scores)

    def _detect(scores, use_chunked):
        flagged = np.zeros(len(scores), dtype=bool)
        det     = np.full(len(scores), np.nan)
        for ei in range(len(scores)):
            for t in range(min(MAX_STEPS, scores.shape[1])):
                s = scores[ei, t]
                if np.isnan(s): continue
                if use_chunked:
                    for ph, (a, b) in PHASE_DEFS.items():
                        if a <= t <= b and s > tau_c[ph]:
                            flagged[ei] = True; det[ei] = t; break
                else:
                    if s > tau_g:
                        flagged[ei] = True; det[ei] = t
                if flagged[ei]: break
        return flagged, det

    def _metrics(sf, ff, fd, use_chunked):
        tp = int(ff.sum()); fp = int(sf.sum())
        recall    = tp / nf if nf > 0 else float("nan")
        fpr       = fp / ns if ns > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))
        det = fd[ff]
        return {"recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
                "median_det": float(np.nanmedian(det)) if len(det) > 0 else float("nan"),
                "frac_before": {k: float((det <= k).sum() / nf) if nf > 0 else float("nan")
                                for k in (10, 20, 30)},
                "tau": tau_g if not use_chunked else tau_c}

    sf_g, _   = _detect(test_succ_scores, False)
    ff_g, fd_g = _detect(test_fail_scores, False)
    sf_c, _   = _detect(test_succ_scores, True)
    ff_c, fd_c = _detect(test_fail_scores, True)

    return {
        "global":  _metrics(sf_g, ff_g, fd_g, False),
        "chunked": _metrics(sf_c, ff_c, fd_c, True),
    }

# ─────────────────────────────────────────────────────────────────────────────
# PER-GROUP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_group(
    ep_data: Dict,
    group_name: str,
) -> Dict[str, Dict[str, dict]]:
    """
    Returns {feat_group_name: {mode_name: result_dict}}.
    result_dict contains: auroc_curve, phase_auroc, peak_auroc, global/chunked conf.
    """
    succ_eps = ep_data.get("success", [])
    fail_eps = ep_data.get("failure", [])

    # 50/50 calibration / test split within the group
    ns = len(succ_eps); nf = len(fail_eps)
    nc_s = ns // 2; nc_f = nf // 2
    cal_succ  = succ_eps[:nc_s];  test_succ = succ_eps[nc_s:]
    cal_fail  = fail_eps[:nc_f];  test_fail = fail_eps[nc_f:]

    all_results: Dict[str, Dict[str, dict]] = {}

    for feat_name, stem, direction, mean_key in FEATURE_GROUPS:
        print(f"    {feat_name} …", flush=True)

        # Load per-token matrices
        def _load(ep_list):
            return get_token_matrix(ep_list, stem)

        cal_succ_tok  = _load(cal_succ)
        cal_fail_tok  = _load(cal_fail)
        test_succ_tok = _load(test_succ)
        test_fail_tok = _load(test_fail)

        # Learned weights from calibration only
        lw = learn_token_weights(cal_fail_tok, cal_succ_tok, direction)

        # All aggregation modes to test
        all_modes = (
            [f"tok{i}" for i in range(N_TOK)] +
            ["max_anom", "near_wt", "far_wt", "mean", "learned"]
        )

        feat_results: Dict[str, dict] = {}
        for mode in all_modes:
            lw_arg = lw if mode == "learned" else None

            # Raw aggregated scores
            test_f_raw = aggregate_tokens(test_fail_tok, mode, direction, lw_arg)
            test_s_raw = aggregate_tokens(test_succ_tok, mode, direction, lw_arg)
            cal_s_raw  = aggregate_tokens(cal_succ_tok,  mode, direction, lw_arg)
            cal_f_raw  = aggregate_tokens(cal_fail_tok,  mode, direction, lw_arg)   # for AUROC only

            # Win_mean scores for conformal
            test_f_wm = win_mean_of(test_f_raw)
            test_s_wm = win_mean_of(test_s_raw)
            cal_s_wm  = win_mean_of(cal_s_raw)

            # AUROC curve (on test set win_mean)
            curve = per_step_auroc(test_f_wm, test_s_wm)
            pa    = phase_auroc_from_scores(test_f_wm, test_s_wm)
            peak  = float(np.nanmax(np.abs(curve - 0.5)) + 0.5) if not np.all(np.isnan(curve)) else float("nan")
            best_step = int(np.nanargmax(np.abs(curve - 0.5))) if not np.all(np.isnan(curve)) else -1

            # Conformal evaluation
            conf = run_conformal(cal_s_wm, test_s_wm, test_f_wm)

            feat_results[mode] = {
                "auroc_curve": curve,
                "phase_auroc": pa,
                "peak_auroc":  peak,
                "best_step":   best_step,
                "global":      conf["global"],
                "chunked":     conf["chunked"],
                "learned_weights": lw if mode == "learned" else None,
            }

        # ── Variance-reduction diagnostics ───────────────────────────────────
        # Use ALL episodes (not just test split) for stable correlation estimates.
        all_succ_tok = get_token_matrix(succ_eps, stem)   # (N_succ, T, 5)
        all_fail_tok = get_token_matrix(fail_eps, stem)   # (N_fail, T, 5)
        feat_results["_token_stats"] = compute_token_stats(
            all_succ_tok, all_fail_tok, direction
        )

        all_results[feat_name] = feat_results

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# INTER-TOKEN CORRELATION AND VARIANCE-REDUCTION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_SCHEMES_VR = {   # aggregation modes to test for variance
    "mean":    W_UNI,
    "near_wt": W_NEAR,
    "far_wt":  W_FAR,
}


def _flat_valid(tok_mat: np.ndarray, a: int, b: int) -> np.ndarray:
    """Extract steps [a, b] from (N, T, 5), flatten to (N*T_phase, 5), drop NaN rows."""
    b_eff = min(b + 1, tok_mat.shape[1])
    seg   = tok_mat[:, a:b_eff, :]             # (N, T_phase, 5)
    flat  = seg.reshape(-1, 5)                 # (N*T_phase, 5)
    valid = ~np.isnan(flat).any(axis=1)
    return flat[valid]


def _corr_stats(flat: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (5×5 correlation matrix, mean off-diagonal correlation)."""
    if len(flat) < 5:
        return np.full((5, 5), np.nan), float("nan")
    corr = np.corrcoef(flat.T)                        # (5, 5)
    off  = corr[np.triu_indices(5, k=1)]              # 10 upper-triangle values
    return corr, float(np.nanmean(off))


def _variance_stats(
    flat: np.ndarray,
    direction: int,
    learned_weights: Optional[np.ndarray],
) -> dict:
    """
    Compute:
      - per-token variance (direction-signed values)
      - variance of each aggregation mode
      - empirical variance-reduction ratio (VRR) for the mean
      - theoretical VRR from correlation (= (1 + 4*ρ̄) / 5)

    VRR = var(mean_over_tokens) / mean(var(token_i))
      ≈ 1.0  when tokens are perfectly correlated (averaging does nothing)
      ≈ 0.2  when tokens are independent (full N-fold variance reduction)
    """
    signed = direction * flat               # (N_flat, 5)
    tok_vars = np.var(signed, axis=0)       # (5,)  per-token variance
    mean_tok_var = float(np.mean(tok_vars))

    agg_vars: Dict[str, float] = {}

    # Named weighted schemes
    for name, w in WEIGHT_SCHEMES_VR.items():
        agg_vals = signed @ w               # (N_flat,)
        agg_vars[name] = float(np.var(agg_vals))

    # max_anom: max of direction-signed values across tokens
    max_vals = signed.max(axis=1)           # (N_flat,)
    agg_vars["max_anom"] = float(np.var(max_vals))

    # learned weights (if provided)
    if learned_weights is not None:
        lw_vals = signed @ learned_weights
        agg_vars["learned"] = float(np.var(lw_vals))

    # Empirical VRR for mean
    vrr_emp = (agg_vars["mean"] / mean_tok_var
               if mean_tok_var > 1e-12 else float("nan"))

    # Theoretical VRR from correlation matrix: (1 + (N-1)*rho_bar) / N
    _, rho_bar = _corr_stats(flat)
    vrr_theory = (1.0 + (N_TOK - 1) * rho_bar) / N_TOK if not np.isnan(rho_bar) else float("nan")

    return {
        "tok_vars":    tok_vars,
        "mean_tok_var": mean_tok_var,
        "agg_vars":    agg_vars,
        "vrr_emp":     vrr_emp,
        "vrr_theory":  vrr_theory,
        "rho_bar":     rho_bar,
    }


def compute_token_stats(
    succ_tok: np.ndarray,          # (N_succ, T, 5)
    fail_tok: np.ndarray,          # (N_fail, T, 5)
    direction: int,
    learned_weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute inter-token correlation and variance-reduction statistics for one
    feature group.  Reported for each phase separately AND all steps pooled.

    Returns dict keyed by (phase_name, label) → stats sub-dict.
    """
    results = {}
    pooled_tok = np.concatenate([succ_tok, fail_tok], axis=0)   # (N_all, T, 5)

    phase_windows = [("all", 0, MAX_STEPS - 1)] + [
        (ph, a, b) for ph, (a, b) in PHASE_DEFS.items()
    ]

    for phase_name, a, b in phase_windows:
        for label, tok in [("success", succ_tok),
                            ("failure", fail_tok),
                            ("pooled",  pooled_tok)]:
            flat = _flat_valid(tok, a, b)
            if len(flat) < 10:
                continue
            corr, rho_bar = _corr_stats(flat)
            vs = _variance_stats(flat, direction, learned_weights)
            results[(phase_name, label)] = {
                "corr_matrix":       corr,
                "mean_off_diag_corr": rho_bar,
                **vs,
                "n_obs":             len(flat),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITING
# ─────────────────────────────────────────────────────────────────────────────

MODES_ORDER = (
    ["tok0", "tok1", "tok2", "tok3", "tok4",
     "mean", "max_anom", "near_wt", "far_wt", "learned"]
)


def write_comparison_table(
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]],
    path: Path,
):
    """
    One table per feature group. Columns: mode × metric.
    Key comparison: does mean beat tok0? does max beat mean?
    """
    lines = [
        "=" * 120,
        "TOKEN AGGREGATION COMPARISON",
        f"alpha={ALPHA}  alpha_chunk={ALPHA_CHUNK:.4f}  win_W={WIN_W}",
        "Groups: " + ", ".join(all_group_results.keys()),
        "(conformal evaluated on within-group 50/50 cal/test split)",
        "=" * 120,
    ]

    for feat_name, stem, direction, mean_key in FEATURE_GROUPS:
        dir_sym = "F>S" if direction > 0 else "F<S"
        lines.append(f"\n{'─'*80}")
        lines.append(f"Feature group: {feat_name}  ({dir_sym})")
        lines.append(f"{'─'*80}")

        for g, group_results in all_group_results.items():
            feat_res = group_results.get(feat_name, {})
            if not feat_res:
                lines.append(f"  {g}: no data"); continue

            lines.append(f"\n  Group: {g}")
            lines.append(
                f"  {'mode':<12} "
                f"{'peak_auc':>8} {'early':>7} {'mid':>7} {'late':>7}  "
                f"{'G_rec':>6} {'G_fpr':>6} {'G_f1':>6}  "
                f"{'C_rec':>6} {'C_fpr':>6} {'C_f1':>6}  "
                f"{'C_med':>6}"
            )
            lines.append("  " + "-" * 100)

            mean_res = feat_res.get("mean", {})
            mean_peak = mean_res.get("peak_auroc", float("nan"))

            for mode in MODES_ORDER:
                res = feat_res.get(mode)
                if res is None: continue
                pa = res["phase_auroc"]
                g_m = res["global"]; c_m = res["chunked"]
                delta_sym = ""
                if mode != "mean":
                    d = res["peak_auroc"] - mean_peak
                    delta_sym = f"  Δ{d:+.3f}"
                lines.append(
                    f"  {mode:<12} "
                    f"{res['peak_auroc']:>8.3f} "
                    f"{pa.get('early',float('nan')):>7.3f} "
                    f"{pa.get('mid',float('nan')):>7.3f} "
                    f"{pa.get('late',float('nan')):>7.3f}  "
                    f"{g_m['recall']:>6.3f} {g_m['fpr']:>6.3f} {g_m['f1']:>6.3f}  "
                    f"{c_m['recall']:>6.3f} {c_m['fpr']:>6.3f} {c_m['f1']:>6.3f}  "
                    f"{c_m['median_det']:>6.1f}"
                    f"{delta_sym}"
                )

            # Best mode summary (exclude internal stats keys)
            mode_keys  = [k for k in feat_res if not k.startswith("_")]
            best_mode  = max(mode_keys, key=lambda m: feat_res[m].get("peak_auroc", 0.0))
            best_peak  = feat_res[best_mode]["peak_auroc"]
            early_best = max(mode_keys,
                             key=lambda m: feat_res[m]["phase_auroc"].get("early", 0.0))
            lines.append(
                f"\n  → Best overall: {best_mode} (peak={best_peak:.3f})  "
                f"Best early: {early_best} "
                f"(early={feat_res[early_best]['phase_auroc'].get('early',float('nan')):.3f})"
            )
            if "learned" in feat_res and feat_res["learned"]["learned_weights"] is not None:
                lw = feat_res["learned"]["learned_weights"]
                lines.append(f"     Learned weights: " +
                              ", ".join(f"t{i}={w:.3f}" for i, w in enumerate(lw)))

    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_summary(
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]],
    path: Path,
):
    """Short summary: for each feature, does mean beat tok0? Does max beat mean?"""
    lines = ["=" * 80, "TOKEN AGGREGATION SUMMARY", "=" * 80,
             f"{'feature':<25} {'group':<22} "
             f"{'best_mode':>12} {'mean_pk':>8} {'tok0_pk':>8} {'max_pk':>8} "
             f"{'mean_beat_tok0':>14} {'max_beat_mean':>13}"]
    lines.append("-" * 80)

    for feat_name, *_ in FEATURE_GROUPS:
        for g, group_res in all_group_results.items():
            fr = group_res.get(feat_name, {})
            if not fr: continue
            mp  = fr.get("mean",     {}).get("peak_auroc", float("nan"))
            t0p = fr.get("tok0",     {}).get("peak_auroc", float("nan"))
            mxp = fr.get("max_anom", {}).get("peak_auroc", float("nan"))
            bm  = max((k for k in fr if not k.startswith("_")), key=lambda m: fr[m].get("peak_auroc", 0.0))
            lines.append(
                f"{feat_name:<25} {g:<22} "
                f"{bm:>12} {mp:>8.3f} {t0p:>8.3f} {mxp:>8.3f} "
                f"{'YES' if mp > t0p else 'NO':>14} "
                f"{'YES' if mxp > mp else 'NO':>13}"
            )
    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_correlation_diagnostics(
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]],
    path: Path,
):
    """
    token_correlation_diagnostics.txt

    For each feature group × group × phase, prints:
      - 5×5 inter-token correlation matrix (pooled over all episodes)
      - Per-outcome (success / failure) average off-diagonal correlation
      - Whether correlations are higher for success or failure
        (higher failure correlation → tokens collapse together during failure,
         meaning mean over tokens amplifies rather than reduces shared drift)
    """
    lines = ["=" * 90,
             "INTER-TOKEN CORRELATION DIAGNOSTICS",
             "Pooled over all (step, episode) pairs within each phase.",
             "rho_bar = mean off-diagonal Pearson correlation among tokens 0-4.",
             "=" * 90]

    for feat_name, stem, direction, _ in FEATURE_GROUPS:
        dir_sym = "F>S" if direction > 0 else "F<S"
        lines.append(f"\n{'─'*70}")
        lines.append(f"{feat_name}  ({dir_sym})")
        lines.append(f"{'─'*70}")

        for g, group_res in all_group_results.items():
            stats_all = group_res.get(feat_name, {}).get("_token_stats", {})
            if not stats_all:
                lines.append(f"  {g}: no data"); continue
            lines.append(f"\n  Group: {g}")

            for phase_name in ["all", "early", "mid", "late"]:
                lines.append(f"\n    Phase: {phase_name}")
                for label in ("success", "failure", "pooled"):
                    key = (phase_name, label)
                    st = stats_all.get(key)
                    if st is None:
                        lines.append(f"      {label:<9}: no data"); continue
                    corr = st["corr_matrix"]
                    rho  = st["mean_off_diag_corr"]
                    n    = st["n_obs"]
                    lines.append(
                        f"      {label:<9}: rho_bar={rho:+.3f}  n={n:6d}"
                    )

                # For pooled phase: print full 5×5 correlation matrix
                st_p = stats_all.get((phase_name, "pooled"))
                if st_p is not None:
                    corr = st_p["corr_matrix"]
                    lines.append("      Correlation matrix (pooled):")
                    lines.append("          " + "    ".join(f"t{i}" for i in range(5)))
                    for i in range(5):
                        row = "  ".join(f"{corr[i,j]:+.3f}" for j in range(5))
                        lines.append(f"      t{i}  {row}")

                # Flag: do success and failure differ in correlation?
                st_s = stats_all.get((phase_name, "success"))
                st_f = stats_all.get((phase_name, "failure"))
                if st_s and st_f:
                    d_rho = st_f["mean_off_diag_corr"] - st_s["mean_off_diag_corr"]
                    flag  = "↑ failure more correlated" if d_rho > 0.05 else (
                            "↓ success more correlated" if d_rho < -0.05 else "≈ similar")
                    lines.append(f"      Δrho (fail-succ) = {d_rho:+.3f}  {flag}")

    path.write_text("\n".join(lines))
    print(f"  → {path}")


def write_variance_reduction(
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]],
    path: Path,
):
    """
    token_variance_reduction.txt

    For each feature group × group × phase:
      - Per-token variance (direction-signed values)
      - Variance of mean, near-weighted, far-weighted, max_anom
      - Empirical VRR  = var(mean) / mean(var(token_i))
      - Theoretical VRR = (1 + 4*rho_bar) / 5
        (derived from the correlation matrix; should match empirical if data is well-behaved)

    Interpretation:
      VRR ≈ 1.0 : mean aggregation gives NO variance reduction (tokens fully correlated)
      VRR ≈ 0.2 : mean gives full N-fold reduction (tokens independent)
      VRR > 1.0 : mean amplifies shared drift (tokens anti-correlated — unusual)
    """
    lines = ["=" * 100,
             "VARIANCE-REDUCTION DIAGNOSTICS FOR TOKEN AGGREGATION",
             "VRR_emp    = var(mean_over_tokens) / mean(var(token_i))",
             "VRR_theory = (1 + 4*rho_bar) / 5  (from correlation matrix)",
             "VRR ≈ 1.0 → averaging does nothing  |  VRR ≈ 0.2 → full N-fold reduction",
             "=" * 100]

    for feat_name, stem, direction, _ in FEATURE_GROUPS:
        dir_sym = "F>S" if direction > 0 else "F<S"
        lines.append(f"\n{'─'*80}")
        lines.append(f"{feat_name}  ({dir_sym})")
        lines.append(f"{'─'*80}")

        for g, group_res in all_group_results.items():
            stats_all = group_res.get(feat_name, {}).get("_token_stats", {})
            if not stats_all:
                lines.append(f"  {g}: no data"); continue
            lines.append(f"\n  Group: {g}")
            lines.append(
                f"  {'phase':<6} {'label':<9} {'n_valid':>7} "
                f"{'rho_bar':>7} "
                + "".join(f"{'var_t'+str(i):>8}" for i in range(5))
                + f"  {'var_mean':>8} {'var_max':>8} {'var_near':>8} {'var_far':>8}"
                f"  {'VRR_emp':>8} {'VRR_thy':>8}"
            )
            lines.append("  " + "-" * 108)

            for phase_name in ["all", "early", "mid", "late"]:
                for label in ("pooled",):   # pooled is the most informative for VRR
                    key = (phase_name, label)
                    st  = stats_all.get(key)
                    if st is None: continue
                    tok_var_str = "".join(f"{v:>8.4f}" for v in st["tok_vars"])
                    av = st["agg_vars"]
                    lines.append(
                        f"  {phase_name:<6} {label:<9} {st['n_obs']:>7d} "
                        f"{st['rho_bar']:>7.3f} "
                        f"{tok_var_str}"
                        f"  {av.get('mean',float('nan')):>8.4f}"
                        f"  {av.get('max_anom',float('nan')):>8.4f}"
                        f"  {av.get('near_wt',float('nan')):>8.4f}"
                        f"  {av.get('far_wt',float('nan')):>8.4f}"
                        f"  {st['vrr_emp']:>8.3f}"
                        f"  {st['vrr_theory']:>8.3f}"
                    )

            # Summary conclusion for this feature×group
            st_all = stats_all.get(("all", "pooled"))
            if st_all:
                rho = st_all["mean_off_diag_corr"]
                vrr = st_all["vrr_emp"]
                if vrr > 0.85:
                    verdict = "Tokens highly correlated — averaging gives minimal variance benefit."
                elif vrr < 0.50:
                    verdict = "Tokens semi-independent — averaging meaningfully reduces variance."
                else:
                    verdict = "Moderate correlation — averaging gives partial variance reduction."
                lines.append(f"\n  ↳ Summary: rho_bar={rho:.3f}, VRR={vrr:.3f}. {verdict}")

    path.write_text("\n".join(lines))
    print(f"  → {path}")


def plot_auroc_curves(
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]],
    feat_name: str,
    path: Path,
):
    """AUROC curves for all modes, for one feature group across groups."""
    n_grp = len(all_group_results)
    fig, axes = plt.subplots(1, n_grp, figsize=(7 * n_grp, 4), sharey=True)
    if n_grp == 1: axes = [axes]
    steps = np.arange(MAX_STEPS)

    MODE_STYLE = {
        "tok0":     ("#1f77b4", "-",  2.5, "tok-0"),
        "tok4":     ("#ff7f0e", "-",  2.5, "tok-4"),
        "mean":     ("#2ca02c", "-",  3.0, "mean"),
        "max_anom": ("#d62728", "--", 2.5, "max"),
        "near_wt":  ("#9467bd", ":",  2.0, "near-wt"),
        "far_wt":   ("#8c564b", ":",  2.0, "far-wt"),
        "learned":  ("#e377c2", "-.", 2.5, "learned"),
    }

    for gi, (g, group_res) in enumerate(all_group_results.items()):
        ax = axes[gi]
        fr = group_res.get(feat_name, {})
        for mode, (color, ls, lw, label) in MODE_STYLE.items():
            res = fr.get(mode)
            if res is None: continue
            curve = res["auroc_curve"]
            valid = ~np.isnan(curve)
            ax.plot(steps[valid], curve[valid], color=color, ls=ls, lw=lw, label=label)

        for ph, (a, b) in PHASE_DEFS.items():
            ax.axvspan(a, b, alpha=0.06,
                       color={"early": "#2c7bb6", "mid": "#fdae61", "late": "#d7191c"}[ph])
        ax.axhline(0.5, color="black", lw=0.7, ls=":")
        ax.axhline(0.6, color="gray",  lw=0.5, ls="--", alpha=0.5)
        ax.set_ylim(0.3, 1.0); ax.set_xlabel("Step"); ax.set_title(g, fontsize=9)
        if gi == 0: ax.set_ylabel("AUROC (win_mean)"); ax.legend(fontsize=7)

    fig.suptitle(f"Token aggregation AUROC — {feat_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); print(f"  → {path}"); plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Loading caches ───────────────────────────────────────────────")
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

    print("\n── Analysing token aggregation modes ────────────────────────────")
    all_group_results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for g, data in all_ep_data.items():
        print(f"  {g}:")
        all_group_results[g] = analyse_group(data, g)

    print("\n── Writing reports ───────────────────────────────────────────────")
    write_comparison_table(all_group_results, OUT_DIR / "token_aggregation_comparison.txt")
    write_summary(all_group_results,          OUT_DIR / "token_aggregation_summary.txt")
    write_correlation_diagnostics(all_group_results,
                                  OUT_DIR / "token_correlation_diagnostics.txt")
    write_variance_reduction(all_group_results,
                             OUT_DIR / "token_variance_reduction.txt")

    print("\n── Plotting ──────────────────────────────────────────────────────")
    for feat_name, *_ in FEATURE_GROUPS:
        safe = feat_name.replace("/", "_")
        plot_auroc_curves(all_group_results, feat_name, OUT_DIR / f"curves_{safe}.png")

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
