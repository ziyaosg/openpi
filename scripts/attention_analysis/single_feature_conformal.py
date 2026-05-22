#!/usr/bin/env python3
"""
single_feature_conformal.py

Builds and evaluates one inductive conformal anomaly detector per feature,
directly from raw .npy step records (no cached pkl needed).

Eight features evaluated:
  1. action_horizon_align  win_mean   (F<S)
  2. action_spread         win_mean   (F>S)
  3. rn_gini_base          win_mean   (F<S)
  4. v_drift_base          win_mean   (F>S)
  5. state_norm            win_mean   (F<S)
  6. v_drift_task          win_mean   (F>S)
  7. agree_gc_rw_base      win_mean   (F>S)
  8. rn_gini_task          slope_to_t (F<S)

Usage:
    module load miniconda && conda activate openpi
    python single_feature_conformal.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (edit these to point at a different dataset)
# ─────────────────────────────────────────────────────────────────────────────

CAL_DIR  = Path("/home/zs377/scratch_pi_tkf6/zs377/policy_records_pi0_fast_libero_20260511_040629")
TEST_DIR = Path("/home/zs377/scratch_pi_tkf6/zs377/policy_records_pi0_fast_libero_20260513_021106")
OUT_DIR  = Path("/home/zs377/scratch_pi_tkf6/zs377/conformal_single_feature")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MAX_STEPS    = 60   # maximum steps we consider per episode
WIN_W        = 5    # causal window for win_mean: [t-WIN_W+1, t] = 5 steps
ALPHA        = 0.10 # desired false positive rate (10% of successes flagged)
N_ACTION_TOK = 5    # pi0fast produces 5 action tokens per step

# ── Attention layer ────────────────────────────────────────────────────────────
# The model stores 2 attention layers: [1, 16].
# Layer 16 is selected here, consistent with step_metrics_plot.py (TARGET_LAYER=16).
# This layer was chosen during visualization work because it shows the clearest
# modality-selective head structure. Index 1 in the stored tensor = layer 16.
TARGET_LAYER_IDX = 1   # index into outputs/debug/attn/weights axis-0

# ── Raw-weights head selection ─────────────────────────────────────────────────
# For agree_gc_rw_base we use the top-8 heads by absolute attention budget on the
# base camera span. Using 8 (not 3) matches conformal_analysis_v2.py: more heads
# give a more robust aggregate for the cosine agreement computation.
# The 3-head ratio-based selection in step_metrics_plot.py is tuned for clean
# single-head visualization, not for aggregate statistics.
RW_TOP_K = 8

# ── Patch grid ─────────────────────────────────────────────────────────────────
PATCH_H = PATCH_W = 16   # each camera image = 16×16 = 256 patches

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE REGISTRY
# Tuple: (feature_name, descriptor, direction)
#   direction = +1 → high value means failure  (F>S)
#   direction = -1 → low  value means failure  (F<S, negate to get score)
# The nonconformity score is always: direction * feature_value
# so that a large positive score means the step looks like a failure.
# ─────────────────────────────────────────────────────────────────────────────

FEATURES: List[Tuple[str, str, int]] = [
    ("action_horizon_align", "win_mean",   -1),
    ("action_spread",        "win_mean",   +1),
    ("rn_gini_base",         "win_mean",   -1),
    ("v_drift_base",         "win_mean",   +1),
    ("state_norm",           "win_mean",   -1),
    ("v_drift_task",         "win_mean",   +1),
    ("agree_gc_rw_base",     "win_mean",   +1),
    ("rn_gini_task",         "slope_to_t", -1),  # running slope from step 0 to t
]

FEAT_KEYS = [f[0] for f in FEATURES]

# ─────────────────────────────────────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def gini(v: np.ndarray) -> float:
    """Gini coefficient. Range [0, 1]; higher = more concentrated in a few elements."""
    a = np.sort(np.abs(v.ravel())).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s < 1e-12 or n == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Returns nan if either vector is near-zero."""
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def auroc_1d(fail_scores: np.ndarray, succ_scores: np.ndarray) -> float:
    """
    Mann-Whitney AUROC = P(fail_score > succ_score).
    0.5 = random, 1.0 = perfect separation (failures always score higher).
    """
    fail_scores = fail_scores[~np.isnan(fail_scores)]
    succ_scores = succ_scores[~np.isnan(succ_scores)]
    if len(fail_scores) == 0 or len(succ_scores) == 0:
        return float("nan")
    nf, ns = len(fail_scores), len(succ_scores)
    all_scores = np.concatenate([fail_scores, succ_scores])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    fail_rank_sum = ranks[:nf].sum()
    return float((fail_rank_sum - nf * (nf + 1) / 2) / (nf * ns))

# ─────────────────────────────────────────────────────────────────────────────
# STEP-LEVEL RAW FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw(
    rec: dict,
    prev_v_centroid_base: Optional[np.ndarray],
    prev_v_centroid_task: Optional[np.ndarray],
) -> Tuple[dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract raw scalar values for all 8 features from one step record.

    Returns
    -------
    feats            : dict mapping feature name → float (nan if unavailable)
    v_centroid_base  : (D,) value-vector centroid for base cam (pass to next step)
    v_centroid_task  : (D,) value-vector centroid for task tokens (pass to next step)
    """
    feats: dict = {}

    # ── Helper: read a (start, end) span from the record ──────────────────────
    def span(key):
        v = rec.get(key)
        if v is None:
            return None
        arr = np.asarray(v).ravel()
        return int(arr[0]), int(arr[1])

    sp_base = span("outputs/debug/spans/image/base_0_rgb")  # (0, 256)
    sp_task = span("outputs/debug/spans/task")               # (768, 785) = 17 task tokens
    # sp_state and sp_wrist not needed for these 8 features

    # ── ACTION FEATURES ───────────────────────────────────────────────────────
    actions    = np.asarray(rec["outputs/actions"]).astype(np.float64)  # (10, 7)
    step_norms = np.linalg.norm(actions, axis=1)                        # (10,)

    # action_spread: std of per-step action norms across the 10-step plan.
    # High in failure = the policy oscillates between large and small moves
    # within a single plan, indicating it doesn't know how hard to push.
    feats["action_spread"] = float(step_norms.std())

    # action_horizon_align: cosine similarity between the first and last
    # action in the 10-step horizon.
    # Low in failure = near-term and far-term actions point in very different
    # directions, meaning the model can't "see" toward a consistent goal.
    feats["action_horizon_align"] = cosine(actions[0], actions[-1])

    # ── STATE FEATURE ─────────────────────────────────────────────────────────
    state = np.asarray(rec["outputs/state"]).astype(np.float64)[:8]  # (8,)

    # state_norm: L2 norm of the robot's 8-dim joint configuration.
    # Low in failure = the robot is still close to its starting pose;
    # it hasn't actually engaged with the task object (stuck or lost).
    feats["state_norm"] = float(np.linalg.norm(state))

    # ── ATTENTION WEIGHTS at TARGET_LAYER ─────────────────────────────────────
    # Shape: (L=2, batch=1, H=8, S=819). Layer 16 is at stored index 1.
    attn_w   = np.asarray(rec["outputs/debug/attn/weights"])
    attn_hxs = attn_w[TARGET_LAYER_IDX, 0]   # (H=8, S=819)

    # ── VALUE VECTORS (last stored layer = layer 16) ───────────────────────────
    # Shape: (L=2, batch=1, S=819, K=1, D=256).
    # conformal_analysis_v2.py uses the last layer for v_drift because it captures
    # the most processed representation. Last stored layer happens to be layer 16,
    # so TARGET_LAYER_IDX and [-1] are equivalent here — but we use [-1] to be
    # explicit that this is "the final processed value vector".
    v_raw = np.asarray(rec["outputs/debug/attn/v"])
    V     = v_raw[-1, 0, :, 0, :]            # (S=819, D=256)

    # ── RN_GINI_BASE: Gini of normalized attention on the base camera ──────────
    # "rn" = raw_alpha/norm, the rollout-normalized attention (more faithful than
    # raw softmax weights which undercount indirect paths).
    # We average the heatmap over all 5 action tokens before computing Gini —
    # this matches conformal_analysis_v2.py and reduces per-token noise.
    # Low Gini in failure = attention is spread uniformly across all 256 patches
    # instead of concentrated on the task-relevant object/region.
    rn_base_list = []
    for i in range(N_ACTION_TOK):
        k = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/base_0_rgb"
        if k in rec:
            rn_base_list.append(np.asarray(rec[k])[0].astype(np.float32))  # (16, 16)
    if rn_base_list:
        feats["rn_gini_base"] = gini(np.mean(rn_base_list, axis=0))
    else:
        feats["rn_gini_base"] = float("nan")

    # ── RN_GINI_TASK: Gini of normalized attention on task (language) tokens ───
    # Same idea for the 17 task tokens instead of image patches.
    # Used as the raw value for feature 8 (slope_to_t), not win_mean.
    # Low Gini in failure = the model attends equally to all words in the
    # instruction, indicating it isn't grounding any specific phrase.
    rn_task_list = []
    for i in range(N_ACTION_TOK):
        k = f"outputs/debug/raw_alpha/norm/action_tokens/{i}/task"
        if k in rec:
            rn_task_list.append(np.asarray(rec[k])[0].astype(np.float32))  # (17,)
    if rn_task_list:
        feats["rn_gini_task"] = gini(np.mean(rn_task_list, axis=0))
    else:
        feats["rn_gini_task"] = float("nan")

    # ── AGREE_GC_RW_BASE: cosine(GradCAM, raw-weights-max) on base camera ─────
    # GradCAM measures which patches CAUSE the output (gradient signal).
    # Raw-weights-max is the maximum attention weight over the top-K most
    # base-focused heads — it measures where the model physically LOOKS.
    # High agreement in failure means both methods point to the same place,
    # suggesting the model's computation has degenerated: it's attending to
    # exactly the places gradient says matter, but those places aren't helping.
    # In success, GradCAM and attention often disagree because the model does
    # richer computation (attention ≠ gradient importance).
    gc_base_list = []
    for i in range(N_ACTION_TOK):
        k = f"outputs/debug/gradcam/action_tokens/{i}/base_0_rgb"
        if k in rec:
            gc_base_list.append(np.abs(np.asarray(rec[k])[0]).astype(np.float32))  # (16, 16)

    if gc_base_list and sp_base is not None:
        gc_avg = np.mean(gc_base_list, axis=0)   # (16, 16)

        # Select top-RW_TOP_K heads by absolute attention budget to base span.
        # "budget" = total attention mass the head puts on base cam tokens.
        # We use budget (not ratio) because for the aggregate max we want the
        # heads that most strongly attend to the camera, regardless of whether
        # they also attend elsewhere.
        base_budget = attn_hxs[:, sp_base[0]:sp_base[1]].sum(axis=1)  # (H,)
        top_heads   = np.argsort(base_budget)[-RW_TOP_K:]
        # Take the maximum across the selected heads: for each patch token,
        # use the highest attention from any of the top heads.
        rw_base     = attn_hxs[top_heads, sp_base[0]:sp_base[1]].max(axis=0)  # (256,)
        rw_base_2d  = rw_base.reshape(PATCH_H, PATCH_W)

        feats["agree_gc_rw_base"] = cosine(gc_avg.ravel(), rw_base_2d.ravel())
    else:
        feats["agree_gc_rw_base"] = float("nan")

    # ── V_DRIFT_BASE: temporal drift of base-cam value-vector centroid ─────────
    # The centroid of the value vectors for base-cam tokens is the model's
    # aggregate "internal representation" of what it sees in the base camera.
    # drift = 1 - cosine(centroid_t, centroid_{t-1}): how much it rotated
    # in representation space between consecutive steps.
    # High drift in failure = the model's internal state is churning;
    # in success the representation evolves smoothly.
    v_centroid_base: Optional[np.ndarray] = None
    if sp_base is not None:
        v_base_seg      = V[sp_base[0]:sp_base[1]]   # (256, 256)
        v_centroid_base = v_base_seg.mean(axis=0)     # (256,) — D=256 value dim

    if v_centroid_base is not None and prev_v_centroid_base is not None:
        feats["v_drift_base"] = 1.0 - cosine(v_centroid_base, prev_v_centroid_base)
    else:
        feats["v_drift_base"] = float("nan")   # step 0 has no previous centroid

    # ── V_DRIFT_TASK: temporal drift of task-token value-vector centroid ───────
    # Same as v_drift_base but for the 17 language-instruction tokens.
    # High drift in failure = the model's internal task encoding is unstable,
    # suggesting it is losing track of what it is supposed to do.
    v_centroid_task: Optional[np.ndarray] = None
    if sp_task is not None:
        v_task_seg      = V[sp_task[0]:sp_task[1]]   # (17, 256)
        v_centroid_task = v_task_seg.mean(axis=0)     # (256,)

    if v_centroid_task is not None and prev_v_centroid_task is not None:
        feats["v_drift_task"] = 1.0 - cosine(v_centroid_task, prev_v_centroid_task)
    else:
        feats["v_drift_task"] = float("nan")

    return feats, v_centroid_base, v_centroid_task


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_episode(data_dir: Path, ep: dict) -> List[dict]:
    """
    Load all steps for one episode and return a list of raw feature dicts.
    The list has length = number of successfully loaded steps (≤ MAX_STEPS).
    Each dict has one key per feature in FEAT_KEYS.
    """
    raw_feats: List[dict] = []
    prev_vb = prev_vt = None   # value centroids for v_drift (passed step-to-step)

    for rel in range(MAX_STEPS):
        global_step = ep["start_idx"] + rel
        if global_step > ep["end_idx"]:
            break
        fpath = data_dir / f"step_{global_step}.npy"
        if not fpath.exists():
            break
        try:
            rec = np.load(fpath, allow_pickle=True).item()
            feats, prev_vb, prev_vt = extract_raw(rec, prev_vb, prev_vt)
        except Exception as e:
            print(f"    [warn] step {global_step}: {e}")
            # Insert all-nan so the episode length is preserved
            feats = {k: float("nan") for k in FEAT_KEYS}
            prev_vb = prev_vt = None
        raw_feats.append(feats)

    return raw_feats


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTOR COMPUTATION (raw per-step values → score per step)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score_sequence(
    raw_feats: List[dict],
    feat_name: str,
    descriptor: str,
    direction: int,
) -> np.ndarray:
    """
    Convert a list of per-step raw feature dicts into a score time series.

    descriptor = "win_mean":
        score[t] = direction * mean(raw[t-WIN_W+1 : t+1])
        Uses a causal window of WIN_W=5 steps.
        The window is right-padded with nans for early steps (t < WIN_W-1),
        so win_mean is just the mean of available values. This avoids NaN
        scores at the start but gives less-stable estimates early on.

    descriptor = "slope_to_t":
        score[t] = direction * slope of linear regression of raw[0..t]
        Requires at least 3 valid observations; returns nan otherwise.
        This captures the cumulative trend from the episode start.
        The signal for rn_gini_task peaks around step 15 (hence "slope@15"),
        but we compute it at every step so the predictor can flag early.

    Returns np.ndarray of shape (T,) where T = len(raw_feats).
    """
    T    = len(raw_feats)
    vals = np.array([r.get(feat_name, float("nan")) for r in raw_feats], dtype=np.float64)
    scores = np.full(T, float("nan"), dtype=np.float64)

    if descriptor == "win_mean":
        for t in range(T):
            lo   = max(0, t - WIN_W + 1)
            win  = vals[lo : t + 1]
            valid = win[~np.isnan(win)]
            if len(valid) > 0:
                scores[t] = direction * float(valid.mean())

    elif descriptor == "slope_to_t":
        # Running linear regression: fit a line to vals[0..t] and take the slope.
        # Minimum 3 valid points required to get a meaningful slope estimate.
        for t in range(T):
            xs  = vals[: t + 1]
            idx = np.where(~np.isnan(xs))[0].astype(float)
            vs  = xs[~np.isnan(xs)]
            if len(vs) >= 3:
                slope = float(np.polyfit(idx, vs, 1)[0])
                scores[t] = direction * slope
            # else leave as nan

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load all episodes from data_dir/client_output/episode_summaries.json.

    Returns:
      success_score_seqs — list of (T, 8) arrays, one per success episode
      failure_score_seqs — list of (T, 8) arrays, one per failure episode

    Column order matches FEATURES list.
    """
    with open(data_dir / "client_output" / "episode_summaries.json") as f:
        episodes = json.load(f)

    success_raw: List[List[dict]] = []
    failure_raw: List[List[dict]] = []

    print(f"  Loading {len(episodes)} episodes from {data_dir.name} …")
    for i, ep in enumerate(episodes):
        label = "success" if ep["success"] else "failure"
        raw = load_episode(data_dir, ep)
        if not raw:
            continue
        if ep["success"]:
            success_raw.append(raw)
        else:
            failure_raw.append(raw)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(episodes)} done")

    print(f"  → {len(success_raw)} success, {len(failure_raw)} failure episodes")

    # Convert to per-feature score sequences
    # Each score_seq[ep] has shape (T, n_features)
    def to_score_seqs(ep_list: List[List[dict]]) -> List[np.ndarray]:
        seqs = []
        for raw in ep_list:
            T      = len(raw)
            scores = np.full((T, len(FEATURES)), float("nan"), dtype=np.float64)
            for fi, (fname, desc, direction) in enumerate(FEATURES):
                scores[:, fi] = compute_score_sequence(raw, fname, desc, direction)
            seqs.append(scores)
        return seqs

    return to_score_seqs(success_raw), to_score_seqs(failure_raw)


# ─────────────────────────────────────────────────────────────────────────────
# AUROC CURVES (diagnostic — verify feature discriminability before calibrating)
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc_curves(
    succ_seqs: List[np.ndarray],
    fail_seqs: List[np.ndarray],
) -> np.ndarray:
    """
    Compute AUROC at each step for each feature.
    Returns array of shape (MAX_STEPS, n_features).
    """
    n_feat = len(FEATURES)
    aurocs = np.full((MAX_STEPS, n_feat), float("nan"), dtype=np.float64)

    for t in range(MAX_STEPS):
        for fi in range(n_feat):
            fail_s = np.array([seq[t, fi] for seq in fail_seqs if t < len(seq)])
            succ_s = np.array([seq[t, fi] for seq in succ_seqs if t < len(seq)])
            aurocs[t, fi] = auroc_1d(fail_s, succ_s)

    return aurocs


# ─────────────────────────────────────────────────────────────────────────────
# CONFORMAL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(succ_seqs: List[np.ndarray]) -> np.ndarray:
    """
    Inductive conformal anomaly detection (ICAD) calibration.

    For each success episode, compute the worst-case (maximum) score across
    all steps. The conformal threshold τ is the (1-alpha) quantile of these
    max scores.

    Guarantee: P(any step of a success episode scores above τ) ≤ alpha.
    This is the marginal coverage guarantee from ICAD (Shafer & Vovk 2008).

    Why max over steps?
    We want to flag an episode if it is EVER anomalous. A success episode
    should never trigger the alarm; calibrating on the max score across all
    steps ensures that even the worst moment in a success episode is below τ
    with probability (1-alpha).

    Returns thresholds of shape (n_features,).
    """
    n_feat = len(FEATURES)
    # max_scores[ep, fi] = max score of success episode ep for feature fi
    max_scores = np.full((len(succ_seqs), n_feat), float("nan"), dtype=np.float64)
    for ei, seq in enumerate(succ_seqs):
        for fi in range(n_feat):
            col = seq[:, fi]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                max_scores[ei, fi] = float(valid.max())

    # Conformal threshold at (1-alpha) quantile.
    # np.nanquantile ignores nan entries (episodes where all steps were nan).
    thresholds = np.nanquantile(max_scores, 1.0 - ALPHA, axis=0)
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    succ_seqs: List[np.ndarray],
    fail_seqs: List[np.ndarray],
    thresholds: np.ndarray,
) -> dict:
    """
    Evaluate each single-feature conformal predictor on a test set.

    A test episode is "flagged" if its score exceeds threshold τ at any step.
    Detection step = first step where score > τ (nan if never flagged).

    Metrics per feature:
      recall    = fraction of failure episodes flagged                 (target ≥ 0.80)
      fpr       = fraction of success episodes flagged                 (target ≤ alpha)
      precision = fraction of flagged episodes that are failures
      f1        = harmonic mean of recall and precision
      median_det = median detection step among correctly-flagged failures

    Returns dict with arrays of shape (n_features,) per metric.
    """
    n_feat = len(FEATURES)

    # Per-episode flag and detection step
    def _check(seqs, label):
        flagged    = np.zeros((len(seqs), n_feat), dtype=bool)
        det_steps  = np.full((len(seqs), n_feat), float("nan"), dtype=np.float64)
        for ei, seq in enumerate(seqs):
            for fi in range(n_feat):
                col = seq[:, fi]
                for t, s in enumerate(col):
                    if not np.isnan(s) and s > thresholds[fi]:
                        flagged[ei, fi]   = True
                        det_steps[ei, fi] = t
                        break  # first flagged step only
        return flagged, det_steps

    fail_flagged, fail_det = _check(fail_seqs, "failure")
    succ_flagged, _        = _check(succ_seqs, "success")

    results = {}
    for fi in range(n_feat):
        n_fail = len(fail_seqs); n_succ = len(succ_seqs)

        tp = int(fail_flagged[:, fi].sum())   # correctly flagged failures
        fp = int(succ_flagged[:, fi].sum())   # incorrectly flagged successes

        recall    = tp / n_fail if n_fail > 0 else float("nan")
        fpr       = fp / n_succ if n_succ > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float("nan"))

        det = fail_det[fail_flagged[:, fi], fi]   # detection steps of TPs only
        median_det = float(np.nanmedian(det)) if len(det) > 0 else float("nan")

        results[fi] = {
            "recall": recall, "fpr": fpr, "precision": precision,
            "f1": f1, "median_det": median_det,
            "tp": tp, "fp": fp, "n_fail": n_fail, "n_succ": n_succ,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_curves(aurocs: np.ndarray, title_suffix: str, fname: str):
    """8-panel AUROC curves (one per feature) with a 0.6 dashed reference line."""
    n_feat = len(FEATURES)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    steps = np.arange(MAX_STEPS)

    for fi, (fname_feat, desc, direction) in enumerate(FEATURES):
        ax = axes[fi]
        curve = aurocs[:, fi]
        ax.plot(steps, curve, color="#1f5fa6", lw=2)
        ax.axhline(0.6, color="gray", lw=1, ls="--", alpha=0.6)
        ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.4)
        ax.set_title(f"{fname_feat}\n({desc})", fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("AUROC")
        ax.grid(True, alpha=0.3)

        # annotate peak AUROC
        peak_t = int(np.nanargmax(curve))
        peak_v = float(np.nanmax(curve))
        ax.annotate(f"peak {peak_v:.3f}@t={peak_t}", xy=(peak_t, peak_v),
                    fontsize=7, color="#b02020",
                    xytext=(peak_t + 2, peak_v - 0.05),
                    arrowprops=dict(arrowstyle="->", color="#b02020", lw=0.8))

    fig.suptitle(f"AUROC curves — {title_suffix}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


def plot_score_distributions(
    cal_succ: List[np.ndarray],
    test_succ: List[np.ndarray],
    test_fail: List[np.ndarray],
    thresholds: np.ndarray,
    eval_step: int = 15,
):
    """
    Histogram of score distributions at a fixed step for each feature.
    Shows calibration-success, test-success, and test-failure distributions.
    """
    n_feat = len(FEATURES)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.ravel()

    def _extract_step(seqs, t, fi):
        vals = [seq[t, fi] for seq in seqs if t < len(seq)]
        return np.array([v for v in vals if not np.isnan(v)])

    for fi, (fname_feat, desc, _) in enumerate(FEATURES):
        ax = axes[fi]
        cs = _extract_step(cal_succ,  eval_step, fi)
        ts = _extract_step(test_succ, eval_step, fi)
        tf = _extract_step(test_fail, eval_step, fi)

        all_vals = np.concatenate([cs, ts, tf])
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) == 0:
            continue
        lo, hi = np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99)
        bins = np.linspace(lo, hi, 30)

        ax.hist(cs, bins=bins, alpha=0.5, color="gray",   label="cal-succ", density=True)
        ax.hist(ts, bins=bins, alpha=0.5, color="#1f5fa6", label="test-succ", density=True)
        ax.hist(tf, bins=bins, alpha=0.5, color="#b02020", label="test-fail", density=True)
        ax.axvline(thresholds[fi], color="black", lw=1.5, ls="--", label=f"τ={thresholds[fi]:.3f}")
        ax.set_title(f"{fname_feat} @ step {eval_step}", fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score distributions at step {eval_step}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    aurocs_cal: np.ndarray,
    aurocs_test: np.ndarray,
    thresholds: np.ndarray,
    results: dict,
):
    """Write a text summary table to OUT_DIR/report.txt and print it."""
    lines = []
    lines.append("=" * 90)
    lines.append("SINGLE-FEATURE CONFORMAL PREDICTOR REPORT")
    lines.append(f"Calibration: {CAL_DIR.name}")
    lines.append(f"Test:        {TEST_DIR.name}")
    lines.append(f"alpha = {ALPHA}  →  FPR target ≤ {ALPHA:.0%}  |  Window W = {WIN_W}")
    lines.append("=" * 90)

    # AUROC summary table
    lines.append("")
    lines.append("AUROC on calibration set (peak and step):")
    hdr = f"  {'Feature':<28} {'desc':<12} {'peak_AUROC':>10} {'peak_step':>10}"
    lines.append(hdr)
    lines.append("  " + "-" * 64)
    for fi, (fname, desc, _) in enumerate(FEATURES):
        peak_t = int(np.nanargmax(aurocs_cal[:, fi]))
        peak_v = float(np.nanmax(aurocs_cal[:, fi]))
        lines.append(f"  {fname:<28} {desc:<12} {peak_v:>10.3f} {peak_t:>10d}")

    lines.append("")
    lines.append("AUROC on test set (peak and step):")
    lines.append(hdr)
    lines.append("  " + "-" * 64)
    for fi, (fname, desc, _) in enumerate(FEATURES):
        peak_t = int(np.nanargmax(aurocs_test[:, fi]))
        peak_v = float(np.nanmax(aurocs_test[:, fi]))
        lines.append(f"  {fname:<28} {desc:<12} {peak_v:>10.3f} {peak_t:>10d}")

    # Conformal evaluation table
    lines.append("")
    lines.append("Conformal predictor evaluation on TEST set:")
    lines.append(f"  {'Feature':<28} {'τ':>8} {'Recall':>8} {'FPR':>8} {'Prec':>8} {'F1':>8} {'MedDet':>8}")
    lines.append("  " + "-" * 80)
    for fi, (fname, desc, _) in enumerate(FEATURES):
        r = results[fi]
        lines.append(
            f"  {fname:<28} {thresholds[fi]:>8.4f} "
            f"{r['recall']:>8.3f} {r['fpr']:>8.3f} "
            f"{r['precision']:>8.3f} {r['f1']:>8.3f} "
            f"{r['median_det']:>8.1f}"
        )
    lines.append("  " + "-" * 80)
    lines.append(f"  Recall = TP/n_fail, FPR = FP/n_succ, MedDet = median step of first flag among TPs")
    lines.append("=" * 90)

    report = "\n".join(lines)
    print(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.txt").write_text(report)
    print(f"\n  → {OUT_DIR / 'report.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n── Step 1: Load calibration set ──────────────────────────────────────")
    cal_succ, cal_fail = load_dataset(CAL_DIR)

    print("\n── Step 2: Load test set ─────────────────────────────────────────────")
    test_succ, test_fail = load_dataset(TEST_DIR)

    print("\n── Step 3: Compute AUROC curves (diagnostic) ─────────────────────────")
    aurocs_cal  = compute_auroc_curves(cal_succ, cal_fail)
    aurocs_test = compute_auroc_curves(test_succ, test_fail)
    plot_auroc_curves(aurocs_cal,  "calibration set", "auroc_calibration.png")
    plot_auroc_curves(aurocs_test, "test set",        "auroc_test.png")

    print("\n── Step 4: Calibrate conformal thresholds ────────────────────────────")
    # Thresholds are calibrated only on the CALIBRATION success episodes.
    # The test set is never used for threshold selection.
    thresholds = calibrate(cal_succ)
    for fi, (fname, desc, _) in enumerate(FEATURES):
        print(f"  {fname:<28} ({desc}) τ = {thresholds[fi]:.4f}")

    print("\n── Step 5: Evaluate on test set ──────────────────────────────────────")
    results = evaluate(test_succ, test_fail, thresholds)

    print("\n── Step 6: Plots ─────────────────────────────────────────────────────")
    plot_score_distributions(cal_succ, test_succ, test_fail, thresholds, eval_step=15)

    print("\n── Step 7: Report ────────────────────────────────────────────────────")
    write_report(aurocs_cal, aurocs_test, thresholds, results)


if __name__ == "__main__":
    main()
