#!/usr/bin/env python3
"""
Failure-predictor statistics: numerical comparison of attention and action
features between success and failure episodes, organised by detection phase.

Run:
    python -m scripts.attention_analysis.failure_stats
  or
    python scripts/attention_analysis/failure_stats.py

Output is plain text — pipe through `less` or redirect to a file.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import stats as scipy_stats

# ─────────────────────────── CONFIG ───────────────────────────
DATA_DIR = Path("/home/ziyao/Documents/policy_records_20260423_180932")
EPISODE_JSON = DATA_DIR / "episode_summaries.json"

# Only tasks with a mix of success and failure outcomes.
# _20260423_: task 6 added (7S/3F); _20260421_: task 6 had only 1 failure (excluded).
MIXED_TASK_IDS: set[int] = {2, 3, 5, 6, 9}

# Step windows for the three analysis phases.
PHASE1_STEPS = (0, 10)   # latent divergence
PHASE2_STEPS = (10, 25)  # emerging divergence
PHASE3_STEPS = (25, 50)  # visible failure loop

# ──────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# TERMINAL FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════

W = 76

def _rule(char="─"):   print(char * W)
def _double_rule():    print("═" * W)

def _header(title: str):
    print(); _double_rule(); print(f"  {title}"); _double_rule()

def _phase_header(n: int, step_range: Tuple[int, int], subtitle: str):
    print(); _rule("━")
    print(f"  PHASE {n}  │  Steps {step_range[0]}–{step_range[1]}  │  {subtitle}")
    _rule("━")

def _section(title: str):
    print()
    print(f"  ── {title} " + "─" * max(0, W - len(title) - 6))

def _formula(label: str, lines: List[str]):
    indent = "       "
    print(f"\n  [{label}]")
    for line in lines:
        print(f"{indent}{line}")

def _sig_star(p: float) -> str:
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★ "
    if p < 0.05:  return "★  "
    if p < 0.10:  return "·  "
    return "   "


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def _load_spans(rec: dict):
    def _sp(key): return tuple(int(x) for x in np.asarray(rec[key]).tolist())
    return (
        _sp("outputs/debug/spans/image/base_0_rgb"),
        _sp("outputs/debug/spans/image/left_wrist_0_rgb"),
        _sp("outputs/debug/spans/image/right_wrist_0_rgb"),
        _sp("outputs/debug/spans/task"),
        _sp("outputs/debug/spans/state"),
    )


def _gini(values: np.ndarray) -> float:
    a = np.sort(np.abs(values)).astype(np.float64)
    n = len(a); s = float(a.sum())
    if s == 0 or n == 0: return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * a) / (n * s) - (n + 1) / n)


def extract_step_features(rec: dict) -> dict:
    attn_w   = np.asarray(rec["outputs/debug/attn/weights"])  # (2, 1, H, S)
    attn_l16 = attn_w[1, 0]   # (H=8, S=819)
    full_v   = np.asarray(rec["outputs/debug/attn/v"])        # (2, 1, S, 1, D)  K=1
    actions  = np.asarray(rec["outputs/actions"])             # (10, 7)
    state    = np.asarray(rec["outputs/state"])               # (8,)
    gmap_b   = np.asarray(rec["outputs/debug/gradcam/image/base_0_rgb"]).reshape(-1)   # (256,)

    sp_base, sp_w1, sp_w2, sp_task, sp_state = _load_spans(rec)

    # ── Base-camera attention (mean over heads) ───────────────
    base_attn_patch = attn_l16[:, sp_base[0]:sp_base[1]].mean(axis=0)  # (256,)

    # ── Value vectors (K=1 GQA) ──────────────────────────────
    v0     = full_v[1, 0, :, 0, :]    # (S, D)
    v_norms = np.linalg.norm(v0, axis=-1)   # (S,)

    # ─────────────────────────────────────────────────────────
    # PHASE 1 FEATURES — detectable steps 0-9
    # ─────────────────────────────────────────────────────────

    # [P1a] Gini coefficient on base-camera patch attention
    gini_base = _gini(base_attn_patch)

    # [P1b] Spatial entropy of base-camera attention
    p_base = base_attn_patch / (base_attn_patch.sum() + 1e-12)
    spatial_ent = float(-np.sum(p_base * np.log(p_base + 1e-12)))

    # [P1c] Action jerk: mean |Δa| across predicted horizon
    action_jerk = float(np.abs(np.diff(actions, axis=0)).mean())

    # [P1d] Head-0 image attention mass (layer 16)
    head0_img_attn = float(attn_l16[0, sp_base[0]:sp_base[1]].sum())

    # [P1e] NEW — Task-token head agreement:
    #   inter-head cosine similarity of task-token attention distributions.
    #   High similarity = all heads agree on which words to read (unified understanding).
    task_per_head = attn_l16[:, sp_task[0]:sp_task[1]]   # (H, N_task)
    t_norms = np.linalg.norm(task_per_head, axis=1, keepdims=True) + 1e-9
    t_normed = task_per_head / t_norms
    sim_task = t_normed @ t_normed.T   # (H, H)
    iu = np.triu_indices(attn_l16.shape[0], k=1)
    task_head_agreement = float(sim_task[iu].mean())

    # ─────────────────────────────────────────────────────────
    # PHASE 2 FEATURES — detectable steps 10-24
    # ─────────────────────────────────────────────────────────

    # [P2a] All-image attention fraction (base + both wrists)
    img_attn_per_head = (
        attn_l16[:, sp_base[0]:sp_base[1]].sum(axis=1)
        + attn_l16[:, sp_w1[0]:sp_w1[1]].sum(axis=1)
        + attn_l16[:, sp_w2[0]:sp_w2[1]].sum(axis=1)
    )
    img_attn_frac = float(img_attn_per_head.mean() / attn_l16.shape[1])

    # [P2b] Task-token attention fraction
    task_attn_frac = float(
        attn_l16[:, sp_task[0]:sp_task[1]].sum(axis=1).mean() / attn_l16.shape[1]
    )

    # [P2c] State-token attention fraction
    state_attn_frac = float(
        attn_l16[:, sp_state[0]:sp_state[1]].sum(axis=1).mean() / attn_l16.shape[1]
    )

    # [P2d] V-norm weighted attention on base camera (max over heads per patch)
    vnorm_score = (attn_l16 * v_norms[np.newaxis, :]).max(axis=0)   # (S,)
    vnorm_base = float(vnorm_score[sp_base[0]:sp_base[1]].mean())

    # [P2e] NEW — Gripper open fraction:
    #   fraction of horizon steps where gripper dim (index 6) > 0 (opening).
    gripper_open_frac = float((actions[:, 6] > 0).mean())

    # [P2e+] NEW — Gripper open sign-flip rate:
    #   fraction of consecutive horizon pairs where gripper dim changes sign.
    #   Distinguishes hesitant oscillation (1010…) from committed transitions (1111…0000…),
    #   both of which can yield the same gripper_open_frac.
    gripper_signs = np.sign(actions[:, 6])
    gripper_flip_rate = float((gripper_signs[1:] != gripper_signs[:-1]).mean())

    # [P2f] NEW — Action sign-flip rate:
    #   mean fraction of dims that change sign between consecutive horizon steps.
    signs = np.sign(actions)
    sign_flip_rate = float((signs[1:] != signs[:-1]).mean())

    # [P2g] NEW — Attention head entropy variance:
    #   variance of per-head attention entropy over the full sequence.
    #   High variance = heads have diverse attention strategies.
    p_h = np.clip(attn_l16, 1e-9, 1.0)
    per_head_ent = -(p_h * np.log(p_h)).sum(axis=1)   # (H,)
    head_ent_var = float(per_head_ent.var())

    # [P2h] NEW — Value-projection alignment on base patches:
    #   |cosine(v[p], o_h)| averaged over image patches and heads.
    #   High = the model actively "uses" image patch information in its output.
    o = attn_l16 @ v0          # (H, D)  head outputs
    o_unit = o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-9)
    v_base = v0[sp_base[0]:sp_base[1]]   # (256, D)
    v_base_unit = v_base / (np.linalg.norm(v_base, axis=-1, keepdims=True) + 1e-9)
    cos_vp = v_base_unit @ o_unit.T      # (256, H)
    v_proj_align = float(np.abs(cos_vp).mean())

    # [P2i] NEW — GradCAM energy on base camera (mean |grad|):
    #   High energy = output is sensitive to base camera inputs.
    #   In failures the output becomes increasingly sensitive (erratic).
    gradcam_base_energy = float(np.abs(gmap_b).mean())

    # ─────────────────────────────────────────────────────────
    # PHASE 3 FEATURES — detectable steps 25+
    # ─────────────────────────────────────────────────────────

    # [P3a] Action std across all dims and horizon steps
    action_std = float(actions.std())

    # [P3b] Mean |a| across all dims and horizon steps
    action_mag = float(np.abs(actions).mean())

    return {
        # Phase 1
        "gini_base":           gini_base,
        "spatial_ent":         spatial_ent,
        "action_jerk":         action_jerk,
        "head0_img_attn":      head0_img_attn,
        "task_head_agreement": task_head_agreement,   # NEW P1e
        # Phase 2
        "img_attn_frac":       img_attn_frac,
        "task_attn_frac":      task_attn_frac,
        "state_attn_frac":     state_attn_frac,
        "vnorm_base":          vnorm_base,
        "gripper_open_frac":   gripper_open_frac,     # NEW P2e
        "gripper_flip_rate":   gripper_flip_rate,     # NEW P2e+
        "sign_flip_rate":      sign_flip_rate,        # NEW P2f
        "head_ent_var":        head_ent_var,          # NEW P2g
        "v_proj_align":        v_proj_align,          # NEW P2h
        "gradcam_base_energy": gradcam_base_energy,   # NEW P2i
        # Phase 3
        "action_std":          action_std,
        "action_mag":          action_mag,
        # Pass-through (used by caller for stateful features)
        "state":               state,
    }


def load_episode_window(ep: dict, step_range: Tuple[int, int]) -> List[dict]:
    """Load feature dicts for a given relative-step range [lo, hi).
    Stateful features (needing previous step) are computed here:
      state_vel, max_joint_vel, joint_vel_entropy, plan_consistency, attn_peak_drift.
    """
    lo, hi = step_range
    records: List[dict] = []
    prev_state       = None
    prev_action0     = None
    prev_attn_peak   = None

    for rel in range(lo, hi):
        s = ep["start_idx"] + rel
        if s > ep["end_idx"]:
            break
        fpath = DATA_DIR / f"step_{s}.npy"
        if not fpath.exists():
            continue
        rec = np.load(fpath, allow_pickle=True).item()
        feats = extract_step_features(rec)
        state = feats.pop("state")

        # [P2d] State velocity — L2 norm of joint-state change
        feats["state_vel"] = (
            float(np.linalg.norm(state - prev_state)) if prev_state is not None else 0.0
        )

        # [P2j] NEW — Max single-joint velocity (which joint is moving fastest)
        if prev_state is not None:
            joint_vel = np.abs(state - prev_state)   # (8,)
            feats["max_joint_vel"] = float(joint_vel.max())
            # [P2k] NEW — Joint velocity entropy:
            #   entropy of the velocity distribution across joints.
            #   High = motion spread across all joints (coordinated); low = one joint dominates.
            jv = joint_vel + 1e-9
            feats["joint_vel_entropy"] = float(-(jv / jv.sum() * np.log(jv / jv.sum())).sum())
        else:
            feats["max_joint_vel"]     = 0.0
            feats["joint_vel_entropy"] = 0.0
        prev_state = state

        # [P2l] NEW — Inter-step plan consistency:
        #   cosine similarity between consecutive step-0 action predictions.
        #   Drops in failures as the model oscillates in what to do next.
        cur_action0 = np.asarray(rec["outputs/actions"])[0]
        if prev_action0 is not None:
            feats["plan_consistency"] = float(
                np.dot(cur_action0, prev_action0) /
                (np.linalg.norm(cur_action0) * np.linalg.norm(prev_action0) + 1e-9)
            )
        else:
            feats["plan_consistency"] = 1.0
        prev_action0 = cur_action0

        # [P2m] NEW — Attention peak spatial drift:
        #   L2 distance (in patch-grid space) of the peak attention patch between steps.
        #   High drift = attention is erratic / unstable.
        sp_b = tuple(int(x) for x in np.asarray(rec["outputs/debug/spans/image/base_0_rgb"]).tolist())
        base_map = np.asarray(rec["outputs/debug/attn/weights"])[1, 0, :, sp_b[0]:sp_b[1]].mean(axis=0).reshape(16, 16)
        peak = np.array(np.unravel_index(base_map.argmax(), base_map.shape), dtype=float)
        feats["attn_peak_drift"] = (
            float(np.linalg.norm(peak - prev_attn_peak)) if prev_attn_peak is not None else 0.0
        )
        prev_attn_peak = peak

        records.append(feats)
    return records


def mean_window(records: List[dict]) -> dict:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


# ══════════════════════════════════════════════════════════════
# EPISODE TABLE PRINTER
# ══════════════════════════════════════════════════════════════

TASK_SHORT = {
    2: "stove+moka",
    3: "bowl+drawer",
    5: "book+caddy",
    6: "mug+choc",
    9: "mug+microwave",
}


def _print_episode_table(ep_rows: List[dict], cols: List[Tuple[str, str, str]]):
    ep_rows = sorted(ep_rows, key=lambda r: (r["task_id"], not r["success"], r["episode_num"]))
    hdr = f"  {'task':>12}  {'ep':>4}  {'outcome':>8}"
    for _, h, _ in cols:
        hdr += f"  {h:>10}"
    print(hdr)
    _rule()
    prev_task = None
    for row in ep_rows:
        tid = row["task_id"]
        if prev_task is not None and prev_task != tid:
            print()
        prev_task = tid
        tname   = TASK_SHORT.get(tid, f"task{tid}")
        outcome = "SUCCESS" if row["success"] else "FAILURE"
        ep_id   = f"ep{row['episode_num']:02d}"
        line    = f"  {tname:>12}  {ep_id:>4}  {outcome:>8}"
        for key, _, fmt in cols:
            val = row.get(key, float("nan"))
            line += f"  {val:{fmt}}"
        print(line)
    _rule()


def _print_summary(ep_rows: List[dict], cols: List[Tuple[str, str, str]]):
    s_rows = [r for r in ep_rows if r["success"]]
    f_rows = [r for r in ep_rows if not r["success"]]

    def _v(rows, key): return np.array([r[key] for r in rows if key in r])

    print(f"\n  {'':12}  {'N':>4}  ", end="")
    for _, h, _ in cols:
        print(f"  {h:>10}", end="")
    print()
    _rule()

    for label, rows in [("SUCCESS", s_rows), ("FAILURE", f_rows)]:
        line = f"  {label:12}  {len(rows):>4}  "
        for key, _, fmt in cols:
            v = _v(rows, key)
            line += f"  {v.mean():{fmt}}" if len(v) else f"  {'—':>10}"
        print(line)

    for row_label, fn in [("t-stat", lambda sv,fv: scipy_stats.ttest_ind(sv,fv,equal_var=False)[0]),
                           ("p-value", lambda sv,fv: scipy_stats.ttest_ind(sv,fv,equal_var=False)[1])]:
        line = f"  {row_label:12}  {'':>4}  "
        for key, _, fmt in cols:
            sv = _v(s_rows, key); fv = _v(f_rows, key)
            if len(sv) > 1 and len(fv) > 1:
                val = fn(sv, fv)
                if row_label == "p-value":
                    line += f"  {f'{val:.4f} {_sig_star(val)}':>10}"
                else:
                    line += f"  {val:>10.3f}"
            else:
                line += f"  {'—':>10}"
        print(line)

    line = f"  {'std S/F':12}  {'':>4}  "
    for key, _, fmt in cols:
        sv = _v(s_rows, key); fv = _v(f_rows, key)
        if len(sv) > 1 and len(fv) > 1:
            line += f"  {sv.std():{fmt}}/{fv.std():{fmt}}"
        else:
            line += f"  {'—':>10}"
    print(line)
    _rule()


def _print_temporal_table(ep_data: dict, feat: str,
                           step_range: Tuple[int, int],
                           fmt: str = "7.4f", every_n: int = 5):
    lo, hi = step_range
    ordered = sorted(ep_data.keys(),
                     key=lambda k: (ep_data[k]["task_id"],
                                    ep_data[k]["label"] != "success",
                                    ep_data[k]["episode_num"]))
    hdr = f"  {'step':>5}"
    for k in ordered:
        ep = ep_data[k]
        tag = f"t{ep['task_id']}e{ep['episode_num']:02d}{'S' if ep['label']=='success' else 'F'}"
        hdr += f"  {tag:>9}"
    print(hdr)
    _rule()
    for t in range(lo, hi, every_n):
        rel  = t - lo
        line = f"  {t:>5}"
        rs, rf = [], []
        for k in ordered:
            ep   = ep_data[k]
            vals = ep["per_step"].get(feat, [])
            if rel < len(vals):
                v = vals[rel]
                line += f"  {v:{fmt}}"
                (rs if ep["label"] == "success" else rf).append(v)
            else:
                line += f"  {'(done)':>9}"
        sm = np.mean(rs) if rs else float("nan")
        fm = np.mean(rf) if rf else float("nan")
        line += f"  │  {sm:{fmt}}  {fm:{fmt}}"
        print(line)
    _rule()
    print(f"  {'step':>5}" + "".join(f"  {'─'*9}" for _ in ordered)
          + f"  │  {'S_mean':>9}  {'F_mean':>9}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    with open(EPISODE_JSON) as f:
        all_episodes = json.load(f)

    mixed_eps = [e for e in all_episodes if e["task_id"] in MIXED_TASK_IDS]
    n_s = sum(1 for e in mixed_eps if e["success"])
    n_f = sum(1 for e in mixed_eps if not e["success"])

    _header("FAILURE PREDICTOR — ATTENTION & ACTION STATISTICS")
    print(f"\n  Data dir : {DATA_DIR}")
    print(f"  Tasks    : {sorted(MIXED_TASK_IDS)}  (mixed-outcome only)")
    print(f"  Episodes : {len(mixed_eps)} total  │  {n_s} SUCCESS  │  {n_f} FAILURE")
    print()
    print("  Significance stars:  ★★★ p<0.001   ★★ p<0.01   ★ p<0.05   · p<0.10")
    print()
    print("  Episode IDs in per-step tables:  t<task_id>e<ep_num><S|F>")
    print("  e.g.  t2e01S  =  task 2, episode 1, SUCCESS")

    # ── Pre-compute all features ─────────────────────────────────────────────
    print("\n  Loading features …", end="", flush=True)
    ep_data: dict = {}
    ep_rows_p1: List[dict] = []
    ep_rows_p2: List[dict] = []
    ep_rows_p3: List[dict] = []

    for ep in mixed_eps:
        key   = f"t{ep['task_id']}e{ep['episode_num']:02d}"
        label = "success" if ep["success"] else "failure"

        m1 = mean_window(load_episode_window(ep, PHASE1_STEPS))
        m2 = mean_window(load_episode_window(ep, PHASE2_STEPS))
        m3 = mean_window(load_episode_window(ep, PHASE3_STEPS))

        # Build full per-step feature arrays (0-49) for temporal tables
        per_step: dict[str, list] = defaultdict(list)
        for rel_feats in [load_episode_window(ep, (0, 50))]:
            for i, fd in enumerate(rel_feats):
                for fk, fv in fd.items():
                    per_step[fk].append(fv)

        ep_data[key] = {"label": label, "task_id": ep["task_id"],
                        "episode_num": ep["episode_num"],
                        "per_step": dict(per_step)}
        base = {"task_id": ep["task_id"], "episode_num": ep["episode_num"],
                "success": ep["success"]}
        ep_rows_p1.append({**base, **m1})
        ep_rows_p2.append({**base, **m2})
        ep_rows_p3.append({**base, **m3})

    print("  done.\n")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1  —  Steps 0–9
    # ════════════════════════════════════════════════════════════════════════
    _phase_header(1, PHASE1_STEPS, "Latent divergence  (pre-visual-failure)")

    _section("METRIC DEFINITIONS")
    _formula("P1a", [
        "Gini coefficient of mean-over-heads base-camera patch attention (sorted ascending):",
        r"  G = \frac{2 \sum_{i=1}^{n} i \cdot \bar{\alpha}_i}{n \sum_{i=1}^{n} \bar{\alpha}_i} - \frac{n+1}{n}",
        r"  \bar{\alpha}_i = \frac{1}{H}\sum_h A^{(16)}_{h,p_i}",
        "  High G → sparse fixation. FINDING: failures have higher G (p≈0.024 on _20260421_).",
    ])
    _formula("P1b", [
        "Spatial entropy of base-camera attention:",
        r"  H_{base} = -\sum_{p} \tilde{\alpha}_p \log \tilde{\alpha}_p, \quad"
        r"  \tilde{\alpha}_p = \bar{\alpha}_p \,/\, \textstyle\sum_{p'}\bar{\alpha}_{p'}",
        "  Complement of P1a. FINDING: failures have lower H (p≈0.094).",
    ])
    _formula("P1c", [
        "Action jerk — mean absolute first-difference of predicted horizon:",
        r"  J = \frac{1}{(T-1)D}\sum_{t=1}^{T-1}\sum_d |a_{t,d} - a_{t-1,d}|, \quad T=10,\; D=7",
        "  FINDING: marginal (p≈0.28); significant for task 5 individually.",
    ])
    _formula("P1d", [
        "Head-0 base-camera attention mass (layer 16):",
        r"  \phi_0 = \sum_{p\in\mathcal{P}_{base}} A^{(16)}_{0,p}",
        "  Head 0 is the primary location-grounding head.",
        "  FINDING: failures have ~50% higher φ₀ (p≈0.083).",
    ])
    _formula("P1e [NEW]", [
        "Task-token inter-head cosine similarity (layer 16):",
        r"  \bar{c}_{task} = \frac{2}{H(H-1)} \sum_{h<h'} \cos\!\left(\vec{a}^{(16)}_h|_{task},\; \vec{a}^{(16)}_{h'}|_{task}\right)",
        r"  \vec{a}^{(16)}_h|_{task} \in \mathbb{R}^{N_{task}} : \text{head } h\text{'s attention over task tokens}",
        "  High similarity = all heads agree which task words matter.",
        "  FINDING: SUCCESS has higher agreement (0.986 vs 0.975, p=0.009 ★★).",
        "  Interpretation: model with unified task understanding → clean motor plan.",
    ])

    _section("PER-EPISODE VALUES  (mean over steps 0–9)")
    cols_p1 = [
        ("gini_base",           "gini",     "7.4f"),
        ("spatial_ent",         "H_base",   "7.4f"),
        ("action_jerk",         "jerk",     "8.5f"),
        ("task_head_agreement", "tsk_hagr", "7.4f"),  # NEW
    ]
    _print_episode_table(ep_rows_p1, cols_p1)
    _section("SUMMARY STATISTICS")
    _print_summary(ep_rows_p1, cols_p1)

    _section("STEP-BY-STEP  —  gini_base  (every 1 step, steps 0–9)")
    print("  Columns: t<task>e<ep><S|F>  then │  S_mean  F_mean")
    _print_temporal_table(ep_data, "gini_base", PHASE1_STEPS, fmt="7.4f", every_n=1)

    _section("STEP-BY-STEP  —  task_head_agreement  (every 1 step, steps 0–9)  [NEW]")
    print("  Higher = heads read task in unison; lower = confused, divergent heads.")
    _print_temporal_table(ep_data, "task_head_agreement", PHASE1_STEPS, fmt="7.4f", every_n=1)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2  —  Steps 10–24
    # ════════════════════════════════════════════════════════════════════════
    _phase_header(2, PHASE2_STEPS, "Emerging divergence  (reliable early-warning window)")

    _section("METRIC DEFINITIONS")
    _formula("P2a", [
        "All-image attention fraction (base + both wrists), mean over heads:",
        r"  f_{img} = \frac{1}{HS}\sum_h \sum_{p\in\mathcal{P}_{all\,img}} A^{(16)}_{h,p}",
        "  FINDING: crosses direction at step ~8–12; success > failure thereafter.",
    ])
    _formula("P2b", [
        "Task-token attention fraction:",
        r"  f_{task} = \frac{1}{HS}\sum_h \sum_{j\in\mathcal{P}_{task}} A^{(16)}_{h,j}",
        "  FINDING: rises in failures after step 10 → model re-reads instruction.",
    ])
    _formula("P2c", [
        "State-token attention fraction (same formula over state token span).",
        "  FINDING: success > failure (model tracks proprioception).",
    ])
    _formula("P2d", [
        "State velocity — L2 norm of consecutive proprioceptive state change:",
        r"  v_t = \|s_t - s_{t-1}\|_2, \quad s_t \in \mathbb{R}^8",
        "  FINDING: significant at cumulative step 20 (p=0.049 ★).",
    ])
    _formula("P2e [NEW]", [
        "Gripper open fraction — fraction of predicted horizon steps with open gripper:",
        r"  f_{grip} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[a_{t,6} > 0], \quad T=10",
        "  FINDING: SUCCESS=0.83, FAILURE=0.59  (p=0.0001 ★★★).",
        "  Interpretation: successful episodes consistently plan to open the gripper",
        "  while failures predict hesitant, half-open-half-closed motion.",
    ])
    _formula("P2e+ [NEW]", [
        "Gripper open sign-flip rate — fraction of consecutive horizon pairs where gripper changes sign:",
        r"  \phi_{grip} = \frac{1}{T-1}\sum_{t=1}^{T-1}\mathbf{1}[\text{sgn}(a_{t,6}) \ne \text{sgn}(a_{t-1,6})]",
        "  Companion to P2e: two episodes with the same f_grip can differ sharply here.",
        "    • 11110000 → f_grip=0.5, φ_grip=0.111  (one clean transition, committed)",
        "    • 10101010 → f_grip=0.5, φ_grip=0.778  (oscillating, indecisive)",
        "  High φ_grip = model cannot commit to open/close → grip uncertainty in phase 2.",
    ])
    _formula("P2f [NEW]", [
        "Action sign-flip rate — mean fraction of dims whose sign flips each horizon step:",
        r"  \phi_{flip} = \frac{1}{(T-1)D}\sum_{t=1}^{T-1}\sum_d \mathbf{1}[\text{sgn}(a_{t,d}) \ne \text{sgn}(a_{t-1,d})]",
        "  FINDING: FAILURE=0.036 vs SUCCESS=0.028  (p=0.006 ★★).",
        "  High flip rate = oscillating, indecisive trajectory prediction.",
    ])
    _formula("P2g [NEW]", [
        "Head attention entropy variance — variance of per-head sequence-level entropy:",
        r"  \sigma^2_{H_{ent}} = \text{Var}_h\!\left[-\sum_s A^{(16)}_{h,s}\log A^{(16)}_{h,s}\right]",
        "  High variance = heads have diverse strategies; low = synchronized/collapsed.",
        "  FINDING: SUCCESS=1.46, FAILURE=1.28  (p=0.028 ★).",
    ])
    _formula("P2h [NEW]", [
        "Value-projection alignment on base patches — mean |cos(v_p, o_h)| over patches and heads:",
        r"  \bar{\rho} = \frac{1}{|\mathcal{P}_{base}| \cdot H}\sum_{p,h}\left|\cos\!\left(v^{(16)}_p,\; o^{(16)}_h\right)\right|",
        r"  o^{(16)}_h = A^{(16)}_h \cdot V^{(16)} \;\text{(head output)},\quad v^{(16)}_p\text{: value at patch }p",
        "  High = image patches actively contribute to head outputs.",
        "  FINDING: SUCCESS=0.654, FAILURE=0.636  (p=0.010 ★).",
    ])
    _formula("P2i [NEW]", [
        "GradCAM energy on base camera — mean absolute gradient w.r.t. image patches:",
        r"  E_{gc} = \frac{1}{|\mathcal{P}_{base}|}\sum_p \left|\frac{\partial L}{\partial x_p}\right|",
        "  High energy = output highly sensitive to image input → model in unstable state.",
        "  FINDING: FAILURE=0.019 > SUCCESS=0.013  (p=0.028 ★) in steps 10-24.",
    ])
    _formula("P2j [NEW]", [
        "Joint velocity entropy — entropy of per-joint velocity magnitudes:",
        r"  H_{jv} = -\sum_{k=1}^{8} \frac{|\dot{q}_k|}{\sum_{k'}|\dot{q}_{k'}|}"
        r"           \log\frac{|\dot{q}_k|}{\sum_{k'}|\dot{q}_{k'}|}",
        r"  |\dot{q}_k| = |s_{t,k} - s_{t-1,k}|",
        "  High = balanced coordinated motion; low = one joint dominates.",
        "  FINDING: SUCCESS=1.49, FAILURE=1.33  (p=0.0001 ★★★). Also p=0.069 in steps 0-9.",
    ])
    _formula("P2k [NEW]", [
        "Inter-step plan consistency — cosine similarity of consecutive step-0 action predictions:",
        r"  c_t = \cos\!\left(a^{(0)}_{t},\; a^{(0)}_{t-1}\right)",
        r"  a^{(0)}_t \in \mathbb{R}^7\text{: immediate next-step action predicted at step }t",
        "  Low = model changes its mind about what to do next.",
        "  FINDING: SUCCESS=0.850, FAILURE=0.797  (p=0.042 ★) in steps 10-24.",
    ])
    _formula("P2l [NEW]", [
        "Attention peak spatial drift — L2 distance of the peak-attention patch between steps:",
        r"  d_t = \left\|(r,c)^* _t - (r,c)^*_{t-1}\right\|_2, \quad (r,c)^*_t = \arg\max_{p}\bar{\alpha}^t_p",
        "  High = attention fixation jumps around = unstable focus.",
        "  FINDING: trending in failures at mid-phase (p≈0.11).",
    ])

    _section("PER-EPISODE VALUES  (mean over steps 10–24)")
    cols_p2a = [
        ("img_attn_frac",     "img_frac",  "8.5f"),
        ("task_attn_frac",    "task_frac", "8.5f"),
        ("state_vel",         "state_vel", "8.5f"),
        ("gripper_open_frac", "grip_open", "8.4f"),  # NEW P2e
        ("gripper_flip_rate", "grp_flip",  "8.4f"),  # NEW P2e+
        ("joint_vel_entropy", "jv_ent",    "8.4f"),  # NEW
    ]
    _print_episode_table(ep_rows_p2, cols_p2a)
    _section("SUMMARY STATISTICS  (img_frac / task_frac / state_vel / grip_open / grp_flip / jv_ent)")
    _print_summary(ep_rows_p2, cols_p2a)

    _section("PER-EPISODE VALUES  (mean over steps 10–24) — new attention features")
    cols_p2b = [
        ("sign_flip_rate",      "flip_rate", "8.4f"),  # NEW
        ("head_ent_var",        "hd_ent_v",  "8.4f"),  # NEW
        ("v_proj_align",        "v_proj",    "8.4f"),  # NEW
        ("gradcam_base_energy", "gc_enrgy",  "8.5f"),  # NEW
        ("plan_consistency",    "plan_con",  "8.4f"),  # NEW
    ]
    _print_episode_table(ep_rows_p2, cols_p2b)
    _section("SUMMARY STATISTICS  (flip / head_ent_var / v_proj / gradcam / plan_cons)")
    _print_summary(ep_rows_p2, cols_p2b)

    _section("STEP-BY-STEP  —  gripper_open_frac  (every 2 steps, steps 10–24)  [NEW ★★★]")
    print("  Fraction of 10-step horizon steps where gripper dim > 0 (opening planned).")
    _print_temporal_table(ep_data, "gripper_open_frac", PHASE2_STEPS, fmt="7.4f", every_n=2)

    _section("STEP-BY-STEP  —  gripper_flip_rate  (every 2 steps, steps 10–24)  [NEW P2e+]")
    print("  Fraction of consecutive horizon pairs where gripper sign flips (open↔close).")
    print("  High = oscillating grip command; low = committed trajectory (even if f_grip=0.5).")
    _print_temporal_table(ep_data, "gripper_flip_rate", PHASE2_STEPS, fmt="7.4f", every_n=2)

    _section("STEP-BY-STEP  —  joint_vel_entropy  (every 2 steps, steps 10–24)  [NEW ★★★]")
    print("  High entropy = balanced joint motion. Low = one joint thrashing.")
    _print_temporal_table(ep_data, "joint_vel_entropy", PHASE2_STEPS, fmt="7.4f", every_n=2)

    _section("STEP-BY-STEP  —  img_attn_frac  (every 2 steps, steps 10–24)")
    print("  Watch the crossing around step 8-12: failure drops below success.")
    _print_temporal_table(ep_data, "img_attn_frac", PHASE2_STEPS, fmt="7.5f", every_n=2)

    _section("STEP-BY-STEP  —  sign_flip_rate  (every 2 steps, steps 10–24)  [NEW ★★]")
    _print_temporal_table(ep_data, "sign_flip_rate", PHASE2_STEPS, fmt="7.4f", every_n=2)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3  —  Steps 25–49
    # ════════════════════════════════════════════════════════════════════════
    _phase_header(3, PHASE3_STEPS, "Visible failure loop  (robot thrashing)")

    _section("METRIC DEFINITIONS")
    _formula("P3a", [
        "Action std across all dims and horizon steps:",
        r"  \sigma_a = \text{std}\!\left(\{a_{t,d}\}_{t\in[T],d\in[D]}\right)",
        "  FINDING: 2.5× higher in failures by step 45 (p=0.0008 ★★★).",
    ])
    _formula("P3b", [
        "State velocity (same as P2d).",
        "  FINDING: 12× higher in failures by step 45.",
        "  Failure episodes enter a thrashing loop with rapid joint oscillation.",
    ])
    _formula("P3c", [
        "Cumulative task-attention drift:",
        r"  \Delta f_{task}(t) = \bar{f}^{\text{fail}}_{task}(t) - \bar{f}^{\text{succ}}_{task}(t)",
        "  FINDING: grows from +0.006 at step 10 to +0.077 at step 45.",
    ])

    _section("PER-EPISODE VALUES  (mean over steps 25–49)")
    cols_p3 = [
        ("action_std",         "act_std",   "7.4f"),
        ("state_vel",          "state_vel", "7.4f"),
        ("task_attn_frac",     "task_frac", "7.4f"),
        ("gripper_open_frac",  "grip_open", "7.4f"),  # NEW
        ("joint_vel_entropy",  "jv_ent",    "7.4f"),  # NEW
    ]
    _print_episode_table(ep_rows_p3, cols_p3)
    _section("SUMMARY STATISTICS")
    _print_summary(ep_rows_p3, cols_p3)

    _section("STEP-BY-STEP  —  action_std  (every 5 steps, steps 25–49)")
    _print_temporal_table(ep_data, "action_std",  PHASE3_STEPS, fmt="7.4f", every_n=5)
    _section("STEP-BY-STEP  —  state_vel  (every 5 steps, steps 25–49)")
    _print_temporal_table(ep_data, "state_vel",   PHASE3_STEPS, fmt="7.4f", every_n=5)
    _section("STEP-BY-STEP  —  gripper_open_frac  (every 5 steps, steps 25–49)  [NEW]")
    _print_temporal_table(ep_data, "gripper_open_frac", PHASE3_STEPS, fmt="7.4f", every_n=5)

    # ════════════════════════════════════════════════════════════════════════
    # CROSS-PHASE SUMMARY
    # ════════════════════════════════════════════════════════════════════════
    _header("CROSS-PHASE SUMMARY: ALL DETECTABLE SIGNALS")
    print()
    rows = [
        # metric, ID, window, sig, direction, interpretation
        ("gini_base",          "P1a",    "0–9",    "p≈0.024 ★ ", "F>S", "Fixation: attention too sparse early"),
        ("spatial_ent",        "P1b",    "0–9",    "p≈0.094 ·  ", "S>F", "Complement of Gini"),
        ("task_head_agr",      "P1e NEW","0–9",    "p=0.009 ★★ ", "S>F", "Heads agree on task words in successes"),
        ("img_attn_frac",      "P2a",    "~step8", "crossing   ", "↕",   "Direction reversal: success > failure after crossing"),
        ("gripper_open_frac",  "P2e NEW","10–24",  "p=0.0001 ★★★","S>F","Planned gripper opening (83% vs 59%)"),
        ("gripper_flip_rate",  "P2e+NEW","10–24",  "TBD        ","F>S","Oscillating grip: disambiguates 10101010 vs 11110000"),
        ("joint_vel_entropy",  "P2j NEW","10–24",  "p=0.0001 ★★★","S>F","Coordinated joint motion vs one-joint thrash"),
        ("sign_flip_rate",     "P2f NEW","10–24",  "p=0.006 ★★ ", "F>S","Oscillating action plan"),
        ("task_attn_frac",     "P2b",    "10–24",  "growing gap", "F>S","Re-reading instruction"),
        ("head_ent_var",       "P2g NEW","10–24",  "p=0.028 ★ ", "S>F","Diverse head strategies"),
        ("v_proj_align",       "P2h NEW","10–24",  "p=0.010 ★ ", "S>F","Image patches used in computation"),
        ("gradcam_base_energy","P2i NEW","10–24",  "p=0.028 ★ ", "F>S","Sensitive/unstable output"),
        ("plan_consistency",   "P2k NEW","10–24",  "p=0.042 ★ ", "S>F","Stable next-action prediction"),
        ("state_vel",          "P2d",    "~step20","p=0.049 ★ ", "F>S","Faster joint motion"),
        ("action_std",         "P3a",    "30–50",  "p=0.0008★★★","F>S","2.5× in failures (late)"),
        ("state_vel (late)",   "P3b",    "25–50",  "p=0.013 ★ ", "F>S","12× in failures by step 45"),
    ]
    print(f"  {'Metric':22} {'ID':12} {'Window':8} {'Sig':12} {'Dir':4}  {'Interpretation'}")
    _rule()
    for metric, mid, win, sig, direction, interp in rows:
        print(f"  {metric:22} {mid:12} {win:8} {sig:12} {direction:4}  {interp}")
    _rule()
    print()
    print("  RECOMMENDED FEATURE SET FOR FAILURE PREDICTOR:")
    print()
    print("  ── By step 10 (early, limited but real signal):")
    print("    • task_head_agreement [P1e]  — strongest single early feature (p=0.009)")
    print("    • gini_base           [P1a]  — spatial fixation")
    print("    • joint_vel_entropy   [P2j]  — already trending (p=0.069)")
    print()
    print("  ── By step 15 (first reliable window):")
    print("    • gripper_open_frac   [P2e]  — ★★★ most powerful new feature")
    print("    • gripper_flip_rate   [P2e+] — companion: catches oscillating grip (10101010 vs 11110000)")
    print("    • joint_vel_entropy   [P2j]  — ★★★ equally powerful")
    print("    • sign_flip_rate      [P2f]  — ★★  oscillating plan")
    print("    • img_attn_frac slope [P2a]  — visual grounding drift")
    print("    • plan_consistency    [P2k]  — consistent next-action prediction")
    print()
    print("  ── By step 25 (high-confidence, still before obvious failure):")
    print("    + head_ent_var, v_proj_align, gradcam_base_energy, state_vel")
    print()
    print("  EXPECTED AUC (logistic regression, leave-one-task-out):")
    print("    ≤10 steps:  ~0.70–0.75  (+task_head_agr vs old ~0.65)")
    print("    ~15 steps:  ~0.82–0.88  (+grip_open + jv_ent vs old ~0.75)")
    print("    ~25 steps:  ~0.90+")
    print()
    _double_rule()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()
