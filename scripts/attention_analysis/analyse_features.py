#!/usr/bin/env python3
"""
analyse_features.py — Full AUROC analysis: phase + windowed (w5, w15) across all groups.
Also produces a phase-stratified analysis where each feature × phase is treated as an
independent feature, surfacing crossing patterns that cancel at the episode level.

Outputs (OUT_DIR = /nfs/roberts/project/pi_tkf6/zs377/feature_analysis):
    auroc_full.csv                  complete numeric table, 1 row per feature (401 rows)
    auroc_phase_split.csv           phase-stratified: 401×3=1203 rows, one per feature×phase
    table_summary.txt               top-50: cross-group summary + per-group best EA
    table_phases.txt                top-50: early/mid/late AUROC per group
    table_windows.txt               top-50: w5 and w15 windowed peak AUROC per group
    table_detail.txt                top-50: full per-group detail
    table_phase_split.txt           top-50 phase-split features (crossing patterns visible)
    feature_clusters.csv            correlation cluster assignments (original features)
    feature_clusters_phase_split.csv  correlation clusters for phase-split features
    candidates.txt                  non-redundant BCE candidates (original features)
    candidates_phase_split.txt      non-redundant BCE candidates (phase-split)
    analysis_results.pkl            full FeaturePhaseResult objects + all stats

Usage:
    python scripts/attention_analysis/analyse_features.py [--top N]
    --top N   features to cluster per analysis (default 120)
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.attention_analysis.features import (
    PHASES_3,
    analyse_all_features,
    effective_auroc,
    feature_names,
    load_group,
)

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT   = Path("/nfs/roberts/project/pi_tkf6/zs377")
ROLLOUTS  = Path("/home/zs377/project_pi_tkf6/zs377/policy_rollouts")
CACHE_DIR = PROJECT / "feature_cache"
OUT_DIR   = PROJECT / "feature_analysis"

GROUPS = [
    ("pi05_libero10",      ROLLOUTS, ["policy_records_pi05_libero_20260512_182606",
                                      "policy_records_pi05_libero_20260512_201032"]),
    ("pi0fast_libero10",   ROLLOUTS, ["policy_records_pi0fast_libero_20260511_040629",
                                      "policy_records_pi0fast_libero_20260513_021106"]),
    ("pi0fast_liberoplus", ROLLOUTS, ["policy_records_pi0fast_liberoplus_20260516_034939",
                                      "policy_records_pi0fast_liberoplus_20260516_034940"]),
    ("pi05_liberoplus",    ROLLOUTS, ["policy_records_pi05_liberoplus_20260522_032743",
                                      "policy_records_pi05_liberoplus_20260522_031717"]),
]
GROUP_NAMES  = [g[0] for g in GROUPS]
GROUP_SHORT  = ["pi05_l10", "pif_l10", "pif_lp", "pi05_lp"]
MAX_STEPS    = 60
WINDOW_SIZES = (5, 15)
PHASES       = list(PHASES_3.keys())   # ["early", "mid", "late"]


# ── per-result extractors ─────────────────────────────────────────────────────

def _eff(a: float) -> float:
    e = effective_auroc(a)
    return e if not np.isnan(e) else 0.5

def phase_eas(res) -> dict[str, float]:
    """Effective AUROC for each phase (mean aggregate)."""
    return {ph: _eff(res.phase_mean_auroc.get(ph, float("nan"))) for ph in PHASES}

def best_ea(res) -> float:
    """Peak effective AUROC over phases + all windowed descriptors."""
    best = 0.5
    for a in list(res.phase_mean_auroc.values()) + list(res.phase_max_auroc.values()):
        best = max(best, _eff(a))
    for pd in res.desc_phase_peak.values():
        for a in pd.values():
            best = max(best, _eff(a))
    return best

def window_peak_ea(res, w: int) -> float:
    """Peak effective AUROC across all descriptors and phases for window width w."""
    best = 0.5
    for key, pd in res.desc_phase_peak.items():
        if not key.endswith(f"_w{w}"):
            continue
        for a in pd.values():
            best = max(best, _eff(a))
    return best

def best_phase_str(res) -> str:
    best_e, best_p = 0.5, "?"
    for ph, a in res.phase_mean_auroc.items():
        e = _eff(a)
        if e > best_e:
            best_e, best_p = e, ph
    return best_p

def direction_str(res) -> str:
    vals = [a for a in res.phase_mean_auroc.values() if not np.isnan(a)]
    return "F>S" if (vals and float(np.nanmean(vals)) > 0.5) else "F<S"

def phase_direction_str(res, ph: str) -> str:
    """Direction (F>S / F<S) for a specific phase."""
    a = res.phase_mean_auroc.get(ph, float("nan"))
    if np.isnan(a):
        return "n/a"
    return "F>S" if a > 0.5 else "F<S"


# ── episode-mean matrices (for clustering) ────────────────────────────────────

def ep_mean_matrix(ep_data: dict, feats: list) -> np.ndarray:
    """(n_episodes, n_feats) matrix of episode-mean feature values."""
    all_eps = ep_data.get("success", []) + ep_data.get("failure", [])
    rows = []
    for ep in all_eps:
        if not ep:
            continue
        rows.append([np.nanmean([s.get(f, np.nan) for s in ep]) for f in feats])
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, len(feats)))

def ep_phase_mean_matrix(ep_data: dict, feats: list, phase_range: tuple) -> np.ndarray:
    """(n_episodes, n_feats) matrix: mean of each feature over steps in phase_range (inclusive)."""
    lo, hi = phase_range
    all_eps = ep_data.get("success", []) + ep_data.get("failure", [])
    rows = []
    for ep in all_eps:
        if not ep:
            continue
        phase_steps = [s for s in ep if lo <= s.get("rel_step", -1) <= hi]
        if phase_steps:
            rows.append([np.nanmean([s.get(f, np.nan) for s in phase_steps]) for f in feats])
        else:
            rows.append([np.nan] * len(feats))
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, len(feats)))


def correlation_clusters(mat: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    n = mat.shape[1]
    labels = np.full(n, -1, dtype=int)
    valid = np.where(~np.all(np.isnan(mat), axis=0))[0]
    if len(valid) < 2:
        for i, v in enumerate(valid):
            labels[v] = i
        return labels
    sub = mat[:, valid].copy()
    col_means = np.nanmean(sub, axis=0)
    ni = np.where(np.isnan(sub))
    sub[ni] = np.take(col_means, ni[1])
    n_v = len(valid)
    dist = np.zeros(n_v * (n_v - 1) // 2)
    idx = 0
    for i in range(n_v):
        for j in range(i + 1, n_v):
            r, _ = pearsonr(sub[:, i], sub[:, j])
            dist[idx] = 1.0 - abs(r) if not np.isnan(r) else 1.0
            idx += 1
    Z = linkage(dist, method="average")
    cl = fcluster(Z, t=1.0 - threshold, criterion="distance")
    for k, v in enumerate(valid):
        labels[v] = int(cl[k])
    return labels


# ── table formatting ──────────────────────────────────────────────────────────

def _fmt(v: float, nan_str: str = " n/a") -> str:
    if np.isnan(v) or v == 0.5:
        return nan_str
    return f"{v:.3f}"

def _bar(v: float, lo: float = 0.5, hi: float = 1.0, width: int = 8) -> str:
    """Tiny ASCII bar for quick visual comparison."""
    if np.isnan(v) or v <= lo:
        return " " * width
    frac = min((v - lo) / (hi - lo), 1.0)
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)

def write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")
    print(f"  {path.name}")

def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=120)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. load ───────────────────────────────────────────────────────────────
    print("Loading cached data …")
    all_data: dict = {}
    all_feats: list | None = None
    for gname, base_dir, run_names in GROUPS:
        ep = load_group(gname, run_names, base_dir, CACHE_DIR,
                        max_steps=MAX_STEPS, use_cache=True)
        all_data[gname] = ep
        fnames = feature_names(ep)
        if all_feats is None:
            all_feats = fnames
        ns, nf = len(ep["success"]), len(ep["failure"])
        print(f"  {gname:25s}: {ns:4d} S  {nf:4d} F")
    assert all_feats is not None

    # ── 2. phase + windowed analysis ─────────────────────────────────────────
    print(f"\nPhase + windowed analysis  (w={WINDOW_SIZES})  — "
          f"{len(all_feats)} features × {len(GROUP_NAMES)} groups …")
    all_results: dict = {}
    for gname, ep_data in all_data.items():
        t0 = time.time()
        print(f"  {gname} …", end=" ", flush=True)
        all_results[gname] = analyse_all_features(
            ep_data, all_feats,
            phase_defs=PHASES_3,
            window_sizes=WINDOW_SIZES,
            max_steps=MAX_STEPS,
        )
        print(f"done in {time.time()-t0:.0f}s")

    # ── 3. build per-feature stats (original, episode-level) ─────────────────
    print("\nBuilding summary rows …")
    stats: list[dict] = []
    for feat in all_feats:
        r: dict = {"feature": feat}
        ea_list = []
        for gname in GROUP_NAMES:
            res = all_results[gname][feat]
            ea  = best_ea(res)
            pea = phase_eas(res)
            w5  = window_peak_ea(res, 5)
            w15 = window_peak_ea(res, 15)
            r[f"ea_{gname}"]    = round(ea,  4)
            r[f"ph_{gname}"]    = best_phase_str(res)
            r[f"dir_{gname}"]   = direction_str(res)
            r[f"dchg_{gname}"]  = int(res.direction_change)
            for ph in PHASES:
                r[f"{ph}_{gname}"] = round(pea[ph], 4)
            r[f"w5_{gname}"]    = round(w5,  4)
            r[f"w15_{gname}"]   = round(w15, 4)
            ea_list.append(ea)
        valid = [e for e in ea_list if e > 0.5]
        r["cross_ea"]       = round(float(np.mean(valid)), 4) if valid else 0.5
        r["n_disc_60"]      = sum(1 for e in ea_list if e >= 0.60)
        r["n_disc_65"]      = sum(1 for e in ea_list if e >= 0.65)
        r["n_disc_70"]      = sum(1 for e in ea_list if e >= 0.70)
        r["any_dir_change"] = int(any(all_results[g][feat].direction_change
                                      for g in GROUP_NAMES))
        stats.append(r)
    stats.sort(key=lambda x: x["cross_ea"], reverse=True)

    # ── 3b. build phase-split stats (each feature × phase = one row) ──────────
    print("Building phase-split rows …")
    phase_stats: list[dict] = []
    for feat in all_feats:
        for ph in PHASES:
            pname = f"{feat}_{ph}"
            r: dict = {"feature": pname, "source_feature": feat, "phase": ph}
            ea_list = []
            for gname in GROUP_NAMES:
                res = all_results[gname][feat]
                pea = phase_eas(res)
                ea  = pea[ph]
                r[f"ea_{gname}"]  = round(ea, 4)
                r[f"dir_{gname}"] = phase_direction_str(res, ph)
                ea_list.append(ea)
            valid = [e for e in ea_list if e > 0.5]
            r["cross_ea"]  = round(float(np.mean(valid)), 4) if valid else 0.5
            r["n_disc_60"] = sum(1 for e in ea_list if e >= 0.60)
            r["n_disc_65"] = sum(1 for e in ea_list if e >= 0.65)
            r["n_disc_70"] = sum(1 for e in ea_list if e >= 0.70)
            phase_stats.append(r)
    phase_stats.sort(key=lambda x: x["cross_ea"], reverse=True)

    # ── 4. save CSVs ──────────────────────────────────────────────────────────
    csv_fields = (
        ["feature", "cross_ea", "n_disc_60", "n_disc_65", "n_disc_70", "any_dir_change"] +
        [f"{k}_{g}" for g in GROUP_NAMES
         for k in ["ea", "ph", "dir", "dchg"] + PHASES + ["w5", "w15"]]
    )
    write_csv(OUT_DIR / "auroc_full.csv", csv_fields, stats)
    print(f"  auroc_full.csv  ({len(stats)} features)")

    ps_fields = (
        ["feature", "source_feature", "phase", "cross_ea",
         "n_disc_60", "n_disc_65", "n_disc_70"] +
        [f"{k}_{g}" for g in GROUP_NAMES for k in ["ea", "dir"]]
    )
    write_csv(OUT_DIR / "auroc_phase_split.csv", ps_fields, phase_stats)
    print(f"  auroc_phase_split.csv  ({len(phase_stats)} features)")

    # ── 5. formatted tables ───────────────────────────────────────────────────
    TOP   = 50
    FW    = 42   # feature name width (original)
    PS_FW = 50   # feature name width (phase-split, includes _early/_mid/_late suffix)

    top_stats = stats[:TOP]

    # ── Table 1: cross-group summary ──────────────────────────────────────────
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║  TABLE 1 — CROSS-GROUP SUMMARY  (top 50 by cross-group effective AUROC)     ║",
        "║  ea = effective AUROC  |  disc = # groups ≥ 0.60/0.65/0.70                 ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
        f"{'Feature':<{FW}} {'cross':>6}  {'disc':>12}  "
        + "  ".join(f"{s:>8}" for s in GROUP_SHORT)
        + "  dir-chg  best-phases",
        f"{'':─<{FW}} {'──────':>6}  {'──────────':>12}  "
        + "  ".join("────────" for _ in GROUP_SHORT)
        + "  ───────  ──────────────────────",
    ]
    for r in top_stats:
        per_g = "  ".join(f"{r[f'ea_{g}']:8.3f}" for g in GROUP_NAMES)
        phases_str = "/".join(r.get(f"ph_{g}", "?") for g in GROUP_NAMES)
        disc = f"{r['n_disc_60']}/{r['n_disc_65']}/{r['n_disc_70']}"
        dc   = "yes" if r["any_dir_change"] else " no"
        lines.append(
            f"{r['feature']:<{FW}} {r['cross_ea']:6.3f}  {disc:>12}  {per_g}  {dc:>7}  {phases_str}"
        )
    lines += ["", "  disc columns: ≥0.60 / ≥0.65 / ≥0.70  |  groups: " + " | ".join(
        f"{s}={g}" for s, g in zip(GROUP_SHORT, GROUP_NAMES))]
    write_table(OUT_DIR / "table_summary.txt", lines)

    # ── Table 2: phase profile ────────────────────────────────────────────────
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║  TABLE 2 — PHASE PROFILE  (effective AUROC per phase, top 50 features)      ║",
        "║  Phases: early=steps 0-10  mid=11-30  late=31-59                           ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
    ]
    grp_hdr = "  ".join(f"{'── '+s+' ──':^23}" for s in GROUP_SHORT)
    ph_hdr  = "  ".join("  ".join(f"{ph:>7}" for ph in PHASES) for _ in GROUP_SHORT)
    lines += [
        f"{'Feature':<{FW}}  {grp_hdr}",
        f"{'':─<{FW}}  {ph_hdr}",
    ]
    for r in top_stats:
        row_vals = []
        for g in GROUP_NAMES:
            row_vals.append("  ".join(f"{r[f'{ph}_{g}']:7.3f}" for ph in PHASES))
        lines.append(f"{r['feature']:<{FW}}  {'  '.join(row_vals)}")
    lines += ["", "  Values ≤ 0.500 mean no discriminative signal (NaN → 0.500)."]
    lines += ["  Groups: " + " | ".join(f"{s}={g}" for s, g in zip(GROUP_SHORT, GROUP_NAMES))]
    write_table(OUT_DIR / "table_phases.txt", lines)

    # ── Table 3: windowed peaks ───────────────────────────────────────────────
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║  TABLE 3 — WINDOWED DESCRIPTOR PEAKS  (top 50 features)                    ║",
        "║  w5 = 5-step causal window   w15 = 15-step causal window                   ║",
        "║  Peak = max eff-AUROC over all descriptors × phases for that window width   ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
    ]
    grp_hdr2 = "  ".join(f"{'── '+s+' ──':^17}" for s in GROUP_SHORT)
    w_hdr    = "  ".join("    w5     w15" for _ in GROUP_SHORT)
    lines += [
        f"{'Feature':<{FW}}  {grp_hdr2}",
        f"{'':─<{FW}}  {w_hdr}",
    ]
    for r in top_stats:
        row_vals = []
        for g in GROUP_NAMES:
            w5  = r[f"w5_{g}"]
            w15 = r[f"w15_{g}"]
            row_vals.append(f"{w5:7.3f}  {w15:7.3f}")
        lines.append(f"{r['feature']:<{FW}}  {'  '.join(row_vals)}")
    lines += [
        "",
        "  Windowed peaks capture transient features that spike in a short window.",
        "  w5 ≫ w15 suggests a sharp onset;  w15 ≫ w5 suggests a slow drift.",
        "  Groups: " + " | ".join(f"{s}={g}" for s, g in zip(GROUP_SHORT, GROUP_NAMES)),
    ]
    write_table(OUT_DIR / "table_windows.txt", lines)

    # ── Table 4: full per-group detail ────────────────────────────────────────
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║  TABLE 4 — FULL PER-GROUP DETAIL  (top 50 features)                        ║",
        "║  Columns per group: best-EA  │ early  mid  late  │  w5   w15  │ dir dchg   ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
    ]
    col_per_g = 8 + 3 + 3*6 + 3 + 2*6 + 3 + 4 + 5
    for r in top_stats:
        lines.append(f"{'─'*FW}  {'─'*6}  {'═'*(col_per_g * len(GROUP_NAMES))}")
        lines.append(f"{r['feature']:<{FW}}  cross={r['cross_ea']:.3f}")
        for g, s in zip(GROUP_NAMES, GROUP_SHORT):
            ea  = r[f"ea_{g}"]
            bar = _bar(ea)
            phs = "  ".join(f"{r[f'{ph}_{g}']:.3f}" for ph in PHASES)
            w5  = r[f"w5_{g}"]
            w15 = r[f"w15_{g}"]
            dc  = "↻" if r[f"dchg_{g}"] else " "
            dr  = r[f"dir_{g}"]
            lines.append(
                f"  {s:>8}: {ea:.3f} {bar}  phases [{phs}]  "
                f"w5={w5:.3f} w15={w15:.3f}  {dr} {dc}"
            )
    write_table(OUT_DIR / "table_detail.txt", lines)

    # ── Table 5: phase-split top-50 ───────────────────────────────────────────
    top_ps = phase_stats[:TOP]
    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        "║  TABLE 5 — PHASE-SPLIT FEATURES  (top 50 by cross-group effective AUROC)    ║",
        "║  Each feature×phase is treated independently — crossing features visible.   ║",
        "║  Suffix: _early=steps 0-10  _mid=steps 11-30  _late=steps 31-59            ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
        "",
        f"{'Feature':<{PS_FW}} {'cross':>6}  {'disc':>12}  "
        + "  ".join(f"{s:>8}" for s in GROUP_SHORT)
        + "  directions",
        f"{'':─<{PS_FW}} {'──────':>6}  {'──────────':>12}  "
        + "  ".join("────────" for _ in GROUP_SHORT)
        + "  ─────────────────────────────",
    ]
    for r in top_ps:
        per_g = "  ".join(f"{r[f'ea_{g}']:8.3f}" for g in GROUP_NAMES)
        dirs  = "/".join(r.get(f"dir_{g}", "n/a") for g in GROUP_NAMES)
        disc  = f"{r['n_disc_60']}/{r['n_disc_65']}/{r['n_disc_70']}"
        lines.append(
            f"{r['feature']:<{PS_FW}} {r['cross_ea']:6.3f}  {disc:>12}  {per_g}  {dirs}"
        )
    lines += [
        "",
        "  Compare to TABLE 1: features absent from TABLE 1 but present here are",
        "  crossing features whose early/late signals cancel at episode level.",
        "  disc: ≥0.60 / ≥0.65 / ≥0.70  |  groups: "
        + " | ".join(f"{s}={g}" for s, g in zip(GROUP_SHORT, GROUP_NAMES)),
    ]
    write_table(OUT_DIR / "table_phase_split.txt", lines)

    # ── 6. clustering (original features) ────────────────────────────────────
    top_n     = min(args.top, len(stats))
    top_feats = [r["feature"] for r in stats[:top_n]]
    print(f"\nClustering top-{top_n} original features …")

    mats = []
    for gname, ep_data in all_data.items():
        m = ep_mean_matrix(ep_data, top_feats)
        if m.shape[0] == 0:
            continue
        std = np.nanstd(m, axis=0)
        std[std < 1e-9] = 1.0
        mats.append((m - np.nanmean(m, axis=0)) / std)
    combined = np.vstack(mats)
    labels = correlation_clusters(combined, threshold=0.85)

    feat_ea = {r["feature"]: r["cross_ea"] for r in stats}
    cluster_best: dict[int, tuple] = {}
    cluster_size: dict[int, int]   = {}
    for i, feat in enumerate(top_feats):
        cl = int(labels[i])
        cluster_size[cl] = cluster_size.get(cl, 0) + 1
        ea = feat_ea[feat]
        if cl not in cluster_best or ea > cluster_best[cl][0]:
            cluster_best[cl] = (ea, feat)

    rep_set = {v[1] for cl, v in cluster_best.items() if cl != -1}
    cluster_rows = []
    for i, feat in enumerate(top_feats):
        cl = int(labels[i])
        cluster_rows.append({
            "feature": feat, "cluster": cl,
            "cluster_size": cluster_size.get(cl, 1),
            "is_representative": int(feat in rep_set),
            "cross_ea": feat_ea[feat],
            "n_disc_60": next(r["n_disc_60"] for r in stats if r["feature"] == feat),
        })
    cluster_rows.sort(key=lambda r: (-r["cluster"], -r["cross_ea"]))
    write_csv(OUT_DIR / "feature_clusters.csv",
              ["feature","cluster","cluster_size","is_representative","cross_ea","n_disc_60"],
              cluster_rows)
    n_clusters = len({r["cluster"] for r in cluster_rows if r["cluster"] != -1})
    print(f"  {n_clusters} clusters  →  feature_clusters.csv")

    # ── 7. candidates (original) ──────────────────────────────────────────────
    candidates = sorted(
        [(ea, feat) for cl, (ea, feat) in cluster_best.items() if cl != -1],
        reverse=True,
    )
    cr_map = {r["feature"]: r for r in cluster_rows}
    fr_map = {r["feature"]: r for r in stats}

    cand_lines = [
        "# Non-redundant feature candidates for episode-level multi-instance BCE",
        f"# {len(candidates)} candidates — best representative per correlation cluster (|r|≥0.85)",
        "# Ranked by cross-group mean effective AUROC",
        "#",
        f"# {'Feature':<{FW}} cross  disc  cl_sz │ " +
        " ".join(f"{s:>8}" for s in GROUP_SHORT),
        "# " + "─" * (FW + 6 + 6 + 6 + 3 + 9 * len(GROUP_NAMES)),
    ]
    for ea, feat in candidates:
        cr = cr_map.get(feat, {})
        fr = fr_map.get(feat, {})
        per_g = " ".join(f"{fr.get(f'ea_{g}', 0.5):8.3f}" for g in GROUP_NAMES)
        cand_lines.append(
            f"  {feat:<{FW}} {ea:.3f}    {fr.get('n_disc_60',0)}     "
            f"{cr.get('cluster_size',1):2d}  │ {per_g}"
        )
    cand_lines += ["", "# Groups: " + " | ".join(f"{s}={g}"
                   for s, g in zip(GROUP_SHORT, GROUP_NAMES))]
    (OUT_DIR / "candidates.txt").write_text("\n".join(cand_lines))
    print(f"  {len(candidates)} candidates  →  candidates.txt")

    # ── 8. clustering (phase-split features) ─────────────────────────────────
    top_ps_n    = min(args.top * 3, len(phase_stats))
    top_ps_list = phase_stats[:top_ps_n]
    print(f"\nClustering top-{top_ps_n} phase-split features …")
    print("  Pre-computing phase matrices …", end=" ", flush=True)

    # (gname, ph) → (n_eps, n_all_feats) phase-mean matrix
    feat_idx_map = {f: i for i, f in enumerate(all_feats)}
    phase_ep_mats: dict = {}
    for gname, ep_data in all_data.items():
        for ph in PHASES:
            phase_ep_mats[(gname, ph)] = ep_phase_mean_matrix(
                ep_data, all_feats, PHASES_3[ph]
            )
    print("done")

    ps_mats = []
    for gname, ep_data in all_data.items():
        all_eps_g = ep_data.get("success", []) + ep_data.get("failure", [])
        n_eps_g   = sum(1 for ep in all_eps_g if ep)
        mat = np.full((n_eps_g, top_ps_n), np.nan)
        for col, r in enumerate(top_ps_list):
            src_idx = feat_idx_map[r["source_feature"]]
            ph_mat  = phase_ep_mats[(gname, r["phase"])]
            mat[:, col] = ph_mat[:, src_idx]
        std = np.nanstd(mat, axis=0)
        std[std < 1e-9] = 1.0
        ps_mats.append((mat - np.nanmean(mat, axis=0)) / std)
    ps_combined = np.vstack(ps_mats)
    ps_labels   = correlation_clusters(ps_combined, threshold=0.85)

    ps_feat_ea = {r["feature"]: r["cross_ea"] for r in phase_stats}
    ps_cluster_best: dict[int, tuple] = {}
    ps_cluster_size: dict[int, int]   = {}
    ps_feats_ordered = [r["feature"] for r in top_ps_list]
    for i, feat in enumerate(ps_feats_ordered):
        cl = int(ps_labels[i])
        ps_cluster_size[cl] = ps_cluster_size.get(cl, 0) + 1
        ea = ps_feat_ea[feat]
        if cl not in ps_cluster_best or ea > ps_cluster_best[cl][0]:
            ps_cluster_best[cl] = (ea, feat)

    ps_rep_set = {v[1] for cl, v in ps_cluster_best.items() if cl != -1}
    ps_src_map  = {r["feature"]: r["source_feature"] for r in phase_stats}
    ps_ph_map   = {r["feature"]: r["phase"] for r in phase_stats}
    ps_disc_map = {r["feature"]: r["n_disc_60"] for r in phase_stats}
    ps_cluster_rows = []
    for i, feat in enumerate(ps_feats_ordered):
        cl = int(ps_labels[i])
        ps_cluster_rows.append({
            "feature":          feat,
            "source_feature":   ps_src_map[feat],
            "phase":            ps_ph_map[feat],
            "cluster":          cl,
            "cluster_size":     ps_cluster_size.get(cl, 1),
            "is_representative": int(feat in ps_rep_set),
            "cross_ea":         ps_feat_ea[feat],
            "n_disc_60":        ps_disc_map[feat],
        })
    ps_cluster_rows.sort(key=lambda r: (-r["cluster"], -r["cross_ea"]))
    write_csv(OUT_DIR / "feature_clusters_phase_split.csv",
              ["feature","source_feature","phase","cluster","cluster_size",
               "is_representative","cross_ea","n_disc_60"],
              ps_cluster_rows)
    n_ps_clusters = len({r["cluster"] for r in ps_cluster_rows if r["cluster"] != -1})
    print(f"  {n_ps_clusters} clusters  →  feature_clusters_phase_split.csv")

    # ── 9. candidates (phase-split) ───────────────────────────────────────────
    ps_candidates = sorted(
        [(ea, feat) for cl, (ea, feat) in ps_cluster_best.items() if cl != -1],
        reverse=True,
    )
    ps_cr_map = {r["feature"]: r for r in ps_cluster_rows}
    ps_fr_map = {r["feature"]: r for r in phase_stats}

    ps_cand_lines = [
        "# Non-redundant phase-split candidates for episode-level multi-instance BCE",
        f"# {len(ps_candidates)} candidates — best per correlation cluster (|r|≥0.85)",
        "# Ranked by cross-group mean effective AUROC",
        "# Feature suffix encodes phase: _early=steps 0-10  _mid=11-30  _late=31-59",
        "#",
        f"# {'Feature':<{PS_FW}} cross  disc  cl_sz │ " +
        " ".join(f"{s:>8}" for s in GROUP_SHORT),
        "# " + "─" * (PS_FW + 6 + 6 + 6 + 3 + 9 * len(GROUP_NAMES)),
    ]
    for ea, feat in ps_candidates:
        cr = ps_cr_map.get(feat, {})
        fr = ps_fr_map.get(feat, {})
        per_g = " ".join(f"{fr.get(f'ea_{g}', 0.5):8.3f}" for g in GROUP_NAMES)
        ps_cand_lines.append(
            f"  {feat:<{PS_FW}} {ea:.3f}    {fr.get('n_disc_60',0)}     "
            f"{cr.get('cluster_size',1):2d}  │ {per_g}"
        )
    ps_cand_lines += ["", "# Groups: " + " | ".join(f"{s}={g}"
                      for s, g in zip(GROUP_SHORT, GROUP_NAMES))]
    (OUT_DIR / "candidates_phase_split.txt").write_text("\n".join(ps_cand_lines))
    print(f"  {len(ps_candidates)} candidates  →  candidates_phase_split.txt")

    # ── 10. save full results ─────────────────────────────────────────────────
    with open(OUT_DIR / "analysis_results.pkl", "wb") as fh:
        pickle.dump({
            "all_results":    all_results,
            "stats":          stats,
            "phase_stats":    phase_stats,
            "cluster_rows":   cluster_rows,
            "ps_cluster_rows": ps_cluster_rows,
            "candidates":     candidates,
            "ps_candidates":  ps_candidates,
            "feature_names":  all_feats,
        }, fh, protocol=4)
    print("  analysis_results.pkl")

    # ── 11. console summary ───────────────────────────────────────────────────
    print()
    for path in [OUT_DIR / f for f in (
            "table_summary.txt", "table_phases.txt", "table_windows.txt",
            "table_detail.txt", "table_phase_split.txt")]:
        print(path.read_text())
        print()

    print("="*72)
    print("TOP 20 BCE CANDIDATES (original features)")
    print("="*72)
    hdr = (f"{'Feature':<{FW}} cross  disc  cl_sz │ " +
           " ".join(f"{s:>8}" for s in GROUP_SHORT))
    print(hdr)
    print("─"*72)
    for ea, feat in candidates[:20]:
        cr = cr_map.get(feat, {})
        fr = fr_map.get(feat, {})
        per_g = " ".join(f"{fr.get(f'ea_{g}',0.5):8.3f}" for g in GROUP_NAMES)
        print(f"{feat:<{FW}} {ea:.3f}    {fr.get('n_disc_60',0)}    "
              f"{cr.get('cluster_size',1):3d}  │ {per_g}")

    print()
    print("="*72)
    print("TOP 20 BCE CANDIDATES (phase-split — crossing features visible)")
    print("="*72)
    ps_hdr = (f"{'Feature':<{PS_FW}} cross  disc  cl_sz │ " +
              " ".join(f"{s:>8}" for s in GROUP_SHORT))
    print(ps_hdr)
    print("─"*72)
    for ea, feat in ps_candidates[:20]:
        cr = ps_cr_map.get(feat, {})
        fr = ps_fr_map.get(feat, {})
        per_g = " ".join(f"{fr.get(f'ea_{g}',0.5):8.3f}" for g in GROUP_NAMES)
        print(f"{feat:<{PS_FW}} {ea:.3f}    {fr.get('n_disc_60',0)}    "
              f"{cr.get('cluster_size',1):3d}  │ {per_g}")

    print()
    print("="*72)
    print("FEATURES ≥ 0.65 EFFECTIVE AUROC  (per group, counts by phase)")
    print("="*72)
    for g in GROUP_NAMES:
        good = [r for r in stats if r[f"ea_{g}"] >= 0.65]
        ph_cnt: dict = {}
        for r in good:
            ph = r[f"ph_{g}"]
            ph_cnt[ph] = ph_cnt.get(ph, 0) + 1
        print(f"  {g:25s}: {len(good):3d} features  {ph_cnt}")

    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
