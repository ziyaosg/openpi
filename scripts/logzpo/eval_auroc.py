import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/logpZO_combined_3000")
RESULTS_DIR = DATA_DIR / "results_2100_v2"
OBS_DIM = 512
HIDDEN_DIM = 1024
N_LAYERS = 6

# ── Network ───────────────────────────────────────────────────────────────
class FlowNet(nn.Module):
    hidden_dim: int
    n_layers: int
    obs_dim: int

    @nn.compact
    def __call__(self, obs, s):
        s_expanded = s[:, None]
        x = jnp.concatenate([obs, s_expanded], axis=-1)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.obs_dim)(x)
        return x

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
all_gc = np.load(DATA_DIR / "all_global_cond.npy", allow_pickle=True)
all_labels = np.load(DATA_DIR / "all_labels.npy")  # True = success

val_idx  = np.load(RESULTS_DIR / "split_val_idx.npy")
test_idx = np.load(RESULTS_DIR / "split_test_idx.npy")

print(f"Val:  {len(val_idx)} episodes")
print(f"Test: {len(test_idx)} episodes")

# ── Score function ────────────────────────────────────────────────────────
def make_score_fn(params, model):
    @jax.jit
    def compute_score(obs_batch):
        s = jnp.zeros(obs_batch.shape[0])
        Z_Ot = obs_batch + model.apply(params, obs_batch, s)
        return jnp.sum(Z_Ot ** 2, axis=-1)
    return compute_score

def compute_scores_for_episodes(compute_score, episode_indices):
    scores = []
    for i in episode_indices:
        s = compute_score(jnp.array(all_gc[i]))
        scores.append(np.array(s))
    return scores

# ── Helper: best F1 ───────────────────────────────────────────────────────
def best_f1_precision_recall(labels, scores):
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    f1_scores = np.where(
        (precision_curve[:-1] + recall_curve[:-1]) > 0,
        2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1]),
        0.0
    )
    if len(f1_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
    best_idx = np.argmax(f1_scores)
    return (
        float(precision_curve[best_idx]),
        float(recall_curve[best_idx]),
        float(f1_scores[best_idx]),
        float(thresholds[best_idx]),
    )

# ── Rolling metrics ───────────────────────────────────────────────────────
def compute_rolling_metrics(scores, episode_indices, setting_name, split_name):
    failure_labels = 1 - all_labels[episode_indices].astype(int)
    n_ep = len(episode_indices)
    max_T = max(len(s) for s in scores)
    baseline_auprc = float(failure_labels.sum()) / n_ep

    print(f"\n  {setting_name} / {split_name}: {n_ep} episodes, {int(failure_labels.sum())} failures")

    success_mean = np.mean([scores[i].mean() for i in range(n_ep) if failure_labels[i] == 0])
    failure_mean = np.mean([scores[i].mean() for i in range(n_ep) if failure_labels[i] == 1])
    print(f"  Sanity — success: {success_mean:.4f}, failure: {failure_mean:.4f}, correct: {failure_mean > success_mean}")

    summary_rows = []

    for t in range(max_T):
        cummax_t = np.array([
            np.max(scores[i][:min(t + 1, len(scores[i]))])
            for i in range(n_ep)
        ])

        auroc = float(roc_auc_score(failure_labels, cummax_t))
        auprc = float(average_precision_score(failure_labels, cummax_t))
        precision, recall, f1, threshold = best_f1_precision_recall(failure_labels, cummax_t)

        summary_rows.append({
            "timestep":           t,
            "auroc":              auroc,
            "auprc":              auprc,
            "precision":          precision,
            "recall":             recall,
            "f1":                 f1,
            "threshold":          threshold,
            "mean_score_success": float(cummax_t[failure_labels == 0].mean()),
            "mean_score_failure": float(cummax_t[failure_labels == 1].mean()),
            "n_episodes":         n_ep,
            "n_failure":          int(failure_labels.sum()),
            "setting":            setting_name,
            "split":              split_name,
        })

        if t % 10 == 0:
            print(f"    t={t:4d}: auroc={auroc:.4f}, auprc={auprc:.4f}, f1={f1:.4f}, p={precision:.4f}, r={recall:.4f}")

    aurocs = [r["auroc"] for r in summary_rows]
    f1s = [r["f1"] for r in summary_rows]
    print(f"  Peak AUROC:        {max(aurocs):.4f} at t={aurocs.index(max(aurocs))}")
    print(f"  Peak F1:           {max(f1s):.4f} at t={f1s.index(max(f1s))}")
    print(f"  Mean AUROC t=0-39: {np.mean(aurocs[:40]):.4f}")
    print(f"  Is monotonic:      {all(aurocs[i] <= aurocs[i+1] + 1e-10 for i in range(len(aurocs)-1))}")
    print(f"  Baseline AUPRC:    {baseline_auprc:.4f}")

    return summary_rows

# ── Run both settings ─────────────────────────────────────────────────────
all_summary_rows = []

for setting_name in ["setting1_full", "setting2_matched"]:
    print(f"\n{'='*50}")
    print(f"Processing {setting_name}")
    print(f"{'='*50}")

    param_path = RESULTS_DIR / f"logpZO_params_{setting_name}.pkl"
    print(f"Loading params from {param_path}...")
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    model = FlowNet(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, obs_dim=OBS_DIM)
    compute_score = make_score_fn(params, model)

    print(f"Scoring val ({len(val_idx)} episodes)...")
    val_scores = compute_scores_for_episodes(compute_score, val_idx)

    print(f"Scoring test ({len(test_idx)} episodes)...")
    test_scores = compute_scores_for_episodes(compute_score, test_idx)

    for split_name, scores, idx in [
        ("val",  val_scores,  val_idx),
        ("test", test_scores, test_idx),
    ]:
        rows = compute_rolling_metrics(scores, idx, setting_name, split_name)
        all_summary_rows.extend(rows)

        out_path = RESULTS_DIR / f"logpZO_summary_{setting_name}_{split_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

# ── Save combined CSV ─────────────────────────────────────────────────────
combined_path = RESULTS_DIR / "logpZO_summary_all.csv"
pd.DataFrame(all_summary_rows).to_csv(combined_path, index=False)
print(f"\nCombined CSV saved to {combined_path}")
print(f"Total rows: {len(all_summary_rows)}")
print("\nAll done.")