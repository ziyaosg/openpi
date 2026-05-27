import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/as4643/scratch_pi_tkf6/as4643/policy_records_pi05_liberoplus_20260524_163155_logpZO_data")
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

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading params...")
with open(DATA_DIR / "logpZO_params.pkl", "rb") as f:
    params = pickle.load(f)
print("Params loaded.")

print("Loading test data...")
test_gc = np.load(DATA_DIR / "test_global_cond.npy", allow_pickle=True)
test_labels = np.load(DATA_DIR / "test_labels.npy")
failure_labels = 1 - test_labels.astype(int)
print(f"Loaded {len(test_gc)} episodes — {int(failure_labels.sum())} failures, {int((1-failure_labels).sum())} successes")

# ── Episode length stats ──────────────────────────────────────────────────
ep_lengths = [len(gc) for gc in test_gc]
max_T = max(ep_lengths)
print(f"\nEpisode length stats:")
print(f"  Min: {min(ep_lengths)}, Max: {max_T}, Mean: {np.mean(ep_lengths):.1f}")
print(f"  Episodes with length > 39: {sum(l > 39 for l in ep_lengths)}")
print(f"  Episodes with length > 79: {sum(l > 79 for l in ep_lengths)}")
print(f"  Full max timestep: {max_T}")

model = FlowNet(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, obs_dim=OBS_DIM)
print(f"\nModel: {N_LAYERS} layers, hidden_dim={HIDDEN_DIM}, obs_dim={OBS_DIM}")

@jax.jit
def compute_score(obs_batch):
    s = jnp.zeros(obs_batch.shape[0])
    Z_Ot = obs_batch + model.apply(params, obs_batch, s)
    return jnp.sum(Z_Ot ** 2, axis=-1)

# ── Compute scores ────────────────────────────────────────────────────────
print("\nComputing scores...")
all_scores = []
for ep_idx, ep_gc in enumerate(test_gc):
    scores = compute_score(jnp.array(ep_gc))
    all_scores.append(np.array(scores))
    if ep_idx % 100 == 0:
        print(f"  Episode {ep_idx}/{len(test_gc)} — T={len(scores)}, score range: [{scores.min():.2f}, {scores.max():.2f}]")

all_scores = np.array(all_scores, dtype=object)
print("All scores computed.")

# ── Sanity check ──────────────────────────────────────────────────────────
success_scores = [all_scores[i].mean() for i in range(len(all_scores)) if failure_labels[i] == 0]
failure_scores_list = [all_scores[i].mean() for i in range(len(all_scores)) if failure_labels[i] == 1]
print(f"\nSanity check:")
print(f"  Mean score — success: {np.mean(success_scores):.4f}")
print(f"  Mean score — failure: {np.mean(failure_scores_list):.4f}")
print(f"  Failure > success: {np.mean(failure_scores_list) > np.mean(success_scores)}")

# ── Build per-timestep rows ───────────────────────────────────────────────
print("\nBuilding output rows (all timesteps)...")
rows = []
for ep_idx, (scores, label, gc) in enumerate(zip(all_scores, failure_labels, test_gc)):
    for t, (score, obs) in enumerate(zip(scores, gc)):
        rows.append({
            "episode": ep_idx,
            "timestep": t,
            "failure": int(label),
            "logpZO_score": float(score),
            "obs_norm": float(np.linalg.norm(obs)),
        })

df_scores = pd.DataFrame(rows)
print(f"Output dataframe shape: {df_scores.shape}")
print(f"Timestep range: {df_scores['timestep'].min()} - {df_scores['timestep'].max()}")

# ── Rolling AUROC over all timesteps ─────────────────────────────────────
print(f"\nComputing rolling AUROC (t=0 to {max_T-1})...")
auroc_by_t = {}
n_episodes_by_t = {}
n_fail_by_t = {}

for t in range(max_T):
    valid_idx = [i for i in range(len(all_scores)) if len(all_scores[i]) > t]
    n_valid = len(valid_idx)
    if n_valid < 10:
        print(f"  t={t}: only {n_valid} episodes remaining, stopping.")
        break

    scores_t = np.array([all_scores[i][t] for i in valid_idx])
    labels_t = failure_labels[valid_idx]
    n_fail = int(labels_t.sum())
    n_episodes_by_t[t] = n_valid
    n_fail_by_t[t] = n_fail

    if labels_t.sum() == 0 or labels_t.sum() == n_valid:
        auroc_by_t[t] = 0.5
        continue

    auroc_by_t[t] = float(roc_auc_score(labels_t, scores_t))

    if t % 10 == 0:
        print(f"  t={t:4d}: auroc={auroc_by_t[t]:.4f}, n_episodes={n_valid}, n_fail={n_fail}")

df_scores["auroc_at_t"] = df_scores["timestep"].map(auroc_by_t)

# ── Summary ───────────────────────────────────────────────────────────────
valid_aurocs = list(auroc_by_t.values())
print(f"\nAUROC summary (all timesteps):")
print(f"  Timesteps evaluated: {len(auroc_by_t)}")
print(f"  Peak AUROC:          {max(valid_aurocs):.4f} at t={max(auroc_by_t, key=auroc_by_t.get)}")
print(f"  Mean AUROC:          {np.mean(valid_aurocs):.4f}")
print(f"  AUROC at t=0:        {auroc_by_t.get(0, float('nan')):.4f}")
print(f"  AUROC at t=39:       {auroc_by_t.get(39, float('nan')):.4f}")

print(f"\nMean logpZO score by outcome:")
print(df_scores.groupby("failure")["logpZO_score"].agg(["mean", "std", "min", "max"]).rename(index={0: "success", 1: "failure"}))

# ── Save ──────────────────────────────────────────────────────────────────
out_path = DATA_DIR / "logpZO_full_output.csv"
df_scores.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# ── Also save AUROC summary as separate CSV ───────────────────────────────
auroc_rows = []
for t, auroc in auroc_by_t.items():
    auroc_rows.append({
        "timestep": t,
        "auroc": auroc,
        "n_episodes": n_episodes_by_t[t],
        "n_failure": n_fail_by_t[t],
    })
df_auroc = pd.DataFrame(auroc_rows)
auroc_path = DATA_DIR / "logpZO_auroc_summary.csv"
df_auroc.to_csv(auroc_path, index=False)
print(f"AUROC summary saved to {auroc_path}")