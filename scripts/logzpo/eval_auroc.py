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

print("Loading test data...")
test_gc = np.load(DATA_DIR / "test_global_cond.npy", allow_pickle=True)
test_labels = np.load(DATA_DIR / "test_labels.npy")
failure_labels = 1 - test_labels.astype(int)

print(f"Loaded {len(test_gc)} episodes — {int(failure_labels.sum())} failures, {int((1-failure_labels).sum())} successes")

model = FlowNet(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, obs_dim=OBS_DIM)

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
np.save(DATA_DIR / "logpZO_scores.npy", all_scores)
print("Scores saved.")

# ── Sanity check ──────────────────────────────────────────────────────────
success_scores = [all_scores[i].mean() for i in range(len(all_scores)) if failure_labels[i] == 0]
failure_scores_list = [all_scores[i].mean() for i in range(len(all_scores)) if failure_labels[i] == 1]
print(f"\nSanity check:")
print(f"  Mean score — success: {np.mean(success_scores):.4f}")
print(f"  Mean score — failure: {np.mean(failure_scores_list):.4f}")
print(f"  Failure > success: {np.mean(failure_scores_list) > np.mean(success_scores)}")

# ── Cumulative max rolling AUROC ──────────────────────────────────────────
print("\nComputing cumulative max rolling AUROC...")
max_T = max(len(s) for s in all_scores)
print(f"Max episode length: {max_T}")

auroc_by_timestep = []
for t in range(max_T):
    # Only episodes that have reached timestep t
    valid_idx = [i for i in range(len(all_scores)) if len(all_scores[i]) > t]
    if len(valid_idx) < 10:
        print(f"  t={t}: only {len(valid_idx)} episodes remaining, stopping.")
        break

    # Cumulative max score up to and including timestep t
    cummax_scores_t = np.array([
        np.max(all_scores[i][:t+1]) for i in valid_idx
    ])
    labels_t = failure_labels[valid_idx]

    if labels_t.sum() == 0 or labels_t.sum() == len(labels_t):
        auroc_by_timestep.append(0.5)
        continue

    auroc = roc_auc_score(labels_t, cummax_scores_t)
    auroc_by_timestep.append(auroc)

    if t % 10 == 0:
        print(f"  t={t:4d}: auroc={auroc:.4f}, n_episodes={len(valid_idx)}, n_fail={int(labels_t.sum())}")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\nAUROC summary:")
print(f"  Timesteps evaluated: {len(auroc_by_timestep)}")
print(f"  Peak AUROC:          {max(auroc_by_timestep):.4f} at t={auroc_by_timestep.index(max(auroc_by_timestep))}")
print(f"  Mean AUROC:          {np.mean(auroc_by_timestep):.4f}")
print(f"  AUROC at t=0:        {auroc_by_timestep[0]:.4f}")
print(f"  AUROC at t=39:       {auroc_by_timestep[min(39, len(auroc_by_timestep)-1)]:.4f}")

# ── Save CSV matching existing format ─────────────────────────────────────
rows = []
for t, auroc in enumerate(auroc_by_timestep):
    valid_idx = [i for i in range(len(all_scores)) if len(all_scores[i]) > t]
    n_fail = int(failure_labels[valid_idx].sum())
    rows.append({
        "timestep": t,
        "auroc": auroc,
        "n_episodes": len(valid_idx),
        "n_failure": n_fail,
    })

df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "logpZO_auroc.csv", index=False)
print(f"\nSaved to {DATA_DIR / 'logpZO_auroc.csv'}")