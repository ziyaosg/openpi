import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/logpZO_combined_3000")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/logpZO_combined_3000/results_2100_v4")
OUTPUT_DIR.mkdir(exist_ok=True)

OBS_DIM = 512
HIDDEN_DIM = 1024
N_LAYERS = 6
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-4
SEED = 42

print("Devices:", jax.devices())

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
print("\nLoading data...")
all_gc = np.load(DATA_DIR / "all_global_cond.npy", allow_pickle=True)
all_labels = np.load(DATA_DIR / "all_labels.npy")  # True = success
n_total = len(all_gc)
print(f"Total: {n_total} episodes, {all_labels.sum()} success, {(~all_labels).sum()} failure")

# ── Split ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

all_success_idx = np.where(all_labels)[0]   # ~2463
all_failure_idx = np.where(~all_labels)[0]  # ~537

rng.shuffle(all_success_idx)
rng.shuffle(all_failure_idx)

n_train_success = 2100
n_val = 450
n_test = 450

# Success split: 2100 train, remainder to val/test
train_success_idx = all_success_idx[:n_train_success]
remaining_success = all_success_idx[n_train_success:]  # ~363

n_remaining_success = len(remaining_success)
n_val_success = n_remaining_success // 2
n_test_success = n_remaining_success - n_val_success

val_success_idx  = remaining_success[:n_val_success]
test_success_idx = remaining_success[n_val_success:]

# Failure split: fill remaining val/test slots
n_val_failure  = n_val - n_val_success
n_test_failure = n_test - n_test_success

val_failure_idx  = all_failure_idx[:n_val_failure]
test_failure_idx = all_failure_idx[n_val_failure:n_val_failure + n_test_failure]

# Combine and shuffle val and test
val_idx  = np.concatenate([val_success_idx, val_failure_idx])
test_idx = np.concatenate([test_success_idx, test_failure_idx])
rng.shuffle(val_idx)
rng.shuffle(test_idx)

print(f"\nSplit:")
print(f"  Train: {len(train_success_idx)} — {len(train_success_idx)} success, 0 failure")
print(f"  Val:   {len(val_idx)} — {n_val_success} success, {n_val_failure} failure")
print(f"  Test:  {len(test_idx)} — {n_test_success} success, {n_test_failure} failure")

# Setting 1: all 2100 successful training episodes
setting1_train_idx = train_success_idx

# Setting 2: 1050 randomly sampled from the 2100
setting2_train_idx = rng.choice(train_success_idx, size=1050, replace=False)

print(f"\nSuccessful episodes in train: {len(train_success_idx)}")
print(f"Setting 1 — full:    {len(setting1_train_idx)} successful episodes")
print(f"Setting 2 — matched: {len(setting2_train_idx)} successful episodes")

# Save split indices
np.save(OUTPUT_DIR / "split_train_success_idx.npy", train_success_idx)
np.save(OUTPUT_DIR / "split_val_idx.npy", val_idx)
np.save(OUTPUT_DIR / "split_test_idx.npy", test_idx)
print("Split indices saved.")

# ── Build flat training arrays ────────────────────────────────────────────
def build_train_X(episode_indices):
    return np.concatenate([all_gc[i] for i in episode_indices], axis=0)

# ── Training functions ────────────────────────────────────────────────────
def flow_matching_loss(params, apply_fn, obs_batch, rng):
    rng_z, rng_s = jax.random.split(rng)
    Z = jax.random.normal(rng_z, obs_batch.shape)
    s = jax.random.uniform(rng_s, (obs_batch.shape[0],))
    s_expanded = s[:, None]
    obs_noisy = obs_batch + s_expanded * (Z - obs_batch)
    target = Z - obs_batch
    pred = apply_fn(params, obs_noisy, s)
    return jnp.mean((pred - target) ** 2)

@jax.jit
def train_step(state, obs_batch, rng):
    loss, grads = jax.value_and_grad(flow_matching_loss)(
        state.params, state.apply_fn, obs_batch, rng
    )
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_flow_network(train_X, setting_name, jax_rng):
    print(f"\n{'='*50}")
    print(f"Training {setting_name}")
    print(f"  Timesteps: {train_X.shape[0]}, obs_dim: {train_X.shape[1]}")
    print(f"{'='*50}")

    n_samples = train_X.shape[0]
    model = FlowNet(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, obs_dim=OBS_DIM)

    # ── Check for existing checkpoint ────────────────────────────────────
    ckpt_path = OUTPUT_DIR / f"logpZO_checkpoint_{setting_name}.pkl"

    jax_rng, init_rng = jax.random.split(jax_rng)
    params = model.init(init_rng, jnp.ones((1, OBS_DIM)), jnp.ones((1,)))
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(LR),
    )
    best_loss = float("inf")
    best_params = None
    start_epoch = 0

    if ckpt_path.exists():
        print(f"  Resuming from checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        state = state.replace(
            step=ckpt["step"],
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
        )
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        best_params = ckpt["best_params"]
        jax_rng = ckpt["jax_rng"]
        print(f"  Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    else:
        print(f"  No checkpoint found, starting fresh.")

    n_batches = n_samples // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        jax_rng, shuffle_rng = jax.random.split(jax_rng)
        perm = jax.random.permutation(shuffle_rng, n_samples)
        train_X_shuffled = train_X[np.array(perm)]

        epoch_loss = 0.0
        for b in range(n_batches):
            obs_batch = jnp.array(
                train_X_shuffled[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            )
            jax_rng, step_rng = jax.random.split(jax_rng)
            state, loss = train_step(state, obs_batch, step_rng)
            epoch_loss += loss

        epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_params = state.params

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS} — loss: {epoch_loss:.4f} (best: {best_loss:.4f})")

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "step": int(state.step),
                "params": state.params,
                "opt_state": state.opt_state,
                "best_loss": best_loss,
                "best_params": best_params,
                "jax_rng": jax_rng,
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  Checkpoint saved at epoch {epoch}")

    print(f"Done. Best loss: {best_loss:.4f}")

    # Clean up checkpoint on successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  Checkpoint deleted (training complete)")

    return best_params, model

# ── Scoring ───────────────────────────────────────────────────────────────
def compute_scores_for_episodes(params, model, episode_indices):
    @jax.jit
    def score_batch(obs_batch):
        s = jnp.zeros(obs_batch.shape[0])
        Z_Ot = obs_batch + model.apply(params, obs_batch, s)
        return jnp.sum(Z_Ot ** 2, axis=-1)

    scores = []
    for i in episode_indices:
        s = score_batch(jnp.array(all_gc[i]))
        scores.append(np.array(s))
    return scores

# ── Rolling AUROC/AUPRC ───────────────────────────────────────────────────
def compute_rolling_metrics(scores, episode_indices, split_name):
    failure_labels = 1 - all_labels[episode_indices].astype(int)
    n_ep = len(episode_indices)
    max_T = max(len(s) for s in scores)
    baseline_auprc = float(failure_labels.sum()) / n_ep

    print(f"\n  {split_name}: {n_ep} episodes, {int(failure_labels.sum())} failures")

    success_mean = np.mean([scores[i].mean() for i in range(n_ep) if failure_labels[i] == 0])
    failure_mean = np.mean([scores[i].mean() for i in range(n_ep) if failure_labels[i] == 1])
    print(f"  Sanity — success: {success_mean:.4f}, failure: {failure_mean:.4f}, correct: {failure_mean > success_mean}")

    auroc_by_t = []
    auprc_by_t = []

    for t in range(max_T):
        cummax_t = np.array([
            np.max(scores[i][:min(t + 1, len(scores[i]))])
            for i in range(n_ep)
        ])

        if failure_labels.sum() == 0 or failure_labels.sum() == n_ep:
            auroc_by_t.append(0.5)
            auprc_by_t.append(baseline_auprc)
            continue

        auroc_by_t.append(float(roc_auc_score(failure_labels, cummax_t)))
        auprc_by_t.append(float(average_precision_score(failure_labels, cummax_t)))

    print(f"  Peak AUROC:        {max(auroc_by_t):.4f} at t={auroc_by_t.index(max(auroc_by_t))}")
    print(f"  Mean AUROC t=0-39: {np.mean(auroc_by_t[:40]):.4f}")
    print(f"  AUROC at t=39:     {auroc_by_t[min(39, len(auroc_by_t)-1)]:.4f}")
    print(f"  Is monotonic:      {all(auroc_by_t[i] <= auroc_by_t[i+1] + 1e-10 for i in range(len(auroc_by_t)-1))}")
    print(f"  Baseline AUPRC:    {baseline_auprc:.4f} (random)")

    return auroc_by_t, auprc_by_t, n_ep, int(failure_labels.sum())

# ── Run both settings ─────────────────────────────────────────────────────
jax_rng = jax.random.PRNGKey(SEED)

for setting_name, train_idx in [
    ("setting1_full",    setting1_train_idx),
    ("setting2_matched", setting2_train_idx),
]:
    train_X = build_train_X(train_idx)
    print(f"\nBuilt train_X for {setting_name}: {train_X.shape}")

    jax_rng, train_rng = jax.random.split(jax_rng)
    params, model = train_flow_network(train_X, setting_name, train_rng)

    with open(OUTPUT_DIR / f"logpZO_params_{setting_name}.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"Params saved: logpZO_params_{setting_name}.pkl")

    print(f"\nScoring val and test...")
    val_scores  = compute_scores_for_episodes(params, model, val_idx)
    test_scores = compute_scores_for_episodes(params, model, test_idx)

    for split_name, scores, idx in [
        ("val",  val_scores,  val_idx),
        ("test", test_scores, test_idx),
    ]:
        auroc_list, auprc_list, n_ep, n_fail = compute_rolling_metrics(
            scores, idx, split_name
        )

        rows = []
        for t, (auroc, auprc) in enumerate(zip(auroc_list, auprc_list)):
            rows.append({
                "timestep":   t,
                "auroc":      auroc,
                "auprc":      auprc,
                "n_episodes": n_ep,
                "n_failure":  n_fail,
                "setting":    setting_name,
                "split":      split_name,
            })

        out_path = OUTPUT_DIR / f"logpZO_auroc_{setting_name}_{split_name}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

print("\nAll done.")