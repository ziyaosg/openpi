import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import pickle
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/as4643/scratch_pi_tkf6/as4643/policy_records_pi05_liberoplus_20260524_163155_logpZO_data")
OUTPUT_DIR = Path("/home/as4643/scratch_pi_tkf6/as4643/policy_records_pi05_liberoplus_20260524_163155_logpZO_data")
OBS_DIM = 512
HIDDEN_DIM = 1024
N_LAYERS = 6
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-4
SEED = 0

# ── Flow matching network ─────────────────────────────────────────────────
class FlowNet(nn.Module):
    """
    Takes (O_t, s) and predicts velocity field.
    Input: obs (OBS_DIM,) concatenated with s (1,) → (OBS_DIM+1,)
    Output: predicted velocity (OBS_DIM,)
    """
    hidden_dim: int
    n_layers: int
    obs_dim: int

    @nn.compact
    def __call__(self, obs, s):
        # s: scalar timestep in [0, 1], shape (B,)
        # obs: shape (B, OBS_DIM)
        s_expanded = s[:, None]  # (B, 1)
        x = jnp.concatenate([obs, s_expanded], axis=-1)  # (B, OBS_DIM+1)
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.obs_dim)(x)
        return x

# ── Training ──────────────────────────────────────────────────────────────
def flow_matching_loss(params, apply_fn, obs_batch, rng):
    rng_z, rng_s = jax.random.split(rng)
    Z = jax.random.normal(rng_z, obs_batch.shape)
    s = jax.random.uniform(rng_s, (obs_batch.shape[0],))
    s_expanded = s[:, None]
    obs_noisy = obs_batch + s_expanded * (Z - obs_batch)
    target = Z - obs_batch
    pred = apply_fn(params, obs_noisy, s)  # ← apply_fn directly, not model.apply
    return jnp.mean((pred - target) ** 2)

@jax.jit
def train_step(state, obs_batch, rng):
    loss, grads = jax.value_and_grad(flow_matching_loss)(
        state.params, state.apply_fn, obs_batch, rng  # ← pass apply_fn not model
    )
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    rng = jax.random.PRNGKey(SEED)

    # Load training data
    train_X = np.load(DATA_DIR / "train_X.npy")  # (23096, 512)
    n_samples = train_X.shape[0]
    print(f"Training on {n_samples} samples, obs_dim={OBS_DIM}")

    # Initialize model
    model = FlowNet(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, obs_dim=OBS_DIM)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.ones((1, OBS_DIM))
    dummy_s = jnp.ones((1,))
    params = model.init(init_rng, dummy_obs, dummy_s)

    # Initialize optimizer
    tx = optax.adam(LR)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # Training loop
    n_batches = n_samples // BATCH_SIZE
    for epoch in range(EPOCHS):
        # Shuffle
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, n_samples)
        train_X_shuffled = train_X[np.array(perm)]

        epoch_loss = 0.0
        for b in range(n_batches):
            obs_batch = jnp.array(
                train_X_shuffled[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            )
            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(state, obs_batch, step_rng)
            epoch_loss += loss

        epoch_loss /= n_batches

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} — loss: {epoch_loss:.4f}")

    # Save params
    with open(OUTPUT_DIR / "logpZO_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print(f"Saved params to {OUTPUT_DIR / 'logpZO_params.pkl'}")

if __name__ == "__main__":
    main()