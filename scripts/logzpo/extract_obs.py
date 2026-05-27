import json
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
ROLLOUT_DIR = Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260524_163155")
OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/policy_records_pi05_liberoplus_20260524_163155_logpZO_data")
# ROLLOUT_DIR = Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260522_032743")
# OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/policy_records_pi05_liberoplus_20260522_032743_logpZO_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load episode summaries ────────────────────────────────────────────────
with open("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260524_163155/client_output/episode_summaries.json") as f:
    episodes = json.load(f)

def extract_global_cond(data):
    """
    v shape:       (2, 968, 1, 256)  → (L, S, K, H)
    weights shape: (2, 8, 10, 968)   → (L, heads, T_action, S)

    Strategy: for each layer, compute attention-weighted sum of V vectors
    across sequence positions, averaged over action tokens and heads.
    Then concatenate across layers → (2 * 256,) = (512,)
    """
    v = data["outputs/debug/attn/v"]        # (2, 968, 1, 256)
    w = data["outputs/debug/attn/weights"]  # (2, 8, 10, 968)

    # Remove K=1 head dim from v → (2, 968, 256)
    v = v[:, :, 0, :]

    # Mean over query heads → (2, 10, 968)
    w = w.mean(axis=1)

    # Mean over action tokens → (2, 968)
    w = w.mean(axis=1)

    # Normalize weights to sum to 1 over sequence dim
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)  # (2, 968)

    # Weighted sum of V over sequence positions → (2, 256)
    # einsum: for each layer l, sum over s: w[l,s] * v[l,s,:]
    global_cond = np.einsum("ls,lsd->ld", w, v)  # (2, 256)

    # Flatten across layers → (512,)
    global_cond = global_cond.reshape(-1)

    return global_cond

# ── Extract ───────────────────────────────────────────────────────────────
train_global_cond = []
test_global_cond = []
test_labels = []

for ep_idx, ep in enumerate(episodes):
    start_idx = ep["start_idx"]
    end_idx = ep["end_idx"]
    success = ep["success"]

    ep_global_cond = []

    for t in range(start_idx, end_idx + 1):
        data = np.load(ROLLOUT_DIR / f"step_{t}.npy", allow_pickle=True).item()
        global_cond = extract_global_cond(data)  # (512,)
        ep_global_cond.append(global_cond)

    ep_global_cond = np.stack(ep_global_cond)  # (T, 512)
    test_global_cond.append(ep_global_cond)
    test_labels.append(success)

    if success:
        train_global_cond.append(ep_global_cond)

    if ep_idx % 10 == 0:
        print(f"Processed {ep_idx}/{len(episodes)} episodes")

# ── Save ──────────────────────────────────────────────────────────────────
train_X = np.concatenate(train_global_cond, axis=0)  # (N_success_steps, 512)
np.save(OUTPUT_DIR / "train_X.npy", train_X)
np.save(OUTPUT_DIR / "test_global_cond.npy", np.array(test_global_cond, dtype=object))
np.save(OUTPUT_DIR / "test_labels.npy", np.array(test_labels))

print(f"Done.")
print(f"Train: {train_X.shape}")
print(f"Test episodes: {len(test_global_cond)}")
print(f"  Success: {sum(test_labels)}, Fail: {len(test_labels) - sum(test_labels)}")