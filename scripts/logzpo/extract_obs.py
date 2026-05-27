import json
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
SOURCES = [
    {
        "rollout_dir": Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260512_182606"),
        "summary": Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260512_182606/client_output/episode_summaries.json"),
    },
    {
        "rollout_dir": Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260512_201032"),
        "summary": Path("/nfs/roberts/scratch/pi_tkf6/zs377/policy_records_pi05_libero_20260512_201032/client_output/episode_summaries.json"),
    },
    {
        "rollout_dir": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260522_032743"),
        "summary": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260522_032743/client_output/episode_summaries.json"),
    },
    {
        "rollout_dir": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260522_031717"),
        "summary": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260522_031717/client_output/episode_summaries.json"),
    },
    {
        "rollout_dir": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260524_163155"),
        "summary": Path("/nfs/roberts/project/pi_tkf6/zs377/policy_records_pi05_liberoplus_20260524_163155/client_output/episode_summaries.json"),
    },
]

OUTPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/logpZO_combined_3000")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Extraction function ───────────────────────────────────────────────────
def extract_global_cond(data):
    v = data["outputs/debug/attn/v"]        # (2, 968, 1, 256)
    w = data["outputs/debug/attn/weights"]  # (2, 8, 10, 968)
    v = v[:, :, 0, :]                       # (2, 968, 256)
    w = w.mean(axis=1)                      # (2, 10, 968)
    w = w.mean(axis=1)                      # (2, 968)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    global_cond = np.einsum("ls,lsd->ld", w, v)  # (2, 256)
    return global_cond.reshape(-1)           # (512,)

# ── Extract across all sources ────────────────────────────────────────────
all_global_cond = []  # one entry per episode, each (T, 512)
all_labels = []       # one bool per episode

total_episodes = 0
total_success = 0
total_failure = 0

for source_idx, source in enumerate(SOURCES):
    rollout_dir = source["rollout_dir"]
    summary_path = source["summary"]

    print(f"\n{'='*50}")
    print(f"Source {source_idx+1}/5: {rollout_dir.name}")

    with open(summary_path) as f:
        episodes = json.load(f)

    print(f"  Episodes: {len(episodes)}")
    source_success = sum(ep["success"] for ep in episodes)
    source_failure = len(episodes) - source_success
    print(f"  Success: {source_success}, Failure: {source_failure}")

    for ep_idx, ep in enumerate(episodes):
        start_idx = ep["start_idx"]
        end_idx = ep["end_idx"]
        success = ep["success"]

        ep_global_cond = []

        for t in range(start_idx, end_idx + 1):
            data = np.load(rollout_dir / f"step_{t}.npy", allow_pickle=True).item()
            global_cond = extract_global_cond(data)
            ep_global_cond.append(global_cond)

        ep_global_cond = np.stack(ep_global_cond)  # (T, 512)
        all_global_cond.append(ep_global_cond)
        all_labels.append(success)

        if ep_idx % 50 == 0:
            print(f"  Episode {ep_idx}/{len(episodes)} — T={len(ep_global_cond)}")

    total_episodes += len(episodes)
    total_success += source_success
    total_failure += source_failure
    print(f"  Done. Running total: {total_episodes} episodes")

# ── Save combined ─────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Total: {total_episodes} episodes, {total_success} success, {total_failure} failure")

all_labels_arr = np.array(all_labels)
all_global_cond_arr = np.array(all_global_cond, dtype=object)

np.save(OUTPUT_DIR / "all_global_cond.npy", all_global_cond_arr)
np.save(OUTPUT_DIR / "all_labels.npy", all_labels_arr)

print(f"Saved:")
print(f"  all_global_cond.npy — {len(all_global_cond_arr)} episodes")
print(f"  all_labels.npy      — {all_labels_arr.shape}, {all_labels_arr.sum()} success")

# ── Sanity check ──────────────────────────────────────────────────────────
ep_lengths = [len(gc) for gc in all_global_cond]
print(f"\nEpisode length stats:")
print(f"  Min: {min(ep_lengths)}, Max: {max(ep_lengths)}, Mean: {np.mean(ep_lengths):.1f}")
any_nan = any(np.isnan(gc).any() for gc in all_global_cond)
print(f"  Any NaN: {any_nan}")