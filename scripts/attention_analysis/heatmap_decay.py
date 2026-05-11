import numpy as np
import matplotlib
matplotlib.use("Agg")  # CRITICAL for clusters (no GUI)

import matplotlib.pyplot as plt
from pathlib import Path

from ..attention_utils.io import load_episode_infos, load_episode_records
from ..attention_utils.keys import get_modality_keys
from ..attention_utils.series import FULL_ATTN_KEY, extract_patches, reduce_vals


# ============================================================
# CONFIG
# ============================================================
DATA_SOURCE = "gradcam"
MODALITY_KEYS = list(get_modality_keys(DATA_SOURCE).values())

REDUCTION_METHOD = "average"
DECAY = 0.7


# ============================================================
# TOKEN MAPS
# ============================================================
def get_token_maps(record, modality_key):
    patches = extract_patches(record, modality_key, DATA_SOURCE, FULL_ATTN_KEY)

    token_maps = []
    for t in range(5):
        try:
            p = patches[t]
            val = reduce_vals(p, REDUCTION_METHOD)
            token_maps.append(val)
        except Exception:
            continue

    return token_maps


# ============================================================
# DECAY FUSION
# ============================================================
def apply_decay(token_maps, decay):
    weights = np.array([decay ** i for i in range(len(token_maps))])

    weighted = [token_maps[i] * weights[i] for i in range(len(token_maps))]
    combined = np.sum(weighted, axis=0)

    return weights, weighted, combined


# ============================================================
# SAVE FUNCTION
# ============================================================
def save_step(ep, step_idx, t, record, out_dir, decay=DECAY):

    token_maps = None

    # SAFE extraction via pipeline (your correct method)
    for k in MODALITY_KEYS:
        try:
            token_maps = get_token_maps(record, k)
            if token_maps:
                break
        except Exception:
            continue

    if not token_maps:
        token_maps = [np.zeros((10, 10)) for _ in range(5)]

    weights, weighted, combined = apply_decay(token_maps, decay)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    for i in range(5):
        axes[i].imshow(weighted[i], cmap="viridis")
        axes[i].set_title(f"T{i} w={weights[i]:.2f}")
        axes[i].axis("off")

    axes[5].imshow(combined, cmap="viridis")
    axes[5].set_title("Decay Fusion")
    axes[5].axis("off")

    fig.suptitle(
        f"{ep.task} | ep={ep.episode_num} | success={ep.success} | step={t}",
        fontsize=8
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(out_dir) / f"step_{step_idx:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"saved {save_path}")


# ============================================================
# MAIN LOOP
# ============================================================
def run(input_dir, episode_json, out_dir):

    episodes = load_episode_infos(episode_json)

    episodes = sorted(episodes, key=lambda e: (e.task_id, e.episode_num))

    for ep in episodes:
        print(f"\nProcessing task={ep.task_id}, ep={ep.episode_num}, success={ep.success}")

        records = load_episode_records(ep, input_dir)

        ep_dir = Path(out_dir) / f"task_{ep.task_id}" / f"ep_{ep.episode_num}" / ("success" if ep.success else "failure")

        for step_idx, (t, record) in enumerate(records):
            save_step(ep, step_idx, t, record, ep_dir)


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":

    STEP_DIR = "/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/policy_records_20260426_100003"
    EPISODE_JSON = "/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/policy_records_20260426_100003/episode_summaries.json"
    आउट_DIR = "/nfs/roberts/scratch/pi_tkf6/as4643/heatmaps_output"

    run(STEP_DIR, EPISODE_JSON, आउट_DIR)