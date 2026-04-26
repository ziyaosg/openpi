from pathlib import Path

from ...attention_utils.keys import get_modality_keys

INPUT_DIR = Path("/nfs/roberts/scratch/pi_tkf6/as4643/policy_records/policy_records_20260426_100003")
OUTPUT_DIR = INPUT_DIR / "plots"
EPISODE_SUMMARIES_JSON = INPUT_DIR / "episode_summaries.json"

# ============================================================
# DATA SOURCE
# "gradcam" -- use GradCAM attribution maps (outputs/debug/gradcam/...)
# "attn"    -- use raw attention weights    (outputs/debug/attn/...)
# ============================================================
DATA_SOURCE = "gradcam"

_KEYS = get_modality_keys(DATA_SOURCE)
IMG1_KEY  = _KEYS["base"]
IMG2_KEY  = _KEYS["left_wrist"]
TASK_KEY  = _KEYS["task"]
STATE_KEY = _KEYS["state"]

ALL_MODALITY_KEYS    = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]
IMAGE_MODALITY_KEYS  = [IMG1_KEY, IMG2_KEY]
VECTOR_MODALITY_KEYS = [TASK_KEY, STATE_KEY]

# Adjustable defaults — change these to switch behaviour globally
REDUCTION_METHOD = "average"   # "average" | "median" | "max" | "min" | "sum" | "sqrt_norm_sum"
NORM_METHOD = "zscore"  # "zscore" | "robust_zscore" | "minmax" | "none"

# Which episode to open in interactive mode (must match episode_summaries.json)
TASK_ID = 2
EPISODE_NUM = 7
