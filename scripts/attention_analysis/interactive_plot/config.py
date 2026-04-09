from pathlib import Path

INPUT_DIR = Path("/home/ziyao/Documents/policy_records_20260407_155916")
OUTPUT_DIR = INPUT_DIR / "plots"
EPISODE_SUMMARIES_JSON = INPUT_DIR / "episode_summaries.json"

# ============================================================
# DATA SOURCE
# "gradcam"  -- use GradCAM attribution maps (outputs/debug/attr/...)
# "raw_attn" -- use raw attention weights    (outputs/debug/raw_attn/...)
# ============================================================
DATA_SOURCE = "gradcam"

_GRADCAM_IMG1  = "outputs/debug/attr/image/right_wrist_0_rgb"
_GRADCAM_IMG2  = "outputs/debug/attr/image/left_wrist_0_rgb"
_GRADCAM_TASK  = "outputs/debug/attr/task/scores"
_GRADCAM_STATE = "outputs/debug/attr/state/scores"

_RAW_ATTN_IMG1  = "outputs/debug/raw_attn/spans/image/right_wrist_0_rgb"
_RAW_ATTN_IMG2  = "outputs/debug/raw_attn/spans/image/left_wrist_0_rgb"
_RAW_ATTN_TASK  = "outputs/debug/raw_attn/spans/task"
_RAW_ATTN_STATE = "outputs/debug/raw_attn/spans/state"

if DATA_SOURCE == "gradcam":
    IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY = (
        _GRADCAM_IMG1, _GRADCAM_IMG2, _GRADCAM_TASK, _GRADCAM_STATE
    )
else:
    IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY = (
        _RAW_ATTN_IMG1, _RAW_ATTN_IMG2, _RAW_ATTN_TASK, _RAW_ATTN_STATE
    )

ALL_MODALITY_KEYS = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]
IMAGE_MODALITY_KEYS = [IMG1_KEY, IMG2_KEY]
VECTOR_MODALITY_KEYS = [TASK_KEY, STATE_KEY]

# Adjustable defaults — change these to switch behaviour globally
REDUCTION_METHOD = "average"   # "average" | "median" | "max" | "min" | "sum" | "sqrt_norm_sum"
NORM_METHOD = "zscore"  # "zscore" | "robust_zscore" | "minmax" | "none"

# Which episode to open in interactive mode (must match episode_summaries.json)
TASK_ID = 2
EPISODE_NUM = 5
