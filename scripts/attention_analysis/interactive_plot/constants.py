from pathlib import Path

INPUT_DIR = Path("/home/annsong/Documents/policy_records_20260324_141715")
OUTPUT_DIR = INPUT_DIR / "plots"
EPISODE_SUMMARIES_JSON = INPUT_DIR / "episode_summaries.json"

IMG1_KEY = "outputs/debug/attr/image/right_wrist_0_rgb"
IMG2_KEY = "outputs/debug/attr/image/left_wrist_0_rgb"
TASK_KEY = "outputs/debug/attr/task/scores"
STATE_KEY = "outputs/debug/attr/state/scores"

ALL_MODALITY_KEYS = [IMG1_KEY, IMG2_KEY, TASK_KEY, STATE_KEY]
IMAGE_MODALITY_KEYS = [IMG1_KEY, IMG2_KEY]
VECTOR_MODALITY_KEYS = [TASK_KEY, STATE_KEY]