from __future__ import annotations

GRADCAM_KEYS: dict[str, str] = {
    "base":       "outputs/debug/attr/image/base_0_rgb",
    "left_wrist": "outputs/debug/attr/image/left_wrist_0_rgb",
    "task":       "outputs/debug/attr/task/scores",
    "state":      "outputs/debug/attr/state/scores",
}

RAW_ATTN_KEYS: dict[str, str] = {
    "base":       "outputs/debug/raw_attn/spans/image/base_0_rgb",
    "left_wrist": "outputs/debug/raw_attn/spans/image/left_wrist_0_rgb",
    "task":       "outputs/debug/raw_attn/spans/task",
    "state":      "outputs/debug/raw_attn/spans/state",
}


# Camera short names — used to construct span/grid record keys
CAM_NAMES: list[str] = ["base_0_rgb", "left_wrist_0_rgb"]

# Input observation image keys (same order as CAM_NAMES)
CAM_IMAGE_KEYS: list[str] = [
    "inputs/observation/image",
    "inputs/observation/wrist_image",
]


def get_modality_keys(data_source: str) -> dict[str, str]:
    """Return the {name: key} mapping for the given data_source ('gradcam' or 'raw_attn')."""
    if data_source == "gradcam":
        return GRADCAM_KEYS
    if data_source == "raw_attn":
        return RAW_ATTN_KEYS
    raise ValueError(f"Unknown data_source: {data_source!r}. Choose 'gradcam' or 'raw_attn'.")
