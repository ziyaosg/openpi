from __future__ import annotations

GRADCAM_KEYS: dict[str, str] = {
    "base":       "outputs/debug/gradcam/image/base_0_rgb",
    "left_wrist": "outputs/debug/gradcam/image/left_wrist_0_rgb",
    "task":       "outputs/debug/gradcam/task",
    "state":      "outputs/debug/gradcam/state",
}

ATTN_KEYS: dict[str, str] = {
    "base":       "outputs/debug/spans/image/base_0_rgb",
    "left_wrist": "outputs/debug/spans/image/left_wrist_0_rgb",
    "task":       "outputs/debug/spans/task",
    "state":      "outputs/debug/spans/state",
}


# Camera short names — used to construct span/grid record keys
CAM_NAMES: list[str] = ["base_0_rgb", "left_wrist_0_rgb"]

# Input observation image keys (same order as CAM_NAMES)
CAM_IMAGE_KEYS: list[str] = [
    "inputs/observation/image",
    "inputs/observation/wrist_image",
]

# Patch grid is fixed for SigLIP So400m/14 (224px / 14px patch = 16 patches per side)
IMAGE_PATCH_GRID: tuple[int, int] = (16, 16)


def get_modality_keys(data_source: str) -> dict[str, str]:
    """Return the {name: key} mapping for the given data_source ('gradcam' or 'attn')."""
    if data_source == "gradcam":
        return GRADCAM_KEYS
    if data_source == "attn":
        return ATTN_KEYS
    raise ValueError(f"Unknown data_source: {data_source!r}. Choose 'gradcam' or 'attn'.")
