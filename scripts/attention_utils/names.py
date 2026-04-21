from __future__ import annotations

import re


def slugify(s: str, max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len].rstrip("_") if len(s) > max_len else s


def short_label(modality_key: str) -> str:
    mapping = {
        # GradCAM attribution keys
        "outputs/debug/gradcam/image/base_0_rgb": "base_rgb",
        "outputs/debug/gradcam/image/left_wrist_0_rgb": "left_wrist_rgb",
        "outputs/debug/gradcam/task": "task",
        "outputs/debug/gradcam/state": "state",
        # Attention span keys
        "outputs/debug/spans/image/base_0_rgb": "base_rgb",
        "outputs/debug/spans/image/left_wrist_0_rgb": "left_wrist_rgb",
        "outputs/debug/spans/task": "task",
        "outputs/debug/spans/state": "state",
    }
    return mapping.get(modality_key, modality_key)


def norm_code(method: str) -> str:
    return {
        "zscore": "z",
        "robust_zscore": "rz",
        "minmax": "mm",
        "none": "raw",
    }.get(method, slugify(method))


def _method_code(method: str) -> str:
    return {
        "average": "avg",
        "median": "med",
        "sqrt_norm_sum": "sqrt",
        "sum": "sum",
        "max": "max",
        "min": "min",
    }.get(method, slugify(method))


def reduction_code(methods: dict[str, str]) -> str:
    unique = set(methods.values())
    if len(unique) == 1:
        return _method_code(next(iter(unique)))
    return "mixed"


def attention_plot_filename(ep, norm_str: str, red_str: str) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_attn_{norm_str}_{red_str}.png"


def patch_dist_filename(ep) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_patch_dist.png"
