from __future__ import annotations

import re


def slugify(s: str, max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def short_label(modality_key: str) -> str:
    mapping = {
        "outputs/debug/attr/image/right_wrist_0_rgb": "right_wrist_rgb",
        "outputs/debug/attr/image/left_wrist_0_rgb": "left_wrist_rgb",
        "outputs/debug/attr/task/scores": "task",
        "outputs/debug/attr/state/scores": "state",
    }
    return mapping.get(modality_key, modality_key)


def norm_code(method: str) -> str:
    return {
        "zscore": "z",
        "robust_zscore": "rz",
        "minmax": "mm",
        "none": "raw",
    }.get(method, slugify(method))


def method_code(method: str) -> str:
    return {
        "average": "avg",
        "sqrt_norm_sum": "sqrt",
        "sum": "sum",
        "max": "max",
        "min": "min",
    }.get(method, slugify(method))


def reduction_code(methods: dict[str, str]) -> str:
    unique = set(methods.values())
    if len(unique) == 1:
        return method_code(next(iter(unique)))
    return "mixed"


def attention_plot_filename(ep, norm_str: str, red_str: str) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_attn_{norm_str}_{red_str}.png"


def correlation_filename(ep, norm_str: str) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_corr_{norm_str}.png"


def slope_correlation_filename(ep, norm_str: str) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_slope_corr_{norm_str}.png"

def changes_in_slope_filename(ep, norm_str: str) -> str:
    return f"t{ep.task_id}_ep{ep.episode_num}_slope_changes_{norm_str}.png"