#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

# ===== DEFAULT CONFIG (you can ignore these if you always pass via CLI) =====
DEFAULT_INPUT_LEFT  = "/data/ziyao/cam_logs/29999/cam_overlays/base_0_rgb"
DEFAULT_INPUT_RIGHT = "/data/ziyao/cam_logs/29999/cam_overlays/left_wrist_0_rgb"
DEFAULT_RUN_DIR     = "/data/ziyao/cam_logs/29999"
DEFAULT_OUTPUT_ROOT = "/data/ziyao/cam_logs/29999/cam_overlays/tasks"

OUTPUT_NAME_PATTERN = "pair_{idx:05d}_{name}.png"
GAP_X, GAP_Y = 16, 16
BACKGROUND = (255, 255, 255)

FRAME_KEYS_FOR_LEFT  = ["frame/base_0_rgb", "frame/agentview_rgb", "frame/base_1_rgb"]
FRAME_KEYS_FOR_RIGHT = ["frame/left_wrist_0_rgb", "frame/wrist_0_rgb", "frame/right_wrist_0_rgb"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STEP_RE = re.compile(r"(step_\d+)", re.IGNORECASE)


# ===== UTILITIES =====
def sanitize_folder_name(name: str, maxlen: int = 140) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9._-]+", "", name)
    return name[:maxlen]


def list_images_sorted(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    imgs.sort(key=lambda p: p.name)
    return imgs


def ensure_task_outdir(base_out: Path, ep: Dict) -> Path:
    """
    Create subfolder like:
    0_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_success
    """
    task_id = ep.get("task_id", "task")
    task    = ep.get("task", f"task_{task_id}")

    base_name = f"{task_id}_{sanitize_folder_name(task)}"
    suffix    = "_success" if ep.get("success", False) else "_failure"
    outdir    = base_out / (base_name + suffix)

    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def array_to_rgb_image(arr: np.ndarray) -> Image.Image:
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    if a.dtype != np.uint8:
        if a.max() <= 1.0:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGB")


def infer_npz_from_overlay(p: Path, run_dir: Path) -> Optional[Path]:
    """
    Given an overlay filename containing 'step_XXXX', infer the corresponding
    raw npz under <run_dir>/raw/step_XXXX.npz.
    """
    m = STEP_RE.search(p.stem)
    if not m:
        return None
    npz = run_dir / "raw" / (m.group(1) + ".npz")
    return npz if npz.exists() else None


def load_first_available_frame(npz: Path, keys: List[str]) -> Optional[Image.Image]:
    try:
        data = np.load(npz, allow_pickle=True)
    except Exception:
        return None
    # Prefer the given list of keys
    for k in keys:
        if k in data:
            return array_to_rgb_image(data[k])
    # Fallback: any frame/*
    for k in data.files:
        if k.startswith("frame/"):
            return array_to_rgb_image(data[k])
    return None


def compose_grid_2x2(tl: Image.Image, tr: Image.Image,
                     bl: Image.Image, br: Image.Image) -> Image.Image:
    r1_h, r2_h = max(tl.height, tr.height), max(bl.height, br.height)
    r1_w, r2_w = tl.width + GAP_X + tr.width, bl.width + GAP_X + br.width
    canvas = Image.new("RGB", (max(r1_w, r2_w), r1_h + GAP_Y + r2_h), BACKGROUND)

    # top row
    canvas.paste(tl, (0, (r1_h - tl.height)//2))
    canvas.paste(tr, (tl.width + GAP_X, (r1_h - tr.height)//2))

    # bottom row
    y2 = r1_h + GAP_Y
    canvas.paste(bl, (0, y2 + (r2_h - bl.height)//2))
    canvas.paste(br, (bl.width + GAP_X, y2 + (r2_h - br.height)//2))
    return canvas


def resize_overlay_to_match(orig: Image.Image, overlay: Image.Image) -> Image.Image:
    return overlay if overlay.size == orig.size else overlay.resize(orig.size, Image.BICUBIC)


def apply_resize(img: Image.Image, args: argparse.Namespace) -> Image.Image:
    """Applies scale or explicit size to the final composed grid."""
    if args.out_size:
        W, H = args.out_size
        return img.resize((W, H), Image.BICUBIC)
    elif abs(args.scale - 1.0) > 1e-6:
        w, h = img.size
        return img.resize((int(w * args.scale), int(h * args.scale)), Image.BICUBIC)
    return img


def build_grid_for_pair(l_overlay: Path, r_overlay: Path, run_dir: Path) -> Optional[Image.Image]:
    try:
        L_ov = Image.open(l_overlay).convert("RGB")
        R_ov = Image.open(r_overlay).convert("RGB")
    except Exception as e:
        print(f"[ERROR] opening overlays {l_overlay.name} / {r_overlay.name}: {e}")
        return None

    npz = infer_npz_from_overlay(l_overlay, run_dir) or infer_npz_from_overlay(r_overlay, run_dir)
    if npz is None or not npz.exists():
        return None

    L_orig = load_first_available_frame(npz, FRAME_KEYS_FOR_LEFT)
    R_orig = load_first_available_frame(npz, FRAME_KEYS_FOR_RIGHT)
    if L_orig is None or R_orig is None:
        return None

    # Resize overlays to match original frames exactly
    L_ov = resize_overlay_to_match(L_orig, L_ov)
    R_ov = resize_overlay_to_match(R_orig, R_ov)

    return compose_grid_2x2(L_ov, R_ov, L_orig, R_orig)


def load_episodes(json_path: str) -> List[Dict]:
    """
    Load episode/task splits from JSON (REQUIRED).

    JSON can be either:
      - a list of dicts
      - or a dict with key 'episodes' or 'tasks' containing a list.

    Each episode must contain 'start_idx' and 'end_idx'.
    """
    if json_path is None:
        raise ValueError(
            "You must provide --episodes-json. No fallback episodes are allowed."
        )

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Episodes JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        if "episodes" in data and isinstance(data["episodes"], list):
            episodes = data["episodes"]
        elif "tasks" in data and isinstance(data["tasks"], list):
            episodes = data["tasks"]
        else:
            raise ValueError(
                f"JSON {json_path} must contain an 'episodes' or 'tasks' list"
            )
    else:
        raise ValueError(f"Invalid episode JSON structure in {json_path}")

    for i, ep in enumerate(episodes):
        if not isinstance(ep, dict):
            raise ValueError(f"Episode {i} is not a dict: {ep}")
        for k in ("start_idx", "end_idx"):
            if k not in ep:
                raise ValueError(f"Episode {i} missing required key '{k}'")

    print(f"[INFO] Loaded {len(episodes)} episodes from {json_path}")
    return episodes


# ===== MAIN =====
def main():
    ap = argparse.ArgumentParser(
        description="Create 2x2 (overlay/original) grids for two cameras, grouped by tasks."
    )
    ap.add_argument("--left-overlays",  default=DEFAULT_INPUT_LEFT,
                    help="Folder with overlays for the left camera (e.g., base_0_rgb).")
    ap.add_argument("--right-overlays", default=DEFAULT_INPUT_RIGHT,
                    help="Folder with overlays for the right camera (e.g., left_wrist_0_rgb).")
    ap.add_argument("--run-dir",        default=DEFAULT_RUN_DIR,
                    help="Pi0FAST run dir containing raw/step_XXXXX.npz.")
    ap.add_argument("--output-root",    default=DEFAULT_OUTPUT_ROOT,
                    help="Root folder where task subfolders will be created.")
    ap.add_argument(
        "--episodes-json",
        required=True,
        help="REQUIRED: JSON file defining episode/task splits (must include start_idx/end_idx).",
    )
    ap.add_argument("--start-idx", type=int, default=None,
                    help="Optional global mode: start index (overrides episodes).")
    ap.add_argument("--end-idx",   type=int, default=None,
                    help="Optional global mode: end index (inclusive; overrides episodes).")
    ap.add_argument(
        "--scale",
        type=float,
        default=2.0,  # <--- make 2x2 grids bigger by default
        help="Scale factor for final grid images (default: 2.0).",
    )
    ap.add_argument("--out-size", type=int, nargs=2, metavar=("W","H"), default=None,
                    help="Resize final grids to exact width/height (W H).")
    args = ap.parse_args()

    Ldir     = Path(args.left_overlays).expanduser().resolve()
    Rdir     = Path(args.right_overlays).expanduser().resolve()
    run_dir  = Path(args.run_dir).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()

    Limgs = list_images_sorted(Ldir)
    Rimgs = list_images_sorted(Rdir)
    n = min(len(Limgs), len(Rimgs))
    out_root.mkdir(parents=True, exist_ok=True)

    if n == 0:
        print(f"[ERROR] No overlay images found in {Ldir} or {Rdir}")
        return

    # ----- Global mode: manual index range (ignores episodes) -----
    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx or 0
        e = args.end_idx if args.end_idx is not None else n - 1
        s = max(0, s)
        e = min(e, n - 1)
        outdir = out_root / "all"
        outdir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i in range(s, e + 1):
            grid = build_grid_for_pair(Limgs[i], Rimgs[i], run_dir)
            if grid is None:
                continue
            grid = apply_resize(grid, args)
            name = OUTPUT_NAME_PATTERN.format(idx=i, name=Limgs[i].stem)
            grid.save(outdir / name)
            count += 1
        print(f"[OK] Global range {s}-{e} → {count} grids → {outdir}")
        return

    # ----- Episode mode: use episodes JSON (required) -----
    episodes = load_episodes(args.episodes_json)

    for ep in episodes:
        s, e = ep["start_idx"], ep["end_idx"]
        if s >= n:
            continue
        e = min(e, n - 1)
        outdir = ensure_task_outdir(out_root, ep)
        count = 0
        for i in range(s, e + 1):
            grid = build_grid_for_pair(Limgs[i], Rimgs[i], run_dir)
            if grid is None:
                continue
            grid = apply_resize(grid, args)
            name = OUTPUT_NAME_PATTERN.format(idx=i, name=Limgs[i].stem)
            grid.save(outdir / name)
            count += 1
        status = "success" if ep.get("success", False) else "failure"
        print(
            f"[OK] task_id={ep.get('task_id')} ({status}) "
            f"→ {count} grids → {outdir}"
        )


if __name__ == "__main__":
    main()
