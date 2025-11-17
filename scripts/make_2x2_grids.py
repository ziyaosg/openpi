#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image

# ===== DEFAULT CONFIG =====
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

# ===== EPISODES (with success flag) =====
EPISODES: List[Dict] = [
    {"episode_num": 1, "task_id": 0, "task": "put both the alphabet soup and the tomato sauce in the basket", "start_idx": 0, "end_idx": 76, "success": True},
    {"episode_num": 1, "task_id": 1, "task": "put both the cream cheese box and the butter in the basket", "start_idx": 77, "end_idx": 126, "success": True},
    {"episode_num": 1, "task_id": 2, "task": "turn on the stove and put the moka pot on it", "start_idx": 127, "end_idx": 230, "success": False},
    {"episode_num": 1, "task_id": 3, "task": "put the black bowl in the bottom drawer of the cabinet and close it", "start_idx": 231, "end_idx": 334, "success": False},
    {"episode_num": 1, "task_id": 4, "task": "put the white mug on the left plate and put the yellow and white mug on the right plate", "start_idx": 335, "end_idx": 383, "success": True},
    {"episode_num": 1, "task_id": 5, "task": "pick up the book and place it in the back compartment of the caddy", "start_idx": 384, "end_idx": 417, "success": True},
    {"episode_num": 1, "task_id": 6, "task": "put the white mug on the plate and put the chocolate pudding to the right of the plate", "start_idx": 418, "end_idx": 460, "success": True},
    {"episode_num": 1, "task_id": 7, "task": "put both the alphabet soup and the cream cheese box in the basket", "start_idx": 461, "end_idx": 509, "success": True},
    {"episode_num": 1, "task_id": 8, "task": "put both moka pots on the stove", "start_idx": 510, "end_idx": 613, "success": False},
    {"episode_num": 1, "task_id": 9, "task": "put the yellow and white mug in the microwave and close it", "start_idx": 614, "end_idx": 717, "success": False},
]


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
    """Creates subfolder like '3_put_the_black_bowl_in..._failure'."""
    name = f'{ep["task_id"]}_{sanitize_folder_name(ep["task"])}'
    suffix = "_success" if ep.get("success", False) else "_failure"
    outdir = base_out / (name + suffix)
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
    for k in keys:
        if k in data:
            return array_to_rgb_image(data[k])
    for k in data.files:
        if k.startswith("frame/"):
            return array_to_rgb_image(data[k])
    return None


def compose_grid_2x2(tl: Image.Image, tr: Image.Image, bl: Image.Image, br: Image.Image) -> Image.Image:
    r1_h, r2_h = max(tl.height, tr.height), max(bl.height, br.height)
    r1_w, r2_w = tl.width + GAP_X + tr.width, bl.width + GAP_X + br.width
    canvas = Image.new("RGB", (max(r1_w, r2_w), r1_h + GAP_Y + r2_h), BACKGROUND)
    canvas.paste(tl, (0, (r1_h - tl.height)//2))
    canvas.paste(tr, (tl.width + GAP_X, (r1_h - tr.height)//2))
    y2 = r1_h + GAP_Y
    canvas.paste(bl, (0, y2 + (r2_h - bl.height)//2))
    canvas.paste(br, (bl.width + GAP_X, y2 + (r2_h - br.height)//2))
    return canvas


def resize_overlay_to_match(orig: Image.Image, overlay: Image.Image) -> Image.Image:
    return overlay if overlay.size == orig.size else overlay.resize(orig.size, Image.BICUBIC)


def build_grid_for_pair(l_overlay: Path, r_overlay: Path, run_dir: Path) -> Optional[Image.Image]:
    try:
        L_ov, R_ov = Image.open(l_overlay).convert("RGB"), Image.open(r_overlay).convert("RGB")
    except Exception as e:
        print(f"[ERROR] opening overlays {l_overlay.name}: {e}")
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


# ===== MAIN =====
def main():
    ap = argparse.ArgumentParser(description="Create 2x2 overlay/original grids, resize, and tag success/failure.")
    ap.add_argument("--left-overlays",  default=DEFAULT_INPUT_LEFT)
    ap.add_argument("--right-overlays", default=DEFAULT_INPUT_RIGHT)
    ap.add_argument("--run-dir",        default=DEFAULT_RUN_DIR)
    ap.add_argument("--output-root",    default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--start-idx", type=int, default=None)
    ap.add_argument("--end-idx",   type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor for final grid.")
    ap.add_argument("--out-size", type=int, nargs=2, metavar=("W","H"), default=None,
                    help="Resize final output to exact width/height.")
    args = ap.parse_args()

    Ldir, Rdir, run_dir, out_root = map(lambda p: Path(p).expanduser().resolve(),
                                        [args.left_overlays, args.right_overlays, args.run_dir, args.output_root])
    Limgs, Rimgs = list_images_sorted(Ldir), list_images_sorted(Rdir)
    n = min(len(Limgs), len(Rimgs))
    out_root.mkdir(parents=True, exist_ok=True)

    # Global mode (manual start/end)
    if args.start_idx is not None or args.end_idx is not None:
        s = args.start_idx or 0
        e = args.end_idx if args.end_idx is not None else n - 1
        outdir = out_root / "all"
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(s, e + 1):
            grid = build_grid_for_pair(Limgs[i], Rimgs[i], run_dir)
            if grid is None:
                continue
            grid = apply_resize(grid, args)
            name = OUTPUT_NAME_PATTERN.format(idx=i, name=Path(Limgs[i]).stem)
            grid.save(outdir / name)
        print(f"[OK] Wrote global range {s}-{e} to {outdir}")
        return

    # Episode mode
    for ep in EPISODES:
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
            name = OUTPUT_NAME_PATTERN.format(idx=i, name=Path(Limgs[i]).stem)
            grid.save(outdir / name)
            count += 1
        status = "success" if ep.get("success", False) else "failure"
        print(f"[OK] task_id={ep['task_id']} ({status}) → {count} grids → {outdir}")


def apply_resize(img: Image.Image, args: argparse.Namespace) -> Image.Image:
    """Applies scale or explicit size to the final composed grid."""
    if args.out_size:
        W, H = args.out_size
        return img.resize((W, H), Image.BICUBIC)
    elif abs(args.scale - 1.0) > 1e-6:
        w, h = img.size
        return img.resize((int(w * args.scale), int(h * args.scale)), Image.BICUBIC)
    return img


if __name__ == "__main__":
    main()
