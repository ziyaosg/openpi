#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

# =========== CONFIG (edit these) ===========

# Exactly two input folders, each containing 718 images ordered by filename
INPUT_FOLDERS: List[str] = [
    "/data/ziyao/cam_logs/29999/cam_overlays/base_0_rgb",
    "/data/ziyao/cam_logs/29999/cam_overlays/left_wrist_0_rgb",
]

# Root for output combined images; subfolders per task go here.
# If None, will create a sibling "combined" folder next to the first INPUT_FOLDERS[0].
OUTPUT_ROOT: str | None = "/data/ziyao/cam_logs/29999/cam_overlays/tasks" # e.g., "/path/to/output_root" or None

# Output filename pattern. "{name}" becomes left image basename (without extension), "{idx}" is zero-based index.
OUTPUT_NAME_PATTERN = "pair_{idx:05d}_{name}.png"

# Visual options for side-by-side composition
GAP_PX = 16                     # empty space between the two images
BACKGROUND = (255, 255, 255)    # RGB background color (e.g., white)
RESIZE_TO_MATCH_HEIGHT = True   # if True, resizes right image to match left image height (keeps aspect)

# Your episode/task list (zero-based indices, end_idx inclusive)
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

# Accepted image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =========== END CONFIG ===========


def sanitize_folder_name(name: str, maxlen: int = 140) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9._-]+", "", name)
    name = re.sub(r"_+", "_", name)
    return name[:maxlen] if len(name) > maxlen else name


def list_images_sorted(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: p.name)  # sort by filename lexicographically
    return files


def ensure_task_outdir(base_out: Path, ep: Dict) -> Path:
    task_name = f'{ep["task_id"]}_{sanitize_folder_name(ep["task"])}'
    outdir = base_out / task_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def compose_side_by_side(img_left: Image.Image, img_right: Image.Image) -> Image.Image:
    # Optionally resize right to match left height
    if RESIZE_TO_MATCH_HEIGHT and img_right.height != img_left.height:
        ratio = img_right.width / img_right.height
        new_w = int(round(ratio * img_left.height))
        img_right = img_right.resize((new_w, img_left.height), Image.BICUBIC)

    w = img_left.width + GAP_PX + img_right.width
    h = max(img_left.height, img_right.height)
    canvas = Image.new("RGB", (w, h), BACKGROUND)

    # center vertically
    y_left = (h - img_left.height) // 2
    y_right = (h - img_right.height) // 2

    canvas.paste(img_left, (0, y_left))
    canvas.paste(img_right, (img_left.width + GAP_PX, y_right))
    return canvas


def main():
    if len(INPUT_FOLDERS) != 2:
        raise ValueError("Please provide exactly two input folders in INPUT_FOLDERS.")

    left_dir = Path(INPUT_FOLDERS[0]).expanduser().resolve()
    right_dir = Path(INPUT_FOLDERS[1]).expanduser().resolve()
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError("One of the INPUT_FOLDERS is not a valid directory.")

    left_imgs = list_images_sorted(left_dir)
    right_imgs = list_images_sorted(right_dir)
    n_left, n_right = len(left_imgs), len(right_imgs)

    if n_left == 0 or n_right == 0:
        raise RuntimeError("No images found in one of the input folders.")
    if n_left != n_right:
        print(f"[WARN] Folder sizes differ: {n_left} vs {n_right}. Will use min length.")
    n = min(n_left, n_right)

    # Determine output root
    if OUTPUT_ROOT:
        base_out = Path(OUTPUT_ROOT).expanduser().resolve()
    else:
        base_out = left_dir.parent / (left_dir.name + "_combined")
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output root: {base_out}")

    # For each episode, process index ranges (inclusive)
    for ep in EPISODES:
        s, e = ep["start_idx"], ep["end_idx"]
        if s < 0 or e < s:
            print(f"[WARN] Bad range for task_id {ep['task_id']}: {s}..{e} (skipped)")
            continue
        if s >= n:
            print(f"[WARN] start_idx {s} >= available pairs {n} for task_id {ep['task_id']} (skipped)")
            continue
        e = min(e, n - 1)

        outdir = ensure_task_outdir(base_out, ep)
        count = 0

        for idx in range(s, e + 1):
            left_path = left_imgs[idx]
            right_path = right_imgs[idx]

            try:
                with Image.open(left_path) as L, Image.open(right_path) as R:
                    L = L.convert("RGB")
                    R = R.convert("RGB")
                    combined = compose_side_by_side(L, R)

                # Build a clean name using the left image's stem
                name_stem = Path(left_path).stem
                out_name = OUTPUT_NAME_PATTERN.format(idx=idx, name=name_stem)
                combined.save(outdir / out_name)
                count += 1
            except Exception as ex:
                print(f"[ERROR] Failed at idx {idx} ({left_path.name}, {right_path.name}): {ex}")

        print(
            f"[OK] task_id={ep['task_id']} "
            f"({ep['task'][:60]}{'...' if len(ep['task'])>60 else ''}) "
            f"→ wrote {count} combined images to {outdir}"
        )

    print("[DONE] All tasks processed.")


if __name__ == "__main__":
    main()
