#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Optional OpenCV for nicer resize; fallback to PIL
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image
    _HAS_CV2 = False


def _resize(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    """Resize to (H, W). Accepts (H,W),(H,W,1),(H,W,3),(H,W,4)."""
    H, W = size_hw
    if _HAS_CV2:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    # PIL fallback
    arr = img
    mode = None
    if img.ndim == 2:
        mode = "L"
    elif img.ndim == 3 and img.shape[-1] == 1:
        arr = img[..., 0]
        mode = "L"
    elif img.ndim == 3 and img.shape[-1] == 3:
        mode = "RGB"
    elif img.ndim == 3 and img.shape[-1] == 4:
        mode = "RGBA"
    else:
        arr = np.squeeze(img)
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            mode = "RGB" if arr.shape[-1] == 3 else "RGBA"
        else:
            raise ValueError(f"Unsupported shape for PIL resize: {img.shape}")
    im = Image.fromarray(
        arr.astype(np.uint8) if arr.dtype != np.float32
        else (np.clip(arr, 0, 1) * 255).astype(np.uint8),
        mode=mode
    )
    im = im.resize((W, H), Image.BILINEAR)
    out = np.array(im)
    if mode == "L" and img.ndim == 3:
        out = out[..., None]
    return out


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim == 3 and cam.shape[-1] == 1:
        cam = cam[..., 0]
    if cam.ndim == 3 and cam.shape[-1] == 3:
        cam = cam.mean(axis=-1)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    fr = np.asarray(frame)
    if fr.dtype != np.uint8:
        fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8) if fr.max() <= 1.0 else np.clip(fr, 0, 255).astype(np.uint8)
    if fr.ndim == 2:
        fr = np.repeat(fr[..., None], 3, axis=-1)
    if fr.ndim == 3 and fr.shape[-1] == 1:
        fr = np.repeat(fr, 3, axis=-1)
    if fr.ndim == 3 and fr.shape[-1] == 4:
        fr = fr[..., :3]
    return fr


def load_manifest_steps(run_dir: str) -> List[str]:
    manifest = os.path.join(run_dir, "manifest.json")
    raw_dir = os.path.join(run_dir, "raw")
    paths = sorted(glob.glob(os.path.join(raw_dir, "step_*.npz")))
    if os.path.exists(manifest):
        try:
            with open(manifest, "r") as f:
                j = json.load(f)
            listed = [os.path.join(run_dir, p["path"]) for p in j.get("steps", []) if "path" in p]
            listed = [p for p in listed if os.path.exists(p)]
            if listed:
                return listed
        except Exception:
            pass
    return paths


def overlay_one(
    npz_path: str,
    *,
    alpha: float,
    cmap_name: str,
    max_cams: int,
    out_dir: Optional[str],
    match_per_cam: bool,
    split_by_camera: bool,
    camera_order: List[str],
    only_cams: Optional[List[str]],
    out_size: Optional[Tuple[int, int]],
    scale: float,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    frame_keys = [k for k in keys if k.startswith("frame/")]
    cam_keys = [k for k in keys if k.startswith("cam/")]

    if not frame_keys and not cam_keys:
        print(f"[WARN] {os.path.basename(npz_path)} has neither frames nor CAMs.")
        return

    if only_cams:
        cam_keys = [ck for ck in cam_keys if ck.split("/", 1)[-1] in only_cams]

    # choose a base frame key if we are NOT matching per cam
    base_key = None
    if not match_per_cam and frame_keys:
        for c in camera_order:
            cand = f"frame/{c}"
            if cand in frame_keys:
                base_key = cand
                break
        if base_key is None:
            base_key = frame_keys[0]

    # prepare base image (if needed)
    base_img = None
    if base_key is not None:
        base_img = _ensure_rgb(np.squeeze(np.asarray(data[base_key])))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # show/save raw base optionally when no CAMs or for debugging (we skip window display here)
    if not cam_keys and base_img is not None and out_dir:
        png = os.path.join(out_dir, os.path.basename(npz_path).replace(".npz", f"__{base_key.replace('/','_')}.png"))
        matplotlib.image.imsave(png, base_img)
        print("Saved", png)
        return

    cmap = plt.get_cmap(cmap_name)
    shown = 0

    # helper: resize overlay to out_size or scale
    def _maybe_resize_overlay(img_float01: np.ndarray) -> np.ndarray:
        if out_size is not None:
            W, H = out_size
            arr = (img_float01 * 255).astype(np.uint8)
            arr = _resize(arr, (H, W))
            return arr.astype(np.float32) / 255.0
        if abs(scale - 1.0) > 1e-6:
            H0, W0 = img_float01.shape[:2]
            arr = (img_float01 * 255).astype(np.uint8)
            arr = _resize(arr, (int(H0 * scale), int(W0 * scale)))
            return arr.astype(np.float32) / 255.0
        return img_float01

    for ck in cam_keys:
        if shown >= max_cams:
            break

        cam_name = ck.split("/", 1)[-1]  # e.g., base_0_rgb
        # choose frame key for this cam
        this_frame_key = None
        if match_per_cam:
            candidate = f"frame/{cam_name}"
            if candidate in frame_keys:
                this_frame_key = candidate
        if this_frame_key is None:
            this_frame_key = base_key  # fallback to global base

        if this_frame_key is None:
            # no frame at all → skip overlay, but could still save heatmap alone if desired
            continue

        base = _ensure_rgb(np.squeeze(np.asarray(data[this_frame_key])))
        base_f = base.astype(np.float32) / 255.0
        H, W = base_f.shape[:2]

        cam = _normalize_cam(np.asarray(data[ck]))
        if cam.shape[:2] != (H, W):
            cam = _resize(cam, (H, W))
            cam = _normalize_cam(cam)
        heat = cmap(cam)[..., :3]

        overlay = np.clip((1.0 - alpha) * base_f + alpha * heat, 0, 1.0)
        overlay = _maybe_resize_overlay(overlay)

        # decide output path
        if out_dir:
            if split_by_camera:
                sub = os.path.join(out_dir, cam_name)
                os.makedirs(sub, exist_ok=True)
                out_png = os.path.join(sub, os.path.basename(npz_path).replace(".npz", ".png"))
            else:
                out_png = os.path.join(
                    out_dir,
                    os.path.basename(npz_path).replace(".npz", f"__{ck.replace('/','_')}.png")
                )
            matplotlib.image.imsave(out_png, overlay)
        shown += 1


def main():
    p = argparse.ArgumentParser(description="View/export Grad-CAM overlays from CAMLogger runs.")
    p.add_argument("--run-dir", required=True, help="Run directory with manifest.json and raw/")
    p.add_argument("--alpha", type=float, default=0.45, help="CAM overlay alpha (0..1)")
    p.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap")
    p.add_argument("--max-steps", type=int, default=512, help="Max steps to process")
    p.add_argument("--max-cams", type=int, default=4, help="Max CAMs per step")
    p.add_argument("--out-dir", type=str, default=None, help="If set, save PNGs here (no GUI windows)")
    p.add_argument("--start", type=int, default=0, help="Start index into sorted steps")

    # New controls
    p.add_argument("--split-by-camera", action="store_true",
                   help="Save into subfolders per camera name (e.g., base_0_rgb/, left_wrist_0_rgb/)")
    p.add_argument("--match-per-cam", action="store_true",
                   help="Overlay each CAM on its own frame if available (frame/<cam_name>)")
    p.add_argument("--camera-order", type=str, default="base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb",
                   help="Fallback order to pick a base frame when not matching per-cam")
    p.add_argument("--only-cams", type=str, default=None,
                   help="Comma list to restrict CAM names (e.g. base_0_rgb,left_wrist_0_rgb)")

    # Resizing options
    p.add_argument("--out-size", type=int, nargs=2, metavar=("W","H"),
                   default=None, help="Resize saved overlays to W H")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor for saved overlays (ignored if --out-size is set)")

    args = p.parse_args()

    steps = load_manifest_steps(args.run_dir)
    if not steps:
        print(f"[ERROR] No steps found under {args.run_dir}")
        return
    steps = steps[args.start: args.start + args.max_steps]

    only_cams = [s.strip() for s in args.only_cams.split(",")] if args.only_cams else None
    camera_order = [s.strip() for s in args.camera_order.split(",") if s.strip()]

    # If out_dir not set, we won’t pop up windows (headless export is typical)
    if args.out_dir is None:
        print("[INFO] No --out-dir provided; using non-interactive export is recommended. Set --out-dir to save PNGs.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Exporting {len(steps)} step(s) → {args.out_dir}")

    for npz_path in steps:
        try:
            overlay_one(
                npz_path,
                alpha=args.alpha,
                cmap_name=args.cmap,
                max_cams=args.max_cams,
                out_dir=args.out_dir,
                match_per_cam=args.match_per_cam,
                split_by_camera=args.split_by_camera,
                camera_order=camera_order,
                only_cams=only_cams,
                out_size=tuple(args.out_size) if args.out_size else None,
                scale=args.scale,
            )
        except Exception as e:
            print(f"[WARN] Failed on {os.path.basename(npz_path)}: {e}")


if __name__ == "__main__":
    main()
