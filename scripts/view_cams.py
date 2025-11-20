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


def _cam_to_2d(cam: np.ndarray) -> np.ndarray:
    """Convert stored CAM to 2D float32 (no normalization)."""
    cam = np.asarray(cam, dtype=np.float32)
    # [H,W,1] → [H,W]
    if cam.ndim == 3 and cam.shape[-1] == 1:
        cam = cam[..., 0]
    # [H,W,3] → average over channels (just in case)
    if cam.ndim == 3 and cam.shape[-1] == 3:
        cam = cam.mean(axis=-1)
    if cam.ndim != 2:
        cam = np.squeeze(cam)
        if cam.ndim != 2:
            raise ValueError(f"CAM is not 2D even after squeeze: shape={cam.shape}")
    return cam


def _normalize_cam(
    cam: np.ndarray,
    *,
    mode: str = "per_image",
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize CAM to [0,1].

    mode="per_image": use per-image min/max (like original behavior).
    mode="global":    use global_min/global_max computed across all steps+cameras.
    """
    cam = _cam_to_2d(cam)

    if mode == "global":
        if global_min is None or global_max is None:
            raise ValueError("global_min/global_max must be provided for mode='global'")
        mn, mx = float(global_min), float(global_max)
    else:  # per_image
        mn, mx = float(cam.min()), float(cam.max())

    denom = mx - mn
    if denom < 1e-8:
        # All zeros or very flat → just return zeros
        return np.zeros_like(cam, dtype=np.float32)

    cam = (cam - mn) / denom
    cam = np.clip(cam, 0.0, 1.0).astype(np.float32)
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
    norm_mode: str,
    global_min: Optional[float],
    global_max: Optional[float],
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

        # ---- NEW: normalization mode aware ----
        raw_cam = np.asarray(data[ck])
        cam = _normalize_cam(
            raw_cam,
            mode=norm_mode,
            global_min=global_min,
            global_max=global_max,
        )

        if cam.shape[:2] != (H, W):
            cam = _resize(cam, (H, W))
            cam = _normalize_cam(
                cam,
                mode=norm_mode,
                global_min=global_min,
                global_max=global_max,
            )
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

    # ---- NEW: normalization mode ----
    p.add_argument(
        "--norm-mode",
        type=str,
        default="per_image",
        choices=["per_image", "global"],
        help=(
            "per_image: normalize each CAM by its own min/max (original behavior). "
            "global: normalize by global min/max across all steps & cameras in this run."
        ),
    )

    args = p.parse_args()

    steps = load_manifest_steps(args.run_dir)
    if not steps:
        print(f"[ERROR] No steps found under {args.run_dir}")
        return
    steps = steps[args.start: args.start + args.max_steps]

    only_cams = [s.strip() for s in args.only_cams.split(",")] if args.only_cams else None
    camera_order = [s.strip() for s in args.camera_order.split(",") if s.strip()]

    if args.out_dir is None:
        print("[INFO] No --out-dir provided; using non-interactive export is recommended. Set --out-dir to save PNGs.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Exporting {len(steps)} step(s) → {args.out_dir}")
    print(f"Normalization mode: {args.norm_mode}")

    # ---- NEW: compute global min/max over all steps+cameras (per run/task) ----
    global_min = None
    global_max = None
    if args.norm_mode == "global":
        gmin = np.inf
        gmax = -np.inf
        count = 0

        for npz_path in steps:
            try:
                data = np.load(npz_path, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] Failed to load {os.path.basename(npz_path)} for global stats: {e}")
                continue

            keys = list(data.keys())
            cam_keys = [k for k in keys if k.startswith("cam/")]

            if only_cams:
                cam_keys = [ck for ck in cam_keys if ck.split("/", 1)[-1] in only_cams]

            for ck in cam_keys:
                cam_raw = _cam_to_2d(np.asarray(data[ck]))
                if cam_raw.size == 0:
                    continue
                local_min = float(cam_raw.min())
                local_max = float(cam_raw.max())
                if local_min < gmin:
                    gmin = local_min
                if local_max > gmax:
                    gmax = local_max
                count += 1

        if count == 0 or not np.isfinite(gmin) or not np.isfinite(gmax):
            print("[WARN] Could not compute global CAM stats; falling back to per-image normalization.")
            global_min = None
            global_max = None
            args.norm_mode = "per_image"
        else:
            global_min = gmin
            global_max = gmax
            print(f"[INFO] Global CAM range across this run: min={global_min:.6f}, max={global_max:.6f}")

    # ---- render overlays ----
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
                norm_mode=args.norm_mode,
                global_min=global_min,
                global_max=global_max,
            )
        except Exception as e:
            print(f"[WARN] Failed on {os.path.basename(npz_path)}: {e}")


if __name__ == "__main__":
    main()
