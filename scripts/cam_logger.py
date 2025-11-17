# scripts/cam_logger.py
import json
import os
import time
from typing import Dict, Optional
import numpy as np
from threading import Lock

class CAMLogger:
    """
    Minimal logger that saves one .npz per step in <run_dir>/raw/
    and keeps a small manifest.json.
    Supports both CAM + frame logging or frames-only fallback.
    """
    _manifest_lock = Lock()  # prevent concurrent manifest corruption

    def __init__(self, run_dir: str, save_frames: bool, frame_size=(224, 224)):
        self.run_dir = os.path.abspath(run_dir)
        self.raw_dir = os.path.join(self.run_dir, "raw")
        self.save_frames = bool(save_frames)
        self.frame_size = tuple(frame_size)

        os.makedirs(self.raw_dir, exist_ok=True)

        # Initialize manifest file if missing
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w") as f:
                json.dump({"created_at": time.time(), "steps": []}, f)


    def log_step(
        self,
        step_like: int,
        *,
        cams: Dict[str, np.ndarray],
        token_id: int,
        frames: Optional[Dict[str, np.ndarray]] = None,
        timestamp: float | None = None,
    ) -> None:
        timestamp = float(timestamp if timestamp is not None else time.time())

        # Ensure dirs still exist (just in case they were cleaned mid-run)
        os.makedirs(self.raw_dir, exist_ok=True)

        # Sanitize cams
        clean_cams = {}
        for k, v in cams.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # e.g., HxWx1
                arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(-1)
            clean_cams[k] = arr

        save_dict = {
            "timestamp": np.array(timestamp, dtype=np.float64),
            "token_id": np.array(token_id, dtype=np.int32),
            "cams_keys": np.array(list(clean_cams.keys()), dtype=object),
        }
        for k, v in clean_cams.items():
            save_dict[f"cam/{k}"] = v

        if self.save_frames and frames:
            for k, v in frames.items():
                arr = np.asarray(v)
                if arr.ndim == 2:
                    arr = arr[..., None]
                save_dict[f"frame/{k}"] = arr

        npz_path = os.path.join(self.raw_dir, f"step_{int(step_like)}.npz")
        tmp_path = os.path.join(self.raw_dir, f"step_{int(step_like)}.tmp.npz")  # <- ends with .npz

        # Write atomically: write to tmp, then replace
        with open(tmp_path, "wb") as f:
            np.savez(f, **save_dict)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, npz_path)

        # Update manifest
        man = {"created_at": time.time(), "steps": []}
        try:
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path, "r") as f:
                    man = json.load(f)
        except Exception:
            pass

        man["steps"].append({
            "step": int(step_like),
            "path": os.path.relpath(npz_path, self.run_dir),
            "timestamp": timestamp,
            "num_cams": len(clean_cams),
            "num_frames": len(frames) if frames else 0,
        })
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(man, f)
