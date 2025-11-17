# scripts/serve_policy.py
import dataclasses
import enum
import logging
import socket
import time
from typing import Any, Optional

import numpy as np
import tyro
from dataclasses import replace

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ----- Grad-CAM logging -----
    cam_run_dir: str = "runs/cam_logs"
    cam_every_k: int = 0          # 0 disables
    cam_save_frames: bool = False # also save RGB frames if present


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint("pi0_aloha", "gs://openpi-assets/checkpoints/pi0_base"),
    EnvMode.ALOHA_SIM: Checkpoint("pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint("pi0_fast_droid", "gs://openpi-assets/checkpoints/pi0_fast_droid"),
    EnvMode.LIBERO: Checkpoint("pi0_fast_libero", "gs://openpi-assets/checkpoints/pi0_fast_libero"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# ---------------------------
# CAM logger wrapper
# ---------------------------
class _MaybeCAMLogger:
    def __init__(self, run_dir: str, save_frames: bool, every_k: int):
        self.logger = None
        self.every_k = every_k
        self._counter = 0
        if every_k > 0:
            try:
                from scripts.cam_logger import CAMLogger
                self.logger = CAMLogger(run_dir=run_dir, save_frames=save_frames, frame_size=(224, 224))
                logging.info("[CAM] Logger enabled at %s (every_k=%d, save_frames=%s)",
                             run_dir, every_k, save_frames)
            except Exception as e:
                logging.warning("Could not initialize CAMLogger: %s", e)

    def enabled(self) -> bool:
        return self.logger is not None and self.every_k > 0

    def maybe_log(self, *, step_like: int, cams: dict[str, np.ndarray], token_id: int,
                  frames: Optional[dict[str, np.ndarray]] = None):
        if not self.enabled():
            return
        if (self._counter % self.every_k) != 0:
            self._counter += 1
            return
        try:
            logging.info("[CAM] log_step: step=%s token_id=%s cams=%s",
                         step_like, token_id, list(cams.keys()))
            self.logger.log_step(step_like, cams=cams, token_id=token_id, frames=frames, timestamp=time.time())
        except Exception as e:
            logging.error("[CAM] log_step failed: %s", e, exc_info=True)
        finally:
            self._counter += 1

    def close(self):
        if self.logger is not None:
            try:
                self.logger.close()
            except Exception:
                pass


class _CAMLoggingPolicy:
    """
    Wraps a policy and, after each .infer(...), computes Grad-CAM best-effort and saves it.
    Works even when the policy doesn't return token IDs (falls back to token_id=1).
    """
    def __init__(self, inner: _policy.Policy, args: Args):
        self._inner = inner
        self._args = args

        # Build run subdir name from checkpoint dir if present
        exp_name = "default"
        try:
            pdir = ""
            if isinstance(args.policy, Checkpoint):
                pdir = args.policy.dir
            elif hasattr(inner, "metadata") and isinstance(inner.metadata, dict):
                pdir = inner.metadata.get("checkpoint_dir", "")
            if pdir:
                exp_name = pdir.strip("/").split("/")[-1] or "default"
        except Exception:
            pass

        run_dir = f"{args.cam_run_dir.rstrip('/')}/{exp_name}"
        self._cam = _MaybeCAMLogger(run_dir=run_dir, save_frames=args.cam_save_frames, every_k=args.cam_every_k)
        self._step = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


    def infer(self, *args, **kwargs):
        """
        1) Run the real policy.
        2) If CAM disabled -> return.
        3) Reuse the policy's own input transform to build an Observation from the SAME request.
        4) Compute Grad-CAM (best effort) and log frames+CAMs.
        """
        out = self._inner.infer(*args, **kwargs)

        # If CAM disabled, return immediately
        if not self._cam.enabled():
            return out

        # --- Extract the raw request dict the same way the server passed it in ---
        req = None
        if args and isinstance(args[0], dict):
            req = args[0]
        elif "data" in kwargs and isinstance(kwargs["data"], dict):
            req = kwargs["data"]

        # Quick debug
        try:
            logging.info("[CAM] request top-level keys: %s",
                        list(req.keys()) if isinstance(req, dict) else type(req))
            logging.info("[CAM] out keys: %s", list(out.keys()) if isinstance(out, dict) else type(out))
        except Exception:
            pass

        # --- Extract frames early (for logging/overlay), from flattened LIBERO keys ---
        images_dict = None
        if isinstance(req, dict):
            flat = {}
            if "observation/image" in req:
                flat["base_0_rgb"] = req["observation/image"]
            if "observation/wrist_image" in req:
                flat["left_wrist_0_rgb"] = req["observation/wrist_image"]
            # Only include right wrist if present
            if "observation/right_wrist_image" in req:
                flat["right_wrist_0_rgb"] = req["observation/right_wrist_image"]
            images_dict = flat if flat else None

        frames = None
        frame_names = set()
        try:
            if self._args.cam_save_frames and images_dict is not None:
                frames = {}
                for k, v in images_dict.items():
                    arr = np.asarray(v)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    frames[k] = arr
                    frame_names.add(k)
        except Exception as e:
            logging.debug("[CAM] Could not parse frames: %s", e)

        # --- Pull generated token ids from response if present (many policies won't have these) ---
        generated_ids = None
        try:
            if isinstance(out, dict):
                if "generated_ids" in out:
                    generated_ids = np.asarray(out["generated_ids"])
                elif "tokens" in out:
                    generated_ids = np.asarray(out["tokens"])
        except Exception as e:
            logging.debug("[CAM] Could not parse generated_ids: %s", e)

        # --- Locate the underlying Policy and its model/config/transform ---
        policy_obj = self._inner
        if hasattr(policy_obj, "_policy"):  # PolicyRecorder wrapper
            policy_obj = getattr(policy_obj, "_policy")

        model = getattr(policy_obj, "model", None) or getattr(policy_obj, "_model", None)
        input_transform = getattr(policy_obj, "_input_transform", None)

        # If we can't find the model or the transform, we can't build an Observation the same way.
        if model is None or input_transform is None or not isinstance(req, dict):
            # still log frames only so you get something
            if frames is not None:
                self._cam.maybe_log(step_like=int(time.time() * 1000), cams={}, token_id=-1, frames=frames)
            return out

        # --- Rebuild the EXACT inputs the policy uses, then an Observation ---
        cams = {}
        tok = -1
        step_idx_val = int(time.time() * 1000)  # unique-ish step id for logging

        try:
            import jax
            import jax.numpy as jnp
            from openpi.models import model as _model

            # Copy the raw request (the policy transform may modify in-place)
            raw_inputs = jax.tree.map(lambda x: x, req)

            # Apply the same input preprocessing the policy uses (LiberoInputs, etc.)
            transformed = input_transform(raw_inputs)

            # Batch & to jax.Array similar to Policy.infer
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)

            # Build Observation via the same helper used by Policy.infer
            observation = _model.Observation.from_dict(batched)

            # --- Choose a token to explain ---
            step_idx = 0
            if generated_ids is not None:
                try:
                    tok = int(np.asarray(generated_ids)[0, step_idx])
                except Exception:
                    tok = 1
            else:
                tok = 1  # safe fallback vocab id

            # --- Run Grad-CAM ---
            if hasattr(model, "gradcam_for_action_token") and generated_ids is not None:
                cams = model.gradcam_for_action_token(
                    rng=None,
                    observation=observation,
                    generated_ids=np.asarray(generated_ids),
                    step_from_start=step_idx,
                )
            elif hasattr(model, "gradcam_for_token"):
                cams = model.gradcam_for_token(
                    rng=None,
                    observation=observation,
                    t_index=step_idx,
                    token_id=tok,
                    stop_at_siglip=True,
                )
            else:
                cams = {}

            # Convert JAX arrays to numpy and keep only valid heatmaps
            if isinstance(cams, dict) and cams:
                clean = {}
                for k, v in cams.items():
                    try:
                        arr = np.array(v)
                        if arr.ndim >= 2:
                            clean[k] = arr
                    except Exception:
                        pass
                cams = clean
            else:
                cams = {}

        except Exception as e:
            logging.debug("[CAM] Grad-CAM compute failed: %s", e, exc_info=True)
            cams = {}

        # Filter CAMs to only those cameras we actually have frames for
        if isinstance(cams, dict) and cams:
            cams = {k: v for k, v in cams.items() if k in frame_names}
            # Ensure plain numpy
            for k, v in list(cams.items()):
                if not isinstance(v, np.ndarray):
                    try:
                        cams[k] = np.array(v)
                    except Exception:
                        cams.pop(k, None)

        # Finally, only call logger if there is *something* to save
        try:
            if cams:
                logging.info("[CAM] log_step: step=%d token_id=%s cams=%s",
                            step_idx_val, str(tok), list(cams.keys()))
                self._cam.maybe_log(step_like=step_idx_val, cams=cams, token_id=tok, frames=frames)
                logging.info("[CAM] logged %d cams; frames=%s", len(cams), "yes" if frames else "no")
            elif frames is not None:
                self._cam.maybe_log(step_like=step_idx_val, cams={}, token_id=-1, frames=frames)
                logging.info("[CAM] frames-only snapshot")
        except Exception as e:
            logging.error("[CAM] log_step failed: %s", e, exc_info=True)

        return out


    def close(self):
        self._cam.close()
        if hasattr(self._inner, "close"):
            self._inner.close()


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata if hasattr(policy, "metadata") else {}

    # Optional behavior recording (writes to policy_records/)
    if args.record:
        from openpi.policies.policy import PolicyRecorder  # same module defines it
        policy = PolicyRecorder(policy, "policy_records")

    # Wrap with CAM logger (no-op if cam_every_k == 0)
    if args.cam_every_k > 0:
        policy = _CAMLoggingPolicy(policy, args)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    try:
        server.serve_forever()
    finally:
        if hasattr(policy, "close"):
            try:
                policy.close()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
