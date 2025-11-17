import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from openpi.models.pi0_fast import Pi0FASTConfig  # noqa: E402


def load_image(path, size=(224, 224)):
    img = np.asarray(Image.open(path).convert("RGB").resize(size), dtype=np.float32) / 255.0
    return jnp.asarray(img[None, ...])  # [1, H, W, 3]


def main():
    rng = jax.random.PRNGKey(0)

    cfg = Pi0FASTConfig()
    model = cfg.create(rng)

    # Build a fresh fake obs and set token fields via dataclasses.replace (no in-place mutation!)
    obs = replace(
        cfg.fake_obs(),
        tokenized_prompt=jnp.ones((1, cfg.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.zeros((1, cfg.max_token_len), bool).at[:, :16].set(True),
        token_ar_mask=jnp.concatenate(
            [
                jnp.zeros((1, 8), jnp.int32),
                jnp.ones((1, 8), jnp.int32),
                jnp.zeros((1, cfg.max_token_len - 16), jnp.int32),
            ],
            axis=1,
        ),
        token_loss_mask=jnp.zeros((1, cfg.max_token_len), bool).at[:, 1:17].set(True),
    )

    # If you want to use real images, do it via replace too (since Observation is frozen):
    # obs = replace(
    #     obs,
    #     images={
    #         "agentview_rgb": load_image("/path/to/agent.png"),
    #         "robot0_eye_in_hand_rgb": load_image("/path/to/wrist.png"),
    #     },
    #     image_masks={
    #         "agentview_rgb": jnp.array([True]),
    #         "robot0_eye_in_hand_rgb": jnp.array([True]),
    #     },
    # )

    print("Running Grad-CAM test ...")
    cams = model.gradcam_for_token(rng, obs, t_index=0, token_id=1, stop_at_siglip=True)

    # os.makedirs("gradcam_outputs", exist_ok=True)
    # for cam_name, cam in cams.items():
    #     out_path = f"gradcam_outputs/cam_{cam_name}.png"
    #     Image.fromarray((cam * 255).astype(np.uint8)).save(out_path)
    #     print(f"Saved {out_path}")
    
    # If you used real images, grab one to overlay. For the dummy test, just synthesize a gray image.
    # Example with a real image you loaded:
    # base_img = np.array(Image.open("/path/to/agent.png").convert("RGB").resize((224,224)), dtype=np.float32)/255.0

    # Fallback: gray canvas if you don't have a real image
    base_img = np.ones((224, 224, 3), dtype=np.float32) * 0.5

    os.makedirs("gradcam_outputs", exist_ok=True)
    for cam_name, cam in cams.items():
        # raw CAM (small grid)
        Image.fromarray((cam * 255).astype(np.uint8)).save(f"gradcam_outputs/cam_{cam_name}_raw.png")
        # overlay (upsampled + blended)
        save_overlay(cam, base_img, f"gradcam_outputs/cam_{cam_name}_overlay.png", alpha=0.45)
        print(f"Saved gradcam_outputs/cam_{cam_name}_raw.png and _overlay.png")

    print("✅ Grad-CAM smoke test completed.")


def save_overlay(cam, rgb_image_np, out_path, alpha=0.45):
    """
    cam: (Hc, Wc) in [0,1]
    rgb_image_np: (H, W, 3) float32 in [0,1]
    """
    H, W = rgb_image_np.shape[:2]
    # Upsample CAM to image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_up = np.array(cam_img).astype(np.float32) / 255.0  # [0,1]

    # Simple "red" heat overlay (no matplotlib dep)
    heat = np.zeros_like(rgb_image_np)
    heat[..., 0] = cam_up  # red channel = cam
    # Blend: (1 - a)*img + a*heat
    overlay = (1 - alpha) * rgb_image_np + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
