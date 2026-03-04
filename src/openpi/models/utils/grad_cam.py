from typing import Tuple
import jax.numpy as jnp


def gradcam_from_patch_tokens(
    patch_tokens: jnp.ndarray,   # (B, Np, D)
    grads: jnp.ndarray,          # (B, Np, D) = dY/d(patch_tokens)
    grid_hw: Tuple[int, int],    # (Hp, Wp) with Hp*Wp == Np
    *,
    relu: bool = True
) -> jnp.ndarray:
    """
    Grad-CAM-style heatmap over ViT patch tokens.

    Returns:
        heatmap: (B, Hp, Wp) in [0,1] if normalize=True
    """
    B, Np, D = patch_tokens.shape
    Hp, Wp = grid_hw
    if Hp * Wp != Np:
        raise ValueError(f"grid_hw={grid_hw} does not match Np={Np}")

    # compute channel weights α (average gradients over patches)
    # alpha_k = mean over spatial locations (patches)
    alpha = jnp.mean(grads, axis=1)  # (B, D)

    # weighted sum across D dims for each patch (CAM per patch)
    # CAM(p) = sum_k alpha_k * A(p,k)
    cam = jnp.sum(patch_tokens * alpha[:, None, :], axis=-1)  # (B, Np)

    # ReLU and reshape to (Hp, Wp)
    if relu:
        cam = jnp.maximum(cam, 0.0)
    cam = cam.reshape((B, Hp, Wp))

    return cam