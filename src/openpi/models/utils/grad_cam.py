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


def raw_alpha_from_patch_tokens(
    grads: jnp.ndarray,          # (B, Np, D) = dY/d(patch_tokens)
    grid_hw: Tuple[int, int],    # (Hp, Wp) with Hp*Wp == Np
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Raw alpha heatmap over patch tokens, returned as two variants:
      - sum:  alpha_ij = sum_k(dY/dA^k_ij)   signed, can cancel across channels
      - norm: alpha_ij = ||dY/dA^k_ij||_2     unsigned gradient magnitude

    Returns:
        (sum_alpha, norm_alpha): each (B, Hp, Wp)
    """
    B, Np, D = grads.shape
    Hp, Wp = grid_hw
    sum_alpha  = jnp.sum(grads, axis=-1).reshape((B, Hp, Wp))           # (B, Hp, Wp)
    norm_alpha = jnp.sqrt(jnp.sum(grads ** 2, axis=-1)).reshape((B, Hp, Wp))  # (B, Hp, Wp)
    return sum_alpha, norm_alpha


def raw_alpha_from_tokens(
    grads: jnp.ndarray,  # (B, Nt, D) = dY/d(token_embeddings)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Raw alpha per token, returned as two variants:
      - sum:  alpha_i = sum_k(dY/dA^k_i)   signed, can cancel across channels
      - norm: alpha_i = ||dY/dA^k_i||_2     unsigned gradient magnitude

    Returns:
        (sum_alpha, norm_alpha): each (B, Nt)
    """
    sum_alpha  = jnp.sum(grads, axis=-1)                       # (B, Nt)
    norm_alpha = jnp.sqrt(jnp.sum(grads ** 2, axis=-1))        # (B, Nt)
    return sum_alpha, norm_alpha