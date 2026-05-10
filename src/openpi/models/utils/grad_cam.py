from __future__ import annotations

from typing import Tuple
import jax.numpy as jnp


def gradcam_from_patch_tokens(
    patch_tokens: jnp.ndarray,   # (B, Np, D)
    grads: jnp.ndarray,          # (B, Np, D) = dY/d(patch_tokens)
    grid_hw: Tuple[int, int],    # (Hp, Wp) with Hp*Wp == Np
    *,
    relu: bool = True
) -> jnp.ndarray:
    """Grad-CAM heatmap over ViT patch tokens. Returns (B, Hp, Wp)."""
    B, Np, D = patch_tokens.shape
    Hp, Wp = grid_hw
    if Hp * Wp != Np:
        raise ValueError(f"grid_hw={grid_hw} does not match Np={Np}")

    alpha = jnp.mean(grads, axis=1)  # (B, D) — global average pooling of gradients
    cam = jnp.sum(patch_tokens * alpha[:, None, :], axis=-1)  # (B, Np)
    if relu:
        cam = jnp.maximum(cam, 0.0)
    return cam.reshape((B, Hp, Wp))


def raw_alpha_from_patch_tokens(
    grads: jnp.ndarray,          # (B, Np, D) = dY/d(patch_tokens)
    grid_hw: Tuple[int, int],    # (Hp, Wp) with Hp*Wp == Np
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Raw alpha heatmap over patch tokens.

    Returns:
        (sum_alpha, norm_alpha): each (B, Hp, Wp)
        - sum_alpha:  sum_k(dY/dA^k_ij) — signed, can cancel across channels
        - norm_alpha: ||dY/dA^k_ij||_2  — unsigned gradient magnitude
    """
    B, Np, D = grads.shape
    Hp, Wp = grid_hw
    sum_alpha  = jnp.sum(grads, axis=-1).reshape((B, Hp, Wp))
    norm_alpha = jnp.sqrt(jnp.sum(grads ** 2, axis=-1)).reshape((B, Hp, Wp))
    return sum_alpha, norm_alpha


def raw_alpha_from_tokens(
    grads: jnp.ndarray,  # (B, Nt, D) = dY/d(token_embeddings)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Raw alpha per token.

    Returns:
        (sum_alpha, norm_alpha): each (B, Nt)
    """
    sum_alpha  = jnp.sum(grads, axis=-1)
    norm_alpha = jnp.sqrt(jnp.sum(grads ** 2, axis=-1))
    return sum_alpha, norm_alpha
