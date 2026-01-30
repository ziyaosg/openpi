import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils


from openpi.models.utils.grad_cam import gradcam_from_patch_tokens


logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    def embed_inputs_with_image_meta(
        self, obs: _model.Observation
    ):
        """
        Like embed_inputs(), but also returns:
          - image_slices: dict[name] = (start_idx, end_idx) within the concatenated token sequence
          - image_grids: dict[name] = (Hp, Wp) patch grid for that camera view
        """
        input_mask = []
        ar_mask = []
        token_embeddings = []

        image_slices: dict[str, tuple[int, int]] = {}
        image_grids: dict[str, tuple[int, int]] = {}

        cur = 0

        # embed images
        for name in obs.images:
            # This runs SigLIP ViT
            image_token_embeddings, img_out = self.PaliGemma.img(obs.images[name], train=False)
            # image_token_embeddings: (B, Np, D)
            token_embeddings.append(image_token_embeddings)

            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

            start = cur
            end = cur + image_token_embeddings.shape[1]
            image_slices[name] = (start, end)
            cur = end

            # Use SigLIP internal grid (h,w) from its conv stem / pre_logits_2d tensor.
            # In siglip.py, out["pre_logits_2d"] has shape (B, h, w, D') after encoder.
            # For pool_type="none" + num_classes=paligemma_width, tokens correspond to h*w patches.
            if "pre_logits_2d" in img_out:
                Hp, Wp = int(img_out["pre_logits_2d"].shape[1]), int(img_out["pre_logits_2d"].shape[2])
            elif "stem" in img_out:
                Hp, Wp = int(img_out["stem"].shape[1]), int(img_out["stem"].shape[2])
            else:
                raise Exception("Shape of image patches [Hp, Wp] cannot be determined.")
            #     # Fallback: assume square-ish; best to avoid this branch.
            #     Hp = int(jnp.sqrt(image_token_embeddings.shape[1]))
            #     Wp = image_token_embeddings.shape[1] // Hp
            image_grids[name] = (Hp, Wp)

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        # This runs the Gemma embedding layer to convert text tokens into embeddings
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
            image_slices,   # where each camera’s tokens live
            image_grids,    # how to reshape each camera’s patch tokens into 2D
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # Compute inputs: one big forward pass of prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
        return_image_grad_cam: bool = False,
    ) -> _model.Actions:
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs
        # prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        if return_image_grad_cam:
            prefix_token_embeddings, prefix_mask, prefix_ar_mask, image_slices, image_grids = \
                self.embed_inputs_with_image_meta(observation)
        else:
            prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
            image_slices, image_grids = {}, {}

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # prepare decoding -- final logit decodes the first token
        # last_logit shape: (B, S, V)
        # B: batch size
        # S: number of prefix tokens (all image tokens + prompt tokens)
        # V: vocab size (action tokens live in this vocab)
        # this is literally the model’s probability distribution over the first action token
        last_logit = prefix_logits[:, -1:]  # shape (B, 1, V)
        grad_cam = None
        if return_image_grad_cam and image_slices:
            # Choose the target token for the FIRST decoded token (step 0).
            # Use greedy choice to make attribution deterministic.
            target_token = jnp.argmax(last_logit, axis=-1)  # (B, 1) typically; picks the highest-scoring token v in the vocab

            # Define score function: takes prefix embeddings -> scalar score
            # We compute y = sum over batch of the chosen token logit at the next-token position.
            def score_fn(prefix_emb):
                """
                    - takes prefix embeddings (same shape as prefix_token_embeddings)
                    - runs a forward pass
                    - returns a single scalar score y
                """
                # IMPORTANT: prefix_mask / prefix_ar_mask come from the real observation and are treated constant here.
                attn = make_attn_mask(prefix_mask, prefix_ar_mask)

                # Align exactly the same way as the actual run
                prefix_emb_aligned, mask_aligned, attn_aligned = left_to_right_align(prefix_emb, prefix_mask, attn)

                # prefill_size2 = prefix_emb_aligned.shape[1]   # total sequence length S
                # prefill_len2 = jnp.sum(mask_aligned, axis=-1) # number of valid tokens (not padding)

                attn_aligned = jnp.pad(attn_aligned, ((0, 0), (0, 0), (0, max_decoding_steps)))
                positions2 = jnp.cumsum(mask_aligned, axis=-1) - 1

                logits2, _, _ = self.PaliGemma.llm(
                    embedded_prefix=prefix_emb_aligned, mask=attn_aligned, positions=positions2, decode=True
                )
                ll = logits2[:, -1:]  # (B, 1, V)

                # For each batch element, this selects the scalar logit corresponding to the chosen token id.
                tid = target_token if target_token.ndim == 2 else target_token[:, None] # (B, 1)
                chosen = jnp.take_along_axis(ll, tid[..., None], axis=-1)  # (B, 1, 1)
                return jnp.sum(chosen)
            
            # Backprop step
            # Gradient wrt *prefix embeddings* (same shape as prefix_token_embeddings)
            # dy/dA = f'(A) = score_fn'(prefix_token_embeddings)
            grads = jax.grad(score_fn)(prefix_token_embeddings)  # (B, S, D)

            # Intercept the image patch tokens A
            # Build per-camera heatmaps from the gradient slice corresponding to each image token segment
            grad_cam = {}
            for cam_name, (s0, s1) in image_slices.items():
                patch_tokens = prefix_token_embeddings[:, s0:s1, :]  # This is A; (B, Np, D)
                patch_grads  = grads[:, s0:s1, :]                    # This is dy/dA; (B, Np, D)
                grid_hw = image_grids[cam_name]                      # Reshape patch tokens into a 2D grid Np = Hp*Wp; (Hp, Wp)

                grad_cam[cam_name] = gradcam_from_patch_tokens(
                    patch_tokens, patch_grads, grid_hw, relu=True
                )

        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        # return output_tokens
        if return_image_grad_cam:
            return output_tokens, {"image_grad_cam_patches": grad_cam}
        return output_tokens

