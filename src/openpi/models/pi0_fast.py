import dataclasses
import logging
from typing import Any
import numpy as np

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
        logger.info(f"[images present] {list(obs.images.keys())}")
        # # embed images
        # for name in obs.images:
        #     image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

        #     token_embeddings.append(image_token_embeddings)
        #     input_mask.append(
        #         einops.repeat(
        #             obs.image_masks[name],
        #             "b -> b s",
        #             s=image_token_embeddings.shape[1],
        #         )
        #     )
        #     # image tokens attend to each other --> AR mask = 0
        #     ar_mask.append(0 * input_mask[-1])

        # embed images
        for name in obs.images:
            image_token_embeddings, aux = self.PaliGemma.img(obs.images[name], train=False)

            # ==== DEBUG STATS ====
            try:
                # pre-projection (ViT features before Dense head)
                if aux is not None and "pre_logits" in aux:
                    pp = aux["pre_logits"]
                    pm = jnp.mean(pp).item()
                    ps = jnp.std(pp).item()
                    logger.info(f"[pre-proj] {name}: mean={pm:.4f} std={ps:.4f}")

                # post-projection (after Dense to VLM width; what the LLM actually sees)
                m = jnp.mean(image_token_embeddings).item()
                s = jnp.std(image_token_embeddings).item()
                logger.info(f"[post-proj] {name}: mean={m:.4f} std={s:.4f}")
            except Exception as e:
                logger.warning(f"norm-stats error for {name}: {e}")

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
    ) -> _model.Actions:
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
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
        last_logit = prefix_logits[:, -1:]
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
        return output_tokens


    # -----------------------------
    # Grad-CAM helpers (inside Pi0FAST)
    # -----------------------------
    def _ordered_image_keys(self, obs: _model.Observation):
        """Stable camera order; auto-adapts to LIBERO keys if present."""
        # Map common aliases -> canonical names used by LIBERO inputs
        libero_aliases = {
            "agentview_rgb": "base_0_rgb",
            "robot0_eye_in_hand_rgb": "left_wrist_0_rgb",
            "wrist_0_rgb": "left_wrist_0_rgb",      # accept our old alias too
            "base_1_rgb": "base_1_rgb",
        }

        imgs = {}
        for k, v in obs.images.items():
            normk = libero_aliases.get(k, k)
            imgs[normk] = v

        # Preferred order; include both wrist names but only ones actually present
        pref = ["base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        return [k for k in pref if k in imgs], imgs


    def _get_image_tokens_and_spans(self, observation: _model.Observation):
        """
        Returns:
        tokens_concat: [B, L_tot, D] concatenated visual tokens from SigLIP
        spans: list of (name, start, end, side) for each camera block
        imgs_norm: normalized image dict used for encoding
        """
        key_order, imgs_norm = self._ordered_image_keys(observation)
        tokens_list, spans = [], []
        start = 0
        for name in key_order:
            # cam_tokens, _ = self.PaliGemma.img(imgs_norm[name], train=False)  # [B, L, D] (pool_type="none")
            cam_tokens, aux = self.PaliGemma.img(imgs_norm[name], train=False)  # [B, L, D] (pool_type="none")
            # ==== DEBUG STATS ====
            try:
                if aux is not None and "pre_logits" in aux:
                    pp = aux["pre_logits"]
                    pm = jnp.mean(pp).item(); ps = jnp.std(pp).item()
                    logger.info(f"[pre-proj] {name}: mean={pm:.4f} std={ps:.4f}")
                m = jnp.mean(cam_tokens).item(); s = jnp.std(cam_tokens).item()
                logger.info(f"[post-proj] {name}: mean={m:.4f} std={s:.4f}")
            except Exception as e:
                logger.warning(f"norm-stats error (spans) for {name}: {e}")


            L = cam_tokens.shape[1]
            side = int(jnp.sqrt(L))
            assert side * side == L, f"Non-square token grid for {name}: L={L}"
            tokens_list.append(cam_tokens)
            spans.append((name, start, start + L, side))
            start += L
        tokens_concat = jnp.concatenate(tokens_list, axis=1) if tokens_list else None
        return tokens_concat, spans, imgs_norm


    def gradcam_for_token(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        t_index: int,
        token_id: int,
        generated_prefix_ids: at.Int | None = None,  # optional teacher-forced generated tokens [B, K]
        stop_at_siglip: bool = True,
    ):
        """
        Grad-CAM over SigLIP patch tokens for ONE output token.

        Args:
        t_index: index within the predicted slice (0 = first predicted token)
        token_id: vocab id to explain at that position
        generated_prefix_ids: optional [B, K] tokens generated AFTER the prompt; if provided,
                                the explanation is conditioned on these K previous generated tokens.
        stop_at_siglip: if True, take grad w.r.t. SigLIP outputs only (faster).

        Returns:
        dict[camera_name] -> (H, W) float32 heatmap in [0,1]
        """
        # Preprocess like normal forward
        observation = _model.preprocess_observation(
            rng, observation, train=False, image_keys=list(observation.images.keys())
        )

        # Visual tokens (A) and spans
        img_tokens_concat, spans, _ = self._get_image_tokens_and_spans(observation)  # [B, L_tot, D]
        if img_tokens_concat is None:
            return {}
        if stop_at_siglip:
            img_tokens_concat = jax.lax.stop_gradient(img_tokens_concat)

        # Text embeddings
        assert observation.tokenized_prompt is not None
        assert observation.tokenized_prompt_mask is not None
        assert observation.token_ar_mask is not None
        txt_ids  = observation.tokenized_prompt
        txt_mask = observation.tokenized_prompt_mask
        txt_ar   = observation.token_ar_mask
        txt_emb  = self.PaliGemma.llm(txt_ids, embed_only=True)  # [B, S_txt, D]

        # Optional generated prefix (teacher-forced)
        gen_len = 0
        if generated_prefix_ids is not None:
            gen_emb  = self.PaliGemma.llm(generated_prefix_ids, embed_only=True)  # [B, K, D]
            gen_len  = generated_prefix_ids.shape[1]
            gen_mask = jnp.ones((gen_emb.shape[0], gen_len), dtype=bool)
            gen_ar   = jnp.ones((gen_emb.shape[0], gen_len), dtype=jnp.int32)     # pure causal
        else:
            gen_emb = None
            gen_mask = None
            gen_ar = None

        # Build masks for [IMG || PROMPT || (GEN?)]
        input_mask_blocks, ar_blocks = [], []
        for name, s, e, _ in spans:
            present = observation.image_masks.get(name, jnp.ones((img_tokens_concat.shape[0],), bool))
            input_mask_blocks.append(jnp.repeat(present[:, None], e - s, axis=1))
            ar_blocks.append(jnp.zeros_like(input_mask_blocks[-1], dtype=jnp.int32))
        input_mask_blocks.append(txt_mask)
        ar_blocks.append(txt_ar)
        if gen_emb is not None:
            input_mask_blocks.append(gen_mask)
            ar_blocks.append(gen_ar)

        input_mask = jnp.concatenate(input_mask_blocks, axis=1)  # [B, S_all]
        ar_mask    = jnp.concatenate(ar_blocks,        axis=1)   # [B, S_all]
        attn_mask  = make_attn_mask(input_mask, ar_mask)         # [B, S_all, S_all]

        # Define score as the logit at target position using one non-decoding forward
        def score_fn(img_tokens_in):
            parts = [img_tokens_in, txt_emb]
            if gen_emb is not None:
                parts.append(gen_emb)
            prefix = jnp.concatenate(parts, axis=1)                           # [B, S_all, D]
            pre_logits, _, _ = self.PaliGemma.llm(
                embedded_prefix=prefix[:, :-1],
                mask=attn_mask[:, :-1, :-1],
                return_prelogits=True,
            )
            target_len = (txt_ids.shape[1] - 1) + gen_len                     # predict next for prompt & gen prefix
            logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -target_len:])
            return logits[0, t_index, token_id]                               # scalar

        # # Grad w.r.t. visual tokens
        # dscore_dA = jax.grad(score_fn)(img_tokens_concat)   # [B, L_tot, D]
        # A         = img_tokens_concat                       # [B, L_tot, D]

        # # Per-camera CAMs
        # cams = {}
        # for name, s, e, side in spans:
        #     A_cam  = A[0, s:e, :]           # [L_cam, D]
        #     dA_cam = dscore_dA[0, s:e, :]   # [L_cam, D]
        #     w      = jnp.mean(dA_cam, axis=0)                           # α_j
        #     cam_f  = jnp.maximum(0.0, jnp.einsum("ld,d->l", A_cam, w))  # ReLU(weighted sum)
        #     cam    = cam_f.reshape(side, side)
        #     cam    = (cam - cam.min()) / (cam.max() + 1e-8)
        #     cams[name] = np.array(cam, dtype=np.float32)
        # return cams


        # Grad w.r.t. visual tokens
        dscore_dA = jax.grad(score_fn)(img_tokens_concat)   # [B, L_tot, D]
        A         = img_tokens_concat                       # [B, L_tot, D]

        # --------- Per-camera raw CAMs (no normalization) ---------
        # Only keep the cameras we care about for visualization
        include_cams = {"base_0_rgb", "left_wrist_0_rgb"}

        cams = {}   # name -> raw (H, W) CAM, non-normalized, ReLU'd

        for name, s, e, side in spans:
            if name not in include_cams:
                continue

            A_cam  = A[0, s:e, :]           # [L_cam, D]
            dA_cam = dscore_dA[0, s:e, :]   # [L_cam, D]

            # Standard Grad-CAM weighting
            w      = jnp.mean(dA_cam, axis=0)                           # α_j
            cam_f  = jnp.maximum(0.0, jnp.einsum("ld,d->l", A_cam, w))  # [L_cam], ReLU

            # Reshape to spatial map; DO NOT normalize here
            cam    = cam_f.reshape(side, side)

            cams[name] = np.asarray(cam, dtype=np.float32)

        return cams



    def gradcam_for_action_token(self, rng, observation, generated_ids, step_from_start: int = 0):
        """
        Explain the token the model actually generated at step `step_from_start` after the prompt.
        Args:
        generated_ids: [B, T] tokens produced by your decode loop
        step_from_start: 0 = first generated token, 1 = second, etc.
        """
        tok = int(generated_ids[0, step_from_start])
        return self.gradcam_for_token(
            rng,
            observation,
            t_index=step_from_start,
            token_id=tok,
            generated_prefix_ids=generated_ids[:, :step_from_start],  # teacher-force prev gens
            stop_at_siglip=True,
        )
