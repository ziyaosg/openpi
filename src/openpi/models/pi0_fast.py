import dataclasses
import logging
from typing import Any, Optional

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
                task_token_len=jax.ShapeDtypeStruct([], jnp.int32),
                task_piece_id=jax.ShapeDtypeStruct([self.max_token_len], jnp.int32),
                task_piece_begin=jax.ShapeDtypeStruct([self.max_token_len], jnp.int32),
                task_piece_end=jax.ShapeDtypeStruct([self.max_token_len], jnp.int32),
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


@dataclasses.dataclass
class Modality:
    name: str # "img/front", "text", "joints"
    type: str # "image", "text", "joints"
    embedding: at.Array # (B, S, D)
    input_mask: at.Array # (B, S)
    ar_mask: at.Array # (B, S)
    meta: Optional[dict[str, Any]] = None # slices, grid


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
    
    def get_patch_grid(self, img_token_embeddings, img_out: dict) -> tuple[int, int]:
        Hp, Wp = (None, None)

        if isinstance(img_out, dict) and "pre_logits_2d" in img_out:
            Hp, Wp = int(img_out["pre_logits_2d"].shape[1]), int(img_out["pre_logits_2d"].shape[2])
        elif isinstance(img_out, dict) and "stem" in img_out:
            Hp, Wp = int(img_out["stem"].shape[1]), int(img_out["stem"].shape[2])
        else:
            Hp, Wp = int(jnp.sqrt(img_token_embeddings.shape[1])), int(img_token_embeddings.shape[1] // Hp)
        
        return Hp, Wp


    def build_image_modality(self, obs: _model.Observation) -> list[Modality]:
        image_modality: list[Modality] = []

        for name in obs.images:
            image_token_embeddings, img_out = self.PaliGemma.img(obs.images[name], train=False)
            input_mask = einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )

            ar_mask = jnp.zeros_like(input_mask, dtype=jnp.int32)  # non-autoregressive
            grid = self.get_patch_grid(
                img_token_embeddings=image_token_embeddings, 
                img_out=img_out
            )

            image_modality.append(Modality(
                name=name,
                type="image",
                embedding=image_token_embeddings,
                input_mask=input_mask,
                ar_mask=ar_mask,
                meta={"grid": grid},
            ))

        return image_modality
    
    def build_full_text_modality(self, obs: _model.Observation) -> Modality:
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"

        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)  # (B, S, D)

        return Modality(
            name="text",
            type="text",
            embedding=tokenized_inputs_embeddings,
            input_mask=obs.tokenized_prompt_mask,
            ar_mask=obs.token_ar_mask,
            meta={}
        )
    
    def build_task_modality(self, obs: _model.Observation, text_modality: Modality) -> Modality:
        assert obs.task_token_len is not None, "task_token_len is required"
        task_token_len = obs.task_token_len

        return Modality(
            name="task",
            type="text",
            embedding=text_modality.embedding[:, :task_token_len, :],
            input_mask=text_modality.input_mask[:, :task_token_len],
            ar_mask=text_modality.ar_mask[:, :task_token_len],
            meta={}
        )

    def build_joint_modality(self, obs: _model.Observation, text_modality: Modality):
        assert obs.task_token_len is not None, "task_token_len is required"
        task_token_len = obs.task_token_len

        return Modality(
            name="joints",
            type="text",
            embedding=text_modality.embedding[:, task_token_len:, :],
            input_mask=text_modality.input_mask[:, task_token_len:],
            ar_mask=text_modality.ar_mask[:, task_token_len:],
            meta={}
        )

    def compute_modality_grads(
        self,
        prefix_emb_unaligned,
        prefix_mask_unaligned,
        prefix_ar_mask_unaligned,
        *,
        max_decoding_steps: int,
        target_token,
    ):
        def score_fn(prefix_emb_unaligned_inner):
            """
            - takes prefix embeddings (same shape as prefix_token_embeddings)
            - runs a forward pass
            - returns a single scalar score y
            """
            
            # IMPORTANT: prefix_mask_unaligned / prefix_ar_mask_unaligned come from the real observation and are treated constant here.
            attn = make_attn_mask(prefix_mask_unaligned, prefix_ar_mask_unaligned)

            # Align exactly the same way as the actual run
            emb_aligned, mask_aligned, attn_aligned = left_to_right_align(
                prefix_emb_unaligned_inner, prefix_mask_unaligned, attn
            )

            # Pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
            attn_aligned = jnp.pad(attn_aligned, ((0, 0), (0, 0), (0, max_decoding_steps)))
            positions = jnp.cumsum(mask_aligned, axis=-1) - 1

            logits, _, _ = self.PaliGemma.llm(
                embedded_prefix=emb_aligned, mask=attn_aligned, positions=positions, decode=True
            )
            ll = logits[:, -1:]  # (B,1,V)

            # For each batch element, select the scalar logit corresponding to the chosen token id.
            tid = target_token if target_token.ndim == 2 else target_token[:, None]  # (B, 1)
            chosen = jnp.take_along_axis(ll, tid[..., None], axis=-1)  # (B, 1, 1)
            return jnp.sum(chosen)

        # Backprop step
        # Gradient wrt *prefix embeddings* (same shape as prefix_token_embeddings)
        # dy/dA = score_fn'(prefix_token_embeddings)
        grads_all = jax.grad(score_fn)(prefix_emb_unaligned)  # (B,S,D)
        return grads_all

    @at.typecheck
    def embed_inputs_with_spans(self, obs: _model.Observation):
        token_embeddings, input_masks, ar_masks = [], [], []
        
        image_modality = self.build_image_modality(obs)
        full_text_modality = self.build_full_text_modality(obs)

        task_modality = self.build_task_modality(obs, full_text_modality) # Name is 'task'
        joint_modality = self.build_joint_modality(obs, full_text_modality) # Name is 'joints'

        modalities = image_modality + [task_modality, joint_modality]

        # modalities = image_modality + [full_text_modality]

        spans = {"image": {}}
        meta = {"image_grids": {}}
        
        cur = 0
        for m in modalities:
            token_embeddings.append(m.embedding)
            input_masks.append(m.input_mask)
            ar_masks.append(m.ar_mask)

            start = cur
            end = start + int(m.embedding.shape[1])
            cur = end

            # Spans
            if m.type == "image":
                spans["image"][m.name] = (start, end)
            else:
                # Possible m.name: "task" and "joints"
                spans[m.name] = (start, end)

            # Image grids
            if m.meta and "grid" in m.meta:
                meta["image_grids"][m.name] = m.meta["grid"]

        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_masks, axis=1),
            jnp.concatenate(ar_masks, axis=1),
            spans,
            meta,
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
        prefix_emb_unaligned, prefix_mask_unaligned, prefix_ar_mask_unaligned, spans, meta = \
            self.embed_inputs_with_spans(observation)
        # prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask_unaligned = make_attn_mask(prefix_mask_unaligned, prefix_ar_mask_unaligned)

        # left to right align all input token sequences
        prefix_emb_aligned, prefix_mask_aligned, prefix_attn_mask_aligned = left_to_right_align(
            prefix_emb_unaligned, prefix_mask_unaligned, prefix_attn_mask_unaligned
        )
        prefill_size = prefix_emb_aligned.shape[1]
        prefill_len = jnp.sum(prefix_mask_aligned, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask_aligned = jnp.pad(prefix_attn_mask_aligned, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask_aligned, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_emb_aligned, mask=prefix_attn_mask_aligned, positions=prefix_positions, decode=True
        )

        # prepare decoding -- final logit decodes the first token
        # last_logit shape: (B, 1, V)
        # B: batch size (typically 1 at inference)
        # V: vocab size (action tokens live in this vocab)
        # this is the model’s logit distribution for the FIRST decoded action token (step 0)
        last_logit = prefix_logits[:, -1:]

        # debug dict containing attributions.
        debug = None

        # choose the target token for the FIRST decoded token (step 0)
        # use greedy choice to make attribution deterministic
        target_token = jnp.argmax(last_logit, axis=-1)  # (B, 1) token id in vocab

        # calculate gradients
        grads = self.compute_modality_grads(
            prefix_emb_unaligned,
            prefix_mask_unaligned,
            prefix_ar_mask_unaligned,
            max_decoding_steps=int(max_decoding_steps),
            target_token=target_token,
        )  # (B,S,D)

        # collect attributions into a single debug payload
        debug = {"attr": {}, "spans": spans, "meta": meta}

        # -----------------------
        # Image Grad-CAM (per view)
        # -----------------------
        image_attr = {}
        for cam_name, (s0, s1) in spans["image"].items():
            patch_tokens = prefix_emb_unaligned[:, s0:s1, :]     # This is A; (B, Np, D)
            patch_grads  = grads[:, s0:s1, :]                    # Slicing for only the image part; this is dy/dA; (B, Np, D)
            grid_hw = meta["image_grids"][cam_name]              # Np = Hp*Wp; (Hp, Wp)

            image_attr[cam_name] = gradcam_from_patch_tokens(
                patch_tokens,
                patch_grads,
                grid_hw,
                relu=True          # keep only positive influence (per classic Grad-CAM)
            )
        debug["attr"]["image"] = image_attr

        # -----------------------
        # Task attribution (per token)
        # -----------------------
        t0, t1 = spans["task"]
        task_token_len = observation.task_token_len

        task_tokens = prefix_emb_unaligned[:, t0:t1, :]   # (B, Nt, D)
        task_grads  = grads[:, t0:t1, :]                  # (B, Nt, D)
        task_scores = jnp.sum(task_tokens * task_grads, axis=-1)  # (B, Nt)

        task_token_ids = observation.tokenized_prompt[:, :task_token_len]

        task_token_mask = None
        if observation.tokenized_prompt_mask is not None:
            task_token_mask = observation.tokenized_prompt_mask[:, :task_token_len]
            task_scores = task_scores * task_token_mask.astype(task_scores.dtype)

        debug["attr"]["task"] = {
            "scores": task_scores,
            "token_ids": task_token_ids,
            "token_mask": task_token_mask,
            "task_piece_id": observation.task_piece_id[:task_token_len],
            "task_piece_begin": observation.task_piece_begin[:task_token_len],
            "task_piece_end": observation.task_piece_end[:task_token_len],
        }

        # -----------------------
        # Joint attribution (per token)
        # -----------------------
        j0, j1 = spans["joints"]

        joint_tokens = prefix_emb_unaligned[:, j0:j1, :]
        joint_grads  = grads[:, j0:j1, :]
        joint_scores = jnp.sum(joint_tokens * joint_grads, axis=-1)

        joint_token_ids = observation.tokenized_prompt[:, task_token_len:]

        joint_token_mask = None
        if observation.tokenized_prompt_mask is not None:
            joint_token_mask = observation.tokenized_prompt_mask[:, task_token_len:]
            joint_scores = joint_scores * joint_token_mask.astype(joint_scores.dtype)

        debug["attr"]["joints"] = {
            "scores": joint_scores,
            "token_ids": joint_token_ids,
            "token_mask": joint_token_mask,
        }
        
        
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
        if debug is not None:
            return output_tokens, debug
        return output_tokens, None
