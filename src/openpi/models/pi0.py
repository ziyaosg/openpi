import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.utils.grad_cam import (
    gradcam_from_patch_tokens,
    raw_alpha_from_patch_tokens,
    raw_alpha_from_tokens,
)
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# SigLIP So400m/14 at 224px → 16×16 patch grid = 256 tokens per image
_N_PATCHES = 256
_PATCH_GRID = (16, 16)

# Layers to record attention from (0-indexed, out of 18 for gemma_2b / gemma_300m).
# Layer 1 gives stable head-selection patterns; layer 16 gives sharper spatial concentration.
_ATTN_LAYERS = [1, 16]


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


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        # Recording-enabled Linen module — same params as PaliGemma.llm, applied directly
        # via _get_llm_vars() when needed. Stored as plain Python attrs so NNX doesn't track them.
        self._paligemma_config = paligemma_config
        self._action_expert_config = action_expert_config
        self._embed_dtype = config.dtype
        self._adarms = config.pi05

    def _get_llm_vars(self) -> dict:
        """Extract current Linen-format variables (params) from the NNX-bridged LLM module.

        The NNX bridge (ToNNX) owns the weights as NNX Variables internally.
        This function converts them back to the Linen {"params": {...}} dict format
        so we can call a recording-enabled Linen module directly via .apply().
        This is necessary because the recording module is a plain Linen object —
        it has no NNX state of its own and must borrow params from the live model.
        """
        from flax.nnx.bridge import variables as bv  # noqa: PLC0415

        nnx_attrs = {name: getattr(self.PaliGemma.llm, name) for name in self.PaliGemma.llm.linen_attributes}
        return bv.nnx_attrs_to_linen_vars(nnx_attrs)

    def _make_record_module(self) -> _gemma.Module:
        """Return a recording-enabled Linen Module with the same architecture as the live LLM.

        This is a fresh stateless Linen object (no params) with record_attn=True.
        When called via .apply(variables, ...), it uses the live params from _get_llm_vars()
        and returns a 3-tuple ([outputs], kv_cache, attn_probs_stacked) instead of the usual
        2-tuple, giving us per-layer attention weights without modifying the normal inference path.
        """
        return _gemma.Module(
            configs=[self._paligemma_config, self._action_expert_config],
            embed_dtype=self._embed_dtype,
            adarms=self._adarms,
            record_attn=True,
        )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _compute_debug_info(
        self,
        observation: _model.Observation,
        prefix_tokens: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        prefix_ar_mask: jnp.ndarray,
        kv_cache,
        noise: jnp.ndarray,
    ) -> dict:
        """Compute gradcam, raw_alpha, and attention-weight debug data for one inference step.

        All gradient-based signals target the action velocity at t=1 (first denoising step,
        pure noise → model sees maximal uncertainty). This is analogous to the FAST model's
        approach of targeting the first decoded action token.

        Layers recorded: _ATTN_LAYERS = [1, 16] out of 18 Gemma layers.
          - Layer 1: head-selection patterns are stable; good for modality attribution.
          - Layer 16: spatial concentration is sharpest; good for image heatmaps.

        Token spans (pi0.5, libero example with 3 cameras):
          - base_0_rgb:        [0   : 256)
          - left_wrist_0_rgb:  [256 : 512)
          - right_wrist_0_rgb: [512 : 768)   (masked, but still present in sequence)
          - task tokens:        [768 : 768+tok_len)
        """
        batch_size = prefix_tokens.shape[0]

        # ── Span bookkeeping ──────────────────────────────────────────────
        spans: dict = {"image": {}}
        meta: dict = {"image_grids": {}}
        cur = 0
        for name in observation.images:
            spans["image"][name] = (cur, cur + _N_PATCHES)
            meta["image_grids"][name] = _PATCH_GRID
            cur += _N_PATCHES

        if observation.tokenized_prompt is not None:
            tok_len = observation.tokenized_prompt.shape[1]
            spans["task"] = (cur, cur + tok_len)

        prefix_len = prefix_tokens.shape[1]  # == cur (+ tok_len if prompt present)

        # ── Gradient-based attribution (GradCAM + raw_alpha) ──────────────
        # Differentiate the sum of action-velocity (at t=1) w.r.t. prefix embeddings.
        # Gradient flows through: prefix_emb → KV cache → suffix attention → v_t.
        timestep_for_grad = jnp.ones(batch_size)

        def score_fn(prefix_emb):
            # Step 1: fill KV cache using the given prefix embeddings.
            # Only PaliGemma expert (index 0) runs; action expert (index 1) is None.
            # This is the same forward pass as the regular prefix fill in sample_actions.
            pfx_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
            pfx_pos = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv = self.PaliGemma.llm([prefix_emb, None], mask=pfx_attn, positions=pfx_pos)

            # Step 2: one action denoising step at t=1 (pure noise input).
            # embed_suffix produces action tokens + adarms_cond (time embedding for adaRMS).
            # Note: embed_suffix does NOT depend on prefix_emb, so JAX correctly traces
            # the gradient path as: prefix_emb → kv → suffix attention → v_t.
            suf_tok, suf_mask, suf_ar_mask, adarms = self.embed_suffix(observation, noise, timestep_for_grad)
            suf_len = suf_tok.shape[1]
            pfx_for_suf = einops.repeat(prefix_mask, "b p -> b s p", s=suf_len)
            suf_attn = make_attn_mask(suf_mask, suf_ar_mask)
            # Full mask: suffix tokens can see all prefix keys (via KV cache) + causal suffix keys.
            full_attn = jnp.concatenate([pfx_for_suf, suf_attn], axis=-1)
            suf_pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1

            # Only action expert runs (index 1), querying prefix K/V from kv cache.
            (_, suf_out), _ = self.PaliGemma.llm(
                [None, suf_tok], mask=full_attn, positions=suf_pos,
                kv_cache=kv, adarms_cond=[None, adarms],
            )
            v_t = self.action_out_proj(suf_out[:, -self.action_horizon :])
            # Scalar target: sum of all predicted action velocities.
            # Gradient w.r.t. prefix_emb tells us how each prefix token influenced the action.
            return jnp.sum(v_t)

        # grads[b, t, d] = how much changing prefix token t's embedding (along dim d)
        # changes the total action velocity. Shape: [B, T_prefix, D].
        grads = jax.grad(score_fn)(prefix_tokens)

        debug: dict = {
            "gradcam": {"image": {}, "task": None},
            "raw_alpha": {
                "summation": {"image": {}, "task": None},
                "norm": {"image": {}, "task": None},
            },
            "tokens": {},
            "attn": {},
            "spans": spans,
        }

        # ── GradCAM for image patches ──────────────────────────────────────
        # GradCAM: weight each patch's embedding by the global-average-pooled gradient
        # (alpha_k = mean over patches of dY/dA_k), then sum over embedding dimension.
        # relu=False: keep negative values; the analysis scripts handle normalization.
        for cam_name, (s0, s1) in spans["image"].items():
            patch_tokens = prefix_tokens[:, s0:s1, :]  # [B, 256, D]
            patch_grads  = grads[:, s0:s1, :]          # [B, 256, D]
            grid = meta["image_grids"][cam_name]        # (16, 16)

            debug["gradcam"]["image"][cam_name] = gradcam_from_patch_tokens(patch_tokens, patch_grads, grid, relu=False)
            # raw_alpha/summation: sum grad over D → signed scalar per patch, can cancel
            # raw_alpha/norm:      L2 norm of grad over D → unsigned magnitude per patch
            ra, rn = raw_alpha_from_patch_tokens(patch_grads, grid)
            debug["raw_alpha"]["summation"]["image"][cam_name] = ra
            debug["raw_alpha"]["norm"]["image"][cam_name] = rn

        # ── GradCAM for task (language) tokens ──────────────────────────────
        # For text tokens we use dot-product(embedding, grad) as the per-token score
        # (the generalization of GradCAM to 1-D token sequences without a spatial grid).
        if "task" in spans:
            t0, t1 = spans["task"]
            task_emb   = prefix_tokens[:, t0:t1, :]  # [B, tok_len, D]
            task_grads = grads[:, t0:t1, :]           # [B, tok_len, D]

            # task_scores[b, t] = sum_d( emb[b,t,d] * grad[b,t,d] ) — element-wise product summed over D
            task_scores          = jnp.sum(task_emb * task_grads, axis=-1)   # [B, tok_len]
            task_ra, task_rn     = raw_alpha_from_tokens(task_grads)

            # Mask out padding tokens so they don't pollute the visualization.
            token_mask = observation.tokenized_prompt_mask
            if token_mask is not None:
                task_scores = task_scores * token_mask.astype(task_scores.dtype)
                task_ra     = task_ra     * token_mask.astype(task_ra.dtype)
                task_rn     = task_rn     * token_mask.astype(task_rn.dtype)

            debug["gradcam"]["task"] = task_scores
            debug["raw_alpha"]["summation"]["task"] = task_ra
            debug["raw_alpha"]["norm"]["task"]      = task_rn
            debug["tokens"]["task"] = {
                "token_ids":  observation.tokenized_prompt,
                "token_mask": token_mask,
            }

            # For pi0.5 with discrete state input (aloha / droid configs), the tokenized_prompt
            # contains both task and state tokens concatenated: "Task: ..., State: {bins};\nAction: ".
            # task_token_len is set by the FAST-style tokenizer when used; for the plain
            # PaligemmaTokenizer used by pi0.5, it is None and we skip the split.
            if observation.task_token_len is not None:
                task_len = observation.task_token_len
                state_len = observation.state_token_len
                state_start = task_len  # state tokens begin right after task tokens
                state_emb   = prefix_tokens[:, t0 + state_start : t0 + state_start + state_len, :]
                state_grads = grads[:, t0 + state_start : t0 + state_start + state_len, :]
                state_scores     = jnp.sum(state_emb * state_grads, axis=-1)
                state_ra, state_rn = raw_alpha_from_tokens(state_grads)
                debug["gradcam"]["state"] = state_scores
                debug["raw_alpha"]["summation"]["state"] = state_ra
                debug["raw_alpha"]["norm"]["state"]      = state_rn

        # ── Raw attention weights and V-cache ─────────────────────────────────
        # We run the exact same first suffix step again, but through the recording-enabled
        # Linen module (record_attn=True) which returns per-layer softmax attention probs.
        # This is a second forward pass — no gradient needed here.
        suf_tok, suf_mask, suf_ar_mask, adarms = self.embed_suffix(
            observation, noise, jnp.ones(batch_size)
        )
        suf_len = suf_tok.shape[1]
        pfx_for_suf = einops.repeat(prefix_mask, "b p -> b s p", s=suf_len)
        suf_attn    = make_attn_mask(suf_mask, suf_ar_mask)
        full_attn   = jnp.concatenate([pfx_for_suf, suf_attn], axis=-1)
        suf_pos     = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suf_mask, axis=-1) - 1

        # Build a recording-enabled Linen module and apply it with the live NNX params.
        # This is a plain Linen .apply() call — completely separate from the NNX-managed
        # inference path, so it doesn't affect model state at all.
        record_module = self._make_record_module()
        variables     = self._get_llm_vars()

        # Returns 3-tuple because record_attn=True:
        #   [None, suf_out]  — hidden states (we discard them here)
        #   kv_new           — updated KV cache (also discarded)
        #   attn_probs       — shape [depth, B, K, G, T_suffix, T_prefix+T_suffix]
        #                      T_suffix = action_horizon (e.g. 10 for libero)
        #                      T_prefix+suffix because the action expert attends to both
        #                      its own suffix tokens AND the prefix (via KV cache concat).
        _outputs, _kv_new, attn_probs = record_module.apply(
            variables,
            [None, suf_tok],   # only action expert (index 1) runs; PaliGemma is None
            suf_pos,
            full_attn,
            [None, adarms],    # adarms_cond: None for PaliGemma, time_emb for action expert
            kv_cache=kv_cache, # prefix KV cache from the regular sample_actions fill
        )

        # Slice to (a) selected layers and (b) prefix-only key positions.
        # attn_probs[l, b, K, G, t_action, :prefix_len] = how much action token t_action
        # attends to prefix token p at layer l, head group (K,G).
        attn_weights = attn_probs[_ATTN_LAYERS, :, :, :, :, :prefix_len]  # [2, B, K, G, T_suf, T_pfx]
        K = attn_weights.shape[2]
        G = attn_weights.shape[3]
        # Flatten (K, G) → H = K*G = 8 heads to match the analysis scripts' expected format.
        attn_weights = attn_weights.reshape(len(_ATTN_LAYERS), batch_size, K * G, suf_len, prefix_len)

        # V vectors from the prefix KV cache filled during the NORMAL inference pass.
        # kv_cache[1] = v_cache shape [depth, B, T_prefix, K, H_dim].
        # These are the value vectors the action tokens weighted during attention,
        # used downstream for value-weighted attention (v_cosine) analysis.
        v_trimmed = kv_cache[1][_ATTN_LAYERS, :, :prefix_len, :, :]  # [2, B, T_pfx, K, H_dim]

        debug["attn"] = {
            "weights": attn_weights,   # [2, B, H=8, T_action, T_prefix]
            "v":       v_trimmed,      # [2, B, T_prefix, K=1, H_dim=256]
            "layers":  jnp.array(_ATTN_LAYERS),
        }

        return debug

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict]:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        debug = self._compute_debug_info(
            observation, prefix_tokens, prefix_mask, prefix_ar_mask, kv_cache, noise
        )
        return x_0, debug
