"""Megakernel-backed talker decoder for Qwen3-TTS.

Replaces the HF transformer backbone for the talker (the expensive 20-layer
part) with the fused CUDA megakernel.  Everything else — embeddings, sub-talker
(code_predictor), and vocoder — still runs via HuggingFace.

Key methods
-----------
step(token_id)            Standard embed-lookup decode step (argmax).
step_embed(inputs_embeds) Decode with a precomputed bfloat16 embedding vector.
                          Used for TTS mixed-embedding decode steps.
inject_kv_cache(past_kv, prefill_len)
                          Copy a HF DynamicCache into the megakernel KV cache
                          after prefill.
hidden_state              Property: normalised hidden state from the last step
                          (float32 [1, 1, hidden_size]) — fed to sub-talker.
"""

import math
from typing import Optional, Tuple

import torch

from .talker_weights import (
    CP_MAX_SEQ_LEN,
    TALKER_HEAD_DIM,
    TALKER_HIDDEN_SIZE,
    TALKER_INTERMEDIATE_SIZE,
    TALKER_MAX_SEQ_LEN,
    TALKER_NUM_KV_HEADS,
    TALKER_NUM_LAYERS,
    TALKER_NUM_Q_HEADS,
    pack_layer_weights,
)

# Derived constants
_Q_SIZE = TALKER_NUM_Q_HEADS * TALKER_HEAD_DIM   # 2048
_KV_SIZE = TALKER_NUM_KV_HEADS * TALKER_HEAD_DIM  # 256
_ATTN_SCALE = 1.0 / math.sqrt(TALKER_HEAD_DIM)


class TalkerDecoder:
    """Stateful megakernel talker decoder.

    Parameters
    ----------
    weights : dict
        From :func:`~talker_weights.load_talker_weights`.
    max_seq_len : int
        KV cache capacity.  Defaults to TALKER_MAX_SEQ_LEN (4096).
    """

    def __init__(self, weights: dict, max_seq_len: int = TALKER_MAX_SEQ_LEN):
        # Build TTS-tuned kernel (compiled once, cached).
        import torch as _torch
        from qwen_megakernel.build import get_tts_extension

        get_tts_extension()  # compile/load to register ops in torch.ops
        self._decode_op = _torch.ops.qwen_tts_megakernel_C.decode

        self._weights = weights
        self._max_seq_len = max_seq_len
        num_layers = weights["num_layers"]
        num_kv = weights["num_kv_heads"]
        head_dim = weights["head_dim"]

        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = pack_layer_weights(
            weights["layer_weights"], num_layers
        )

        self._num_layers = num_layers
        self._position = 0
        self._attn_scale = _ATTN_SCALE

        # KV cache — [layers, kv_heads, max_seq, head_dim]
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._k_cache = torch.zeros(
            num_layers, num_kv, max_seq_len, head_dim, **bf16
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers
        f32 = dict(dtype=torch.float32, device="cuda")
        kv_size = num_kv * head_dim
        q_size = TALKER_NUM_Q_HEADS * head_dim

        self._hidden = torch.empty(TALKER_HIDDEN_SIZE, **bf16)
        self._act = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        self._res = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        self._q = torch.empty(q_size, **f32)
        self._k = torch.empty(kv_size, **f32)
        self._v = torch.empty(kv_size, **f32)
        self._attn_out = torch.empty(q_size, **f32)
        self._mlp_inter = torch.empty(TALKER_INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        # LM-head reduction buffers — sized for VOCAB=3072, LM_NUM_BLOCKS=16
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    # ------------------------------------------------------------------
    # Core decode
    # ------------------------------------------------------------------

    def _call_kernel(self, token_id: int) -> int:
        """Run one megakernel decode step. Returns next codec token id."""
        self._decode_op(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            self._num_layers,
            self._position,
            self._max_seq_len,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def step(self, token_id: int) -> int:
        """Standard token-id decode step (embed lookup inside kernel)."""
        return self._call_kernel(token_id)

    def step_embed(
        self, inputs_embeds: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """Decode with a precomputed embedding (TTS mixed-embed mode).

        Parameters
        ----------
        inputs_embeds : torch.Tensor
            Shape [hidden_size] or [1, hidden_size] or [1, 1, hidden_size],
            dtype bfloat16.

        Returns
        -------
        codec_token_id : int
            Argmax over vocab.
        hidden_state : torch.Tensor
            Normalised hidden state, shape [1, 1, hidden_size], float32 on GPU.
            Feed to the sub-talker as `past_hidden`.
        """
        # Copy embedding into hidden buffer; kernel reads it when token_id = -1.
        emb = inputs_embeds.view(-1).to(torch.bfloat16)
        self._hidden.copy_(emb)

        # Sentinel -1 tells kernel to use hidden_buffer as layer-0 input.
        token_id = self._call_kernel(-1)

        # _norm_out is populated by the kernel after the final RMSNorm.
        # Clone before next step overwrites it.
        hidden = self._norm_out.clone().to(torch.float32).unsqueeze(0).unsqueeze(0)
        return token_id, hidden

    # ------------------------------------------------------------------
    # KV cache injection (after HF prefill)
    # ------------------------------------------------------------------

    def inject_kv_cache(self, past_key_values, prefill_len: int) -> None:
        """Copy a HuggingFace DynamicCache into the megakernel KV cache.

        Call this after running the HF model for prefill, then set
        ``self._position = prefill_len`` before the decode loop.

        The HF model stores K post-QK-norm+RoPE and V directly, which is
        exactly the same layout the megakernel uses.
        """
        for layer in range(self._num_layers):
            if hasattr(past_key_values, "key_cache"):
                k = past_key_values.key_cache[layer]   # [1, num_kv, T, head_dim]
                v = past_key_values.value_cache[layer]
            else:
                k, v = past_key_values[layer]           # legacy tuple format

            seq = k.shape[-2]
            self._k_cache[layer, :, :seq, :] = k[0].to(torch.bfloat16)
            self._v_cache[layer, :, :seq, :] = v[0].to(torch.bfloat16)

        self._position = prefill_len

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def hidden_state(self) -> torch.Tensor:
        """Normalised hidden state from the last step: [1, 1, hidden_size] f32."""
        return self._norm_out.unsqueeze(0).unsqueeze(0)

    def reset(self) -> None:
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def soft_reset(self) -> None:
        """Reset position only — KV values are overwritten by prefill_embeds."""
        self._position = 0

    def prefill_embeds(self, embeds: torch.Tensor) -> None:
        """Fill the KV cache for the prefill sequence using the megakernel.

        Runs the decode kernel once per token at positions 0..T-1.
        Replaces the HF backbone forward pass + inject_kv_cache.

        Parameters
        ----------
        embeds : [T, hidden_size] bfloat16 on CUDA
        """
        seq = embeds.view(-1, TALKER_HIDDEN_SIZE).to(torch.bfloat16)
        T = seq.shape[0]
        for t in range(T):
            self._hidden.copy_(seq[t])
            self._call_kernel(-1)

    @property
    def position(self) -> int:
        return self._position


class CodePredictorKernel:
    """Megakernel-accelerated code predictor (5-layer sub-talker).

    Reuses the same compiled TTS CUDA kernel as TalkerDecoder but with
    num_layers=5. The kernel LM-head output is discarded; per-group
    lm_head[g] weights from HF are applied to _norm_out in Python via
    F.linear — no HuggingFace DynamicCache overhead per frame.

    Call predict() once per TTS codec frame to generate all
    (num_code_groups - 1) sub-tokens.
    """

    def __init__(self, cp_weights: dict):
        import torch as _torch
        from qwen_megakernel.build import get_tts_extension

        get_tts_extension()
        self._decode_op = _torch.ops.qwen_tts_megakernel_C.decode

        num_layers = cp_weights["num_layers"]
        num_kv = cp_weights["num_kv_heads"]
        head_dim = cp_weights["head_dim"]

        self._num_layers = num_layers
        self._attn_scale = 1.0 / math.sqrt(head_dim)
        self._position = 0

        self._embed_weight = cp_weights["dummy_embed"]
        self._final_norm_weight = cp_weights["final_norm_weight"]
        self._lm_head_weight = cp_weights["dummy_lm_head"]
        self._cos_table = cp_weights["cos_table"]
        self._sin_table = cp_weights["sin_table"]
        self._layer_weights_packed = pack_layer_weights(cp_weights["layer_weights"], num_layers)

        self._codec_embedding_weights = cp_weights["codec_embedding_weights"]
        self._lm_head_weights = cp_weights["lm_head_weights"]
        self._num_code_groups = cp_weights["num_code_groups"]
        self._proj_weight = cp_weights["proj_weight"]
        self._proj_bias = cp_weights["proj_bias"]

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._k_cache = torch.zeros(num_layers, num_kv, CP_MAX_SEQ_LEN, head_dim, **bf16)
        self._v_cache = torch.zeros_like(self._k_cache)

        f32 = dict(dtype=torch.float32, device="cuda")
        q_size = TALKER_NUM_Q_HEADS * head_dim
        kv_size = num_kv * head_dim

        self._hidden = torch.empty(TALKER_HIDDEN_SIZE, **bf16)
        self._act = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        self._res = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        self._q = torch.empty(q_size, **f32)
        self._k = torch.empty(kv_size, **f32)
        self._v = torch.empty(kv_size, **f32)
        self._attn_out = torch.empty(q_size, **f32)
        self._mlp_inter = torch.empty(TALKER_INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(TALKER_HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def _step(self, embed: torch.Tensor) -> None:
        """Project (if needed), copy embed to hidden buffer, run one kernel step."""
        e = embed.view(-1).to(torch.bfloat16)
        if self._proj_weight is not None:
            e = torch.nn.functional.linear(
                e.unsqueeze(0).to(self._proj_weight.dtype),
                self._proj_weight,
                self._proj_bias,
            ).view(-1).to(torch.bfloat16)
        self._hidden.copy_(e)
        self._decode_op(
            self._out_token,
            -1,  # sentinel: kernel reads from hidden_buffer instead of embed table
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            self._num_layers,
            self._position,
            CP_MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1

    def predict(
        self,
        sub_input: torch.Tensor,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> torch.Tensor:
        """Generate all sub-tokens for one codec frame.

        Parameters
        ----------
        sub_input : [1, 2, hidden_size]
            Row 0: talker's normalised hidden state (from mk_decoder._norm_out).
            Row 1: backbone codec_embedding of the first codec token (group 0).

        Returns
        -------
        torch.Tensor : [1, num_code_groups - 1] int64
        """
        self._position = 0
        n = self._num_code_groups - 1  # 31 for 32-group model

        # Prefill — 2 tokens; KV written at positions 0 and 1.
        self._step(sub_input[0, 0])  # talker hidden
        self._step(sub_input[0, 1])  # group-0 backbone codec embed
        # _norm_out now holds RMSNorm(hidden[1]) — used for lm_head[0].

        dtype = self._lm_head_weights[0].dtype
        hidden = self._norm_out.unsqueeze(0).to(dtype)  # [1, D]
        token = self._pick(hidden, self._lm_head_weights[0], do_sample, top_k, top_p, temperature)
        tokens = [token]

        for g in range(1, n):
            # Embed previous token using per-group codec embedding.
            embed = torch.nn.functional.embedding(
                token.squeeze(-1),                        # [1]
                self._codec_embedding_weights[g - 1],    # [vocab, D]
            )  # [1, D]
            self._step(embed)
            hidden = self._norm_out.unsqueeze(0).to(self._lm_head_weights[g].dtype)
            token = self._pick(hidden, self._lm_head_weights[g], do_sample, top_k, top_p, temperature)
            tokens.append(token)

        return torch.cat(tokens, dim=-1)  # [1, n]

    @staticmethod
    def _pick(
        hidden: torch.Tensor,
        lm_head_w: torch.Tensor,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> torch.Tensor:
        logits = torch.nn.functional.linear(hidden, lm_head_w)  # [1, vocab]
        if not do_sample:
            return logits.argmax(-1, keepdim=True)  # [1, 1]
        logits = logits / temperature
        if top_k > 1:
            cutoff, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < cutoff[:, -1:]] = float("-inf")
        if top_p < 1.0:
            sorted_logits, sort_idx = torch.sort(logits, descending=True)
            cum_prob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cum_prob - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
            logits.scatter_(-1, sort_idx, sorted_logits)
        return torch.multinomial(torch.softmax(logits, dim=-1), 1)
