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
        from qwen_megakernel.build import get_tts_extension

        ext = get_tts_extension()
        self._decode_op = ext.decode

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

    @property
    def position(self) -> int:
        return self._position
