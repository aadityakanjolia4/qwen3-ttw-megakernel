"""Load Qwen3-TTS talker backbone weights into megakernel format.

The TTS model has two transformer stacks:
  - talker.model  (backbone, 20 layers): replaced by megakernel
  - talker.code_predictor (sub-talker, 5 layers): kept in HF for sampling

This module extracts the backbone weights and builds the RoPE tables.
"""

import math
import struct

import torch


# Talker backbone architecture constants (Qwen3-TTS-0.6B defaults).
# These MUST match the compile-time constants in the TTS kernel build.
TALKER_NUM_LAYERS = 28
TALKER_NUM_KV_HEADS = 8
TALKER_NUM_Q_HEADS = 16
TALKER_HEAD_DIM = 128
TALKER_HIDDEN_SIZE = 1024
TALKER_INTERMEDIATE_SIZE = 3072
TALKER_VOCAB_SIZE = 3072
TALKER_MAX_SEQ_LEN = 4096

# Code predictor resets each frame: 2 prefill + 31 decode = 33 max positions.
CP_MAX_SEQ_LEN = 64


def load_talker_weights(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    verbose: bool = True,
):
    """Load a Qwen3-TTS model from HuggingFace and extract talker backbone weights.

    Returns
    -------
    weights : dict
        Tensors in megakernel layout.
    hf_model : Qwen3TTSForConditionalGeneration
        Full HF model kept alive (for sub-talker, vocoder, speaker encoder).
    talker_cfg : Qwen3TTSTalkerConfig
        Talker architecture config (used to validate constants).
    """
    import os

    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoConfig, AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging

    if not verbose:
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass

    from qwen_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
    )

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    if verbose:
        print(f"Loading {model_name} ...")
    _load_kwargs = dict(dtype=torch.bfloat16, device_map="cuda")
    try:
        hf_model = AutoModel.from_pretrained(model_name, local_files_only=True, **_load_kwargs)
    except EnvironmentError:
        hf_model = AutoModel.from_pretrained(model_name, **_load_kwargs)
    hf_model.eval()

    talker_cfg = hf_model.config.talker_config
    talker = hf_model.talker  # Qwen3TTSTalkerForConditionalGeneration
    backbone = talker.model   # Qwen3TTSTalkerModel

    num_layers = talker_cfg.num_hidden_layers
    num_kv = talker_cfg.num_key_value_heads
    head_dim = getattr(talker_cfg, "head_dim",
                       talker_cfg.hidden_size // talker_cfg.num_attention_heads)

    if verbose:
        print(
            f"Talker: {num_layers}L, hidden={talker_cfg.hidden_size}, "
            f"ffn={talker_cfg.intermediate_size}, "
            f"Q={talker_cfg.num_attention_heads} KV={num_kv} heads"
        )

    # Use the model's actual inv_freq for RoPE (robust to different rope_theta).
    inv_freq = backbone.rotary_emb.inv_freq.float().cpu()  # [head_dim//2]
    positions = torch.arange(TALKER_MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)             # [max_seq, head_dim//2]
    cos_table = (
        torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    )
    sin_table = (
        torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    )

    # Per-layer weights (11 tensors in megakernel order).
    backbone_sd = backbone.state_dict()
    layer_weights = []
    for i in range(num_layers):
        p = f"layers.{i}."
        layer_weights.extend(
            [
                backbone_sd[p + "input_layernorm.weight"].contiguous(),
                backbone_sd[p + "self_attn.q_proj.weight"].contiguous(),
                backbone_sd[p + "self_attn.k_proj.weight"].contiguous(),
                backbone_sd[p + "self_attn.v_proj.weight"].contiguous(),
                backbone_sd[p + "self_attn.q_norm.weight"].contiguous(),
                backbone_sd[p + "self_attn.k_norm.weight"].contiguous(),
                backbone_sd[p + "self_attn.o_proj.weight"].contiguous(),
                backbone_sd[p + "post_attention_layernorm.weight"].contiguous(),
                backbone_sd[p + "mlp.gate_proj.weight"].contiguous(),
                backbone_sd[p + "mlp.up_proj.weight"].contiguous(),
                backbone_sd[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    embed_weight = backbone_sd["codec_embedding.weight"].contiguous()
    final_norm_weight = backbone_sd["norm.weight"].contiguous()
    # codec_head is directly on the talker (not backbone)
    lm_head_weight = talker.state_dict()["codec_head.weight"].contiguous()

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        num_layers=num_layers,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=talker_cfg.hidden_size,
    )

    return weights, hf_model, talker_cfg


def load_code_predictor_weights(hf_model) -> dict:
    """Extract code predictor weights into megakernel format.

    The 5-layer code predictor shares the same architecture as the talker
    backbone (hidden=1024, ffn=3072, 16Q/8KV heads, head_dim=128), so the
    same compiled TTS kernel handles it with num_layers=5.

    The kernel LM-head output is ignored; per-group lm_head[g] weights are
    applied in Python via F.linear on _norm_out.
    """
    cp = hf_model.talker.code_predictor
    cp_model = cp.model
    cp_cfg = cp.config

    num_layers = cp_cfg.num_hidden_layers
    num_kv = cp_cfg.num_key_value_heads
    head_dim = cp_cfg.head_dim

    inv_freq = cp_model.rotary_emb.inv_freq.float().cpu()
    positions = torch.arange(CP_MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    sd = cp_model.state_dict()
    layer_weights = []
    for i in range(num_layers):
        p = f"layers.{i}."
        layer_weights.extend([
            sd[p + "input_layernorm.weight"].contiguous(),
            sd[p + "self_attn.q_proj.weight"].contiguous(),
            sd[p + "self_attn.k_proj.weight"].contiguous(),
            sd[p + "self_attn.v_proj.weight"].contiguous(),
            sd[p + "self_attn.q_norm.weight"].contiguous(),
            sd[p + "self_attn.k_norm.weight"].contiguous(),
            sd[p + "self_attn.o_proj.weight"].contiguous(),
            sd[p + "post_attention_layernorm.weight"].contiguous(),
            sd[p + "mlp.gate_proj.weight"].contiguous(),
            sd[p + "mlp.up_proj.weight"].contiguous(),
            sd[p + "mlp.down_proj.weight"].contiguous(),
        ])

    final_norm_weight = sd["norm.weight"].contiguous()

    # Dummy tensors for the kernel's embed lookup and LM-head — always skipped
    # via sentinel -1, but the kernel signature still needs valid pointers.
    dummy_embed = torch.zeros(TALKER_VOCAB_SIZE, TALKER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    dummy_lm_head = torch.zeros(TALKER_VOCAB_SIZE, TALKER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Optional input projection (created only when cp_hidden != talker_hidden).
    proj_weight = None
    proj_bias = None
    if hasattr(cp, "small_to_mtp_projection") and hasattr(cp.small_to_mtp_projection, "weight"):
        proj_weight = cp.small_to_mtp_projection.weight.contiguous()
        proj_bias = cp.small_to_mtp_projection.bias.contiguous()

    codec_embedding_weights = [cp_model.codec_embedding[g].weight for g in range(len(cp_model.codec_embedding))]
    lm_head_weights = [cp.lm_head[g].weight for g in range(len(cp.lm_head))]

    return dict(
        layer_weights=layer_weights,
        final_norm_weight=final_norm_weight,
        dummy_embed=dummy_embed,
        dummy_lm_head=dummy_lm_head,
        cos_table=cos_table,
        sin_table=sin_table,
        num_layers=num_layers,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        hidden_size=cp_cfg.hidden_size,
        vocab_size=cp_cfg.vocab_size,
        codec_embedding_weights=codec_embedding_weights,
        lm_head_weights=lm_head_weights,
        num_code_groups=cp_cfg.num_code_groups,
        proj_weight=proj_weight,
        proj_bias=proj_bias,
    )


def pack_layer_weights(layer_weights: list, num_layers: int) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a GPU blob of LDGLayerWeights."""
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(num_layers * struct_bytes)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()
