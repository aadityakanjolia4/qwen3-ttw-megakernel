"""JIT compilation of the megakernel CUDA extension.

Two build profiles:
  - default (Qwen3-0.6B LM):  INTERMEDIATE_SIZE=3072, NUM_KV_HEADS=8,  VOCAB=3072
  - tts_talker (TTS talker):  INTERMEDIATE_SIZE=2048, NUM_KV_HEADS=2,  VOCAB=3072
"""

import os
from torch.utils.cpp_extension import load

_module_lm = None
_module_tts = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "../csrc")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


# RTX 5090 (sm_120) base flags shared by both profiles.
_CUDA_BASE_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-arch=sm_120a",
    f"-I{_CSRC}",
]

_SOURCES = [
    os.path.join(_CSRC, "torch_bindings.cpp"),
    os.path.join(_CSRC, "kernel.cu"),
]


def _build_flags(
    *,
    intermediate_size: int,
    num_kv_heads: int,
    vocab_size: int,
    num_blocks: int,
    block_size: int,
    lm_num_blocks: int,
    lm_block_size: int,
    lm_rows_per_warp: int,
    attn_blocks: int,
) -> list[str]:
    return [
        f"-DLDG_INTERMEDIATE_SIZE={intermediate_size}",
        f"-DLDG_NUM_KV_HEADS={num_kv_heads}",
        f"-DLDG_VOCAB_SIZE={vocab_size}",
        f"-DLDG_NUM_BLOCKS={num_blocks}",
        f"-DLDG_BLOCK_SIZE={block_size}",
        f"-DLDG_LM_NUM_BLOCKS={lm_num_blocks}",
        f"-DLDG_LM_BLOCK_SIZE={lm_block_size}",
        f"-DLDG_LM_ROWS_PER_WARP={lm_rows_per_warp}",
        f"-DLDG_ATTN_BLOCKS={attn_blocks}",
        f"-DLDG_PREFETCH_QK={_env_int('LDG_PREFETCH_QK', 0)}",
        f"-DLDG_PREFETCH_THREAD_STRIDE={_env_int('LDG_PREFETCH_THREAD_STRIDE', 10)}",
        f"-DLDG_PREFETCH_DOWN={_env_int('LDG_PREFETCH_DOWN', 1)}",
        f"-DLDG_PREFETCH_ELEM_STRIDE={_env_int('LDG_PREFETCH_ELEM_STRIDE', 1)}",
        f"-DLDG_PREFETCH_BLOCK_STRIDE={_env_int('LDG_PREFETCH_BLOCK_STRIDE', 1)}",
        f"-DLDG_PREFETCH_GATE={_env_int('LDG_PREFETCH_GATE', 1)}",
        f"-DLDG_PREFETCH_UP={_env_int('LDG_PREFETCH_UP', 1)}",
        "-DLDG_USE_UINT4",
        "-DLDG_ATTENTION_VEC4",
        "-DLDG_WEIGHT_LDCS",
        "-DLDG_MLP_SMEM",
    ]


# Original Qwen3-0.6B LM profile (kept for backward compatibility).
KERNEL_FLAGS = _build_flags(
    intermediate_size=_env_int("LDG_INTERMEDIATE_SIZE", 3072),
    num_kv_heads=_env_int("LDG_NUM_KV_HEADS", 8),
    vocab_size=_env_int("LDG_VOCAB_SIZE", 3072),
    num_blocks=_env_int("LDG_NUM_BLOCKS", 128),
    block_size=_env_int("LDG_BLOCK_SIZE", 512),
    lm_num_blocks=_env_int("LDG_LM_NUM_BLOCKS", 16),
    lm_block_size=_env_int("LDG_LM_BLOCK_SIZE", 384),
    lm_rows_per_warp=_env_int("LDG_LM_ROWS_PER_WARP", 2),
    attn_blocks=16,
)

CUDA_FLAGS = _CUDA_BASE_FLAGS + KERNEL_FLAGS

# TTS talker profile (Qwen3-TTS-12Hz-0.6B-CustomVoice): 28L, ffn=3072, KV=8 heads.
TTS_KERNEL_FLAGS = _build_flags(
    intermediate_size=_env_int("LDG_TTS_INTERMEDIATE_SIZE", 3072),
    num_kv_heads=_env_int("LDG_TTS_NUM_KV_HEADS", 8),
    vocab_size=3072,
    num_blocks=_env_int("LDG_NUM_BLOCKS", 128),
    block_size=_env_int("LDG_BLOCK_SIZE", 512),
    lm_num_blocks=16,
    lm_block_size=384,
    lm_rows_per_warp=2,
    attn_blocks=16,
)

TTS_CUDA_FLAGS = _CUDA_BASE_FLAGS + TTS_KERNEL_FLAGS


def get_extension():
    """Build (or return cached) the default megakernel extension (Qwen3-0.6B LM)."""
    global _module_lm
    if _module_lm is not None:
        return _module_lm

    _module_lm = load(
        name="qwen_megakernel_C",
        sources=_SOURCES,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module_lm


def get_tts_extension():
    """Build (or return cached) the TTS talker megakernel extension."""
    global _module_tts
    if _module_tts is not None:
        return _module_tts

    _module_tts = load(
        name="qwen_tts_megakernel_C",
        sources=_SOURCES,
        extra_cuda_cflags=TTS_CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module_tts
