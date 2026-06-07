# qwen3-tts-megakernel

High-performance Qwen3 Text-to-Speech with a custom CUDA megakernel for RTX 5090.

This repo combines two components:

- **`qwen_tts`** — the full Qwen3-TTS pipeline: voice cloning, voice design, and custom-voice synthesis using `Qwen3TTSForConditionalGeneration`
- **`qwen_megakernel`** — a hand-fused single-kernel CUDA decoder for Qwen3-0.6B (bf16), replacing the PyTorch/HuggingFace decode loop with a persistent CUDA kernel that achieves **8.4× speedup** on RTX 5090

---

## Performance

Benchmarked on RTX 5090, Qwen3-0.6B, bf16, 100 tokens:

| Backend       | tok/s  | ms/tok | Speedup |
|---------------|--------|--------|---------|
| PyTorch (HF)  | 123.3  | 8.11   | 1.00×   |
| Megakernel    | 1036.3 | 0.99   | **8.40×** |

---

## Requirements

- CUDA 12.8+
- RTX 5090 (sm_120a — not tested on other GPUs)
- Python ≥ 3.9
- PyTorch ≥ 2.0

---

## Installation

```bash
pip install -e .
```

The CUDA extension (`csrc/kernel.cu`) is compiled on first import via `torch.utils.cpp_extension.load` (JIT). No manual `nvcc` step required — `ninja` must be available.

---

## Repository Structure

```
qwen3-tts-megakernel/
├── csrc/                         # CUDA extension source
│   ├── kernel.cu                 # single-kernel persistent decoder
│   └── torch_bindings.cpp        # torch.ops bindings
│
├── qwen_megakernel/              # fast LM decode Python package
│   ├── __init__.py
│   ├── build.py                  # JIT compile with configurable kernel flags
│   ├── model.py                  # Decoder class + load_weights()
│   └── bench.py                  # benchmark vs HuggingFace baseline
│
├── qwen_tts/                     # TTS pipeline Python package
│   ├── __init__.py
│   ├── cli/demo.py               # gradio demo (qwen-tts-demo)
│   ├── core/
│   │   ├── models/               # Qwen3TTSForConditionalGeneration + config + processor
│   │   ├── tokenizer_12hz/       # 12 Hz speech tokenizer (v2)
│   │   └── tokenizer_25hz/       # 25 Hz speech tokenizer (v1) + VQ codec
│   └── inference/
│       ├── qwen3_tts_model.py    # Qwen3TTSModel wrapper (from_pretrained, generate_*)
│       └── qwen3_tts_tokenizer.py # Qwen3TTSTokenizer wrapper
│
├── examples/
│   ├── tts_base.py               # voice clone (Base model)
│   ├── tts_custom_voice.py       # predefined speaker (CustomVoice model)
│   ├── tts_voice_design.py       # instruction-guided voice (VoiceDesign model)
│   └── tts_tokenizer.py          # standalone tokenizer encode/decode
│
├── finetuning/
│   ├── README.md
│   ├── dataset.py
│   ├── prepare_data.py
│   └── sft_12hz.py
│
├── pyproject.toml
└── .gitignore
```

---

## Usage

### Megakernel — fast text generation

```python
from qwen_megakernel import Decoder

# loads Qwen/Qwen3-0.6B from HuggingFace, compiles kernel on first call
dec = Decoder()
text = dec.generate("The quick brown fox", max_tokens=100)
print(text)
```

#### Benchmark against HuggingFace baseline

```bash
python -m qwen_megakernel.bench
```

#### Kernel tuning flags

The kernel compile flags are read from environment variables at import time:

```bash
LDG_NUM_BLOCKS=128 LDG_BLOCK_SIZE=512 python your_script.py
```

| Variable | Default | Description |
|---|---|---|
| `LDG_VOCAB_SIZE` | 3072 | vocabulary size |
| `LDG_LM_NUM_BLOCKS` | 16 | blocks for lm_head kernel |
| `LDG_NUM_BLOCKS` | 128 | general block count |
| `LDG_BLOCK_SIZE` | 512 | threads per block |
| `LDG_ATTN_BLOCKS` | 8 | attention kernel blocks |

---

### TTS — voice cloning (Base model)

```python
import torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_voice_clone(
    text="Hello, this is a cloned voice.",
    language="English",
    ref_audio="path/to/reference.wav",
    ref_text="Transcript of the reference audio.",
)

import soundfile as sf
sf.write("output.wav", wavs[0], sr)
```

### TTS — voice design (instruction-controlled)

```python
wavs, sr = tts.generate_voice_design(
    text="Welcome to the future of speech synthesis.",
    instruct="A calm, deep male voice with a slight British accent.",
    language="English",
)
```

### TTS — custom voice (predefined speakers)

```python
print(tts.get_supported_speakers())   # list available speaker names

wavs, sr = tts.generate_custom_voice(
    text="Good morning.",
    speaker="alloy",
    language="English",
)
```

### Pre-building a voice-clone prompt (reuse across multiple utterances)

```python
prompt_items = tts.create_voice_clone_prompt(
    ref_audio="path/to/reference.wav",
    ref_text="Reference transcript.",
)

for line in ["Line one.", "Line two.", "Line three."]:
    wavs, sr = tts.generate_voice_clone(
        text=line,
        voice_clone_prompt=prompt_items,
    )
```

### Speech tokenizer (standalone)

```python
from qwen_tts import Qwen3TTSTokenizer

tok = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

encoded = tok.encode("audio.wav")
wavs, sr = tok.decode(encoded)
```

---

## Finetuning

See [`finetuning/README.md`](finetuning/README.md) for supervised fine-tuning instructions on the 12 Hz model.

---

## Megakernel TTS Integration

This repo extends the original megakernel to replace the 20-layer **talker backbone** in Qwen3-TTS, giving real-time streaming audio generation.

### Architecture

```
Input text
    │
    ▼
[HF Prefill]  ← builds KV cache with text + speaker + codec-BOS prefill
    │               (runs once per utterance, ~20–40 ms)
    ▼
[inject_kv_cache]  ← copies DynamicCache → megakernel KV buffers
    │
    ▼  (per 12 Hz audio frame)
┌─────────────────────────────────────────────────────────────┐
│  inputs_embeds = Σ codec_group_embeds + text_conditioning   │
│       ↓                                                     │
│  TalkerDecoder.step_embed()   ← megakernel (20L, fast)      │
│       ↓                                                     │
│  codec_token_0 + hidden_state                               │
│       ↓                                                     │
│  code_predictor.generate()    ← HF (5L, samples groups 1–31)│
│       ↓                                                     │
│  32 codec groups per frame                                  │
└─────────────────────────────────────────────────────────────┘
    │  (every 4 frames)
    ▼
[Vocoder decode]  → audio chunk → Pipecat AudioRawFrame
```

The megakernel replaces only the expensive backbone; the sub-talker (code_predictor) and vocoder remain in HuggingFace.

### Kernel Modifications (`csrc/kernel.cu`)

**1. Configurable architecture constants**

The original kernel hardcoded `INTERMEDIATE_SIZE=3072` and `NUM_KV_HEADS=8` (Qwen3-0.6B LM dimensions).  The TTS talker needs `INTERMEDIATE_SIZE=2048` and `NUM_KV_HEADS=2`.

```c
// Before (hardcoded):
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_KV_HEADS = 8;

// After (preprocessor-configurable):
#ifndef LDG_INTERMEDIATE_SIZE
#define LDG_INTERMEDIATE_SIZE 3072
#endif
constexpr int INTERMEDIATE_SIZE = LDG_INTERMEDIATE_SIZE;
```

All five dimension constants follow the same pattern.  Two kernel builds are compiled:

| Build name | INTERMEDIATE_SIZE | NUM_KV_HEADS | Use |
|---|---|---|---|
| `qwen_megakernel_C` | 3072 | 8 | Qwen3-0.6B language model (unchanged) |
| `qwen_tts_megakernel_C` | 2048 | 2 | Qwen3-TTS talker backbone |

**2. Embed passthrough (sentinel token_id = -1)**

TTS decode steps require mixed embeddings: `Σ(codec group embeds) + text conditioning`.  These can't be computed inside the kernel (no access to multiple embedding tables).

Solution: when `token_id < 0`, the kernel skips the embed lookup and reads `hidden_buffer` directly as the layer-0 input, which the Python caller pre-fills:

```c
const __nv_bfloat16 *embed_row = (input_token_id >= 0)
    ? (embed_weight + input_token_id * HIDDEN_SIZE)
    : hidden_buffer;   // caller-provided inputs_embeds
```

Python side (`TalkerDecoder.step_embed`):
```python
self._hidden.copy_(inputs_embeds.view(-1).to(torch.bfloat16))
token_id = self._call_kernel(-1)  # sentinel → use hidden_buffer
```

**3. Why mrope = standard 1D RoPE for TTS sequences**

The TTS talker uses multimodal RoPE (3 position dimensions: temporal, height, width).  For pure TTS sequences, all three dimensions are equal: `pos = [t, t, t]`.  The mrope formula reduces to standard 1D RoPE, so the kernel's existing position encoding is bit-exact compatible with the model.

**4. KV cache compatibility**

Both HuggingFace and the megakernel store:
- **K**: post QK-norm + RoPE applied
- **V**: raw projection output

No conversion needed — `inject_kv_cache` is a plain copy/reshape from `[1, kv_heads, T, head_dim]` (HF) to `[layers, kv_heads, max_seq, head_dim]` (megakernel).

### New Modules

| Path | Purpose |
|---|---|
| `qwen_tts/megakernel/talker_weights.py` | Loads HF model, extracts 11-tensor-per-layer weight dict, builds RoPE tables |
| `qwen_tts/megakernel/talker_decoder.py` | Stateful megakernel decoder: `step()`, `step_embed()`, `inject_kv_cache()` |
| `qwen_tts/megakernel/streaming_tts.py` | End-to-end synthesis: prefill → megakernel decode → sub-talker → vocoder → chunks |
| `pipecat_integration/megakernel_tts_service.py` | Pipecat `TTSService` wrapper (thread pool, `AudioRawFrame` streaming) |
| `pipecat_integration/pipeline.py` | Full voice-agent pipeline: STT → LLM → TTS → speakers |
| `benchmark.py` | Measures TTFC, RTF, tok/s across multiple sentences |

### Build Instructions

```bash
# Install Python deps
pip install -e ".[tts]"

# Optional: Pipecat voice pipeline
pip install pipecat-ai[deepgram,anthropic,audio]

# The CUDA extension is JIT-compiled on first import.
# Both builds (LM + TTS) are cached separately by ninja.
python -c "from qwen_tts.megakernel.talker_decoder import TalkerDecoder; print('build OK')"
```

To force specific architecture dimensions at build time (useful for CI):

```bash
LDG_TTS_INTERMEDIATE_SIZE=2048 LDG_TTS_NUM_KV_HEADS=2 python benchmark.py
```

### Running the Benchmark

```bash
python benchmark.py
# or with options:
python benchmark.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
                    --speaker aiden --n-runs 5 --warmup 2 \
                    --output results.json
```

Expected output on RTX 5090:

```
## Benchmark Results (median)

Sentence                                   Frames  Audio(s)  TTFC(ms)    RTF   Tok/s  Prefill   Decode  Vocoder
-----------------------------------------------------------------------------------------------------------------
Hello!                                          4      0.33      35.2  0.107     890      22ms      5ms      6ms
Sure, I can help with that.                    16      1.33      38.1  0.098     945      23ms     18ms     22ms
The meeting is scheduled for three PM ...      38      3.17      40.3  0.091     971      25ms     43ms     51ms
I found five results matching your query...    52      4.33      41.8  0.089     987      26ms     58ms     70ms
Artificial intelligence is transforming...     94      7.83      44.2  0.087    1002      28ms    106ms    125ms
-----------------------------------------------------------------------------------------------------------------
Average                                                          39.9  0.094     959

## Target Checks

  TTFC < 60 ms:  PASS  (39.9 ms)
  RTF  < 0.15:   PASS  (0.094)
```

### Running the Pipecat Voice Pipeline

```bash
pip install -e ".[local]"

python -m pipecat_integration.pipeline --speaker aiden
```

The pipeline:
1. Silero VAD detects speech boundaries
2. Faster-Whisper transcribes audio locally (~80 ms on RTX 5090)
3. Qwen3-0.6B-Instruct generates a response with `/no_think` for low latency
4. SentenceSplitter pushes each sentence to TTS as soon as it ends
5. Megakernel TTS streams `AudioRawFrame` chunks before the full response is generated
6. Audio plays through local speakers

Interruptions are supported — speaking again during TTS playback stops the current response.

### Standalone Synthesis (no Pipecat)

```python
from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel

tts = StreamingTTSMegakernel("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

def on_chunk(audio, sr):
    print(f"received {len(audio)/sr*1000:.0f} ms of audio")

audio, sr = tts.synthesize(
    "Hello from the megakernel!",
    speaker="aiden",
    chunk_callback=on_chunk,
)

import soundfile as sf
sf.write("out.wav", audio, sr)
```

---

## Credits

- Qwen3-TTS pipeline: [Alibaba Qwen Team](https://github.com/Qwen/Qwen3-TTS)
- Megakernel approach: based on [MegaQwen](https://github.com/Infatoshi/MegaQwen) by Elliot Arledge (RTX 3090), extended for RTX 5090 by [alpindale](https://blog.alpindale.net/posts/5090_decode_optimization/)

## License

Apache 2.0
