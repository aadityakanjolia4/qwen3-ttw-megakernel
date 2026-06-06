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

## Credits

- Qwen3-TTS pipeline: [Alibaba Qwen Team](https://github.com/Qwen/Qwen3-TTS)
- Megakernel approach: based on [MegaQwen](https://github.com/Infatoshi/MegaQwen) by Elliot Arledge (RTX 3090), extended for RTX 5090 by [alpindale](https://blog.alpindale.net/posts/5090_decode_optimization/)

## License

Apache 2.0
