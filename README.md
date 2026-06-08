# qwen3-tts-megakernel

Real-time voice assistant pipeline: browser mic → WebSocket → Whisper STT → Qwen3-0.6B LLM → Qwen3-TTS megakernel → browser audio.

Optimised for RTX 5090 (sm_120). Achieves **~48 ms TTFC** and **RTF ~0.23** on the full end-to-end path.

---

## Quick Start

```bash
git clone https://github.com/aadityakanjolia4/qwen3-ttw-megakernel
cd qwen3-tts-megakernel
pip install -e .
```

---

## Start the Server

```bash
python -m pipecat_integration.server \
    --host 0.0.0.0 \
    --port 8081 \
    --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --speaker aiden \
    --whisper-model base
```

Then open `http://<your-server-ip>:8081` in a browser. The page streams mic audio over WebSocket and plays back TTS audio in real time.

---

## Benchmark

Place `input1.wav`, `input2.wav`, `input3.wav` in the project root, then run:

```bash
python benchmark_pipeline.py
```

Loads all models once and runs the full STT → LLM → TTS chain for each file, printing a JSON list:

```json
[
  {
    "file": "input1.wav",
    "ttfc_ms": 48.4,
    "rtf": 0.232,
    "decode_tps": 331.3
  },
  {
    "file": "input2.wav",
    "ttfc_ms": 47.5,
    "rtf": 0.227,
    "decode_tps": 343.9
  },
  {
    "file": "input3.wav",
    "ttfc_ms": 47.3,
    "rtf": 0.227,
    "decode_tps": 337.4
  }
]
```

| File | TTFC (ms) | RTF | Decode tok/s |
|------|----------:|----:|-------------:|
| input1.wav | 48.4 | 0.232 | 331.3 |
| input2.wav | 47.5 | 0.227 | 343.9 |
| input3.wav | 47.3 | 0.227 | 337.4 |

- **TTFC** — time from start of synthesis to first audio chunk delivered
- **RTF** — real-time factor: `wall_time / audio_duration` (< 1.0 = faster than real-time)
- **Decode tok/s** — codec tokens decoded per second by the megakernel talker backbone

---

## Pipeline Architecture

```
Browser mic (PCM 16kHz)
        │  WebSocket (binary PCM)
        ▼
  Silero VAD
        │
        ▼
  Faster-Whisper STT
        │  TranscriptionFrame
        ▼
  Qwen3-0.6B LLM  (/no_think for low latency)
        │  TextFrames → full response
        ▼
  MegakernelTTSService
    ├─ prefill (HF, builds KV cache)
    ├─ talker decode (megakernel, 20-layer backbone)
    ├─ sub-talker (HF, 5-layer code_predictor)
    └─ vocoder (every CHUNK_FRAMES=4 frames → audio chunk)
        │  OutputAudioRawFrame (PCM 16kHz)
        ▼
  Browser speaker (WebSocket binary PCM)
```

---

## Changes from [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)

The upstream repo is a **pure text LM decoder** — it makes next-token generation fast for Qwen3-0.6B via a single fused CUDA kernel. There is no TTS, no audio, no streaming, no pipecat. This project adapted that kernel for the Qwen3-TTS talker backbone and built a full real-time voice pipeline on top.

### `csrc/kernel.cu` — made architecture configurable

The upstream hardcodes all model dimensions for Qwen3-0.6B (LM):

```c
constexpr int HIDDEN_SIZE       = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_KV_HEADS      = 8;
```

The TTS talker backbone has different dimensions (`INTERMEDIATE_SIZE=2048`, `NUM_KV_HEADS=2`). Every constant was wrapped in `#ifndef` preprocessor guards so the same `.cu` file compiles into two separate ops without touching any kernel logic:

```c
#ifndef LDG_INTERMEDIATE_SIZE
#define LDG_INTERMEDIATE_SIZE 3072
#endif
constexpr int INTERMEDIATE_SIZE = LDG_INTERMEDIATE_SIZE;
```

### `qwen_megakernel/build.py` — second TTS build target

Added `get_tts_extension()` which compiles a second shared library `qwen_tts_megakernel_C` with TTS-specific flags (`INTERMEDIATE_SIZE=2048`, `NUM_KV_HEADS=2`, smaller LM head blocks for the 3072-token codec vocab). Both ops coexist in the same Python process.

### `qwen_tts/megakernel/` — entire TTS decode stack (new)

The upstream has no TTS code. These three files were written from scratch to plug the megakernel into the Qwen3-TTS pipeline:

| File | Purpose |
|------|---------|
| `talker_weights.py` | Loads the TTS talker weights from HuggingFace, extracts per-layer tensors and RoPE tables, loads code predictor weights separately |
| `talker_decoder.py` | Stateful megakernel decoder for the 20-layer talker backbone. Handles prefill, per-step decode, KV cache injection, and the 5-layer sub-talker (code predictor) for generating all 32 codec groups per frame |
| `streaming_tts.py` | End-to-end synthesis: text → prefill → talker decode → sub-talker → vocoder → audio chunks. Key parameters: `CHUNK_FRAMES=4` (vocoder called every 4 frames = 333 ms of audio per call, reducing RTF from ~0.38 to ~0.10), `OVERLAP_FRAMES=4` (context overlap to prevent boundary glitches), `torch.compile(mode="reduce-overhead")` on the vocoder decoder (reduces per-call cost from ~75 ms to ~30 ms) |

### `pipecat_integration/` — real-time pipeline (new)

Not present in the upstream at all. Built from scratch:

| File | Purpose |
|------|---------|
| `server.py` | FastAPI WebSocket server. Pre-loads all models at startup. `PCMSerializer` bridges raw binary PCM (browser) ↔ pipecat frames. `PipelineLogger` observer streams logs and transcripts to the browser as JSON over the same WebSocket |
| `megakernel_tts_service.py` | Pipecat `TTSService` wrapper. Runs synthesis in a thread pool executor, streams `OutputAudioRawFrame` chunks via `asyncio.Queue` as they arrive. Uses `TextAggregationMode.SENTENCE` so the full LLM response is synthesised in one call — matching `test_pipeline.py` latency |
| `qwen3_llm_service.py` | Local Qwen3-0.6B LLM service (no API, no Ollama). Two-thread pattern: one thread runs `model.generate()`, another reads the `TextIteratorStreamer` and strips `<think>` blocks. Appends `/no_think` to disable chain-of-thought for voice responses |

---

## Requirements

- CUDA 12.8+
- RTX 5090 (sm_120a) — not tested on other GPUs
- Python ≥ 3.9
- PyTorch ≥ 2.7

---

## License

Apache 2.0
