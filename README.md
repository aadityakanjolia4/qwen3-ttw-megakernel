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

`pip install -e .` handles everything:
- Installs all dependencies including `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`
- Writes a `.pth` hook to site-packages so CUDA libraries are preloaded at every Python startup — no manual `export LD_LIBRARY_PATH` needed

---

## Start the Server

```bash
python -m pipecat_integration.server \
    --host 0.0.0.0 \
    --port 8080 \
    --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --speaker aiden \
    --whisper-model base
```

Then open `http://<your-server-ip>:8080` in a browser. The page streams mic audio over WebSocket and plays back TTS audio in real time.

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
- **RTF** — real-time factor: `wall_time / audio_duration` (lower is faster; < 1.0 means faster than real-time)
- **Decode tok/s** — codec tokens decoded per second by the megakernel talker backbone

---

## How We Reduced TTFC and RTF

### 1. `torch.compile` on the vocoder decoder

The vocoder decoder (HiFi-GAN style) launches ~40 CUDA kernels per call. Without compilation, each call took ~75 ms.

```python
self._speech_tokenizer.model.decoder = torch.compile(
    self._speech_tokenizer.model.decoder,
    mode="reduce-overhead",
    fullgraph=False,
)
```

`reduce-overhead` mode captures CUDA graphs, reducing per-call overhead from ~75 ms to ~30 ms.

### 2. `CHUNK_FRAMES` — amortise vocoder overhead

The vocoder is called once every `CHUNK_FRAMES` codec frames. Each frame is 83 ms of audio at 12 Hz.

| CHUNK_FRAMES | Audio per call | Vocoder cost | RTF |
|:---:|---:|---:|---:|
| 1 | 83 ms | ~30 ms | ~0.38 |
| 4 | 333 ms | ~30 ms | ~0.09 |

Setting `CHUNK_FRAMES = 4` means each 30 ms vocoder call produces 333 ms of audio — an 11× amortisation — bringing RTF from ~0.38 down to ~0.09 on RTX 5090.

### 3. `OVERLAP_FRAMES` — continuity across chunks

`OVERLAP_FRAMES = 4` carries codec context from the previous chunk into each vocoder call, preventing audio discontinuities at chunk boundaries without any extra latency cost.

### 4. Warmup loop for CUDA graph capture

`torch.compile(mode='reduce-overhead')` captures CUDA graphs on first use for each unique input shape. The warmup loop covers all shapes from `T=1` to `T=CHUNK_FRAMES + OVERLAP_FRAMES` so the first real synthesis call hits fully compiled paths:

```python
for t in range(1, CHUNK_FRAMES + OVERLAP_FRAMES + 1):
    _warmup_vocoder(t)
```

### 5. `TextAggregationMode.SENTENCE` + single `synthesize()` call

Rather than splitting the LLM response into sentences and calling TTS once per sentence (which causes a re-prefill penalty each time), the pipeline accumulates the full LLM response and calls `synthesize()` once. This matches `test_pipeline.py` behaviour and keeps TTFC deterministic.

### 6. GPU L2 cache keep-warm

Between turns the GPU L2 cache goes cold, adding ~25 ms to the next turn's TTFC. After each response the server synthesises a short silent phrase to keep the vocoder weights in L2:

```python
tts.synthesize("Okay.", speaker=speaker, language="English", chunk_callback=lambda a, s: None)
```

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

## Requirements

- CUDA 12.8+
- RTX 5090 (sm_120) — not tested on other GPUs
- Python ≥ 3.9
- PyTorch ≥ 2.7

---

## License

Apache 2.0
