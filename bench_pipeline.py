"""Benchmark: input1.wav / input2.wav / input3.wav → STT → LLM → TTS → JSON list

Runs automatically:  python bench_pipeline.py
"""

import json
import re
import time

import torch

INPUT_FILES   = ["input1.wav", "input2.wav", "input3.wav"]
WHISPER_MODEL = "base"
LLM_MODEL     = "Qwen/Qwen3-0.6B"
TTS_MODEL     = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKER       = "aiden"


# ---------------------------------------------------------------------------
# Loaders (called once, shared across all inputs)
# ---------------------------------------------------------------------------

def load_whisper():
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)


def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def load_tts():
    from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel
    return StreamingTTSMegakernel(model_name=TTS_MODEL, verbose=False)


# ---------------------------------------------------------------------------
# Per-file benchmark
# ---------------------------------------------------------------------------

def run_one(audio_path, whisper, llm_model, llm_tok, tts):
    device = next(llm_model.parameters()).device

    # STT
    segments, _ = whisper.transcribe(audio_path, beam_size=5)
    transcript = " ".join(s.text for s in segments).strip()

    # LLM
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Reply in one or two short sentences. No markdown."},
        {"role": "user", "content": transcript + " /no_think"},
    ]
    prompt = llm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out_ids = llm_model.generate(
            **inputs, max_new_tokens=120, temperature=0.7,
            do_sample=True, pad_token_id=llm_tok.eos_token_id,
        )
    response = llm_tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # TTS
    first_chunk: list[float] = []
    def on_chunk(audio, sr):
        if not first_chunk:
            first_chunk.append(time.perf_counter())

    stats: dict = {}
    t0 = time.perf_counter()
    audio, sr = tts.synthesize(
        response, speaker=SPEAKER, language="English",
        chunk_callback=on_chunk, stats_out=stats,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    audio_ms = max(len(audio) / sr * 1000, 1)
    ttfc_ms = (first_chunk[0] - t0) * 1000 if first_chunk else None

    return {
        "file": audio_path,
        "ttfc_ms": round(ttfc_ms, 1) if ttfc_ms is not None else None,
        "rtf": round(wall_ms / audio_ms, 3),
        "decode_tps": round(stats.get("decode_tps", 0), 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading models...", flush=True)
    whisper = load_whisper()
    llm_model, llm_tok = load_llm()
    tts = load_tts()
    print("Models ready. Running benchmark...\n", flush=True)

    results = []
    for f in INPUT_FILES:
        print(f"  {f} ...", flush=True)
        results.append(run_one(f, whisper, llm_model, llm_tok, tts))

    print(json.dumps(results, indent=2))
