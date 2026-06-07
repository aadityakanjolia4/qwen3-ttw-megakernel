"""
Full pipeline test: audio file → Whisper STT → Qwen3 LLM → Megakernel TTS → output.wav

Usage:
  python test_pipeline.py input.wav
  python test_pipeline.py input.wav --whisper-model small --speaker aiden
  python test_pipeline.py input.wav --tts-model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

No WebRTC, no browser needed. Tests the exact same components the server uses.
"""

import argparse
import sys
import time

import numpy as np
import soundfile as sf
import torch


def step(name):
    print(f"\n[{name}]")


def check_gpu():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory/1024**3:.1f} GB VRAM)")
    else:
        print("WARNING: No CUDA — running on CPU")


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def run_stt(audio_path: str, model_size: str) -> str:
    step("STT  (Faster-Whisper)")
    from faster_whisper import WhisperModel

    print(f"  Model : {model_size}")
    print(f"  Input : {audio_path}")

    def _load_model():
        # int8_float16 avoids the cuBLAS dependency while still using the GPU.
        if torch.cuda.is_available():
            try:
                m = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                print("  Device: cuda (int8_float16)")
                return m
            except Exception as e:
                print(f"  CUDA load failed ({e}), falling back to CPU")
        m = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("  Device: cpu (int8)")
        return m

    t0 = time.perf_counter()
    model = _load_model()
    load_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join(s.text for s in segments).strip()
    transcribe_ms = (time.perf_counter() - t1) * 1000

    print(f"  Loaded in {load_ms:.0f} ms, transcribed in {transcribe_ms:.0f} ms")
    print(f"  Detected language: {info.language} ({info.language_probability:.0%})")
    print(f"  Transcript: {transcript!r}")
    return transcript


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def run_llm(transcript: str, model_name: str) -> str:
    step("LLM  (Qwen3)")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"  Model : {model_name}")
    print(f"  Input : {transcript!r}")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {load_ms:.0f} ms")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Reply in two sentences maximum. "
                "Never use bullet points, lists, or markdown."
            ),
        },
        {"role": "user", "content": transcript + " /no_think"},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    t1 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
        )
    gen_ms = (time.perf_counter() - t1) * 1000

    new_tokens = out_ids.shape[1] - inputs["input_ids"].shape[1]
    response = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"  Response: {response!r}")
    print(f"  {new_tokens} tokens in {gen_ms:.0f} ms  ({new_tokens/(gen_ms/1000):.1f} tok/s)")
    return response


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def run_tts(text: str, model_name: str, speaker: str, out_path: str):
    step("TTS  (Megakernel)")
    from pipecat_integration.megakernel_tts_service import MegakernelTTSService

    print(f"  Model  : {model_name}")
    print(f"  Speaker: {speaker}")
    print(f"  Input  : {text!r}")

    tts = MegakernelTTSService(
        model_name=model_name,
        speaker=speaker,
        sample_rate=16000,
        verbose=True,
    )

    t0 = time.perf_counter()
    audio, sr = tts.synthesize(text)
    elapsed = (time.perf_counter() - t0) * 1000

    duration_ms = len(audio) / sr * 1000
    sf.write(out_path, audio, sr)

    print(f"  RTF   : {elapsed/duration_ms:.2f}x  ({elapsed:.0f} ms to produce {duration_ms:.0f} ms of audio)")
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: wav → STT → LLM → TTS → wav")
    parser.add_argument("input", help="Input audio file (wav/mp3/flac)")
    parser.add_argument("--output", default="pipeline_output.wav")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--llm-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--skip-llm", action="store_true", help="Use a hardcoded prompt instead of running LLM")
    args = parser.parse_args()

    print("=" * 55)
    print("  Full pipeline test: audio → STT → LLM → TTS → audio")
    print("=" * 55)
    check_gpu()

    total_t0 = time.perf_counter()

    # Step 1: STT
    try:
        transcript = run_stt(args.input, args.whisper_model)
    except Exception as e:
        print(f"\nSTT failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Step 2: LLM
    if args.skip_llm:
        response = f"You said: {transcript}"
        print(f"\n[LLM] Skipped — using passthrough: {response!r}")
    else:
        try:
            response = run_llm(transcript, args.llm_model)
        except Exception as e:
            print(f"\nLLM failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            sys.exit(1)

    # Step 3: TTS
    try:
        run_tts(response, args.tts_model, args.speaker, args.output)
    except Exception as e:
        print(f"\nTTS failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    total_ms = (time.perf_counter() - total_t0) * 1000
    print(f"\n{'='*55}")
    print(f"  Total end-to-end: {total_ms:.0f} ms")
    print(f"  Output: {args.output}")
    print("=" * 55)


if __name__ == "__main__":
    main()
