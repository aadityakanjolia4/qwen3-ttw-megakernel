"""Benchmark: input.wav → STT → LLM → TTS → JSON metrics

Usage:
  python bench_pipeline.py input.wav
  python bench_pipeline.py input.wav --whisper-model small --runs 3
"""

import argparse
import json
import re
import time

import torch


def bench_stt(audio_path, model_size):
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join(s.text for s in segments).strip()
    return {"stt_ms": round((time.perf_counter() - t0) * 1000, 1), "transcript": transcript}


def bench_llm(transcript, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Reply in one or two short sentences. No markdown."},
        {"role": "user", "content": transcript + " /no_think"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=120, temperature=0.7,
                                  do_sample=True, pad_token_id=tok.eos_token_id)
    llm_ms = (time.perf_counter() - t0) * 1000
    n_new = out_ids.shape[1] - inputs["input_ids"].shape[1]
    response = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return {
        "llm_ms": round(llm_ms, 1),
        "llm_tokens": n_new,
        "llm_tps": round(n_new / (llm_ms / 1000), 1),
        "response": response,
    }


def bench_tts(text, model_name, speaker, tts_instance=None):
    if tts_instance is None:
        from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel
        tts_instance = StreamingTTSMegakernel(model_name=model_name, verbose=False)

    first_chunk: list[float] = []
    def on_chunk(audio, sr):
        if not first_chunk:
            first_chunk.append(time.perf_counter())

    stats: dict = {}
    t0 = time.perf_counter()
    audio, sr = tts_instance.synthesize(
        text, speaker=speaker, language="English",
        chunk_callback=on_chunk, stats_out=stats,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    audio_ms = max(len(audio) / sr * 1000, 1)
    ttfc_ms = (first_chunk[0] - t0) * 1000 if first_chunk else None

    return {
        "ttfc_ms": round(ttfc_ms, 1) if ttfc_ms is not None else None,
        "rtf": round(wall_ms / audio_ms, 3),
        "decode_tps": round(stats.get("decode_tps", 0), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input wav file")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--runs", type=int, default=1, help="Repeat TTS N times and average")
    args = parser.parse_args()

    stt = bench_stt(args.input, args.whisper_model)
    llm = bench_llm(stt["transcript"], args.llm_model)

    from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel
    tts = StreamingTTSMegakernel(model_name=args.tts_model, verbose=False)

    runs = [bench_tts(llm["response"], args.tts_model, args.speaker, tts)
            for _ in range(args.runs)]

    if args.runs == 1:
        metrics = runs[0]
    else:
        metrics = {}
        for k in runs[0]:
            vals = [r[k] for r in runs if r[k] is not None]
            metrics[k] = round(sum(vals) / len(vals), 1) if vals else None

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
