"""End-to-end benchmark for the Qwen3-TTS megakernel integration.

Measures:
  - Kernel throughput      (codec tokens/sec from the talker backbone)
  - TTFC                   (time-to-first-audio-chunk, ms)
  - RTF                    (real-time factor = wall_time / audio_duration)
  - End-to-end latency     (prefill + decode + vocoder, ms)

Usage
-----
  python benchmark.py
  python benchmark.py --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \\
                      --speaker aiden --n-runs 5 --warmup 1

Results are printed to stdout as a markdown table and optionally saved to
results.json for CI comparison.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Test sentences (representative voice-agent responses)
# ---------------------------------------------------------------------------

SENTENCES = [
    "Hello!",  # very short — stresses TTFC
    "Sure, I can help with that.",
    "The meeting is scheduled for three PM tomorrow in Conference Room B.",
    "I found five results matching your query. Would you like me to read them out?",
    (
        "Artificial intelligence is transforming the way we interact with technology, "
        "enabling more natural and intuitive experiences across a wide range of applications."
    ),
]


@dataclass
class RunResult:
    sentence: str
    char_len: int
    codec_frames: int
    audio_dur_s: float
    wall_s: float
    ttfc_ms: float
    rtf: float
    tokens_per_sec: float  # codec tokens/sec from backbone alone
    prefill_ms: float
    decode_ms: float
    vocoder_ms: float


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class Benchmarker:
    def __init__(self, model_name: str, speaker: str, verbose: bool = True):
        self.verbose = verbose

        # Import here so CUDA init happens on demand.
        from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel

        if verbose:
            print(f"Loading model: {model_name}")

        self._tts = StreamingTTSMegakernel(
            model_name=model_name, verbose=verbose, max_new_tokens=4096
        )
        self._speaker = speaker
        self._sr = None  # set on first synthesis

    def _run_one(self, sentence: str) -> RunResult:
        """Synthesize one sentence and collect detailed timing."""
        tts = self._tts
        talker = tts._talker
        backbone = tts._backbone
        mk_decoder = tts._mk_decoder

        device = next(talker.parameters()).device
        dtype = next(talker.parameters()).dtype

        timings = {}

        with torch.inference_mode():
            # ---- Prefill --------------------------------------------------
            t0 = time.perf_counter()
            prefill_embeds, trailing_text_hidden, tts_pad_embed = (
                tts._build_prefill_embeds(sentence, self._speaker, "English", device, dtype)
            )
            prefill_len = prefill_embeds.shape[1]

            from transformers.cache_utils import DynamicCache

            past_kv = DynamicCache()
            outputs = backbone(
                inputs_embeds=prefill_embeds,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=False,
            )
            past_kv = outputs.past_key_values
            mk_decoder.reset()
            mk_decoder.inject_kv_cache(past_kv, prefill_len)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter()
            timings["prefill_ms"] = (t_prefill - t0) * 1000

            # ---- Decode loop ---------------------------------------------
            from qwen_tts.megakernel.streaming_tts import CHUNK_FRAMES

            codec_bos_id = tts._codec_bos_id
            codec_eos_id = tts._codec_eos_id
            num_code_groups = tts._num_code_groups

            codec_frames = []
            last_codec_groups = None
            step = 0

            codec_bos_embed = backbone.codec_embedding(
                torch.tensor([[codec_bos_id]], device=device, dtype=torch.long)
            ).squeeze(1)

            audio_chunks = []
            ttfc_ms = None
            t_decode_start = time.perf_counter()
            vocoder_total_s = 0.0

            for _ in range(tts.max_new_tokens):
                if last_codec_groups is None:
                    step_embed = codec_bos_embed
                else:
                    step_embed = backbone.codec_embedding(
                        last_codec_groups[:1]
                    )
                    for g in range(1, num_code_groups):
                        emb_g = tts._code_predictor.get_input_embeddings()[g - 1](
                            last_codec_groups[g : g + 1]
                        )
                        step_embed = step_embed + emb_g

                if step < trailing_text_hidden.shape[1]:
                    step_embed = step_embed + trailing_text_hidden[0, step]
                else:
                    step_embed = step_embed + tts_pad_embed.squeeze()

                codec_token_0, mk_hidden = mk_decoder.step_embed(
                    step_embed.to(torch.bfloat16)
                )

                if codec_token_0 == codec_eos_id:
                    break

                last_id_hidden = backbone.codec_embedding(
                    torch.tensor([[codec_token_0]], device=device, dtype=torch.long)
                )
                sub_input = torch.cat([mk_hidden.to(dtype), last_id_hidden], dim=1)
                sub_result = tts._code_predictor.generate(
                    inputs_embeds=sub_input,
                    max_new_tokens=num_code_groups - 1,
                    do_sample=False,
                    top_k=1,
                )
                extra_groups = sub_result.sequences if hasattr(sub_result, "sequences") else sub_result
                all_groups = torch.cat(
                    [torch.tensor([[codec_token_0]], device=device), extra_groups], dim=-1
                ).squeeze(0)

                codec_frames.append(all_groups)
                last_codec_groups = all_groups
                step += 1

                if len(codec_frames) % CHUNK_FRAMES == 0:
                    t_voc0 = time.perf_counter()
                    chunk_codes = torch.stack(codec_frames[-CHUNK_FRAMES:], dim=0)
                    audio_chunk, sr = tts._decode_audio_chunk(chunk_codes)
                    torch.cuda.synchronize()
                    vocoder_total_s += time.perf_counter() - t_voc0
                    audio_chunks.append(audio_chunk)
                    self._sr = sr

                    if ttfc_ms is None:
                        torch.cuda.synchronize()
                        ttfc_ms = (time.perf_counter() - t0) * 1000

            # Remainder
            remainder = len(codec_frames) % CHUNK_FRAMES
            if remainder > 0:
                t_voc0 = time.perf_counter()
                chunk_codes = torch.stack(codec_frames[-remainder:], dim=0)
                audio_chunk, sr = tts._decode_audio_chunk(chunk_codes)
                torch.cuda.synchronize()
                vocoder_total_s += time.perf_counter() - t_voc0
                audio_chunks.append(audio_chunk)
                if self._sr is None:
                    self._sr = sr

        t_end = time.perf_counter()
        torch.cuda.synchronize()

        decode_ms = (t_end - t_decode_start) * 1000 - (vocoder_total_s * 1000)
        wall_s = t_end - t0
        audio_dur_s = len(codec_frames) / 12.0  # 12 Hz codec
        rtf = wall_s / max(audio_dur_s, 1e-6)
        # codec_token_0 is one token per backbone step; total backbone tokens = len(codec_frames)
        tokens_per_sec = len(codec_frames) / max(decode_ms / 1000.0, 1e-6)

        if ttfc_ms is None:
            ttfc_ms = wall_s * 1000  # fallback (sentence too short for one chunk)

        return RunResult(
            sentence=sentence[:60],
            char_len=len(sentence),
            codec_frames=len(codec_frames),
            audio_dur_s=audio_dur_s,
            wall_s=wall_s,
            ttfc_ms=ttfc_ms,
            rtf=rtf,
            tokens_per_sec=tokens_per_sec,
            prefill_ms=timings["prefill_ms"],
            decode_ms=decode_ms,
            vocoder_ms=vocoder_total_s * 1000,
        )

    def run(self, sentences: List[str], n_runs: int = 3, warmup: int = 1) -> List[RunResult]:
        results = []

        # Warmup
        for i in range(warmup):
            if self.verbose:
                print(f"[warmup {i+1}/{warmup}] {sentences[0][:40]!r}")
            self._run_one(sentences[0])

        torch.cuda.synchronize()

        for s in sentences:
            run_results = []
            for r in range(n_runs):
                if self.verbose:
                    print(f"  run {r+1}/{n_runs}: {s[:50]!r}")
                run_results.append(self._run_one(s))

            # Aggregate: median across runs for this sentence.
            med = RunResult(
                sentence=run_results[0].sentence,
                char_len=run_results[0].char_len,
                codec_frames=int(statistics.median(r.codec_frames for r in run_results)),
                audio_dur_s=statistics.median(r.audio_dur_s for r in run_results),
                wall_s=statistics.median(r.wall_s for r in run_results),
                ttfc_ms=statistics.median(r.ttfc_ms for r in run_results),
                rtf=statistics.median(r.rtf for r in run_results),
                tokens_per_sec=statistics.median(r.tokens_per_sec for r in run_results),
                prefill_ms=statistics.median(r.prefill_ms for r in run_results),
                decode_ms=statistics.median(r.decode_ms for r in run_results),
                vocoder_ms=statistics.median(r.vocoder_ms for r in run_results),
            )
            results.append(med)

        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(results: List[RunResult]):
    header = (
        f"{'Sentence':<42} {'Frames':>6} {'Audio(s)':>8} "
        f"{'TTFC(ms)':>9} {'RTF':>6} {'Tok/s':>7} "
        f"{'Prefill':>8} {'Decode':>8} {'Vocoder':>8}"
    )
    sep = "-" * len(header)
    print()
    print("## Benchmark Results (median)")
    print()
    print(header)
    print(sep)

    for r in results:
        label = (r.sentence[:39] + "…") if len(r.sentence) > 40 else r.sentence
        print(
            f"{label:<42} {r.codec_frames:>6} {r.audio_dur_s:>8.2f} "
            f"{r.ttfc_ms:>9.1f} {r.rtf:>6.3f} {r.tokens_per_sec:>7.0f} "
            f"{r.prefill_ms:>7.0f}ms {r.decode_ms:>7.0f}ms {r.vocoder_ms:>7.0f}ms"
        )

    print(sep)
    avg_ttfc = statistics.mean(r.ttfc_ms for r in results)
    avg_rtf = statistics.mean(r.rtf for r in results)
    avg_tps = statistics.mean(r.tokens_per_sec for r in results)
    print(f"{'Average':<42} {'':>6} {'':>8} {avg_ttfc:>9.1f} {avg_rtf:>6.3f} {avg_tps:>7.0f}")
    print()

    # Pass/fail against targets.
    ttfc_target = 60.0
    rtf_target = 0.15

    print("## Target Checks")
    print()
    ttfc_pass = avg_ttfc < ttfc_target
    rtf_pass = avg_rtf < rtf_target
    print(f"  TTFC < {ttfc_target:.0f} ms:  {'PASS' if ttfc_pass else 'FAIL'}  ({avg_ttfc:.1f} ms)")
    print(f"  RTF  < {rtf_target:.2f}:     {'PASS' if rtf_pass else 'FAIL'}  ({avg_rtf:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel benchmark")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="HuggingFace model id",
    )
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--n-runs", type=int, default=3, help="Runs per sentence (median taken)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument(
        "--sentences",
        nargs="+",
        default=None,
        help="Custom sentences to benchmark (overrides defaults)",
    )
    args = parser.parse_args()

    sentences = args.sentences or SENTENCES

    bencher = Benchmarker(model_name=args.model, speaker=args.speaker, verbose=True)
    results = bencher.run(sentences, n_runs=args.n_runs, warmup=args.warmup)

    print_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
