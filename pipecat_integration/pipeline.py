"""Voice pipeline with local audio transport.

Stack
-----
  Mic → Silero VAD
    → Faster-Whisper STT      (local, ~80 ms on RTX 5090)
    → Qwen3-0.6B-Instruct     (local HF, /no_think, sentence streaming)
    → Megakernel TTS          (local, TTFC ~40 ms)
    → Speakers

Setup
-----
  pip install "pipecat-ai[silero,audio]" faster-whisper

Run
---
  python -m pipecat_integration.pipeline
"""

import argparse
import asyncio
import sys


# ---------------------------------------------------------------------------
# Transport builders
# ---------------------------------------------------------------------------

def _build_local_transport():
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except ImportError:
        print(
            "Local audio transport not available.\n"
            "Run:  pip install 'pipecat-ai[silero,audio]'"
        )
        sys.exit(1)

    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
    )
    transport = LocalAudioTransport(params)
    sample_rate = params.audio_out_sample_rate or 16000
    return transport, sample_rate


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    tts_model: str,
    speaker: str,
    whisper_model: str,
):
    try:
        import pipecat  # noqa: F401
    except ImportError:
        print("pipecat not installed. Run:\n  pip install 'pipecat-ai[silero,audio]'")
        sys.exit(1)

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_response import (
        LLMAssistantResponseAggregator,
        LLMUserResponseAggregator,
    )
    from pipecat.services.whisper.stt import WhisperSTTService

    from pipecat_integration.megakernel_tts_service import MegakernelTTSService
    from pipecat_integration.qwen3_llm_service import Qwen3LLMService, SentenceSplitter

    transport, sample_rate = _build_local_transport()

    # STT: Faster-Whisper (local).
    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=whisper_model))

    # LLM: Qwen3-0.6B-Instruct (local HF, direct inference, /no_think).
    llm = Qwen3LLMService(
        model_name="Qwen/Qwen3-0.6B",
        max_new_tokens=120,
        temperature=0.7,
        enable_thinking=False,
    )

    # Sentence splitter: TTS starts on sentence 1 while LLM generates sentence 2+.
    splitter = SentenceSplitter()

    # TTS: Megakernel-accelerated Qwen3-TTS (local).
    tts = MegakernelTTSService(
        model_name=tts_model,
        speaker=speaker,
        sample_rate=sample_rate,
        verbose=True,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Reply in two sentences maximum. "
                "Never use bullet points, lists, or markdown."
            ),
        }
    ]

    user_agg = LLMUserResponseAggregator(messages)
    assistant_agg = LLMAssistantResponseAggregator(messages)

    pipeline = Pipeline(
        [
            transport.input(),   # mic → VAD frames
            stt,                 # audio → TranscriptionFrame
            user_agg,            # accumulate user turn into LLMMessagesFrame
            llm,                 # LLMMessagesFrame → streaming TextFrames
            splitter,            # TextFrames → sentence-boundary TextFrames
            tts,                 # sentence TextFrame → AudioRawFrame chunks
            transport.output(),  # speakers
            assistant_agg,       # accumulate assistant turn
        ]
    )

    return pipeline, transport


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _download_whisper(model_name: str):
    """Pre-download Whisper model files before the pipeline starts."""
    try:
        from faster_whisper import WhisperModel
        print(f"[Startup] Downloading Whisper '{model_name}' ...")
        _probe = WhisperModel(model_name, device="cpu", compute_type="int8")
        del _probe
        print("[Startup] Whisper downloaded.")
    except Exception as exc:
        print(f"[Startup] Whisper pre-download skipped: {exc}")


async def main(args):
    # Pre-download Whisper before building the pipeline (TTS + LLM load inside
    # build_pipeline already, so this covers the only remaining lazy download).
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _download_whisper, args.whisper_model)

    pipeline, transport = build_pipeline(
        tts_model=args.tts_model,
        speaker=args.speaker,
        whisper_model=args.whisper_model,
    )

    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, enable_metrics=True),
    )

    runner = PipelineRunner()

    print(
        f"\nPipeline ready — local audio\n"
        f"  STT : Faster-Whisper {args.whisper_model}\n"
        f"  LLM : Qwen3-0.6B-Instruct (HF direct, /no_think)\n"
        f"  TTS : Megakernel {args.tts_model} speaker={args.speaker}\n\n"
        f"Speak into your microphone.  Ctrl+C to stop.\n"
    )

    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel voice pipeline")
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    )
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="tiny | base | small | medium | large-v3",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
