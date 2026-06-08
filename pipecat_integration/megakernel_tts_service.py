"""Pipecat TTSService backed by the megakernel talker decoder.

Implements the pipecat.services.ai_services.TTSService interface:

  async def run_tts(self, sentence: str) -> AsyncGenerator[Frame, None]

Each audio frame is a pipecat AudioRawFrame with 16-bit PCM @ 16 kHz (or the
model's native sample rate).  Frames are pushed as soon as each CHUNK_FRAMES
chunk is decoded, giving sub-second latency on RTX 5090.

Usage
-----
  from pipecat_integration import MegakernelTTSService
  tts = MegakernelTTSService(
      model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
      speaker="aiden",
  )
  pipeline = Pipeline([..., tts, ...])
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import numpy as np

# Pipecat imports — gracefully degrade if pipecat is not installed.
try:
    from pipecat.frames.frames import EndFrame, Frame, OutputAudioRawFrame
    from pipecat.services.tts_service import TTSService, TTSSettings, TextAggregationMode

    _PIPECAT_AVAILABLE = True
except ImportError:
    _PIPECAT_AVAILABLE = False

    class TTSService:  # type: ignore[no-redef]
        """Stub for environments without pipecat installed."""
        pass

    class OutputAudioRawFrame:  # type: ignore[no-redef]
        def __init__(self, audio, sample_rate, num_channels):
            self.audio = audio
            self.sample_rate = sample_rate
            self.num_channels = num_channels


class MegakernelTTSService(TTSService if _PIPECAT_AVAILABLE else object):
    """Pipecat-compatible TTS service using the megakernel talker backbone.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id of the Qwen3-TTS model.
    speaker : str
        Speaker name (for CustomVoice models).
    language : str
        Synthesis language.
    sample_rate : int
        Output audio sample rate (resampled if needed).
    verbose : bool
        Print timing info per utterance.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "aiden",
        language: str = "English",
        sample_rate: int = 16000,
        verbose: bool = True,
        tts_instance: Optional[object] = None,
        log_callback=None,
        timing: Optional[dict] = None,
    ):
        if _PIPECAT_AVAILABLE:
            super().__init__(
                sample_rate=sample_rate,
                settings=TTSSettings(
                    model=model_name,
                    voice=speaker,
                    language=language,
                ),
                # SENTENCE mode accumulates the full LLM response before calling
                # run_tts — one synthesize call for the full text, matching test_pipeline.
                # The megakernel streams internally at CHUNK_FRAMES=1, so TTFC is
                # prefill(full_text) + 1 decode step + 1 vocoder call.
                text_aggregation_mode=TextAggregationMode.SENTENCE,
            )

        self._speaker = speaker
        self._language = language
        self._target_sr = sample_rate
        self._verbose = verbose
        self._model_name = model_name
        self._log_callback = log_callback  # Callable[[dict], None] | None
        self._timing = timing              # shared dict with "vad_end_ts" written by observer

        # Accept a pre-loaded instance (passed from server startup) or lazy-load.
        self._tts: Optional[object] = tts_instance

    def _ensure_loaded(self):
        if self._tts is None:
            from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel

            self._tts = StreamingTTSMegakernel(
                model_name=self._model_name, verbose=self._verbose
            )

    # ------------------------------------------------------------------
    # Pipecat TTSService interface
    # ------------------------------------------------------------------

    async def run_tts(self, sentence: str, context_id: Optional[str] = None) -> AsyncGenerator:
        """Synthesize sentence and yield AudioRawFrame chunks as they arrive.

        This is called by pipecat's pipeline per TTS sentence segment.
        """
        self._ensure_loaded()

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        total_samples = 0
        first_chunk_ts: list[float] = []  # mutable container for thread callback

        t_synth_start: list[float] = []  # captured inside the executor thread

        def on_chunk(audio: np.ndarray, sr: int) -> None:
            nonlocal total_samples
            if not first_chunk_ts:
                first_chunk_ts.append(time.perf_counter())
            total_samples += len(audio)
            pcm = _to_pcm16(audio, src_sr=sr, dst_sr=self._target_sr)
            frame = OutputAudioRawFrame(
                audio=pcm.tobytes(),
                sample_rate=self._target_sr,
                num_channels=1,
            )
            asyncio.run_coroutine_threadsafe(queue.put(frame), loop)

        def _synth():
            t_synth_start.append(time.perf_counter())
            return self._tts.synthesize(
                sentence, self._speaker, self._language, on_chunk
            )

        t0 = time.perf_counter()
        synthesis_future = loop.run_in_executor(None, _synth)

        ttfc_logged = False
        done = False
        while not done:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield frame
                if not ttfc_logged and first_chunk_ts and self._log_callback is not None:
                    ttfc_logged = True
                    exec_delay = (t_synth_start[0] - t0) * 1000 if t_synth_start else 0
                    synth_ttfc  = (first_chunk_ts[0] - t_synth_start[0]) * 1000 if t_synth_start else 0
                    total_ttfc  = (first_chunk_ts[0] - t0) * 1000
                    print(
                        f"[TTFC] executor_delay={exec_delay:.1f}ms  "
                        f"synth={synth_ttfc:.1f}ms  total={total_ttfc:.1f}ms",
                        flush=True,
                    )
                    self._log_callback({"type": "log", "msg": f"TTFC: {total_ttfc:.0f} ms"})
            except asyncio.TimeoutError:
                if synthesis_future.done():
                    done = True

        # Drain any remaining frames that arrived right at the end.
        while not queue.empty():
            yield await queue.get()

        # Propagate any GPU/synthesis exception.
        await synthesis_future

        elapsed = time.perf_counter() - t0

        if self._log_callback is not None and total_samples > 0:
            audio_duration = total_samples / self._target_sr
            rtf = elapsed / audio_duration
            self._log_callback({"type": "log", "msg": f"RTF: {rtf:.3f}  ({elapsed*1000:.0f}ms for {audio_duration:.2f}s audio)"})
            if self._verbose:
                ttfc_str = f"{(first_chunk_ts[0] - t0)*1000:.0f}ms" if first_chunk_ts else "n/a"
                print(f"[TTS] '{sentence[:40]}' TTFC={ttfc_str} RTF={rtf:.3f}")

    # ------------------------------------------------------------------
    # Standalone (non-pipecat) usage
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        chunk_callback=None,
    ) -> tuple:
        """Direct synthesis without pipecat.  Returns (audio_np, sample_rate)."""
        self._ensure_loaded()
        return self._tts.synthesize(
            text,
            speaker=self._speaker,
            language=self._language,
            chunk_callback=chunk_callback,
        )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _to_pcm16(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Convert float32 audio to int16 PCM, resampling if necessary."""
    if src_sr != dst_sr:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr)
        except ImportError:
            pass  # skip resampling if librosa unavailable

    # Clip and convert to int16.
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)
