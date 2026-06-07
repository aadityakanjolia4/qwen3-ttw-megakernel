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
    from pipecat.frames.frames import AudioRawFrame, EndFrame, Frame, TTSStartedFrame, TTSStoppedFrame
    from pipecat.services.ai_services import TTSService

    _PIPECAT_AVAILABLE = True
except ImportError:
    _PIPECAT_AVAILABLE = False

    class TTSService:  # type: ignore[no-redef]
        """Stub for environments without pipecat installed."""
        pass

    class AudioRawFrame:  # type: ignore[no-redef]
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
    ):
        if _PIPECAT_AVAILABLE:
            super().__init__()

        self._speaker = speaker
        self._language = language
        self._target_sr = sample_rate
        self._verbose = verbose
        self._model_name = model_name

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

    async def run_tts(self, sentence: str) -> AsyncGenerator:
        """Synthesize sentence and yield AudioRawFrame chunks as they arrive.

        This is called by pipecat's pipeline per TTS sentence segment.
        """
        self._ensure_loaded()

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def on_chunk(audio: np.ndarray, sr: int) -> None:
            """Callback from synthesis worker → push frame to async queue."""
            pcm = _to_pcm16(audio, src_sr=sr, dst_sr=self._target_sr)
            frame = AudioRawFrame(
                audio=pcm.tobytes(),
                sample_rate=self._target_sr,
                num_channels=1,
            )
            # Thread-safe put (synthesis runs in executor thread).
            asyncio.run_coroutine_threadsafe(queue.put(frame), loop)

        if _PIPECAT_AVAILABLE:
            yield TTSStartedFrame()

        # Run blocking synthesis in a thread pool executor so we don't block
        # the asyncio event loop.
        t0 = time.perf_counter()
        synthesis_future = loop.run_in_executor(
            None,
            self._tts.synthesize,
            sentence,
            self._speaker,
            self._language,
            on_chunk,
        )

        # Drain queue as chunks arrive, until synthesis is done.
        done = False
        while not done:
            try:
                # Poll queue with short timeout.
                frame = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield frame
            except asyncio.TimeoutError:
                if synthesis_future.done():
                    done = True

        # Drain any remaining frames that arrived right at the end.
        while not queue.empty():
            yield await queue.get()

        if _PIPECAT_AVAILABLE:
            yield TTSStoppedFrame()

        if self._verbose:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[MegakernelTTSService] '{sentence[:40]}' → {elapsed:.0f} ms total")

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
