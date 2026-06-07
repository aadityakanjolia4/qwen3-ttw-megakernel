"""Local Qwen3-Instruct LLM service for Pipecat — no Ollama, no API keys.

Uses HuggingFace TextIteratorStreamer so tokens reach the pipeline immediately.
Combined with sentence-level streaming, TTS starts on the first complete
sentence while the model is still generating the rest.

Key speed tricks
----------------
- Direct HF inference: no Ollama subprocess or HTTP round-trip (~20–50 ms saved)
- /no_think suffix: disables Qwen3's chain-of-thought for voice responses
- max_new_tokens=120: voice answers are short; hard cap prevents runaway gen
- bf16 on CUDA: same dtype as the TTS megakernel weights already in VRAM
"""

import asyncio
import queue as stdlib_queue
import re
import threading
from typing import Optional

import torch

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# LLM service
# ---------------------------------------------------------------------------


class Qwen3LLMService(FrameProcessor):
    """Wraps Qwen3-Instruct with streaming token output for Pipecat.

    Listens for LLMMessagesFrame, generates tokens via TextIteratorStreamer,
    and pushes TextFrame chunks downstream.  Sentence splitting (see below)
    ensures TTS receives whole sentences rather than sub-word fragments.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        enable_thinking: bool = False,
        model=None,
        tokenizer=None,
    ):
        super().__init__()
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._enable_thinking = enable_thinking

        if model is not None and tokenizer is not None:
            # Accept pre-loaded model+tokenizer (passed from server startup).
            self._model = model
            self._tokenizer = tokenizer
            self._device = next(model.parameters()).device.type
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = _best_device()
            dtype = torch.float16 if self._device == "mps" else (
                torch.bfloat16 if self._device == "cuda" else torch.float32
            )

            print(f"[Qwen3LLM] Loading {model_name} on {self._device} ({dtype}) ...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(self._device)
            self._model.eval()
            print("[Qwen3LLM] Ready.")

    # ------------------------------------------------------------------
    # Pipecat FrameProcessor interface
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import LLMContextFrame

        if isinstance(frame, LLMContextFrame):
            await self._generate(frame.context.messages)
        else:
            await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def _generate(self, messages: list):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            TextFrame,
        )
        from transformers import TextIteratorStreamer

        # /no_think disables Qwen3's chain-of-thought — essential for
        # low-latency voice responses.
        if not self._enable_thinking and messages:
            msgs = list(messages)
            last = msgs[-1]
            if last.get("role") == "user" and "/no_think" not in last["content"]:
                msgs[-1] = {**last, "content": last["content"] + " /no_think"}
            messages = msgs

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            do_sample=self._temperature > 0,
        )

        token_queue: stdlib_queue.Queue = stdlib_queue.Queue()

        def _run():
            try:
                self._model.generate(**gen_kwargs)
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        await self.push_frame(LLMFullResponseStartFrame())

        while True:
            token: Optional[str] = await loop.run_in_executor(None, token_queue.get)
            if token is None:
                break
            if token:
                await self.push_frame(TextFrame(token))

        await self.push_frame(LLMFullResponseEndFrame())
        thread.join()


# ---------------------------------------------------------------------------
# Sentence splitter — pushes complete sentences to TTS immediately
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


class SentenceSplitter(FrameProcessor):
    """Buffers streaming TextFrames and pushes each complete sentence at once.

    Without this, the TTS service receives individual sub-word tokens and
    either buffers everything (high latency) or tries to synthesise fragments
    (bad audio).  With sentence splitting the TTS starts on the first complete
    sentence while the LLM is still generating the second.
    """

    def __init__(self):
        super().__init__()
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import LLMFullResponseEndFrame, TextFrame

        if isinstance(frame, TextFrame):
            self._buffer += frame.text
            sentences, self._buffer = _split_complete(self._buffer)
            for s in sentences:
                s = s.strip()
                if s:
                    await self.push_frame(TextFrame(s))
        elif isinstance(frame, LLMFullResponseEndFrame):
            tail = self._buffer.strip()
            if tail:
                await self.push_frame(TextFrame(tail))
            self._buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


def _split_complete(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, leftover_buffer)."""
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [], text
    return parts[:-1], parts[-1]


# ---------------------------------------------------------------------------
# User context aggregator — pipecat 1.3.0 removed LLMUserResponseAggregator
# ---------------------------------------------------------------------------


class LLMUserContextAggregator(FrameProcessor):
    """Sends system prompt + current user turn only (no history) for low latency."""

    def __init__(self, messages: list):
        super().__init__()
        self._system_messages = [m for m in messages if m.get("role") == "system"]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import LLMContext, LLMContextFrame, TranscriptionFrame

        if isinstance(frame, TranscriptionFrame):
            msgs = self._system_messages + [{"role": "user", "content": frame.text}]
            await self.push_frame(LLMContextFrame(context=LLMContext(messages=msgs)))
        else:
            await self.push_frame(frame, direction)
