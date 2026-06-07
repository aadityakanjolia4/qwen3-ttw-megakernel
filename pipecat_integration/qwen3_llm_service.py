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


# ---------------------------------------------------------------------------
# LLM service
# ---------------------------------------------------------------------------


class Qwen3LLMService:
    """Wraps Qwen3-Instruct with streaming token output for Pipecat.

    Implements pipecat's FrameProcessor protocol.  Listens for
    LLMMessagesFrame, generates tokens via TextIteratorStreamer, and pushes
    TextFrame chunks downstream.  Sentence splitting (see below) ensures TTS
    receives whole sentences rather than sub-word fragments.
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

        # Reuse a single push frame function once wired into the pipeline.
        self._push_frame = None

    # ------------------------------------------------------------------
    # Pipecat FrameProcessor interface
    # ------------------------------------------------------------------

    async def process_frame(self, frame, direction):
        from pipecat.frames.frames import (
            LLMFullResponseEndFrame,
            LLMFullResponseStartFrame,
            LLMMessagesFrame,
            TextFrame,
        )

        if isinstance(frame, LLMMessagesFrame):
            await self._generate(frame.messages)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame, direction=None):
        if self._push_frame is not None:
            if direction is not None:
                await self._push_frame(frame, direction)
            else:
                await self._push_frame(frame)

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
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")

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

        # Run generate() in a background thread; pull tokens via a queue
        # so we don't block the asyncio event loop.
        token_queue: stdlib_queue.Queue = stdlib_queue.Queue()

        def _run():
            try:
                self._model.generate(**gen_kwargs)
            finally:
                token_queue.put(None)  # sentinel

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        await self.push_frame(LLMFullResponseStartFrame())

        while True:
            token: Optional[str] = await loop.run_in_executor(
                None, token_queue.get
            )
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


class SentenceSplitter:
    """Buffers streaming TextFrames and pushes each complete sentence at once.

    Without this, the TTS service receives individual sub-word tokens and
    either buffers everything (high latency) or tries to synthesise fragments
    (bad audio).  With sentence splitting the TTS starts on the first complete
    sentence while the LLM is still generating the second.
    """

    def __init__(self):
        self._buffer = ""
        self._push_frame = None

    async def process_frame(self, frame, direction):
        from pipecat.frames.frames import LLMFullResponseEndFrame, TextFrame

        if isinstance(frame, TextFrame):
            self._buffer += frame.text
            sentences, self._buffer = _split_complete(self._buffer)
            for s in sentences:
                s = s.strip()
                if s:
                    await self.push_frame(TextFrame(s))
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Flush whatever remains.
            tail = self._buffer.strip()
            if tail:
                await self.push_frame(TextFrame(tail))
            self._buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def push_frame(self, frame, direction=None):
        if self._push_frame is not None:
            if direction is not None:
                await self._push_frame(frame, direction)
            else:
                await self._push_frame(frame)


def _split_complete(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, leftover_buffer)."""
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [], text
    # Last element may be an incomplete sentence — keep it in the buffer.
    return parts[:-1], parts[-1]
