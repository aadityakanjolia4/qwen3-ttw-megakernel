"""FastAPI server — WebSocket transport + pipeline runner.

Run on Vast.ai:
  pip install -e ".[webrtc]"
  python -m pipecat_integration.server --port 8080

Then open  http://<vast-ip>:8080  in your browser.
Works through ngrok / SSH TCP tunnels (WebSocket, no UDP required).
"""

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from pipecat_integration.megakernel_tts_service import MegakernelTTSService
from pipecat_integration.qwen3_llm_service import Qwen3LLMService


# ---------------------------------------------------------------------------
# Custom binary PCM serializer
# ---------------------------------------------------------------------------

class PCMSerializer(FrameSerializer):
    """Binary PCM ↔ pipecat frames.

    Browser → Server: raw little-endian 16-bit PCM at 16 kHz mono
    Server → Browser: raw little-endian 16-bit PCM at 16 kHz mono
    Text messages (JSON): forwarded as-is for control / log / transcript
    """

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes) and data:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
        return None


# ---------------------------------------------------------------------------
# Pipeline observer — sends log/transcript JSON to the browser
# ---------------------------------------------------------------------------

class PipelineLogger(BaseObserver):
    def __init__(self, websocket: WebSocket, timing: dict, tts_instance=None, speaker: str = "aiden"):
        super().__init__()
        self._ws = websocket
        self._timing = timing
        self._tts_instance = tts_instance
        self._speaker = speaker
        self._llm_buf: list[str] = []
        self._seen_ids: set[int] = set()

    def _log(self, msg: str):
        asyncio.create_task(self._send({"type": "log", "msg": msg}))

    def _transcript(self, role: str, text: str):
        asyncio.create_task(self._send({"type": "transcript", "role": role, "text": text}))

    async def _send(self, obj: dict):
        try:
            await self._ws.send_text(json.dumps(obj))
        except Exception:
            pass

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        fid = id(frame)
        if fid in self._seen_ids:
            return
        self._seen_ids.add(fid)
        if len(self._seen_ids) > 2000:
            self._seen_ids.clear()

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._timing["vad_end_ts"] = time.perf_counter()
        elif isinstance(frame, TranscriptionFrame):
            self._log(f"STT: \"{frame.text}\"")
            self._transcript("user", frame.text)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_buf = []
        elif isinstance(frame, TextFrame):
            self._llm_buf.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            full = "".join(self._llm_buf).strip()
            if full:
                self._transcript("assistant", full)
            self._llm_buf = []
        elif isinstance(frame, TTSStartedFrame):
            self._log("TTS: started")
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Keep GPU cache warm so the next turn's vocoder call stays fast
            if self._tts_instance is not None:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _keep_warm, self._tts_instance, self._speaker)
        elif isinstance(frame, MetricsFrame):
            pass


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

_cfg: dict = {}
_loaded_tts = None
_loaded_llm_model = None
_loaded_llm_tokenizer = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_tts():
    import torch
    if not torch.cuda.is_available():
        print("[Startup] No CUDA GPU — TTS skipped.")
        return None
    from qwen_tts.megakernel.streaming_tts import StreamingTTSMegakernel
    return StreamingTTSMegakernel(model_name=_cfg["tts_model"], verbose=True)


def _load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[Startup] Loading LLM {model_name} on {device} ({dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    print("[Startup] LLM ready.")
    return model, tokenizer


def _download_whisper(model_name: str):
    try:
        from faster_whisper import WhisperModel
        print(f"[Startup] Downloading Whisper '{model_name}' ...")
        _probe = WhisperModel(model_name, device="cpu", compute_type="int8")
        del _probe
        print("[Startup] Whisper downloaded.")
    except Exception as exc:
        print(f"[Startup] Whisper pre-download skipped: {exc}")


def _warmup_tts(tts, speaker: str):
    if tts is None:
        return
    print("[Startup] Warming up TTS (compiling CUDA kernels) ...")
    try:
        tts.synthesize("Hello.", speaker=speaker, language="English", chunk_callback=lambda a, s: None)
        print("[Startup] TTS warmup done.")
    except Exception as exc:
        print(f"[Startup] TTS warmup failed (non-fatal): {exc}")


def _keep_warm(tts, speaker: str):
    """Synthesize a short dummy phrase to keep GPU L2 cache warm between turns."""
    if tts is None:
        return
    try:
        tts.synthesize("Okay.", speaker=speaker, language="English", chunk_callback=lambda a, s: None)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_tts, _loaded_llm_model, _loaded_llm_tokenizer

    loop = asyncio.get_event_loop()
    print("\n[Startup] Loading all models — this runs once at boot.\n")

    _loaded_tts = await loop.run_in_executor(None, _load_tts)
    _loaded_llm_model, _loaded_llm_tokenizer = await loop.run_in_executor(None, _load_llm)
    await loop.run_in_executor(None, _download_whisper, _cfg["whisper_model"])
    await loop.run_in_executor(None, _warmup_tts, _loaded_tts, _cfg["speaker"])

    print("\n[Startup] All models ready — accepting connections.\n")
    yield


app = FastAPI(lifespan=lifespan)
CLIENT_HTML = Path(__file__).parent.parent / "client" / "index.html"


@app.get("/health")
async def health():
    import torch
    cuda = torch.cuda.is_available()
    return JSONResponse({
        "status": "ok",
        "tts": _loaded_tts is not None,
        "llm": _loaded_llm_model is not None,
        "cuda": cuda,
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    return CLIENT_HTML.read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected — starting pipeline.")
    try:
        await _run_pipeline(websocket)
    except Exception as exc:
        print(f"[Pipeline] Error: {exc}")
    print("WebSocket client disconnected.")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _make_log_cb(ws: WebSocket):
    def cb(msg: dict):
        asyncio.create_task(ws.send_text(json.dumps(msg)))
    return cb


async def _run_pipeline(websocket: WebSocket):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            serializer=PCMSerializer(sample_rate=16000),
        ),
    )

    sample_rate = 16000

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=_cfg["whisper_model"]))

    system_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant speaking on a phone call. "
                "Keep every reply to one or two short sentences. "
                "Speak naturally — no bullet points, no lists, no markdown, no XML tags, no code."
            ),
        }
    ]

    llm = Qwen3LLMService(
        max_new_tokens=120,
        temperature=0.7,
        enable_thinking=False,
        model=_loaded_llm_model,
        tokenizer=_loaded_llm_tokenizer,
        system_messages=system_messages,
    )

    timing = {"vad_end_ts": 0.0}
    obs = PipelineLogger(websocket, timing, tts_instance=_loaded_tts, speaker=_cfg["speaker"])

    if _loaded_tts is not None:
        tts = MegakernelTTSService(
            speaker=_cfg["speaker"],
            sample_rate=sample_rate,
            verbose=False,
            tts_instance=_loaded_tts,
            log_callback=_make_log_cb(websocket),
            timing=timing,
        )
        stages = [
            transport.input(), vad, stt, llm,
            tts, transport.output(),
        ]
    else:
        print("[Pipeline] CPU-only mode — no TTS.")
        stages = [
            transport.input(), vad, stt, llm, splitter,
            transport.output(),
        ]

    pipeline = Pipeline(stages)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
        enable_rtvi=False,
        observers=[obs],
    )
    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--whisper-model", default="base")
    args = parser.parse_args()

    _cfg.update(
        tts_model=args.tts_model,
        speaker=args.speaker,
        whisper_model=args.whisper_model,
    )

    print(
        f"\nStarting WebSocket server\n"
        f"  STT : Faster-Whisper {args.whisper_model}\n"
        f"  LLM : Qwen3-0.6B\n"
        f"  TTS : {args.tts_model}\n\n"
        f"  Open: http://<vast-ip>:{args.port}\n"
    )
    uvicorn.run(app, host=args.host, port=args.port)
