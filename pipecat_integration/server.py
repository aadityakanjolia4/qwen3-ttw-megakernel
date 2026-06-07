"""FastAPI server — WebSocket audio pipeline.

Works through SSH tunnels and Vast.ai HTTP proxy (TCP only, no UDP needed).

Run on Vast.ai:
  pip install -e ".[webrtc]"
  python -m pipecat_integration.server --port 8080

Then open  http://localhost:8080  (via SSH tunnel) in your browser.
"""

import argparse
import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.workers.task import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from pipecat_integration.megakernel_tts_service import MegakernelTTSService
from pipecat_integration.qwen3_llm_service import LLMUserContextAggregator, Qwen3LLMService, SentenceSplitter


# ---------------------------------------------------------------------------
# WebSocket channel — thin wrapper so TTS service / logger can send JSON msgs
# ---------------------------------------------------------------------------

class WsChannel:
    def __init__(self, ws: WebSocket):
        self._ws = ws

    def send_app_message(self, msg: dict):
        asyncio.create_task(self._send(msg))

    async def _send(self, msg: dict):
        try:
            await self._ws.send_text(json.dumps(msg))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pipeline observer — forwards log/transcript events to the browser
# ---------------------------------------------------------------------------

class PipelineLogger(BaseObserver):
    def __init__(self, channel: WsChannel, timing: dict):
        super().__init__()
        self._ch = channel
        self._timing = timing
        self._llm_buf: list[str] = []
        self._seen_ids: set[int] = set()

    def _log(self, msg: str):
        self._ch.send_app_message({"type": "log", "msg": msg})

    def _transcript(self, role: str, text: str):
        self._ch.send_app_message({"type": "transcript", "role": role, "text": text})

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        fid = id(frame)
        if fid in self._seen_ids:
            return
        self._seen_ids.add(fid)
        if len(self._seen_ids) > 2000:
            self._seen_ids.clear()

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._timing["vad_end_ts"] = time.perf_counter()

        elif isinstance(frame, TranscriptionFrame):
            self._log(f"STT: \"{frame.text}\"")
            self._transcript("user", frame.text)

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._log("LLM: generating response")
            self._llm_buf = []

        elif isinstance(frame, TextFrame):
            self._llm_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            full = "".join(self._llm_buf).strip()
            if full:
                self._transcript("assistant", full)
            self._llm_buf = []

        elif isinstance(frame, TTSStartedFrame):
            self._log("TTS: synthesis started")

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
        print("[Startup] No CUDA GPU — TTS skipped (CPU-only mode).")
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_tts, _loaded_llm_model, _loaded_llm_tokenizer

    loop = asyncio.get_event_loop()
    print("\n[Startup] Loading all models — this runs once at boot.\n")

    _loaded_tts = await loop.run_in_executor(None, _load_tts)
    _loaded_llm_model, _loaded_llm_tokenizer = await loop.run_in_executor(None, _load_llm)
    await loop.run_in_executor(None, _download_whisper, _cfg["whisper_model"])

    print("\n[Startup] All models ready — accepting connections.\n")
    yield


app = FastAPI(lifespan=lifespan)

CLIENT_HTML = Path(__file__).parent.parent / "client" / "index.html"


@app.get("/health")
async def health():
    import torch
    cuda = torch.cuda.is_available()
    llm_ready = _loaded_llm_model is not None
    tts_ready = _loaded_tts is not None
    all_ready = llm_ready and (tts_ready or not cuda)
    return JSONResponse({
        "status": "ok" if all_ready else "loading",
        "tts": tts_ready,
        "llm": llm_ready,
        "cuda": cuda,
        "mode": "gpu" if cuda else "cpu-only (TTS disabled)",
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    return CLIENT_HTML.read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await _run_pipeline(websocket)
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as exc:
        print(f"Pipeline error: {exc}")


async def _run_pipeline(websocket: WebSocket):
    channel = WsChannel(websocket)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    sample_rate = 16000

    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=_cfg["whisper_model"]))

    llm = Qwen3LLMService(
        max_new_tokens=120,
        temperature=0.7,
        enable_thinking=False,
        model=_loaded_llm_model,
        tokenizer=_loaded_llm_tokenizer,
    )

    splitter = SentenceSplitter()

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

    user_agg = LLMUserContextAggregator(messages)
    assistant_agg = LLMFullResponseAggregator()

    timing = {"vad_end_ts": 0.0}
    obs = PipelineLogger(channel, timing)

    if _loaded_tts is not None:
        tts = MegakernelTTSService(
            speaker=_cfg["speaker"],
            sample_rate=sample_rate,
            verbose=False,
            tts_instance=_loaded_tts,
            connection=channel,
            timing=timing,
        )
        stages = [
            transport.input(), stt, user_agg, llm, splitter,
            tts, transport.output(), assistant_agg,
        ]
    else:
        print("[Pipeline] Running in CPU-only mode — no TTS output.")
        stages = [
            transport.input(), stt, user_agg, llm, splitter,
            transport.output(), assistant_agg,
        ]

    pipeline = Pipeline(stages)
    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            observers=[obs],
        ),
    )
    runner = WorkerRunner()
    await runner.add_workers(task)

    print("WebSocket client connected — pipeline running.")
    await runner.run()
    print("WebSocket pipeline finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel WebSocket server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--speaker", default="aiden")
    parser.add_argument("--whisper-model", default="base", help="tiny|base|small|medium|large-v3")
    args = parser.parse_args()

    _cfg.update(
        tts_model=args.tts_model,
        speaker=args.speaker,
        whisper_model=args.whisper_model,
    )

    print(
        f"\nStarting WebSocket server\n"
        f"  STT : Faster-Whisper {args.whisper_model}\n"
        f"  LLM : Qwen3-0.6B-Instruct\n"
        f"  TTS : Megakernel {args.tts_model} speaker={args.speaker}\n\n"
        f"  Open in browser:  http://localhost:{args.port}\n"
        f"  (SSH tunnel: ssh -L {args.port}:localhost:{args.port} user@vast-ip)\n"
    )

    uvicorn.run(app, host=args.host, port=args.port)
