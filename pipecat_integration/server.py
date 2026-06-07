"""FastAPI server — WebRTC signaling + pipeline runner.

Run on Vast.ai:
  pip install -e ".[webrtc]"
  python -m pipecat_integration.server --port 8080

Then open  http://<vast-ip>:8080  in your browser.
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCParams, SmallWebRTCTransport

from pipecat_integration.megakernel_tts_service import MegakernelTTSService
from pipecat_integration.qwen3_llm_service import Qwen3LLMService, SentenceSplitter

# ---------------------------------------------------------------------------
# Global config (set by CLI args before uvicorn starts)
# ---------------------------------------------------------------------------
_cfg: dict = {}

# Pre-loaded model singletons — populated in lifespan, reused per connection.
_loaded_tts = None       # StreamingTTSMegakernel
_loaded_llm_model = None
_loaded_llm_tokenizer = None


# ---------------------------------------------------------------------------
# Startup: download + load all models before accepting connections
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
    """Pre-download Whisper model files so per-connection load is instant."""
    try:
        from faster_whisper import WhisperModel
        print(f"[Startup] Downloading Whisper '{model_name}' ...")
        # CPU + int8 is the cheapest way to trigger the file download.
        _probe = WhisperModel(model_name, device="cpu", compute_type="int8")
        del _probe
        print("[Startup] Whisper downloaded.")
    except Exception as exc:
        print(f"[Startup] Whisper pre-download skipped: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_tts, _loaded_llm_model, _loaded_llm_tokenizer

    loop = asyncio.get_event_loop()

    # Run blocking model loads in a thread pool so the event loop stays alive.
    print("\n[Startup] Loading all models — this runs once at boot.\n")

    _loaded_tts = await loop.run_in_executor(None, _load_tts)
    _loaded_llm_model, _loaded_llm_tokenizer = await loop.run_in_executor(None, _load_llm)
    await loop.run_in_executor(None, _download_whisper, _cfg["whisper_model"])

    print("\n[Startup] All models ready — accepting connections.\n")
    yield
    # (no teardown needed)


app = FastAPI(lifespan=lifespan)

CLIENT_HTML = Path(__file__).parent.parent / "client" / "index.html"


@app.get("/health")
async def health():
    import torch
    cuda = torch.cuda.is_available()
    llm_ready = _loaded_llm_model is not None
    tts_ready = _loaded_tts is not None
    # In CPU-only mode TTS is intentionally skipped — still "ok" for local testing.
    all_ready = llm_ready and (tts_ready or not cuda)
    return JSONResponse({
        "status": "ok" if all_ready else "loading",
        "tts": tts_ready,
        "llm": llm_ready,
        "whisper": _cfg.get("whisper_model", "unknown"),
        "cuda": cuda,
        "mode": "gpu" if cuda else "cpu-only (TTS disabled)",
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    return CLIENT_HTML.read_text()


STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
]


@app.post("/offer")
async def offer(request: Request):
    data = await request.json()

    connection = SmallWebRTCConnection(ice_servers=STUN_SERVERS)
    await connection.initialize(sdp=data["sdp"], type=data["type"])

    asyncio.create_task(_run_pipeline(connection))

    return JSONResponse(connection.get_answer())


async def _run_pipeline(connection: SmallWebRTCConnection):
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=SmallWebRTCParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    sample_rate = 16000  # SmallWebRTC default (16 kHz mono PCM)

    # Whisper model files are already cached from startup; this load is fast.
    stt = WhisperSTTService(settings=WhisperSTTService.Settings(model=_cfg["whisper_model"]))

    # Pass pre-loaded model weights — no download or GPU init on connect.
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

    user_agg = LLMUserResponseAggregator(messages)
    assistant_agg = LLMAssistantResponseAggregator(messages)

    if _loaded_tts is not None:
        tts = MegakernelTTSService(
            speaker=_cfg["speaker"],
            sample_rate=sample_rate,
            verbose=False,
            tts_instance=_loaded_tts,
        )
        stages = [
            transport.input(), stt, user_agg, llm, splitter,
            tts, transport.output(), assistant_agg,
        ]
    else:
        # CPU-only mode: pipeline runs without TTS (STT + LLM only).
        print("[Pipeline] Running in CPU-only mode — no TTS output.")
        stages = [
            transport.input(), stt, user_agg, llm, splitter,
            transport.output(), assistant_agg,
        ]

    pipeline = Pipeline(stages)

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))
    runner = PipelineRunner()

    print("WebRTC client connected — pipeline running.")
    await runner.run(task)
    print("WebRTC client disconnected.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS megakernel WebRTC server")
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
        f"\nStarting WebRTC server\n"
        f"  STT : Faster-Whisper {args.whisper_model}\n"
        f"  LLM : Qwen3-0.6B-Instruct\n"
        f"  TTS : Megakernel {args.tts_model} speaker={args.speaker}\n\n"
        f"  Open in browser:  http://{args.host}:{args.port}\n"
        f"  (Use the Vast.ai public IP/port if running remotely)\n"
    )

    uvicorn.run(app, host=args.host, port=args.port)
