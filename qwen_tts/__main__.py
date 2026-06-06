import argparse
import sys
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qwen_tts",
        description="Qwen3-TTS command-line inference.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--language", default="English", help="Language (default: English).")
    parser.add_argument(
        "--mode",
        choices=["custom", "design", "clone"],
        default="custom",
        help=(
            "Synthesis mode:\n"
            "  custom  — predefined speaker (default)\n"
            "  design  — describe the voice with --instruct\n"
            "  clone   — clone a voice from --ref-audio\n"
        ),
    )
    parser.add_argument("--speaker", default="aiden", help="Speaker name for --mode custom (default: Chelsie).")
    parser.add_argument("--instruct", default="", help="Voice description for --mode design/custom.")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path/URL for --mode clone.")
    parser.add_argument("--ref-text", default=None, help="Transcript of the reference audio for --mode clone.")
    parser.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"], help="Model size (default: 0.6B).")
    parser.add_argument("--model", default=None, help="HuggingFace repo or local path (overrides --size).")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0).")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable FlashAttention-2.")
    parser.add_argument("--output", default="output.wav", help="Output .wav file path (default: output.wav).")
    return parser


_DEFAULT_MODELS = {
    ("custom", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("design", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
    ("clone",  "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("custom", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("clone",  "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    model_path = args.model or _DEFAULT_MODELS[(args.mode, args.size)]
    attn = "flash_attention_2" if not args.no_flash_attn else "eager"

    print(f"Loading model: {model_path}")
    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=args.device,
        dtype=_DTYPE_MAP[args.dtype],
    )

    if args.mode == "custom":
        wavs, sr = tts.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct or None,
        )
    elif args.mode == "design":
        wavs, sr = tts.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct,
        )
    elif args.mode == "clone":
        if not args.ref_audio:
            parser.error("--mode clone requires --ref-audio")
        wavs, sr = tts.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text or "",
        )

    sf.write(args.output, wavs[0], sr)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
