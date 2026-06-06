

import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = tts.generate_custom_voice(
    text="Good morning.",
    speaker="alloy",
    language="English",
)
sf.write("output.wav", wavs[0], sr)

