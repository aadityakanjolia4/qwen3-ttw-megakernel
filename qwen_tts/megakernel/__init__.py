"""Megakernel-accelerated talker decoder for Qwen3-TTS."""

from .talker_decoder import TalkerDecoder
from .talker_weights import load_talker_weights

__all__ = ["TalkerDecoder", "load_talker_weights"]
