"""Streaming TTS synthesis using the megakernel talker backbone.

Architecture
------------
Prefill (HF model):
  Text tokens + speaker prompt + codec BOS → KV cache + first hidden state.

Decode loop (per audio frame):
  1. Compute inputs_embeds in Python:
       embed = sum(codec_group_embeds) + text_conditioning
  2. Megakernel step_embed() → (codec_token_0, hidden_state)   ← talker (28L)
  3. CodePredictorKernel.predict() → codec_tokens_1..N-1        ← kernel (5L)
  4. Collect all N codec groups → audio_codes frame
  5. Every CHUNK_FRAMES frames, decode audio via vocoder and yield.

Performance targets (RTX 5090, Qwen3-TTS-12Hz-0.6B):
  TTFC    < 60 ms   (prefill + first vocoder decode)
  RTF     < 0.15    (1 s audio generated in < 150 ms)
"""

import time
from typing import Callable, Generator, Optional

import numpy as np
import torch

from .talker_decoder import CodePredictorKernel, TalkerDecoder
from .talker_weights import TALKER_MAX_SEQ_LEN, load_code_predictor_weights, load_talker_weights

# Vocoder decode is called every CHUNK_FRAMES codec frames.
# At 12 Hz, 1 frame = 83 ms of audio → minimum TTFC.
CHUNK_FRAMES = 1

# Context frames prepended to each vocoder call to eliminate boundary artifacts.
# The neural vocoder has a large receptive field; decoding single frames in isolation
# causes edge transients (noise/clicks at word boundaries). Including OVERLAP_FRAMES
# of prior context gives the model the signal it needs — those samples are trimmed
# from the output before yielding.
OVERLAP_FRAMES = 4


class StreamingTTSMegakernel:
    """Streaming TTS with megakernel-accelerated talker backbone.

    Parameters
    ----------
    model_name : str
        HuggingFace repo id (or local path) of the Qwen3-TTS model.
    verbose : bool
        Print load progress.
    max_new_tokens : int
        Maximum number of codec frames to generate.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        verbose: bool = True,
        max_new_tokens: int = 2048,
    ):
        self.max_new_tokens = max_new_tokens

        # Load HF model + extract talker weights.
        weights, self._hf_model, self._talker_cfg = load_talker_weights(
            model_name, verbose=verbose
        )

        self._talker = self._hf_model.talker
        self._backbone = self._talker.model
        self._speech_tokenizer = self._hf_model.speech_tokenizer

        self._num_code_groups = self._talker_cfg.num_code_groups

        # Cache processor so _build_prefill_embeds doesn't reload it every call.
        from transformers import AutoProcessor
        _model_id = (
            self._hf_model.config._name_or_path
            if hasattr(self._hf_model.config, "_name_or_path") and self._hf_model.config._name_or_path
            else model_name
        )
        try:
            self._processor = AutoProcessor.from_pretrained(_model_id, local_files_only=True, fix_mistral_regex=True)
        except (EnvironmentError, TypeError):
            self._processor = AutoProcessor.from_pretrained(_model_id, fix_mistral_regex=True)

        # Build megakernel decoder from the same weights.
        self._mk_decoder = TalkerDecoder(weights)

        # Build megakernel code predictor (replaces HF DynamicCache loop).
        cp_weights = load_code_predictor_weights(self._hf_model)
        self._cp_kernel = CodePredictorKernel(cp_weights)
        # Codec embedding tables for groups 1..N-1 (used when building next-frame input).
        self._cp_codec_embed_weights = cp_weights["codec_embedding_weights"]

        # Token IDs from talker config.
        self._codec_eos_id = self._talker_cfg.codec_eos_token_id
        self._codec_bos_id = self._talker_cfg.codec_bos_id
        self._codec_nothink_id = self._talker_cfg.codec_nothink_id
        self._codec_think_bos_id = self._talker_cfg.codec_think_bos_id
        self._codec_think_eos_id = self._talker_cfg.codec_think_eos_id
        self._codec_pad_id = self._talker_cfg.codec_pad_id

        device = next(self._talker.parameters()).device
        dtype  = next(self._talker.parameters()).dtype

        # Pre-compute static decode embeds (reused every synthesis call).
        with torch.no_grad():
            cfg_ = self._hf_model.config
            _tts_tok = torch.tensor(
                [[cfg_.tts_bos_token_id, cfg_.tts_eos_token_id, cfg_.tts_pad_token_id]],
                device=device, dtype=torch.long,
            )
            _bos_e, _eos_e, _pad_e = self._talker.text_projection(
                self._backbone.text_embedding(_tts_tok)
            ).to(dtype).chunk(3, dim=1)
            self._tts_bos_embed = _bos_e.detach()   # [1, 1, D]
            self._tts_eos_embed = _eos_e.detach()   # [1, 1, D]
            self._tts_pad_embed = _pad_e.detach()   # [1, 1, D]

            self._codec_bos_embed = self._backbone.codec_embedding(
                torch.tensor([[self._codec_bos_id]], device=device, dtype=torch.long)
            ).squeeze(1).to(dtype).detach()  # [1, D]

        # Compile the vocoder decoder with CUDA graph capture (reduce-overhead mode).
        # The 12Hz transformer decoder has ~40 kernel launches per call; torch.compile
        # fuses them into a single CUDA graph replay, cutting per-call overhead from
        # ~75ms to <10ms. Warmup runs all overlap sizes (1..1+OVERLAP_FRAMES) so every
        # shape is compiled before the first real request.
        if torch.cuda.is_available():
            try:
                self._speech_tokenizer.model.decoder = torch.compile(
                    self._speech_tokenizer.model.decoder,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                if verbose:
                    print("Vocoder decoder compiled (torch.compile reduce-overhead)")
            except Exception as _e:
                if verbose:
                    print(f"torch.compile skipped: {_e}")

        # Warm every input shape the decode loop will use (T = 1 .. 1+OVERLAP_FRAMES).
        # First call triggers compilation per shape; subsequent calls hit the CUDA graph.
        if verbose:
            print("Warming vocoder for all overlap sizes ...")
        for _nf in range(1, OVERLAP_FRAMES + 2):
            _dummy = torch.zeros(_nf, self._num_code_groups, dtype=torch.long, device=device)
            _warmup_audio, _ = self._decode_audio_chunk(_dummy)
        self._samples_per_frame = len(_warmup_audio) // (OVERLAP_FRAMES + 1)
        torch.cuda.synchronize()

        if verbose:
            print(
                f"Megakernel TTS ready — talker {self._talker_cfg.num_hidden_layers}L "
                f"(megakernel) + code predictor {self._talker_cfg.code_predictor_config.num_hidden_layers}L "
                f"(megakernel) [vocoder compiled + warmed]"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = "aiden",
        language: str = "English",
        chunk_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        stats_out: Optional[dict] = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to speech, optionally streaming chunks.

        Parameters
        ----------
        text : str
        speaker : str, optional
            Speaker name for CustomVoice models.
        language : str
        chunk_callback : callable, optional
            Called with (audio_chunk: np.ndarray, sample_rate: int) as each
            chunk becomes available.  Use for real-time streaming.
        do_sample : bool
            If True, use sampling for sub-talker (megakernel always uses argmax
            for the main talker step — see README for details).

        Returns
        -------
        audio : np.ndarray
            Full synthesised audio waveform.
        sample_rate : int
        """
        t_start = time.perf_counter()

        # Estimate max frames from word count (12 Hz, ~13 frames/word, 2s padding).
        word_count = max(1, len(text.split()))
        max_steps = min(self.max_new_tokens, word_count * 13 + 24)

        device = next(self._talker.parameters()).device
        dtype = next(self._talker.parameters()).dtype

        # ---- 1. Build prefill embeddings (same as HF model) ----------------
        prefill_embeds, trailing_text_hidden = (
            self._build_prefill_embeds(text, speaker, language, device, dtype)
        )
        t_build_embeds = time.perf_counter()

        # ---- 2. Megakernel prefill (replaces HF backbone forward) -----------
        prefill_len = prefill_embeds.shape[1]

        self._mk_decoder.soft_reset()
        self._mk_decoder.prefill_embeds(prefill_embeds.squeeze(0))  # [T, hidden]
        t_kv_inject = time.perf_counter()

        print(
            f"[TTFC breakdown] build_embeds={1000*(t_build_embeds-t_start):.1f}ms "
            f"mk_prefill={1000*(t_kv_inject-t_build_embeds):.1f}ms "
            f"prefill_tokens={prefill_len}"
        )

        t_prefill_done = t_kv_inject

        # ---- 3. Decode loop --------------------------------------------------
        codec_frames = []  # list of [num_code_groups] tensors
        audio_chunks = []

        # First decode step uses the hidden state from prefill.
        # Subsequent steps use the hidden state from the megakernel.

        codec_bos_embed = self._codec_bos_embed

        last_codec_groups = None  # Will be set after first step

        ttfc_reported = False
        generation_step = 0
        t_decode_accum = 0.0
        t_vocoder_accum = 0.0

        for step in range(max_steps):
            # Compute inputs_embeds for this decode step.
            if last_codec_groups is None:
                # Very first step: use codec_bos embedding.
                step_embed = codec_bos_embed  # [1, hidden]
            else:
                # Standard step: sum of all code group embeddings.
                step_embed = self._backbone.codec_embedding(
                    last_codec_groups[:1]
                )  # group 0 embed [1, hidden]
                for g in range(1, self._num_code_groups):
                    emb_g = torch.nn.functional.embedding(
                        last_codec_groups[g : g + 1],
                        self._cp_codec_embed_weights[g - 1],
                    )  # [1, hidden]
                    step_embed = step_embed + emb_g

            # Add text conditioning.
            if generation_step < trailing_text_hidden.shape[1]:
                step_embed = (
                    step_embed + trailing_text_hidden[0, generation_step]
                )
            else:
                step_embed = step_embed + self._tts_pad_embed.squeeze()

            # Megakernel step: fast transformer backbone.
            _t0 = time.perf_counter()
            codec_token_0, mk_hidden = self._mk_decoder.step_embed(
                step_embed.to(torch.bfloat16)
            )

            if codec_token_0 == self._codec_eos_id:
                break

            # Code predictor: generate remaining codec groups via megakernel.
            last_id_hidden = self._backbone.codec_embedding(
                torch.tensor([[codec_token_0]], device=device, dtype=torch.long)
            )  # [1, 1, hidden]

            sub_input = torch.cat([mk_hidden.to(dtype), last_id_hidden], dim=1)
            extra_groups = self._cp_kernel.predict(
                sub_input, do_sample, top_k, top_p, temperature
            )  # [1, num_code_groups-1]
            t_decode_accum += time.perf_counter() - _t0
            all_groups = torch.cat(
                [
                    torch.tensor([[codec_token_0]], device=device),
                    extra_groups,
                ],
                dim=-1,
            ).squeeze(0)  # [num_code_groups]

            codec_frames.append(all_groups)
            last_codec_groups = all_groups
            generation_step += 1

            # Yield audio chunk every CHUNK_FRAMES frames.
            if len(codec_frames) % CHUNK_FRAMES == 0:
                # Include OVERLAP_FRAMES of prior context so the vocoder has
                # signal from neighboring frames; boundary transients are eliminated.
                ctx_start = max(0, len(codec_frames) - CHUNK_FRAMES - OVERLAP_FRAMES)
                chunk_codes = torch.stack(
                    codec_frames[ctx_start:], dim=0
                )  # [ctx + CHUNK_FRAMES, num_code_groups]
                _tv0 = time.perf_counter()
                audio_with_ctx, sr = self._decode_audio_chunk(chunk_codes)
                t_vocoder_accum += time.perf_counter() - _tv0
                ctx_frames = len(codec_frames) - CHUNK_FRAMES - ctx_start
                audio_chunk = audio_with_ctx[ctx_frames * self._samples_per_frame:]

                if not ttfc_reported:
                    ttfc_ms = (time.perf_counter() - t_start) * 1000
                    decode_ms = t_decode_accum * 1000
                    vocoder_ms = t_vocoder_accum * 1000
                    print(
                        f"[Megakernel TTS] TTFC: {ttfc_ms:.1f} ms  "
                        f"(decode={decode_ms:.1f}ms vocoder={vocoder_ms:.1f}ms)"
                    )
                    ttfc_reported = True

                if chunk_callback is not None:
                    chunk_callback(audio_chunk, sr)
                audio_chunks.append(audio_chunk)

        # Decode remaining frames (< CHUNK_FRAMES).
        remainder = len(codec_frames) % CHUNK_FRAMES
        if remainder > 0:
            ctx_start = max(0, len(codec_frames) - remainder - OVERLAP_FRAMES)
            chunk_codes = torch.stack(codec_frames[ctx_start:], dim=0)
            audio_with_ctx, sr = self._decode_audio_chunk(chunk_codes)
            ctx_frames = len(codec_frames) - remainder - ctx_start
            audio_chunk = audio_with_ctx[ctx_frames * self._samples_per_frame:]
            if chunk_callback is not None:
                chunk_callback(audio_chunk, sr)
            audio_chunks.append(audio_chunk)

        t_end = time.perf_counter()
        total_audio_s = len(codec_frames) / 12.0  # 12 Hz codec
        wall_s = t_end - t_start
        rtf = wall_s / max(total_audio_s, 1e-6)
        decode_tps = len(codec_frames) / t_decode_accum if t_decode_accum > 0 else 0
        print(
            f"[Megakernel TTS] {len(codec_frames)} frames | "
            f"decode={decode_tps:.0f} tok/s | "
            f"wall={wall_s*1000:.0f} ms | audio={total_audio_s*1000:.0f} ms | RTF={rtf:.3f}"
        )

        if stats_out is not None:
            stats_out["frames"] = len(codec_frames)
            stats_out["decode_tps"] = decode_tps
            stats_out["rtf"] = rtf
            stats_out["wall_ms"] = wall_s * 1000
            stats_out["audio_ms"] = total_audio_s * 1000

        full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0)
        return full_audio, sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prefill_embeds(self, text, speaker, language, device, dtype):
        """Build prefill inputs_embeds and trailing text conditioning.

        Mirrors Qwen3TTSForConditionalGeneration.generate() streaming prefill exactly:
          role(3) + interleaved(tts_pad*N + tts_bos + codec_prefix[:-1]) + (text_tok_0 + codec_bos)
        trailing_text_hidden = text_proj(text[4:-5]) + tts_eos_embed
        """
        text_str = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._processor(text=text_str, return_tensors="pt", padding=True)[
            "input_ids"
        ].to(device)  # [1, T]

        talker = self._talker
        backbone = self._backbone
        cfg = self._hf_model.config
        tcfg = self._talker_cfg

        # Use pre-computed static embeds (no GPU op needed).
        tts_bos_embed = self._tts_bos_embed
        tts_eos_embed = self._tts_eos_embed
        tts_pad_embed = self._tts_pad_embed

        # Speaker embed (via codec embedding — same as HF model).
        speaker_embed = None
        if speaker is not None and speaker.lower() in (tcfg.spk_id or {}):
            spk_id_val = tcfg.spk_id[speaker.lower()]
            speaker_embed = talker.get_input_embeddings()(
                torch.tensor(spk_id_val, device=device, dtype=torch.long)
            )  # [D] or [N, D]

        # Language id.
        codec_language_id = (tcfg.codec_language_id or {}).get(language.lower(), None)

        # Codec prefix tokens (nothink path or language path).
        if codec_language_id is None:
            codec_prefill = [[tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id]]
        else:
            codec_prefill = [[getattr(tcfg, "codec_think_id", tcfg.codec_nothink_id),
                              tcfg.codec_think_bos_id, codec_language_id, tcfg.codec_think_eos_id]]

        codec_emb_0 = talker.get_input_embeddings()(
            torch.tensor(codec_prefill, device=device, dtype=torch.long)
        )  # [1, 3 or 4, D]
        codec_emb_1 = talker.get_input_embeddings()(
            torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]], device=device, dtype=torch.long)
        )  # [1, 2, D]

        if speaker_embed is not None:
            codec_input_embedding = torch.cat(
                [codec_emb_0, speaker_embed.view(1, 1, -1), codec_emb_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat([codec_emb_0, codec_emb_1], dim=1)

        # Role embed: first 3 tokens (<|im_start|>assistant\n).
        role_embed = talker.text_projection(
            backbone.text_embedding(input_ids[:, :3])
        )  # [1, 3, D]

        # Interleaved block: (tts_pad × (len-2) + tts_bos) + codec_prefix[:-1].
        n_combined = codec_input_embedding.shape[1] - 2
        pad_bos_block = torch.cat(
            [tts_pad_embed.expand(-1, n_combined, -1), tts_bos_embed], dim=1
        )  # [1, n_combined+1, D]
        combined_block = pad_bos_block + codec_input_embedding[:, :-1]  # [1, n_combined+1, D]

        # First text token fused with codec_bos.
        first_text = talker.text_projection(
            backbone.text_embedding(input_ids[:, 3:4])
        )  # [1, 1, D]
        first_combined = first_text + codec_input_embedding[:, -1:]  # [1, 1, D]

        prefill_embeds = torch.cat([role_embed, combined_block, first_combined], dim=1)

        # Trailing text: tokens 4..-5 projected + tts_eos appended.
        # Guards against very short texts where 4:-5 would be empty.
        body_ids = input_ids[:, 4:-5]
        if body_ids.shape[1] > 0:
            body_embed = talker.text_projection(backbone.text_embedding(body_ids))
            trailing_text_hidden = torch.cat([body_embed, tts_eos_embed], dim=1)
        else:
            trailing_text_hidden = tts_eos_embed

        return prefill_embeds.to(dtype), trailing_text_hidden.to(dtype)

    def _decode_audio_chunk(self, codec_frames: torch.Tensor):
        """Decode codec frames to audio via the vocoder.

        Parameters
        ----------
        codec_frames : torch.Tensor
            Shape [T, num_code_groups] on CUDA.

        Returns
        -------
        audio : np.ndarray
        sample_rate : int
        """
        audio_codes = codec_frames  # [T, Q]
        wavs, sr = self._speech_tokenizer.decode(
            [{"audio_codes": audio_codes}]
        )
        return wavs[0], sr


def load_streaming_tts(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    verbose: bool = True,
) -> StreamingTTSMegakernel:
    """Convenience constructor."""
    return StreamingTTSMegakernel(model_name=model_name, verbose=verbose)
