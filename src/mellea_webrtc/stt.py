
"""Speech-to-text backends."""

import asyncio
import logging
import os
from typing import Protocol

import torch

logger = logging.getLogger(__name__)

STT_BACKEND = os.environ.get("STT_BACKEND", "whisper")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")


class STTBackend(Protocol):
    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        ...


class WhisperSTT:
    """STT backend using faster-whisper (CPU-friendly)."""

    def __init__(self, model_size: str = WHISPER_MODEL) -> None:
        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model: %s", model_size)
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        loop = asyncio.get_event_loop()
        audio_np = audio_16k.numpy()
        logger.debug("Audio stats: shape=%s min=%.4f max=%.4f", audio_np.shape, audio_np.min(), audio_np.max())

        def _run():
            segments, info = self._model.transcribe(
                audio_np,
                beam_size=3,
                language=WHISPER_LANGUAGE,
                vad_filter=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.debug("Whisper detected language: %s (%.2f)", info.language, info.language_probability)
            return text

        text = await loop.run_in_executor(None, _run)
        logger.debug("STT: %r", text)
        return text


class MLXWhisperSTT:
    """STT backend using mlx-whisper (Apple Silicon accelerated)."""

    def __init__(self, model_repo: str = "mlx-community/whisper-base.en-mlx") -> None:
        import mlx_whisper

        self._mlx_whisper = mlx_whisper
        self._model_repo = model_repo
        logger.info("Loading mlx-whisper model: %s", model_repo)

    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        loop = asyncio.get_event_loop()
        audio_np = audio_16k.numpy()

        def _run():
            result = self._mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=self._model_repo,
                language="en",
            )
            return result["text"].strip()

        text = await loop.run_in_executor(None, _run)
        logger.debug("STT: %r", text)
        return text


class GraniteSpeechSTT:
    """STT backend using IBM Granite Speech (CUDA/MPS/CPU)."""

    def __init__(self) -> None:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        model_name = os.environ.get("GRANITE_MODEL", "ibm-granite/granite-4.0-1b-speech")
        logger.info("Loading Granite Speech model: %s on %s", model_name, self._device)
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )

    async def transcribe(self, audio_16k: torch.Tensor) -> str:
        loop = asyncio.get_event_loop()
        wav = audio_16k.unsqueeze(0)  # (1, samples) as expected by processor

        def _run():
            user_prompt = "<|audio|>can you transcribe the speech into a written format?"
            chat = [{"role": "user", "content": user_prompt}]
            text_prompt = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self._processor(
                text_prompt, wav, device=self._device, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
                )
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = output_ids[0, num_input_tokens:].unsqueeze(0)
            text = self._tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )[0].strip()
            return text

        text = await loop.run_in_executor(None, _run)
        logger.debug("STT: %r", text)
        return text


def create_stt_backend() -> STTBackend:
    backend = STT_BACKEND.lower()
    if backend == "granite":
        return GraniteSpeechSTT()
    if backend == "mlx":
        return MLXWhisperSTT()
    return WhisperSTT()
