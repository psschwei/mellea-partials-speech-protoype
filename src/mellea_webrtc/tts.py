"""Kokoro TTS wrapper."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

TTS_VOICE = os.environ.get("TTS_VOICE", "bf_emma")
TTS_SAMPLE_RATE = 24000


@dataclass
class TTSResult:
    """Synthesis result with timing info."""
    chunks: list[np.ndarray]
    first_phoneme: str | None          # first phoneme string produced
    first_phoneme_elapsed_ms: float    # ms from synthesis start to first phoneme
    first_audio_elapsed_ms: float      # ms from synthesis start to first audio chunk


class TextToSpeech:
    """Kokoro TTS: synthesizes text to 24kHz float32 numpy audio."""

    def __init__(self) -> None:
        from kokoro import KPipeline

        logger.info("Loading Kokoro TTS pipeline (lang='b', voice=%s)", TTS_VOICE)
        self._pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
        self._voice = TTS_VOICE

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize text into a TTSResult with audio chunks and timing."""
        loop = asyncio.get_event_loop()

        def _run():
            chunks = []
            t_start = time.perf_counter()
            first_phoneme = None
            first_phoneme_elapsed_ms = 0.0
            first_audio_elapsed_ms = 0.0

            generator = self._pipeline(text, voice=self._voice, speed=1.0)
            for graphemes, phonemes, audio in generator:
                if first_phoneme is None and phonemes:
                    first_phoneme = phonemes
                    first_phoneme_elapsed_ms = (time.perf_counter() - t_start) * 1000
                if audio is not None and len(audio) > 0:
                    if not chunks:
                        first_audio_elapsed_ms = (time.perf_counter() - t_start) * 1000
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    chunks.append(audio)

            return TTSResult(
                chunks=chunks,
                first_phoneme=first_phoneme,
                first_phoneme_elapsed_ms=first_phoneme_elapsed_ms,
                first_audio_elapsed_ms=first_audio_elapsed_ms,
            )

        result = await loop.run_in_executor(None, _run)
        logger.debug(
            "TTS synthesized %d chunk(s) for %r  [first_phoneme=%.1fms  first_audio=%.1fms]",
            len(result.chunks), text[:40],
            result.first_phoneme_elapsed_ms, result.first_audio_elapsed_ms,
        )
        return result
