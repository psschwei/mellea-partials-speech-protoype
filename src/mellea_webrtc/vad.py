"""Silero VAD wrapper for utterance boundary detection."""

import logging
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

logger = logging.getLogger(__name__)

VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512          # ~32ms at 16kHz (silero requires 512 or 256)
SILENCE_THRESHOLD_SAMPLES = 5600  # 350ms at 16kHz
MIN_UTTERANCE_SAMPLES = 4000      # 250ms at 16kHz


class VoiceActivityDetector:
    """Detects utterance boundaries using Silero VAD.

    Feed 16kHz float32 samples via push(). When a complete utterance is
    detected (speech followed by 600ms silence), on_utterance is called
    with the full utterance tensor.
    """

    def __init__(self, on_utterance) -> None:
        self._on_utterance = on_utterance  # async callable(tensor)
        self._model = load_silero_vad()
        self._model.eval()

        self._buffer: list[torch.Tensor] = []     # incoming PCM chunks
        self._speech: list[torch.Tensor] = []     # accumulated speech
        self._silence_samples = 0
        self._in_speech = False

    def push(self, samples: torch.Tensor) -> list[torch.Tensor]:
        """Process new samples; returns list of complete utterance tensors."""
        utterances = []
        self._buffer.append(samples)

        # Process in VAD_WINDOW_SAMPLES chunks
        while True:
            total = sum(t.shape[0] for t in self._buffer)
            if total < VAD_WINDOW_SAMPLES:
                break

            # Collect exactly VAD_WINDOW_SAMPLES
            window_parts = []
            remaining = VAD_WINDOW_SAMPLES
            while remaining > 0 and self._buffer:
                chunk = self._buffer.pop(0)
                if chunk.shape[0] <= remaining:
                    window_parts.append(chunk)
                    remaining -= chunk.shape[0]
                else:
                    window_parts.append(chunk[:remaining])
                    self._buffer.insert(0, chunk[remaining:])
                    remaining = 0

            window = torch.cat(window_parts)
            confidence = self._model(window, VAD_SAMPLE_RATE).item()
            is_speech = confidence > 0.5

            if is_speech:
                self._in_speech = True
                self._silence_samples = 0
                self._speech.append(window)
            else:
                if self._in_speech:
                    self._silence_samples += VAD_WINDOW_SAMPLES
                    self._speech.append(window)  # keep trailing silence

                    if self._silence_samples >= SILENCE_THRESHOLD_SAMPLES:
                        utterance = torch.cat(self._speech)
                        # Trim trailing silence
                        speech_end = len(utterance) - self._silence_samples
                        utterance = utterance[:speech_end]

                        if utterance.shape[0] >= MIN_UTTERANCE_SAMPLES:
                            logger.debug(
                                "Utterance detected: %.2fs",
                                utterance.shape[0] / VAD_SAMPLE_RATE,
                            )
                            utterances.append(utterance)
                        else:
                            logger.debug("Utterance too short, discarding")

                        self._speech = []
                        self._silence_samples = 0
                        self._in_speech = False

        return utterances

    def reset(self) -> None:
        self._buffer.clear()
        self._speech.clear()
        self._silence_samples = 0
        self._in_speech = False
