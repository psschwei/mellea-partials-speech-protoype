"""LLM integration via Mellea-partials stream_with_chunking."""

import asyncio
import logging
import os
import re

from mellea.backends.openai import OpenAIBackend
from mellea.backends.model_options import ModelOption
from mellea.stdlib.components.instruction import Instruction
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.context import SimpleContext

from mellea_partial import StreamChunkingResult, stream_with_chunking
from mellea_partial.chunking import ChunkingStrategy

logger = logging.getLogger(__name__)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "granite-4.0-micro@q8_0")
GUARDIAN_MODEL = os.environ.get("GUARDIAN_MODEL", "granite-guardian-3.3-8b")

# System prompt for the voice assistant
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Respond naturally and conversationally. "
    "Keep responses concise — typically 2-4 sentences. "
    "Avoid markdown, bullet points, or numbered lists; use plain prose only."
)


_CLAUSE_SPLIT = re.compile(r"(?<=[,;:\-\u2014.!?])(?=\s+)")
_MIN_CLAUSE_LEN = 20


class ClauseChunking(ChunkingStrategy):
    """Splits on clause boundaries (commas, semicolons, dashes, sentence-end).

    Merges fragments shorter than _MIN_CLAUSE_LEN into the next chunk so that
    TTS receives reasonably-sized phrases rather than single words.
    """

    def split(self, text: str) -> list[str]:
        raw = _CLAUSE_SPLIT.split(text)
        if len(raw) <= 1:
            return raw

        merged: list[str] = []
        buf = ""
        for part in raw:
            buf += part
            if len(buf) >= _MIN_CLAUSE_LEN:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] += buf
            else:
                merged.append(buf)
        return merged


class MarkdownFreeRequirement(Requirement):
    """Rejects chunks that contain markdown formatting (bad for TTS)."""

    def __init__(self):
        super().__init__(
            description="The response must not contain markdown formatting.",
            check_only=True,
        )

    async def validate(self, backend, ctx, *, format=None, model_options=None):
        text = ctx.last_output().value or ""
        has_markdown = bool(re.search(r"(\*\*|\*|#+|`|\[.+\]\(.+\)|^\s*[-*]\s)", text, re.MULTILINE))
        return ValidationResult(result=not has_markdown)


class GuardianRequirement(Requirement):
    """Quick-check requirement that calls Granite Guardian to detect harmful content."""

    def __init__(self):
        super().__init__(
            description="The response must not contain harmful, toxic, or unsafe content.",
            check_only=True,
            output_to_bool=(lambda x: "yes" not in str(x).lower())
        )
        self._guardian_backend = OpenAIBackend(
            model_id=GUARDIAN_MODEL,
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
        )

    async def validate(self, backend, ctx, *, format=None, model_options=None):
        last_output = ctx.last_output()
        chunk_text = last_output.value
        if not chunk_text or not chunk_text.strip():
            return ValidationResult(result=True)
        try:
            test_ctx = SimpleContext()
            test_ctx = test_ctx.add(last_output)
            return await super().validate(
                self._guardian_backend, test_ctx, format=format, model_options=model_options
            )
        except Exception:
            logger.warning("Guardian check failed, allowing chunk: %r", chunk_text, exc_info=True)
            return ValidationResult(result=True)


class DeferredGuardianRequirement(Requirement):
    """Wraps GuardianRequirement but skips the check for the first chunk.

    The first chunk passes through immediately for lower TTFW. Guardian runs
    on the first chunk asynchronously (fire-and-forget, logs a warning on failure).
    Subsequent chunks are checked synchronously as before.
    """

    def __init__(self):
        super().__init__(
            description="Deferred guardian: skip first chunk, check rest.",
            check_only=True,
        )
        self._inner = GuardianRequirement()
        self._chunk_count = 0

    async def validate(self, backend, ctx, *, format=None, model_options=None):
        self._chunk_count += 1
        if self._chunk_count == 1:
            # Fire-and-forget Guardian on the first chunk
            asyncio.ensure_future(self._async_check_first(backend, ctx, format=format, model_options=model_options))
            return ValidationResult(result=True)
        return await self._inner.validate(backend, ctx, format=format, model_options=model_options)

    async def _async_check_first(self, backend, ctx, *, format=None, model_options=None):
        try:
            result = await self._inner.validate(backend, ctx, format=format, model_options=model_options)
            if not result.result:
                logger.warning("Deferred Guardian: first chunk FAILED check (already sent to TTS)")
        except Exception:
            logger.warning("Deferred Guardian: first chunk check errored", exc_info=True)


def _make_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id=LM_STUDIO_MODEL,
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model_options={ModelOption.STREAM: True,
                       SYSTEM_PROMPT: "You are a helpful chat assistant. Keep answers short."},
    )


async def generate_response(user_text: str, sentence_queue: asyncio.Queue[str | None]) -> StreamChunkingResult:
    """Stream LLM response sentence-by-sentence into sentence_queue.

    Puts each sentence string onto the queue, then puts None as sentinel.
    Returns the StreamChunkingResult so callers can cancel _task if needed.
    """
    logger.debug("LLM input: %r", user_text)
    backend = _make_backend()

    instruction = Instruction(
        description=f"{user_text}",
    )

    async def on_failure(chunk: str, ctx, requirements, results) -> tuple[bool, str]:
        return True, "Sorry, some things are not right."

    ctx = SimpleContext()
    result = await stream_with_chunking(
        instruction,
        backend,
        ctx,
        chunking=ClauseChunking(),
        quick_check_requirements=[DeferredGuardianRequirement(), MarkdownFreeRequirement()],
        quick_repair=on_failure,
    )

    async for sentence in result.astream():
        sentence = sentence.strip()
        if sentence:
            logger.debug("LLM sentence: %r", sentence)
            await sentence_queue.put(sentence)

    if result.failed_chunk:
        logger.warning("Guardian stopped streaming at chunk: %r", result.failed_chunk)

    logger.debug("LLM complete. Full text: %r", result.full_text)
    return result
