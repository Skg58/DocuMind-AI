"""
DocuMind AI — Generation Service
==================================
Grounded LLM answer generation (non-streaming and SSE streaming) with
TTL-based response caching.  Uses the ``google-genai`` SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator
import re
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.exceptions import GenerationError

logger = logging.getLogger(__name__)


GLOBAL_SUMMARY_PATTERNS = (
    r"\bsummar(y|ize|ise)\b",
    r"\boverview\b",
    r"\bmain argument\b",
    r"\bwhole document\b",
    r"\bentire document\b",
    r"\bkey takeaways\b",
    r"\btakeaways\b",
    r"\bgive me the gist\b",
    r"\boutline\b",
)


def classify_intent(query: str) -> str:
    """Classify broad document requests separately from precision questions."""
    normalized = query.lower().strip()
    if any(re.search(pattern, normalized) for pattern in GLOBAL_SUMMARY_PATTERNS):
        return "GLOBAL_SUMMARY"
    return "SPECIFIC_QA"


async def rewrite_query(
    query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Rewrite a conversational follow-up into a standalone document question."""
    if not chat_history:
        return query
    recent = chat_history[-4:]
    history_text = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in recent
        if turn.get("role") and turn.get("content")
    )
    if not history_text:
        return query
    prompt = (
        "Rewrite the latest user question as one standalone question using the conversation "
        "only for necessary context. Preserve names and technical terms. "
        "Do not answer it. Return only the rewritten question.\n\n"
        f"CONVERSATION:\n{history_text}\n\nLATEST QUESTION:\n{query}"
    )
    try:
        client = _get_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=128,
            ),
        )
        rewritten = (response.text or "").strip()
        return rewritten or query
    except Exception as exc:
        logger.warning("Standalone query rewrite failed; using the original query: %s", exc)
        return query

# ---------------------------------------------------------------------------
#  Clients (lazy-initialised on first use)
# ---------------------------------------------------------------------------
_sync_client: genai.Client | None = None
_async_client_attr: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return (and lazily create) the synchronous ``genai.Client``."""
    global _sync_client  # noqa: PLW0603
    if _sync_client is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError("GEMINI_API_KEY is not configured.")
        _sync_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _sync_client


def _get_async_client() -> genai.Client:
    """Return (and lazily create) the async-capable ``genai.Client``."""
    global _async_client_attr  # noqa: PLW0603
    if _async_client_attr is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError("GEMINI_API_KEY is not configured.")
        _async_client_attr = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _async_client_attr




# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Construction
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = (
    "You are DocuMind AI, a precise document question-answering assistant.\n\n"
    "RULES — follow them strictly:\n"
    "1. ONLY use information found in the CONTEXT CHUNKS provided below.\n"
    "2. Cite your sources inline using the format [Page X] where X is the page number.\n"
    "3. If the answer to the question CANNOT be found in the context, reply:\n"
    "   \"I'm sorry, but the provided document does not contain information to answer this question.\"\n"
    "4. NEVER fabricate, hallucinate, or use external knowledge.\n"
    "5. Be concise, well-structured, and professional.\n"
)


def build_grounded_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a grounded user prompt from the query and retrieved chunks.

    Parameters
    ----------
    query : str
        User's natural-language question.
    context_chunks : list[dict]
        Retrieved chunks; each must have ``"text"`` and ``"page_num"``.

    Returns
    -------
    str
        Fully-formatted prompt string ready to send to the LLM.
    """
    context_parts: list[str] = []
    for chunk in context_chunks:
        page = chunk.get("page_num", "?")
        text = chunk.get("text", "")
        context_parts.append(f"--- [Page {page}] ---\n{text}\n")

    context_block = "\n".join(context_parts)

    prompt = (
        "CONTEXT CHUNKS:\n"
        f"{context_block}\n\n"
        "USER QUESTION:\n"
        f"{query}\n\n"
        "Provide a detailed, well-cited answer based ONLY on the context above."
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════
#  Non-streaming Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate a grounded answer (non-streaming).

    Parameters
    ----------
    query : str
    context_chunks : list[dict]

    Returns
    -------
    str
        Full answer text.
    """
    if not context_chunks:
        return "I'm sorry, but the provided document does not contain information to answer this question."

    client = _get_client()
    user_prompt = build_grounded_prompt(query, context_chunks)

    try:
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
            ),
        )
        answer = response.text or ""
        logger.info("Generated answer (%d chars) for query='%s'.", len(answer), query[:60])
        return answer

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, exc_info=True)
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"LLM generation failed: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════
#  SSE Streaming Generation
# ═══════════════════════════════════════════════════════════════════════════

async def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
) -> AsyncGenerator[str, None]:
    """Stream a grounded answer as Server-Sent Events (SSE).

    Each yielded string is a complete SSE frame::

        data: {"token": "..."}\n\n

    The stream ends with::

        data: [DONE]\n\n

    Parameters
    ----------
    query : str
    context_chunks : list[dict]

    Yields
    ------
    str
        SSE-formatted data lines.
    """
    if not context_chunks:
        payload = json.dumps(
            {
                "token": (
                    "I'm sorry, but the provided document does not contain information "
                    "to answer this question."
                )
            }
        )
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        return

    client = _get_async_client()
    user_prompt = build_grounded_prompt(query, context_chunks)

    try:
        stream = await client.aio.models.generate_content_stream(
            model=settings.LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
            ),
        )
        async for chunk in stream:
            token = chunk.text or ""
            if token:
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"

    except asyncio.CancelledError:
        logger.info("Gemini streaming generation was cancelled.")
        raise
    except Exception as exc:
        logger.error("Streaming generation failed: %s", exc, exc_info=True)
        error_payload = json.dumps({"error": "Answer generation failed."})
        yield f"data: {error_payload}\n\n"

    # Signal end of stream.
    yield "data: [DONE]\n\n"