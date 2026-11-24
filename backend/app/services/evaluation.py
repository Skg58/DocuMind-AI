"""
DocuMind AI — Evaluation Service
==================================
Retrieval quality and generation faithfulness evaluation pipeline.
Computes Precision@K, MRR, LLM-judged faithfulness, and relevance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder, SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.exceptions import GenerationError
from app.services.generation import generate_answer
from app.services.retrieval import hybrid_retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Sample Test Set
#  NOTE: ``doc_id`` values are placeholders — replace them with IDs of
#  actually-ingested documents before running a meaningful evaluation.
# ---------------------------------------------------------------------------

SAMPLE_TEST_SET: list[dict[str, Any]] = [
    {
        "query": "What is the purpose of the document?",
        "doc_id": "placeholder-doc-001",
        "expected_pages": [1, 2],
        "expected_keywords": ["purpose", "objective", "overview"],
    },
    {
        "query": "What methodology was used in the study?",
        "doc_id": "placeholder-doc-001",
        "expected_pages": [3, 4, 5],
        "expected_keywords": ["methodology", "approach", "method", "design"],
    },
    {
        "query": "What are the main findings and results?",
        "doc_id": "placeholder-doc-001",
        "expected_pages": [6, 7, 8],
        "expected_keywords": ["findings", "results", "analysis", "outcome"],
    },
    {
        "query": "What limitations are discussed?",
        "doc_id": "placeholder-doc-001",
        "expected_pages": [9, 10],
        "expected_keywords": ["limitations", "constraints", "challenges"],
    },
    {
        "query": "What conclusions and recommendations are made?",
        "doc_id": "placeholder-doc-001",
        "expected_pages": [10, 11, 12],
        "expected_keywords": ["conclusion", "recommendation", "future work"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════

def precision_at_k(
    retrieved_pages: list[int],
    relevant_pages: list[int],
    k: int,
) -> float:
    """Compute Precision@K.

    Parameters
    ----------
    retrieved_pages : list[int]
        Ordered list of page numbers from retrieval results.
    relevant_pages : list[int]
        Ground-truth relevant page numbers.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Precision@K ∈ [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_pages[:k]
    relevant_set = set(relevant_pages)
    hits = sum(1 for p in top_k if p in relevant_set)
    return hits / k


def mean_reciprocal_rank(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Parameters
    ----------
    retrieved_pages : list[int]
        Ordered list of page numbers from retrieval results.
    relevant_pages : list[int]
        Ground-truth relevant page numbers.

    Returns
    -------
    float
        MRR ∈ [0.0, 1.0].  Returns 0.0 if no relevant page is found.
    """
    relevant_set = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_set:
            return 1.0 / rank
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  LLM-based Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _llm_judge(prompt: str) -> float:
    """Send a judgement prompt to Gemini and parse a 0.0–1.0 score.

    The prompt must instruct the model to return **only** a decimal number
    between 0.0 and 1.0.
    """
    if not settings.GEMINI_API_KEY:
        raise GenerationError("GEMINI_API_KEY is required for LLM-as-a-Judge evaluation.")
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=16,
            ),
        )
        raw = (response.text or "").strip()
        score = float(raw)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError) as exc:
        logger.error("LLM judge returned non-numeric output: %s", exc)
        raise GenerationError("LLM judge returned an invalid score.") from exc
    except Exception as exc:
        logger.error("LLM judge call failed: %s", exc, exc_info=True)
        raise GenerationError("LLM judge evaluation failed.") from exc


def llm_faithfulness_score(answer: str, context: str) -> float:
    """Use Gemini to judge whether *answer* is faithful to *context*.

    Parameters
    ----------
    answer : str
        Generated answer text.
    context : str
        Concatenated context that was provided to the generator.

    Returns
    -------
    float
        Faithfulness score ∈ [0.0, 1.0].
    """
    prompt = (
        "You are an impartial evaluator.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "Rate how faithful the ANSWER is to the CONTEXT. A score of 1.0 means "
        "every claim in the answer is fully supported by the context. A score "
        "of 0.0 means the answer is entirely unsupported or contradicts the "
        "context.\n\n"
        "Reply with ONLY a single decimal number between 0.0 and 1.0."
    )
    return _llm_judge(prompt)


def llm_relevance_score(query: str, answer: str) -> float:
    """Use Gemini to judge whether *answer* is relevant to *query*.

    Parameters
    ----------
    query : str
        User query.
    answer : str
        Generated answer.

    Returns
    -------
    float
        Relevance score ∈ [0.0, 1.0].
    """
    prompt = (
        "You are an impartial evaluator.\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "ANSWER:\n"
        f"{answer}\n\n"
        "Rate how relevant and helpful the ANSWER is to the QUESTION. "
        "A score of 1.0 means the answer perfectly addresses the question. "
        "A score of 0.0 means the answer is completely irrelevant.\n\n"
        "Reply with ONLY a single decimal number between 0.0 and 1.0."
    )
    return _llm_judge(prompt)


# ═══════════════════════════════════════════════════════════════════════════
#  Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_evaluation(
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    sparse_model: SparseTextEmbedding,
    qdrant_client: QdrantClient,
    session_id: str = "",
    test_set: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the complete evaluation pipeline on a test set.

    For each test case the pipeline:
    1. Retrieves chunks via :func:`hybrid_retrieve`.
    2. Generates an answer via :func:`generate_answer`.
    3. Computes Precision@K, MRR, faithfulness, and relevance.

    Missing documents are gracefully skipped and noted.

    Parameters
    ----------
    embedding_model : SentenceTransformer
    reranker_model : CrossEncoder
    test_set : list[dict] | None
        Custom test set; falls back to :data:`SAMPLE_TEST_SET`.

    Returns
    -------
    dict
        ``{"per_query": [...], "averages": {...}, "skipped": [...]}``
    """
    if test_set is None:
        test_set = SAMPLE_TEST_SET

    per_query_results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    total_precision = 0.0
    total_mrr = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_latency_ms = 0.0
    evaluated_count = 0

    for case in test_set:
        query: str = case["query"]
        doc_id: str = case["doc_id"]
        expected_pages: list[int] = case.get("expected_pages", [])

        logger.info("Evaluating query='%s' on doc_id='%s'.", query[:60], doc_id)

        # --- Retrieval --------------------------------------------------------
        try:
            retrieval_started = time.perf_counter()
            retrieved = await hybrid_retrieve(
                query,
                doc_id,
                session_id,
                qdrant_client,
                embedding_model,
                sparse_model,
                reranker_model,
                settings.TOP_K_RETRIEVAL,
                settings.TOP_K_RERANK,
            )
            latency_ms = (time.perf_counter() - retrieval_started) * 1000
            total_latency_ms += latency_ms
        except ValueError as exc:
            logger.warning("Skipping evaluation for doc_id='%s': %s", doc_id, exc)
            skipped.append({"doc_id": doc_id, "query": query, "reason": str(exc)})
            continue

        retrieved_pages = [r["chunk"]["page_num"] for r in retrieved]

        # --- Generation -------------------------------------------------------
        context_chunks = [r["chunk"] for r in retrieved]
        try:
            answer = await asyncio.to_thread(generate_answer, query, context_chunks)
        except GenerationError as exc:
            logger.warning("Generation failed for query='%s': %s", query[:60], exc)
            answer = ""

        # --- Retrieval Metrics ------------------------------------------------
        k = settings.TOP_K_RERANK
        p_at_k = precision_at_k(retrieved_pages, expected_pages, k)
        mrr = mean_reciprocal_rank(retrieved_pages, expected_pages)

        # --- LLM-based Metrics ------------------------------------------------
        context_text = "\n\n".join(c["text"] for c in context_chunks)
        faithfulness = (
            await asyncio.to_thread(llm_faithfulness_score, answer, context_text)
            if answer
            else 0.0
        )
        relevance = (
            await asyncio.to_thread(llm_relevance_score, query, answer) if answer else 0.0
        )

        result_entry: dict[str, Any] = {
            "query": query,
            "doc_id": doc_id,
            "num_retrieved": len(retrieved),
            "retrieved_pages": retrieved_pages,
            "answer_length": len(answer),
            "latency_ms": round(latency_ms, 2),
            f"precision_at_{k}": p_at_k,
            "mrr": mrr,
            "faithfulness": faithfulness,
            "relevance": relevance,
        }
        per_query_results.append(result_entry)

        total_precision += p_at_k
        total_mrr += mrr
        total_faithfulness += faithfulness
        total_relevance += relevance
        evaluated_count += 1

        logger.info(
            "  P@%d=%.3f  MRR=%.3f  Faith=%.3f  Rel=%.3f",
            k, p_at_k, mrr, faithfulness, relevance,
        )

    # --- Averages -------------------------------------------------------------
    if evaluated_count > 0:
        averages: dict[str, float] = {
            "avg_precision_at_k": total_precision / evaluated_count,
            "avg_mrr": total_mrr / evaluated_count,
            "avg_faithfulness": total_faithfulness / evaluated_count,
            "avg_relevance": total_relevance / evaluated_count,
            "avg_latency_ms": total_latency_ms / evaluated_count,
            "num_evaluated": evaluated_count,
        }
    else:
        averages = {
            "avg_precision_at_k": 0.0,
            "avg_mrr": 0.0,
            "avg_faithfulness": 0.0,
            "avg_relevance": 0.0,
            "avg_latency_ms": 0.0,
            "num_evaluated": 0,
        }

    evaluation_report: dict[str, Any] = {
        "per_query": per_query_results,
        "averages": averages,
        "skipped": skipped,
    }

    logger.info(
        "Evaluation complete: %d evaluated, %d skipped. Averages: %s",
        evaluated_count,
        len(skipped),
        averages,
    )
    return evaluation_report
