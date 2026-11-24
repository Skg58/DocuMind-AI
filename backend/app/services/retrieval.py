"""Adaptive Qdrant hybrid retrieval and cross-encoder reranking."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Compatibility RRF helper for tests and evaluation tooling."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for results in results_list:
        for item in results:
            chunk = item["chunk"]
            chunk_id = chunk["chunk_id"]
            rank = int(item["rank"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = chunk
    fused = [
        {"chunk": chunks[chunk_id], "rrf_score": score}
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: item["rrf_score"], reverse=True)
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank
    return fused


def active_filter(session_id: str, doc_id: str | None = None) -> models.Filter:
    conditions: list[models.FieldCondition] = [
        models.FieldCondition(
            key="session_id",
            match=models.MatchValue(value=session_id),
        ),
        models.FieldCondition(
            key="expires_at",
            range=models.Range(gt=float(time.time())),
        ),
    ]
    if doc_id:
        conditions.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=doc_id),
            )
        )
    return models.Filter(must=conditions)


def cross_encoder_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    reranker_model: CrossEncoder,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(query, candidate["chunk"]["text"]) for candidate in candidates]
    scores = reranker_model.predict(pairs, show_progress_bar=False)
    for candidate, score in zip(candidates, scores):
        candidate["ce_score"] = float(score)
    candidates.sort(key=lambda item: item["ce_score"], reverse=True)
    return candidates[:top_k]


async def hybrid_retrieve(
    query: str,
    doc_id: str,
    session_id: str,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    reranker_model: CrossEncoder,
    top_k_retrieval: int = 20,
    top_k_rerank: int = 5,
) -> list[dict[str, Any]]:
    dense = np.asarray(
        (
            await asyncio.to_thread(
                embedding_model.encode,
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )[0],
        dtype=np.float32,
    ).tolist()
    sparse = next(iter(await asyncio.to_thread(sparse_model.embed, [query])))

    response = await asyncio.to_thread(
        qdrant_client.query_points,
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense,
                using="dense",
                limit=top_k_retrieval,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
                using="bm25",
                limit=top_k_retrieval,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=active_filter(session_id, doc_id),
        limit=top_k_retrieval,
        with_payload=True,
    )

    candidates = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}
        candidates.append({
            "chunk": {
                "chunk_id": str(point.id),
                "text": payload.get("text", ""),
                "page_num": int(payload.get("page_number", 1)),
                "chunk_index": int(payload.get("chunk_index", 0)),
            },
            "rrf_score": float(getattr(point, "score", 0.0)),
            "rank": rank,
        })
    return await asyncio.to_thread(
        cross_encoder_rerank,
        query,
        candidates,
        reranker_model,
        top_k_rerank,
    )


def _breadth_first_chunks(points: list[Any], limit: int = 15) -> list[dict[str, Any]]:
    chunks = []
    seen: set[str] = set()
    for point in points:
        payload = point.payload or {}
        chunk_id = str(point.id)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunks.append({
            "chunk": {
                "chunk_id": chunk_id,
                "text": payload.get("text", ""),
                "page_num": int(payload.get("page_number", 1)),
                "chunk_index": int(payload.get("chunk_index", 0)),
            }
        })
    chunks.sort(
        key=lambda item: (
            item["chunk"]["page_num"],
            item["chunk"]["chunk_index"],
        )
    )
    if len(chunks) <= limit:
        return chunks

    selected_indices: set[int] = set()
    opening = min(3, len(chunks))
    closing = min(3, len(chunks))
    middle_slots = max(0, limit - opening - closing)
    selected_indices.update(range(opening))
    selected_indices.update(range(len(chunks) - closing, len(chunks)))
    if middle_slots:
        middle = list(range(opening, len(chunks) - closing))
        if middle:
            selected_indices.update(
                middle[round(index * (len(middle) - 1) / max(1, middle_slots - 1))]
                for index in range(middle_slots)
            )
    return [chunks[index] for index in sorted(selected_indices)][:limit]


async def global_retrieve(
    doc_id: str,
    session_id: str,
    qdrant_client: QdrantClient,
) -> list[dict[str, Any]]:
    points, _ = await asyncio.to_thread(
        qdrant_client.scroll,
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=active_filter(session_id, doc_id),
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    return _breadth_first_chunks(points)
