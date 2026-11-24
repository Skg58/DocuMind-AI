"""PDF parsing, recursive chunking, and stateless Qdrant indexing."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any

import fitz
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.storage import r2_storage

logger = logging.getLogger(__name__)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_pdf(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{filename}': {exc}") from exc

    pages: list[dict[str, Any]] = []
    total_pages = len(document)
    try:
        for page_index in range(total_pages):
            text = (document.load_page(page_index).get_text("text") or "").strip()
            if text:
                pages.append({"page_num": page_index + 1, "text": text})
    finally:
        document.close()

    if not pages:
        raise ValueError(f"PDF '{filename}' contains no extractable text across {total_pages} page(s).")
    return pages


def _split_text(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        return [segment for paragraph in paragraphs for segment in _split_text(paragraph, max_len)]
    sentences = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if len(sentences) > 1:
        return [segment for sentence in sentences for segment in _split_text(sentence, max_len)]

    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in text.split():
        if len(word) > max_len:
            if current:
                segments.append(" ".join(current))
                current = []
                current_len = 0
            segments.extend(word[index:index + max_len] for index in range(0, len(word), max_len))
            continue
        word_len = len(word) + (1 if current else 0)
        if current_len + word_len > max_len and current:
            segments.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len
    if current:
        segments.append(" ".join(current))
    return segments


def recursive_chunk(
    pages: list[dict[str, Any]],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    atomic_segments: list[tuple[int, str]] = []
    for page in pages:
        for part in _split_text(page["text"], chunk_size):
            atomic_segments.append((page["page_num"], part))

    chunks: list[dict[str, Any]] = []
    buffer = ""
    buffer_page = atomic_segments[0][0] if atomic_segments else 1

    for page_num, segment in atomic_segments:
        candidate = f"{buffer} {segment}".strip() if buffer else segment
        if len(candidate) <= chunk_size:
            buffer = candidate
            buffer_page = page_num
            continue

        if buffer:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": buffer,
                "page_num": buffer_page,
                "source": f"page-{buffer_page}",
                "chunk_index": len(chunks),
            })
            available_overlap = min(chunk_overlap, len(buffer), max(0, chunk_size - len(segment) - 1))
            overlap = buffer[-available_overlap:].strip() if available_overlap else ""
            buffer = f"{overlap} {segment}".strip() if overlap else segment
        else:
            buffer = segment
        buffer_page = page_num

    if buffer:
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": buffer,
            "page_num": buffer_page,
            "source": f"page-{buffer_page}",
            "chunk_index": len(chunks),
        })
    return chunks


def _active_filter(session_id: str, doc_id: str | None = None) -> models.Filter:
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


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    session_id: str,
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
) -> dict[str, Any]:
    doc_id = str(uuid.uuid4())
    pages = parse_pdf(file_bytes, filename)
    chunks = recursive_chunk(pages)
    if not chunks:
        raise ValueError("The PDF produced no indexable chunks.")

    texts = [chunk["text"] for chunk in chunks]
    dense_vectors = await asyncio.to_thread(
        embedding_model.encode,
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sparse_vectors = await asyncio.to_thread(sparse_model.embed, texts)
    sparse_vectors = list(sparse_vectors)
    expires_at = int(time.time()) + 30 * 60

    points: list[models.PointStruct] = []
    for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
        points.append(models.PointStruct(
            id=chunk["chunk_id"],
            vector={
                "dense": np.asarray(dense_vector, dtype=np.float32).tolist(),
                "bm25": models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                ),
            },
            payload={
                "session_id": session_id,
                "doc_id": doc_id,
                "filename": filename,
                "page_number": chunk["page_num"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "expires_at": expires_at,
            },
        ))

    await asyncio.to_thread(
        qdrant_client.upsert,
        collection_name=settings.QDRANT_COLLECTION,
        points=points,
        wait=True,
    )
    await r2_storage.upload_pdf(session_id, doc_id, file_bytes)
    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": len(pages),
    }


async def list_documents(session_id: str, qdrant_client: QdrantClient) -> list[dict[str, Any]]:
    points, _ = await asyncio.to_thread(
        qdrant_client.scroll,
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=_active_filter(session_id),
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    documents: dict[str, dict[str, Any]] = {}
    for point in points:
        payload = point.payload or {}
        doc_id = str(payload.get("doc_id", ""))
        if not doc_id:
            continue
        record = documents.setdefault(doc_id, {
            "doc_id": doc_id,
            "filename": payload.get("filename", "unknown.pdf"),
            "num_chunks": 0,
            "num_pages": 0,
        })
        record["num_chunks"] += 1
        record["num_pages"] = max(record["num_pages"], int(payload.get("page_number", 0)))
    return sorted(documents.values(), key=lambda item: item["filename"].lower())


async def delete_document(session_id: str, doc_id: str, qdrant_client: QdrantClient) -> bool:
    exists = await list_documents(session_id, qdrant_client)
    if not any(document["doc_id"] == doc_id for document in exists):
        return False
    await asyncio.to_thread(
        qdrant_client.delete,
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=_active_filter(session_id, doc_id)
        ),
        wait=True,
    )
    await r2_storage.delete_document(session_id, doc_id)
    return True


async def clear_session(session_id: str, qdrant_client: QdrantClient) -> bool:
    await asyncio.to_thread(
        qdrant_client.delete,
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=_active_filter(session_id)
        ),
        wait=True,
    )
    await r2_storage.clear_session(session_id)
    return True


async def cleanup_expired(qdrant_client: QdrantClient) -> int:
    now = float(time.time())
    expired_filter = models.Filter(must=[
        models.FieldCondition(
            key="expires_at",
            range=models.Range(lte=now),
        )
    ])
    points, _ = await asyncio.to_thread(
        qdrant_client.scroll,
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=expired_filter,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    session_documents: set[tuple[str, str]] = set()
    for point in points:
        payload = point.payload or {}
        session_documents.add((str(payload.get("session_id", "")), str(payload.get("doc_id", ""))))
    await asyncio.to_thread(
        qdrant_client.delete,
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=expired_filter),
        wait=True,
    )
    for session_id, doc_id in session_documents:
        if session_id and doc_id:
            await r2_storage.delete_document(session_id, doc_id)
    return len(points)
