"""Pytest fixtures and test environment configuration for DocuMind AI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure the backend root is on sys.path so `app.*` imports resolve
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Mock heavy C-extension libraries that may not be installed in the test env
for _mod_name in [
    "qdrant_client",
    "qdrant_client.models",
    "fastembed",
    "boto3",
    "botocore",
    "botocore.config",
    "fitz",
    "sentence_transformers",
    "google",
    "google.genai",
    "google.genai.types",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from fastapi.testclient import TestClient
from app.main import app


class MockModelsContainer:
    """Container holding mocked embedding and reranker models."""

    def __init__(self, embedding_model: MagicMock, reranker_model: MagicMock) -> None:
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Fixture providing a list of 10 sample chunk dictionaries with varying page numbers."""
    chunks: List[Dict[str, Any]] = []
    topics = [
        "executive summary and mission",
        "system architecture overview",
        "data ingestion pipeline",
        "recursive character chunking",
        "dense vector representation",
        "BM25 inverted index sparse scoring",
        "reciprocal rank fusion algorithm",
        "cross-encoder reranking strategy",
        "grounded generation with Gemini",
        "retrieval and answer evaluation metrics",
    ]

    for i, topic in enumerate(topics):
        page_num = (i % 4) + 1
        chunks.append(
            {
                "chunk_id": f"chunk-uuid-{i:03d}",
                "text": f"Detailed section about {topic}. Explains design and implementation.",
                "page_num": page_num,
                "source": f"page-{page_num}",
                "chunk_index": i,
                "score": round(0.95 - (i * 0.04), 3),
            }
        )
    return chunks


@pytest.fixture
def sample_pages() -> List[Dict[str, Any]]:
    """Fixture providing parsed page dictionaries simulating parse_pdf output."""
    return [
        {
            "page_num": 1,
            "text": (
                "Introduction to DocuMind AI.\n\n"
                "DocuMind AI is an enterprise document question-answering platform. "
                "It combines dense vector embeddings and BM25 sparse keyword indices "
                "with reciprocal rank fusion to achieve state-of-the-art document retrieval."
            ),
        },
        {
            "page_num": 2,
            "text": (
                "Architecture and Chunking.\n\n"
                "Documents are parsed page-by-page and chunked recursively by paragraph, "
                "sentence, and word boundaries. Overlap is maintained between consecutive "
                "chunks to preserve semantic context across chunk splits."
            ),
        },
        {
            "page_num": 3,
            "text": (
                "Retrieval and Reranking.\n\n"
                "Dense search is performed using FAISS IndexFlatIP inner-product search. "
                "Sparse search uses BM25Okapi scoring. The top candidates from both "
                "are merged via Reciprocal Rank Fusion (RRF) and scored with a CrossEncoder."
            ),
        },
        {
            "page_num": 4,
            "text": (
                "Grounded Synthesis and Evaluation.\n\n"
                "Grounded responses are generated with strict inline citations. "
                "The pipeline computes Precision@K, Mean Reciprocal Rank (MRR), "
                "faithfulness scores, and answer relevance."
            ),
        },
    ]


@pytest.fixture
def mock_models() -> MockModelsContainer:
    """Fixture providing mocked embedding and cross-encoder reranker models."""
    mock_emb = MagicMock()

    def fake_encode(texts, *args, **kwargs) -> np.ndarray:
        count = len(texts) if isinstance(texts, list) else 1
        return np.ones((count, 384), dtype=np.float32)

    mock_emb.encode.side_effect = fake_encode

    mock_rerank = MagicMock()

    def fake_predict(pairs, *args, **kwargs) -> np.ndarray:
        return np.array([0.98 - (idx * 0.05) for idx in range(len(pairs))], dtype=np.float32)

    mock_rerank.predict.side_effect = fake_predict

    return MockModelsContainer(embedding_model=mock_emb, reranker_model=mock_rerank)


@pytest.fixture
def client(mock_models: MockModelsContainer) -> Generator[TestClient, None, None]:
    """Fixture providing a FastAPI TestClient configured with mocked ML models."""
    app.state.embedding_model = mock_models.embedding_model
    app.state.reranker_model = mock_models.reranker_model

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
