"""Unit tests for the recursive document chunking engine."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.services.ingestion import recursive_chunk


def test_recursive_chunk_basic(sample_pages: List[Dict[str, Any]]) -> None:
    """Test that recursive chunking creates chunks with all required metadata fields."""
    chunks = recursive_chunk(sample_pages, chunk_size=200, chunk_overlap=30)

    assert isinstance(chunks, list)
    assert len(chunks) > 0, "Expected chunks to be produced from sample pages"

    expected_keys = {"chunk_id", "text", "page_num", "source", "chunk_index"}
    for idx, chunk in enumerate(chunks):
        assert expected_keys.issubset(chunk.keys()), f"Chunk {idx} missing keys: {expected_keys - chunk.keys()}"
        assert isinstance(chunk["chunk_id"], str) and len(chunk["chunk_id"]) > 0
        assert isinstance(chunk["text"], str) and len(chunk["text"].strip()) > 0
        assert isinstance(chunk["page_num"], int) and chunk["page_num"] >= 1
        assert isinstance(chunk["source"], str) and f"page-{chunk['page_num']}" in chunk["source"]
        assert chunk["chunk_index"] == idx, f"Chunk index should be sequential (got {chunk['chunk_index']}, expected {idx})"


def test_recursive_chunk_overlap() -> None:
    """Test that consecutive chunks share overlapping text when chunk_overlap > 0."""
    sentences = [
        "First sentence begins the narrative on neural network fundamentals.",
        "Second sentence details transformer attention mechanisms and heads.",
        "Third sentence explores feedforward layers and positional encodings.",
        "Fourth sentence discusses encoder-decoder sequence architectures.",
        "Fifth sentence analyzes autoregressive causal generation methods.",
    ]
    pages = [{"page_num": 1, "text": " ".join(sentences)}]

    chunk_size = 140
    chunk_overlap = 35

    chunks = recursive_chunk(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 2, f"Expected at least 2 chunks to test overlap, got {len(chunks)}"

    for i in range(len(chunks) - 1):
        prev_chunk_text = chunks[i]["text"]
        next_chunk_text = chunks[i + 1]["text"]

        tail = prev_chunk_text[-chunk_overlap:].strip()
        words_in_tail = [w for w in tail.split() if len(w) > 3]
        shared_words = [w for w in words_in_tail if w in next_chunk_text]
        assert len(shared_words) > 0, (
            f"Expected overlap between chunk {i} and {i + 1}. "
            f"Tail of prev: '{tail}' | Next chunk: '{next_chunk_text[:60]}...'"
        )


def test_recursive_chunk_empty() -> None:
    """Test chunking with empty page input returns an empty list without error."""
    chunks = recursive_chunk([], chunk_size=256, chunk_overlap=32)
    assert isinstance(chunks, list)
    assert len(chunks) == 0


def test_recursive_chunk_rejects_invalid_overlap() -> None:
    """Reject overlap values that cannot produce forward progress."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        recursive_chunk([{"page_num": 1, "text": "content"}], chunk_size=10, chunk_overlap=10)


def test_recursive_chunk_single_page() -> None:
    """Test chunking with a single page produces correctly attributed chunk(s)."""
    single_page_text = "DocuMind AI provides high-precision question-answering over PDF documents."
    pages = [{"page_num": 42, "text": single_page_text}]

    chunks = recursive_chunk(pages, chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["page_num"] == 42
    assert chunk["source"] == "page-42"
    assert chunk["chunk_index"] == 0
    assert chunk["text"] == single_page_text


def test_chunk_size_limit() -> None:
    """Verify that chunks adhere approximately to the specified chunk_size limit."""
    long_text = (
        "Artificial intelligence and deep learning have revolutionized Natural Language Processing. "
        "Dense embedding models map discrete tokens into dense vector spaces capturing latent semantics. "
        "Retrieval-Augmented Generation bridges static parameter memory and dynamic external corpora. "
        "By decomposing documents into manageable semantic units, fine-grained retrieval is achieved. "
        "Each chunk must maintain enough context while avoiding token bloat during generation."
    )
    pages = [{"page_num": 1, "text": long_text}]

    target_chunk_size = 120
    chunk_overlap = 20

    chunks = recursive_chunk(pages, chunk_size=target_chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) >= 3, f"Expected multiple chunks, got {len(chunks)}"

    for idx, c in enumerate(chunks):
        chunk_len = len(c["text"])
        assert chunk_len <= target_chunk_size, (
            f"Chunk {idx} length ({chunk_len}) exceeded target size ({target_chunk_size})"
        )


def test_chunk_size_limit_with_oversized_word() -> None:
    """Split unbroken tokens rather than emitting chunks larger than the limit."""
    chunks = recursive_chunk(
        [{"page_num": 1, "text": "x" * 25}],
        chunk_size=10,
        chunk_overlap=2,
    )
    assert len(chunks) == 3
    assert all(len(chunk["text"]) <= 10 for chunk in chunks)
