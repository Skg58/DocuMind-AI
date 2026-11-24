"""Unit tests for the Reciprocal Rank Fusion (RRF) algorithm."""

from __future__ import annotations

import pytest

from app.services.retrieval import reciprocal_rank_fusion


def test_rrf_single_list() -> None:
    """Test RRF fusion with a single ranked result list."""
    single_list = [
        {"chunk": {"chunk_id": "c1", "text": "Chunk 1"}, "score": 0.95, "rank": 1},
        {"chunk": {"chunk_id": "c2", "text": "Chunk 2"}, "score": 0.85, "rank": 2},
        {"chunk": {"chunk_id": "c3", "text": "Chunk 3"}, "score": 0.75, "rank": 3},
    ]

    fused = reciprocal_rank_fusion([single_list], k=60)

    assert len(fused) == 3
    assert [item["chunk"]["chunk_id"] for item in fused] == ["c1", "c2", "c3"]
    assert [item["rank"] for item in fused] == [1, 2, 3]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"] > fused[2]["rrf_score"]


def test_rrf_multiple_lists() -> None:
    """Test RRF with two distinct result lists, verifying correct fusion and ranking."""
    list_a = [
        {"chunk": {"chunk_id": "c1", "text": "Alpha"}, "score": 0.90, "rank": 1},
        {"chunk": {"chunk_id": "c2", "text": "Beta"}, "score": 0.80, "rank": 2},
    ]
    list_b = [
        {"chunk": {"chunk_id": "c3", "text": "Gamma"}, "score": 12.5, "rank": 1},
        {"chunk": {"chunk_id": "c4", "text": "Delta"}, "score": 10.2, "rank": 2},
    ]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)

    assert len(fused) == 4
    chunk_ids = [item["chunk"]["chunk_id"] for item in fused]
    assert set(chunk_ids) == {"c1", "c2", "c3", "c4"}

    expected_top_score = 1.0 / 61
    expected_second_score = 1.0 / 62

    scores_by_id = {item["chunk"]["chunk_id"]: item["rrf_score"] for item in fused}
    assert scores_by_id["c1"] == pytest.approx(expected_top_score)
    assert scores_by_id["c3"] == pytest.approx(expected_top_score)
    assert scores_by_id["c2"] == pytest.approx(expected_second_score)
    assert scores_by_id["c4"] == pytest.approx(expected_second_score)


def test_rrf_overlapping() -> None:
    """Test RRF when documents overlap across multiple result lists."""
    list_dense = [
        {"chunk": {"chunk_id": "c1", "text": "Consensus Winner"}, "score": 0.92, "rank": 1},
        {"chunk": {"chunk_id": "c2", "text": "Dense Only"}, "score": 0.82, "rank": 2},
    ]
    list_sparse = [
        {"chunk": {"chunk_id": "c1", "text": "Consensus Winner"}, "score": 15.0, "rank": 1},
        {"chunk": {"chunk_id": "c3", "text": "Sparse Only"}, "score": 11.0, "rank": 2},
    ]

    fused = reciprocal_rank_fusion([list_dense, list_sparse], k=60)

    assert len(fused) == 3
    assert fused[0]["chunk"]["chunk_id"] == "c1"
    assert fused[0]["rank"] == 1
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    expected_c1_score = (1.0 / 61) + (1.0 / 61)
    expected_c2_score = 1.0 / 62
    assert fused[0]["rrf_score"] == pytest.approx(expected_c1_score)
    assert fused[1]["rrf_score"] == pytest.approx(expected_c2_score)
    assert fused[2]["rrf_score"] == pytest.approx(expected_c2_score)


def test_rrf_empty() -> None:
    """Test RRF handles empty input lists cleanly."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_scores() -> None:
    """Verify that RRF score calculation mathematically conforms to sum(1 / (k + rank))."""
    k = 50
    list_1 = [
        {"chunk": {"chunk_id": "c_other", "text": "Other"}, "score": 1.0, "rank": 1},
        {"chunk": {"chunk_id": "c_multi", "text": "Target"}, "score": 0.8, "rank": 2},
    ]
    list_2 = [
        {"chunk": {"chunk_id": "c_other", "text": "Other"}, "score": 1.0, "rank": 1},
        {"chunk": {"chunk_id": "c_a", "text": "A"}, "score": 0.7, "rank": 2},
        {"chunk": {"chunk_id": "c_b", "text": "B"}, "score": 0.6, "rank": 3},
        {"chunk": {"chunk_id": "c_multi", "text": "Target"}, "score": 0.5, "rank": 4},
    ]
    list_3 = [
        {"chunk": {"chunk_id": "c_multi", "text": "Target"}, "score": 0.99, "rank": 1},
    ]

    fused = reciprocal_rank_fusion([list_1, list_2, list_3], k=k)

    expected_multi_score = (1.0 / (k + 2)) + (1.0 / (k + 4)) + (1.0 / (k + 1))
    multi_entry = next(item for item in fused if item["chunk"]["chunk_id"] == "c_multi")

    assert multi_entry["rrf_score"] == pytest.approx(expected_multi_score, rel=1e-7)
    assert multi_entry["rank"] == 1, "c_multi should achieve the highest fused rank"
