"""Pydantic v2 data models and schemas for DocuMind AI API.

This module defines all request, response, and intermediate data transfer
objects (DTOs) used across DocuMind AI endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    """Response payload returned upon successful document ingestion and indexing."""

    model_config = ConfigDict(populate_by_name=True)

    doc_id: str = Field(..., description="Unique UUID assigned to the ingested document")
    filename: str = Field(..., description="Original filename of the ingested PDF")
    num_chunks: int = Field(..., description="Total count of text chunks created from the document")
    num_pages: int = Field(..., description="Count of pages containing extractable text")
    message: str = Field(
        default="Document ingested and indexed successfully",
        description="Human-readable status summary message",
    )
    session_id: str | None = Field(
        default=None,
        description="Opaque private browser-session identifier",
    )


class QueryRequest(BaseModel):
    """Request payload for semantic querying and answer generation."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language question or query string to answer from document context",
    )
    doc_id: str | None = Field(
        default=None,
        description="Unique identifier of the target ingested document to search within",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant context chunks to retrieve and consider",
    )
    chat_history: List[Dict[str, str]] = Field(
        default_factory=list,
        max_length=4,
        description="Last four client-side conversation turns",
    )


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    """Metadata and snippet of a retrieved context chunk used to support an answer."""

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(..., description="Unique identifier of the source chunk")
    text: str = Field(..., description="Verbatim text content of the retrieved chunk")
    page_num: int = Field(..., description="1-based page number where the chunk originated")
    score: float = Field(..., description="Retrieval / reranker relevance score")


class QueryResponse(BaseModel):
    """Response payload containing generated answer and supporting source citations."""

    model_config = ConfigDict(populate_by_name=True)

    answer: str = Field(..., description="Grounded answer synthesized by the LLM")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="List of supporting source chunks retrieved for answering the query",
    )
    query: str = Field(..., description="Original user query")
    doc_id: str = Field(..., description="Identifier of the queried document")
    cached: bool = Field(
        default=False,
        description="Indicates whether the answer was served directly from cache",
    )


class EvaluationResult(BaseModel):
    """Response payload summarizing evaluation pipeline metrics and query results."""

    model_config = ConfigDict(populate_by_name=True)

    metrics: Dict[str, Any] = Field(
        ...,
        description="Aggregate evaluation metrics such as Precision@K, MRR, Faithfulness, Relevance",
    )
    per_query_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed evaluation results broken down per evaluation query",
    )
    summary: str = Field(..., description="Human-readable summary of the evaluation execution")


class HealthResponse(BaseModel):
    """Response payload reporting application health and model availability."""

    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(default="healthy", description="Current health status of the application")
    version: str = Field(..., description="Application semantic version string")
    models_loaded: bool = Field(
        ...,
        description="Flag indicating whether embedding and reranking models are loaded into memory",
    )


class ErrorResponse(BaseModel):
    """Standardized error payload returned upon client or server exceptions."""

    model_config = ConfigDict(populate_by_name=True)

    detail: str = Field(..., description="Detailed description of the error encountered")
    error_type: str = Field(..., description="Exception class or category name of the error")
