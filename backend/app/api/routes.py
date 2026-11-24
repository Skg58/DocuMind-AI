"""API endpoints and route definitions for DocuMind AI.

This module provides routes for document ingestion, semantic querying (standard and
streaming via SSE), model and pipeline evaluation, health checks, and document cataloging.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import (
    DocuMindError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.api.schemas import (
    ErrorResponse,
    EvaluationResult,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services import evaluation, generation, ingestion, retrieval

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["DocuMind AI"])

# Maximum allowable file size in bytes
MAX_FILE_BYTES: int = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def _session_id(request: Request) -> str:
    """Read or create a private, opaque browser-session identifier."""
    supplied = request.headers.get("X-Session-ID", "")
    try:
        return str(uuid.UUID(supplied))
    except (ValueError, AttributeError):
        return str(uuid.uuid4())


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    responses={
        413: {"model": ErrorResponse, "description": "Uploaded file exceeds maximum size limit"},
        415: {"model": ErrorResponse, "description": "Unsupported file format (non-PDF)"},
        500: {"model": ErrorResponse, "description": "Ingestion processing failure"},
    },
)
async def ingest_document_endpoint(
    request: Request,
    file: UploadFile = File(..., description="PDF document file to ingest and index"),
) -> IngestResponse:
    """Ingest a PDF document, extract text, split into recursive chunks, and build indices.

    Validates that the file is a PDF and does not exceed the configured maximum size (15MB).
    """
    session_id = _session_id(request)
    filename = file.filename or "unknown.pdf"
    content_type = file.content_type or ""

    # Validate file type (must be PDF based on filename or MIME type)
    is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"
    if not is_pdf:
        logger.warning(
            "Rejected non-PDF upload attempt: filename='%s', content_type='%s'",
            filename,
            content_type,
        )
        raise UnsupportedFileTypeError(
            detail=f"Unsupported file format '{content_type or filename}'. Only PDF files are supported.",
            status_code=415,
        )

    # Read file content and validate size limit
    file_bytes = await file.read(MAX_FILE_BYTES + 1)
    if len(file_bytes) > MAX_FILE_BYTES:
        logger.warning(
            "Rejected oversized upload: filename='%s', size=%d bytes (max=%d bytes)",
            filename,
            len(file_bytes),
            MAX_FILE_BYTES,
        )
        raise FileTooLargeError(
            detail=(
                f"File size ({len(file_bytes)} bytes) exceeds the maximum allowed "
                f"limit of {settings.MAX_FILE_SIZE_MB}MB."
            ),
            status_code=413,
        )

    if not file_bytes:
        raise ValidationError(detail="The uploaded PDF file is empty (0 bytes).", status_code=400)
    if not file_bytes.startswith(b"%PDF-"):
        raise ValidationError(detail="The uploaded file does not have a valid PDF signature.")

    try:
        embedding_model = getattr(request.app.state, "embedding_model", None)
        sparse_model = getattr(request.app.state, "sparse_model", None)
        qdrant_client = getattr(request.app.state, "qdrant_client", None)
        result = await ingestion.ingest_document(
            file_bytes=file_bytes,
            filename=filename,
            session_id=session_id,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            sparse_model=sparse_model,
        )
        response = IngestResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            num_chunks=result["num_chunks"],
            num_pages=result["num_pages"],
            message="Document successfully ingested, chunked, and indexed.",
            session_id=session_id,
        )
        response_headers = {"X-Session-ID": session_id}
        return JSONResponse(
            status_code=200,
            content=response.model_dump(),
            headers=response_headers,
        )
    except DocuMindError:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error during document ingestion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Document ID not found"},
        500: {"model": ErrorResponse, "description": "Query retrieval or generation failure"},
    },
)
async def query_document_endpoint(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    """Run adaptive query routing and return a grounded answer."""
    try:
        session_id = _session_id(request)
        qdrant_client = getattr(request.app.state, "qdrant_client", None)
        embedding_model = getattr(request.app.state, "embedding_model", None)
        sparse_model = getattr(request.app.state, "sparse_model", None)
        reranker_model = getattr(request.app.state, "reranker_model", None)
        rewritten = await generation.rewrite_query(payload.query, payload.chat_history)
        intent = generation.classify_intent(rewritten)

        documents = await ingestion.list_documents(session_id, qdrant_client)
        doc_id = payload.doc_id
        if doc_id is None and documents:
            doc_id = documents[0]["doc_id"]
        if doc_id is None:
            raise HTTPException(status_code=404, detail="Upload a document before querying.")

        if intent == "GLOBAL_SUMMARY":
            retrieved = await retrieval.global_retrieve(doc_id, session_id, qdrant_client)
        else:
            retrieved = await retrieval.hybrid_retrieve(
                rewritten,
                doc_id,
                session_id,
                qdrant_client,
                embedding_model,
                sparse_model,
                reranker_model,
                settings.TOP_K_RETRIEVAL,
                payload.top_k,
            )

        sources: List[SourceChunk] = []
        context_chunks: List[Dict[str, Any]] = []
        for item in retrieved:
            chunk = item.get("chunk", item)
            score = float(item.get("ce_score", item.get("rrf_score", item.get("score", 0.0))))
            context_chunks.append(chunk)
            sources.append(
                SourceChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    text=chunk.get("text", ""),
                    page_num=int(chunk.get("page_num", 1)),
                    score=score,
                )
            )
        answer = await run_in_threadpool(
            generation.generate_answer,
            rewritten,
            context_chunks,
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=rewritten,
            doc_id=doc_id,
            cached=False,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except DocuMindError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except Exception as exc:
        logger.error("Error during document query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer for query: {exc}",
        )


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Document ID not found"},
        500: {"model": ErrorResponse, "description": "Streaming retrieval or generation failure"},
    },
)
async def query_document_stream_endpoint(
    payload: QueryRequest,
    request: Request,
) -> StreamingResponse:
    """Stream a grounded query answer as Server-Sent Events (SSE).

    First sends an initial event containing retrieved source chunks:
    `data: {"sources": [...]}\\n\\n`
    Followed by token events:
    `data: {"token": "..."}\\n\\n`
    Ending with:
    `data: [DONE]\\n\\n`
    """
    session_id = _session_id(request)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    embedding_model = getattr(request.app.state, "embedding_model", None)
    sparse_model = getattr(request.app.state, "sparse_model", None)
    reranker_model = getattr(request.app.state, "reranker_model", None)

    try:
        rewritten = await generation.rewrite_query(payload.query, payload.chat_history)
        intent = generation.classify_intent(rewritten)
        documents = await ingestion.list_documents(session_id, qdrant_client)
        doc_id = payload.doc_id or (documents[0]["doc_id"] if documents else None)
        if doc_id is None:
            raise HTTPException(status_code=404, detail="Upload a document before querying.")
        if intent == "GLOBAL_SUMMARY":
            retrieved_results = await retrieval.global_retrieve(doc_id, session_id, qdrant_client)
        else:
            retrieved_results = await retrieval.hybrid_retrieve(
                rewritten,
                doc_id,
                session_id,
                qdrant_client,
                embedding_model,
                sparse_model,
                reranker_model,
                settings.TOP_K_RETRIEVAL,
                payload.top_k,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except DocuMindError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except Exception as exc:
        logger.error("Error during retrieval for stream: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    sources: List[Dict[str, Any]] = []
    context_chunks: List[Dict[str, Any]] = []

    for item in retrieved_results:
        chunk = item.get("chunk", item)
        score = float(item.get("ce_score", item.get("rrf_score", item.get("score", 0.0))))
        context_chunks.append(chunk)
        sources.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "page_num": int(chunk.get("page_num", 1)),
                "score": score,
            }
        )

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events for sources and streamed response tokens."""
        answer_parts: list[str] = []
        generation_failed = False
        try:
            # Send initial sources metadata event
            sources_event = json.dumps({"sources": sources})
            yield f"data: {sources_event}\n\n"

            # Stream answer tokens from the generation service
            async for token_frame in generation.generate_answer_stream(
                rewritten, context_chunks
            ):
                if token_frame.startswith("data: ") and token_frame.strip() != "data: [DONE]":
                    try:
                        event_data = json.loads(token_frame[6:].strip())
                    except json.JSONDecodeError:
                        event_data = {}
                    if "token" in event_data:
                        answer_parts.append(str(event_data["token"]))
                    if "error" in event_data:
                        generation_failed = True
                yield token_frame

        except Exception as exc:
            logger.error("Error in streaming response generator: %s", exc, exc_info=True)
            err_frame = json.dumps({"error": str(exc)})
            yield f"data: {err_frame}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/evaluate",
    response_model=EvaluationResult,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Evaluation execution failure"},
    },
)
async def evaluate_pipeline_endpoint(
    request: Request,
) -> EvaluationResult:
    """Run retrieval quality and generation faithfulness evaluation pipeline."""
    try:
        embedding_model = getattr(request.app.state, "embedding_model", None)
        reranker_model = getattr(request.app.state, "reranker_model", None)
        sparse_model = getattr(request.app.state, "sparse_model", None)
        qdrant_client = getattr(request.app.state, "qdrant_client", None)
        session_id = _session_id(request)

        raw_result = await evaluation.run_evaluation(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            sparse_model=sparse_model,
            qdrant_client=qdrant_client,
            session_id=session_id,
        )

        metrics = raw_result.get("averages", {})
        per_query = raw_result.get("per_query", [])
        skipped = raw_result.get("skipped", [])
        num_eval = metrics.get("num_evaluated", len(per_query))

        summary = (
            f"Evaluation completed for {num_eval} queries ({len(skipped)} skipped). "
            f"Avg Precision@K: {metrics.get('avg_precision_at_k', 0.0):.2f}, "
            f"Avg MRR: {metrics.get('avg_mrr', 0.0):.2f}, "
            f"Avg Faithfulness: {metrics.get('avg_faithfulness', 0.0):.2f}, "
            f"Avg Relevance: {metrics.get('avg_relevance', 0.0):.2f}."
        )

        return EvaluationResult(
            metrics=metrics,
            per_query_results=per_query,
            summary=summary,
        )
    except DocuMindError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except Exception as exc:
        logger.error("Evaluation pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation pipeline failed: {exc}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
async def health_check_endpoint(
    request: Request,
) -> HealthResponse:
    """Check backend operational health and ML model readiness."""
    emb = getattr(request.app.state, "embedding_model", None)
    rerank = getattr(request.app.state, "reranker_model", None)
    sparse = getattr(request.app.state, "sparse_model", None)
    qdrant = getattr(request.app.state, "qdrant_client", None)
    models_loaded = bool(emb is not None and rerank is not None and sparse is not None and qdrant is not None)

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded=models_loaded,
    )


@router.get(
    "/documents",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
)
async def list_documents_endpoint(request: Request) -> List[Dict[str, Any]]:
    """Return catalog of all ingested document IDs and their associated metadata."""
    session_id = _session_id(request)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    return await ingestion.list_documents(session_id, qdrant_client)


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document_endpoint(request: Request, doc_id: str) -> Response:
    """Remove a document and all of its private indexes immediately."""
    session_id = _session_id(request)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    removed = await ingestion.delete_document(session_id, doc_id, qdrant_client)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found in this session.")
    return Response(status_code=204)


@router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session_endpoint(request: Request) -> Response:
    session_id = _session_id(request)
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    await ingestion.clear_session(session_id, qdrant_client)
    return Response(status_code=204)
