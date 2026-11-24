"""Integration tests for DocuMind FastAPI endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import settings


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["version"] == settings.APP_VERSION


def test_ingest_invalid_file_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("document.txt", b"text", "text/plain")},
        headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
    )
    assert response.status_code == 415


def test_ingest_invalid_pdf_signature(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("document.pdf", b"not a PDF", "application/pdf")},
        headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
    )
    assert response.status_code == 400


def test_ingest_too_large(client: TestClient) -> None:
    size = settings.MAX_FILE_SIZE_MB * 1024 * 1024 + 1
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("large.pdf", b"%" + b"0" * (size - 1), "application/pdf")},
    )
    assert response.status_code == 413


def test_documents_are_session_filtered(client: TestClient) -> None:
    with patch(
        "app.services.ingestion.list_documents",
        new_callable=AsyncMock,
        return_value=[{"doc_id": "doc-a", "filename": "a.pdf", "num_chunks": 2, "num_pages": 2}],
    ):
        response = client.get(
            "/api/v1/documents",
            headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
        )
    assert response.status_code == 200
    assert response.json()[0]["doc_id"] == "doc-a"


def test_specific_query_route(client: TestClient) -> None:
    retrieved = [{
        "chunk": {"chunk_id": "c1", "text": "The method uses vector search.", "page_num": 3},
        "ce_score": 0.91,
    }]
    with (
        patch("app.services.ingestion.list_documents", new_callable=AsyncMock,
              return_value=[{"doc_id": "doc-a", "filename": "a.pdf"}]),
        patch("app.services.generation.rewrite_query", new_callable=AsyncMock,
              return_value="What method does the document use?"),
        patch("app.services.generation.classify_intent", return_value="SPECIFIC_QA"),
        patch("app.services.retrieval.hybrid_retrieve", new_callable=AsyncMock,
              return_value=retrieved),
        patch("app.services.generation.generate_answer",
              return_value="It uses vector search [Page 3]."),
    ):
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What method?",
                "doc_id": "doc-a",
                "top_k": 5,
                "chat_history": [{"role": "user", "content": "What is in this paper?"}],
            },
            headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
        )
    assert response.status_code == 200
    assert "[Page 3]" in response.json()["answer"]


def test_global_summary_route(client: TestClient) -> None:
    retrieved = [
        {"chunk": {"chunk_id": "c1", "text": "Opening thesis.", "page_num": 1}},
        {"chunk": {"chunk_id": "c2", "text": "Closing conclusion.", "page_num": 8}},
    ]
    with (
        patch("app.services.ingestion.list_documents", new_callable=AsyncMock,
              return_value=[{"doc_id": "doc-a", "filename": "a.pdf"}]),
        patch("app.services.generation.rewrite_query", new_callable=AsyncMock,
              return_value="Summarize the main argument."),
        patch("app.services.generation.classify_intent", return_value="GLOBAL_SUMMARY"),
        patch("app.services.retrieval.global_retrieve", new_callable=AsyncMock,
              return_value=retrieved),
        patch("app.services.generation.generate_answer",
              return_value="The document argues for a new approach [Page 1] [Page 8]."),
    ):
        response = client.post(
            "/api/v1/query",
            json={"query": "Summarize the main argument", "doc_id": "doc-a", "top_k": 5},
            headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
        )
    assert response.status_code == 200
    assert "[Page 1]" in response.json()["answer"]


def test_unknown_document_query_returns_404(client: TestClient) -> None:
    with patch("app.services.ingestion.list_documents", new_callable=AsyncMock, return_value=[]):
        response = client.post(
            "/api/v1/query",
            json={"query": "Summarize it", "chat_history": []},
            headers={"X-Session-ID": "11111111-1111-4111-8111-111111111111"},
        )
    assert response.status_code == 404
