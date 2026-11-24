"""FastAPI application lifespan management for DocuMind AI."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from sentence_transformers import CrossEncoder, SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _ensure_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    if not any(collection.name == settings.QDRANT_COLLECTION for collection in collections):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    # Create payload indexes for all filtered fields
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="expires_at",
        field_schema=models.PayloadSchemaType.FLOAT,
    )





@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for startup and shutdown events.

    Initializes structured logging, ensures required storage directories exist,
    and loads ML models (embeddings and reranker) into application state as singletons.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Transfers execution control to the application lifecycle.
    """
    # 1. Initialize logging
    setup_logging(settings.LOG_LEVEL)
    logger.info("Starting DocuMind AI backend service...")

    # 2. Document indexes are intentionally not written to disk.

    app.state.embedding_model = None
    app.state.reranker_model = None
    app.state.sparse_model = None
    app.state.qdrant_client = None
    app.state.cleanup_task = None

    async def load_embedding_model() -> SentenceTransformer:
        return await asyncio.to_thread(
            SentenceTransformer,
            settings.EMBEDDING_MODEL,
            device="cpu",
        )

    async def load_reranker_model() -> CrossEncoder:
        return await asyncio.to_thread(
            CrossEncoder,
            settings.RERANKER_MODEL,
            device="cpu",
        )

    try:
        # 3. Load models on worker threads so Uvicorn remains able to receive
        # lifespan shutdown and health requests while the first-time download occurs.
        logger.info("Loading embedding model '%s'...", settings.EMBEDDING_MODEL)
        embedding_task = asyncio.create_task(load_embedding_model())
        app.state.embedding_model = await asyncio.shield(embedding_task)
        logger.info("Embedding model loaded successfully.")

        # 4. Load cross-encoder reranker model
        logger.info("Loading reranker model '%s'...", settings.RERANKER_MODEL)
        reranker_task = asyncio.create_task(load_reranker_model())
        app.state.reranker_model = await asyncio.shield(reranker_task)
        logger.info("Reranker model loaded successfully.")

        logger.info("Loading sparse model '%s'...", settings.SPARSE_MODEL)
        app.state.sparse_model = await asyncio.to_thread(
            SparseTextEmbedding,
            model_name=settings.SPARSE_MODEL,
        )

        app.state.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
        try:
            await asyncio.to_thread(_ensure_collection, app.state.qdrant_client)
        except Exception as exc:
            logger.warning(
                "Qdrant is unavailable at startup; document operations require it: %s",
                exc,
            )

        async def cleanup_expired_periodically() -> None:
            from app.services.ingestion import cleanup_expired
            while True:
                await asyncio.sleep(300)
                try:
                    removed = await cleanup_expired(app.state.qdrant_client)
                    if removed:
                        logger.info("Expired %d Qdrant points.", removed)
                except Exception:
                    logger.exception("Periodic Qdrant cleanup failed.")

        app.state.cleanup_task = asyncio.create_task(cleanup_expired_periodically())

        logger.info("DocuMind AI startup sequence completed.")
        yield
    except asyncio.CancelledError:
        logger.info("DocuMind AI startup or shutdown lifecycle was cancelled.")
        raise
    except Exception:
        logger.exception("DocuMind AI startup failed while loading ML models.")
        raise
    finally:
        # Shutdown sequence
        logger.info("Initiating DocuMind AI shutdown sequence...")
        cleanup_task = getattr(app.state, "cleanup_task", None)
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        app.state.embedding_model = None
        app.state.reranker_model = None
        app.state.sparse_model = None
        app.state.qdrant_client = None
        logger.info("DocuMind AI shutdown complete.")
