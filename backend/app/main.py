"""Main FastAPI application module for DocuMind AI."""

from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import DocuMindError
from app.core.lifespan import lifespan
from app.core.rate_limit import RateLimitMiddleware
from app.api.routes import router


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
# Configure CORS outside middleware that may return an early response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],
)


# Global exception handler for DocuMindError
@app.exception_handler(DocuMindError)
async def documind_exception_handler(request: Request, exc: DocuMindError) -> JSONResponse:
    """Handle custom DocuMind domain exceptions globally across all endpoints."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


# Include API router
app.include_router(router)


# Health check endpoint at root '/'
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """Root health check endpoint returning service status and metadata."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
