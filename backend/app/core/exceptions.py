"""Custom domain exceptions for DocuMind AI."""


class DocuMindError(Exception):
    """Base exception class for all DocuMind AI errors."""

    def __init__(
        self,
        detail: str = "An internal DocuMind AI error occurred",
        status_code: int = 500,
    ) -> None:
        """Initialize the base DocuMind exception.

        Args:
            detail: Human-readable error message or detail.
            status_code: HTTP status code representing the error (default: 500).
        """
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.detail)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(detail='{self.detail}', status_code={self.status_code})"


class IngestionError(DocuMindError):
    """Raised when PDF parsing, text extraction, chunking, or indexing fails."""

    def __init__(
        self,
        detail: str = "Failed to ingest document",
        status_code: int = 500,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)


class RetrievalError(DocuMindError):
    """Raised when document retrieval, hybrid search, or reranking fails."""

    def __init__(
        self,
        detail: str = "Failed to retrieve documents",
        status_code: int = 500,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)


class GenerationError(DocuMindError):
    """Raised when LLM generation, answer synthesis, or API communication fails."""

    def __init__(
        self,
        detail: str = "Failed to generate LLM response",
        status_code: int = 500,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)


class ValidationError(DocuMindError):
    """Raised when request payload or parameter validation fails."""

    def __init__(
        self,
        detail: str = "Validation error",
        status_code: int = 400,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)


class FileTooLargeError(ValidationError):
    """Raised when an uploaded document exceeds the maximum allowable file size."""

    def __init__(
        self,
        detail: str = "File size exceeds the maximum allowed limit",
        status_code: int = 413,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)


class UnsupportedFileTypeError(ValidationError):
    """Raised when an uploaded document has an unsupported file extension or MIME type."""

    def __init__(
        self,
        detail: str = "Unsupported file type",
        status_code: int = 415,
    ) -> None:
        super().__init__(detail=detail, status_code=status_code)
