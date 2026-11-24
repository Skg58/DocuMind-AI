"""Structured JSON logging configuration for DocuMind AI."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Custom logging formatter that renders records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as a JSON string.

        Args:
            record: The LogRecord instance to format.

        Returns:
            str: JSON-formatted log entry string.
        """
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Include exception trace if present
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if supplied in record.__dict__
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "taskName",
        }
        extra_keys = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extra_keys:
            log_payload["extra"] = extra_keys

        return json.dumps(log_payload, default=str)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger with the structured JSON formatter.

    Args:
        level: Logging level as a string (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: The configured root logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Attach stdout stream handler with JsonFormatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(JsonFormatter())

    root_logger.addHandler(console_handler)

    return root_logger
