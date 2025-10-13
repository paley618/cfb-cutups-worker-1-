"""Structured logging utilities with request-level context."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

REQUEST_ID_HEADER = "X-Request-ID"

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_REQUEST_DEBUG: ContextVar[bool] = ContextVar("request_debug", default=False)

_DEFAULT_LOG_KEYS: Iterable[str] = (
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
)

_BASE_LEVEL = logging.INFO


class RequestLevelFilter(logging.Filter):
    """Filter that enforces the configured base level unless debug is enabled."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        if record.levelno >= _BASE_LEVEL:
            return True
        return bool(_REQUEST_DEBUG.get())


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON with contextual metadata."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
            "module": record.name,
        }

        request_id = getattr(record, "request_id", None) or _REQUEST_ID.get()
        if request_id:
            payload["request_id"] = request_id

        job_id = getattr(record, "job_id", None)
        if job_id:
            payload["job_id"] = job_id

        for key, value in record.__dict__.items():
            if key in _DEFAULT_LOG_KEYS or key in {"request_id", "job_id"}:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level_name: str) -> None:
    """Configure root logging to emit structured JSON lines."""

    global _BASE_LEVEL

    level = getattr(logging, level_name.upper(), logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO
    _BASE_LEVEL = level

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestLevelFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def bind_request_context(request_id: str, debug: bool = False) -> Tuple[Token, Token]:
    """Bind the current request context for downstream log records."""

    return _REQUEST_ID.set(request_id), _REQUEST_DEBUG.set(debug)


def reset_request_context(token_request: Optional[Token], token_debug: Optional[Token]) -> None:
    """Reset the request context using the provided tokens."""

    if token_request is not None:
        _REQUEST_ID.reset(token_request)
    if token_debug is not None:
        _REQUEST_DEBUG.reset(token_debug)


def current_request_id() -> Optional[str]:
    """Return the request identifier currently in scope, if any."""

    return _REQUEST_ID.get()


def is_request_debug_enabled() -> bool:
    """Return whether debug logging is enabled for the active request."""

    return bool(_REQUEST_DEBUG.get())
