"""Public, sanitized conversion errors.

Errors raised from this package are safe to serialize in an API response. Raw
loader exceptions and tracebacks must never cross the service boundary.
"""

from __future__ import annotations

from typing import Any


class ConversionError(Exception):
    """An actionable error that is safe to return to an untrusted client."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


class ResourceLimitError(ConversionError):
    """A request or decoded payload exceeded a configured resource limit."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message, status_code=413, details=details)
