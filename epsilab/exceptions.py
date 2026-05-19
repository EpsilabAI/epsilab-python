"""Exception hierarchy for the Epsilab SDK.

All exceptions inherit from :class:`EpsilabError` so callers can
use a single ``except EpsilabError`` to handle any SDK error.
"""

from __future__ import annotations

from typing import Optional


class EpsilabError(Exception):
    """Base exception for all Epsilab SDK errors."""


class AuthError(EpsilabError):
    """Raised when the API key is missing or invalid (401/403)."""


class RateLimitError(EpsilabError):
    """Raised when the API rate limit is exceeded (429)."""

    def __init__(self, message: str, retry_after: Optional[int] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InsufficientCreditsError(EpsilabError):
    """Raised when the account has insufficient credits (402)."""

    def __init__(self, message: str, needed: int = 0, balance: int = 0) -> None:
        super().__init__(message)
        self.needed = needed
        self.balance = balance


class ApiError(EpsilabError):
    """Raised for non-success HTTP responses not covered by specific types."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
