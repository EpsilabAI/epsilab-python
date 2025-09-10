class EpsilabError(Exception):
    """Base exception for Epsilab SDK."""


class AuthError(EpsilabError):
    """Authentication or authorization failed (HTTP 401/403)."""


class SubscriptionRequiredError(EpsilabError):
    """Subscription is required (HTTP 402)."""


class RateLimitError(EpsilabError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ApiError(EpsilabError):
    """Generic API error (any other 4xx/5xx)."""

