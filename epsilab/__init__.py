"""Epsilab Python SDK — evaluate, compare, and improve AI models.

Quick start::

    from epsilab import Epsilab

    client = Epsilab(api_key="sk-...")
    eval = client.create_evaluation(["openai/gpt-4o", "google/gemini-2.5-flash"])
"""

from .client import EpsilabClient as Epsilab
from . import models
from .exceptions import (
    ApiError,
    AuthError,
    EpsilabError,
    InsufficientCreditsError,
    RateLimitError,
)

__all__ = [
    "Epsilab",
    "models",
    "EpsilabError",
    "AuthError",
    "InsufficientCreditsError",
    "RateLimitError",
    "ApiError",
]

__version__ = "0.6.0"
