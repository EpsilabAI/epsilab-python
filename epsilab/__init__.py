"""Epsilab Python SDK for the RL Environment Hub and Marketplace.

Search, run, and export training data from verified RL environments,
or publish your own and earn from usage.

Quick start::

    from epsilab import Epsilab

    client = Epsilab(api_key="sk-...")
    envs = client.search_environments(domain="coding", min_quality_score=0.8)
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
from .models import (
    EnvironmentListing,
    EnvironmentRelease,
    EnvironmentSession,
    EnvironmentStepResult,
    RLSession,
    RLStepResult,
    RLTrajectory,
)

__all__ = [
    "Epsilab",
    "models",
    "EpsilabError",
    "AuthError",
    "InsufficientCreditsError",
    "RateLimitError",
    "ApiError",
    "RLSession",
    "RLStepResult",
    "RLTrajectory",
    "EnvironmentListing",
    "EnvironmentRelease",
    "EnvironmentSession",
    "EnvironmentStepResult",
]

__version__ = "0.11.2"
