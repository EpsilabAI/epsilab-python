"""Epsilab Python SDK for the RL Environment Hub and Marketplace.

Discover, run, and export training data from published RL environments,
inspect their quality evidence, or publish your own.

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
    AgentEpisodeResult,
    AgentRunContext,
    AgentTraceEvent,
    AgentToolCall,
    AgentTurn,
    AgentUsage,
    ApplicationTool,
    ApplicationToolRelease,
    Dataset,
    DatasetRelease,
    EnvironmentListing,
    EnvironmentRelease,
    EnvironmentSession,
    EnvironmentStepResult,
    RLSession,
    RLStepResult,
    RLTrajectory,
    RLTrajectoryStep,
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
    "RLTrajectoryStep",
    "AgentToolCall",
    "AgentUsage",
    "AgentTurn",
    "AgentRunContext",
    "AgentTraceEvent",
    "AgentEpisodeResult",
    "EnvironmentListing",
    "EnvironmentRelease",
    "EnvironmentSession",
    "EnvironmentStepResult",
    "ApplicationTool",
    "ApplicationToolRelease",
    "Dataset",
    "DatasetRelease",
]

__version__ = "0.17.22"
