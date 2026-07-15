"""Data models returned by the Epsilab API.

All models are plain ``dataclasses`` with ``from_dict`` / ``to_dict``
helpers for JSON round-tripping. Import them directly or via
``epsilab.models``:

    >>> from epsilab.models import RunSummary, GapSummary
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ── Evaluation runs ──────────────────────────────────────────────────


@dataclass
class RunSummary:
    """Status and metadata for an evaluation run.

    Attributes:
        run_id: Unique identifier for this run.
        status: Current state — ``queued``, ``running``, ``completed``,
            ``failed``, or ``cancelled``.
        name: Human-readable display name.
        target_model: The primary model being evaluated.
        reference_models: Models used for comparison.
        task_count: Number of tasks in the evaluation.
        gap_count: Number of capability gaps found.
        error: Error message if the run failed.
        created_at: ISO-8601 creation timestamp.
        started_at: ISO-8601 timestamp when execution began.
        completed_at: ISO-8601 timestamp when execution finished.
        progress: Progress details when the run is active (tasks
            completed/total, percent, elapsed/remaining seconds).
        estimated_credits: Estimated credit cost for this run.
        resumed_from: Run ID this was resumed/retried from, if any.
        evaluation_id: Parent evaluation ID, if part of a multi-model
            evaluation.
    """

    run_id: str
    status: str
    name: Optional[str] = None
    target_model: Optional[str] = None
    reference_models: Optional[List[str]] = None
    task_count: int = 0
    gap_count: int = 0
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    estimated_credits: Optional[int] = None
    resumed_from: Optional[str] = None
    evaluation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunSummary":
        return cls(
            run_id=data["run_id"],
            status=data["status"],
            name=data.get("name"),
            target_model=data.get("target_model"),
            reference_models=data.get("reference_models"),
            task_count=data.get("task_count", 0),
            gap_count=data.get("gap_count", 0),
            error=data.get("error"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            progress=data.get("progress"),
            estimated_credits=data.get("estimated_credits"),
            resumed_from=data.get("resumed_from"),
            evaluation_id=data.get("evaluation_id"),
        )


# ── Gaps and artifacts ───────────────────────────────────────────────


@dataclass
class GapSummary:
    """A capability gap found during evaluation.

    Attributes:
        gap_id: Unique identifier for this gap.
        capability: The capability area (e.g. ``"coding"``, ``"math"``).
        alpha_score: Composite gap severity score (0.0–1.0).
        target_score: Your model's score on this capability.
        reference_score: Best reference model's score.
        priority: Gap priority level (``critical``, ``high``,
            ``medium``, ``low``).
        description: Human-readable description of the gap.
        verification: How this gap was verified (``judge``,
            ``execution``, ``exact_match``, ``human``).
        review_status: Human review status (``pending``,
            ``in_progress``, ``approved``, or ``None``).
    """

    gap_id: str
    capability: str
    alpha_score: float
    target_score: float
    reference_score: float
    priority: Optional[str] = None
    description: Optional[str] = None
    verification: Optional[str] = None
    review_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GapSummary":
        return cls(
            gap_id=data["gap_id"],
            capability=data["capability"],
            alpha_score=data.get("alpha_score", 0.0),
            target_score=data.get("target_score", 0.0),
            reference_score=data.get("reference_score", 0.0),
            priority=data.get("priority"),
            description=data.get("description"),
            verification=data.get("verification"),
            review_status=data.get("review_status"),
        )


@dataclass
class ArtifactSummary:
    """A generated training artifact (e.g. preference pair, test case).

    Attributes:
        artifact_id: Unique identifier.
        artifact_type: Type of artifact. One of:

            - ``preference_pair`` — DPO/RLHF chosen/rejected pair
            - ``gold_answer`` — verified correct output (SFT-ready)
            - ``trajectory`` — full agent execution trace
            - ``refined_trajectory`` — compressed, verified trajectory
              with redundant steps removed (higher quality for training)
            - ``test_case`` — executable test

        gap_id: The gap this artifact addresses, if any.
        content: Artifact payload (prompt, chosen/rejected, etc.).
            For ``refined_trajectory`` artifacts, includes:

            - ``prompt`` — the original task prompt
            - ``refined_trajectory`` — compressed step sequence
            - ``original_step_count`` — steps before compression
            - ``refined_step_count`` — steps after compression
            - ``compression_ratio`` — ratio of refined/original steps
            - ``final_output`` — model's final answer
            - ``score`` — verification score (0-1)
            - ``domain`` — task domain
            - ``capability`` — capability area

        metadata: Additional metadata about the artifact.
    """

    artifact_id: str
    artifact_type: str
    gap_id: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_refined(self) -> bool:
        """Whether this is a refined (compressed) trajectory artifact."""
        return self.artifact_type == "refined_trajectory"

    @property
    def compression_ratio(self) -> Optional[float]:
        """Compression ratio for refined trajectories (0-1, lower=more compressed)."""
        if self.artifact_type != "refined_trajectory":
            return None
        return self.content.get("compression_ratio")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactSummary":
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            gap_id=data.get("gap_id"),
            content=data.get("content", {}),
            metadata=data.get("metadata", {}),
        )


# ── Custom tasks ─────────────────────────────────────────────────────


@dataclass
class CustomTaskUploadResult:
    """Result of uploading custom evaluation tasks.

    Attributes:
        uploaded: Number of tasks successfully uploaded.
        task_ids: IDs assigned to the uploaded tasks.
        task_names: Names of the uploaded tasks.
        skipped_duplicates: Number of duplicate tasks skipped.
        source: Task source label (always ``"custom"``).
    """

    uploaded: int
    task_ids: List[str]
    task_names: List[str]
    skipped_duplicates: int = 0
    source: str = "custom"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomTaskUploadResult":
        return cls(
            uploaded=data["uploaded"],
            task_ids=data.get("task_ids", []),
            task_names=data.get("task_names", []),
            skipped_duplicates=data.get("skipped_duplicates", 0),
            source=data.get("source", "custom"),
        )


# ── Usage and billing ────────────────────────────────────────────────


@dataclass
class UsageRecord:
    """Monthly usage summary.

    Attributes:
        period: Month in ``YYYY-MM`` format.
        run_count: Number of evaluation runs.
        total_prompt_tokens: Total prompt tokens consumed.
        total_completion_tokens: Total completion tokens consumed.
        total_cost_usd: Total API cost in USD.
    """

    period: str
    run_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageRecord":
        return cls(
            period=data["period"],
            run_count=data.get("run_count", 0),
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
        )


# ── Multi-model evaluations ─────────────────────────────────────────


@dataclass
class ModelEstimate:
    """Per-model cost breakdown within a cost estimate.

    Attributes:
        model_id: Model identifier.
        task_count: Number of tasks this model will be evaluated on.
        credits: Estimated credit cost for this model.
        fresh_tasks: Number of tasks to be newly evaluated.
        cached_tasks: Number of tasks with existing results.
        usd_per_task: Estimated USD cost per task.
        usd_total: Estimated total USD API cost.
    """

    model_id: str
    task_count: int
    credits: int
    fresh_tasks: int = 0
    cached_tasks: int = 0
    usd_per_task: Optional[float] = None
    usd_total: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEstimate":
        return cls(
            model_id=data["model_id"],
            task_count=data.get("task_count", 0),
            credits=data.get("credits", 0),
            fresh_tasks=data.get("fresh_tasks", 0),
            cached_tasks=data.get("cached_tasks", 0),
            usd_per_task=data.get("usd_per_task"),
            usd_total=data.get("usd_total"),
        )


@dataclass
class CostEstimate:
    """Estimated cost for a planned evaluation.

    Returned by :meth:`~epsilab.Epsilab.estimate_evaluation_cost`.

    Attributes:
        task_count: Total number of tasks in the evaluation.
        total_credits: Total credit cost across all models.
        balance: Current account credit balance.
        sufficient: Whether the balance covers the cost.
        per_model: Breakdown by model.
    """

    task_count: int
    total_credits: int
    balance: int
    sufficient: bool
    per_model: List[ModelEstimate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_model"] = [m.to_dict() for m in self.per_model]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEstimate":
        return cls(
            task_count=data.get("task_count", 0),
            total_credits=data.get("total_credits", 0),
            balance=data.get("balance", 0),
            sufficient=data.get("sufficient", False),
            per_model=[ModelEstimate.from_dict(m) for m in data.get("per_model", [])],
        )


@dataclass
class EvaluationRunResult:
    """Individual run created as part of a multi-model evaluation.

    Attributes:
        run_id: Unique run identifier.
        model_id: The model being evaluated in this run.
        harness: Agent harness used, if any.
        status: Current run status.
        estimated_credits: Estimated credit cost for this run.
    """

    run_id: str
    model_id: str
    harness: Optional[str] = None
    status: str = "pending"
    estimated_credits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRunResult":
        return cls(
            run_id=data["run_id"],
            model_id=data["model_id"],
            harness=data.get("harness"),
            status=data.get("status", "pending"),
            estimated_credits=data.get("estimated_credits", 0),
        )


@dataclass
class EvaluationResult:
    """Result from creating a multi-model evaluation.

    Returned by :meth:`~epsilab.Epsilab.create_evaluation`.

    Attributes:
        evaluation_id: Unique evaluation identifier.
        name: Display name for the evaluation.
        total_models: Number of models in the evaluation.
        total_estimated_credits: Total estimated credit cost.
        runs: Individual runs created for this evaluation.
    """

    evaluation_id: str
    name: Optional[str]
    total_models: int
    total_estimated_credits: int
    runs: List[EvaluationRunResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["runs"] = [r.to_dict() for r in self.runs]
        return d

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            evaluation_id=data["evaluation_id"],
            name=data.get("name"),
            total_models=data.get("total_models", 0),
            total_estimated_credits=data.get("total_estimated_credits", 0),
            runs=[EvaluationRunResult.from_dict(r) for r in data.get("runs", [])],
        )


# ── RL environments ──────────────────────────────────────────────────


@dataclass
class RLSession:
    """An active or completed RL environment session.

    Attributes:
        session_id: Unique session identifier.
        task_id: The task this session is running.
        env_type: Environment type (single_turn, code_sandbox, agent_workflow, simulation).
        status: Session state — ``active``, ``completed``, ``truncated``, ``failed``.
        observation: Current/initial observation text.
        reward_mode: Reward mode (binary, continuous, partial_credit).
        total_reward: Cumulative reward across all steps.
        steps_taken: Number of steps completed so far.
        info: Additional metadata from the environment.
    """

    session_id: str
    task_id: str
    env_type: str
    status: str
    observation: str = ""
    reward_mode: str = "continuous"
    total_reward: Optional[float] = None
    steps_taken: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLSession":
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            env_type=data["env_type"],
            status=data["status"],
            observation=data.get("observation", ""),
            reward_mode=data.get("reward_mode", "continuous"),
            total_reward=data.get("total_reward"),
            steps_taken=data.get("steps_taken", 0),
            info=data.get("info", {}),
        )


@dataclass
class RLStepResult:
    """Result of taking an action in an RL environment.

    Attributes:
        observation: Text observation from the environment.
        reward: Reward for this step (None for intermediate steps).
        terminated: Whether the episode ended naturally (goal reached).
        truncated: Whether the episode was cut short (max steps, error).
        info: Step metadata (scores, latency, terminal_reason, etc.).
    """

    observation: str
    reward: Optional[float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @property
    def done(self) -> bool:
        """True if the episode is over (terminated or truncated)."""
        return self.terminated or self.truncated

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLStepResult":
        return cls(
            observation=data.get("observation", ""),
            reward=data.get("reward"),
            terminated=data.get("terminated", False),
            truncated=data.get("truncated", False),
            info=data.get("info", {}),
        )


@dataclass
class RLTrajectory:
    """Full trajectory for a completed RL session.

    Attributes:
        session_id: Session that produced this trajectory.
        task_id: Task the session ran.
        env_type: Environment type.
        status: Final session status.
        total_reward: Cumulative reward.
        steps_taken: Total steps.
        steps: List of step records (action_hash, observation, reward, terminated, truncated).
    """

    session_id: str
    task_id: str
    env_type: str
    status: str
    total_reward: Optional[float] = None
    steps_taken: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLTrajectory":
        return cls(
            session_id=data["session_id"],
            task_id=data["task_id"],
            env_type=data["env_type"],
            status=data["status"],
            total_reward=data.get("total_reward"),
            steps_taken=data.get("steps_taken", 0),
            steps=data.get("steps", []),
        )


# ── Environment Hub & Marketplace ────────────────────────────────────


@dataclass
class EnvironmentListing:
    """A listed environment available through the marketplace.

    Attributes:
        listing_id: Unique listing identifier.
        namespace_id: Owning namespace identifier.
        slug: URL-safe listing slug.
        title: Human-readable title.
        summary: Short description.
        visibility: ``private``, ``unlisted``, or ``public``.
        moderation_state: ``pending``, ``approved``, or ``suspended``.
        recommended_release_id: Currently recommended release, if any.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
    """

    listing_id: str
    namespace_id: str
    slug: str
    title: str
    summary: str = ""
    visibility: str = "private"
    moderation_state: str = "pending"
    recommended_release_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentListing":
        return cls(
            listing_id=str(data["listing_id"]),
            namespace_id=str(data.get("namespace_id", "")),
            slug=data.get("slug", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            visibility=data.get("visibility", "private"),
            moderation_state=data.get("moderation_state", "pending"),
            recommended_release_id=_opt_str(data.get("recommended_release_id")),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class EnvironmentSession:
    """A hosted environment session for running episodes.

    Attributes:
        session_id: Unique session identifier.
        deployment_id: Deployment this session runs on.
        task_id: Task being executed.
        status: ``provisioning``, ``active``, ``completed``, ``failed``,
            ``cancelled``, or ``truncated``.
        session_token: Bearer token for step requests (only on create).
        session_token_expires_at: Token expiry timestamp.
        observation: Current observation (available after provisioning).
        reward: Cumulative reward so far.
        steps_taken: Number of steps completed.
        seed: Reproducibility seed, if provided.
        created_at: ISO-8601 creation timestamp.
    """

    session_id: str
    deployment_id: str
    task_id: str
    status: str
    session_token: Optional[str] = None
    session_token_expires_at: Optional[str] = None
    observation: Optional[str] = None
    reward: Optional[float] = None
    steps_taken: int = 0
    seed: Optional[int] = None
    created_at: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Whether the session is ready for stepping."""
        return self.status == "active"

    @property
    def is_terminal(self) -> bool:
        """Whether the session has ended."""
        return self.status in ("completed", "failed", "cancelled", "truncated")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentSession":
        return cls(
            session_id=str(data["session_id"]),
            deployment_id=str(data.get("deployment_id", "")),
            task_id=data.get("task_id", ""),
            status=data.get("status", "provisioning"),
            session_token=data.get("session_token"),
            session_token_expires_at=data.get("session_token_expires_at"),
            observation=data.get("observation"),
            reward=data.get("reward"),
            steps_taken=data.get("steps_taken", 0),
            seed=data.get("seed"),
            created_at=data.get("created_at"),
        )


@dataclass
class EnvironmentStepResult:
    """Result of taking an action in a hosted environment session.

    Attributes:
        observation: Observation after the action.
        reward: Reward for this step (``None`` for dense-off intermediate steps).
        terminated: Episode ended naturally (goal reached, game over).
        truncated: Episode was cut short (max steps, timeout).
        info: Additional step metadata.
    """

    observation: str
    reward: Optional[float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """True if the episode is over (terminated or truncated)."""
        return self.terminated or self.truncated

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentStepResult":
        return cls(
            observation=data.get("observation", ""),
            reward=data.get("reward"),
            terminated=data.get("terminated", False),
            truncated=data.get("truncated", False),
            info=data.get("info", {}),
        )


@dataclass
class EnvironmentRelease:
    """An immutable, content-addressed environment release.

    Attributes:
        release_id: Unique release identifier.
        listing_id: Parent listing.
        release_version: Semantic version string.
        protocol_version: Protocol version (e.g. ``0.4.1``).
        status: ``quarantined``, ``qualified``, or ``revoked``.
        content_digest: Content-addressed SHA-256 digest.
        created_at: ISO-8601 creation timestamp.
    """

    release_id: str
    listing_id: str
    release_version: str
    protocol_version: str
    status: str = "quarantined"
    content_digest: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnvironmentRelease":
        return cls(
            release_id=str(data["release_id"]),
            listing_id=str(data.get("listing_id", "")),
            release_version=data.get("release_version", ""),
            protocol_version=data.get("protocol_version", ""),
            status=data.get("status", "quarantined"),
            content_digest=data.get("content_digest"),
            created_at=data.get("created_at"),
        )


def _opt_str(val: Any) -> Optional[str]:
    """Convert a value to str or None."""
    return str(val) if val is not None else None
