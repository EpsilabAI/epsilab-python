"""Tests for data models.

Covers ``from_dict`` / ``to_dict`` / ``to_json`` round-trips and
default-value handling for every public model class.
"""

import json

from epsilab.models import (
    ArtifactSummary,
    CostEstimate,
    CustomTaskUploadResult,
    EnvironmentListing,
    EnvironmentRelease,
    EnvironmentSession,
    EnvironmentStepResult,
    EvaluationResult,
    EvaluationRunResult,
    GapSummary,
    ModelEstimate,
    RLSession,
    RLStepResult,
    RLTrajectory,
    RunSummary,
    UsageRecord,
)


class TestRunSummary:
    def test_from_dict_minimal(self):
        r = RunSummary.from_dict({"run_id": "r1", "status": "pending"})
        assert r.run_id == "r1"
        assert r.status == "pending"
        assert r.task_count == 0
        assert r.target_model is None
        assert r.reference_models is None
        assert r.progress is None
        assert r.estimated_credits is None
        assert r.evaluation_id is None

    def test_from_dict_full(self):
        r = RunSummary.from_dict(
            {
                "run_id": "r1",
                "status": "completed",
                "name": "test eval",
                "target_model": "openai/gpt-4o",
                "reference_models": [
                    "google/gemini-2.5-flash",
                    "deepseek/deepseek-v4-flash",
                ],
                "task_count": 10,
                "gap_count": 3,
                "created_at": "2026-05-01T00:00:00",
                "estimated_credits": 40,
                "resumed_from": "r0",
                "evaluation_id": "ev-1",
                "progress": {
                    "tasks_completed": 10,
                    "tasks_total": 10,
                    "percent": 100.0,
                },
            }
        )
        assert r.gap_count == 3
        assert r.name == "test eval"
        assert r.target_model == "openai/gpt-4o"
        assert len(r.reference_models) == 2
        assert r.estimated_credits == 40
        assert r.resumed_from == "r0"
        assert r.evaluation_id == "ev-1"
        assert r.progress["percent"] == 100.0
        assert r.created_at == "2026-05-01T00:00:00"

    def test_roundtrip(self):
        r = RunSummary(run_id="r1", status="completed", task_count=5)
        d = r.to_dict()
        assert d["task_count"] == 5
        r2 = RunSummary.from_dict(json.loads(r.to_json()))
        assert r2.run_id == r.run_id


class TestGapSummary:
    def test_from_dict(self):
        g = GapSummary.from_dict(
            {
                "gap_id": "g1",
                "capability": "coding",
                "alpha_score": 0.35,
                "target_score": 0.45,
                "reference_score": 0.8,
            }
        )
        assert g.alpha_score == 0.35
        assert g.capability == "coding"

    def test_defaults(self):
        g = GapSummary.from_dict({"gap_id": "g1", "capability": "math"})
        assert g.alpha_score == 0.0
        assert g.description is None
        assert g.verification is None
        assert g.review_status is None

    def test_verification_and_review_status(self):
        g = GapSummary.from_dict(
            {
                "gap_id": "g1",
                "capability": "coding",
                "alpha_score": 0.5,
                "target_score": 0.3,
                "reference_score": 0.8,
                "verification": "execution",
                "review_status": "approved",
            }
        )
        assert g.verification == "execution"
        assert g.review_status == "approved"

    def test_roundtrip_with_new_fields(self):
        g = GapSummary.from_dict(
            {
                "gap_id": "g1",
                "capability": "math",
                "alpha_score": 0.4,
                "target_score": 0.5,
                "reference_score": 0.9,
                "priority": "high",
                "verification": "judge",
                "review_status": "pending",
            }
        )
        d = g.to_dict()
        assert d["verification"] == "judge"
        assert d["review_status"] == "pending"
        g2 = GapSummary.from_dict(d)
        assert g2.verification == g.verification
        assert g2.review_status == g.review_status


class TestArtifactSummary:
    def test_from_dict(self):
        a = ArtifactSummary.from_dict(
            {
                "artifact_id": "a1",
                "artifact_type": "preference_pair",
                "content": {"prompt": "hello"},
            }
        )
        assert a.artifact_type == "preference_pair"
        assert a.content == {"prompt": "hello"}

    def test_refined_trajectory(self):
        a = ArtifactSummary.from_dict(
            {
                "artifact_id": "a2",
                "artifact_type": "refined_trajectory",
                "content": {
                    "prompt": "implement binary search",
                    "refined_trajectory": [{"action": "write code"}],
                    "original_step_count": 8,
                    "refined_step_count": 5,
                    "compression_ratio": 0.625,
                    "final_output": "def binary_search(arr, target): ...",
                    "score": 1.0,
                    "domain": "coding",
                    "capability": "algorithms",
                },
            }
        )
        assert a.is_refined is True
        assert a.compression_ratio == 0.625
        assert a.content["refined_step_count"] == 5

    def test_non_refined_properties(self):
        a = ArtifactSummary.from_dict(
            {
                "artifact_id": "a3",
                "artifact_type": "preference_pair",
                "content": {"prompt": "hello"},
            }
        )
        assert a.is_refined is False
        assert a.compression_ratio is None


class TestCustomTaskUploadResult:
    def test_from_dict(self):
        r = CustomTaskUploadResult.from_dict(
            {
                "uploaded": 3,
                "task_ids": ["t1", "t2", "t3"],
                "task_names": ["Task one", "Task two", "Task three"],
                "source": "custom",
            }
        )
        assert r.uploaded == 3
        assert len(r.task_ids) == 3
        assert r.task_names == ["Task one", "Task two", "Task three"]
        assert r.source == "custom"

    def test_roundtrip(self):
        r = CustomTaskUploadResult(
            uploaded=2,
            task_ids=["a", "b"],
            task_names=["Alpha", "Beta"],
            source="custom",
        )
        d = r.to_dict()
        assert d["uploaded"] == 2
        r2 = CustomTaskUploadResult.from_dict(d)
        assert r2.task_ids == ["a", "b"]
        assert r2.task_names == ["Alpha", "Beta"]

    def test_defaults(self):
        r = CustomTaskUploadResult.from_dict(
            {
                "uploaded": 1,
                "task_ids": ["x"],
            }
        )
        assert r.task_names == []
        assert r.source == "custom"


class TestUsageRecord:
    def test_from_dict(self):
        u = UsageRecord.from_dict(
            {
                "period": "2026-05",
                "run_count": 12,
                "total_cost_usd": 3.50,
            }
        )
        assert u.run_count == 12
        assert u.total_cost_usd == 3.50


class TestModelEstimate:
    def test_from_dict(self):
        m = ModelEstimate.from_dict(
            {
                "model_id": "openai/gpt-4o",
                "task_count": 25,
                "credits": 40,
                "fresh_tasks": 20,
                "cached_tasks": 5,
                "usd_per_task": 0.01,
                "usd_total": 0.20,
            }
        )
        assert m.model_id == "openai/gpt-4o"
        assert m.credits == 40
        assert m.fresh_tasks == 20
        assert m.cached_tasks == 5

    def test_roundtrip(self):
        m = ModelEstimate(model_id="m/a", task_count=10, credits=20)
        d = m.to_dict()
        m2 = ModelEstimate.from_dict(d)
        assert m2.model_id == "m/a"
        assert m2.credits == 20


class TestCostEstimate:
    def test_from_dict(self):
        e = CostEstimate.from_dict(
            {
                "task_count": 25,
                "total_credits": 65,
                "balance": 500,
                "sufficient": True,
                "per_model": [
                    {
                        "model_id": "m/a",
                        "task_count": 25,
                        "credits": 40,
                        "fresh_tasks": 20,
                        "cached_tasks": 5,
                    },
                    {
                        "model_id": "m/b",
                        "task_count": 25,
                        "credits": 25,
                        "fresh_tasks": 25,
                        "cached_tasks": 0,
                    },
                ],
            }
        )
        assert e.total_credits == 65
        assert e.sufficient is True
        assert len(e.per_model) == 2
        assert e.per_model[0].model_id == "m/a"

    def test_roundtrip(self):
        e = CostEstimate(
            task_count=10,
            total_credits=30,
            balance=100,
            sufficient=True,
            per_model=[ModelEstimate(model_id="m/a", task_count=10, credits=30)],
        )
        d = e.to_dict()
        e2 = CostEstimate.from_dict(d)
        assert e2.total_credits == 30
        assert len(e2.per_model) == 1


class TestEvaluationResult:
    def test_from_dict(self):
        r = EvaluationResult.from_dict(
            {
                "evaluation_id": "ev-1",
                "name": "test eval",
                "total_models": 3,
                "total_estimated_credits": 100,
                "runs": [
                    {
                        "run_id": "r1",
                        "model_id": "openai/gpt-4o",
                        "harness": "codex",
                        "status": "queued",
                        "estimated_credits": 100,
                    },
                ],
            }
        )
        assert r.evaluation_id == "ev-1"
        assert r.name == "test eval"
        assert r.total_models == 3
        assert len(r.runs) == 1
        assert r.runs[0].run_id == "r1"
        assert r.runs[0].harness == "codex"

    def test_roundtrip(self):
        r = EvaluationResult(
            evaluation_id="ev-2",
            name="roundtrip",
            total_models=2,
            total_estimated_credits=50,
            runs=[EvaluationRunResult(run_id="r1", model_id="m/a")],
        )
        d = r.to_dict()
        r2 = EvaluationResult.from_dict(d)
        assert r2.evaluation_id == "ev-2"
        assert r2.runs[0].run_id == "r1"

    def test_to_json(self):
        r = EvaluationResult(
            evaluation_id="ev-3",
            name=None,
            total_models=1,
            total_estimated_credits=10,
        )
        j = json.loads(r.to_json())
        assert j["evaluation_id"] == "ev-3"


# ── RL environments ──────────────────────────────────────────────────


class TestRLSession:
    def test_from_dict_full(self):
        s = RLSession.from_dict(
            {
                "session_id": "sess-123",
                "task_id": "task-456",
                "env_type": "code_sandbox",
                "status": "active",
                "observation": "Write fibonacci...",
                "reward_mode": "partial_credit",
                "total_reward": 0.5,
                "steps_taken": 2,
                "info": {"max_steps": 10},
            }
        )
        assert s.session_id == "sess-123"
        assert s.task_id == "task-456"
        assert s.env_type == "code_sandbox"
        assert s.status == "active"
        assert s.observation == "Write fibonacci..."
        assert s.reward_mode == "partial_credit"
        assert s.total_reward == 0.5
        assert s.steps_taken == 2
        assert s.info["max_steps"] == 10

    def test_from_dict_defaults(self):
        s = RLSession.from_dict(
            {
                "session_id": "s1",
                "task_id": "t1",
                "env_type": "single_turn",
                "status": "active",
            }
        )
        assert s.observation == ""
        assert s.reward_mode == "continuous"
        assert s.total_reward is None
        assert s.steps_taken == 0
        assert s.info == {}

    def test_serialization(self):
        session = RLSession(
            session_id="s1",
            task_id="t1",
            env_type="single_turn",
            status="active",
            info={"max_steps": 1},
        )
        assert RLSession.from_dict(session.to_dict()) == session
        assert json.loads(session.to_json())["info"] == {"max_steps": 1}


class TestRLStepResult:
    def test_from_dict(self):
        r = RLStepResult.from_dict(
            {
                "observation": "2/3 tests pass",
                "reward": 0.67,
                "terminated": False,
                "truncated": False,
                "info": {"tests_passed": 2, "tests_total": 3},
            }
        )
        assert r.observation == "2/3 tests pass"
        assert r.reward == 0.67
        assert r.terminated is False
        assert r.truncated is False
        assert r.done is False

    def test_done_when_terminated(self):
        r = RLStepResult.from_dict(
            {
                "observation": "done",
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
            }
        )
        assert r.done is True

    def test_done_when_truncated(self):
        r = RLStepResult.from_dict(
            {
                "observation": "max steps",
                "reward": 0.0,
                "terminated": False,
                "truncated": True,
            }
        )
        assert r.done is True

    def test_defaults(self):
        r = RLStepResult.from_dict({})
        assert r.observation == ""
        assert r.reward is None
        assert r.terminated is False
        assert r.truncated is False
        assert r.info == {}

    def test_serialization(self):
        result = RLStepResult(
            observation="next",
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"step": 1},
        )
        assert RLStepResult.from_dict(result.to_dict()) == result
        assert json.loads(result.to_json())["reward"] == 0.5


class TestRLTrajectory:
    def test_from_dict(self):
        t = RLTrajectory.from_dict(
            {
                "session_id": "sess-001",
                "task_id": "task-001",
                "env_type": "simulation",
                "status": "completed",
                "total_reward": 3.14,
                "steps_taken": 5,
                "steps": [
                    {"step_idx": 0, "action_hash": "a1", "reward": 0.5},
                    {"step_idx": 1, "action_hash": "a2", "reward": 1.0},
                ],
            }
        )
        assert t.session_id == "sess-001"
        assert t.env_type == "simulation"
        assert t.total_reward == 3.14
        assert t.steps_taken == 5
        assert len(t.steps) == 2
        assert t.steps[0]["action_hash"] == "a1"

    def test_defaults(self):
        t = RLTrajectory.from_dict(
            {
                "session_id": "s1",
                "task_id": "t1",
                "env_type": "single_turn",
                "status": "failed",
            }
        )
        assert t.total_reward is None
        assert t.steps_taken == 0
        assert t.steps == []

    def test_serialization(self):
        trajectory = RLTrajectory(
            session_id="s1",
            task_id="t1",
            env_type="single_turn",
            status="completed",
            steps=[{"step_idx": 0, "reward": 1.0}],
        )
        assert RLTrajectory.from_dict(trajectory.to_dict()) == trajectory
        assert json.loads(trajectory.to_json())["steps"][0]["reward"] == 1.0


# ── Environment Hub & Marketplace ────────────────────────────────────


class TestEnvironmentListing:
    def test_from_dict_full(self):
        l = EnvironmentListing.from_dict(
            {
                "listing_id": "lst-001",
                "namespace_id": "ns-001",
                "slug": "code-sandbox-v1",
                "title": "Code Sandbox v1",
                "summary": "A sandboxed code execution environment",
                "visibility": "public",
                "moderation_state": "approved",
                "recommended_release_id": "rel-001",
                "created_at": "2026-06-01T00:00:00",
                "updated_at": "2026-06-15T00:00:00",
            }
        )
        assert l.listing_id == "lst-001"
        assert l.namespace_id == "ns-001"
        assert l.slug == "code-sandbox-v1"
        assert l.title == "Code Sandbox v1"
        assert l.visibility == "public"
        assert l.moderation_state == "approved"
        assert l.recommended_release_id == "rel-001"
        assert l.created_at == "2026-06-01T00:00:00"

    def test_from_dict_defaults(self):
        l = EnvironmentListing.from_dict(
            {
                "listing_id": "lst-002",
                "namespace_id": "ns-002",
                "slug": "test",
                "title": "Test",
            }
        )
        assert l.summary == ""
        assert l.visibility == "private"
        assert l.moderation_state == "pending"
        assert l.recommended_release_id is None

    def test_roundtrip(self):
        listing = EnvironmentListing(
            listing_id="lst-003",
            namespace_id="ns-003",
            slug="roundtrip",
            title="Roundtrip Test",
        )
        d = listing.to_dict()
        assert d["listing_id"] == "lst-003"
        l2 = EnvironmentListing.from_dict(d)
        assert l2.listing_id == listing.listing_id
        assert l2.slug == listing.slug

    def test_to_json(self):
        listing = EnvironmentListing(
            listing_id="lst-004",
            namespace_id="ns-004",
            slug="json-test",
            title="JSON Test",
        )
        j = json.loads(listing.to_json())
        assert j["listing_id"] == "lst-004"


class TestEnvironmentSession:
    def test_from_dict_full(self):
        s = EnvironmentSession.from_dict(
            {
                "session_id": "sess-001",
                "deployment_id": "dep-001",
                "task_id": "task-001",
                "status": "active",
                "session_token": "tok-abc",
                "session_token_expires_at": "2026-06-01T01:00:00",
                "observation": "Write a function...",
                "reward": 0.5,
                "steps_taken": 3,
                "seed": 42,
                "created_at": "2026-06-01T00:00:00",
            }
        )
        assert s.session_id == "sess-001"
        assert s.deployment_id == "dep-001"
        assert s.status == "active"
        assert s.session_token == "tok-abc"
        assert s.observation == "Write a function..."
        assert s.reward == 0.5
        assert s.steps_taken == 3
        assert s.seed == 42
        assert s.is_active is True
        assert s.is_terminal is False

    def test_from_dict_defaults(self):
        s = EnvironmentSession.from_dict(
            {
                "session_id": "sess-002",
                "deployment_id": "dep-002",
                "task_id": "task-002",
                "status": "provisioning",
            }
        )
        assert s.session_token is None
        assert s.observation is None
        assert s.reward is None
        assert s.steps_taken == 0
        assert s.seed is None
        assert s.is_active is False
        assert s.is_terminal is False

    def test_terminal_states(self):
        for status in ("completed", "failed", "cancelled", "truncated"):
            s = EnvironmentSession.from_dict(
                {
                    "session_id": "s1",
                    "deployment_id": "d1",
                    "task_id": "t1",
                    "status": status,
                }
            )
            assert s.is_terminal is True
            assert s.is_active is False

    def test_roundtrip(self):
        session = EnvironmentSession(
            session_id="sess-003",
            deployment_id="dep-003",
            task_id="task-003",
            status="active",
            session_token="tok-xyz",
        )
        d = session.to_dict()
        s2 = EnvironmentSession.from_dict(d)
        assert s2.session_id == session.session_id
        assert s2.session_token == session.session_token

    def test_to_json(self):
        session = EnvironmentSession(
            session_id="sess-004",
            deployment_id="dep-004",
            task_id="task-004",
            status="completed",
        )
        j = json.loads(session.to_json())
        assert j["status"] == "completed"


class TestEnvironmentStepResult:
    def test_from_dict_full(self):
        r = EnvironmentStepResult.from_dict(
            {
                "observation": "2/3 tests pass",
                "reward": 0.67,
                "terminated": False,
                "truncated": False,
                "info": {"tests_passed": 2, "tests_total": 3},
            }
        )
        assert r.observation == "2/3 tests pass"
        assert r.reward == 0.67
        assert r.terminated is False
        assert r.truncated is False
        assert r.done is False

    def test_done_when_terminated(self):
        r = EnvironmentStepResult.from_dict(
            {
                "observation": "done",
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
            }
        )
        assert r.done is True

    def test_done_when_truncated(self):
        r = EnvironmentStepResult.from_dict(
            {
                "observation": "max steps",
                "reward": 0.0,
                "terminated": False,
                "truncated": True,
            }
        )
        assert r.done is True

    def test_defaults(self):
        r = EnvironmentStepResult.from_dict({})
        assert r.observation == ""
        assert r.reward is None
        assert r.terminated is False
        assert r.truncated is False
        assert r.info == {}

    def test_roundtrip(self):
        result = EnvironmentStepResult(
            observation="next",
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"step": 1},
        )
        assert EnvironmentStepResult.from_dict(result.to_dict()) == result
        assert json.loads(result.to_json())["reward"] == 0.5


class TestEnvironmentRelease:
    def test_from_dict_full(self):
        r = EnvironmentRelease.from_dict(
            {
                "release_id": "rel-001",
                "listing_id": "lst-001",
                "release_version": "1.2.0",
                "protocol_version": "0.4.1",
                "status": "qualified",
                "content_digest": "sha256:abc123",
                "created_at": "2026-06-01T00:00:00",
            }
        )
        assert r.release_id == "rel-001"
        assert r.listing_id == "lst-001"
        assert r.release_version == "1.2.0"
        assert r.protocol_version == "0.4.1"
        assert r.status == "qualified"
        assert r.content_digest == "sha256:abc123"

    def test_defaults(self):
        r = EnvironmentRelease.from_dict(
            {
                "release_id": "rel-002",
                "listing_id": "lst-002",
            }
        )
        assert r.release_version == ""
        assert r.protocol_version == ""
        assert r.status == "quarantined"
        assert r.content_digest is None

    def test_roundtrip(self):
        release = EnvironmentRelease(
            release_id="rel-003",
            listing_id="lst-003",
            release_version="2.0.0",
            protocol_version="0.5.0",
            status="qualified",
        )
        d = release.to_dict()
        r2 = EnvironmentRelease.from_dict(d)
        assert r2.release_id == release.release_id
        assert r2.status == release.status
