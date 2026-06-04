"""Tests for data models.

Covers ``from_dict`` / ``to_dict`` / ``to_json`` round-trips and
default-value handling for every public model class.
"""

import json

from epsilab.models import (
    ArtifactSummary,
    CostEstimate,
    CustomTaskUploadResult,
    EvaluationResult,
    EvaluationRunResult,
    GapSummary,
    ModelEstimate,
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
        r = RunSummary.from_dict({
            "run_id": "r1", "status": "completed",
            "name": "test eval",
            "target_model": "openai/gpt-4o",
            "reference_models": ["google/gemini-2.5-flash", "deepseek/deepseek-v4-flash"],
            "task_count": 10, "gap_count": 3,
            "created_at": "2026-05-01T00:00:00",
            "estimated_credits": 40,
            "resumed_from": "r0",
            "evaluation_id": "ev-1",
            "progress": {"tasks_completed": 10, "tasks_total": 10, "percent": 100.0},
        })
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
        g = GapSummary.from_dict({
            "gap_id": "g1", "capability": "coding",
            "alpha_score": 0.35, "target_score": 0.45,
            "reference_score": 0.8,
        })
        assert g.alpha_score == 0.35
        assert g.capability == "coding"

    def test_defaults(self):
        g = GapSummary.from_dict({"gap_id": "g1", "capability": "math"})
        assert g.alpha_score == 0.0
        assert g.description is None
        assert g.verification is None
        assert g.review_status is None

    def test_verification_and_review_status(self):
        g = GapSummary.from_dict({
            "gap_id": "g1", "capability": "coding",
            "alpha_score": 0.5, "target_score": 0.3,
            "reference_score": 0.8,
            "verification": "execution",
            "review_status": "approved",
        })
        assert g.verification == "execution"
        assert g.review_status == "approved"

    def test_roundtrip_with_new_fields(self):
        g = GapSummary.from_dict({
            "gap_id": "g1", "capability": "math",
            "alpha_score": 0.4, "target_score": 0.5,
            "reference_score": 0.9, "priority": "high",
            "verification": "judge", "review_status": "pending",
        })
        d = g.to_dict()
        assert d["verification"] == "judge"
        assert d["review_status"] == "pending"
        g2 = GapSummary.from_dict(d)
        assert g2.verification == g.verification
        assert g2.review_status == g.review_status


class TestArtifactSummary:
    def test_from_dict(self):
        a = ArtifactSummary.from_dict({
            "artifact_id": "a1",
            "artifact_type": "preference_pair",
            "content": {"prompt": "hello"},
        })
        assert a.artifact_type == "preference_pair"
        assert a.content == {"prompt": "hello"}

    def test_refined_trajectory(self):
        a = ArtifactSummary.from_dict({
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
        })
        assert a.is_refined is True
        assert a.compression_ratio == 0.625
        assert a.content["refined_step_count"] == 5

    def test_non_refined_properties(self):
        a = ArtifactSummary.from_dict({
            "artifact_id": "a3",
            "artifact_type": "preference_pair",
            "content": {"prompt": "hello"},
        })
        assert a.is_refined is False
        assert a.compression_ratio is None


class TestCustomTaskUploadResult:
    def test_from_dict(self):
        r = CustomTaskUploadResult.from_dict({
            "uploaded": 3,
            "task_ids": ["t1", "t2", "t3"],
            "task_names": ["Task one", "Task two", "Task three"],
            "source": "custom",
        })
        assert r.uploaded == 3
        assert len(r.task_ids) == 3
        assert r.task_names == ["Task one", "Task two", "Task three"]
        assert r.source == "custom"

    def test_roundtrip(self):
        r = CustomTaskUploadResult(
            uploaded=2, task_ids=["a", "b"],
            task_names=["Alpha", "Beta"],
            source="custom",
        )
        d = r.to_dict()
        assert d["uploaded"] == 2
        r2 = CustomTaskUploadResult.from_dict(d)
        assert r2.task_ids == ["a", "b"]
        assert r2.task_names == ["Alpha", "Beta"]

    def test_defaults(self):
        r = CustomTaskUploadResult.from_dict({
            "uploaded": 1, "task_ids": ["x"],
        })
        assert r.task_names == []
        assert r.source == "custom"


class TestUsageRecord:
    def test_from_dict(self):
        u = UsageRecord.from_dict({
            "period": "2026-05",
            "run_count": 12,
            "total_cost_usd": 3.50,
        })
        assert u.run_count == 12
        assert u.total_cost_usd == 3.50


class TestModelEstimate:
    def test_from_dict(self):
        m = ModelEstimate.from_dict({
            "model_id": "openai/gpt-4o",
            "task_count": 25,
            "credits": 40,
            "fresh_tasks": 20,
            "cached_tasks": 5,
            "usd_per_task": 0.01,
            "usd_total": 0.20,
        })
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
        e = CostEstimate.from_dict({
            "task_count": 25,
            "total_credits": 65,
            "balance": 500,
            "sufficient": True,
            "per_model": [
                {"model_id": "m/a", "task_count": 25, "credits": 40,
                 "fresh_tasks": 20, "cached_tasks": 5},
                {"model_id": "m/b", "task_count": 25, "credits": 25,
                 "fresh_tasks": 25, "cached_tasks": 0},
            ],
        })
        assert e.total_credits == 65
        assert e.sufficient is True
        assert len(e.per_model) == 2
        assert e.per_model[0].model_id == "m/a"

    def test_roundtrip(self):
        e = CostEstimate(
            task_count=10, total_credits=30, balance=100, sufficient=True,
            per_model=[ModelEstimate(model_id="m/a", task_count=10, credits=30)],
        )
        d = e.to_dict()
        e2 = CostEstimate.from_dict(d)
        assert e2.total_credits == 30
        assert len(e2.per_model) == 1


class TestEvaluationResult:
    def test_from_dict(self):
        r = EvaluationResult.from_dict({
            "evaluation_id": "ev-1",
            "name": "test eval",
            "total_models": 3,
            "total_estimated_credits": 100,
            "runs": [
                {"run_id": "r1", "model_id": "openai/gpt-4o",
                 "harness": "codex", "status": "queued",
                 "estimated_credits": 100},
            ],
        })
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
            evaluation_id="ev-3", name=None,
            total_models=1, total_estimated_credits=10,
        )
        j = json.loads(r.to_json())
        assert j["evaluation_id"] == "ev-3"
