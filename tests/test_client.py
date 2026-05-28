"""Tests for EpsilabClient using httpx mock transport.

Each test class covers one client method, using httpx.MockTransport
to simulate API responses without network calls.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from epsilab import Epsilab
from epsilab.exceptions import (
    ApiError,
    AuthError,
    InsufficientCreditsError,
    RateLimitError,
)
from epsilab.models import CostEstimate, EvaluationResult, RunSummary


def _json_response(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=body,
        request=httpx.Request("GET", "http://test"),
    )


def _handler(responses: dict[str, httpx.Response]):
    """Create a mock transport that maps (method, path) -> response."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.raw_path.decode()}"
        for pattern, resp in responses.items():
            if key == pattern or key.startswith(pattern):
                return resp
        return httpx.Response(404, json={"error": "not found"}, request=request)

    return handle


def _make_client(transport, max_retries=0):
    client = Epsilab.__new__(Epsilab)
    client._client = httpx.Client(transport=transport, base_url="http://test")
    client._api_key = "test"
    client._max_retries = max_retries
    client._backoff_base = 0.0  # no real sleep in tests
    return client


class TestCreateRun:
    def test_basic(self):
        transport = httpx.MockTransport(
            lambda req: _json_response({"run_id": "r1", "status": "pending"})
        )
        client = _make_client(transport)

        run = client.create_run("my-model")
        assert run.run_id == "r1"
        assert isinstance(run, RunSummary)

    def test_with_byom(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response({"run_id": "r2", "status": "pending"})

        client = _make_client(httpx.MockTransport(capture))

        client.create_run(
            "internal-v3",
            base_url="https://my.api.com/v1",
            api_key="sk-custom",
        )
        assert captured["body"]["target_model"] == "internal-v3"
        assert captured["body"]["target_config"]["base_url"] == "https://my.api.com/v1"
        assert captured["body"]["target_config"]["api_key"] == "sk-custom"


class TestGetRun:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "run_id": "r1",
                        "status": "completed",
                        "gap_count": 5,
                        "target_model": "openai/gpt-4o",
                        "reference_models": ["google/gemini-2.5-flash"],
                        "estimated_credits": 40,
                        "evaluation_id": "ev-1",
                    }
                )
            )
        )

        run = client.get_run("r1")
        assert run.status == "completed"
        assert run.gap_count == 5
        assert run.target_model == "openai/gpt-4o"
        assert run.reference_models == ["google/gemini-2.5-flash"]
        assert run.estimated_credits == 40
        assert run.evaluation_id == "ev-1"

    def test_url_encodes_run_id(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.raw_path.decode()
            return _json_response({"run_id": "r/1", "status": "completed"})

        client = _make_client(httpx.MockTransport(capture))
        client.get_run("r/1")
        assert captured["path"] == "/v1/runs/r%2F1"


class TestListRuns:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "runs": [
                            {"run_id": "r1", "status": "completed"},
                            {"run_id": "r2", "status": "pending"},
                        ],
                        "total": 2,
                    }
                )
            )
        )

        runs = client.list_runs()
        assert len(runs) == 2
        assert runs[0].run_id == "r1"


class TestDeleteRun:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(lambda req: httpx.Response(204, request=req))
        )

        result = client.delete_run("r1")
        assert result is None


class TestGetGaps:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "gaps": [
                            {
                                "gap_id": "g1",
                                "capability": "coding",
                                "alpha_score": 0.35,
                                "target_score": 0.4,
                                "reference_score": 0.8,
                            },
                        ],
                        "total": 1,
                    }
                )
            )
        )

        gaps = client.get_gaps("r1")
        assert len(gaps) == 1
        assert gaps[0].capability == "coding"


class TestGetArtifacts:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "artifacts": [
                            {"artifact_id": "a1", "artifact_type": "preference_pair"},
                        ],
                        "total": 1,
                    }
                )
            )
        )

        arts = client.get_artifacts("r1")
        assert len(arts) == 1


class TestExportRun:
    def test_to_file(self, tmp_path):
        ndjson = '{"prompt":"a","chosen":"b"}\n{"prompt":"c","chosen":"d"}\n'
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(200, text=ndjson, request=req)
            )
        )

        out = tmp_path / "export.jsonl"
        result = client.export_run("r1", "dpo", path=str(out))
        assert result is None
        assert out.exists()
        assert out.read_text().count("\n") == 2

    def test_to_memory(self):
        ndjson = '{"prompt":"a"}\n{"prompt":"b"}\n'
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(200, text=ndjson, request=req)
            )
        )

        result = client.export_run("r1", "dpo")
        assert isinstance(result, str)
        assert "prompt" in result


class TestStreamExport:
    def test_to_memory(self):
        ndjson = '{"prompt":"a","chosen":"b"}\n{"prompt":"c","chosen":"d"}\n'
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(200, text=ndjson, request=req)
            )
        )

        result = client.stream_export("r1", "dpo")
        assert isinstance(result, str)
        assert "prompt" in result

    def test_to_file(self, tmp_path):
        ndjson = '{"prompt":"a"}\n'
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(200, text=ndjson, request=req)
            )
        )

        out = tmp_path / "streamed.jsonl"
        result = client.stream_export("r1", "sft", path=str(out))
        assert result is None
        assert out.exists()

    def test_filter_params(self):
        def handler(req):
            assert "min_score_gap" in str(req.url)
            assert "min_chosen_score" in str(req.url)
            return httpx.Response(200, text="{}\n", request=req)

        client = _make_client(httpx.MockTransport(handler))
        client.stream_export(
            "r1", "dpo",
            min_score_gap=0.2,
            min_chosen_score=0.5,
        )

    def test_uses_stream_endpoint(self):
        def handler(req):
            assert "/export/stream" in str(req.url)
            return httpx.Response(200, text="{}\n", request=req)

        client = _make_client(httpx.MockTransport(handler))
        client.stream_export("r1", "grpo")


class TestResumeRun:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"run_id": "r3", "status": "pending"})
            )
        )

        run = client.resume_run("r1")
        assert run.run_id == "r3"

    def test_with_new_credentials(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"run_id": "r3", "status": "pending"})

        client = _make_client(httpx.MockTransport(capture))

        client.resume_run("r1", base_url="https://new.api.com/v1", api_key="sk-new")
        assert captured["body"]["target_config"]["base_url"] == "https://new.api.com/v1"


class TestGetUsage:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "usage": [
                            {
                                "period": "2026-05",
                                "run_count": 10,
                                "total_cost_usd": 5.0,
                            }
                        ],
                    }
                )
            )
        )

        usage = client.get_usage()
        assert len(usage) == 1
        assert usage[0].period == "2026-05"


class TestRetryRun:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"run_id": "r2", "status": "pending"})
            )
        )

        run = client.retry_run("r1")
        assert run.run_id == "r2"


class TestWaitForCompletion:
    def test_immediate_completion(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"run_id": "r1", "status": "completed"})
            )
        )

        run = client.wait_for_completion("r1", poll_interval=0, timeout=5)
        assert run.status == "completed"

    def test_timeout(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"run_id": "r1", "status": "running"})
            )
        )

        with pytest.raises(TimeoutError):
            client.wait_for_completion("r1", poll_interval=0, timeout=0)


# ── Multi-model evaluations ──────────────────────────────────────────


class TestCreateEvaluation:
    def test_basic_string_models(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "evaluation_id": "ev-1",
                        "name": "test",
                        "total_models": 2,
                        "total_estimated_credits": 50,
                        "runs": [
                            {
                                "run_id": "r1",
                                "model_id": "openai/gpt-4o",
                                "status": "queued",
                                "estimated_credits": 50,
                            },
                        ],
                    },
                    202,
                )
            )
        )

        result = client.create_evaluation(
            ["openai/gpt-4o", "google/gemini-2.5-flash"],
            name="test",
        )
        assert isinstance(result, EvaluationResult)
        assert result.evaluation_id == "ev-1"
        assert result.total_models == 2
        assert len(result.runs) == 1
        assert result.runs[0].model_id == "openai/gpt-4o"

    def test_dict_models_with_harness(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "evaluation_id": "ev-2",
                    "total_models": 2,
                    "total_estimated_credits": 80,
                    "runs": [{"run_id": "r1", "model_id": "m/a", "status": "queued"}],
                },
                202,
            )

        client = _make_client(httpx.MockTransport(capture))

        client.create_evaluation(
            [
                {"model_id": "m/a", "harness": "codex"},
                "m/b",
            ],
            default_harness="openhands",
            max_tasks=10,
            domains=["coding"],
        )
        body = captured["body"]
        assert body["models"][0]["model_id"] == "m/a"
        assert body["models"][0]["harness"] == "codex"
        assert body["models"][1]["model_id"] == "m/b"
        assert body["default_harness"] == "openhands"
        assert body["max_tasks"] == 10
        assert body["domains"] == ["coding"]

    def test_insufficient_credits(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(
                    402,
                    json={
                        "detail": "Insufficient credits. Need ~100 but balance is 5."
                    },
                    request=req,
                )
            )
        )

        with pytest.raises(InsufficientCreditsError):
            client.create_evaluation(["m/a"])


class TestEstimateEvaluationCost:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "task_count": 25,
                        "total_credits": 65,
                        "balance": 500,
                        "sufficient": True,
                        "per_model": [
                            {
                                "model_id": "openai/gpt-4o",
                                "task_count": 25,
                                "credits": 40,
                                "fresh_tasks": 20,
                                "cached_tasks": 5,
                                "usd_per_task": 0.01,
                                "usd_total": 0.20,
                            },
                            {
                                "model_id": "google/gemini-2.5-flash",
                                "task_count": 25,
                                "credits": 25,
                                "fresh_tasks": 25,
                                "cached_tasks": 0,
                                "usd_per_task": 0.005,
                                "usd_total": 0.125,
                            },
                        ],
                    }
                )
            )
        )

        estimate = client.estimate_evaluation_cost(
            ["openai/gpt-4o", "google/gemini-2.5-flash"],
            max_tasks=25,
        )
        assert isinstance(estimate, CostEstimate)
        assert estimate.task_count == 25
        assert estimate.total_credits == 65
        assert estimate.sufficient is True
        assert len(estimate.per_model) == 2
        assert estimate.per_model[0].model_id == "openai/gpt-4o"
        assert estimate.per_model[0].cached_tasks == 5
        assert estimate.per_model[1].fresh_tasks == 25

    def test_sends_correct_body(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "task_count": 10,
                    "total_credits": 20,
                    "balance": 100,
                    "sufficient": True,
                    "per_model": [],
                }
            )

        client = _make_client(httpx.MockTransport(capture))

        client.estimate_evaluation_cost(
            ["m/a"],
            task_source="custom",
            human_verified_only=True,
        )
        body = captured["body"]
        assert body["task_source"] == "custom"
        assert body["human_verified_only"] is True


# ── Error handling ────────────────────────────────────────────────────


class TestErrors:
    def test_auth_error(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(401, text="unauthorized", request=req)
            )
        )

        with pytest.raises(AuthError):
            client.list_runs()

    def test_rate_limit(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(
                    429,
                    text="too many requests",
                    headers={"Retry-After": "30"},
                    request=req,
                )
            )
        )

        with pytest.raises(RateLimitError) as exc_info:
            client.list_runs()
        assert exc_info.value.retry_after == 30

    def test_api_error(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(500, text="internal error", request=req)
            )
        )

        with pytest.raises(ApiError) as exc_info:
            client.list_runs()
        assert exc_info.value.status_code == 500

    def test_insufficient_credits(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(
                    402,
                    json={"detail": "Insufficient credits"},
                    request=req,
                )
            )
        )

        with pytest.raises(InsufficientCreditsError):
            client.create_run("m/a")

    def test_malformed_retry_after_does_not_crash(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(
                    429,
                    text="too many requests",
                    headers={"Retry-After": "soon"},
                    request=req,
                )
            )
        )

        with pytest.raises(RateLimitError) as exc_info:
            client.list_runs()
        assert exc_info.value.retry_after is None


class TestSetApiKey:
    def test_updates_header(self):
        client = _make_client(
            httpx.MockTransport(lambda req: _json_response({"runs": [], "total": 0}))
        )
        client._api_key = None

        client.set_api_key("sk-new")
        assert client._api_key == "sk-new"
        assert client._client.headers["Authorization"] == "Bearer sk-new"


class TestConfiguration:
    def test_reads_process_environment(self, monkeypatch):
        monkeypatch.setenv("EPSILAB_API_KEY", "sk-env")
        monkeypatch.setenv("EPSILAB_API_BASE", "https://env.example.com")
        client = Epsilab()
        try:
            assert client.api_base == "https://env.example.com"
            assert client._client.headers["Authorization"] == "Bearer sk-env"
        finally:
            client.close()

    def test_dotenv_is_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("EPSILAB_API_KEY", raising=False)
        monkeypatch.delenv("EPSILAB_API_BASE", raising=False)
        (tmp_path / ".env").write_text(
            "EPSILAB_API_KEY=sk-dotenv\nEPSILAB_API_BASE=https://dotenv.example.com\n",
            encoding="utf-8",
        )

        client = Epsilab()
        try:
            assert client.api_base == "https://api.epsilab.com"
            assert "Authorization" not in client._client.headers
        finally:
            client.close()

    def test_dotenv_can_be_enabled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("EPSILAB_API_KEY", raising=False)
        monkeypatch.delenv("EPSILAB_API_BASE", raising=False)
        (tmp_path / ".env").write_text(
            "EPSILAB_API_KEY=sk-dotenv\nEPSILAB_API_BASE=https://dotenv.example.com\n",
            encoding="utf-8",
        )

        client = Epsilab(load_dotenv=True)
        try:
            assert client.api_base == "https://dotenv.example.com"
            assert client._client.headers["Authorization"] == "Bearer sk-dotenv"
        finally:
            client.close()


class TestContextManager:
    def test_enters_and_exits(self):
        transport = httpx.MockTransport(lambda req: _json_response({}))
        with Epsilab.__new__(Epsilab) as client:
            client._client = httpx.Client(transport=transport, base_url="http://test")
            client._api_key = "test"


class TestUploadCustomTasks:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "uploaded": 2,
                        "task_ids": ["t1", "t2"],
                        "task_names": ["Write tests", "Fix bug"],
                        "source": "custom",
                    },
                    201,
                )
            )
        )

        result = client.upload_custom_tasks(
            [
                {
                    "domain": "coding",
                    "capability": "testing",
                    "prompt": "Write tests",
                    "ground_truth": "pass",
                },
                {
                    "domain": "coding",
                    "capability": "debugging",
                    "prompt": "Fix bug",
                    "rubric": "Code compiles",
                },
            ]
        )
        assert result.uploaded == 2
        assert len(result.task_ids) == 2
        assert result.task_names == ["Write tests", "Fix bug"]
        assert result.source == "custom"

    def test_sends_tasks_only(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "uploaded": 1,
                    "task_ids": ["t1"],
                    "task_names": ["Write tests"],
                },
                201,
            )

        client = _make_client(httpx.MockTransport(capture))

        client.upload_custom_tasks(
            [
                {
                    "domain": "legal",
                    "capability": "analysis",
                    "prompt": "Analyze contract",
                    "rubric": "Accurate",
                }
            ],
        )
        assert "visibility" not in captured["body"]
        assert len(captured["body"]["tasks"]) == 1

    def test_insufficient_credits(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(
                    402, json={"detail": "Insufficient credits"}, request=req
                )
            )
        )

        with pytest.raises(InsufficientCreditsError):
            client.upload_custom_tasks(
                [
                    {
                        "domain": "coding",
                        "capability": "testing",
                        "prompt": "Test",
                        "ground_truth": "pass",
                    },
                ]
            )


class TestListTasks:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "tasks": [
                            {
                                "task_id": "t1",
                                "domain": "coding",
                                "capability": "testing",
                            },
                        ],
                        "total": 1,
                    }
                )
            )
        )

        result = client.list_tasks(source="custom")
        assert result["total"] == 1
        assert len(result["tasks"]) == 1


class TestDeleteTask:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(lambda req: httpx.Response(204, request=req))
        )

        result = client.delete_task("t1")
        assert result is None


class TestClassifyTasks:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "tasks": [
                            {
                                "prompt": "Fix this bug",
                                "domain": "coding",
                                "capability": "debugging",
                                "difficulty": "hard",
                                "verification": "judge",
                            },
                        ],
                        "domains_found": ["coding"],
                    }
                )
            )
        )

        result = client.classify_tasks([{"prompt": "Fix this bug"}])
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["domain"] == "coding"
        assert "coding" in result["domains_found"]

    def test_sends_correct_body(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            captured["path"] = req.url.raw_path.decode()
            return _json_response({"tasks": [], "domains_found": []})

        client = _make_client(httpx.MockTransport(capture))

        client.classify_tasks(
            [
                {"prompt": "Task 1", "expected_answer": "Answer 1"},
                {"prompt": "Task 2", "rubric": "Must be good"},
            ]
        )
        assert captured["path"] == "/v1/tasks/classify"
        assert len(captured["body"]["tasks"]) == 2
        assert captured["body"]["tasks"][0]["expected_answer"] == "Answer 1"


# ── New endpoints ─────────────────────────────────────────────────────


class TestListModels:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "models": [
                            {
                                "model_id": "openai/gpt-4o",
                                "provider": "openai",
                                "name": "GPT-4o",
                            },
                        ],
                    }
                )
            )
        )
        result = client.list_models(search="gpt")
        assert len(result["models"]) == 1
        assert result["models"][0]["provider"] == "openai"


class TestCancelRun:
    def test_success(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"run_id": "r1", "status": "cancelled"})
            )
        )
        run = client.cancel_run("r1")
        assert run.status == "cancelled"


class TestSuggestScope:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "domains": ["coding", "math"],
                        "task_scope": "standard",
                    }
                )
            )
        )
        result = client.suggest_scope("test coding and math abilities")
        assert "coding" in result["domains"]

    def test_sends_instructions(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response({"domains": [], "task_scope": "standard"})

        client = _make_client(httpx.MockTransport(capture))
        client.suggest_scope("evaluate reasoning")
        assert captured["body"]["instructions"] == "evaluate reasoning"


class TestRequestReview:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"status": "submitted", "gaps_requested": 2})
            )
        )
        result = client.request_review("r1", ["g1", "g2"], message="Please check these")
        assert result["gaps_requested"] == 2

    def test_sends_correct_body(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response({"status": "submitted"})

        client = _make_client(httpx.MockTransport(capture))
        client.request_review("r1", ["g1"], message="urgent")
        assert captured["body"]["gap_ids"] == ["g1"]
        assert captured["body"]["message"] == "urgent"


class TestForge:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(lambda req: _json_response({"tasks_created": 5}))
        )
        result = client.forge("r1")
        assert result["tasks_created"] == 5


class TestGetTask:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "task_id": "t1",
                        "domain": "coding",
                        "capability": "testing",
                    }
                )
            )
        )
        result = client.get_task("t1")
        assert result["task_id"] == "t1"


class TestCreateTask:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "task_id": "t-new",
                    "domain": "coding",
                },
                201,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_task(
            {
                "domain": "coding",
                "capability": "testing",
                "prompt": "Write a test",
                "rubric": "Tests pass",
            }
        )
        assert result["task_id"] == "t-new"
        assert captured["body"]["domain"] == "coding"


class TestGetTaskUploadLimits:
    def test_basic(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {"max_file_size": 10485760, "max_tasks": 100}
                )
            )
        )
        result = client.get_task_upload_limits()
        assert result["max_tasks"] == 100


class TestLeaderboard:
    def test_leaderboard(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {"models": [{"model_id": "m/a", "score": 0.85}]}
                )
            )
        )
        result = client.get_leaderboard()
        assert len(result["models"]) == 1

    def test_domain_leaderboard(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {"domains": {"coding": [{"model_id": "m/a"}]}}
                )
            )
        )
        result = client.get_domain_leaderboard()
        assert "coding" in result["domains"]

    def test_cost_analysis(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {"models": [{"model_id": "m/a", "cost_per_score": 0.01}]}
                )
            )
        )
        result = client.get_cost_analysis()
        assert len(result["models"]) == 1

    def test_precomputed_insights(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response({"coding": {"best_model": "m/a"}})
            )
        )
        result = client.get_precomputed_insights()
        assert result["coding"]["best_model"] == "m/a"


class TestApiKeys:
    def test_list(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "keys": [{"key_id": "k1", "label": "dev"}],
                    }
                )
            )
        )
        result = client.list_api_keys()
        assert len(result["keys"]) == 1

    def test_create(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response({"key_id": "k2", "key": "sk-new-key"}, 201)

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_api_key(label="CI key")
        assert result["key_id"] == "k2"
        assert captured["body"]["label"] == "CI key"

    def test_revoke(self):
        client = _make_client(
            httpx.MockTransport(lambda req: httpx.Response(204, request=req))
        )
        result = client.revoke_api_key("k1")
        assert result is None


class TestGetUsagePeriod:
    def test_with_period(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["params"] = str(req.url)
            return _json_response(
                {
                    "usage": [{"period": "2026-04", "run_count": 5}],
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        usage = client.get_usage(period="2026-04")
        assert len(usage) == 1
        assert usage[0].period == "2026-04"
        assert "period=2026-04" in captured["params"]


# ── Retry behaviour ───────────────────────────────────────────────────


class TestRetryMechanism:
    """Verify automatic retry on 429 and 5xx responses."""

    def test_retries_on_429_then_succeeds(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return httpx.Response(
                    429,
                    text="rate limited",
                    headers={"Retry-After": "0"},
                    request=req,
                )
            return _json_response({"runs": [], "total": 0})

        client = _make_client(httpx.MockTransport(handler), max_retries=3)
        runs = client.list_runs()
        assert runs == []
        assert call_count["n"] == 3

    def test_retries_on_500_then_succeeds(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(500, text="internal error", request=req)
            return _json_response({"runs": [], "total": 0})

        client = _make_client(httpx.MockTransport(handler), max_retries=2)
        runs = client.list_runs()
        assert runs == []
        assert call_count["n"] == 2

    def test_raises_after_max_retries_exhausted(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: httpx.Response(502, text="bad gateway", request=req)
            ),
            max_retries=2,
        )
        with pytest.raises(ApiError) as exc_info:
            client.list_runs()
        assert exc_info.value.status_code == 502

    def test_no_retry_on_4xx(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(400, text="bad request", request=req)

        client = _make_client(httpx.MockTransport(handler), max_retries=3)
        with pytest.raises(ApiError):
            client.list_runs()
        assert call_count["n"] == 1

    def test_no_retry_when_disabled(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(500, text="error", request=req)

        client = _make_client(httpx.MockTransport(handler), max_retries=0)
        with pytest.raises(ApiError):
            client.list_runs()
        assert call_count["n"] == 1

    def test_raw_request_retries(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(503, text="unavailable", request=req)
            return httpx.Response(200, text='{"data":"ok"}\n', request=req)

        client = _make_client(httpx.MockTransport(handler), max_retries=2)
        text = client.export_run("r1", "dpo")
        assert "ok" in text
        assert call_count["n"] == 2

    def test_retries_transport_errors(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("connection reset", request=req)
            return _json_response({"runs": [], "total": 0})

        client = _make_client(httpx.MockTransport(handler), max_retries=2)
        assert client.list_runs() == []
        assert call_count["n"] == 2


# ── Auto-paginating iterators ────────────────────────────────────────


class TestIterRuns:
    def test_single_page(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {
                        "runs": [{"run_id": "r1", "status": "completed"}],
                        "total": 1,
                    }
                )
            )
        )
        runs = list(client.iter_runs(page_size=10))
        assert len(runs) == 1
        assert runs[0].run_id == "r1"

    def test_multi_page(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response(
                    {
                        "runs": [
                            {"run_id": "r1", "status": "completed"},
                            {"run_id": "r2", "status": "completed"},
                        ],
                    }
                )
            return _json_response({"runs": [{"run_id": "r3", "status": "pending"}]})

        client = _make_client(httpx.MockTransport(handler))
        runs = list(client.iter_runs(page_size=2))
        assert [r.run_id for r in runs] == ["r1", "r2", "r3"]
        assert call_count["n"] == 2

    def test_empty(self):
        client = _make_client(
            httpx.MockTransport(lambda req: _json_response({"runs": []}))
        )
        assert list(client.iter_runs()) == []


class TestIterArtifacts:
    def test_paginates(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response(
                    {
                        "artifacts": [
                            {"artifact_id": "a1", "artifact_type": "preference_pair"},
                            {"artifact_id": "a2", "artifact_type": "preference_pair"},
                        ],
                    }
                )
            return _json_response({"artifacts": []})

        client = _make_client(httpx.MockTransport(handler))
        arts = list(client.iter_artifacts("r1", page_size=2))
        assert len(arts) == 2


class TestIterTasks:
    def test_paginates(self):
        call_count = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _json_response(
                    {
                        "tasks": [{"task_id": "t1"}, {"task_id": "t2"}],
                        "total": 3,
                    }
                )
            return _json_response({"tasks": [{"task_id": "t3"}], "total": 3})

        client = _make_client(httpx.MockTransport(handler))
        tasks = list(client.iter_tasks(page_size=2))
        assert [t["task_id"] for t in tasks] == ["t1", "t2", "t3"]
        assert call_count["n"] == 2


class TestCodeHygiene:
    """Basic sanity checks for SDK source quality."""

    def test_consistent_api_version(self):
        """All API paths should use the same version prefix."""
        import pathlib
        import re

        src = pathlib.Path(__file__).parent.parent / "epsilab"
        api_path_re = re.compile(r'["\'](/v\d+/[^\s"\'{}]+)')
        for py in src.glob("*.py"):
            text = py.read_text()
            for match in api_path_re.finditer(text):
                path = match.group(1)
                assert path.startswith("/v1/"), (
                    f"Inconsistent API version in {py.name}: {match.group()}"
                )

    def test_env_vars_use_sdk_prefix(self):
        """Env vars referenced in source should use the SDK prefix."""
        import pathlib
        import re

        src = pathlib.Path(__file__).parent.parent / "epsilab"
        env_re = re.compile(r'env\.get\(["\']([A-Z_]+)')
        for py in src.glob("*.py"):
            text = py.read_text()
            for match in env_re.finditer(text):
                var = match.group(1)
                assert var.startswith("EPSILAB_"), (
                    f"Unexpected env var in {py.name}: {var}"
                )
