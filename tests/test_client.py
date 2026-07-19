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
from epsilab.models import (
    AgentRunContext,
    AgentToolCall,
    AgentTurn,
    AgentUsage,
    ApplicationTool,
    ApplicationToolRelease,
    CostEstimate,
    EnvironmentListing,
    EnvironmentRelease,
    EnvironmentSession,
    EnvironmentStepResult,
    EvaluationResult,
    RunSummary,
)


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
            "r1",
            "dpo",
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
        import epsilab.client as _client_mod
        monkeypatch.setattr(_client_mod, "_CREDENTIALS_FILE", tmp_path / "nonexistent.json")

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

    def test_loads_stored_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EPSILAB_API_KEY", raising=False)
        monkeypatch.delenv("EPSILAB_API_BASE", raising=False)
        import epsilab.client as _client_mod

        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(
            '{"profiles":{"default":{"api_key":"sk-stored"}},"active_profile":"default","api_base":"https://stored.example.com"}',
            encoding="utf-8",
        )
        monkeypatch.setattr(_client_mod, "_CREDENTIALS_FILE", creds_file)

        client = Epsilab()
        try:
            assert client._client.headers["Authorization"] == "Bearer sk-stored"
            assert client.api_base == "https://stored.example.com"
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


# ── Voice evaluations ────────────────────────────────────────────────


class TestRegisterVoiceAsset:
    def test_registers_asset(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"asset": {"asset_id": "a1", "uri": "gs://bucket/audio.wav"}},
                status=201,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.register_voice_asset(
            "a1", "gs://bucket/audio.wav", language="en", duration_s=3.5
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/voice/assets/register"
        assert captured["body"]["asset_id"] == "a1"
        assert captured["body"]["language"] == "en"
        assert captured["body"]["duration_s"] == 3.5
        assert result["asset"]["asset_id"] == "a1"


class TestCreateVoiceTask:
    def test_creates_task(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"task_id": "vt-1", "domain": "voice", "verification": "wer"},
                status=201,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_voice_task(
            "vt-1",
            "voice_asr",
            "Transcribe this audio clip.",
            "speech_recognition",
            assets=[{"asset_id": "a1", "uri": "gs://bucket/clip.wav"}],
            ground_truth="Hello world",
        )
        assert captured["body"]["task_id"] == "vt-1"
        assert captured["body"]["task_type"] == "voice_asr"
        assert captured["body"]["ground_truth"] == "Hello world"
        assert len(captured["body"]["assets"]) == 1
        assert result["task_id"] == "vt-1"


class TestCreateVoiceRun:
    def test_creates_run(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"run_id": "vr-1", "status": "queued"},
                status=202,
            )

        client = _make_client(httpx.MockTransport(capture))
        run = client.create_voice_run(
            "openai/whisper-large-v3",
            reference_models=["deepgram/nova-2"],
            task_ids=["vt-1", "vt-2"],
            name="ASR eval",
        )
        assert captured == {
            "method": "POST",
            "path": "/v1/voice/runs",
            "body": {
                "target_model": "openai/whisper-large-v3",
                "reference_models": ["deepgram/nova-2"],
                "task_ids": ["vt-1", "vt-2"],
                "name": "ASR eval",
                "reference_mode": "best_on_task",
                "reference_top_k": 3,
                "exploratory": False,
            },
        }
        assert run.run_id == "vr-1"
        assert run.status == "queued"


class TestGetVoiceSlices:
    def test_returns_slices(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "run_id": "vr-1",
                    "slices": [{"name": "noisy", "wer": 0.12, "count": 5}],
                    "total_results": 20,
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_voice_slices("vr-1")
        assert captured == {"method": "GET", "path": "/v1/voice/runs/vr-1/slices"}
        assert len(result["slices"]) == 1
        assert result["total_results"] == 20


class TestGetVoiceTimeline:
    def test_returns_timeline(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "run_id": "vr-1",
                    "task_id": "vt-1",
                    "model_alias": "target_model",
                    "event_timeline": [{"t": 0.0, "type": "chunk"}],
                    "output_assets": [],
                    "scenario_checks": {},
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_voice_timeline("vr-1", "vt-1")
        assert captured == {
            "method": "GET",
            "path": "/v1/voice/runs/vr-1/timeline/vt-1",
        }
        assert result["model_alias"] == "target_model"
        assert len(result["event_timeline"]) == 1


class TestRouteVoice:
    def test_routes_workload(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "strategy": "quality_first",
                    "confidence": 0.85,
                    "primary": {"model_id": "openai/whisper-large-v3"},
                    "candidates": [{"model_id": "openai/whisper-large-v3"}],
                    "explanation": "Best WER on similar tasks",
                    "router_id": "global",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.route_voice(
            "Transcribe a noisy phone call",
            task_type="voice_asr",
            language="en",
            max_latency_s=2.0,
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/voice/route"
        assert captured["body"]["language"] == "en"
        assert captured["body"]["max_latency_s"] == 2.0
        assert result["confidence"] == 0.85
        assert result["primary"]["model_id"] == "openai/whisper-large-v3"


# ── RL environments ──────────────────────────────────────────────────


class TestCreateRLSession:
    def test_creates_session(self):
        transport = httpx.MockTransport(
            lambda req: _json_response(
                {
                    "session_id": "sess-001",
                    "task_id": "fib-task",
                    "env_type": "code_sandbox",
                    "status": "active",
                    "observation": "Implement fibonacci...",
                    "reward_mode": "partial_credit",
                    "info": {"max_steps": 5},
                }
            )
        )
        client = _make_client(transport)
        session = client.create_rl_session(
            "fib-task", env_type="code_sandbox", reward_mode="partial_credit"
        )
        assert session.session_id == "sess-001"
        assert session.task_id == "fib-task"
        assert session.env_type == "code_sandbox"
        assert session.status == "active"
        assert "fibonacci" in session.observation.lower()
        assert session.reward_mode == "partial_credit"

    def test_sends_optional_params(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "session_id": "sess-002",
                    "task_id": "sim-task",
                    "env_type": "simulation",
                    "status": "active",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        client.create_rl_session(
            "sim-task", env_type="simulation", seed=42, max_steps=100
        )
        assert captured == {
            "method": "POST",
            "path": "/v1/rl/sessions",
            "body": {
                "task_id": "sim-task",
                "env_type": "simulation",
                "reward_mode": "continuous",
                "seed": 42,
                "max_steps": 100,
            },
        }


class TestRLStep:
    def test_intermediate_step(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "observation": "1/3 tests passed.",
                    "reward": None,
                    "terminated": False,
                    "truncated": False,
                    "info": {"step": 1, "tests_passed": 1, "tests_total": 3},
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.rl_step("sess-001", "def fib(n): return n")
        assert captured == {
            "method": "POST",
            "path": "/v1/rl/sessions/sess-001/step",
            "body": {"action": "def fib(n): return n"},
        }
        assert result.reward is None
        assert result.done is False
        assert result.info["tests_passed"] == 1

    def test_terminal_step(self):
        transport = httpx.MockTransport(
            lambda req: _json_response(
                {
                    "observation": "3/3 tests passed.",
                    "reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {"all_passed": True},
                }
            )
        )
        client = _make_client(transport)
        result = client.rl_step("sess-001", "def fib(n): ...")
        assert result.reward == 1.0
        assert result.terminated is True
        assert result.done is True


class TestGetRLTrajectory:
    def test_retrieves_trajectory(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "session_id": "sess-001",
                    "task_id": "fib-task",
                    "env_type": "code_sandbox",
                    "status": "completed",
                    "total_reward": 1.0,
                    "steps_taken": 2,
                    "steps": [
                        {
                            "step_idx": 0,
                            "action_hash": "abc123",
                            "reward": None,
                            "terminated": False,
                            "truncated": False,
                        },
                        {
                            "step_idx": 1,
                            "action_hash": "def456",
                            "reward": 1.0,
                            "terminated": True,
                            "truncated": False,
                        },
                    ],
                    "trace_events": [
                        {
                            "event_id": "evt-1",
                            "event_idx": 0,
                            "event_type": "reasoning",
                            "payload": {"content": "plan"},
                        }
                    ],
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        traj = client.get_rl_trajectory("sess-001")
        assert captured == {
            "method": "GET",
            "path": "/v1/rl/sessions/sess-001/trajectory",
        }
        assert traj.session_id == "sess-001"
        assert traj.total_reward == 1.0
        assert len(traj.steps) == 2
        assert traj.steps[1]["reward"] == 1.0
        assert traj.trace_events[0].event_type == "reasoning"


class TestVerifyRLTrajectory:
    def test_verified_trajectory(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "verified": True,
                    "steps_replayed": 3,
                    "divergences": [],
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.verify_rl_trajectory("sess-001")
        assert captured == {
            "method": "POST",
            "path": "/v1/rl/sessions/sess-001/verify",
        }
        assert result["verified"] is True
        assert result["divergences"] == []


class TestGetRLCurriculum:
    def test_returns_curriculum_batch(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response(
                {
                    "curriculum": {
                        "frontier_tasks": ["t1", "t2"],
                        "exploration_tasks": ["t3"],
                        "retention_tasks": [],
                        "training_total": 3,
                    },
                    "stats_summary": {
                        "total_tasks_available": 100,
                        "frontier_count": 10,
                    },
                    "difficulty_profile": {},
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_rl_curriculum(env_type="code_sandbox", batch_size=4)
        assert captured == {
            "method": "GET",
            "path": "/v1/rl/curriculum",
            "params": {"batch_size": "4", "env_type": "code_sandbox"},
        }
        assert len(result["curriculum"]["frontier_tasks"]) == 2
        assert result["stats_summary"]["frontier_count"] == 10


class TestExportRLSessions:
    def test_exports_grpo_format(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "schema_version": "rl-export-v1",
                    "format": "grpo",
                    "n_records": 1,
                    "total_records": 1,
                    "records": [{"prompt": "...", "completions": []}],
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.export_rl_sessions("grpo", env_type="code_sandbox")
        assert captured == {
            "method": "POST",
            "path": "/v1/rl/exports",
            "body": {"format": "grpo", "env_type": "code_sandbox"},
        }
        assert result["format"] == "grpo"
        assert len(result["records"]) == 1
        assert result["schema_version"] == "rl-export-v1"


class TestCloseRLSession:
    def test_closes_session(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "session_id": "sess-001",
                    "status": "failed",
                    "total_reward": 0.0,
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.close_rl_session("sess-001", reason="test_complete")
        assert captured == {
            "method": "POST",
            "path": "/v1/rl/sessions/sess-001/close",
            "body": {"reason": "test_complete"},
        }
        assert result["status"] == "failed"


class TestListRLEnvironments:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({
                "environments": [{"env_id": "t1", "env_type": "code_sandbox"}],
                "total": 1,
            })

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_rl_environments(domain="coding", limit=10)
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/rl/environments"
        assert captured["params"]["domain"] == "coding"
        assert captured["params"]["limit"] == "10"
        assert result["total"] == 1


class TestListRLSessions:
    def test_with_filters(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({
                "sessions": [{"session_id": "s1", "status": "completed"}],
                "total": 1,
            })

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_rl_sessions(status="completed", task_id="t1")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/rl/sessions"
        assert captured["params"]["status"] == "completed"
        assert captured["params"]["task_id"] == "t1"
        assert result["sessions"][0]["status"] == "completed"


class TestGetRLStats:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({
                "session_counts": {"total": 50, "completed": 40},
                "reward_distribution": {"mean": 0.72},
                "task_count": 10,
            })

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_rl_stats(env_type="code_sandbox")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/rl/stats"
        assert captured["params"]["env_type"] == "code_sandbox"
        assert result["session_counts"]["total"] == 50


class TestGetMatrixModels:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({
                "models": [{"model_id": "openai/gpt-4o", "score": 0.85}],
            })

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_models(modality="text", min_tasks=10)
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/models"
        assert captured["params"]["modality"] == "text"
        assert captured["params"]["min_tasks"] == "10"
        assert result["models"][0]["model_id"] == "openai/gpt-4o"


class TestGetMatrixModelProfile:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({
                "model_id": "openai/gpt-4o",
                "coverage": {"total_tasks": 100},
                "strengths": ["coding", "math"],
            })

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_model_profile("openai/gpt-4o")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/models/openai/gpt-4o/profile"
        assert result["strengths"] == ["coding", "math"]


class TestGetMatrixGaps:
    def test_with_filters(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"gaps": [{"domain": "coding", "gap": 0.15}]})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_gaps(model_id="openai/gpt-4o", domain="coding")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/gaps"
        assert captured["params"]["model_id"] == "openai/gpt-4o"
        assert captured["params"]["domain"] == "coding"
        assert result["gaps"][0]["gap"] == 0.15


class TestGetMatrixDomains:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"domains": [{"domain": "coding", "avg_score": 0.8}]})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_domains(model_id="openai/gpt-4o")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/domains"
        assert result["domains"][0]["domain"] == "coding"


class TestGetMatrixScores:
    def test_with_pagination(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"scores": [], "total": 0})

        client = _make_client(httpx.MockTransport(capture))
        client.get_matrix_scores(domain="math", limit=100, offset=50)
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/scores"
        assert captured["params"]["domain"] == "math"
        assert captured["params"]["limit"] == "100"
        assert captured["params"]["offset"] == "50"


class TestGetMatrixArtifacts:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"artifacts": [{"type": "dpo"}], "total": 1})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_artifacts(artifact_type="dpo", model_id="openai/gpt-4o")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/artifacts"
        assert captured["params"]["artifact_type"] == "dpo"
        assert result["artifacts"][0]["type"] == "dpo"


class TestGetMatrixInsights:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"rankings": [], "recommendations": []})

        client = _make_client(httpx.MockTransport(capture))
        client.get_matrix_insights(model_id="openai/gpt-4o", refresh=True)
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/insights"
        assert captured["params"]["model_id"] == "openai/gpt-4o"
        assert captured["params"]["refresh"] == "true"


class TestGetMatrixCoverage:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"coverage": {}})

        client = _make_client(httpx.MockTransport(capture))
        client.get_matrix_coverage(
            domain="coding", models=["openai/gpt-4o", "google/gemini-2.5-flash"]
        )
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/coverage"
        assert captured["params"]["domain"] == "coding"
        assert captured["params"]["models"] == "openai/gpt-4o,google/gemini-2.5-flash"


class TestGetMatrixModelGaps:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"gaps": [{"capability": "reasoning", "gap": 0.2}]})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_model_gaps("openai/gpt-4o", min_gap=0.1, limit=20)
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/models/openai/gpt-4o/gaps"
        assert captured["params"]["min_gap"] == "0.1"
        assert captured["params"]["limit"] == "20"
        assert result["gaps"][0]["gap"] == 0.2


class TestGetMatrixModelCapabilities:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"capabilities": [{"name": "coding", "score": 0.9}]})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_matrix_model_capabilities("openai/gpt-4o", domain="coding")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/matrix/models/openai/gpt-4o/capabilities"
        assert result["capabilities"][0]["score"] == 0.9


# ══════════════════════════════════════════════════════════════════════
# Environment Hub & Marketplace
# ══════════════════════════════════════════════════════════════════════


class TestListEnvironmentListings:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            captured["authorization"] = req.headers.get("Authorization")
            return _json_response(
                [
                    {
                        "listing_id": "lst-1",
                        "namespace": "community",
                        "slug": "my-env",
                        "title": "My Env",
                        "listing_revision": 2,
                        "is_owner": False,
                        "release_id": "rel-1",
                    }
                ]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_environment_listings(
            query="agent tools",
            domain="software-engineering",
            sort_by="stars",
            limit=10,
            offset=5,
        )
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-listings"
        assert captured["params"]["limit"] == "10"
        assert captured["params"]["offset"] == "5"
        assert captured["params"]["q"] == "agent tools"
        assert captured["params"]["domain"] == "software-engineering"
        assert captured["params"]["sort_by"] == "stars"
        assert len(result) == 1
        assert isinstance(result[0], EnvironmentListing)
        assert result[0].listing_id == "lst-1"
        assert result[0].slug == "my-env"
        assert result[0].namespace == "community"
        assert result[0].revision == 2
        assert result[0].release_id == "rel-1"
        assert captured["authorization"] is None

    def test_get_direct_listing(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            return _json_response(
                {
                    "listing_id": "lst-unlisted",
                    "namespace": "team",
                    "slug": "preview",
                    "title": "Preview",
                    "visibility": "unlisted",
                    "is_owner": False,
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        listing = client.get_environment_listing("lst-unlisted")
        assert captured["path"] == "/v1/environment-listings/lst-unlisted"
        assert listing.visibility == "unlisted"


class TestApplicationTools:
    def test_discovery_and_release(self):
        def capture(req: httpx.Request) -> httpx.Response:
            if req.url.path == "/v1/application-tools":
                assert req.url.params["q"] == "engineering"
                return _json_response(
                    [
                        {
                            "tool_id": "tool-1",
                            "namespace_id": "ns-1",
                            "slug": "workspace",
                            "title": "Workspace",
                            "category": "engineering",
                        }
                    ]
                )
            if req.url.path == "/v1/application-tool-releases/rel-1":
                return _json_response(
                    {
                        "release_id": "rel-1",
                        "tool_id": "tool-1",
                        "release_version": "1.0.0",
                        "content_digest": "sha256:content",
                        "qualification_state": "qualified",
                        "artifact_digest": "sha256:artifact",
                        "appsuite_version": "0.1.0",
                        "plugin_names": ["github"],
                        "seed_schema_digest": "sha256:seed",
                        "interface_schema_digest": "sha256:interface",
                        "license_id": "apache-2.0",
                        "manifest": {},
                    }
                )
            return _json_response({}, status=404)

        client = _make_client(httpx.MockTransport(capture))
        tools = client.list_application_tools(query="engineering")
        release = client.get_application_tool_release("rel-1")
        assert isinstance(tools[0], ApplicationTool)
        assert isinstance(release, ApplicationToolRelease)

    def test_create_defaults_public_and_update(self):
        requests = []

        def capture(req: httpx.Request) -> httpx.Response:
            body = json.loads(req.content)
            requests.append((req.method, req.url.path, body))
            return _json_response(
                {
                    "tool_id": "tool-1",
                    "namespace_id": "ns-1",
                    "slug": "workspace",
                    "title": body.get("title", "Workspace"),
                    "summary": body.get("summary", ""),
                    "category": body.get("category", "engineering"),
                    "tags": body.get("tags", []),
                    "visibility": body.get("visibility", "public"),
                    "revision": 2,
                },
                status=201 if req.method == "POST" else 200,
            )

        client = _make_client(httpx.MockTransport(capture))
        created = client.create_application_tool(
            namespace_id="ns-1",
            slug="workspace",
            title="Workspace",
            category="engineering",
        )
        updated = client.update_application_tool(
            "tool-1",
            expected_revision=1,
            visibility="unlisted",
        )
        assert requests[0][2]["visibility"] == "public"
        assert created.visibility == "public"
        assert updated.visibility == "unlisted"


class TestListPublicListings:
    def test_with_query(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response([{"listing_id": "lst-pub", "title": "Public Env"}])

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_public_listings(query="coding", sort_by="popular")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/public/listings"
        assert captured["params"]["query"] == "coding"
        assert captured["params"]["sort_by"] == "popular"
        assert result[0]["title"] == "Public Env"


class TestSearchEnvironments:
    def test_with_filters(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response([{"listing_id": "lst-1", "score": 0.95}])

        client = _make_client(httpx.MockTransport(capture))
        result = client.search_environments(
            query="math", domain="math", min_quality_score=0.8
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-search"
        assert captured["body"]["query"] == "math"
        assert captured["body"]["domain"] == "math"
        assert captured["body"]["min_quality_score"] == 0.8
        assert result[0]["score"] == 0.95


class TestGetEnvironmentRelease:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "release_id": "rel-1",
                    "listing_id": "lst-1",
                    "release_version": "1.0.0",
                    "protocol_version": "0.4.1",
                    "status": "qualified",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_environment_release("rel-1")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-releases/rel-1"
        assert isinstance(result, EnvironmentRelease)
        assert result.release_version == "1.0.0"
        assert result.status == "qualified"


class TestCreateEnvironmentSession:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "session_id": "sess-1",
                    "deployment_id": "dep-1",
                    "task_id": "task-1",
                    "status": "active",
                    "session_token": "tok-abc",
                    "observation": "Write fibonacci...",
                },
                status=202,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_environment_session(
            "dep-1", task_id="task-1", seed=42
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-deployments/dep-1/sessions"
        assert captured["body"]["task_id"] == "task-1"
        assert captured["body"]["seed"] == 42
        assert isinstance(result, EnvironmentSession)
        assert result.session_id == "sess-1"
        assert result.session_token == "tok-abc"
        assert result.observation == "Write fibonacci..."


class TestWaitForEnvironmentSession:
    def test_preserves_create_only_session_token_after_polling(self):
        def capture(req: httpx.Request) -> httpx.Response:
            assert req.method == "GET"
            assert req.url.path == "/v1/environment-sessions/sess-1"
            return _json_response(
                {
                    "session_id": "sess-1",
                    "task_id": "task-1",
                    "status": "active",
                    "observation": "ready",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        initial = EnvironmentSession(
            session_id="sess-1",
            task_id="task-1",
            status="provisioning",
            session_token="secret-token",
            session_token_expires_at="2026-07-19T00:00:00Z",
        )
        result = client.wait_for_session(initial, poll_interval=0, timeout=1)

        assert result.status == "active"
        assert result.session_token == "secret-token"
        assert result.session_token_expires_at == "2026-07-19T00:00:00Z"


class TestGetEnvironmentSession:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "session_id": "sess-1",
                    "deployment_id": "dep-1",
                    "task_id": "task-1",
                    "status": "completed",
                    "reward": 1.0,
                    "steps_taken": 5,
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_environment_session("sess-1")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-sessions/sess-1"
        assert isinstance(result, EnvironmentSession)
        assert result.is_terminal is True
        assert result.reward == 1.0


class TestEnvironmentStep:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "observation": "2/3 tests pass",
                    "reward": 0.67,
                    "terminated": False,
                    "truncated": False,
                    "info": {"tests_passed": 2},
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.environment_step("sess-1", "def fib(n): ...")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-sessions/sess-1/step"
        assert captured["body"]["action"] == "def fib(n): ..."
        assert isinstance(result, EnvironmentStepResult)
        assert result.observation == "2/3 tests pass"
        assert result.reward == 0.67
        assert result.done is False

    def test_structured_action_is_serialized_deterministically(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "observation": "done",
                    "reward": 1.0,
                    "terminated": True,
                    "truncated": False,
                    "info": {},
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        client.environment_step(
            "sess-1",
            {"content": "answer", "action_type": "submit"},
        )

        assert json.loads(captured["body"]["action"]) == {
            "action_type": "submit",
            "content": "answer",
        }

    def test_rejects_unsupported_action_type(self):
        client = _make_client(httpx.MockTransport(lambda req: _json_response({})))
        with pytest.raises(TypeError, match="string or dictionary"):
            client.environment_step("sess-1", ["not", "valid"])  # type: ignore[arg-type]


class TestCancelEnvironmentSession:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"session_id": "sess-1", "status": "cancelled"})

        client = _make_client(httpx.MockTransport(capture))
        result = client.cancel_environment_session("sess-1")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-sessions/sess-1/cancel"
        assert result["status"] == "cancelled"


class TestRefreshSessionToken:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                {
                    "session_id": "sess-1",
                    "session_token": "tok-new",
                    "session_token_expires_at": "2026-06-01T02:00:00",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.refresh_session_token("sess-1")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-sessions/sess-1/token"
        assert result["session_token"] == "tok-new"


class TestRunEnvironmentEpisode:
    def test_full_episode(self):
        call_count = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            path = req.url.path
            if "/sessions" in path and req.method == "POST" and "/step" not in path and "/cancel" not in path:
                return _json_response(
                    {
                        "session_id": "sess-ep",
                        "deployment_id": "dep-1",
                        "task_id": "task-1",
                        "status": "active",
                        "session_token": "tok-ep",
                        "observation": "initial",
                    },
                    status=202,
                )
            elif "/step" in path:
                return _json_response(
                    {
                        "observation": "done",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                    }
                )
            elif req.method == "GET":
                return _json_response(
                    {
                        "session_id": "sess-ep",
                        "deployment_id": "dep-1",
                        "task_id": "task-1",
                        "status": "completed",
                        "reward": 1.0,
                        "steps_taken": 1,
                    }
                )
            return _json_response({}, 404)

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_environment_episode(
            "dep-1",
            task_id="task-1",
            policy_fn=lambda obs, info: "my action",
        )
        assert isinstance(result, EnvironmentSession)
        assert result.status == "completed"
        assert result.reward == 1.0
        assert call_count == 3

    def test_waits_for_provisioning_before_first_policy_action(self):
        session_reads = 0
        observations = []

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal session_reads
            path = req.url.path
            if path == "/v1/environment-deployments/dep-1/sessions" and req.method == "POST":
                return _json_response(
                    {
                        "session_id": "sess-wait",
                        "task_id": "task-1",
                        "status": "provisioning",
                        "session_token": "tok-wait",
                    },
                    status=202,
                )
            if path == "/v1/environment-sessions/sess-wait" and req.method == "GET":
                session_reads += 1
                if session_reads == 1:
                    return _json_response(
                        {
                            "session_id": "sess-wait",
                            "task_id": "task-1",
                            "status": "active",
                            "observation": "ready observation",
                        }
                    )
                return _json_response(
                    {
                        "session_id": "sess-wait",
                        "task_id": "task-1",
                        "status": "completed",
                        "total_reward": 1.0,
                    }
                )
            if path == "/v1/environment-sessions/sess-wait/step":
                assert req.headers["X-RL-Session-Token"] == "tok-wait"
                return _json_response(
                    {
                        "observation": "done",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                    }
                )
            return _json_response({}, 404)

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_environment_episode(
            "dep-1",
            task_id="task-1",
            policy_fn=lambda observation, _info: observations.append(observation) or "answer",
        )

        assert observations == ["ready observation"]
        assert result.status == "completed"

    def test_cancels_session_at_policy_step_limit(self):
        cancelled = False

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal cancelled
            path = req.url.path
            if path == "/v1/environment-deployments/dep-1/sessions":
                return _json_response(
                    {
                        "session_id": "sess-limit",
                        "task_id": "task-1",
                        "status": "active",
                        "session_token": "tok-limit",
                        "observation": "continue",
                    },
                    status=202,
                )
            if path == "/v1/environment-sessions/sess-limit/step":
                return _json_response(
                    {
                        "observation": "still running",
                        "reward": None,
                        "terminated": False,
                        "truncated": False,
                    }
                )
            if path == "/v1/environment-sessions/sess-limit/cancel":
                cancelled = True
                return _json_response({"session_id": "sess-limit", "status": "cancelled"})
            if path == "/v1/environment-sessions/sess-limit":
                return _json_response(
                    {
                        "session_id": "sess-limit",
                        "task_id": "task-1",
                        "status": "cancelled" if cancelled else "active",
                    }
                )
            return _json_response({}, 404)

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_environment_episode(
            "dep-1",
            task_id="task-1",
            policy_fn=lambda _observation, _info: "continue",
            max_steps=1,
        )

        assert cancelled is True
        assert result.status == "cancelled"


class TestLongHorizonAgentRunner:
    def test_records_100_reasoning_turns_before_first_environment_action(self):
        trace_events = []
        step_calls = []
        contexts = []

        def handler(req: httpx.Request) -> httpx.Response:
            path = req.url.path
            if path == "/v1/environment-deployments/dep-long/sessions":
                body = json.loads(req.content)
                assert body["model"] == "glm-5.2"
                assert body["agent_id"] == "prime-intellect"
                return _json_response(
                    {
                        "session_id": "sess-long",
                        "task_id": "task-long",
                        "status": "active",
                        "session_token": "tok-long",
                        "observation": "initial workspace",
                        "steps_taken": 0,
                    },
                    status=202,
                )
            if path == "/v1/environment-sessions/sess-long/trace-events":
                assert req.headers["X-RL-Session-Token"] == "tok-long"
                assert len(req.headers["Idempotency-Key"]) >= 8
                event = json.loads(req.content)
                trace_events.append(event)
                return _json_response({"event_idx": len(trace_events) - 1, **event})
            if path == "/v1/environment-sessions/sess-long/step":
                body = json.loads(req.content)
                step_calls.append(json.loads(body["action"]))
                return _json_response(
                    {
                        "observation": "submitted",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                        "info": {},
                    }
                )
            if path == "/v1/environment-sessions/sess-long" and req.method == "GET":
                return _json_response(
                    {
                        "session_id": "sess-long",
                        "task_id": "task-long",
                        "status": "completed",
                        "total_reward": 1.0,
                        "steps_taken": 1,
                    }
                )
            return _json_response({}, 404)

        def model_fn(context: AgentRunContext) -> AgentTurn:
            contexts.append((context.turn_index, context.environment_steps, context.observation))
            if context.turn_index < 100:
                return AgentTurn(
                    reasoning=f"planning turn {context.turn_index}",
                    provider="prime-intellect",
                    model="glm-5.2",
                    provider_request_id=f"pi-{context.turn_index}",
                    usage=AgentUsage(input_tokens=10, output_tokens=20),
                )
            return AgentTurn(
                message="Ready to submit.",
                tool_calls=[
                    AgentToolCall(
                        call_id="call-submit",
                        name="submit",
                        arguments={},
                    )
                ],
                provider="prime-intellect",
                model="glm-5.2",
                provider_request_id="pi-100",
                usage=AgentUsage(input_tokens=11, output_tokens=21),
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_agent_episode(
            "dep-long",
            task_id="task-long",
            model_fn=model_fn,
            model="glm-5.2",
            agent_id="prime-intellect",
            max_turns=500,
        )

        assert result.stop_reason == "environment_terminal"
        assert result.turns_completed == 101
        assert result.environment_steps == 1
        assert result.input_tokens == 1011
        assert result.output_tokens == 2021
        assert contexts[99] == (99, 0, "initial workspace")
        assert contexts[100] == (100, 0, "initial workspace")
        assert step_calls == [{"input": {}, "tool": "submit"}]
        event_types = [event["event_type"] for event in trace_events]
        assert event_types.count("model_request") == 101
        assert event_types.count("reasoning") == 100
        assert event_types.count("tool_call") == 1
        assert event_types.count("tool_result") == 1
        assert trace_events[-1]["event_type"] == "lifecycle"

    def test_cancellation_between_reasoning_turns_records_trace_and_cleans_up(self):
        trace_types = []
        cancelled = False
        model_calls = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal cancelled
            path = req.url.path
            if path == "/v1/environment-deployments/dep-cancel/sessions":
                return _json_response(
                    {
                        "session_id": "sess-cancel",
                        "task_id": "task-cancel",
                        "status": "active",
                        "session_token": "tok-cancel",
                        "observation": "initial",
                    },
                    status=202,
                )
            if path.endswith("/trace-events"):
                trace_types.append(json.loads(req.content)["event_type"])
                return _json_response({})
            if path.endswith("/cancel"):
                cancelled = True
                return _json_response({"status": "cancelled"})
            if path == "/v1/environment-sessions/sess-cancel":
                return _json_response(
                    {
                        "session_id": "sess-cancel",
                        "task_id": "task-cancel",
                        "status": "cancelled" if cancelled else "active",
                    }
                )
            if path.endswith("/step"):
                pytest.fail("reasoning-only turns must not step the environment")
            return _json_response({}, 404)

        def model_fn(_context: AgentRunContext) -> AgentTurn:
            nonlocal model_calls
            model_calls += 1
            return AgentTurn(reasoning="still thinking")

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_agent_episode(
            "dep-cancel",
            task_id="task-cancel",
            model_fn=model_fn,
            cancel_check=lambda: model_calls >= 2,
        )

        assert model_calls == 2
        assert result.stop_reason == "cancelled"
        assert result.environment_steps == 0
        assert cancelled is True
        assert "cancellation" in trace_types

    def test_turn_limit_is_the_only_sdk_rollout_budget(self):
        client = _make_client(httpx.MockTransport(lambda req: _json_response({})))
        with pytest.raises(ValueError, match="between 1 and 500"):
            client.run_agent_episode(
                "dep",
                task_id="task",
                model_fn=lambda _context: AgentTurn(reasoning="thinking"),
                max_turns=501,
            )


class TestRunBatch:
    def test_drives_provisioned_sessions_and_returns_final_records(self):
        stepped = False
        policy_contexts = []

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal stepped
            path = req.url.path
            if path == "/v1/environment-batches" and req.method == "POST":
                return _json_response({"batch_id": "bat-run", "status": "queued"}, status=202)
            if path == "/v1/environment-batches/bat-run/sessions":
                return _json_response(
                    [
                        {
                            "session_id": "sess-batch",
                            "task_id": "task-batch",
                            "seed": 7,
                            "session_status": "completed" if stepped else "active",
                            "total_reward": 1.0 if stepped else None,
                        }
                    ]
                )
            if path == "/v1/environment-batches/bat-run":
                return _json_response(
                    {
                        "batch_id": "bat-run",
                        "status": "completed" if stepped else "running",
                        "sessions_requested": 1,
                        "sessions_completed": 1 if stepped else 0,
                        "sessions_failed": 0,
                    }
                )
            if path == "/v1/environment-sessions/sess-batch/token":
                return _json_response(
                    {
                        "session_id": "sess-batch",
                        "session_token": "tok-batch",
                        "session_token_expires_at": "2026-07-18T12:00:00Z",
                    }
                )
            if path == "/v1/environment-sessions/sess-batch/step":
                assert req.headers["X-RL-Session-Token"] == "tok-batch"
                stepped = True
                return _json_response(
                    {
                        "observation": "complete",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                    }
                )
            if path == "/v1/environment-sessions/sess-batch":
                return _json_response(
                    {
                        "session_id": "sess-batch",
                        "task_id": "task-batch",
                        "seed": 7,
                        "status": "completed" if stepped else "active",
                        "observation": "batch observation",
                        "total_reward": 1.0 if stepped else None,
                    }
                )
            return _json_response({}, 404)

        client = _make_client(httpx.MockTransport(handler))
        result = client.run_batch(
            deployment_id="dep-1",
            name="SDK batch",
            task_seed_pairs=[{"task_id": "task-batch", "seed": 7}],
            policy_fn=lambda observation, info: policy_contexts.append((observation, info)) or "answer",
            poll_interval=0.05,
        )

        assert result["status"] == "completed"
        assert result["sessions"][0]["total_reward"] == 1.0
        assert policy_contexts[0][0] == "batch observation"
        assert policy_contexts[0][1]["task_id"] == "task-batch"
        assert policy_contexts[0][1]["seed"] == 7


class TestListEntitlements:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                [{"entitlement_id": "ent-1", "listing_id": "lst-1"}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_entitlements()
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-entitlements"
        assert result[0]["entitlement_id"] == "ent-1"


class TestGrantEntitlement:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["body"] = json.loads(req.content)
            return _json_response({"entitlement_id": "ent-new"}, status=201)

        client = _make_client(httpx.MockTransport(capture))
        result = client.grant_entitlement(
            grantee_tenant_id="tenant-buyer",
            listing_id="lst-1",
            license_id="lic-1",
        )
        assert captured["method"] == "POST"
        assert captured["body"]["grantee_tenant_id"] == "tenant-buyer"
        assert result["entitlement_id"] == "ent-new"


class TestCreateEnvironmentExport:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"export_id": "exp-1", "status": "pending"}, status=202
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_environment_export(
            deployment_id="dep-1", format="dpo"
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-exports"
        assert captured["body"]["deployment_id"] == "dep-1"
        assert captured["body"]["format"] == "dpo"
        assert result["export_id"] == "exp-1"


class TestCreateBatch:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"batch_id": "bat-1", "status": "pending"}, status=202
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_batch(
            deployment_id="dep-1",
            name="test batch",
            task_seed_pairs=[{"task_id": "t1", "seed": 42}],
        )
        assert captured["method"] == "POST"
        assert captured["body"]["name"] == "test batch"
        assert len(captured["body"]["task_seed_pairs"]) == 1
        assert result["batch_id"] == "bat-1"


class TestCreateDispute:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"dispute_id": "disp-1", "status": "open"}, status=201
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_dispute(
            session_id="sess-1",
            deployment_id="dep-1",
            release_id="rel-1",
            dispute_type="reward_error",
            summary="Reward was incorrect",
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-disputes"
        assert captured["body"]["dispute_type"] == "reward_error"
        assert result["dispute_id"] == "disp-1"


class TestListQualityReports:
    def test_with_filters(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response(
                [{"report_id": "rpt-1", "status": "completed"}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_quality_reports(
            release_id="rel-1", report_type="full_qualification"
        )
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-quality-reports"
        assert captured["params"]["release_id"] == "rel-1"
        assert captured["params"]["report_type"] == "full_qualification"
        assert result[0]["report_id"] == "rpt-1"


class TestListQualityBadges:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                [{"badge_id": "badge-1", "badge_type": "gold"}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_quality_badges(release_id="rel-1")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-quality-badges"
        assert result[0]["badge_type"] == "gold"


class TestListSessionCharges:
    def test_with_filter(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response(
                [{"charge_id": "chg-1", "amount_cents": 100}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_session_charges(session_id="sess-1")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-session-charges"
        assert captured["params"]["session_id"] == "sess-1"
        assert result[0]["amount_cents"] == 100


class TestListInvoices:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                [{"invoice_id": "inv-1", "total_cents": 5000}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_invoices()
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-invoices"
        assert result[0]["total_cents"] == 5000


class TestCreateReview:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response({"review_id": "rev-1"})

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_review(
            listing_id="lst-1",
            listing_owner_tenant_id="tenant-creator",
            rating=5,
            title="Excellent environment",
            usage_hours=12.5,
            privacy_cleared=True,
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/reviews"
        assert captured["body"]["rating"] == 5
        assert captured["body"]["privacy_cleared"] is True
        assert "usage_hours" not in captured["body"]
        assert result["review_id"] == "rev-1"


class TestCreatePurchase:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response({"purchase_id": "pur-1", "status": "completed"})

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_purchase(
            listing_id="lst-1",
            license_version_id="license-v1",
            payment_reference="legacy-reference",
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/purchases"
        assert captured["body"]["license_version_id"] == "license-v1"
        assert "payment_reference" not in captured["body"]
        assert result["purchase_id"] == "pur-1"


class TestCreateNamespace:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"namespace_id": "ns-1", "slug": "my-org"}, status=201
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_namespace(slug="my-org", display_name="My Org")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-namespaces"
        assert captured["body"]["slug"] == "my-org"
        assert result["namespace_id"] == "ns-1"


class TestCreateListing:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "listing_id": "lst-new",
                    "namespace_id": "ns-1",
                    "slug": "my-env",
                    "title": "My Environment",
                },
                status=201,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_listing(
            namespace_id="ns-1", slug="my-env", title="My Environment"
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-listings"
        assert captured["body"]["visibility"] == "public"
        assert isinstance(result, EnvironmentListing)
        assert result.listing_id == "lst-new"


class TestUpdateListing:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            captured["idempotency_key"] = req.headers.get("Idempotency-Key")
            return _json_response(
                {
                    "listing_id": "lst-1",
                    "namespace_id": "ns-1",
                    "slug": "my-env",
                    "title": "Updated Title",
                    "visibility": "public",
                }
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.update_listing(
            "lst-1",
            expected_revision=2,
            title="Updated Title",
            visibility="public",
            idempotency_key="listing-update-1",
        )
        assert captured["method"] == "PATCH"
        assert captured["path"] == "/v1/environment-listings/lst-1"
        assert captured["body"]["expected_revision"] == 2
        assert captured["idempotency_key"] == "listing-update-1"
        assert isinstance(result, EnvironmentListing)
        assert result.title == "Updated Title"


class TestCreateDeployment:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"deployment_id": "dep-1", "alias": "prod"}, status=201
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_deployment(
            listing_id="lst-1",
            alias="prod",
            environment_release_id="rel-1",
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-deployments"
        assert captured["body"]["alias"] == "prod"
        assert result["deployment_id"] == "dep-1"


class TestCreateEnvironmentRelease:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {
                    "release_id": "rel-new",
                    "listing_id": "lst-1",
                    "release_version": "1.0.0",
                    "protocol_version": "0.4.1",
                    "qualification_state": "qualified",
                },
                status=201,
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_environment_release(
            listing_id="lst-1",
            release_version="1.0.0",
            protocol_version="0.4.1",
            runtime_ref="ghcr.io/my-org/env:1.0.0",
            runtime_digest="sha256:" + "a" * 64,
            task_pack_release_id="tp-1",
            verifier_release_id="ver-1",
            action_schema_digest="sha256:" + "b" * 64,
            observation_schema_digest="sha256:" + "c" * 64,
            application_tools=[
                {
                    "tool_release_id": "tool-release-1",
                    "alias": "github",
                    "configuration_digest": "sha256:" + "d" * 64,
                }
            ],
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/environment-releases"
        assert captured["body"]["application_tools"] == [
            {
                "tool_release_id": "tool-release-1",
                "alias": "github",
                "configuration_digest": "sha256:" + "d" * 64,
            }
        ]
        assert isinstance(result, EnvironmentRelease)
        assert result.release_id == "rel-new"
        assert result.status == "qualified"


class TestGetCreatorAggregates:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response(
                [{"release_id": "rel-1", "total_sessions": 150, "pass_rate": 0.82}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_creator_aggregates()
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/environment-creator-aggregates"
        assert result[0]["total_sessions"] == 150


class TestCreatorProfile:
    def test_create(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response({"profile_id": "prof-1", "display_name": "AI Labs"})

        client = _make_client(httpx.MockTransport(capture))
        result = client.create_creator_profile(
            display_name="AI Labs", bio="We build RL environments"
        )
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/creator-profiles"
        assert captured["body"]["display_name"] == "AI Labs"
        assert result["display_name"] == "AI Labs"

    def test_get(self):
        client = _make_client(
            httpx.MockTransport(
                lambda req: _json_response(
                    {"profile_id": "prof-1", "display_name": "AI Labs"}
                )
            )
        )
        result = client.get_creator_profile()
        assert result["display_name"] == "AI Labs"

    def test_update(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["body"] = json.loads(req.content)
            return _json_response({"profile_id": "prof-1", "is_public": True})

        client = _make_client(httpx.MockTransport(capture))
        client.update_creator_profile(is_public=True)
        assert captured["method"] == "PATCH"
        assert captured["body"]["is_public"] is True


class TestCreatorSettlement:
    def test_get_account(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            return _json_response({"balance_cents": 15000, "currency": "usd"})

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_creator_account()
        assert captured["path"] == "/v1/creator-account"
        assert result["balance_cents"] == 15000

    def test_list_royalty_rules(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            return _json_response([{"rule_id": "rule-1", "type": "per_session"}])

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_royalty_rules()
        assert captured["path"] == "/v1/creator-royalty-rules"
        assert result[0]["type"] == "per_session"

    def test_list_accruals(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            return _json_response([{"accrual_id": "acc-1", "amount_cents": 500}])

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_accruals(status="pending")
        assert captured["path"] == "/v1/creator-accruals"
        assert result[0]["amount_cents"] == 500


class TestListAdapters:
    def test_with_filter(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response(
                [{"adapter_id": "adp-1", "protocol_family": "gymnasium"}]
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.list_adapters(protocol_family="gymnasium")
        assert captured["method"] == "GET"
        assert captured["path"] == "/v1/adapters"
        assert captured["params"]["protocol_family"] == "gymnasium"
        assert result[0]["adapter_id"] == "adp-1"


class TestGetAdapter:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            return _json_response(
                {"adapter_id": "adp-1", "name": "Gymnasium Adapter"}
            )

        client = _make_client(httpx.MockTransport(capture))
        result = client.get_adapter("adp-1")
        assert captured["path"] == "/v1/adapters/adp-1"
        assert result["name"] == "Gymnasium Adapter"


class TestCheckAdapterEquivalence:
    def test_basic(self):
        captured = {}

        def capture(req: httpx.Request) -> httpx.Response:
            captured["path"] = req.url.path
            captured["params"] = dict(req.url.params)
            return _json_response({"equivalent": True, "diff_count": 0})

        client = _make_client(httpx.MockTransport(capture))
        result = client.check_adapter_equivalence("adp-1", "ver-1")
        assert captured["path"] == "/v1/adapters/adp-1/equivalence/check"
        assert captured["params"]["version_id"] == "ver-1"
        assert result["equivalent"] is True


class TestRevokeEntitlement:
    def test_path_and_method(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"entitlement_id": "ent-1", "status": "revoked"})

        client = _make_client(httpx.MockTransport(handler))
        result = client.revoke_entitlement("ent-1")
        assert captured["method"] == "POST"
        assert "/ent-1/revoke" in captured["path"]
        assert result["status"] == "revoked"


class TestGetEnvironmentExport:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response({"export_id": "exp-1", "status": "completed"})

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_environment_export("exp-1")
        assert "/v1/environment-exports/exp-1" in captured["path"]
        assert result["export_id"] == "exp-1"


class TestGetBatch:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                {"batch_id": "bat-1", "status": "running", "progress": 50}
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_batch("bat-1")
        assert "/v1/environment-batches/bat-1" in captured["path"]
        assert result["progress"] == 50


class TestGetBatchSessions:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response([{"session_id": "s1"}, {"session_id": "s2"}])

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_batch_sessions("bat-1")
        assert "/bat-1/sessions" in captured["path"]
        assert len(result) == 2


class TestCancelBatch:
    def test_path_and_method(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"batch_id": "bat-1", "status": "cancelled"})

        client = _make_client(httpx.MockTransport(handler))
        result = client.cancel_batch("bat-1")
        assert captured["method"] == "POST"
        assert "/bat-1/cancel" in captured["path"]
        assert result["status"] == "cancelled"


class TestGetBatchComparison:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response({"comparison": {"mean_reward": 0.72}})

        client = _make_client(httpx.MockTransport(handler))
        client.get_batch_comparison("bat-1")
        assert "/bat-1/comparison" in captured["path"]


class TestGetDispute:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response({"dispute_id": "dis-1", "status": "open"})

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_dispute("dis-1")
        assert "/v1/environment-disputes/dis-1" in captured["path"]
        assert result["status"] == "open"


class TestGetSessionAudit:
    def test_path_and_params(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"event": "step", "ts": "2026-01-01T00:00:00"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_session_audit("sess-1", event_type="step", limit=10)
        assert "/sess-1/audit" in captured["path"]
        assert "event_type=step" in captured["query"]
        assert len(result) == 1


class TestGetQualityReport:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                {"report_id": "rpt-1", "status": "completed"}
            )

        client = _make_client(httpx.MockTransport(handler))
        client.get_quality_report("rpt-1")
        assert "/v1/environment-quality-reports/rpt-1" in captured["path"]


class TestGetQualityChecks:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                [{"check_id": "chk-1", "passed": True}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_quality_checks("rpt-1")
        assert "/rpt-1/checks" in captured["path"]
        assert result[0]["passed"] is True


class TestListContaminationFindings:
    def test_path_and_filters(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"finding_id": "f1", "finding_type": "data_leak"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.list_contamination_findings(
            release_id="rel-1", finding_type="data_leak"
        )
        assert "release_id=rel-1" in captured["query"]
        assert "finding_type=data_leak" in captured["query"]
        assert len(result) == 1


class TestListBenchmarkResults:
    def test_path_and_filters(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"model_tag": "gpt-4", "score": 0.85}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.list_benchmark_results(
            release_id="rel-1", model_tag="gpt-4"
        )
        assert "model_tag=gpt-4" in captured["query"]
        assert result[0]["score"] == 0.85


class TestListLicenseVersions:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"license_version_id": "lv-1", "license_id": "apache-2.0"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.list_license_versions("rel-1")
        assert "release_id=rel-1" in captured["query"]
        assert result[0]["license_id"] == "apache-2.0"


class TestGetLicenseVersion:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                {"license_version_id": "lv-1", "text": "Apache License..."}
            )

        client = _make_client(httpx.MockTransport(handler))
        client.get_license_version("lv-1")
        assert "/lv-1" in captured["path"]


class TestListChargeAdjustments:
    def test_path_and_filter(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"adjustment_id": "adj-1", "type": "refund"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_charge_adjustments(charge_id="chg-1")
        assert "charge_id=chg-1" in captured["query"]


class TestGetInvoice:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                {"invoice_id": "inv-1", "total_cents": 5000}
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_invoice("inv-1")
        assert "/inv-1" in captured["path"]
        assert result["total_cents"] == 5000


class TestGetInvoiceLineItems:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                [{"line_item_id": "li-1", "amount_cents": 1500}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.get_invoice_line_items("inv-1")
        assert "/inv-1/line-items" in captured["path"]


class TestGetChargeSummary:
    def test_with_date_params(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                {"total_cents": 10000, "session_count": 42}
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_charge_summary(
            since="2026-01-01", until="2026-06-30"
        )
        assert "since=2026-01-01" in captured["query"]
        assert "until=2026-06-30" in captured["query"]
        assert result["session_count"] == 42


class TestListNotifications:
    def test_unread_filter(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"notification_id": "n1", "type": "new_review", "read": False}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.list_notifications(unread_only=True)
        assert "unread_only=true" in captured["query"]
        assert len(result) == 1


class TestMarkNotificationRead:
    def test_path_and_method(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            return _json_response({"notification_id": "n1", "read": True})

        client = _make_client(httpx.MockTransport(handler))
        client.mark_notification_read("n1")
        assert captured["method"] == "POST"
        assert "/n1/read" in captured["path"]


class TestCreateDeploymentRevision:
    def test_path_and_body(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"deployment_id": "dep-1", "alias": "production"}, status=201
            )

        client = _make_client(httpx.MockTransport(handler))
        client.create_deployment_revision(
            "dep-1",
            environment_release_id="rel-2",
            export_policy="full",
        )
        assert captured["method"] == "POST"
        assert "/dep-1/revisions" in captured["path"]
        assert captured["body"]["environment_release_id"] == "rel-2"
        assert captured["body"]["export_policy"] == "full"


class TestCreateQualityReport:
    def test_path_and_body(self):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content)
            return _json_response(
                {"report_id": "rpt-1", "status": "pending"}, status=202
            )

        client = _make_client(httpx.MockTransport(handler))
        client.create_quality_report(
            release_id="rel-1",
            report_type="full_qualification",
            deployment_id="dep-1",
        )
        assert captured["body"]["release_id"] == "rel-1"
        assert captured["body"]["report_type"] == "full_qualification"
        assert captured["body"]["deployment_id"] == "dep-1"

    def test_rejects_unsupported_report_type_before_request(self):
        client = _make_client(httpx.MockTransport(lambda request: pytest.fail("unexpected request")))
        with pytest.raises(ValueError, match="report_type must be one of"):
            client.create_quality_report(
                release_id="rel-1",
                report_type="qualification",
            )


class TestCreateTaskPackRelease:
    def test_path_and_body(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response({"release_id": "tp-1"}, status=201)

        client = _make_client(httpx.MockTransport(handler))
        result = client.create_task_pack_release(
            namespace_id="ns-1",
            name="my-tasks",
            release_version="1.0.0",
            artifact_ref="ghcr.io/tasks:1.0",
            artifact_digest="sha256:aaa",
            usage_policy="open",
            license_id="apache-2.0",
            members=[{"task_id": "t1"}],
        )
        assert captured["method"] == "POST"
        assert "/v1/task-pack-releases" in captured["path"]
        assert captured["body"]["name"] == "my-tasks"
        assert captured["body"]["members"] == [{"task_id": "t1"}]
        assert result["release_id"] == "tp-1"


class TestCreateVerifierRelease:
    def test_path_and_body(self):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content)
            return _json_response({"release_id": "ver-1"}, status=201)

        client = _make_client(httpx.MockTransport(handler))
        result = client.create_verifier_release(
            namespace_id="ns-1",
            name="my-verifier",
            release_version="1.0.0",
            runtime_ref="ghcr.io/ver:1.0",
            runtime_digest="sha256:bbb",
            source_digest="sha256:src",
            evidence_schema_digest="sha256:evi",
            reward_mode="continuous",
            reward_min=0.0,
            reward_max=1.0,
            timeout_seconds=30,
            nondeterministic=True,
        )
        assert captured["body"]["reward_mode"] == "continuous"
        assert captured["body"]["reward_min"] == 0.0
        assert captured["body"]["reward_max"] == 1.0
        assert captured["body"]["timeout_seconds"] == 30
        assert captured["body"]["nondeterministic"] is True
        assert result["release_id"] == "ver-1"


class TestUpdateCreatorProfile:
    def test_patch_method_and_body(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["body"] = json.loads(req.content)
            return _json_response({"display_name": "Updated Name", "is_public": True})

        client = _make_client(httpx.MockTransport(handler))
        result = client.update_creator_profile(
            display_name="Updated Name",
            bio="New bio",
            is_public=True,
        )
        assert captured["method"] == "PATCH"
        assert captured["body"]["display_name"] == "Updated Name"
        assert captured["body"]["is_public"] is True
        assert result["display_name"] == "Updated Name"


class TestCreateChangelog:
    def test_body_and_optional_fields(self):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content)
            return _json_response({"changelog_id": "cl-1"}, status=201)

        client = _make_client(httpx.MockTransport(handler))
        client.create_changelog(
            release_id="rel-1",
            version_label="1.2.0",
            summary="Added new tasks",
            body="Full details here",
            breaking_changes=True,
            notify_buyers=True,
            listing_id="lst-1",
        )
        assert captured["body"]["release_id"] == "rel-1"
        assert captured["body"]["version_label"] == "1.2.0"
        assert captured["body"]["breaking_changes"] is True
        assert captured["body"]["notify_buyers"] is True
        assert captured["body"]["listing_id"] == "lst-1"


class TestListChangelogs:
    def test_path_and_params(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"changelog_id": "cl-1", "summary": "v1.2.0"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_changelogs("rel-1", limit=10)
        assert "/rel-1" in captured["path"]
        assert "limit=10" in captured["query"]


class TestGetCreatorAccount:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                {"balance_cents": 15000, "status": "active"}
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_creator_account()
        assert "/v1/creator-account" in captured["path"]
        assert result["balance_cents"] == 15000


class TestListRoyaltyRules:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                [{"rule_id": "rr-1", "rate_bps": 2000}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_royalty_rules()
        assert "/v1/creator-royalty-rules" in captured["path"]


class TestListSettlementAdjustments:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response([{"adjustment_id": "adj-1"}])

        client = _make_client(httpx.MockTransport(handler))
        client.list_settlement_adjustments()
        assert "/v1/creator-adjustments" in captured["path"]


class TestListPayoutBatches:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response([{"payout_id": "po-1", "status": "completed"}])

        client = _make_client(httpx.MockTransport(handler))
        client.list_payout_batches()
        assert "/v1/creator-payout-batches" in captured["path"]


class TestListCreatorStatements:
    def test_path(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            return _json_response(
                [{"statement_id": "stmt-1", "period": "2026-06"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_creator_statements()
        assert "/v1/creator-statements" in captured["path"]


class TestListAdapterVersions:
    def test_path_and_filter(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"version_id": "v1", "status": "active"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_adapter_versions("adp-1", status="active")
        assert "/adp-1/versions" in captured["path"]
        assert "status=active" in captured["query"]


class TestGetAdapterConformance:
    def test_path_and_params(self):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"test_id": "ct-1", "passed": True}]
            )

        client = _make_client(httpx.MockTransport(handler))
        result = client.get_adapter_conformance(
            "adp-1", version_id="v1", status="passed"
        )
        assert "/adp-1/conformance" in captured["path"]
        assert "version_id=v1" in captured["query"]
        assert result[0]["passed"] is True


class TestReportAdapterUsage:
    def test_body(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["body"] = json.loads(req.content)
            return _json_response({"ok": True})

        client = _make_client(httpx.MockTransport(handler))
        client.report_adapter_usage(
            "adp-1",
            version_id="v1",
            event_type="session_start",
            metadata={"env": "test"},
        )
        assert captured["method"] == "POST"
        assert captured["body"]["version_id"] == "v1"
        assert captured["body"]["event_type"] == "session_start"
        assert captured["body"]["metadata"] == {"env": "test"}


class TestListBatches:
    def test_filters(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"batch_id": "bat-1"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_batches(deployment_id="dep-1", status="completed")
        assert "deployment_id=dep-1" in captured["query"]
        assert "status=completed" in captured["query"]


class TestListDisputesFilters:
    def test_filters(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response([{"dispute_id": "dis-1"}])

        client = _make_client(httpx.MockTransport(handler))
        client.list_disputes(release_id="rel-1", status="open")
        assert "release_id=rel-1" in captured["query"]
        assert "status=open" in captured["query"]


class TestRequestPublish:
    def test_path_and_body(self):
        captured = {}

        def handler(req):
            captured["method"] = req.method
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content)
            return _json_response({"request_id": "pr-1", "status": "pending"})

        client = _make_client(httpx.MockTransport(handler))
        client.request_publish("lst-1")
        assert captured["method"] == "POST"
        assert "/moderation/publish-request" in captured["path"]
        assert captured["body"]["listing_id"] == "lst-1"


class TestListAccruals:
    def test_with_status_filter(self):
        captured = {}

        def handler(req):
            captured["query"] = str(req.url.query)
            return _json_response(
                [{"accrual_id": "acc-1", "status": "pending"}]
            )

        client = _make_client(httpx.MockTransport(handler))
        client.list_accruals(status="pending")
        assert "status=pending" in captured["query"]


class TestAllMarketplacePathsUseV1OrKnownPrefix:
    """Smoke test: all marketplace endpoint paths must use /v1/ or a known non-versioned prefix."""

    KNOWN_PREFIXES = (
        "/v1/",
        "/public/",
        "/reviews",
        "/purchases",
        "/notifications",
        "/creator-profiles",
        "/moderation/",
        "/changelogs",
        "/vulnerabilities",
        "/adapters",
    )

    def test_all_paths_are_valid(self):
        captured_paths = []

        def capture(req: httpx.Request) -> httpx.Response:
            captured_paths.append(req.url.path)
            return _json_response(
                {
                    "listing_id": "x",
                    "namespace_id": "x",
                    "slug": "x",
                    "title": "x",
                    "release_id": "x",
                    "release_version": "x",
                    "protocol_version": "x",
                    "session_id": "x",
                    "deployment_id": "x",
                    "task_id": "x",
                    "status": "active",
                    "observation": "",
                    "reward": None,
                    "terminated": False,
                    "truncated": False,
                }
            )

        client = _make_client(httpx.MockTransport(capture))

        client.list_environment_listings()
        client.list_public_listings()
        client.search_environments()
        client.get_environment_release("r1")
        client.create_environment_session("d1", task_id="t1")
        client.get_environment_session("s1")
        client.environment_step("s1", "act")
        client.cancel_environment_session("s1")
        client.refresh_session_token("s1")
        client.list_entitlements()
        client.grant_entitlement(
            grantee_tenant_id="t", listing_id="l", license_id="lic"
        )
        client.create_environment_export(deployment_id="d", format="dpo")
        client.list_environment_exports()
        client.create_batch(
            deployment_id="d",
            name="b",
            task_seed_pairs=[{"task_id": "t1"}],
        )
        client.list_batches()
        client.create_dispute(
            session_id="s",
            deployment_id="d",
            release_id="r",
            dispute_type="t",
            summary="s",
        )
        client.list_disputes()
        client.list_quality_reports()
        client.list_quality_badges()
        client.list_session_charges()
        client.list_invoices()
        client.create_review(
            listing_id="l",
            listing_owner_tenant_id="t",
            rating=5,
            title="great",
        )
        client.list_reviews("l1")
        client.create_purchase(
            listing_id="l",
            license_version_id="license-v1",
        )
        client.list_purchases()
        client.create_namespace(slug="ns", display_name="NS")
        client.create_listing(namespace_id="ns", slug="s", title="T")
        client.create_deployment(
            listing_id="l", alias="a", environment_release_id="r"
        )
        client.get_creator_aggregates()
        client.create_creator_profile(display_name="N")
        client.get_creator_profile()
        client.request_publish("l1")
        client.get_creator_account()
        client.list_royalty_rules()
        client.list_accruals()
        client.list_adapters()
        client.get_adapter("a1")

        for path in captured_paths:
            assert any(
                path.startswith(p) for p in self.KNOWN_PREFIXES
            ), f"Unexpected path prefix: {path}"
