"""Epsilab Python SDK — main client module.

Provides :class:`EpsilabClient`, the primary interface for submitting
evaluations, retrieving results, managing tasks, and exporting
training data.  Importable as ``from epsilab import Epsilab``.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import quote

import httpx
from dotenv import dotenv_values

from .exceptions import ApiError, AuthError, InsufficientCreditsError, RateLimitError
from .models import (
    ArtifactSummary,
    CostEstimate,
    CustomTaskUploadResult,
    EvaluationResult,
    GapSummary,
    RLSession,
    RLStepResult,
    RLTrajectory,
    RunSummary,
    UsageRecord,
)

logger = logging.getLogger("epsilab")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0
_DEFAULT_BACKOFF_MAX = 60.0


class EpsilabClient:
    """Client for the Epsilab evaluation API.

    Args:
        api_base: Base URL for the API. Defaults to ``EPSILAB_API_BASE``
            env var or the production URL.
        api_key: API key for authentication. Defaults to ``EPSILAB_API_KEY``
            env var.
        timeout_seconds: HTTP timeout. Defaults to ``EPSILAB_HTTP_TIMEOUT``
            env var or 120 seconds.
        max_retries: Number of automatic retries for rate-limit (429) and
            transient server errors (5xx). Set to ``0`` to disable.
            Defaults to 3.
        backoff_base: Initial backoff delay in seconds for retries.
            Each subsequent retry doubles the delay (with jitter).
            For 429 responses, ``Retry-After`` header takes precedence.
            Defaults to 1.0.
        load_dotenv: If true, also read configuration from a local ``.env``
            file. Disabled by default so importing the SDK in an untrusted
            working directory cannot redirect API requests.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        load_dotenv: bool = False,
    ) -> None:
        env: Dict[str, Optional[str]] = {}
        if load_dotenv:
            try:
                env = dotenv_values(".env")
            except Exception:
                pass

        def config_value(name: str) -> Optional[str]:
            return os.getenv(name) or env.get(name)

        self.api_base = (
            api_base or config_value("EPSILAB_API_BASE") or "https://api.epsilab.com"
        ).rstrip("/")

        env_timeout = config_value("EPSILAB_HTTP_TIMEOUT")
        self.timeout_seconds = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else int(env_timeout)
            if env_timeout
            else 120
        )

        self._api_key: Optional[str] = api_key or config_value("EPSILAB_API_KEY")
        self._max_retries = max_retries
        self._backoff_base = backoff_base

        self._client = httpx.Client(
            base_url=self.api_base,
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
        )
        if self._api_key:
            self._client.headers["Authorization"] = f"Bearer {self._api_key}"

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EpsilabClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key
        self._client.headers["Authorization"] = f"Bearer {api_key}"

    # ── HTTP (with retry) ────────────────────────────────────────────

    @staticmethod
    def _path_segment(value: str) -> str:
        return quote(value, safe="")

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())

    @staticmethod
    def _retry_after_for_error(value: Optional[str]) -> Optional[int]:
        parsed = EpsilabClient._parse_retry_after(value)
        return math.ceil(parsed) if parsed is not None else None

    def _sleep_for_retry(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None:
            delay = float(retry_after)
        else:
            delay = self._backoff_base * (2**attempt)
            delay = min(delay, _DEFAULT_BACKOFF_MAX)
            delay *= 0.5 + random.random()  # jitter: 50–150% of base
        logger.debug("Retry %d: sleeping %.1fs", attempt + 1, delay)
        time.sleep(delay)

    def _buffered_request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute a buffered HTTP request with automatic retry.

        Retries on transport errors and retryable HTTP status codes
        (429, 500, 502, 503, 504). Returns the final response.
        """
        cleaned_params = (
            {k: str(v) for k, v in params.items() if v is not None} if params else None
        )

        last_resp: Optional[httpx.Response] = None
        last_error: Optional[httpx.TransportError] = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.request(
                    method,
                    path,
                    params=cleaned_params,
                    json=json_body,
                )
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep_for_retry(attempt, None)
                continue
            last_resp = resp

            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                break
            if attempt >= self._max_retries:
                break

            retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
            self._sleep_for_retry(attempt, retry_after)

        if last_resp is not None:
            return last_resp
        raise ApiError(0, str(last_error) if last_error else "Request failed")

    def _check_response_errors(self, resp: httpx.Response) -> None:
        """Raise appropriate exceptions for error HTTP status codes."""
        if resp.status_code in (401, 403):
            raise AuthError(resp.text)
        if resp.status_code == 402:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise InsufficientCreditsError(detail)
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After")
            raise RateLimitError(resp.text, self._retry_after_for_error(retry))
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        resp = self._buffered_request_with_retry(
            method,
            path,
            params=params,
            json_body=json_body,
        )
        self._check_response_errors(resp)
        if resp.status_code == 204:
            return None
        return resp.json()

    def _request_raw(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Like _request but returns raw response text (for NDJSON/exports)."""
        resp = self._buffered_request_with_retry(method, path, params=params)
        self._check_response_errors(resp)
        return resp.text

    def _stream_raw_to_path(
        self,
        method: str,
        path: str,
        output_path: Path,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Like _request_raw, but streams a successful response to a file."""
        cleaned_params = (
            {k: str(v) for k, v in params.items() if v is not None} if params else None
        )

        last_error: Optional[httpx.TransportError] = None
        for attempt in range(self._max_retries + 1):
            try:
                with self._client.stream(
                    method,
                    path,
                    params=cleaned_params,
                ) as resp:
                    if (
                        resp.status_code in _RETRYABLE_STATUS_CODES
                        and attempt < self._max_retries
                    ):
                        retry_after = self._parse_retry_after(
                            resp.headers.get("Retry-After")
                        )
                        resp.read()
                        self._sleep_for_retry(attempt, retry_after)
                        continue

                    if resp.status_code in (401, 403):
                        raise AuthError(resp.read().decode("utf-8", errors="replace"))
                    if resp.status_code == 429:
                        text = resp.read().decode("utf-8", errors="replace")
                        raise RateLimitError(
                            text,
                            self._retry_after_for_error(
                                resp.headers.get("Retry-After")
                            ),
                        )
                    if resp.status_code >= 400:
                        text = resp.read().decode("utf-8", errors="replace")
                        raise ApiError(resp.status_code, text)

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
                    try:
                        with tmp.open("wb") as f:
                            for chunk in resp.iter_bytes():
                                f.write(chunk)
                        tmp.rename(output_path)
                    except BaseException:
                        tmp.unlink(missing_ok=True)
                        raise
                    return
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep_for_retry(attempt, None)
                continue

        raise ApiError(0, str(last_error) if last_error else "Request failed")

    # ── models ───────────────────────────────────────────────────────

    def list_models(
        self,
        *,
        search: Optional[str] = None,
        provider: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Browse available models with live pricing.

        Args:
            search: Free-text search filter (e.g. ``"gpt-4"``).
            provider: Filter by provider (e.g. ``"openai"``,
                ``"anthropic"``, ``"google"``).
            limit: Max results to return.

        Returns:
            Dict with ``models`` list (each has ``model_id``,
            ``provider``, ``name``, pricing info, etc.).
        """
        return self._request(
            "GET",
            "/v1/models",
            params={"search": search, "provider": provider, "limit": limit},
        )

    # ── runs ─────────────────────────────────────────────────────────

    def create_run(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tasks: Optional[int] = None,
        domains: Optional[List[str]] = None,
        force: bool = False,
    ) -> RunSummary:
        """Submit a single model for evaluation.

        For multi-model evaluations, use :meth:`create_evaluation` instead.
        """
        body: Dict[str, Any] = {"target_model": model_name}
        if base_url:
            target_config: Dict[str, str] = {"base_url": base_url}
            if api_key:
                target_config["api_key"] = api_key
            body["target_config"] = target_config
        if max_tasks is not None:
            body["max_tasks"] = max_tasks
        if domains:
            body["domains"] = domains
        if force:
            body["force"] = True
        data = self._request("POST", "/v1/runs", json_body=body)
        return RunSummary.from_dict(data)

    def get_run(self, run_id: str) -> RunSummary:
        """Get the current status of a run."""
        data = self._request("GET", f"/v1/runs/{self._path_segment(run_id)}")
        return RunSummary.from_dict(data)

    def list_runs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[RunSummary]:
        """List your evaluation runs (single page).

        For automatic pagination, use :meth:`iter_runs`.
        """
        data = self._request(
            "GET",
            "/v1/runs",
            params={"status": status, "limit": limit, "offset": offset},
        )
        return [RunSummary.from_dict(r) for r in data.get("runs", [])]

    def iter_runs(
        self,
        *,
        status: Optional[str] = None,
        page_size: int = 100,
    ) -> Iterator[RunSummary]:
        """Iterate over all your evaluation runs, auto-paginating.

        Yields :class:`~epsilab.models.RunSummary` objects one at a time.

        Args:
            status: Filter by status (``queued``, ``running``,
                ``completed``, ``failed``).
            page_size: Number of runs to fetch per API call.
        """
        offset = 0
        while True:
            page = self.list_runs(status=status, limit=page_size, offset=offset)
            if not page:
                break
            yield from page
            if len(page) < page_size:
                break
            offset += len(page)

    def cancel_run(self, run_id: str) -> RunSummary:
        """Cancel a queued or running evaluation.

        A proportional credit refund is issued for unfinished tasks.
        """
        data = self._request("POST", f"/v1/runs/{self._path_segment(run_id)}/cancel")
        return RunSummary.from_dict(data)

    def retry_run(self, run_id: str) -> RunSummary:
        """Retry a failed run, reusing previously completed results."""
        data = self._request("POST", f"/v1/runs/{self._path_segment(run_id)}/retry")
        return RunSummary.from_dict(data)

    def resume_run(
        self,
        run_id: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> RunSummary:
        """Resume a failed or cancelled run.

        Optionally provide new model credentials if the originals
        caused the failure.
        """
        body: Dict[str, Any] = {}
        if base_url or api_key:
            tc: Dict[str, str] = {}
            if base_url:
                tc["base_url"] = base_url
            if api_key:
                tc["api_key"] = api_key
            body["target_config"] = tc
        data = self._request(
            "POST",
            f"/v1/runs/{self._path_segment(run_id)}/resume",
            json_body=body or None,
        )
        return RunSummary.from_dict(data)

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its associated results.

        Data may be recoverable for a limited time — contact support
        if you need to restore a deleted run.
        """
        self._request("DELETE", f"/v1/runs/{self._path_segment(run_id)}")

    def wait_for_completion(
        self,
        run_id: str,
        *,
        poll_interval: int = 10,
        timeout: int = 3600,
    ) -> RunSummary:
        """Block until a run completes or fails.

        Raises:
            TimeoutError: If the run does not complete within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            run = self.get_run(run_id)
            if run.status in ("completed", "failed"):
                return run
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")
            time.sleep(poll_interval)

    # ── multi-model evaluations ──────────────────────────────────────

    def create_evaluation(
        self,
        models: List[Union[str, Dict[str, Any]]],
        *,
        name: Optional[str] = None,
        task_source: str = "db",
        domains: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        instructions: Optional[str] = None,
        human_verified_only: bool = False,
        default_harness: Optional[str] = None,
    ) -> EvaluationResult:
        """Create a multi-model evaluation.

        Evaluates multiple models on the same task set in a single run.
        The first model becomes the target; others become the reference
        panel for side-by-side comparison.

        Args:
            models: Model IDs to evaluate. Each entry can be a plain
                string (``"openai/gpt-4o"``) or a dict with optional
                fields (``{"model_id": "...", "harness": "codex"}``).
            name: Display name for the evaluation.
            task_source: Where to source tasks from. ``"db"`` (default)
                uses the standard task library. ``"custom"`` uses only
                your uploaded tasks. ``"platform"`` uses the curated
                benchmark set.
            domains: Filter tasks by domain (``coding``, ``math``, etc.).
            max_tasks: Cap the number of tasks evaluated.
            instructions: Custom evaluation instructions.
            human_verified_only: Only use human-verified tasks.
            default_harness: Default agent harness for models that don't
                specify one (``openhands``, ``codex``, ``hermes``,
                ``swe-agent``).

        Returns:
            An :class:`~epsilab.models.EvaluationResult` with the
            evaluation ID and run details.

        Raises:
            InsufficientCreditsError: If the account cannot afford the
                evaluation.
        """
        model_entries = []
        for m in models:
            if isinstance(m, str):
                model_entries.append({"model_id": m})
            else:
                model_entries.append(m)

        body: Dict[str, Any] = {"models": model_entries}
        if name is not None:
            body["name"] = name
        if task_source != "db":
            body["task_source"] = task_source
        if domains:
            body["domains"] = domains
        if max_tasks is not None:
            body["max_tasks"] = max_tasks
        if instructions:
            body["instructions"] = instructions
        if human_verified_only:
            body["human_verified_only"] = True
        if default_harness:
            body["default_harness"] = default_harness

        data = self._request("POST", "/v1/evaluations", json_body=body)
        return EvaluationResult.from_dict(data)

    def estimate_evaluation_cost(
        self,
        models: List[Union[str, Dict[str, Any]]],
        *,
        task_source: str = "db",
        domains: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        human_verified_only: bool = False,
        default_harness: Optional[str] = None,
    ) -> CostEstimate:
        """Estimate the credit cost of an evaluation before running it.

        Args:
            models: Same format as :meth:`create_evaluation`.
            task_source: Task source (see :meth:`create_evaluation`).
            domains: Filter tasks by domain.
            max_tasks: Cap the number of tasks.
            human_verified_only: Only use human-verified tasks.
            default_harness: Default agent harness.

        Returns:
            A :class:`~epsilab.models.CostEstimate` with per-model
            breakdowns, total credits, and whether the balance is
            sufficient.
        """
        model_entries = []
        for m in models:
            if isinstance(m, str):
                model_entries.append({"model_id": m})
            else:
                model_entries.append(m)

        body: Dict[str, Any] = {"models": model_entries}
        if task_source != "db":
            body["task_source"] = task_source
        if domains:
            body["domains"] = domains
        if max_tasks is not None:
            body["max_tasks"] = max_tasks
        if human_verified_only:
            body["human_verified_only"] = True
        if default_harness:
            body["default_harness"] = default_harness

        data = self._request("POST", "/v1/evaluations/estimate", json_body=body)
        return CostEstimate.from_dict(data)

    def suggest_scope(self, instructions: str) -> Dict[str, Any]:
        """Get AI-generated evaluation scope suggestions.

        Given a natural-language description of what you want to evaluate,
        returns suggested domains, task filters, and scope parameters.

        Args:
            instructions: Plain-text description of evaluation goals
                (e.g. ``"test my model on coding and math"``).

        Returns:
            Dict with suggested ``domains``, ``task_scope``, and other
            parameters for use with :meth:`create_evaluation`.
        """
        return self._request(
            "POST",
            "/v1/evaluations/suggest-scope",
            json_body={"instructions": instructions},
        )

    # ── gaps, insights & artifacts ───────────────────────────────────

    def get_gaps(self, run_id: str) -> List[GapSummary]:
        """Get capability gaps found in a completed run."""
        data = self._request("GET", f"/v1/runs/{self._path_segment(run_id)}/gaps")
        return [GapSummary.from_dict(g) for g in data.get("gaps", [])]

    def get_artifacts(
        self,
        run_id: str,
        *,
        artifact_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ArtifactSummary]:
        """Get generated artifacts for a run (single page).

        For automatic pagination, use :meth:`iter_artifacts`.
        """
        data = self._request(
            "GET",
            f"/v1/runs/{self._path_segment(run_id)}/artifacts",
            params={"artifact_type": artifact_type, "limit": limit, "offset": offset},
        )
        return [ArtifactSummary.from_dict(a) for a in data.get("artifacts", [])]

    def iter_artifacts(
        self,
        run_id: str,
        *,
        artifact_type: Optional[str] = None,
        page_size: int = 100,
    ) -> Iterator[ArtifactSummary]:
        """Iterate over all artifacts for a run, auto-paginating.

        Args:
            run_id: The run to fetch artifacts for.
            artifact_type: Filter by type. Common values:

                - ``preference_pair`` — DPO/RLHF training pairs
                - ``gold_answer`` — verified correct outputs (SFT)
                - ``trajectory`` — full agent execution traces
                - ``refined_trajectory`` — compressed, verified traces
                - ``test_case`` — executable test cases

            page_size: Number of artifacts to fetch per API call.
        """
        offset = 0
        while True:
            page = self.get_artifacts(
                run_id,
                artifact_type=artifact_type,
                limit=page_size,
                offset=offset,
            )
            if not page:
                break
            yield from page
            if len(page) < page_size:
                break
            offset += len(page)

    def get_refined_trajectories(
        self,
        run_id: str,
        *,
        page_size: int = 100,
    ) -> List[ArtifactSummary]:
        """Get all refined trajectory artifacts for a run.

        Refined trajectories are compressed, verified versions of agent
        execution traces with redundant error-recovery cycles removed.
        They produce higher-quality training signal than raw trajectories.

        Each artifact's ``content`` includes:
            - ``refined_trajectory``: the compressed step sequence
            - ``compression_ratio``: how much the trajectory was reduced
            - ``original_step_count`` / ``refined_step_count``: step counts
            - ``prompt``, ``final_output``, ``score``, ``domain``

        Args:
            run_id: The run to fetch refined trajectories from.
            page_size: Number of artifacts to fetch per API call.

        Returns:
            List of refined trajectory artifacts. Empty if the run
            produced no refined trajectories (e.g. no passing workflow
            tasks, or refinement not enabled for your tier).

        Example::

            trajectories = client.get_refined_trajectories(run_id)
            for t in trajectories:
                ratio = t.compression_ratio
                print(f"  {t.content['domain']}: "
                      f"{t.content['original_step_count']}→"
                      f"{t.content['refined_step_count']} steps "
                      f"({ratio:.0%} of original)")
        """
        return list(
            self.iter_artifacts(
                run_id,
                artifact_type="refined_trajectory",
                page_size=page_size,
            )
        )

    def get_insights(self, run_id: str) -> Dict[str, Any]:
        """Get rich analytics for a completed run.

        Returns model rankings, J1/J2/J3 scoring metrics, percentiles,
        win counts, gap priority breakdowns, per-model cost and
        token statistics, and capability breakdowns.
        """
        return self._request("GET", f"/v1/runs/{self._path_segment(run_id)}/insights")

    def request_review(
        self,
        run_id: str,
        gap_ids: List[str],
        *,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request human review for specific capability gaps.

        Args:
            run_id: The run containing the gaps.
            gap_ids: IDs of gaps to request review for.
            message: Optional message for the reviewer.

        Returns:
            Confirmation dict with review request details.
        """
        body: Dict[str, Any] = {"gap_ids": gap_ids}
        if message:
            body["message"] = message
        return self._request(
            "POST",
            f"/v1/runs/{self._path_segment(run_id)}/request-review",
            json_body=body,
        )

    def forge(self, run_id: str) -> Dict[str, Any]:
        """Generate new evaluation tasks targeting gaps found in a run.

        Args:
            run_id: A completed run with identified gaps.

        Returns:
            Dict with details about the generated tasks.
        """
        return self._request("POST", f"/v1/runs/{self._path_segment(run_id)}/forge")

    # ── cross-run analytics ──────────────────────────────────────────

    def get_leaderboard(self) -> Dict[str, Any]:
        """Get cross-run model leaderboard.

        Ranks all models you've evaluated across runs by overall score.
        """
        return self._request("GET", "/v1/leaderboard")

    def get_domain_leaderboard(self) -> Dict[str, Any]:
        """Get per-domain model scores across all your runs."""
        return self._request("GET", "/v1/leaderboard/domains")

    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost-efficiency rankings with live pricing.

        Compares model performance against API cost to identify
        the best value options.
        """
        return self._request("GET", "/v1/cost-analysis")

    def get_precomputed_insights(self) -> Dict[str, Any]:
        """Get per-domain best-model recommendations."""
        return self._request("GET", "/v1/insights/precomputed")

    # ── routing ──────────────────────────────────────────────────────

    def route(
        self,
        prompt: str,
        *,
        strategy: str = "quality_first",
        max_candidates: int = 5,
        router_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a model-harness recommendation for a task.

        The trained router predicts which model-harness pair will score
        best on the given prompt, based on patterns learned from past
        evaluation results.

        Args:
            prompt: The task prompt to route.
            strategy: Routing strategy. One of:

                - ``quality_first`` (default): highest predicted score
                - ``cost_first``: cheapest model above quality floor
                - ``balanced``: best score/cost ratio (Pareto-efficient)
                - ``selective``: quality-first but downgrades when safe (best cost/quality trade-off)
                - ``cascade``: cheapest first (for try-cheap-escalate patterns)
                - ``latency_first``: lowest latency, best score among ties
            max_candidates: Number of ranked alternatives to return (1-20).
            router_name: Use a specific named custom router (e.g.
                ``"legal-team"``). Uses your default router if omitted.

        Returns:
            Dict with:
                - ``confidence``: prediction confidence (0-1)
                - ``primary``: top recommended model (dict with model_id, harness, score, cost, etc.)
                - ``candidates``: ranked list of alternatives
                - ``strategy``: the strategy used
                - ``explanation``: human-readable recommendation

        Example::

            rec = client.route("Write a Python function that validates email addresses")
            print(rec["primary"]["model_id"])  # e.g. "anthropic/claude-opus-4.6"
            print(rec["primary"]["harness"])   # e.g. "direct"

            # Use selective routing with a custom router
            rec = client.route(
                "Draft an NDA for software contractors",
                strategy="selective",
                router_name="legal-team",
            )
        """
        body: Dict[str, Any] = {
            "prompt": prompt,
            "strategy": strategy,
            "max_candidates": max_candidates,
        }
        if router_name:
            body["router_name"] = router_name
        return self._request("POST", "/v1/route", json_body=body)

    def get_routing_policy(self) -> Dict[str, Any]:
        """Get summary statistics about the current routing policy.

        Returns domains covered, number of models evaluated, top models
        globally, and when the policy was last generated.

        Returns:
            Dict with ``n_models``, ``n_domains``, ``domains`` (list),
            ``top_models`` (list), and ``generated_at``.
        """
        return self._request("GET", "/v1/route/policy")

    def get_domain_insights(
        self,
        domain: str,
        *,
        quality_floor: float = 0.0,
    ) -> Dict[str, Any]:
        """Get rich model comparison insights for a specific domain.

        Answers "what's the best model for X?" with structured data
        about best quality, cheapest passing, best value, and the
        Pareto frontier.

        Args:
            domain: The domain to get insights for (e.g. ``"coding"``,
                ``"law"``, ``"math"``).
            quality_floor: Minimum score threshold (0-1) for a model to
                be considered "passing". Default 0 (all models included).

        Returns:
            Dict with ``best_model``, ``cheapest_passing``, ``best_value``,
            ``pareto_frontier``, ``comparison`` narrative, and counts.
        """
        return self._request(
            "GET",
            f"/v1/route/insights/{self._path_segment(domain)}",
            params={"quality_floor": quality_floor} if quality_floor > 0 else None,
        )

    def create_routing_mask(
        self,
        name: str,
        *,
        allowed_providers: Optional[List[str]] = None,
        blocked_providers: Optional[List[str]] = None,
        allowed_models: Optional[List[str]] = None,
        blocked_models: Optional[List[str]] = None,
        max_cost_per_request_usd: Optional[float] = None,
        min_quality_score: float = 0.0,
        max_latency_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create or update a routing mask for your organization.

        A mask filters the router's output with your constraints:
        provider whitelists, cost ceilings, quality floors, etc.

        Args:
            name: Human-readable name for this mask config.
            allowed_providers: Only use models from these providers.
            blocked_providers: Never use models from these providers.
            allowed_models: Only use these specific model IDs.
            blocked_models: Never use these specific model IDs.
            max_cost_per_request_usd: Maximum cost per request.
            min_quality_score: Minimum quality score (0-1).
            max_latency_s: Maximum acceptable latency in seconds.

        Returns:
            Confirmation dict with mask ID and name.

        Example::

            client.create_routing_mask(
                "production",
                allowed_providers=["openai", "anthropic"],
                max_cost_per_request_usd=0.05,
                min_quality_score=0.8,
            )
        """
        body: Dict[str, Any] = {
            "name": name,
            "allowed_providers": allowed_providers or [],
            "blocked_providers": blocked_providers or [],
            "allowed_models": allowed_models or [],
            "blocked_models": blocked_models or [],
            "max_cost_per_request_usd": max_cost_per_request_usd,
            "min_quality_score": min_quality_score,
            "max_latency_s": max_latency_s,
        }
        return self._request("POST", "/v1/route/mask", json_body=body)

    def get_routing_mask(self) -> Dict[str, Any]:
        """Get your current routing mask configuration.

        Returns:
            Dict with ``mask`` (the mask config, or None if not configured).
        """
        return self._request("GET", "/v1/route/mask")

    def train_router(
        self,
        *,
        router_name: str = "default",
        domains: Optional[List[str]] = None,
        cost_weight: float = 0.02,
    ) -> Dict[str, Any]:
        """Train a custom router using your evaluation data.

        Queues a background training job. The router learns which models
        perform best on your specific workload from your past evaluation
        results. Requires at least 50 evaluated tasks.

        Args:
            router_name: Name for this router (e.g. ``"default"``,
                ``"legal-team"``, ``"coding"``). Allows multiple routers
                per organization.
            domains: Optional domain filter — only train on tasks from
                these domains (e.g. ``["law", "compliance"]``).
            cost_weight: How much to penalize expensive models (0-1).
                0 = pure quality, 1 = heavy cost preference. Default 0.02.

        Returns:
            Dict with ``job_id``, ``status`` ("queued"), ``router_name``,
            and ``poll_url`` for checking progress.

        Example::

            job = client.train_router(router_name="legal-team", domains=["law"])
            # Poll until complete
            result = client.get_training_job(job["job_id"])
        """
        body: Dict[str, Any] = {
            "router_name": router_name,
            "cost_weight": cost_weight,
        }
        if domains:
            body["domains"] = domains
        return self._request("POST", "/v1/route/train", json_body=body)

    def get_training_job(self, job_id: str) -> Dict[str, Any]:
        """Poll the status of a router training job.

        Args:
            job_id: The job ID returned by :meth:`train_router`.

        Returns:
            Dict with ``status`` (queued/running/completed/failed),
            ``router_name``, ``domains``, timestamps, and ``result``
            (when completed) or ``error`` (when failed).
        """
        return self._request("GET", f"/v1/route/train/{job_id}")

    def wait_for_training(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Wait for a router training job to complete.

        Args:
            job_id: The job ID returned by :meth:`train_router`.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Final job status dict (same as :meth:`get_training_job`).

        Raises:
            TimeoutError: If the job doesn't complete within ``timeout``.
            ApiError: If the job fails.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get_training_job(job_id)
            if job["status"] == "completed":
                return job
            if job["status"] == "failed":
                raise ApiError(
                    f"Router training failed: {job.get('error', 'unknown error')}",
                    status_code=400,
                    response_body=job,
                )
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Router training job {job_id} did not complete within {timeout}s"
        )

    def list_routers(self) -> Dict[str, Any]:
        """List all custom routers trained for your organization.

        Returns:
            Dict with ``routers`` (list of trained routers with metadata)
            and ``recent_jobs`` (recent training job summaries).
        """
        return self._request("GET", "/v1/route/routers")

    def delete_router(self, router_name: str) -> Dict[str, Any]:
        """Delete a named custom router.

        The ``"default"`` router cannot be deleted (re-train it instead).

        Args:
            router_name: Name of the router to delete.

        Returns:
            Confirmation dict.
        """
        return self._request(
            "DELETE", f"/v1/route/routers/{self._path_segment(router_name)}"
        )

    def get_router_performance(self, *, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get the router's quality and cost metrics by domain.

        Shows how the router compares to the best single model on each
        domain — quality retention percentage and cost savings.

        Args:
            domain: Optional domain filter (e.g. ``"law"``). Returns
                all domains if omitted.

        Returns:
            Dict with ``overall_quality_vs_reference``, ``overall_cost_savings_pct``,
            ``domains`` (per-domain breakdown), and ``reference_model``.

        Example::

            perf = client.get_router_performance(domain="coding")
            print(f"Quality: {perf['overall_quality_vs_reference']}%")
            print(f"Cost savings: {perf['overall_cost_savings_pct']}%")
        """
        params = {"domain": domain} if domain else None
        return self._request("GET", "/v1/route/performance", params=params)

    # ── custom tasks ──────────────────────────────────────────────────

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get details for a specific task.

        Args:
            task_id: The task ID.

        Returns:
            Task dict. Content availability depends on your plan.
        """
        return self._request("GET", f"/v1/tasks/{self._path_segment(task_id)}")

    def create_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a single custom evaluation task.

        Args:
            task: Task dict with ``domain``, ``capability``, ``prompt``,
                and at least one scoring field (``ground_truth`` or
                ``rubric``).

        Returns:
            The created task with its assigned ID.
        """
        return self._request("POST", "/v1/tasks", json_body=task)

    def upload_custom_tasks(
        self,
        tasks: List[Dict[str, Any]],
    ) -> CustomTaskUploadResult:
        """Batch upload custom evaluation tasks.

        Tasks are private to your organization.

        Args:
            tasks: List of task dicts. Max per call determined by
                :meth:`get_task_upload_limits`. Each must contain
                ``domain``, ``capability``, ``prompt``, and at least
                one scoring field (``ground_truth`` or ``rubric``).

        Returns:
            Upload result with task IDs.

        Raises:
            InsufficientCreditsError: If the account has insufficient credits.
        """
        data = self._request(
            "POST",
            "/v1/tasks/upload",
            json_body={"tasks": tasks},
        )
        return CustomTaskUploadResult.from_dict(data)

    def get_task_upload_limits(self) -> Dict[str, Any]:
        """Get upload constraints (max file size, max tasks per batch).

        Returns:
            Dict with ``max_file_size`` and ``max_tasks`` fields.
        """
        return self._request("GET", "/v1/tasks/upload/limits")

    def classify_tasks(
        self,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Auto-classify tasks by domain and capability.

        Accepts simple prompt + expected_answer pairs and returns
        classified tasks with inferred domain, capability, and difficulty.

        Args:
            tasks: List of dicts with ``prompt`` (required), ``expected_answer``
                (optional), and ``rubric`` (optional).

        Returns:
            Dict with ``tasks`` (list of classified tasks) and
            ``domains_found`` (unique domains detected).
        """
        return self._request(
            "POST",
            "/v1/tasks/classify",
            json_body={"tasks": tasks},
        )

    def list_tasks(
        self,
        *,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List available evaluation tasks (single page).

        For automatic pagination, use :meth:`iter_tasks`.

        Args:
            domain: Filter by domain (coding, math, legal, etc.).
            source: Filter by source (``system`` or ``custom``).
            limit: Max results per page.
            offset: Pagination offset.
        """
        return self._request(
            "GET",
            "/v1/tasks",
            params={
                "domain": domain,
                "source": source,
                "limit": limit,
                "offset": offset,
            },
        )

    def iter_tasks(
        self,
        *,
        domain: Optional[str] = None,
        source: Optional[str] = None,
        page_size: int = 100,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over all available tasks, auto-paginating.

        Args:
            domain: Filter by domain (coding, math, legal, etc.).
            source: Filter by source (``system`` or ``custom``).
            page_size: Number of tasks to fetch per API call.
        """
        offset = 0
        while True:
            data = self.list_tasks(
                domain=domain,
                source=source,
                limit=page_size,
                offset=offset,
            )
            tasks = data.get("tasks", [])
            if not tasks:
                break
            yield from tasks
            if len(tasks) < page_size:
                break
            offset += len(tasks)

    def delete_task(self, task_id: str) -> None:
        """Delete a custom task you own."""
        self._request("DELETE", f"/v1/tasks/{self._path_segment(task_id)}")

    # ── RL environments ─────────────────────────────────────────────

    def create_rl_session(
        self,
        task_id: str,
        *,
        env_type: str = "single_turn",
        reward_mode: str = "continuous",
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> "RLSession":
        """Create an RL environment session and get the initial observation.

        Args:
            task_id: The task to use for this session.
            env_type: Environment type — ``single_turn``, ``code_sandbox``,
                ``agent_workflow``, or ``simulation``. Availability may vary
                by account.
            reward_mode: Reward mode — ``binary``, ``continuous``, or
                ``partial_credit``.
            seed: Optional seed for reproducible episodes (simulation envs).
            max_steps: Override default max steps for multi-step envs.

        Returns:
            An :class:`~epsilab.models.RLSession` with the initial observation.
        """
        body: Dict[str, Any] = {
            "task_id": task_id,
            "env_type": env_type,
            "reward_mode": reward_mode,
        }
        if seed is not None:
            body["seed"] = seed
        if max_steps is not None:
            body["max_steps"] = max_steps
        data = self._request("POST", "/v1/rl/sessions", json_body=body)
        return RLSession.from_dict(data)

    def rl_step(
        self,
        session_id: str,
        action: str,
    ) -> "RLStepResult":
        """Take an action in an RL environment session.

        Args:
            session_id: The session to step.
            action: The action to take. Format depends on environment type:

                - **single_turn**: Free-text response.
                - **code_sandbox**: Python code (complete function).
                - **simulation**: JSON-encoded action dict.
                - **agent_workflow**: JSON tool call.

        Returns:
            An :class:`~epsilab.models.RLStepResult` with the observation,
            reward, and terminal flags.
        """
        data = self._request(
            "POST",
            f"/v1/rl/sessions/{self._path_segment(session_id)}/step",
            json_body={"action": action},
        )
        return RLStepResult.from_dict(data)

    def get_rl_trajectory(self, session_id: str) -> "RLTrajectory":
        """Get the full trajectory for a completed RL session.

        Actions are returned as hashes for privacy. Observations and
        rewards are returned in full.

        Args:
            session_id: The session to retrieve.

        Returns:
            An :class:`~epsilab.models.RLTrajectory` with all steps.
        """
        data = self._request(
            "GET", f"/v1/rl/sessions/{self._path_segment(session_id)}/trajectory"
        )
        return RLTrajectory.from_dict(data)

    def verify_rl_trajectory(self, session_id: str) -> Dict[str, Any]:
        """Replay a completed session and verify trajectory integrity.

        Replays stored actions in a fresh environment and checks that the
        resulting trajectory matches. Only deterministically verifiable
        environments support this operation.

        Args:
            session_id: The session to verify.

        Returns:
            Dict with ``verified`` (bool), ``steps_replayed``, and
            any ``divergences`` found.
        """
        return self._request(
            "POST", f"/v1/rl/sessions/{self._path_segment(session_id)}/verify"
        )

    def get_rl_curriculum(
        self,
        *,
        env_type: Optional[str] = None,
        batch_size: int = 64,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get an adaptive curriculum batch of tasks for RL training.

        Returns tasks selected from prior performance where training signal
        is expected to be strongest.

        Args:
            env_type: Environment type filter (single_turn, code_sandbox,
                agent_workflow, simulation).
            batch_size: Number of tasks to return (1-512, default 64).
            domain: Optional domain filter.
            seed: Optional seed for reproducible sampling.

        Returns:
            Dict with ``curriculum`` (frontier/exploration/retention task
            lists), ``stats_summary``, and ``difficulty_profile``.
        """
        params: Dict[str, Any] = {"batch_size": batch_size}
        if env_type:
            params["env_type"] = env_type
        if domain:
            params["domain"] = domain
        if seed is not None:
            params["seed"] = seed
        return self._request("GET", "/v1/rl/curriculum", params=params)

    def export_rl_sessions(
        self,
        format: str = "grpo",
        *,
        env_type: Optional[str] = None,
        domain: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        min_sessions: Optional[int] = None,
        min_score_gap: Optional[float] = None,
        pass_threshold: Optional[float] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export completed RL sessions as training data.

        Args:
            format: Export format — ``grpo``, ``dpo``, ``kto``, or
                ``process_supervision``.
            env_type: Filter by environment type.
            domain: Filter by task domain.
            task_ids: Filter by specific task IDs.
            min_sessions: Minimum sessions per task needed to form
                groups (relevant for GRPO/DPO).
            min_score_gap: Minimum score gap between chosen/rejected
                for DPO pairs.
            pass_threshold: Score threshold to consider a session
                as positive/passing.
            limit: Maximum number of records to return (max 500).
            offset: Pagination offset.

        Returns:
            Dict with ``records`` (list of training examples),
            ``format``, ``total``, and ``diagnostics``.
        """
        body: Dict[str, Any] = {"format": format}
        if env_type:
            body["env_type"] = env_type
        if domain:
            body["domain"] = domain
        if task_ids:
            body["task_ids"] = task_ids
        if min_sessions is not None:
            body["min_sessions"] = min_sessions
        if min_score_gap is not None:
            body["min_score_gap"] = min_score_gap
        if pass_threshold is not None:
            body["pass_threshold"] = pass_threshold
        if limit is not None:
            body["limit"] = limit
        if offset is not None:
            body["offset"] = offset
        return self._request("POST", "/v1/rl/exports", json_body=body)

    def close_rl_session(
        self,
        session_id: str,
        *,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually close an active RL session.

        Useful for abandoning a session early without waiting for TTL
        reaping.

        Args:
            session_id: The session to close.
            reason: Optional close reason (default: ``manual_close``).

        Returns:
            Dict with ``session_id``, ``status``, and ``reason``.
        """
        body: Dict[str, Any] = {}
        if reason:
            body["reason"] = reason
        return self._request(
            "POST",
            f"/v1/rl/sessions/{self._path_segment(session_id)}/close",
            json_body=body,
        )

    # ── exports ──────────────────────────────────────────────────────

    def export_run(
        self,
        run_id: str,
        format: str,
        *,
        path: Optional[str] = None,
    ) -> Any:
        """Export training data or artifacts for a completed run.

        Args:
            run_id: The run to export.
            format: Export format. One of:

                Training data:
                    ``dpo``, ``quality_dpo``, ``sft``, ``kto``, ``grpo``,
                    ``sharegpt``, ``process_supervision``.
                Raw:
                    ``jsonl``, ``artifacts``.
                Other:
                    ``report``, ``yaml``, ``pytest``, ``all`` (zip archive).

                The ``sharegpt`` and ``process_supervision`` formats
                automatically include refined trajectories when available.
                Refined trajectories are compressed, verified versions of
                agent execution traces — they produce higher-quality
                training signal with fewer redundant steps.

            path: If provided, write the response to this file path.

        Returns:
            The raw text response, or *None* if *path* was provided.
        """
        if path:
            p = Path(path)
            self._stream_raw_to_path(
                "GET",
                f"/v1/runs/{self._path_segment(run_id)}/export",
                p,
                params={"format": format},
            )
            return None
        text = self._request_raw(
            "GET",
            f"/v1/runs/{self._path_segment(run_id)}/export",
            params={"format": format},
        )
        return text

    def stream_export(
        self,
        run_id: str,
        format: str,
        *,
        path: Optional[str] = None,
        min_score_gap: Optional[float] = None,
        min_chosen_score: Optional[float] = None,
        max_rejected_score: Optional[float] = None,
    ) -> Any:
        """Stream training data as newline-delimited JSON (NDJSON).

        Like :meth:`export_run` but uses the streaming endpoint, which
        avoids buffering the full dataset in memory. Supports quality
        filter parameters.

        Only line-based formats are supported: ``jsonl``, ``dpo``,
        ``quality_dpo``, ``sft``, ``kto``, ``grpo``, ``sharegpt``,
        ``process_supervision``.

        The ``process_supervision`` format includes refined trajectory
        records (marked with ``"refined": true``) when available. These
        are higher-quality training signal with fewer redundant steps.

        Args:
            run_id: The run to export.
            format: Export format (see above).
            path: If provided, stream the response to this file path.
            min_score_gap: Only include pairs where the chosen/rejected
                score gap exceeds this threshold.
            min_chosen_score: Only include pairs where the chosen score
                is at least this value.
            max_rejected_score: Only include pairs where the rejected
                score is at most this value.

        Returns:
            The raw NDJSON text, or *None* if *path* was provided.
        """
        params: Dict[str, Any] = {"format": format}
        if min_score_gap is not None:
            params["min_score_gap"] = min_score_gap
        if min_chosen_score is not None:
            params["min_chosen_score"] = min_chosen_score
        if max_rejected_score is not None:
            params["max_rejected_score"] = max_rejected_score

        if path:
            p = Path(path)
            self._stream_raw_to_path(
                "GET",
                f"/v1/runs/{self._path_segment(run_id)}/export/stream",
                p,
                params=params,
            )
            return None
        text = self._request_raw(
            "GET",
            f"/v1/runs/{self._path_segment(run_id)}/export/stream",
            params=params,
        )
        return text

    # ── API keys ─────────────────────────────────────────────────────

    def list_api_keys(self) -> Dict[str, Any]:
        """List your API keys (secrets are not included)."""
        return self._request("GET", "/v1/api-keys")

    def create_api_key(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Create a new API key.

        Args:
            label: Human-readable label for the key.

        Returns:
            Dict with ``key_id`` and the raw ``key`` (shown only once).
        """
        body: Dict[str, Any] = {}
        if label:
            body["label"] = label
        return self._request("POST", "/v1/api-keys", json_body=body)

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key.

        Args:
            key_id: The key ID to revoke.
        """
        self._request("DELETE", f"/v1/api-keys/{self._path_segment(key_id)}")

    # ── usage ────────────────────────────────────────────────────────

    def get_usage(self, *, period: Optional[str] = None) -> List[UsageRecord]:
        """Get your monthly usage summary.

        Args:
            period: Optional month filter in ``YYYY-MM`` format.
        """
        data = self._request(
            "GET",
            "/v1/usage",
            params={"period": period} if period else None,
        )
        return [UsageRecord.from_dict(u) for u in data.get("usage", [])]

    # ── billing / credits ────────────────────────────────────────────

    def get_credit_balance(self) -> Dict[str, Any]:
        """Get current credit balance (balance, lifetime_purchased, lifetime_used)."""
        return self._request("GET", "/v1/billing/balance")

    def get_credit_ledger(self, *, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get credit transaction history."""
        return self._request(
            "GET",
            "/v1/billing/ledger",
            params={"limit": limit, "offset": offset},
        )
