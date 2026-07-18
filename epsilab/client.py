"""Epsilab Python SDK, main client module.

Provides :class:`EpsilabClient`, the primary interface for the
RL Environment Hub and Marketplace, model evaluations, and training
data export.  Importable as ``from epsilab import Epsilab``.
"""

from __future__ import annotations

import json
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
    ApplicationTool,
    ApplicationToolRelease,
    ArtifactSummary,
    CostEstimate,
    CustomTaskUploadResult,
    EnvironmentListing,
    EnvironmentRelease,
    EnvironmentSession,
    EnvironmentStepResult,
    EvaluationResult,
    GapSummary,
    RLSession,
    RLStepResult,
    RLTrajectory,
    RunSummary,
    UsageRecord,
)

logger = logging.getLogger("epsilab")
_request_logger = logging.getLogger("epsilab.http")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0
_CREDENTIALS_FILE = Path.home() / ".epsilab" / "credentials.json"


def _load_stored_credentials() -> tuple[Optional[str], Optional[str]]:
    """Read API key and base URL from ~/.epsilab/credentials.json (set by ``epsilab login``)."""
    try:
        if not _CREDENTIALS_FILE.exists():
            return None, None
        data = json.loads(_CREDENTIALS_FILE.read_text())
        profile = data.get("active_profile", "default")
        profiles = data.get("profiles", {})
        creds = profiles.get(profile, {})
        return creds.get("api_key"), data.get("api_base")
    except Exception:
        return None, None
_QUALITY_REPORT_TYPES = frozenset(
    {
        "protocol_conformance",
        "startup_cleanup",
        "reset_independence",
        "verifier_repeatability",
        "adversarial",
        "contamination",
        "benchmark",
        "full_qualification",
    }
)
_DEFAULT_BACKOFF_MAX = 60.0


class EpsilabClient:
    """Client for the Epsilab evaluation API.

    Args:
        api_base: Base URL for the API. Defaults to ``EPSILAB_API_BASE``
            env var or the production URL.
        api_key: API key for authentication. Resolved in order:
            explicit argument, ``EPSILAB_API_KEY`` env var, then
            ``~/.epsilab/credentials.json`` (set by ``epsilab login``).
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

        resolved_key = api_key or config_value("EPSILAB_API_KEY")
        if not resolved_key:
            resolved_key, stored_base = _load_stored_credentials()
            if stored_base and not api_base and not config_value("EPSILAB_API_BASE"):
                self.api_base = stored_base.rstrip("/")
        self._api_key: Optional[str] = resolved_key
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

    @staticmethod
    def _auto_idem_key() -> str:
        import uuid
        return uuid.uuid4().hex

    def _buffered_request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
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
                    headers=extra_headers,
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

    _deprecation_warned: set = set()

    def _check_deprecation(self, resp: httpx.Response, path: str) -> None:
        """Emit a one-time warning if the server signals deprecation."""
        if resp.headers.get("Deprecation") != "true":
            return
        key = path.split("?")[0]
        if key in self._deprecation_warned:
            return
        self._deprecation_warned.add(key)
        import warnings

        sunset = resp.headers.get("Sunset", "soon")
        warnings.warn(
            f"Epsilab API endpoint '{key}' is deprecated and will be "
            f"removed after {sunset}. Migrate to the RL Environment Hub. "
            f"See https://docs.epsilab.com/migration/environments",
            DeprecationWarning,
            stacklevel=4,
        )

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
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        t0 = time.monotonic()
        _request_logger.debug("%s %s", method, path)
        resp = self._buffered_request_with_retry(
            method,
            path,
            params=params,
            json_body=json_body,
            extra_headers=extra_headers,
        )
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _request_logger.debug(
            "%s %s -> %d (%dms)", method, path, resp.status_code, elapsed_ms
        )
        self._check_deprecation(resp, path)
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
        self._check_deprecation(resp, path)
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

    # ── runs (deprecated — use RL environments) ────────────────────

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

        .. deprecated::
            Use RL environments instead. See ``examples/run_environment.py``.
        """
        import warnings
        warnings.warn(
            "create_run() is deprecated. Use RL environments instead. "
            "See examples/run_environment.py",
            DeprecationWarning, stacklevel=2,
        )
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

    # ── multi-model evaluations (deprecated — use RL environments) ──

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

        .. deprecated::
            Use RL environments instead. See ``examples/run_environment.py``.

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
        import warnings
        warnings.warn(
            "create_evaluation() is deprecated. Use RL environments instead. "
            "See examples/run_environment.py",
            DeprecationWarning, stacklevel=2,
        )
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

        .. deprecated::
            Use RL environments instead.
        """
        import warnings
        warnings.warn(
            "estimate_evaluation_cost() is deprecated. Use RL environments instead.",
            DeprecationWarning, stacklevel=2,
        )
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

    # ── gaps, insights & artifacts (deprecated) ─────────────────────

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

    # ── cross-run analytics (deprecated) ─────────────────────────────

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

    # ── routing (deprecated) ──────────────────────────────────────────

    def route(
        self,
        prompt: str,
        *,
        strategy: str = "quality_first",
        max_candidates: int = 5,
        router_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a model-harness recommendation for a task.

        .. deprecated::
            Routing is deprecated.

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
        import warnings
        warnings.warn(
            "route() is deprecated.",
            DeprecationWarning, stacklevel=2,
        )
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

    # ── voice evaluations (deprecated) ──────────────────────────────

    def register_voice_asset(
        self,
        asset_id: str,
        uri: str,
        *,
        kind: str = "input_audio",
        format: str = "wav",
        sample_rate: int = 16000,
        duration_s: float = 0.0,
        channels: int = 1,
        codec: Optional[str] = None,
        language: Optional[str] = None,
        locale: Optional[str] = None,
        synthetic: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a voice asset for use in voice evaluation tasks.

        .. deprecated::
            Voice evaluations are deprecated.

        Args:
            asset_id: Unique identifier for this asset.
            uri: URI pointing to the audio file (``gs://``, ``s3://``,
                or ``https://``).
            kind: Asset role — ``input_audio``, ``reference_audio``, etc.
            format: Audio format (``wav``, ``mp3``, ``flac``, ``ogg``).
            sample_rate: Sample rate in Hz.
            duration_s: Duration in seconds.
            channels: Number of audio channels.
            codec: Optional codec name.
            language: BCP-47 language tag.
            locale: Locale variant (e.g. ``en-US``).
            synthetic: Whether this audio is synthetic/TTS-generated.
            metadata: Additional metadata dict.

        Returns:
            Dict with the validated ``asset`` record.
        """
        import warnings
        warnings.warn(
            "Voice evaluations are deprecated.",
            DeprecationWarning, stacklevel=2,
        )
        body: Dict[str, Any] = {
            "asset_id": asset_id,
            "uri": uri,
            "kind": kind,
            "format": format,
            "sample_rate": sample_rate,
            "duration_s": duration_s,
            "channels": channels,
            "synthetic": synthetic,
        }
        if codec:
            body["codec"] = codec
        if language:
            body["language"] = language
        if locale:
            body["locale"] = locale
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/v1/voice/assets/register", json_body=body)

    def create_voice_task(
        self,
        task_id: str,
        task_type: str,
        prompt: str,
        capability: str,
        *,
        assets: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
        domain: str = "voice",
        verification: str = "wer",
        difficulty: str = "hard",
        input_modality: str = "audio",
        output_modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a voice evaluation task.

        Args:
            task_id: Unique task identifier.
            task_type: Voice task type (e.g. ``voice_asr``, ``voice_tts``,
                ``voice_emotion``, ``voice_speaker_id``).
            prompt: Task prompt or instructions.
            capability: The capability being tested.
            assets: List of voice asset dicts (each must include
                ``asset_id`` and ``uri``).
            ground_truth: Expected output for scoring.
            domain: Task domain (default ``voice``).
            verification: Verification method (default ``wer``).
            difficulty: Task difficulty level.
            input_modality: Input type (``audio``, ``text``).
            output_modality: Output type (``text``, ``audio``).
            metadata: Additional metadata dict.

        Returns:
            Dict with the created task details.
        """
        body: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": task_type,
            "prompt": prompt,
            "capability": capability,
            "domain": domain,
            "verification": verification,
            "difficulty": difficulty,
            "input_modality": input_modality,
            "output_modality": output_modality,
        }
        if assets:
            body["assets"] = assets
        if ground_truth:
            body["ground_truth"] = ground_truth
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/v1/voice/tasks", json_body=body)

    def create_voice_run(
        self,
        target_model: str,
        *,
        target_config: Optional[Dict[str, str]] = None,
        reference_models: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        name: Optional[str] = None,
        max_tasks: Optional[int] = None,
        reference_mode: str = "best_on_task",
        reference_top_k: int = 3,
        pre_registered_reference: Optional[str] = None,
        exploratory: bool = False,
    ) -> "RunSummary":
        """Create a voice evaluation run.

        Evaluates a voice model against reference models on voice tasks.
        Results include per-slice metrics, event timelines, and gap
        detection for voice capabilities.

        Args:
            target_model: The voice model to evaluate.
            target_config: Optional custom endpoint (``base_url`` and
                ``api_key``) for BYOM voice models.
            reference_models: Models to compare against.
            task_ids: Specific task IDs to evaluate on. If omitted,
                tasks are selected automatically.
            task_type: Filter tasks by voice task type.
            name: Display name for the run.
            max_tasks: Maximum number of tasks to evaluate.
            reference_mode: Comparator strategy — ``best_on_task``,
                ``best_model``, ``top_k_mean``, ``panel_mean``, or
                ``pre_registered``.
            reference_top_k: Number of top references for ``top_k_mean``.
            pre_registered_reference: Required when reference_mode is
                ``pre_registered``.
            exploratory: If true, includes broader task sampling.

        Returns:
            A :class:`~epsilab.models.RunSummary` for the queued run.
        """
        body: Dict[str, Any] = {
            "target_model": target_model,
            "reference_mode": reference_mode,
            "reference_top_k": reference_top_k,
            "exploratory": exploratory,
        }
        if target_config:
            body["target_config"] = target_config
        if reference_models:
            body["reference_models"] = reference_models
        if task_ids:
            body["task_ids"] = task_ids
        if task_type:
            body["task_type"] = task_type
        if name:
            body["name"] = name
        if max_tasks is not None:
            body["max_tasks"] = max_tasks
        if pre_registered_reference:
            body["pre_registered_reference"] = pre_registered_reference
        data = self._request("POST", "/v1/voice/runs", json_body=body)
        return RunSummary.from_dict(data)

    def get_voice_slices(self, run_id: str) -> Dict[str, Any]:
        """Get per-slice voice metrics for a completed voice run.

        Returns quality scores broken down by audio characteristics
        (speaker, noise level, accent, etc.).

        Args:
            run_id: The voice run to query.

        Returns:
            Dict with ``slices`` (list of metric dicts) and
            ``total_results``.
        """
        return self._request(
            "GET", f"/v1/voice/runs/{self._path_segment(run_id)}/slices"
        )

    def get_voice_timeline(
        self,
        run_id: str,
        task_id: str,
        *,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the event timeline for a voice task result.

        Returns the sequence of processing events (audio chunks,
        transcription segments, latency markers) for replay and
        debugging.

        Args:
            run_id: The voice run.
            task_id: The specific task within the run.
            model_id: Optional model filter if multiple models produced
                results for the same task.

        Returns:
            Dict with ``event_timeline``, ``output_assets``,
            ``scenario_checks``, and ``model_alias``.
        """
        params: Optional[Dict[str, Any]] = None
        if model_id:
            params = {"model_id": model_id}
        return self._request(
            "GET",
            f"/v1/voice/runs/{self._path_segment(run_id)}/timeline/{self._path_segment(task_id)}",
            params=params,
        )

    def route_voice(
        self,
        prompt: str,
        *,
        task_type: str = "voice_asr",
        strategy: str = "quality_first",
        max_candidates: int = 5,
        input_modality: str = "audio",
        output_modality: str = "text",
        language: Optional[str] = None,
        locale: Optional[str] = None,
        max_latency_s: Optional[float] = None,
        max_cost_per_request_usd: Optional[float] = None,
        router_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a voice workload to the best model candidates.

        Uses evaluation history and constraints to recommend models
        for a given voice task.

        Args:
            prompt: Task description or scenario to route.
            task_type: Voice task type (e.g. ``voice_asr``, ``voice_tts``).
            strategy: Routing strategy — ``quality_first``,
                ``cost_first``, ``balanced``, ``latency_first``,
                ``cascade``, or ``selective``.
            max_candidates: Maximum models to return (1-20).
            input_modality: Input type (``audio`` or ``text``).
            output_modality: Output type (``text`` or ``audio``).
            language: BCP-47 language constraint.
            locale: Locale constraint.
            max_latency_s: Maximum acceptable latency in seconds.
            max_cost_per_request_usd: Maximum cost per request.
            router_name: Use a specific trained router.

        Returns:
            Dict with ``primary`` (top pick), ``candidates`` (ranked
            list), ``strategy``, ``confidence``, and ``explanation``.
        """
        body: Dict[str, Any] = {
            "prompt": prompt,
            "task_type": task_type,
            "strategy": strategy,
            "max_candidates": max_candidates,
            "input_modality": input_modality,
            "output_modality": output_modality,
        }
        if language:
            body["language"] = language
        if locale:
            body["locale"] = locale
        if max_latency_s is not None:
            body["max_latency_s"] = max_latency_s
        if max_cost_per_request_usd is not None:
            body["max_cost_per_request_usd"] = max_cost_per_request_usd
        if router_name:
            body["router_name"] = router_name
        return self._request("POST", "/v1/voice/route", json_body=body)

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

    def list_rl_environments(
        self,
        *,
        domain: Optional[str] = None,
        capability: Optional[str] = None,
        difficulty: Optional[str] = None,
        verification: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List available RL environments (tasks suitable for RL training).

        Args:
            domain: Filter by domain (e.g. ``coding``, ``math``).
            capability: Filter by capability.
            difficulty: Filter by difficulty level.
            verification: Filter by verification type
                (``hidden_tests``, ``simulation``, ``rubric``).
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            Dict with ``environments`` list and ``total`` count.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if domain:
            params["domain"] = domain
        if capability:
            params["capability"] = capability
        if difficulty:
            params["difficulty"] = difficulty
        if verification:
            params["verification"] = verification
        return self._request("GET", "/v1/rl/environments", params=params)

    def list_rl_sessions(
        self,
        *,
        status: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List your RL sessions.

        Args:
            status: Filter by status (``active``, ``completed``, ``failed``,
                ``closed``, ``truncated``).
            task_id: Filter by task ID.
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            Dict with ``sessions`` list and ``total`` count.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if task_id:
            params["task_id"] = task_id
        return self._request("GET", "/v1/rl/sessions", params=params)

    def get_rl_stats(
        self,
        *,
        env_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get RL environment statistics — completion rates, reward distribution.

        Args:
            env_type: Filter by environment type.
            domain: Filter by domain.

        Returns:
            Dict with ``session_counts``, ``reward_distribution``,
            ``difficulty_profile``, and ``task_count``.
        """
        params: Dict[str, Any] = {}
        if env_type:
            params["env_type"] = env_type
        if domain:
            params["domain"] = domain
        return self._request("GET", "/v1/rl/stats", params=params)

    # ── capability matrix (deprecated) ──────────────────────────────

    def get_matrix_models(
        self,
        *,
        modality: Optional[str] = None,
        min_tasks: int = 5,
    ) -> Dict[str, Any]:
        """List all models you've evaluated, with aggregated stats.

        .. deprecated::
            Capability matrix is deprecated.

        Args:
            modality: Filter by modality (``text``, ``voice``).
            min_tasks: Minimum tasks evaluated to include a model (default 5).

        Returns:
            Dict with model stats across all evaluations.
        """
        import warnings
        warnings.warn(
            "Capability matrix methods are deprecated.",
            DeprecationWarning, stacklevel=2,
        )
        params: Dict[str, Any] = {"min_tasks": min_tasks}
        if modality:
            params["modality"] = modality
        return self._request("GET", "/v1/matrix/models", params=params)

    def get_matrix_model_profile(self, model_id: str) -> Dict[str, Any]:
        """Get a detailed profile for a specific model (enterprise only).

        Includes coverage breakdown, strengths, weaknesses, and domain scores.

        Args:
            model_id: The model identifier (e.g. ``openai/gpt-4o``).

        Returns:
            Dict with coverage, domain_scores, strengths, weaknesses.
        """
        return self._request(
            "GET", f"/v1/matrix/models/{self._path_segment(model_id)}/profile"
        )

    def get_matrix_model_gaps(
        self,
        model_id: str,
        *,
        domain: Optional[str] = None,
        min_gap: float = 0.05,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get capability gaps for a specific model.

        Shows tasks/capabilities where this model underperforms relative
        to other models you've evaluated.

        Args:
            model_id: The model to analyze.
            domain: Filter by domain.
            min_gap: Minimum gap score to include (0.0-1.0, default 0.05).
            limit: Max results (1-200, default 50).

        Returns:
            Dict with ``gaps`` list showing where the model underperforms.
        """
        params: Dict[str, Any] = {"min_gap": min_gap, "limit": limit}
        if domain:
            params["domain"] = domain
        return self._request(
            "GET", f"/v1/matrix/models/{self._path_segment(model_id)}/gaps", params=params
        )

    def get_matrix_model_capabilities(
        self,
        model_id: str,
        *,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get per-capability breakdown for a model.

        Args:
            model_id: The model to analyze.
            domain: Filter by domain.

        Returns:
            Dict with per-capability scores and gap analysis.
        """
        params: Dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return self._request(
            "GET",
            f"/v1/matrix/models/{self._path_segment(model_id)}/capabilities",
            params=params,
        )

    def get_matrix_gaps(
        self,
        *,
        model_id: Optional[str] = None,
        domain: Optional[str] = None,
        capability: Optional[str] = None,
        min_alpha: Optional[float] = None,
        priority: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get cross-model capability gaps.

        Args:
            model_id: Show gaps where this model underperforms.
            domain: Filter by domain.
            capability: Filter by capability.
            min_alpha: Minimum gap significance threshold.
            priority: Filter by priority level.
            limit: Max results (1-500, default 100).

        Returns:
            Dict with ``gaps`` list and aggregation metadata.
        """
        params: Dict[str, Any] = {"limit": limit}
        if model_id:
            params["model_id"] = model_id
        if domain:
            params["domain"] = domain
        if capability:
            params["capability"] = capability
        if min_alpha is not None:
            params["min_alpha"] = min_alpha
        if priority:
            params["priority"] = priority
        return self._request("GET", "/v1/matrix/gaps", params=params)

    def get_matrix_domains(
        self,
        *,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get per-domain score breakdown across all evaluations.

        Args:
            model_id: Filter to a specific model's domain scores.

        Returns:
            Dict with per-domain performance aggregates.
        """
        params: Dict[str, Any] = {}
        if model_id:
            params["model_id"] = model_id
        return self._request("GET", "/v1/matrix/domains", params=params)

    def get_matrix_scores(
        self,
        *,
        model_id: Optional[str] = None,
        domain: Optional[str] = None,
        capability: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get raw scores from the capability matrix (enterprise only).

        Args:
            model_id: Filter by model.
            domain: Filter by domain.
            capability: Filter by capability.
            modality: Filter by modality (``text``, ``voice``).
            limit: Max results (1-2000, default 500).
            offset: Pagination offset.

        Returns:
            Dict with ``scores`` list of per-task model scores.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id:
            params["model_id"] = model_id
        if domain:
            params["domain"] = domain
        if capability:
            params["capability"] = capability
        if modality:
            params["modality"] = modality
        return self._request("GET", "/v1/matrix/scores", params=params)

    def get_matrix_artifacts(
        self,
        *,
        model_id: Optional[str] = None,
        domain: Optional[str] = None,
        artifact_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get training artifacts from the capability matrix.

        Args:
            model_id: Filter to artifacts involving this model.
            domain: Filter by domain.
            artifact_type: Filter by type (``sft``, ``dpo``, ``kto``, ``grpo``).
            limit: Max results (1-500, default 100).
            offset: Pagination offset.

        Returns:
            Dict with ``artifacts`` list.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if model_id:
            params["model_id"] = model_id
        if domain:
            params["domain"] = domain
        if artifact_type:
            params["artifact_type"] = artifact_type
        return self._request("GET", "/v1/matrix/artifacts", params=params)

    def get_matrix_insights(
        self,
        *,
        modality: Optional[str] = None,
        min_tasks: int = 5,
        model_id: Optional[str] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get high-level insights from the capability matrix (enterprise only).

        Identifies overall patterns, model rankings, and recommended
        training priorities.

        Args:
            modality: Filter by modality (``text``, ``voice``).
            min_tasks: Minimum tasks for model inclusion (default 5).
            model_id: Target model perspective.
            refresh: Force recomputation of insights cache.

        Returns:
            Dict with rankings, patterns, and recommendations.
        """
        params: Dict[str, Any] = {"min_tasks": min_tasks}
        if modality:
            params["modality"] = modality
        if model_id:
            params["model_id"] = model_id
        if refresh:
            params["refresh"] = "true"
        return self._request("GET", "/v1/matrix/insights", params=params)

    def get_matrix_coverage(
        self,
        *,
        domain: Optional[str] = None,
        models: Optional[List[str]] = None,
        modality: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get evaluation coverage matrix (enterprise only).

        Shows how many tasks each model has been evaluated on per domain.

        Args:
            domain: Filter by domain.
            models: Filter to specific model IDs.
            modality: Filter by modality (``text``, ``voice``).

        Returns:
            Dict with coverage matrix data.
        """
        params: Dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if models:
            params["models"] = ",".join(models)
        if modality:
            params["modality"] = modality
        return self._request("GET", "/v1/matrix/coverage", params=params)

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

    # ══════════════════════════════════════════════════════════════════
    # Environment Hub & Marketplace
    # ══════════════════════════════════════════════════════════════════

    # ── Discovery & catalog ──────────────────────────────────────────

    def list_environment_listings(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List["EnvironmentListing"]:
        """Browse environment listings on the hub.

        Returns public listings, your own listings, and shared listings you can discover.

        Args:
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            List of :class:`~epsilab.models.EnvironmentListing`.
        """
        data = self._request(
            "GET",
            "/v1/environment-listings",
            params={"limit": limit, "offset": offset},
        )
        items = data if isinstance(data, list) else data.get("listings", data.get("items", []))
        return [EnvironmentListing.from_dict(d) for d in items]

    def get_environment_listing(self, listing_id: str) -> "EnvironmentListing":
        """Get a hub listing by ID, including an unlisted listing reached by direct link."""
        data = self._request(
            "GET",
            f"/v1/environment-listings/{self._path_segment(listing_id)}",
        )
        return EnvironmentListing.from_dict(data)

    def list_application_tools(
        self,
        *,
        query: Optional[str] = None,
        plugin: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List["ApplicationTool"]:
        """Browse public Application Tools; authentication is optional."""
        data = self._request(
            "GET",
            "/v1/application-tools",
            params={"q": query, "plugin": plugin, "limit": limit, "offset": offset},
        )
        items = data if isinstance(data, list) else data.get("items", [])
        return [ApplicationTool.from_dict(item) for item in items]

    def get_application_tool(self, tool_id: str) -> "ApplicationTool":
        """Get a public or unlisted Application Tool by ID."""
        data = self._request("GET", f"/v1/application-tools/{self._path_segment(tool_id)}")
        return ApplicationTool.from_dict(data)

    def get_application_tool_release(self, release_id: str) -> "ApplicationToolRelease":
        """Get the recommended public or unlisted Application Tool release by ID."""
        data = self._request(
            "GET",
            f"/v1/application-tool-releases/{self._path_segment(release_id)}",
        )
        return ApplicationToolRelease.from_dict(data)

    def list_public_listings(
        self,
        *,
        query: Optional[str] = None,
        sort_by: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Browse the public marketplace catalog.

        Args:
            query: Free-text search query.
            sort_by: Sort order — ``popular``, ``newest``, ``quality``.
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            List of public listing summaries.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["query"] = query
        if sort_by:
            params["sort_by"] = sort_by
        return self._request("GET", "/v1/public/listings", params=params)

    def search_environments(
        self,
        *,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        env_type: Optional[str] = None,
        badge_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search environments by quality, domain, difficulty, and more.

        Uses the quality-weighted search index for ranked results.

        Args:
            query: Free-text search.
            domain: Filter by domain (e.g. ``coding``, ``math``).
            env_type: Filter by environment type.
            badge_type: Filter by quality badge (e.g. ``gold``).
            difficulty: Filter by difficulty level.
            min_quality_score: Minimum quality score (0.0-1.0).
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            Ranked list of matching environments.
        """
        body: Dict[str, Any] = {}
        if query:
            body["query"] = query
        if domain:
            body["domain"] = domain
        if env_type:
            body["env_type"] = env_type
        if badge_type:
            body["badge_type"] = badge_type
        if difficulty:
            body["difficulty"] = difficulty
        if min_quality_score is not None:
            body["min_quality_score"] = min_quality_score
        return self._request(
            "POST",
            "/v1/environment-search",
            json_body=body,
            params={"limit": limit, "offset": offset},
        )

    def get_environment_release(self, release_id: str) -> "EnvironmentRelease":
        """Get details of a specific environment release.

        Args:
            release_id: The release ID to retrieve.

        Returns:
            An :class:`~epsilab.models.EnvironmentRelease`.
        """
        data = self._request(
            "GET",
            f"/v1/environment-releases/{self._path_segment(release_id)}",
        )
        return EnvironmentRelease.from_dict(data)

    # ── Image upload ──────────────────────────────────────────────────

    def upload_image(self, tarball_path: str, *, tag: str) -> Dict[str, Any]:
        """Upload a Docker image tarball to the platform.

        The platform receives, validates, and stores the image.
        No external credentials are required.

        Args:
            tarball_path: Path to a ``docker save`` tarball (``.tar``).
            tag: Image tag (e.g. ``my-env:0.1.0``).

        Returns:
            Dict with ``image_ref``, ``content_digest``, and ``size_bytes``.
        """
        import pathlib

        path = pathlib.Path(tarball_path)
        if not path.exists():
            raise ApiError(0, f"Tarball not found: {tarball_path}")

        t0 = time.monotonic()
        _request_logger.debug("POST /v1/environment-images (uploading %s)", path.name)

        with open(path, "rb") as f:
            resp = self._client.post(
                "/v1/environment-images",
                files={"image": (path.name, f, "application/x-tar")},
                params={"tag": tag},
                timeout=httpx.Timeout(1800.0, connect=30.0),
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _request_logger.debug(
            "POST /v1/environment-images -> %d (%dms)", resp.status_code, elapsed_ms
        )
        self._check_response_errors(resp)
        return resp.json()

    # ── Hosted sessions ──────────────────────────────────────────────

    def create_environment_session(
        self,
        deployment_id: str,
        *,
        task_id: str,
        seed: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentSession":
        """Create a hosted environment session.

        Provisions a sandboxed environment instance and returns the
        initial observation once ready.

        Args:
            deployment_id: The deployment to run on.
            task_id: Task to execute in this session.
            seed: Optional seed for reproducible episodes.
            idempotency_key: Unique key to prevent duplicate creation.

        Returns:
            An :class:`~epsilab.models.EnvironmentSession` with the
            session token for subsequent step calls.
        """
        body: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            body["seed"] = seed
        headers: Dict[str, str] = {
            "Idempotency-Key": idempotency_key or self._auto_idem_key(),
        }
        data = self._request(
            "POST",
            f"/v1/environment-deployments/{self._path_segment(deployment_id)}/sessions",
            json_body=body,
            extra_headers=headers,
        )
        return EnvironmentSession.from_dict(data)

    def get_environment_session(self, session_id: str) -> "EnvironmentSession":
        """Get the current state of an environment session.

        Args:
            session_id: The session to inspect.

        Returns:
            An :class:`~epsilab.models.EnvironmentSession`.
        """
        data = self._request(
            "GET",
            f"/v1/environment-sessions/{self._path_segment(session_id)}",
        )
        return EnvironmentSession.from_dict(data)

    def wait_for_session(
        self,
        session: "EnvironmentSession",
        *,
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> "EnvironmentSession":
        """Poll until a provisioning session becomes active (or terminal).

        Args:
            session: The session returned by :meth:`create_environment_session`.
            poll_interval: Seconds between status polls.
            timeout: Maximum seconds to wait before raising ``TimeoutError``.

        Returns:
            The updated :class:`~epsilab.models.EnvironmentSession`.
        """
        import time as _time

        deadline = _time.monotonic() + timeout
        while session.status == "provisioning":
            if _time.monotonic() > deadline:
                raise TimeoutError(f"Session {session.session_id} did not become active within {timeout}s")
            _time.sleep(poll_interval)
            session = self.get_environment_session(session.session_id)
        return session

    def environment_step(
        self,
        session_id: str,
        action: str,
        *,
        session_token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentStepResult":
        """Take an action in a hosted environment session.

        The ``session_token`` authenticates this request independently
        of the API key; it was returned when the session was created.

        Args:
            session_id: Session to step.
            action: Action string (format depends on environment).
            session_token: Session-scoped bearer token. If not provided,
                the default API key authorization is used.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            An :class:`~epsilab.models.EnvironmentStepResult` with
            observation, reward, and terminal flags.
        """
        extra_headers: Dict[str, str] = {
            "Idempotency-Key": idempotency_key or self._auto_idem_key(),
        }
        if session_token:
            extra_headers["X-RL-Session-Token"] = session_token
        data = self._request(
            "POST",
            f"/v1/environment-sessions/{self._path_segment(session_id)}/step",
            json_body={"action": action},
            extra_headers=extra_headers,
        )
        return EnvironmentStepResult.from_dict(data)

    def cancel_environment_session(
        self,
        session_id: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an active environment session.

        Args:
            session_id: Session to cancel.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            Dict with the final session state.
        """
        return self._request(
            "POST",
            f"/v1/environment-sessions/{self._path_segment(session_id)}/cancel",
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def refresh_session_token(self, session_id: str) -> Dict[str, Any]:
        """Refresh the session token for a long-running session.

        Args:
            session_id: Session whose token to refresh.

        Returns:
            Dict with ``session_id``, ``session_token``, and
            ``session_token_expires_at``.
        """
        return self._request(
            "POST",
            f"/v1/environment-sessions/{self._path_segment(session_id)}/token",
        )

    def run_environment_episode(
        self,
        deployment_id: str,
        *,
        task_id: str,
        policy_fn: Any,
        seed: Optional[int] = None,
        max_steps: int = 100,
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentSession":
        """Run a complete environment episode using your policy function.

        Convenience wrapper that creates a session, steps until done or
        ``max_steps``, then returns the final session. Your ``policy_fn``
        receives ``(observation: str, info: dict)`` and returns an action
        string.

        Args:
            deployment_id: Deployment to run on.
            task_id: Task to execute.
            policy_fn: Callable ``(observation, info) -> action``.
            seed: Optional seed for reproducibility.
            max_steps: Safety limit on steps.
            idempotency_key: Unique key for session creation.

        Returns:
            The final :class:`~epsilab.models.EnvironmentSession`.
        """
        session = self.create_environment_session(
            deployment_id,
            task_id=task_id,
            seed=seed,
            idempotency_key=idempotency_key,
        )
        token = session.session_token
        obs = session.observation or ""
        info: Dict[str, Any] = {}

        for _ in range(max_steps):
            action = policy_fn(obs, info)
            result = self.environment_step(
                session.session_id,
                action,
                session_token=token,
            )
            obs = result.observation
            info = result.info
            if result.done:
                break

        return self.get_environment_session(session.session_id)

    # ── Entitlements ─────────────────────────────────────────────────

    def list_entitlements(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List environment entitlements for your account.

        Returns entitlements you've been granted (as a buyer) or
        that you've issued (as a creator).

        Args:
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            List of entitlement records.
        """
        return self._request(
            "GET",
            "/v1/environment-entitlements",
            params={"limit": limit, "offset": offset},
        )

    def grant_entitlement(
        self,
        *,
        grantee_tenant_id: str,
        listing_id: str,
        license_id: str,
        permissions: Optional[List[str]] = None,
        starts_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grant a buyer access to one of your environment listings.

        Creator-only operation.

        Args:
            grantee_tenant_id: Tenant ID of the buyer to grant.
            listing_id: Your listing to grant access to.
            license_id: License under which access is granted.
            permissions: Optional permission list (default: all).
            starts_at: ISO-8601 timestamp when access begins.
            expires_at: ISO-8601 timestamp when access expires.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created entitlement record.
        """
        body: Dict[str, Any] = {
            "grantee_tenant_id": grantee_tenant_id,
            "listing_id": listing_id,
            "license_id": license_id,
        }
        if permissions:
            body["permissions"] = permissions
        if starts_at:
            body["starts_at"] = starts_at
        if expires_at:
            body["expires_at"] = expires_at
        return self._request(
            "POST",
            "/v1/environment-entitlements",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def revoke_entitlement(
        self,
        entitlement_id: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Revoke a previously granted entitlement.

        Creator-only operation.

        Args:
            entitlement_id: The entitlement to revoke.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The updated entitlement record.
        """
        return self._request(
            "POST",
            f"/v1/environment-entitlements/{self._path_segment(entitlement_id)}/revoke",
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    # ── Exports & batches ────────────────────────────────────────────

    def create_environment_export(
        self,
        *,
        deployment_id: str,
        format: str,
        filter_env_type: Optional[str] = None,
        filter_domain: Optional[str] = None,
        filter_task_ids: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start an export job for environment session data.

        Args:
            deployment_id: Deployment to export from.
            format: Export format (``dpo``, ``sft``, ``grpo``, ``jsonl``, etc.).
            filter_env_type: Filter sessions by environment type.
            filter_domain: Filter sessions by domain.
            filter_task_ids: Filter to specific task IDs.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            Export job record with ``export_id`` and ``status``.
        """
        body: Dict[str, Any] = {
            "deployment_id": deployment_id,
            "format": format,
        }
        if filter_env_type:
            body["filter_env_type"] = filter_env_type
        if filter_domain:
            body["filter_domain"] = filter_domain
        if filter_task_ids:
            body["filter_task_ids"] = filter_task_ids
        return self._request(
            "POST",
            "/v1/environment-exports",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def get_environment_export(self, export_id: str) -> Dict[str, Any]:
        """Get the status and details of an export job.

        Args:
            export_id: The export job to inspect.

        Returns:
            Export job record.
        """
        return self._request(
            "GET",
            f"/v1/environment-exports/{self._path_segment(export_id)}",
        )

    def list_environment_exports(
        self,
        *,
        deployment_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List export jobs.

        Args:
            deployment_id: Filter by deployment.
            status: Filter by status (``pending``, ``completed``, ``failed``).
            limit: Max results (1-200, default 50).
            offset: Pagination offset.

        Returns:
            List of export job records.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if deployment_id:
            params["deployment_id"] = deployment_id
        if status:
            params["status"] = status
        return self._request("GET", "/v1/environment-exports", params=params)

    def create_batch(
        self,
        *,
        deployment_id: str,
        name: str,
        task_seed_pairs: List[Dict[str, Any]],
        max_credits: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a batch evaluation across multiple tasks.

        Args:
            deployment_id: Deployment to run on.
            name: Human-readable batch name.
            task_seed_pairs: List of ``{"task_id": ..., "seed": ...}`` dicts.
            max_credits: Optional credit cap.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            Batch record with ``batch_id`` and ``status``.
        """
        body: Dict[str, Any] = {
            "deployment_id": deployment_id,
            "name": name,
            "task_seed_pairs": task_seed_pairs,
        }
        if max_credits is not None:
            body["max_credits"] = max_credits
        return self._request(
            "POST", "/v1/environment-batches", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status and progress.

        Args:
            batch_id: The batch to inspect.
        """
        return self._request(
            "GET",
            f"/v1/environment-batches/{self._path_segment(batch_id)}",
        )

    def list_batches(
        self,
        *,
        deployment_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List batch jobs.

        Args:
            deployment_id: Filter by deployment.
            status: Filter by status.
            limit: Max results (1-200, default 50).
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if deployment_id:
            params["deployment_id"] = deployment_id
        if status:
            params["status"] = status
        return self._request("GET", "/v1/environment-batches", params=params)

    def get_batch_sessions(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get the sessions produced by a batch.

        Args:
            batch_id: The batch to inspect.
        """
        return self._request(
            "GET",
            f"/v1/environment-batches/{self._path_segment(batch_id)}/sessions",
        )

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel a running batch.

        Args:
            batch_id: The batch to cancel.
        """
        return self._request(
            "POST",
            f"/v1/environment-batches/{self._path_segment(batch_id)}/cancel",
        )

    def get_batch_comparison(self, batch_id: str) -> Dict[str, Any]:
        """Get the comparison report for a completed batch.

        Args:
            batch_id: The batch to inspect.
        """
        return self._request(
            "GET",
            f"/v1/environment-batches/{self._path_segment(batch_id)}/comparison",
        )

    # ── Disputes & audit ─────────────────────────────────────────────

    def create_dispute(
        self,
        *,
        session_id: str,
        deployment_id: str,
        release_id: str,
        dispute_type: str,
        summary: str,
        severity: Optional[str] = None,
        evidence_digest: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """File a dispute about a session outcome.

        Args:
            session_id: The session being disputed.
            deployment_id: Deployment the session ran on.
            release_id: Release version involved.
            dispute_type: Type of dispute (e.g. ``reward_error``,
                ``environment_bug``, ``verifier_disagreement``).
            summary: Description of the issue.
            severity: Optional severity (``low``, ``medium``, ``high``,
                ``critical``).
            evidence_digest: SHA-256 digest of supporting evidence.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created dispute record.
        """
        body: Dict[str, Any] = {
            "session_id": session_id,
            "deployment_id": deployment_id,
            "release_id": release_id,
            "dispute_type": dispute_type,
            "summary": summary,
        }
        if severity:
            body["severity"] = severity
        if evidence_digest:
            body["evidence_digest"] = evidence_digest
        return self._request(
            "POST", "/v1/environment-disputes", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def get_dispute(self, dispute_id: str) -> Dict[str, Any]:
        """Get details of a dispute.

        Args:
            dispute_id: The dispute to inspect.
        """
        return self._request(
            "GET",
            f"/v1/environment-disputes/{self._path_segment(dispute_id)}",
        )

    def list_disputes(
        self,
        *,
        release_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List disputes.

        Args:
            release_id: Filter by release.
            status: Filter by status.
            limit: Max results (1-200, default 50).
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        if status:
            params["status"] = status
        return self._request("GET", "/v1/environment-disputes", params=params)

    def get_session_audit(
        self,
        session_id: str,
        *,
        event_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get the audit trail for a session.

        Args:
            session_id: Session to audit.
            event_type: Filter by event type.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type
        return self._request(
            "GET",
            f"/v1/environment-sessions/{self._path_segment(session_id)}/audit",
            params=params,
        )

    # ── Quality & badges ─────────────────────────────────────────────

    def list_quality_reports(
        self,
        *,
        release_id: Optional[str] = None,
        report_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List environment quality reports.

        Args:
            release_id: Filter by release.
            report_type: Filter by report type (e.g. ``qualification``,
                ``regression``, ``benchmark``).
            status: Filter by status (``pending``, ``running``, ``completed``).
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        if report_type:
            params["report_type"] = report_type
        if status:
            params["status"] = status
        return self._request("GET", "/v1/environment-quality-reports", params=params)

    def get_quality_report(self, report_id: str) -> Dict[str, Any]:
        """Get a quality report.

        Args:
            report_id: The report to retrieve.
        """
        return self._request(
            "GET",
            f"/v1/environment-quality-reports/{self._path_segment(report_id)}",
        )

    def get_quality_checks(self, report_id: str) -> List[Dict[str, Any]]:
        """Get individual checks within a quality report.

        Args:
            report_id: The parent report.
        """
        return self._request(
            "GET",
            f"/v1/environment-quality-reports/{self._path_segment(report_id)}/checks",
        )

    def list_quality_badges(
        self,
        *,
        release_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List quality badges awarded to releases.

        Args:
            release_id: Filter by release.
            status: Filter by badge status.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        if status:
            params["status"] = status
        return self._request("GET", "/v1/environment-quality-badges", params=params)

    def list_contamination_findings(
        self,
        *,
        release_id: Optional[str] = None,
        finding_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List contamination findings for releases.

        Args:
            release_id: Filter by release.
            finding_type: Filter by finding type.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        if finding_type:
            params["finding_type"] = finding_type
        return self._request("GET", "/v1/environment-contamination", params=params)

    def list_benchmark_results(
        self,
        *,
        report_id: Optional[str] = None,
        release_id: Optional[str] = None,
        model_tag: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List benchmark results for environment releases.

        Args:
            report_id: Filter by quality report.
            release_id: Filter by release.
            model_tag: Filter by model used in the benchmark.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if report_id:
            params["report_id"] = report_id
        if release_id:
            params["release_id"] = release_id
        if model_tag:
            params["model_tag"] = model_tag
        return self._request("GET", "/v1/environment-benchmark-results", params=params)

    # ── Environment billing ──────────────────────────────────────────

    def list_license_versions(
        self,
        release_id: str,
    ) -> List[Dict[str, Any]]:
        """List license versions for a release.

        Args:
            release_id: The release to query.
        """
        return self._request(
            "GET",
            "/v1/environment-license-versions",
            params={"release_id": release_id},
        )

    def get_license_version(self, license_version_id: str) -> Dict[str, Any]:
        """Get a specific license version.

        Args:
            license_version_id: The license version to retrieve.
        """
        return self._request(
            "GET",
            f"/v1/environment-license-versions/{self._path_segment(license_version_id)}",
        )

    def list_session_charges(
        self,
        *,
        session_id: Optional[str] = None,
        billable_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """List session charges on your account.

        Args:
            session_id: Filter to a specific session.
            billable_only: If True, only billable charges.
        """
        params: Dict[str, Any] = {}
        if session_id:
            params["session_id"] = session_id
        if billable_only is not None:
            params["billable_only"] = str(billable_only).lower()
        return self._request("GET", "/v1/environment-session-charges", params=params)

    def list_charge_adjustments(
        self,
        *,
        charge_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List charge adjustments (credits, refunds, dispute resolutions).

        Args:
            charge_id: Filter to a specific charge.
        """
        params: Dict[str, Any] = {}
        if charge_id:
            params["charge_id"] = charge_id
        return self._request("GET", "/v1/environment-charge-adjustments", params=params)

    def list_invoices(self) -> List[Dict[str, Any]]:
        """List environment invoices."""
        return self._request("GET", "/v1/environment-invoices")

    def get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Get a specific invoice.

        Args:
            invoice_id: The invoice to retrieve.
        """
        return self._request(
            "GET",
            f"/v1/environment-invoices/{self._path_segment(invoice_id)}",
        )

    def get_invoice_line_items(self, invoice_id: str) -> List[Dict[str, Any]]:
        """Get line items for an invoice.

        Args:
            invoice_id: The invoice to inspect.
        """
        return self._request(
            "GET",
            f"/v1/environment-invoices/{self._path_segment(invoice_id)}/line-items",
        )

    def get_charge_summary(
        self,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get an aggregated charge summary for a time range.

        Args:
            since: ISO-8601 start date.
            until: ISO-8601 end date.
        """
        params: Dict[str, Any] = {}
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return self._request("GET", "/v1/environment-charge-summary", params=params)

    # ── Reviews & purchases ──────────────────────────────────────────

    def create_review(
        self,
        *,
        listing_id: str,
        rating: int,
        title: str,
        body: Optional[str] = None,
        usage_hours: Optional[float] = None,
        privacy_cleared: bool = False,
        idempotency_key: Optional[str] = None,
        listing_owner_tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit a review for an environment listing.

        Args:
            listing_id: The listing to review.
            rating: Rating (1-5).
            title: Review title.
            body: Optional review body text.
            usage_hours: Deprecated and ignored. Verified usage is derived by the service.
            privacy_cleared: Confirm the review contains no private session data.
            idempotency_key: Unique key for at-most-once delivery.
            listing_owner_tenant_id: Deprecated, ignored.

        Returns:
            The created review record.
        """
        payload: Dict[str, Any] = {
            "listing_id": listing_id,
            "rating": rating,
            "title": title,
        }
        if body:
            payload["body"] = body
        payload["privacy_cleared"] = privacy_cleared
        headers = {"Idempotency-Key": idempotency_key or self._auto_idem_key()}
        return self._request("POST", "/v1/reviews", json_body=payload, extra_headers=headers)

    def list_reviews(
        self,
        listing_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List reviews for a listing.

        Args:
            listing_id: The listing to get reviews for.
            limit: Max results.
            offset: Pagination offset.
        """
        return self._request(
            "GET",
            f"/v1/reviews/{self._path_segment(listing_id)}",
            params={"limit": limit, "offset": offset},
        )

    def create_purchase(
        self,
        *,
        listing_id: str,
        license_version_id: str,
        payment_reference: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        listing_owner_tenant_id: Optional[str] = None,
        amount_cents: Optional[int] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Purchase access to an environment listing.

        Args:
            listing_id: The listing to purchase.
            license_version_id: License version to purchase under.
            payment_reference: Deprecated and ignored. Payment details are resolved by the service.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The purchase record.
        """
        payload: Dict[str, Any] = {
            "listing_id": listing_id,
            "license_version_id": license_version_id,
        }
        headers = {"Idempotency-Key": idempotency_key or self._auto_idem_key()}
        return self._request("POST", "/v1/purchases", json_body=payload, extra_headers=headers)

    def list_purchases(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List your purchases.

        Args:
            status: Filter by purchase status.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._request("GET", "/v1/purchases", params=params)

    # ── Notifications ────────────────────────────────────────────────

    def list_notifications(
        self,
        *,
        unread_only: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List marketplace notifications.

        Args:
            unread_only: If True, only unread notifications.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if unread_only is not None:
            params["unread_only"] = str(unread_only).lower()
        return self._request("GET", "/v1/notifications", params=params)

    def mark_notification_read(self, notification_id: str) -> Dict[str, Any]:
        """Mark a notification as read.

        Args:
            notification_id: The notification to mark.
        """
        return self._request(
            "POST",
            f"/v1/notifications/{self._path_segment(notification_id)}/read",
        )

    # ── Platform config ────────────────────────────────────────────────

    def get_platform_config(self) -> Dict[str, Any]:
        """Fetch platform configuration (image upload support, API version, etc.)."""
        return self._request("GET", "/v1/platform/config")

    # ── Creator: publishing ─────────────────────────────────────────

    def create_namespace(
        self,
        *,
        slug: str,
        display_name: str,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an environment namespace (creator operation).

        A namespace groups your listings (e.g. ``my-org/env-name``).

        Args:
            slug: URL-safe namespace slug (3-64 chars).
            display_name: Human-readable name.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created namespace record.
        """
        return self._request(
            "POST",
            "/v1/environment-namespaces",
            json_body={"slug": slug, "display_name": display_name},
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def create_listing(
        self,
        *,
        namespace_id: str,
        slug: str,
        title: str,
        summary: Optional[str] = None,
        readme: Optional[str] = None,
        visibility: str = "public",
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentListing":
        """Create an environment listing (creator operation).

        Args:
            namespace_id: Namespace to create the listing in.
            slug: URL-safe listing slug.
            title: Listing title.
            summary: Short description.
            readme: Long-form description (Markdown, up to 32 000 chars).
            visibility: ``private``, ``unlisted``, ``shared``, or ``public``.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created :class:`~epsilab.models.EnvironmentListing`.
        """
        body: Dict[str, Any] = {
            "namespace_id": namespace_id,
            "slug": slug,
            "title": title,
        }
        if summary:
            body["summary"] = summary
        if readme:
            body["readme"] = readme
        if visibility:
            body["visibility"] = visibility
        data = self._request(
            "POST", "/v1/environment-listings", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )
        return EnvironmentListing.from_dict(data)

    def create_application_tool(
        self,
        *,
        namespace_id: str,
        slug: str,
        title: str,
        category: str,
        summary: str = "",
        readme: str = "",
        tags: Optional[List[str]] = None,
        visibility: str = "public",
        idempotency_key: Optional[str] = None,
    ) -> "ApplicationTool":
        """Create an Application Tool listing."""
        body: Dict[str, Any] = {
            "namespace_id": namespace_id,
            "slug": slug,
            "title": title,
            "summary": summary,
            "category": category,
            "tags": tags or [],
            "visibility": visibility,
        }
        if readme:
            body["readme"] = readme
        data = self._request(
            "POST",
            "/v1/application-tools",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )
        return ApplicationTool.from_dict(data)

    def create_application_tool_release(
        self,
        *,
        tool_id: str,
        release_version: str,
        artifact_ref: str,
        artifact_digest: str,
        appsuite_version: str,
        plugin_names: List[str],
        seed_schema_digest: str,
        interface_schema_digest: str,
        license_id: str,
        manifest: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish an immutable Application Tool release."""
        return self._request(
            "POST",
            "/v1/application-tool-releases",
            json_body={
                "tool_id": tool_id,
                "release_version": release_version,
                "artifact_ref": artifact_ref,
                "artifact_digest": artifact_digest,
                "appsuite_version": appsuite_version,
                "plugin_names": plugin_names,
                "seed_schema_digest": seed_schema_digest,
                "interface_schema_digest": interface_schema_digest,
                "license_id": license_id,
                "manifest": manifest,
            },
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def update_application_tool(
        self,
        tool_id: str,
        *,
        expected_revision: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        readme: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> "ApplicationTool":
        """Update mutable Application Tool metadata."""
        body: Dict[str, Any] = {"expected_revision": expected_revision}
        for key, value in {
            "title": title,
            "summary": summary,
            "readme": readme,
            "category": category,
            "tags": tags,
            "visibility": visibility,
        }.items():
            if value is not None:
                body[key] = value
        data = self._request(
            "PATCH",
            f"/v1/application-tools/{self._path_segment(tool_id)}",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )
        return ApplicationTool.from_dict(data)

    def update_listing(
        self,
        listing_id: str,
        *,
        expected_revision: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        readme: Optional[str] = None,
        visibility: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentListing":
        """Update a listing's metadata (creator operation).

        Requires ``expected_revision`` for optimistic concurrency control.

        Args:
            listing_id: The listing to update.
            expected_revision: Current revision number (prevents conflicts).
            title: New title.
            summary: New summary.
            readme: New long-form description (Markdown, up to 32 000 chars).
            visibility: New visibility.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The updated :class:`~epsilab.models.EnvironmentListing`.
        """
        body: Dict[str, Any] = {"expected_revision": expected_revision}
        if title:
            body["title"] = title
        if summary:
            body["summary"] = summary
        if readme is not None:
            body["readme"] = readme
        if visibility:
            body["visibility"] = visibility
        data = self._request(
            "PATCH",
            f"/v1/environment-listings/{self._path_segment(listing_id)}",
            json_body=body,
        )
        return EnvironmentListing.from_dict(data)

    def create_task_pack_release(
        self,
        *,
        namespace_id: str,
        name: str,
        release_version: str,
        artifact_ref: str,
        artifact_digest: str,
        usage_policy: str,
        license_id: str,
        members: Optional[List[Dict[str, Any]]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a task pack release (creator operation).

        A task pack defines the set of tasks available in the environment.

        Args:
            namespace_id: Namespace this release belongs to.
            name: Task pack name.
            release_version: Semantic version.
            artifact_ref: OCI or URI reference to the task pack artifact.
            artifact_digest: SHA-256 content digest.
            usage_policy: Usage policy identifier.
            license_id: License governing this task pack.
            members: Optional list of task member definitions.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created release record.
        """
        body: Dict[str, Any] = {
            "namespace_id": namespace_id,
            "name": name,
            "release_version": release_version,
            "artifact_ref": artifact_ref,
            "artifact_digest": artifact_digest,
            "usage_policy": usage_policy,
            "license_id": license_id,
        }
        if members is not None:
            body["members"] = members
        return self._request(
            "POST", "/v1/task-pack-releases", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def create_verifier_release(
        self,
        *,
        namespace_id: str,
        name: str,
        release_version: str,
        runtime_ref: str,
        runtime_digest: str,
        source_digest: str,
        evidence_schema_digest: str,
        reward_mode: str,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
        nondeterministic: Optional[bool] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a verifier release (creator operation).

        A verifier provides the reward function for the environment.

        Args:
            namespace_id: Namespace this release belongs to.
            name: Verifier name.
            release_version: Semantic version.
            runtime_ref: OCI or URI reference to the verifier runtime.
            runtime_digest: SHA-256 digest of the runtime image.
            source_digest: SHA-256 digest of the verifier source code.
            evidence_schema_digest: Digest of the evidence JSON schema.
            reward_mode: ``binary``, ``continuous``, or ``partial_credit``.
            reward_min: Minimum reward value (for continuous mode).
            reward_max: Maximum reward value (for continuous mode).
            timeout_seconds: Verification timeout.
            nondeterministic: Whether the verifier is nondeterministic.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created release record.
        """
        body: Dict[str, Any] = {
            "namespace_id": namespace_id,
            "name": name,
            "release_version": release_version,
            "runtime_ref": runtime_ref,
            "runtime_digest": runtime_digest,
            "source_digest": source_digest,
            "evidence_schema_digest": evidence_schema_digest,
            "reward_mode": reward_mode,
        }
        if reward_min is not None:
            body["reward_min"] = reward_min
        if reward_max is not None:
            body["reward_max"] = reward_max
        if timeout_seconds is not None:
            body["timeout_seconds"] = timeout_seconds
        if nondeterministic is not None:
            body["nondeterministic"] = nondeterministic
        return self._request(
            "POST", "/v1/verifier-releases", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def create_environment_release(
        self,
        *,
        listing_id: str,
        release_version: str,
        protocol_version: str,
        runtime_ref: str,
        runtime_digest: str,
        task_pack_release_id: str,
        verifier_release_id: str,
        action_schema_digest: str,
        observation_schema_digest: str,
        resource_policy: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> "EnvironmentRelease":
        """Register an environment release (creator operation).

        Bundles a runtime, task pack, and verifier into a deployable,
        content-addressed release.

        Args:
            listing_id: Parent listing.
            release_version: Semantic version.
            protocol_version: Protocol version (e.g. ``0.4.1``).
            runtime_ref: OCI/URI reference to the environment runtime.
            runtime_digest: SHA-256 digest of the runtime image.
            task_pack_release_id: Task pack release to include.
            verifier_release_id: Verifier release to include.
            action_schema_digest: Digest of the action JSON schema.
            observation_schema_digest: Digest of the observation JSON schema.
            resource_policy: Optional resource limits (cpu, memory, gpu).
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created :class:`~epsilab.models.EnvironmentRelease`.
        """
        body: Dict[str, Any] = {
            "listing_id": listing_id,
            "release_version": release_version,
            "protocol_version": protocol_version,
            "runtime_ref": runtime_ref,
            "runtime_digest": runtime_digest,
            "task_pack_release_id": task_pack_release_id,
            "verifier_release_id": verifier_release_id,
            "action_schema_digest": action_schema_digest,
            "observation_schema_digest": observation_schema_digest,
        }
        if resource_policy:
            body["resource_policy"] = resource_policy
        data = self._request(
            "POST", "/v1/environment-releases", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )
        return EnvironmentRelease.from_dict(data)

    # ── Creator: deployments ─────────────────────────────────────────

    def create_deployment(
        self,
        *,
        listing_id: str,
        alias: str,
        environment_release_id: str,
        allowed_split: Optional[str] = None,
        network_policy: Optional[str] = None,
        trace_policy: Optional[str] = None,
        export_policy: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a deployment for a release (creator operation).

        A deployment makes a release available for hosted sessions.

        Args:
            listing_id: Parent listing.
            alias: Human-readable deployment name.
            environment_release_id: Release to deploy.
            allowed_split: Data-split access policy.
            network_policy: Network policy (``isolated``, ``egress_only``).
            trace_policy: Trace collection policy.
            export_policy: Export permission policy.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created deployment record.
        """
        body: Dict[str, Any] = {
            "listing_id": listing_id,
            "alias": alias,
            "environment_release_id": environment_release_id,
        }
        if allowed_split:
            body["allowed_split"] = allowed_split
        if network_policy:
            body["network_policy"] = network_policy
        if trace_policy:
            body["trace_policy"] = trace_policy
        if export_policy:
            body["export_policy"] = export_policy
        return self._request(
            "POST", "/v1/environment-deployments", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    def create_deployment_revision(
        self,
        deployment_id: str,
        *,
        environment_release_id: str,
        allowed_split: Optional[str] = None,
        network_policy: Optional[str] = None,
        trace_policy: Optional[str] = None,
        export_policy: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new revision for an existing deployment (creator operation).

        Revisions allow updating the release or policies without creating
        a new deployment.

        Args:
            deployment_id: Deployment to revise.
            environment_release_id: New release to deploy.
            allowed_split: Updated data-split policy.
            network_policy: Updated network policy.
            trace_policy: Updated trace policy.
            export_policy: Updated export policy.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The updated deployment record.
        """
        body: Dict[str, Any] = {
            "environment_release_id": environment_release_id,
        }
        if allowed_split:
            body["allowed_split"] = allowed_split
        if network_policy:
            body["network_policy"] = network_policy
        if trace_policy:
            body["trace_policy"] = trace_policy
        if export_policy:
            body["export_policy"] = export_policy
        return self._request(
            "POST",
            f"/v1/environment-deployments/{self._path_segment(deployment_id)}/revisions",
            json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    # ── Creator: quality management ──────────────────────────────────

    def create_quality_report(
        self,
        *,
        release_id: str,
        report_type: str,
        deployment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a quality report for a release (creator operation).

        Args:
            release_id: Release to evaluate.
            report_type: One of ``protocol_conformance``, ``startup_cleanup``,
                ``reset_independence``, ``verifier_repeatability``,
                ``adversarial``, ``contamination``, ``benchmark``, or
                ``full_qualification``.
            deployment_id: Optional deployment to test against.
            config: Optional report configuration.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created quality report record.
        """
        if report_type not in _QUALITY_REPORT_TYPES:
            allowed = ", ".join(sorted(_QUALITY_REPORT_TYPES))
            raise ValueError(f"report_type must be one of: {allowed}")
        body: Dict[str, Any] = {
            "release_id": release_id,
            "report_type": report_type,
        }
        if deployment_id:
            body["deployment_id"] = deployment_id
        if config:
            body["config"] = config
        return self._request(
            "POST", "/v1/environment-quality-reports", json_body=body,
            extra_headers={"Idempotency-Key": idempotency_key or self._auto_idem_key()},
        )

    # ── Creator: analytics ───────────────────────────────────────────

    def get_creator_aggregates(
        self,
        *,
        release_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get aggregated usage analytics for your releases (creator operation).

        Returns session counts, unique buyers, reward distributions, and
        pass rates per release.

        Args:
            release_id: Filter to a specific release.
            limit: Max results.
            offset: Pagination offset.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if release_id:
            params["release_id"] = release_id
        return self._request(
            "GET",
            "/v1/environment-creator-aggregates",
            params=params,
        )

    # ── Creator: profiles & moderation ───────────────────────────────

    def create_creator_profile(
        self,
        *,
        display_name: str,
        bio: Optional[str] = None,
        website_url: Optional[str] = None,
        avatar_url: Optional[str] = None,
        contact_email: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create your creator profile (creator operation).

        Args:
            display_name: Public display name.
            bio: Short biography.
            website_url: Link to your website.
            avatar_url: URL to your avatar image.
            contact_email: Contact email for buyers.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created profile record.
        """
        body: Dict[str, Any] = {"display_name": display_name}
        if bio:
            body["bio"] = bio
        if website_url:
            body["website_url"] = website_url
        if avatar_url:
            body["avatar_url"] = avatar_url
        if contact_email:
            body["contact_email"] = contact_email
        headers = {"Idempotency-Key": idempotency_key or self._auto_idem_key()}
        return self._request("POST", "/v1/creator-profiles", json_body=body, extra_headers=headers)

    def get_creator_profile(self) -> Dict[str, Any]:
        """Get your creator profile."""
        return self._request("GET", "/v1/creator-profiles/me")

    def update_creator_profile(
        self,
        *,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
        website_url: Optional[str] = None,
        avatar_url: Optional[str] = None,
        contact_email: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update your creator profile.

        Args:
            display_name: Updated display name.
            bio: Updated biography.
            website_url: Updated website URL.
            avatar_url: Updated avatar URL.
            contact_email: Updated contact email.
            is_public: Whether to make the profile public.

        Returns:
            The updated profile record.
        """
        body: Dict[str, Any] = {}
        if display_name:
            body["display_name"] = display_name
        if bio:
            body["bio"] = bio
        if website_url:
            body["website_url"] = website_url
        if avatar_url:
            body["avatar_url"] = avatar_url
        if contact_email:
            body["contact_email"] = contact_email
        if is_public is not None:
            body["is_public"] = is_public
        return self._request("PATCH", "/v1/creator-profiles/me", json_body=body)

    def request_publish(
        self,
        listing_id: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a listing on the environment hub (creator operation).

        Args:
            listing_id: The listing to publish publicly.
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The publish request record.
        """
        headers = {"Idempotency-Key": idempotency_key or self._auto_idem_key()}
        return self._request(
            "POST",
            "/v1/moderation/publish-request",
            json_body={"listing_id": listing_id},
            extra_headers=headers,
        )

    def create_changelog(
        self,
        *,
        release_id: str,
        version_label: str,
        summary: str,
        body: Optional[str] = None,
        breaking_changes: Optional[bool] = None,
        notify_buyers: Optional[bool] = None,
        listing_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a changelog entry for a release (creator operation).

        Args:
            release_id: The release this changelog describes.
            version_label: Version label (e.g. ``1.2.0``).
            summary: Short summary of changes.
            body: Full changelog body (Markdown).
            breaking_changes: Whether this release has breaking changes.
            notify_buyers: Whether to notify existing buyers.
            listing_id: Parent listing (for cross-referencing).
            idempotency_key: Unique key for at-most-once delivery.

        Returns:
            The created changelog record.
        """
        payload: Dict[str, Any] = {
            "release_id": release_id,
            "version_label": version_label,
            "summary": summary,
        }
        if body:
            payload["body"] = body
        if breaking_changes is not None:
            payload["breaking_changes"] = breaking_changes
        if notify_buyers is not None:
            payload["notify_buyers"] = notify_buyers
        if listing_id:
            payload["listing_id"] = listing_id
        headers = {"Idempotency-Key": idempotency_key or self._auto_idem_key()}
        return self._request("POST", "/v1/changelogs", json_body=payload, extra_headers=headers)

    def list_changelogs(
        self,
        release_id: str,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List changelog entries for a release.

        Args:
            release_id: The release to get changelogs for.
            limit: Max results.
        """
        return self._request(
            "GET",
            f"/v1/changelogs/{self._path_segment(release_id)}",
            params={"limit": limit},
        )

    # ── Creator: settlement ──────────────────────────────────────────

    def get_creator_account(self) -> Dict[str, Any]:
        """Get your creator settlement account.

        Returns balance, payout info, and account status.
        """
        return self._request("GET", "/v1/creator-account")

    def list_royalty_rules(self) -> List[Dict[str, Any]]:
        """List royalty rules configured for your releases."""
        return self._request("GET", "/v1/creator-royalty-rules")

    def list_accruals(
        self,
        *,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List royalty accruals for your releases.

        Args:
            status: Filter by status (``pending``, ``settled``, ``paid``).
        """
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        return self._request("GET", "/v1/creator-accruals", params=params)

    def list_settlement_adjustments(self) -> List[Dict[str, Any]]:
        """List adjustments applied to your settlement account."""
        return self._request("GET", "/v1/creator-adjustments")

    def list_payout_batches(self) -> List[Dict[str, Any]]:
        """List payout batches (scheduled and completed transfers)."""
        return self._request("GET", "/v1/creator-payout-batches")

    def list_creator_statements(self) -> List[Dict[str, Any]]:
        """List period settlement statements."""
        return self._request("GET", "/v1/creator-statements")

    # ── Adapters ─────────────────────────────────────────────────────

    def list_adapters(
        self,
        *,
        protocol_family: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List protocol adapters.

        Adapters provide compatibility shims between different
        environment protocols (e.g. OpenAI Gym → Epsilab Protocol).

        Args:
            protocol_family: Filter by protocol family.
            status: Filter by adapter status.
            limit: Max results.
        """
        params: Dict[str, Any] = {"limit": limit}
        if protocol_family:
            params["protocol_family"] = protocol_family
        if status:
            params["status"] = status
        return self._request("GET", "/v1/adapters", params=params)

    def get_adapter(self, adapter_id: str) -> Dict[str, Any]:
        """Get adapter details.

        Args:
            adapter_id: The adapter to inspect.
        """
        return self._request(
            "GET",
            f"/v1/adapters/{self._path_segment(adapter_id)}",
        )

    def list_adapter_versions(
        self,
        adapter_id: str,
        *,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List versions of an adapter.

        Args:
            adapter_id: The adapter.
            status: Filter by version status.
        """
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        return self._request(
            "GET",
            f"/v1/adapters/{self._path_segment(adapter_id)}/versions",
            params=params,
        )

    def get_adapter_conformance(
        self,
        adapter_id: str,
        *,
        version_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get conformance test results for an adapter.

        Args:
            adapter_id: The adapter.
            version_id: Filter by version.
            status: Filter by result status.
            limit: Max results.
        """
        params: Dict[str, Any] = {"limit": limit}
        if version_id:
            params["version_id"] = version_id
        if status:
            params["status"] = status
        return self._request(
            "GET",
            f"/v1/adapters/{self._path_segment(adapter_id)}/conformance",
            params=params,
        )

    def check_adapter_equivalence(
        self,
        adapter_id: str,
        version_id: str,
    ) -> Dict[str, Any]:
        """Check behavioral equivalence for an adapter version.

        Args:
            adapter_id: The adapter.
            version_id: The version to check.
        """
        return self._request(
            "GET",
            f"/v1/adapters/{self._path_segment(adapter_id)}/equivalence/check",
            params={"version_id": version_id},
        )

    def report_adapter_usage(
        self,
        adapter_id: str,
        *,
        version_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Report adapter usage telemetry.

        Args:
            adapter_id: The adapter being used.
            version_id: Adapter version.
            event_type: Event type (e.g. ``session_start``, ``session_end``).
            metadata: Optional event metadata.
        """
        body: Dict[str, Any] = {
            "version_id": version_id,
            "event_type": event_type,
        }
        if metadata:
            body["metadata"] = metadata
        return self._request(
            "POST",
            "/v1/adapters/usage",
            json_body=body,
        )
