import os
import httpx
from typing import Optional, Dict, Any, List
from dotenv import dotenv_values
from . import models
from .exceptions import AuthError, SubscriptionRequiredError, RateLimitError, ApiError


class EpsilabClient:
    """
    Minimal Python client for the Epsilab Website API (pass-through to Strategy-Engine).

    - Auth via API key sent as X-API-Key
    - Persistent HTTP/2 connection via httpx.Client
    - Targets the Website API base (e.g. https://www.epsilab.ai/api/ext/v1)
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        enable_http2: Optional[bool] = None,
    ) -> None:
        # Read variables from .env without mutating process env
        env = {}
        try:
            env = dotenv_values()
        except Exception:
            env = {}
        # Prefer Website API base (pass-through proxy). Example: https://www.epsilab.ai/api/ext/v1
        self.api_base = (
            (api_base or env.get("EPSILAB_API_BASE") or "https://www.epsilab.ai/api/ext/v1").rstrip("/")
        )
        # Default to 300s, can be overridden by param or env EPSILAB_HTTP_TIMEOUT
        env_timeout = env.get("EPSILAB_HTTP_TIMEOUT")
        self.timeout_seconds = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else int(env_timeout) if env_timeout else 300
        )
        # API key auth (preferred for service usage)
        self._api_key: Optional[str] = api_key or env.get("EPSILAB_API_KEY")
        # Decide HTTP/2 support (auto-detect unless explicitly set)
        use_http2 = False
        if enable_http2 is None:
            try:
                import h2  # type: ignore  # noqa: F401
                use_http2 = True
            except Exception:
                use_http2 = False
        else:
            if enable_http2:
                try:
                    import h2  # type: ignore  # noqa: F401
                    use_http2 = True
                except Exception:
                    use_http2 = False
            else:
                use_http2 = False

        # Persistent client with keep-alive
        self._client = httpx.Client(base_url=self.api_base, timeout=self._make_timeout(), http2=use_http2)
        # Apply auth headers if present
        if self._api_key:
            self._client.headers.update({"X-API-Key": self._api_key})

    def _make_timeout(self) -> httpx.Timeout:
        # Longer read timeout for long-running operations; moderate connects/writes
        return httpx.Timeout(read=float(self.timeout_seconds), connect=30.0, write=30.0, pool=60.0)

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> httpx.Response:
        rel = path.lstrip("/")
        resp = self._client.request(method.upper(), rel, params=params, json=json)
        # Central error mapping
        if resp.status_code >= 400:
            msg = resp.text
            if resp.status_code in (401, 403):
                raise AuthError(msg)
            if resp.status_code == 402:
                raise SubscriptionRequiredError("Pro plan required for API access")
            if resp.status_code == 429:
                retry_after = None
                try:
                    ra = resp.headers.get("Retry-After")
                    retry_after = float(ra) if ra is not None else None
                except Exception:
                    retry_after = None
                raise RateLimitError("Rate limit exceeded", retry_after)
            raise ApiError(f"API error {resp.status_code}: {msg}")
        return resp

    # Context manager support for deterministic close
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # -----------------------------
    # Authentication helpers
    # -----------------------------
    def set_api_key(self, api_key: str) -> None:
        """Use a per-user API key (recommended for headless/service clients)."""
        self._api_key = api_key
        self._client.headers.update({"X-API-Key": api_key})

    # -----------------------------
    # Strategy-Engine API: Live Portfolio
    # -----------------------------

    def get_live_latest(self, *, force: bool = False, tolerance_minutes: Optional[int] = None, return_results_if_fresh: bool = False) -> models.LiveLatest:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params: Dict[str, Any] = {"force": force, "return_results_if_fresh": return_results_if_fresh}
        if tolerance_minutes is not None:
            params["tolerance_minutes"] = int(tolerance_minutes)
        resp = self._request("GET", "api/portfolio/live/latest", params=params)
        return models.parse_live_latest(resp.json())

    def get_live_status(self) -> models.LiveStatus:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        resp = self._request("GET", "api/portfolio/live/status")
        return models.LiveStatus.from_dict(resp.json())

    def get_live_trades(self, *, status: Optional[str] = None, include_positions: bool = False, limit: int = 100) -> List[models.PortfolioTrade]:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params: Dict[str, Any] = {"limit": int(limit), "include_positions": bool(include_positions)}
        if status:
            params["status"] = status
        resp = self._request("GET", "api/portfolio/live/trades", params=params)
        return models.parse_trades(resp.json())

    def get_live_members(self, *, run_id: Optional[str] = None) -> List[models.PortfolioMember]:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params: Dict[str, Any] = {}
        if run_id:
            params["run_id"] = run_id
        resp = self._request("GET", "api/portfolio/live/members", params=params)
        return models.parse_members(resp.json())

    def get_live_equity(self, *, limit: int = 200) -> List[models.EquityPoint]:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params = {"limit": int(limit)}
        resp = self._request("GET", "api/portfolio/live/equity", params=params)
        return models.parse_equity(resp.json())

    def get_portfolio_signals(self, *, run_id: Optional[str] = None, limit: int = 1000, offset: int = 0) -> List[models.PortfolioSignal]:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params: Dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
        if run_id:
            params["run_id"] = run_id
        resp = self._request("GET", "api/portfolio/signals", params=params)
        return models.parse_signals(resp.json())

    def get_portfolio_weights(self, *, run_id: Optional[str] = None, limit: int = 1000, offset: int = 0) -> List[models.PortfolioWeight]:
        if not self._api_key:
            raise RuntimeError("Not authenticated. Set EPSILAB_API_KEY or call set_api_key().")
        params: Dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
        if run_id:
            params["run_id"] = run_id
        resp = self._request("GET", "api/portfolio/weights", params=params)
        return models.parse_weights(resp.json())