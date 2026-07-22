"""Epsilab CLI for managing environments, evaluations, and the marketplace.

Usage::

    epsilab init my-environment       # scaffold a working environment
    epsilab deploy                    # build, push, and register (one command)
    epsilab run owner/environment     # run a hosted environment
    epsilab login
    epsilab whoami
    epsilab env list / search / create / push / deploy / session / batch / export
    epsilab run create / list / status / export / cancel
    epsilab rl envs / session / step / export / stats
    epsilab route <prompt>
    epsilab task create / list
    epsilab namespace create
    epsilab profile show / create
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from ._prompt import confirm, info, is_interactive, select, select_or_create, status, step, text
from .client import EpsilabClient
from .exceptions import ApiError, AuthError, EpsilabError

_cli_logger = logging.getLogger("epsilab.cli")

_CONFIG_DIR = Path.home() / ".epsilab"
_CONFIG_FILE = _CONFIG_DIR / "credentials.json"

_DEFAULT_PROFILE = "default"

_DASHBOARD_URL = "https://app.epsilab.com"
_DOCS_URL = "https://app.epsilab.com"
_MAX_LOCAL_OPENENV_STEPS = 10_000
_LEGACY_PLATFORM_OPENENV_STEPS = 200
_DEFAULT_ENVIRONMENT_RESOURCE_POLICY = {
    "cpu_millis": 1000,
    "memory_bytes": 512 * 1024 * 1024,
    "architecture": "amd64",
    "network_policy": "deny",
    "runtime_interface": "openenv",
}


def _normalize_slug(value: str) -> str:
    """Return a registry-compatible slug derived from user-facing text."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64].rstrip("-")


# ── Helpers ──────────────────────────────────────────────────────────


def _load_config() -> Dict[str, Any]:
    if _CONFIG_FILE.exists():
        return json.loads(_CONFIG_FILE.read_text())
    return {}


def _save_config(config: Dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
    _CONFIG_FILE.chmod(0o600)


def _active_profile() -> str:
    """Return the active profile name.

    Priority: EPSILAB_PROFILE env var > config active_profile > "default".
    """
    env_profile = os.environ.get("EPSILAB_PROFILE")
    if env_profile:
        return env_profile
    config = _load_config()
    return config.get("active_profile", _DEFAULT_PROFILE)


def _resolve_api_key(profile: Optional[str] = None) -> Optional[str]:
    key = os.environ.get("EPSILAB_API_KEY")
    if key:
        return key
    config = _load_config()
    name = profile or _active_profile()
    profiles = config.get("profiles", {})
    if name in profiles:
        return profiles[name].get("api_key")
    # Backwards compatibility: fall back to top-level api_key
    return config.get("api_key")


def _get_client(profile: Optional[str] = None) -> EpsilabClient:
    api_key = _resolve_api_key(profile)
    if not api_key:
        _err(
            "Not authenticated. Run 'epsilab login' or set EPSILAB_API_KEY.\n"
            f"  Create an API key at {_DASHBOARD_URL} (Settings > API Keys)"
        )
    api_base = os.environ.get("EPSILAB_API_BASE")
    return EpsilabClient(api_key=api_key, api_base=api_base)


def _platform_openenv_max_steps(client: EpsilabClient) -> int:
    """Resolve the target Foundation horizon, failing closed for legacy APIs."""
    config = client.get_platform_config()
    try:
        value = config["environment_limits"]["runtimes"]["openenv"]["max_steps"]
    except (KeyError, TypeError):
        return _LEGACY_PLATFORM_OPENENV_STEPS
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Foundation returned an invalid OpenEnv max_steps capability")
    return value


def _environment_horizon_errors(
    tasks: list[dict[str, Any]],
    *,
    max_steps: int,
    source: str,
) -> list[str]:
    errors: list[str] = []
    for index, task in enumerate(tasks):
        task_id = task.get("task_id", f"task #{index + 1}")
        value = task.get("max_steps", 50)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= max_steps:
            errors.append(
                f"{task_id}: max_steps must be an integer in [1, {max_steps}] "
                f"for {source}"
            )
    return errors


def _err(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _friendly_error(exc: Exception) -> str:
    """Return a user-friendly error message from an SDK exception."""
    from .exceptions import ApiError, AuthError, RateLimitError, InsufficientCreditsError

    if isinstance(exc, AuthError):
        return "Authentication failed. Run: epsilab login"
    if isinstance(exc, RateLimitError):
        retry = f" (retry in {exc.retry_after}s)" if exc.retry_after else ""
        return f"Rate limited{retry}. Please wait and try again."
    if isinstance(exc, InsufficientCreditsError):
        return "Insufficient credits. Check your account at epsilab.com."
    if isinstance(exc, ApiError):
        code = exc.status_code
        try:
            import json as _json
            body = _json.loads(str(exc).split(": ", 1)[-1])
            if isinstance(body, dict) and "detail" in body:
                detail = body["detail"]
                if isinstance(detail, str):
                    return detail
                if isinstance(detail, list):
                    return "; ".join(
                        d.get("msg", str(d)) for d in detail if isinstance(d, dict)
                    )
        except Exception:
            pass
        msg = str(exc)
        if code == 404:
            return "Not found. Check the ID and try again."
        if code == 409:
            return "Already exists."
        if code == 422:
            return msg.split(": ", 1)[-1] if ": " in msg else msg
        return f"Request failed ({code}). Use -v for details."
    return str(exc)


def _ok(msg: str) -> None:
    print(msg)


def _warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def _table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    """Print a simple aligned table."""
    if not rows:
        print("  (none)")
        return
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    header = "  ".join(c.upper().ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def _json_out(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


# ── Commands ─────────────────────────────────────────────────────────


def _browser_login() -> Optional[str]:
    """Start a local HTTP server, open the browser for auth, and wait for the
    API key callback.  Returns the API key or None on timeout/failure."""
    import http.server
    import threading
    import urllib.parse

    result: Dict[str, Optional[str]] = {"key": None}
    port = 0  # let the OS pick a free port

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/callback" and "key" in params:
                result["key"] = params["key"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                    b"<h2>Authenticated!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p>"
                    b"</body></html>"
                )
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass  # suppress noisy logs

    server = http.server.HTTPServer(("127.0.0.1", port), CallbackHandler)
    actual_port = server.server_address[1]

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    auth_url = f"{_DASHBOARD_URL}?cli_auth={actual_port}"
    _ok("  Opening browser for authentication...")
    webbrowser.open(auth_url)
    _ok("  Waiting for login (press Ctrl+C to cancel)...\n")

    try:
        thread.join(timeout=120)
    except KeyboardInterrupt:
        print()
        server.server_close()
        return None

    server.server_close()
    return result["key"]


def cmd_login(args: argparse.Namespace) -> None:
    api_key = args.api_key
    if not api_key:
        if is_interactive():
            step("Log in to Epsilab")

            method = select("How would you like to authenticate?", [
                {"value": "browser", "label": "Log in with browser (opens epsilab.com)"},
                {"value": "paste", "label": "Paste an existing API key"},
            ], default="browser")

            if method == "browser":
                api_key = _browser_login()
                if not api_key:
                    _err("Browser login cancelled or timed out.")
            else:
                info(f"Create an API key at {_DASHBOARD_URL} (Settings > API Keys)")
                api_key = text("API key", required=True)
        else:
            _ok(f"Create an API key at {_DASHBOARD_URL} (Settings > API Keys)")
            try:
                api_key = input("Enter your Epsilab API key: ").strip()
            except KeyboardInterrupt:
                print()
                sys.exit(1)
    if not api_key:
        _err("API key cannot be empty.")

    profile_name = getattr(args, "profile", None) or _DEFAULT_PROFILE
    if is_interactive() and not getattr(args, "profile", None):
        label_or_profile = text("Profile name", default=_DEFAULT_PROFILE)
        if label_or_profile:
            profile_name = label_or_profile

    client = EpsilabClient(api_key=api_key)
    try:
        client.get_usage()
        client.close()
        status("API key validated")
    except AuthError:
        client.close()
        _err(
            "Invalid API key. Generate a new one at "
            f"{_DASHBOARD_URL} (Settings > API Keys)"
        )
    except Exception:
        pass

    config = _load_config()
    profiles = config.setdefault("profiles", {})
    label = getattr(args, "label", None) or ""
    profiles[profile_name] = {"api_key": api_key}
    if label:
        profiles[profile_name]["label"] = label
    config["active_profile"] = profile_name
    _save_config(config)
    _ok(f"\nAuthenticated as profile '{profile_name}'.")
    _ok(f"Credentials saved to {_CONFIG_FILE}")


def cmd_logout(args: argparse.Namespace) -> None:
    profile_name = getattr(args, "profile", None) or _active_profile()
    config = _load_config()
    profiles = config.get("profiles", {})
    if profile_name in profiles:
        del profiles[profile_name]
        if config.get("active_profile") == profile_name:
            remaining = list(profiles.keys())
            config["active_profile"] = remaining[0] if remaining else _DEFAULT_PROFILE
        _ok(f"Logged out of profile '{profile_name}'. Credentials removed.")
    elif "api_key" in config:
        config.pop("api_key", None)
        _ok("Logged out. Credentials removed.")
    else:
        _ok(f"No credentials found for profile '{profile_name}'.")
    _save_config(config)


def cmd_whoami(args: argparse.Namespace) -> None:
    profile_name = _active_profile()
    client = _get_client()
    try:
        client.get_usage()
        _ok(f"Authenticated (profile: {profile_name})")
        try:
            profile = client.get_creator_profile()
            _ok(f"Creator profile: {profile.get('display_name', 'unnamed')}")
        except Exception:
            _ok("No creator profile configured.")
            _ok(f"  Set one up at {_DASHBOARD_URL} (Settings > Profile)")
            _ok("  Or run: epsilab profile create \"Your Name\"")
    except AuthError:
        _err(
            "Invalid API key. Generate a new one at "
            f"{_DASHBOARD_URL} (Settings > API Keys)"
        )
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── env commands ─────────────────────────────────────────────────────


def cmd_env_list(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        listings = client.list_environment_listings(limit=args.limit)
        if args.json:
            _json_out([listing.to_dict() for listing in listings])
        else:
            _ok(f"Environments ({len(listings)}):\n")
            if not listings:
                _ok(f"  No environments found. Browse the hub at {_DASHBOARD_URL}\n")
            rows = [
                {
                    "id": listing.listing_id,
                    "slug": listing.slug,
                    "title": listing.title,
                    "visibility": listing.visibility,
                    "status": listing.moderation_state,
                }
                for listing in listings
            ]
            _table(rows, ["id", "slug", "title", "visibility", "status"])
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_search(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.min_quality is not None:
            results = client.search_environments(
                query=args.query,
                domain=args.domain,
                min_quality_score=args.min_quality,
                limit=args.limit,
            )
        else:
            results = [
                listing.to_dict()
                for listing in client.list_environment_listings(
                    query=args.query,
                    domain=args.domain,
                    limit=args.limit,
                )
            ]
        if args.json:
            _json_out(results)
        else:
            _ok(f"Found {len(results)} environment(s):\n")
            if not results:
                _ok(f"  Browse all available environments at {_DASHBOARD_URL}\n")
            rows = [
                {
                    "id": r.get("listing_id", r.get("release_id", "?")),
                    "title": r.get("title", "untitled"),
                    "domain": r.get("domain", "-"),
                    "score": f"{r.get('quality_score', 0):.2f}" if r.get("quality_score") else "-",
                }
                for r in results
            ]
            _table(rows, ["id", "title", "domain", "score"])
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def _interactive_namespace(client: EpsilabClient) -> str:
    """Interactively select or create a namespace."""
    listings = client.list_environment_listings(limit=100)
    seen: dict[str, str] = {}
    for li in listings:
        ns_id = getattr(li, "namespace_id", None)
        ns_slug = getattr(li, "namespace_slug", None) or str(ns_id or "")[:12]
        if ns_id and str(ns_id) not in seen:
            seen[str(ns_id)] = ns_slug

    if seen:
        choices = [{"value": nid, "label": f"{slug} ({nid[:8]}...)"} for nid, slug in seen.items()]

        def _create() -> str:
            slug = text("Namespace slug", required=True)
            display = text("Display name", default=slug)
            ns = client.create_namespace(slug=slug, display_name=display)
            ns_id = ns.get("namespace_id", "?")
            status(f"Created namespace: {ns_id}")
            return str(ns_id)

        return select_or_create("Select namespace", choices, create_label="Create new namespace", create_fn=_create)
    else:
        _ok("No namespaces found. Let's create one.")
        slug = text("Namespace slug", required=True)
        display = text("Display name", default=slug)
        ns = client.create_namespace(slug=slug, display_name=display)
        ns_id = ns.get("namespace_id", "?")
        status(f"Created namespace: {ns_id}")
        return str(ns_id)


def cmd_env_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        namespace_id = args.namespace_id
        slug = args.slug
        title = args.title
        summary = args.summary
        visibility = args.visibility

        if is_interactive() and not all([namespace_id, slug, title]):
            step("Create a new environment listing")

            if not namespace_id:
                namespace_id = _interactive_namespace(client)

            if not slug:
                slug = text("Listing slug", required=True,
                            validator=lambda s: "Must be 3+ chars" if len(s) < 3 else None)

            if not title:
                title = text("Listing title", required=True)

            if not summary:
                summary = text("Short description", default="")

            visibility = select("Visibility", [
                {"value": "public", "label": "Public — visible in marketplace"},
                {"value": "shared", "label": "Shared — visible to collaborators"},
                {"value": "unlisted", "label": "Unlisted — accessible via direct link"},
                {"value": "private", "label": "Private — only you"},
            ], default=visibility or "public")

        if not namespace_id:
            _err(
                "--namespace-id is required. Create one with:\n"
                "  epsilab namespace create <slug>\n"
                f"Or manage namespaces at {_DASHBOARD_URL} (My Environments)"
            )

        listing = client.create_listing(
            namespace_id=namespace_id,
            slug=slug,
            title=title,
            summary=summary,
            visibility=visibility,
        )
        _ok(f"\nCreated listing: {listing.listing_id}")
        _ok(f"  slug: {listing.slug}")
        _ok(f"  title: {listing.title}")
        _ok(f"  visibility: {listing.visibility}")
        _ok("\nNext steps:")
        _ok("  cd your-environment-directory/")
        _ok("  epsilab deploy")
        _ok(f"\nManage this listing at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def _deterministic_idem_key(prefix: str, **fields: object) -> str:
    """Derive a stable idempotency key from content fields."""
    import hashlib
    canonical = json.dumps(fields, sort_keys=True, default=str)
    h = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"{prefix}-{h}"


# ── deploy (top-level) ──────────────────────────────────────────────

_PROJECT_DIR = ".epsilab"
_PROJECT_FILE = "project.json"


def _hash_build_context(directory: Path) -> str:
    """Compute a content hash of the build context for incremental build detection."""
    import hashlib

    h = hashlib.sha256()
    dockerignore = directory / ".dockerignore"
    ignore_patterns: set[str] = set()
    if dockerignore.exists():
        for line in dockerignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignore_patterns.add(line)

    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__", ".epsilab", "node_modules"})
        for fname in sorted(files):
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(directory))
            if any(rel.startswith(p.strip("*/")) for p in ignore_patterns if not p.startswith("!")):
                continue
            try:
                h.update(rel.encode())
                h.update(fpath.read_bytes())
            except (OSError, PermissionError):
                continue
    return h.hexdigest()


def _detect_project_type(directory: Path) -> dict:
    """Auto-detect whether a directory is an environment or an application tool."""
    detected: dict = {"directory": str(directory.resolve()), "type": "unknown"}
    if (directory / "Dockerfile").exists():
        detected["dockerfile"] = True
    if (directory / "server.py").exists():
        detected["server"] = True
    if (directory / "environment.py").exists():
        detected["environment"] = True
    if (directory / "plugin.py").exists():
        detected["plugin"] = True
    if (directory / "api.py").exists():
        detected["api"] = True
    if (directory / "state.py").exists():
        detected["state"] = True
    tasks_path = directory / "tasks.json"
    if tasks_path.exists():
        try:
            tasks = json.loads(tasks_path.read_text())
            if isinstance(tasks, list):
                detected["tasks"] = tasks
                detected["task_count"] = len(tasks)
        except (json.JSONDecodeError, OSError):
            pass
    if (directory / "verifier.py").exists():
        detected["verifier"] = True
    if (directory / "pyproject.toml").exists():
        detected["pyproject"] = True

    if detected.get("plugin") and detected.get("api") and detected.get("state"):
        detected["type"] = "tool"
    elif detected.get("dockerfile"):
        detected["type"] = "environment"
    return detected


def _load_project(directory: Path) -> dict | None:
    """Load .epsilab/project.json if it exists."""
    project_file = directory / _PROJECT_DIR / _PROJECT_FILE
    if project_file.exists():
        try:
            return json.loads(project_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_project(directory: Path, project: dict) -> None:
    """Save project config to .epsilab/project.json."""
    project_dir = directory / _PROJECT_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / _PROJECT_FILE
    project_file.write_text(json.dumps(project, indent=2) + "\n")
    gitignore = project_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# managed by epsilab cli\nproject.json\n.build_hash\n.build_info.json\n")


def _environment_resource_policy(config: dict) -> dict:
    raw = config.get("resource_policy")
    if raw is None:
        return dict(_DEFAULT_ENVIRONMENT_RESOURCE_POLICY)
    if not isinstance(raw, dict):
        raise ValueError("resource_policy must be an object")
    unknown = set(raw) - set(_DEFAULT_ENVIRONMENT_RESOURCE_POLICY)
    if unknown:
        raise ValueError(f"resource_policy has unsupported field(s): {', '.join(sorted(unknown))}")
    policy = {**_DEFAULT_ENVIRONMENT_RESOURCE_POLICY, **raw}
    cpu_millis = policy["cpu_millis"]
    memory_bytes = policy["memory_bytes"]
    if isinstance(cpu_millis, bool) or not isinstance(cpu_millis, int) or not 100 <= cpu_millis <= 64_000:
        raise ValueError("resource_policy.cpu_millis must be an integer between 100 and 64000")
    if (
        isinstance(memory_bytes, bool)
        or not isinstance(memory_bytes, int)
        or not 64 * 1024 * 1024 <= memory_bytes <= 256 * 1024 * 1024 * 1024
    ):
        raise ValueError("resource_policy.memory_bytes must be an integer between 64 MiB and 256 GiB")
    if policy["architecture"] not in {"amd64", "arm64"}:
        raise ValueError("resource_policy.architecture must be amd64 or arm64")
    if policy["network_policy"] not in {"deny", "egress_allowlist"}:
        raise ValueError("resource_policy.network_policy is unsupported")
    if policy["runtime_interface"] not in {"foundation_native", "openenv"}:
        raise ValueError("resource_policy.runtime_interface must be foundation_native or openenv")
    return policy


def _environment_reward_mode(config: dict) -> str:
    reward_mode = config.get("reward_mode", "continuous")
    if reward_mode not in {"binary", "continuous", "partial_credit"}:
        raise ValueError("reward_mode must be binary, continuous, or partial_credit")
    return reward_mode


def _environment_qualification_config(config: dict) -> dict[str, Any] | None:
    """Validate the optional hosted qualification profile in project.json."""
    raw = config.get("qualification")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("qualification must be an object")
    unknown = set(raw) - {"enabled", "task_id", "smoke_actions", "repetitions", "seed"}
    if unknown:
        raise ValueError(
            f"qualification has unsupported field(s): {', '.join(sorted(unknown))}"
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("qualification.enabled must be true or false")
    if not enabled:
        return None
    task_id = raw.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 255:
        raise ValueError("qualification.task_id must contain 1-255 characters")
    actions = raw.get("smoke_actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 50:
        raise ValueError("qualification.smoke_actions must contain 1-50 actions")
    for action in actions:
        if not isinstance(action, (str, dict)) or not action:
            raise ValueError("qualification.smoke_actions must contain strings or objects")
        try:
            encoded = json.dumps(action, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("qualification.smoke_actions must contain finite JSON") from exc
        if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
            raise ValueError("qualification smoke action is too large")
    repetitions = raw.get("repetitions", 3)
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= 20
    ):
        raise ValueError("qualification.repetitions must be an integer from 1 through 20")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("qualification.seed must be an integer from 0 through 2147483647")
    return {
        "task_id": task_id.strip(),
        "smoke_actions": actions,
        "repetitions": repetitions,
        "seed": seed,
    }


def _docker_build_and_upload(
    client: EpsilabClient, directory: Path, image_tag: str,
    *, build_context: Path | None = None, build_args: dict | None = None,
    named_contexts: dict[str, Path] | None = None,
) -> dict:
    """Build a Docker image locally and upload via the API.

    Returns dict with image_ref, registry_digest, content_digest, size_bytes.
    Users never need container registry credentials.
    """
    import subprocess
    import tempfile

    ctx = build_context or directory
    local_tag = f"epsilab-local/{image_tag.split('/')[-1] if '/' in image_tag else image_tag}"

    cmd = ["docker", "build", "--platform", "linux/amd64"]
    for k, v in (build_args or {}).items():
        cmd.extend(["--build-arg", f"{k}={v}"])
    for name, path in sorted((named_contexts or {}).items()):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None:
            raise ValueError(f"Invalid Docker build context name: {name!r}")
        cmd.extend(["--build-context", f"{name}={path}"])
    cmd.extend(["-t", local_tag, "-f", str(directory / "Dockerfile"), str(ctx)])
    _ok("  Building image ...")
    _cli_logger.debug("docker build: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _err(f"Docker build failed:\n{result.stderr}")

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tarball_path = tmp.name

    try:
        _ok("  Saving image ...")
        save = subprocess.run(
            ["docker", "save", "-o", tarball_path, local_tag],
            capture_output=True, text=True, timeout=300,
        )
        if save.returncode != 0:
            _err(f"Docker save failed:\n{save.stderr}")

        size_mb = Path(tarball_path).stat().st_size / (1024 * 1024)
        _ok(f"  Uploading image ({size_mb:.0f} MiB) ...")

        upload_result = client.upload_image(tarball_path, tag=image_tag)

        _ok("  Uploaded successfully")
        return upload_result
    finally:
        Path(tarball_path).unlink(missing_ok=True)
        subprocess.run(["docker", "rmi", local_tag],
                       capture_output=True, timeout=30)


def _content_digest(path: Path) -> str:
    """SHA-256 digest of a file's contents."""
    import hashlib
    h = hashlib.sha256(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _resolve_namespace(client: EpsilabClient, directory: Path, *, auto: bool = False, explicit_id: str | None = None) -> str:
    """Resolve a namespace ID: use explicit > saved > auto-select > create."""
    if explicit_id:
        return explicit_id

    ns_map: dict[str, str] = {}
    namespace_usage: dict[str, int] = {}
    listings = client.list_environment_listings(limit=100)
    for li in listings:
        if li.is_owner and li.namespace_id and li.namespace_id not in ns_map:
            ns_map[li.namespace_id] = li.namespace or li.namespace_id[:12]
        if li.is_owner and li.namespace_id:
            namespace_usage[li.namespace_id] = namespace_usage.get(li.namespace_id, 0) + 1
    try:
        tools = client.list_application_tools(limit=100)
        for t in tools:
            ns_id = getattr(t, "namespace_id", None)
            ns_slug = getattr(t, "namespace", None)
            if ns_id and ns_id not in ns_map:
                ns_map[ns_id] = ns_slug or ns_id[:12]
    except Exception:
        pass

    if len(ns_map) == 1:
        ns_id = next(iter(ns_map))
        _ok(f"  Using namespace: {ns_map[ns_id]}")
        return ns_id

    if len(ns_map) > 1 and auto:
        ns_id = max(ns_map, key=lambda value: (namespace_usage.get(value, 0), ns_map[value]))
        _ok(f"  Using namespace: {ns_map[ns_id]}")
        return ns_id

    if len(ns_map) > 1 and is_interactive() and not auto:
        choices = [{"value": k, "label": v} for k, v in ns_map.items()]
        return select_or_create(
            "Which namespace?", choices,
            create_label="Create new namespace",
            create_fn=lambda: _create_namespace(client, directory, auto=False),
        )

    return _create_namespace(client, directory, auto=auto)


def _create_namespace(client: EpsilabClient, directory: Path, *, auto: bool = False) -> str:
    """Create a namespace, interactively or from the directory name."""
    default_slug = ""
    try:
        profile = client.get_creator_profile()
        default_slug = str(profile.get("display_name", ""))
    except EpsilabError:
        _cli_logger.debug("Creator profile unavailable while selecting a namespace", exc_info=True)
    default_slug = _normalize_slug(default_slug)
    if len(default_slug) < 3:
        default_slug = _normalize_slug(directory.parent.name)
    if len(default_slug) < 3:
        default_slug = _normalize_slug(directory.name)
    if len(default_slug) < 3:
        default_slug = "epsilab"
    if auto or not is_interactive():
        slug = default_slug
        display_name = slug.replace("-", " ").title()
    else:
        slug = text("Namespace slug", default=default_slug)
        display_name = text("Display name", default=slug.replace("-", " ").title())
    try:
        ns = client.create_namespace(slug=slug, display_name=display_name)
        namespace_id = str(ns.get("namespace_id", ns.get("id")))
        _ok(f"  Created namespace: {slug}")
        return namespace_id
    except ApiError as e:
        if e.status_code == 409:
            for li in client.list_environment_listings(limit=100):
                if li.namespace and li.namespace == slug:
                    _ok(f"  Using namespace: {slug}")
                    return li.namespace_id
            try:
                for t in client.list_application_tools(limit=100):
                    ns_slug = getattr(t, "namespace", None)
                    ns_id = getattr(t, "namespace_id", None)
                    if ns_slug == slug and ns_id:
                        _ok(f"  Using namespace: {slug}")
                        return ns_id
            except Exception:
                pass
            _err(f"Namespace '{slug}' already exists but could not be resolved.\n"
                 "  Pass --namespace-id explicitly.")
        raise


def _resolve_listing(
    client: EpsilabClient,
    namespace_id: str,
    slug: str,
    title: str,
    summary: str,
    domain: Optional[str] = None,
    tags: Optional[list[str]] = None,
    visibility: str = "public",
    enforce_visibility: bool = False,
) -> Any:
    """Find an existing listing by slug or create a new one."""
    listings = client.list_environment_listings(limit=100)
    existing = next(
        (li for li in listings if li.slug == slug and li.namespace_id == namespace_id),
        None,
    )
    if existing:
        _ok(f"  Found existing listing: {existing.slug} ({existing.listing_id[:8]}...)")
        updates: dict[str, Any] = {}
        if domain:
            updates["domain"] = domain
        if tags:
            updates["tags"] = tags
        if enforce_visibility and existing.visibility != visibility:
            updates["visibility"] = visibility
        if updates:
            try:
                existing = client.update_listing(
                    existing.listing_id,
                    expected_revision=existing.revision,
                    **updates,
                )
                _ok("  Updated listing: " + ", ".join(sorted(updates)))
            except EpsilabError as exc:
                _warn(f"  Could not update existing listing: {_friendly_error(exc)}")
        return existing
    try:
        listing = client.create_listing(
            namespace_id=namespace_id, slug=slug,
            title=title, summary=summary, visibility=visibility,
            domain=domain, tags=tags,
        )
        _ok(f"  Created listing: {listing.slug}")
        return listing
    except ApiError as e:
        if e.status_code == 409:
            refreshed = client.list_environment_listings(limit=100)
            found = next((li for li in refreshed if li.slug == slug), None)
            if found:
                _ok(f"  Linked to existing listing: {found.slug}")
                return found
        raise


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy an environment or application tool. Auto-detects project type."""
    deploy_start = time.monotonic()
    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        _err(f"Not a directory: {directory}")

    detected = _detect_project_type(directory)
    project_type = detected["type"]
    _cli_logger.debug("Detected project type: %s in %s", project_type, directory)

    if project_type == "unknown":
        files = [f.name for f in directory.iterdir() if f.is_file()]
        if not files:
            _err(
                f"Directory is empty: {directory}\n\n"
                "  To create an RL environment:\n"
                "    epsilab init my-environment\n\n"
                "  An environment needs at minimum:\n"
                "    Dockerfile     — builds the runtime container\n"
                "    tasks.json     — defines the task set\n"
                "    server.py      — OpenEnv-compatible HTTP server\n\n"
                "  An application tool needs at minimum:\n"
                "    plugin.py      — AppPlugin subclass\n"
                "    api.py         — API route handlers\n"
                "    state.py       — deterministic state model"
            )
        has_any = {k for k in ("dockerfile", "plugin", "api", "state", "server", "tasks") if detected.get(k)}
        hints = []
        if detected.get("dockerfile") and not detected.get("tasks"):
            hints.append("  Found Dockerfile but no tasks.json — add a tasks.json with your task definitions")
        if detected.get("plugin") and not detected.get("api"):
            hints.append("  Found plugin.py but no api.py — add API route handlers")
        if detected.get("api") and not detected.get("plugin"):
            hints.append("  Found api.py but no plugin.py — add an AppPlugin subclass")
        hint_str = "\n".join(hints) if hints else ""
        found_str = ", ".join(sorted(has_any)) if has_any else "no recognized project files"
        _err(
            f"Could not detect project type in {directory}\n"
            f"  Found: {found_str}\n\n"
            "  Environment needs:        Dockerfile + tasks.json\n"
            "  Application Tool needs:   plugin.py + api.py + state.py\n"
            + (f"\n{hint_str}\n" if hint_str else "") +
            "\n  Run 'epsilab init my-environment' to scaffold a new environment."
        )

    if project_type == "tool":
        step("Detected application tool")
        status("plugin.py", ok=True)
        status("api.py", ok=detected.get("api", False))
        status("state.py", ok=detected.get("state", False))
        status("environment.py", ok=detected.get("environment", False))
        status("server.py", ok=detected.get("server", False))
    else:
        step("Detected environment")
        status("Dockerfile", ok=True)
        status("server.py", ok=detected.get("server", False))
        tc = detected.get("task_count", 0)
        status(f"tasks.json ({tc} task{'s' if tc != 1 else ''})", ok=bool(detected.get("tasks")))
        status("verifier.py", ok=detected.get("verifier", False))
        if not detected.get("tasks"):
            info("Warning: no tasks.json found — release will have no tasks")

    project = _load_project(directory)
    existing_listing_id = (project or {}).get("listing_id")
    client = _get_client()
    try:
        project_id = (project or {}).get("listing_id") or (project or {}).get("tool_id")
        if project and project_id:
            pid = str(project_id)
            _ok(f"\n  Linked to {project.get('slug', '?')} ({pid[:8]}...)")
        if not project or not project.get("namespace_id"):
            step("Set up project")
            if not project and not is_interactive() and not args.yes:
                _err("No .epsilab/project.json found. Run interactively or use --yes.")

            namespace_id = _resolve_namespace(
                client, directory,
                auto=args.yes,
                explicit_id=(project or {}).get("namespace_id") or getattr(args, "namespace_id", None),
            )

            slug = (project or {}).get("slug") or (project or {}).get("name") or directory.name
            title = (project or {}).get("title") or slug.replace("-", " ").replace("_", " ").title()
            summary = (project or {}).get("summary", "")
            if not project and is_interactive() and not args.yes:
                slug = text("Slug", default=slug)
                title = text("Title", default=title)
                summary = text("Summary", default="")

            if project_type == "tool":
                category = _infer_tool_category(directory)
                if is_interactive() and not args.yes:
                    category = text("Category", default=category)
                tool = _resolve_tool(client, namespace_id, slug, title, summary, category)
                project = {
                    **(project or {}),
                    "type": "tool",
                    "tool_id": tool.tool_id,
                    "namespace_id": namespace_id,
                    "slug": tool.slug,
                    "title": tool.title,
                }
            else:
                listing = _resolve_listing(
                    client, namespace_id, slug, title, summary,
                    domain=(project or {}).get("domain"),
                    tags=(project or {}).get("tags"),
                    visibility=(
                        getattr(args, "visibility", None)
                        or (project or {}).get("visibility")
                        or "public"
                    ),
                    enforce_visibility=bool(getattr(args, "visibility", None)),
                )
                project = {
                    **(project or {}),
                    "type": "environment",
                    "listing_id": listing.listing_id,
                    "namespace_id": namespace_id,
                    "slug": listing.slug,
                    "title": listing.title,
                    "visibility": listing.visibility,
                    "namespace": listing.namespace or (project or {}).get("namespace", ""),
                }
            _save_project(directory, project)
            _ok("  Saved to .epsilab/project.json")

        elif project_type == "environment" and not project.get("listing_id"):
            step("Set up project")
            namespace_id = project["namespace_id"]
            slug = project.get("slug") or project.get("name") or directory.name
            title = project.get("title") or slug.replace("-", " ").replace("_", " ").title()
            listing = _resolve_listing(
                client,
                namespace_id,
                slug,
                title,
                project.get("summary", ""),
                domain=project.get("domain"),
                tags=project.get("tags"),
                visibility=getattr(args, "visibility", None) or project.get("visibility") or "public",
                enforce_visibility=bool(getattr(args, "visibility", None)),
            )
            project.update(
                {
                    "type": "environment",
                    "listing_id": listing.listing_id,
                    "slug": listing.slug,
                    "title": listing.title,
                    "visibility": listing.visibility,
                    "namespace": listing.namespace or project.get("namespace", ""),
                }
            )
            _save_project(directory, project)
            _ok("  Saved listing to .epsilab/project.json")

        if (
            project_type == "environment"
            and existing_listing_id
            and getattr(args, "visibility", None)
        ):
            listing = client.get_environment_listing(existing_listing_id)
            if listing.visibility != args.visibility:
                listing = client.update_listing(
                    existing_listing_id,
                    expected_revision=listing.revision,
                    visibility=args.visibility,
                )
                _ok(f"  Visibility: {listing.visibility}")
            project["visibility"] = listing.visibility
            _save_project(directory, project)

        namespace_id = project["namespace_id"]

        if project.get("type") == "tool" or project_type == "tool":
            _deploy_tool(args, client, directory, detected, project, namespace_id, deploy_start)
        else:
            _deploy_environment(args, client, directory, detected, project, namespace_id, deploy_start)

    except EpsilabError as e:
        _err(_friendly_error(e))
    except ValueError as e:
        _err(str(e))
    finally:
        client.close()


def _infer_tool_category(directory: Path) -> str:
    """Guess a tool category from the directory name."""
    name = directory.name.lower()
    categories = {
        "github": "source-control", "git": "source-control",
        "slack": "messaging", "linear": "project-management",
        "calendar": "scheduling", "gmail": "email", "email": "email",
        "jira": "project-management", "confluence": "documentation",
    }
    for keyword, cat in categories.items():
        if keyword in name:
            return cat
    return "general"


def _resolve_tool(client: EpsilabClient, namespace_id: str, slug: str, title: str, summary: str, category: str) -> Any:
    """Find an existing tool by slug or create a new one."""
    tools = client.list_application_tools(limit=100)
    existing = next(
        (t for t in tools if t.slug == slug and t.namespace_id == namespace_id),
        None,
    )
    if existing:
        _ok(f"  Found existing tool: {existing.slug} ({existing.tool_id[:8]}...)")
        return existing
    try:
        tool = client.create_application_tool(
            namespace_id=namespace_id, slug=slug,
            title=title, summary=summary,
            category=category, visibility="public",
        )
        _ok(f"  Created tool: {tool.slug}")
        return tool
    except ApiError as e:
        if e.status_code == 409:
            refreshed = client.list_application_tools(limit=100)
            found = next((t for t in refreshed if t.slug == slug), None)
            if found:
                _ok(f"  Linked to existing tool: {found.slug}")
                return found
        raise


def _environment_plugin_slugs(
    project: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    uses_appsuite: bool,
) -> list[str]:
    """Return configured or task-derived AppSuite plugin slugs."""
    configured = project.get("plugins")
    if configured is not None:
        if not isinstance(configured, list) or not all(
            isinstance(item, str) for item in configured
        ):
            raise ValueError("project plugins must be a list of plugin slugs")
        return sorted(set(configured))
    if not uses_appsuite:
        return []

    inferred: set[str] = set()
    for task in tasks:
        workspace = task.get("workspace")
        if not isinstance(workspace, dict):
            continue
        for field in ("seeds", "actors"):
            values = workspace.get(field)
            if isinstance(values, dict):
                inferred.update(str(slug) for slug in values)
    return sorted(inferred)


def _resolve_tool_bindings(client: EpsilabClient, plugin_slugs: list[str]) -> list[dict[str, str]]:
    """Resolve published plugin slugs to release binding payloads."""
    if not plugin_slugs:
        return []
    import hashlib as _hashlib

    tools = client.list_application_tools(limit=100)
    bindings: list[dict[str, str]] = []
    empty_configuration_digest = "sha256:" + _hashlib.sha256(b"{}").hexdigest()
    for reference in sorted(set(plugin_slugs)):
        namespace, separator, slug = reference.partition("/")
        if not separator:
            slug = namespace
            namespace = ""
        matches = [
            tool
            for tool in tools
            if tool.slug == slug and (not namespace or tool.namespace == namespace)
        ]
        if not matches:
            _warn(f"  Published tool '{reference}' not found; skipping its catalog binding")
            continue
        if len(matches) > 1:
            owners = ", ".join(sorted(f"{tool.namespace or '?'}/{tool.slug}" for tool in matches))
            raise ValueError(
                f"Tool '{reference}' is ambiguous ({owners}); configure it as <owner>/<slug>."
            )
        tool = matches[0]
        rec_release_id = tool.recommended_release_id
        if not rec_release_id:
            try:
                rec_release_id = client.get_application_tool(tool.tool_id).recommended_release_id
            except EpsilabError:
                _cli_logger.debug("Could not refresh application tool %s", tool.tool_id, exc_info=True)
        if not rec_release_id:
            _warn(f"  Published tool '{reference}' has no active release; skipping its catalog binding")
            continue
        bindings.append(
            {
                "tool_release_id": str(rec_release_id),
                "alias": slug,
                "configuration_digest": empty_configuration_digest,
            }
        )
    return bindings


def _deploy_tool(
    args: argparse.Namespace, client: EpsilabClient,
    directory: Path, detected: dict, project: dict, namespace_id: str,
    deploy_start: float = 0.0,
) -> None:
    """Build and register an Application Tool release."""
    import subprocess

    version = args.version or project.get("version") or "1.0.0"
    if is_interactive() and not args.yes and not args.version:
        version = text("Version", default=version)

    step("Building package")
    pkg_root = directory
    while pkg_root.parent != pkg_root:
        if (pkg_root / "pyproject.toml").exists():
            break
        pkg_root = pkg_root.parent
    else:
        pkg_root = directory

    dist_dir = pkg_root / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(pkg_root)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        _err(f"Package build failed:\n{result.stderr}")

    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        _err("No wheel found after build")
    wheel = wheels[0]
    artifact_digest = _content_digest(wheel)
    status(f"Built: {wheel.name}", ok=True)
    status(f"Digest: {artifact_digest[:20]}...", ok=True)

    plugin_names = _detect_plugin_names(directory)

    step("Registering release")
    source_digest = _content_digest(directory / "plugin.py")
    tool_id = project["tool_id"]
    idem = _deterministic_idem_key(
        "tool-rel", tool_id=tool_id, version=version, digest=artifact_digest,
    )
    rel = client.create_application_tool_release(
        tool_id=tool_id,
        release_version=version,
        artifact_ref=f"object://local/{wheel.name}",
        artifact_digest=artifact_digest,
        appsuite_version="0.2.0",
        plugin_names=plugin_names,
        seed_schema_digest=source_digest,
        interface_schema_digest=source_digest,
        license_id="apache-2.0",
        manifest={"plugins": plugin_names, "version": version},
        idempotency_key=idem,
    )
    release_id = rel.get("release_id", rel.get("id", "?"))
    status(f"Release: {release_id}", ok=True)

    deploy_elapsed = time.monotonic() - deploy_start
    print()
    _ok(f"Deployed {project['slug']} v{version} ({deploy_elapsed:.1f}s)")
    _ok(f"  Release:  {release_id}")
    _ok(f"  Package:  {wheel.name}")
    _ok(f"  Plugins:  {', '.join(plugin_names)}")


def _detect_plugin_names(directory: Path) -> list[str]:
    """Derive slug-format plugin names from the directory."""
    name = _normalize_slug(directory.name)
    return [name] if name else ["plugin"]


def _deploy_environment(
    args: argparse.Namespace, client: EpsilabClient,
    directory: Path, detected: dict, project: dict, namespace_id: str,
    deploy_start: float = 0.0,
) -> None:
    """Build, upload, and register an environment release."""
    listing_id = project["listing_id"]
    resource_policy = _environment_resource_policy(project)
    reward_mode = _environment_reward_mode(project)
    qualification_config = (
        _environment_qualification_config(project) if args.prod else None
    )

    version = args.version or project.get("version") or "1.0.0"
    if is_interactive() and not args.yes and not args.version and not project.get("version"):
        version = text("Version", default=version)

    image_tag = f"{project['slug']}:{version}"
    tasks = detected.get("tasks", [])
    if tasks:
        horizon_errors = _environment_horizon_errors(
            tasks,
            max_steps=_platform_openenv_max_steps(client),
            source="the target Foundation platform",
        )
        if horizon_errors:
            raise ValueError("Environment is not deployable:\n- " + "\n- ".join(horizon_errors))

    step("Building and uploading")

    # Incremental build: skip if the build context hash matches a previous build
    build_cache_dir = directory / ".epsilab"
    build_hash_file = build_cache_dir / ".build_hash"
    build_info_file = build_cache_dir / ".build_info.json"
    skip_build = False
    cached_upload = None

    current_hash = _hash_build_context(directory)
    if (
        not getattr(args, "force", False)
        and build_hash_file.exists()
        and build_info_file.exists()
        and build_hash_file.read_text().strip() == current_hash
    ):
        try:
            cached_upload = json.loads(build_info_file.read_text())
            if cached_upload.get("image_ref"):
                skip_build = True
                _ok(f"  Build context unchanged (hash={current_hash[:12]}...) — skipping build")
                _ok("  Use --force to rebuild")
        except (json.JSONDecodeError, KeyError):
            pass

    build_args = {}
    named_contexts: dict[str, Path] = {}
    shared_dir = directory.parent.parent / "_shared" if (directory.parent.parent / "_shared").exists() else None
    dockerfile = (directory / "Dockerfile").read_text()
    if shared_dir and "ENV_PATH" in dockerfile:
        rel_env_path = str(directory.relative_to(directory.parent.parent))
        build_args["ENV_PATH"] = rel_env_path
        build_context = directory.parent.parent
    else:
        build_context = None

    if "--from=appsuite" in dockerfile.lower():
        appsuite_root = Path(
            os.environ.get(
                "EPSILAB_APPSUITE_ROOT",
                str((build_context or directory).parent / "AppSuite"),
            )
        ).expanduser().resolve()
        if not (
            (appsuite_root / "pyproject.toml").is_file()
            and (appsuite_root / "src" / "epsilab_apps" / "__init__.py").is_file()
        ):
            raise ValueError(
                "This environment requires an AppSuite checkout. Set "
                "EPSILAB_APPSUITE_ROOT to its directory."
            )
        named_contexts["appsuite"] = appsuite_root

    if skip_build and cached_upload:
        upload = cached_upload
    else:
        upload = _docker_build_and_upload(
            client, directory, image_tag,
            build_context=build_context,
            build_args=build_args,
            named_contexts=named_contexts,
        )
        # Cache build info for incremental builds
        try:
            build_cache_dir.mkdir(parents=True, exist_ok=True)
            build_hash_file.write_text(current_hash)
            build_info_file.write_text(json.dumps(upload, default=str))
        except OSError:
            pass

    oci_ref = upload["image_ref"]
    if "@sha256:" in oci_ref:
        digest = "sha256:" + oci_ref.split("@sha256:")[-1]
    else:
        digest = upload.get("content_digest", upload.get("image_ref", "").split("@")[-1])
    status(f"Image: {project['slug']}:{version}", ok=True)
    if digest:
        status(f"Digest: {digest[:20]}...", ok=True)

    step("Registering release")

    if tasks:
        _ok(f"  Creating {len(tasks)} tasks ...")
        for t in tasks:
            task_body = {
                "task_id": t["task_id"],
                "domain": t.get("domain", project.get("domain", project["slug"])),
                "capability": t.get("capability", project["slug"]),
                "prompt": t.get("prompt", t.get("title", t["task_id"])),
                "verification": t.get("verification", "judge"),
                "difficulty": t.get("difficulty", "medium"),
                "max_steps": t.get("max_steps", 50),
            }
            if t.get("ground_truth") or t.get("expected_fix") or t.get("expected_answer"):
                task_body["ground_truth"] = (
                    t.get("ground_truth")
                    or t.get("expected_fix")
                    or t.get("expected_answer", "")
                )
            try:
                client.create_task(task_body)
            except ApiError as e:
                if e.status_code != 409:
                    raise
        status(f"{len(tasks)} tasks registered", ok=True)

    members = []
    for t in tasks:
        members.append({
            "task_id": t["task_id"],
            "lineage_group_id": t.get("lineage_group_id", t["task_id"]),
            "split": t.get("split", "train"),
        })

    source_digest = _content_digest(directory / "verifier.py") if detected.get("verifier") else digest

    tp_idem = _deterministic_idem_key("tp", namespace_id=namespace_id, slug=project["slug"], version=version, digest=digest)
    try:
        tp = client.create_task_pack_release(
            namespace_id=namespace_id,
            name=f"{project['slug']}-tasks",
            release_version=version,
            artifact_ref=oci_ref,
            artifact_digest=digest,
            usage_policy="training",
            license_id="apache-2.0",
            members=members,
            idempotency_key=tp_idem,
        )
    except ApiError as e:
        if e.status_code != 409:
            raise
        _err(
            f"Release version {version} already exists with different task-pack content. "
            "Choose a new version with --version."
        )
    tp_release_id = str(tp.get("release_id", tp.get("id", "")))
    status(f"Task pack: {len(members)} tasks", ok=True)

    ver_idem = _deterministic_idem_key("ver", namespace_id=namespace_id, slug=project["slug"], version=version, digest=digest)
    try:
        ver = client.create_verifier_release(
            namespace_id=namespace_id,
            name=f"{project['slug']}-verifier",
            release_version=version,
            runtime_ref=oci_ref,
            runtime_digest=digest,
            source_digest=source_digest,
            evidence_schema_digest=source_digest,
            reward_mode=reward_mode,
            idempotency_key=ver_idem,
        )
    except ApiError as e:
        if e.status_code != 409:
            raise
        _err(
            f"Release version {version} already exists with different verifier content. "
            "Choose a new version with --version."
        )
    ver_release_id = str(ver.get("release_id", ver.get("id", "")))
    status("Verifier registered", ok=True)

    plugin_slugs = _environment_plugin_slugs(
        project,
        tasks,
        uses_appsuite="appsuite" in named_contexts,
    )
    if plugin_slugs:
        project["plugins"] = plugin_slugs
    tool_bindings = _resolve_tool_bindings(client, plugin_slugs) if plugin_slugs else []
    if tool_bindings:
        status(f"Resolved {len(tool_bindings)} tool binding(s)", ok=True)

    env_idem = _deterministic_idem_key("env", listing_id=listing_id, version=version, digest=digest)
    try:
        release = client.create_environment_release(
            listing_id=listing_id,
            release_version=version,
            protocol_version="0.4.1",
            runtime_ref=oci_ref,
            runtime_digest=digest,
            task_pack_release_id=tp_release_id,
            verifier_release_id=ver_release_id,
            action_schema_digest=source_digest,
            observation_schema_digest=source_digest,
            resource_policy=resource_policy,
            application_tools=tool_bindings or None,
            idempotency_key=env_idem,
        )
    except ApiError as e:
        if e.status_code != 409:
            raise
        _err(
            f"Environment version {version} already exists with different content. "
            "Choose a new version with --version."
        )
    if release is not None:
        release_id = getattr(release, "release_id", None) or (release.get("release_id") if isinstance(release, dict) else "") or ""
    else:
        release_id = ""
    status(f"Release: {release_id}", ok=True)

    quality_report_id = ""
    if qualification_config is not None:
        step("Checking hosted compatibility")
        try:
            report = client.create_quality_report(
                release_id=str(release_id),
                report_type="hosted_execution",
                config=qualification_config,
                idempotency_key=_deterministic_idem_key(
                    "hosted-quality",
                    release_id=release_id,
                    config=qualification_config,
                ),
            )
            quality_report_id = str(report.get("report_id", ""))
            status("Hosted qualification queued", ok=True)
        except EpsilabError as exc:
            _warn(
                "The release was published, but hosted qualification could not be queued: "
                f"{_friendly_error(exc)}\n"
                f"  Retry with: epsilab env qualify {release_id}"
            )

    deploy_id = ""
    if args.prod:
        step("Deploying")
        deploy_alias = f"prod-v{version.replace('.', '-')}"
        try:
            deploy_result = client.create_deployment(
                listing_id=listing_id,
                environment_release_id=str(release_id),
                alias=deploy_alias,
                allowed_split="train",
                export_policy="training_allowed",
            )
        except ApiError as e:
            if e.status_code != 409:
                raise
            refreshed = client.get_environment_listing(listing_id)
            if not refreshed.deployment_id:
                raise
            deploy_result = {"deployment_id": refreshed.deployment_id}
        deploy_id = str(deploy_result.get("deployment_id", deploy_result.get("id", "")))
        status(f"Deployed: {deploy_id}", ok=True)

    project.update(
        {
            "version": version,
            "environment_release_id": release_id,
            "deployment_id": deploy_id or project.get("deployment_id", ""),
            "quality_report_id": quality_report_id or project.get("quality_report_id", ""),
        }
    )
    try:
        refreshed_listing = client.get_environment_listing(listing_id)
        if refreshed_listing.namespace:
            project["namespace"] = refreshed_listing.namespace
        if refreshed_listing.deployment_id:
            project["deployment_id"] = refreshed_listing.deployment_id
    except EpsilabError:
        _cli_logger.debug("Could not refresh listing metadata after deploy", exc_info=True)
    _save_project(directory, project)

    deploy_elapsed = time.monotonic() - deploy_start
    print()
    _ok(f"Deployed {project['slug']}@{version} ({deploy_elapsed:.1f}s)")
    _ok(f"  Release:  {release_id}")
    _ok(f"  Tasks:    {len(members)}")
    if args.prod:
        _ok("  Status:   Published")
        if quality_report_id:
            _ok("  Hosting:  Qualification queued")
            _ok(f"  Check:    epsilab env status {release_id}")
        owner = project.get("namespace") or project.get("namespace_slug")
        if owner:
            label = "Run after qualification" if quality_report_id else "Run it"
            _ok(f"  {label}: epsilab run {owner}/{project['slug']}")
        else:
            label = "Run after qualification" if quality_report_id else "Run it"
            _ok(f"  {label}: epsilab run <owner>/{project['slug']}")
    else:
        _ok("\n  Run with --prod to deploy for hosted execution.")


def _find_manifest(args: argparse.Namespace) -> tuple[dict, Path | None]:
    """Locate and load the manifest, with interactive fallback."""
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            _err(f"Manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text()), manifest_path

    default_path = Path("epsilab.json")
    if default_path.exists():
        if is_interactive():
            if confirm(f"Found {default_path} — use it?"):
                return json.loads(default_path.read_text()), default_path
        else:
            return json.loads(default_path.read_text()), default_path

    if is_interactive():
        path_str = text("Path to manifest file", default="epsilab.json")
        p = Path(path_str)
        if p.exists():
            return json.loads(p.read_text()), p
        _err(f"Manifest not found: {p}")

    return {}, None


def _interactive_listing(client: EpsilabClient) -> str:
    """Interactively select from existing listings."""
    listings = client.list_environment_listings(limit=100)
    if not listings:
        _err("No listings found. Create one first:\n  epsilab env create")
    choices = [
        {"value": li.listing_id, "label": f"{li.slug} — {li.title} ({li.visibility})"}
        for li in listings
    ]
    return select("Select listing", choices)


def cmd_env_push(args: argparse.Namespace) -> None:
    """Register a new environment release from a manifest file or CLI args."""
    import re as _re

    client = _get_client()
    try:
        manifest, manifest_path = _find_manifest(args)

        listing_id = args.listing_id or manifest.get("listing_id")
        if not listing_id and is_interactive():
            listing_id = _interactive_listing(client)
        if not listing_id:
            _err("--listing-id is required (or set listing_id in manifest).")

        version = args.version or manifest.get("release_version")
        if not version and is_interactive():
            version = text("Release version", default="0.1.0", required=True,
                           validator=lambda v: None if _re.match(r"^\d+\.\d+\.\d+", v) else "Must be semver (e.g. 0.1.0)")
        if not version:
            _err("--version is required (or set release_version in manifest).")

        for cli_arg, label in [
            (args.runtime_digest, "--runtime-digest"),
            (args.task_pack_digest, "--task-pack-digest"),
            (args.verifier_digest, "--verifier-digest"),
        ]:
            if cli_arg and not _re.match(r"^sha256:[0-9a-f]{64}$", cli_arg):
                _err(f"Invalid {label}: must be sha256:<64 hex chars>, got: {cli_arg}")

        env_config = manifest.get("environment", {})
        tp_config = manifest.get("task_pack", {})
        ver_config = manifest.get("verifier", {})
        env_config = manifest.get("environment", {})

        namespace_id = args.namespace_id or manifest.get("namespace_id")
        if not namespace_id:
            _err("--namespace-id is required (or set namespace_id in manifest).")

        _ok(f"Pushing environment release v{version} to {listing_id}...")

        tp_name = tp_config.get("name", f"{listing_id}-tasks")
        tp_artifact_ref = args.task_pack_ref or tp_config.get("artifact_ref", "")
        tp_artifact_digest = args.task_pack_digest or tp_config.get("artifact_digest", "")
        tp_usage_policy = tp_config.get("usage_policy", "training")
        tp_license_id = args.license or tp_config.get("license_id", "apache-2.0")
        tp_members = tp_config.get("members", [])

        tp_idem = _deterministic_idem_key(
            "tp", namespace_id=namespace_id, name=tp_name,
            version=version, digest=tp_artifact_digest,
            members=tp_members,
        )
        tp = client.create_task_pack_release(
            namespace_id=namespace_id,
            name=tp_name,
            release_version=version,
            artifact_ref=tp_artifact_ref,
            artifact_digest=tp_artifact_digest,
            usage_policy=tp_usage_policy,
            license_id=tp_license_id,
            members=tp_members,
            idempotency_key=tp_idem,
        )
        _ok(f"  Task pack registered: {tp.get('release_id', tp.get('id', '?'))}")

        ver_name = ver_config.get("name", f"{listing_id}-verifier")
        ver_runtime_ref = args.verifier_ref or ver_config.get("runtime_ref", "")
        ver_runtime_digest = args.verifier_digest or ver_config.get("runtime_digest", "")
        ver_source_digest = ver_config.get("source_digest", "")
        ver_evidence_schema_digest = ver_config.get("evidence_schema_digest", "")
        ver_reward_mode = _environment_reward_mode(ver_config)

        ver_idem = _deterministic_idem_key(
            "ver", namespace_id=namespace_id, name=ver_name,
            version=version, digest=ver_runtime_digest,
        )
        ver = client.create_verifier_release(
            namespace_id=namespace_id,
            name=ver_name,
            release_version=version,
            runtime_ref=ver_runtime_ref,
            runtime_digest=ver_runtime_digest,
            source_digest=ver_source_digest,
            evidence_schema_digest=ver_evidence_schema_digest,
            reward_mode=ver_reward_mode,
            idempotency_key=ver_idem,
        )
        _ok(f"  Verifier registered: {ver.get('release_id', ver.get('id', '?'))}")

        env_runtime_ref = args.runtime_ref or env_config.get("runtime_ref", "")
        env_runtime_digest = args.runtime_digest or env_config.get("runtime_digest", "")
        tp_release_id = str(tp.get("release_id", tp.get("id", "")))
        ver_release_id = str(ver.get("release_id", ver.get("id", "")))
        env_action_schema = env_config.get("action_schema_digest", "")
        env_obs_schema = env_config.get("observation_schema_digest", "")
        env_resource_policy = _environment_resource_policy(env_config)

        push_plugin_slugs = manifest.get("plugins", env_config.get("plugins", []))
        push_tool_bindings = _resolve_tool_bindings(client, push_plugin_slugs) if push_plugin_slugs else []
        if push_tool_bindings:
            _ok(f"  Resolved {len(push_tool_bindings)} tool binding(s)")

        env_idem = _deterministic_idem_key(
            "env", listing_id=listing_id, version=version,
            runtime_digest=env_runtime_digest,
            tp_release_id=tp_release_id, ver_release_id=ver_release_id,
        )
        release = client.create_environment_release(
            listing_id=listing_id,
            release_version=version,
            protocol_version=env_config.get("protocol_version", args.protocol_version or "0.4.1"),
            runtime_ref=env_runtime_ref,
            runtime_digest=env_runtime_digest,
            task_pack_release_id=tp_release_id,
            verifier_release_id=ver_release_id,
            action_schema_digest=env_action_schema,
            observation_schema_digest=env_obs_schema,
            resource_policy=env_resource_policy,
            application_tools=push_tool_bindings or None,
            idempotency_key=env_idem,
        )
        _ok(f"  Environment release registered: {release.release_id}")
        _ok(f"\nPushed v{version} successfully.")
        _ok(f"Release ID: {release.release_id}")
        _ok(f"Status: {release.status}")

        if release.status == "qualified":
            _ok("\nRelease is live according to the listing's visibility setting.")
        _ok(f"\nView release details at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_deploy(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        release_id = args.release_id
        listing_id = args.listing_id
        alias = args.alias

        if is_interactive() and not args.revision and not all([release_id, listing_id]):
            step("Deploy an environment release")

            if not listing_id:
                listing_id = _interactive_listing(client)

            if not release_id:
                release_id = text("Release ID to deploy", required=True)

            if not alias:
                alias = text("Deployment alias", default="production")

        if args.revision:
            dep = client.create_deployment_revision(
                args.deployment_id,
                environment_release_id=release_id,
                export_policy=args.export_policy,
            )
            _ok(f"Deployment revised: {dep.get('deployment_id', '?')}")
        else:
            if not listing_id:
                _err("--listing-id is required for new deployments.")
            dep = client.create_deployment(
                listing_id=listing_id,
                alias=alias or "production",
                environment_release_id=release_id,
                export_policy=args.export_policy,
            )
            _ok(f"\nDeployed: {dep.get('deployment_id', '?')}")
        _ok(f"  alias: {dep.get('alias', '-')}")
        _ok(f"  release: {release_id}")
        _ok(f"\nManage deployments at {_DASHBOARD_URL} (My Environments)")
        if args.json:
            _json_out(dep)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_grant(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        ent = client.grant_entitlement(
            grantee_tenant_id=args.tenant_id,
            listing_id=args.listing_id,
            license_id=args.license or "apache-2.0",
            expires_at=args.expires_at,
        )
        _ok(f"Access shared: {ent.get('entitlement_id', '?')}")
        _ok(f"  organization: {args.tenant_id}")
        _ok(f"  environment:  {args.listing_id}")
        env_title = ent.get("environment_title") or ent.get("environment_slug") or ""
        if env_title:
            _ok(f"  ({env_title})")
        _ok(f"\nManage shared access at {_DASHBOARD_URL} (Environments > Shared)")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_revoke(args: argparse.Namespace) -> None:
    """Revoke shared access to an environment."""
    client = _get_client()
    try:
        ent = client.revoke_entitlement(args.entitlement_id)
        _ok(f"Access revoked: {args.entitlement_id}")
        env_title = ent.get("environment_title") or ent.get("environment_slug") or ""
        if env_title:
            _ok(f"  environment: {env_title}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def _resolve_listing_for_update(client: EpsilabClient, target: str) -> Any:
    """Resolve a creator listing by ID or exact owner/name."""
    if "/" not in target:
        return client.get_environment_listing(target)
    if target.count("/") != 1:
        raise ValueError("Environment must be a listing ID or <owner>/<name>.")
    owner, slug = (part.strip().lower() for part in target.split("/", 1))
    if not owner or not slug:
        raise ValueError("Environment must be a listing ID or <owner>/<name>.")
    matches = [
        listing
        for listing in client.list_environment_listings(limit=100)
        if listing.slug.lower() == slug
        and (listing.namespace or "").lower() == owner
        and listing.is_owner
    ]
    if not matches:
        raise ValueError(f"Owned environment '{target}' was not found.")
    if len(matches) > 1:
        raise ValueError(f"Owned environment '{target}' is ambiguous; use its listing ID.")
    return matches[0]


def cmd_env_visibility(args: argparse.Namespace) -> None:
    """Change how a creator listing is exposed on the hub."""
    client = _get_client()
    try:
        listing = _resolve_listing_for_update(client, args.target)
        if listing.visibility == args.visibility:
            updated = listing
        else:
            client.update_listing(
                listing.listing_id,
                expected_revision=listing.revision,
                visibility=args.visibility,
            )
            updated = client.get_environment_listing(listing.listing_id)
        if args.json:
            _json_out(updated.to_dict())
        else:
            _ok(f"Visibility updated: {updated.namespace or '?'}/{updated.slug}")
            _ok(f"  visibility: {updated.visibility}")
            if updated.visibility == "private":
                _ok("  The environment is now visible only to your organization.")
    except (EpsilabError, ValueError) as exc:
        _err(_friendly_error(exc))
    finally:
        client.close()


def cmd_env_shared(args: argparse.Namespace) -> None:
    """List shared environments."""
    client = _get_client()
    try:
        entitlements = client.list_entitlements(limit=100)
        active = [e for e in entitlements if e.get("status") == "active"]
        if not active:
            _ok("No shared environments.")
            return
        _ok(f"Shared environments ({len(active)} active):\n")
        for e in active:
            env_name = e.get("environment_namespace", "?") + "/" + e.get("environment_slug", "?")
            env_title = e.get("environment_title", "")
            access_role = e.get("access_role")
            if access_role == "owner":
                counterpart_label = "shared with"
                counterpart = e.get("grantee_name") or str(e.get("grantee_tenant_id", "?"))[:12]
            else:
                counterpart_label = "shared by"
                counterpart = e.get("owner_name") or str(e.get("owner_tenant_id", "?"))[:12]
            perms = ", ".join(e.get("permissions", []))
            _ok(f"  {env_name}")
            if env_title:
                _ok(f"    {env_title}")
            _ok(f"    {counterpart_label}: {counterpart}  |  permissions: {perms}")
            _ok(f"    id: {e.get('entitlement_id', '?')}")
            _ok("")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_status(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        release = client.get_environment_release(args.release_id)
        _ok(f"Release: {release.release_id}")
        _ok(f"  version: {release.release_version}")
        _ok(f"  protocol: {release.protocol_version}")
        _ok(f"  publication: {release.status}")
        if release.content_digest:
            _ok(f"  digest: {release.content_digest}")
        if release.created_at:
            _ok(f"  created: {release.created_at}")

        reports = client.list_quality_reports(
            release_id=args.release_id,
            report_type="hosted_execution",
            limit=1,
        )
        if reports:
            report = reports[0]
            hosted_status = report.get("hosted_qualification_status")
            if not hosted_status:
                if report.get("status") in {"queued", "running", "failed"}:
                    hosted_status = report["status"]
                elif report.get("status") == "completed" and report.get("fail_count") == 0:
                    hosted_status = "qualified"
                else:
                    hosted_status = "failed"
            _ok(f"  hosted execution: {hosted_status}")
            if report.get("hosted_qualification_expires_at"):
                _ok(f"  qualification expires: {report['hosted_qualification_expires_at']}")
            if hosted_status == "failed" and report.get("error_code"):
                _ok(f"  qualification error: {report['error_code']}")
        else:
            _ok("  hosted execution: not requested")

        badges = client.list_quality_badges(release_id=args.release_id, limit=10)
        if badges:
            _ok("\n  Quality badges:")
            for b in badges:
                _ok(f"    - {b.get('badge_type', '?')} ({b.get('status', '?')})")
        else:
            _ok("\n  No quality badges yet. Run a qualification report:")
            _ok(f"    epsilab env qualify {args.release_id}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_qualify(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        config: dict[str, Any] | None = None
        if args.report_type == "hosted_execution":
            project = _load_project(Path.cwd()) or {}
            config = _environment_qualification_config(project)
            if args.action:
                config = dict(config or {})
                config["smoke_actions"] = [
                    json.loads(_encode_environment_action(value, action_type=args.action_type))
                    for value in args.action
                ]
            if args.task:
                config = dict(config or {})
                config["task_id"] = args.task
            if args.repetitions is not None:
                config = dict(config or {})
                config["repetitions"] = args.repetitions
            if args.seed is not None:
                config = dict(config or {})
                config["seed"] = args.seed
            if not config or not config.get("task_id") or not config.get("smoke_actions"):
                raise ValueError(
                    "Hosted qualification needs a task and terminal smoke action. "
                    "Run from a project created by 'epsilab init', or pass --task and --action."
                )
            config = _environment_qualification_config({"qualification": config})

        report = client.create_quality_report(
            release_id=args.release_id,
            report_type=args.report_type,
            config=config,
            idempotency_key=_deterministic_idem_key(
                "hosted-quality" if args.report_type == "hosted_execution" else "quality",
                release_id=args.release_id,
                report_type=args.report_type,
                config=config or {},
            ),
        )
        _ok(f"Quality report started: {report.get('report_id', '?')}")
        _ok(f"  type: {args.report_type}")
        _ok(f"  status: {report.get('status', 'pending')}")
        _ok("\nCheck progress with:")
        _ok(f"  epsilab env status {args.release_id}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    except ValueError as e:
        _err(str(e))
    finally:
        client.close()


# ── namespace commands ───────────────────────────────────────────────


def cmd_task_create(args: argparse.Namespace) -> None:
    """Create a single task from CLI args or a JSON file."""
    client = _get_client()
    try:
        if args.file:
            task_data = json.loads(Path(args.file).read_text())
            if isinstance(task_data, list):
                result = client.upload_custom_tasks(task_data)
                _ok(f"Uploaded {len(task_data)} tasks")
                _ok(f"  created: {result.created}  skipped: {result.skipped}")
                return
        elif is_interactive() and not args.task_id:
            step("Create a new task")
            task_id = text("Task ID", required=True)
            prompt_text = text("Task prompt", required=True)
            domain = text("Domain", default="general")
            capability = text("Capability", default="general")
            difficulty = select("Difficulty", [
                {"value": "easy", "label": "Easy"},
                {"value": "medium", "label": "Medium"},
                {"value": "hard", "label": "Hard"},
            ], default="medium")
            expected = text("Expected answer (optional)", default="")
            task_data = {
                "task_id": task_id,
                "prompt": prompt_text,
                "domain": domain,
                "capability": capability,
                "difficulty": difficulty,
            }
            if expected:
                task_data["expected_answer"] = expected
        else:
            task_data = {
                "task_id": args.task_id,
                "prompt": args.prompt,
                "domain": args.domain or "general",
                "capability": args.capability or "general",
                "difficulty": args.difficulty or "medium",
            }
            if args.expected_answer:
                task_data["expected_answer"] = args.expected_answer

        result = client.create_task(task_data)
        _ok(f"Task created: {result.get('task_id', '?')}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_task_list(args: argparse.Namespace) -> None:
    """List tasks visible to the current tenant."""
    client = _get_client()
    try:
        result = client.list_tasks(
            domain=args.domain,
            limit=args.limit,
            offset=args.offset,
        )
        tasks = result.get("tasks", result) if isinstance(result, dict) else result
        if not tasks:
            _ok("No tasks found.")
            return
        for t in tasks:
            tid = t.get("task_id", "?")
            domain = t.get("domain", "")
            diff = t.get("difficulty", "")
            prompt = (t.get("prompt", "") or "")[:80]
            _ok(f"  {tid}  [{domain}/{diff}]  {prompt}")
        _ok(f"\n{len(tasks)} task(s)")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_namespace_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        slug = args.slug
        display_name = args.display_name

        if is_interactive() and not slug:
            step("Create a new namespace")
            slug = text("Namespace slug (3-64 chars)", required=True,
                        validator=lambda s: "Must be 3-64 chars" if len(s) < 3 or len(s) > 64 else None)
            if not display_name:
                display_name = text("Display name", default=slug)

        if not slug:
            _err("slug is required.")

        ns = client.create_namespace(
            slug=slug,
            display_name=display_name or slug,
        )
        ns_id = ns.get("namespace_id", "?")
        _ok(f"\nNamespace created: {ns_id}")
        _ok(f"  slug: {slug}")
        _ok("\nDeploy an environment or tool:")
        _ok("  cd your-project/ && epsilab deploy")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── profile commands ─────────────────────────────────────────────────


def cmd_profile_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        profile = client.get_creator_profile()
        if args.json:
            _json_out(profile)
        else:
            _ok("Creator Profile:")
            _ok(f"  name: {profile.get('display_name', '-')}")
            _ok(f"  bio: {profile.get('bio', '-')}")
            _ok(f"  website: {profile.get('website_url', '-')}")
            _ok(f"  public: {profile.get('is_public', False)}")
    except ApiError as e:
        if e.status_code == 404:
            _ok("No creator profile yet.")
            _ok("  Create one with: epsilab profile create \"Your Name\"")
            _ok(f"  Or set it up at {_DASHBOARD_URL} (Settings > Profile)")
        else:
            _err(str(e))
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_profile_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        display_name = args.display_name
        bio = args.bio
        website = args.website
        email = args.email

        if is_interactive() and not display_name:
            step("Set up your creator profile")
            display_name = text("Display name", required=True)
            bio = bio or text("Short bio", default="")
            website = website or text("Website URL", default="")
            email = email or text("Contact email", default="")

        if not display_name:
            _err("display_name is required.")

        profile = client.create_creator_profile(
            display_name=display_name,
            bio=bio,
            website_url=website,
            contact_email=email,
        )
        _ok(f"\nCreator profile created: {profile.get('display_name', '?')}")
        _ok(f"Manage your profile at {_DASHBOARD_URL} (Settings > Profile)")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── env init ─────────────────────────────────────────────────────────


_TASKS_TEMPLATE = """\
[
  {
    "task_id": "%(slug)s-easy-train-001",
    "domain": "general",
    "capability": "instruction-following",
    "prompt": "Reply with the exact phrase: hello epsilab",
    "expected_answer": "hello epsilab",
    "verification": "symbolic",
    "difficulty": "easy",
    "split": "train",
    "max_steps": 3,
    "pass_threshold": 1.0
  }
]
"""

_DOCKERFILE_TEMPLATE = """\
FROM python:3.12-slim@sha256:46cb7cc2877e60fbd5e21a9ae6115c30ace7a077b9f8772da879e4590c18c2e3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps openenv-core==0.3.0
COPY . /app/
RUN mkdir -p /opt/epsilab && mv /app/verifier.py /opt/epsilab/verifier.py

USER 65532:65532
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=2s --retries=6 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)"]

CMD ["python", "-B", "/app/server.py"]
"""

_REQUIREMENTS_TEMPLATE = """\
fastapi==0.139.0
fastmcp==3.4.4
websockets==16.1
uvicorn==0.51.0
pydantic==2.13.4
"""

_ENVIRONMENT_TEMPLATE = '''\
"""A deterministic OpenEnv environment ready for local and hosted execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openenv.core import Action, Environment, Observation, State
from openenv.core.env_server.types import EnvironmentMetadata
from pydantic import Field


class TextAction(Action):
    content: str = Field(..., description="The agent response")
    action_type: str = Field(default="submit", description="respond or submit")


class TextObservation(Observation):
    content: str
    context: dict[str, Any] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class TaskState(State):
    task_id: str = ""
    task: dict[str, Any] = Field(default_factory=dict)
    submitted: bool = False


class ExampleEnvironment(Environment[TextAction, TextObservation, TaskState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())
        self._state = TaskState()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
    ) -> TextObservation:
        del seed
        task = self._resolve_task(task_id)
        self._state = TaskState(
            episode_id=episode_id,
            task_id=task["task_id"],
            task=task,
        )
        return TextObservation(
            content=task["prompt"],
            context={"task_id": task["task_id"], "difficulty": task["difficulty"]},
        )

    def step(self, action: TextAction) -> TextObservation:
        if not self._state.task:
            self.reset()
        if self._state.submitted:
            return TextObservation(
                content="Episode already completed.",
                done=True,
                terminated=True,
                info={"error": "already_completed"},
            )
        if action.action_type != "submit":
            self._state = self._state.model_copy(
                update={"step_count": self._state.step_count + 1}
            )
            return TextObservation(
                content="Submit your final answer when ready.",
                info={"step": self._state.step_count},
            )

        expected = str(self._state.task["expected_answer"]).strip().casefold()
        reward = 1.0 if action.content.strip().casefold() == expected else 0.0
        self._state = self._state.model_copy(update={"submitted": True})
        return TextObservation(
            content="Correct." if reward == 1.0 else "That answer is not correct.",
            reward=reward,
            done=True,
            terminated=True,
            info={"passed": reward == 1.0},
        )

    @property
    def state(self) -> TaskState:
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        return None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="__SLUG__",
            description="A deterministic example environment",
            version="1.0.0",
        )

    def _resolve_task(self, task_id: str | None) -> dict[str, Any]:
        if task_id is None:
            return self._tasks[0]
        for task in self._tasks:
            if task.get("task_id") == task_id:
                return task
        raise ValueError(f"unknown task_id: {task_id}")
'''

_SERVER_TEMPLATE = '''\
"""HTTP entry point for the environment."""

from openenv.core import create_fastapi_app

from environment import ExampleEnvironment, TextAction, TextObservation


app = create_fastapi_app(
    ExampleEnvironment,
    TextAction,
    TextObservation,
    max_concurrent_envs=8,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

_VERIFIER_TEMPLATE = '''\
"""Replay a completed trajectory and emit a strict verification result."""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from environment import ExampleEnvironment, TextAction

_MAX_INPUT_BYTES = 2 * 1024 * 1024


def _main() -> None:
    raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
    if len(raw) > _MAX_INPUT_BYTES:
        _invalid("trajectory_too_large")
        return
    try:
        trajectory = json.loads(raw)
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be an object")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("trajectory has no steps")

        environment = ExampleEnvironment()
        try:
            environment.reset(
                seed=trajectory.get("seed") if isinstance(trajectory.get("seed"), int) else None,
                task_id=str(trajectory["task_id"]),
            )
            result = None
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise ValueError(f"step {index} must be an object")
                action = step.get("action")
                if isinstance(action, str):
                    action = json.loads(action)
                if not isinstance(action, dict):
                    raise ValueError(f"step {index} action must be an object")
                result = environment.step(TextAction.model_validate(action))
                _verify_claim(index, step, result)
                if (result.terminated or result.truncated) and index != len(steps) - 1:
                    raise ValueError("trajectory contains actions after termination")
            if result is None or result.reward is None or not (result.terminated or result.truncated):
                raise ValueError("trajectory is not terminal")
            if isinstance(result.reward, bool):
                raise ValueError("reward is invalid")
            reward = float(result.reward)
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise ValueError("reward is invalid")
            print(json.dumps({
                "status": "valid",
                "reward": reward,
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "tests_passed": 1 if reward == 1.0 else 0,
                "tests_total": 1,
            }, separators=(",", ":")))
        finally:
            environment.close()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        _invalid("invalid_trajectory")
    except Exception:
        _invalid("verifier_failure")


def _verify_claim(index: int, claimed: dict[str, Any], replayed: Any) -> None:
    reward = claimed.get("reward")
    if reward is None or replayed.reward is None:
        if reward is not None or replayed.reward is not None:
            raise ValueError(f"reward diverged at step {index}")
    elif not math.isclose(float(reward), float(replayed.reward), abs_tol=1e-6):
        raise ValueError(f"reward diverged at step {index}")
    if claimed.get("terminated") is not bool(replayed.terminated):
        raise ValueError(f"termination diverged at step {index}")
    if claimed.get("truncated") is not bool(replayed.truncated):
        raise ValueError(f"truncation diverged at step {index}")
    observation = claimed.get("observation")
    if isinstance(observation, dict):
        observation = observation.get("content")
    if isinstance(observation, str) and observation != replayed.content:
        raise ValueError(f"observation diverged at step {index}")


def _invalid(reason: str) -> None:
    print(json.dumps({"status": "invalid", "reason": reason}, separators=(",", ":")))


if __name__ == "__main__":
    _main()
'''


def cmd_env_verify(args: argparse.Namespace) -> None:
    """Run local preflight checks on an environment project before pushing."""
    import re
    import subprocess
    import urllib.request

    target = Path(args.directory or ".")
    manifest_path = target / (args.manifest or "epsilab.json")
    dockerfile = target / "Dockerfile"
    server_py = target / "server.py"
    environment_py = target / "environment.py"
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    _ok("Verifying environment project...\n")

    tasks_path = target / "tasks.json"
    if tasks_path.exists():
        try:
            tasks_value = json.loads(tasks_path.read_text())
            if not isinstance(tasks_value, list) or any(
                not isinstance(task, dict) for task in tasks_value
            ):
                errors.append("tasks.json must contain a list of task objects")
            else:
                horizon_errors = _environment_horizon_errors(
                    tasks_value,
                    max_steps=_MAX_LOCAL_OPENENV_STEPS,
                    source="the OpenEnv contract",
                )
                errors.extend(horizon_errors)
                if not horizon_errors:
                    longest = max(
                        (int(task.get("max_steps", 50)) for task in tasks_value),
                        default=0,
                    )
                    passed.append(
                        f"Task horizons are valid (maximum {longest}; OpenEnv limit "
                        f"{_MAX_LOCAL_OPENENV_STEPS})"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"tasks.json is not valid JSON: {exc}")

    # ── 1. Manifest checks (optional — deploy flow doesn't use manifests) ──
    if not manifest_path.exists():
        passed.append("No manifest file (not required for epsilab deploy)")
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
            passed.append("Manifest is valid JSON")
        except json.JSONDecodeError as e:
            errors.append(f"Manifest is not valid JSON: {e}")
            manifest = None

        if manifest is not None:
            for field in ("listing_id", "namespace_id", "release_version"):
                val = manifest.get(field)
                if not val or not str(val).strip():
                    warnings.append(f"Manifest field '{field}' is empty (required for push)")
                else:
                    passed.append(f"Manifest field '{field}' is set")

            version = manifest.get("release_version", "")
            if version and not re.match(r"^\d+\.\d+\.\d+", version):
                warnings.append(f"release_version '{version}' does not look like semver (X.Y.Z)")

            env_config = manifest.get("environment", {})
            if not isinstance(env_config, dict):
                errors.append("Manifest 'environment' section is not an object")
            else:
                for digest_field in ("runtime_digest", "action_schema_digest", "observation_schema_digest"):
                    val = env_config.get(digest_field, "")
                    if val and not re.match(r"^sha256:[0-9a-f]{64}$", val):
                        errors.append(f"environment.{digest_field} is not a valid sha256 digest: {val}")
                    elif val:
                        passed.append(f"environment.{digest_field} format is valid")

                ref = env_config.get("runtime_ref", "")
                if ref:
                    passed.append("environment.runtime_ref is set")

            tp_config = manifest.get("task_pack", {})
            if isinstance(tp_config, dict):
                digest = tp_config.get("artifact_digest", "")
                if digest and not re.match(r"^sha256:[0-9a-f]{64}$", digest):
                    errors.append(f"task_pack.artifact_digest is not valid: {digest}")

            ver_config = manifest.get("verifier", {})
            if isinstance(ver_config, dict):
                digest = ver_config.get("runtime_digest", "")
                if digest and not re.match(r"^sha256:[0-9a-f]{64}$", digest):
                    errors.append(f"verifier.runtime_digest is not valid: {digest}")

    # ── 2. File structure checks ──────────────────────────────────
    if environment_py.exists():
        passed.append("environment.py exists")
        try:
            compile(environment_py.read_text(), str(environment_py), "exec")
            passed.append("environment.py is valid Python")
        except SyntaxError as exc:
            errors.append(f"environment.py is invalid Python: {exc}")

    if server_py.exists():
        passed.append("server.py exists")
        source = server_py.read_text()
        uses_openenv = "create_fastapi_app" in source
        if "/reset" not in source and not uses_openenv:
            errors.append("server.py does not contain a /reset endpoint")
        else:
            passed.append("server.py references /reset endpoint")
        if "/step" not in source and not uses_openenv:
            errors.append("server.py does not contain a /step endpoint")
        else:
            passed.append("server.py references /step endpoint")
        if "8000" not in source and "PORT" not in source:
            warnings.append("server.py does not reference port 8000 or PORT")
    else:
        warnings.append("server.py not found (expected for container environments)")

    if dockerfile.exists():
        passed.append("Dockerfile exists")
        df_content = dockerfile.read_text()
        if "EXPOSE" not in df_content:
            warnings.append("Dockerfile does not contain EXPOSE directive")
    else:
        warnings.append("Dockerfile not found")

    # ── 3. Docker build check (if --build) ────────────────────────
    image_tag = None
    if args.build and dockerfile.exists():
        _ok("Building Docker image...")
        tag = f"epsilab-verify-{int(time.time())}"
        result = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            errors.append(f"Docker build failed:\n{result.stderr[-500:]}")
        else:
            passed.append("Docker build succeeded")
            image_tag = tag

            inspect = subprocess.run(
                ["docker", "inspect", "--format", "{{.Size}}", tag],
                capture_output=True, text=True,
            )
            if inspect.returncode == 0:
                size_mb = int(inspect.stdout.strip()) / (1024 * 1024)
                passed.append(f"Image size: {size_mb:.0f} MB")
                if size_mb > 2048:
                    warnings.append(
                        f"Image is {size_mb:.0f} MB — large images slow down session provisioning. "
                        "Consider multi-stage builds, .dockerignore, or --no-cache-dir for pip."
                    )
                elif size_mb > 1024:
                    warnings.append(f"Image is {size_mb:.0f} MB — consider optimizing if possible")

            digest_out = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", tag],
                capture_output=True, text=True,
            )
            if digest_out.returncode == 0:
                digest = digest_out.stdout.strip()
                _ok(f"  Image digest: {digest}")

            # Validate USER directive: if Dockerfile specifies a non-root user,
            # verify the image is configured to run as that user
            user_directive = None
            for line in df_content.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("USER ") and not stripped.startswith("#"):
                    user_directive = stripped.split(None, 1)[1].strip()  # last USER wins
            if user_directive:
                user_inspect = subprocess.run(
                    ["docker", "inspect", "--format", "{{.Config.User}}", tag],
                    capture_output=True, text=True,
                )
                if user_inspect.returncode == 0:
                    actual_user = user_inspect.stdout.strip()
                    if actual_user and actual_user != "root" and actual_user != "0":
                        passed.append(f"Container runs as non-root user: {actual_user}")
                    elif not actual_user or actual_user in ("root", "0"):
                        warnings.append(
                            f"Dockerfile has USER {user_directive} but image runs as root — "
                            "verify the USER directive is not overridden"
                        )

    # ── 4. Protocol smoke test (if --test or --build) ─────────────
    if args.test and image_tag:
        _ok("Running protocol smoke test...")
        container_id = None
        try:
            run_result = subprocess.run(
                ["docker", "run", "-d", "--rm", "-p", "127.0.0.1::8000", image_tag],
                capture_output=True, text=True, timeout=30,
            )
            if run_result.returncode != 0:
                errors.append(f"Failed to start container: {run_result.stderr}")
            else:
                container_id = run_result.stdout.strip()
                port_result = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}',
                        container_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if port_result.returncode != 0 or not port_result.stdout.strip().isdigit():
                    raise RuntimeError("could not resolve the container health port")
                base_url = f"http://127.0.0.1:{port_result.stdout.strip()}"

                deadline = time.monotonic() + 30
                while True:
                    try:
                        with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("environment did not become healthy within 30 seconds")
                        time.sleep(0.25)

                try:
                    reset_payload = json.dumps(
                        {"task_id": json.loads((target / "tasks.json").read_text())[0]["task_id"], "seed": 42}
                    ).encode()
                    req = urllib.request.Request(
                        f"{base_url}/reset",
                        data=reset_payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        reset_body = json.loads(resp.read())
                        if "observation" not in reset_body:
                            errors.append("POST /reset response missing 'observation' field")
                        else:
                            passed.append("POST /reset returns valid response with 'observation'")
                    repeat_req = urllib.request.Request(
                        f"{base_url}/reset",
                        data=reset_payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(repeat_req, timeout=10) as resp:
                        if json.loads(resp.read()) != reset_body:
                            errors.append("POST /reset is not deterministic for the same task and seed")
                        else:
                            passed.append("POST /reset is deterministic")
                except Exception as e:
                    errors.append(f"POST /reset failed: {e}")

                try:
                    req = urllib.request.Request(
                        f"{base_url}/step",
                        data=json.dumps(
                            {"action": {"content": "hello epsilab", "action_type": "submit"}}
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        step_body = json.loads(resp.read())
                        required = {"observation", "reward", "done"}
                        missing = required - set(step_body.keys())
                        if missing:
                            errors.append(f"POST /step response missing fields: {missing}")
                        else:
                            passed.append("POST /step returns all required fields")
                        if "reward" in step_body:
                            if not isinstance(step_body["reward"], (int, float)):
                                errors.append(f"reward must be numeric, got {type(step_body['reward']).__name__}")
                        if "done" in step_body and not isinstance(step_body["done"], bool):
                            errors.append(f"done must be bool, got {type(step_body['done']).__name__}")
                except Exception as e:
                    errors.append(f"POST /step failed: {e}")

                # ── 4b. Tool smoke test ───────────────────────────
                # Re-reset so we have a fresh observation with tools
                tools_to_probe: dict[str, list[str]] = {}
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(
                            f"{base_url}/reset",
                            data=reset_payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        ),
                        timeout=10,
                    ) as resp:
                        obs = json.loads(resp.read())
                    obs_text = obs.get("observation", "")
                    if isinstance(obs_text, str):
                        try:
                            obs_data = json.loads(obs_text)
                        except (json.JSONDecodeError, TypeError):
                            obs_data = {}
                    elif isinstance(obs_text, dict):
                        obs_data = obs_text
                    else:
                        obs_data = {}
                    ctx = obs_data.get("context", {}) if isinstance(obs_data, dict) else {}
                    raw_tools = ctx.get("tools", {})
                    if isinstance(raw_tools, dict):
                        tools_to_probe = raw_tools
                except Exception:
                    pass

                _TOOL_PROBES: dict[str, dict] = {
                    "files": {"plugin": "browser", "method": "files.list", "args": {}},
                    "preview": {"plugin": "browser", "method": "preview.screenshot", "args": {}},
                    "audit": {"plugin": "browser", "method": "audit.performance", "args": {"url": "about:blank"}},
                    "calendar": {"plugin": "calendar", "method": "calendars.list", "args": {}},
                    "github": {"plugin": "github", "method": "repos.list", "args": {}},
                    "pagerduty": {"plugin": "pagerduty", "method": "incidents.list", "args": {}},
                    "support": {"plugin": "support", "method": "tickets.list", "args": {}},
                    "gmail": {"plugin": "gmail", "method": "messages.list", "args": {"userId": "me"}},
                }

                if tools_to_probe:
                    _ok(f"  Probing {len(tools_to_probe)} tool(s): {', '.join(sorted(tools_to_probe))}")
                    for tool_name in sorted(tools_to_probe):
                        probe = _TOOL_PROBES.get(tool_name)
                        if probe is None:
                            probe = {
                                "plugin": tool_name,
                                "method": f"{tool_name}.list"
                                    if any(m.endswith(".list") for m in tools_to_probe.get(tool_name, []))
                                    else (tools_to_probe.get(tool_name, ["unknown"])[0]),
                                "args": {},
                            }
                        try:
                            step_req = urllib.request.Request(
                                f"{base_url}/step",
                                data=json.dumps({"action": probe}).encode(),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urllib.request.urlopen(step_req, timeout=30) as resp:
                                probe_body = json.loads(resp.read())
                            probe_obs = probe_body.get("observation", "")
                            if isinstance(probe_obs, str) and "error" in probe_obs.lower():
                                try:
                                    probe_detail = json.loads(probe_obs)
                                    err_msg = probe_detail.get("error", probe_obs[:200])
                                except (json.JSONDecodeError, TypeError):
                                    err_msg = probe_obs[:200]
                                errors.append(f"Tool '{tool_name}' returned error: {err_msg}")
                            else:
                                passed.append(f"Tool '{tool_name}' responds without error")
                        except Exception as e:
                            errors.append(f"Tool '{tool_name}' probe failed: {e}")

        finally:
            if container_id:
                subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True, timeout=15,
                )

    elif args.test and not image_tag:
        warnings.append("--test requires --build to create a testable image")

    # ── 5. Manifest digest consistency (if --build) ───────────────
    if image_tag and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            env_digest = m.get("environment", {}).get("runtime_digest", "")
            if env_digest:
                digest_out = subprocess.run(
                    ["docker", "inspect", "--format", "{{.Id}}", image_tag],
                    capture_output=True, text=True,
                )
                if digest_out.returncode == 0:
                    local_digest = digest_out.stdout.strip()
                    if local_digest == env_digest:
                        passed.append("Image digest matches manifest runtime_digest")
                    else:
                        warnings.append(
                            f"Local image digest ({local_digest[:20]}...) differs from "
                            f"manifest runtime_digest ({env_digest[:20]}...) — "
                            "rebuild and redeploy to update"
                        )
        except Exception:
            pass

    # ── Report ────────────────────────────────────────────────────
    print()
    if passed:
        for msg in passed:
            print(f"  \033[32m✓\033[0m {msg}")
    if warnings:
        print()
        for msg in warnings:
            print(f"  \033[33m⚠\033[0m {msg}")
    if errors:
        print()
        for msg in errors:
            print(f"  \033[31m✗\033[0m {msg}")

    print()
    total = len(passed) + len(warnings) + len(errors)
    if errors:
        _err(
            f"Verification failed: {len(errors)} error(s), "
            f"{len(warnings)} warning(s), {len(passed)} passed "
            f"out of {total} checks."
        )
    elif warnings:
        _ok(
            f"Verification passed with warnings: {len(warnings)} warning(s), "
            f"{len(passed)} passed out of {total} checks."
        )
    else:
        _ok(f"Verification passed: all {total} checks passed.")

    # Cleanup verify image
    if image_tag:
        subprocess.run(
            ["docker", "rmi", image_tag],
            capture_output=True, timeout=30,
        )


def cmd_env_test(args: argparse.Namespace) -> None:
    """Run a local multi-step session against a containerized environment."""
    import subprocess
    import urllib.request

    target = Path(args.directory or ".")
    dockerfile = target / "Dockerfile"
    tasks_path = target / "tasks.json"

    if not dockerfile.exists():
        _err("No Dockerfile found in the target directory.")
        return
    if not tasks_path.exists():
        _err("No tasks.json found in the target directory.")
        return

    tasks = json.loads(tasks_path.read_text())
    if not tasks:
        _err("tasks.json is empty.")
        return

    task_id = args.task or tasks[0]["task_id"]
    task = next((t for t in tasks if t["task_id"] == task_id), None)
    if task is None:
        _err(f"Task '{task_id}' not found in tasks.json. Available: {[t['task_id'] for t in tasks]}")
        return

    max_steps = args.steps or int(task.get("max_steps", 5))
    _ok(f"Testing environment: {target.name}")
    _ok(f"  Task:      {task_id}")
    _ok(f"  Max steps: {max_steps}")

    # Build image
    tag = f"epsilab-test-{int(time.time())}"
    _ok("\nBuilding Docker image...")
    build = subprocess.run(
        ["docker", "build", "-t", tag, "."],
        cwd=str(target), capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        _err(f"Docker build failed:\n{build.stderr[-1000:]}")
        return
    _ok("  Build succeeded")

    container_id = None
    try:
        run_result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "127.0.0.1::8000", tag],
            capture_output=True, text=True, timeout=30,
        )
        if run_result.returncode != 0:
            _err(f"Failed to start container: {run_result.stderr}")
            return
        container_id = run_result.stdout.strip()

        port_result = subprocess.run(
            ["docker", "inspect", "--format",
             '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}',
             container_id],
            capture_output=True, text=True, timeout=15,
        )
        if port_result.returncode != 0 or not port_result.stdout.strip().isdigit():
            _err("Could not resolve container port.")
            return
        base_url = f"http://127.0.0.1:{port_result.stdout.strip()}"

        # Wait for health
        _ok("  Waiting for container health...")
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    _err("Environment did not become healthy within 60 seconds.")
                    return
                time.sleep(0.5)
        _ok("  Container healthy")

        # Reset
        _ok(f"\n  POST /reset  task={task_id}")
        reset_payload = json.dumps({"task_id": task_id, "seed": 42}).encode()
        req = urllib.request.Request(
            f"{base_url}/reset", data=reset_payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            reset_body = json.loads(resp.read())
        if "observation" not in reset_body:
            _err("POST /reset response missing 'observation'")
            return
        obs = reset_body["observation"]
        _ok(f"    observation: {str(obs)[:200]}{'...' if len(str(obs)) > 200 else ''}")

        # Extract tools from observation
        tools: dict = {}
        try:
            obs_data = json.loads(obs) if isinstance(obs, str) else obs
            if isinstance(obs_data, dict):
                tools = obs_data.get("context", {}).get("tools", {})
        except (json.JSONDecodeError, TypeError):
            pass

        if tools:
            _ok(f"    tools: {', '.join(sorted(tools))}")

        # Run steps
        _TOOL_ACTIONS: dict[str, list[dict]] = {
            "files": [
                {"plugin": "browser", "method": "files.list", "args": {}},
                {"plugin": "browser", "method": "files.write", "args": {
                    "path": "index.html", "content": "<html><body><h1>Test</h1></body></html>"}},
            ],
            "preview": [
                {"plugin": "browser", "method": "preview.screenshot", "args": {}},
            ],
            "audit": [
                {"plugin": "browser", "method": "audit.performance", "args": {"url": "about:blank"}},
            ],
            "calendar": [
                {"plugin": "calendar", "method": "calendars.list", "args": {}},
            ],
            "github": [
                {"plugin": "github", "method": "repos.list", "args": {}},
            ],
        }

        # Build a sequence of actions: use tool-specific ones if available, else generic
        actions: list[dict] = []
        for tool_name in sorted(tools):
            tool_actions = _TOOL_ACTIONS.get(tool_name, [])
            actions.extend(tool_actions)
        if not actions:
            actions.append(_default_environment_smoke_action(task))

        actions = actions[:max_steps]
        steps_completed = 0
        done = False

        for step_idx, action in enumerate(actions):
            method = action.get("method", action.get("action_type", "?"))
            _ok(f"\n  step {step_idx + 1}/{len(actions)}  {method}")
            step_req = urllib.request.Request(
                f"{base_url}/step",
                data=json.dumps({"action": action}).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(step_req, timeout=60) as resp:
                    step_body = json.loads(resp.read())
            except Exception as e:
                _err(f"    Step failed: {e}")
                break

            steps_completed += 1
            step_obs = str(step_body.get("observation", ""))[:200]
            reward = step_body.get("reward")
            step_done = step_body.get("done", False)
            _ok(f"    reward={reward}  done={step_done}  obs_len={len(str(step_body.get('observation', '')))}")

            if "error" in step_obs.lower():
                try:
                    err_data = json.loads(step_body.get("observation", ""))
                    err_msg = err_data.get("error", step_obs)
                except (json.JSONDecodeError, TypeError):
                    err_msg = step_obs
                _err(f"    Tool error: {err_msg}")
                break

            if step_done:
                done = True
                break

        # Summary
        print(f"\n{'=' * 60}")
        _ok(f"  Steps completed: {steps_completed}/{len(actions)}")
        if done:
            _ok(f"  Session terminated: reward={reward}")
        else:
            _ok(f"  Session did not terminate (ran all {steps_completed} steps)")
        print(f"{'=' * 60}")

    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=15)
            _ok("\n  Container stopped")
    # Cleanup image
    subprocess.run(["docker", "rmi", tag], capture_output=True, timeout=30)


def _default_environment_smoke_action(task: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic local smoke action for a task without tools."""
    configured = task.get("smoke_action")
    if isinstance(configured, dict) and configured:
        return dict(configured)
    expected = task.get("expected_answer")
    if isinstance(expected, (str, int, float, bool)):
        return {"content": str(expected), "action_type": "submit"}
    return {"content": "test step", "action_type": "submit"}


def cmd_env_init(args: argparse.Namespace) -> None:
    slug = args.slug
    target_dir = args.directory

    if is_interactive() and not slug:
        step("Initialize a new environment project")
        slug = text("Environment slug", default="my-environment", required=True)
        if not target_dir:
            target_dir = text("Project directory", default=slug)

    raw_slug = slug or "my-environment"
    slug = _normalize_slug(raw_slug)
    if len(slug) < 3 or len(slug) > 64:
        _err("Environment name must produce a 3-64 character slug.")
    target = Path(target_dir or slug)

    if target.exists() and any(target.iterdir()):
        if is_interactive():
            if not confirm(f"Directory '{target}' is not empty. Continue anyway?", default=False):
                sys.exit(0)
        else:
            _err(f"Directory '{target}' already exists and is not empty.")

    target.mkdir(parents=True, exist_ok=True)

    (target / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE)
    (target / "environment.py").write_text(
        _ENVIRONMENT_TEMPLATE.replace("__SLUG__", slug)
    )
    (target / "server.py").write_text(_SERVER_TEMPLATE)
    (target / "verifier.py").write_text(_VERIFIER_TEMPLATE)
    (target / "tasks.json").write_text(_TASKS_TEMPLATE % {"slug": slug})
    (target / "requirements.txt").write_text(_REQUIREMENTS_TEMPLATE)
    (target / ".dockerignore").write_text(
        ".git\n.epsilab\n__pycache__\n*.py[cod]\n.pytest_cache\n"
    )
    _save_project(
        target,
        {
            "type": "environment",
            "name": slug,
            "slug": slug,
            "title": slug.replace("-", " ").title(),
            "summary": "A deterministic OpenEnv environment.",
            "version": "1.0.0",
            "visibility": "public",
            "reward_mode": "continuous",
            "qualification": {
                "task_id": f"{slug}-easy-train-001",
                "smoke_actions": [
                    {"content": "hello epsilab", "action_type": "submit"}
                ],
                "repetitions": 3,
                "seed": 0,
            },
            "resource_policy": dict(_DEFAULT_ENVIRONMENT_RESOURCE_POLICY),
            "namespace_id": "",
            "listing_id": "",
        },
    )

    _ok(f"\nInitialized environment project in {target}/")
    _ok("")
    _ok(f"  {target}/environment.py — environment logic")
    _ok(f"  {target}/verifier.py    — deterministic trajectory verifier")
    _ok(f"  {target}/tasks.json     — task definitions")
    _ok(f"  {target}/Dockerfile     — production container")
    _ok("")
    _ok("Next steps:")
    _ok("  1. Implement your environment logic in environment.py")
    _ok("  2. Add your tasks to tasks.json")
    _ok(f"  3. Deploy:  cd {target} && epsilab deploy")
    _ok("")
    _ok("The deploy command handles everything: building, uploading, and registering.")
    _ok("No registry credentials or manual image pushing required.")
    _ok("")
    _ok(f"Documentation:  {_DASHBOARD_URL} (Documentation > Quick Start)")


# ── env session commands ──────────────────────────────────────────────


def cmd_env_session_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        deployment_id = args.deployment_id
        task_id = args.task_id

        if is_interactive() and not deployment_id:
            step("Create an environment session")
            deployment_id = text("Deployment ID", required=True)
            if not task_id:
                task_id = text("Task ID", required=True)

        if not deployment_id or not task_id:
            _err("--deployment-id and --task-id are required.")

        t0 = time.monotonic()
        session = client.create_environment_session(
            deployment_id=deployment_id,
            task_id=task_id,
            seed=args.seed,
        )
        elapsed = time.monotonic() - t0
        _ok(f"Session created: {session.session_id}")
        _ok(f"  status:     {session.status}")
        _ok(f"  task:       {session.task_id}")
        _ok(f"  deployment: {deployment_id}")
        _ok(f"  latency:    {elapsed:.1f}s")
        if session.seed is not None:
            _ok(f"  seed:       {session.seed}")
        if session.observation:
            _ok(f"  observation: {session.observation[:300]}")
        if session.session_token:
            _ok(f"  token:      {session.session_token[:8]}{'*' * 12}")
        _ok("")
        _ok(f"Next: epsilab env session step {session.session_id} \"<action>\" --session-token <token>")
        if args.json:
            safe = session.to_dict()
            if "session_token" in safe:
                safe["session_token"] = safe["session_token"][:8] + "***"
            _json_out(safe)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_session_step(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        session_id = args.session_id
        action = args.action

        if is_interactive() and not session_id:
            step("Take an action in an environment session")
            session_id = text("Session ID", required=True)
            if not action:
                action = text("Action", required=True)

        if not session_id or not action:
            _err("session_id and action are required.")

        t0 = time.monotonic()
        result = client.environment_step(
            session_id=session_id,
            action=action,
            session_token=args.session_token,
        )
        elapsed = time.monotonic() - t0

        reward_str = f"{result.reward:.4f}" if result.reward is not None else "-"
        status_parts = []
        if result.terminated:
            status_parts.append("TERMINAL")
        if result.truncated:
            status_parts.append("TRUNCATED")
        status_label = " | ".join(status_parts) if status_parts else "continue"

        _ok(f"Step result ({elapsed:.1f}s)")
        _ok(f"  reward:  {reward_str}")
        _ok(f"  status:  {status_label}")
        if result.observation:
            obs_lines = result.observation.split("\n")
            preview = "\n           ".join(obs_lines[:10])
            _ok(f"  observe: {preview}")
            if len(obs_lines) > 10:
                _ok(f"           ... ({len(obs_lines) - 10} more lines)")
        if result.info:
            _ok(f"  info:    {json.dumps(result.info, default=str)[:200]}")
        if result.terminated or result.truncated:
            _ok("")
            _ok("Session complete.")
        if args.json:
            _json_out({"observation": result.observation, "reward": result.reward,
                        "terminated": result.terminated, "truncated": result.truncated,
                        "info": result.info})
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_session_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        session = client.get_environment_session(args.session_id)
        _ok(f"Session: {session.session_id}")
        _ok(f"  status:      {session.status}")
        _ok(f"  task:        {session.task_id}")
        _ok(f"  env_type:    {getattr(session, 'env_type', '-')}")
        _ok(f"  reward_mode: {getattr(session, 'reward_mode', '-')}")
        _ok(f"  steps:       {getattr(session, 'steps_taken', '-')}")
        total = getattr(session, "total_reward", None)
        _ok(f"  reward:      {total:.4f}" if total is not None else "  reward:      -")
        if getattr(session, "deployment_id", None):
            _ok(f"  deployment:  {session.deployment_id}")
        if getattr(session, "seed", None) is not None:
            _ok(f"  seed:        {session.seed}")
        if getattr(session, "terminal_reason", None):
            _ok(f"  terminal:    {session.terminal_reason}")
        created = getattr(session, "created_at", None)
        if created:
            _ok(f"  created:     {str(created)[:19]}")
        closed = getattr(session, "closed_at", None)
        if closed:
            _ok(f"  closed:      {str(closed)[:19]}")
        if session.observation:
            _ok(f"  observation: {session.observation[:300]}")
        if args.json:
            _json_out(session.to_dict())
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_session_cancel(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.cancel_environment_session(args.session_id)
        _ok(f"Session cancelled: {args.session_id}")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def _resolve_environment_listing(client: EpsilabClient, target: str) -> Any:
    """Resolve an owner/name reference to one deployed environment listing."""
    if target.count("/") != 1:
        raise ValueError("Environment must be written as <owner>/<name>.")
    owner, slug = (part.strip().lower() for part in target.split("/", 1))
    if not owner or not slug:
        raise ValueError("Environment must be written as <owner>/<name>.")

    listings = client.list_environment_listings(limit=100)
    exact = [
        listing
        for listing in listings
        if listing.slug.lower() == slug
        and (listing.namespace or "").lower() == owner
    ]
    if not exact:
        slug_matches = [
            listing
            for listing in listings
            if listing.slug.lower() == slug and not listing.namespace
        ]
        if len(slug_matches) == 1:
            exact = slug_matches
    if not exact:
        raise ValueError(f"Environment '{target}' was not found.")
    if len(exact) > 1:
        raise ValueError(f"Environment '{target}' is ambiguous; use its exact owner/name.")

    listing = exact[0]
    if not listing.deployment_id:
        raise ValueError(f"Environment '{target}' does not have an active deployment.")
    return listing


def _resolve_environment_task(
    client: EpsilabClient,
    *,
    slug: str,
    explicit_task_id: str | None,
    published_tasks: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve a task for a listing without inventing arbitrary UUIDs."""
    if explicit_task_id:
        return explicit_task_id

    published_candidates = [
        (str(task.get("split") or ""), task_id)
        for task in (published_tasks or [])
        if isinstance(task, dict)
        and isinstance((task_id := task.get("task_id")), str)
        and task_id
    ]
    if published_candidates:
        return sorted(
            set(published_candidates),
            key=lambda item: (item[0] != "train", item[1]),
        )[0][1]

    candidates: list[str] = []
    try:
        for task in client.iter_tasks(source="custom", page_size=100):
            task_id = task.get("task_id")
            capability = task.get("capability")
            if isinstance(task_id, str) and (
                task_id.startswith(f"{slug}-") or capability == slug
            ):
                candidates.append(task_id)
    except EpsilabError:
        _cli_logger.debug("Task discovery failed for %s", slug, exc_info=True)
    if candidates:
        return sorted(
            set(candidates),
            key=lambda task_id: ("-train-" not in task_id, task_id),
        )[0]
    raise ValueError(
        f"No published tasks were found for '{slug}'. Pass --task <task-id> explicitly."
    )


def _encode_environment_action(value: str, *, action_type: str) -> str:
    """Encode plain text or validate an explicitly structured action."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("Action cannot be empty.")
    if stripped.startswith("{"):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Structured action is not valid JSON: {exc.msg}.") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Structured action must be a JSON object.")
        return json.dumps(decoded, separators=(",", ":"), sort_keys=True)
    return json.dumps(
        {"content": value, "action_type": action_type},
        separators=(",", ":"),
        sort_keys=True,
    )


def _interactive_action(raw: str, default_type: str) -> tuple[str, str]:
    """Parse optional action types used by the terminal loop."""
    stripped = raw.strip()
    if not stripped.startswith("/"):
        return raw, default_type
    command, _, content = stripped[1:].partition(" ")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", command):
        raise ValueError("Action type must use letters, numbers, underscores, or hyphens.")
    if not content.strip():
        raise ValueError(f"/{command} requires action content.")
    return content, command


def _public_step_info(info_data: Any) -> dict[str, Any]:
    """Return customer-relevant verification fields only."""
    if not isinstance(info_data, dict):
        return {}
    allowed = {
        "passed",
        "score",
        "terminal_reason",
        "tests_passed",
        "tests_total",
        "verification_authority",
        "verification_evidence_digest",
        "verification_status",
    }
    return {key: info_data[key] for key in sorted(allowed & info_data.keys())}


def _display_environment_observation(label: str, observation: str | None) -> None:
    value = observation if observation not in {None, ""} else "(no observation)"
    _ok(f"\n{label}:\n{value}")


def _display_environment_step(result: Any) -> None:
    _display_environment_observation("Observation", result.observation)
    reward = "pending" if result.reward is None else f"{result.reward:.4f}"
    state = "terminated" if result.terminated else "truncated" if result.truncated else "active"
    _ok(f"\nReward: {reward}")
    _ok(f"Status: {state}")
    verification = _public_step_info(result.info)
    if verification:
        _ok("Verification:")
        for key, value in verification.items():
            _ok(f"  {key}: {value}")


def cmd_run_environment(args: argparse.Namespace) -> None:
    """Run one hosted environment by its public owner/name reference."""
    client = _get_client()
    session = None
    terminal = False
    interrupted = False
    try:
        listing = _resolve_environment_listing(client, args.target)
        published_tasks: list[dict[str, Any]] = []
        if args.task is None:
            try:
                published_tasks = client.get_environment_listing(listing.listing_id).tasks
            except EpsilabError:
                _cli_logger.debug(
                    "Published task discovery failed for %s",
                    listing.listing_id,
                    exc_info=True,
                )
        task_id = _resolve_environment_task(
            client,
            slug=listing.slug,
            explicit_task_id=args.task,
            published_tasks=published_tasks,
        )
        owner = listing.namespace or args.target.split("/", 1)[0]
        canonical_name = f"{owner}/{listing.slug}"

        if not args.json:
            _ok(f"Starting {canonical_name}")
            _ok(f"Task: {task_id}")
        session = client.create_environment_session(
            listing.deployment_id,
            task_id=task_id,
            seed=args.seed,
        )
        session_token = session.session_token
        session = client.wait_for_session(session, timeout=args.timeout)
        if session.session_token is None:
            session.session_token = session_token
        if session.is_terminal:
            terminal = True
            raise RuntimeError(
                f"Session ended during startup ({session.status}: "
                f"{session.terminal_reason or 'no reason provided'})."
            )
        if not session.is_active:
            raise RuntimeError(f"Session entered unexpected state '{session.status}'.")

        if not args.json:
            _ok(f"Session: {session.session_id}")
            _display_environment_observation("Observation", session.observation)

        results: list[dict[str, Any]] = []

        def take_action(raw_action: str, action_type: str) -> bool:
            nonlocal terminal
            encoded = _encode_environment_action(raw_action, action_type=action_type)
            result = client.environment_step(
                session.session_id,
                encoded,
                session_token=session.session_token,
            )
            public_result = result.to_dict()
            public_result["info"] = _public_step_info(result.info)
            results.append(public_result)
            terminal = result.done
            if not args.json:
                _display_environment_step(result)
            if result.done and result.reward is None:
                reason = _public_step_info(result.info).get(
                    "terminal_reason", "unknown error"
                )
                raise RuntimeError(f"Environment action failed ({reason}).")
            return result.done

        if args.action is not None:
            take_action(args.action, args.action_type)
        else:
            if args.json:
                raise ValueError("--json requires --action for non-interactive output.")
            _ok("\nEnter an action. Prefix with /<type> to select an action type; /quit exits.")
            while not terminal:
                try:
                    raw_action = input("action> ")
                except EOFError:
                    break
                if raw_action.strip().lower() in {"/quit", "quit", "exit"}:
                    break
                try:
                    content, action_type = _interactive_action(raw_action, args.action_type)
                    take_action(content, action_type)
                except ValueError as exc:
                    _ok(f"Invalid action: {exc}")

        if args.json:
            _json_out(
                {
                    "environment": canonical_name,
                    "task_id": task_id,
                    "session_id": session.session_id,
                    "results": results,
                }
            )
    except KeyboardInterrupt:
        interrupted = True
        _ok("\nInterrupted.")
    except (EpsilabError, TimeoutError, RuntimeError, ValueError) as exc:
        _err(_friendly_error(exc))
    finally:
        if session is not None and not terminal:
            try:
                client.cancel_environment_session(session.session_id)
                if not args.json:
                    _ok("Session closed.")
            except EpsilabError:
                _cli_logger.warning(
                    "Could not close environment session %s",
                    session.session_id,
                    exc_info=True,
                )
        client.close()
        if interrupted:
            return


def cmd_env_publish(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        listing_id = args.listing_id
        if is_interactive() and not listing_id:
            step("Publish a listing to the marketplace")
            listing_id = _interactive_listing(client)
        if not listing_id:
            _err("listing_id is required.")

        result = client.request_publish(listing_id)
        _ok(f"Listing published: {listing_id}")
        _ok(f"  status: {result.get('status', 'published')}")
        _ok("\nThe listing is now publicly available on the hub.")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_review(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        listing_id = args.listing_id
        if args.list_reviews:
            if not listing_id:
                _err("listing_id is required for --list.")
            reviews = client.list_reviews(listing_id, limit=args.limit)
            if not reviews:
                _ok("No reviews yet.")
                return
            for r in reviews:
                stars = "*" * r.get("rating", 0)
                _ok(f"  {stars}  {r.get('title', 'untitled')}")
                if r.get("body"):
                    _ok(f"    {r['body'][:120]}")
            return

        rating = args.rating
        title = args.title

        if is_interactive() and not all([rating, title]):
            step("Leave a review")
            if not listing_id:
                listing_id = text("Listing ID", required=True)
            if not rating:
                rating = int(select("Rating", [
                    {"value": "5", "label": "★★★★★ Excellent"},
                    {"value": "4", "label": "★★★★  Great"},
                    {"value": "3", "label": "★★★   Good"},
                    {"value": "2", "label": "★★    Fair"},
                    {"value": "1", "label": "★     Poor"},
                ]))
            if not title:
                title = text("Review title", required=True)

        if not rating or not title:
            _err("--rating and --title are required.")

        result = client.create_review(
            listing_id=listing_id,
            rating=int(rating),
            title=title,
            body=args.body,
            usage_hours=args.usage_hours,
        )
        _ok(f"Review submitted: {result.get('review_id', '?')}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_purchase(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.list_purchases:
            purchases = client.list_purchases(limit=args.limit)
            if not purchases:
                _ok("No purchases yet.")
                return
            rows = [{"id": p.get("purchase_id", "?"), "listing": p.get("listing_id", "?"),
                      "status": p.get("status", "?"), "amount": p.get("amount_cents", 0)}
                    for p in purchases]
            _table(rows, ["id", "listing", "status", "amount"])
            return

        listing_id = args.listing_id
        license_version_id = getattr(args, "license_version_id", None)

        if is_interactive() and not listing_id:
            step("Purchase environment access")
            listing_id = text("Listing ID", required=True)
            if not license_version_id:
                license_version_id = text("License version ID", required=True)

        if not listing_id or not license_version_id:
            _err("listing_id and --license-version-id are required.")

        result = client.create_purchase(
            listing_id=listing_id,
            license_version_id=license_version_id,
        )
        _ok(f"Purchase created: {result.get('purchase_id', '?')}")
        _ok(f"  status: {result.get('status', '?')}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── env batch & export commands ──────────────────────────────────────


def cmd_env_batch_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        deployment_id = args.deployment_id
        name = args.name

        if is_interactive() and not deployment_id:
            step("Create a batch evaluation")
            deployment_id = text("Deployment ID", required=True)
            if not name:
                name = text("Batch name", required=True)

        if not deployment_id or not name:
            _err("--deployment-id and --name are required.")

        if args.file:
            pairs = json.loads(Path(args.file).read_text())
        elif args.tasks:
            pairs = [{"task_id": t, "seed": args.seed or 42} for t in args.tasks.split(",")]
        else:
            _err("--tasks or --file is required (comma-separated task IDs or JSON file).")

        result = client.create_batch(
            deployment_id=deployment_id,
            name=name,
            task_seed_pairs=pairs,
            max_credits=args.max_credits,
        )
        _ok(f"Batch created: {result.get('batch_id', '?')}")
        _ok(f"  tasks: {len(pairs)}")
        _ok(f"  run with your policy: client.drive_batch({result.get('batch_id', '?')!r}, policy_fn=policy)")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_batch_list(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        batches = client.list_batches(
            deployment_id=args.deployment_id,
            status=args.status,
            limit=args.limit,
        )
        if not batches:
            _ok("No batches found.")
            return
        rows = [{"id": b.get("batch_id", "?"), "name": b.get("name", "?"),
                  "status": b.get("status", "?"),
                  "sessions": f"{b.get('sessions_completed', 0)}/{b.get('sessions_requested', 0)}"}
                for b in batches]
        _table(rows, ["id", "name", "status", "sessions"])
        if args.json:
            _json_out(batches)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_batch_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        batch = client.get_batch(args.batch_id)
        _ok(f"Batch: {batch.get('batch_id', '?')}")
        _ok(f"  name: {batch.get('name', '?')}")
        _ok(f"  status: {batch.get('status', '?')}")
        _ok(
            "  sessions: "
            f"{batch.get('sessions_completed', 0)}/{batch.get('sessions_requested', 0)} completed, "
            f"{batch.get('sessions_failed', 0)} failed"
        )

        if args.sessions:
            sessions = client.get_batch_sessions(args.batch_id)
            _ok(f"\nSessions ({len(sessions)}):")
            for s in sessions[:20]:
                _ok(f"  {s.get('session_id', '?')}  task={s.get('task_id', '?')}  status={s.get('status', '?')}")

        if args.comparison:
            comp = client.get_batch_comparison(args.batch_id)
            _ok("\nComparison:")
            _json_out(comp)

        if args.json:
            _json_out(batch)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_batch_cancel(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.cancel_batch(args.batch_id)
        _ok(f"Batch cancelled: {args.batch_id}")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_export_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        deployment_id = args.deployment_id
        fmt = args.format

        if is_interactive() and not deployment_id:
            step("Export training data from environment sessions")
            deployment_id = text("Deployment ID", required=True)
            if not fmt:
                fmt = select("Export format", [
                    {"value": "grpo", "label": "GRPO — group relative policy optimization"},
                    {"value": "dpo", "label": "DPO — direct preference optimization"},
                    {"value": "sft", "label": "SFT — supervised fine-tuning"},
                    {"value": "jsonl", "label": "JSONL — raw session data"},
                ], default="grpo")

        if not deployment_id or not fmt:
            _err("--deployment-id and --format are required.")

        result = client.create_environment_export(
            deployment_id=deployment_id,
            format=fmt,
            filter_domain=args.domain,
        )
        _ok(f"Export job created: {result.get('export_id', '?')}")
        _ok(f"  format: {fmt}")
        _ok(f"  status: {result.get('status', 'pending')}")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_export_list(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        exports = client.list_environment_exports(
            deployment_id=args.deployment_id,
            status=args.status,
            limit=args.limit,
        )
        if not exports:
            _ok("No exports found.")
            return
        rows = [{"id": e.get("export_id", "?"), "format": e.get("format", "?"),
                  "status": e.get("status", "?")}
                for e in exports]
        _table(rows, ["id", "format", "status"])
        if args.json:
            _json_out(exports)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_env_export_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        export = client.get_environment_export(args.export_id)
        _ok(f"Export: {export.get('export_id', '?')}")
        _ok(f"  format: {export.get('format', '?')}")
        _ok(f"  status: {export.get('status', '?')}")
        if export.get("download_url"):
            _ok(f"  download: {export['download_url']}")
        if args.json:
            _json_out(export)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── run commands ─────────────────────────────────────────────────────


def cmd_run_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        model = args.model

        if is_interactive() and not model:
            step("Create an evaluation run")
            model = text("Model name (e.g. gpt-4, claude-3-opus)", required=True)

        if not model:
            _err("model is required.")

        domains = args.domains.split(",") if args.domains else None

        run = client.create_run(
            model,
            max_tasks=args.max_tasks,
            domains=domains,
            force=args.force,
        )
        _ok(f"Run created: {run.run_id}")
        _ok(f"  model: {run.target_model}")
        _ok(f"  status: {run.status}")
        _ok(f"  tasks: {run.task_count}")
        if run.estimated_credits:
            _ok(f"  estimated credits: {run.estimated_credits}")

        if args.wait:
            _ok("\nWaiting for completion...")
            run = client.wait_for_completion(
                run.run_id,
                poll_interval=args.poll_interval or 10,
                timeout=args.timeout or 3600,
            )
            _ok(f"Run {run.status}: {run.run_id}")

        if args.json:
            _json_out(run.to_dict())
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_run_list(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        runs = client.list_runs(
            status=args.status,
            limit=args.limit,
            offset=args.offset,
        )
        if not runs:
            _ok("No runs found.")
            return
        rows = [{"id": r.run_id, "model": r.target_model or "?",
                  "status": r.status, "tasks": r.task_count,
                  "progress": f"{(r.progress or 0) * 100:.0f}%"}
                for r in runs]
        _table(rows, ["id", "model", "status", "tasks", "progress"])
        if args.json:
            _json_out([r.to_dict() for r in runs])
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_run_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        run = client.get_run(args.run_id)
        _ok(f"Run: {run.run_id}")
        _ok(f"  model: {run.target_model}")
        _ok(f"  status: {run.status}")
        _ok(f"  tasks: {run.task_count}")
        _ok(f"  progress: {(run.progress or 0) * 100:.0f}%")
        if run.gap_count:
            _ok(f"  gaps found: {run.gap_count}")

        if args.gaps:
            gaps = client.get_gaps(args.run_id)
            if gaps:
                _ok(f"\nCapability gaps ({len(gaps)}):")
                for g in gaps[:20]:
                    _ok(f"  {g.capability}: alpha={g.alpha_score:.3f}  priority={g.priority}")

        if args.json:
            _json_out(run.to_dict())
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_run_cancel(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        run = client.cancel_run(args.run_id)
        _ok(f"Run cancelled: {run.run_id}")
        _ok(f"  status: {run.status}")
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_run_export(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        run_id = args.run_id
        fmt = args.format

        if is_interactive() and not fmt:
            fmt = select("Export format", [
                {"value": "grpo", "label": "GRPO — group relative policy optimization"},
                {"value": "dpo", "label": "DPO — direct preference optimization"},
                {"value": "sft", "label": "SFT — supervised fine-tuning"},
                {"value": "kto", "label": "KTO — Kahneman-Tversky optimization"},
                {"value": "jsonl", "label": "JSONL — raw results"},
                {"value": "report", "label": "Report — human-readable summary"},
            ], default="grpo")

        if not fmt:
            _err("--format is required.")

        output_path = args.output
        if args.stream:
            result = client.stream_export(
                run_id, fmt, path=output_path,
                min_score_gap=args.min_score_gap,
            )
        else:
            result = client.export_run(run_id, fmt, path=output_path)

        if output_path:
            _ok(f"Exported to {output_path}")
        elif isinstance(result, str):
            print(result)
        else:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_run_eval(args: argparse.Namespace) -> None:
    """Multi-model evaluation."""
    client = _get_client()
    try:
        models = args.models.split(",") if args.models else None

        if is_interactive() and not models:
            step("Create a multi-model evaluation")
            models_str = text("Models (comma-separated)", required=True)
            models = [m.strip() for m in models_str.split(",")]

        if not models:
            _err("--models is required (comma-separated).")

        domains = args.domains.split(",") if args.domains else None

        if args.estimate:
            est = client.estimate_evaluation_cost(
                models=models, domains=domains, max_tasks=args.max_tasks,
            )
            _ok("Cost estimate:")
            _ok(f"  tasks: {est.task_count}")
            _ok(f"  total credits: {est.total_credits}")
            _ok(f"  balance: {est.balance}")
            _ok(f"  sufficient: {est.sufficient}")
            if hasattr(est, "per_model") and est.per_model:
                for pm in est.per_model:
                    _ok(f"    {pm.model_id}: {pm.credits} credits")
            return

        result = client.create_evaluation(
            models=models,
            name=args.name,
            domains=domains,
            max_tasks=args.max_tasks,
        )
        _ok(f"Evaluation created: {result.evaluation_id}")
        _ok(f"  models: {result.total_models}")
        _ok(f"  estimated credits: {result.total_estimated_credits}")
        for run in result.runs:
            _ok(f"  run {run.run_id}: {run.model_id} ({run.status})")

        if args.wait:
            _ok("\nWaiting for all runs to complete...")
            for run in result.runs:
                summary = client.wait_for_completion(run.run_id, timeout=args.timeout or 7200)
                _ok(f"  {summary.target_model}: {summary.status}")

        if args.json:
            _json_out({"evaluation_id": result.evaluation_id,
                        "runs": [r.to_dict() for r in result.runs]})
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── rl commands ──────────────────────────────────────────────────────


def cmd_rl_envs(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.list_rl_environments(
            domain=args.domain, capability=args.capability,
            difficulty=args.difficulty, limit=args.limit,
        )
        envs = result.get("environments", result) if isinstance(result, dict) else result
        if not envs:
            _ok("No RL environments found.")
            return
        for e in envs:
            _ok(f"  {e.get('task_id', '?')}  [{e.get('domain', '')}/{e.get('difficulty', '')}]"
                f"  {(e.get('prompt', '') or '')[:60]}")
        _ok(f"\n{len(envs)} environment(s)")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_session(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        task_id = args.task_id

        if is_interactive() and not task_id:
            step("Create an RL training session")
            task_id = text("Task ID", required=True)

        if not task_id:
            _err("task_id is required.")

        t0 = time.monotonic()
        session = client.create_rl_session(
            task_id=task_id,
            env_type=args.env_type,
            reward_mode=args.reward_mode,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        elapsed = time.monotonic() - t0
        _ok(f"RL session created: {session.session_id}")
        _ok(f"  task:        {session.task_id}")
        _ok(f"  env_type:    {session.env_type}")
        _ok(f"  reward_mode: {session.reward_mode}")
        _ok(f"  status:      {session.status}")
        _ok(f"  latency:     {elapsed:.1f}s")
        if session.seed is not None:
            _ok(f"  seed:        {session.seed}")
        if session.observation:
            _ok(f"  observation: {session.observation[:300]}")
        _ok("")
        _ok(f"Next: epsilab rl step {session.session_id} \"<your action>\"")
        if args.json:
            _json_out(session.to_dict())
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_step(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        t0 = time.monotonic()
        result = client.rl_step(args.session_id, args.action)
        elapsed = time.monotonic() - t0

        reward_str = f"{result.reward:.4f}" if result.reward is not None else "-"
        status_parts = []
        if result.terminated:
            status_parts.append("TERMINAL")
        if result.truncated:
            status_parts.append("TRUNCATED")
        status_label = " | ".join(status_parts) if status_parts else "continue"

        _ok(f"Step result ({elapsed:.1f}s)")
        _ok(f"  reward:  {reward_str}")
        _ok(f"  status:  {status_label}")
        if result.observation:
            obs_lines = result.observation.split("\n")
            preview = "\n           ".join(obs_lines[:10])
            _ok(f"  observe: {preview}")
            if len(obs_lines) > 10:
                _ok(f"           ... ({len(obs_lines) - 10} more lines)")
        if result.info:
            _ok(f"  info:    {json.dumps(result.info, default=str)[:200]}")

        if result.terminated or result.truncated:
            _ok("")
            _ok("Session complete. View trajectory:")
            _ok(f"  epsilab rl trajectory {args.session_id}")
        if args.json:
            _json_out({"observation": result.observation, "reward": result.reward,
                        "terminated": result.terminated, "truncated": result.truncated,
                        "info": result.info})
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_trajectory(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.verify:
            t0 = time.monotonic()
            result = client.verify_rl_trajectory(args.session_id)
            elapsed = time.monotonic() - t0
            status_icon = "PASS" if result.get("verified") or result.get("status") == "valid" else "FAIL"
            _ok(f"Verification: {status_icon} ({elapsed:.1f}s)")
            if result.get("steps_replayed"):
                _ok(f"  steps replayed: {result['steps_replayed']}")
            if result.get("divergences"):
                _ok(f"  divergences:    {len(result['divergences'])}")
                for d in result["divergences"][:5]:
                    _ok(f"    step {d.get('step', '?')}: {d.get('field', '?')} mismatch")
            if args.json:
                _json_out(result)
            return

        traj = client.get_rl_trajectory(args.session_id)
        total = traj.total_reward
        _ok(f"Trajectory: {traj.session_id}")
        _ok(f"  task:         {traj.task_id}")
        _ok(f"  env_type:     {traj.env_type}")
        _ok(f"  steps:        {len(traj.steps)}")
        _ok(f"  total reward: {total:.4f}" if total is not None else "  total reward: -")
        _ok("")

        for s in traj.steps:
            r = s.get("reward")
            reward_str = f"{r:+.4f}" if r is not None else "     -"
            flags = []
            if s.get("terminated"):
                flags.append("TERMINAL")
            if s.get("truncated"):
                flags.append("TRUNCATED")
            latency = s.get("latency_ms")
            lat_str = f" ({latency}ms)" if latency else ""
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            _ok(f"  Step {s['step_idx']:>2}  reward={reward_str}{lat_str}{flag_str}")
            obs = s.get("observation")
            if obs and not args.json:
                obs_preview = obs[:120].replace("\n", " ")
                _ok(f"           {obs_preview}{'...' if len(obs) > 120 else ''}")

        if args.json:
            _json_out(traj.to_dict())
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_sessions(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.list_rl_sessions(
            status=args.status, task_id=args.task_id,
            limit=args.limit,
        )
        sessions = result.get("sessions", result) if isinstance(result, dict) else result
        total = result.get("total", len(sessions)) if isinstance(result, dict) else len(sessions)
        if not sessions:
            _ok("No RL sessions found.")
            return

        _ok(f"Sessions ({len(sessions)} of {total})")
        _ok("")
        rows = []
        for s in sessions:
            reward = s.get("total_reward")
            rows.append({
                "id": s.get("session_id", "?")[:12],
                "task": s.get("task_id", "?")[:40],
                "status": s.get("status", "?"),
                "steps": str(s.get("steps_taken", 0)),
                "reward": f"{reward:.3f}" if reward is not None else "-",
                "created": s.get("created_at", "?")[:19].replace("T", " "),
            })
        _table(rows, ["id", "task", "status", "steps", "reward", "created"])

        completed = sum(1 for s in sessions if s.get("status") == "completed")
        active = sum(1 for s in sessions if s.get("status") in ("active", "provisioning"))
        rewards = [s["total_reward"] for s in sessions if s.get("total_reward") is not None]
        _ok("")
        if rewards:
            _ok(f"  completed: {completed}  active: {active}  avg_reward: {sum(rewards)/len(rewards):.3f}")
        else:
            _ok(f"  completed: {completed}  active: {active}")

        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_export(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        fmt = args.format

        if is_interactive() and not fmt:
            fmt = select("Export format", [
                {"value": "grpo", "label": "GRPO — group relative policy optimization"},
                {"value": "dpo", "label": "DPO — direct preference optimization"},
                {"value": "sft", "label": "SFT — supervised fine-tuning"},
            ], default="grpo")

        result = client.export_rl_sessions(
            format=fmt or "grpo",
            domain=args.domain,
            min_score_gap=args.min_score_gap,
            limit=args.limit,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, default=str))
            _ok(f"Exported to {args.output}")
        else:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_stats(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        stats = client.get_rl_stats(domain=args.domain, env_type=args.env_type)
        _json_out(stats)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


def cmd_rl_curriculum(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.get_rl_curriculum(
            batch_size=args.batch_size,
            domain=args.domain,
            seed=args.seed,
        )
        tasks = result.get("tasks", result) if isinstance(result, dict) else result
        if isinstance(tasks, list):
            for t in tasks:
                _ok(f"  {t.get('task_id', '?')}  [{t.get('domain', '')}/{t.get('difficulty', '')}]")
            _ok(f"\n{len(tasks)} task(s) in curriculum batch")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── route command ────────────────────────────────────────────────────


def cmd_route(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        prompt = args.prompt

        if is_interactive() and not prompt:
            step("Route a prompt to the best model")
            prompt = text("Prompt", required=True)

        if not prompt:
            _err("prompt is required.")

        result = client.route(
            prompt=prompt,
            strategy=args.strategy,
            max_candidates=args.max_candidates,
            router_name=args.router,
        )

        rec = result.get("recommendation", result)
        _ok(f"Recommended model: {rec.get('model_id', '?')}")
        _ok(f"  strategy: {args.strategy}")
        if rec.get("harness"):
            _ok(f"  harness: {rec['harness']}")
        if rec.get("confidence"):
            _ok(f"  confidence: {rec['confidence']:.2f}")

        candidates = result.get("candidates", [])
        if candidates:
            _ok(f"\nCandidates ({len(candidates)}):")
            for c in candidates:
                _ok(f"  {c.get('model_id', '?')}: score={c.get('score', '?')}"
                    f"  cost={c.get('cost_usd', '?')}")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(_friendly_error(e))
    finally:
        client.close()


# ── Parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epsilab",
        description=(
            "Epsilab CLI for managing RL environments and the marketplace.\n"
            f"Dashboard: {_DASHBOARD_URL}  |  Docs: {_DASHBOARD_URL} (Documentation)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"epsilab {__version__}"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (HTTP requests, timing, debug info)",
    )
    parser.add_argument(
        "--profile", "-p",
        help="Named profile to use (default: active profile or 'default')",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── login / logout / whoami ──────────────────────────────────
    login_p = sub.add_parser(
        "login",
        help="Authenticate with your API key",
        description=f"Log in with an API key from {_DASHBOARD_URL} (Settings > API Keys)",
    )
    login_p.add_argument("--api-key", help="API key (or enter interactively)")
    login_p.add_argument("--label", help="Optional label for this key (e.g. 'production', 'ci')")
    login_p.set_defaults(func=cmd_login)

    logout_p = sub.add_parser("logout", help="Remove stored credentials")
    logout_p.set_defaults(func=cmd_logout)

    whoami_p = sub.add_parser("whoami", help="Show current authentication status")
    whoami_p.set_defaults(func=cmd_whoami)

    # ── init (top-level) ────────────────────────────────────────
    root_init_p = sub.add_parser(
        "init",
        help="Scaffold a new environment project",
        description="Create a deterministic OpenEnv project that is ready to build and deploy.",
    )
    root_init_p.add_argument("slug", nargs="?", help="Environment name (default: my-environment)")
    root_init_p.add_argument("-d", "--directory", help="Target directory")
    root_init_p.set_defaults(func=cmd_env_init)

    # ── deploy (top-level) ──────────────────────────────────────
    deploy_p = sub.add_parser(
        "deploy",
        help="Build, push, and register an environment or tool (one command)",
        description=(
            "Deploy an RL environment or Application Tool from a directory.\n"
            "Auto-detects the project type from files present:\n\n"
            "  Environment:       Dockerfile + tasks.json\n"
            "  Application Tool:  plugin.py + api.py + state.py\n\n"
            "First time: interactive setup creates the listing and saves\n"
            "config to .epsilab/project.json. Subsequent runs reuse it.\n\n"
            "  epsilab deploy              # deploy current directory\n"
            "  epsilab deploy ./my-env     # deploy a specific directory\n"
            "  epsilab deploy --no-host    # register without hosting\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    deploy_p.add_argument("directory", nargs="?", default=".", help="Environment directory (default: .)")
    deploy_p.add_argument("--version", help="Release version (default: 1.0.0)")
    deploy_p.add_argument("--namespace-id", help="Namespace ID (skips namespace selection)")
    deploy_p.add_argument(
        "--visibility",
        choices=["private", "unlisted", "shared", "public"],
        help="Visibility for a new listing, or update an existing linked listing",
    )
    deploy_p.add_argument(
        "--prod",
        dest="prod",
        action="store_true",
        default=True,
        help="Deploy for hosted execution (default)",
    )
    deploy_p.add_argument(
        "--no-host",
        dest="prod",
        action="store_false",
        help="Register the release without creating a hosted deployment",
    )
    deploy_p.add_argument("--yes", "-y", action="store_true", help="Skip prompts (use defaults)")
    deploy_p.add_argument("--force", action="store_true", help="Force rebuild even if context unchanged")
    deploy_p.set_defaults(func=cmd_deploy)

    # ── env ──────────────────────────────────────────────────────
    env_p = sub.add_parser("env", help="Manage environments")
    env_sub = env_p.add_subparsers(dest="env_command", help="Environment commands")

    # env init
    init_p = env_sub.add_parser("init", help="Scaffold a new environment project")
    init_p.add_argument("slug", nargs="?", help="Environment slug (default: my-environment)")
    init_p.add_argument("-d", "--directory", help="Target directory")
    init_p.set_defaults(func=cmd_env_init)

    # env verify
    verify_p = env_sub.add_parser(
        "verify",
        help="Run local preflight checks before pushing",
        description=(
            "Validates manifest format, file structure, Docker build, "
            "and protocol compliance locally before pushing."
        ),
    )
    verify_p.add_argument(
        "-d", "--directory", default=".",
        help="Environment project directory (default: current)",
    )
    verify_p.add_argument(
        "--manifest", default="epsilab.json",
        help="Manifest filename (default: epsilab.json)",
    )
    verify_p.add_argument(
        "--build", action="store_true",
        help="Build the Docker image and check size/digest",
    )
    verify_p.add_argument(
        "--test", action="store_true",
        help="Run a protocol smoke test (requires --build)",
    )
    verify_p.set_defaults(func=cmd_env_verify)

    # env test
    test_p = env_sub.add_parser(
        "test",
        help="Run a local multi-step session against a containerized environment",
        description=(
            "Builds the Docker image, starts the container, and runs a "
            "multi-step session locally. Exercises each tool in the "
            "observation to verify dependencies are installed and functional."
        ),
    )
    test_p.add_argument(
        "-d", "--directory", default=".",
        help="Environment project directory (default: current)",
    )
    test_p.add_argument(
        "--task", help="Task ID to test (default: first task in tasks.json)",
    )
    test_p.add_argument(
        "--steps", type=int,
        help="Max steps to run (default: from task max_steps)",
    )
    test_p.set_defaults(func=cmd_env_test)

    # env list
    list_p = env_sub.add_parser("list", help="List environment listings")
    list_p.add_argument("--limit", type=int, default=50)
    list_p.add_argument("--json", action="store_true", help="Output as JSON")
    list_p.set_defaults(func=cmd_env_list)

    # env search
    search_p = env_sub.add_parser("search", help="Search the marketplace")
    search_p.add_argument("query", nargs="?", help="Search query")
    search_p.add_argument("--domain", help="Filter by domain")
    search_p.add_argument("--min-quality", type=float, help="Minimum quality score")
    search_p.add_argument("--limit", type=int, default=20)
    search_p.add_argument("--json", action="store_true", help="Output as JSON")
    search_p.set_defaults(func=cmd_env_search)

    # env create
    create_p = env_sub.add_parser("create", help="Create a new environment listing")
    create_p.add_argument("slug", nargs="?", help="URL-safe listing slug (prompted if omitted)")
    create_p.add_argument("title", nargs="?", help="Listing title (prompted if omitted)")
    create_p.add_argument("--namespace-id", help="Namespace ID (prompted if omitted)")
    create_p.add_argument("--summary", help="Short description")
    create_p.add_argument(
        "--visibility",
        choices=["private", "unlisted", "shared", "public"],
        default="public",
    )
    create_p.set_defaults(func=cmd_env_create)

    # env push
    push_p = env_sub.add_parser(
        "push", help="Register a new environment release"
    )
    push_p.add_argument("--manifest", "-f", help="Path to epsilab.json manifest")
    push_p.add_argument("--listing-id", help="Listing ID")
    push_p.add_argument("--namespace-id", help="Namespace ID")
    push_p.add_argument("--version", help="Release version (semver)")
    push_p.add_argument("--license", default="apache-2.0", help="License ID")
    push_p.add_argument("--protocol-version", default="0.4.1")
    push_p.add_argument("--runtime-ref", help="Environment runtime OCI reference")
    push_p.add_argument("--runtime-digest", help="Runtime image SHA-256 digest")
    push_p.add_argument("--task-pack-ref", help="Task pack artifact reference")
    push_p.add_argument("--task-pack-digest", help="Task pack SHA-256 digest")
    push_p.add_argument("--verifier-ref", help="Verifier runtime reference")
    push_p.add_argument("--verifier-digest", help="Verifier runtime SHA-256 digest")
    push_p.set_defaults(func=cmd_env_push)

    # env deploy
    deploy_p = env_sub.add_parser("deploy", help="Deploy a release")
    deploy_p.add_argument("--release-id", help="Release ID to deploy (prompted if omitted)")
    deploy_p.add_argument("--listing-id", help="Listing ID (for new deployments)")
    deploy_p.add_argument("--alias", default="production", help="Deployment alias")
    deploy_p.add_argument("--deployment-id", help="Existing deployment (for revisions)")
    deploy_p.add_argument("--revision", action="store_true", help="Update existing deployment")
    deploy_p.add_argument("--export-policy", help="Export permission policy")
    deploy_p.add_argument("--json", action="store_true", help="Output as JSON")
    deploy_p.set_defaults(func=cmd_env_deploy)

    # env grant (share access)
    grant_p = env_sub.add_parser("grant", help="Share access to an environment")
    grant_p.add_argument("listing_id", help="Listing ID")
    grant_p.add_argument("tenant_id", help="Organization's tenant ID")
    grant_p.add_argument("--license", default="apache-2.0", help="License ID")
    grant_p.add_argument("--expires-at", help="Expiry date (ISO-8601)")
    grant_p.set_defaults(func=cmd_env_grant)

    # env revoke (revoke shared access)
    revoke_p = env_sub.add_parser("revoke", help="Revoke shared access")
    revoke_p.add_argument("entitlement_id", help="Entitlement ID to revoke")
    revoke_p.set_defaults(func=cmd_env_revoke)

    # env shared (list shared environments)
    shared_p = env_sub.add_parser("shared", help="List shared environments")
    shared_p.set_defaults(func=cmd_env_shared)

    # env visibility
    visibility_p = env_sub.add_parser(
        "visibility",
        help="Change an environment's hub visibility",
    )
    visibility_p.add_argument("target", help="Listing ID or <owner>/<name>")
    visibility_p.add_argument(
        "visibility",
        choices=["private", "unlisted", "shared", "public"],
    )
    visibility_p.add_argument("--json", action="store_true", help="Output as JSON")
    visibility_p.set_defaults(func=cmd_env_visibility)

    # env status
    status_p = env_sub.add_parser("status", help="Show release status and quality")
    status_p.add_argument("release_id", help="Release ID")
    status_p.set_defaults(func=cmd_env_status)

    # env qualify
    qualify_p = env_sub.add_parser("qualify", help="Start a quality report")
    qualify_p.add_argument("release_id", help="Release ID to qualify")
    qualify_p.add_argument(
        "--report-type",
        choices=[
            "protocol_conformance",
            "startup_cleanup",
            "reset_independence",
            "verifier_repeatability",
            "adversarial",
            "contamination",
            "benchmark",
            "hosted_execution",
            "full_qualification",
        ],
        default="hosted_execution",
    )
    qualify_p.add_argument("--task", help="Task ID used by the hosted smoke check")
    qualify_p.add_argument(
        "--action",
        action="append",
        help="Terminal smoke action (plain text or a JSON object; repeatable)",
    )
    qualify_p.add_argument(
        "--action-type",
        default="submit",
        help="Action type for plain-text --action values (default: submit)",
    )
    qualify_p.add_argument("--repetitions", type=int, choices=range(1, 21))
    qualify_p.add_argument("--seed", type=int)
    qualify_p.set_defaults(func=cmd_env_qualify)

    # env publish
    publish_p = env_sub.add_parser("publish", help="Publish listing to the environment hub")
    publish_p.add_argument("listing_id", nargs="?", help="Listing ID (prompted if omitted)")
    publish_p.set_defaults(func=cmd_env_publish)

    # env session
    session_p = env_sub.add_parser("session", help="Manage hosted environment sessions")
    session_sub = session_p.add_subparsers(dest="session_command", help="Session commands")

    sess_create_p = session_sub.add_parser("create", help="Create a session")
    sess_create_p.add_argument("--deployment-id", help="Deployment ID")
    sess_create_p.add_argument("--task-id", help="Task ID")
    sess_create_p.add_argument("--seed", type=int, help="Random seed")
    sess_create_p.add_argument("--json", action="store_true")
    sess_create_p.set_defaults(func=cmd_env_session_create)

    sess_step_p = session_sub.add_parser("step", help="Take an action")
    sess_step_p.add_argument("session_id", nargs="?", help="Session ID")
    sess_step_p.add_argument("action", nargs="?", help="Action string")
    sess_step_p.add_argument("--session-token", help="Session token")
    sess_step_p.add_argument("--json", action="store_true")
    sess_step_p.set_defaults(func=cmd_env_session_step)

    sess_show_p = session_sub.add_parser("show", help="Show session state")
    sess_show_p.add_argument("session_id", help="Session ID")
    sess_show_p.add_argument("--json", action="store_true")
    sess_show_p.set_defaults(func=cmd_env_session_show)

    sess_cancel_p = session_sub.add_parser("cancel", help="Cancel a session")
    sess_cancel_p.add_argument("session_id", help="Session ID")
    sess_cancel_p.add_argument("--json", action="store_true")
    sess_cancel_p.set_defaults(func=cmd_env_session_cancel)

    # env batch
    batch_p = env_sub.add_parser("batch", help="Manage batch evaluations")
    batch_sub = batch_p.add_subparsers(dest="batch_command", help="Batch commands")

    batch_create_p = batch_sub.add_parser("create", help="Create a batch evaluation")
    batch_create_p.add_argument("--deployment-id", help="Deployment ID")
    batch_create_p.add_argument("--name", help="Batch name")
    batch_create_p.add_argument("--tasks", help="Comma-separated task IDs")
    batch_create_p.add_argument("--file", "-f", help="JSON file with task-seed pairs")
    batch_create_p.add_argument("--seed", type=int, default=42, help="Default seed for tasks")
    batch_create_p.add_argument("--max-credits", type=int, help="Credit limit")
    batch_create_p.add_argument("--json", action="store_true")
    batch_create_p.set_defaults(func=cmd_env_batch_create)

    batch_list_p = batch_sub.add_parser("list", help="List batches")
    batch_list_p.add_argument("--deployment-id", help="Filter by deployment")
    batch_list_p.add_argument("--status", help="Filter by status")
    batch_list_p.add_argument("--limit", type=int, default=50)
    batch_list_p.add_argument("--json", action="store_true")
    batch_list_p.set_defaults(func=cmd_env_batch_list)

    batch_show_p = batch_sub.add_parser("show", help="Show batch details")
    batch_show_p.add_argument("batch_id", help="Batch ID")
    batch_show_p.add_argument("--sessions", action="store_true", help="Show sessions")
    batch_show_p.add_argument("--comparison", action="store_true", help="Show comparison report")
    batch_show_p.add_argument("--json", action="store_true")
    batch_show_p.set_defaults(func=cmd_env_batch_show)

    batch_cancel_p = batch_sub.add_parser("cancel", help="Cancel a batch")
    batch_cancel_p.add_argument("batch_id", help="Batch ID")
    batch_cancel_p.add_argument("--json", action="store_true")
    batch_cancel_p.set_defaults(func=cmd_env_batch_cancel)

    # env export
    export_p = env_sub.add_parser("export", help="Export training data from sessions")
    export_sub = export_p.add_subparsers(dest="export_command", help="Export commands")

    export_create_p = export_sub.add_parser("create", help="Create an export job")
    export_create_p.add_argument("--deployment-id", help="Deployment ID")
    export_create_p.add_argument("--format", help="Export format (grpo, dpo, sft, jsonl)")
    export_create_p.add_argument("--domain", help="Filter by domain")
    export_create_p.add_argument("--json", action="store_true")
    export_create_p.set_defaults(func=cmd_env_export_create)

    export_list_p = export_sub.add_parser("list", help="List export jobs")
    export_list_p.add_argument("--deployment-id", help="Filter by deployment")
    export_list_p.add_argument("--status", help="Filter by status")
    export_list_p.add_argument("--limit", type=int, default=50)
    export_list_p.add_argument("--json", action="store_true")
    export_list_p.set_defaults(func=cmd_env_export_list)

    export_show_p = export_sub.add_parser("show", help="Show export job status")
    export_show_p.add_argument("export_id", help="Export ID")
    export_show_p.add_argument("--json", action="store_true")
    export_show_p.set_defaults(func=cmd_env_export_show)

    # env review
    review_p = env_sub.add_parser("review", help="Review an environment listing")
    review_p.add_argument("listing_id", nargs="?", help="Listing ID")
    review_p.add_argument("--rating", type=int, choices=[1, 2, 3, 4, 5], help="Star rating")
    review_p.add_argument("--title", help="Review title")
    review_p.add_argument("--body", help="Review body")
    review_p.add_argument("--owner-tenant-id", help="Listing owner's tenant ID")
    review_p.add_argument("--usage-hours", type=float, help="Hours spent using environment")
    review_p.add_argument("--list", dest="list_reviews", action="store_true", help="List reviews instead")
    review_p.add_argument("--limit", type=int, default=20)
    review_p.set_defaults(func=cmd_env_review)

    # env purchase
    purchase_p = env_sub.add_parser("purchase", help="Purchase environment access")
    purchase_p.add_argument("listing_id", nargs="?", help="Listing ID")
    purchase_p.add_argument("--license-version-id", help="License version ID")
    purchase_p.add_argument("--list", dest="list_purchases", action="store_true", help="List purchases")
    purchase_p.add_argument("--limit", type=int, default=50)
    purchase_p.add_argument("--json", action="store_true")
    purchase_p.set_defaults(func=cmd_env_purchase)

    # ── run ──────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run an environment or manage evaluation runs",
        description=(
            "Run a hosted environment by owner/name, or use an evaluation-run subcommand.\n\n"
            "  epsilab run epsilab/bug-hunter\n"
            "  epsilab run epsilab/bug-hunter --action '<answer>'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_sub = run_p.add_subparsers(
        dest="run_command",
        help="Run commands",
        metavar="{create,list,show,cancel,export,eval}",
    )

    run_environment_p = run_sub.add_parser("__environment", prog="epsilab run")
    run_environment_p.add_argument("target", help="Environment as <owner>/<name>")
    run_environment_p.add_argument("--action", help="Take one action and exit")
    run_environment_p.add_argument(
        "--action-type",
        default="submit",
        help="Action type for plain-text input, such as submit or fill (default: submit)",
    )
    run_environment_p.add_argument("--task", help="Task ID (default: first train task)")
    run_environment_p.add_argument("--seed", type=int, default=42, help="Episode seed")
    run_environment_p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for session readiness (default: 120)",
    )
    run_environment_p.add_argument("--json", action="store_true", help="Output one-step result as JSON")
    run_environment_p.set_defaults(func=cmd_run_environment)

    run_create_p = run_sub.add_parser("create", help="Create a single-model evaluation run")
    run_create_p.add_argument("model", nargs="?", help="Model name (e.g. gpt-4)")
    run_create_p.add_argument("--max-tasks", type=int, help="Max tasks to evaluate")
    run_create_p.add_argument("--domains", help="Comma-separated domains")
    run_create_p.add_argument("--force", action="store_true", help="Force re-evaluation")
    run_create_p.add_argument("--wait", action="store_true", help="Wait for completion")
    run_create_p.add_argument("--poll-interval", type=int, help="Seconds between polls")
    run_create_p.add_argument("--timeout", type=int, help="Max wait seconds")
    run_create_p.add_argument("--json", action="store_true")
    run_create_p.set_defaults(func=cmd_run_create)

    run_list_p = run_sub.add_parser("list", help="List runs")
    run_list_p.add_argument("--status", help="Filter by status")
    run_list_p.add_argument("--limit", type=int, default=20)
    run_list_p.add_argument("--offset", type=int, default=0)
    run_list_p.add_argument("--json", action="store_true")
    run_list_p.set_defaults(func=cmd_run_list)

    run_show_p = run_sub.add_parser("show", help="Show run details and gaps")
    run_show_p.add_argument("run_id", help="Run ID")
    run_show_p.add_argument("--gaps", action="store_true", help="Show capability gaps")
    run_show_p.add_argument("--json", action="store_true")
    run_show_p.set_defaults(func=cmd_run_show)

    run_cancel_p = run_sub.add_parser("cancel", help="Cancel a run")
    run_cancel_p.add_argument("run_id", help="Run ID")
    run_cancel_p.set_defaults(func=cmd_run_cancel)

    run_export_p = run_sub.add_parser("export", help="Export training data from a run")
    run_export_p.add_argument("run_id", help="Run ID")
    run_export_p.add_argument("--format", help="Export format (grpo, dpo, sft, kto, jsonl, report)")
    run_export_p.add_argument("--output", "-o", help="Output file path")
    run_export_p.add_argument("--stream", action="store_true", help="Use streaming export")
    run_export_p.add_argument("--min-score-gap", type=float, help="Minimum score gap filter")
    run_export_p.set_defaults(func=cmd_run_export)

    run_eval_p = run_sub.add_parser("eval", help="Multi-model evaluation")
    run_eval_p.add_argument("--models", help="Comma-separated model names")
    run_eval_p.add_argument("--name", help="Evaluation name")
    run_eval_p.add_argument("--domains", help="Comma-separated domains")
    run_eval_p.add_argument("--max-tasks", type=int, help="Max tasks per model")
    run_eval_p.add_argument("--estimate", action="store_true", help="Estimate cost only")
    run_eval_p.add_argument("--wait", action="store_true", help="Wait for completion")
    run_eval_p.add_argument("--timeout", type=int, help="Max wait seconds")
    run_eval_p.add_argument("--json", action="store_true")
    run_eval_p.set_defaults(func=cmd_run_eval)

    # ── rl ───────────────────────────────────────────────────────
    rl_p = sub.add_parser("rl", help="RL training environments")
    rl_sub = rl_p.add_subparsers(dest="rl_command", help="RL commands")

    rl_envs_p = rl_sub.add_parser("envs", help="List available RL environments")
    rl_envs_p.add_argument("--domain", help="Filter by domain")
    rl_envs_p.add_argument("--capability", help="Filter by capability")
    rl_envs_p.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    rl_envs_p.add_argument("--limit", type=int, default=50)
    rl_envs_p.add_argument("--json", action="store_true")
    rl_envs_p.set_defaults(func=cmd_rl_envs)

    rl_session_p = rl_sub.add_parser("session", help="Create an RL session")
    rl_session_p.add_argument("task_id", nargs="?", help="Task ID")
    rl_session_p.add_argument("--env-type", default="single_turn", help="Environment type")
    rl_session_p.add_argument("--reward-mode", default="continuous", help="Reward mode")
    rl_session_p.add_argument("--seed", type=int, help="Random seed")
    rl_session_p.add_argument("--max-steps", type=int, help="Max steps")
    rl_session_p.add_argument("--json", action="store_true")
    rl_session_p.set_defaults(func=cmd_rl_session)

    rl_step_p = rl_sub.add_parser("step", help="Take a step in an RL session")
    rl_step_p.add_argument("session_id", help="Session ID")
    rl_step_p.add_argument("action", help="Action string")
    rl_step_p.add_argument("--json", action="store_true")
    rl_step_p.set_defaults(func=cmd_rl_step)

    rl_traj_p = rl_sub.add_parser("trajectory", help="Get or verify a trajectory")
    rl_traj_p.add_argument("session_id", help="Session ID")
    rl_traj_p.add_argument("--verify", action="store_true", help="Verify trajectory integrity")
    rl_traj_p.add_argument("--json", action="store_true")
    rl_traj_p.set_defaults(func=cmd_rl_trajectory)

    rl_sessions_p = rl_sub.add_parser("sessions", help="List RL sessions")
    rl_sessions_p.add_argument("--status", help="Filter by status")
    rl_sessions_p.add_argument("--task-id", help="Filter by task")
    rl_sessions_p.add_argument("--limit", type=int, default=50)
    rl_sessions_p.add_argument("--json", action="store_true")
    rl_sessions_p.set_defaults(func=cmd_rl_sessions)

    rl_export_p = rl_sub.add_parser("export", help="Export RL sessions as training data")
    rl_export_p.add_argument("--format", help="Export format (grpo, dpo, sft)")
    rl_export_p.add_argument("--domain", help="Filter by domain")
    rl_export_p.add_argument("--min-score-gap", type=float, help="Min score gap")
    rl_export_p.add_argument("--limit", type=int, help="Max sessions")
    rl_export_p.add_argument("--output", "-o", help="Output file path")
    rl_export_p.set_defaults(func=cmd_rl_export)

    rl_stats_p = rl_sub.add_parser("stats", help="Show RL statistics")
    rl_stats_p.add_argument("--domain", help="Filter by domain")
    rl_stats_p.add_argument("--env-type", help="Filter by environment type")
    rl_stats_p.set_defaults(func=cmd_rl_stats)

    rl_curriculum_p = rl_sub.add_parser("curriculum", help="Get adaptive curriculum batch")
    rl_curriculum_p.add_argument("--batch-size", type=int, default=64)
    rl_curriculum_p.add_argument("--domain", help="Filter by domain")
    rl_curriculum_p.add_argument("--seed", type=int, help="Random seed")
    rl_curriculum_p.add_argument("--json", action="store_true")
    rl_curriculum_p.set_defaults(func=cmd_rl_curriculum)

    # ── route ────────────────────────────────────────────────────
    route_p = sub.add_parser("route", help="Route a prompt to the best model")
    route_p.add_argument("prompt", nargs="?", help="The prompt to route")
    route_p.add_argument("--strategy", default="quality_first",
                         choices=["quality_first", "cost_first", "balanced"],
                         help="Routing strategy")
    route_p.add_argument("--max-candidates", type=int, default=5)
    route_p.add_argument("--router", help="Named router to use")
    route_p.add_argument("--json", action="store_true")
    route_p.set_defaults(func=cmd_route)

    # ── task ─────────────────────────────────────────────────────
    task_p = sub.add_parser("task", help="Manage tasks")
    task_sub = task_p.add_subparsers(dest="task_command", help="Task commands")

    task_create_p = task_sub.add_parser("create", help="Create a task")
    task_create_p.add_argument("task_id", nargs="?", help="Task ID")
    task_create_p.add_argument("--prompt", help="Task prompt text")
    task_create_p.add_argument("--domain", help="Task domain (e.g. economics)")
    task_create_p.add_argument("--capability", help="Task capability (e.g. resource_management)")
    task_create_p.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Difficulty level")
    task_create_p.add_argument("--expected-answer", help="Expected answer text")
    task_create_p.add_argument("--file", "-f", help="JSON file with task(s) — single object or array")
    task_create_p.set_defaults(func=cmd_task_create)

    task_list_p = task_sub.add_parser("list", help="List tasks")
    task_list_p.add_argument("--domain", help="Filter by domain")
    task_list_p.add_argument("--limit", type=int, default=50)
    task_list_p.add_argument("--offset", type=int, default=0)
    task_list_p.set_defaults(func=cmd_task_list)

    # ── namespace ────────────────────────────────────────────────
    ns_p = sub.add_parser("namespace", help="Manage namespaces")
    ns_sub = ns_p.add_subparsers(dest="ns_command", help="Namespace commands")

    ns_create_p = ns_sub.add_parser("create", help="Create a namespace")
    ns_create_p.add_argument("slug", nargs="?", help="Namespace slug (prompted if omitted)")
    ns_create_p.add_argument("--display-name", help="Human-readable name")
    ns_create_p.set_defaults(func=cmd_namespace_create)

    # ── profile ──────────────────────────────────────────────────
    profile_p = sub.add_parser("profile", help="Manage your creator profile")
    profile_sub = profile_p.add_subparsers(dest="profile_command")

    profile_show_p = profile_sub.add_parser("show", help="Show your creator profile")
    profile_show_p.add_argument("--json", action="store_true")
    profile_show_p.set_defaults(func=cmd_profile_show)

    profile_create_p = profile_sub.add_parser("create", help="Create your creator profile")
    profile_create_p.add_argument("display_name", nargs="?", help="Public display name (prompted if omitted)")
    profile_create_p.add_argument("--bio", help="Short biography")
    profile_create_p.add_argument("--website", help="Website URL")
    profile_create_p.add_argument("--email", help="Contact email")
    profile_create_p.set_defaults(func=cmd_profile_create)

    return parser


def _normalize_cli_argv(argv: List[str]) -> List[str]:
    """Map the public environment-run syntax onto the existing run group."""
    normalized = list(argv)
    index = 0
    while index < len(normalized):
        token = normalized[index]
        if token in {"--profile", "-p"}:
            index += 2
            continue
        if token in {"--verbose", "-v"}:
            index += 1
            continue
        if token == "run":
            next_index = index + 1
            if next_index == len(normalized):
                normalized.extend(["__environment", "--help"])
                break
            if normalized[next_index] in {"-h", "--help"}:
                normalized.insert(next_index, "__environment")
                break
            if (
                next_index < len(normalized)
                and "/" in normalized[next_index]
                and not normalized[next_index].startswith("-")
            ):
                normalized.insert(next_index, "__environment")
            break
        break
    return normalized


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed_argv = _normalize_cli_argv(list(argv) if argv is not None else sys.argv[1:])
    args = parser.parse_args(parsed_argv)

    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    if not args.command:
        parser.print_help()
        print(f"\nGet started: {_DASHBOARD_URL}")
        sys.exit(0)

    if args.command == "env" and not getattr(args, "env_command", None):
        parser.parse_args(["env", "--help"])
        sys.exit(0)

    if args.command == "run" and not getattr(args, "run_command", None):
        parser.parse_args(["run", "--help"])
        sys.exit(0)

    if args.command == "rl" and not getattr(args, "rl_command", None):
        parser.parse_args(["rl", "--help"])
        sys.exit(0)

    if args.command == "namespace" and not getattr(args, "ns_command", None):
        parser.parse_args(["namespace", "--help"])
        sys.exit(0)

    if args.command == "profile" and not getattr(args, "profile_command", None):
        parser.parse_args(["profile", "--help"])
        sys.exit(0)

    global_profile = getattr(args, "profile", None)
    if global_profile:
        os.environ["EPSILAB_PROFILE"] = global_profile

    func = getattr(args, "func", None)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
