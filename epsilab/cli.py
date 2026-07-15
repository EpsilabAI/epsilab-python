"""Epsilab CLI for managing environments and the marketplace.

Usage::

    epsilab login
    epsilab whoami
    epsilab env list
    epsilab env push ...
    epsilab env deploy ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .client import EpsilabClient
from .exceptions import ApiError, AuthError, EpsilabError

_CONFIG_DIR = Path.home() / ".epsilab"
_CONFIG_FILE = _CONFIG_DIR / "credentials.json"

_DEFAULT_PROFILE = "default"

_DASHBOARD_URL = "https://epsilab.com"
_DOCS_URL = "https://epsilab.com"  # docs are in-app


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


def _err(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str) -> None:
    print(msg)


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


def cmd_login(args: argparse.Namespace) -> None:
    api_key = args.api_key
    if not api_key:
        _ok(f"Create an API key at {_DASHBOARD_URL} (Settings > API Keys)")
        api_key = input("Enter your Epsilab API key: ").strip()
    if not api_key:
        _err("API key cannot be empty.")

    profile_name = getattr(args, "profile", None) or _DEFAULT_PROFILE

    client = EpsilabClient(api_key=api_key)
    try:
        client.get_usage()
        client.close()
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
    _ok(f"Authenticated as profile '{profile_name}'.")
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
        _err(str(e))
    finally:
        client.close()


# ── env commands ─────────────────────────────────────────────────────


def cmd_env_list(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        listings = client.list_environment_listings(limit=args.limit)
        if args.json:
            _json_out([l.to_dict() for l in listings])
        else:
            _ok(f"Your environments ({len(listings)}):\n")
            if not listings:
                _ok("  No environments yet. Get started:")
                _ok("    epsilab env init my-env")
                _ok(f"  Or browse the marketplace at {_DASHBOARD_URL}\n")
            rows = [
                {
                    "id": l.listing_id,
                    "slug": l.slug,
                    "title": l.title,
                    "visibility": l.visibility,
                    "status": l.moderation_state,
                }
                for l in listings
            ]
            _table(rows, ["id", "slug", "title", "visibility", "status"])
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_search(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        results = client.search_environments(
            query=args.query,
            domain=args.domain,
            min_quality_score=args.min_quality,
            limit=args.limit,
        )
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
        _err(str(e))
    finally:
        client.close()


def cmd_env_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if not args.namespace_id:
            _err(
                "--namespace-id is required. Create one with:\n"
                "  epsilab namespace create <slug>\n"
                f"Or manage namespaces at {_DASHBOARD_URL} (My Environments)"
            )
        listing = client.create_listing(
            namespace_id=args.namespace_id,
            slug=args.slug,
            title=args.title,
            summary=args.summary,
            visibility=args.visibility,
        )
        _ok(f"Created listing: {listing.listing_id}")
        _ok(f"  slug: {listing.slug}")
        _ok(f"  title: {listing.title}")
        _ok(f"  visibility: {listing.visibility}")
        _ok(f"\nManage this listing at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_push(args: argparse.Namespace) -> None:
    """Register a new environment release from a manifest file or CLI args."""
    import re as _re

    client = _get_client()
    try:
        if args.manifest:
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                _err(f"Manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {}

        listing_id = args.listing_id or manifest.get("listing_id")
        if not listing_id:
            _err("--listing-id is required (or set listing_id in manifest).")

        version = args.version or manifest.get("release_version")
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

        tp = client.create_task_pack_release(
            namespace_id=namespace_id,
            name=tp_config.get("name", f"{listing_id}-tasks"),
            release_version=version,
            artifact_ref=args.task_pack_ref or tp_config.get("artifact_ref", ""),
            artifact_digest=args.task_pack_digest or tp_config.get("artifact_digest", ""),
            usage_policy=tp_config.get("usage_policy", "open"),
            license_id=args.license or tp_config.get("license_id", "apache-2.0"),
            members=tp_config.get("members"),
        )
        _ok(f"  Task pack registered: {tp.get('release_id', tp.get('id', '?'))}")

        ver = client.create_verifier_release(
            namespace_id=namespace_id,
            name=ver_config.get("name", f"{listing_id}-verifier"),
            release_version=version,
            runtime_ref=args.verifier_ref or ver_config.get("runtime_ref", ""),
            runtime_digest=args.verifier_digest or ver_config.get("runtime_digest", ""),
            source_digest=ver_config.get("source_digest", ""),
            evidence_schema_digest=ver_config.get("evidence_schema_digest", ""),
            reward_mode=ver_config.get("reward_mode", "binary"),
        )
        _ok(f"  Verifier registered: {ver.get('release_id', ver.get('id', '?'))}")

        release = client.create_environment_release(
            listing_id=listing_id,
            release_version=version,
            protocol_version=env_config.get("protocol_version", args.protocol_version or "0.4.1"),
            runtime_ref=args.runtime_ref or env_config.get("runtime_ref", ""),
            runtime_digest=args.runtime_digest or env_config.get("runtime_digest", ""),
            task_pack_release_id=str(tp.get("release_id", tp.get("id", ""))),
            verifier_release_id=str(ver.get("release_id", ver.get("id", ""))),
            action_schema_digest=env_config.get("action_schema_digest", ""),
            observation_schema_digest=env_config.get("observation_schema_digest", ""),
            resource_policy=env_config.get("resource_policy"),
        )
        _ok(f"  Environment release registered: {release.release_id}")
        _ok(f"\nPushed v{version} successfully.")
        _ok(f"Release ID: {release.release_id}")
        _ok(f"Status: {release.status}")

        if release.status == "quarantined":
            _ok("\nThe release is quarantined pending qualification.")
            _ok("Run a quality report to qualify it:")
            _ok(f"  epsilab env qualify {release.release_id}")
        _ok(f"\nView release details at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_deploy(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.revision:
            dep = client.create_deployment_revision(
                args.deployment_id,
                environment_release_id=args.release_id,
                export_policy=args.export_policy,
            )
            _ok(f"Deployment revised: {dep.get('deployment_id', '?')}")
        else:
            if not args.listing_id:
                _err("--listing-id is required for new deployments.")
            dep = client.create_deployment(
                listing_id=args.listing_id,
                alias=args.alias or "production",
                environment_release_id=args.release_id,
                export_policy=args.export_policy,
            )
            _ok(f"Deployed: {dep.get('deployment_id', '?')}")
        _ok(f"  alias: {dep.get('alias', '-')}")
        _ok(f"  release: {args.release_id}")
        _ok(f"\nManage deployments at {_DASHBOARD_URL} (My Environments)")
        if args.json:
            _json_out(dep)
    except EpsilabError as e:
        _err(str(e))
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
        _ok(f"Entitlement granted: {ent.get('entitlement_id', '?')}")
        _ok(f"  tenant: {args.tenant_id}")
        _ok(f"  listing: {args.listing_id}")
        _ok(f"\nManage entitlements at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_status(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        release = client.get_environment_release(args.release_id)
        _ok(f"Release: {release.release_id}")
        _ok(f"  version: {release.release_version}")
        _ok(f"  protocol: {release.protocol_version}")
        _ok(f"  status: {release.status}")
        if release.content_digest:
            _ok(f"  digest: {release.content_digest}")
        if release.created_at:
            _ok(f"  created: {release.created_at}")

        badges = client.list_quality_badges(release_id=args.release_id, limit=10)
        if badges:
            _ok(f"\n  Quality badges:")
            for b in badges:
                _ok(f"    - {b.get('badge_type', '?')} ({b.get('status', '?')})")
        else:
            _ok("\n  No quality badges yet. Run a qualification report:")
            _ok(f"    epsilab env qualify {args.release_id}")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_qualify(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        report = client.create_quality_report(
            release_id=args.release_id,
            report_type=args.report_type,
        )
        _ok(f"Quality report started: {report.get('report_id', '?')}")
        _ok(f"  type: {args.report_type}")
        _ok(f"  status: {report.get('status', 'pending')}")
        _ok(f"\nCheck progress with:")
        _ok(f"  epsilab env status {args.release_id}")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


# ── namespace commands ───────────────────────────────────────────────


def cmd_namespace_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        ns = client.create_namespace(
            slug=args.slug,
            display_name=args.display_name or args.slug,
        )
        ns_id = ns.get("namespace_id", "?")
        _ok(f"Namespace created: {ns_id}")
        _ok(f"  slug: {args.slug}")
        _ok(f"\nNext step: create a listing in this namespace:")
        _ok(f"  epsilab env create --namespace-id {ns_id} <slug> \"<title>\"")
    except EpsilabError as e:
        _err(str(e))
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
            _ok(f"Creator Profile:")
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
        _err(str(e))
    finally:
        client.close()


def cmd_profile_create(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        profile = client.create_creator_profile(
            display_name=args.display_name,
            bio=args.bio,
            website_url=args.website,
            contact_email=args.email,
        )
        _ok(f"Creator profile created: {profile.get('display_name', '?')}")
        _ok(f"\nManage your profile at {_DASHBOARD_URL} (Settings > Profile)")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


# ── env init ─────────────────────────────────────────────────────────


_MANIFEST_TEMPLATE = """\
{
  "namespace_id": "",
  "listing_id": "",
  "release_version": "0.1.0",
  "task_pack": {
    "name": "%(slug)s-tasks",
    "artifact_ref": "",
    "artifact_digest": "",
    "usage_policy": "open",
    "license_id": "apache-2.0"
  },
  "verifier": {
    "name": "%(slug)s-verifier",
    "runtime_ref": "",
    "runtime_digest": "",
    "source_digest": "",
    "evidence_schema_digest": "",
    "reward_mode": "binary"
  },
  "environment": {
    "protocol_version": "0.4.1",
    "runtime_ref": "",
    "runtime_digest": "",
    "action_schema_digest": "",
    "observation_schema_digest": ""
  }
}
"""

_DOCKERFILE_TEMPLATE = """\
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

EXPOSE 8080
CMD ["python", "server.py"]
"""

_SERVER_TEMPLATE = """\
\"\"\"Minimal environment server implementing the Epsilab protocol.

The environment must expose:
    POST /reset   -> {\"observation\": str}
    POST /step    -> {\"observation\": str, \"reward\": float,
                      \"terminated\": bool, \"truncated\": bool}
\"\"\"

import json
from http.server import HTTPServer, BaseHTTPRequestHandler


class EnvironmentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        if self.path == "/reset":
            response = {"observation": "implement your initial observation here"}
        elif self.path == "/step":
            action = body.get("action", "")
            response = {
                "observation": f"received action: {action}",
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), EnvironmentHandler)
    print("Environment server running on :8080")
    server.serve_forever()
"""


def cmd_env_verify(args: argparse.Namespace) -> None:
    """Run local preflight checks on an environment project before pushing."""
    import hashlib
    import re
    import subprocess
    import time
    import urllib.request

    target = Path(args.directory or ".")
    manifest_path = target / (args.manifest or "epsilab.json")
    dockerfile = target / "Dockerfile"
    server_py = target / "server.py"
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    _ok("Verifying environment project...\n")

    # ── 1. Manifest checks ────────────────────────────────────────
    if not manifest_path.exists():
        errors.append(f"Manifest not found: {manifest_path}")
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
                if ref and not ref.startswith("oci://"):
                    warnings.append(f"environment.runtime_ref does not start with oci:// — may not be accepted")
                elif ref:
                    passed.append("environment.runtime_ref looks like a valid OCI reference")

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
    if server_py.exists():
        passed.append("server.py exists")
        source = server_py.read_text()
        if "/reset" not in source:
            errors.append("server.py does not contain a /reset endpoint")
        else:
            passed.append("server.py references /reset endpoint")
        if "/step" not in source:
            errors.append("server.py does not contain a /step endpoint")
        else:
            passed.append("server.py references /step endpoint")
        if "8080" not in source and "PORT" not in source:
            warnings.append("server.py does not reference port 8080 or PORT")
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
                if size_mb > 4096:
                    warnings.append(f"Image is {size_mb:.0f} MB — consider optimizing (>4 GB)")

            digest_out = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", tag],
                capture_output=True, text=True,
            )
            if digest_out.returncode == 0:
                digest = digest_out.stdout.strip()
                _ok(f"  Image digest: {digest}")

    # ── 4. Protocol smoke test (if --test or --build) ─────────────
    if args.test and image_tag:
        _ok("Running protocol smoke test...")
        container_id = None
        try:
            run_result = subprocess.run(
                ["docker", "run", "-d", "--rm", "-p", "18080:8080", image_tag],
                capture_output=True, text=True, timeout=30,
            )
            if run_result.returncode != 0:
                errors.append(f"Failed to start container: {run_result.stderr}")
            else:
                container_id = run_result.stdout.strip()
                time.sleep(3)

                try:
                    req = urllib.request.Request(
                        "http://localhost:18080/reset",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        reset_body = json.loads(resp.read())
                        if "observation" not in reset_body:
                            errors.append("POST /reset response missing 'observation' field")
                        else:
                            passed.append("POST /reset returns valid response with 'observation'")
                except Exception as e:
                    errors.append(f"POST /reset failed: {e}")

                try:
                    req = urllib.request.Request(
                        "http://localhost:18080/step",
                        data=json.dumps({"action": "test_action"}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        step_body = json.loads(resp.read())
                        required = {"observation", "reward", "terminated", "truncated"}
                        missing = required - set(step_body.keys())
                        if missing:
                            errors.append(f"POST /step response missing fields: {missing}")
                        else:
                            passed.append("POST /step returns all required fields")
                        if "reward" in step_body:
                            if not isinstance(step_body["reward"], (int, float)):
                                errors.append(f"reward must be numeric, got {type(step_body['reward']).__name__}")
                        if "terminated" in step_body:
                            if not isinstance(step_body["terminated"], bool):
                                errors.append(f"terminated must be bool, got {type(step_body['terminated']).__name__}")
                        if "truncated" in step_body:
                            if not isinstance(step_body["truncated"], bool):
                                errors.append(f"truncated must be bool, got {type(step_body['truncated']).__name__}")
                except Exception as e:
                    errors.append(f"POST /step failed: {e}")

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
                            "the manifest digest should match the pushed registry image"
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


def cmd_env_init(args: argparse.Namespace) -> None:
    slug = args.slug or "my-environment"
    target = Path(args.directory or slug)

    if target.exists() and any(target.iterdir()):
        _err(f"Directory '{target}' already exists and is not empty.")

    target.mkdir(parents=True, exist_ok=True)

    (target / "epsilab.json").write_text(_MANIFEST_TEMPLATE % {"slug": slug})
    (target / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE)
    (target / "server.py").write_text(_SERVER_TEMPLATE)
    (target / "requirements.txt").write_text("")

    _ok(f"Initialized environment project in {target}/")
    _ok(f"")
    _ok(f"  {target}/epsilab.json   — release manifest (fill in refs and digests)")
    _ok(f"  {target}/Dockerfile     — container image template")
    _ok(f"  {target}/server.py      — minimal environment server")
    _ok(f"")
    _ok(f"Next steps:")
    _ok(f"  1. Implement your environment logic in server.py")
    _ok(f"  2. Build and push your container image")
    _ok(f"  3. Fill in epsilab.json with image refs and content digests")
    _ok(f"  4. Create a namespace:  epsilab namespace create {slug}")
    _ok(f"  5. Create a listing:    epsilab env create --namespace-id <ns-id> {slug} \"My Environment\"")
    _ok(f"  6. Push a release:      epsilab env push --manifest epsilab.json --listing-id <listing-id>")
    _ok(f"  7. Deploy:              epsilab env deploy --listing-id <listing-id> --release-id <rel-id>")
    _ok(f"")
    _ok(f"Documentation:  {_DASHBOARD_URL} (Documentation > Quick Start)")
    _ok(f"Dashboard:      {_DASHBOARD_URL} (My Environments)")


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

    # env list
    list_p = env_sub.add_parser("list", help="List your environment listings")
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
    create_p.add_argument("slug", help="URL-safe listing slug")
    create_p.add_argument("title", help="Listing title")
    create_p.add_argument("--namespace-id", required=True, help="Namespace ID")
    create_p.add_argument("--summary", help="Short description")
    create_p.add_argument(
        "--visibility",
        choices=["private", "unlisted", "public"],
        default="private",
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
    deploy_p.add_argument("--release-id", required=True, help="Release ID to deploy")
    deploy_p.add_argument("--listing-id", help="Listing ID (for new deployments)")
    deploy_p.add_argument("--alias", default="production", help="Deployment alias")
    deploy_p.add_argument("--deployment-id", help="Existing deployment (for revisions)")
    deploy_p.add_argument("--revision", action="store_true", help="Update existing deployment")
    deploy_p.add_argument("--export-policy", help="Export permission policy")
    deploy_p.add_argument("--json", action="store_true", help="Output as JSON")
    deploy_p.set_defaults(func=cmd_env_deploy)

    # env grant
    grant_p = env_sub.add_parser("grant", help="Grant access to a buyer")
    grant_p.add_argument("listing_id", help="Listing ID")
    grant_p.add_argument("tenant_id", help="Buyer's tenant ID")
    grant_p.add_argument("--license", default="apache-2.0", help="License ID")
    grant_p.add_argument("--expires-at", help="Expiry date (ISO-8601)")
    grant_p.set_defaults(func=cmd_env_grant)

    # env status
    status_p = env_sub.add_parser("status", help="Show release status and quality")
    status_p.add_argument("release_id", help="Release ID")
    status_p.set_defaults(func=cmd_env_status)

    # env qualify
    qualify_p = env_sub.add_parser("qualify", help="Start a quality report")
    qualify_p.add_argument("release_id", help="Release ID to qualify")
    qualify_p.add_argument(
        "--report-type",
        choices=["qualification", "regression", "benchmark"],
        default="qualification",
    )
    qualify_p.set_defaults(func=cmd_env_qualify)

    # ── namespace ────────────────────────────────────────────────
    ns_p = sub.add_parser("namespace", help="Manage namespaces")
    ns_sub = ns_p.add_subparsers(dest="ns_command", help="Namespace commands")

    ns_create_p = ns_sub.add_parser("create", help="Create a namespace")
    ns_create_p.add_argument("slug", help="Namespace slug (3-64 chars)")
    ns_create_p.add_argument("--display-name", help="Human-readable name")
    ns_create_p.set_defaults(func=cmd_namespace_create)

    # ── profile ──────────────────────────────────────────────────
    profile_p = sub.add_parser("profile", help="Manage your creator profile")
    profile_sub = profile_p.add_subparsers(dest="profile_command")

    profile_show_p = profile_sub.add_parser("show", help="Show your creator profile")
    profile_show_p.add_argument("--json", action="store_true")
    profile_show_p.set_defaults(func=cmd_profile_show)

    profile_create_p = profile_sub.add_parser("create", help="Create your creator profile")
    profile_create_p.add_argument("display_name", help="Public display name")
    profile_create_p.add_argument("--bio", help="Short biography")
    profile_create_p.add_argument("--website", help="Website URL")
    profile_create_p.add_argument("--email", help="Contact email")
    profile_create_p.set_defaults(func=cmd_profile_create)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        print(f"\nGet started: {_DASHBOARD_URL}")
        sys.exit(0)

    if args.command == "env" and not getattr(args, "env_command", None):
        parser.parse_args(["env", "--help"])
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
