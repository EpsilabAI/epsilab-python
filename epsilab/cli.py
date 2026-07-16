"""Epsilab CLI for managing environments, evaluations, and the marketplace.

Usage::

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
import os
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from ._prompt import confirm, info, is_interactive, select, select_or_create, status, step, text
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
    _ok(f"  Opening browser for authentication...")
    webbrowser.open(auth_url)
    _ok(f"  Waiting for login (press Ctrl+C to cancel)...\n")

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
                {"value": "private", "label": "Private — only you and grantees"},
                {"value": "unlisted", "label": "Unlisted — accessible via direct link"},
                {"value": "public", "label": "Public — visible in marketplace"},
            ], default=visibility or "private")

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
        _ok(f"\nNext steps:")
        _ok(f"  epsilab env push --manifest epsilab.json --listing-id {listing.listing_id}")
        _ok(f"\nManage this listing at {_DASHBOARD_URL} (My Environments)")
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def _deterministic_idem_key(prefix: str, **fields: object) -> str:
    """Derive a stable idempotency key from content fields."""
    import hashlib
    canonical = json.dumps(fields, sort_keys=True, default=str)
    h = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    return f"{prefix}-{h}"


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
        ver_reward_mode = ver_config.get("reward_mode", "binary")

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
        env_resource_policy = env_config.get("resource_policy")

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
            idempotency_key=env_idem,
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
        _err(str(e))
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
        _err(str(e))
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
        _ok(f"\nNext step: create a listing in this namespace:")
        _ok(f"  epsilab env create --namespace-id {ns_id} <slug> \"<title>\"")

        if is_interactive() and confirm("Create a listing in this namespace now?"):
            listing_slug = text("Listing slug", required=True)
            listing_title = text("Listing title", required=True)
            listing_summary = text("Short description", default="")
            visibility = select("Visibility", [
                {"value": "private", "label": "Private — only you and grantees"},
                {"value": "unlisted", "label": "Unlisted — accessible via direct link"},
                {"value": "public", "label": "Public — visible in marketplace"},
            ], default="private")

            listing = client.create_listing(
                namespace_id=str(ns_id),
                slug=listing_slug,
                title=listing_title,
                summary=listing_summary or None,
                visibility=visibility,
            )
            _ok(f"\nCreated listing: {listing.listing_id}")
            _ok(f"  slug: {listing.slug}")
            _ok(f"\nReady to push:")
            _ok(f"  epsilab env push --manifest epsilab.json --listing-id {listing.listing_id}")
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
    "usage_policy": "training",
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
    slug = args.slug
    target_dir = args.directory

    if is_interactive() and not slug:
        step("Initialize a new environment project")
        slug = text("Environment slug", default="my-environment", required=True)
        if not target_dir:
            target_dir = text("Project directory", default=slug)

    slug = slug or "my-environment"
    target = Path(target_dir or slug)

    if target.exists() and any(target.iterdir()):
        if is_interactive():
            if not confirm(f"Directory '{target}' is not empty. Continue anyway?", default=False):
                sys.exit(0)
        else:
            _err(f"Directory '{target}' already exists and is not empty.")

    target.mkdir(parents=True, exist_ok=True)

    (target / "epsilab.json").write_text(_MANIFEST_TEMPLATE % {"slug": slug})
    (target / "Dockerfile").write_text(_DOCKERFILE_TEMPLATE)
    (target / "server.py").write_text(_SERVER_TEMPLATE)
    (target / "requirements.txt").write_text("")

    _ok(f"\nInitialized environment project in {target}/")
    _ok("")
    _ok(f"  {target}/epsilab.json   — release manifest (fill in refs and digests)")
    _ok(f"  {target}/Dockerfile     — container image template")
    _ok(f"  {target}/server.py      — minimal environment server")
    _ok("")
    _ok("Next steps:")
    _ok(f"  1. Implement your environment logic in server.py")
    _ok(f"  2. Build and push your container image")
    _ok(f"  3. Fill in epsilab.json with image refs and content digests")
    _ok(f"  4. Create a namespace:  epsilab namespace create {slug}")
    _ok(f"  5. Create a listing:    epsilab env create --namespace-id <ns-id> {slug} \"My Environment\"")
    _ok(f"  6. Push a release:      epsilab env push --manifest epsilab.json --listing-id <listing-id>")
    _ok(f"  7. Deploy:              epsilab env deploy --listing-id <listing-id> --release-id <rel-id>")
    _ok("")
    _ok(f"Documentation:  {_DASHBOARD_URL} (Documentation > Quick Start)")
    _ok(f"Dashboard:      {_DASHBOARD_URL} (My Environments)")


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

        session = client.create_environment_session(
            deployment_id=deployment_id,
            task_id=task_id,
            seed=args.seed,
        )
        _ok(f"Session created: {session.session_id}")
        _ok(f"  status: {session.status}")
        _ok(f"  task: {session.task_id}")
        if session.observation:
            _ok(f"  observation: {session.observation[:200]}")
        if session.session_token:
            _ok(f"  token: {session.session_token[:20]}...")
        if args.json:
            _json_out(session.to_dict())
    except EpsilabError as e:
        _err(str(e))
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

        result = client.environment_step(
            session_id=session_id,
            action=action,
            session_token=args.session_token,
        )
        _ok(f"Reward: {result.reward}")
        _ok(f"Terminated: {result.terminated}  Truncated: {result.truncated}")
        if result.observation:
            _ok(f"Observation: {result.observation[:300]}")
        if result.info:
            _ok(f"Info: {json.dumps(result.info, default=str)[:200]}")
        if args.json:
            _json_out({"observation": result.observation, "reward": result.reward,
                        "terminated": result.terminated, "truncated": result.truncated,
                        "info": result.info})
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_session_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        session = client.get_environment_session(args.session_id)
        _ok(f"Session: {session.session_id}")
        _ok(f"  status: {session.status}")
        _ok(f"  deployment: {session.deployment_id}")
        _ok(f"  task: {session.task_id}")
        if session.observation:
            _ok(f"  observation: {session.observation[:300]}")
        if args.json:
            _json_out(session.to_dict())
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
    finally:
        client.close()


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
        _ok(f"Publish request submitted for listing: {listing_id}")
        _ok(f"  status: {result.get('status', 'pending_review')}")
        _ok(f"\nThe listing will be reviewed by the moderation team.")
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(str(e))
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
                  "sessions": b.get("total_sessions", "?")}
                for b in batches]
        _table(rows, ["id", "name", "status", "sessions"])
        if args.json:
            _json_out(batches)
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_env_batch_show(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        batch = client.get_batch(args.batch_id)
        _ok(f"Batch: {batch.get('batch_id', '?')}")
        _ok(f"  name: {batch.get('name', '?')}")
        _ok(f"  status: {batch.get('status', '?')}")
        _ok(f"  sessions: {batch.get('total_sessions', '?')}")

        if args.sessions:
            sessions = client.get_batch_sessions(args.batch_id)
            _ok(f"\nSessions ({len(sessions)}):")
            for s in sessions[:20]:
                _ok(f"  {s.get('session_id', '?')}  task={s.get('task_id', '?')}  status={s.get('status', '?')}")

        if args.comparison:
            comp = client.get_batch_comparison(args.batch_id)
            _ok(f"\nComparison:")
            _json_out(comp)

        if args.json:
            _json_out(batch)
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
    finally:
        client.close()


def cmd_run_cancel(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        run = client.cancel_run(args.run_id)
        _ok(f"Run cancelled: {run.run_id}")
        _ok(f"  status: {run.status}")
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
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
            _ok(f"Cost estimate:")
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
        _err(str(e))
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
        _err(str(e))
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

        session = client.create_rl_session(
            task_id=task_id,
            env_type=args.env_type,
            reward_mode=args.reward_mode,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        _ok(f"RL session created: {session.session_id}")
        _ok(f"  task: {session.task_id}")
        _ok(f"  status: {session.status}")
        if session.observation:
            _ok(f"  observation: {session.observation[:300]}")
        if args.json:
            _json_out(session.to_dict())
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_rl_step(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        result = client.rl_step(args.session_id, args.action)
        _ok(f"Reward: {result.reward}")
        _ok(f"Terminated: {result.terminated}  Truncated: {result.truncated}")
        if result.observation:
            _ok(f"Observation: {result.observation[:300]}")
        if args.json:
            _json_out({"observation": result.observation, "reward": result.reward,
                        "terminated": result.terminated, "truncated": result.truncated,
                        "info": result.info})
    except EpsilabError as e:
        _err(str(e))
    finally:
        client.close()


def cmd_rl_trajectory(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.verify:
            result = client.verify_rl_trajectory(args.session_id)
            _ok(f"Verification: {result.get('status', '?')}")
            if result.get("mismatches"):
                _ok(f"  mismatches: {result['mismatches']}")
            if args.json:
                _json_out(result)
            return

        traj = client.get_rl_trajectory(args.session_id)
        _ok(f"Trajectory: {traj.session_id}")
        _ok(f"  task: {traj.task_id}")
        _ok(f"  steps: {len(traj.steps)}")
        _ok(f"  total reward: {traj.total_reward}")
        if args.json:
            _json_out(traj.to_dict())
    except EpsilabError as e:
        _err(str(e))
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
        if not sessions:
            _ok("No RL sessions found.")
            return
        for s in sessions:
            _ok(f"  {s.get('session_id', '?')}  task={s.get('task_id', '?')}"
                f"  status={s.get('status', '?')}  reward={s.get('total_reward', '?')}")
        if args.json:
            _json_out(result)
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
    finally:
        client.close()


def cmd_rl_stats(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        stats = client.get_rl_stats(domain=args.domain, env_type=args.env_type)
        _json_out(stats)
    except EpsilabError as e:
        _err(str(e))
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
        _err(str(e))
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
        _err(str(e))
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
    create_p.add_argument("slug", nargs="?", help="URL-safe listing slug (prompted if omitted)")
    create_p.add_argument("title", nargs="?", help="Listing title (prompted if omitted)")
    create_p.add_argument("--namespace-id", help="Namespace ID (prompted if omitted)")
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
    deploy_p.add_argument("--release-id", help="Release ID to deploy (prompted if omitted)")
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

    # env publish
    publish_p = env_sub.add_parser("publish", help="Submit listing for marketplace review")
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
    run_p = sub.add_parser("run", help="Manage evaluation runs")
    run_sub = run_p.add_subparsers(dest="run_command", help="Run commands")

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
