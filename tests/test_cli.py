"""Tests for the Epsilab CLI.

Uses httpx.MockTransport to intercept API calls and tmpdir for config
file isolation.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from epsilab.cli import (
    _active_profile,
    _deploy_environment,
    _docker_build_and_upload,
    _encode_environment_action,
    _default_environment_smoke_action,
    _environment_horizon_errors,
    _environment_plugin_slugs,
    _environment_qualification_config,
    _environment_resource_policy,
    _environment_reward_mode,
    _get_client,
    _interactive_action,
    _normalize_cli_argv,
    _normalize_slug,
    _platform_openenv_max_steps,
    _public_step_info,
    _resolve_listing,
    _create_namespace,
    _resolve_environment_task,
    _resolve_tool_bindings,
    _load_config,
    _resolve_api_key,
    _save_config,
    _table,
    build_parser,
    cmd_env_init,
    cmd_env_verify,
    main,
)
from epsilab.models import ApplicationTool


def _json_response(body, status=200):
    return httpx.Response(
        status, json=body, request=httpx.Request("GET", "http://test")
    )


def _mock_client(handler):
    """Patch _get_client to return a client with mock transport."""
    from epsilab.client import EpsilabClient

    client = EpsilabClient.__new__(EpsilabClient)
    client._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    client._api_key = "test-key"
    client._max_retries = 0
    client._backoff_base = 0.0
    return client


class TestParser:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_global_profile_flag(self):
        args = build_parser().parse_args(["--profile", "staging", "whoami"])
        assert args.profile == "staging"
        assert args.command == "whoami"

    def test_login_command(self):
        args = build_parser().parse_args(["login", "--api-key", "sk-test"])
        assert args.command == "login"
        assert args.api_key == "sk-test"

    def test_login_with_label(self):
        args = build_parser().parse_args(["login", "--api-key", "sk-test", "--label", "production"])
        assert args.label == "production"

    def test_root_init_command(self):
        args = build_parser().parse_args(["init", "my-environment"])
        assert args.command == "init"
        assert args.slug == "my-environment"

    def test_deploy_help_matches_current_defaults(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["deploy", "--help"])

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "Release version (default: 1.0.0)" in output
        assert "epsilab deploy --no-host" in output

    def test_deploy_accepts_listing_visibility(self):
        args = build_parser().parse_args(["deploy", "--visibility", "unlisted"])
        assert args.visibility == "unlisted"

    def test_generated_slugs_match_registry_contract(self):
        assert _normalize_slug("Build.Output_2026") == "build-output-2026"
        assert _normalize_slug("-" + "a" * 80 + "-") == "a" * 64

    def test_environment_visibility_command(self):
        args = build_parser().parse_args([
            "env", "visibility", "epsilab/demo", "private",
        ])
        assert args.target == "epsilab/demo"
        assert args.visibility == "private"

    def test_public_run_syntax_is_normalized_without_affecting_run_subcommands(self):
        assert _normalize_cli_argv(["run", "epsilab/bug-hunter", "--action", "x"]) == [
            "run",
            "__environment",
            "epsilab/bug-hunter",
            "--action",
            "x",
        ]
        assert _normalize_cli_argv(["run", "list"]) == ["run", "list"]
        assert _normalize_cli_argv(["run", "--help"]) == [
            "run",
            "__environment",
            "--help",
        ]
        assert _normalize_cli_argv(["run"]) == ["run", "__environment", "--help"]

    def test_environment_run_accepts_custom_action_type(self):
        args = build_parser().parse_args(
            _normalize_cli_argv(
                ["run", "epsilab/form-filler", "--action", "name: Ada", "--action-type", "fill"]
            )
        )
        assert args.action_type == "fill"

    def test_interactive_run_accepts_environment_defined_action_type(self):
        assert _interactive_action("/check_logs order-service", "submit") == (
            "order-service",
            "check_logs",
        )
        with pytest.raises(ValueError, match="letters, numbers"):
            _interactive_action("/bad.type value", "submit")

    def test_environment_task_discovery_uses_published_capability(self):
        class Client:
            def iter_tasks(self, **_kwargs):
                return iter([
                    {"task_id": "unrelated-task", "capability": "other"},
                    {"task_id": "form-contact-001", "capability": "form-filler"},
                ])

        assert _resolve_environment_task(
            Client(),
            slug="form-filler",
            explicit_task_id=None,
        ) == "form-contact-001"

    def test_environment_task_discovery_prefers_published_train_split(self):
        class Client:
            def iter_tasks(self, **_kwargs):
                raise AssertionError("generic task discovery should not be needed")

        assert _resolve_environment_task(
            Client(),
            slug="workflow",
            explicit_task_id=None,
            published_tasks=[
                {"task_id": "workflow-development-001", "split": "development"},
                {"task_id": "workflow-train-002", "split": "train"},
                {"task_id": "workflow-train-001", "split": "train"},
            ],
        ) == "workflow-train-001"

    def test_environment_task_discovery_never_invents_task_ids(self):
        class Client:
            def iter_tasks(self, **_kwargs):
                return iter([])

        with pytest.raises(ValueError, match="Pass --task"):
            _resolve_environment_task(
                Client(),
                slug="workflow",
                explicit_task_id=None,
            )

    def test_env_list_command(self):
        args = build_parser().parse_args(["env", "list", "--limit", "10"])
        assert args.command == "env"
        assert args.env_command == "list"
        assert args.limit == 10

    def test_env_search_command(self):
        args = build_parser().parse_args(
            ["env", "search", "coding", "--domain", "math"]
        )
        assert args.command == "env"
        assert args.env_command == "search"
        assert args.query == "coding"
        assert args.domain == "math"

    def test_env_create_command(self):
        args = build_parser().parse_args(
            ["env", "create", "my-env", "My Environment", "--namespace-id", "ns-1"]
        )
        assert args.slug == "my-env"
        assert args.title == "My Environment"
        assert args.namespace_id == "ns-1"

    def test_env_push_with_manifest(self):
        args = build_parser().parse_args(
            ["env", "push", "--manifest", "epsilab.json", "--listing-id", "lst-1"]
        )
        assert args.manifest == "epsilab.json"
        assert args.listing_id == "lst-1"

    def test_env_deploy_command(self):
        args = build_parser().parse_args(
            [
                "env", "deploy",
                "--release-id", "rel-1",
                "--listing-id", "lst-1",
                "--alias", "staging",
            ]
        )
        assert args.release_id == "rel-1"
        assert args.alias == "staging"

    def test_env_grant_command(self):
        args = build_parser().parse_args(
            ["env", "grant", "lst-1", "tenant-buyer"]
        )
        assert args.listing_id == "lst-1"
        assert args.tenant_id == "tenant-buyer"

    def test_env_status_command(self):
        args = build_parser().parse_args(["env", "status", "rel-1"])
        assert args.release_id == "rel-1"

    def test_namespace_create(self):
        args = build_parser().parse_args(
            ["namespace", "create", "my-org", "--display-name", "My Org"]
        )
        assert args.slug == "my-org"
        assert args.display_name == "My Org"

    def test_profile_show(self):
        args = build_parser().parse_args(["profile", "show", "--json"])
        assert args.json is True

    def test_profile_create(self):
        args = build_parser().parse_args(
            ["profile", "create", "My Name", "--bio", "hello"]
        )
        assert args.display_name == "My Name"
        assert args.bio == "hello"


class TestNamespaceSetup:
    def test_auto_namespace_normalizes_parent_directory(self, tmp_path):
        project = tmp_path / "Build.Output_2026" / "demo-environment"
        project.mkdir(parents=True)
        captured = {}

        class Client:
            def get_creator_profile(self):
                return {}

            def create_namespace(self, *, slug, display_name):
                captured.update(slug=slug, display_name=display_name)
                return {"namespace_id": "ns-1"}

        assert _create_namespace(Client(), project, auto=True) == "ns-1"
        assert captured == {
            "slug": "build-output-2026",
            "display_name": "Build Output 2026",
        }


class TestEnvironmentSmokeAction:
    def test_uses_explicit_smoke_action(self):
        action = {"tool": "submit", "input": {"answer": 42}}
        assert _default_environment_smoke_action({"smoke_action": action}) == action

    def test_uses_exact_answer_for_text_task(self):
        assert _default_environment_smoke_action({"expected_answer": "hello epsilab"}) == {
            "content": "hello epsilab",
            "action_type": "submit",
        }

    def test_falls_back_to_protocol_probe(self):
        assert _default_environment_smoke_action({}) == {
            "content": "test step",
            "action_type": "submit",
        }


class TestConfigFile:
    def test_save_and_load(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path):
            _save_config({"api_key": "sk-test"})
            assert config_file.exists()
            config = json.loads(config_file.read_text())
            assert config["api_key"] == "sk-test"
            assert oct(config_file.stat().st_mode)[-3:] == "600"

            loaded = _load_config()
            assert loaded["api_key"] == "sk-test"

    def test_load_missing(self, tmp_path):
        with patch("epsilab.cli._CONFIG_FILE", tmp_path / "nope.json"):
            assert _load_config() == {}


class TestEnvInit:
    def test_creates_scaffold(self, tmp_path):
        target = tmp_path / "test-env"
        args = build_parser().parse_args(["env", "init", "test-env"])
        args.directory = str(target)
        cmd_env_init(args)

        assert (target / "tasks.json").exists()
        assert (target / "Dockerfile").exists()
        assert (target / "environment.py").exists()
        assert (target / "server.py").exists()
        assert (target / "verifier.py").exists()
        assert (target / "requirements.txt").exists()
        assert (target / ".epsilab" / "project.json").exists()

        dockerfile = (target / "Dockerfile").read_text()
        assert "mv /app/verifier.py /opt/epsilab/verifier.py" in dockerfile
        assert "ln -s /app /opt/epsilab" not in dockerfile

        tasks = json.loads((target / "tasks.json").read_text())
        assert isinstance(tasks, list)
        assert tasks[0]["task_id"] == "test-env-easy-train-001"
        project = json.loads((target / ".epsilab" / "project.json").read_text())
        assert project["listing_id"] == ""
        assert project["namespace_id"] == ""
        assert project["version"] == "1.0.0"
        assert project["qualification"] == {
            "task_id": "test-env-easy-train-001",
            "smoke_actions": [
                {"content": "hello epsilab", "action_type": "submit"}
            ],
            "repetitions": 3,
            "seed": 0,
        }
        assert project["resource_policy"] == {
            "cpu_millis": 1000,
            "memory_bytes": 512 * 1024 * 1024,
            "architecture": "amd64",
            "network_policy": "deny",
            "runtime_interface": "openenv",
        }

    @pytest.mark.parametrize(
        ("policy", "message"),
        [
            ({"cpu_millis": True}, "cpu_millis"),
            ({"memory_bytes": 1024}, "memory_bytes"),
            ({"architecture": "x86"}, "architecture"),
            ({"unknown": 1}, "unsupported field"),
        ],
    )
    def test_rejects_invalid_resource_policy(self, policy, message):
        with pytest.raises(ValueError, match=message):
            _environment_resource_policy({"resource_policy": policy})

    @pytest.mark.parametrize(
        ("qualification", "message"),
        [
            ({"enabled": "yes"}, "enabled"),
            ({"task_id": "task", "smoke_actions": []}, "smoke_actions"),
            (
                {"task_id": "task", "smoke_actions": ["done"], "repetitions": 0},
                "repetitions",
            ),
            (
                {"task_id": "task", "smoke_actions": ["done"], "seed": -1},
                "seed",
            ),
        ],
    )
    def test_rejects_invalid_qualification_profile(self, qualification, message):
        with pytest.raises(ValueError, match=message):
            _environment_qualification_config({"qualification": qualification})

    def test_refuses_nonempty_dir(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        (target / "file.txt").write_text("occupied")

        args = build_parser().parse_args(["env", "init", "existing"])
        args.directory = str(target)
        with pytest.raises(SystemExit):
            cmd_env_init(args)


class TestEnvList:
    def test_table_output(self, capsys):
        def handler(req):
            return _json_response(
                [
                    {
                        "listing_id": "lst-1",
                        "namespace_id": "ns-1",
                        "slug": "my-env",
                        "title": "My Env",
                        "visibility": "private",
                        "moderation_state": "pending",
                    }
                ]
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "list"])

        out = capsys.readouterr().out
        assert "my-env" in out
        assert "My Env" in out

    def test_json_output(self, capsys):
        def handler(req):
            return _json_response(
                [
                    {
                        "listing_id": "lst-1",
                        "namespace_id": "ns-1",
                        "slug": "my-env",
                        "title": "Test",
                    }
                ]
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "list", "--json"])

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["listing_id"] == "lst-1"


class TestEnvSearch:
    def test_basic(self, capsys):
        def handler(req):
            return _json_response(
                [{"listing_id": "lst-1", "title": "Code Sandbox", "domain": "coding"}]
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "search", "coding"])

        out = capsys.readouterr().out
        assert "Code Sandbox" in out


class TestEnvDeploy:
    def test_new_deployment(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"deployment_id": "dep-1", "alias": "production"}, status=201
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(
                [
                    "env", "deploy",
                    "--release-id", "rel-1",
                    "--listing-id", "lst-1",
                ]
            )

        out = capsys.readouterr().out
        assert "dep-1" in out
        assert captured["body"]["alias"] == "production"


class TestNamespaceCreate:
    def test_basic(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"namespace_id": "ns-new", "slug": "my-org"}, status=201
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["namespace", "create", "my-org"])

        out = capsys.readouterr().out
        assert "ns-new" in out
        assert captured["body"]["slug"] == "my-org"


class TestEnvStatus:
    def test_basic(self, capsys):
        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            if "quality-badges" in req.url.path:
                return _json_response([{"badge_type": "gold", "status": "active"}])
            if "quality-reports" in req.url.path:
                return _json_response([])
            return _json_response(
                {
                    "release_id": "rel-1",
                    "listing_id": "lst-1",
                    "release_version": "1.0.0",
                    "protocol_version": "0.4.1",
                    "status": "qualified",
                    "content_digest": "sha256:abc",
                    "created_at": "2026-06-01T00:00:00",
                }
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "status", "rel-1"])

        out = capsys.readouterr().out
        assert "qualified" in out
        assert "1.0.0" in out
        assert "gold" in out


class TestMainNoCommand:
    def test_no_args_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


class TestResolveApiKey:
    def test_from_env_var(self):
        with patch.dict(os.environ, {"EPSILAB_API_KEY": "sk-from-env"}):
            assert _resolve_api_key() == "sk-from-env"

    def test_from_profile_config(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "default",
            "profiles": {"default": {"api_key": "sk-from-profile"}},
        }))
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            os.environ.pop("EPSILAB_API_KEY", None)
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _resolve_api_key() == "sk-from-profile"

    def test_named_profile(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "default",
            "profiles": {
                "default": {"api_key": "sk-default"},
                "staging": {"api_key": "sk-staging"},
            },
        }))
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            os.environ.pop("EPSILAB_API_KEY", None)
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _resolve_api_key("staging") == "sk-staging"

    def test_env_profile_selects_profile(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "default",
            "profiles": {
                "default": {"api_key": "sk-default"},
                "ci": {"api_key": "sk-ci"},
            },
        }))
        with patch.dict(os.environ, {"EPSILAB_PROFILE": "ci"}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            assert _resolve_api_key() == "sk-ci"

    def test_backwards_compat_toplevel_api_key(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({"api_key": "sk-legacy"}))
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            os.environ.pop("EPSILAB_API_KEY", None)
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _resolve_api_key() == "sk-legacy"

    def test_env_var_takes_precedence(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "profiles": {"default": {"api_key": "sk-from-config"}},
        }))
        with patch.dict(os.environ, {"EPSILAB_API_KEY": "sk-from-env"}), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            assert _resolve_api_key() == "sk-from-env"

    def test_returns_none_when_missing(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", tmp_path / "nope.json"):
            os.environ.pop("EPSILAB_API_KEY", None)
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _resolve_api_key() is None


class TestGetClientErrors:
    def test_exits_without_auth(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", tmp_path / "nope.json"):
            os.environ.pop("EPSILAB_API_KEY", None)
            with pytest.raises(SystemExit):
                _get_client()


class TestTableHelper:
    def test_empty_rows(self, capsys):
        _table([], ["col_a", "col_b"])
        out = capsys.readouterr().out
        assert "(none)" in out

    def test_aligned_output(self, capsys):
        rows = [
            {"name": "short", "val": "1"},
            {"name": "a longer name", "val": "2"},
        ]
        _table(rows, ["name", "val"])
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        assert len(lines) == 4
        assert "NAME" in lines[0]
        assert "VAL" in lines[0]
        assert "short" in lines[2]
        assert "a longer name" in lines[3]


class TestLogin:
    def test_with_api_key_flag(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.get_usage.return_value = []
            instance.close.return_value = None
            main(["login", "--api-key", "sk-test-key"])

        out = capsys.readouterr().out
        assert "Authenticated" in out
        assert "default" in out
        saved = json.loads(config_file.read_text())
        assert saved["profiles"]["default"]["api_key"] == "sk-test-key"
        assert saved["active_profile"] == "default"

    def test_named_profile(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.get_usage.return_value = []
            instance.close.return_value = None
            main(["--profile", "staging", "login", "--api-key", "sk-staging"])

        saved = json.loads(config_file.read_text())
        assert saved["profiles"]["staging"]["api_key"] == "sk-staging"
        assert saved["active_profile"] == "staging"
        out = capsys.readouterr().out
        assert "staging" in out

    def test_with_label(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.get_usage.return_value = []
            instance.close.return_value = None
            main(["login", "--api-key", "sk-prod", "--label", "production"])

        saved = json.loads(config_file.read_text())
        assert saved["profiles"]["default"]["label"] == "production"

    def test_rejects_invalid_key(self, capsys, tmp_path):
        from epsilab.exceptions import AuthError as _AuthError

        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.get_usage.side_effect = _AuthError("bad key")
            instance.close.return_value = None
            with pytest.raises(SystemExit):
                main(["login", "--api-key", "sk-bad"])

        assert not config_file.exists()


class TestLogout:
    def test_removes_active_profile(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "default",
            "profiles": {"default": {"api_key": "sk-old"}},
        }))
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path):
            os.environ.pop("EPSILAB_PROFILE", None)
            main(["logout"])

        out = capsys.readouterr().out
        assert "Logged out" in out
        saved = json.loads(config_file.read_text())
        assert "default" not in saved.get("profiles", {})

    def test_removes_named_profile(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "staging",
            "profiles": {
                "default": {"api_key": "sk-default"},
                "staging": {"api_key": "sk-staging"},
            },
        }))
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path):
            os.environ.pop("EPSILAB_PROFILE", None)
            main(["--profile", "staging", "logout"])

        saved = json.loads(config_file.read_text())
        assert "staging" not in saved["profiles"]
        assert saved["active_profile"] == "default"

    def test_backwards_compat_toplevel_key(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({"api_key": "sk-old", "other": "data"}))
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path):
            os.environ.pop("EPSILAB_PROFILE", None)
            main(["logout"])

        saved = json.loads(config_file.read_text())
        assert "api_key" not in saved
        assert saved["other"] == "data"


class TestWhoami:
    def test_with_creator_profile(self, capsys):
        call_paths = []

        def handler(req):
            call_paths.append(req.url.path)
            if "usage" in req.url.path:
                return _json_response({"usage": []})
            if "creator-profiles" in req.url.path:
                return _json_response({"display_name": "Test Org"})
            return _json_response({})

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client), \
             patch("epsilab.cli._active_profile", return_value="default"):
            main(["whoami"])

        out = capsys.readouterr().out
        assert "Authenticated" in out
        assert "default" in out
        assert "Test Org" in out

    def test_without_creator_profile(self, capsys):
        def handler(req):
            if "usage" in req.url.path:
                return _json_response({"usage": []})
            if "creator-profiles" in req.url.path:
                return _json_response({"error": "not found"}, status=404)
            return _json_response({})

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client), \
             patch("epsilab.cli._active_profile", return_value="staging"):
            main(["whoami"])

        out = capsys.readouterr().out
        assert "Authenticated" in out
        assert "staging" in out
        assert "No creator profile" in out


class TestEnvCreate:
    def test_basic(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {
                    "listing_id": "lst-new",
                    "namespace_id": "ns-1",
                    "slug": "my-env",
                    "title": "My Environment",
                    "visibility": "public",
                },
                status=201,
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "create", "my-env", "My Environment", "--namespace-id", "ns-1"])

        out = capsys.readouterr().out
        assert "lst-new" in out
        assert "my-env" in out
        assert captured["body"]["slug"] == "my-env"
        assert captured["body"]["title"] == "My Environment"
        assert captured["body"]["visibility"] == "public"

    def test_with_visibility(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {
                    "listing_id": "lst-pub",
                    "namespace_id": "ns-1",
                    "slug": "pub-env",
                    "title": "Public Env",
                    "visibility": "public",
                },
                status=201,
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main([
                "env", "create", "pub-env", "Public Env",
                "--namespace-id", "ns-1", "--visibility", "public",
            ])

        assert captured["body"]["visibility"] == "public"


class TestEnvVisibility:
    def test_explicit_deploy_visibility_updates_rediscovered_listing(self):
        existing = SimpleNamespace(
            listing_id="lst-1",
            namespace_id="ns-1",
            namespace="epsilab",
            slug="demo-env",
            title="Demo Environment",
            visibility="public",
            revision=7,
        )
        captured = {}

        class Client:
            def list_environment_listings(self, **_kwargs):
                return [existing]

            def update_listing(self, listing_id, **kwargs):
                captured["listing_id"] = listing_id
                captured["kwargs"] = kwargs
                return SimpleNamespace(**{
                    **vars(existing),
                    "visibility": kwargs["visibility"],
                    "revision": 8,
                })

            def create_listing(self, **_kwargs):
                raise AssertionError("existing listing should be reused")

        result = _resolve_listing(
            Client(),
            "ns-1",
            "demo-env",
            "Demo Environment",
            "",
            visibility="unlisted",
            enforce_visibility=True,
        )

        assert result.visibility == "unlisted"
        assert captured == {
            "listing_id": "lst-1",
            "kwargs": {"expected_revision": 7, "visibility": "unlisted"},
        }

    def test_updates_listing_without_database_access(self, capsys):
        captured = {}
        listing = {
            "listing_id": "lst-1",
            "namespace_id": "ns-1",
            "namespace": "epsilab",
            "slug": "demo-env",
            "title": "Demo Environment",
            "visibility": "unlisted",
            "listing_revision": 3,
            "is_owner": True,
        }

        def handler(req):
            if req.method == "GET" and req.url.path == "/v1/environment-listings/lst-1":
                return _json_response({
                    **listing,
                    "visibility": captured.get("visibility", listing["visibility"]),
                    "listing_revision": 4 if "visibility" in captured else 3,
                })
            if req.method == "PATCH" and req.url.path == "/v1/environment-listings/lst-1":
                captured["body"] = json.loads(req.content)
                captured["visibility"] = captured["body"]["visibility"]
                return _json_response({
                    "listing_id": listing["listing_id"],
                    "namespace_id": listing["namespace_id"],
                    "slug": listing["slug"],
                    "title": listing["title"],
                    "visibility": captured["visibility"],
                    "listing_revision": 4,
                })
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "visibility", "lst-1", "private"])

        assert captured["body"] == {
            "expected_revision": 3,
            "visibility": "private",
        }
        output = capsys.readouterr().out
        assert "Visibility updated: epsilab/demo-env" in output
        assert "visible only to your organization" in output


class TestEnvGrant:
    def test_basic(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"entitlement_id": "ent-1"}, status=201)

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "grant", "lst-1", "tenant-buyer"])

        out = capsys.readouterr().out
        assert "ent-1" in out
        assert "tenant-buyer" in out
        assert captured["body"]["grantee_tenant_id"] == "tenant-buyer"
        assert captured["body"]["listing_id"] == "lst-1"

    def test_with_license_and_expiry(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"entitlement_id": "ent-2"}, status=201)

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main([
                "env", "grant", "lst-1", "tenant-2",
                "--license", "mit",
                "--expires-at", "2027-01-01T00:00:00Z",
            ])

        assert captured["body"]["license_id"] == "mit"
        assert captured["body"]["expires_at"] == "2027-01-01T00:00:00Z"

    def test_shared_command_labels_owned_and_received_access(self, capsys):
        def handler(req):
            assert req.method == "GET"
            return _json_response([
                {
                    "entitlement_id": "owned-1",
                    "environment_namespace": "my-team",
                    "environment_slug": "owned-env",
                    "environment_title": "Owned Environment",
                    "owner_tenant_id": "tenant-me",
                    "grantee_tenant_id": "tenant-partner",
                    "grantee_name": "Partner Org",
                    "permissions": ["discover", "execute"],
                    "status": "active",
                    "access_role": "owner",
                },
                {
                    "entitlement_id": "received-1",
                    "environment_namespace": "research",
                    "environment_slug": "received-env",
                    "environment_title": "Received Environment",
                    "owner_tenant_id": "tenant-lab",
                    "grantee_tenant_id": "tenant-me",
                    "owner_name": "Research Lab",
                    "permissions": ["execute"],
                    "status": "active",
                    "access_role": "recipient",
                },
            ])

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "shared"])

        output = capsys.readouterr().out
        assert "shared with: Partner Org" in output
        assert "shared by: Research Lab" in output


class TestEnvQualify:
    def test_basic(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"report_id": "rpt-1", "status": "pending"}, status=202
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(
                [
                    "env",
                    "qualify",
                    "rel-1",
                    "--task",
                    "task-1",
                    "--action",
                    "correct answer",
                ]
            )

        out = capsys.readouterr().out
        assert "rpt-1" in out
        assert "hosted_execution" in out
        assert captured["body"]["report_type"] == "hosted_execution"
        assert captured["body"]["release_id"] == "rel-1"
        assert captured["body"]["config"] == {
            "task_id": "task-1",
            "smoke_actions": [
                {"action_type": "submit", "content": "correct answer"}
            ],
            "repetitions": 3,
            "seed": 0,
        }

    def test_with_report_type(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"report_id": "rpt-2", "status": "pending"}, status=202
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "qualify", "rel-1", "--report-type", "benchmark"])

        out = capsys.readouterr().out
        assert "benchmark" in out

    def test_uses_scaffolded_project_profile(self, tmp_path, monkeypatch):
        target = tmp_path / "qualified-env"
        init_args = build_parser().parse_args(
            ["init", "qualified-env", "--directory", str(target)]
        )
        cmd_env_init(init_args)
        monkeypatch.chdir(target)
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"report_id": "rpt-project", "status": "queued"}, status=202
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "qualify", "rel-project"])

        assert captured["body"]["config"] == {
            "task_id": "qualified-env-easy-train-001",
            "smoke_actions": [
                {"content": "hello epsilab", "action_type": "submit"}
            ],
            "repetitions": 3,
            "seed": 0,
        }


class TestHostedQualificationStatus:
    def test_displays_current_hosted_qualification_without_signing_details(self, capsys):
        class Client:
            def get_environment_release(self, _release_id):
                return SimpleNamespace(
                    release_id="rel-1",
                    release_version="1.0.0",
                    protocol_version="0.4.1",
                    status="qualified",
                    content_digest="sha256:" + "a" * 64,
                    created_at=None,
                )

            def list_quality_reports(self, **_kwargs):
                return [
                    {
                        "report_id": "rpt-1",
                        "status": "completed",
                        "fail_count": 0,
                        "hosted_qualification_status": "qualified",
                        "hosted_qualification_expires_at": "2026-10-01T00:00:00Z",
                        "key_id": "must-not-display",
                    }
                ]

            def list_quality_badges(self, **_kwargs):
                return []

            def close(self):
                return None

        with patch("epsilab.cli._get_client", return_value=Client()):
            main(["env", "status", "rel-1"])

        out = capsys.readouterr().out
        assert "publication: qualified" in out
        assert "hosted execution: qualified" in out
        assert "qualification expires: 2026-10-01T00:00:00Z" in out
        assert "must-not-display" not in out


class TestEnvPush:
    def test_from_manifest(self, capsys, tmp_path):
        manifest = {
            "namespace_id": "ns-1",
            "listing_id": "lst-1",
            "release_version": "1.0.0",
            "task_pack": {
                "name": "my-tasks",
                "artifact_ref": "ghcr.io/tasks:1.0",
                "artifact_digest": "sha256:aaa",
            },
            "verifier": {
                "name": "my-verifier",
                "runtime_ref": "ghcr.io/ver:1.0",
                "runtime_digest": "sha256:bbb",
            },
            "environment": {
                "protocol_version": "0.4.1",
                "runtime_ref": "ghcr.io/env:1.0",
                "runtime_digest": "sha256:ccc",
            },
        }
        manifest_path = tmp_path / "epsilab.json"
        manifest_path.write_text(json.dumps(manifest))

        call_count = 0

        def handler(req):
            nonlocal call_count
            call_count += 1
            path = req.url.path
            if "task-pack" in path:
                return _json_response({"release_id": "tp-1"}, status=201)
            if "verifier" in path:
                return _json_response({"release_id": "ver-1"}, status=201)
            if "environment-releases" in path:
                return _json_response(
                    {
                        "release_id": "rel-1",
                        "listing_id": "lst-1",
                        "release_version": "1.0.0",
                        "protocol_version": "0.4.1",
                        "qualification_state": "qualified",
                    },
                    status=201,
                )
            return _json_response({})

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "push", "--manifest", str(manifest_path)])

        out = capsys.readouterr().out
        assert "Pushing environment release v1.0.0" in out
        assert "Task pack registered: tp-1" in out
        assert "Verifier registered: ver-1" in out
        assert "rel-1" in out
        assert "qualified" in out
        assert "according to the listing's visibility" in out
        assert call_count == 3

    def test_missing_listing_id_exits(self, capsys):
        with patch("epsilab.cli._get_client", return_value=_mock_client(lambda r: _json_response({}))):
            with pytest.raises(SystemExit):
                main(["env", "push", "--version", "1.0.0", "--namespace-id", "ns-1"])

    def test_missing_manifest_file_exits(self, capsys):
        with patch("epsilab.cli._get_client", return_value=_mock_client(lambda r: _json_response({}))):
            with pytest.raises(SystemExit):
                main(["env", "push", "--manifest", "/nonexistent/epsilab.json"])


class TestEnvDeployRevision:
    def test_revision(self, capsys):
        captured = {}

        def handler(req):
            captured["path"] = req.url.path
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response(
                {"deployment_id": "dep-1", "alias": "production"}, status=201
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main([
                "env", "deploy",
                "--release-id", "rel-2",
                "--deployment-id", "dep-1",
                "--revision",
            ])

        out = capsys.readouterr().out
        assert "revised" in out.lower() or "dep-1" in out
        assert "revisions" in captured["path"]

    def test_deploy_without_listing_id_exits(self, capsys):
        with patch("epsilab.cli._get_client", return_value=_mock_client(lambda r: _json_response({}))):
            with pytest.raises(SystemExit):
                main(["env", "deploy", "--release-id", "rel-1"])

    def test_deploy_json_output(self, capsys):
        def handler(req):
            return _json_response(
                {"deployment_id": "dep-1", "alias": "staging"}, status=201
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main([
                "env", "deploy",
                "--release-id", "rel-1",
                "--listing-id", "lst-1",
                "--json",
            ])

        out = capsys.readouterr().out
        assert "dep-1" in out


class TestEnvStatusNoBadges:
    def test_no_badges(self, capsys):
        def handler(req):
            if "quality-badges" in req.url.path:
                return _json_response([])
            if "quality-reports" in req.url.path:
                return _json_response([])
            return _json_response(
                {
                    "release_id": "rel-1",
                    "listing_id": "lst-1",
                    "release_version": "2.0.0",
                    "protocol_version": "0.4.1",
                    "status": "quarantined",
                }
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "status", "rel-1"])

        out = capsys.readouterr().out
        assert "quarantined" in out
        assert "2.0.0" in out
        assert "gold" not in out


class TestEnvSearchJson:
    def test_json_output(self, capsys):
        def handler(req):
            return _json_response(
                [{"listing_id": "lst-1", "title": "Env A", "quality_score": 0.95}]
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "search", "coding", "--json"])

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["listing_id"] == "lst-1"

    def test_empty_results(self, capsys):
        client = _mock_client(lambda r: _json_response([]))
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "search", "nonexistent"])

        out = capsys.readouterr().out
        assert "0 environment" in out


class TestEnvListEmpty:
    def test_empty(self, capsys):
        client = _mock_client(lambda r: _json_response([]))
        with patch("epsilab.cli._get_client", return_value=client):
            main(["env", "list"])

        out = capsys.readouterr().out
        assert "0" in out
        assert "(none)" in out


class TestProfileShow:
    def test_table_output(self, capsys):
        def handler(req):
            return _json_response(
                {
                    "display_name": "Test Creator",
                    "bio": "We build environments",
                    "website_url": "https://example.com",
                    "is_public": True,
                }
            )

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["profile", "show"])

        out = capsys.readouterr().out
        assert "Test Creator" in out
        assert "We build environments" in out
        assert "https://example.com" in out

    def test_json_output(self, capsys):
        def handler(req):
            return _json_response({"display_name": "JSON Org", "is_public": False})

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["profile", "show", "--json"])

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["display_name"] == "JSON Org"

    def test_404_shows_message(self, capsys):
        def handler(req):
            return _json_response({"error": "not found"}, status=404)

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["profile", "show"])

        out = capsys.readouterr().out
        assert "No creator profile" in out


class TestProfileCreate:
    def test_basic(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"display_name": "New Org"})

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["profile", "create", "New Org", "--bio", "We do RL", "--website", "https://rl.co"])

        out = capsys.readouterr().out
        assert "New Org" in out
        assert captured["body"]["display_name"] == "New Org"
        assert captured["body"]["bio"] == "We do RL"
        assert captured["body"]["website_url"] == "https://rl.co"


class TestNamespaceCreateWithDisplayName:
    def test_uses_display_name(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"namespace_id": "ns-1"}, status=201)

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["namespace", "create", "my-ns", "--display-name", "My Namespace"])

        assert captured["body"]["display_name"] == "My Namespace"

    def test_defaults_display_name_to_slug(self, capsys):
        captured = {}

        def handler(req):
            captured["body"] = json.loads(req.content) if req.content else {}
            return _json_response({"namespace_id": "ns-1"}, status=201)

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["namespace", "create", "auto-name"])

        assert captured["body"]["display_name"] == "auto-name"


class TestActiveProfile:
    def test_from_env_var(self):
        with patch.dict(os.environ, {"EPSILAB_PROFILE": "ci"}):
            assert _active_profile() == "ci"

    def test_from_config(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({"active_profile": "staging"}))
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _active_profile() == "staging"

    def test_defaults_to_default(self, tmp_path):
        with patch.dict(os.environ, {}, clear=True), \
             patch("epsilab.cli._CONFIG_FILE", tmp_path / "nope.json"):
            os.environ.pop("EPSILAB_PROFILE", None)
            assert _active_profile() == "default"

    def test_env_var_overrides_config(self, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({"active_profile": "staging"}))
        with patch.dict(os.environ, {"EPSILAB_PROFILE": "prod"}), \
             patch("epsilab.cli._CONFIG_FILE", config_file):
            assert _active_profile() == "prod"


class TestMultipleProfiles:
    def test_login_two_profiles(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.get_usage.return_value = []
            instance.close.return_value = None

            main(["login", "--api-key", "sk-default-key"])
            main(["--profile", "staging", "login", "--api-key", "sk-staging-key"])

        saved = json.loads(config_file.read_text())
        assert saved["profiles"]["default"]["api_key"] == "sk-default-key"
        assert saved["profiles"]["staging"]["api_key"] == "sk-staging-key"
        assert saved["active_profile"] == "staging"

    def test_logout_falls_back_to_remaining(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "staging",
            "profiles": {
                "default": {"api_key": "sk-default"},
                "staging": {"api_key": "sk-staging"},
                "ci": {"api_key": "sk-ci"},
            },
        }))
        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path):
            os.environ.pop("EPSILAB_PROFILE", None)
            main(["--profile", "staging", "logout"])

        saved = json.loads(config_file.read_text())
        assert "staging" not in saved["profiles"]
        assert saved["active_profile"] in ("default", "ci")

    def test_global_profile_flag_threads_through(self, capsys, tmp_path):
        config_file = tmp_path / "credentials.json"
        config_file.write_text(json.dumps({
            "active_profile": "default",
            "profiles": {
                "default": {"api_key": "sk-default"},
                "ci": {"api_key": "sk-ci"},
            },
        }))

        def handler(req):
            return _json_response([])

        with patch("epsilab.cli._CONFIG_FILE", config_file), \
             patch("epsilab.cli._CONFIG_DIR", tmp_path), \
             patch("epsilab.cli.EpsilabClient") as MockClient:
            instance = MockClient.return_value
            instance.list_environment_listings.return_value = []
            instance.close.return_value = None
            main(["--profile", "ci", "env", "list"])
            assert MockClient.call_args[1].get("api_key") == "sk-ci" or \
                   MockClient.call_args[0][0] if MockClient.call_args[0] else True


class TestEnvInitDefaultSlug:
    def test_default_slug(self, tmp_path):
        target = tmp_path / "my-environment"
        args = build_parser().parse_args(["env", "init"])
        args.directory = str(target)
        cmd_env_init(args)

        tasks = json.loads((target / "tasks.json").read_text())
        assert isinstance(tasks, list)
        assert tasks[0]["task_id"] == "my-environment-easy-train-001"

    def test_generated_python_files_are_valid(self, tmp_path):
        target = tmp_path / "valid-env"
        args = build_parser().parse_args(["env", "init", "valid-env"])
        args.directory = str(target)
        cmd_env_init(args)

        for filename in ("environment.py", "server.py", "verifier.py"):
            source = (target / filename).read_text()
            compile(source, filename, "exec")

    def test_tasks_json_is_valid(self, tmp_path):
        target = tmp_path / "json-env"
        args = build_parser().parse_args(["env", "init", "json-env"])
        args.directory = str(target)
        cmd_env_init(args)

        tasks = json.loads((target / "tasks.json").read_text())
        assert isinstance(tasks, list)
        assert len(tasks) >= 1
        assert "task_id" in tasks[0]
        assert "prompt" in tasks[0]


class TestEnvironmentRunHelpers:
    def test_plain_action_uses_selected_type(self):
        encoded = json.loads(_encode_environment_action("inspect this", action_type="analyze"))
        assert encoded == {"action_type": "analyze", "content": "inspect this"}

    def test_structured_action_is_preserved(self):
        encoded = _encode_environment_action(
            '{"tool":"search","input":{"q":"status"}}',
            action_type="submit",
        )
        assert json.loads(encoded) == {"tool": "search", "input": {"q": "status"}}

    def test_invalid_structured_action_is_rejected(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _encode_environment_action("{bad", action_type="submit")

    def test_step_info_excludes_internal_fields(self):
        assert _public_step_info(
            {
                "passed": True,
                "verification_authority": "independent_verifier",
                "verification_status": "verified",
                "provider_key": "internal",
                "runtime_config": {"secret": True},
            }
        ) == {
            "passed": True,
            "verification_authority": "independent_verifier",
            "verification_status": "verified",
        }


class TestApplicationToolBindings:
    @staticmethod
    def _tool(
        slug: str,
        *,
        namespace: str = "epsilab",
        release_id: str | None = None,
    ) -> ApplicationTool:
        return ApplicationTool(
            tool_id=f"tool-{namespace}-{slug}",
            namespace_id=f"namespace-{namespace}",
            namespace=namespace,
            slug=slug,
            title=slug.title(),
            category="application",
            recommended_release_id=release_id,
        )

    def test_infers_plugins_from_composed_task_workspace(self):
        tasks = [
            {
                "workspace": {
                    "actors": {"gmail": "token", "slack": "token"},
                    "seeds": {"gmail": {}, "support": {}, "slack": {}},
                }
            }
        ]

        assert _environment_plugin_slugs({}, tasks, uses_appsuite=True) == [
            "gmail",
            "slack",
            "support",
        ]
        assert _environment_plugin_slugs({}, tasks, uses_appsuite=False) == []
        assert _environment_plugin_slugs(
            {"plugins": ["epsilab-apps/github", "slack", "slack"]},
            tasks,
            uses_appsuite=True,
        ) == ["epsilab-apps/github", "slack"]

    def test_resolves_api_contract_alias_and_canonical_empty_configuration(self):
        class Client:
            def list_application_tools(self, **_kwargs):
                return [
                    TestApplicationToolBindings._tool(
                        "github",
                        namespace="epsilab-apps",
                        release_id="release-github",
                    ),
                    TestApplicationToolBindings._tool(
                        "slack",
                        release_id="release-slack",
                    ),
                ]

        bindings = _resolve_tool_bindings(
            Client(),
            ["slack", "epsilab-apps/github"],
        )
        empty_digest = (
            "sha256:44136fa355b3678a1146ad16f7e8649e"
            "94fb4fc21fe77e8310c060f61caaff8a"
        )

        assert bindings == [
            {
                "tool_release_id": "release-github",
                "alias": "github",
                "configuration_digest": empty_digest,
            },
            {
                "tool_release_id": "release-slack",
                "alias": "slack",
                "configuration_digest": empty_digest,
            },
        ]

    def test_ambiguous_unqualified_tool_requires_owner(self):
        class Client:
            def list_application_tools(self, **_kwargs):
                return [
                    TestApplicationToolBindings._tool(
                        "slack", namespace="one", release_id="release-one"
                    ),
                    TestApplicationToolBindings._tool(
                        "slack", namespace="two", release_id="release-two"
                    ),
                ]

        with pytest.raises(ValueError, match="configure it as <owner>/<slug>"):
            _resolve_tool_bindings(Client(), ["slack"])


class TestRunEnvironmentCommand:
    def test_one_step_flow_uses_listing_task_discovery(self):
        requested_paths = []

        def handler(req):
            requested_paths.append(req.url.path)
            if req.method == "GET" and req.url.path == "/v1/environment-listings":
                return _json_response(
                    [
                        {
                            "listing_id": "lst-1",
                            "namespace_id": "ns-1",
                            "namespace": "epsilab",
                            "slug": "bug-hunter",
                            "title": "Bug Hunter",
                            "deployment_id": "dep-1",
                        }
                    ]
                )
            if req.method == "GET" and req.url.path == "/v1/environment-listings/lst-1":
                return _json_response(
                    {
                        "listing_id": "lst-1",
                        "namespace_id": "ns-1",
                        "namespace": "epsilab",
                        "slug": "bug-hunter",
                        "title": "Bug Hunter",
                        "deployment_id": "dep-1",
                        "tasks": [
                            {
                                "task_id": "bug-hunter-train-001",
                                "name": "Fix the average",
                                "domain": "coding",
                                "capability": "debugging",
                                "difficulty": "easy",
                                "verification": "hidden_tests",
                                "split": "train",
                            }
                        ],
                    }
                )
            if req.method == "POST" and req.url.path.endswith("/sessions"):
                assert json.loads(req.content)["task_id"] == "bug-hunter-train-001"
                return _json_response(
                    {
                        "session_id": "sess-discovery",
                        "task_id": "bug-hunter-train-001",
                        "status": "active",
                        "session_token": "session-secret",
                        "observation": "Find and fix the bug.",
                    },
                    status=202,
                )
            if req.method == "POST" and req.url.path.endswith("/step"):
                return _json_response(
                    {
                        "observation": "Submission received.",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                        "info": {"passed": True},
                    }
                )
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(["run", "epsilab/bug-hunter", "--action", "fixed code"])

        assert "/v1/tasks" not in requested_paths

    def test_one_step_flow_resolves_listing_task_and_uses_session_token(self, capsys):
        captured = {}

        def handler(req):
            if req.method == "GET" and req.url.path == "/v1/environment-listings":
                return _json_response(
                    [
                        {
                            "listing_id": "lst-1",
                            "namespace_id": "ns-1",
                            "namespace": "epsilab",
                            "slug": "bug-hunter",
                            "title": "Bug Hunter",
                            "deployment_id": "dep-1",
                        }
                    ]
                )
            if req.method == "GET" and req.url.path == "/v1/tasks":
                return _json_response(
                    {"tasks": [{"task_id": "bug-hunter-easy-train-001"}]}
                )
            if req.method == "POST" and req.url.path.endswith("/sessions"):
                return _json_response(
                    {
                        "session_id": "sess-1",
                        "task_id": "bug-hunter-easy-train-001",
                        "status": "active",
                        "session_token": "session-secret",
                        "observation": "Find and fix the bug.",
                    },
                    status=202,
                )
            if req.method == "POST" and req.url.path.endswith("/step"):
                captured["token"] = req.headers.get("x-rl-session-token")
                captured["action"] = json.loads(req.content)["action"]
                return _json_response(
                    {
                        "observation": "Submission received.",
                        "reward": 1.0,
                        "terminated": True,
                        "truncated": False,
                        "info": {
                            "passed": True,
                            "verification_authority": "independent_verifier",
                            "verification_status": "verified",
                            "provider_key": "not-public",
                        },
                    }
                )
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(
                [
                    "run",
                    "epsilab/bug-hunter",
                    "--action",
                    "fixed code",
                    "--task",
                    "bug-hunter-easy-train-001",
                ]
            )

        assert captured["token"] == "session-secret"
        assert json.loads(captured["action"]) == {
            "action_type": "submit",
            "content": "fixed code",
        }
        output = capsys.readouterr().out
        assert "Reward: 1.0000" in output
        assert "verification_authority: independent_verifier" in output
        assert "verification_status: verified" in output
        assert "not-public" not in output

    def test_nonterminal_one_step_session_is_cancelled(self):
        cancelled = []

        def handler(req):
            if req.method == "GET" and req.url.path == "/v1/environment-listings":
                return _json_response(
                    [
                        {
                            "listing_id": "lst-1",
                            "namespace_id": "ns-1",
                            "namespace": "epsilab",
                            "slug": "workflow",
                            "title": "Workflow",
                            "deployment_id": "dep-1",
                        }
                    ]
                )
            if req.method == "POST" and req.url.path.endswith("/sessions"):
                return _json_response(
                    {
                        "session_id": "sess-2",
                        "task_id": "workflow-task",
                        "status": "active",
                        "session_token": "token",
                        "observation": "Ready.",
                    },
                    status=202,
                )
            if req.method == "POST" and req.url.path.endswith("/step"):
                return _json_response(
                    {
                        "observation": "Continue.",
                        "reward": None,
                        "terminated": False,
                        "truncated": False,
                        "info": {},
                    }
                )
            if req.method == "POST" and req.url.path.endswith("/cancel"):
                cancelled.append(req.url.path)
                return _json_response({"session_id": "sess-2", "status": "cancelled"})
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            main(
                [
                    "run",
                    "epsilab/workflow",
                    "--task",
                    "workflow-task",
                    "--action",
                    "inspect",
                    "--action-type",
                    "analyze",
                ]
            )

        assert cancelled == ["/v1/environment-sessions/sess-2/cancel"]

    def test_terminal_step_without_reward_exits_with_failure(self, capsys):
        def handler(req):
            if req.method == "GET" and req.url.path == "/v1/environment-listings":
                return _json_response(
                    [
                        {
                            "listing_id": "lst-1",
                            "namespace_id": "ns-1",
                            "namespace": "epsilab",
                            "slug": "bug-hunter",
                            "title": "Bug Hunter",
                            "deployment_id": "dep-1",
                        }
                    ]
                )
            if req.method == "POST" and req.url.path.endswith("/sessions"):
                return _json_response(
                    {
                        "session_id": "sess-error",
                        "task_id": "bug-hunter-task",
                        "status": "active",
                        "session_token": "token",
                        "observation": "Ready.",
                    },
                    status=202,
                )
            if req.method == "POST" and req.url.path.endswith("/step"):
                return _json_response(
                    {
                        "observation": "The hosted environment could not complete this action.",
                        "reward": None,
                        "terminated": False,
                        "truncated": True,
                        "info": {
                            "terminal_reason": "error",
                            "provider_key": "not-public",
                        },
                    }
                )
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        with patch("epsilab.cli._get_client", return_value=client):
            with pytest.raises(SystemExit) as exc_info:
                main(
                    [
                        "run",
                        "epsilab/bug-hunter",
                        "--task",
                        "bug-hunter-task",
                        "--action",
                        '{"unexpected":true}',
                    ]
                )

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Environment action failed (error)." in captured.err
        assert "not-public" not in captured.out + captured.err


class TestRootDeployCommand:
    @pytest.mark.parametrize("reward_mode", ["binary", "continuous", "partial_credit"])
    def test_environment_reward_mode_accepts_supported_values(self, reward_mode):
        assert _environment_reward_mode({"reward_mode": reward_mode}) == reward_mode

    def test_environment_reward_mode_defaults_to_continuous(self):
        assert _environment_reward_mode({}) == "continuous"

    def test_environment_reward_mode_rejects_unknown_values(self):
        with pytest.raises(ValueError, match="reward_mode"):
            _environment_reward_mode({"reward_mode": "score"})

    def test_platform_horizon_capability_is_enforced_before_build(self, tmp_path):
        directory = tmp_path / "environment"
        directory.mkdir()
        (directory / "Dockerfile").write_text("FROM scratch\n")

        class Client:
            def get_platform_config(self):
                return {
                    "environment_limits": {
                        "runtimes": {"openenv": {"max_steps": 499}}
                    }
                }

        args = build_parser().parse_args(["deploy", str(directory), "--yes"])
        with pytest.raises(ValueError, match="max_steps must be an integer in \\[1, 499\\]"):
            _deploy_environment(
                args,
                Client(),
                directory,
                {"tasks": [{"task_id": "long-task", "max_steps": 500}]},
                {"listing_id": "listing", "slug": "long-env", "version": "1.0.0"},
                "namespace",
            )

    def test_docker_build_uses_named_context(self, tmp_path):
        directory = tmp_path / "environment"
        appsuite = tmp_path / "AppSuite"
        directory.mkdir()
        appsuite.mkdir()
        (directory / "Dockerfile").write_text("FROM scratch\n")

        class Client:
            def upload_image(self, _path, *, tag):
                return {"image_ref": f"registry.example/{tag}"}

        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=completed) as run:
            _docker_build_and_upload(
                Client(),
                directory,
                "demo:1.0.0",
                named_contexts={"appsuite": appsuite},
            )

        build_command = run.call_args_list[0].args[0]
        context_index = build_command.index("--build-context")
        assert build_command[context_index + 1] == f"appsuite={appsuite}"

    def test_composed_deploy_passes_current_appsuite_checkout(
        self,
        tmp_path,
        monkeypatch,
    ):
        catalog = tmp_path / "env-catalog"
        directory = catalog / "composed" / "demo"
        appsuite = tmp_path / "AppSuite"
        directory.mkdir(parents=True)
        (catalog / "_shared").mkdir()
        (appsuite / "src" / "epsilab_apps").mkdir(parents=True)
        (appsuite / "pyproject.toml").write_text("[project]\nname='epsilab-apps'\n")
        (appsuite / "src" / "epsilab_apps" / "__init__.py").write_text("")
        (directory / "Dockerfile").write_text(
            "ARG ENV_PATH\nCOPY --from=appsuite src /tmp/AppSuite/src\n"
        )
        captured = {}

        def upload(*_args, **kwargs):
            captured.update(kwargs)
            return {
                "image_ref": "registry.example/demo@sha256:" + "a" * 64,
                "content_digest": "sha256:" + "a" * 64,
            }

        class Client:
            def create_task(self, **_kwargs):
                raise AssertionError("no tasks expected")

            def create_task_pack_release(self, **_kwargs):
                return {"release_id": "pack-1"}

            def create_verifier_release(self, **_kwargs):
                return {"release_id": "ver-1"}

            def create_environment_release(self, **_kwargs):
                return {"release_id": "rel-1"}

            def create_deployment(self, **_kwargs):
                return {"deployment_id": "dep-1"}

            def get_environment_listing(self, _listing_id):
                return SimpleNamespace(namespace="epsilab", deployment_id="dep-1")

        monkeypatch.setattr("epsilab.cli._docker_build_and_upload", upload)
        monkeypatch.setattr("epsilab.cli._save_project", lambda *_args: None)
        args = build_parser().parse_args(["deploy", str(directory), "--yes"])
        project = {
            "listing_id": "listing-1",
            "namespace_id": "namespace-1",
            "slug": "demo",
            "title": "Demo",
            "version": "1.0.0",
        }

        _deploy_environment(
            args,
            Client(),
            directory,
            {"tasks": []},
            project,
            "namespace-1",
        )

        assert captured["build_context"] == catalog
        assert captured["build_args"] == {"ENV_PATH": "composed/demo"}
        assert captured["named_contexts"] == {"appsuite": appsuite}

    def test_repairs_empty_project_ids_and_creates_openenv_deployment(
        self,
        tmp_path,
        capsys,
    ):
        target = tmp_path / "demo-env"
        init_args = build_parser().parse_args(["init", "demo-env", "-d", str(target)])
        cmd_env_init(init_args)
        project_file = target / ".epsilab" / "project.json"
        project_config = json.loads(project_file.read_text())
        project_config["resource_policy"] = {
            "cpu_millis": 1750,
            "memory_bytes": 1024 * 1024 * 1024,
            "architecture": "amd64",
            "network_policy": "deny",
            "runtime_interface": "openenv",
        }
        project_file.write_text(json.dumps(project_config))
        captured = {
            "paths": [],
            "environment_body": None,
            "listing_body": None,
            "verifier_body": None,
        }

        existing_listing = {
            "listing_id": "lst-existing",
            "namespace_id": "ns-1",
            "namespace": "epsilab",
            "slug": "existing",
            "title": "Existing",
            "is_owner": True,
        }
        deployed_listing = {
            "listing_id": "lst-new",
            "namespace_id": "ns-1",
            "namespace": "epsilab",
            "slug": "demo-env",
            "title": "Demo Env",
            "is_owner": True,
            "release_id": "rel-1",
            "release_version": "1.0.0",
            "deployment_id": "dep-1",
        }

        def handler(req):
            captured["paths"].append((req.method, req.url.path))
            if req.method == "GET" and req.url.path == "/v1/platform/config":
                return _json_response(
                    {
                        "environment_limits": {
                            "runtimes": {"openenv": {"max_steps": 10_000}}
                        }
                    }
                )
            if req.method == "GET" and req.url.path == "/v1/environment-listings":
                return _json_response([existing_listing])
            if req.method == "GET" and req.url.path == "/v1/application-tools":
                return _json_response([])
            if req.method == "POST" and req.url.path == "/v1/environment-listings":
                captured["listing_body"] = json.loads(req.content)
                return _json_response({
                    **deployed_listing,
                    "visibility": captured["listing_body"]["visibility"],
                }, status=201)
            if req.method == "POST" and req.url.path == "/v1/tasks":
                return _json_response({"task_id": "demo-env-easy-train-001"}, status=201)
            if req.method == "POST" and req.url.path == "/v1/task-pack-releases":
                return _json_response({"release_id": "pack-1"}, status=201)
            if req.method == "POST" and req.url.path == "/v1/verifier-releases":
                captured["verifier_body"] = json.loads(req.content)
                return _json_response({"release_id": "ver-1"}, status=201)
            if req.method == "POST" and req.url.path == "/v1/environment-releases":
                captured["environment_body"] = json.loads(req.content)
                return _json_response(
                    {
                        "release_id": "rel-1",
                        "listing_id": "lst-new",
                        "release_version": "1.0.0",
                        "protocol_version": "0.4.1",
                        "status": "qualified",
                    },
                    status=201,
                )
            if req.method == "POST" and req.url.path == "/v1/environment-quality-reports":
                captured["quality_body"] = json.loads(req.content)
                return _json_response(
                    {"report_id": "quality-1", "status": "queued"},
                    status=202,
                )
            if req.method == "POST" and req.url.path == "/v1/environment-deployments":
                return _json_response({"deployment_id": "dep-1"}, status=201)
            if req.method == "GET" and req.url.path == "/v1/environment-listings/lst-new":
                return _json_response(deployed_listing)
            raise AssertionError(f"unexpected request: {req.method} {req.url.path}")

        client = _mock_client(handler)
        upload = {
            "image_ref": "registry.example/demo-env@sha256:" + "a" * 64,
            "content_digest": "sha256:" + "a" * 64,
        }
        with (
            patch("epsilab.cli._get_client", return_value=client),
            patch("epsilab.cli._docker_build_and_upload", return_value=upload),
        ):
            main(["deploy", str(target), "--yes", "--visibility", "unlisted"])

        project = json.loads((target / ".epsilab" / "project.json").read_text())
        assert project["namespace_id"] == "ns-1"
        assert project["listing_id"] == "lst-new"
        assert project["deployment_id"] == "dep-1"
        assert project["visibility"] == "unlisted"
        assert captured["listing_body"]["visibility"] == "unlisted"
        assert captured["verifier_body"]["reward_mode"] == "continuous"
        assert captured["environment_body"]["resource_policy"] == project_config["resource_policy"]
        assert captured["quality_body"] == {
            "release_id": "rel-1",
            "report_type": "hosted_execution",
            "config": project_config["qualification"],
        }
        assert project["quality_report_id"] == "quality-1"
        assert ("POST", "/v1/environment-deployments") in captured["paths"]
        output = capsys.readouterr().out
        assert "Deployed demo-env@1.0.0" in output
        assert "Hosting:  Qualification queued" in output
        assert "epsilab run epsilab/demo-env" in output


class TestEnvVerify:
    def test_verify_scaffolded_project(self, tmp_path, capsys):
        target = tmp_path / "verify-env"
        init_args = build_parser().parse_args(["env", "init", "verify-env"])
        init_args.directory = str(target)
        cmd_env_init(init_args)

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        cmd_env_verify(verify_args)
        out = capsys.readouterr().out
        assert "server.py exists" in out
        assert "Dockerfile exists" in out
        assert "/reset" in out
        assert "/step" in out

    def test_verify_no_manifest_passes(self, tmp_path, capsys):
        target = tmp_path / "empty-env"
        target.mkdir()

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        cmd_env_verify(verify_args)
        out = capsys.readouterr().out
        assert "not required" in out or "passed" in out.lower()

    def test_verify_invalid_digest(self, tmp_path, capsys):
        target = tmp_path / "bad-digest"
        target.mkdir()
        manifest = {
            "listing_id": "test",
            "namespace_id": "ns",
            "release_version": "1.0.0",
            "environment": {"runtime_digest": "sha256:tooshort"},
        }
        (target / "epsilab.json").write_text(json.dumps(manifest))

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        with pytest.raises(SystemExit):
            cmd_env_verify(verify_args)
        out = capsys.readouterr().out
        assert "not a valid sha256 digest" in out

    def test_verify_valid_digest(self, tmp_path, capsys):
        target = tmp_path / "good-digest"
        target.mkdir()
        digest = "sha256:" + "a" * 64
        manifest = {
            "listing_id": "test",
            "namespace_id": "ns",
            "release_version": "1.0.0",
            "environment": {"runtime_digest": digest},
        }
        (target / "epsilab.json").write_text(json.dumps(manifest))

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        cmd_env_verify(verify_args)
        out = capsys.readouterr().out
        assert "runtime_digest format is valid" in out

    def test_verify_accepts_500_step_task(self, tmp_path, capsys):
        target = tmp_path / "long-env"
        target.mkdir()
        (target / "tasks.json").write_text(
            json.dumps([{"task_id": "long-task", "max_steps": 500}])
        )

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        cmd_env_verify(verify_args)

        assert "maximum 500" in capsys.readouterr().out

    def test_verify_rejects_task_above_openenv_limit(self, tmp_path, capsys):
        target = tmp_path / "too-long-env"
        target.mkdir()
        (target / "tasks.json").write_text(
            json.dumps([{"task_id": "too-long", "max_steps": 10_001}])
        )

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        with pytest.raises(SystemExit):
            cmd_env_verify(verify_args)

        assert "max_steps must be an integer in [1, 10000]" in capsys.readouterr().out


def test_legacy_platform_without_capabilities_keeps_200_step_guard() -> None:
    class Client:
        def get_platform_config(self):
            return {"api_version": "v1"}

    assert _platform_openenv_max_steps(Client()) == 200
    assert _environment_horizon_errors(
        [{"task_id": "long-task", "max_steps": 500}],
        max_steps=200,
        source="the target Foundation platform",
    ) == [
        "long-task: max_steps must be an integer in [1, 200] for the target Foundation platform"
    ]


class TestSubcommandHelpExits:
    @pytest.mark.parametrize("argv", [
        ["env"],
        ["namespace"],
        ["profile"],
    ])
    def test_shows_help(self, argv):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
        assert exc_info.value.code == 0


class TestAllCommandsHaveFunc:
    """Every parsed subcommand should have a func attribute."""

    COMMAND_SETS = [
        ["login"],
        ["logout"],
        ["whoami"],
        ["env", "init"],
        ["env", "list"],
        ["env", "search"],
        ["env", "create", "s", "t", "--namespace-id", "n"],
        ["env", "verify"],
        ["env", "push"],
        ["env", "deploy", "--release-id", "r"],
        ["env", "grant", "l", "t"],
        ["env", "status", "r"],
        ["env", "qualify", "r"],
        ["namespace", "create", "s"],
        ["profile", "show"],
        ["profile", "create", "name"],
    ]

    @pytest.mark.parametrize("argv", COMMAND_SETS)
    def test_func_exists(self, argv):
        args = build_parser().parse_args(argv)
        assert hasattr(args, "func"), f"No func for {argv}"
        assert callable(args.func)
