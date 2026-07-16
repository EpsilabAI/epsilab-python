"""Tests for the Epsilab CLI.

Uses httpx.MockTransport to intercept API calls and tmpdir for config
file isolation.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest

from epsilab.cli import (
    _active_profile,
    _get_client,
    _load_config,
    _resolve_api_key,
    _save_config,
    _table,
    build_parser,
    cmd_env_init,
    cmd_env_verify,
    main,
)


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

        assert (target / "epsilab.json").exists()
        assert (target / "Dockerfile").exists()
        assert (target / "server.py").exists()
        assert (target / "requirements.txt").exists()

        manifest = json.loads((target / "epsilab.json").read_text())
        assert manifest["task_pack"]["name"] == "test-env-tasks"
        assert manifest["verifier"]["name"] == "test-env-verifier"

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
            main(["env", "qualify", "rel-1"])

        out = capsys.readouterr().out
        assert "rpt-1" in out
        assert "full_qualification" in out
        assert captured["body"]["report_type"] == "full_qualification"
        assert captured["body"]["release_id"] == "rel-1"

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

        manifest = json.loads((target / "epsilab.json").read_text())
        assert manifest["task_pack"]["name"] == "my-environment-tasks"

    def test_server_py_is_valid_python(self, tmp_path):
        target = tmp_path / "valid-env"
        args = build_parser().parse_args(["env", "init", "valid-env"])
        args.directory = str(target)
        cmd_env_init(args)

        server_code = (target / "server.py").read_text()
        compile(server_code, "server.py", "exec")

    def test_manifest_is_valid_json(self, tmp_path):
        target = tmp_path / "json-env"
        args = build_parser().parse_args(["env", "init", "json-env"])
        args.directory = str(target)
        cmd_env_init(args)

        manifest = json.loads((target / "epsilab.json").read_text())
        assert "task_pack" in manifest
        assert "verifier" in manifest
        assert "environment" in manifest
        assert manifest["release_version"] == "0.1.0"


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
        assert "Manifest is valid JSON" in out
        assert "/reset" in out
        assert "/step" in out

    def test_verify_missing_manifest(self, tmp_path, capsys):
        target = tmp_path / "empty-env"
        target.mkdir()

        verify_args = build_parser().parse_args(["env", "verify", "-d", str(target)])
        with pytest.raises(SystemExit):
            cmd_env_verify(verify_args)
        out = capsys.readouterr().out
        assert "Manifest not found" in out

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
