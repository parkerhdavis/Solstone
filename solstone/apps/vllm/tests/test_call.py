# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for vLLM CLI commands (``sol call vllm ...``)."""

from __future__ import annotations

import json as jsonlib
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from apps.vllm import call as vllm_call

runner = CliRunner()


# ---------------------------------------------------------------------------
# _build_serve_argv — pure-function tests (no subprocess, no docker)
# ---------------------------------------------------------------------------


class TestBuildServeArgv:
    def test_minimal_entry(self):
        argv = vllm_call._build_serve_argv(
            "test-model",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "test-model",
                "model": "Qwen/Qwen3.5-2B",
            },
        )
        # Sanity: must be a docker-run argv
        assert argv[0] == "docker"
        assert argv[1] == "run"
        assert "--rm" in argv
        # Container name defaults from friendly name
        assert "vllm-test-model" in argv
        # Port mapping derived from base_url
        assert "8000:8000" in argv
        # Image defaults
        assert vllm_call.DEFAULT_IMAGE in argv
        # The model gets passed to vllm serve as the positional
        entrypoint_cmd = argv[-1]
        assert "vllm serve Qwen/Qwen3.5-2B" in entrypoint_cmd
        assert "--served-model-name test-model" in entrypoint_cmd
        # No audio extras pip install when needs_audio_extras is omitted
        assert "pip install" not in entrypoint_cmd

    def test_audio_extras_prepends_pip(self):
        argv = vllm_call._build_serve_argv(
            "omni",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "omni",
                "model": "nvidia/Some-Omni-Model",
                "needs_audio_extras": True,
            },
        )
        entrypoint_cmd = argv[-1]
        assert entrypoint_cmd.startswith("pip install --no-cache-dir 'vllm[audio]' &&")
        assert "vllm serve" in entrypoint_cmd

    def test_custom_port_overrides_base_url_port(self):
        argv = vllm_call._build_serve_argv(
            "x",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "X/Y",
                "port": 8001,
            },
        )
        # Port mapping uses the explicit port, not the base_url port
        assert "8001:8000" in argv
        assert "8000:8000" not in argv

    def test_vllm_args_appended_to_serve(self):
        argv = vllm_call._build_serve_argv(
            "x",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "X/Y",
                "vllm_args": [
                    "--reasoning-parser",
                    "qwen3_xml",
                    "--max-model-len",
                    "65536",
                ],
            },
        )
        entrypoint_cmd = argv[-1]
        assert "--reasoning-parser qwen3_xml" in entrypoint_cmd
        assert "--max-model-len 65536" in entrypoint_cmd

    def test_extra_docker_args_inserted(self):
        argv = vllm_call._build_serve_argv(
            "x",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "X/Y",
                "extra_docker_args": ["-e", "FOO=bar", "-v", "/tmp/extra:/extra"],
            },
        )
        # Inserted before the entrypoint section
        assert "FOO=bar" in argv
        assert "/tmp/extra:/extra" in argv

    def test_missing_model_raises(self):
        with pytest.raises(Exception):
            vllm_call._build_serve_argv(
                "x",
                {"base_url": "http://localhost:8000", "served_model_name": "x"},
            )

    def test_custom_container_name(self):
        argv = vllm_call._build_serve_argv(
            "x",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "X/Y",
                "container_name": "my-custom-name",
            },
        )
        assert "my-custom-name" in argv
        assert "vllm-x" not in argv

    def test_torch_compile_cache_mount_present(self):
        # Phase 0 spike notes gotcha #5 — torch.compile cache must persist
        # across container restarts to skip the ~8s recompile.
        argv = vllm_call._build_serve_argv(
            "x",
            {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "X/Y",
            },
        )
        # Find the -v entries
        v_entries = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        # One should mount a host path to /root/.cache/vllm
        assert any(":/root/.cache/vllm" in v for v in v_entries), (
            f"expected /root/.cache/vllm mount in {v_entries}"
        )


# ---------------------------------------------------------------------------
# _resolve_server_entry — config lookup
# ---------------------------------------------------------------------------


class TestResolveServerEntry:
    def test_no_servers_raises(self):
        with patch.object(vllm_call, "_load_servers", return_value={}):
            with pytest.raises(Exception):
                vllm_call._resolve_server_entry(None)

    def test_single_server_auto_picks(self):
        with patch.object(
            vllm_call,
            "_load_servers",
            return_value={"only": {"base_url": "http://x"}},
        ):
            name, entry = vllm_call._resolve_server_entry(None)
            assert name == "only"
            assert entry["base_url"] == "http://x"

    def test_multiple_servers_requires_name(self):
        with patch.object(
            vllm_call,
            "_load_servers",
            return_value={"a": {}, "b": {}},
        ):
            with pytest.raises(Exception):
                vllm_call._resolve_server_entry(None)

    def test_explicit_name_picks_that_one(self):
        with patch.object(
            vllm_call,
            "_load_servers",
            return_value={"a": {"base_url": "http://a"}, "b": {"base_url": "http://b"}},
        ):
            name, entry = vllm_call._resolve_server_entry("b")
            assert name == "b"
            assert entry["base_url"] == "http://b"

    def test_unknown_name_raises(self):
        with patch.object(
            vllm_call,
            "_load_servers",
            return_value={"a": {}},
        ):
            with pytest.raises(Exception):
                vllm_call._resolve_server_entry("does-not-exist")


# ---------------------------------------------------------------------------
# CLI: list / status — text + JSON output
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_no_servers_friendly_message(self):
        with patch.object(vllm_call, "_load_servers", return_value={}):
            result = runner.invoke(vllm_call.app, ["list"])
        assert result.exit_code == 0
        assert "No vLLM servers configured" in result.output

    def test_lists_configured_servers(self):
        with patch.object(
            vllm_call,
            "_load_servers",
            return_value={
                "nemotron-omni": {
                    "base_url": "http://localhost:8000",
                    "served_model_name": "nemotron-omni",
                    "model": "nvidia/Some-Repo",
                }
            },
        ):
            result = runner.invoke(vllm_call.app, ["list"])
        assert result.exit_code == 0
        assert "nemotron-omni" in result.output
        assert "http://localhost:8000" in result.output
        assert "nvidia/Some-Repo" in result.output

    def test_json_output_returns_raw_servers(self):
        servers = {
            "x": {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "Y/Z",
            }
        }
        with patch.object(vllm_call, "_load_servers", return_value=servers):
            result = runner.invoke(vllm_call.app, ["list", "--json"])
        assert result.exit_code == 0
        assert jsonlib.loads(result.output) == servers


class TestStatusCommand:
    def test_no_servers_friendly_message(self):
        with patch.object(vllm_call, "_load_servers", return_value={}):
            result = runner.invoke(vllm_call.app, ["status"])
        assert result.exit_code == 0
        assert "No vLLM servers configured" in result.output

    def test_pings_each_configured_server(self):
        servers = {
            "up-one": {"base_url": "http://localhost:8000"},
            "down-one": {"base_url": "http://localhost:8001"},
        }

        class _Resp:
            def __init__(self, data):
                self._data = data
                self.status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return self._data

        def fake_get(url, timeout=None):
            if url.startswith("http://localhost:8000"):
                return _Resp({"data": [{"id": "served-here"}]})
            raise RuntimeError("connection refused")

        with patch.object(vllm_call, "_load_servers", return_value=servers):
            with patch("httpx.get", side_effect=fake_get):
                result = runner.invoke(vllm_call.app, ["status", "--json"])
        assert result.exit_code == 0
        out = jsonlib.loads(result.output)
        assert out["up-one"]["reachable"] is True
        assert "served-here" in out["up-one"]["served_models"]
        assert out["down-one"]["reachable"] is False
        assert "connection refused" in out["down-one"]["error"]


# ---------------------------------------------------------------------------
# Smoke: serve fails fast when docker isn't available
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_missing_docker_exits_2(self):
        servers = {
            "x": {
                "base_url": "http://localhost:8000",
                "served_model_name": "x",
                "model": "Y/Z",
            }
        }
        with patch.object(vllm_call, "_load_servers", return_value=servers):
            with patch.object(vllm_call.shutil, "which", return_value=None):
                result = runner.invoke(vllm_call.app, ["serve"])
        assert result.exit_code == 2
        assert "docker" in result.output.lower()
