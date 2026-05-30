# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for benchmark CLI commands (``sol call benchmark ...``)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from typer.testing import CliRunner

from solstone.apps.benchmark import call as benchmark_call
from solstone.think.benchmark import estimate as est_mod
from solstone.think.call import call_app

runner = CliRunner()


def _fake_smi_result(stdout: str):
    class _R:
        pass

    r = _R()
    r.returncode = 0
    r.stdout = stdout
    r.stderr = ""
    return r


class TestProfile:
    def test_profile_writes_health_file(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            result = runner.invoke(call_app, ["benchmark", "profile", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_class"] == "cpu-only"
        assert payload["gpus"] == []

        health_file = journal_override / "health" / "hardware.json"
        assert health_file.exists()

    def test_profile_detects_nvidia_gpu(self, journal_override):
        stdout = "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
        with patch.object(subprocess, "run", return_value=_fake_smi_result(stdout)):
            result = runner.invoke(call_app, ["benchmark", "profile", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_class"] == "rtx-4090"
        assert len(payload["gpus"]) == 1
        assert payload["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"

    def test_profile_text_output_is_human_readable(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            result = runner.invoke(call_app, ["benchmark", "profile"])
        assert result.exit_code == 0
        assert "Platform:" in result.output
        assert "CPU:" in result.output
        assert "Hardware class:" in result.output


class TestListModels:
    def test_lists_without_hardware_probe(self, journal_override):
        with patch.object(benchmark_call, "_list_installed_models", return_value=set()):
            result = runner.invoke(call_app, ["benchmark", "list-models", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["hardware_probed"] is False
        assert len(payload["models"]) > 0
        for row in payload["models"]:
            assert row["installed"] is False

    def test_marks_installed_models(self, journal_override):
        # Seed with a probed-hardware file so we don't get the "unprobed" note.
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            runner.invoke(call_app, ["benchmark", "profile"])

        fake_installed = {"local/qwen3.5-4b"}
        with patch.object(
            benchmark_call, "_list_installed_models", return_value=fake_installed
        ):
            result = runner.invoke(call_app, ["benchmark", "list-models", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        by_id = {row["model_id"]: row for row in payload["models"]}
        assert by_id["local/qwen3.5-4b"]["installed"] is True
        assert by_id["local/qwen3.6-35b-a3b"]["installed"] is False


class TestEstimate:
    def test_requires_probed_hardware(self, journal_override):
        result = runner.invoke(
            call_app, ["benchmark", "estimate", "local/nemotron-3-nano-omni"]
        )
        assert result.exit_code == 1
        assert (
            "probe" in result.output.lower() or "probe" in (result.stderr or "").lower()
        )

    def test_rejects_unknown_model(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app, ["benchmark", "estimate", "local/not-a-real-model"]
        )
        assert result.exit_code == 1

    def test_estimates_known_model(self, journal_override):
        stdout = "NVIDIA GeForce RTX 4090, 24564, 550.144.03\n"
        with patch.object(subprocess, "run", return_value=_fake_smi_result(stdout)):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app,
            ["benchmark", "estimate", "local/nemotron-3-nano-omni", "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["model_id"] == "local/nemotron-3-nano-omni"
        assert payload["hardware_class"] == "rtx-4090"
        # Served/candidate rows ship with empty benchmarks until the Spark
        # head-to-head (Plan phase B3), so confidence is unknown for now;
        # the assertion accepts the full set so it stays valid post-B3.
        assert payload["confidence"] in ("unknown", "measured", "interpolated")

    def test_task_time_estimate(self, journal_override, monkeypatch):
        # Seed a dgx-spark chat_reply measurement so the direct-measurement
        # path (a measured wall-clock wins over the formula -> confidence
        # "measured") is exercised deterministically, independent of what the
        # live registry has been measured for. The served/candidate rows ship
        # with empty benchmarks until the Spark head-to-head (Plan phase B3).
        fake_registry = {
            "models": {
                "local/nemotron-3-nano-omni": {
                    "label": "Nemotron",
                    "served": True,
                    "tier_hint": 1,
                    "size_gb": 35.1,
                    "capabilities": ["generate", "cogitate", "vision"],
                    "vram_required_gb": 48,
                    "benchmarks": {
                        "dgx-spark": {
                            "output_tok_s": 90.0,
                            "prompt_tok_s": 900.0,
                            "tasks": {"chat_reply": {"seconds": 1.2}},
                        }
                    },
                }
            }
        }
        monkeypatch.setattr(benchmark_call, "load_registry", lambda: fake_registry)
        monkeypatch.setattr(est_mod, "load_registry", lambda: fake_registry)

        stdout = "NVIDIA DGX Spark, 0, 580.142\n"
        with patch.object(subprocess, "run", return_value=_fake_smi_result(stdout)):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app,
            [
                "benchmark",
                "estimate",
                "local/nemotron-3-nano-omni",
                "--task",
                "chat_reply",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["task_id"] == "chat_reply"
        assert payload["seconds"] is not None
        # A direct task measurement exists for this hardware class, so it
        # wins over the tok/s formula -> "measured" (ground truth).
        assert payload["confidence"] == "measured"

    def test_task_time_rejects_unknown_task(self, journal_override):
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            runner.invoke(call_app, ["benchmark", "profile"])
        result = runner.invoke(
            call_app,
            [
                "benchmark",
                "estimate",
                "local/nemotron-3-nano-omni",
                "--task",
                "definitely_not_a_task",
            ],
        )
        assert result.exit_code == 1


class TestTasks:
    def test_lists_catalog(self):
        result = runner.invoke(call_app, ["benchmark", "tasks", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "tasks" in payload
        # At minimum, the seed catalog should include chat_reply and screen_frame.
        assert "chat_reply" in payload["tasks"]
        assert "screen_frame" in payload["tasks"]
        assert payload["tasks"]["screen_frame"]["mode"] == "vision"
