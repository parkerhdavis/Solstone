# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.benchmark.estimate — hardware-class resolution + estimates."""

from __future__ import annotations

import pytest

from think.benchmark import estimate as est_mod
from think.benchmark.estimate import (
    estimate_output_tok_s,
    estimate_task_time_s,
    list_prevetted_models,
    resolve_hardware_class,
)

FAKE_REFERENCE = {
    "classes": {
        "rtx-4090": {
            "label": "NVIDIA GeForce RTX 4090",
            "fp16_tflops": 165.0,
            "mem_bandwidth_gbs": 1000.0,
            "vram_gb": 24,
        },
        "rtx-3090": {
            "label": "NVIDIA GeForce RTX 3090",
            "fp16_tflops": 70.0,
            "mem_bandwidth_gbs": 900.0,
            "vram_gb": 24,
        },
        "dgx-spark": {
            "label": "NVIDIA DGX Spark (GB10)",
            "fp16_tflops": 500.0,
            "mem_bandwidth_gbs": 273.0,
            "vram_gb": 128,
        },
        "cpu-only": {
            "label": "CPU-only",
            "fp16_tflops": 0.0,
            "mem_bandwidth_gbs": 0.0,
            "vram_gb": 0,
        },
    },
    "aliases": {
        "NVIDIA GeForce RTX 4090": "rtx-4090",
        "NVIDIA GeForce RTX 3090": "rtx-3090",
        "NVIDIA DGX Spark": "dgx-spark",
    },
}


FAKE_REGISTRY = {
    "models": {
        "ollama-local/measured-model:1b": {
            "label": "Measured",
            "tier_hint": 3,
            "size_gb": 1.0,
            "capabilities": ["generate"],
            "vram_required_gb": 2,
            "benchmarks": {
                "rtx-3090": {"output_tok_s": 50.0, "prompt_tok_s": 1000.0},
            },
        },
        "ollama-local/unmeasured:9b": {
            "label": "Unmeasured",
            "tier_hint": 2,
            "size_gb": 5.5,
            "capabilities": ["generate"],
            "vram_required_gb": 8,
            "benchmarks": {},
        },
        "ollama-local/huge-vision:72b": {
            "label": "Huge vision",
            "tier_hint": 1,
            "size_gb": 40.0,
            "capabilities": ["vision"],
            "vram_required_gb": 44,
            "benchmarks": {
                "dgx-spark": {"output_tok_s": 30.0, "prompt_tok_s": 200.0},
            },
        },
    },
}


FAKE_TASKS = {
    "tasks": {
        "chat_reply": {
            "label": "Chat reply",
            "prompt_tokens": 500,
            "output_tokens": 200,
            "mode": "text",
            "tier_role": "cogitate",
            "ui_priority": 1,
        },
        "screen_frame": {
            "label": "Screen frame",
            "prompt_tokens": 200,
            "output_tokens": 400,
            "mode": "vision",
            "tier_role": "vision",
            "ui_priority": 1,
        },
    },
}


@pytest.fixture(autouse=True)
def patch_loaders(monkeypatch):
    """Replace the cached loaders with fixtures for every test."""
    monkeypatch.setattr(est_mod, "load_reference", lambda: FAKE_REFERENCE)
    monkeypatch.setattr(est_mod, "load_registry", lambda: FAKE_REGISTRY)
    monkeypatch.setattr(est_mod, "load_tasks", lambda: FAKE_TASKS)


class TestResolveHardwareClass:
    def test_exact_alias_hit(self):
        assert resolve_hardware_class("NVIDIA GeForce RTX 4090") == "rtx-4090"

    def test_fuzzy_substring_hit(self):
        # Case-insensitive substring match against aliases.
        assert resolve_hardware_class("nvidia geforce rtx 3090") == "rtx-3090"

    def test_unknown_falls_back_to_cpu_only(self):
        assert resolve_hardware_class("NVIDIA Totally Fake GPU 999") == "cpu-only"

    def test_none_yields_cpu_only(self):
        assert resolve_hardware_class(None) == "cpu-only"

    def test_empty_yields_cpu_only(self):
        assert resolve_hardware_class("") == "cpu-only"


class TestEstimate:
    def test_measured_exact_match(self):
        est = estimate_output_tok_s("ollama-local/measured-model:1b", "rtx-3090")
        assert est.confidence == "measured"
        assert est.tok_s == 50.0
        assert est.source_class == "rtx-3090"

    def test_interpolated_when_different_class(self):
        # rtx-4090 target: 165 * 1000 = 165000
        # rtx-3090 source: 70 * 900 = 63000
        # scale factor: 165000 / 63000 ≈ 2.619
        # expected: 50 * 2.619 ≈ 131.0
        est = estimate_output_tok_s("ollama-local/measured-model:1b", "rtx-4090")
        assert est.confidence == "interpolated"
        assert est.source_class == "rtx-3090"
        assert est.tok_s is not None
        assert 125 < est.tok_s < 135

    def test_unknown_when_model_has_no_benchmarks(self):
        est = estimate_output_tok_s("ollama-local/unmeasured:9b", "rtx-4090")
        assert est.confidence == "unknown"
        assert est.tok_s is None
        assert est.source_class is None

    def test_unknown_when_cpu_only(self):
        est = estimate_output_tok_s("ollama-local/measured-model:1b", "cpu-only")
        assert est.confidence == "unknown"
        assert est.tok_s is None

    def test_unknown_for_missing_model(self):
        est = estimate_output_tok_s("ollama-local/not-in-registry:1b", "rtx-4090")
        assert est.confidence == "unknown"
        assert est.tok_s is None


class TestListPrevettedModels:
    def test_marks_vram_overflow(self):
        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        by_id = {row["model_id"]: row for row in rows}
        # 24 GB VRAM fits the small/medium models but not the 44 GB vision model.
        assert by_id["ollama-local/measured-model:1b"]["fits_in_vram"] is True
        assert by_id["ollama-local/unmeasured:9b"]["fits_in_vram"] is True
        assert by_id["ollama-local/huge-vision:72b"]["fits_in_vram"] is False

    def test_returns_all_models_with_estimates(self):
        hardware = {"gpus": [{"name": "NVIDIA DGX Spark", "vram_gb": 128}]}
        rows = list_prevetted_models(hardware)
        assert len(rows) == len(FAKE_REGISTRY["models"])
        for row in rows:
            assert "estimate" in row
            assert "confidence" in row["estimate"]

    def test_no_hardware_yields_all_unknown_cpu_only(self):
        rows = list_prevetted_models(None)
        for row in rows:
            assert row["estimate"]["hardware_class"] == "cpu-only"
            assert row["estimate"]["confidence"] == "unknown"

    def test_attaches_task_times_by_capability(self):
        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        by_id = {row["model_id"]: row for row in rows}

        # Text-capable model: gets chat_reply but not screen_frame (vision-only)
        measured = by_id["ollama-local/measured-model:1b"]
        assert "chat_reply" in measured["tasks"]
        assert "screen_frame" not in measured["tasks"]
        assert measured["tasks"]["chat_reply"]["seconds"] is not None

        # Vision-only model: gets screen_frame but not chat_reply (text task)
        vision = by_id["ollama-local/huge-vision:72b"]
        assert "chat_reply" not in vision["tasks"]
        assert "screen_frame" in vision["tasks"]


class TestEstimateTaskTime:
    def test_measured_task_time(self):
        # measured-model:1b on rtx-3090 has prompt_tok_s=1000, output_tok_s=50.
        # chat_reply: 500 prompt / 200 output
        # seconds = 500/1000 + 200/50 = 0.5 + 4.0 = 4.5
        est = estimate_task_time_s(
            "ollama-local/measured-model:1b", "rtx-3090", "chat_reply"
        )
        assert est.confidence == "measured"
        assert est.seconds is not None
        assert abs(est.seconds - 4.5) < 0.01
        assert est.source_class == "rtx-3090"

    def test_interpolated_task_time(self):
        # Same model, but rtx-4090 has no measurement — interpolates from rtx-3090.
        est = estimate_task_time_s(
            "ollama-local/measured-model:1b", "rtx-4090", "chat_reply"
        )
        assert est.confidence == "interpolated"
        assert est.seconds is not None
        assert est.seconds > 0
        # Should be FASTER than rtx-3090 since rtx-4090 has more throughput.
        assert est.seconds < 4.5

    def test_unknown_when_model_has_no_benchmarks(self):
        est = estimate_task_time_s(
            "ollama-local/unmeasured:9b", "rtx-4090", "chat_reply"
        )
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_unknown_for_missing_task(self):
        est = estimate_task_time_s(
            "ollama-local/measured-model:1b", "rtx-3090", "no_such_task"
        )
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_unknown_for_missing_model(self):
        est = estimate_task_time_s(
            "ollama-local/not-in-registry:1b", "rtx-3090", "chat_reply"
        )
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_vision_task_uses_vision_prompt_tok_s(self):
        # huge-vision:72b has prompt_tok_s=200, output_tok_s=30 on dgx-spark.
        # screen_frame: 200 prompt / 400 output
        # seconds = 200/200 + 400/30 = 1 + 13.33 = 14.33
        est = estimate_task_time_s(
            "ollama-local/huge-vision:72b", "dgx-spark", "screen_frame"
        )
        assert est.confidence == "measured"
        assert est.seconds is not None
        assert 13 < est.seconds < 16
