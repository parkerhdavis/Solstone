# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for solstone.think.benchmark.estimate — hardware-class resolution + estimates."""

from __future__ import annotations

import json

import pytest

from solstone.think.benchmark import estimate as est_mod
from solstone.think.benchmark.estimate import (
    estimate_output_tok_s,
    estimate_task_time_s,
    list_prevetted_models,
    resolve_hardware_class,
)
from solstone.think.models import LOCAL_MODEL

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
        "local/measured-model:1b": {
            "label": "Measured",
            "tier_hint": 3,
            "size_gb": 1.0,
            "capabilities": ["generate"],
            "vram_required_gb": 2,
            "benchmarks": {
                "rtx-3090": {"output_tok_s": 50.0, "prompt_tok_s": 1000.0},
            },
        },
        "local/unmeasured:9b": {
            "label": "Unmeasured",
            "tier_hint": 2,
            "size_gb": 5.5,
            "capabilities": ["generate"],
            "vram_required_gb": 8,
            "benchmarks": {},
        },
        "local/huge-vision:72b": {
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
        est = estimate_output_tok_s("local/measured-model:1b", "rtx-3090")
        assert est.confidence == "measured"
        assert est.tok_s == 50.0
        assert est.source_class == "rtx-3090"

    def test_interpolated_when_different_class(self):
        # rtx-4090 target: 165 * 1000 = 165000
        # rtx-3090 source: 70 * 900 = 63000
        # scale factor: 165000 / 63000 ≈ 2.619
        # expected: 50 * 2.619 ≈ 131.0
        est = estimate_output_tok_s("local/measured-model:1b", "rtx-4090")
        assert est.confidence == "interpolated"
        assert est.source_class == "rtx-3090"
        assert est.tok_s is not None
        assert 125 < est.tok_s < 135

    def test_unknown_when_model_has_no_benchmarks(self):
        est = estimate_output_tok_s("local/unmeasured:9b", "rtx-4090")
        assert est.confidence == "unknown"
        assert est.tok_s is None
        assert est.source_class is None

    def test_unknown_when_cpu_only(self):
        est = estimate_output_tok_s("local/measured-model:1b", "cpu-only")
        assert est.confidence == "unknown"
        assert est.tok_s is None

    def test_unknown_for_missing_model(self):
        est = estimate_output_tok_s("local/not-in-registry:1b", "rtx-4090")
        assert est.confidence == "unknown"
        assert est.tok_s is None


class TestListPrevettedModels:
    def test_marks_vram_overflow(self):
        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        by_id = {row["model_id"]: row for row in rows}
        # 24 GB VRAM fits the small/medium models but not the 44 GB vision model.
        assert by_id["local/measured-model:1b"]["fits_in_vram"] is True
        assert by_id["local/unmeasured:9b"]["fits_in_vram"] is True
        assert by_id["local/huge-vision:72b"]["fits_in_vram"] is False

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

    def test_segment_estimate_is_attached_to_each_row(self, monkeypatch):
        # Patch in a minimal scenario whose talents map cleanly to the
        # FAKE_TASKS catalog used elsewhere in this file.
        fake_segments = {
            "scenarios": {
                "solo_active": {
                    "audio_minutes": 5,
                    "qualified_frames": 3,
                    "fixed_overhead_s": 3.0,
                    "talents": [
                        {"task_id": "chat_reply", "count": 1},
                    ],
                }
            }
        }
        monkeypatch.setattr(
            est_mod,
            "load_segments",
            fake_segments.__call__ if False else (lambda: fake_segments),
        )

        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        for row in rows:
            assert "segment_estimate" in row
            seg = row["segment_estimate"]
            assert seg["scenario"] == "solo_active"
            assert "tier_models" in seg
            assert "self_attributed_tiers" in seg
            assert "confidence" in seg

    def test_segment_default_tier_models_pick_smallest_per_capability(
        self, monkeypatch
    ):
        fake_segments = {
            "scenarios": {
                "solo_active": {
                    "audio_minutes": 5,
                    "qualified_frames": 1,
                    "fixed_overhead_s": 3.0,
                    "talents": [{"task_id": "chat_reply", "count": 1}],
                }
            }
        }
        monkeypatch.setattr(est_mod, "load_segments", lambda: fake_segments)

        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        by_id = {r["model_id"]: r for r in rows}

        # The smallest generate model in FAKE_REGISTRY is measured-model:1b
        # (1 GB) — it should be the comparison baseline for vision-only rows.
        # The smallest vision model is huge-vision:72b (only one, 40 GB).
        vision_row = by_id["local/huge-vision:72b"]
        seg = vision_row["segment_estimate"]
        # This row attributes itself to the vision tier; generate/cogitate
        # come from the smallest applicable registry models.
        assert seg["self_attributed_tiers"] == ["vision"]
        assert seg["tier_models"]["vision"] == "local/huge-vision:72b"
        assert seg["tier_models"]["generate"] == "local/measured-model:1b"

        # A generate-capable row attributes itself to generate (no
        # cogitate capability in this fixture, so cogitate doesn't appear).
        gen_row = by_id["local/measured-model:1b"]
        gen_seg = gen_row["segment_estimate"]
        assert "generate" in gen_seg["self_attributed_tiers"]
        assert gen_seg["tier_models"]["generate"] == "local/measured-model:1b"

    def test_segment_total_unknown_when_audio_lane_unmeasured(self, monkeypatch):
        # transcriber=None (default) leaves the audio lane unknown, which
        # downgrades total_seconds to None per the existing contract.
        fake_segments = {
            "scenarios": {
                "solo_active": {
                    "audio_minutes": 5,
                    "qualified_frames": 1,
                    "fixed_overhead_s": 3.0,
                    "talents": [{"task_id": "chat_reply", "count": 1}],
                }
            }
        }
        monkeypatch.setattr(est_mod, "load_segments", lambda: fake_segments)

        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)  # transcriber=None
        for row in rows:
            seg = row["segment_estimate"]
            assert seg["audio_seconds"] is None
            assert seg["total_seconds"] is None

    def test_attaches_task_times_by_capability(self):
        hardware = {"gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24}]}
        rows = list_prevetted_models(hardware)
        by_id = {row["model_id"]: row for row in rows}

        # Text-capable model: gets chat_reply but not screen_frame (vision-only)
        measured = by_id["local/measured-model:1b"]
        assert "chat_reply" in measured["tasks"]
        assert "screen_frame" not in measured["tasks"]
        assert measured["tasks"]["chat_reply"]["seconds"] is not None

        # Vision-only model: gets screen_frame but not chat_reply (text task)
        vision = by_id["local/huge-vision:72b"]
        assert "chat_reply" not in vision["tasks"]
        assert "screen_frame" in vision["tasks"]


class TestEstimateTaskTime:
    def test_formula_task_time_is_interpolated_even_with_measured_tok_s(self):
        # measured-model:1b on rtx-3090 has measured tok/s but no direct
        # task measurement. Formula-derived time => "interpolated".
        # chat_reply: 500 prompt / 200 output
        # seconds = 500/1000 + 200/50 = 0.5 + 4.0 = 4.5
        est = estimate_task_time_s("local/measured-model:1b", "rtx-3090", "chat_reply")
        assert est.confidence == "interpolated"
        assert est.seconds is not None
        assert abs(est.seconds - 4.5) < 0.01

    def test_direct_task_measurement_yields_measured(self):
        # Stub a direct task measurement on measured-model:1b for rtx-3090.
        registry = {
            "models": {
                "local/measured-model:1b": {
                    **FAKE_REGISTRY["models"]["local/measured-model:1b"],
                    "benchmarks": {
                        "rtx-3090": {
                            "output_tok_s": 50.0,
                            "prompt_tok_s": 1000.0,
                            "tasks": {
                                "chat_reply": {"seconds": 3.8},
                            },
                        },
                    },
                },
            },
        }
        # Override the autouse loader for just this test.
        original = est_mod.load_registry
        est_mod.load_registry = lambda: registry
        try:
            est = estimate_task_time_s(
                "local/measured-model:1b", "rtx-3090", "chat_reply"
            )
        finally:
            est_mod.load_registry = original
        assert est.confidence == "measured"
        assert est.seconds == 3.8
        assert est.source_class == "rtx-3090"

    def test_interpolated_task_time_cross_hardware(self):
        # rtx-4090 has no direct measurement — falls back to formula with
        # interpolated tok/s, which is also "interpolated" confidence.
        est = estimate_task_time_s("local/measured-model:1b", "rtx-4090", "chat_reply")
        assert est.confidence == "interpolated"
        assert est.seconds is not None
        assert est.seconds > 0
        assert est.seconds < 4.5  # rtx-4090 is faster than rtx-3090

    def test_unknown_when_model_has_no_benchmarks(self):
        est = estimate_task_time_s("local/unmeasured:9b", "rtx-4090", "chat_reply")
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_unknown_for_missing_task(self):
        est = estimate_task_time_s(
            "local/measured-model:1b", "rtx-3090", "no_such_task"
        )
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_unknown_for_missing_model(self):
        est = estimate_task_time_s("local/not-in-registry:1b", "rtx-3090", "chat_reply")
        assert est.confidence == "unknown"
        assert est.seconds is None

    def test_vision_task_uses_vision_prompt_tok_s(self):
        # huge-vision:72b has prompt_tok_s=200, output_tok_s=30 on dgx-spark.
        # screen_frame: 200 prompt / 400 output
        # seconds = 200/200 + 400/30 = 1 + 13.33 = 14.33
        # Formula-derived => "interpolated" even with measured tok/s.
        est = estimate_task_time_s("local/huge-vision:72b", "dgx-spark", "screen_frame")
        assert est.confidence == "interpolated"
        assert est.seconds is not None
        assert 13 < est.seconds < 16


class TestRegistryIntegrity:
    """Invariants on the real shipped models.json (read from disk, so the
    autouse fake-loader fixture doesn't apply)."""

    def _real_registry(self) -> dict:
        return json.loads(est_mod._REGISTRY_FILE.read_text())

    def test_served_flag_matches_local_model(self):
        # Drift guard: exactly one registry row carries served=true and it is
        # the model pinned in think/models.py LOCAL_MODEL. The served flag is
        # documentation/UI data; this keeps it from silently disagreeing with
        # the single source of truth for what the provider actually serves.
        models = self._real_registry()["models"]
        served = [mid for mid, spec in models.items() if spec.get("served")]
        assert served == [LOCAL_MODEL], (
            f"served-flagged rows {served} must be exactly [{LOCAL_MODEL}]"
        )

    def test_candidates_are_capability_supersets_of_nothing_unexpected(self):
        # Every local row uses the tasks.json tier_role vocabulary and never
        # claims audio (unsupported on the bundle — runs through Whisper STT).
        models = self._real_registry()["models"]
        allowed = {"generate", "cogitate", "vision"}
        for mid, spec in models.items():
            caps = set(spec.get("capabilities") or [])
            assert caps <= allowed, (
                f"{mid} has unexpected capabilities {caps - allowed}"
            )
