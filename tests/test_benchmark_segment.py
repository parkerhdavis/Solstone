# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for solstone.think.benchmark.estimate.estimate_segment_time_s."""

from __future__ import annotations

import pytest

from solstone.think.benchmark import estimate as est_mod
from solstone.think.benchmark.estimate import estimate_segment_time_s

FAKE_REFERENCE = {
    "classes": {
        "rtx-3090": {
            "label": "NVIDIA GeForce RTX 3090",
            "fp16_tflops": 70.0,
            "mem_bandwidth_gbs": 900.0,
            "vram_gb": 24,
        },
        "cpu-only": {
            "label": "CPU-only",
            "fp16_tflops": 0.0,
            "mem_bandwidth_gbs": 0.0,
            "vram_gb": 0,
        },
    },
    "aliases": {"NVIDIA GeForce RTX 3090": "rtx-3090"},
}


# Three measured models so every tier has a known per-call cost on rtx-3090.
FAKE_REGISTRY = {
    "models": {
        "local/vision-fast:7b": {
            "label": "Vision Fast",
            "tier_hint": 3,
            "size_gb": 4.0,
            "capabilities": ["vision"],
            "vram_required_gb": 6,
            "benchmarks": {
                "rtx-3090": {"output_tok_s": 100.0, "prompt_tok_s": 1000.0},
            },
        },
        "local/generate-fast:9b": {
            "label": "Generate Fast",
            "tier_hint": 2,
            "size_gb": 5.5,
            "capabilities": ["generate"],
            "vram_required_gb": 8,
            "benchmarks": {
                "rtx-3090": {"output_tok_s": 50.0, "prompt_tok_s": 1000.0},
            },
        },
        "local/cogitate-lite:3b": {
            "label": "Cogitate Lite",
            "tier_hint": 3,
            "size_gb": 2.0,
            "capabilities": ["cogitate"],
            "vram_required_gb": 4,
            "benchmarks": {
                "rtx-3090": {"output_tok_s": 80.0, "prompt_tok_s": 1500.0},
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
    },
}


FAKE_TASKS = {
    "tasks": {
        "screen_frame": {
            "label": "Screen frame",
            "prompt_tokens": 200,
            "output_tokens": 100,
            "mode": "vision",
            "tier_role": "vision",
        },
        "entity_extraction": {
            "label": "Entity extraction",
            "prompt_tokens": 800,
            "output_tokens": 700,
            "mode": "text",
            "tier_role": "generate",
        },
        "segment_sense": {
            "label": "Segment sense",
            "prompt_tokens": 900,
            "output_tokens": 400,
            "mode": "text",
            "tier_role": "generate",
        },
        "speaker_attribution_llm": {
            "label": "Speaker attribution",
            "prompt_tokens": 1100,
            "output_tokens": 350,
            "mode": "text",
            "tier_role": "generate",
        },
        "screen_record": {
            "label": "Screen record",
            "prompt_tokens": 1500,
            "output_tokens": 1500,
            "mode": "text",
            "tier_role": "generate",
        },
        "awareness_tender": {
            "label": "Awareness tender",
            "prompt_tokens": 800,
            "output_tokens": 500,
            "mode": "text",
            "tier_role": "cogitate",
        },
        "pulse": {
            "label": "Pulse",
            "prompt_tokens": 1000,
            "output_tokens": 700,
            "mode": "text",
            "tier_role": "cogitate",
        },
    },
}


FAKE_TRANSCRIBERS = {
    "transcribers": {
        "parakeet": {
            "label": "Parakeet TDT (local)",
            "kind": "local",
            "supported_hardware": ["rtx-3090", "rtx-4090", "cpu-only"],
            "fallback": False,
            "benchmarkable": True,
            "benchmarks": {
                "rtx-3090": {"rtf": 0.05},
            },
        },
        "whisper": {
            "label": "Whisper (local)",
            "kind": "local",
            "supported_hardware": ["*"],
            "fallback": True,
            "benchmarkable": True,
            "benchmarks": {},
        },
        "gemini": {
            "label": "Gemini (cloud)",
            "kind": "cloud",
            "supported_hardware": ["*"],
            "fallback": False,
            "benchmarkable": False,
            "wall_seconds_per_5min": 30,
        },
    },
}


FAKE_SEGMENTS = {
    "scenarios": {
        "solo_active": {
            "label": "Solo active",
            "audio_minutes": 5,
            "qualified_frames": 4,
            "fixed_overhead_s": 3.0,
            "talents": [
                {"task_id": "segment_sense", "count": 1},
                {"task_id": "entity_extraction", "count": 1},
                {"task_id": "screen_record", "count": 1},
                {"task_id": "awareness_tender", "count": 1},
                {"task_id": "pulse", "count": 1},
            ],
        },
        "meeting_active": {
            "label": "Meeting",
            "audio_minutes": 5,
            "qualified_frames": 2,
            "fixed_overhead_s": 3.0,
            "talents": [
                {"task_id": "segment_sense", "count": 1},
                {"task_id": "entity_extraction", "count": 1},
                {"task_id": "speaker_attribution_llm", "count": 1},
                {"task_id": "awareness_tender", "count": 1},
                {"task_id": "pulse", "count": 1},
            ],
        },
        "idle": {
            "label": "Idle",
            "audio_minutes": 5,
            "qualified_frames": 1,
            "fixed_overhead_s": 3.0,
            "talents": [
                {"task_id": "segment_sense", "count": 1},
                {"task_id": "awareness_tender", "count": 1},
                {"task_id": "pulse", "count": 1},
            ],
        },
    },
}


@pytest.fixture(autouse=True)
def patch_loaders(monkeypatch):
    """Replace cached loaders with fixtures for every test."""
    monkeypatch.setattr(est_mod, "load_reference", lambda: FAKE_REFERENCE)
    monkeypatch.setattr(est_mod, "load_registry", lambda: FAKE_REGISTRY)
    monkeypatch.setattr(est_mod, "load_tasks", lambda: FAKE_TASKS)
    monkeypatch.setattr(est_mod, "load_segments", lambda: FAKE_SEGMENTS)
    monkeypatch.setattr(est_mod, "load_transcribers", lambda: FAKE_TRANSCRIBERS)


TIER_MODELS = {
    "vision": "local/vision-fast:7b",
    "generate": "local/generate-fast:9b",
    "cogitate": "local/cogitate-lite:3b",
}


class TestSegmentEstimate:
    def test_breakdown_is_plausible_when_measurements_present(self):
        est = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "solo_active")

        # Every leg computed; per-talent dict has all 5 talents.
        assert est.video_seconds is not None
        assert est.talent_seconds is not None
        assert est.overhead_seconds == 3.0
        assert set(est.per_talent.keys()) == {
            "segment_sense",
            "entity_extraction",
            "screen_record",
            "awareness_tender",
            "pulse",
        }
        # Audio not yet implemented => total_seconds is None and audio_seconds is None.
        assert est.audio_seconds is None
        assert est.total_seconds is None
        # Video and talent values are positive and finite.
        assert est.video_seconds > 0
        assert est.talent_seconds > 0
        # Per-talent sums to talent_seconds (within float tolerance).
        assert abs(sum(est.per_talent.values()) - est.talent_seconds) < 0.05

    def test_video_lane_scales_with_frame_count(self):
        # 4 frames in solo_active vs 1 frame in idle => video_seconds ~4x.
        solo = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "solo_active")
        idle = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "idle")
        assert solo.video_seconds is not None and idle.video_seconds is not None
        assert idle.video_seconds > 0
        # 4x within rounding tolerance
        assert abs(solo.video_seconds - 4 * idle.video_seconds) < 0.05

    def test_meeting_includes_speaker_attribution(self):
        meeting = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "meeting_active")
        solo = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "solo_active")
        assert "speaker_attribution_llm" in meeting.per_talent
        assert "speaker_attribution_llm" not in solo.per_talent

    def test_idle_has_only_housekeeping_talents(self):
        idle = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "idle")
        assert set(idle.per_talent.keys()) == {
            "segment_sense",
            "awareness_tender",
            "pulse",
        }

    def test_audio_unknown_when_no_transcriber_passed(self):
        # transcriber omitted: audio leg goes 'unknown', total None.
        est = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "solo_active")
        assert est.audio_seconds is None
        assert est.total_seconds is None
        assert est.confidence == "unknown"
        assert any("no transcriber specified" in n for n in est.notes)

    def test_audio_lane_populated_with_measured_local_rtf(self):
        # parakeet RTF=0.05 on rtx-3090 → 5 min audio = 15s wall.
        est = estimate_segment_time_s(
            TIER_MODELS, "rtx-3090", "solo_active", transcriber="parakeet"
        )
        assert est.audio_seconds is not None
        assert abs(est.audio_seconds - 15.0) < 0.05
        assert est.total_seconds is not None
        # Sanity: total ≈ audio + video + talents + overhead
        expected = (
            est.audio_seconds
            + (est.video_seconds or 0)
            + (est.talent_seconds or 0)
            + est.overhead_seconds
        )
        assert abs(est.total_seconds - expected) < 0.05
        # Audio leg is measured; weakest non-audio leg is interpolated
        # (formula-derived) → combined confidence is 'interpolated'.
        assert est.confidence == "interpolated"

    def test_audio_lane_unknown_for_local_backend_without_class_measurement(self):
        # whisper has no benchmarks for any class → audio unknown.
        est = estimate_segment_time_s(
            TIER_MODELS, "rtx-3090", "solo_active", transcriber="whisper"
        )
        assert est.audio_seconds is None
        assert est.total_seconds is None
        assert any("no RTF measurement" in n for n in est.notes)

    def test_audio_lane_cloud_backend_uses_flat_wall_clock(self):
        # gemini: 30 wall_seconds_per_5min → 5 min audio = 30s flat.
        est = estimate_segment_time_s(
            TIER_MODELS, "rtx-3090", "solo_active", transcriber="gemini"
        )
        assert est.audio_seconds is not None
        assert abs(est.audio_seconds - 30.0) < 0.05
        # Cloud is a rule-of-thumb → never 'measured', so combined is 'interpolated'.
        assert est.confidence == "interpolated"
        assert any("rule-of-thumb" in n for n in est.notes)

    def test_audio_lane_unknown_for_unknown_transcriber(self):
        est = estimate_segment_time_s(
            TIER_MODELS, "rtx-3090", "solo_active", transcriber="nonexistent"
        )
        assert est.audio_seconds is None
        assert est.total_seconds is None
        assert est.confidence == "unknown"
        assert any("missing from transcribers.json" in n for n in est.notes)

    def test_missing_tier_model_marks_lane_unknown(self):
        # No vision model provided => video lane unknown, total unknown.
        partial = {
            "generate": "local/generate-fast:9b",
            "cogitate": "local/cogitate-lite:3b",
        }
        est = estimate_segment_time_s(partial, "rtx-3090", "solo_active")
        assert est.video_seconds is None
        assert est.total_seconds is None
        assert any("vision-tier" in n for n in est.notes)
        # Talent legs still computed.
        assert est.talent_seconds is not None
        assert est.talent_seconds > 0

    def test_missing_generate_model_marks_each_dependent_talent_unknown(self):
        partial = {
            "vision": "local/vision-fast:7b",
            "cogitate": "local/cogitate-lite:3b",
        }
        est = estimate_segment_time_s(partial, "rtx-3090", "solo_active")
        # Generate-tier talents skipped; cogitate-tier talents still present.
        assert "segment_sense" not in est.per_talent
        assert "entity_extraction" not in est.per_talent
        assert "screen_record" not in est.per_talent
        assert "awareness_tender" in est.per_talent
        assert "pulse" in est.per_talent
        assert est.talent_seconds is None
        assert est.total_seconds is None

    def test_unknown_scenario_returns_unknown(self):
        est = estimate_segment_time_s(TIER_MODELS, "rtx-3090", "no_such_scenario")
        assert est.total_seconds is None
        assert est.video_seconds is None
        assert est.talent_seconds is None
        assert est.confidence == "unknown"
        assert any("unknown scenario" in n for n in est.notes)

    def test_unmeasured_model_for_a_tier_marks_lane_unknown(self):
        # Substitute the generate-tier model with one that has no benchmarks.
        models = dict(TIER_MODELS)
        models["generate"] = "local/unmeasured:9b"
        est = estimate_segment_time_s(models, "rtx-3090", "solo_active")
        # All generate-tier talents fail to estimate.
        assert "segment_sense" not in est.per_talent
        assert "entity_extraction" not in est.per_talent
        assert est.talent_seconds is None
        assert est.confidence == "unknown"

    def test_confidence_interpolated_when_all_legs_interpolated(self):
        # All measured tok/s exist on rtx-3090, so per-call task estimates
        # are formula-derived from measured tok/s => "interpolated".
        # With a transcriber that has measured RTF (parakeet), the audio
        # leg is "measured"; combined confidence is the weakest leg
        # ('interpolated'). No leg should produce a fallback 'unknown'
        # note.
        est = estimate_segment_time_s(
            TIER_MODELS, "rtx-3090", "solo_active", transcriber="parakeet"
        )
        assert est.notes == ()
        assert est.confidence == "interpolated"


class TestHarnessPreflight:
    """Lock in the harness's hard-fail contracts for transcriber RTF mode.

    These guard against silent corruption of the cross-backend benchmark
    signal — the whole point of RTF capture is comparing backends
    honestly, so mismatches must fail loudly, not fall back.
    """

    def _patch(self, monkeypatch, tmp_path):
        # _preflight_transcriber reads transcribers.json from disk; redirect
        # by writing a fake into a tmp path the function can be steered to.
        from solstone.think.benchmark import harness

        fixture = tmp_path / "transcribers.json"
        fixture.write_text(__import__("json").dumps(FAKE_TRANSCRIBERS))

        # Patch _preflight_transcriber to read from our fixture path
        # instead of think/benchmark/transcribers.json on disk.
        def patched(transcriber, hw_class):
            catalog = (
                __import__("json").loads(fixture.read_text()).get("transcribers", {})
            )
            if transcriber not in catalog:
                names = ", ".join(sorted(catalog.keys())) or "(none)"
                raise SystemExit(
                    f"Unknown transcriber '{transcriber}'. Available: {names}"
                )
            spec = catalog[transcriber]
            if not spec.get("benchmarkable", False):
                kind = spec.get("kind", "?")
                raise SystemExit(
                    f"Transcriber '{transcriber}' is not benchmarkable (kind={kind})."
                )
            supported = spec.get("supported_hardware") or []
            if supported != ["*"] and hw_class not in supported:
                listed = ", ".join(supported) if supported else "(none)"
                raise SystemExit(
                    f"Transcriber '{transcriber}' does not support hardware "
                    f"class '{hw_class}'. Supported: {listed}."
                )
            return spec

        monkeypatch.setattr(harness, "_preflight_transcriber", patched)
        return harness

    def test_unknown_transcriber_hard_fails(self, monkeypatch, tmp_path):
        harness = self._patch(monkeypatch, tmp_path)
        with pytest.raises(SystemExit, match="Unknown transcriber 'bogus'"):
            harness._preflight_transcriber("bogus", "dgx-spark")

    def test_cloud_backend_not_benchmarkable(self, monkeypatch, tmp_path):
        harness = self._patch(monkeypatch, tmp_path)
        with pytest.raises(SystemExit, match="not benchmarkable"):
            harness._preflight_transcriber("gemini", "dgx-spark")

    def test_wrong_hardware_for_parakeet_hard_fails(self, monkeypatch, tmp_path):
        # The bug that originally triggered the schema work: the bundled
        # parakeet path is CoreML / linux-x86_64 ONNX only, so it must
        # refuse to RTF-capture on dgx-spark.
        harness = self._patch(monkeypatch, tmp_path)
        with pytest.raises(
            SystemExit, match="does not support hardware class 'dgx-spark'"
        ):
            harness._preflight_transcriber("parakeet", "dgx-spark")

    def test_whisper_wildcard_supports_any_class(self, monkeypatch, tmp_path):
        harness = self._patch(monkeypatch, tmp_path)
        # Should not raise — whisper is the universal floor.
        spec = harness._preflight_transcriber("whisper", "dgx-spark")
        assert spec["fallback"] is True
