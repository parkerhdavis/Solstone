# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the settings app's benchmark endpoints.

Covers the two new routes added in Phase 5 of the segment-time
benchmark project:

- ``GET /app/settings/api/benchmark/scenarios`` — exposes the
  ``segment.json`` catalog so the UI can populate the scenario picker.
- ``GET /app/settings/api/benchmark/segment`` — returns a
  ``SegmentEstimate`` for the chosen scenario and (optionally) tier
  models / transcriber, used by the providers tab's "Background
  processing" card.
"""

from __future__ import annotations

from pathlib import Path

from convey import create_app


def _settings_client(journal_dir: Path):
    app = create_app(str(journal_dir))
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# /api/benchmark/scenarios
# ---------------------------------------------------------------------------


class TestScenariosRoute:
    def test_returns_scenarios_catalog(self, journal_copy):
        client = _settings_client(journal_copy)
        resp = client.get("/app/settings/api/benchmark/scenarios")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "scenarios" in body
        # The repo ships solo_active / meeting_active / idle.
        assert "solo_active" in body["scenarios"]
        # Each scenario carries the fields the UI renders.
        spec = body["scenarios"]["solo_active"]
        assert "label" in spec
        assert "qualified_frames" in spec
        assert "talents" in spec


# ---------------------------------------------------------------------------
# /api/benchmark/segment
# ---------------------------------------------------------------------------


class TestSegmentRoute:
    def test_default_scenario_returns_full_breakdown(self, journal_copy):
        client = _settings_client(journal_copy)
        resp = client.get("/app/settings/api/benchmark/segment")
        assert resp.status_code == 200
        body = resp.get_json()
        # Lane keys are always present, even when individual values are None.
        assert "audio_seconds" in body
        assert "video_seconds" in body
        assert "talent_seconds" in body
        assert "overhead_seconds" in body
        assert "per_talent" in body
        assert "confidence" in body
        assert "tier_models" in body
        # Default tier-model fill should populate all three tier roles
        # from the smallest registry models (vision/generate/cogitate
        # all have at least one capable model in the shipped registry).
        for tier in ("vision", "generate", "cogitate"):
            assert tier in body["tier_models"]
        assert body["scenario"] == "solo_active"

    def test_specified_scenario_passes_through(self, journal_copy):
        client = _settings_client(journal_copy)
        resp = client.get("/app/settings/api/benchmark/segment?scenario=meeting_active")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["scenario"] == "meeting_active"

    def test_unknown_scenario_returns_400(self, journal_copy):
        client = _settings_client(journal_copy)
        resp = client.get("/app/settings/api/benchmark/segment?scenario=does_not_exist")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body

    def test_explicit_tier_model_overrides_default(self, journal_copy):
        # When the caller supplies an explicit per-tier model, the
        # endpoint must use it verbatim (this is how the UI will pass
        # the user's selected models from the providers tab).
        client = _settings_client(journal_copy)
        resp = client.get(
            "/app/settings/api/benchmark/segment"
            "?scenario=solo_active"
            "&generate=ollama-local/qwen3.5:35b-a3b-bf16"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["tier_models"]["generate"] == "ollama-local/qwen3.5:35b-a3b-bf16"

    def test_transcriber_query_param_carries_through(self, journal_copy):
        client = _settings_client(journal_copy)
        resp = client.get(
            "/app/settings/api/benchmark/segment"
            "?scenario=solo_active&transcriber=whisper"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["transcriber"] == "whisper"
