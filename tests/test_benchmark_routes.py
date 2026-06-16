# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the benchmark app's HTTP endpoints.

Fork-only benchmark routes. These live in the standalone ``benchmark`` app
(an API-only app over ``solstone.think.benchmark``); a benchmark workspace UI
can be layered on later.

- ``GET /app/benchmark/api/scenarios`` — exposes the
  ``segment.json`` catalog so the UI can populate the scenario picker.
- ``GET /app/benchmark/api/segment`` — returns a
  ``SegmentEstimate`` for the chosen scenario and (optionally) tier
  models / transcriber.
- ``GET /app/benchmark/api/models`` — pre-vetted local models with
  task-time estimates and the served-model flag.
"""

from __future__ import annotations

from pathlib import Path

from solstone.convey import create_app


def _client(journal_dir: Path):
    app = create_app(str(journal_dir))
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# /api/benchmark/scenarios
# ---------------------------------------------------------------------------


class TestScenariosRoute:
    def test_returns_scenarios_catalog(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/scenarios")
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
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/segment")
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
        # The registry now carries vision-capable local models (the served
        # Nemotron Omni plus the qwen candidates), so the default tier-model
        # fill populates all three roles from local models.
        for tier in ("generate", "cogitate", "vision"):
            assert tier in body["tier_models"]
        assert body["scenario"] == "solo_active"

    def test_specified_scenario_passes_through(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/segment?scenario=meeting_active")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["scenario"] == "meeting_active"

    def test_unknown_scenario_returns_400(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/segment?scenario=does_not_exist")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body

    def test_explicit_tier_model_overrides_default(self, journal_copy):
        # When the caller supplies an explicit per-tier model, the
        # endpoint must use it verbatim (this is how the UI will pass
        # the user's selected models from the providers tab).
        client = _client(journal_copy)
        resp = client.get(
            "/app/benchmark/api/segment"
            "?scenario=solo_active"
            "&generate=local/qwen3.6-35b-a3b"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["tier_models"]["generate"] == "local/qwen3.6-35b-a3b"

    def test_transcriber_query_param_carries_through(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get(
            "/app/benchmark/api/segment?scenario=solo_active&transcriber=whisper"
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["transcriber"] == "whisper"

    def test_carries_budget_and_group_fit(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/segment")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "budget_gb" in body
        assert "group_fit" in body
        for key in ("budget_gb", "footprint_gb", "fits", "per_model_gb", "notes"):
            assert key in body["group_fit"]

    def test_rejects_invalid_budget(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/segment?budget=not-a-number")
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# /api/benchmark/models
# ---------------------------------------------------------------------------


class TestModelsRoute:
    def test_rows_carry_served_flag(self, journal_copy):
        # The providers-tab benchmark card uses `served` to show the single
        # served model and keep candidate rows non-switchable, so the row must
        # carry it and exactly the pinned LOCAL_MODEL must be flagged.
        from solstone.think.models import LOCAL_MODEL

        client = _client(journal_copy)
        resp = client.get("/app/benchmark/api/models")
        assert resp.status_code == 200
        body = resp.get_json()
        by_id = {m["model_id"]: m for m in body["models"]}
        assert by_id[LOCAL_MODEL]["served"] is True
        served = [m["model_id"] for m in body["models"] if m.get("served")]
        assert served == [LOCAL_MODEL]
