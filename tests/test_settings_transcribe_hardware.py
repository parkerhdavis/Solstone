# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the Settings app's /api/transcribe hardware-compat enrichment.

Fork-only enrichment: ``/app/settings/api/transcribe`` decorates each backend
with ``supported_hardware`` (from ``transcribers.json``) and carries a probed
hardware class, so the transcription tab can flag mismatches like
parakeet-on-aarch64. Upstream kept the transcribe surface in Settings; this
enrichment is re-applied on top of it after the 0.6.4 merge.
"""

from __future__ import annotations

from pathlib import Path

from solstone.convey import create_app


def _client(journal_dir: Path):
    app = create_app(str(journal_dir))
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# /api/transcribe — hardware-compat enrichment for the settings UI
# ---------------------------------------------------------------------------


class TestTranscribeRoute:
    """The transcription tab's compat warning depends on
    /api/transcribe carrying (a) per-backend supported_hardware from
    transcribers.json and (b) a probed hardware class. Both are
    fork-only enrichments added so the UI can flag mismatches like
    parakeet-on-aarch64."""

    def test_response_carries_hardware_payload(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/settings/api/transcribe")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "hardware" in body
        # 'class' is always present (defaults to "cpu-only" when probe fails),
        # 'probed' tells the UI whether to trust it.
        assert "class" in body["hardware"]
        assert "probed" in body["hardware"]

    def test_each_backend_has_supported_hardware_key(self, journal_copy):
        client = _client(journal_copy)
        resp = client.get("/app/settings/api/transcribe")
        body = resp.get_json()
        names = {b["name"] for b in body.get("backends", [])}
        # Backends declared in transcribers.json carry a list; backends
        # not declared there carry None. Either way the key must exist
        # so the UI's _isBackendCompatible never sees `undefined`.
        for backend in body.get("backends", []):
            assert "supported_hardware" in backend, (
                f"backend {backend['name']!r} missing supported_hardware"
            )
        # parakeet's transcribers.json entry explicitly lists supported
        # hardware classes (it's the case the UI warning fires on).
        if "parakeet" in names:
            parakeet = next(b for b in body["backends"] if b["name"] == "parakeet")
            assert isinstance(parakeet["supported_hardware"], list)
            # dgx-spark must NOT be in the list — that's the bug the
            # warning surfaces.
            assert "dgx-spark" not in parakeet["supported_hardware"]
        # whisper is the universal floor → ['*']
        if "whisper" in names:
            whisper = next(b for b in body["backends"] if b["name"] == "whisper")
            assert whisper["supported_hardware"] == ["*"]
