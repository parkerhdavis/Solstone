# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.providers.asr.parakeet_nim and the ASR provider package.

Covers (1) the package-level registry, (2) the parakeet-nim provider's
HTTP request shaping + response normalization, and (3) the failure
contracts that exist to keep the cross-backend benchmark signal honest
(no silent fallbacks, hard-fail on unreachable endpoint or bad payload).
"""

from __future__ import annotations

import io
import json
import wave

import httpx
import numpy as np
import pytest

from think.providers.asr import (
    ASR_PROVIDER_METADATA,
    ASR_PROVIDER_REGISTRY,
    get_asr_provider_module,
)
from think.providers.asr import parakeet_nim
from think.providers.asr.parakeet_nim import ParakeetNimError, transcribe
from think.providers.asr.shared import (
    assign_sequential_ids,
    encode_wav_pcm16,
    empty_transcript,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_parakeet_nim_is_registered(self):
        assert "parakeet-nim" in ASR_PROVIDER_REGISTRY
        assert (
            ASR_PROVIDER_REGISTRY["parakeet-nim"]
            == "think.providers.asr.parakeet_nim"
        )

    def test_metadata_has_required_keys(self):
        meta = ASR_PROVIDER_METADATA["parakeet-nim"]
        assert meta["kind"] == "local-http"
        assert meta["endpoint_env"] == "PARAKEET_NIM_URL"
        assert meta["supported_hardware"] == ["dgx-spark"]

    def test_get_module_returns_provider(self):
        mod = get_asr_provider_module("parakeet-nim")
        assert mod is parakeet_nim
        assert callable(mod.transcribe)

    def test_get_module_unknown_provider_raises_valueerror(self):
        # ValueError, not SystemExit — the provider layer doesn't drive
        # user-facing CLI exits; that's the harness preflight's job.
        with pytest.raises(ValueError, match="Unknown ASR provider"):
            get_asr_provider_module("bogus")


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


class TestEncodeWavPcm16:
    def test_roundtrip(self):
        # Generate a 0.2s sine wave at 16kHz; encode; reload via wave; verify.
        sample_rate = 16000
        n = 3200
        t = np.arange(n) / sample_rate
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        blob = encode_wav_pcm16(audio, sample_rate)
        with wave.open(io.BytesIO(blob), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == sample_rate
            assert wf.getnframes() == n

    def test_clips_out_of_range(self):
        # A naive int16 cast of values > 1.0 wraps; encode_wav_pcm16
        # clips first so downstream gain stages can't smuggle in noise.
        sample_rate = 16000
        audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        blob = encode_wav_pcm16(audio, sample_rate)
        with wave.open(io.BytesIO(blob), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        assert samples[0] == 32767  # clipped from 2.0
        assert samples[1] == -32767  # clipped from -2.0

    def test_rejects_non_mono(self):
        with pytest.raises(ValueError, match="mono"):
            encode_wav_pcm16(np.zeros((2, 16000), dtype=np.float32), 16000)


# ---------------------------------------------------------------------------
# Helpers — fake httpx transport
# ---------------------------------------------------------------------------


def _make_client_with_response(
    json_body: dict | None = None,
    *,
    status_code: int = 200,
    text_body: str | None = None,
    raise_on_send: Exception | None = None,
    capture: dict | None = None,
):
    """Build an httpx.MockTransport that returns a fixed response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if raise_on_send is not None:
            raise raise_on_send
        if capture is not None:
            capture["url"] = str(request.url)
            capture["content_type"] = request.headers.get("content-type", "")
            capture["body_len"] = len(request.content or b"")
        if json_body is not None:
            return httpx.Response(status_code, json=json_body)
        return httpx.Response(status_code, text=text_body or "")

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def _patch_client_factory(monkeypatch, client: httpx.Client) -> None:
    """Make ``httpx.Client(...)`` return our pre-built mock client.

    The provider does ``with httpx.Client(timeout=...) as client:`` —
    we replace the constructor to return the already-set-up mock.
    """
    monkeypatch.setattr(parakeet_nim.httpx, "Client", lambda **kwargs: client)


# ---------------------------------------------------------------------------
# Normalization — verbose_json + diarization
# ---------------------------------------------------------------------------


SAMPLE_RATE = 16000
SHORT_AUDIO = np.zeros(SAMPLE_RATE // 4, dtype=np.float32)  # 0.25s of silence


class TestNormalization:
    def test_verbose_json_with_segments_and_words(self, monkeypatch):
        body = {
            "text": "Hello world. How are you?",
            "language": "en-US",
            "duration": 3.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "Hello world.",
                    "speaker": 0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5,
                         "probability": 0.99},
                        {"word": "world", "start": 0.6, "end": 1.2,
                         "probability": 0.97},
                    ],
                },
                {
                    "start": 1.5,
                    "end": 3.0,
                    "text": "How are you?",
                    "speaker": 1,
                    "words": [
                        {"word": "How", "start": 1.5, "end": 1.8},
                        {"word": "are", "start": 1.85, "end": 2.0},
                        {"word": "you", "start": 2.05, "end": 3.0},
                    ],
                },
            ],
        }
        client = _make_client_with_response(body)
        _patch_client_factory(monkeypatch, client)

        statements = transcribe(SHORT_AUDIO, SAMPLE_RATE, {})

        assert len(statements) == 2
        assert statements[0]["id"] == 1
        assert statements[1]["id"] == 2
        assert statements[0]["text"] == "Hello world."
        assert statements[0]["start"] == 0.0
        assert statements[0]["end"] == 1.2
        # Speaker mapped through int → int.
        assert statements[0]["speaker"] == 0
        assert statements[1]["speaker"] == 1
        # Word-level data preserved with probabilities when present.
        assert statements[0]["words"][0]["word"] == "Hello"
        assert statements[0]["words"][0]["probability"] == 0.99
        # Words without probability still parse cleanly.
        assert "probability" not in statements[1]["words"][0]

    def test_speaker_string_label_normalized_to_int(self, monkeypatch):
        # Sortformer outputs may use "speaker_0" / "speaker_1" labels; the
        # provider converts those to 1-indexed ints to match the existing
        # observe.transcribe schema.
        body = {
            "text": "ok",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "ok",
                    "speaker": "speaker_0",
                    "words": [],
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "yep",
                    "speaker": "speaker_2",
                    "words": [],
                },
            ],
        }
        client = _make_client_with_response(body)
        _patch_client_factory(monkeypatch, client)

        statements = transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert statements[0]["speaker"] == 1  # speaker_0 -> 1
        assert statements[1]["speaker"] == 3  # speaker_2 -> 3

    def test_segments_missing_required_fields_are_dropped(self, monkeypatch):
        # A segment without text/start/end is dropped entirely rather than
        # being passed through with garbage values.
        body = {
            "text": "hello",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 2.0},  # no text, no end — drop
                {"start": 3.0, "end": 4.0, "text": "world"},
            ],
        }
        client = _make_client_with_response(body)
        _patch_client_factory(monkeypatch, client)

        statements = transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert [s["text"] for s in statements] == ["hello", "world"]
        # IDs renumbered after the drop.
        assert [s["id"] for s in statements] == [1, 2]

    def test_plain_text_response_synthesizes_single_statement(
        self, monkeypatch, caplog
    ):
        # When the NIM doesn't honor verbose_json (no segments key), fall
        # back to a single statement covering the full audio. Loses
        # timing precision but keeps downstream agents working.
        body = {"text": "the whole transcript as one string"}
        client = _make_client_with_response(body)
        _patch_client_factory(monkeypatch, client)

        with caplog.at_level("WARNING"):
            statements = transcribe(SHORT_AUDIO, SAMPLE_RATE, {})

        assert len(statements) == 1
        assert statements[0]["text"] == "the whole transcript as one string"
        assert statements[0]["speaker"] is None
        # Warning logged so the maintainer knows the fast path failed.
        assert any("no usable segments" in r.message for r in caplog.records)

    def test_empty_response_yields_empty_transcript(self, monkeypatch):
        body = {"text": "", "segments": []}
        client = _make_client_with_response(body)
        _patch_client_factory(monkeypatch, client)
        statements = transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert statements == empty_transcript()


# ---------------------------------------------------------------------------
# Failure contracts
# ---------------------------------------------------------------------------


class TestFailureContracts:
    def test_unreachable_endpoint_hard_fails(self, monkeypatch):
        # No silent fallback — the provider raises ParakeetNimError so the
        # caller sees the exact endpoint URL and can fix the deploy.
        client = _make_client_with_response(
            raise_on_send=httpx.ConnectError("Connection refused")
        )
        _patch_client_factory(monkeypatch, client)
        with pytest.raises(ParakeetNimError, match="unreachable"):
            transcribe(SHORT_AUDIO, SAMPLE_RATE, {})

    def test_http_error_includes_status_and_body(self, monkeypatch):
        client = _make_client_with_response(
            text_body="model not found", status_code=503
        )
        _patch_client_factory(monkeypatch, client)
        with pytest.raises(ParakeetNimError, match="HTTP 503"):
            transcribe(SHORT_AUDIO, SAMPLE_RATE, {})

    def test_non_json_response_hard_fails(self, monkeypatch):
        # Some misconfigured proxies return HTML 200; the provider must
        # not pretend that's a transcript.
        client = _make_client_with_response(text_body="<html>oops</html>")
        _patch_client_factory(monkeypatch, client)
        with pytest.raises(ParakeetNimError, match="non-JSON"):
            transcribe(SHORT_AUDIO, SAMPLE_RATE, {})

    def test_empty_audio_skips_http_call(self, monkeypatch):
        # Silent files shouldn't burn an HTTP round-trip on the NIM —
        # they're just empty transcripts.
        called = {"count": 0}

        def handler(request):
            called["count"] += 1
            return httpx.Response(200, json={"text": "should not be reached"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        _patch_client_factory(monkeypatch, client)

        statements = transcribe(np.array([], dtype=np.float32), SAMPLE_RATE, {})
        assert statements == empty_transcript()
        assert called["count"] == 0


# ---------------------------------------------------------------------------
# Request wiring
# ---------------------------------------------------------------------------


class TestRequestWiring:
    def test_endpoint_resolves_from_config_first(self, monkeypatch):
        capture: dict = {}
        client = _make_client_with_response({"text": ""}, capture=capture)
        _patch_client_factory(monkeypatch, client)
        # Even with PARAKEET_NIM_URL set, config should win.
        monkeypatch.setenv("PARAKEET_NIM_URL", "http://from-env:9000")
        transcribe(
            SHORT_AUDIO, SAMPLE_RATE, {"endpoint": "http://from-config:9000"}
        )
        assert "from-config" in capture["url"]
        assert "/v1/audio/transcriptions" in capture["url"]

    def test_endpoint_resolves_from_env_when_no_config(self, monkeypatch):
        capture: dict = {}
        client = _make_client_with_response({"text": ""}, capture=capture)
        _patch_client_factory(monkeypatch, client)
        monkeypatch.setenv("PARAKEET_NIM_URL", "http://from-env:9000")
        transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert "from-env" in capture["url"]

    def test_endpoint_default_when_neither(self, monkeypatch):
        capture: dict = {}
        client = _make_client_with_response({"text": ""}, capture=capture)
        _patch_client_factory(monkeypatch, client)
        monkeypatch.delenv("PARAKEET_NIM_URL", raising=False)
        transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert "localhost:9000" in capture["url"]

    def test_request_carries_multipart_audio_body(self, monkeypatch):
        capture: dict = {}
        client = _make_client_with_response({"text": ""}, capture=capture)
        _patch_client_factory(monkeypatch, client)
        transcribe(SHORT_AUDIO, SAMPLE_RATE, {})
        assert capture["content_type"].startswith("multipart/form-data")
        # Body length grows with the audio fixture (sanity check that the
        # WAV blob made it into the request).
        assert capture["body_len"] > 1000


# ---------------------------------------------------------------------------
# assign_sequential_ids helper (called from the provider; small smoke test)
# ---------------------------------------------------------------------------


def test_assign_sequential_ids_renumbers_in_order():
    out = assign_sequential_ids(
        [
            {"text": "a", "start": 0.0, "end": 1.0},
            {"text": "b", "start": 1.0, "end": 2.0},
            {"text": "c", "start": 2.0, "end": 3.0},
        ]
    )
    assert [s["id"] for s in out] == [1, 2, 3]
    # Original fields preserved.
    assert [s["text"] for s in out] == ["a", "b", "c"]
