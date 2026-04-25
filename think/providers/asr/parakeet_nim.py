# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Parakeet TDT via NVIDIA NIM container (HTTP).

Talks to a Parakeet ASR NIM (``nvcr.io/nim/nvidia/parakeet-0-6b-tdt``)
running locally or remotely. The NIM exposes an OpenAI-compatible
``POST /v1/audio/transcriptions`` endpoint on port 9000 by default and
ships with Sortformer speaker diarization and Silero VAD bundled.

This provider exists because the bundled in-process Parakeet path
(``observe/transcribe/parakeet.py``) does not run on ``linux/aarch64``
+ Blackwell hosts (DGX Spark). See the project's Transcription Backend
Architecture doc for the full rationale; the short version is that
NIM is the deployment vector NVIDIA paves for the Spark, and we want
the ASR speed there without dragging NeMo / aarch64 wheel chains into
Solstone's Python codebase.

Hard-fails on unreachable endpoint and on malformed responses rather
than silently falling back. Cross-backend benchmark integrity depends
on the failure being loud, not graceful.

**Field-mapping caveat — to be verified in Phase 3b.** The exact
JSON shape of NVIDIA's diarization extension to the OpenAI Whisper
verbose_json response is not fully documented at the time of writing.
This provider reads the response through the constants in
``_RESPONSE_FIELDS`` below; if the live NIM uses different field
names, fix the constants — do not restructure the provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import numpy as np

from think.providers.asr.shared import (
    Statement,
    Word,
    assign_sequential_ids,
    empty_transcript,
    encode_wav_pcm16,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:9000"
_TRANSCRIBE_PATH = "/v1/audio/transcriptions"
_DEFAULT_TIMEOUT_S = 600.0
_DEFAULT_LANGUAGE = "en-US"
_DEFAULT_RESPONSE_FORMAT = "verbose_json"


# Centralized response field paths. Phase 3b will verify these against
# a live NIM and patch as needed; isolating them here means that fix is
# a one-line edit, not a provider rewrite.
_RESPONSE_FIELDS = {
    # Top-level container of per-segment results.
    "segments": "segments",
    # Top-level full transcript (used as a fallback when the NIM doesn't
    # return segments, e.g. when verbose_json is unsupported).
    "text": "text",
    # Per-segment fields.
    "segment_start": "start",
    "segment_end": "end",
    "segment_text": "text",
    "segment_words": "words",
    "segment_speaker": "speaker",
    # Per-word fields (inside segment_words).
    "word_text": "word",
    "word_start": "start",
    "word_end": "end",
    "word_probability": "probability",
}


class ParakeetNimError(RuntimeError):
    """Raised when the NIM endpoint is unreachable or returns a bad response."""


def _resolve_endpoint(config: dict[str, Any]) -> str:
    """Resolve the NIM base URL from config, env, or the default.

    Precedence: config["endpoint"] > PARAKEET_NIM_URL env > default.
    The endpoint is normalized to drop a trailing slash.
    """
    raw = config.get("endpoint") or os.environ.get(
        "PARAKEET_NIM_URL", _DEFAULT_ENDPOINT
    )
    return str(raw).rstrip("/")


def _post_audio(
    client: httpx.Client,
    endpoint: str,
    wav_bytes: bytes,
    config: dict[str, Any],
) -> dict[str, Any]:
    """POST a WAV blob to ``/v1/audio/transcriptions`` and return parsed JSON.

    Hard-fails with ``ParakeetNimError`` on connection failure, non-2xx
    status, or non-JSON body. The error message includes the endpoint
    URL so a misconfigured Cumulus deploy is immediately diagnosable.
    """
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    form: dict[str, str] = {
        "language": config.get("language", _DEFAULT_LANGUAGE),
        "response_format": config.get(
            "response_format", _DEFAULT_RESPONSE_FORMAT
        ),
    }
    if "model" in config:
        form["model"] = str(config["model"])
    if config.get("timestamp_granularities"):
        form["timestamp_granularities[]"] = ",".join(
            config["timestamp_granularities"]
        )

    url = f"{endpoint}{_TRANSCRIBE_PATH}"
    try:
        response = client.post(url, files=files, data=form)
    except httpx.HTTPError as exc:
        raise ParakeetNimError(
            f"parakeet-nim unreachable at {url}: {exc.__class__.__name__}: "
            f"{exc}. Hard-fail: provider does not silently fall back to "
            f"another backend."
        ) from exc

    if response.status_code != 200:
        raise ParakeetNimError(
            f"parakeet-nim returned HTTP {response.status_code} from {url}: "
            f"{response.text[:500]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ParakeetNimError(
            f"parakeet-nim returned non-JSON from {url}: "
            f"{response.text[:200]}"
        ) from exc


def _normalize_word(raw: dict[str, Any]) -> Word | None:
    """Build a Word dict from the NIM's per-word entry; ``None`` if unparseable."""
    text = raw.get(_RESPONSE_FIELDS["word_text"])
    start = raw.get(_RESPONSE_FIELDS["word_start"])
    end = raw.get(_RESPONSE_FIELDS["word_end"])
    if text is None or start is None or end is None:
        return None
    word: Word = {
        "word": str(text),
        "start": float(start),
        "end": float(end),
    }
    prob = raw.get(_RESPONSE_FIELDS["word_probability"])
    if isinstance(prob, (int, float)):
        word["probability"] = float(prob)
    return word


def _normalize_segment(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Build a partial Statement (without ``id``) from one NIM segment.

    Returns ``None`` for segments missing the required text/start/end
    fields — those are dropped rather than passed through with garbage
    timestamps.
    """
    text = raw.get(_RESPONSE_FIELDS["segment_text"])
    start = raw.get(_RESPONSE_FIELDS["segment_start"])
    end = raw.get(_RESPONSE_FIELDS["segment_end"])
    if text is None or start is None or end is None:
        return None

    statement: dict[str, Any] = {
        "start": float(start),
        "end": float(end),
        "text": str(text).strip(),
    }

    raw_words = raw.get(_RESPONSE_FIELDS["segment_words"])
    if isinstance(raw_words, list) and raw_words:
        words: list[Word] = []
        for w in raw_words:
            if not isinstance(w, dict):
                continue
            normalized = _normalize_word(w)
            if normalized is not None:
                words.append(normalized)
        statement["words"] = words if words else None
    else:
        statement["words"] = None

    raw_speaker = raw.get(_RESPONSE_FIELDS["segment_speaker"])
    if isinstance(raw_speaker, int):
        statement["speaker"] = raw_speaker
    elif isinstance(raw_speaker, str) and raw_speaker.isdigit():
        statement["speaker"] = int(raw_speaker)
    elif isinstance(raw_speaker, str) and raw_speaker.lower().startswith(
        "speaker_"
    ):
        # Sortformer outputs may use "speaker_0", "speaker_1" labels;
        # store the numeric suffix to keep the schema int-typed.
        suffix = raw_speaker.split("_", 1)[1]
        statement["speaker"] = int(suffix) + 1 if suffix.isdigit() else None
    else:
        statement["speaker"] = None

    return statement


def _normalize_response(body: dict[str, Any]) -> list[Statement]:
    """Map the NIM's JSON response to ``observe.transcribe`` statements.

    Handles two shapes:

    1. ``verbose_json`` with ``segments``: one Statement per segment,
       words and speaker carried through when present.
    2. Plain ``{"text": ...}`` with no segments: synthesizes a single
       statement covering the full audio. Loses timing precision, but
       keeps the rest of the pipeline working when the NIM doesn't
       honour ``response_format=verbose_json``.
    """
    raw_segments = body.get(_RESPONSE_FIELDS["segments"])
    if isinstance(raw_segments, list) and raw_segments:
        partials: list[dict[str, Any]] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            normalized = _normalize_segment(seg)
            if normalized is not None:
                partials.append(normalized)
        if partials:
            return assign_sequential_ids(partials)

    fallback_text = body.get(_RESPONSE_FIELDS["text"])
    if isinstance(fallback_text, str) and fallback_text.strip():
        logger.warning(
            "parakeet-nim response had no usable segments; falling back to "
            "single-statement synthesis. Verify response_format support and "
            "_RESPONSE_FIELDS in this provider."
        )
        return assign_sequential_ids(
            [
                {
                    "start": 0.0,
                    "end": 0.0,
                    "text": fallback_text.strip(),
                    "words": None,
                    "speaker": None,
                }
            ]
        )

    return empty_transcript()


def transcribe(
    audio: np.ndarray, sample_rate: int, config: dict[str, Any]
) -> list[Statement]:
    """Transcribe ``audio`` via the Parakeet NIM container.

    Returns the same statement-list shape as the bundled
    ``observe.transcribe`` backends. Raises ``ParakeetNimError`` (a
    ``RuntimeError`` subclass) when the endpoint is unreachable or the
    response is malformed; callers that want to swallow these must do
    so explicitly. There is intentionally no silent fallback to
    another backend at this layer.
    """
    if audio.size == 0:
        return empty_transcript()

    endpoint = _resolve_endpoint(config)
    timeout = float(config.get("timeout_s", _DEFAULT_TIMEOUT_S))
    wav_bytes = encode_wav_pcm16(audio, sample_rate)

    with httpx.Client(timeout=timeout) as client:
        body = _post_audio(client, endpoint, wav_bytes, config)

    return _normalize_response(body)


__all__ = [
    "transcribe",
    "ParakeetNimError",
]
