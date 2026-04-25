# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared helpers for ASR providers.

Encodes the statement schema that ``observe.transcribe`` already
consumes (see ``observe/transcribe/__init__.py`` docstring for the
authoritative description). ASR providers in this package return
``list[Statement]`` matching that schema so downstream entity
extraction, meeting detection, and journal-write paths don't care
which backend produced the data.
"""

from __future__ import annotations

import io
import wave
from typing import Any, TypedDict

import numpy as np


class Word(TypedDict, total=False):
    """One word with timestamps. Mirrors ``observe.transcribe`` words."""

    word: str
    start: float
    end: float
    probability: float


class Statement(TypedDict, total=False):
    """One transcript statement. Mirrors ``observe.transcribe`` statements.

    ``id`` is sequential 1-indexed across the audio. ``speaker`` is the
    1-indexed Sortformer speaker id when diarization is available, or
    ``None`` for backends that don't diarize. ``words`` carries
    word-level timestamps when the backend exposes them.
    """

    id: int
    start: float
    end: float
    text: str
    words: list[Word] | None
    speaker: int | None


def encode_wav_pcm16(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono buffer as a 16-bit PCM WAV byte blob.

    Used to form the multipart upload body for NIMs that accept WAV.
    Audio is clipped to [-1, 1] before quantization so out-of-range
    values from upstream gain stages don't wrap around as noise.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"encode_wav_pcm16 expects mono audio (1-D); got shape {audio.shape}"
        )
    clipped = np.clip(audio, -1.0, 1.0)
    int_audio = (clipped * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int_audio.tobytes())
    return buf.getvalue()


def empty_transcript() -> list[Statement]:
    """Canonical empty result. Backends use this when audio yields no speech."""
    return []


def assign_sequential_ids(statements: list[dict[str, Any]]) -> list[Statement]:
    """Renumber ``id`` starting at 1 in input order.

    Providers often build statements without worrying about IDs; this
    helper applies the contract the rest of the pipeline expects.
    """
    out: list[Statement] = []
    for idx, st in enumerate(statements, start=1):
        out.append({**st, "id": idx})  # type: ignore[typeddict-item]
    return out


__all__ = [
    "Word",
    "Statement",
    "encode_wav_pcm16",
    "empty_transcript",
    "assign_sequential_ids",
]
