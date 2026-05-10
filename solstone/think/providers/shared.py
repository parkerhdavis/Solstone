# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared utilities and types for AI providers.

This module contains:
- Event TypedDicts emitted by providers during talent execution
- GenerateResult TypedDict returned by run_generate/run_agenerate
- ContentBlock types for multimodal messages (text, image, audio)
- JSONEventCallback for event emission
- Utility functions for common provider operations

Multimodal message shape
------------------------
A user message's ``content`` field can be either:

1. A plain string (legacy, text-only).
2. A list of ContentBlock dicts: ``TextBlock``, ``ImageBlock``, or
   ``AudioBlock``.

Provider modules translate the second form into their wire format. Ollama
extracts ``ImageBlock`` data into the per-message ``images`` array and
concatenates ``TextBlock`` text; ``AudioBlock`` is unsupported. vLLM (and
other OpenAI-compatible endpoints) pass the blocks through as
``image_url`` / ``input_audio`` content blocks. Providers that cannot
serve a given modality must raise ``NotImplementedError`` with a clear
message.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Literal, Optional, Union

from typing_extensions import Required, TypedDict

from solstone.think.utils import now_ms

# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------


class ToolStartEvent(TypedDict, total=False):
    """Event emitted when a tool starts."""

    event: Literal["tool_start"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    call_id: Optional[str]  # Unique ID to pair with tool_end event
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ToolEndEvent(TypedDict, total=False):
    """Event emitted when a tool finishes."""

    event: Literal["tool_end"]
    ts: int
    tool: str
    args: Optional[dict[str, Any]]
    result: Any
    call_id: Optional[str]  # Matches the call_id from tool_start
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class StartEvent(TypedDict, total=False):
    """Event emitted when a talent run begins."""

    event: Required[Literal["start"]]
    ts: Required[int]
    prompt: Required[str]
    name: Required[str]
    model: Required[str]
    provider: Required[str]
    session_id: Optional[str]  # CLI session ID for continuation
    chat_id: Optional[str]  # Chat ID for reverse lookup
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class FinishEvent(TypedDict, total=False):
    """Event emitted when a talent run finishes successfully."""

    event: Required[Literal["finish"]]
    ts: Required[int]
    result: Required[str]
    usage: Optional[dict[str, Any]]
    cli_session_id: Optional[str]  # Provider CLI session/thread ID for resume
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ErrorEvent(TypedDict, total=False):
    """Event emitted when an error occurs."""

    event: Literal["error"]
    ts: int
    error: str
    trace: Optional[str]
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class TalentUpdatedEvent(TypedDict, total=False):
    """Event emitted when the talent context changes."""

    event: Required[Literal["talent_updated"]]
    ts: Required[int]
    talent: Required[str]
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class ThinkingEvent(TypedDict, total=False):
    """Event emitted when thinking/reasoning summaries are available.

    For Anthropic models, may include a signature for verification when
    passing thinking blocks back during tool use continuations.
    For redacted thinking, summary will contain "[redacted]" and
    redacted_data will contain the encrypted content.
    """

    event: Required[Literal["thinking"]]
    ts: Required[int]
    summary: Required[str]
    model: Optional[str]
    signature: Optional[str]  # Anthropic thinking block signature
    redacted_data: Optional[str]  # Encrypted data for redacted thinking
    raw: Optional[list[dict[str, Any]]]  # Original provider JSON event(s)


class FallbackEvent(TypedDict, total=False):
    """Event emitted when provider fallback occurs."""

    event: Required[Literal["fallback"]]
    ts: Required[int]
    original_provider: Required[str]
    backup_provider: Required[str]
    reason: Required[str]  # "preflight" or "on_failure"
    error: Optional[str]  # Error message for on_failure case


Event = Union[
    ToolStartEvent,
    ToolEndEvent,
    StartEvent,
    FinishEvent,
    ErrorEvent,
    ThinkingEvent,
    TalentUpdatedEvent,
    FallbackEvent,
]


# ---------------------------------------------------------------------------
# Multimodal Content Blocks
# ---------------------------------------------------------------------------
#
# A message's content is either a plain string (legacy text-only) or a list
# of ContentBlock dicts. Providers translate to their wire format.


class TextBlock(TypedDict):
    """Text content block."""

    type: Literal["text"]
    text: str


class ImageBlock(TypedDict, total=False):
    """Image content block.

    ``data`` is base64-encoded image bytes. ``mime`` is the IANA media type
    (e.g. ``image/jpeg``, ``image/png``); providers that don't carry mime
    in their wire format may default to JPEG.
    """

    type: Required[Literal["image"]]
    data: Required[str]
    mime: str


class AudioBlock(TypedDict, total=False):
    """Audio content block.

    ``data`` is base64-encoded audio bytes. ``format`` is the container
    (``wav``, ``flac``, ``mp3``, ``ogg``). ``sample_rate`` is optional and
    only meaningful for raw-PCM-in-WAV; most providers infer it from the
    container.
    """

    type: Required[Literal["audio"]]
    data: Required[str]
    format: Required[str]
    sample_rate: int


ContentBlock = Union[TextBlock, ImageBlock, AudioBlock]


def _is_content_block_list(content: Any) -> bool:
    """Return True if *content* is a list of ContentBlock dicts.

    Used by providers to detect the multimodal form vs. the legacy string
    form. A list of strings (the older ``[str, str, ...]`` shorthand) is
    *not* a content-block list and should be joined to a string by the
    caller before this returns True.
    """
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return isinstance(first, dict) and first.get("type") in {"text", "image", "audio"}


# ---------------------------------------------------------------------------
# Usage Schema
# ---------------------------------------------------------------------------

# Canonical keys for the normalized usage dict returned by all providers.
# log_token_usage() passes through exactly these keys (when present and non-zero).
USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cache_creation_tokens",
        "requests",
    }
)

# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class GenerateResult(TypedDict, total=False):
    """Result from provider run_generate/run_agenerate functions.

    Structured result that allows the wrapper to handle cross-cutting concerns
    like token logging and JSON validation centrally.

    The thinking field contains dicts with: summary (str), signature (optional str),
    redacted_data (optional str for Anthropic redacted thinking).
    """

    text: Required[str]  # Response text
    usage: Optional[dict]  # Normalized usage dict (input_tokens, output_tokens, etc.)
    finish_reason: Optional[str]  # Normalized: "stop", "max_tokens", "safety", etc.
    thinking: Optional[list]  # List of thinking block dicts
    schema_validation: Optional[dict]  # Validation result when json_schema is supplied


# ---------------------------------------------------------------------------
# JSONEventCallback
# ---------------------------------------------------------------------------


class JSONEventCallback:
    """Emit JSON events via a callback."""

    def __init__(self, callback: Optional[Callable[[Event], None]] = None) -> None:
        self.callback = callback

    def emit(self, data: Event) -> None:
        if "ts" not in data:
            data = {**data, "ts": now_ms()}
        if self.callback:
            self.callback(data)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Raw Event Trimming
# ---------------------------------------------------------------------------

# Structural keys preserved when trimming oversized raw events.
_RAW_STRUCTURAL_KEYS = frozenset(
    {
        "type",
        "id",
        "tool_id",
        "tool_name",
        "role",
        "event_type",
        "timestamp",
    }
)

_RAW_BYTE_LIMIT = 16_384  # 16 KB


def safe_raw(
    events: list[dict[str, Any]],
    limit: int = _RAW_BYTE_LIMIT,
) -> list[dict[str, Any]]:
    """Return *events* as-is if small enough, otherwise a trimmed version.

    When the JSON-serialized size exceeds *limit* bytes, each event is reduced
    to its structural keys and a ``_raw_trimmed`` dict is appended with the
    original byte count and the limit that was applied.
    """
    serialized = json.dumps(events, ensure_ascii=False)
    if len(serialized.encode("utf-8")) <= limit:
        return events

    trimmed = [
        {k: v for k, v in e.items() if k in _RAW_STRUCTURAL_KEYS} for e in events
    ]
    trimmed.append(
        {"_raw_trimmed": {"original_bytes": len(serialized), "limit": limit}}
    )
    return trimmed


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
#
# Providers that participate in `think.benchmark.harness` expose two
# functions:
#
#   bench_ensure_installed(model, *, allow_pull) -> None
#       Verify the model is locally available; optionally trigger a pull.
#       Raise SystemExit on failure with a clear remediation message.
#
#   bench_run_once(model, *, prompt, image_b64=None, audio_b64=None,
#                  audio_format="wav", max_output_tokens) -> BenchmarkResult
#       Send one synchronous benchmark request and return a normalized
#       BenchmarkResult. The provider is responsible for any provider-
#       specific request shaping (Ollama caps num_ctx; vLLM relies on
#       max_model_len set at serve time; etc.).
#
# The harness owns the *policy* (which prompts, which media, when to run)
# and the providers own the *transport* (how to reach the model and
# extract timing). This keeps harness code provider-agnostic and lets new
# providers slot in without harness changes.


class BenchmarkResult(TypedDict, total=False):
    """One benchmark request's outcome, normalized across providers.

    ``elapsed_s`` is wall-clock around the provider's HTTP round-trip.
    ``native_output_tok_s`` / ``native_prompt_tok_s`` are the provider's
    own server-side counters when available (Ollama reports nanosecond
    eval durations; vLLM does not). Harness reporting prefers native when
    present and falls back to ``output_tokens / elapsed_s`` otherwise.
    """

    elapsed_s: Required[float]
    prompt_tokens: Required[int]
    output_tokens: Required[int]
    native_output_tok_s: Optional[float]
    native_prompt_tok_s: Optional[float]
    finish_reason: Optional[str]
    text: Required[str]
    raw: Required[dict[str, Any]]


__all__ = [
    "AudioBlock",
    "BenchmarkResult",
    "ContentBlock",
    "Event",
    "GenerateResult",
    "ImageBlock",
    "JSONEventCallback",
    "TextBlock",
    "ThinkingEvent",
    "USAGE_KEYS",
    "_is_content_block_list",
    "safe_raw",
]
