#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Ollama (Local) provider for LLM generation and tool-calling agents.

This module provides the Ollama provider for run_generate/run_agenerate
(text generation) and run_cogitate (tool-calling agents).

**Generation** uses Ollama's native ``/api/chat`` endpoint via ``httpx``
for reliable control over the ``think`` parameter, which the OpenAI-compatible
endpoint silently ignores on models like Qwen3.5.

**Cogitate** uses the OpenCode CLI (``opencode run --format json``) as a
subprocess, following the same CLIRunner + translate pattern as the Google,
OpenAI, and Anthropic providers. OpenCode connects to local Ollama via its
OpenAI-compatible endpoint and handles tool execution internally.

Common Parameters
-----------------
contents : str or list
    The content to send to the model.
model : str
    Model name with ``ollama-local/`` prefix (e.g., ``ollama-local/qwen3.5:9b``).
    The prefix is stripped before sending to the Ollama API.
temperature : float
    Temperature for generation (default: 0.3).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction for the model.
json_output : bool
    Whether to request JSON response format.
thinking_budget : int, optional
    Token budget for model thinking. When > 0, enables Ollama's ``think``
    parameter. When None or 0, thinking is explicitly disabled.
timeout_s : float, optional
    Request timeout in seconds.
**kwargs
    Additional provider-specific options (absorbed for forward compatibility).

Environment Variables
---------------------
OLLAMA_BASE_URL : str
    Base URL for the Ollama server (default: ``http://localhost:11434``).
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import httpx

from solstone.think.models import OLLAMA_FLASH
from solstone.think.utils import now_ms

from .cli import CLIRunner, QuotaExhaustedError, ThinkingAggregator, assemble_prompt
from .shared import (
    BenchmarkResult,
    GenerateResult,
    JSONEventCallback,
    _is_content_block_list,
    classify_provider_error,
    safe_raw,
)

LOG = logging.getLogger("solstone.think.providers.ollama")

_OLLAMA_LOCAL_PREFIX = "ollama-local/"
_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def _get_base_url() -> str:
    """Get Ollama base URL from environment or default."""
    return os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _get_client() -> httpx.Client:
    """Get or create cached sync httpx client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(
            base_url=_get_base_url(),
            timeout=_DEFAULT_TIMEOUT,
        )
    return _sync_client


def _get_async_client() -> httpx.AsyncClient:
    """Get or create cached async httpx client."""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            base_url=_get_base_url(),
            timeout=_DEFAULT_TIMEOUT,
        )
    return _async_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_model_prefix(model: str) -> str:
    """Strip the ``ollama-local/`` prefix for the Ollama API.

    The Ollama API expects bare model names like ``qwen3.5:9b``, but
    Solstone uses the ``ollama-local/`` prefix for provider routing.
    """
    if model.startswith(_OLLAMA_LOCAL_PREFIX):
        return model[len(_OLLAMA_LOCAL_PREFIX) :]
    return model


def _translate_content_blocks(content: list[dict[str, Any]]) -> dict[str, Any]:
    """Translate a list of ContentBlock dicts to Ollama's per-message format.

    Ollama's /api/chat expects each message to have a string ``content``
    field plus an optional ``images: [<base64>, ...]`` array. Audio is not
    supported.

    Parameters
    ----------
    content
        A list of ContentBlock dicts (TextBlock / ImageBlock / AudioBlock).

    Returns
    -------
    dict
        ``{"content": "<concatenated text>", "images": [...]}`` (images key
        only present when at least one ImageBlock was supplied).

    Raises
    ------
    NotImplementedError
        If any AudioBlock is present — Ollama does not accept audio input.
    """
    text_parts: list[str] = []
    images: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image":
            data = block.get("data")
            if not data:
                raise ValueError("ImageBlock missing 'data' (base64-encoded bytes)")
            images.append(data)
        elif btype == "audio":
            raise NotImplementedError(
                "Ollama does not support audio input. Use a vLLM-served "
                "multimodal model (e.g. nemotron3:33b via vllm-local/) for "
                "audio benchmarking."
            )
        else:
            raise ValueError(f"Unknown content-block type: {btype!r}")

    out: dict[str, Any] = {"content": "\n".join(text_parts)}
    if images:
        out["images"] = images
    return out


def _build_messages(
    contents: Any,
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Convert contents and system instruction to chat messages.

    Parameters
    ----------
    contents
        One of:

        - A plain string (legacy, text-only user message).
        - A list of strings (joined with newlines into a single user message).
        - A list of message dicts with a ``role`` key. Each message's
          ``content`` may itself be a string OR a list of ContentBlock
          dicts (text/image/audio).
    system_instruction
        Optional system prompt, prepended as a system message.

    Returns
    -------
    list[dict[str, Any]]
        Messages in ``[{role, content, [images]}, ...]`` format ready for
        Ollama's ``/api/chat``. The ``images`` field is added per-message
        when ImageBlocks were translated.
    """
    messages: list[dict[str, Any]] = []

    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            for raw_msg in contents:
                msg: dict[str, Any] = {"role": raw_msg["role"]}
                content = raw_msg.get("content")
                if _is_content_block_list(content):
                    msg.update(_translate_content_blocks(content))
                else:
                    msg["content"] = (
                        content if isinstance(content, str) else str(content)
                    )
                # Preserve any pre-set images key on the raw message
                # (e.g. callers that already constructed the Ollama-native
                # shape); content-block translation above wins if present.
                if "images" in raw_msg and "images" not in msg:
                    msg["images"] = raw_msg["images"]
                messages.append(msg)
        else:
            messages.append(
                {"role": "user", "content": "\n".join(str(c) for c in contents)}
            )
    else:
        messages.append({"role": "user", "content": str(contents)})

    return messages


def _build_request_body(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    json_output: bool,
    thinking_budget: int | None,
    json_schema: dict | None = None,
) -> dict[str, Any]:
    """Build the native Ollama /api/chat request body.

    Parameters
    ----------
    model
        Bare model name (prefix already stripped).
    messages
        Chat messages list.
    temperature
        Sampling temperature.
    max_output_tokens
        Maximum response tokens (``num_predict`` in Ollama).
    json_output
        Whether to request JSON response format.
    thinking_budget
        Thinking token budget; > 0 enables, None/0 disables.

    Returns
    -------
    dict
        Request body for ``POST /api/chat``.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_output_tokens,
        },
    }

    # Thinking control: this is the reason we use the native API.
    # The OpenAI-compat endpoint ignores this parameter.
    if thinking_budget is not None and thinking_budget > 0:
        body["think"] = True
    else:
        body["think"] = False

    if json_schema is not None:
        body["format"] = json_schema
    elif json_output:
        body["format"] = "json"

    return body


def _normalize_finish_reason(data: dict[str, Any]) -> str | None:
    """Normalize Ollama's done_reason to standard values.

    Returns ``"stop"``, ``"max_tokens"``, or None.
    """
    if not data.get("done"):
        return None

    reason = data.get("done_reason", "")
    if reason == "stop":
        return "stop"
    elif reason == "length":
        return "max_tokens"
    elif reason:
        return reason
    return "stop"  # done=True with no reason implies normal completion


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    """Extract normalized usage dict from native Ollama response.

    Ollama uses ``prompt_eval_count`` and ``eval_count`` instead of the
    OpenAI-style ``prompt_tokens`` / ``completion_tokens``.
    """
    input_tokens = data.get("prompt_eval_count", 0) or 0
    output_tokens = data.get("eval_count", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_thinking(data: dict[str, Any]) -> list | None:
    """Extract thinking content from native Ollama response.

    The native API returns a ``thinking`` field on the message when
    thinking is enabled.
    """
    message = data.get("message", {})
    thinking = message.get("thinking")
    if thinking and isinstance(thinking, str) and thinking.strip():
        return [{"summary": thinking.strip()}]
    return None


def _parse_response(data: dict[str, Any]) -> GenerateResult:
    """Parse the native Ollama /api/chat response into GenerateResult."""
    message = data.get("message", {})
    text = message.get("content", "")

    return GenerateResult(
        text=text,
        usage=_extract_usage(data),
        finish_reason=_normalize_finish_reason(data),
        thinking=_extract_thinking(data),
    )


# ---------------------------------------------------------------------------
# run_generate / run_agenerate
# ---------------------------------------------------------------------------


def run_generate(
    contents: str | list[Any],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text synchronously via local Ollama.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_client()
    api_model = _strip_model_prefix(model)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        api_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
        json_schema,
    )

    response = client.post(
        "/api/chat",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json())


async def run_agenerate(
    contents: str | list[Any],
    model: str,
    temperature: float = 0.3,
    max_output_tokens: int = 8192 * 2,
    system_instruction: str | None = None,
    json_output: bool = False,
    thinking_budget: int | None = None,
    json_schema: dict | None = None,
    timeout_s: float | None = None,
    **kwargs: Any,
) -> GenerateResult:
    """Generate text asynchronously via local Ollama.

    Returns GenerateResult with text, usage, finish_reason, and thinking.
    See module docstring for parameter details.
    """
    client = _get_async_client()
    api_model = _strip_model_prefix(model)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        api_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
        json_schema,
    )

    response = await client.post(
        "/api/chat",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json())


# ---------------------------------------------------------------------------
# run_cogitate via OpenCode CLI
# ---------------------------------------------------------------------------


def _translate_opencode(
    event: dict[str, Any],
    aggregator: ThinkingAggregator,
    callback: JSONEventCallback,
    usage_out: dict[str, Any],
) -> str | None:
    """Translate an OpenCode JSONL event into our standard Event types.

    Args:
        event: Raw JSONL event dict from ``opencode run --format json``.
        aggregator: ThinkingAggregator for buffering text.
        callback: JSONEventCallback for emitting events.
        usage_out: Mutable dict to receive usage stats from step_finish events.

    Returns:
        The CLI session ID from step_start events, or None.
    """
    event_type = event.get("type")
    part = event.get("part", {})

    # -- step_start: capture session ID ------------------------------------
    if event_type == "step_start":
        return event.get("sessionID")

    # -- text: accumulate assistant text -----------------------------------
    if event_type == "text":
        text = part.get("text", "")
        if text:
            aggregator.accumulate(text)
        return None

    # -- tool_use: emit tool_start + tool_end ------------------------------
    # OpenCode reports tools as already completed, so we emit both events
    # back-to-back from a single JSONL line.
    if event_type == "tool_use":
        aggregator.flush_as_thinking(raw_events=[event])

        tool_name = part.get("tool", "")
        call_id = part.get("callID", "")
        state = part.get("state", {})
        tool_input = state.get("input", {})
        tool_output = state.get("output", "")

        callback.emit(
            {
                "event": "tool_start",
                "tool": tool_name,
                "args": tool_input,
                "call_id": call_id,
                "raw": safe_raw([event]),
                "ts": now_ms(),
            }
        )
        callback.emit(
            {
                "event": "tool_end",
                "tool": tool_name,
                "args": tool_input,
                "result": tool_output,
                "call_id": call_id,
                "raw": safe_raw([event]),
                "ts": now_ms(),
            }
        )
        return None

    # -- step_finish: capture usage ----------------------------------------
    if event_type == "step_finish":
        tokens = part.get("tokens")
        if tokens and usage_out is not None:
            input_tokens = tokens.get("input", 0)
            output_tokens = tokens.get("output", 0)
            total_tokens = tokens.get("total", 0)
            # Accumulate across steps (OpenCode emits one per turn)
            usage_out["input_tokens"] = usage_out.get("input_tokens", 0) + input_tokens
            usage_out["output_tokens"] = (
                usage_out.get("output_tokens", 0) + output_tokens
            )
            usage_out["total_tokens"] = usage_out.get("total_tokens", 0) + total_tokens
            reasoning = tokens.get("reasoning", 0)
            if reasoning:
                usage_out["reasoning_tokens"] = (
                    usage_out.get("reasoning_tokens", 0) + reasoning
                )
            cache = tokens.get("cache", {})
            cached_read = cache.get("read", 0)
            if cached_read:
                usage_out["cached_tokens"] = (
                    usage_out.get("cached_tokens", 0) + cached_read
                )
        return None

    # Unknown event type — log and skip
    LOG.debug("Unknown OpenCode CLI event type: %s", event_type)
    return None


def _build_opencode_env() -> dict[str, str]:
    """Build environment dict for the OpenCode subprocess.

    Sets ``OPENAI_API_KEY`` to a placeholder if not already set, since
    OpenCode's OpenAI-compatible provider requires it even for local Ollama.
    """
    env = os.environ.copy()
    if not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "ollama"
    return env


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    """Run a prompt with tool-calling support via OpenCode CLI + local Ollama.

    Uses the OpenCode CLI as a subprocess agent, which connects to the local
    Ollama instance and provides built-in tools (bash, read, glob, grep, etc.).

    Args:
        config: Complete configuration dictionary including prompt, system_instruction,
            user_instruction, extra_context, model, etc.
        on_event: Optional event callback
    """
    model = _strip_model_prefix(config.get("model", OLLAMA_FLASH))
    session_id = config.get("session_id")
    callback = JSONEventCallback(on_event)

    try:
        # Check that OpenCode CLI is available
        import shutil

        if not shutil.which("opencode"):
            raise RuntimeError(
                "Cogitate requires OpenCode CLI (opencode). "
                "Install from https://opencode.ai and configure it with a local "
                "Ollama provider. Generate works without it."
            )

        # Assemble prompt from config fields
        prompt_body, system_instruction = assemble_prompt(
            config,
            sol_tool_name="bash" if not config.get("write") else None,
        )

        # OpenCode has no --system-prompt flag; prepend to prompt body
        if system_instruction:
            prompt_body = system_instruction + "\n\n" + prompt_body

        # Build CLI command.
        # --title skips OpenCode's title-generation LLM call (avoids delays).
        agent_name = config.get("name", "sol-agent")
        cmd = [
            "opencode",
            "run",
            "--format",
            "json",
            "--title",
            agent_name,
            "-m",
            f"ollama/{model}",
        ]

        # Resume from previous session if continuing
        if session_id:
            cmd.extend(["--session", session_id])

        # Mutable container for usage accumulation
        usage: dict[str, Any] = {}

        def translate(
            event: dict[str, Any], agg: ThinkingAggregator, cb: JSONEventCallback
        ) -> str | None:
            return _translate_opencode(event, agg, cb, usage)

        aggregator = ThinkingAggregator(callback, model=model)
        cwd_value = config.get("cwd")
        runner = CLIRunner(
            cmd=cmd,
            prompt_text=prompt_body,
            translate=translate,
            callback=callback,
            aggregator=aggregator,
            cwd=Path(cwd_value) if cwd_value else None,
            env=_build_opencode_env(),
            # Local models are slower than cloud APIs; allow more time for
            # the first event (model loading + initial inference).
            first_event_timeout=120,
        )
        runner.provider = "ollama"

        result = await runner.run()

        # Emit finish event (CLIRunner does not emit one)
        finish_event: dict[str, Any] = {
            "event": "finish",
            "result": result,
            "ts": now_ms(),
        }
        if usage:
            finish_event["usage"] = usage
        if runner.cli_session_id:
            finish_event["cli_session_id"] = runner.cli_session_id
        callback.emit(finish_event)
        return result
    except QuotaExhaustedError:
        raise
    except Exception as exc:
        callback.emit(
            {
                "event": "error",
                "error": str(exc),
                "reason_code": classify_provider_error(exc, "ollama"),
                "provider": "ollama",
                "trace": traceback.format_exc(),
            }
        )
        setattr(exc, "_evented", True)
        raise


# ---------------------------------------------------------------------------
# list_models / validate_key
# ---------------------------------------------------------------------------


def list_models() -> list[dict]:
    """List available models from the local Ollama instance.

    Returns
    -------
    list[dict]
        List of model info dicts from the Ollama ``/api/tags`` endpoint.
    """
    client = _get_client()
    response = client.get("/api/tags")
    response.raise_for_status()
    return response.json().get("models", [])


def validate_key(api_key: str) -> dict:
    """Check that the local Ollama instance is reachable.

    The ``api_key`` parameter is ignored — Ollama requires no authentication.
    Connectivity is validated by hitting the version endpoint.

    Returns ``{"valid": True}`` if reachable, ``{"valid": False, "error": "..."}``
    if not.
    """
    try:
        base_url = _get_base_url()
        response = httpx.get(f"{base_url}/api/version", timeout=5)
        response.raise_for_status()
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Benchmark interface (think.benchmark.harness uses these)
# ---------------------------------------------------------------------------
#
# The harness owns the benchmark policy (prompts, media, when to run) and
# delegates transport to providers. See think/providers/shared.py for the
# BenchmarkResult contract these functions return.

# Cap context window during benchmarking to keep the compute graph
# tractable on unified-memory hosts. Ollama's default is very large
# (256K for recent Qwen builds), which inflates the KV cache + compute
# graph enough to OOM big models on the Spark. 8K is plenty for the fixed
# benchmark prompt + image tokens + 256-token completion.
_BENCHMARK_NUM_CTX = 8192


def _native_tok_s(body: dict[str, Any]) -> tuple[float | None, float | None]:
    """Compute (output_tok_s, prompt_tok_s) from Ollama's native counters.

    Ollama reports eval and prompt-eval durations in nanoseconds. Returns
    (None, None) for missing fields so callers can fall back to wall-clock
    cleanly.
    """
    eval_count = body.get("eval_count") or 0
    eval_duration_ns = body.get("eval_duration") or 0
    prompt_eval_count = body.get("prompt_eval_count") or 0
    prompt_eval_duration_ns = body.get("prompt_eval_duration") or 0

    output_tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else None
    prompt_tok_s = (
        (prompt_eval_count / (prompt_eval_duration_ns / 1e9))
        if prompt_eval_duration_ns
        else None
    )
    return output_tok_s, prompt_tok_s


def bench_ensure_installed(model: str, *, allow_pull: bool) -> None:
    """Verify the Ollama model is installed locally; optionally pull it.

    Raises SystemExit with a clear remediation message when the model is
    missing and pulling is not allowed.
    """
    bare_model = _strip_model_prefix(model)
    client = _get_client()
    response = client.get("/api/tags", timeout=10.0)
    response.raise_for_status()
    installed = {m.get("name") for m in response.json().get("models", [])}

    if bare_model in installed:
        return

    if not allow_pull:
        raise SystemExit(
            f"Model '{bare_model}' not installed. Run `ollama pull {bare_model}` "
            f"first, or pass --pull."
        )

    print(f"Pulling {bare_model}…", file=sys.stderr)
    with client.stream(
        "POST",
        "/api/pull",
        json={"name": bare_model},
        timeout=None,
    ) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if line:
                print(line, file=sys.stderr)


def bench_run_once(
    model: str,
    *,
    prompt: str,
    image_b64: str | None = None,
    audio_b64: str | None = None,
    audio_format: str = "wav",
    max_output_tokens: int = 256,
) -> BenchmarkResult:
    """Send one benchmark request and return a normalized BenchmarkResult.

    Parameters
    ----------
    model
        Model id with the ``ollama-local/`` prefix (e.g.
        ``ollama-local/qwen3.5:9b``).
    prompt
        The user-message text. The harness assembles this from a fixture
        or synthetic prompt.
    image_b64
        Optional base64-encoded image bytes. When supplied, attached as
        an ImageBlock so prompt-eval token accounting includes image
        encoding cost.
    audio_b64
        Optional base64-encoded audio bytes. Ollama does not support
        audio input — supplying this raises NotImplementedError via
        :func:`_translate_content_blocks`.
    audio_format
        Audio container (``wav``, ``flac``, etc.). Only used when
        audio_b64 is supplied.
    max_output_tokens
        ``num_predict`` ceiling for the response.

    Returns
    -------
    BenchmarkResult
        Normalized result. ``native_output_tok_s`` and
        ``native_prompt_tok_s`` are populated from Ollama's nanosecond
        counters when present.
    """
    bare_model = _strip_model_prefix(model)
    client = _get_client()

    # Build a single user message via _build_messages so the multimodal
    # translation path (image -> images: [...]) is exercised consistently
    # with run_generate.
    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64 is not None:
        blocks.append({"type": "image", "data": image_b64, "mime": "image/jpeg"})
    if audio_b64 is not None:
        blocks.append({"type": "audio", "data": audio_b64, "format": audio_format})
    messages = _build_messages([{"role": "user", "content": blocks}])

    body = {
        "model": bare_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_output_tokens,
            "num_ctx": _BENCHMARK_NUM_CTX,
        },
    }

    start = time.perf_counter()
    response = client.post("/api/chat", json=body, timeout=600.0)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    raw = response.json()

    output_tok_s, prompt_tok_s = _native_tok_s(raw)
    return BenchmarkResult(
        elapsed_s=elapsed,
        prompt_tokens=int(raw.get("prompt_eval_count") or 0),
        output_tokens=int(raw.get("eval_count") or 0),
        native_output_tok_s=output_tok_s,
        native_prompt_tok_s=prompt_tok_s,
        finish_reason=_normalize_finish_reason(raw),
        text=(raw.get("message") or {}).get("content", ""),
        raw=raw,
    )


__all__ = [
    "bench_ensure_installed",
    "bench_run_once",
    "list_models",
    "run_agenerate",
    "run_cogitate",
    "run_generate",
    "validate_key",
]
