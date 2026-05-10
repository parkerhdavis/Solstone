#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""vLLM (Local) provider for LLM generation.

vLLM serves an OpenAI-compatible HTTP API. We talk to it via raw httpx
rather than the openai SDK so the request/response shape (especially
``extra_body.chat_template_kwargs`` and content blocks for image/audio)
stays transparent and easy to evolve.

Why a separate provider from ``openai``
---------------------------------------
- Connectivity model: vLLM has no API key and is reached at a
  user-configured ``base_url`` (env ``VLLM_BASE_URL``, default
  ``http://localhost:8000``), not at api.openai.com.
- Capability mix: vLLM-served models are typically multimodal
  (Nemotron-3-Nano-Omni, Qwen-Omni, Qwen-VL, etc.). The benchmark layer
  needs to route ``vllm-local/<served_name>`` ids to this provider.
- Thinking control: vLLM honors per-request
  ``extra_body.chat_template_kwargs.enable_thinking`` for V3-class
  reasoning models, distinct from OpenAI's ``reasoning.effort`` field.

What this provider supports
---------------------------
- ``run_generate`` / ``run_agenerate`` — sync/async text generation via
  ``/v1/chat/completions``. Multimodal content blocks (image, audio) are
  translated to OpenAI-compat content arrays.
- ``list_models`` / ``validate_key`` — connectivity checks against the
  configured server.
- ``bench_run_once`` / ``bench_ensure_installed`` — the harness
  benchmark interface (see ``think/providers/shared.py``).
- ``run_cogitate`` is intentionally *not* implemented in this phase.
  vLLM is OpenAI-compat so OpenCode could in principle drive it, but
  verifying that round-trip is its own piece of work and is out of scope
  for the multimodal-benchmark branch.

Common Parameters
-----------------
contents : str or list
    String, list of strings, or list of message dicts. A message dict's
    ``content`` may be a string OR a list of ContentBlocks (TextBlock,
    ImageBlock, AudioBlock) per ``think/providers/shared.py``.
model : str
    Model id with the ``vllm-local/`` prefix (e.g.
    ``vllm-local/nemotron-omni``). The prefix is stripped to produce the
    served-model-name sent to the vLLM server.
temperature : float
    Sampling temperature (default 0.3).
max_output_tokens : int
    Maximum tokens for the model's response output.
system_instruction : str, optional
    System instruction prepended as a system-role message.
json_output : bool
    Whether to request a JSON response (sets ``response_format`` to
    ``{"type": "json_object"}``).
json_schema : dict, optional
    Structured-output JSON schema (sets ``response_format`` to
    ``{"type": "json_schema", ...}``). Wins over ``json_output``.
thinking_budget : int, optional
    When > 0, set ``chat_template_kwargs.enable_thinking=True`` (V3-class
    reasoning models honor this). When None or 0, explicitly disabled.
timeout_s : float, optional
    Per-request timeout in seconds.
**kwargs
    Absorbed for forward compatibility.

Environment Variables
---------------------
VLLM_BASE_URL : str
    Base URL of the vLLM server, used as the fallback when no
    ``journal.json → providers.vllm.servers`` entry is configured for a
    given model id (default ``http://localhost:8000``).

Multi-server routing
--------------------
For multiple vLLM servers (one per loaded model — vLLM can't hot-swap),
configure ``journal.json → providers.vllm.servers`` as a map from
*friendly name* (the part after ``vllm-local/`` in a model id) to a
server descriptor:

    "providers": {
      "vllm": {
        "servers": {
          "nemotron-omni": {
            "base_url": "http://localhost:8000",
            "served_model_name": "nemotron-omni"
          },
          "qwen3.5:35b-a3b": {
            "base_url": "http://localhost:8001",
            "served_model_name": "qwen3.5:35b-a3b"
          }
        }
      }
    }

When a model id is resolved (e.g. ``vllm-local/nemotron-omni``), the
provider strips the ``vllm-local/`` prefix to produce the friendly name,
looks it up in ``servers`` to get the ``base_url`` + ``served_model_name``,
and uses that for the chat-completions call. ``served_model_name``
defaults to the friendly name when omitted.

When no ``servers`` config exists, the env-var fallback applies and the
served name is the friendly name itself — the legacy single-server
behavior is preserved exactly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from .shared import (
    BenchmarkResult,
    GenerateResult,
    _is_content_block_list,
)

LOG = logging.getLogger("solstone.think.providers.vllm")

_VLLM_LOCAL_PREFIX = "vllm-local/"
_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 300.0  # vLLM cold-first-request can be ~20s; leave headroom

# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

# Per-base_url client caches. Each configured vLLM server gets its own
# httpx client to keep connection pools independent. Env-var fallback uses
# the same cache keyed by the resolved env URL.
_sync_clients: dict[str, httpx.Client] = {}
_async_clients: dict[str, httpx.AsyncClient] = {}


def _get_env_base_url() -> str:
    """Return the env-var fallback base URL (env-var or default)."""
    return os.getenv("VLLM_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _get_servers_config() -> dict[str, dict[str, Any]]:
    """Read ``providers.vllm.servers`` from the journal config.

    Returns an empty dict when no config is present, which signals
    callers to use the env-var single-server fallback.
    """
    # Lazy import to avoid circular: solstone.think.utils imports providers in places.
    from solstone.think.utils import get_config

    return get_config().get("providers", {}).get("vllm", {}).get("servers", {}) or {}


def _resolve_server(model: str) -> tuple[str, str]:
    """Resolve a vllm-local/ model id to ``(base_url, served_model_name)``.

    Lookup order:

    1. Strip ``vllm-local/`` prefix → friendly name.
    2. Look up ``providers.vllm.servers[<friendly>]`` in journal config.
       If present, use its ``base_url`` + ``served_model_name`` (the
       latter defaults to the friendly name when omitted).
    3. Otherwise, fall back to env-var single-server: ``VLLM_BASE_URL``
       (or default ``http://localhost:8000``) with the friendly name as
       the served-model-name.

    Raises no exceptions for missing config — the env-var path always
    succeeds at producing a tuple. Whether that server actually responds
    is checked at request time.
    """
    friendly = _strip_model_prefix(model)
    servers = _get_servers_config()
    entry = servers.get(friendly)
    if entry:
        base_url = str(entry.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
        served = str(entry.get("served_model_name") or friendly)
        return base_url, served
    return _get_env_base_url(), friendly


def _get_client(base_url: str | None = None) -> httpx.Client:
    """Get or create a cached sync httpx client for the given base URL.

    When ``base_url`` is omitted, the env-var fallback URL is used —
    preserving the pre-multi-server call shape for callers that don't
    have a model context (e.g. ``validate_key``).
    """
    url = (base_url or _get_env_base_url()).rstrip("/")
    client = _sync_clients.get(url)
    if client is None:
        client = httpx.Client(base_url=url, timeout=_DEFAULT_TIMEOUT)
        _sync_clients[url] = client
    return client


def _get_async_client(base_url: str | None = None) -> httpx.AsyncClient:
    """Async counterpart of ``_get_client``."""
    url = (base_url or _get_env_base_url()).rstrip("/")
    client = _async_clients.get(url)
    if client is None:
        client = httpx.AsyncClient(base_url=url, timeout=_DEFAULT_TIMEOUT)
        _async_clients[url] = client
    return client


def _all_configured_base_urls() -> list[str]:
    """Return the set of base URLs for all configured servers, or the env-var
    fallback URL when no ``providers.vllm.servers`` config is present.

    Used by callers that need to enumerate across all servers (list_models,
    validate_key, status checks) rather than route by model id.
    """
    servers = _get_servers_config()
    if servers:
        urls: list[str] = []
        seen = set()
        for entry in servers.values():
            url = str(entry.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
            if url not in seen:
                seen.add(url)
                urls.append(url)
        return urls
    return [_get_env_base_url()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_model_prefix(model: str) -> str:
    """Strip the ``vllm-local/`` prefix to produce the served-model-name.

    The vLLM server matches the ``model`` field in chat-completions
    requests against the ``--served-model-name`` set at server startup.
    Solstone uses the ``vllm-local/`` prefix for provider routing.
    """
    if model.startswith(_VLLM_LOCAL_PREFIX):
        return model[len(_VLLM_LOCAL_PREFIX) :]
    return model


def _translate_content_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Solstone ContentBlocks into OpenAI-compat content array entries.

    Mapping:
      TextBlock  -> {"type": "text", "text": ...}
      ImageBlock -> {"type": "image_url",
                     "image_url": {"url": "data:<mime>;base64,..."}}
      AudioBlock -> {"type": "input_audio",
                     "input_audio": {"data": <base64>, "format": <wav|...>}}

    The OpenAI ``input_audio`` shape is what vLLM 0.20.0+ accepts for
    multimodal audio models (confirmed in Phase 0 against
    nemotron-omni). The alternate ``audio_url`` shape with ``data:`` URIs
    also works on vLLM but is not used here for portability with other
    OpenAI-compat backends.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            data = block.get("data")
            if not data:
                raise ValueError("ImageBlock missing 'data' (base64-encoded bytes)")
            mime = block.get("mime") or "image/jpeg"
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        elif btype == "audio":
            data = block.get("data")
            audio_format = block.get("format")
            if not data:
                raise ValueError("AudioBlock missing 'data' (base64-encoded bytes)")
            if not audio_format:
                raise ValueError("AudioBlock missing 'format' (e.g. 'wav')")
            out.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": audio_format},
                }
            )
        else:
            raise ValueError(f"Unknown content-block type: {btype!r}")
    return out


def _build_messages(
    contents: Any,
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Convert contents and system instruction to OpenAI-compat chat messages.

    Parameters
    ----------
    contents
        One of:
        - A plain string (legacy text-only user message).
        - A list of strings (joined with newlines into a single user message).
        - A list of message dicts with ``role`` keys. Each message's
          ``content`` may be a string OR a list of ContentBlock dicts.
    system_instruction
        Optional system prompt, prepended as a system-role message.

    Returns
    -------
    list[dict[str, Any]]
        Messages in ``[{role, content}, ...]`` format. Multimodal
        messages have ``content`` as a list of OpenAI-compat content-array
        entries (``{type: text|image_url|input_audio, ...}``).
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
                    msg["content"] = _translate_content_blocks(content)
                elif isinstance(content, str):
                    msg["content"] = content
                else:
                    msg["content"] = str(content)
                messages.append(msg)
        else:
            messages.append(
                {"role": "user", "content": "\n".join(str(c) for c in contents)}
            )
    else:
        messages.append({"role": "user", "content": str(contents)})

    return messages


def _build_request_body(
    served_model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    json_output: bool,
    thinking_budget: int | None,
    json_schema: dict | None,
) -> dict[str, Any]:
    """Build the OpenAI-compat /v1/chat/completions request body."""
    body: dict[str, Any] = {
        "model": served_model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "stream": False,
    }

    # Thinking control: vLLM's OpenAI-compat endpoint honors per-request
    # chat_template_kwargs via the extra_body sidecar. Always pin it so
    # behavior doesn't depend on whatever default the server was started
    # with.
    enable_thinking = bool(thinking_budget and thinking_budget > 0)
    body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    if json_schema is not None:
        title = json_schema.get("title") if isinstance(json_schema, dict) else None
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": title or "response",
                "schema": json_schema,
                "strict": True,
            },
        }
    elif json_output:
        body["response_format"] = {"type": "json_object"}

    return body


def _normalize_finish_reason(choice: dict[str, Any]) -> str | None:
    """Normalize OpenAI-style finish_reason to standard values."""
    reason = choice.get("finish_reason")
    if reason == "stop":
        return "stop"
    if reason == "length":
        return "max_tokens"
    if reason == "tool_calls":
        return "stop"
    if reason == "content_filter":
        return "safety"
    return reason


def _extract_thinking(message: dict[str, Any]) -> list | None:
    """Pull V3 reasoning content out of the response message.

    Per Phase 0 findings, vLLM 0.20.0 with ``--reasoning-parser
    nemotron_v3`` populates ``message.reasoning`` (not
    ``reasoning_content``) when thinking ran. We surface it as a
    Solstone thinking-block list for parity with other providers. Older
    parsers that use ``reasoning_content`` are also accepted as a
    fallback.
    """
    raw = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(raw, str) and raw.strip():
        return [{"summary": raw.strip()}]
    return None


def _extract_text(message: dict[str, Any]) -> str:
    """Coalesce ``content`` and ``reasoning`` into a single text response.

    Per Phase 0 finding: when thinking is enabled and the prompt is
    descriptive, V3 reasoning models can emit *all* output through the
    ``reasoning`` channel and leave ``content=None``. Most callers want
    "the model's response" regardless of which channel it came through,
    so we return content when present and fall back to reasoning. The
    structured ``thinking`` field on GenerateResult still carries the
    reasoning separately for callers that need to distinguish.
    """
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        return reasoning
    return ""


def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
    """Extract a normalized usage dict from a chat-completions response."""
    usage = data.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    out: dict[str, int] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    # Optional reasoning-token breakdown from completion_tokens_details.
    details = usage.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    if reasoning:
        out["reasoning_tokens"] = reasoning
    return out


def _parse_response(data: dict[str, Any]) -> GenerateResult:
    """Parse a vLLM chat-completions response into GenerateResult."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    return GenerateResult(
        text=_extract_text(message),
        usage=_extract_usage(data),
        finish_reason=_normalize_finish_reason(choice),
        thinking=_extract_thinking(message),
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
    """Generate text synchronously via the configured vLLM server."""
    base_url, served_model = _resolve_server(model)
    client = _get_client(base_url)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        served_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
        json_schema,
    )

    response = client.post(
        "/v1/chat/completions",
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
    """Generate text asynchronously via the configured vLLM server."""
    base_url, served_model = _resolve_server(model)
    client = _get_async_client(base_url)
    messages = _build_messages(contents, system_instruction)
    body = _build_request_body(
        served_model,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        thinking_budget,
        json_schema,
    )

    response = await client.post(
        "/v1/chat/completions",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json())


# ---------------------------------------------------------------------------
# list_models / validate_key
# ---------------------------------------------------------------------------


def list_models() -> list[dict]:
    """List models served across all configured vLLM servers.

    With ``providers.vllm.servers`` configured, queries each unique
    ``base_url`` and returns the union of their served models. Without
    config, falls back to the env-var single-server case.

    Each returned entry has the ``vllm-local/`` prefix prepended to its
    id so it's directly usable as a model parameter elsewhere in
    Solstone, mirroring how ``ollama.list_models`` exposes prefixed ids.
    Unreachable servers are debug-logged and skipped — listing succeeds
    with whatever is reachable.
    """
    out: list[dict] = []
    for base_url in _all_configured_base_urls():
        try:
            response = httpx.get(f"{base_url}/v1/models", timeout=5.0)
            response.raise_for_status()
        except Exception as exc:
            LOG.debug("vLLM server unreachable at %s: %s", base_url, exc)
            continue
        for entry in response.json().get("data") or []:
            item = dict(entry)
            served_id = item.get("id")
            if isinstance(served_id, str) and not served_id.startswith(
                _VLLM_LOCAL_PREFIX
            ):
                item["id"] = f"{_VLLM_LOCAL_PREFIX}{served_id}"
            out.append(item)
    return out


def validate_key(api_key: str) -> dict:
    """Check that at least one configured vLLM server is reachable.

    The ``api_key`` parameter is ignored — vLLM does not require auth by
    default. With ``providers.vllm.servers`` configured, returns
    ``valid: True`` if *any* server responds; ``unreachable`` lists
    servers that didn't respond. Without config, validates the single
    env-var server.
    """
    urls = _all_configured_base_urls()
    reachable: list[str] = []
    unreachable: list[dict[str, str]] = []
    for url in urls:
        try:
            response = httpx.get(f"{url}/v1/models", timeout=5)
            response.raise_for_status()
            reachable.append(url)
        except Exception as exc:
            unreachable.append({"base_url": url, "error": str(exc)})
    if reachable:
        result: dict[str, Any] = {"valid": True, "reachable": reachable}
        if unreachable:
            result["unreachable"] = unreachable
        return result
    # No reachable servers — return the most informative error.
    if unreachable:
        return {"valid": False, "error": unreachable[0]["error"]}
    return {"valid": False, "error": "no vLLM servers configured or reachable"}


# ---------------------------------------------------------------------------
# Benchmark interface
# ---------------------------------------------------------------------------
#
# vLLM does not report server-side eval/prompt-eval durations, unlike
# Ollama. Tok/s comes from wall-clock × token counts at the harness
# level. native_output_tok_s / native_prompt_tok_s are therefore always
# None; the harness's _bench_tok_s helper falls back accordingly.


def bench_ensure_installed(model: str, *, allow_pull: bool) -> None:
    """Verify the requested model is currently served by the vLLM server.

    vLLM has no pull-on-demand: the model is loaded at server startup and
    can only be changed by stopping/starting the server with new
    arguments. ``allow_pull`` is therefore meaningless for this provider.
    Raises SystemExit with remediation guidance if the model is missing
    or the server is unreachable.
    """
    base_url, served_model = _resolve_server(model)
    try:
        response = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"vLLM server unreachable at {base_url}: {exc}. "
            f"Start a vLLM server (see ~/solstone-vllm-spike/serve.sh for "
            f"the Phase 0 reference invocation) before running the harness."
        ) from exc

    served_ids = {m.get("id") for m in (response.json().get("data") or [])}
    if served_model in served_ids:
        return

    served_list = ", ".join(sorted(s for s in served_ids if isinstance(s, str)))
    if allow_pull:
        # vLLM has no pull endpoint; surface the no-op explicitly.
        LOG.warning(
            "vLLM does not support model pulling — --pull is a no-op for "
            "vllm-local/ ids. Configure --served-model-name on the running "
            "vLLM server instead."
        )
    raise SystemExit(
        f"Model '{served_model}' is not currently served by vLLM at {base_url}. "
        f"Currently served: {served_list or '(none)'}. Restart the vLLM "
        f"server with --served-model-name {served_model} (or a matching id) "
        f"to make it available."
    )


def bench_run_once(
    model: str,
    *,
    prompt: str,
    image_b64: str | None = None,
    audio_b64: str | None = None,
    audio_format: str = "wav",
    max_output_tokens: int = 256,
) -> BenchmarkResult:
    """Send one benchmark request to the vLLM server and return a BenchmarkResult.

    Wall-clock timing is measured around the HTTP round-trip; vLLM does
    not expose server-side eval-duration counters, so
    ``native_output_tok_s`` and ``native_prompt_tok_s`` are always None.
    Harness reporting falls back to ``output_tokens / elapsed_s`` for the
    output rate and leaves the prompt rate at 0 (see
    ``solstone.think.benchmark.harness._bench_tok_s``).

    Thinking is explicitly disabled for benchmark requests so the wall-
    clock and token counts measure pure response generation, not silent
    reasoning that may dwarf the response. Talents that want thinking on
    use ``run_generate`` directly.
    """
    base_url, served_model = _resolve_server(model)
    client = _get_client(base_url)

    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_b64 is not None:
        blocks.append({"type": "image", "data": image_b64, "mime": "image/jpeg"})
    if audio_b64 is not None:
        blocks.append({"type": "audio", "data": audio_b64, "format": audio_format})
    messages = _build_messages([{"role": "user", "content": blocks}])

    body = _build_request_body(
        served_model,
        messages,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
        json_output=False,
        thinking_budget=None,  # benchmarks measure direct output, not reasoning
        json_schema=None,
    )

    start = time.perf_counter()
    response = client.post("/v1/chat/completions", json=body, timeout=600.0)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    raw = response.json()

    usage = raw.get("usage") or {}
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    return BenchmarkResult(
        elapsed_s=elapsed,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        native_output_tok_s=None,
        native_prompt_tok_s=None,
        finish_reason=_normalize_finish_reason(choice),
        text=_extract_text(message),
        raw=raw,
    )


__all__ = [
    "bench_ensure_installed",
    "bench_run_once",
    "list_models",
    "run_agenerate",
    "run_generate",
    "validate_key",
]
