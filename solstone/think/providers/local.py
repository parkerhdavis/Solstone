# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Bundled local provider backed by llama-server on 127.0.0.1.

The module must remain importable before the local runtime or GGUF files exist.
Network clients and daemon startup are created only inside provider functions.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from solstone.think.models import LOCAL_MODEL
from solstone.think.providers._image import encode_image_part, is_image_part
from solstone.think.providers.shared import (
    GenerateResult,
    classify_provider_error,
    safe_raw,
)

LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0
_LOCAL_PREFIX = "local/"


@dataclass(frozen=True)
class LocalModelSpec:
    model_id: str
    repo: str
    filename: str
    revision: str
    sha256: str
    size_bytes: int
    min_ram_bytes: int
    mmproj_filename: str | None = None
    mmproj_sha256: str | None = None


# Fork: the single bundled local model is NVIDIA Nemotron 3 Nano Omni (Q8_0),
# a vision-capable omni model, paired with its mmproj projector. Upstream ships
# a text-only qwen2.5-coder here; the fork leans into local vision so the one
# always-on local daemon serves both text/agentic cogitate AND image input.
# Audio is NOT wired: llama-server (b9291) rejects Nemotron audio input ("audio
# input is not supported"), so audio stays on the Whisper STT pipeline. The
# supervisor passes --mmproj at launch when the spec carries one (upstream
# machinery, unchanged). See docs/FORK.md "Local vision".
LOCAL_MODEL_SPECS: dict[str, LocalModelSpec] = {
    LOCAL_MODEL: LocalModelSpec(
        model_id=LOCAL_MODEL,
        repo="ggml-org/NVIDIA-Nemotron-3-Nano-Omni",
        filename="nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf",
        revision="main",
        sha256="98e5cbdb3cb9bd172ddfeb164edb3fea049364750eea2fc20d1011e640748571",
        size_bytes=33_585_495_872,
        min_ram_bytes=48 * 1024**3,
        mmproj_filename="mmproj-nemotron-3-nano-omni-ga_v1.0.gguf",
        mmproj_sha256="797d096c07c80a5d49ec3793b6d96889fa394a1207e0aa558effebde6928c2a9",
    ),
}


class LocalProviderError(RuntimeError):
    """Local provider failure with a recovery reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def normalize_model_id(model: str | None) -> str:
    model_id = str(model or LOCAL_MODEL)
    if model_id.startswith("openai/"):
        model_id = model_id[len("openai/") :]
    if not model_id.startswith(_LOCAL_PREFIX):
        raise LocalProviderError(
            "unsupported_model",
            f"Local provider model must start with {_LOCAL_PREFIX!r}: {model_id}",
        )
    return LOCAL_MODEL


def _contains_image(value: Any) -> bool:
    if is_image_part(value):
        return True
    if isinstance(value, dict):
        return any(_contains_image(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_image(item) for item in value)
    return False


def _image_content_part(part: Any) -> dict[str, Any]:
    media_type, b64 = encode_image_part(part)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }


def _content_parts(value: Any) -> list[dict[str, Any]]:
    if is_image_part(value):
        return [_image_content_part(value)]
    if isinstance(value, list | tuple):
        parts: list[dict[str, Any]] = []
        for item in value:
            parts.extend(_content_parts(item))
        return parts
    return [{"type": "text", "text": str(value)}]


def _message_content(value: Any) -> str | list[dict[str, Any]]:
    if _contains_image(value):
        return _content_parts(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "\n".join(str(item) for item in value)
    return str(value)


def _build_messages(
    contents: str | list[Any],
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            for item in contents:
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                messages.append({"role": role, "content": _message_content(content)})
        else:
            messages.append({"role": "user", "content": _message_content(contents)})
    else:
        messages.append({"role": "user", "content": str(contents)})
    return messages


def _build_request_body(
    model_id: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_output_tokens: int,
    json_output: bool,
    json_schema: dict | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "local_schema",
                "schema": json_schema,
                "strict": True,
            },
        }
    elif json_output:
        body["response_format"] = {"type": "json_object"}
    return body


def _extract_usage(data: dict[str, Any]) -> dict[str, int] | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _parse_response(data: dict[str, Any], requested_model: str) -> GenerateResult:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalProviderError("provider_response_invalid", "No response from model.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LocalProviderError(
            "provider_response_invalid", "Malformed model response."
        )
    message = choice.get("message")
    text = ""
    if isinstance(message, dict):
        content = message.get("content", "")
        text = content if isinstance(content, str) else ""
    return GenerateResult(
        text=text,
        model=data.get("model")
        if isinstance(data.get("model"), str)
        else requested_model,
        usage=_extract_usage(data),
        finish_reason=choice.get("finish_reason"),
        thinking=None,
    )


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
    del thinking_budget, kwargs
    from solstone.think.providers import local_server

    model_id = normalize_model_id(model)
    messages = _build_messages(contents, system_instruction)
    server = local_server.connect()
    body = _build_request_body(
        model_id,
        messages,
        temperature,
        max_output_tokens,
        json_output,
        json_schema,
    )

    import httpx

    response = httpx.post(
        f"{server.base_url}/v1/chat/completions",
        json=body,
        timeout=timeout_s or _DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return _parse_response(response.json(), model_id)


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
    return await asyncio.to_thread(
        run_generate,
        contents,
        model,
        temperature,
        max_output_tokens,
        system_instruction,
        json_output,
        thinking_budget,
        json_schema,
        timeout_s,
        **kwargs,
    )


async def run_cogitate(
    config: dict[str, Any],
    on_event: Callable[[dict], None] | None = None,
) -> str:
    from solstone.think.providers import local_server, openhands

    config = {**config, "model": normalize_model_id(config.get("model", LOCAL_MODEL))}
    try:
        local_server.connect()
        return await openhands.run_cogitate(config, on_event=on_event)
    except Exception as exc:
        if on_event and not getattr(exc, "_evented", False):
            reason_code = getattr(exc, "reason_code", None) or classify_provider_error(
                exc, "local"
            )
            on_event(
                {
                    "event": "error",
                    "error": str(exc),
                    "reason_code": reason_code,
                    "provider": "local",
                    "trace": traceback.format_exc(),
                    "raw": safe_raw([{"reason_code": reason_code}]),
                }
            )
            setattr(exc, "_evented", True)
        raise


def list_models(provider: str = "local") -> list[dict[str, Any]]:
    del provider
    return [
        {
            "name": spec.model_id,
            "model": spec.model_id,
            "repo": spec.repo,
            "filename": spec.filename,
            "size_bytes": spec.size_bytes,
            "min_ram_bytes": spec.min_ram_bytes,
        }
        for spec in LOCAL_MODEL_SPECS.values()
    ]


def validate_key(provider: str = "local", api_key: str = "") -> dict[str, Any]:
    del provider, api_key
    try:
        run_generate(
            "Say OK",
            model=LOCAL_MODEL,
            temperature=0,
            max_output_tokens=8,
            timeout_s=10,
        )
        return {"valid": True}
    except Exception as exc:
        return {
            "valid": False,
            "error": str(exc),
            "reason_code": getattr(exc, "reason_code", None)
            or classify_provider_error(exc, "local"),
        }


def bench_ensure_installed(model: str, *, allow_pull: bool) -> None:
    """Ensure the bundled binary + GGUF for ``model`` are installed for benchmarking.

    The local bundle supports pull-on-demand: when ``allow_pull`` is set, any
    missing artifact is downloaded via the bundle's installer. Raises SystemExit
    with remediation guidance when something is missing and pulling is disabled.
    """
    from solstone.think.providers import local_install

    model_id = normalize_model_id(model)
    readiness = local_install.inspect_readiness(model_id)
    if readiness["binary_installed"] and readiness["model_installed"]:
        return

    missing = []
    if not readiness["binary_installed"]:
        missing.append("llama-server binary")
    if not readiness["model_installed"]:
        missing.append(f"model {model_id}")
    if not allow_pull:
        raise SystemExit(
            "Local provider not ready for benchmarking — missing: "
            f"{', '.join(missing)}. Re-run with --pull to download via the bundle."
        )

    if not readiness["binary_installed"]:
        local_install.install_llama_server()
    if not readiness["model_installed"]:
        local_install.install_model(model_id)


def bench_run_once(
    model: str,
    *,
    prompt: str,
    image_b64: str | None = None,
    audio_b64: str | None = None,
    audio_format: str = "wav",
    max_output_tokens: int = 256,
) -> BenchmarkResult:
    """Send one benchmark request to the bundled llama-server; return a BenchmarkResult.

    Connect-only: this attaches to the supervisor-owned local daemon (it does
    not spawn its own server), so the daemon must be running. Production image
    input works through ``run_generate`` (Nemotron Omni + mmproj), but the
    benchmark harness's image/audio measurement path is not wired yet — see
    docs/FORK.md follow-ups — so image/audio benchmark args raise here for now.
    Audio never benchmarks on the bundle: llama-server rejects Nemotron audio
    input, so audio runs through the Whisper STT pipeline. llama-server reports
    per-request ``timings``, so native output/prompt tok/s are populated when
    present — the harness prefers these over the wall-clock-derived rate.
    """
    import time

    import httpx

    from solstone.think.providers import local_server

    if image_b64 is not None or audio_b64 is not None:
        raise LocalProviderError(
            "unsupported_capability",
            "Image/audio benchmarking is not wired through the harness yet; "
            "text-only benchmark requests are supported.",
        )
    del audio_format

    model_id = normalize_model_id(model)
    messages = _build_messages(prompt, None)
    server = local_server.connect()
    body = _build_request_body(model_id, messages, 0.2, max_output_tokens, False, None)
    # Disable llama-server's prompt cache for benchmarking: the harness reuses
    # the same prompt across warmup + measured runs, and a cached prefix makes
    # the native prompt_per_second timing reflect near-zero work (misleading).
    # Production generates keep caching (run_generate omits this).
    body["cache_prompt"] = False

    start = time.perf_counter()
    response = httpx.post(
        f"{server.base_url}/v1/chat/completions", json=body, timeout=600.0
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    raw = response.json()

    usage = _extract_usage(raw) or {"input_tokens": 0, "output_tokens": 0}
    choices = raw.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    text = content if isinstance(content, str) else ""

    timings = raw.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    native_output = timings.get("predicted_per_second")
    native_prompt = timings.get("prompt_per_second")

    return BenchmarkResult(
        elapsed_s=elapsed,
        prompt_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        native_output_tok_s=(
            float(native_output) if isinstance(native_output, int | float) else None
        ),
        native_prompt_tok_s=(
            float(native_prompt) if isinstance(native_prompt, int | float) else None
        ),
        finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        text=text,
        raw=raw,
    )


__all__ = [
    "LOCAL_MODEL_SPECS",
    "LocalModelSpec",
    "LocalProviderError",
    "normalize_model_id",
    "run_generate",
    "run_agenerate",
    "run_cogitate",
    "list_models",
    "validate_key",
    "bench_ensure_installed",
    "bench_run_once",
]
