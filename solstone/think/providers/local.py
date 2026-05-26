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

from solstone.think.models import LOCAL_FLASH, LOCAL_LITE, LOCAL_PRO
from solstone.think.providers.shared import (
    BenchmarkResult,
    GenerateResult,
    _is_content_block_list,
    classify_provider_error,
    safe_raw,
)

LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0
_LOCAL_PREFIX = "local/"


# Fork-only model id for the experimental omni model (vision on the bundle;
# audio stays on Whisper — llama-server b9291 rejects Nemotron audio input,
# see the Phase C spike in the migration plan).
LOCAL_OMNI = "local/nemotron-3-nano-omni"


@dataclass(frozen=True)
class LocalModelSpec:
    model_id: str
    repo: str
    filename: str
    revision: str
    sha256: str
    size_bytes: int
    min_ram_bytes: int
    # Multimodal projector (mmproj) for vision-capable models; None for
    # text-only. When set, local_install downloads it alongside the main GGUF
    # and local_server passes --mmproj at launch (libmtmd vision path).
    mmproj_filename: str | None = None
    mmproj_sha256: str | None = None
    mmproj_size_bytes: int = 0

    @property
    def supports_vision(self) -> bool:
        return bool(self.mmproj_filename)


LOCAL_MODEL_SPECS: dict[str, LocalModelSpec] = {
    LOCAL_LITE: LocalModelSpec(
        model_id=LOCAL_LITE,
        repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        revision="main",
        sha256="509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
        size_bytes=4_683_073_536,
        min_ram_bytes=12 * 1024**3,
    ),
    LOCAL_PRO: LocalModelSpec(
        model_id=LOCAL_PRO,
        repo="giladgd/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M-GGUF",
        filename="qwen3-coder-30b-a3b-instruct-q4_k_m.gguf",
        revision="main",
        sha256="ab4fc2b27b2043483a9e346c802809dfbe9b775efbeea7ca74dc2fd1aa4a0f71",
        size_bytes=18_556_688_704,
        min_ram_bytes=32 * 1024**3,
    ),
    # Fork-only: NVIDIA Nemotron 3 Nano Omni (Q8_0) + its vision mmproj, from
    # ggml-org. Vision-capable on the bundle; audio is NOT wired (llama-server
    # b9291 returns "audio input is not supported" — audio stays on Whisper).
    LOCAL_OMNI: LocalModelSpec(
        model_id=LOCAL_OMNI,
        repo="ggml-org/NVIDIA-Nemotron-3-Nano-Omni",
        filename="nemotron-3-nano-omni-ga_v1.0-Q8_0.gguf",
        revision="main",
        sha256="98e5cbdb3cb9bd172ddfeb164edb3fea049364750eea2fc20d1011e640748571",
        size_bytes=33_585_495_872,
        min_ram_bytes=48 * 1024**3,
        mmproj_filename="mmproj-nemotron-3-nano-omni-ga_v1.0.gguf",
        mmproj_sha256="797d096c07c80a5d49ec3793b6d96889fa394a1207e0aa558effebde6928c2a9",
        mmproj_size_bytes=1_587_540_672,
    ),
}


class LocalProviderError(RuntimeError):
    """Local provider failure with a recovery reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def normalize_model_id(model: str | None) -> str:
    model_id = str(model or LOCAL_FLASH)
    if model_id.startswith("openai/"):
        model_id = model_id[len("openai/") :]
    if not model_id.startswith(_LOCAL_PREFIX):
        raise LocalProviderError(
            "unsupported_model",
            f"Local provider model must start with {_LOCAL_PREFIX!r}: {model_id}",
        )
    if model_id not in LOCAL_MODEL_SPECS:
        raise LocalProviderError(
            "unsupported_model", f"Unsupported local model: {model_id}"
        )
    return model_id


def _translate_content_blocks(
    content: list[dict[str, Any]], *, supports_vision: bool
) -> list[dict[str, Any]]:
    """Translate Solstone ContentBlocks into llama-server OpenAI-compat entries.

    TextBlock  -> {"type": "text", "text": ...}
    ImageBlock -> {"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}

    Images require a vision-capable model (one with an mmproj projector); on a
    text-only model they raise ``unsupported_capability``. AudioBlock always
    raises: llama-server b9291 rejects Nemotron audio input ("audio input is
    not supported"), so audio stays on the Whisper STT pipeline (Phase C spike).
    """
    out: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            if not supports_vision:
                raise LocalProviderError(
                    "unsupported_capability",
                    "Image input requires a vision-capable local model "
                    "(one with an mmproj projector, e.g. local/nemotron-3-nano-omni).",
                )
            data = block.get("data")
            if not data:
                raise LocalProviderError(
                    "provider_request_invalid",
                    "ImageBlock missing 'data' (base64-encoded bytes).",
                )
            mime = block.get("mime") or "image/jpeg"
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        elif btype == "audio":
            raise LocalProviderError(
                "unsupported_capability",
                "The local bundle does not support audio input (llama-server "
                "rejects it for this model); audio runs through Whisper STT.",
            )
        else:
            raise LocalProviderError(
                "provider_request_invalid", f"Unknown content-block type: {btype!r}"
            )
    return out


def _build_messages(
    contents: str | list[Any],
    system_instruction: str | None = None,
    *,
    supports_vision: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
    elif _is_content_block_list(contents):
        messages.append(
            {
                "role": "user",
                "content": _translate_content_blocks(
                    contents, supports_vision=supports_vision
                ),
            }
        )
    elif isinstance(contents, list):
        if contents and isinstance(contents[0], dict) and "role" in contents[0]:
            for item in contents:
                role = str(item.get("role", "user"))
                content = item.get("content", "")
                if _is_content_block_list(content):
                    messages.append(
                        {
                            "role": role,
                            "content": _translate_content_blocks(
                                content, supports_vision=supports_vision
                            ),
                        }
                    )
                elif isinstance(content, str):
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": role, "content": str(content)})
        else:
            messages.append(
                {"role": "user", "content": "\n".join(str(item) for item in contents)}
            )
    else:
        messages.append({"role": "user", "content": str(contents)})
    return messages


def _build_request_body(
    model_id: str,
    messages: list[dict[str, str]],
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
    messages = _build_messages(
        contents,
        system_instruction,
        supports_vision=LOCAL_MODEL_SPECS[model_id].supports_vision,
    )
    server = local_server.ensure_running(model_id)
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

    model_id = normalize_model_id(config.get("model", LOCAL_FLASH))
    try:
        local_server.ensure_running(model_id, on_event=on_event)
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
            model=LOCAL_FLASH,
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

    The v1 local provider is text-only, so image/audio blocks raise
    ``unsupported_capability`` (multimodal arrives in Phase C via libmtmd /
    ``--mmproj``). llama-server reports per-request ``timings``, so native
    output/prompt tok/s are populated when present — the harness prefers these
    over the wall-clock-derived rate.
    """
    import time

    import httpx

    from solstone.think.providers import local_server

    if image_b64 is not None or audio_b64 is not None:
        raise LocalProviderError(
            "unsupported_capability",
            "The local provider is text-only (v1); image/audio benchmarking "
            "lands in Phase C (libmtmd / --mmproj).",
        )
    del audio_format

    model_id = normalize_model_id(model)
    messages = _build_messages(prompt, None)
    server = local_server.ensure_running(model_id)
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
