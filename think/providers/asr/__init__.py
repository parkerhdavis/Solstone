# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""ASR (speech-to-text) provider backends.

Mirrors the structural shape of ``think.providers`` (LLM providers) for
ASR backends that talk to remote/containerized services rather than
running inference in-process. The point of this package is to keep the
Solstone Python codebase platform-portable: the bundled in-process
backends (``observe.transcribe.parakeet`` and
``observe.transcribe.whisper``) couple model code to platform-specific
wheels (CoreML on macOS, ONNX on linux-x86_64, faster-whisper everywhere
via Python). This package houses the providers that talk to model code
that lives **on the other side of a network boundary** — which is what
makes them platform-portable on Solstone's side.

Currently registered:

- ``parakeet-nim``: NVIDIA Parakeet TDT served by the NIM container
  (HTTP, OpenAI-compatible ``/v1/audio/transcriptions``). Required on
  ``linux/aarch64`` + Blackwell hosts (DGX Spark) where the bundled
  parakeet path doesn't run. See
  ``docs/PROVIDERS.md`` and the project Transcription Backend
  Architecture doc for the architectural rationale.

Each provider module exports a single function::

    transcribe(
        audio: np.ndarray,    # float32, mono, target sample rate
        sample_rate: int,     # typically 16000
        config: dict,         # backend-specific config from journal.json
    ) -> list[dict]

Returning the same statement dicts that ``observe.transcribe`` already
consumes (id / start / end / text / words / speaker). Callers normally
go through ``observe.transcribe.transcribe(backend, ...)``, which
dispatches by registered backend name; the entries in
``observe.transcribe.BACKEND_REGISTRY`` for these providers point at
the corresponding ``think.providers.asr.*`` module path.

Anti-patterns this package exists to prevent (see project memory and
docs/PROVIDERS.md):

- Importing NeMo / NVIDIA SDKs into Solstone Python. The HTTP boundary
  is load-bearing.
- "Unifying" ``parakeet`` and ``parakeet-nim``. They're distinct
  backends with different supported_hardware.
- Silently falling back to whisper when a NIM endpoint is unreachable.
  Hard-fail at the provider level; corruption of the cross-backend
  benchmarking signal is a worse outcome than a noisy error.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

ASR_PROVIDER_REGISTRY: dict[str, str] = {
    "parakeet-nim": "think.providers.asr.parakeet_nim",
}


ASR_PROVIDER_METADATA: dict[str, dict[str, Any]] = {
    "parakeet-nim": {
        "label": "Parakeet TDT (NVIDIA NIM, HTTP)",
        "kind": "local-http",
        "endpoint_env": "PARAKEET_NIM_URL",
        "endpoint_default": "http://localhost:9000",
        "supported_hardware": ["dgx-spark"],
        "notes": (
            "Containerized service. Hard-fails when the endpoint is "
            "unreachable rather than falling back to another backend."
        ),
    },
}


def get_asr_provider_module(provider: str) -> ModuleType:
    """Return the provider module for ``provider``.

    Raises ``ValueError`` (not ``SystemExit``) when the provider isn't
    registered, so callers in the dispatch layer can decide how to
    surface the error. Harness preflight has its own ``SystemExit``
    path keyed off ``transcribers.json`` instead — that file is the
    user-facing surface.
    """
    if provider not in ASR_PROVIDER_REGISTRY:
        valid = ", ".join(sorted(ASR_PROVIDER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown ASR provider: {provider!r}. Registered: {valid}"
        )
    return import_module(ASR_PROVIDER_REGISTRY[provider])


__all__ = [
    "ASR_PROVIDER_REGISTRY",
    "ASR_PROVIDER_METADATA",
    "get_asr_provider_module",
]
