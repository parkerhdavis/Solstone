# Fork Changelog

This document tracks significant changes made on this fork of Solstone.


## Field Journal Test Content

**Files:** `setup_field_journal.sh`, `docs/FIELD_JOURNAL.md`

This fork uses [solpbc/field_journal](https://github.com/solpbc/field_journal) — a public-domain media corpus — as its journal content, making this instance a dedicated testing and development environment rather than a personal capture one. `setup_field_journal.sh` at the repo root copies days from a local field_journal clone (default `~/Field_Journal/`) into `journal/chronicle/`. Setup lives in a standalone script rather than the `Makefile` so shared files stay convergent with upstream. See `docs/FIELD_JOURNAL.md` for the full workflow.


## Local-Model Benchmark Heuristic

**Files:** `apps/benchmark/`, `think/benchmark/`, `tests/test_benchmark_estimate.py`

Added a `benchmark` app and supporting `think/benchmark/` module that estimates expected output tok/s for pre-vetted Ollama models on the user's hardware without requiring the models to be pulled. A reference table of measured tok/s per canonical hardware class (see `think/benchmark/reference.json`) is interpolated by FP16 throughput × memory bandwidth when the exact hardware isn't listed. The registry (`models.json`) covers text and vision models across tiers, with direct wall-clock measurements taken on DGX Spark used to ground the task-time heuristics.

The `sol call benchmark` CLI exposes `profile` (probe + cache host hardware), `list-models` (pre-vetted + installed models with tok/s and task-time estimates), `estimate <model-id>` (single-model estimate, optionally `--task <task_id>` for a wall-clock estimate against a reference workload), and `tasks` (show the reference-task catalog). A harness (`think/benchmark/harness.py`) runs the fixture-backed reference tasks (`fixtures/*.txt`) to produce new measurements that feed back into the registry. The settings UI surfaces per-model tok/s alongside the cogitate details panel, with generic tier labels and a recommended-models section for quick orientation.


---

*The following changes originated on this fork and have since been merged upstream.*


## Ollama (Local) Provider

> **Merged upstream.** The `think/providers/ollama.py` provider, tests, and
> associated config/docs changes are now part of upstream Solstone.

Added a new `ollama` provider that routes text generation requests to a local
Ollama instance via its native `/api/chat` endpoint, removing the hard
dependency on cloud API keys for the `generate` workload. Ollama models use
the `ollama-local/` prefix (e.g., `ollama-local/qwen3.5:9b`) to leave room for
a future `ollama-cloud/` variant. `run_cogitate()` shells out to the OpenCode
CLI, which connects to Ollama via its OpenAI-compatible endpoint and handles
tool execution internally. OpenCode provider config lives at the user level
(`~/.config/opencode/opencode.json`), per upstream's guidance in
`docs/PROVIDERS.md`.


## Makefile NVM/npx Lookup

> **Merged upstream.** The `NVM_BIN` detection block is now in the upstream
> `Makefile`.

Added `NVM_BIN` detection so `npx` can be found outside interactive shells
(e.g., nvm-managed Node installs).


## WebSocket HTTPS Support

> **Merged upstream** in commit [`27b0745`](https://github.com/solpbc/solstone/commit/27b0745fded2c507b5ccb94df906434c5bc7818d)

**File:** `convey/static/websocket.js`

The WebSocket connection URL was hardcoded to `ws://`, which causes a mixed
content error when the dashboard is served over HTTPS. The browser blocks
insecure WebSocket connections from HTTPS pages. Changed to auto-detect the
protocol (`wss:` for HTTPS, `ws:` for HTTP) based on `location.protocol`.
