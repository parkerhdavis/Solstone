# Fork Divergence

This document catalogs how this fork (`parkerhdavis/Solstone`) currently diverges from upstream (`solpbc/solstone`). It is a **map of the live divergence surface**, not a running changelog — when a fork change lands upstream or is removed, it drops out of the sections below into [Historical context](#historical-context) rather than accreting.

The fork is a dev/testing instance running on a DGX Spark (GB10, aarch64 Linux). Almost all divergence clusters around two things upstream doesn't carry:

1. **Spark adaptations** that make upstream's bundled `local` (llama-server) provider run GPU-offloaded on the Spark's aarch64 + CUDA hardware and serve images.
2. A **local-model benchmarking** feature that estimates inference speed/feasibility per host *before* weights are pulled.

Everything else is small. For the capability/routing view of providers, see [PROVIDER_MATRIX.md](PROVIDER_MATRIX.md).


## ⏱️ Local-model benchmarking & hardware probe

The fork's largest divergence, and a feature upstream has no equivalent for: a speed/feasibility heuristic that tells you what a local model will do on *this* host before you download it, plus a per-segment background-processing estimate.

![AI Providers settings tab with a benchmark card for Qwen 3.5 2B showing tok/s and per-task time estimates split into foreground and background sections.](../.github/model-benchmark-2026-04-25.png)

**Files:** `solstone/apps/benchmark/` (the `sol call benchmark` app), `solstone/think/benchmark/` (estimator, harness, fixtures, and the `models.json` / `reference.json` / `segment.json` / `tasks.json` / `transcribers.json` registries), `solstone/think/hardware.py` (host probe; owns `journal/health/hardware.json`), `solstone/think/providers/__init__.py` (`list_installed_local_models()`), `solstone/apps/settings/routes.py` (`/api/benchmark/*` endpoints + `_fresh_hardware`), `solstone/apps/settings/workspace.html` (providers-tab benchmark cards), and tests `tests/test_benchmark_estimate.py`, `tests/test_benchmark_segment.py`, `tests/test_hardware.py`, `tests/test_settings_benchmark_routes.py`.

- **Per-model tok/s estimate.** `solstone/think/hardware.py` probes CPU / RAM / NVIDIA GPUs and caches the result. The estimator reads `reference.json` (measured tok/s per canonical hardware class) and `models.json` (pre-vetted local models with measured numbers); when the host's exact class isn't measured it interpolates by `fp16_tflops × mem_bandwidth` and reports `confidence="interpolated"`. Direct wall-clock measurements were taken on the DGX Spark.
- **Per-segment background estimate.** Layered on the tok/s heuristic: the wall-clock seconds to fully process one 5-minute segment, decomposed into **Audio** (STT real-time-factor), **Video** (frame describe), **Talents** (per-segment LLM talents), plus a fixed overhead. Scenarios (`solo_active`, `meeting_active`, `idle`) live in `segment.json`.
- **Transcriber RTF surface.** STT cost is real-time-factor, not tok/s, so `transcribers.json` declares each backend's `supported_hardware` / `fallback` / `benchmarkable` axes and the harness has a `--transcriber` mode. Whisper's measured RTF on the Spark is 0.182 (`medium.en` / CUDA / float16).
- **CLI** (`sol call benchmark`): `profile`, `list-models`, `estimate`, `segment`, `scenarios`, `tasks`.
- **Settings UI.** The AI > Providers tab renders per-model tok/s alongside a "Background processing" card (scenario picker, lane breakdown, collapsible per-talent disclosure). The benchmark API endpoints re-probe hardware on each request, so a transient `nvidia-smi` failure (driver warmup at boot, container-startup contention) can't poison the cache and leave the UI showing "tok/s unknown" indefinitely.

![Background processing card showing a 5-minute Solo active segment broken into Audio, Video frames, Talents, and Overhead lanes.](../.github/segment-benchmark-2026-04-25.png)


## 🔧 Spark llama.cpp local-bundle adaptations

Upstream ships a bundled `local` (llama-server, GGUF over an OpenAI endpoint) provider, but no `aarch64-unknown-linux-gnu` binary and no image intake. The fork makes that provider production-ready on the Spark and extends it to vision.

**Files:** `solstone/think/providers/local_install.py` (CUDA-pinned aarch64 artifact + `projector_path`), `solstone/think/providers/local.py` (Nemotron Omni spec + `_translate_content_blocks` + benchmark interface), `solstone/think/providers/local_server.py` (`--mmproj` at launch), `solstone/think/models.py`, `Makefile` (aarch64 extras), and tests `tests/test_local.py`, `tests/test_local_install.py`.

- **aarch64 CUDA pin.** `LLAMA_SERVER_PINS` gains an `aarch64-unknown-linux-gnu` entry (fork-hosted CUDA build: llama.cpp `b9291`, CUDA 13.0, sm_121 / GB10) so the bundle runs GPU-offloaded on the Spark instead of being unavailable there. The `Makefile` skips the parakeet-onnx-cuda extras on any arm64 host (no arm64 wheels for its `nvidia-*` deps), syncing only `pdf` + `whisper`; x86_64 Linux keeps the full set.
- **Image input.** A fork-only `local/nemotron-3-nano-omni` (Q8_0 + mmproj) model spec, `supports_vision()` gating, and `_translate_content_blocks` mapping `ImageBlock → image_url` data URIs. `local_server` passes `--mmproj` for vision-capable models and `local_install` downloads + sha256-verifies the projector GGUF alongside the main weights. Image requests against a text-only model raise `unsupported_capability`.
- **Audio stays on Whisper.** A spike confirmed llama-server rejects Nemotron audio input (HTTP 500), so there is no audio-in LLM on the bundle — audio runs through the Whisper STT pipeline. This is the fork's one local-omni gap.


## 🧩 Multimodal content-block message shape

**Files:** `solstone/think/providers/shared.py`.

The fork added `TextBlock` / `ImageBlock` / `AudioBlock` TypedDicts (plus a `BenchmarkResult` type) so providers and the benchmark harness can pass structured multimodal content instead of a string-only `messages` payload. Still load-bearing after the vLLM removal: consumed by `solstone/think/benchmark/harness.py` (audio/image benchmark tasks) and `solstone/think/providers/local.py` (image intake).


## ⚠️ Transcription-compatibility UI + tab attention

**Files:** `solstone/apps/settings/workspace.html`, `solstone/apps/settings/routes.py`, `tests/baselines/api/settings/transcribe.json`.

When the configured transcribe backend's `supported_hardware` list doesn't include the host's hardware class, the settings transcription tab shows a callout naming the host (e.g. "NVIDIA DGX Spark (GB10)") and suggesting a compatible alternative — preferring Whisper as the universal floor. The same condition flips a reusable **tab-attention** dot: a CSS `::after` indicator keyed off `data-attention="true"` that any settings tab opts into via `setTabAttention('<section>', true|false)`. `/app/settings/api/transcribe` carries the host class + each backend's `supported_hardware` so the client evaluates compatibility without a second round-trip.

![Settings transcription tab with an attention dot in the side-nav and a callout warning that Parakeet does not list NVIDIA DGX Spark (GB10) as supported, suggesting Whisper instead.](../.github/tab-attention-2026-04-25.png)


## 📚 Reference docs

**Files:** `docs/PROVIDER_MATRIX.md` (fork-only), `docs/PROVIDERS.md` (fork-added "Benchmark Surface" section).

`PROVIDER_MATRIX.md` is a fork-only capability/routing reference for which provider serves which surface on the Spark. `PROVIDERS.md` gains a section documenting the `sol call benchmark` surface and the harness workflow for seeding measurements on new hardware.


## 🩹 Minor fork patches

Small carried patches, all upstream candidates:

- `solstone/think/services/cli.py` — match argparse invalid-choice errors on the service name itself rather than version-specific quoting (`choose from scout` vs `choose from 'scout'`), so the `unknown_service` exit code is stable across Python versions.
- `solstone/think/doctor.py`, `solstone/think/models.py` — residual comment/docstring deltas left after the vLLM-checks removal; near-trivial.


## 🧪 Deployment stance (not code divergence)

The fork uses [solpbc/field_journal](https://github.com/solpbc/field_journal) — a public-domain media corpus — as its journal content, making this a testing/development instance rather than a personal-capture one. The `setup_field_journal.sh` tooling itself converged with upstream (PR #44); *using* it is a deployment choice, not a code difference.


---

# Historical context

Things first built on this fork that upstream has since implemented (in one form or another), or that the fork has since removed. Kept for context, not as active divergence.

**vLLM provider — removed (Phase D, 2026-05-25).** For a stretch the fork ran a fork-only vLLM provider (`solstone/think/providers/vllm.py`, the `sol call vllm` app, doctor advisory checks, a multi-server `providers.vllm.servers` config schema, and server-per-model Docker lifecycle under the supervisor) to get **audio-in and aarch64 multimodal local inference** that upstream's bundled `local` couldn't serve at all. Once the `local` bundle covered production text + image on the Spark and Whisper covered audio, vLLM's remaining edges (NVFP4 throughput, speculative video) didn't justify what had become the single largest source of fork↔upstream merge friction, so it was removed. Git history preserves it; an archival writeup lives in the Obsidian `vLLM Implementation Archive`. The multimodal content-block shape (above) was introduced for it and outlived it.

**Ollama provider — converged upstream, then superseded.** The fork's original `ollama` provider — its *first* major divergence, a local `generate` backend that removed the hard cloud-API-key dependency — was merged upstream. Upstream then replaced it wholesale with the bundled `local` (llama-server) + `mlx` providers (PR #44, 2026-05-25), and the fork followed: dropped `ollama`, adopted `local`. This is the reason local inference is no longer fork-divergent *in shape* — only in the Spark-specific adaptations above. It's also why this doc shrank: a large early chunk of fork divergence is now upstream's own implementation.

**Other converged-upstream changes:** the `setup_field_journal.sh` + `docs/FIELD_JOURNAL.md` test-content tooling (upstream independently added an equivalent in PR #44, now canonical); the `Makefile` `NVM_BIN` detection block (so `npx` resolves under nvm-managed Node outside interactive shells); and the WebSocket `ws://` → `wss://` auto-detect fix for HTTPS dashboards (`solstone/convey/static/websocket.js`, upstream commit `27b0745`).
