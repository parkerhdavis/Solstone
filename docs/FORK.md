# Fork Divergence

This document catalogs how this fork (`parkerhdavis/Solstone`) currently diverges from upstream (`solpbc/solstone-journal`, formerly `solpbc/solstone`). It is a **map of the live divergence surface**, not a running changelog — when a fork change lands upstream or is removed, it drops out of the sections below into [Historical context](#historical-context) rather than accreting.

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

Upstream ships a bundled `local` (llama-server, GGUF over an OpenAI endpoint) provider as a **single fixed model behind a supervisor-owned, always-on daemon** with a connect-only client (upstream commit `22034e83`). Upstream also built the vision rails — `image_url` content translation in `local.py`, optional mmproj threading through `local_install`, and a `--mmproj` launch flag in the supervisor — but ships only a **text-only** `qwen2.5-coder-7b` on them. The fork keeps that exact architecture and makes two changes: it provides the `aarch64-unknown-linux-gnu` CUDA binary upstream lacks, and it sets the one fixed model to a **vision-capable** one so the rails actually carry images.

**Files:** `solstone/think/models.py` (`LOCAL_MODEL` → Qwen3.6-35B-A3B), `solstone/think/providers/local.py` (Qwen3.6-35B-A3B `LOCAL_MODEL_SPECS` entry + the benchmark interface), `solstone/think/providers/local_install.py` (aarch64 CUDA pin + url-override), `solstone/think/benchmark/` (the fork-only benchmark/preset framework + the standalone head-to-head harness), and tests `tests/test_local.py`, `tests/test_local_install.py`.

- **aarch64 CUDA pin.** `LLAMA_SERVER_PINS` gains an `aarch64-unknown-linux-gnu` entry with an explicit `url` override (fork-hosted CUDA build: llama.cpp `b9291`, CUDA 13.0, sm_121 / GB10) so the bundle runs GPU-offloaded on the Spark instead of being unavailable there. This is the single irreplaceable Spark divergence — no upstream prebuilt exists for this platform.
- **Vision-capable single model.** `LOCAL_MODEL = "local/qwen3.6-35b-a3b"` (Q8_0 + `mmproj-BF16`), a 35B-A3B MoE VLM (~3B active), replacing upstream's smaller `qwen3.5-4b` VLM default. Everything downstream is upstream machinery, unchanged: `local_install` downloads + sha256-verifies the projector GGUF alongside the weights, the supervisor passes `--mmproj` at launch because the spec carries one, and `local.py` translates `image_url` image content. The one always-on local daemon therefore serves both text/agentic cogitate **and** image input. Footprint: ~37 GB weights + 0.9 GB mmproj, min-RAM 48 GiB — fine on the 128 GB Spark. (Was Nemotron 3 Nano Omni until the head-to-head below repointed it.)
- **Audio stays on Whisper.** The bundle serves no audio input through llama-server (`b9291`) — Qwen3.6 carries no audio modality, and the prior Nemotron Omni's audio was rejected by llama-server ("audio input is not supported") — so audio runs through the Whisper STT pipeline.

> **Served-model choice — resolved (2026-05-30).** A three-way Q8 head-to-head on the Spark (Nemotron 3 Nano Omni vs. `qwen3.5-4b` vs. `qwen3.6-35b-a3b`, via the fork-only standalone benchmark harness) repointed the served model from Nemotron to **Qwen3.6-35B-A3B**: equal tok/s + footprint (both ~3B-active MoE, ~58–60 tok/s, bandwidth-bound on the GB10), a measured + published agentic/vision quality edge, and the same family as upstream's qwen baseline (so the shared `--jinja` / `enable_thinking=false` chat-template machinery is tuned for it), while shedding Nemotron's unused omni audio/video. The pinned `b9291` build loads Qwen3.6's Qwen2.5-VL mmproj ViT (gate passed — no binary rebuild). Still a single served model (upstream's architecture); a multi-model local layer remains a possible future re-divergence. Details + numbers: Obsidian `90-99 Agents/Transient/2026-05-30 Local Inference & Benchmarking Plan`.


## 🧩 Multimodal content-block message shape

**Files:** `solstone/think/providers/shared.py`.

The fork added `TextBlock` / `ImageBlock` / `AudioBlock` TypedDicts (plus a `BenchmarkResult` type) so the benchmark harness can pass structured multimodal content instead of a string-only `messages` payload. Still fork-only — upstream's `shared.py` carries neither. Consumed by `solstone/think/benchmark/harness.py` (audio/image benchmark tasks) and the `bench_run_once` return type in `solstone/think/providers/local.py`. Note: production image intake in `local.py` now rides on upstream's own `_image` helpers (`encode_image_part` → `image_url`), not these types — the 0.4.4 merge converged the provider-side translation onto upstream (see [Historical context](#historical-context)).


## ⚠️ Transcription-compatibility UI

**Files:** `solstone/apps/settings/workspace.html`, `solstone/apps/settings/routes.py`, `tests/baselines/api/settings/transcribe.json`.

When the configured transcribe backend's `supported_hardware` list doesn't include the host's hardware class, the settings transcription tab shows a callout naming the host (e.g. "NVIDIA DGX Spark (GB10)") and suggesting a compatible alternative — preferring Whisper as the universal floor. It surfaces this via the **tab-attention** dot (`data-attention="true"` + `setTabAttention(...)`). `/app/settings/api/transcribe` carries the host class + each backend's `supported_hardware` so the client evaluates compatibility without a second round-trip.

> The tab-attention dot mechanism itself **converged upstream** in the 0.4.4 merge — upstream independently added a byte-identical `setTabAttention` + `data-attention` CSS. The fork no longer carries its own copy; only the transcription-tab *use* of it (the hardware-compat callout) remains fork divergence.

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

**Local-provider architecture + vision translation — converged upstream (0.4.4 merge, 2026-05-30).** Upstream commit `22034e83` rewrote the local provider into a supervisor-owned, always-on llama-server daemon with a connect-only client and a single fixed model, and added the vision rails (`image_url` content translation, mmproj threading, `--mmproj` launch). The fork had carried its own multi-model registry, lazy `ensure_running` spawn path, `_translate_content_blocks`, and `supports_vision()` gating; all of that was dropped in favor of upstream's implementation. What remains fork-divergent narrowed to the aarch64 CUDA pin and the *choice* of a vision-capable single model (see [Spark llama.cpp local-bundle adaptations](#-spark-llamacpp-local-bundle-adaptations)). The `LocalModelSpec.mmproj_filename` / `mmproj_sha256` fields the fork introduced are now upstream's.

**Tab-attention dot — converged upstream (0.4.4 merge).** Upstream added a byte-identical `setTabAttention` + `data-attention` indicator; the fork dropped its duplicate. Only the transcription-tab use of it remains (see [Transcription-compatibility UI](#-transcription-compatibility-ui)).

**Makefile install extras — converged upstream (0.4.10 merge, 2026-06-03).** The fork carried a platform-conditional `EXTRAS_ARGS` that avoided `--all-extras` on any arm64 host (Darwin arm64 + Linux aarch64) to dodge parakeet-onnx-cuda's arm64-less `nvidia-*` wheels, syncing only `pdf` + `whisper`. Upstream commit `b780d52e` solved the same root cause more generally — a platform-agnostic `--extra all` (= pdf + whisper) plus a dedicated per-host parakeet step gated on `x86_64`, so non-x86_64 Linux (the Spark) skips parakeet on its own. The fork's conditional became redundant; the `Makefile` now matches upstream byte-for-byte.

**Other converged-upstream changes:** the `setup_field_journal.sh` + `docs/FIELD_JOURNAL.md` test-content tooling (upstream independently added an equivalent in PR #44, now canonical); the `Makefile` `NVM_BIN` detection block (so `npx` resolves under nvm-managed Node outside interactive shells); and the WebSocket `ws://` → `wss://` auto-detect fix for HTTPS dashboards (`solstone/convey/static/websocket.js`, upstream commit `27b0745`).
