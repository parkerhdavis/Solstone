# Fork Changelog

This document tracks significant changes made on this fork of Solstone.

## 🪡 vLLM (Local) Provider — Multimodal Local Inference

![AI Providers settings tab showing Generate routed to vLLM (Local) at the Fast/Lightweight tier with Ollama as backup. Benchmark card reads "Qwen 3.5 2B (AWQ-Int4) — vLLM-served (LITE-tier promotable)" with a "not installed" badge and "98 tok/s measured" — the same per-tier details panel that was previously Ollama-only.](../.github/vllm-provider-add-2026-04-30.png)

**Files:** `solstone/think/providers/vllm.py`, `solstone/apps/vllm/` (CLI app + tests), `solstone/think/providers/__init__.py`, `solstone/think/providers/ollama.py` (content-block intake), `solstone/think/providers/shared.py`, `solstone/think/benchmark/harness.py`, `solstone/think/benchmark/models.json` (vllm-local entries), `solstone/think/models.py` (`VLLM_PRO/FLASH/LITE`), `solstone/think/supervisor.py` (`start_vllm_servers`), `solstone/think/doctor.py` (vLLM advisory checks), `solstone/apps/benchmark/call.py` (provider-aware `_list_installed_models`), `solstone/apps/settings/workspace.html` + `solstone/apps/settings/routes.py` (UI parity)

Added vLLM as a peer local provider alongside Ollama, scoped to the multimodal-omni capabilities Ollama can't serve at all — most importantly, audio input. Ollama's GGUF-only stack doesn't accept audio content blocks; vLLM's HuggingFace path does. The integration was sequenced as Phases 0-4 (spike → provider → benchmark → tier promotion → server lifecycle) and is now operationally durable on the DGX Spark.

**Provider module (`solstone/think/providers/vllm.py`).** OpenAI-compatible client speaking to a local vLLM server over `/v1/chat/completions`. Consumes text + image + audio content blocks. Multi-server config schema in `journal.json → providers.vllm.servers` lets a single Solstone instance route to multiple vLLM processes — necessary because vLLM pins one model per process and has no hot-swap. `run_generate`, `run_agenerate`, `bench_run_once`, `bench_ensure_installed`, `validate_key`, `list_models` are implemented; `run_cogitate` is deferred to Phase 5.

**Tier promotion (`solstone/think/models.py`).** `VLLM_PRO` (qwen3.5:35b-a3b bf16), `VLLM_FLASH` (qwen3.5:9b AWQ-Int4), `VLLM_LITE` (qwen3.5:2b AWQ-Int4) constants alongside the Ollama tier defaults. Journal context can now route any tier to vLLM-served models by config without the harness needing per-tier wiring.

**`sol call vllm` CLI (`solstone/apps/vllm/`).** Wraps the `docker run` invocation we'd otherwise hand-launch from a spike workspace. `serve` spawns `docker run --rm` with the right flags (`--gpus all`, `--ipc=host`, `/root/.cache/huggingface` mount for the model cache, `/root/.cache/vllm` mount so torch.compile artifacts persist across restarts) and translates SIGINT/SIGTERM into `docker stop` with a 30s grace window. `list` and `status` enumerate configured servers and ping each `/v1/models`. Server-per-model; `--name <name>` selects which entry to operate on.

**Supervisor integration (`solstone/think/supervisor.py`).** `start_vllm_servers()` reads `providers.vllm.servers` from journal config and registers one `vllm-<name>` managed process per entry, each running `sol call vllm serve --name <name> -v` under the standard restart policy. No-op when no servers are configured. Opt-out via `--no-vllm`. vLLM containers now get the same lifecycle treatment as cortex/link/sense/convey — restart-on-crash, callosum events, unified logs.

**Doctor advisory checks (`solstone/think/doctor_vllm.py`).** Three new linux-only advisory checks: `vllm_docker_available`, `vllm_nvidia_smi`, `vllm_servers_reachable`. Lives in a fork-only module; `solstone/think/doctor.py` imports `VLLM_CHECKS` and extends its registry. Tests in `tests/test_doctor_vllm.py`. All stdlib-only (urllib for HTTP) so doctor stays runnable on a fresh clone before `uv sync` — `scripts/doctor.py` is now a stdlib-only bootstrap shim that delegates to `solstone.think.doctor.main`, so the same checks run via `python3 scripts/doctor.py` pre-install or `sol doctor` once the venv exists. Skips cleanly when there's no journal config or no `providers.vllm.servers` section.

**Benchmark integration.** vLLM models registered in `solstone/think/benchmark/models.json` under the `vllm-local/` prefix with measured numbers on `dgx-spark` (Nemotron-3-Nano-Omni, Qwen3.5 35B/9B/2B, Qwen2.5-VL 7B). Harness genericized via `_resolve_provider(model)` so dispatch goes through the provider's `bench_run_once` interface — vLLM and Ollama use the same harness path. The shared `solstone.think.providers.list_installed_local_models()` queries both Ollama (`/api/tags`) and vLLM (`/v1/models`); the settings + benchmark CLI both call it.

**Settings UI parity.** The providers tab's per-tier benchmark details panel renders for vllm selection the same way it does for ollama — same tier-anchor lookup, same per-task seconds, same fits-in-vram check. Rows filtered by `model_id` prefix; the recommended-models list correctly skips vLLM rows (vLLM "install" means editing journal config, not `ollama pull`).

**Operational docs (Obsidian, not in repo).** `Spark vLLM Host Prerequisites` and `Spark Memory Budget for vLLM Coexistence` capture the host-setup steps and the unified-memory budget math respectively. Headline finding from the budget exercise: vLLM's footprint is dominated by `--gpu-memory-utilization` × 121.7 GiB unified memory, not weight size, so two-vLLM coexistence requires dropping `gpu-mem-util` to ~0.4 from the default 0.8.


## 🧩 Multimodal Content-Block Message Shape

**Files:** `solstone/think/providers/shared.py`, `solstone/think/providers/ollama.py`, `solstone/think/providers/vllm.py`, `solstone/think/benchmark/harness.py`, `solstone/think/benchmark/tasks.json`, `solstone/think/benchmark/fixtures/audio_30s.wav`, `solstone/think/benchmark/fixtures/README.md`, `solstone/think/benchmark/estimate.py`

Replaced the harness's string-only `messages` payload with content-block lists (`TextBlock`, `ImageBlock`, `AudioBlock` TypedDicts in `solstone/think/providers/shared.py`). Providers consume the blocks in their native shapes — Ollama maps `ImageBlock` to its `images: [...]` field and raises `NotImplementedError` on `AudioBlock`; vLLM emits OpenAI-style `image_url` and `input_audio` blocks. The harness auto-detects audio mode from the task spec and base64-encodes audio fixtures via `_load_audio_b64`. `audio_transcribe` and `audio_summarize` task entries reference `solstone/think/benchmark/fixtures/audio_30s.wav` — the first audio benchmark fixture, a public-domain LibriVox clip (Tom Sawyer chapter 1) with provenance documented in `fixtures/README.md`. The estimator's `_task_applies_to_model` now gates audio tasks on the model's `audio` capability.


## 📊 Provider Matrix Reference

**File:** `docs/PROVIDER_MATRIX.md`

New top-level reference doc capturing the per-provider × per-capability routing matrix as the local-provider count went from one (Ollama) to two (Ollama + vLLM). Two tables: (1) per-layer/task → which provider can serve it, (2) per-provider → which roles it covers. Documents the multi-server `providers.vllm.servers` config schema, the cyankiwi AWQ trust profile for FLASH/LITE quants, and the tool-call parser per model family (`qwen3_xml` for Qwen3.5, `hermes` for Nemotron-3-Nano-Omni). Useful when reasoning about which provider serves which surface, especially as cogitate / generate / vision / audio start routing through different providers per journal context.


## ⏱️ Local Model Benchmarking

![AI Providers settings tab with a benchmark card for Qwen 3.5 2B showing 87 tok/s and per-task time estimates split into foreground (Chat reply, Voice reply, Search query, Agent turn) and background (Entity extraction, Todo extraction, Meeting summary, Activity clustering, Daily insights, Segment sense, Speaker attribution, Screen record, Awareness tender, Pulse) sections.](../.github/model-benchmark-2026-04-25.png)

**Files:** `solstone/apps/benchmark/`, `solstone/think/benchmark/`, `tests/test_benchmark_estimate.py`

Added a `benchmark` app and supporting `solstone/think/benchmark/` module that estimates expected output tok/s for pre-vetted Ollama models on the user's hardware without requiring the models to be pulled. A reference table of measured tok/s per canonical hardware class (see `solstone/think/benchmark/reference.json`) is interpolated by FP16 throughput × memory bandwidth when the exact hardware isn't listed. The registry (`models.json`) covers text and vision models across tiers, with direct wall-clock measurements taken on DGX Spark used to ground the task-time heuristics.

The `sol call benchmark` CLI exposes `profile` (probe + cache host hardware), `list-models` (pre-vetted + installed models with tok/s and task-time estimates), `estimate <model-id>` (single-model estimate, optionally `--task <task_id>` for a wall-clock estimate against a reference workload), and `tasks` (show the reference-task catalog). A harness (`solstone/think/benchmark/harness.py`) runs the fixture-backed reference tasks (`fixtures/*.txt`) to produce new measurements that feed back into the registry. The settings UI surfaces per-model tok/s alongside the cogitate details panel, with generic tier labels and a recommended-models section for quick orientation. The same details panel renders for both Ollama and vLLM provider selections (see vLLM section above).

The settings benchmark API endpoints always re-probe the host's hardware on each request rather than trusting the cached probe at `journal/health/hardware.json`. This self-heals the case where the cache was poisoned by a transient `nvidia-smi` failure (driver warmup at boot, container-startup contention) — without it, a single bad probe could leave the providers UI showing "tok/s unknown" and "may not fit" indefinitely.


## ⌛ Segment-Time Background Processing Benchmarks

![Background processing card showing a 2m 31s estimate for a 5-minute Solo active segment, broken down into Audio, Video frames, Talents, and Overhead lanes.](../.github/segment-benchmark-2026-04-25.png)

**Files:** `solstone/apps/benchmark/`, `solstone/think/benchmark/segment.json`,
`solstone/think/benchmark/estimate.py`, `solstone/apps/settings/routes.py`,
`solstone/apps/settings/workspace.html`, `tests/test_benchmark_segment.py`,
`tests/test_settings_benchmark_routes.py`

Layered a **semantic benchmark** on top of the existing per-task
tok/s heuristics: the estimated wall-clock seconds it takes the host
to fully process one 5-minute audio + screen-recording segment.
Decomposed into three lanes plus a small fixed overhead:

- **Audio** — local STT cost (real-time-factor × audio length).
- **Video** — `screen_frame` × qualified-frames-per-segment.
- **Talents** — per-segment LLM talents (`segment_sense`,
  `entity_extraction`, conditionally `screen_record` /
  `speaker_attribution_llm`, plus housekeeping `awareness_tender` /
  `pulse`).

Three named scenarios live in `solstone/think/benchmark/segment.json`:
`solo_active`, `meeting_active`, `idle`. Each scenario fixes a
qualified-frame count, a talent recipe, and a fixed overhead.

`estimate_segment_time_s(tier_models, hardware_class, scenario,
transcriber)` composes the lanes via the existing tok/s estimator
plus a transcriber RTF lookup and returns a `SegmentEstimate` with
the per-lane breakdown, per-talent rows, weakest-leg confidence, and
notes.

**CLI surface (`sol call benchmark`):**

- `segment` — total + lane breakdown + per-talent rows for a
  scenario.
- `scenarios` — list scenarios from `segment.json`.
- `list-models` — now leads with a `5-MIN SEGMENT` column, per row,
  computed by attributing this row's model to whichever tier roles
  it can serve and using the smallest registry model per tier as the
  comparison baseline. Old per-task `TOK/S` + `TASK TIMES` columns
  move behind `--detailed`.

**Settings UI:** new "Background processing" card on the AI >
Providers tab, fed by `GET /app/settings/api/benchmark/scenarios`
and `GET /app/settings/api/benchmark/segment`. Scenario picker,
headline total + confidence chip, lane breakdown rows, collapsible
per-talent disclosure, notes footer.

Five fork-only fixture files under `solstone/think/benchmark/fixtures/` for
the new per-segment talents (`segment_sense`,
`speaker_attribution_llm`, `screen_record`, `awareness_tender`,
`pulse`), with measured token counts in `tasks.json` (qwen
tokenizer, consistent across model sizes).


## ⚠️ Transcription Compatibility Warning + Reusable Tab Attention

![Settings transcription tab with an attention dot in the side-nav and a callout warning that Parakeet does not list NVIDIA DGX Spark (GB10) as supported, suggesting Whisper instead.](../.github/tab-attention-2026-04-25.png)

**Files:** `solstone/apps/settings/workspace.html`, `solstone/apps/settings/routes.py`,
`tests/test_settings_benchmark_routes.py`

Quality-of-life pass on the settings UI:

- **Reusable tab attention indicator.** A small dot appears in the
  side-nav next to any tab that has unresolved problems. Tabs opt in
  by calling `setTabAttention('<section>', true|false)` from their
  data loader. The indicator is just a CSS `::after` pseudo-element
  keyed off `data-attention="true"` — any future tab can adopt it
  without new framework.
- **Transcription compatibility check.** When the configured
  transcribe backend's `supported_hardware` list (sourced from
  `transcribers.json`) doesn't include the host's hardware class,
  the transcription tab shows a callout naming the host
  (e.g. "NVIDIA DGX Spark (GB10)") and suggesting a compatible
  alternative — preferring whisper since it's the universal floor.
  The same condition flips the tab's attention dot on. This catches
  the parakeet-on-aarch64 case the host's running Solstone wouldn't
  otherwise surface in the UI.

`/app/settings/api/transcribe` was extended to carry the host's
hardware class + each backend's `supported_hardware` list so the
client can do the compat evaluation without a second round-trip.


## 🎤 Transcriber RTF Benchmarking + `transcribers.json`

**Files:** `solstone/think/benchmark/transcribers.json`,
`solstone/think/benchmark/harness.py`, `solstone/think/benchmark/estimate.py`,
`solstone/apps/benchmark/call.py`, `tests/test_benchmark_segment.py`

Added a parallel benchmark surface for STT backends, since
transcription doesn't fit a tok/s model — its cost is real-time
factor (RTF = `wall_seconds / audio_seconds`) on local backends and a
flat per-5min wall-clock heuristic on cloud backends. New
`solstone/think/benchmark/transcribers.json` declares each backend with three
orthogonal axes:

- `supported_hardware` — explicit list of hardware-class keys from
  `reference.json`, or `["*"]` for any. Machine-readable, enforced.
- `fallback` — production runtime fallback eligibility (whisper is
  `true`; parakeet is `false`).
- `benchmarkable` — whether the harness should run RTF capture.
  Cloud backends are not benchmarkable in the RTF sense.

`solstone/think/benchmark/harness.py` gained a `--transcriber <backend>
--audio-fixture <path> --class <hw>` mode. Hard-fails before any
transcription work runs when the chosen transcriber doesn't list the
host's hardware class, or when a cloud backend is asked for RTF.

The audio lane of `estimate_segment_time_s` resolves via these RTFs.
Whisper's RTF on `dgx-spark` is measured (0.182, `medium.en` /
`cuda` / `float16`) — `faster-whisper` auto-detects CUDA via
CTranslate2 on the Spark, so the in-process whisper backend is GPU
accelerated by default rather than CPU-bound.


## 📓 Field Journal Test Content

**Files:** `setup_field_journal.sh`, `docs/FIELD_JOURNAL.md`

This fork uses [solpbc/field_journal](https://github.com/solpbc/field_journal) — a public-domain media corpus — as its journal content, making this instance a dedicated testing and development environment rather than a personal capture one. `setup_field_journal.sh` at the repo root copies days from a local field_journal clone (default `~/Field_Journal/`) into `journal/chronicle/`. Setup lives in a standalone script rather than the `Makefile` so shared files stay convergent with upstream. See `docs/FIELD_JOURNAL.md` for the full workflow.


---

*The following changes originated on this fork and have since been merged upstream.*


## Ollama (Local) Provider

> **Merged upstream.** The `solstone/think/providers/ollama.py` provider, tests, and
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

**File:** `solstone/convey/static/websocket.js`

The WebSocket connection URL was hardcoded to `ws://`, which causes a mixed
content error when the dashboard is served over HTTPS. The browser blocks
insecure WebSocket connections from HTTPS pages. Changed to auto-detect the
protocol (`wss:` for HTTPS, `ws:` for HTTP) based on `location.protocol`.
