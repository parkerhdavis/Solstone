# Fork Changelog

This document tracks significant changes made on this fork of Solstone.


## Segment-Time Benchmark — semantic 5-min processing metric

**Files:** `apps/benchmark/`, `think/benchmark/segment.json`,
`think/benchmark/estimate.py`, `apps/settings/routes.py`,
`apps/settings/workspace.html`, `tests/test_benchmark_segment.py`,
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

Three named scenarios live in `think/benchmark/segment.json`:
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

Five fork-only fixture files under `think/benchmark/fixtures/` for
the new per-segment talents (`segment_sense`,
`speaker_attribution_llm`, `screen_record`, `awareness_tender`,
`pulse`), with measured token counts in `tasks.json` (qwen
tokenizer, consistent across model sizes).


## Transcriber RTF Benchmarking + `transcribers.json`

**Files:** `think/benchmark/transcribers.json`,
`think/benchmark/harness.py`, `think/benchmark/estimate.py`,
`apps/benchmark/call.py`, `tests/test_benchmark_segment.py`

Added a parallel benchmark surface for STT backends, since
transcription doesn't fit a tok/s model — its cost is real-time
factor (RTF = `wall_seconds / audio_seconds`) on local backends and a
flat per-5min wall-clock heuristic on cloud backends. New
`think/benchmark/transcribers.json` declares each backend with three
orthogonal axes:

- `supported_hardware` — explicit list of hardware-class keys from
  `reference.json`, or `["*"]` for any. Machine-readable, enforced.
- `fallback` — production runtime fallback eligibility (whisper is
  `true`; parakeet is `false`).
- `benchmarkable` — whether the harness should run RTF capture.
  Cloud backends are not benchmarkable in the RTF sense.

`think/benchmark/harness.py` gained a `--transcriber <backend>
--audio-fixture <path> --class <hw>` mode. Hard-fails before any
transcription work runs when the chosen transcriber doesn't list the
host's hardware class, or when a cloud backend is asked for RTF.

The audio lane of `estimate_segment_time_s` resolves via these RTFs.
Whisper's RTF on `dgx-spark` is measured (0.182, `medium.en` /
`cuda` / `float16`) — `faster-whisper` auto-detects CUDA via
CTranslate2 on the Spark, so the in-process whisper backend is GPU
accelerated by default rather than CPU-bound.


## Transcription Hardware-Compat Warning + Reusable Tab Attention

**Files:** `apps/settings/workspace.html`, `apps/settings/routes.py`,
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


## Local-Model Benchmark Heuristic

**Files:** `apps/benchmark/`, `think/benchmark/`, `tests/test_benchmark_estimate.py`

Added a `benchmark` app and supporting `think/benchmark/` module that estimates expected output tok/s for pre-vetted Ollama models on the user's hardware without requiring the models to be pulled. A reference table of measured tok/s per canonical hardware class (see `think/benchmark/reference.json`) is interpolated by FP16 throughput × memory bandwidth when the exact hardware isn't listed. The registry (`models.json`) covers text and vision models across tiers, with direct wall-clock measurements taken on DGX Spark used to ground the task-time heuristics.

The `sol call benchmark` CLI exposes `profile` (probe + cache host hardware), `list-models` (pre-vetted + installed models with tok/s and task-time estimates), `estimate <model-id>` (single-model estimate, optionally `--task <task_id>` for a wall-clock estimate against a reference workload), and `tasks` (show the reference-task catalog). A harness (`think/benchmark/harness.py`) runs the fixture-backed reference tasks (`fixtures/*.txt`) to produce new measurements that feed back into the registry. The settings UI surfaces per-model tok/s alongside the cogitate details panel, with generic tier labels and a recommended-models section for quick orientation.


## Field Journal Test Content

**Files:** `setup_field_journal.sh`, `docs/FIELD_JOURNAL.md`

This fork uses [solpbc/field_journal](https://github.com/solpbc/field_journal) — a public-domain media corpus — as its journal content, making this instance a dedicated testing and development environment rather than a personal capture one. `setup_field_journal.sh` at the repo root copies days from a local field_journal clone (default `~/Field_Journal/`) into `journal/chronicle/`. Setup lives in a standalone script rather than the `Makefile` so shared files stay convergent with upstream. See `docs/FIELD_JOURNAL.md` for the full workflow.


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
