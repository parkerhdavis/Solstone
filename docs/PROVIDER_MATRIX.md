# Provider & Model Capability Matrix

A reference for *which provider does what* in Solstone's pipelines, and *what each registered provider/model can be used for*. Two tables, opposite axes:

- **Routing matrix** — for every layer/task in Solstone, which providers handle it today and what happens if any one provider is unavailable.
- **Capability matrix** — for every registered provider/model, which roles it can fill and where the caveats are.

This is a *capability* doc, not an implementation guide. For the developer-facing "how to add a new provider" walkthrough, see [PROVIDERS.md](PROVIDERS.md). For the deeper architectural story — including the spike-then-decide arc that landed vLLM as a peer provider — see the Obsidian notes under `40-69 Projects/43 sol pbc/solstone/20-29 Tech/Local Models & Benchmarking` and `90-99 Agents/Transient/vLLM vs Ollama Comparison`.

The matrix reflects the system as of the vLLM peer-provider promotion (2026-04-30). Re-check when adding new providers, models, or modalities.


# Why two tables

The same provider information answers two different questions, and they want different shapes:

- *"For this task, which providers can serve it?"* — the **routing** view. Useful when adding a new talent or wiring a journal config: you want to see all the candidates for a given role and pick one.
- *"For this provider/model, what can I use it for?"* — the **capability** view. Useful when evaluating a new model, debugging a routing decision, or understanding why some features can't fall back gracefully when a particular provider is unavailable.

Each table omits information the other surfaces. Read both for the full picture.


# Routing matrix: per layer/task → provider

For each layer or task in Solstone's pipelines, this shows the providers wired today, the routing pattern (specialist vs tier-routed vs vLLM-only), and the graceful-degradation behavior if vLLM specifically is unavailable.

**Patterns:**

- **Specialist** — one provider, by design; not pluggable per-call (e.g., `wespeaker` is the speaker-embedding implementation, period).
- **Journal-pluggable** — journal config picks the provider per surface or per context; tier system handles fallback. Default routing in `think/models.py::TYPE_DEFAULTS` and `PROVIDER_DEFAULTS`.
- **vLLM-only** — feature requires vLLM; absence of vLLM means the feature is unavailable (degrades gracefully — feature missing, rest of system intact).

| Layer / Task | Provider(s) | Pattern | If vLLM down? |
|--------------|-------------|---------|---------------|
| Audio transcription (live capture) | `faster-whisper` (in-process) | specialist | unaffected |
| Audio transcription (import: Plaud / Rev.ai) | configured per journal | specialist | unaffected |
| Speaker embedding | `wespeaker` (in-process) | specialist | unaffected |
| Speaker attribution L1–L3 (acoustic) | in-process pipeline | specialist | unaffected |
| Speaker attribution L4 (LLM fallback) | Generate (tier-routed) | journal-pluggable | unaffected — uses configured Generate provider |
| Screen frame categorize | Generate w/ vision (tier-routed) | journal-pluggable | unaffected |
| Screen frame extract (selected frames only) | Generate w/ vision (tier-routed) | journal-pluggable | unaffected |
| Segment synthesis (sense, entities, todos, summary) | Generate (tier-routed) | journal-pluggable | unaffected |
| Daily insights, muse generators | Generate (tier-routed) | journal-pluggable | unaffected |
| Voice-brain (text response) | Generate (tier-routed) | journal-pluggable | unaffected |
| Cogitate (chat agent, daily briefing, agent talents) | Cogitate (tier-routed via OpenCode); Ollama default | journal-pluggable | unaffected — Ollama keeps serving; vLLM via OpenCode is mechanically verified but not yet wired into `vllm.py::run_cogitate` |
| OCR / PDF text extraction | Tesseract / pypdf / PaddleOCR (in-process) | specialist | unaffected |
| Embeddings (search & similarity) | provider-specific or `nomic-embed-text` | specialist or journal-pluggable | unaffected |
| Multimodal-augmented meeting summary (planned B/C feature) | `nemotron-omni` via vLLM | vLLM-only | feature unavailable; existing segment record intact |
| Voice-brain audio-tone interpretation (planned B/C) | `nemotron-omni` via vLLM | vLLM-only (opt-in) | feature unavailable; voice-brain falls back to text-only |
| Cross-modal search (planned B/C) | `nemotron-omni` via vLLM | vLLM-only (opt-in) | feature unavailable; standard FTS search keeps working |

**Read:** Everything currently load-bearing is journal-pluggable or specialist. vLLM only appears as a critical path for new additive capabilities that didn't exist before. If vLLM is down, Solstone degrades feature-by-feature, never wholesale.


# Capability matrix: per provider/model → roles

For each registered provider and model, what can it be used for. Marker key:

- ✓ — capable + currently wired in Solstone's provider abstraction
- ◐ — capable but with caveats (not wired, not yet verified, partial support, or notable performance considerations)
- ✗ — not capable or not applicable

## Cloud providers (tier-routed, require API key + outbound network)

| Provider | Text Gen | Cogitate | Vision in | Audio in | Embeddings | Notes |
|----------|:--------:|:--------:|:---------:|:--------:|:----------:|-------|
| Google Gemini (PRO / FLASH / LITE) | ✓ | ✓ (Gemini agent CLI) | ✓ (PIL Image native) | ✗ | ✓ (separate model) | Default for `generate` and `cogitate` per `TYPE_DEFAULTS`. Vision is best-in-class for the existing describe pipeline. |
| OpenAI (GPT-5.4 + variants) | ✓ | ✓ (Codex CLI) | ✓ | ◐ (gpt-4o-audio family exists; not currently wired) | ✓ | Reasoning-effort suffix support (`-high`, `-low`, `-xhigh`). |
| Anthropic (Claude Opus / Sonnet / Haiku 4.x) | ✓ | ✓ (`claude` CLI) | ✓ | ✗ | ✗ | Default `backup` provider for both `generate` and `cogitate`. No native embeddings surface. |

## Local — Ollama (daemon, lazy-load, GGUF Q4_K_M defaults)

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `ollama-local/qwen3.5:35b-a3b-bf16` (PRO) | ✓ | ✓ | ✗ | ✗ | `OLLAMA_PRO`. Heavy cogitate workloads (chat agent, daily briefing). 70 GB pinned when loaded. |
| `ollama-local/qwen3.5:9b` (FLASH) | ✓ | ✓ | ✗ | ✗ | `OLLAMA_FLASH`. Workhorse for most talents. |
| `ollama-local/qwen3.5:2b` (LITE) | ✓ | ✗ | ✗ | ✗ | `OLLAMA_LITE`. `models.json` notes "skip for cogitate" — too small for reliable tool-calling. |
| `ollama-local/qwen2.5vl:7b` | ◐ (vision-anchored only) | ✗ | ✓ | ✗ | Vision tier for the screen-frame describe pipeline. |
| `ollama-local/qwen2.5vl:32b` / `:72b` | ◐ | ✗ | ✓ | ✗ | Larger vision models. 72B is severely memory-bandwidth-bound on Spark per registry notes — 32B is ~2.3× faster. |

## Local — vLLM (server-per-model, AWQ / bf16 / NVFP4 quants)

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `vllm-local/qwen3.5:35b-a3b` (bf16) | ✓ | ◐ (mechanically verified Phase 5; not yet wired into `vllm.py::run_cogitate`) | ✗ | ✗ | `VLLM_PRO`. Same role as Ollama PRO. Use Ollama unless explicitly wanted. |
| `vllm-local/qwen3.5:9b-awq` | ✓ | ◐ (same caveat) | ✗ | ✗ | `VLLM_FLASH`. ~5% slower or comparable to Ollama FLASH. **Caveat:** thinking-enabled accuracy can degrade up to 33% on reasoning tasks per kaitchup benchmarks (model thinks more, hits max-output). |
| `vllm-local/qwen3.5:2b-awq` | ✓ | ✗ | ✗ | ✗ | `VLLM_LITE`. **Notably faster than Ollama equivalent on Spark** (+13% synth, −36% chat_reply). |
| `vllm-local/qwen2.5vl:7b-awq` | ◐ | ✗ | ✓ | ✗ | Vision counterpart of Ollama qwen2.5vl:7b. ~5–15% faster on Spark at apples-to-apples quant. |
| `vllm-local/nemotron-omni` (NVFP4) | ✓ | ◐ (untested with multimodal + tools combined) | ✓ | **✓** | **The only model with audio-input capability.** Also handles text + image in the same call. Video claimed by NVIDIA, untested by us. ~21 GB pinned. The reason vLLM is in the system at all. |

## In-process specialists (not LLM providers; no tier system)

| Specialist | Role | Notes |
|------------|------|-------|
| `faster-whisper` (large-v3 / medium / etc.) | Audio transcription, live capture | Default `transcribe.backend`. Produces sentence-level `audio.jsonl` + sentence embeddings used by speaker-attribution flywheel. |
| Parakeet TDT | Audio transcription, alternative | In-process (CoreML / linux-x86_64 ONNX). Deferred on aarch64 Spark per [Transcription Backend Architecture] (Obsidian). |
| Rev.ai | Audio transcription, cloud import only | Used for imported audio paths only, not live capture. |
| `wespeaker` (resnet34) | Per-sentence speaker voiceprint embeddings (256-dim) | The substrate for the voiceprint-learning flywheel. Purpose-built for speaker similarity; nemotron's audio encoder is not a substitute. |
| Resemblyzer | Speaker embeddings, older path | Some legacy paths use this; the wespeaker pipeline is the modern one. |
| pyannote-audio | VAD / diarization | Sentence-boundary segmentation. |
| Tesseract / PaddleOCR / pypdf / pdf2image | OCR + PDF text extraction | Document pipeline. nemotron-omni claims OCR capability but specialist tools are faster on the document workload. |
| OpenCV (`cv2`) | Video frame extraction | Preprocessing for the screen describe pipeline. Not AI. |


# Cross-cutting observations

Three load-bearing facts to internalize when reasoning about which provider belongs where:

1. **Audio input is genuinely vLLM-only.** Across every other provider — cloud or local — audio is either text-via-Whisper (specialist path, no fusion with other modalities) or not supported. `nemotron-omni` is the only ✓ for audio-in. Any feature that needs *audio understanding* (tone, non-speech sounds, acoustic context) goes through vLLM by necessity. This is the unique-value reason vLLM exists in Solstone.

2. **Cogitate today still routes through Ollama** even though vLLM cogitate is mechanically verified. The ◐ marks on vLLM cogitate reflect that OpenCode-against-vLLM works (Phase 5 validation), but `vllm.py::run_cogitate` isn't wired yet. Small piece of code work whenever someone wants to land it as a real production-routable surface.

3. **The cloud↔local fallback story is symmetric for text/vision/cogitate** but **asymmetric for audio.** Text/vision/cogitate failures degrade gracefully — if cloud is down, Ollama or vLLM cover it; if vLLM is down, cloud or Ollama cover it. But if vLLM is down and a feature relies on `nemotron-omni`'s audio understanding, that feature has no fallback — the closest substitute is "Whisper transcribes, Generate-tier LLM interprets the text," which loses the audio-direct capability the feature was using.

The third point is the architectural constraint that shapes where vLLM-only features should live: **opt-in additive capabilities** (the multimodal augmentation goes missing when vLLM is down, the rest of the segment record is intact) rather than **critical-path replacements** (where vLLM down = synthesis stalls = downstream features stall).

This is also why segment synthesis stays on Generate-tier-routed providers, not on vLLM-served `nemotron-omni`. Segment processing is the substrate-producing pipeline that downstream features (speaker attribution flywheel, search index, timeline UI) depend on; it has to be available whenever Solstone is running.


# Related

- [PROVIDERS.md](PROVIDERS.md) — implementation guide for adding a new provider.
- [THINK.md](THINK.md) — module overview for `think/`.
- `think/providers/__init__.py` — canonical `PROVIDER_REGISTRY` and `PROVIDER_METADATA`.
- `think/models.py` — tier constants, `PROVIDER_DEFAULTS`, `TYPE_DEFAULTS`, talent-context resolution.
- `think/benchmark/models.json` — pre-vetted local models with measured tok/s and per-task wall-clock.
- Obsidian: `20-29 Tech/Local Models & Benchmarking` — narrative companion covering tier mapping, journal-config patterns, and the benchmarking surface.
- Obsidian: `20-29 Tech/Solstone AI Modalities Overview` — modality-by-modality inventory of Solstone's AI usage.
- Obsidian: `90-99 Agents/Transient/vLLM Provider & Multimodal Benchmark Plan` — the full plan that produced the vLLM peer-provider integration.
- Obsidian: `90-99 Agents/Transient/vLLM vs Ollama Comparison` — the replace-or-coexist analysis behind the current architecture.
