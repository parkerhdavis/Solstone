# Provider & Model Capability Matrix

A reference for *which provider does what* in Solstone's pipelines, and *what each registered provider/model can be used for*. Two tables, opposite axes:

- **Routing matrix** — for every layer/task in Solstone, which providers handle it today and what happens if any one provider is unavailable.
- **Capability matrix** — for every registered provider/model, which roles it can fill and where the caveats are.

This is a *capability* doc, not an implementation guide. For the developer-facing "how to add a new provider" walkthrough, see [PROVIDERS.md](PROVIDERS.md).

The matrix reflects the system **after the llama.cpp local-bundle migration (Phases A–D, 2026-05-25)**: the PR #44 upstream merge dropped `ollama` for the bundled `local` (llama-server) provider + an `mlx` provider + the `openhands` cogitate façade, and this fork then made `local` the Spark's production backend and **removed the fork-only vLLM provider** entirely. Re-check when adding new providers, models, or modalities.

> **Migration complete.** The fork added an `aarch64-unknown-linux-gnu` CUDA pin so the `local` bundle runs GPU-offloaded on the DGX Spark (GB10), extended it to **image** (Nemotron 3 Nano Omni via `--mmproj`), and repointed production `generate`/`cogitate` from vLLM to `local`. vLLM has been removed (it was fork-only and superseded). **Audio is not served on the bundle** — llama-server rejects Nemotron audio input — so audio stays on the Whisper STT pipeline. So on the Spark today: `local` serves text + image; Whisper serves audio; cloud providers cover tier fallback. Full history in the Obsidian `90-99 Agents/Transient/llama.cpp Local Bundle Migration Plan`.


# Why two tables

The same provider information answers two different questions, and they want different shapes:

- *"For this provider/model, what can I use it for?"* — the **capability** view. Useful when evaluating a new model, debugging a routing decision, or understanding why some features can't fall back gracefully when a particular provider is unavailable.
- *"For this task, which providers can serve it?"* — the **routing** view. Useful when adding a new talent or wiring a journal config: you want to see all the candidates for a given role and pick one.

Each table omits information the other surfaces. Read both for the full picture.


# Capability matrix: per provider/model → roles

For each registered provider and model, what can it be used for. Marker key:

- ✓ — capable + currently wired in Solstone's provider abstraction
- ◐ — capable but with caveats (not wired, not yet verified, partial support, or notable performance considerations)
- ✗ — not capable or not applicable

## Cloud providers (tier-routed, require API key + outbound network)

Cogitate for all three cloud providers now runs through the **`openhands` façade** (`openhands-sdk` + LiteLLM), installed on-demand via `solstone/think/providers/bundled.py`. The registry maps `google`/`openai`/`anthropic` → `solstone.think.providers.openhands` for the agentic loop; `run_generate`/`run_agenerate` stay in the per-provider modules.

| Provider | Text Gen | Cogitate | Vision in | Audio in | Embeddings | Notes |
|----------|:--------:|:--------:|:---------:|:--------:|:----------:|-------|
| Google Gemini (PRO / FLASH / LITE) | ✓ | ✓ (openhands) | ✓ (PIL Image native) | ✗ | ✓ (separate model) | Default for `generate` and `cogitate` per `TYPE_DEFAULTS`. Vision is best-in-class for the existing describe pipeline. |
| OpenAI (GPT-5.4 + variants) | ✓ | ✓ (openhands) | ✓ | ◐ (gpt-4o-audio family exists; not wired) | ✓ | Reasoning-effort suffix support (`-high`, `-low`, `-xhigh`). |
| Anthropic (Claude Opus / Sonnet / Haiku 4.x) | ✓ | ✓ (openhands) | ✓ | ✗ | ✗ | Default `backup` provider for both `generate` and `cogitate`. No native embeddings surface. |

## Local — llama.cpp bundle (`local`, GGUF over llama-server's OpenAI endpoint)

Upstream's bundled local provider, **the production local backend on the Spark**. Downloads a pinned `llama-server` binary (`solstone/think/providers/local_install.py`), lazily starts it (`local_server.py`), and serves GGUF weights over the OpenAI endpoint. Serves **text + image**; the fork extended `local.py`'s content-block intake to images (`ImageBlock` → `image_url`, gated on a model carrying an mmproj projector) and wires `--mmproj` at launch. **Audio is not served** — llama-server rejects Nemotron audio input, so `local.py` raises `unsupported_capability` and audio runs through Whisper.

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `local/qwen3-coder-30b-a3b-q4_k_m` (PRO) | ✓ | ✓ (openhands against local endpoint) | ✗ | ✗ | Upstream's PRO default. ~95 tok/s on the GB10 (MoE, ~3B active). |
| `local/qwen2.5-coder-7b` (FLASH / LITE) | ✓ | ◐ | ✗ | ✗ | Upstream's FLASH/LITE default. ~95 tok/s on the GB10. |
| `local/nemotron-3-nano-omni` (Q8_0 + mmproj) | ✓ | ◐ | ✓ (image, via mtmd `--mmproj`) | ✗ (llama-server rejects audio for this model — HTTP 500; audio stays on Whisper) | Fork-only. The Spark's vision model for the describe pipeline. From `ggml-org/NVIDIA-Nemotron-3-Nano-Omni`. |

Platform pins (`LLAMA_SERVER_PINS`): `aarch64-apple-darwin` ✓, `x86_64-unknown-linux-gnu` ✓ (CPU build), `aarch64-unknown-linux-gnu` ✓ (Spark/GB10 — fork-hosted CUDA build, sm_121, llama.cpp `b9291` / CUDA 13.0; do **not** build against CUDA 13.2: gibberish output for Nemotron per Unsloth).

## Local — MLX (`mlx`, Apple Silicon only)

Upstream's Apple-Silicon provider (mlx-vlm). Vision-capable. **Not applicable to this fork's Spark** (aarch64 Linux, not macOS) — listed for completeness.

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `qwen3.5:9b-mlx-8bit` / `gemma-4-26b-...-mlx-4bit` | ✓ | ✗ (MLX has no cogitate) | ✓ | ✗ | macOS ARM64 + mlx-vlm only. Irrelevant on the Spark. |

> **vLLM removed (Phase D).** The fork-only vLLM provider (`vllm-local/*` models, server-per-model AWQ/bf16/NVFP4) was deleted once `local` covered production text + image and Whisper covered audio. Its speculative-only remaining capabilities (NVFP4 throughput, theoretical video) didn't justify the fork↔upstream merge friction; git history preserves it if resurrected.

## In-process specialists (not LLM providers; no tier system)

| Specialist | Role | Notes |
|------------|------|-------|
| `faster-whisper` (large-v3 / medium / etc.) | Audio transcription, live capture | Default `transcribe.backend` on the Spark. Produces sentence-level `audio.jsonl` + sentence embeddings used by speaker-attribution flywheel. |
| Parakeet TDT | Audio transcription, alternative | In-process (CoreML / linux-x86_64 ONNX). Deferred on aarch64 Spark per [Transcription Backend Architecture] (Obsidian). |
| Rev.ai | Audio transcription, cloud import only | Used for imported audio paths only, not live capture. |
| `wespeaker` (resnet34) | Per-sentence speaker voiceprint embeddings (256-dim) | The substrate for the voiceprint-learning flywheel. Purpose-built for speaker similarity; an omni model's audio encoder is not a substitute. |
| Resemblyzer | Speaker embeddings, older path | Some legacy paths use this; the wespeaker pipeline is the modern one. |
| pyannote-audio | VAD / diarization | Sentence-boundary segmentation. |
| Tesseract / PaddleOCR / pypdf / pdf2image | OCR + PDF text extraction | Document pipeline. Specialist tools are faster on the document workload than an omni model's OCR. |
| OpenCV (`cv2`) | Video frame extraction | Preprocessing for the screen describe pipeline. Not AI. |


# Routing matrix: per layer/task → provider

For each layer or task, this shows the providers wired today and the graceful-degradation behavior. "Generate" / "Cogitate" rows are tier-routed via `think/models.py::TYPE_DEFAULTS` + `PROVIDER_DEFAULTS`; the configured local provider on the Spark is the **`local` bundle** (production `generate`/`cogitate` primary).

**Patterns:**

- **Specialist** — one provider, by design; not pluggable per-call.
- **Journal-pluggable** — journal config picks the provider per surface or per context; tier system handles fallback.
- **Local-only** — feature requires a local multimodal model; absence means the feature is unavailable (degrades gracefully).

| Layer / Task | Provider(s) | Pattern | Degradation |
|--------------|-------------|---------|-------------|
| Audio transcription (live capture) | `faster-whisper` (in-process) | specialist | unaffected by LLM-provider state |
| Audio transcription (import: Plaud / Rev.ai) | configured per journal | specialist | unaffected |
| Speaker embedding | `wespeaker` (in-process) | specialist | unaffected |
| Speaker attribution L1–L3 (acoustic) | in-process pipeline | specialist | unaffected |
| Speaker attribution L4 (LLM fallback) | Generate (tier-routed) | journal-pluggable | uses configured Generate provider |
| Screen frame categorize | Generate w/ vision (tier-routed) | journal-pluggable | needs a vision-capable provider (cloud, or `local/nemotron-3-nano-omni` on the Spark) |
| Screen frame extract (selected frames only) | Generate w/ vision (tier-routed) | journal-pluggable | same |
| Segment synthesis (sense, entities, todos, summary) | Generate (tier-routed) | journal-pluggable | cloud or local text covers it |
| Daily insights, muse generators | Generate (tier-routed) | journal-pluggable | same |
| Voice-brain (text response) | Generate (tier-routed) | journal-pluggable | same |
| Cogitate (chat agent, daily briefing, agent talents) | Cogitate (tier-routed; cloud via openhands, local via configured local provider) | journal-pluggable | cloud or local covers it |
| OCR / PDF text extraction | Tesseract / pypdf / PaddleOCR (in-process) | specialist | unaffected |
| Embeddings (search & similarity) | provider-specific or `nomic-embed-text` | specialist or journal-pluggable | unaffected |
| Multimodal-augmented meeting summary (audio fusion) | audio-omni model — **no local option** (llama-server rejects Nemotron audio); cloud-only if ever wired | local-only | feature unavailable on the bundle; Whisper transcript + Generate covers the text path |
| Voice-brain audio-tone interpretation (planned B/C) | audio-omni model (opt-in) | local-only | feature unavailable; voice-brain falls back to text-only |
| Cross-modal search (planned B/C) | audio-omni model (opt-in) | local-only | feature unavailable; FTS search keeps working |

**Read:** Everything currently load-bearing is journal-pluggable or specialist. Audio/omni features are additive — they go missing if the omni model is down, but the rest of the segment record stays intact.


# Local bundle serving model

The `local` bundle is single-server: `local_server.ensure_running()` manages one `llama-server` process and swaps models by restart, with the port tracked via `read_service_port("local")`. Reattach verifies the served model (`/v1/models`) before reusing a running server, so an in-process model switch never silently serves the previously-loaded model. Vision-capable models additionally launch with `--mmproj <projector>`; projector GGUFs are downloaded + sha256-verified alongside the main weights in `local_install.install_model`.


# Cross-cutting observations

Load-bearing facts when reasoning about which provider belongs where:

1. **Audio understanding has no local model.** Audio runs through the Whisper STT specialist path (transcript, no fusion with other modalities). The Phase C spike confirmed llama-server rejects Nemotron Omni audio input over HTTP ("audio input is not supported"), so there is no audio-in LLM on the bundle — the closest substitute is "Whisper transcribes, a Generate-tier LLM interprets the text," which loses the audio-direct capability. Revisit if llama.cpp wires Nemotron audio dispatch.

2. **`local` now serves image; audio is the fork's one omni gap.** The fork extended `local.py`'s content-block intake to images (`ImageBlock` → `image_url`, gated on an mmproj projector) — this is fork-only code, diverging from upstream's text-only `local`. Audio was intentionally not wired (the spike's gate failed).

3. **The cloud↔local fallback story is symmetric for text/cogitate and now vision, asymmetric for audio.** Text/cogitate/vision degrade gracefully — cloud or the `local` bundle cover each other (vision via `local/nemotron-3-nano-omni` or a cloud VLM). Audio understanding has no local fallback: the substitute is "Whisper transcribes, Generate-tier LLM interprets the text."

This is why audio/omni features should stay **opt-in additive capabilities** (missing when the omni model is down, rest of the segment record intact) rather than **critical-path replacements**. Segment synthesis stays on tier-routed Generate providers, not an omni model, because it's the substrate-producing pipeline that downstream features depend on — it has to be available whenever Solstone is running.


# Related

- [PROVIDERS.md](PROVIDERS.md) — implementation guide for adding a new provider.
- [THINK.md](THINK.md) — module overview for `think/`.
- `think/providers/__init__.py` — canonical `PROVIDER_REGISTRY` and `PROVIDER_METADATA`.
- `think/providers/local.py`, `local_install.py`, `local_server.py` — the bundled llama.cpp provider.
- `think/models.py` — tier constants, `PROVIDER_DEFAULTS`, `TYPE_DEFAULTS`, talent-context resolution.
- `think/benchmark/models.json` — pre-vetted local models with measured tok/s and per-task wall-clock.
- Obsidian: `90-99 Agents/Transient/llama.cpp Local Bundle Migration Plan` — the staged plan (Phases A–D) that moved Spark production inference onto the `local` bundle and removed vLLM.
- Obsidian: `20-29 Tech/Solstone AI Modalities Overview` — modality-by-modality inventory of Solstone's AI usage.
- Obsidian: `90-99 Agents/Transient/vLLM Provider & Multimodal Benchmark Plan` — how the (now-removed) vLLM provider was built; historical, and the content-block work Phase C reused for `local.py`.
