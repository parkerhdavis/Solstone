# Provider & Model Capability Matrix

A reference for *which provider does what* in Solstone's pipelines, and *what each registered provider/model can be used for*. Two tables, opposite axes:

- **Routing matrix** — for every layer/task in Solstone, which providers handle it today and what happens if any one provider is unavailable.
- **Capability matrix** — for every registered provider/model, which roles it can fill and where the caveats are.

This is a *capability* doc, not an implementation guide. For the developer-facing "how to add a new provider" walkthrough, see [PROVIDERS.md](PROVIDERS.md).

The matrix reflects the system **as of the PR #44 upstream merge (2026-05-25)**, which dropped the `ollama` provider in favour of upstream's bundled `local` (llama.cpp / llama-server) provider, added an `mlx` provider, and routed cloud cogitate through a unified `openhands` façade. Re-check when adding new providers, models, or modalities.

> **Fork-in-transition note.** Upstream's `local` bundle pins prebuilt `llama-server` binaries only for `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu` — there is **no `aarch64-unknown-linux-gnu` pin**, so the bundle cannot install on this fork's DGX Spark (GB10) yet. Until that pin lands, **vLLM remains the only working local backend on the Spark**, and is doing more than the "additive-features-only" role this matrix originally described. The migration that makes the `local` bundle the Spark's production engine (and demotes vLLM to an optional experimental module) is planned in the Obsidian note `90-99 Agents/Transient/llama.cpp Local Bundle Migration Plan`. Rows below mark **current** vs **planned** where they differ.


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

Upstream's bundled local provider. Downloads a pinned `llama-server` binary (`solstone/think/providers/local_install.py`), lazily starts it (`local_server.py`), and serves GGUF weights. **Text-only in v1** — `local.py` explicitly raises `unsupported_capability` on vision and has no audio path. Platform support is the pinned-binary set; the Spark needs a fork-added CUDA pin (planned — see migration plan).

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `local/qwen3-coder-30b-a3b-q4_k_m` (PRO) | ✓ | ✓ (openhands against local endpoint) | ✗ | ✗ | Upstream's current PRO default. Planned to be replaced by the NVIDIA-supported set (Qwen3.6-35B-A3B, etc.). |
| `local/qwen2.5-coder-7b` (FLASH / LITE) | ✓ | ◐ | ✗ | ✗ | Upstream's current FLASH/LITE default. |
| *planned:* Nemotron 3 Nano Omni, Qwen3.6-35B-A3B / 27B, Gemma 4 family | ✓ | ✓ | ◐ (image, via mtmd `--mmproj` — fork Phase C) | ◐ (audio, via mtmd — experimental, fork Phase C) | NVIDIA's Spark-supported GGUF matrix. Image/audio require extending the text-only `local` provider to multimodal content blocks + projector files. |

Platform pins (`LLAMA_SERVER_PINS`): `aarch64-apple-darwin` ✓, `x86_64-unknown-linux-gnu` ✓ (CPU build), `aarch64-unknown-linux-gnu` (Spark/GB10) ✗ **planned** (fork-hosted CUDA build, sm_121 — do **not** build against CUDA 13.2: gibberish output for Nemotron per Unsloth).

## Local — MLX (`mlx`, Apple Silicon only)

Upstream's Apple-Silicon provider (mlx-vlm). Vision-capable. **Not applicable to this fork's Spark** (aarch64 Linux, not macOS) — listed for completeness.

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `qwen3.5:9b-mlx-8bit` / `gemma-4-26b-...-mlx-4bit` | ✓ | ✗ (MLX has no cogitate) | ✓ | ✗ | macOS ARM64 + mlx-vlm only. Irrelevant on the Spark. |

## Local — vLLM (`vllm`, server-per-model; **fork-only, transitional**)

Fork-only provider. Server-per-model (AWQ / bf16 / NVFP4 quants), multimodal content blocks. **Currently the only working local backend on the Spark** (pending the `local` CUDA pin). Slated to be demoted to an optional spin-up/down experimental module once the `local` bundle covers text + image + audio on the Spark (migration plan, Phase D). Kept for what llama.cpp can't do: NVFP4 throughput and (theoretical) video.

| Model | Text Gen | Cogitate | Vision in | Audio in | Best fit / caveats |
|-------|:--------:|:--------:|:---------:|:--------:|--------------------|
| `vllm-local/qwen3.5:35b-a3b` (bf16) | ✓ | ◐ (mechanically verified; `vllm.py::run_cogitate` not wired) | ✗ | ✗ | `VLLM_PRO`. |
| `vllm-local/qwen3.5:9b-awq` | ✓ | ◐ (same caveat) | ✗ | ✗ | `VLLM_FLASH`. **Caveat:** thinking-enabled accuracy can degrade up to 33% on reasoning tasks per kaitchup benchmarks. |
| `vllm-local/qwen3.5:2b-awq` | ✓ | ✗ | ✗ | ✗ | `VLLM_LITE`. Notably faster than the Ollama-era equivalent on Spark. |
| `vllm-local/qwen2.5vl:7b-awq` | ◐ | ✗ | ✓ | ✗ | Vision tier for the describe pipeline on the Spark today. |
| `vllm-local/nemotron-omni` (NVFP4) | ✓ | ◐ (untested with multimodal + tools combined) | ✓ | **✓** | The only currently-wired model with audio-input capability. Text + image too. Video claimed by NVIDIA, untested. The capability the fork's vLLM exists for — and the one Phase C aims to reproduce on llama.cpp. |

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

For each layer or task, this shows the providers wired today and the graceful-degradation behavior. "Generate" / "Cogitate" rows are tier-routed via `think/models.py::TYPE_DEFAULTS` + `PROVIDER_DEFAULTS`; the configured local provider on the Spark is **vLLM today, the `local` bundle once its CUDA pin lands**.

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
| Screen frame categorize | Generate w/ vision (tier-routed) | journal-pluggable | needs a vision-capable provider (cloud, or vLLM on the Spark today) |
| Screen frame extract (selected frames only) | Generate w/ vision (tier-routed) | journal-pluggable | same |
| Segment synthesis (sense, entities, todos, summary) | Generate (tier-routed) | journal-pluggable | cloud or local text covers it |
| Daily insights, muse generators | Generate (tier-routed) | journal-pluggable | same |
| Voice-brain (text response) | Generate (tier-routed) | journal-pluggable | same |
| Cogitate (chat agent, daily briefing, agent talents) | Cogitate (tier-routed; cloud via openhands, local via configured local provider) | journal-pluggable | cloud or local covers it |
| OCR / PDF text extraction | Tesseract / pypdf / PaddleOCR (in-process) | specialist | unaffected |
| Embeddings (search & similarity) | provider-specific or `nomic-embed-text` | specialist or journal-pluggable | unaffected |
| Multimodal-augmented meeting summary (planned B/C feature) | audio-omni model (vLLM today; `local`/llama.cpp targeted) | local-only | feature unavailable; segment record intact |
| Voice-brain audio-tone interpretation (planned B/C) | audio-omni model (opt-in) | local-only | feature unavailable; voice-brain falls back to text-only |
| Cross-modal search (planned B/C) | audio-omni model (opt-in) | local-only | feature unavailable; FTS search keeps working |

**Read:** Everything currently load-bearing is journal-pluggable or specialist. Audio/omni features are additive — they go missing if the omni model is down, but the rest of the segment record stays intact.


# vLLM multi-server config (current; fork-only)

While vLLM remains in the system, it serves one model per process (no hot-swap), so production routing across multiple vLLM-served models requires multiple containers, each on its own port. Configure them in `journal.json → providers.vllm.servers` as a map from *friendly name* (the part after `vllm-local/` in a model id) to a server descriptor:

```json
{
  "providers": {
    "vllm": {
      "servers": {
        "nemotron-omni": {
          "base_url": "http://localhost:8000",
          "served_model_name": "nemotron-omni"
        },
        "qwen3.5:35b-a3b": {
          "base_url": "http://localhost:8001",
          "served_model_name": "qwen3.5:35b-a3b"
        }
      }
    }
  }
}
```

When the provider receives a request for `vllm-local/<friendly>`, it strips the `vllm-local/` prefix to produce the friendly name, looks it up in `servers`, and uses the resolved `base_url` + `served_model_name`. `served_model_name` defaults to the friendly name when omitted. When `providers.vllm.servers` is absent (or lacks a friendly name), the provider falls back to env-var single-server: `VLLM_BASE_URL` (default `http://localhost:8000`).

`list_models()` and `validate_key()` enumerate across all configured servers; `build_provider_status()` reports `configured: true` if any configured server is reachable and names each unreachable URL in `issues`.

By contrast, the `local` bundle is single-server: `local_server.ensure_running()` manages one `llama-server` process and swaps models by restart, with the port tracked via `read_service_port("local")`.


# Cross-cutting observations

Load-bearing facts when reasoning about which provider belongs where:

1. **Audio input is currently a single-model capability.** Across cloud and the text-only `local` bundle, audio is either text-via-Whisper (specialist path, no fusion with other modalities) or unsupported. `vllm-local/nemotron-omni` is the only ✓ for audio-in today. The migration plan aims to reproduce this on llama.cpp (Nemotron Omni GGUF via libmtmd `--mmproj`), but llama.cpp audio is flagged "highly experimental" — so audio is a validation gate, not a given.

2. **The text-only `local` provider is a v1 design choice, and a fork extension point.** Vision/audio on the bundle (Phase C) means porting the content-block plumbing already built for `vllm.py` into `local.py` — fork-only code, diverging from upstream's text-only `local`.

3. **The cloud↔local fallback story is symmetric for text/cogitate, asymmetric for vision and audio on the Spark.** Text/cogitate degrade gracefully (cloud or local cover each other). Vision on the Spark currently needs vLLM or cloud (the `local` bundle is text-only and, until its CUDA pin, absent on the Spark). Audio understanding has no local fallback if the omni model is down — the closest substitute is "Whisper transcribes, Generate-tier LLM interprets the text," which loses the audio-direct capability.

This is why audio/omni features should stay **opt-in additive capabilities** (missing when the omni model is down, rest of the segment record intact) rather than **critical-path replacements**. Segment synthesis stays on tier-routed Generate providers, not an omni model, because it's the substrate-producing pipeline that downstream features depend on — it has to be available whenever Solstone is running.


# Related

- [PROVIDERS.md](PROVIDERS.md) — implementation guide for adding a new provider.
- [THINK.md](THINK.md) — module overview for `think/`.
- `think/providers/__init__.py` — canonical `PROVIDER_REGISTRY` and `PROVIDER_METADATA`.
- `think/providers/local.py`, `local_install.py`, `local_server.py` — the bundled llama.cpp provider.
- `think/models.py` — tier constants, `PROVIDER_DEFAULTS`, `TYPE_DEFAULTS`, talent-context resolution.
- `think/benchmark/models.json` — pre-vetted local models with measured tok/s and per-task wall-clock.
- Obsidian: `90-99 Agents/Transient/llama.cpp Local Bundle Migration Plan` — the staged plan to move Spark production inference onto the `local` bundle and demote vLLM.
- Obsidian: `20-29 Tech/Solstone AI Modalities Overview` — modality-by-modality inventory of Solstone's AI usage.
- Obsidian: `90-99 Agents/Transient/vLLM Provider & Multimodal Benchmark Plan` — how the vLLM provider was built (the content-block work Phase C reuses).
