# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark harness — measure a model's speed on this host.

Maintainer-only. Run by hand to seed ``models.json`` /
``transcribers.json`` with real measurements; not invoked by the live
pipeline. Connect-only: the bundled llama-server must already be running
under the supervisor (``sol supervisor``) before running an LLM benchmark —
the harness attaches to that daemon and does not spawn its own server.

Three modes:

1. **Synthetic benchmark** (default) — sends a fixed prompt, records
   output tok/s and prompt tok/s. Good for populating the baseline
   tok/s used by the formula estimator.

2. **Task benchmark** (``--task <task_id>``) — reads the fixture under
   ``fixtures/<task_id>.txt``, runs it end-to-end, records wall-clock
   seconds. Output pastes directly into the model's
   ``benchmarks.<class>.tasks.<task_id>`` entry. This is the
   ground-truth source for task-time heuristics — no token formulas,
   no interpolation.

3. **Transcriber RTF** (``--transcriber <backend> --audio-fixture
   <path>``) — runs the configured in-process STT backend against an
   audio file of any length and records
   ``RTF = wall_seconds / audio_seconds``. Output pastes into
   ``transcribers.json`` →
   ``transcribers[backend].benchmarks[<class>]``. Powers the audio
   lane of the segment-time semantic benchmark.

Usage::

    # Synthetic tok/s benchmark (bundled llama-server; --pull to fetch the GGUF)
    python -m solstone.think.benchmark.harness --model local/qwen3.6-35b-a3b \\
        --class dgx-spark --pull

    # Task-time benchmark (vision flag auto-applied when task.mode=vision, so
    # this measures the mmproj image-encoder cost via the production image path)
    python -m solstone.think.benchmark.harness --model local/qwen3.6-35b-a3b \\
        --class dgx-spark --task screen_frame

    # Transcriber RTF (point at any mono 16kHz audio file)
    python -m solstone.think.benchmark.harness --transcriber whisper \\
        --audio-fixture /path/to/audio.wav --class dgx-spark

The script never writes ``models.json`` or ``transcribers.json``
directly — the maintainer reviews the output snippet before committing.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

from solstone.think.providers.shared import BenchmarkResult

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Fixed text prompt used for text-mode runs. ~150-200 tokens of input,
# asks for ~200 tokens of output.
_TEXT_PROMPT = (
    "You are a benchmarking fixture. Write a focused, concrete 200-word "
    "technical paragraph explaining the tradeoffs between dense and "
    "sparse-mixture-of-experts transformer architectures for on-device "
    "inference. Discuss parameter count, active-parameter count, memory "
    "bandwidth, and typical quantization strategies. Do not include "
    "headings, lists, or citations — a single paragraph of prose."
)

# Prompt used in vision mode alongside the canned image. Asks for enough
# output to make the output-tok/s number stable.
_VISION_PROMPT = (
    "Describe this image in a focused 200-word paragraph. Cover the "
    "geometric shapes, colors, layout, and any text visible. Do not "
    "include headings, lists, or citations — a single paragraph of prose."
)

# Prompt used in synthetic audio mode alongside the canned audio fixture.
_AUDIO_PROMPT = (
    "Describe what you hear in this audio in a focused 200-word "
    "paragraph. Cover the speaker (one or many), the content of any "
    "speech, the tone and pace, and any non-speech sounds. Do not "
    "include headings or lists — a single paragraph of prose."
)

# Default audio fixture for synthetic audio mode (when no --task is
# specified). Resolved relative to _FIXTURES_DIR.
_DEFAULT_AUDIO_FIXTURE = "audio_30s.wav"

_MAX_OUTPUT_TOKENS = 256
_WARMUP_RUNS = 1
_MEASURE_RUNS = 3


def _build_canned_image_b64() -> str:
    """Generate a deterministic 512x512 JPEG and return it as base64.

    Uses PIL to draw a mix of shapes, a gradient, and text so the
    vision encoder has non-trivial content to process. Same image every
    run so prompt-eval numbers are comparable.
    """
    from PIL import Image, ImageDraw, ImageFont

    size = 512
    img = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Gradient band
    for y in range(0, 128):
        shade = int(80 + (y / 128) * 120)
        draw.line([(0, y), (size, y)], fill=(shade, shade // 2, 200 - shade // 2))

    # Geometric shapes
    draw.rectangle([40, 180, 200, 340], fill=(200, 60, 60), outline=(0, 0, 0), width=3)
    draw.ellipse([280, 180, 460, 360], fill=(60, 140, 200), outline=(0, 0, 0), width=3)
    draw.polygon(
        [(256, 380), (400, 480), (112, 480)],
        fill=(80, 180, 90),
        outline=(0, 0, 0),
    )

    # Text so the encoder sees character content
    try:
        font = ImageFont.load_default()
        draw.text((20, 20), "solstone benchmark fixture", fill=(20, 20, 20), font=font)
        draw.text(
            (20, 490), "shapes: square, circle, triangle", fill=(20, 20, 20), font=font
        )
    except OSError:
        # If default font missing in some minimal env, skip text.
        pass

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_provider(model: str) -> ModuleType:
    """Resolve the provider module for a given model id.

    Routes by prefix:
        ``local/...`` -> ``solstone.think.providers.local`` (bundled llama-server)

    Raises SystemExit with a clear message for unknown prefixes so the
    harness fails fast rather than reaching some half-implemented path.
    """
    if model.startswith("local/"):
        return importlib.import_module("solstone.think.providers.local")
    raise SystemExit(
        f"Cannot resolve benchmark provider for model id: {model!r}. "
        f"Expected a 'local/' prefix."
    )


def _load_audio_b64(audio_fixture: str | None) -> tuple[str, str]:
    """Load an audio fixture from ``fixtures/`` and return (base64, format).

    ``audio_fixture`` is a filename relative to ``_FIXTURES_DIR`` (e.g.
    ``audio_30s.wav``). ``None`` means "use the default fixture." The
    return ``format`` is the file extension lowercased without the dot,
    matching the OpenAI ``input_audio.format`` field (``wav``, ``flac``,
    ``mp3``, etc.).
    """
    name = audio_fixture or _DEFAULT_AUDIO_FIXTURE
    path = _FIXTURES_DIR / name
    if not path.exists():
        raise SystemExit(
            f"No audio fixture at {path}. Add one before running audio-mode harness."
        )
    suffix = path.suffix.lstrip(".").lower() or "wav"
    return base64.b64encode(path.read_bytes()).decode("ascii"), suffix


def run_once(
    model: str,
    *,
    vision: bool = False,
    audio: bool = False,
    audio_fixture: str | None = None,
    prompt_override: str | None = None,
    max_output_tokens: int = _MAX_OUTPUT_TOKENS,
    base_url: str | None = None,
) -> BenchmarkResult:
    """Send one benchmark request via the model's provider.

    When ``vision=True``, include a canned image in the user message so
    the prompt-eval count captures image-encoder cost. When
    ``audio=True``, include the audio fixture (default
    ``_DEFAULT_AUDIO_FIXTURE``, or whichever ``audio_fixture`` names) so
    prompt-eval captures audio-encoder cost. ``vision`` and ``audio`` may
    be combined for omni models. When ``prompt_override`` is set, use
    that text instead of the default mode-specific prompt — used by
    task-mode runs that read a fixture.

    Returns a normalized BenchmarkResult; provider-specific transport,
    response shape, and tok/s computation live in the provider module.
    """
    provider = _resolve_provider(model)
    if prompt_override is not None:
        prompt_text = prompt_override
    elif audio:
        prompt_text = _AUDIO_PROMPT
    elif vision:
        prompt_text = _VISION_PROMPT
    else:
        prompt_text = _TEXT_PROMPT
    image_b64 = _build_canned_image_b64() if vision else None
    audio_b64: str | None = None
    audio_format = "wav"
    if audio:
        audio_b64, audio_format = _load_audio_b64(audio_fixture)
    return provider.bench_run_once(
        model,
        prompt=prompt_text,
        image_b64=image_b64,
        audio_b64=audio_b64,
        audio_format=audio_format,
        max_output_tokens=max_output_tokens,
        base_url=base_url,
    )


def _load_task(task_id: str) -> dict[str, Any]:
    """Load a task spec from ``tasks.json`` by id."""
    tasks_file = Path(__file__).parent / "tasks.json"
    catalog = json.loads(tasks_file.read_text()).get("tasks", {})
    if task_id not in catalog:
        raise SystemExit(
            f"Task '{task_id}' not in tasks.json. Available: "
            f"{', '.join(sorted(catalog.keys()))}"
        )
    return catalog[task_id]


def _load_fixture(task_id: str) -> str:
    """Load the fixture prompt text for ``task_id``."""
    path = _FIXTURES_DIR / f"{task_id}.txt"
    if not path.exists():
        raise SystemExit(
            f"No fixture at {path}. Add one before running task-mode harness "
            f"for '{task_id}'."
        )
    return path.read_text()


def ensure_installed(model: str, *, allow_pull: bool) -> None:
    """Verify the model is installed locally; optionally trigger a pull."""
    _resolve_provider(model).bench_ensure_installed(model, allow_pull=allow_pull)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a bundled local (llama-server) model."
    )
    parser.add_argument(
        "--model",
        help=(
            "Model ID, e.g. local/qwen2.5-coder-7b. "
            "Required unless --transcriber is set."
        ),
    )
    parser.add_argument(
        "--class",
        dest="hw_class",
        required=True,
        help="Hardware class key for this host (see think/benchmark/reference.json)",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help=(
            "Request an install if the model is not already available. "
            "Pulls the GGUF (and any mmproj) via the bundle for local/ models."
        ),
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help=(
            "Vision mode: include a canned image in the prompt. Use for VLMs "
            "so prompt-eval captures image-encoder cost. Auto-enabled when "
            "--task specifies a vision-mode task."
        ),
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help=(
            "Audio mode: include the default audio fixture (audio_30s.wav, a "
            "30s LibriVox PD speech clip) in the prompt. Use for omni models "
            "so prompt-eval captures audio-encoder cost. Auto-enabled when "
            "--task specifies an audio-mode task. Requires a provider that "
            "accepts audio input (the bundled local provider does not)."
        ),
    )
    parser.add_argument(
        "--task",
        help=(
            "Task id from tasks.json. Runs the fixture under "
            "fixtures/<task_id>.txt end-to-end and records wall-clock seconds "
            "as the measured task-time (printed as a paste-ready snippet)."
        ),
    )
    parser.add_argument(
        "--transcriber",
        help=(
            "STT backend to benchmark for RTF (parakeet, whisper, gemini, "
            "revai). Mutually exclusive with --model. Requires --audio-fixture."
        ),
    )
    parser.add_argument(
        "--audio-fixture",
        help=(
            "Path to an audio file (any length, mono 16kHz preferred). "
            "Used with --transcriber to measure real-time-factor."
        ),
    )
    args = parser.parse_args()

    if args.transcriber:
        if not args.audio_fixture:
            raise SystemExit("--transcriber requires --audio-fixture <path>")
        return _run_transcriber_mode(args)

    if not args.model:
        raise SystemExit("--model is required (or use --transcriber)")

    ensure_installed(args.model, allow_pull=args.pull)

    if args.task:
        return _run_task_mode(args)

    return _run_tok_s_mode(args)


def _bench_tok_s(result: BenchmarkResult) -> tuple[float, float]:
    """Extract (output_tok_s, prompt_tok_s) from a BenchmarkResult.

    Prefers the provider's native server-side counters when available.
    Falls back to wall-clock output rate (output_tokens / elapsed_s) when
    native is missing — and leaves prompt_tok_s at 0.0 in that case, since
    wall-clock prompt-eval rate would conflate prefill with decode and
    report a misleading number. Providers that lack native prompt-eval
    timing will need a separate measurement pass for prompt rate.
    """
    elapsed = result.get("elapsed_s") or 0.0
    output_tokens = result.get("output_tokens") or 0
    native_out = result.get("native_output_tok_s")
    native_prompt = result.get("native_prompt_tok_s")

    out_rate = (
        native_out
        if native_out is not None
        else (output_tokens / elapsed if elapsed > 0 else 0.0)
    )
    prompt_rate = native_prompt if native_prompt is not None else 0.0
    return out_rate, prompt_rate


def _run_tok_s_mode(args: argparse.Namespace) -> int:
    """Synthetic-prompt tok/s benchmark."""
    if args.audio:
        mode = "audio"
    elif args.vision:
        mode = "vision"
    else:
        mode = "text_only"
    print(
        f"Benchmarking {args.model} on class '{args.hw_class}' in {mode} mode "
        f"({_WARMUP_RUNS} warmup + {_MEASURE_RUNS} measured runs)…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(args.model, vision=args.vision, audio=args.audio)

    output_rates: list[float] = []
    prompt_rates: list[float] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        result = run_once(args.model, vision=args.vision, audio=args.audio)
        out_rate, prompt_rate = _bench_tok_s(result)
        output_rates.append(out_rate)
        prompt_rates.append(prompt_rate)
        print(
            f"    output {out_rate:.1f} tok/s, prompt {prompt_rate:.1f} tok/s",
            file=sys.stderr,
        )

    median_output = sorted(output_rates)[len(output_rates) // 2]
    median_prompt = sorted(prompt_rates)[len(prompt_rates) // 2]

    snippet = {
        args.hw_class: {
            "output_tok_s": round(median_output, 1),
            "prompt_tok_s": round(median_prompt, 1),
            "measured_at": date.today().isoformat(),
            "mode": mode,
        }
    }
    print(
        f"\n# Paste into think/benchmark/models.json "
        f"-> models['{args.model}'].benchmarks:",
        file=sys.stderr,
    )
    print(json.dumps(snippet, indent=2))
    return 0


def _run_task_mode(args: argparse.Namespace) -> int:
    """End-to-end task-fixture benchmark; records wall-clock seconds."""
    task_spec = _load_task(args.task)
    fixture = _load_fixture(args.task)
    task_mode = task_spec.get("mode")
    vision = task_mode == "vision" or args.vision
    audio = task_mode == "audio" or args.audio
    audio_fixture = task_spec.get("audio_fixture") if task_mode == "audio" else None
    expected_output_tokens = int(task_spec.get("output_tokens") or 400)

    print(
        f"Benchmarking task '{args.task}' with {args.model} on class "
        f"'{args.hw_class}' (mode={task_mode}, presence="
        f"{task_spec.get('presence')}); {_WARMUP_RUNS} warmup + "
        f"{_MEASURE_RUNS} measured runs…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(
            args.model,
            vision=vision,
            audio=audio,
            audio_fixture=audio_fixture,
            prompt_override=fixture,
            max_output_tokens=expected_output_tokens,
        )

    elapsed_samples: list[float] = []
    prompt_token_samples: list[int] = []
    output_token_samples: list[int] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        result = run_once(
            args.model,
            vision=vision,
            audio=audio,
            audio_fixture=audio_fixture,
            prompt_override=fixture,
            max_output_tokens=expected_output_tokens,
        )
        elapsed = result["elapsed_s"]
        prompt_tokens = int(result.get("prompt_tokens") or 0)
        output_tokens = int(result.get("output_tokens") or 0)
        elapsed_samples.append(elapsed)
        prompt_token_samples.append(prompt_tokens)
        output_token_samples.append(output_tokens)
        print(
            f"    wall={elapsed:.2f}s  prompt_tokens={prompt_tokens}  "
            f"output_tokens={output_tokens}",
            file=sys.stderr,
        )

    median_elapsed = sorted(elapsed_samples)[len(elapsed_samples) // 2]
    median_prompt_tokens = sorted(prompt_token_samples)[len(prompt_token_samples) // 2]
    median_output_tokens = sorted(output_token_samples)[len(output_token_samples) // 2]

    snippet = {
        args.task: {
            "seconds": round(median_elapsed, 2),
            "prompt_tokens": median_prompt_tokens,
            "output_tokens": median_output_tokens,
            "measured_at": date.today().isoformat(),
        }
    }
    print(
        f"\n# Paste into think/benchmark/models.json "
        f"-> models['{args.model}'].benchmarks['{args.hw_class}'].tasks:",
        file=sys.stderr,
    )
    print(json.dumps(snippet, indent=2))
    return 0


def _preflight_transcriber(transcriber: str, hw_class: str) -> dict[str, Any]:
    """Validate (transcriber, hardware_class) pair against transcribers.json.

    Hard-fails with a specific error before any transcription work runs:

    - Unknown transcriber name → fail with the available list.
    - ``benchmarkable: false`` (cloud backends) → fail; harness has no
      meaningful RTF to capture for cloud backends.
    - ``hw_class`` not in ``supported_hardware`` (and the list is not
      ``["*"]``) → fail. This is what stops a Spark machine from
      "successfully" benchmarking the in-process parakeet backend
      (which doesn't run on aarch64).

    Returns the resolved transcriber spec for downstream use.
    """
    transcribers_file = Path(__file__).parent / "transcribers.json"
    catalog = json.loads(transcribers_file.read_text()).get("transcribers", {})

    if transcriber not in catalog:
        names = ", ".join(sorted(catalog.keys())) or "(none)"
        raise SystemExit(f"Unknown transcriber '{transcriber}'. Available: {names}")

    spec = catalog[transcriber]

    if not spec.get("benchmarkable", False):
        kind = spec.get("kind", "?")
        raise SystemExit(
            f"Transcriber '{transcriber}' is not benchmarkable (kind={kind}). "
            f"Cloud backends use a flat wall_seconds_per_5min rule of thumb "
            f"in transcribers.json — RTF capture is meaningless when network "
            f"latency dominates wall-clock."
        )

    supported = spec.get("supported_hardware") or []
    if supported != ["*"] and hw_class not in supported:
        listed = ", ".join(supported) if supported else "(none)"
        raise SystemExit(
            f"Transcriber '{transcriber}' does not support hardware class "
            f"'{hw_class}'. Supported: {listed}.\n"
            f"This guard prevents corrupting the benchmarking signal by "
            f"running the wrong backend on a host it can't actually serve."
        )

    return spec


def _run_transcriber_mode(args: argparse.Namespace) -> int:
    """Real-time-factor benchmark for an STT backend on an audio fixture."""
    _preflight_transcriber(args.transcriber, args.hw_class)

    audio_path = Path(args.audio_fixture)
    if not audio_path.exists():
        raise SystemExit(f"Audio fixture not found: {audio_path}")

    # Lazy imports — observe.transcribe pulls heavy deps that aren't
    # needed for the LLM benchmark modes.
    from solstone.observe.transcribe import transcribe as stt_transcribe
    from solstone.observe.utils import SAMPLE_RATE, load_audio
    from solstone.think.utils import get_config

    audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
    sample_rate = SAMPLE_RATE
    audio_seconds = float(len(audio)) / float(sample_rate)
    if audio_seconds <= 0:
        raise SystemExit(f"Audio fixture has zero duration: {audio_path}")

    config = get_config()
    backend_config = (config.get("transcribe") or {}).get(args.transcriber, {}) or {}

    print(
        f"Benchmarking transcriber '{args.transcriber}' on class "
        f"'{args.hw_class}' against {audio_path.name} ({audio_seconds:.1f}s "
        f"audio); {_WARMUP_RUNS} warmup + {_MEASURE_RUNS} measured runs…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        stt_transcribe(args.transcriber, audio, sample_rate, backend_config)

    elapsed_samples: list[float] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        start = time.perf_counter()
        stt_transcribe(args.transcriber, audio, sample_rate, backend_config)
        elapsed = time.perf_counter() - start
        elapsed_samples.append(elapsed)
        rtf = elapsed / audio_seconds
        print(
            f"    wall={elapsed:.2f}s  audio={audio_seconds:.1f}s  rtf={rtf:.3f}",
            file=sys.stderr,
        )

    median_elapsed = sorted(elapsed_samples)[len(elapsed_samples) // 2]
    median_rtf = median_elapsed / audio_seconds

    snippet = {
        args.hw_class: {
            "rtf": round(median_rtf, 3),
            "audio_seconds_measured": round(audio_seconds, 1),
            "wall_seconds_measured": round(median_elapsed, 2),
            "measured_at": date.today().isoformat(),
        }
    }
    print(
        f"\n# Paste into think/benchmark/transcribers.json "
        f"-> transcribers['{args.transcriber}'].benchmarks:",
        file=sys.stderr,
    )
    print(json.dumps(snippet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
