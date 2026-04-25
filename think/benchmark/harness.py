# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark harness — measure a model's speed on this host.

Maintainer-only. Run by hand to seed ``models.json`` /
``transcribers.json`` with real measurements; not invoked by the live
pipeline.

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
   <path>``) — runs the configured STT backend against an audio file
   of any length and records ``RTF = wall_seconds / audio_seconds``.
   Output pastes into ``transcribers.json`` →
   ``transcribers[backend].benchmarks[<class>]``. Powers the audio
   lane of the segment-time semantic benchmark.

Usage::

    # Synthetic tok/s benchmark
    python -m think.benchmark.harness --model ollama-local/qwen3.5:9b \\
        --class rtx-4090

    # Task-time benchmark (vision flag auto-applied when task.mode=vision)
    python -m think.benchmark.harness --model ollama-local/qwen3.5:9b \\
        --class rtx-4090 --task chat_reply

    # Transcriber RTF (point at any mono 16kHz audio file)
    python -m think.benchmark.harness --transcriber parakeet \\
        --audio-fixture /path/to/audio.wav --class rtx-4090

The script never writes ``models.json`` or ``transcribers.json``
directly — the maintainer reviews the output snippet before committing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

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

_MAX_OUTPUT_TOKENS = 256
_WARMUP_RUNS = 1
_MEASURE_RUNS = 3

# Cap context window to keep the compute graph tractable. Ollama otherwise
# defaults to a very large context (256K for recent Qwen builds), which
# inflates the KV cache + compute graph enough to OOM big models on
# unified-memory systems. 8K is plenty for the fixed benchmark prompt +
# image tokens + 256-token completion.
_BENCHMARK_NUM_CTX = 8192


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


def run_once(
    model: str,
    *,
    vision: bool = False,
    prompt_override: str | None = None,
    max_output_tokens: int = _MAX_OUTPUT_TOKENS,
) -> tuple[dict[str, Any], float]:
    """Send one completion request; return ``(response_body, wall_clock_s)``.

    When ``vision=True``, include a canned image in the user message so
    the prompt-eval count captures image-encoder cost. When
    ``prompt_override`` is set, use that text instead of the default
    benchmark prompt — used by task-mode runs that read a fixture.
    """
    from think.providers.ollama import _get_client, _strip_model_prefix

    bare_model = _strip_model_prefix(model)
    client = _get_client()

    prompt_text = prompt_override or (_VISION_PROMPT if vision else _TEXT_PROMPT)
    if vision:
        message: dict[str, Any] = {
            "role": "user",
            "content": prompt_text,
            "images": [_build_canned_image_b64()],
        }
    else:
        message = {"role": "user", "content": prompt_text}

    start = time.perf_counter()
    response = client.post(
        "/api/chat",
        json={
            "model": bare_model,
            "messages": [message],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_output_tokens,
                "num_ctx": _BENCHMARK_NUM_CTX,
            },
        },
        timeout=600.0,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed


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
    from think.providers.ollama import _get_client, _strip_model_prefix

    bare_model = _strip_model_prefix(model)
    client = _get_client()
    response = client.get("/api/tags", timeout=10.0)
    response.raise_for_status()
    installed = {m.get("name") for m in response.json().get("models", [])}

    if bare_model in installed:
        return

    if not allow_pull:
        raise SystemExit(
            f"Model '{bare_model}' not installed. Run `ollama pull {bare_model}` "
            f"first, or pass --pull."
        )

    print(f"Pulling {bare_model}…", file=sys.stderr)
    with client.stream(
        "POST",
        "/api/pull",
        json={"name": bare_model},
        timeout=None,
    ) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines():
            if line:
                print(line, file=sys.stderr)


def tok_s_from_response(body: dict[str, Any]) -> tuple[float, float]:
    """Compute (output_tok_s, prompt_tok_s) from a single Ollama response.

    Ollama reports durations in nanoseconds.
    """
    eval_count = body.get("eval_count") or 0
    eval_duration_ns = body.get("eval_duration") or 0
    prompt_eval_count = body.get("prompt_eval_count") or 0
    prompt_eval_duration_ns = body.get("prompt_eval_duration") or 0

    output_tok_s = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
    prompt_tok_s = (
        (prompt_eval_count / (prompt_eval_duration_ns / 1e9))
        if prompt_eval_duration_ns
        else 0.0
    )
    return output_tok_s, prompt_tok_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama model.")
    parser.add_argument(
        "--model",
        help="Model ID, e.g. ollama-local/qwen3.5:9b. Required unless --transcriber is set.",
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
        help="Pull the model via Ollama API if not already installed.",
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


def _run_tok_s_mode(args: argparse.Namespace) -> int:
    """Synthetic-prompt tok/s benchmark."""
    mode = "vision" if args.vision else "text_only"
    print(
        f"Benchmarking {args.model} on class '{args.hw_class}' in {mode} mode "
        f"({_WARMUP_RUNS} warmup + {_MEASURE_RUNS} measured runs)…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(args.model, vision=args.vision)

    output_rates: list[float] = []
    prompt_rates: list[float] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        body, _elapsed = run_once(args.model, vision=args.vision)
        out_rate, prompt_rate = tok_s_from_response(body)
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
    vision = task_spec.get("mode") == "vision" or args.vision
    expected_output_tokens = int(task_spec.get("output_tokens") or 400)

    print(
        f"Benchmarking task '{args.task}' with {args.model} on class "
        f"'{args.hw_class}' (mode={task_spec.get('mode')}, presence="
        f"{task_spec.get('presence')}); {_WARMUP_RUNS} warmup + "
        f"{_MEASURE_RUNS} measured runs…",
        file=sys.stderr,
    )

    for i in range(_WARMUP_RUNS):
        print(f"  warmup {i + 1}/{_WARMUP_RUNS}…", file=sys.stderr)
        run_once(
            args.model,
            vision=vision,
            prompt_override=fixture,
            max_output_tokens=expected_output_tokens,
        )

    elapsed_samples: list[float] = []
    prompt_token_samples: list[int] = []
    output_token_samples: list[int] = []
    for i in range(_MEASURE_RUNS):
        print(f"  run {i + 1}/{_MEASURE_RUNS}…", file=sys.stderr)
        body, elapsed = run_once(
            args.model,
            vision=vision,
            prompt_override=fixture,
            max_output_tokens=expected_output_tokens,
        )
        prompt_tokens = int(body.get("prompt_eval_count") or 0)
        output_tokens = int(body.get("eval_count") or 0)
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
      "successfully" benchmarking the wrong parakeet backend.

    Returns the resolved transcriber spec for downstream use.
    """
    transcribers_file = Path(__file__).parent / "transcribers.json"
    catalog = json.loads(transcribers_file.read_text()).get("transcribers", {})

    if transcriber not in catalog:
        names = ", ".join(sorted(catalog.keys())) or "(none)"
        raise SystemExit(
            f"Unknown transcriber '{transcriber}'. Available: {names}"
        )

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


def _ensure_nim_reachable(transcriber: str, spec: dict[str, Any]) -> None:
    """For NIM-backed transcribers, hard-fail if the endpoint is unreachable.

    Silent fallback to another backend would corrupt the benchmark
    signal — the whole point of cross-backend RTF capture is comparing
    backends honestly. If the NIM container isn't running, the right
    answer is "fix the deployment", not "measure something else".
    """
    if spec.get("kind") != "local-http":
        return

    import os

    if transcriber == "parakeet-nim":
        url = os.environ.get("PARAKEET_NIM_URL", "http://localhost:9000")
    else:
        # Generic local-http: require an explicit env var per transcriber.
        env_key = f"{transcriber.upper().replace('-', '_')}_URL"
        url = os.environ.get(env_key)
        if not url:
            raise SystemExit(
                f"Transcriber '{transcriber}' is local-http but no endpoint "
                f"URL is configured (set {env_key})."
            )

    try:
        import httpx
    except ImportError as exc:
        raise SystemExit(
            f"Cannot reach {transcriber}: httpx not installed ({exc})"
        ) from exc

    try:
        with httpx.Client(timeout=5.0) as client:
            client.get(url)
    except Exception as exc:
        raise SystemExit(
            f"{transcriber} container not running at {url} ({exc.__class__.__name__}: "
            f"{exc}). Hard-fail: harness will not silently fall back to another "
            f"backend, since that would corrupt the cross-backend RTF signal."
        ) from exc


def _run_transcriber_mode(args: argparse.Namespace) -> int:
    """Real-time-factor benchmark for an STT backend on an audio fixture."""
    spec = _preflight_transcriber(args.transcriber, args.hw_class)
    _ensure_nim_reachable(args.transcriber, spec)

    audio_path = Path(args.audio_fixture)
    if not audio_path.exists():
        raise SystemExit(f"Audio fixture not found: {audio_path}")

    # Lazy imports — observe.transcribe pulls heavy deps that aren't
    # needed for the LLM benchmark modes.
    from observe.transcribe import transcribe as stt_transcribe
    from observe.utils import SAMPLE_RATE, load_audio
    from think.utils import get_config

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
            f"    wall={elapsed:.2f}s  audio={audio_seconds:.1f}s  "
            f"rtf={rtf:.3f}",
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
