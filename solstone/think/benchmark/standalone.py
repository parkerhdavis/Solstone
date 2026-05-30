# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Standalone head-to-head benchmark harness (fork-only, maintainer-only).

The default harness (``harness.py``) is connect-only: it attaches to the
supervisor-owned llama-server, which serves the single pinned ``LOCAL_MODEL``
and coerces every request to it (``normalize_model_id``). That is correct for
measuring the *served* model but cannot measure a *candidate* (a non-served
registry row) — a request "for" a candidate would just re-measure the served
model (the wrong-provenance trap that retired the old qwen-coder numbers).

This module benchmarks an arbitrary local GGUF by launching its OWN
llama-server: the bundled b9291 binary with the exact flags the supervisor uses
(``solstone/think/supervisor.py`` — ``--jinja``, ``--mmproj``, and the binary's
default full-GPU-offload), on a throwaway port. It then drives
``harness.run_once`` against that server's ``base_url``
(``bench_run_once(base_url=...)``, which skips the coercion) and prints a
paste-ready ``models.json`` benchmark snippet. Used for the multi-candidate
head-to-head (Local Inference & Benchmarking Plan, phase B3); never invoked by
the live pipeline.

Launching the server with ``--mmproj`` and waiting for ``/health`` is also the
**mmproj gate**: if the pinned b9291 build cannot load a model's vision
projector, the server exits during load and that surfaces here as a startup
error.

Stop the supervisor daemon first (``sol supervisor`` / ``sol up``) so this
server has the GPU to itself; otherwise the two contend for memory.

Usage::

    # Installed model (resolves artifacts from the provider's spec/cache):
    python -m solstone.think.benchmark.standalone \\
        --model local/nemotron-3-nano-omni --class dgx-spark

    # Candidate not yet in the spec table — point at downloaded artifacts:
    python -m solstone.think.benchmark.standalone --model local/qwen3.6-35b-a3b \\
        --gguf /path/model.gguf --mmproj /path/mmproj.gguf --class dgx-spark
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from solstone.think.benchmark import harness

_HOST = "127.0.0.1"
_READY_TIMEOUT_S = 600.0  # large GGUFs can take minutes to load + warm the GPU
_HEALTH_POLL_S = 2.0
# Representative head-to-head tasks: a foreground cogitate turn, an agentic
# turn, a background generate extraction, and the vision frame. Override with
# --tasks, or sweep everything applicable with --all-tasks.
_DEFAULT_TASKS = ("chat_reply", "agent_turn", "entity_extraction", "screen_frame")


def _resolve_binary() -> Path:
    """Path to the bundled llama-server for this host (model-independent)."""
    from solstone.think.models import LOCAL_MODEL
    from solstone.think.providers import local_install

    readiness = local_install.inspect_readiness(LOCAL_MODEL)
    binary = readiness.get("binary_path")
    if not binary or not Path(binary).exists():
        raise SystemExit(
            "Bundled llama-server binary not installed. Install the local bundle "
            "first (e.g. run `sol call benchmark profile` then a --pull, or start "
            "the supervisor once)."
        )
    return Path(binary)


def _resolve_artifacts(
    model_id: str, gguf: str | None, mmproj: str | None
) -> tuple[Path, Path | None]:
    """Return (gguf_path, mmproj_path|None) for the model.

    Explicit ``--gguf`` wins (required for candidates not in the provider's spec
    table). Otherwise resolve the installed artifacts from the local bundle.
    """
    if gguf:
        gguf_path = Path(gguf)
        if not gguf_path.exists():
            raise SystemExit(f"--gguf not found: {gguf_path}")
        return gguf_path, (Path(mmproj) if mmproj else None)

    from solstone.think.providers import local_install

    readiness = local_install.inspect_readiness(model_id)
    model_path = readiness.get("model_path")
    if not readiness.get("gguf_installed") or not model_path:
        raise SystemExit(
            f"{model_id} is not installed and no --gguf given. Pass --gguf "
            "(and --mmproj for vision) pointing at downloaded artifacts."
        )
    mmproj_path = readiness.get("mmproj_path")
    return Path(model_path), (Path(mmproj_path) if mmproj_path else None)


def launch_server(
    binary: Path, gguf: Path, mmproj: Path | None, model_id: str, port: int
) -> subprocess.Popen[str]:
    """Launch a standalone llama-server, mirroring the supervisor's flags."""
    cmd = [
        str(binary),
        "-m",
        str(gguf),
        "--alias",
        model_id,
        "--host",
        _HOST,
        "--port",
        str(port),
        "--jinja",
    ]
    if mmproj is not None:
        cmd += ["--mmproj", str(mmproj)]
    print(f"launching: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def wait_for_health(base_url: str, proc: subprocess.Popen[str]) -> None:
    """Block until the server reports healthy, or raise with its load log.

    A non-zero exit during load is the mmproj-gate / arch-unsupported signal:
    the pinned b9291 build could not load this gguf/mmproj pair.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = (proc.stdout.read() if proc.stdout else "") or ""
            raise SystemExit(
                f"llama-server exited (code {proc.returncode}) during load — "
                f"the pinned b9291 build may not support this gguf/mmproj.\n"
                f"--- last server output ---\n{tail[-3000:]}"
            )
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(_HEALTH_POLL_S)
    raise SystemExit(f"llama-server did not become healthy within {_READY_TIMEOUT_S}s")


def _median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def _measure_tok_s(model_id: str, base_url: str, vision: bool) -> dict[str, Any]:
    mode = "vision" if vision else "text_only"
    for i in range(harness._WARMUP_RUNS):
        print(f"  [{mode}] warmup {i + 1}/{harness._WARMUP_RUNS}…", file=sys.stderr)
        harness.run_once(model_id, vision=vision, base_url=base_url)
    outs: list[float] = []
    prompts: list[float] = []
    for i in range(harness._MEASURE_RUNS):
        res = harness.run_once(model_id, vision=vision, base_url=base_url)
        out_rate, prompt_rate = harness._bench_tok_s(res)
        outs.append(out_rate)
        prompts.append(prompt_rate)
        print(
            f"  [{mode}] run {i + 1}/{harness._MEASURE_RUNS}: "
            f"output {out_rate:.1f} tok/s, prompt {prompt_rate:.1f} tok/s",
            file=sys.stderr,
        )
    return {
        "output_tok_s": round(_median(outs), 1),
        "prompt_tok_s": round(_median(prompts), 1),
        "measured_at": date.today().isoformat(),
        "mode": mode,
    }


def _measure_task(model_id: str, base_url: str, task_id: str) -> dict[str, Any] | None:
    task_spec = harness._load_task(task_id)
    task_mode = task_spec.get("mode")
    if task_mode == "audio":
        print(
            f"  [{task_id}] skipped (audio unsupported on the bundle)", file=sys.stderr
        )
        return None
    fixture = harness._load_fixture(task_id)
    vision = task_mode == "vision"
    expected = int(task_spec.get("output_tokens") or 400)
    for _ in range(harness._WARMUP_RUNS):
        harness.run_once(
            model_id,
            vision=vision,
            prompt_override=fixture,
            max_output_tokens=expected,
            base_url=base_url,
        )
    elapsed: list[float] = []
    ptoks: list[float] = []
    otoks: list[float] = []
    for _ in range(harness._MEASURE_RUNS):
        res = harness.run_once(
            model_id,
            vision=vision,
            prompt_override=fixture,
            max_output_tokens=expected,
            base_url=base_url,
        )
        elapsed.append(res.get("elapsed_s") or 0.0)
        ptoks.append(float(res.get("prompt_tokens") or 0))
        otoks.append(float(res.get("output_tokens") or 0))
    secs = round(_median(elapsed), 2)
    print(f"  [{task_id}] {secs}s (mode={task_mode})", file=sys.stderr)
    return {
        "seconds": secs,
        "prompt_tokens": int(_median(ptoks)),
        "output_tokens": int(_median(otoks)),
        "measured_at": date.today().isoformat(),
    }


def benchmark(
    model_id: str,
    hw_class: str,
    gguf: Path,
    mmproj: Path | None,
    tasks: list[str],
) -> dict[str, Any]:
    """Launch the model standalone, measure, tear down; return the snippet."""
    from solstone.think.utils import find_available_port

    binary = _resolve_binary()
    port = find_available_port(_HOST)
    base_url = f"http://{_HOST}:{port}"
    has_vision = mmproj is not None

    proc = launch_server(binary, gguf, mmproj, model_id, port)
    try:
        print("waiting for /health (load can take minutes)…", file=sys.stderr)
        wait_for_health(base_url, proc)
        print("server healthy — measuring…", file=sys.stderr)

        entry: dict[str, Any] = _measure_tok_s(model_id, base_url, vision=False)
        if has_vision:
            entry["vision"] = _measure_tok_s(model_id, base_url, vision=True)

        task_results: dict[str, Any] = {}
        for task_id in tasks:
            result = _measure_task(model_id, base_url, task_id)
            if result is not None:
                task_results[task_id] = result
        if task_results:
            entry["tasks"] = task_results
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return {hw_class: entry}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone per-candidate llama-server benchmark (fork-only)."
    )
    parser.add_argument("--model", required=True, help="Model id, e.g. local/...")
    parser.add_argument(
        "--class", dest="hw_class", required=True, help="Hardware class key."
    )
    parser.add_argument(
        "--gguf", help="Path to the model GGUF (required for non-installed candidates)."
    )
    parser.add_argument(
        "--mmproj", help="Path to the vision projector GGUF (enables vision)."
    )
    parser.add_argument(
        "--tasks", help="Comma-separated task ids (default: a representative set)."
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Benchmark every non-audio task in tasks.json.",
    )
    args = parser.parse_args()

    gguf_path, mmproj_path = _resolve_artifacts(args.model, args.gguf, args.mmproj)

    if args.all_tasks:
        from solstone.think.benchmark.estimate import load_tasks

        catalog = load_tasks().get("tasks", {})
        tasks = [tid for tid, spec in catalog.items() if spec.get("mode") != "audio"]
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = list(_DEFAULT_TASKS)

    snippet = benchmark(args.model, args.hw_class, gguf_path, mmproj_path, tasks)
    print(
        f"\n# Paste into think/benchmark/models.json "
        f"-> models['{args.model}'].benchmarks:",
        file=sys.stderr,
    )
    print(json.dumps(snippet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
