# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark CLI — local-model speed heuristics for the Ollama provider.

Verbs:

- ``sol call benchmark profile`` — probe host hardware, cache result.
- ``sol call benchmark list-models`` — pre-vetted + installed models with
  estimated output tok/s and task-time heuristics.
- ``sol call benchmark estimate <model-id>`` — single-model tok/s estimate.
  With ``--task <task_id>``, returns a wall-clock estimate for that task.
- ``sol call benchmark tasks`` — show the reference-task catalog.

Writes only to ``journal/health/hardware.json`` (via ``think.hardware``);
the pre-vetted registry, reference, and task tables are in-repo static data.
"""

from __future__ import annotations

import json as jsonlib
import logging
from typing import Any

import typer

from think.benchmark import (
    estimate_output_tok_s,
    estimate_task_time_s,
    list_prevetted_models,
    load_registry,
    load_tasks,
    resolve_hardware_class,
)
from think.hardware import load_hardware, probe_hardware

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="benchmark",
    help="Estimate local-model performance without running the models.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("profile")
def profile(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Probe CPU / RAM / NVIDIA GPUs and cache the result.

    Writes to ``journal/health/hardware.json``. Safe to re-run.
    """
    payload = probe_hardware()
    hardware_class = _resolved_class(payload)
    payload_out = dict(payload)
    payload_out["hardware_class"] = hardware_class

    if json:
        typer.echo(jsonlib.dumps(payload_out, indent=2))
        return

    cpu = payload.get("cpu", {}) or {}
    typer.echo(f"Platform:       {payload.get('platform', 'unknown')}")
    typer.echo(f"CPU:            {cpu.get('model', 'unknown')}")
    typer.echo(f"  cores/threads: {cpu.get('cores', 0)} / {cpu.get('threads', 0)}")
    typer.echo(f"RAM:            {payload.get('ram_gb', 0)} GB")

    gpus = payload.get("gpus") or []
    if gpus:
        typer.echo("GPUs:")
        for gpu in gpus:
            vram = gpu.get("vram_gb")
            if vram is None or gpu.get("unified_memory"):
                vram_str = "unified memory"
            else:
                vram_str = f"{vram} GB VRAM"
            typer.echo(f"  - {gpu.get('name')}  {vram_str}  driver {gpu.get('driver')}")
    else:
        typer.echo("GPUs:           none detected (CPU-only)")

    typer.echo(f"Hardware class: {hardware_class}")


@app.command("list-models")
def list_models(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """List pre-vetted models with installed status + speed estimates."""
    hardware = load_hardware()
    installed = _list_installed_models()

    rows = list_prevetted_models(hardware)
    for row in rows:
        row["installed"] = row["model_id"] in installed

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "hardware_probed": hardware is not None,
                    "hardware_class": (
                        rows[0]["estimate"]["hardware_class"] if rows else "cpu-only"
                    ),
                    "models": rows,
                },
                indent=2,
            )
        )
        return

    if hardware is None:
        typer.echo(
            "Note: hardware not yet probed — run 'sol call benchmark profile' "
            "for accurate estimates. Showing registry with unknown confidence."
        )
        typer.echo("")

    header = (
        f"{'MODEL':40} {'INSTALLED':9} {'TIER':4} {'SIZE':>5} "
        f"{'TOK/S':>6} {'CONF':12} {'TASK TIMES':40}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        est = row["estimate"]
        tok_s = "?" if est["tok_s"] is None else f"{est['tok_s']:.0f}"
        task_summary = _format_task_summary(row.get("tasks", {}))
        typer.echo(
            f"{_truncate(row['model_id'], 40):40} "
            f"{'yes' if row['installed'] else 'no':9} "
            f"{row.get('tier_hint') or '-':<4} "
            f"{row.get('size_gb') or '?':>5} "
            f"{tok_s:>6} "
            f"{est['confidence']:12} "
            f"{task_summary:40}"
        )


def _format_task_summary(tasks: dict[str, dict[str, Any]]) -> str:
    """Render the top 2 tasks (by ui_priority) as a compact inline string."""
    if not tasks:
        return ""
    sorted_tasks = sorted(
        tasks.items(),
        key=lambda kv: (kv[1].get("ui_priority") or 99, kv[0]),
    )[:2]
    parts: list[str] = []
    for task_id, info in sorted_tasks:
        label = info.get("label") or task_id
        seconds = info.get("seconds")
        if seconds is None:
            parts.append(f"{label}: ?")
        else:
            parts.append(f"{label}: {_format_seconds(seconds)}")
    return ", ".join(parts)


def _format_seconds(seconds: float) -> str:
    """Human-friendly duration: 0.9s / 4s / 1m 20s / 2m / 1h 5m."""
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(round(seconds))}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(round(seconds - minutes * 60))
        if secs == 0:
            return f"{minutes}m"
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int(round((seconds - hours * 3600) / 60))
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}m"


@app.command("estimate")
def estimate(
    model_id: str = typer.Argument(..., help="Model ID, e.g. ollama-local/qwen3.5:9b"),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Task ID from tasks.json for a wall-clock estimate (e.g. chat_reply).",
    ),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Estimate tok/s — or, with ``--task``, wall-clock seconds for a task."""
    hardware = load_hardware()
    if hardware is None:
        typer.echo(
            "Hardware not yet probed. Run 'sol call benchmark profile' first.",
            err=True,
        )
        raise typer.Exit(1)

    registry = load_registry()
    if model_id not in registry.get("models", {}):
        typer.echo(
            f"Model '{model_id}' is not in the pre-vetted registry "
            f"(think/benchmark/models.json).",
            err=True,
        )
        raise typer.Exit(1)

    hardware_class = _resolved_class(hardware)

    if task is not None:
        tasks = load_tasks().get("tasks", {})
        if task not in tasks:
            typer.echo(
                f"Task '{task}' not in tasks.json. Run 'sol call benchmark tasks' "
                f"to see the catalog.",
                err=True,
            )
            raise typer.Exit(1)

        task_est = estimate_task_time_s(model_id, hardware_class, task)
        if json:
            typer.echo(
                jsonlib.dumps(
                    {
                        "model_id": task_est.model_id,
                        "task_id": task_est.task_id,
                        "hardware_class": task_est.hardware_class,
                        "seconds": task_est.seconds,
                        "confidence": task_est.confidence,
                        "source_class": task_est.source_class,
                    },
                    indent=2,
                )
            )
            return

        seconds_str = (
            "unknown" if task_est.seconds is None else _format_seconds(task_est.seconds)
        )
        typer.echo(f"Model:          {model_id}")
        typer.echo(f"Task:           {tasks[task].get('label') or task}")
        typer.echo(f"Hardware class: {hardware_class}")
        typer.echo(f"Estimate:       {seconds_str}")
        typer.echo(f"Confidence:     {task_est.confidence}")
        if task_est.source_class and task_est.source_class != hardware_class:
            typer.echo(f"Source class:   {task_est.source_class} (interpolated)")
        return

    est = estimate_output_tok_s(model_id, hardware_class)

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "model_id": est.model_id,
                    "hardware_class": est.hardware_class,
                    "tok_s": est.tok_s,
                    "confidence": est.confidence,
                    "source_class": est.source_class,
                },
                indent=2,
            )
        )
        return

    tok_s = "unknown" if est.tok_s is None else f"{est.tok_s:.1f} tok/s"
    typer.echo(f"Model:          {model_id}")
    typer.echo(f"Hardware class: {hardware_class}")
    typer.echo(f"Estimate:       {tok_s}")
    typer.echo(f"Confidence:     {est.confidence}")
    if est.source_class and est.source_class != hardware_class:
        typer.echo(f"Source class:   {est.source_class} (interpolated)")


@app.command("tasks")
def tasks(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Show the reference-task catalog (what task-time estimates are based on)."""
    catalog = load_tasks().get("tasks", {})
    if json:
        typer.echo(jsonlib.dumps({"tasks": catalog}, indent=2))
        return

    header = f"{'TASK':22} {'MODE':8} {'ROLE':10} {'PROMPT':>7} {'OUTPUT':>7} LABEL"
    typer.echo(header)
    typer.echo("-" * len(header))
    for task_id, spec in sorted(
        catalog.items(),
        key=lambda kv: (kv[1].get("ui_priority") or 99, kv[0]),
    ):
        typer.echo(
            f"{task_id:22} "
            f"{spec.get('mode') or '-':8} "
            f"{spec.get('tier_role') or '-':10} "
            f"{spec.get('prompt_tokens') or 0:>7} "
            f"{spec.get('output_tokens') or 0:>7} "
            f"{spec.get('label') or ''}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_class(hardware: dict[str, Any]) -> str:
    """Resolve the hardware-class key from a probe payload."""
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only"
    return resolve_hardware_class(gpus[0].get("name"))


def _list_installed_models() -> set[str]:
    """Query Ollama ``/api/tags`` and return installed model IDs with prefix.

    Returns an empty set if Ollama is unreachable (not a hard error — the
    CLI should still work without Ollama running).
    """
    try:
        from think.providers.ollama import _OLLAMA_LOCAL_PREFIX, _get_client
    except ImportError:
        return set()

    try:
        client = _get_client()
        response = client.get("/api/tags", timeout=5.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        logger.debug("Ollama /api/tags unreachable: %s", exc)
        return set()

    installed: set[str] = set()
    for entry in data.get("models", []) or []:
        name = entry.get("name")
        if name:
            installed.add(f"{_OLLAMA_LOCAL_PREFIX}{name}")
    return installed


def _truncate(text: str, width: int) -> str:
    """Truncate with ellipsis so long model IDs don't break the table."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"
