# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark CLI — local-model speed heuristics for the bundled local (llama.cpp) provider.

Verbs:

- ``sol call benchmark profile`` — probe host hardware, cache result.
- ``sol call benchmark list-models`` — pre-vetted + installed models with
  estimated output tok/s and task-time heuristics.
- ``sol call benchmark estimate <model-id>`` — single-model tok/s estimate.
  With ``--task <task_id>``, returns a wall-clock estimate for that task.
- ``sol call benchmark tasks`` — show the reference-task catalog.

Writes only to ``journal/health/hardware.json`` (via ``solstone.think.hardware``);
the pre-vetted registry, reference, and task tables are in-repo static data.
"""

from __future__ import annotations

import json as jsonlib
import logging
from typing import Any

import typer

from solstone.think.benchmark import (
    estimate_group_fit,
    estimate_output_tok_s,
    estimate_segment_time_s,
    estimate_task_time_s,
    list_prevetted_models,
    load_registry,
    load_segments,
    load_tasks,
    resolve_hardware_class,
    resolve_memory_budget_gb,
)
from solstone.think.hardware import load_hardware, probe_hardware

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
    scenario: str = typer.Option(
        "solo_active",
        "--scenario",
        help=(
            "Scenario from segment.json used for the headline 5-min segment "
            "estimate. Defaults to solo_active."
        ),
    ),
    transcriber: str | None = typer.Option(
        None,
        "--transcriber",
        help=(
            "STT backend for the audio lane. Defaults to the configured "
            "transcribe.backend from the user's journal config."
        ),
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        help=(
            "Show drill-down columns (tok/s, per-task seconds) alongside "
            "the headline segment-time. Off by default since the segment-time "
            "is the more useful glance metric."
        ),
    ),
) -> None:
    """List pre-vetted models with installed status + segment-time estimates.

    The headline column is **5-min segment time** for the chosen
    scenario, computed using this row's model for whichever tier roles
    it can serve and the smallest registry model for the other tiers.
    Pass ``--detailed`` to also see per-token tok/s and per-task
    seconds (the old default columns).
    """
    hardware = load_hardware()
    installed = _list_installed_models()
    resolved_transcriber = transcriber or _configured_transcriber()

    rows = list_prevetted_models(
        hardware, scenario=scenario, transcriber=resolved_transcriber
    )
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
                    "scenario": scenario,
                    "transcriber": resolved_transcriber,
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

    typer.echo(
        f"Scenario: {scenario}    Transcriber: {resolved_transcriber or '(none)'}"
    )
    typer.echo("")

    if detailed:
        header = (
            f"{'MODEL':40} {'INSTALLED':9} {'TIER':4} {'SIZE':>5} "
            f"{'5-MIN SEGMENT':>14} {'CONF':12} "
            f"{'TOK/S':>6} {'TASK TIMES':40}"
        )
    else:
        header = (
            f"{'MODEL':40} {'INSTALLED':9} {'TIER':4} {'SIZE':>5} "
            f"{'5-MIN SEGMENT':>14} {'CONF':12} {'ATTRIBUTED':20}"
        )
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        seg = row.get("segment_estimate") or {}
        seg_seconds = seg.get("total_seconds")
        seg_str = "?" if seg_seconds is None else _format_seconds(seg_seconds)
        seg_conf = seg.get("confidence", "unknown")
        if detailed:
            est = row["estimate"]
            tok_s = "?" if est["tok_s"] is None else f"{est['tok_s']:.0f}"
            task_summary = _format_task_summary(row.get("tasks", {}))
            typer.echo(
                f"{_truncate(row['model_id'], 40):40} "
                f"{'yes' if row['installed'] else 'no':9} "
                f"{row.get('tier_hint') or '-':<4} "
                f"{row.get('size_gb') or '?':>5} "
                f"{seg_str:>14} "
                f"{seg_conf:12} "
                f"{tok_s:>6} "
                f"{task_summary:40}"
            )
        else:
            attributed = ",".join(seg.get("self_attributed_tiers") or []) or "-"
            typer.echo(
                f"{_truncate(row['model_id'], 40):40} "
                f"{'yes' if row['installed'] else 'no':9} "
                f"{row.get('tier_hint') or '-':<4} "
                f"{row.get('size_gb') or '?':>5} "
                f"{seg_str:>14} "
                f"{seg_conf:12} "
                f"{_truncate(attributed, 20):20}"
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


def _format_lane(seconds: float | None) -> str:
    """Lane row: '?' when unmeasured, formatted seconds otherwise."""
    if seconds is None:
        return "?"
    return _format_seconds(seconds)


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
    model_id: str = typer.Argument(..., help="Model ID, e.g. local/qwen3.6-35b-a3b"),
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
            f"(solstone/think/benchmark/models.json).",
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


@app.command("segment")
def segment(
    scenario: str = typer.Option(
        "solo_active",
        "--scenario",
        help="Scenario from segment.json (e.g. solo_active, meeting_active, idle).",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision",
        help="Model to attribute the vision tier (screen_frame) to.",
    ),
    generate_model: str | None = typer.Option(
        None,
        "--generate",
        help="Model to attribute the generate tier to (entity_extraction etc.).",
    ),
    cogitate_model: str | None = typer.Option(
        None,
        "--cogitate",
        help="Model to attribute the cogitate tier to (pulse, awareness_tender).",
    ),
    transcriber: str | None = typer.Option(
        None,
        "--transcriber",
        help=(
            "STT backend for the audio lane (parakeet / whisper / gemini / "
            "revai). Defaults to the configured 'transcribe.backend' from "
            "the user's journal config."
        ),
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help=(
            "Memory budget in GB Solstone may use on this host (drives the "
            "group-fit check). Defaults to the host default: full machine minus "
            "a small OS reserve on unified-memory hosts, full VRAM otherwise."
        ),
    ),
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Estimate wall-clock seconds to fully process one 5-minute segment.

    Headline semantic benchmark — decomposes into audio (transcriber
    RTF), video (screen-frame × qualified_frames), and per-segment
    talents. Pass one model per tier the scenario actually uses;
    missing tiers are flagged in the notes. The ``group_fit`` block reports
    whether the distinct models in the active group co-resident fit the memory
    budget (a model loaded once serves every tier it fills).
    """
    hardware = load_hardware()
    if hardware is None:
        typer.echo(
            "Hardware not yet probed. Run 'sol call benchmark profile' first.",
            err=True,
        )
        raise typer.Exit(1)

    hardware_class = _resolved_class(hardware)
    tier_models: dict[str, str] = {}
    if vision_model:
        tier_models["vision"] = vision_model
    if generate_model:
        tier_models["generate"] = generate_model
    if cogitate_model:
        tier_models["cogitate"] = cogitate_model

    resolved_transcriber = transcriber or _configured_transcriber()
    est = estimate_segment_time_s(
        tier_models, hardware_class, scenario, transcriber=resolved_transcriber
    )
    budget_gb = resolve_memory_budget_gb(hardware, budget)
    group = estimate_group_fit(tier_models, scenario, budget_gb=budget_gb)

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "scenario": est.scenario,
                    "hardware_class": est.hardware_class,
                    "total_seconds": est.total_seconds,
                    "audio_seconds": est.audio_seconds,
                    "video_seconds": est.video_seconds,
                    "talent_seconds": est.talent_seconds,
                    "overhead_seconds": est.overhead_seconds,
                    "per_talent": est.per_talent,
                    "confidence": est.confidence,
                    "notes": list(est.notes),
                    "tier_models": tier_models,
                    "transcriber": resolved_transcriber,
                    "budget_gb": budget_gb,
                    "group_fit": {
                        "budget_gb": group.budget_gb,
                        "footprint_gb": group.footprint_gb,
                        "fits": group.fits,
                        "per_model_gb": group.per_model_gb,
                        "notes": list(group.notes),
                    },
                },
                indent=2,
            )
        )
        return

    scenarios_catalog = load_segments().get("scenarios", {})
    scenario_label = (scenarios_catalog.get(scenario) or {}).get("label") or scenario
    total = (
        "unknown" if est.total_seconds is None else _format_seconds(est.total_seconds)
    )
    typer.echo(f"Scenario:       {scenario_label} ({scenario})")
    typer.echo(f"Hardware class: {hardware_class}")
    typer.echo(f"Transcriber:    {resolved_transcriber or '(none)'}")
    typer.echo(f"Total:          {total}  ({est.confidence})")
    typer.echo("")
    typer.echo(f"  Audio (5 min):  {_format_lane(est.audio_seconds)}")
    typer.echo(f"  Video frames:   {_format_lane(est.video_seconds)}")
    typer.echo(f"  Talents:        {_format_lane(est.talent_seconds)}")
    typer.echo(f"  Overhead:       {_format_seconds(est.overhead_seconds)}")
    typer.echo("")
    if group.fits is None:
        fit_str = "budget unknown (host not probed)"
    else:
        fit_str = "fits" if group.fits else "OVER budget"
    budget_str = "?" if group.budget_gb is None else f"{group.budget_gb:.0f} GB"
    typer.echo(
        f"  Group footprint: {group.footprint_gb:.1f} GB / {budget_str} budget "
        f"({fit_str})"
    )
    if est.per_talent:
        typer.echo("")
        typer.echo("  Per talent:")
        for task_id, seconds in est.per_talent.items():
            typer.echo(f"    {task_id:28} {_format_seconds(seconds)}")
    if est.notes:
        typer.echo("")
        typer.echo("  Notes:")
        for note in est.notes:
            typer.echo(f"    - {note}")


@app.command("scenarios")
def scenarios(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """List segment scenarios from segment.json."""
    catalog = load_segments().get("scenarios", {})
    if json:
        typer.echo(jsonlib.dumps({"scenarios": catalog}, indent=2))
        return

    header = f"{'SCENARIO':18} {'FRAMES':>6} {'TALENTS':>8} LABEL / DESCRIPTION"
    typer.echo(header)
    typer.echo("-" * len(header))
    for scenario_id, spec in catalog.items():
        frames = spec.get("qualified_frames") or 0
        talents = len(spec.get("talents") or [])
        label = spec.get("label") or scenario_id
        desc = spec.get("description") or ""
        typer.echo(f"{scenario_id:18} {frames:>6} {talents:>8} {label} — {desc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configured_transcriber() -> str | None:
    """Read ``transcribe.backend`` from the user's journal config.

    Returns ``None`` when no journal config is reachable — callers
    should fall through to leaving the audio lane unmeasured. The
    Solstone default is ``parakeet`` when the key is absent (matches
    ``observe/transcribe/main.py``).
    """
    try:
        from solstone.think.utils import get_config
    except ImportError:
        return None
    try:
        config = get_config()
    except Exception as exc:
        logger.debug("get_config() unavailable: %s", exc)
        return None
    backend = (config.get("transcribe") or {}).get("backend")
    if isinstance(backend, str) and backend:
        return backend
    return "parakeet"


def _resolved_class(hardware: dict[str, Any]) -> str:
    """Resolve the hardware-class key from a probe payload."""
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only"
    return resolve_hardware_class(gpus[0].get("name"))


def _list_installed_models() -> set[str]:
    """Return the set of model IDs currently usable across local providers.

    Thin wrapper that delegates to the shared
    ``solstone.think.providers.list_installed_local_models()``. Kept as a
    module-private name so existing test patches on
    ``apps.benchmark.call._list_installed_models`` continue to work.
    """
    from solstone.think.providers import list_installed_local_models

    return list_installed_local_models()


def _truncate(text: str, width: int) -> str:
    """Truncate with ellipsis so long model IDs don't break the table."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"
