# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Estimator: map (hardware class, model) -> expected output tok/s.

When a measurement exists for the user's exact hardware class, that wins.
Otherwise the estimator picks the closest class in the model's benchmark
table (by ``fp16_tflops * mem_bandwidth_gbs``) and scales the measured
tok/s by the ratio. If the model has no measurements at all — or the
user is ``cpu-only`` and the model requires VRAM — the estimate is
returned with ``confidence="unknown"``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent
_REFERENCE_FILE = _DATA_DIR / "reference.json"
_REGISTRY_FILE = _DATA_DIR / "models.json"
_TASKS_FILE = _DATA_DIR / "tasks.json"


Confidence = Literal["measured", "interpolated", "unknown"]


@dataclass(frozen=True)
class Estimate:
    """Single-model speed estimate."""

    model_id: str
    hardware_class: str
    tok_s: float | None
    confidence: Confidence
    source_class: str | None


@dataclass(frozen=True)
class TaskEstimate:
    """Wall-clock time estimate for a specific task on a specific model."""

    model_id: str
    task_id: str
    hardware_class: str
    seconds: float | None
    confidence: Confidence
    source_class: str | None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_reference() -> dict[str, Any]:
    """Load ``reference.json`` (cached)."""
    return json.loads(_REFERENCE_FILE.read_text())


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    """Load ``models.json`` (cached)."""
    return json.loads(_REGISTRY_FILE.read_text())


@lru_cache(maxsize=1)
def load_tasks() -> dict[str, Any]:
    """Load ``tasks.json`` (cached)."""
    return json.loads(_TASKS_FILE.read_text())


# ---------------------------------------------------------------------------
# Hardware class resolution
# ---------------------------------------------------------------------------


def resolve_hardware_class(gpu_name: str | None) -> str:
    """Map an ``nvidia-smi`` GPU name to a canonical hardware class.

    Tries (1) exact alias match, (2) case-insensitive substring match
    against class labels, (3) fallback to ``"cpu-only"``.
    """
    if not gpu_name:
        return "cpu-only"

    ref = load_reference()
    aliases: dict[str, str] = ref.get("aliases", {})
    classes: dict[str, dict[str, Any]] = ref.get("classes", {})

    if gpu_name in aliases:
        return aliases[gpu_name]

    needle = gpu_name.lower()
    for alias_name, class_key in aliases.items():
        if alias_name.lower() in needle or needle in alias_name.lower():
            return class_key
    for class_key, class_spec in classes.items():
        label = str(class_spec.get("label", "")).lower()
        if label and (label in needle or needle in label):
            return class_key

    return "cpu-only"


def _class_throughput(class_key: str) -> float:
    """Return the TFLOPs × bandwidth product used as the interpolation proxy."""
    classes = load_reference().get("classes", {})
    spec = classes.get(class_key, {})
    return float(spec.get("fp16_tflops", 0.0)) * float(
        spec.get("mem_bandwidth_gbs", 0.0)
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def estimate_output_tok_s(model_id: str, hardware_class: str) -> Estimate:
    """Estimate output tok/s for ``model_id`` on ``hardware_class``."""
    return _estimate_metric_tok_s(model_id, hardware_class, "output_tok_s")


def _estimate_metric_tok_s(model_id: str, hardware_class: str, metric: str) -> Estimate:
    """Generic tok/s estimate for any metric in a model's benchmark entry.

    Used for both ``output_tok_s`` (generation speed) and ``prompt_tok_s``
    (prompt-eval speed, which captures image-encoder cost in vision mode).
    """
    registry = load_registry()
    model = registry.get("models", {}).get(model_id)
    if model is None:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    benchmarks: dict[str, dict[str, Any]] = model.get("benchmarks", {}) or {}

    if hardware_class in benchmarks:
        tok_s = benchmarks[hardware_class].get(metric)
        if isinstance(tok_s, (int, float)):
            return Estimate(
                model_id=model_id,
                hardware_class=hardware_class,
                tok_s=float(tok_s),
                confidence="measured",
                source_class=hardware_class,
            )

    if hardware_class == "cpu-only" or not benchmarks:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    target = _class_throughput(hardware_class)
    if target <= 0:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    best_source: str | None = None
    best_source_throughput: float = 0.0
    best_distance: float = float("inf")
    for source_class, bench in benchmarks.items():
        tok_s = bench.get(metric)
        if not isinstance(tok_s, (int, float)):
            continue
        source_throughput = _class_throughput(source_class)
        if source_throughput <= 0:
            continue
        distance = abs(source_throughput - target)
        if distance < best_distance:
            best_distance = distance
            best_source = source_class
            best_source_throughput = source_throughput

    if best_source is None:
        return Estimate(
            model_id=model_id,
            hardware_class=hardware_class,
            tok_s=None,
            confidence="unknown",
            source_class=None,
        )

    source_tok_s = float(benchmarks[best_source][metric])
    scaled = source_tok_s * (target / best_source_throughput)
    return Estimate(
        model_id=model_id,
        hardware_class=hardware_class,
        tok_s=round(scaled, 1),
        confidence="interpolated",
        source_class=best_source,
    )


def estimate_task_time_s(
    model_id: str, hardware_class: str, task_id: str
) -> TaskEstimate:
    """Estimate wall-clock seconds for ``task_id`` on this model+hardware.

    Resolution order:

    1. **Measured** — if a direct wall-clock measurement exists at
       ``models.json -> models[model].benchmarks[hw_class].tasks[task].seconds``,
       return it. This is the ground-truth source (see
       ``harness.py --task``).
    2. **Interpolated** — use ``prompt_tokens / prompt_tok_s +
       output_tokens / output_tok_s`` with the estimator's fallback
       interpolation across hardware classes. Confidence is the weaker
       of the two underlying tok/s estimates.
    3. **Unknown** — either side can't be estimated.
    """
    tasks = load_tasks().get("tasks", {})
    task = tasks.get(task_id)
    if task is None:
        return TaskEstimate(
            model_id=model_id,
            task_id=task_id,
            hardware_class=hardware_class,
            seconds=None,
            confidence="unknown",
            source_class=None,
        )

    registry = load_registry()
    model = registry.get("models", {}).get(model_id)
    if model is not None:
        bench = (model.get("benchmarks") or {}).get(hardware_class) or {}
        measured = (bench.get("tasks") or {}).get(task_id) or {}
        direct_seconds = measured.get("seconds")
        if isinstance(direct_seconds, (int, float)):
            return TaskEstimate(
                model_id=model_id,
                task_id=task_id,
                hardware_class=hardware_class,
                seconds=float(direct_seconds),
                confidence="measured",
                source_class=hardware_class,
            )

    prompt_tokens = float(task.get("prompt_tokens") or 0)
    output_tokens = float(task.get("output_tokens") or 0)

    prompt_est = _estimate_metric_tok_s(model_id, hardware_class, "prompt_tok_s")
    output_est = _estimate_metric_tok_s(model_id, hardware_class, "output_tok_s")

    if (
        prompt_est.tok_s is None
        or output_est.tok_s is None
        or prompt_est.tok_s <= 0
        or output_est.tok_s <= 0
    ):
        return TaskEstimate(
            model_id=model_id,
            task_id=task_id,
            hardware_class=hardware_class,
            seconds=None,
            confidence="unknown",
            source_class=None,
        )

    seconds = (prompt_tokens / prompt_est.tok_s) + (output_tokens / output_est.tok_s)

    # Formula-derived task times are always "interpolated" at best —
    # "measured" is reserved for direct task-time measurements (handled
    # earlier via the benchmarks[hw].tasks[task].seconds path).
    rank = {"measured": 2, "interpolated": 1, "unknown": 0}
    combined_rank = min(rank[prompt_est.confidence], rank[output_est.confidence])
    combined_conf: Confidence = "interpolated" if combined_rank >= 1 else "unknown"

    # When either leg was interpolated, the source class is the one
    # that was actually interpolated from (prefer output since it's the
    # longer leg usually).
    source_class = (
        output_est.source_class
        if output_est.confidence == "interpolated"
        else prompt_est.source_class
    )

    return TaskEstimate(
        model_id=model_id,
        task_id=task_id,
        hardware_class=hardware_class,
        seconds=round(seconds, 2),
        confidence=combined_conf,
        source_class=source_class,
    )


# ---------------------------------------------------------------------------
# Registry listing
# ---------------------------------------------------------------------------


def list_prevetted_models(hardware: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return each pre-vetted model with estimates + VRAM fit flag.

    Each row carries the raw ``estimate`` (output tok/s) plus a
    ``tasks`` dict mapping task_id → task-time estimate (see
    ``estimate_task_time_s``). Only tasks whose mode matches a
    capability of the model are attached.

    ``hardware`` is the cached probe from ``think.hardware.load_hardware()``;
    pass ``None`` for a hardware-agnostic listing (estimates all unknown).
    """
    hardware_class, user_vram_gb = _user_hardware(hardware)
    registry = load_registry()
    tasks = load_tasks().get("tasks", {})
    rows: list[dict[str, Any]] = []

    for model_id, spec in registry.get("models", {}).items():
        vram_required = float(spec.get("vram_required_gb") or 0)
        estimate = estimate_output_tok_s(model_id, hardware_class)

        capabilities = spec.get("capabilities", []) or []
        task_rows: dict[str, dict[str, Any]] = {}
        for task_id, task_spec in tasks.items():
            if not _task_applies_to_model(task_spec, capabilities):
                continue
            task_est = estimate_task_time_s(model_id, hardware_class, task_id)
            task_rows[task_id] = {
                "label": task_spec.get("label"),
                "seconds": task_est.seconds,
                "confidence": task_est.confidence,
                "ui_priority": task_spec.get("ui_priority"),
                "tier_role": task_spec.get("tier_role"),
            }

        rows.append(
            {
                "model_id": model_id,
                "label": spec.get("label"),
                "tier_hint": spec.get("tier_hint"),
                "size_gb": spec.get("size_gb"),
                "capabilities": capabilities,
                "vram_required_gb": vram_required,
                "fits_in_vram": (user_vram_gb is None or user_vram_gb >= vram_required),
                "notes": spec.get("notes"),
                "estimate": {
                    "tok_s": estimate.tok_s,
                    "confidence": estimate.confidence,
                    "source_class": estimate.source_class,
                    "hardware_class": estimate.hardware_class,
                },
                "tasks": task_rows,
            }
        )
    return rows


def _task_applies_to_model(task_spec: dict[str, Any], capabilities: list[str]) -> bool:
    """A vision task requires a vision capability; other tasks apply broadly."""
    mode = task_spec.get("mode", "text")
    if mode == "vision":
        return "vision" in capabilities
    # Text tasks: apply to any model that can generate text (generate or cogitate).
    return any(cap in ("generate", "cogitate") for cap in capabilities)


def _user_hardware(
    hardware: dict[str, Any] | None,
) -> tuple[str, float | None]:
    """Resolve (hardware_class, effective_vram_gb) from a probe payload.

    On unified-memory systems (Spark, Jetson) the GPU reports no discrete
    VRAM; effective VRAM for fit checks is the system RAM.
    """
    if not hardware:
        return "cpu-only", None
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only", 0.0
    primary = gpus[0]
    hardware_class = resolve_hardware_class(primary.get("name"))

    has_unified = any(g.get("unified_memory") for g in gpus)
    if has_unified:
        # Use system RAM as the memory ceiling; keep a small reserve for the OS.
        ram_gb = float(hardware.get("ram_gb") or 0)
        effective_vram = max(ram_gb - 8.0, 0.0) if ram_gb else None
        return hardware_class, effective_vram

    total_vram = sum(float(g.get("vram_gb") or 0) for g in gpus)
    return hardware_class, total_vram
