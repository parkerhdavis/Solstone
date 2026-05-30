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
_SEGMENTS_FILE = _DATA_DIR / "segment.json"
_TRANSCRIBERS_FILE = _DATA_DIR / "transcribers.json"

# Default OS/headroom reserve on unified-memory hosts (Spark/Jetson), where the
# GPU shares system RAM. The resource budget (resolve_memory_budget_gb) defaults
# to total RAM minus this reserve; an explicit operator budget overrides it. This
# replaces the formerly-hardcoded 8 GB inline in _user_hardware.
_DEFAULT_OS_RESERVE_GB = 8.0


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


@dataclass(frozen=True)
class SegmentEstimate:
    """Estimated wall-clock time to fully process one 5-minute segment.

    Decomposed into three lanes — audio (local STT), video (screen-frame
    description × qualified_frames), and talent (segment-scoped LLM
    talents) — plus a fixed orchestration/decode overhead. ``per_talent``
    maps each talent task_id to its computed seconds (count × per-call
    estimate). ``confidence`` is the weakest leg; if any required leg is
    unknown, ``total_seconds`` is None.
    """

    scenario: str
    hardware_class: str
    total_seconds: float | None
    audio_seconds: float | None
    video_seconds: float | None
    talent_seconds: float | None
    overhead_seconds: float
    per_talent: dict[str, float]
    confidence: Confidence
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GroupFit:
    """Whether the model GROUP for one segment co-resident fits a memory budget.

    Solstone runs several models concurrently across modalities per segment
    (frame-describe + the per-segment talents — STT runs separately). A model
    loaded once serves every tier_role it fills, so ``footprint_gb`` sums each
    *distinct* active model's ``vram_required_gb`` once. ``fits`` is None when no
    budget is resolvable (e.g. cpu-only / un-probed host). The STT backend's
    footprint is not folded in (``transcribers.json`` tracks RTF, not size) —
    called out in ``notes`` rather than silently dropped.
    """

    budget_gb: float | None
    footprint_gb: float
    fits: bool | None
    per_model_gb: dict[str, float]
    notes: tuple[str, ...]


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


@lru_cache(maxsize=1)
def load_segments() -> dict[str, Any]:
    """Load ``segment.json`` (cached)."""
    return json.loads(_SEGMENTS_FILE.read_text())


@lru_cache(maxsize=1)
def load_transcribers() -> dict[str, Any]:
    """Load ``transcribers.json`` (cached)."""
    return json.loads(_TRANSCRIBERS_FILE.read_text())


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
# Segment-time estimation
# ---------------------------------------------------------------------------


_CONFIDENCE_RANK: dict[Confidence, int] = {
    "measured": 2,
    "interpolated": 1,
    "unknown": 0,
}


def _estimate_audio_lane(
    transcriber: str | None,
    audio_seconds_total: float,
    hardware_class: str,
) -> tuple[float | None, Confidence, str | None]:
    """Return ``(audio_seconds, confidence, note_or_None)`` for the audio lane.

    Resolution:

    - ``transcriber=None`` → ``(None, "unknown", explanatory note)``. Callers
      that don't know which backend is configured fall through to this
      pathway; the segment estimator will downgrade total confidence
      accordingly.
    - Local backend with measured RTF for ``hardware_class`` →
      ``rtf * audio_seconds_total``, ``confidence="measured"``.
    - Local backend without a measurement on this class → ``(None,
      "unknown", note)``. RTF doesn't interpolate cleanly across
      hardware classes the way LLM tok/s does, so we don't try.
    - Cloud backend → flat ``wall_seconds_per_5min * (audio/300)``,
      ``confidence="interpolated"`` (rule-of-thumb, not measured here).
    - Unknown transcriber name → ``(None, "unknown", note)``.
    """
    if transcriber is None:
        return (
            None,
            "unknown",
            "audio lane unknown: no transcriber specified (pass --transcriber "
            "or read transcribe.backend from config)",
        )

    transcribers = load_transcribers().get("transcribers", {})
    spec = transcribers.get(transcriber)
    if spec is None:
        return (
            None,
            "unknown",
            f"audio lane unknown: transcriber '{transcriber}' missing from "
            f"transcribers.json",
        )

    kind = spec.get("kind")
    if kind == "local":
        bench = (spec.get("benchmarks") or {}).get(hardware_class) or {}
        rtf = bench.get("rtf")
        if isinstance(rtf, (int, float)):
            return (
                round(float(rtf) * audio_seconds_total, 2),
                "measured",
                None,
            )
        return (
            None,
            "unknown",
            f"audio lane unknown: no RTF measurement for transcriber "
            f"'{transcriber}' on '{hardware_class}' (run "
            f"`python -m solstone.think.benchmark.harness --transcriber {transcriber} "
            f"--audio-fixture <path> --class {hardware_class}`)",
        )

    if kind == "cloud":
        wall_per_5min = spec.get("wall_seconds_per_5min")
        if isinstance(wall_per_5min, (int, float)):
            seconds = float(wall_per_5min) * (audio_seconds_total / 300.0)
            return (
                round(seconds, 2),
                "interpolated",
                f"audio lane uses a flat rule-of-thumb wall-clock for "
                f"cloud backend '{transcriber}' (network-bound, not "
                f"hardware-derived)",
            )
        return (
            None,
            "unknown",
            f"audio lane unknown: cloud transcriber '{transcriber}' has "
            f"no wall_seconds_per_5min in transcribers.json",
        )

    return (
        None,
        "unknown",
        f"audio lane unknown: transcriber '{transcriber}' has unrecognized "
        f"kind '{kind}'",
    )


def estimate_segment_time_s(
    tier_models: dict[str, str],
    hardware_class: str,
    scenario: str = "solo_active",
    transcriber: str | None = None,
) -> SegmentEstimate:
    """Estimate wall-clock seconds to process one 5-min segment.

    ``tier_models`` maps a task's ``tier_role`` (``vision`` / ``generate``
    / ``cogitate``) to the model_id that handles that tier on this host.
    The segment recipe (frame count, talent list, fixed overhead) comes
    from ``segment.json``. The returned ``SegmentEstimate`` exposes a
    per-lane breakdown so callers can render the headline number plus
    the structural detail behind it.

    The audio lane resolves via ``transcriber``: pass the configured
    backend name (``parakeet`` / ``whisper`` / ``gemini`` / ``revai``)
    and the estimator looks up RTF (local) or wall-seconds-per-5min
    (cloud) from ``transcribers.json``. Pass ``None`` to opt out and
    leave the audio lane unmeasured (downgrades total confidence to
    ``unknown``).
    """
    segments = load_segments().get("scenarios", {})
    spec = segments.get(scenario)
    if spec is None:
        return SegmentEstimate(
            scenario=scenario,
            hardware_class=hardware_class,
            total_seconds=None,
            audio_seconds=None,
            video_seconds=None,
            talent_seconds=None,
            overhead_seconds=0.0,
            per_talent={},
            confidence="unknown",
            notes=(f"unknown scenario '{scenario}'",),
        )

    tasks_catalog = load_tasks().get("tasks", {})
    notes: list[str] = []
    leg_confidences: list[Confidence] = []

    # --- Video lane ---------------------------------------------------------
    qualified_frames = int(spec.get("qualified_frames") or 0)
    video_seconds: float | None = 0.0
    if qualified_frames > 0:
        screen_frame_spec = tasks_catalog.get("screen_frame", {})
        vision_model = tier_models.get(screen_frame_spec.get("tier_role") or "vision")
        if vision_model is None:
            video_seconds = None
            leg_confidences.append("unknown")
            notes.append(
                "video lane unknown: no vision-tier model provided for screen_frame"
            )
        else:
            per_frame = estimate_task_time_s(
                vision_model, hardware_class, "screen_frame"
            )
            if per_frame.seconds is None:
                video_seconds = None
                leg_confidences.append("unknown")
                notes.append(
                    f"video lane unknown: no estimate for screen_frame on "
                    f"{vision_model}"
                )
            else:
                video_seconds = round(per_frame.seconds * qualified_frames, 2)
                leg_confidences.append(per_frame.confidence)

    # --- Talent lane --------------------------------------------------------
    per_talent: dict[str, float] = {}
    talent_seconds: float | None = 0.0
    talent_unknown = False
    for entry in spec.get("talents", []) or []:
        task_id = entry.get("task_id")
        count = int(entry.get("count") or 1)
        task_spec = tasks_catalog.get(task_id)
        if task_spec is None:
            talent_unknown = True
            leg_confidences.append("unknown")
            notes.append(f"talent '{task_id}' missing from tasks.json")
            continue
        tier_role = task_spec.get("tier_role")
        model_id = tier_models.get(tier_role)
        if model_id is None:
            talent_unknown = True
            leg_confidences.append("unknown")
            notes.append(
                f"talent '{task_id}' unknown: no '{tier_role}'-tier model provided"
            )
            continue
        per_call = estimate_task_time_s(model_id, hardware_class, task_id)
        if per_call.seconds is None:
            talent_unknown = True
            leg_confidences.append("unknown")
            notes.append(f"talent '{task_id}' unknown: no estimate on {model_id}")
            continue
        seconds = round(per_call.seconds * count, 2)
        per_talent[task_id] = seconds
        leg_confidences.append(per_call.confidence)

    if talent_unknown:
        talent_seconds = None
    else:
        talent_seconds = round(sum(per_talent.values()), 2)

    # --- Audio lane ---------------------------------------------------------
    audio_minutes = float(spec.get("audio_minutes") or 5)
    audio_seconds_total = audio_minutes * 60.0
    audio_seconds, audio_conf, audio_note = _estimate_audio_lane(
        transcriber, audio_seconds_total, hardware_class
    )
    leg_confidences.append(audio_conf)
    if audio_note:
        notes.append(audio_note)

    overhead_seconds = float(spec.get("fixed_overhead_s") or 0.0)

    if audio_seconds is None or video_seconds is None or talent_seconds is None:
        total_seconds: float | None = None
    else:
        total_seconds = round(
            audio_seconds + video_seconds + talent_seconds + overhead_seconds, 2
        )

    if not leg_confidences:
        combined: Confidence = "unknown"
    else:
        worst_rank = min(_CONFIDENCE_RANK[c] for c in leg_confidences)
        combined = next(c for c, rank in _CONFIDENCE_RANK.items() if rank == worst_rank)

    return SegmentEstimate(
        scenario=scenario,
        hardware_class=hardware_class,
        total_seconds=total_seconds,
        audio_seconds=audio_seconds,
        video_seconds=video_seconds,
        talent_seconds=talent_seconds,
        overhead_seconds=overhead_seconds,
        per_talent=per_talent,
        confidence=combined,
        notes=tuple(notes),
    )


def estimate_group_fit(
    tier_models: dict[str, str],
    scenario: str = "solo_active",
    *,
    budget_gb: float | None = None,
) -> GroupFit:
    """Estimate whether the segment's active model group fits ``budget_gb``.

    The active group is the set of *distinct* models ``tier_models`` maps the
    scenario's live lanes to — the vision model (when the scenario describes
    frames) plus the model behind each per-segment talent's ``tier_role``. A
    model that fills several roles (e.g. the single served local model) counts
    once. Footprint is each distinct model's ``vram_required_gb`` from the
    registry; ``budget_gb`` is the resolved resource budget (see
    ``resolve_memory_budget_gb``).
    """
    spec = load_segments().get("scenarios", {}).get(scenario) or {}
    tasks_catalog = load_tasks().get("tasks", {})
    registry_models = load_registry().get("models", {})

    active: set[str] = set()
    if int(spec.get("qualified_frames") or 0) > 0:
        vision_model = tier_models.get("vision")
        if vision_model:
            active.add(vision_model)
    for entry in spec.get("talents", []) or []:
        task_spec = tasks_catalog.get(entry.get("task_id")) or {}
        model_id = tier_models.get(task_spec.get("tier_role"))
        if model_id:
            active.add(model_id)

    per_model_gb = {
        model_id: float(
            registry_models.get(model_id, {}).get("vram_required_gb") or 0.0
        )
        for model_id in sorted(active)
    }
    footprint_gb = round(sum(per_model_gb.values()), 1)
    fits = None if budget_gb is None else footprint_gb <= budget_gb
    return GroupFit(
        budget_gb=budget_gb,
        footprint_gb=footprint_gb,
        fits=fits,
        per_model_gb=per_model_gb,
        notes=(
            "footprint sums distinct active-group models' vram_required_gb "
            "(a model loaded once serves every tier_role it fills); the STT "
            "backend footprint is not included (transcribers.json tracks RTF, "
            "not size)",
        ),
    )


# ---------------------------------------------------------------------------
# Registry listing
# ---------------------------------------------------------------------------


def _pick_default_tier_models(registry: dict[str, Any]) -> dict[str, str]:
    """Pick the smallest model per tier-role from the registry.

    Used as the comparison baseline in ``list_prevetted_models``: each
    row's own model takes the tier slot(s) it has capabilities for, and
    every other tier defaults to whatever's smallest (cheapest) and
    capable in the registry. That makes the per-row segment-time number
    a like-for-like comparison — only this model varies, the rest of
    the pipeline is held constant.

    The registry uses ``capabilities`` matching the ``tier_role`` field
    in ``tasks.json``: ``vision`` / ``generate`` / ``cogitate``.
    """
    by_tier: dict[str, list[tuple[float, str]]] = {}
    for model_id, spec in (registry.get("models") or {}).items():
        size = float(spec.get("size_gb") or float("inf"))
        for cap in spec.get("capabilities") or []:
            by_tier.setdefault(cap, []).append((size, model_id))
    return {
        tier: sorted(candidates)[0][1]
        for tier, candidates in by_tier.items()
        if candidates
    }


def list_prevetted_models(
    hardware: dict[str, Any] | None,
    *,
    scenario: str = "solo_active",
    transcriber: str | None = None,
    budget_gb: float | None = None,
) -> list[dict[str, Any]]:
    """Return each pre-vetted model with estimates + budget fit flags.

    Each row carries the raw ``estimate`` (output tok/s), the
    per-task ``tasks`` dict (see ``estimate_task_time_s``), **and** a
    ``segment_estimate`` populated by ``estimate_segment_time_s`` — which now
    also carries a ``group_fit`` block (see ``estimate_group_fit``): whether the
    segment's whole active model group co-resident fits the resolved budget.

    ``budget_gb`` is the resource budget Solstone may use on this host (see
    ``resolve_memory_budget_gb``); ``None`` uses the host default. Both the
    per-row ``fits_in_vram`` flag and the segment ``group_fit`` are evaluated
    against it.

    The segment estimate uses the row's own model for whichever tier
    roles it can serve (``vision`` / ``generate`` / ``cogitate``) and
    falls back to the smallest registry model for the other tiers — so
    each row is a like-for-like comparison where only this model
    varies. ``self_attributed_tiers`` records which tiers came from
    this row's model vs. the comparison baseline.

    ``hardware`` is the cached probe from ``solstone.think.hardware.load_hardware()``;
    pass ``None`` for a hardware-agnostic listing (estimates all unknown).
    ``transcriber`` selects the audio-lane backend; pass ``None`` to
    leave the audio lane unmeasured (downgrades segment confidence to
    ``unknown``).
    """
    hardware_class, user_budget_gb = _user_hardware(hardware, budget_gb)
    registry = load_registry()
    tasks = load_tasks().get("tasks", {})
    default_tier_models = _pick_default_tier_models(registry)
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

        # Build the per-row tier-model map: this model fills the slots
        # it can serve, defaults fill the rest.
        tier_models: dict[str, str] = dict(default_tier_models)
        self_attributed: list[str] = []
        for cap in capabilities:
            if cap in tier_models:
                tier_models[cap] = model_id
                self_attributed.append(cap)

        seg = estimate_segment_time_s(
            tier_models, hardware_class, scenario, transcriber=transcriber
        )
        group = estimate_group_fit(tier_models, scenario, budget_gb=user_budget_gb)

        rows.append(
            {
                "model_id": model_id,
                "label": spec.get("label"),
                # served == the one model think/models.py pins as LOCAL_MODEL;
                # candidate rows are evaluation-only and the UI must not present
                # them as switchable (see models.json _comment).
                "served": bool(spec.get("served")),
                "tier_hint": spec.get("tier_hint"),
                "size_gb": spec.get("size_gb"),
                "capabilities": capabilities,
                "vram_required_gb": vram_required,
                "fits_in_vram": (
                    user_budget_gb is None or user_budget_gb >= vram_required
                ),
                "notes": spec.get("notes"),
                "estimate": {
                    "tok_s": estimate.tok_s,
                    "confidence": estimate.confidence,
                    "source_class": estimate.source_class,
                    "hardware_class": estimate.hardware_class,
                },
                "tasks": task_rows,
                "segment_estimate": {
                    "scenario": seg.scenario,
                    "total_seconds": seg.total_seconds,
                    "audio_seconds": seg.audio_seconds,
                    "video_seconds": seg.video_seconds,
                    "talent_seconds": seg.talent_seconds,
                    "overhead_seconds": seg.overhead_seconds,
                    "confidence": seg.confidence,
                    "tier_models": tier_models,
                    "self_attributed_tiers": self_attributed,
                    "notes": list(seg.notes),
                    "group_fit": {
                        "budget_gb": group.budget_gb,
                        "footprint_gb": group.footprint_gb,
                        "fits": group.fits,
                        "per_model_gb": group.per_model_gb,
                        "notes": list(group.notes),
                    },
                },
            }
        )
    return rows


def _task_applies_to_model(task_spec: dict[str, Any], capabilities: list[str]) -> bool:
    """Gate tasks by model capability.

    Vision tasks require the ``vision`` capability; audio tasks require
    the ``audio`` capability. Text tasks apply broadly to any model that
    can generate text (``generate`` or ``cogitate``).
    """
    mode = task_spec.get("mode", "text")
    if mode == "vision":
        return "vision" in capabilities
    if mode == "audio":
        return "audio" in capabilities
    # Text tasks: apply to any model that can generate text (generate or cogitate).
    return any(cap in ("generate", "cogitate") for cap in capabilities)


def _machine_memory(
    hardware: dict[str, Any] | None,
) -> tuple[str, float | None, bool]:
    """Resolve (hardware_class, total_usable_memory_gb, is_unified).

    ``total_usable_memory_gb`` is system RAM on unified-memory hosts (Spark,
    Jetson — the GPU shares RAM and reports no discrete VRAM) or summed discrete
    VRAM otherwise. ``None`` means an un-probed host or cpu-only with no GPU.
    """
    if not hardware:
        return "cpu-only", None, False
    gpus = hardware.get("gpus") or []
    if not gpus:
        return "cpu-only", 0.0, False
    hardware_class = resolve_hardware_class(gpus[0].get("name"))
    if any(g.get("unified_memory") for g in gpus):
        ram_gb = float(hardware.get("ram_gb") or 0)
        return hardware_class, (ram_gb if ram_gb else None), True
    total_vram = sum(float(g.get("vram_gb") or 0) for g in gpus)
    return hardware_class, total_vram, False


def resolve_memory_budget_gb(
    hardware: dict[str, Any] | None,
    budget_gb: float | None = None,
) -> float | None:
    """How much memory Solstone may use on this host — the first-class budget.

    ``budget_gb`` is an explicit operator cap ("how much of this machine to give
    the service"), clamped to the machine total. When ``None``, the default
    budget is the full machine minus a small OS reserve on unified-memory hosts
    (``_DEFAULT_OS_RESERVE_GB``), or the full discrete VRAM otherwise. Returns
    ``None`` when no memory is resolvable (un-probed / cpu-only-no-GPU host),
    the "ceiling unknown" sentinel fit checks already understand.
    """
    _, total, is_unified = _machine_memory(hardware)
    if budget_gb is not None:
        capped = float(budget_gb)
        if total is not None:
            capped = min(capped, total)
        return max(capped, 0.0)
    if total is None:
        return None
    if is_unified:
        return max(total - _DEFAULT_OS_RESERVE_GB, 0.0)
    return total


def _user_hardware(
    hardware: dict[str, Any] | None,
    budget_gb: float | None = None,
) -> tuple[str, float | None]:
    """Resolve (hardware_class, memory_budget_gb) from a probe payload.

    The second element is the memory ceiling used for fit checks — Solstone's
    resolved resource budget for this host (see ``resolve_memory_budget_gb``),
    not the raw machine total. ``budget_gb`` threads an explicit operator cap
    through; ``None`` uses the host default.
    """
    hardware_class, _, _ = _machine_memory(hardware)
    return hardware_class, resolve_memory_budget_gb(hardware, budget_gb)
