# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Benchmark app HTTP API.

API-only app (no workspace page yet): a thin JSON surface over the fork-only
local-model benchmarking heuristics in ``solstone.think.benchmark`` — the same
data behind the ``sol call benchmark`` CLI, exposed to UI clients. A standalone
benchmark workspace can be layered on later by adding ``workspace.html``.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from solstone.convey.reasons import INVALID_REQUEST_VALUE, SERVICE_OPERATION_FAILED
from solstone.convey.utils import error_response

logger = logging.getLogger(__name__)

benchmark_bp = Blueprint("app:benchmark", __name__, url_prefix="/app/benchmark")


def _fresh_hardware() -> dict[str, Any] | None:
    """Return the host's current hardware probe for benchmark app endpoints.

    Always attempts a fresh probe so a stale cache (e.g. an empty-GPU
    snapshot written during driver warmup at boot) self-heals on the
    next page load. Falls back to the cached probe if a fresh one
    raises. Returns None only if both fail.
    """
    from solstone.think.hardware import load_hardware, probe_hardware

    try:
        return probe_hardware()
    except Exception as exc:
        logger.warning("fresh hardware probe failed; using cache: %s", exc)
        return load_hardware()


def _configured_transcriber() -> str | None:
    """Read ``transcribe.backend`` from journal config; ``None`` on failure.

    Mirrors the helper in ``apps/benchmark/call.py``. Kept here as a
    duplicate (rather than imported across app boundaries) so the
    settings app doesn't grow a dependency on the benchmark app.
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


@benchmark_bp.route("/api/models")
def get_benchmark_models() -> Any:
    """Return pre-vetted local models with task-time estimates.

    Powers the local-model tier annotations + "recommended models you
    don't have yet" section in the providers UI.

    Response shape::

        {
          "hardware": {"probed": bool, "class": str, "label": str | null,
                       "platform": str, "ram_gb": float, "gpus": [...]},
          "tasks": {<task_id>: {label, description, mode, presence,
                                tier_role, ui_priority}},
          "models": [
            {"model_id": ..., "label": ..., "tier_hint": 1|2|3,
             "installed": bool, "fits_in_vram": bool, "size_gb": N,
             "capabilities": [...], "notes": ..., "vram_required_gb": N,
             "estimate": {"tok_s": N, "confidence": str,
                          "hardware_class": str, "source_class": str},
             "tasks": {<task_id>: {label, seconds, confidence,
                                   ui_priority, tier_role}}}
          ]
        }
    """
    try:
        from solstone.think.benchmark import list_prevetted_models, load_tasks
        from solstone.think.benchmark.estimate import load_reference
    except ImportError as exc:
        return error_response(
            SERVICE_OPERATION_FAILED, detail=f"benchmark module unavailable: {exc}"
        )

    hardware = _fresh_hardware()

    rows = list_prevetted_models(hardware)

    # Attach installed flag from the local bundle's cache; models whose
    # weights aren't installed stay installed=False.
    from solstone.think.providers import list_installed_local_models

    installed_ids = list_installed_local_models()
    for row in rows:
        row["installed"] = row["model_id"] in installed_ids

    # Resolve hardware class once for UI convenience.
    hw_class = rows[0]["estimate"]["hardware_class"] if rows else "cpu-only"
    ref_label = load_reference().get("classes", {}).get(hw_class, {}).get("label")

    return jsonify(
        {
            "hardware": {
                "probed": hardware is not None,
                "class": hw_class,
                "label": ref_label,
                "platform": (hardware or {}).get("platform"),
                "ram_gb": (hardware or {}).get("ram_gb"),
                "gpus": (hardware or {}).get("gpus", []),
            },
            "tasks": load_tasks().get("tasks", {}),
            "models": rows,
        }
    )


@benchmark_bp.route("/api/scenarios")
def get_benchmark_scenarios() -> Any:
    """Return the segment-time scenario catalog from ``segment.json``.

    Powers the scenario picker on the providers UI's "Background
    processing" card. Shape mirrors ``segment.json`` directly so the
    UI can render labels / descriptions / qualified_frames /
    talents counts without a second round-trip.
    """
    try:
        from solstone.think.benchmark import load_segments
    except ImportError as exc:
        return error_response(
            SERVICE_OPERATION_FAILED, detail=f"benchmark module unavailable: {exc}"
        )

    return jsonify(load_segments())


@benchmark_bp.route("/api/segment")
def get_benchmark_segment() -> Any:
    """Return a SegmentEstimate for the chosen scenario + tier-model picks.

    Query parameters
    ----------------
    scenario : str, default "solo_active"
        Scenario id from ``segment.json``.
    vision, generate, cogitate : str, optional
        Per-tier model ids. Any tier left unset falls back to the
        smallest registry model with that capability — same comparison
        baseline ``list_prevetted_models`` uses for the per-row
        segment column. This makes the endpoint useful before the
        user has explicitly picked per-tier models.
    transcriber : str, optional
        STT backend for the audio lane. Defaults to
        ``transcribe.backend`` from journal config (typically
        ``parakeet`` or ``whisper``).
    budget : float, optional
        Memory budget in GB Solstone may use on this host. Defaults to
        the host default (full machine minus a small OS reserve on
        unified-memory hosts). Drives ``group_fit``.

    Response shape mirrors ``SegmentEstimate`` plus the resolved
    ``tier_models``, ``transcriber``, ``hardware_class``, the resolved
    ``budget_gb``, and a ``group_fit`` block (whether the segment's whole
    active model group co-resident fits the budget) so the UI knows exactly
    what was estimated.
    """
    try:
        from solstone.think.benchmark import (
            estimate_group_fit,
            estimate_segment_time_s,
            load_registry,
            load_segments,
            resolve_memory_budget_gb,
        )
        from solstone.think.benchmark.estimate import (
            _pick_default_tier_models,
            resolve_hardware_class,
        )
    except ImportError as exc:
        return error_response(
            SERVICE_OPERATION_FAILED, detail=f"benchmark module unavailable: {exc}"
        )

    scenario = request.args.get("scenario", "solo_active")
    if scenario not in (load_segments().get("scenarios") or {}):
        return error_response(
            INVALID_REQUEST_VALUE, detail=f"unknown scenario: {scenario!r}"
        )

    hardware = _fresh_hardware()

    gpus = (hardware or {}).get("gpus") or []
    hardware_class = resolve_hardware_class(gpus[0].get("name")) if gpus else "cpu-only"

    # Build tier_models: explicit query-param picks override the
    # smallest-registry-model baseline.
    tier_models = dict(_pick_default_tier_models(load_registry()))
    for tier in ("vision", "generate", "cogitate"):
        override = request.args.get(tier)
        if override:
            tier_models[tier] = override

    transcriber = request.args.get("transcriber") or _configured_transcriber()

    budget_arg = request.args.get("budget")
    budget_override: float | None = None
    if budget_arg:
        try:
            budget_override = float(budget_arg)
        except ValueError:
            return error_response(
                INVALID_REQUEST_VALUE, detail=f"invalid budget: {budget_arg!r}"
            )
    budget_gb = resolve_memory_budget_gb(hardware, budget_override)

    est = estimate_segment_time_s(
        tier_models, hardware_class, scenario, transcriber=transcriber
    )
    group = estimate_group_fit(tier_models, scenario, budget_gb=budget_gb)

    return jsonify(
        {
            "scenario": est.scenario,
            "hardware_class": est.hardware_class,
            "transcriber": transcriber,
            "tier_models": tier_models,
            "budget_gb": budget_gb,
            "total_seconds": est.total_seconds,
            "audio_seconds": est.audio_seconds,
            "video_seconds": est.video_seconds,
            "talent_seconds": est.talent_seconds,
            "overhead_seconds": est.overhead_seconds,
            "per_talent": est.per_talent,
            "confidence": est.confidence,
            "notes": list(est.notes),
            "group_fit": {
                "budget_gb": group.budget_gb,
                "footprint_gb": group.footprint_gb,
                "fits": group.fits,
                "per_model_gb": group.per_model_gb,
                "notes": list(group.notes),
            },
        }
    )
