# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Fork-only vLLM advisory checks for sol doctor (linux only).

Kept out of ``solstone.think.doctor`` so upstream merges don't collide
in the middle of that file. ``doctor`` imports ``VLLM_CHECKS`` near the
end of its module init and extends its registry; ``doctor`` is the
authoritative entry point.

This module is fork-only and intentionally has no upstream counterpart.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from solstone.think.doctor import (
    Args,
    Check,
    CheckResult,
    ROOT,
    make_result,
    run_probe,
    unexpected_output_result,
)


def _load_vllm_servers() -> tuple[str, dict | None]:
    """Resolve providers.vllm.servers from the journal config.

    Stdlib-only: looks at ``$SOLSTONE_JOURNAL`` first, then a source-tree
    journal at ``<repo>/journal``. Returns one of:

    - ``("no_journal", None)`` — no config file exists (pre-install, etc.)
    - ``("unreadable", None)`` — file exists but can't be parsed as JSON
    - ``("no_section", None)`` — file exists but no providers.vllm.servers
    - ``("ok", {...})`` — section present (may be an empty dict)
    """
    env_path = os.environ.get("SOLSTONE_JOURNAL")
    journal = Path(env_path) if env_path else ROOT / "journal"
    config_path = journal / "config" / "journal.json"
    if not config_path.is_file():
        return "no_journal", None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unreadable", None
    providers = data.get("providers") if isinstance(data, dict) else None
    vllm = providers.get("vllm") if isinstance(providers, dict) else None
    servers = vllm.get("servers") if isinstance(vllm, dict) else None
    if not isinstance(servers, dict):
        return "no_section", None
    return "ok", servers


def vllm_docker_available_check(args: Args) -> CheckResult:
    del args
    from solstone.think.doctor import CHECK_MAP

    check = CHECK_MAP["vllm_docker_available"]
    if shutil.which("docker") is None:
        return make_result(
            check,
            "warn",
            "docker not on PATH; required only if you intend to run a local vLLM server",
            "install Docker (https://docs.docker.com/engine/install/) and add your user to the docker group",
        )
    return make_result(check, "ok", "docker found on PATH")


def vllm_nvidia_smi_check(args: Args) -> CheckResult:
    del args
    from solstone.think.doctor import CHECK_MAP

    check = CHECK_MAP["vllm_nvidia_smi"]
    if shutil.which("nvidia-smi") is None:
        return make_result(
            check,
            "warn",
            "nvidia-smi not on PATH; install the NVIDIA driver if you intend to run vLLM on this host",
            "install the NVIDIA driver appropriate for this platform (e.g. NVIDIA Container Toolkit on the DGX Spark)",
        )
    probe = run_probe(
        check,
        ["nvidia-smi", "-L"],
        timeout=5.0,
        fix="check the NVIDIA driver install; `nvidia-smi -L` should list at least one GPU",
    )
    if isinstance(probe, CheckResult):
        return probe
    gpu_lines = [
        line for line in probe.stdout.splitlines() if line.strip().startswith("GPU ")
    ]
    if not gpu_lines:
        return unexpected_output_result(
            check,
            probe.stdout,
            fix="check the NVIDIA driver install; `nvidia-smi -L` should list at least one GPU",
        )
    return make_result(
        check,
        "ok",
        f"nvidia-smi reports {len(gpu_lines)} GPU(s)",
    )


def vllm_servers_reachable_check(args: Args) -> CheckResult:
    del args
    from solstone.think.doctor import CHECK_MAP

    check = CHECK_MAP["vllm_servers_reachable"]
    state, servers = _load_vllm_servers()
    if state == "no_journal":
        return make_result(check, "skip", "no journal config found")
    if state == "unreadable":
        return make_result(check, "skip", "journal config not readable as JSON")
    if state == "no_section":
        return make_result(check, "skip", "no providers.vllm.servers configured")
    if not servers:
        return make_result(check, "skip", "providers.vllm.servers is empty")
    unreachable: list[str] = []
    reachable: list[str] = []
    for name, entry in sorted(servers.items()):
        base_url = (entry or {}).get("base_url") if isinstance(entry, dict) else None
        if not isinstance(base_url, str) or not base_url:
            unreachable.append(f"{name} (no base_url)")
            continue
        url = base_url.rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if 200 <= resp.status < 300:
                    reachable.append(name)
                else:
                    unreachable.append(f"{name} (HTTP {resp.status})")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            unreachable.append(f"{name} ({type(exc).__name__})")
    if unreachable:
        return make_result(
            check,
            "warn",
            f"unreachable: {', '.join(unreachable)}"
            + (f"; reachable: {', '.join(reachable)}" if reachable else ""),
            "start the supervisor (`sol supervisor`) or run `sol call vllm serve` manually",
        )
    return make_result(
        check,
        "ok",
        f"all {len(reachable)} configured server(s) reachable: {', '.join(reachable)}",
    )


VLLM_CHECKS: list = [
    (
        Check("vllm_docker_available", "advisory", ("linux",)),
        vllm_docker_available_check,
    ),
    (
        Check("vllm_nvidia_smi", "advisory", ("linux",)),
        vllm_nvidia_smi_check,
    ),
    (
        Check("vllm_servers_reachable", "advisory", ("linux",)),
        vllm_servers_reachable_check,
    ),
]
