# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""vLLM CLI — manage vLLM containers configured in journal.json.

Verbs:

- ``sol call vllm list`` — print configured vLLM servers.
- ``sol call vllm status`` — ping each configured server's ``/v1/models``
  and report up/down + loaded models.
- ``sol call vllm serve [--name <friendly>] [-v]`` — spawn the docker
  container for a configured server in the foreground. Streams container
  logs to stdout so the supervisor's ``DailyLogWriter`` captures them.
  On SIGINT/SIGTERM, ``docker stop``\\s the container and waits for
  graceful exit.

Configuration lives at ``journal.json → providers.vllm.servers``. Each
entry is a map keyed by *friendly name* (the part after ``vllm-local/``
in a model id), and supports these fields:

  base_url            (required) HTTP base URL the provider routes to,
                      e.g. ``http://localhost:8000``. Port is parsed from
                      this when ``port`` is omitted.
  served_model_name   (required) Value of ``--served-model-name`` passed
                      to ``vllm serve``. The provider sends this as the
                      ``model`` field in chat-completions requests.
  model               (required) The model the container loads. Either an
                      HF repo path (``Qwen/Qwen3.5-9B``) or an in-container
                      local path. Becomes the positional argument to
                      ``vllm serve``.
  port                (optional) Host port to publish the container on.
                      Defaults to the port parsed from base_url.
  image               (optional) Container image. Defaults to
                      ``vllm/vllm-openai:v0.20.0``.
  vllm_args           (optional) List of additional flags appended to
                      ``vllm serve <model> --port 8000 --served-model-name <name>``.
                      Per-model parser/quant/multimodal-quota flags go here.
  needs_audio_extras  (optional, bool) When true, prepends
                      ``pip install --no-cache-dir vllm[audio] && `` to
                      the container entrypoint command. Required for any
                      omni model that takes audio input (the upstream
                      vllm/vllm-openai image doesn't bundle the audio
                      extras).
  container_name      (optional) Docker container name for stop-targeting.
                      Defaults to ``vllm-<friendly>``.
  hf_cache_host       (optional) Host path mounted into the container at
                      ``/root/.cache/huggingface``. Defaults to
                      ``$HOME/.cache/huggingface``.
  vllm_cache_host     (optional) Host path mounted into the container at
                      ``/root/.cache/vllm`` for torch.compile artifacts.
                      Defaults to ``$HOME/.cache/vllm``. Persisting this
                      across container restarts skips the ~8s torch.compile
                      step on every restart (Phase 0 spike notes gotcha
                      #5).
  shm_size            (optional) Container shared-memory size. Defaults to
                      ``16g``.
  extra_docker_args   (optional) List of additional ``docker run`` args
                      appended after the standard set. Use for additional
                      volume mounts, env vars, or runtime flags.

The container is launched with ``docker run --rm`` so a clean shutdown
(Ctrl-C, ``docker stop``, or supervisor SIGTERM) cleans up the container
state. ``--restart`` policies are intentionally not used — the supervisor
is the restart authority for managed services.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="vllm",
    help="Manage vLLM containers configured in journal.json.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "vllm/vllm-openai:v0.20.0"
DEFAULT_SHM_SIZE = "16g"
DEFAULT_HF_CACHE_HOST = os.path.expanduser("~/.cache/huggingface")
DEFAULT_VLLM_CACHE_HOST = os.path.expanduser("~/.cache/vllm")
DEFAULT_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_servers() -> dict[str, dict[str, Any]]:
    """Read providers.vllm.servers from the journal config."""
    from solstone.think.utils import get_config

    return get_config().get("providers", {}).get("vllm", {}).get("servers", {}) or {}


def _resolve_server_entry(name: str | None) -> tuple[str, dict[str, Any]]:
    """Pick which configured server to operate on.

    When ``name`` is given, that's the entry. When omitted and exactly one
    server is configured, that's the entry (common single-server case).
    Otherwise raises typer.BadParameter with the available names.
    """
    servers = _load_servers()
    if not servers:
        raise typer.BadParameter(
            "No vLLM servers configured. Add at least one entry under "
            "providers.vllm.servers in journal.json. See "
            "docs/PROVIDER_MATRIX.md or apps/vllm/call.py docstring for "
            "the schema."
        )
    if name:
        if name not in servers:
            raise typer.BadParameter(
                f"No vLLM server '{name}' configured. Available: "
                f"{', '.join(sorted(servers.keys()))}"
            )
        return name, servers[name]
    if len(servers) == 1:
        only_name = next(iter(servers))
        return only_name, servers[only_name]
    raise typer.BadParameter(
        f"Multiple vLLM servers configured ({', '.join(sorted(servers.keys()))}); "
        f"specify --name <friendly>."
    )


def _port_from_base_url(base_url: str, default: int = 8000) -> int:
    """Extract the port from a base_url, defaulting to 8000."""
    parsed = urlparse(base_url)
    return parsed.port or default


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _build_serve_argv(name: str, entry: dict[str, Any]) -> list[str]:
    """Build the docker-run argv for a configured server."""
    if "model" not in entry:
        raise typer.BadParameter(
            f"vLLM server '{name}' is missing required field 'model' "
            f"(HF repo path or in-container local path)."
        )
    if "served_model_name" not in entry and "base_url" not in entry:
        raise typer.BadParameter(
            f"vLLM server '{name}' must specify either served_model_name "
            f"or base_url so the chat-completions model field can be set."
        )

    model = str(entry["model"])
    served_name = str(entry.get("served_model_name") or name)
    image = str(entry.get("image") or DEFAULT_IMAGE)
    port = int(entry.get("port") or _port_from_base_url(str(entry.get("base_url", ""))))
    container_name = str(entry.get("container_name") or f"vllm-{name}")
    shm_size = str(entry.get("shm_size") or DEFAULT_SHM_SIZE)
    hf_cache_host = os.path.expanduser(
        str(entry.get("hf_cache_host") or DEFAULT_HF_CACHE_HOST)
    )
    vllm_cache_host = os.path.expanduser(
        str(entry.get("vllm_cache_host") or DEFAULT_VLLM_CACHE_HOST)
    )
    needs_audio = bool(entry.get("needs_audio_extras", False))
    vllm_args = list(entry.get("vllm_args") or [])
    extra_docker = list(entry.get("extra_docker_args") or [])

    # Ensure cache host paths exist so docker doesn't create them as root.
    os.makedirs(hf_cache_host, exist_ok=True)
    os.makedirs(vllm_cache_host, exist_ok=True)

    # vllm serve invocation that runs inside the container shell.
    serve_cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        "8000",
        "--served-model-name",
        served_name,
        *vllm_args,
    ]
    serve_str = " ".join(shlex.quote(a) for a in serve_cmd)

    if needs_audio:
        # Upstream vllm/vllm-openai doesn't bundle audio extras; install at
        # container start. For production we'd derive an image with these
        # baked in (Phase 4 follow-up); for now the runtime install is fast
        # (~14s) and works.
        entrypoint_cmd = f"pip install --no-cache-dir 'vllm[audio]' && {serve_str}"
    else:
        entrypoint_cmd = serve_str

    docker_argv: list[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--ipc=host",
        f"--shm-size={shm_size}",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-p",
        f"{port}:8000",
        "-v",
        f"{hf_cache_host}:/root/.cache/huggingface",
        "-v",
        f"{vllm_cache_host}:/root/.cache/vllm",
        *extra_docker,
        "--entrypoint",
        "/bin/bash",
        image,
        "-c",
        entrypoint_cmd,
    ]
    return docker_argv


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("list")
def list_servers(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """List configured vLLM servers."""
    servers = _load_servers()
    if json:
        import json as jsonlib

        typer.echo(jsonlib.dumps(servers, indent=2))
        return
    if not servers:
        typer.echo(
            "No vLLM servers configured. Add entries under "
            "providers.vllm.servers in journal.json."
        )
        return
    for name, entry in sorted(servers.items()):
        base_url = entry.get("base_url", "?")
        model = entry.get("model", "?")
        served = entry.get("served_model_name") or name
        typer.echo(f"  {name}")
        typer.echo(f"    base_url:          {base_url}")
        typer.echo(f"    served_model_name: {served}")
        typer.echo(f"    model:             {model}")


@app.command("status")
def status(
    json: bool = typer.Option(False, "--json", help="Emit JSON instead of text."),
) -> None:
    """Ping each configured server's /v1/models endpoint."""
    import httpx

    servers = _load_servers()
    if not servers:
        typer.echo("No vLLM servers configured.")
        raise typer.Exit(code=0)

    results: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(servers.items()):
        base_url = str(entry.get("base_url") or "")
        if not base_url:
            results[name] = {"reachable": False, "error": "no base_url configured"}
            continue
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}/v1/models", timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            served_ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
            results[name] = {
                "reachable": True,
                "base_url": base_url,
                "served_models": served_ids,
            }
        except Exception as exc:
            results[name] = {
                "reachable": False,
                "base_url": base_url,
                "error": str(exc),
            }

    if json:
        import json as jsonlib

        typer.echo(jsonlib.dumps(results, indent=2))
        return

    for name, r in results.items():
        if r["reachable"]:
            served = ", ".join(r.get("served_models", [])) or "(none)"
            typer.echo(f"  {name}  ✓  {r['base_url']}  served: {served}")
        else:
            typer.echo(
                f"  {name}  ✗  {r.get('base_url', '?')}  {r.get('error', 'unreachable')}"
            )


@app.command("serve")
def serve(
    name: str | None = typer.Option(
        None,
        "--name",
        help=(
            "Friendly name of the configured server to launch. Optional when "
            "exactly one server is configured."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Verbose logging on this side; the "
        "container's stdout/stderr always streams through regardless.",
    ),
) -> None:
    """Spawn the docker container for a configured vLLM server in the foreground.

    Streams container logs to stdout/stderr (so the supervisor's
    DailyLogWriter captures them). On SIGINT/SIGTERM, gracefully stops
    the container and waits for clean exit before returning.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if shutil.which("docker") is None:
        typer.echo(
            "ERROR: 'docker' not found on PATH. Install Docker + nvidia-container-toolkit "
            "before serving vLLM. See ~/Obsidian/.../20-29 Tech/Spark Technical Guidance.",
            err=True,
        )
        raise typer.Exit(code=2)

    resolved_name, entry = _resolve_server_entry(name)
    container_name = str(entry.get("container_name") or f"vllm-{resolved_name}")
    docker_argv = _build_serve_argv(resolved_name, entry)

    logger.info(
        "Launching vLLM server %s as container %s", resolved_name, container_name
    )
    logger.info("docker argv: %s", " ".join(shlex.quote(a) for a in docker_argv))

    # Spawn the container in the foreground; let docker run own stdout/stderr.
    # We use Popen so we can intercept SIGINT/SIGTERM and docker stop the
    # container before waiting on the child to finish.
    proc = subprocess.Popen(docker_argv, stdout=sys.stdout, stderr=sys.stderr)

    stop_requested = {"value": False}

    def _request_stop(signum: int, _frame: Any) -> None:
        if stop_requested["value"]:
            # Second signal — escalate to direct kill.
            logger.warning("Second signal received, killing docker subprocess")
            proc.kill()
            return
        stop_requested["value"] = True
        logger.info(
            "Signal %d received; calling docker stop %s", signum, container_name
        )
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("docker stop timed out; killing docker subprocess")
            proc.kill()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    rc = proc.wait()
    raise typer.Exit(code=rc)


if __name__ == "__main__":
    app()
