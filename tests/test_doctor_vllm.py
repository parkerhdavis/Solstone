# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for fork-only vLLM advisory checks (solstone.think.doctor_vllm)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def doctor_vllm():
    # Ensure doctor.py's registry is built (so CHECK_MAP has the vLLM
    # entries) before doctor_vllm functions are called.
    from solstone.think import doctor  # noqa: F401
    from solstone.think import doctor_vllm as module

    yield module


def args(doctor_vllm, *, port: int = 5015):
    from solstone.think import doctor

    return doctor.Args(verbose=False, json=False, jsonl=False, port=port)


class TestVllmDockerAvailable:
    def test_ok_when_docker_present(self, doctor_vllm, monkeypatch):
        monkeypatch.setattr(doctor_vllm.shutil, "which", lambda name: "/usr/bin/docker")
        result = doctor_vllm.vllm_docker_available_check(args(doctor_vllm))
        assert result.status == "ok"
        assert "docker found" in result.detail

    def test_warn_when_docker_missing(self, doctor_vllm, monkeypatch):
        monkeypatch.setattr(doctor_vllm.shutil, "which", lambda name: None)
        result = doctor_vllm.vllm_docker_available_check(args(doctor_vllm))
        assert result.status == "warn"
        assert "docker not on PATH" in result.detail


class TestVllmNvidiaSmi:
    def test_warn_when_nvidia_smi_missing(self, doctor_vllm, monkeypatch):
        monkeypatch.setattr(doctor_vllm.shutil, "which", lambda name: None)
        result = doctor_vllm.vllm_nvidia_smi_check(args(doctor_vllm))
        assert result.status == "warn"
        assert "nvidia-smi not on PATH" in result.detail

    def test_ok_when_nvidia_smi_lists_gpus(self, doctor_vllm, monkeypatch):
        from solstone.think import doctor

        monkeypatch.setattr(doctor_vllm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            doctor_vllm,
            "run_probe",
            lambda *_a, **_kw: doctor.ProbeOutput(
                "GPU 0: NVIDIA GB10 (UUID: GPU-...)\n", "", 0
            ),
        )
        result = doctor_vllm.vllm_nvidia_smi_check(args(doctor_vllm))
        assert result.status == "ok"
        assert "1 GPU" in result.detail

    def test_fail_when_no_gpu_lines(self, doctor_vllm, monkeypatch):
        from solstone.think import doctor

        monkeypatch.setattr(doctor_vllm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(
            doctor_vllm,
            "run_probe",
            lambda *_a, **_kw: doctor.ProbeOutput("Driver Version: 580.00\n", "", 0),
        )
        result = doctor_vllm.vllm_nvidia_smi_check(args(doctor_vllm))
        assert result.status == "fail"


class TestVllmServersReachable:
    def _write_journal(self, tmp_path: Path, payload: dict | None) -> Path:
        journal = tmp_path / "journal"
        config_dir = journal / "config"
        config_dir.mkdir(parents=True)
        if payload is not None:
            (config_dir / "journal.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return journal

    def test_skip_when_no_journal(self, doctor_vllm, monkeypatch, tmp_path):
        monkeypatch.delenv("SOLSTONE_JOURNAL", raising=False)
        monkeypatch.setattr(doctor_vllm, "ROOT", tmp_path)
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "skip"
        assert "no journal config found" in result.detail

    def test_skip_when_section_missing(self, doctor_vllm, monkeypatch, tmp_path):
        journal = self._write_journal(tmp_path, {"providers": {"generate": {}}})
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "skip"
        assert "no providers.vllm.servers configured" in result.detail

    def test_skip_when_servers_empty(self, doctor_vllm, monkeypatch, tmp_path):
        journal = self._write_journal(
            tmp_path, {"providers": {"vllm": {"servers": {}}}}
        )
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "skip"
        assert "is empty" in result.detail

    def test_skip_when_unreadable(self, doctor_vllm, monkeypatch, tmp_path):
        journal = tmp_path / "journal"
        (journal / "config").mkdir(parents=True)
        (journal / "config" / "journal.json").write_text("not json", encoding="utf-8")
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "skip"
        assert "not readable as JSON" in result.detail

    def test_ok_when_all_reachable(self, doctor_vllm, monkeypatch, tmp_path):
        journal = self._write_journal(
            tmp_path,
            {
                "providers": {
                    "vllm": {
                        "servers": {
                            "alpha": {"base_url": "http://localhost:8000"},
                            "beta": {"base_url": "http://localhost:8001"},
                        }
                    }
                }
            },
        )
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        monkeypatch.setattr(
            doctor_vllm.urllib.request, "urlopen", lambda *_a, **_kw: _Resp()
        )
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "ok"
        assert "alpha" in result.detail and "beta" in result.detail

    def test_warn_when_one_unreachable(self, doctor_vllm, monkeypatch, tmp_path):
        journal = self._write_journal(
            tmp_path,
            {
                "providers": {
                    "vllm": {
                        "servers": {
                            "alpha": {"base_url": "http://localhost:8000"},
                            "beta": {"base_url": "http://localhost:8001"},
                        }
                    }
                }
            },
        )
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def fake_urlopen(url, **_kw):
            if "8001" in url:
                raise doctor_vllm.urllib.error.URLError("connection refused")
            return _Resp()

        monkeypatch.setattr(doctor_vllm.urllib.request, "urlopen", fake_urlopen)
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "warn"
        assert "beta" in result.detail
        assert "alpha" in result.detail  # included as the reachable list

    def test_warn_when_entry_missing_base_url(self, doctor_vllm, monkeypatch, tmp_path):
        journal = self._write_journal(
            tmp_path,
            {"providers": {"vllm": {"servers": {"alpha": {}}}}},
        )
        monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
        result = doctor_vllm.vllm_servers_reachable_check(args(doctor_vllm))
        assert result.status == "warn"
        assert "no base_url" in result.detail
