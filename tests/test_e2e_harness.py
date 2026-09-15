from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from nmesh.runtime import engine

HARNESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "e2e.py"


def _harness():
    spec = importlib.util.spec_from_file_location("nmesh_e2e_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _installed(tag: str, exe: Path) -> engine.InstalledEngine:
    return engine.InstalledEngine(
        backend="llamacpp",
        tag=tag,
        variant="cpu",
        exe=exe,
        version_line=None,
        sha256="0" * 64,
        asset=f"{tag}.zip",
        installed_at=0.0,
        flags=(),
    )


def test_backend_uses_engine_installed_by_nmesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    exe = tmp_path / "engines" / "llamacpp" / "b1" / "llama-server"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.delenv("NMESH_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("NMESH_LLAMA_CPP", raising=False)
    monkeypatch.setattr(harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(engine, "active", lambda: _installed("b1", exe))
    monkeypatch.setattr(engine, "installed", lambda: [_installed("b1", exe)])

    assert harness._backend() == exe.resolve()


def test_backend_prefers_active_engine_over_other_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    older = tmp_path / "b1" / "llama-server"
    newer = tmp_path / "b2" / "llama-server"
    for exe in (older, newer):
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
    monkeypatch.delenv("NMESH_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("NMESH_LLAMA_CPP", raising=False)
    monkeypatch.setattr(harness.shutil, "which", lambda _name: None)
    monkeypatch.setattr(engine, "active", lambda: _installed("b2", newer))
    monkeypatch.setattr(
        engine,
        "installed",
        lambda: [_installed("b1", older), _installed("b2", newer)],
    )

    assert harness._backend() == newer.resolve()


def test_model_source_reads_configured_nmesh_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    models = tmp_path / "models"
    models.mkdir()
    model = models / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    model.write_text("", encoding="utf-8")
    monkeypatch.delenv("NMESH_E2E_MODEL", raising=False)
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))

    assert harness._model_source("qwen2.5-1.5b-instruct", "q4_k_m") == model.resolve()
