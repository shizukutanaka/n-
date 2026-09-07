from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from nmesh import cli
from nmesh.runtime import engine

ATOM = """
<feed>
  <entry><id>tag:github.com,2008:Repository/1/b10830</id></entry>
  <entry><id>tag:github.com,2008:Repository/1/v0.4.0</id></entry>
  <entry><id>tag:github.com,2008:Repository/1/b10829</id></entry>
</feed>
"""
ASSETS = """
<a href="/ggml-org/llama.cpp/releases/download/b10830/llama-b10830-bin-win-cpu-x64.zip">cpu</a>
<a href="/ggml-org/llama.cpp/releases/download/b10830/llama-b10830-bin-win-vulkan-x64.zip">vulkan</a>
<a href="/ggml-org/llama.cpp/releases/download/b10830/llama-b10830-bin-win-cuda-12.4-x64.zip">cuda</a>
<a href="/ggml-org/llama.cpp/releases/download/b10830/cudart-llama-bin-win-cuda-12.4-x64.zip">cudart</a>
<a href="/ggml-org/llama.cpp/releases/download/b10830/llama-b10830-bin-ubuntu-vulkan-x64.tar.gz">linux</a>
<a href="/ggml-org/llama.cpp/releases/download/b10830/llama-b10830-bin-ubuntu-x64.tar.gz">linux cpu</a>
"""


def test_build_tags_filter_nightly_release() -> None:
    assert engine.build_tags(fetch=lambda _: ATOM) == ["b10830", "b10829"]


def test_published_assets_are_sorted_and_deduplicated() -> None:
    duplicate = ASSETS + ASSETS.splitlines()[1]
    assets = engine.published_assets("b10830", fetch=lambda _: duplicate)
    assert assets == sorted(set(assets))
    assert len(assets) == 6


def test_linux_nvidia_uses_vulkan_with_warning() -> None:
    asset, warning = engine.select_asset(
        "b10830",
        ["llama-b10830-bin-ubuntu-vulkan-x64.tar.gz"],
        system="linux",
        machine="x86_64",
        accelerator="nvidia",
    )
    assert asset.variant == "vulkan"
    assert "no ubuntu-cuda asset" in warning


def test_explicit_unpublished_variant_reports_assets() -> None:
    with pytest.raises(ValueError, match="published assets"):
        engine.select_asset(
            "b10830",
            ["llama-b10830-bin-win-cpu-x64.zip"],
            system="windows",
            machine="AMD64",
            accelerator=None,
            variant="vulkan",
        )


def test_install_manifest_and_active_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine, "engines_dir", lambda: tmp_path)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("llama-server.exe", "fake")
    payload = archive.getvalue()

    def download(_url: str, path: Path) -> None:
        path.write_bytes(payload)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type(
            "Result", (), {"stdout": "", "stderr": "version: test build", "returncode": 1}
        )(),
    )
    monkeypatch.setattr("nmesh.probe.caps.llamacpp_caps", lambda _: None)
    item, warnings = engine.install(
        "b10830",
        dest=tmp_path,
        fetch=lambda _: ASSETS,
        download=download,
        system="windows",
        machine="AMD64",
        accelerator=None,
    )
    assert item.version_line == "version: test build"
    assert warnings
    manifest = json.loads((tmp_path / "b10830" / "manifest.json").read_text())
    assert manifest["sha256"]
    assert engine.active().tag == "b10830"
    assert engine.installed()[0].exe.name == "llama-server.exe"


def test_tar_member_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        source.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        engine._extract_archive(archive, tmp_path / "install")


def test_models_rm_refuses_a_planned_file(monkeypatch, tmp_path: Path, capsys) -> None:
    model = tmp_path / "models" / "qwen-q4_k_m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    service = type("Service", (), {"model_ref": str(model)})()
    plan = type("Plan", (), {"services": [service]})()
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    args = type(
        "Args",
        (),
        {"models_command": "rm", "name": model.name, "force": False, "json": False},
    )()

    assert cli._models(args) == 1
    assert model.exists()
    assert "--force" in capsys.readouterr().err
