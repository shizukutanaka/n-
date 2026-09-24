from __future__ import annotations

import io
import json
import os
import tarfile
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

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


def test_darwin_selects_macos_asset() -> None:
    asset, warning = engine.select_asset(
        "b10830",
        ["llama-b10830-bin-macos-x64.tar.gz"],
        system="Darwin",
        machine="x86_64",
        accelerator=None,
    )
    assert asset.asset == "llama-b10830-bin-macos-x64.tar.gz"
    assert warning is None


def test_bare_cuda_selects_highest_published_version() -> None:
    asset, _ = engine.select_asset(
        "b10830",
        [
            "llama-b10830-bin-win-cuda-12.4-x64.zip",
            "llama-b10830-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip",
        ],
        system="windows",
        machine="AMD64",
        accelerator=None,
        variant="cuda",
    )
    assert asset.variant == "cuda-13.3"
    assert asset.extra_assets == ("cudart-llama-bin-win-cuda-13.3-x64.zip",)


def test_bare_rocm_selects_highest_published_version() -> None:
    asset, _ = engine.select_asset(
        "b10830",
        [
            "llama-b10830-bin-ubuntu-rocm-9.0-x64.tar.gz",
            "llama-b10830-bin-ubuntu-rocm-10.0-x64.tar.gz",
        ],
        system="linux",
        machine="x86_64",
        accelerator=None,
        variant="rocm",
    )
    assert asset.variant == "rocm-10.0"


def test_explicit_cuda_version_selects_matching_asset() -> None:
    asset, _ = engine.select_asset(
        "b10830",
        [
            "llama-b10830-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ],
        system="windows",
        machine="AMD64",
        accelerator=None,
        variant="cuda-12.4",
    )
    assert asset.asset == "llama-b10830-bin-win-cuda-12.4-x64.zip"
    assert asset.extra_assets == ("cudart-llama-bin-win-cuda-12.4-x64.zip",)


def test_install_selects_shallowest_executable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(engine, "engines_dir", lambda: tmp_path)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("nested/llama-server.exe", "nested")
        source.writestr("llama-server.exe", "top")
    payload = archive.getvalue()

    def download(_url: str, path: Path) -> None:
        path.write_bytes(payload)

    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type(
            "Result", (), {"stdout": "", "stderr": "version: test", "returncode": 0}
        )(),
    )
    monkeypatch.setattr("nmesh.probe.caps.llamacpp_caps", lambda _: None)
    item, _ = engine.install(
        "b10830",
        dest=tmp_path,
        fetch=lambda _: ASSETS,
        download=download,
        system="windows",
        machine="AMD64",
        accelerator=None,
    )
    assert item.exe == (tmp_path / "b10830" / "llama-server.exe").resolve()


def test_install_skips_release_with_no_assets(monkeypatch, tmp_path: Path) -> None:
    """The newest tag can reach the feed before binaries finish uploading;
    install must take the newest tag that actually publishes assets."""
    monkeypatch.setattr(engine, "engines_dir", lambda: tmp_path)
    monkeypatch.setattr(engine, "build_tags", lambda fetch=None: ["b10830", "b10829"])
    monkeypatch.setattr(
        engine,
        "published_assets",
        lambda tag, fetch=None: [] if tag == "b10830" else ["linux cpu"],
    )
    monkeypatch.setattr(
        engine,
        "select_asset",
        lambda tag, assets, **kwargs: (SimpleNamespace(
            asset="fake.tar.gz", url="https://example/fake.tar.gz",
            extra_assets=(), variant="cpu",
        ), None),
    )
    archive = io.BytesIO()
    info = tarfile.TarInfo("llama-server")
    info.size = 4
    with tarfile.open(fileobj=archive, mode="w:gz") as source:
        source.addfile(info, io.BytesIO(b"fake"))
    payload = archive.getvalue()
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type(
            "Result", (), {"stdout": "", "stderr": "version: x", "returncode": 0}
        )(),
    )
    monkeypatch.setattr("nmesh.probe.caps.llamacpp_caps", lambda _: None)
    item, warnings = engine.install(
        dest=tmp_path,
        download=lambda _url, path: path.write_bytes(payload),
        system="linux",
        machine="x86_64",
        accelerator=None,
    )
    assert item.tag == "b10829"
    assert warnings[0] == "b10830 published no assets"


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


def _install_with_exit_code(
    monkeypatch, tmp_path: Path, system: str, returncode: int
) -> list[str]:
    monkeypatch.setattr(engine, "engines_dir", lambda: tmp_path)
    exe_name = "llama-server.exe" if system == "windows" else "llama-server"
    archive = io.BytesIO()
    if system == "windows":
        with zipfile.ZipFile(archive, "w") as source:
            source.writestr(exe_name, "fake")
    else:
        with tarfile.open(fileobj=archive, mode="w:gz") as source:
            info = tarfile.TarInfo(exe_name)
            info.size = 4
            source.addfile(info, io.BytesIO(b"fake"))
    payload = archive.getvalue()
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: type(
            "Result", (), {"stdout": "", "stderr": "", "returncode": returncode}
        )(),
    )
    monkeypatch.setattr("nmesh.probe.caps.llamacpp_caps", lambda _: None)
    _, warnings = engine.install(
        "b10830",
        dest=tmp_path,
        fetch=lambda _: ASSETS,
        download=lambda _url, path: path.write_bytes(payload),
        system=system,
        machine="AMD64",
        accelerator=None,
    )
    return warnings


def test_install_warns_when_binary_fails_to_launch(monkeypatch, tmp_path: Path) -> None:
    # Real-world case: llama.cpp win binaries exit 0xC0000005 when the
    # Microsoft Visual C++ Redistributable is missing/outdated
    # (ggml-org/llama.cpp#11479, #10944, #13767) — the warning must name it.
    warnings = _install_with_exit_code(
        monkeypatch, tmp_path, system="windows", returncode=3221225477
    )
    launch_warning = next(w for w in warnings if "failed to launch" in w)
    assert "3221225477" in launch_warning
    assert "Visual C++ Redistributable" in launch_warning


def test_install_launch_warning_has_no_vcredist_hint_off_windows(
    monkeypatch, tmp_path: Path
) -> None:
    warnings = _install_with_exit_code(
        monkeypatch, tmp_path, system="linux", returncode=1
    )
    launch_warning = next(w for w in warnings if "failed to launch" in w)
    assert "Visual C++" not in launch_warning


def test_tar_member_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        source.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        engine._extract_archive(archive, tmp_path / "install")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs Windows privileges")
def test_tar_relative_symlink_chain_is_extracted(tmp_path: Path) -> None:
    archive = tmp_path / "links.tar.gz"
    content = b"shared library"
    with tarfile.open(archive, "w:gz") as source:
        library = tarfile.TarInfo("lib/libx.so.0.1")
        library.mode = 0o755
        library.size = len(content)
        source.addfile(library, io.BytesIO(content))
        soname = tarfile.TarInfo("lib/libx.so.0")
        soname.type = tarfile.SYMTYPE
        soname.linkname = "libx.so.0.1"
        source.addfile(soname)
        linker_name = tarfile.TarInfo("lib/libx.so")
        linker_name.type = tarfile.SYMTYPE
        linker_name.linkname = "libx.so.0"
        source.addfile(linker_name)
        executable = tarfile.TarInfo("bin/llama-server")
        executable.mode = 0o755
        executable.size = 3
        source.addfile(executable, io.BytesIO(b"bin"))

    destination = tmp_path / "install"
    engine._extract_archive(archive, destination)

    assert (destination / "lib/libx.so").is_symlink()
    assert os.readlink(destination / "lib/libx.so") == "libx.so.0"
    assert os.readlink(destination / "lib/libx.so.0") == "libx.so.0.1"
    assert (destination / "lib/libx.so").read_bytes() == content
    assert os.access(destination / "bin/llama-server", os.X_OK)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs Windows privileges")
def test_tar_symlink_escaping_destination_is_rejected(tmp_path: Path) -> None:
    for name, linkname in (("relative.tar.gz", "../../outside"), ("absolute.tar.gz", "/etc/passwd")):
        archive = tmp_path / name
        with tarfile.open(archive, "w:gz") as source:
            link = tarfile.TarInfo("lib/evil")
            link.type = tarfile.SYMTYPE
            link.linkname = linkname
            source.addfile(link)
        with pytest.raises(ValueError, match="escapes"):
            engine._extract_archive(archive, tmp_path / "install")


def test_tar_hardlink_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "hardlink.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        link = tarfile.TarInfo("link")
        link.type = tarfile.LNKTYPE
        link.linkname = "a"
        source.addfile(link)
    with pytest.raises(ValueError, match="link is not allowed"):
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


def test_models_local_lists_downloaded_weights(monkeypatch, tmp_path: Path, capsys) -> None:
    model = tmp_path / "models" / "qwen-q4_k_m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    monkeypatch.setattr(cli, "nmesh_home", lambda: tmp_path)
    args = type(
        "Args",
        (),
        {"models_command": "local", "json": True},
    )()

    assert cli._models(args) == 0
    assert str(model) in json.loads(capsys.readouterr().out)[0]["path"]


def test_models_fetch_downloads_planned_services(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    model_path = tmp_path / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    service = type(
        "Service",
        (),
        {
            "model_id": "qwen2.5-0.5b-instruct",
            "name": "chat",
            "backend": "llamacpp",
            "quant": "q4_k_m",
            "model_ref": str(model_path),
        },
    )()
    plan = type("Plan", (), {"services": [service], "warnings": []})()
    acquired = type(
        "Acquired",
        (),
        {"path": model_path, "model_ref": None, "warning": None},
    )()
    seen: list[object] = []

    def fake_acquire(item: object) -> object:
        seen.append(item)
        return acquired

    monkeypatch.setattr(cli, "_make_plan", lambda args: plan)
    monkeypatch.setattr(cli, "acquire", fake_acquire)
    args = type(
        "Args",
        (),
        {
            "models_command": "fetch",
            "model": "qwen2.5-0.5b-instruct",
            "json": True,
        },
    )()

    assert cli._models(args) == 0
    assert seen == [service]
    out = json.loads(capsys.readouterr().out)
    assert out[0]["model"] == "qwen2.5-0.5b-instruct"
    assert out[0]["path"] == str(model_path)


def test_models_fetch_rejects_unknown_model(capsys) -> None:
    args = type(
        "Args",
        (),
        {"models_command": "fetch", "model": "no-such-model", "json": False},
    )()

    assert cli._models(args) == 1
    assert "no-such-model" in capsys.readouterr().err


def test_models_fetch_reports_when_plan_cannot_place(
    monkeypatch, capsys
) -> None:
    plan = type(
        "Plan",
        (),
        {"services": [], "warnings": ["does not fit"]},
    )()
    monkeypatch.setattr(cli, "_make_plan", lambda args: plan)
    args = type(
        "Args",
        (),
        {
            "models_command": "fetch",
            "model": "qwen2.5-0.5b-instruct",
            "json": False,
        },
    )()

    assert cli._models(args) == 1
    err = capsys.readouterr().err
    assert "does not fit" in err
    assert "qwen2.5-0.5b-instruct" in err


def test_unload_empty_result_reports_reason(monkeypatch, capsys) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"unloaded": [], "results": [{"service": "chat", "reason": "idle"}]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    args = type(
        "Args",
        (),
        {"port": 18058, "service": "chat", "json": True},
    )()

    assert cli._unload(args) == 1
    assert "not unloaded" in capsys.readouterr().err


def test_unload_not_owned_points_to_foreign_down(monkeypatch, capsys) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"unloaded": [], "results": [{"service": "chat", "reason": "not_owned"}]}'

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    args = type(
        "Args",
        (),
        {"port": 18058, "service": "chat", "json": False},
    )()

    assert cli._unload(args) == 1
    assert "nmesh down" in capsys.readouterr().err


def test_unload_404_preserves_unknown_service(monkeypatch, capsys) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:18058/admin/unload/chat",
        404,
        "not found",
        {},
        None,
    )
    monkeypatch.setattr(
        cli.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    args = type(
        "Args",
        (),
        {"port": 18058, "service": "chat", "json": False},
    )()

    assert cli._unload(args) == 1
    assert capsys.readouterr().err.strip() == "Unknown service: chat"


def test_engine_install_accepts_backend_positional(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    item = engine.InstalledEngine(
        "llamacpp", "b10830", "cpu", tmp_path / "llama-server.exe",
        None, "sha", "asset.zip", 0.0, (),
    )
    seen: dict[str, object] = {}

    def fake_install(**kwargs: object) -> tuple[engine.InstalledEngine, list[str]]:
        seen.update(kwargs)
        return item, []

    monkeypatch.setattr(engine, "install", fake_install)

    assert cli.main(["engine", "install", "llamacpp", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["installed"]["tag"] == "b10830"
    assert seen == {"tag": None, "variant": "auto"}


def test_engine_install_rejects_unknown_backend(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["engine", "install", "vllm"])
    assert error.value.code == 2
