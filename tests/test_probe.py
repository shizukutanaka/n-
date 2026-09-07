from __future__ import annotations

from pathlib import Path

from nmesh.probe import (
    GPUInfo,
    Tier,
    classify_tier,
    detector,
    parse_nvidia_smi,
    parse_rocm_smi,
)


def test_nvidia_smi_parser() -> None:
    text = "0, NVIDIA GeForce RTX 3060, 12288, 10000, 8.6\n1, A100, 81920, 80000, 8.0"
    gpus = parse_nvidia_smi(text)
    assert len(gpus) == 2
    assert gpus[0].total_vram_bytes == 12288 * 1024**2
    assert gpus[0].compute_capability == (8, 6)


def test_rocm_smi_parser() -> None:
    text = '{"GPU[0]": {"VRAM Total Memory (B)": 8589934592, "VRAM Total Used Memory (B)": 1024}}'
    gpus = parse_rocm_smi(text)
    assert len(gpus) == 1
    assert gpus[0].vendor == "amd"
    assert gpus[0].total_vram_bytes == 8589934592


def test_tier_boundaries() -> None:
    gpu = GPUInfo(0, "GPU", "nvidia", 4 * 1024**3, 4 * 1024**3, None, False)
    assert classify_tier([gpu]) == Tier.T1_LOW
    assert classify_tier([]) == Tier.T0_CPU
    assert classify_tier([gpu, gpu]) == Tier.T5_SERVER


def test_backend_env_binary_wins_over_path(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setenv("NMESH_LLAMACPP_BIN", str(binary))
    monkeypatch.setattr(
        detector.shutil,
        "which",
        lambda value: "C:/path/llama-server.exe" if value == "llama-server" else None,
    )
    monkeypatch.setattr(
        detector,
        "_run",
        lambda command: ("version: test", "") if command[0] == str(binary) else (None, None),
    )
    monkeypatch.setattr(detector, "llamacpp_caps", lambda path: None)

    backends, _, paths, _ = detector._detect_backends([], [])

    assert backends["llamacpp"] == "version: test"
    assert paths["llamacpp"] == str(binary.resolve())


def test_missing_backend_env_binary_does_not_fall_back_to_path(
    monkeypatch,
) -> None:
    missing = "C:/missing/llama-server.exe"
    monkeypatch.setenv("NMESH_LLAMACPP_BIN", missing)
    monkeypatch.setattr(
        detector.shutil,
        "which",
        lambda value: (
            "C:/path/llama-server.exe"
            if value == "llama-server"
            else None
        ),
    )

    warnings: list[str] = []
    params: list[dict[str, str]] = []
    backends, _, paths, _ = detector._detect_backends(warnings, params)

    assert backends["llamacpp"] is None
    assert "llamacpp" not in paths
    assert "warn.backend_binary_missing" in warnings
    assert {"backend": "llamacpp", "path": missing} in params


def test_backend_binary_is_found_in_nmesh_home_bin(monkeypatch, tmp_path: Path) -> None:
    binary = tmp_path / "bin" / "llama-server.exe"
    binary.parent.mkdir()
    binary.write_text("", encoding="utf-8")
    monkeypatch.delenv("NMESH_LLAMACPP_BIN", raising=False)
    monkeypatch.setattr(detector, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(detector.shutil, "which", lambda value: None)

    resolved = detector._resolve_backend_binary("llamacpp", "llama-server", [], [])

    assert resolved == binary.resolve()


def test_active_managed_engine_precedes_nmesh_home_bin(monkeypatch, tmp_path: Path) -> None:
    managed = tmp_path / "engines" / "llama-server.exe"
    managed.parent.mkdir(parents=True)
    managed.write_text("", encoding="utf-8")
    fallback = tmp_path / "bin" / "llama-server.exe"
    fallback.parent.mkdir()
    fallback.write_text("", encoding="utf-8")
    monkeypatch.delenv("NMESH_LLAMACPP_BIN", raising=False)
    monkeypatch.setattr(detector.shutil, "which", lambda _: None)
    monkeypatch.setattr(detector, "nmesh_home", lambda: tmp_path)
    monkeypatch.setattr(
        "nmesh.runtime.engine.active",
        lambda: type("Engine", (), {"exe": managed})(),
    )

    assert detector._resolve_backend_binary("llamacpp", "llama-server", [], []) == (
        managed.resolve()
    )


def test_backend_version_line_prefers_line_containing_version() -> None:
    assert detector._version_line(
        "Warning: could not connect to a running Ollama instance\n"
        "Warning: client version is 0.33.2\n",
        None,
    ) == "Warning: client version is 0.33.2"
