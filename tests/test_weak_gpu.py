from __future__ import annotations

import json
from dataclasses import replace

from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan
from nmesh.probe import (
    GPUInfo,
    HardwareProfile,
    Tier,
    detector,
    parse_devices,
    parse_windows_adapters,
)
from nmesh.probe.generic_gpu import detect_linux_sysfs

GIB = 1024**3


def _profile(gpus: list[GPUInfo]) -> HardwareProfile:
    return HardwareProfile(
        "windows", "CPU", 4, 8, 32 * GIB, 32 * GIB, 100 * GIB, False, gpus,
        {"llamacpp": "test"}, Tier.T3_HIGH if gpus else Tier.T0_CPU,
    )


def test_parse_llamacpp_device_reports() -> None:
    assert parse_devices("Available devices:\n  (none)\n") == ()
    assert parse_devices("Available devices:\n  Vulkan0: Intel Arc\n  CUDA0: RTX\n") == (
        "Vulkan0: Intel Arc", "CUDA0: RTX"
    )
    assert parse_devices("usage: llama-server") is None


def test_windows_generic_gpu_prefers_64_bit_registry_memory() -> None:
    payload = json.dumps([
        {
            "Name": "Intel Arc A770",
            "AdapterRAM": 4293918720,
            "DriverVersion": "1",
            "PNPDeviceID": "PCI\\VEN_8086&DEV_56A0",
        },
        {
            "Name": "Intel UHD Graphics",
            "AdapterRAM": 1073741824,
            "DriverVersion": "1",
            "PNPDeviceID": "PCI\\VEN_8086&DEV_4680",
        },
    ])
    gpus = parse_windows_adapters(
        payload,
        {"PCI\\VEN_8086&DEV_56A0": 16 * GIB},
    )
    assert [gpu.name for gpu in gpus] == ["Intel Arc A770", "Intel UHD Graphics"]
    assert gpus[0].total_vram_bytes == 16 * GIB
    assert gpus[0].vram_source == "registry"
    assert gpus[1].total_vram_bytes == GIB
    assert gpus[1].vram_source == "unknown"
    assert Tier.T3_HIGH == detector.classify_tier([gpus[0]])
    assert Tier.T0_CPU == detector.classify_tier([gpus[1]])


def test_linux_generic_gpu_reads_amd_vram_and_does_not_invent_intel_vram(tmp_path) -> None:
    amd = tmp_path / "card0" / "device"
    intel = tmp_path / "card1" / "device"
    amd.mkdir(parents=True)
    intel.mkdir(parents=True)
    (amd / "vendor").write_text("0x1002\n", encoding="ascii")
    (amd / "mem_info_vram_total").write_text(str(8 * GIB), encoding="ascii")
    (intel / "vendor").write_text("0x8086\n", encoding="ascii")
    gpus = detect_linux_sysfs(tmp_path)
    assert gpus[0].vendor == "amd"
    assert gpus[0].total_vram_bytes == 8 * GIB
    assert gpus[0].vram_source == "sysfs"
    assert gpus[1].vendor == "intel"
    assert gpus[1].total_vram_bytes == 0
    assert gpus[1].vram_source == "unknown"


def test_known_cpu_only_llamacpp_forces_cpu_placement() -> None:
    model = next(item for item in load_catalog() if item.id == "qwen2.5-1.5b-instruct")
    profile = replace(
        _profile([GPUInfo(0, "Arc", "intel", 16 * GIB, 16 * GIB, None, False)]),
        backend_gpu_devices={"llamacpp": ()},
    )
    plan = build_plan(profile, [model], Policy(roles=["chat"]))
    service = plan.services[0]
    assert service.backend == "llamacpp"
    assert service.n_gpu_layers == 0
    assert service.memory.gpu_bytes == 0
    assert service.memory.cpu_bytes >= service.memory.weight_bytes
    assert not any(flag in service.launch.argv for flag in ("-ngl", "--gpu-layers", "--n-gpu-layers"))
    assert any("no GPU backend" in warning and "Vulkan" in warning for warning in plan.warnings)


def test_unknown_llamacpp_device_probe_preserves_gpu_placement() -> None:
    model = next(item for item in load_catalog() if item.id == "qwen2.5-1.5b-instruct")
    profile = replace(
        _profile([GPUInfo(0, "GPU", "nvidia", 16 * GIB, 16 * GIB, None, False)]),
        backend_gpu_devices={},
    )
    plan = build_plan(profile, [model], Policy(roles=["chat"]))
    service = plan.services[0]
    assert service.n_gpu_layers > 0
    assert service.memory.gpu_bytes > 0
    assert "-ngl" in service.launch.argv


def test_generic_detection_is_skipped_when_specialized_detection_succeeds(monkeypatch) -> None:
    gpu = GPUInfo(0, "NVIDIA", "nvidia", 8 * GIB, 8 * GIB, None, False)
    monkeypatch.setattr(detector, "_detect_nvidia", lambda warnings: [gpu])
    monkeypatch.setattr(detector, "_detect_rocm", lambda warnings: [])
    monkeypatch.setattr(
        detector, "detect_generic", lambda os_name: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(
        detector, "_detect_backends", lambda warnings: ({"llamacpp": None}, {}, {}, {})
    )
    profile = detector.detect_hardware()
    assert profile.gpus[0].name == "NVIDIA"


def test_low_vram_generic_gpu_remains_visible_with_explanation(monkeypatch) -> None:
    gpu = GPUInfo(0, "Intel UHD", "intel", GIB, GIB, None, True)
    monkeypatch.setattr(detector, "_detect_nvidia", lambda warnings: [])
    monkeypatch.setattr(detector, "_detect_rocm", lambda warnings: [])
    monkeypatch.setattr(detector, "detect_generic", lambda os_name: [gpu])
    monkeypatch.setattr(
        detector, "_detect_backends", lambda warnings: ({"llamacpp": None}, {}, {}, {})
    )
    profile = detector.detect_hardware()
    assert profile.gpus == [gpu]
    assert profile.tier == Tier.T0_CPU
    assert "less than 2 GiB dedicated VRAM" in profile.warnings[0]
