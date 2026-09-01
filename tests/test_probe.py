from __future__ import annotations

from nmesh.probe import GPUInfo, Tier, classify_tier, parse_nvidia_smi, parse_rocm_smi


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
