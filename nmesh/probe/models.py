from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Tier(str, Enum):
    T0_CPU = "T0_CPU"
    T1_LOW = "T1_LOW"
    T2_MID = "T2_MID"
    T3_HIGH = "T3_HIGH"
    T4_WORKSTATION = "T4_WORKSTATION"
    T5_SERVER = "T5_SERVER"


Vendor = Literal["nvidia", "amd", "intel", "apple", "unknown"]
OperatingSystem = Literal["windows", "linux", "macos"]


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vendor: Vendor
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: tuple[int, int] | None
    driving_display: bool


@dataclass(frozen=True)
class HardwareProfile:
    os: OperatingSystem
    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    free_disk_bytes: int
    unified_memory: bool
    gpus: list[GPUInfo]
    available_backends: dict[str, str | None]
    tier: Tier
    warnings: list[str] = field(default_factory=list)
    backend_flags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    backend_paths: Mapping[str, str] = field(default_factory=dict)


def classify_tier(
    gpus: list[GPUInfo], unified_memory: bool = False, total_ram_bytes: int = 0
) -> Tier:
    effective_vram = sum(gpu.total_vram_bytes for gpu in gpus)
    if unified_memory:
        effective_vram = int(total_ram_bytes * 0.70)
    if len(gpus) >= 2 or effective_vram >= 48 * 1024**3:
        return Tier.T5_SERVER
    if effective_vram < 2 * 1024**3:
        return Tier.T0_CPU
    if effective_vram < 6 * 1024**3:
        return Tier.T1_LOW
    if effective_vram < 12 * 1024**3:
        return Tier.T2_MID
    if effective_vram < 24 * 1024**3:
        return Tier.T3_HIGH
    return Tier.T4_WORKSTATION
