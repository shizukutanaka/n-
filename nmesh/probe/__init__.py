"""Hardware detection and tier classification."""

from .detector import detect_hardware, parse_nvidia_smi, parse_rocm_smi
from .models import GPUInfo, HardwareProfile, Tier, classify_tier

__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "Tier",
    "classify_tier",
    "detect_hardware",
    "parse_nvidia_smi",
    "parse_rocm_smi",
]
