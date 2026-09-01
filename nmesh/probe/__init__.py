"""Hardware detection and tier classification."""

from .caps import BackendCaps, llamacpp_caps, parse_help
from .detector import detect_hardware, parse_nvidia_smi, parse_rocm_smi
from .models import GPUInfo, HardwareProfile, Tier, classify_tier

__all__ = [
    "BackendCaps",
    "GPUInfo",
    "HardwareProfile",
    "Tier",
    "classify_tier",
    "detect_hardware",
    "llamacpp_caps",
    "parse_help",
    "parse_nvidia_smi",
    "parse_rocm_smi",
]
