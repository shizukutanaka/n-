"""Hardware detection and tier classification."""

from .caps import BackendCaps, llamacpp_caps, parse_devices, parse_help
from .detector import detect_hardware, parse_nvidia_smi, parse_rocm_smi
from .generic_gpu import detect_generic, detect_linux_sysfs, parse_windows_adapters
from .models import GPUInfo, HardwareProfile, Tier, classify_tier

__all__ = [
    "BackendCaps",
    "GPUInfo",
    "HardwareProfile",
    "Tier",
    "classify_tier",
    "detect_generic",
    "detect_hardware",
    "detect_linux_sysfs",
    "llamacpp_caps",
    "parse_devices",
    "parse_help",
    "parse_nvidia_smi",
    "parse_rocm_smi",
    "parse_windows_adapters",
]
