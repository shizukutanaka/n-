from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .models import GPUInfo, Vendor, VramSource

_SATURATED_ADAPTER_RAM = 4095 * 1024**2
_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _vendor(value: object) -> Vendor | None:
    text = str(value).strip().lower()
    if text in {"0x8086", "8086"}:
        return "intel"
    if text in {"0x1002", "1002"}:
        return "amd"
    return None


def _number(value: object) -> int:
    try:
        if isinstance(value, (str, bytes, bytearray, int, float)):
            return int(value)
        return 0
    except (TypeError, ValueError):
        return 0


def _registry_size(
    pnp: str, name: str, registry_sizes: Mapping[str, int]
) -> int:
    candidates = (pnp.lower(), name.lower())
    for key, value in registry_sizes.items():
        normalized = str(key).lower()
        if any(
            candidate and (
                normalized == candidate
                or candidate.startswith(normalized)
                or normalized.startswith(candidate)
            )
            for candidate in candidates
        ):
            size = _number(value)
            if size:
                return size
    return 0


def parse_windows_adapters(
    payload: str | list[object] | dict[str, object],
    registry_sizes: Mapping[str, int] | None = None,
) -> list[GPUInfo]:
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return []
    else:
        value = payload
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    registry_sizes = registry_sizes or {}
    gpus: list[GPUInfo] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or f"Video adapter {index}")
        if any(
            marker in name.lower()
            for marker in ("iddsampledriver", "microsoft basic display", "remote display")
        ):
            continue
        adapter_ram = _number(item.get("AdapterRAM"))
        pnp = str(item.get("PNPDeviceID") or "")
        registry_ram = _registry_size(pnp, name, registry_sizes)
        if registry_ram > 0:
            total = registry_ram
            source: VramSource = "registry"
        elif 0 < adapter_ram < _SATURATED_ADAPTER_RAM:
            total = adapter_ram
            source = "unknown"
        else:
            total = 0
            source = "unknown"
        pnp_vendor = re.search(r"VEN_([0-9A-F]{4})", pnp, re.IGNORECASE)
        vendor = _vendor(pnp_vendor.group(1)) if pnp_vendor else None
        if vendor is None:
            vendor = _vendor(item.get("Vendor"))
        if vendor is None:
            lowered = name.lower()
            vendor = "intel" if "intel" in lowered else "amd" if "amd" in lowered else "unknown"
        gpus.append(
            GPUInfo(index, name, vendor, total, total, None, True, source)
        )
    return gpus


def _read_sysfs_card(card: Path, index: int) -> GPUInfo | None:
    try:
        vendor_id = card.joinpath("device", "vendor").read_text(encoding="ascii").strip()
    except OSError:
        return None
    vendor = _vendor(vendor_id)
    if vendor is None:
        return None
    try:
        name = card.joinpath("device", "uevent").read_text(encoding="utf-8")
        product = next(
            (line.split("=", 1)[1] for line in name.splitlines() if line.startswith("DRIVER=")),
            f"{vendor.title()} GPU {index}",
        )
    except OSError:
        product = f"{vendor.title()} GPU {index}"
    total = 0
    source: VramSource = "unknown"
    try:
        total = _number(
            card.joinpath("device", "mem_info_vram_total").read_text(encoding="ascii").strip()
        )
    except OSError:
        pass
    if total:
        source = "sysfs"
    return GPUInfo(index, product, vendor, total, total, None, True, source)


def detect_linux_sysfs(root: Path = Path("/sys/class/drm")) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for index, card in enumerate(sorted(root.glob("card*/"))):
        gpu = _read_sysfs_card(card, index)
        if gpu is not None:
            gpus.append(gpu)
    return gpus


def _registry_sizes() -> dict[str, int]:
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return {}
    result: dict[str, int] = {}
    try:
        for index in range(winreg.QueryInfoKey(root)[0]):
            try:
                subkey_name = winreg.EnumKey(root, index)
                subkey = winreg.OpenKey(root, subkey_name)
                value, _ = winreg.QueryValueEx(subkey, "HardwareInformation.qwMemorySize")
                pnp, _ = winreg.QueryValueEx(subkey, "MatchingDeviceId")
                result[str(pnp)] = _number(value)
                try:
                    name, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                    result[str(name)] = _number(value)
                except OSError:
                    pass
            except OSError:
                continue
    finally:
        winreg.CloseKey(root)
    return result


def detect_windows() -> list[GPUInfo]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return parse_windows_adapters(result.stdout, _registry_sizes())


def detect_generic(os_name: str | None = None) -> list[GPUInfo]:
    current = os_name or platform.system().lower()
    if current == "windows":
        return detect_windows()
    if current == "linux":
        return detect_linux_sysfs()
    return []
