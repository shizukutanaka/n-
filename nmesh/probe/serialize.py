from __future__ import annotations

from collections.abc import Mapping

from .models import (
    GPUInfo,
    HardwareProfile,
    OperatingSystem,
    Tier,
    Vendor,
    VramSource,
)

_OS = {"windows", "linux", "macos"}
_VENDORS = {"nvidia", "amd", "intel", "apple", "unknown"}


def _operating_system(value: str) -> OperatingSystem:
    if value == "windows":
        return "windows"
    if value == "linux":
        return "linux"
    if value == "macos":
        return "macos"
    raise ValueError("profile field 'os' must be windows, linux, or macos")


def _vendor(value: str) -> Vendor:
    if value == "nvidia":
        return "nvidia"
    if value == "amd":
        return "amd"
    if value == "intel":
        return "intel"
    if value == "apple":
        return "apple"
    if value == "unknown":
        return "unknown"
    raise ValueError("profile field 'vendor' is invalid")


def _vram_source(value: str) -> VramSource:
    if value == "nvml":
        return "nvml"
    if value == "smi":
        return "smi"
    if value == "registry":
        return "registry"
    if value == "sysfs":
        return "sysfs"
    if value == "unknown":
        return "unknown"
    raise ValueError("profile field 'vram_source' is invalid")


def _field(data: Mapping[str, object], name: str) -> object:
    if name not in data:
        raise ValueError(f"profile field {name!r} is missing")
    return data[name]


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"profile field {name!r} must be an integer")
    if value < 0:
        raise ValueError(f"profile field {name!r} must be non-negative")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"profile field {name!r} must be an object")
    return value


def _boolean_field(data: Mapping[str, object], name: str) -> bool:
    value = _field(data, name)
    if not isinstance(value, bool):
        raise TypeError(f"profile field {name!r} must be a boolean")
    return value


def _string_map(
    values: Mapping[str, object],
    name: str,
    *,
    nullable: bool = False,
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"profile field {name} keys must be strings")
        if value is None and nullable:
            result[key] = None
        elif not isinstance(value, str):
            raise TypeError(f"profile field {name}.{key} must be a string")
        else:
            result[key] = value
    return result


def _required_string_map(
    values: Mapping[str, object],
    name: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError(f"profile field {name} keys must be strings")
        if not isinstance(value, str):
            raise TypeError(f"profile field {name}.{key} must be a string")
        result[key] = value
    return result


def profile_from_dict(data: Mapping[str, object]) -> HardwareProfile:
    if not isinstance(data, Mapping):
        raise TypeError("profile must be an object")
    os_value = _field(data, "os")
    if not isinstance(os_value, str):
        raise TypeError("profile field 'os' must be a string")
    os_name = _operating_system(os_value)
    gpus_value = _field(data, "gpus")
    if not isinstance(gpus_value, list):
        raise TypeError("profile field 'gpus' must be a list")
    gpus: list[GPUInfo] = []
    for index, raw in enumerate(gpus_value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"profile field 'gpus[{index}]' must be an object")
        vendor = _field(raw, "vendor")
        if not isinstance(vendor, str):
            raise TypeError(f"profile field 'gpus[{index}].vendor' must be a string")
        vendor = _vendor(vendor)
        cap = raw.get("compute_capability")
        if cap is not None:
            if not isinstance(cap, (list, tuple)):
                raise TypeError(
                    f"profile field 'gpus[{index}].compute_capability' must be a list"
                )
            if len(cap) != 2:
                raise ValueError(f"profile field 'gpus[{index}].compute_capability' is invalid")
            for item in cap:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise TypeError(
                        f"profile field 'gpus[{index}].compute_capability' "
                        "must contain integers"
                    )
                if item < 0:
                    raise ValueError(
                        f"profile field 'gpus[{index}].compute_capability' "
                        "must be non-negative"
                    )
            capability = (cap[0], cap[1])
        else:
            capability = None
        driving_display = _field(raw, "driving_display")
        if not isinstance(driving_display, bool):
            raise TypeError(
                f"profile field 'gpus[{index}].driving_display' must be a boolean"
            )
        vram_source_value = raw.get("vram_source", "unknown")
        if not isinstance(vram_source_value, str):
            raise TypeError(
                f"profile field 'gpus[{index}].vram_source' must be a string"
            )
        vram_source = _vram_source(vram_source_value)
        gpu_name = _field(raw, "name")
        if not isinstance(gpu_name, str):
            raise TypeError(f"profile field 'gpus[{index}].name' must be a string")
        gpus.append(GPUInfo(
            _nonnegative_int(_field(raw, "index"), f"gpus[{index}].index"),
            gpu_name,
            vendor,
            _nonnegative_int(_field(raw, "total_vram_bytes"), f"gpus[{index}].total_vram_bytes"),
            _nonnegative_int(_field(raw, "free_vram_bytes"), f"gpus[{index}].free_vram_bytes"),
            capability,
            driving_display,
            vram_source,
        ))
    available = _mapping(data.get("available_backends", {}), "available_backends")
    flags_raw = _mapping(data.get("backend_flags", {}), "backend_flags")
    paths_raw = _mapping(data.get("backend_paths", {}), "backend_paths")
    devices_raw = _mapping(data.get("backend_gpu_devices", {}), "backend_gpu_devices")
    flags: dict[str, tuple[str, ...]] = {}
    for name, values in flags_raw.items():
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"profile field 'backend_flags.{name}' must be a list")
        if not all(isinstance(item, str) for item in values):
            raise TypeError(f"profile field 'backend_flags.{name}' must contain strings")
        flags[str(name)] = tuple(values)
    devices: dict[str, tuple[str, ...]] = {}
    for name, values in devices_raw.items():
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"profile field 'backend_gpu_devices.{name}' must be a list")
        if not all(isinstance(item, str) for item in values):
            raise TypeError(
                f"profile field 'backend_gpu_devices.{name}' must contain strings"
            )
        devices[str(name)] = tuple(values)
    # Doctor JSON carries localized prose, not stable warning keys, so re-feeding it
    # would print untranslatable text as if it were a key.
    tier_value = _field(data, "tier")
    try:
        tier = Tier(tier_value)
    except TypeError as error:
        raise TypeError("profile field 'tier' must be a string") from error
    except ValueError as error:
        raise ValueError("profile field 'tier' is invalid") from error
    return HardwareProfile(
        os_name, str(_field(data, "cpu_name")),
        _nonnegative_int(_field(data, "physical_cores"), "physical_cores"),
        _nonnegative_int(_field(data, "logical_cores"), "logical_cores"),
        _nonnegative_int(_field(data, "total_ram_bytes"), "total_ram_bytes"),
        _nonnegative_int(_field(data, "available_ram_bytes"), "available_ram_bytes"),
        _nonnegative_int(_field(data, "free_disk_bytes"), "free_disk_bytes"),
        _boolean_field(data, "unified_memory"), gpus,
        _string_map(available, "available_backends", nullable=True),
        tier, [], flags,
        _required_string_map(paths_raw, "backend_paths"), devices, [],
    )
