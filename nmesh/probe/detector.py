from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import psutil

from .caps import llamacpp_caps
from .models import GPUInfo, HardwareProfile, OperatingSystem, classify_tier


def _append_warning(
    warnings: list[str],
    warning_params: list[dict[str, str]] | None,
    key: str,
    **params: object,
) -> None:
    warnings.append(key)
    if warning_params is not None:
        warning_params.append({name: str(value) for name, value in params.items()})


def _run(command: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    return result.stdout, result.stderr


def _parse_cc(value: str) -> tuple[int, int] | None:
    parts = value.strip().split(".")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            index = int(fields[0])
            total = int(float(fields[2]) * 1024**2)
            free = int(float(fields[3]) * 1024**2)
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=index,
                name=fields[1],
                vendor="nvidia",
                total_vram_bytes=total,
                free_vram_bytes=free,
                compute_capability=_parse_cc(fields[4]),
                driving_display=False,
            )
        )
    return gpus


def _number(value: object) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return int(float(cleaned))
        except ValueError:
            return 0
    return 0


def _find_vram(value: object) -> tuple[int, int]:
    if isinstance(value, dict):
        total = 0
        free = 0
        for key, item in value.items():
            lowered = key.lower()
            if "total" in lowered:
                total = max(total, _number(item))
            if "free" in lowered or "avail" in lowered:
                free = max(free, _number(item))
            nested_total, nested_free = _find_vram(item)
            total = max(total, nested_total)
            free = max(free, nested_free)
        return total, free
    if isinstance(value, list):
        totals = [_find_vram(item) for item in value]
        return max((item[0] for item in totals), default=0), max(
            (item[1] for item in totals), default=0
        )
    return 0, 0


def parse_rocm_smi(text: str) -> list[GPUInfo]:
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    gpus: list[GPUInfo] = []
    for key, value in payload.items():
        total, free = _find_vram(value)
        if total == 0:
            continue
        index = _number(key)
        gpus.append(
            GPUInfo(
                index=index,
                name=f"AMD GPU {index}",
                vendor="amd",
                total_vram_bytes=total,
                free_vram_bytes=free or total,
                compute_capability=None,
                driving_display=False,
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.index)


def _detect_nvidia(
    warnings: list[str], warning_params: list[dict[str, str]] | None = None
) -> list[GPUInfo]:
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            gpus: list[GPUInfo] = []
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    capability: tuple[int, int] | None = (major, minor)
                except (AttributeError, RuntimeError, ValueError):
                    capability = None
                gpus.append(
                    GPUInfo(index, str(name), "nvidia", memory.total, memory.free, capability, False)
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        output, _ = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            return parse_nvidia_smi(output)
        _append_warning(warnings, warning_params, "warn.nvidia_unavailable")
        return []


def _detect_rocm(
    warnings: list[str], warning_params: list[dict[str, str]] | None = None
) -> list[GPUInfo]:
    output, _ = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not output:
        return []
    gpus = parse_rocm_smi(output)
    if not gpus:
        _append_warning(warnings, warning_params, "warn.rocm_parse")
    return gpus


def _detect_backends(
    warnings: list[str],
    warning_params: list[dict[str, str]] | None = None,
) -> tuple[dict[str, str | None], dict[str, tuple[str, ...]], dict[str, str]]:
    backends: dict[str, str | None] = {
        "ollama": None,
        "llamacpp": None,
        "vllm": None,
        "mlx": None,
    }
    flags: dict[str, tuple[str, ...]] = {}
    paths: dict[str, str] = {}
    commands: dict[str, list[str]] = {
        "ollama": ["ollama", "--version"],
        "llamacpp": ["llama-server", "--version"],
        "vllm": ["vllm", "--version"],
    }
    for name, command in commands.items():
        if shutil.which(command[0]) is None:
            continue
        executable = Path(shutil.which(command[0]) or command[0]).resolve()
        paths[name] = str(executable)
        output, error = _run(command)
        version_text = (output or error or "").strip()
        if version_text:
            backends[name] = version_text.splitlines()[0]
        if name == "llamacpp":
            caps = llamacpp_caps(str(executable))
            if caps is not None:
                flags[name] = tuple(sorted(caps.flags))
    python_executable = shutil.which("python") or shutil.which("python3")
    if python_executable:
        output, error = _run([python_executable, "-c", "import mlx_lm; print('installed')"])
        if output and "installed" in output:
            backends["mlx"] = "installed"
        elif error and "No module named" not in error:
            _append_warning(warnings, warning_params, "warn.mlx_check")
    return backends, flags, paths


def _os_name() -> OperatingSystem:
    name = platform.system().lower()
    if name == "windows":
        return "windows"
    if name == "darwin":
        return "macos"
    return "linux"


def _mark_display(gpus: list[GPUInfo], os_name: OperatingSystem) -> list[GPUInfo]:
    if not gpus:
        return gpus
    if os_name == "linux":
        driving = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    else:
        driving = True
    first = gpus[0]
    return [
        GPUInfo(
            gpu.index,
            gpu.name,
            gpu.vendor,
            gpu.total_vram_bytes,
            gpu.free_vram_bytes,
            gpu.compute_capability,
            driving if gpu.index == first.index else gpu.driving_display,
        )
        for gpu in gpus
    ]


def detect_hardware() -> HardwareProfile:
    warnings: list[str] = []
    warning_params: list[dict[str, str]] = []
    os_name = _os_name()
    try:
        cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or physical_cores
        memory = psutil.virtual_memory()
        total_ram = int(memory.total)
        available_ram = int(memory.available)
        root = Path.home().anchor or Path.cwd().anchor or "."
        free_disk = int(psutil.disk_usage(root).free)
    except Exception as error:  # noqa: BLE001
        _append_warning(
            warnings, warning_params, "warn.system_probe", error=error
        )
        cpu_name = platform.machine() or "Unknown CPU"
        physical_cores = logical_cores = 1
        total_ram = available_ram = free_disk = 0

    unified = os_name == "macos" and platform.machine().lower() in {"arm64", "aarch64"}
    gpus: list[GPUInfo] = []
    if unified:
        total = int(total_ram * 0.70)
        gpus = [GPUInfo(0, "Apple Silicon", "apple", total, total, None, True)]
    else:
        gpus = _detect_nvidia(warnings, warning_params)
        if not gpus:
            gpus = _detect_rocm(warnings, warning_params)
    gpus = _mark_display(gpus, os_name)
    backends, backend_flags, backend_paths = _detect_backends(
        warnings, warning_params
    )
    tier = classify_tier(gpus, unified, total_ram)
    return HardwareProfile(
        os_name,
        cpu_name,
        physical_cores,
        logical_cores,
        total_ram,
        available_ram,
        free_disk,
        unified,
        gpus,
        backends,
        tier,
        warnings,
        backend_flags,
        backend_paths,
        warning_params,
    )
