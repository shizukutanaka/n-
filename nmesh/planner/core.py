from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from nmesh import __version__
from nmesh.catalog import ModelSpec
from nmesh.i18n import t
from nmesh.paths import nmesh_home
from nmesh.probe import GPUInfo, HardwareProfile, Tier

BPW = {
    "f16": 16.0, "q8_0": 8.5, "q6_k": 6.6, "q5_k_m": 5.7,
    "q4_k_m": 4.85, "q4_0": 4.55, "q3_k_m": 3.9, "q2_k": 3.35,
}
QUANT_PENALTY = {
    "f16": 0.0, "q8_0": 0.5, "q6_k": 1.0, "q5_k_m": 2.0,
    "q4_k_m": 3.5, "q4_0": 5.0, "q3_k_m": 9.0, "q2_k": 16.0,
}
GIB = 1024**3
PLAN_PATH = nmesh_home() / "plan.json"
INSTALL_HINTS = {
    "ollama": "install.ollama",
    "llamacpp": "install.llamacpp",
    "vllm": "install.vllm",
    "mlx": "install.mlx",
}


@dataclass(frozen=True)
class MemoryEstimate:
    weight_bytes: float
    per_layer_bytes: float
    kv_bytes_per_tok: float
    kv_cache_bytes: float
    compute_overhead: float
    total_bytes: float
    vram_budget: float
    ram_budget: float
    disk_needed: float
    cpu_bytes: float = 0.0
    gpu_bytes: float = 0.0
    n_gpu_layers: int = 0
    parallel_slots: int = 1


@dataclass
class Policy:
    roles: list[str] = field(default_factory=lambda: ["chat", "code", "embed"])
    min_decode_tps: float = 8.0
    max_context: int | None = None
    prefer: str = "balanced"
    allow_download_gb: float = 60.0
    kv_quant: str = "f16"
    budget_source: str = "total"
    parallel_slots: int | None = None
    lang: str = "en"
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchSpec:
    argv: list[str]
    env: dict[str, str]
    health_url: str | None
    shared_daemon: bool = False


@dataclass(frozen=True)
class PlannedService:
    name: str
    roles: list[str]
    model_id: str
    model_ref: str
    download_repo: str | None
    quant: str
    backend: str
    context: int
    port: int
    gpu_indices: list[int]
    n_gpu_layers: int | None
    resident: bool
    memory: MemoryEstimate
    decode_tps: float
    estimated: bool
    launch: LaunchSpec
    languages: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class RoutingRules:
    mode: str
    role_to_service: dict[str, str]
    aliases: dict[str, str]


@dataclass(frozen=True)
class Plan:
    created_at: str
    nmesh_version: str
    profile: HardwareProfile
    tier: Tier
    policy: Policy
    services: list[PlannedService]
    swap_group: list[str]
    routing: RoutingRules
    warnings: list[str]
    install_hints: list[str]
    total_download_bytes: int
    runnable: bool = True


def _profile_budgets(profile: HardwareProfile | None, source: str = "total") -> tuple[float, float]:
    if source not in {"total", "free"}:
        raise ValueError(f"Unknown budget source: {source}")
    if profile is None:
        return 0.0, 0.0
    total_vram = sum(
        (gpu.free_vram_bytes if source == "free" else gpu.total_vram_bytes)
        for gpu in profile.gpus
    )
    if profile.unified_memory:
        total_vram = int(
            (profile.available_ram_bytes if source == "free" else profile.total_ram_bytes) * 0.70
        )
    display = any(gpu.driving_display for gpu in profile.gpus)
    ram = profile.available_ram_bytes if source == "free" else profile.total_ram_bytes
    return max(total_vram * 0.92 - (0.8 * GIB if display else 0.0), 0.0), ram * 0.70


def free_budgets(profile: HardwareProfile) -> tuple[float, float]:
    return _profile_budgets(profile, "free")


def _gpu_budget(gpu: GPUInfo, source: str = "total") -> float:
    if source not in {"total", "free"}:
        raise ValueError(f"Unknown budget source: {source}")
    vram = gpu.free_vram_bytes if source == "free" else gpu.total_vram_bytes
    return max(vram * 0.92 - (0.8 * GIB if gpu.driving_display else 0.0), 0.0)


def estimate_memory(
    model: ModelSpec, quant: str, context: int, parallel_slots: int = 1,
    profile: HardwareProfile | None = None, kv_quant: str = "f16",
    budget_source: str = "total",
) -> MemoryEstimate:
    if quant not in BPW:
        raise ValueError(f"Unsupported quantization: {quant}")
    if kv_quant not in {"f16", "q8_0"}:
        raise ValueError(f"Unsupported KV quantization: {kv_quant}")
    weight_bytes = model.params * BPW[quant] / 8
    per_layer_bytes = weight_bytes / model.n_layers
    kv_elem_bytes = {"f16": 2, "q8_0": 1}[kv_quant]
    kv_bytes_per_tok = 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_elem_bytes
    kv_cache_bytes = kv_bytes_per_tok * context * parallel_slots
    compute_overhead = 0.06 * weight_bytes + 320 * 1024**2
    vram_budget, ram_budget = _profile_budgets(profile, budget_source)
    return MemoryEstimate(
        weight_bytes, per_layer_bytes, kv_bytes_per_tok, kv_cache_bytes, compute_overhead,
        weight_bytes + kv_cache_bytes + compute_overhead, vram_budget, ram_budget,
        weight_bytes * 1.05,
    )


def solve_gpu_layers(memory: MemoryEstimate, n_layers: int) -> int:
    if memory.per_layer_bytes <= 0:
        return 0
    raw = math.floor(
        (memory.vram_budget - memory.kv_cache_bytes - memory.compute_overhead)
        / memory.per_layer_bytes
    )
    return max(0, min(n_layers, raw))


def _gpu_bandwidth(gpu: GPUInfo) -> float:
    known = {"4090": 1008.0, "3060": 360.0, "1650": 192.0, "a100": 1555.0, "h100": 2039.0}
    for name, bandwidth in known.items():
        if name in gpu.name.lower():
            return bandwidth
    return {"nvidia": 400.0, "amd": 350.0, "apple": 200.0, "intel": 200.0}.get(gpu.vendor, 200.0)


def _throughput(model: ModelSpec, memory: MemoryEstimate, layers: int,
                profile: HardwareProfile) -> float:
    gpu_frac = layers / model.n_layers
    if gpu_frac == 0 or not profile.gpus:
        effective = 40.0
    else:
        gpu_bw = sum(_gpu_bandwidth(gpu) for gpu in profile.gpus) / len(profile.gpus)
        effective = 1.0 / (gpu_frac / gpu_bw + (1.0 - gpu_frac) / 40.0)
    return 0.75 * effective * 1e9 / memory.weight_bytes


def _backend(profile: HardwareProfile, model: ModelSpec, layers: int) -> tuple[str, bool]:
    nvidia = any(gpu.vendor == "nvidia" for gpu in profile.gpus)
    full_gpu = layers >= model.n_layers
    if profile.os == "linux" and nvidia and full_gpu and "hf" in model.sources:
        name = "vllm"
    elif profile.unified_memory and "hf" in model.sources:
        name = "mlx"
    elif "hf_gguf" in model.sources:
        name = "llamacpp"
    elif "ollama" in model.sources:
        name = "ollama"
    elif profile.os == "linux" and "hf" in model.sources:
        name = "vllm"
    elif profile.unified_memory and "hf" in model.sources:
        name = "mlx"
    else:
        return "", False
    return name, bool(profile.available_backends.get(name))


def _source_for(backend: str, model: ModelSpec, quant: str) -> str:
    if backend in {"vllm", "mlx"}:
        return model.sources["hf"]
    if backend == "llamacpp":
        return str(nmesh_home() / "models" / f"{model.id}-{quant}.gguf")
    return model.sources["ollama"]


def _service_port_base() -> int:
    try:
        return int(os.environ.get("NMESH_SERVICE_PORT_BASE", "18010"))
    except ValueError:
        return 18010


def _has_source(backend: str, model: ModelSpec) -> bool:
    return {
        "vllm": "hf",
        "mlx": "hf",
        "llamacpp": "hf_gguf",
        "ollama": "ollama",
    }.get(backend, "") in model.sources


def _launch(
    backend: str,
    model: ModelSpec,
    quant: str,
    context: int,
    port: int,
    layers: int,
    tensor_parallel: int,
    slots: int = 1,
    gpu_fraction: float | None = None,
    backend_flags: frozenset[str] | None = None,
    warnings: list[str] | None = None,
    gpu_devices: tuple[str, ...] | None = None,
    language: str = "en",
) -> LaunchSpec:
    ref = _source_for(backend, model, quant)
    if backend == "ollama":
        return LaunchSpec(["ollama", "serve"], {}, "http://127.0.0.1:11434/api/tags", True)
    if backend == "vllm":
        argv = ["vllm", "serve", ref, "--host", "127.0.0.1", "--port", str(port),
                "--max-model-len", str(context), "--max-num-seqs", str(slots)]
        if tensor_parallel > 1:
            argv += ["--tensor-parallel-size", str(tensor_parallel)]
        if gpu_fraction is not None:
            argv += ["--gpu-memory-utilization", f"{gpu_fraction:.3f}"]
    elif backend == "mlx":
        argv = ["python", "-m", "mlx_lm.server", "--model", ref, "--port", str(port)]
    else:
        known = backend_flags is not None
        parallel = not known or any(
            flag in backend_flags for flag in ("-np", "--parallel")
        )
        gpu_layers = (
            gpu_devices != ()
            and (not known or _supports_gpu_layers(backend_flags))
        )
        tensor_split = not known or "--tensor-split" in backend_flags
        argv = [
            "llama-server", "-m", ref, "-c",
            str(context * slots if parallel else context),
        ]
        if parallel:
            argv += ["--parallel", str(slots)]
        elif warnings is not None:
            warnings.append(
                t("warn.parallel_unsupported", language)
            )
        argv += ["--port", str(port)]
        if gpu_layers:
            argv += ["-ngl", str(layers)]
        elif warnings is not None and gpu_devices != ():
            warnings.append(
                t("warn.gpu_layers_unsupported", language)
            )
        if tensor_parallel > 1 and tensor_split:
            argv += ["--tensor-split", ",".join(["1"] * tensor_parallel)]
        elif tensor_parallel > 1 and warnings is not None:
            warnings.append(t("warn.tensor_split_unsupported", language))
    health_path = "/health" if backend == "llamacpp" else "/v1/models"
    return LaunchSpec(argv, {}, f"http://127.0.0.1:{port}{health_path}")


@dataclass(frozen=True)
class _Candidate:
    model: ModelSpec
    quant: str
    context: int
    memory: MemoryEstimate
    n_gpu_layers: int
    decode_tps: float
    backend: str
    installed: bool
    score: float
    estimated: bool


def _bench_value(cache: Mapping[object, float] | None, model: ModelSpec, quant: str,
                 backend: str, gpu_name: str, layers: int) -> float | None:
    if cache is None:
        return None
    for key in (
        (model.id, quant, backend, gpu_name, layers),
        f"{model.id}|{quant}|{backend}|{gpu_name}|{layers}",
        f"{model.id}:{quant}:{backend}:{gpu_name}:{layers}",
    ):
        if key in cache:
            return float(cache[key])
    return None


def _candidate_for(model: ModelSpec, profile: HardwareProfile, policy: Policy,
                   cache: Mapping[object, float] | None) -> list[_Candidate]:
    initial = min(model.max_context, policy.max_context or 8192)
    contexts = list(dict.fromkeys(context for context in (initial, 4096, 2048) if context <= initial))
    candidates: list[_Candidate] = []
    for quant in BPW:
        for context in contexts:
            base = estimate_memory(
                model, quant, context, profile=profile, kv_quant=policy.kv_quant,
                budget_source=policy.budget_source,
            )
            if model.roles == ["embed"]:
                activation = min(0.02 * base.weight_bytes * math.ceil(context / 512), 512 * 1024**2)
                overhead = base.compute_overhead + activation
                base = replace(
                    base,
                    kv_bytes_per_tok=0.0,
                    kv_cache_bytes=0.0,
                    compute_overhead=overhead,
                    total_bytes=base.weight_bytes + overhead,
                )
            layers = solve_gpu_layers(base, model.n_layers) if profile.gpus else 0
            if profile.tier == Tier.T0_CPU:
                layers = 0
            gpu_bytes, cpu_bytes = _split_memory(base, model.n_layers, layers)
            if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                continue
            backend, installed = _backend(profile, model, layers)
            if backend == "llamacpp" and profile.backend_gpu_devices.get("llamacpp") == ():
                layers = 0
                gpu_bytes, cpu_bytes = _split_memory(base, model.n_layers, layers)
                if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                    continue
            if not _has_source(backend, model):
                continue
            gpu_name = profile.gpus[0].name if profile.gpus else "cpu"
            bench = _bench_value(cache, model, quant, backend, gpu_name, layers)
            memory = MemoryEstimate(**{**asdict(base), "cpu_bytes": cpu_bytes,
                                       "gpu_bytes": gpu_bytes, "n_gpu_layers": layers})
            tps = bench if bench is not None else _throughput(model, memory, layers, profile)
            if tps < policy.min_decode_tps:
                continue
            wq, ws = {"quality": (1.0, 0.1), "speed": (0.5, 1.0),
                      "balanced": (1.0, 0.25)}.get(policy.prefer, (1.0, 0.25))
            score = (model.quality - QUANT_PENALTY[quant]) * wq + min(tps, 30) / 30 * 100 * ws
            if policy.languages:
                covers = set(policy.languages).issubset(model.languages)
                score *= 1.0 if covers else 0.7
            candidates.append(_Candidate(model, quant, context, memory, layers, tps,
                                         backend, installed, score, bench is None))
            break
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _split_memory(memory: MemoryEstimate, model_layers: int, layers: int) -> tuple[float, float]:
    on_gpu = layers > 0
    gpu_bytes = memory.per_layer_bytes * layers + (
        memory.kv_cache_bytes + memory.compute_overhead if on_gpu else 0.0
    )
    cpu_bytes = memory.per_layer_bytes * (model_layers - layers) + (
        0.0 if on_gpu else memory.kv_cache_bytes + memory.compute_overhead
    )
    return gpu_bytes, cpu_bytes


GPU_LAYER_FLAGS = ("-ngl", "--gpu-layers", "--n-gpu-layers")


def _supports_gpu_layers(
    flags: frozenset[str] | tuple[str, ...] | None,
) -> bool:
    return flags is None or any(flag in flags for flag in GPU_LAYER_FLAGS)


def _plan_group(group: list[str], pools: dict[str, list[_Candidate]]) -> _Candidate | None:
    if not group:
        return None
    ids = {candidate.model.id for candidate in pools.get(group[0], [])}
    for role in group[1:]:
        ids &= {candidate.model.id for candidate in pools.get(role, [])}
    return next((candidate for candidate in pools[group[0]] if candidate.model.id in ids), None)


def _add_service(group: list[str], candidate: _Candidate, profile: HardwareProfile,
                 services: list[PlannedService], swap_group: list[str],
                 role_to_service: dict[str, str], hints: list[str],
                 budget_source: str, warnings: list[str],
                 language: str = "en") -> None:
    if not candidate.installed:
        hints.append(t(INSTALL_HINTS[candidate.backend], language))
    indices: list[int] = []
    tensor_parallel = 1
    name = group[0]
    port = _service_port_base() + len(services)
    layers = candidate.n_gpu_layers
    memory = candidate.memory
    backend_flags = profile.backend_flags.get(candidate.backend)
    gpu_devices = profile.backend_gpu_devices.get(candidate.backend)
    if (
        candidate.backend == "llamacpp"
        and (not _supports_gpu_layers(backend_flags) or gpu_devices == ())
    ):
        model_layers = max(
            1, round(memory.weight_bytes / memory.per_layer_bytes)
        )
        layers = 0
        gpu_bytes, cpu_bytes = _split_memory(memory, model_layers, layers)
        memory = replace(
            memory, n_gpu_layers=layers, gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes
        )
    launch = _launch(
        candidate.backend, candidate.model, candidate.quant, candidate.context,
        port, layers or 0, tensor_parallel,
        backend_flags=backend_flags,
        warnings=warnings,
        gpu_devices=gpu_devices,
        language=language,
    )
    if candidate.backend == "llamacpp" and "hf_gguf" in candidate.model.sources:
        launch = replace(launch, env={"NMESH_HF_REPO": candidate.model.sources["hf_gguf"]})
    service = PlannedService(
        name, group, candidate.model.id, _source_for(candidate.backend, candidate.model, candidate.quant),
        candidate.model.sources.get("hf_gguf") or candidate.model.sources.get("hf"),
        candidate.quant, candidate.backend, candidate.context,
        11434 if candidate.backend == "ollama" else port, indices,
        None if candidate.backend in {"vllm", "mlx", "ollama"} else layers,
        profile.tier not in {Tier.T0_CPU, Tier.T1_LOW} or not services,
        memory, candidate.decode_tps, candidate.estimated, launch,
        candidate.model.languages,
    )
    services.append(service)
    for role in group:
        role_to_service[role] = name
    if not service.resident:
        swap_group.append(name)


def _rebuild_launch(service: PlannedService, tensor_parallel: int,
                    layers: int | None,
                    backend_flags: frozenset[str] | None = None,
                    warnings: list[str] | None = None,
                    gpu_devices: tuple[str, ...] | None = None,
                    language: str = "en") -> LaunchSpec:
    argv = list(service.launch.argv)
    if service.backend == "vllm":
        if tensor_parallel > 1:
            if "--tensor-parallel-size" in argv:
                argv[argv.index("--tensor-parallel-size") + 1] = str(tensor_parallel)
            else:
                argv.extend(["--tensor-parallel-size", str(tensor_parallel)])
        elif "--tensor-parallel-size" in argv:
            index = argv.index("--tensor-parallel-size")
            del argv[index:index + 2]
    elif service.backend == "llamacpp":
        if tensor_parallel > 1:
            value = ",".join(["1"] * tensor_parallel)
            supported = backend_flags is None or "--tensor-split" in backend_flags
            if supported and "--tensor-split" in argv:
                argv[argv.index("--tensor-split") + 1] = value
            elif supported:
                argv.extend(["--tensor-split", value])
            else:
                if "--tensor-split" in argv:
                    index = argv.index("--tensor-split")
                    del argv[index:index + 2]
                if warnings is not None:
                    warnings.append(
                        t("warn.tensor_split_unsupported", language)
                    )
        elif "--tensor-split" in argv:
            index = argv.index("--tensor-split")
            del argv[index:index + 2]
        gpu_layer_flags = GPU_LAYER_FLAGS
        gpu_layers_supported = gpu_devices != () and _supports_gpu_layers(backend_flags)
        if layers is not None and gpu_layers_supported:
            for flag in gpu_layer_flags:
                if flag in argv:
                    argv[argv.index(flag) + 1] = str(layers)
                    break
        elif not gpu_layers_supported:
            removed = False
            for flag in gpu_layer_flags:
                if flag in argv:
                    index = argv.index(flag)
                    del argv[index:index + 2]
                    removed = True
            if removed and warnings is not None and gpu_devices != ():
                warnings.append(
                    t("warn.gpu_layers_unsupported", language)
                )
    return replace(service.launch, argv=argv)


def _cpu_llamacpp_service(
    service: PlannedService,
    backend_flags: frozenset[str] | tuple[str, ...] | None,
    warnings: list[str],
    gpu_devices: tuple[str, ...] | None = None,
    language: str = "en",
) -> PlannedService:
    if (
        service.backend != "llamacpp"
        or (_supports_gpu_layers(backend_flags) and gpu_devices != ())
    ):
        return service
    model_layers = max(
        1, round(service.memory.weight_bytes / service.memory.per_layer_bytes)
    )
    memory = replace(service.memory, n_gpu_layers=0)
    gpu_bytes, cpu_bytes = _split_memory(memory, model_layers, 0)
    memory = replace(memory, gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes)
    launch = _rebuild_launch(
        service, 1, 0, backend_flags, warnings,
        gpu_devices=gpu_devices, language=language,
    )
    return replace(service, n_gpu_layers=0, memory=memory, launch=launch)


def _place_services(
    services: list[PlannedService],
    profile: HardwareProfile,
    policy: Policy,
    swap_group: Sequence[str],
    warnings: list[str],
) -> list[PlannedService]:
    if not profile.gpus:
        return services
    services = [
        _cpu_llamacpp_service(
            service, profile.backend_flags.get(service.backend), warnings,
            profile.backend_gpu_devices.get(service.backend), policy.lang,
        )
        for service in services
    ]
    indices = [gpu.index for gpu in profile.gpus]
    budgets = {
        gpu.index: _gpu_budget(gpu, policy.budget_source) for gpu in profile.gpus
    }
    ram_budget = _profile_budgets(profile, policy.budget_source)[1]
    ram_used = 0.0
    remaining = dict(budgets)
    swap_reserved = {index: 0.0 for index in indices}
    swap_names = set(swap_group)
    ordered = sorted(
        services,
        key=lambda item: (item.name in swap_names, -item.memory.gpu_bytes),
    )
    placed: dict[str, PlannedService] = {}
    for service in ordered:
        if service.memory.gpu_bytes <= 0:
            placed[service.name] = service
            ram_used += service.memory.cpu_bytes
            continue
        is_swap = service.name in swap_names
        fits = [
            index for index in indices
            if service.memory.gpu_bytes
            <= remaining[index] + (swap_reserved[index] if is_swap else 0.0) + 1
        ]
        placement_budget = 0.0
        if fits:
            target = min(fits, key=lambda index: (remaining[index], index))
            assigned = [target]
            tensor_parallel = 1
            placement_budget = remaining[target] + (
                swap_reserved[target] if is_swap else 0.0
            )
            committed = service.memory.gpu_bytes
            if is_swap:
                committed = max(swap_reserved[target], committed)
                remaining[target] -= committed - swap_reserved[target]
                swap_reserved[target] = committed
            else:
                remaining[target] -= committed
        elif service.backend == "vllm" and len(indices) > 1:
            assigned = indices
            tensor_parallel = len(indices)
            committed = service.memory.gpu_bytes / tensor_parallel
            for index in indices:
                if is_swap:
                    new_reserved = max(swap_reserved[index], committed)
                    remaining[index] -= new_reserved - swap_reserved[index]
                    swap_reserved[index] = new_reserved
                else:
                    remaining[index] -= committed
        elif service.backend in {"ollama", "vllm", "mlx"}:
            total_bytes = service.memory.cpu_bytes + service.memory.gpu_bytes
            if ram_used + total_bytes <= ram_budget + 1:
                placed[service.name] = replace(
                    service,
                    gpu_indices=[],
                    memory=replace(
                        service.memory,
                        gpu_bytes=0.0,
                        cpu_bytes=total_bytes,
                    ),
                )
                ram_used += total_bytes
                warnings.append(
                    t("warn.backend_placement_estimate", policy.lang,
                      service=service.name, backend=service.backend)
                )
                continue
            target = max(indices, key=lambda index: (remaining[index], -index))
            assigned = [target]
            tensor_parallel = 1
            placement_budget = remaining[target] + (
                swap_reserved[target] if is_swap else 0.0
            )
            committed = service.memory.gpu_bytes
            remaining[target] -= committed
            warnings.append(
                t("warn.gpu_over_budget", policy.lang, service=service.name,
                  committed=committed, budget=budgets[target])
            )
        else:
            target = max(indices, key=lambda index: (remaining[index], -index))
            assigned = [target]
            tensor_parallel = 1
            placement_budget = remaining[target] + (
                swap_reserved[target] if is_swap else 0.0
            )
            committed = service.memory.gpu_bytes
            remaining[target] -= committed

        current = replace(service, gpu_indices=assigned)
        if (
            current.backend == "llamacpp"
            and current.n_gpu_layers is not None
            and len(assigned) == 1
        ):
            model_layers = max(
                1, round(
                    current.memory.weight_bytes / current.memory.per_layer_bytes
                ),
            )
            target_budget = placement_budget
            card_memory = replace(current.memory, vram_budget=target_budget)
            layers = solve_gpu_layers(card_memory, model_layers)
            if layers < current.n_gpu_layers:
                adjusted = replace(card_memory, n_gpu_layers=layers)
                gpu_bytes, cpu_bytes = _split_memory(
                    adjusted, model_layers, layers
                )
                if (
                    cpu_bytes <= adjusted.ram_budget + 1
                    and gpu_bytes <= target_budget + 1
                ):
                    old_bytes = current.memory.gpu_bytes
                    current = replace(
                        current,
                        n_gpu_layers=layers,
                        memory=replace(
                            adjusted, gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes
                        ),
                    )
                    if is_swap:
                        committed = max(swap_reserved[assigned[0]], gpu_bytes)
                        remaining[assigned[0]] += (
                            swap_reserved[assigned[0]] - committed
                        )
                        swap_reserved[assigned[0]] = committed
                    else:
                        remaining[assigned[0]] += old_bytes - gpu_bytes
                else:
                    warnings.append(
                        t("warn.layers_reduced", policy.lang, service=service.name,
                          layers=layers, previous=current.n_gpu_layers)
                    )
                    if service.memory.gpu_bytes > target_budget:
                        warnings.append(
                            t("warn.gpu_over_budget", policy.lang,
                              service=service.name,
                              committed=service.memory.gpu_bytes,
                              budget=budgets[assigned[0]])
                        )
            else:
                current = replace(
                    current, memory=card_memory
                )
        if tensor_parallel != 1 or current.n_gpu_layers != service.n_gpu_layers:
            current = replace(
                current,
                launch=_rebuild_launch(
                    current, tensor_parallel, current.n_gpu_layers,
                    profile.backend_flags.get(current.backend), warnings,
                    gpu_devices=profile.backend_gpu_devices.get(current.backend),
                    language=policy.lang,
                ),
            )
        placed[service.name] = current
        ram_used += current.memory.cpu_bytes
    return [placed[service.name] for service in services]


SLOT_CAPS = {"llamacpp": 8, "vllm": 32, "mlx": 1, "ollama": 1}
SLOT_SPARE_FRACTION = 0.5


def _assign_slots(
    services: list[PlannedService],
    profile: HardwareProfile,
    policy: Policy,
    swap_group: Sequence[str] = (),
    warnings: list[str] | None = None,
) -> list[PlannedService]:
    vram_budget = {
        gpu.index: _gpu_budget(gpu, policy.budget_source) for gpu in profile.gpus
    }
    ram_budget = _profile_budgets(profile, policy.budget_source)[1]
    swap_names = set(swap_group)

    def domain(service: PlannedService) -> tuple[str, tuple[int, ...]]:
        if service.gpu_indices:
            return "gpu", tuple(service.gpu_indices)
        if profile.gpus and service.n_gpu_layers is not None and service.n_gpu_layers > 0:
            return "gpu", tuple(gpu.index for gpu in profile.gpus)
        return "cpu", ()

    def budget(key: tuple[str, tuple[int, ...]]) -> float:
        kind, indices = key
        if kind == "cpu":
            return ram_budget
        return sum(vram_budget.get(index, 0.0) for index in indices)

    def effective_layers(service: PlannedService, model_layers: int) -> int:
        if service.n_gpu_layers is not None:
            return service.n_gpu_layers
        return model_layers if service.gpu_indices else 0

    grouped: dict[tuple[str, tuple[int, ...]], list[PlannedService]] = {}
    for service in services:
        grouped.setdefault(domain(service), []).append(service)
    usage: dict[tuple[str, tuple[int, ...]], float] = {}
    swap_usage: dict[tuple[str, tuple[int, ...]], dict[str, float]] = {}
    swap_accounted: dict[tuple[str, tuple[int, ...]], float] = {}
    for key, members in grouped.items():
        resident = [item for item in members if item.name not in swap_names]
        swapped = [item for item in members if item.name in swap_names]
        usage[key] = sum(
            item.memory.gpu_bytes if key[0] == "gpu" else item.memory.cpu_bytes
            for item in resident
        )
        if swapped:
            swap_usage[key] = {
                item.name: (
                    item.memory.gpu_bytes
                    if key[0] == "gpu"
                    else item.memory.cpu_bytes
                )
                for item in swapped
            }
            swap_accounted[key] = max(swap_usage[key].values(), default=0.0)
            usage[key] += swap_accounted[key]

    result: list[PlannedService] = []
    for service in services:
        cap = SLOT_CAPS.get(service.backend, 1)
        model_layers = max(
            1, round(service.memory.weight_bytes / service.memory.per_layer_bytes)
        )
        full_gpu = (
            service.n_gpu_layers is None
            or service.n_gpu_layers >= model_layers
        )
        eligible = (
            cap > 1
            and full_gpu
            and service.memory.kv_bytes_per_tok > 0
        )
        if service.backend == "llamacpp":
            flags = profile.backend_flags.get("llamacpp")
            if flags is not None and not any(
                flag in flags for flag in ("-np", "--parallel")
            ):
                eligible = False
        requested = policy.parallel_slots
        if requested is not None:
            requested = max(1, requested)
        slots = 1
        key = domain(service)
        leftover = max(budget(key) - usage[key], 0.0)
        if eligible and requested != 1:
            kv_per_slot = service.memory.kv_bytes_per_tok * service.context
            if kv_per_slot > 0:
                if requested is None:
                    available_extra = math.floor(
                        leftover * SLOT_SPARE_FRACTION / kv_per_slot
                    )
                    slots = min(cap, 1 + max(available_extra, 0))
                else:
                    available = math.floor(leftover / kv_per_slot)
                    slots = min(cap, requested, 1 + max(available, 0))
                if requested is not None and slots != requested and warnings is not None:
                    warnings.append(
                        t("warn.slots_clamped", policy.lang, service=service.name,
                          requested=requested, slots=slots)
                    )
        slots = max(1, slots)
        if slots > 1:
            kv_cache_bytes = service.memory.kv_bytes_per_tok * service.context * slots
            rewritten = replace(
                service.memory,
                kv_cache_bytes=kv_cache_bytes,
                total_bytes=(
                    service.memory.weight_bytes
                    + kv_cache_bytes
                    + service.memory.compute_overhead
                ),
                parallel_slots=slots,
            )
            layers = effective_layers(service, model_layers)
            gpu_bytes, cpu_bytes = _split_memory(rewritten, model_layers, layers)
            domain_fit = gpu_bytes <= budget(key) + 1 if key[0] == "gpu" else True
            if not domain_fit or cpu_bytes > ram_budget + 1:
                slots = 1
        kv_cache_bytes = service.memory.kv_bytes_per_tok * service.context * slots
        rewritten = replace(
            service.memory,
            kv_cache_bytes=kv_cache_bytes,
            total_bytes=(
                service.memory.weight_bytes
                + kv_cache_bytes
                + service.memory.compute_overhead
            ),
            parallel_slots=slots,
        )
        gpu_bytes, cpu_bytes = _split_memory(
            rewritten,
            model_layers,
            effective_layers(service, model_layers),
        )
        rewritten = replace(rewritten, gpu_bytes=gpu_bytes, cpu_bytes=cpu_bytes)
        gpu_fraction = None
        if service.backend == "vllm" and profile.gpus:
            indices = service.gpu_indices or [gpu.index for gpu in profile.gpus]
            total_vram = sum(
                gpu.total_vram_bytes for gpu in profile.gpus if gpu.index in indices
            )
            if total_vram > 0:
                gpu_fraction = min(
                    0.95, max(0.10, rewritten.gpu_bytes / total_vram)
                )
        launch = _rewrite_launch(
            service, slots, gpu_fraction,
            profile.backend_flags.get(service.backend), warnings, policy.lang,
        )
        result.append(replace(service, memory=rewritten, launch=launch))
        old_bytes = (
            service.memory.gpu_bytes if key[0] == "gpu" else service.memory.cpu_bytes
        )
        new_bytes = rewritten.gpu_bytes if key[0] == "gpu" else rewritten.cpu_bytes
        if service.name in swap_names:
            swap_usage.setdefault(key, {})[service.name] = new_bytes
            new_accounted = max(swap_usage[key].values())
            usage[key] += new_accounted - swap_accounted.get(key, 0.0)
            swap_accounted[key] = new_accounted
        else:
            usage[key] += new_bytes - old_bytes
    return result


def _rewrite_launch(
    service: PlannedService, slots: int, gpu_fraction: float | None,
    backend_flags: frozenset[str] | None = None,
    warnings: list[str] | None = None,
    language: str = "en",
) -> LaunchSpec:
    argv = list(service.launch.argv)
    if service.backend == "llamacpp":
        known = backend_flags is not None
        parallel = not known or any(
            flag in backend_flags for flag in ("-np", "--parallel")
        )
        if "-c" in argv:
            argv[argv.index("-c") + 1] = str(
                service.context * slots if parallel else service.context
            )
        else:
            argv.extend([
                "-c", str(service.context * slots if parallel else service.context)
            ])
        if parallel and "--parallel" in argv:
            argv[argv.index("--parallel") + 1] = str(slots)
        elif parallel:
            argv.extend(["--parallel", str(slots)])
        elif "--parallel" in argv:
            index = argv.index("--parallel")
            del argv[index:index + 2]
        if not parallel and warnings is not None and slots > 1:
            warnings.append(
                t("warn.parallel_clamped", language, requested=slots)
            )
    elif service.backend == "vllm":
        if "--max-num-seqs" in argv:
            argv[argv.index("--max-num-seqs") + 1] = str(slots)
        else:
            argv.extend(["--max-num-seqs", str(slots)])
        if gpu_fraction is not None:
            formatted = f"{gpu_fraction:.3f}"
            if "--gpu-memory-utilization" in argv:
                argv[argv.index("--gpu-memory-utilization") + 1] = formatted
            else:
                argv.extend(["--gpu-memory-utilization", formatted])
    return replace(service.launch, argv=argv)


def build_plan(profile: HardwareProfile, catalog: Sequence[ModelSpec],
               policy: Policy | None = None,
               bench_cache: Mapping[object, float] | None = None) -> Plan:
    selected = policy or Policy()
    roles = list(dict.fromkeys(selected.roles))
    warning_params = profile.warning_params
    warnings = [
        t(
            warning,
            selected.lang,
            **(warning_params[index] if index < len(warning_params) else {}),
        )
        for index, warning in enumerate(profile.warnings)
    ]
    if profile.backend_gpu_devices.get("llamacpp") == () and profile.gpus:
        warnings.append(
            t("warn.backend_no_gpu", selected.lang)
        )
    hints: list[str] = []
    for model in catalog:
        if (
            any(role in model.roles for role in roles)
            and not any(source in model.sources for source in ("hf", "hf_gguf", "ollama"))
        ):
            warnings.append(
                t("warn.no_source", selected.lang, model=model.id)
            )
    pools = {role: sorted(
        (candidate for model in catalog if role in model.roles
         for candidate in _candidate_for(model, profile, selected, bench_cache)),
        key=lambda item: item.score, reverse=True,
    ) for role in roles}
    if profile.tier in {Tier.T0_CPU, Tier.T1_LOW, Tier.T2_MID, Tier.T3_HIGH}:
        groups = [[role for role in roles if role in {"chat", "code"}]]
        groups += [[role] for role in roles if role == "embed"]
    else:
        groups = [[role] for role in roles]
    services: list[PlannedService] = []
    swap_group: list[str] = []
    role_to_service: dict[str, str] = {}
    total_download = 0
    for group in [item for item in groups if item]:
        candidate = _plan_group(group, pools)
        if candidate is None and len(group) > 1:
            for role in group:
                role_candidate = pools.get(role, [])
                if role_candidate:
                    _add_service(
                        [role], role_candidate[0], profile, services, swap_group,
                        role_to_service, hints, selected.budget_source, warnings,
                        selected.lang,
                    )
                    total_download += int(role_candidate[0].memory.disk_needed)
                else:
                    warnings.append(t("warn.no_candidate", selected.lang, role=role))
            continue
        if candidate is None:
            warnings.append(t("warn.no_candidate", selected.lang, role=group[0]))
            continue
        _add_service(
            group, candidate, profile, services, swap_group, role_to_service, hints,
            selected.budget_source, warnings, selected.lang,
        )
        total_download += int(candidate.memory.disk_needed)
    services = _place_services(services, profile, selected, swap_group, warnings)
    services = _assign_slots(services, profile, selected, swap_group, warnings)
    if total_download > selected.allow_download_gb * GIB:
        warnings.append(t("warn.download_budget", selected.lang))
    if selected.languages:
        requested = set(selected.languages)
        for service in services:
            missing = sorted(requested - set(service.languages))
            if missing:
                warnings.append(t(
                    "warn.language_coverage", selected.lang,
                    model=service.model_id, languages=", ".join(missing),
                ))
    covered = set(role_to_service)
    runnable = bool(services) and covered >= set(roles) and not hints
    return Plan(
        datetime.now(timezone.utc).isoformat(), __version__, profile, profile.tier, selected,
        services, swap_group,
        RoutingRules("rules", role_to_service, {"nmesh-auto": role_to_service.get("chat", "")}),
        list(dict.fromkeys(warnings)), list(dict.fromkeys(hints)),
        total_download, runnable,
    )


def _gpu_from_dict(data: object) -> GPUInfo:
    if not isinstance(data, dict):
        raise TypeError("Invalid GPU data")
    cap = data.get("compute_capability")
    return GPUInfo(
        int(data["index"]), str(data["name"]), str(data["vendor"]),
        int(data["total_vram_bytes"]), int(data["free_vram_bytes"]),
        tuple(cap) if isinstance(cap, list) else cap, bool(data["driving_display"]),
        str(data.get("vram_source", "unknown")),
    )


def _plan_from_dict(data: dict[str, object]) -> Plan:
    pd = data["profile"]
    if not isinstance(pd, dict):
        raise TypeError("Invalid profile")
    profile = HardwareProfile(
        str(pd["os"]), str(pd["cpu_name"]), int(pd["physical_cores"]), int(pd["logical_cores"]),
        int(pd["total_ram_bytes"]), int(pd["available_ram_bytes"]), int(pd["free_disk_bytes"]),
        bool(pd["unified_memory"]), [_gpu_from_dict(item) for item in pd["gpus"]],
        {str(k): v if isinstance(v, str) else None for k, v in pd["available_backends"].items()},
        Tier(str(pd["tier"])), [str(x) for x in pd.get("warnings", [])],
        {
            str(name): tuple(str(flag) for flag in flags)
            for name, flags in pd.get("backend_flags", {}).items()
        },
        {str(name): str(path) for name, path in pd.get("backend_paths", {}).items()},
        {
            str(name): (
                tuple(str(device) for device in devices)
                if isinstance(devices, list)
                else None
            )
            for name, devices in pd.get("backend_gpu_devices", {}).items()
        },
        [
            {str(key): str(value) for key, value in params.items()}
            if isinstance(params, dict)
            else {}
            for params in pd.get("warning_params", [])
        ],
    )
    pol = data["policy"]
    if not isinstance(pol, dict):
        raise TypeError("Invalid policy")
    policy = Policy([str(x) for x in pol["roles"]], float(pol["min_decode_tps"]),
                    int(pol["max_context"]) if pol["max_context"] is not None else None,
                    str(pol["prefer"]), float(pol["allow_download_gb"]),
                    str(pol.get("kv_quant", "f16")), str(pol.get("budget_source", "total")),
                    int(pol["parallel_slots"]) if pol.get("parallel_slots") is not None else None,
                    str(pol.get("lang", "en")),
                    tuple(str(x) for x in pol.get("languages", [])),
                    )
    services: list[PlannedService] = []
    for item in data["services"]:
        sd = item
        if not isinstance(sd, dict):
            raise TypeError("Invalid service")
        md, ld = sd["memory"], sd["launch"]
        if not isinstance(md, dict) or not isinstance(ld, dict):
            raise TypeError("Invalid nested data")
        memory_values = {str(k): v for k, v in md.items()}
        memory_values["n_gpu_layers"] = int(memory_values["n_gpu_layers"])
        memory_values["parallel_slots"] = int(memory_values.get("parallel_slots", 1))
        memory = MemoryEstimate(**memory_values)
        launch = LaunchSpec(
            [str(x) for x in ld["argv"]], {str(k): str(v) for k, v in ld["env"].items()},
            str(ld["health_url"]) if ld["health_url"] else None, bool(ld.get("shared_daemon", False)),
        )
        services.append(PlannedService(
            str(sd["name"]), [str(x) for x in sd["roles"]], str(sd["model_id"]),
            str(sd.get("model_ref", sd["model_id"])),
            str(sd["download_repo"]) if sd.get("download_repo") is not None else None,
            str(sd["quant"]), str(sd["backend"]),
            int(sd["context"]), int(sd["port"]), [int(x) for x in sd["gpu_indices"]],
            int(sd["n_gpu_layers"]) if sd["n_gpu_layers"] is not None else None, bool(sd["resident"]),
            memory, float(sd["decode_tps"]), bool(sd["estimated"]), launch,
            tuple(str(x) for x in sd.get("languages", ["en"])),
        ))
    rd = data["routing"]
    if not isinstance(rd, dict):
        raise TypeError("Invalid routing")
    routing = RoutingRules(str(rd["mode"]), {str(k): str(v) for k, v in rd["role_to_service"].items()},
                           {str(k): str(v) for k, v in rd["aliases"].items()})
    return Plan(
        str(data["created_at"]), str(data["nmesh_version"]), profile, Tier(str(data["tier"])),
        policy, services, [str(x) for x in data["swap_group"]], routing,
        [str(x) for x in data["warnings"]], [str(x) for x in data["install_hints"]],
        int(data["total_download_bytes"]), bool(data.get("runnable", True)),
    )


def save_plan(plan: Plan, path: Path | None = None) -> Path:
    target = path or PLAN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        payload = asdict(plan)
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def load_plan(path: Path | None = None) -> Plan | None:
    target = path or PLAN_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return _plan_from_dict(payload) if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


plan = build_plan
