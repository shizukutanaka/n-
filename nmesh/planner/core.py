from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nmesh import __version__
from nmesh.catalog import ModelSpec
from nmesh.probe import GPUInfo, HardwareProfile, Tier

BPW: dict[str, float] = {
    "f16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_k_m": 5.7,
    "q4_k_m": 4.85,
    "q4_0": 4.55,
    "q3_k_m": 3.9,
    "q2_k": 3.35,
}
QUANT_PENALTY: dict[str, float] = {
    "f16": 0.0,
    "q8_0": 0.5,
    "q6_k": 1.0,
    "q5_k_m": 2.0,
    "q4_k_m": 3.5,
    "q4_0": 5.0,
    "q3_k_m": 9.0,
    "q2_k": 16.0,
}
GIB = 1024**3


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


@dataclass
class Policy:
    roles: list[str] = field(default_factory=lambda: ["chat", "code", "embed"])
    min_decode_tps: float = 8.0
    max_context: int | None = None
    prefer: str = "balanced"
    allow_download_gb: float = 60.0


@dataclass(frozen=True)
class LaunchSpec:
    argv: list[str]
    env: dict[str, str]
    health_url: str | None


@dataclass(frozen=True)
class PlannedService:
    name: str
    roles: list[str]
    model_id: str
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


def _profile_budgets(profile: HardwareProfile | None) -> tuple[float, float]:
    if profile is None:
        return 0.0, 0.0
    total_vram = sum(gpu.total_vram_bytes for gpu in profile.gpus)
    if profile.unified_memory:
        total_vram = int(profile.total_ram_bytes * 0.70)
    display = any(gpu.driving_display for gpu in profile.gpus)
    vram_budget = total_vram * 0.92 - (0.8 * GIB if display else 0.0)
    return max(vram_budget, 0.0), profile.total_ram_bytes * 0.70


def estimate_memory(
    model: ModelSpec,
    quant: str,
    context: int,
    parallel_slots: int = 1,
    profile: HardwareProfile | None = None,
) -> MemoryEstimate:
    if quant not in BPW:
        raise ValueError(f"Unsupported quantization: {quant}")
    weight_bytes = model.params * BPW[quant] / 8
    per_layer_bytes = weight_bytes / model.n_layers
    kv_elem_bytes = 1 if quant == "q8_0" else 2
    kv_bytes_per_tok = 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_elem_bytes
    kv_cache_bytes = kv_bytes_per_tok * context * parallel_slots
    compute_overhead = 0.06 * weight_bytes + 320 * 1024**2
    total_bytes = weight_bytes + kv_cache_bytes + compute_overhead
    vram_budget, ram_budget = _profile_budgets(profile)
    return MemoryEstimate(
        weight_bytes=weight_bytes,
        per_layer_bytes=per_layer_bytes,
        kv_bytes_per_tok=kv_bytes_per_tok,
        kv_cache_bytes=kv_cache_bytes,
        compute_overhead=compute_overhead,
        total_bytes=total_bytes,
        vram_budget=vram_budget,
        ram_budget=ram_budget,
        disk_needed=weight_bytes * 1.05,
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
    known = {
        "4090": 1008.0,
        "3060": 360.0,
        "1650": 192.0,
        "a100": 1555.0,
        "h100": 2039.0,
    }
    lowered = gpu.name.lower()
    for model_name, bandwidth in known.items():
        if model_name in lowered:
            return bandwidth
    return {"nvidia": 400.0, "amd": 350.0, "apple": 200.0, "intel": 200.0}.get(
        gpu.vendor, 200.0
    )


def _throughput(
    model: ModelSpec, memory: MemoryEstimate, n_gpu_layers: int, profile: HardwareProfile
) -> float:
    gpu_frac = n_gpu_layers / model.n_layers
    cpu_frac = 1.0 - gpu_frac
    cpu_bw = 40.0
    if gpu_frac == 0 or not profile.gpus:
        effective_bw = cpu_bw
    else:
        gpu_bw = sum(_gpu_bandwidth(gpu) for gpu in profile.gpus) / len(profile.gpus)
        effective_bw = 1.0 / (gpu_frac / gpu_bw + cpu_frac / cpu_bw)
    return 0.75 * effective_bw * 1e9 / memory.weight_bytes


def _backend(profile: HardwareProfile, model: ModelSpec, n_gpu_layers: int) -> str | None:
    available = profile.available_backends
    nvidia = any(gpu.vendor == "nvidia" for gpu in profile.gpus)
    full_gpu = n_gpu_layers >= model.n_layers
    non_gguf = "hf" in model.sources
    if profile.os == "linux" and nvidia and full_gpu and non_gguf and available.get("vllm"):
        return "vllm"
    if profile.unified_memory and available.get("mlx"):
        return "mlx"
    if available.get("llamacpp"):
        return "llamacpp"
    if available.get("ollama"):
        return "ollama"
    return None


def _launch(
    backend: str,
    model: ModelSpec,
    quant: str,
    context: int,
    port: int,
    n_gpu_layers: int,
    tensor_parallel_size: int = 1,
) -> LaunchSpec:
    source = model.sources.get("ollama", model.id)
    if backend == "ollama":
        argv = ["ollama", "serve"]
    elif backend == "vllm":
        argv = [
            "vllm",
            "serve",
            source,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(context),
        ]
        if tensor_parallel_size > 1:
            argv.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    elif backend == "mlx":
        argv = ["python", "-m", "mlx_lm.server", "--model", source, "--port", str(port)]
    else:
        argv = [
            "llama-server",
            "-m",
            source,
            "-c",
            str(context),
            "--port",
            str(port),
            "-ngl",
            str(n_gpu_layers),
        ]
    return LaunchSpec(argv, {}, f"http://127.0.0.1:{port}/health")


@dataclass(frozen=True)
class _Candidate:
    model: ModelSpec
    quant: str
    context: int
    memory: MemoryEstimate
    n_gpu_layers: int
    decode_tps: float
    backend: str | None
    score: float


def _bench_value(
    cache: Mapping[str, float] | None, model: ModelSpec, quant: str, backend: str
) -> float | None:
    if cache is None:
        return None
    keys = (
        f"{model.id}|{quant}|{backend}",
        f"{model.id}:{quant}:{backend}",
        f"{model.id},{quant},{backend}",
    )
    for key in keys:
        value = cache.get(key)
        if value is not None:
            return float(value)
    return None


def _candidate_for(
    model: ModelSpec,
    profile: HardwareProfile,
    policy: Policy,
    cache: Mapping[str, float] | None,
) -> list[_Candidate]:
    contexts: list[int] = []
    initial = min(model.max_context, policy.max_context or 8192)
    for context in (initial, 4096, 2048):
        if context <= initial and context not in contexts:
            contexts.append(context)
    candidates: list[_Candidate] = []
    for quant in BPW:
        for context in contexts:
            memory = estimate_memory(model, quant, context, profile=profile)
            n_gpu_layers = solve_gpu_layers(memory, model.n_layers)
            if not profile.gpus or profile.tier == Tier.T0_CPU:
                n_gpu_layers = 0
            gpu_bytes = (
                memory.kv_cache_bytes
                + memory.compute_overhead
                + memory.per_layer_bytes * n_gpu_layers
            )
            cpu_bytes = memory.weight_bytes - memory.per_layer_bytes * n_gpu_layers
            if gpu_bytes > memory.vram_budget + 1 or cpu_bytes > memory.ram_budget + 1:
                continue
            backend = _backend(profile, model, n_gpu_layers)
            if backend is None:
                continue
            memory = MemoryEstimate(
                **{
                    **asdict(memory),
                    "cpu_bytes": cpu_bytes,
                    "gpu_bytes": gpu_bytes,
                    "n_gpu_layers": n_gpu_layers,
                }
            )
            bench = _bench_value(cache, model, quant, backend)
            decode_tps = bench if bench is not None else _throughput(
                model, memory, n_gpu_layers, profile
            )
            if decode_tps < policy.min_decode_tps:
                continue
            weights = {"quality": (1.0, 0.25), "speed": (0.4, 1.0), "balanced": (1.0, 0.6)}
            weight_quality, weight_speed = weights.get(policy.prefer, weights["balanced"])
            quality_adj = model.quality - QUANT_PENALTY[quant]
            normalized_tps = min(decode_tps, 60) / 60 * 100
            score = quality_adj * weight_quality + normalized_tps * weight_speed
            candidates.append(
                _Candidate(
                    model, quant, context, memory, n_gpu_layers, decode_tps, backend, score
                )
            )
            break
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _smallest_fallback(
    models: Sequence[ModelSpec], role: str, profile: HardwareProfile, policy: Policy
) -> _Candidate | None:
    supported = [model for model in models if role in model.roles]
    if not supported:
        return None
    model = min(supported, key=lambda item: item.params)
    quant = "q4_k_m"
    context = min(model.max_context, policy.max_context or 2048)
    memory = estimate_memory(model, quant, context, profile=profile)
    n_gpu_layers = solve_gpu_layers(memory, model.n_layers) if profile.gpus else 0
    backend = _backend(profile, model, n_gpu_layers) or "unavailable"
    memory = MemoryEstimate(
        **{
            **asdict(memory),
            "cpu_bytes": memory.weight_bytes - memory.per_layer_bytes * n_gpu_layers,
            "gpu_bytes": memory.kv_cache_bytes
            + memory.compute_overhead
            + memory.per_layer_bytes * n_gpu_layers,
            "n_gpu_layers": n_gpu_layers,
        }
    )
    return _Candidate(
        model,
        quant,
        context,
        memory,
        n_gpu_layers,
        _throughput(model, memory, n_gpu_layers, profile),
        backend,
        model.quality - QUANT_PENALTY[quant],
    )


def build_plan(
    profile: HardwareProfile,
    catalog: Sequence[ModelSpec],
    policy: Policy | None = None,
    bench_cache: Mapping[str, float] | None = None,
) -> Plan:
    selected_policy = policy or Policy()
    roles = list(dict.fromkeys(selected_policy.roles))
    warnings = list(profile.warnings)
    install_hints: list[str] = []
    all_candidates: dict[str, list[_Candidate]] = {}
    for role in roles:
        role_models = [model for model in catalog if role in model.roles]
        combined = [model for model in role_models if all(item in model.roles for item in roles if item != "embed")]
        models = combined if role in {"chat", "code"} and combined else role_models
        candidates: list[_Candidate] = []
        for model in models:
            candidates.extend(_candidate_for(model, profile, selected_policy, bench_cache))
        if profile.tier == Tier.T1_LOW and role in {"chat", "code"}:
            partial = [candidate for candidate in candidates if candidate.n_gpu_layers < candidate.model.n_layers]
            if partial:
                candidates = partial
        all_candidates[role] = sorted(candidates, key=lambda item: item.score, reverse=True)

    resident_roles: list[list[str]]
    if profile.tier in {Tier.T0_CPU, Tier.T1_LOW}:
        main_roles = [role for role in roles if role in {"chat", "code"}]
        resident_roles = [main_roles[:1]] if main_roles else []
        if len(main_roles) > 1:
            resident_roles[0] = main_roles
        resident_roles.extend([[role] for role in roles if role == "embed"])
    elif profile.tier in {Tier.T2_MID, Tier.T3_HIGH}:
        main_roles = [role for role in roles if role in {"chat", "code"}]
        resident_roles = [main_roles] if main_roles else []
        resident_roles.extend([[role] for role in roles if role == "embed"])
    else:
        resident_roles = [[role] for role in roles]

    services: list[PlannedService] = []
    swap_group: list[str] = []
    role_to_service: dict[str, str] = {}
    total_download = 0
    for group in resident_roles:
        if not group:
            continue
        pool = [candidate for candidate in all_candidates.get(group[0], []) if all(
            candidate.model.id in {item.model.id for item in all_candidates.get(role, [])}
            for role in group
        )]
        candidate = pool[0] if pool else None
        if candidate is None:
            candidate = _smallest_fallback(catalog, group[0], profile, selected_policy)
        if candidate is None:
            warnings.append(f"No catalog model supports role(s): {', '.join(group)}")
            continue
        resident = profile.tier not in {Tier.T0_CPU, Tier.T1_LOW} or not services
        name = group[0]
        backend = candidate.backend or "unavailable"
        if candidate.backend is None:
            install_hints.append("Install llama-server, ollama, vllm, or mlx_lm.")
        gpu_indices = [gpu.index for gpu in profile.gpus]
        if profile.tier == Tier.T5_SERVER and len(gpu_indices) > 1:
            gpu_indices = [gpu.index for gpu in profile.gpus]
        elif gpu_indices:
            gpu_indices = [gpu_indices[0]]
        launch = _launch(
            backend,
            candidate.model,
            candidate.quant,
            candidate.context,
            18010 + len(services),
            candidate.n_gpu_layers,
            len(profile.gpus) if profile.tier == Tier.T5_SERVER else 1,
        )
        service = PlannedService(
            name=name,
            roles=group,
            model_id=candidate.model.id,
            quant=candidate.quant,
            backend=backend,
            context=candidate.context,
            port=18010 + len(services),
            gpu_indices=gpu_indices,
            n_gpu_layers=None if backend in {"vllm", "mlx"} else candidate.n_gpu_layers,
            resident=resident,
            memory=candidate.memory,
            decode_tps=candidate.decode_tps,
            estimated=_bench_value(bench_cache, candidate.model, candidate.quant, backend) is None,
            launch=launch,
        )
        services.append(service)
        total_download += int(candidate.memory.disk_needed)
        for role in group:
            role_to_service[role] = name
        if not resident:
            swap_group.append(name)

    if not services:
        warnings.append("No runnable services were found.")
    if total_download > selected_policy.allow_download_gb * GIB:
        warnings.append("Planned downloads exceed policy.allow_download_gb.")
    routing = RoutingRules("rules", role_to_service, {"nmesh-auto": role_to_service.get("chat", "")})
    return Plan(
        created_at=datetime.now(timezone.utc).isoformat(),
        nmesh_version=__version__,
        profile=profile,
        tier=profile.tier,
        policy=selected_policy,
        services=services,
        swap_group=swap_group,
        routing=routing,
        warnings=warnings,
        install_hints=list(dict.fromkeys(install_hints)),
        total_download_bytes=total_download,
        runnable=bool(services) and not install_hints,
    )


def _gpu_from_dict(data: object) -> GPUInfo:
    if not isinstance(data, dict):
        raise TypeError("Invalid GPU data")
    capability = data.get("compute_capability")
    parsed_capability = tuple(capability) if isinstance(capability, list) else capability
    return GPUInfo(
        int(data["index"]),
        str(data["name"]),
        str(data["vendor"]),  # type: ignore[arg-type]
        int(data["total_vram_bytes"]),
        int(data["free_vram_bytes"]),
        parsed_capability,
        bool(data["driving_display"]),
    )


def _plan_from_dict(data: dict[str, object]) -> Plan:
    profile_data = data["profile"]
    if not isinstance(profile_data, dict):
        raise TypeError("Invalid profile data")
    profile = HardwareProfile(
        str(profile_data["os"]),  # type: ignore[arg-type]
        str(profile_data["cpu_name"]),
        int(profile_data["physical_cores"]),
        int(profile_data["logical_cores"]),
        int(profile_data["total_ram_bytes"]),
        int(profile_data["available_ram_bytes"]),
        int(profile_data["free_disk_bytes"]),
        bool(profile_data["unified_memory"]),
        [_gpu_from_dict(item) for item in profile_data["gpus"]],  # type: ignore[index]
        {str(key): value if isinstance(value, str) else None for key, value in profile_data["available_backends"].items()},  # type: ignore[union-attr]
        Tier(str(data["tier"])),
        [str(item) for item in profile_data.get("warnings", [])],  # type: ignore[union-attr]
    )
    policy_data = data["policy"]
    if not isinstance(policy_data, dict):
        raise TypeError("Invalid policy data")
    policy = Policy(
        [str(role) for role in policy_data["roles"]],
        float(policy_data["min_decode_tps"]),
        int(policy_data["max_context"]) if policy_data["max_context"] is not None else None,
        str(policy_data["prefer"]),
        float(policy_data["allow_download_gb"]),
    )
    services: list[PlannedService] = []
    for item in data["services"]:  # type: ignore[union-attr]
        service_data = item
        if not isinstance(service_data, dict):
            raise TypeError("Invalid service data")
        memory_data = service_data["memory"]
        launch_data = service_data["launch"]
        if not isinstance(memory_data, dict) or not isinstance(launch_data, dict):
            raise TypeError("Invalid service nested data")
        memory = MemoryEstimate(**{str(key): float(value) for key, value in memory_data.items()})
        launch = LaunchSpec(
            [str(arg) for arg in launch_data["argv"]],
            {str(key): str(value) for key, value in launch_data["env"].items()},
            str(launch_data["health_url"]) if launch_data["health_url"] else None,
        )
        services.append(
            PlannedService(
                str(service_data["name"]),
                [str(role) for role in service_data["roles"]],
                str(service_data["model_id"]),
                str(service_data["quant"]),
                str(service_data["backend"]),
                int(service_data["context"]),
                int(service_data["port"]),
                [int(index) for index in service_data["gpu_indices"]],
                int(service_data["n_gpu_layers"]) if service_data["n_gpu_layers"] is not None else None,
                bool(service_data["resident"]),
                memory,
                float(service_data["decode_tps"]),
                bool(service_data["estimated"]),
                launch,
            )
        )
    routing_data = data["routing"]
    if not isinstance(routing_data, dict):
        raise TypeError("Invalid routing data")
    routing = RoutingRules(
        str(routing_data["mode"]),
        {str(key): str(value) for key, value in routing_data["role_to_service"].items()},
        {str(key): str(value) for key, value in routing_data["aliases"].items()},
    )
    return Plan(
        str(data["created_at"]),
        str(data["nmesh_version"]),
        profile,
        Tier(str(data["tier"])),
        policy,
        services,
        [str(item) for item in data["swap_group"]],
        routing,
        [str(item) for item in data["warnings"]],
        [str(item) for item in data["install_hints"]],
        int(data["total_download_bytes"]),
        bool(data.get("runnable", True)),
    )


def save_plan(plan: Plan, path: Path | None = None) -> Path:
    target = path or (Path.home() / ".nmesh" / "plan.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(plan), indent=2, default=str), encoding="utf-8")
    return target


def load_plan(path: Path | None = None) -> Plan | None:
    target = path or (Path.home() / ".nmesh" / "plan.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _plan_from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


plan = build_plan
