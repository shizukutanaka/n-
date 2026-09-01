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
PLAN_PATH = Path.home() / ".nmesh" / "plan.json"
INSTALL_HINTS = {
    "ollama": "Install Ollama: https://ollama.com/download",
    "llamacpp": "Install llama.cpp: winget install llama.cpp / brew install llama.cpp / build from source",
    "vllm": "Install vLLM: pip install vllm",
    "mlx": "Install MLX-LM: pip install mlx-lm",
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


@dataclass
class Policy:
    roles: list[str] = field(default_factory=lambda: ["chat", "code", "embed"])
    min_decode_tps: float = 8.0
    max_context: int | None = None
    prefer: str = "balanced"
    allow_download_gb: float = 60.0
    kv_quant: str = "f16"
    budget_source: str = "total"


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
        name = "llamacpp"
    return name, bool(profile.available_backends.get(name))


def _source_for(backend: str, model: ModelSpec, quant: str) -> str:
    if backend in {"vllm", "mlx"}:
        return model.sources["hf"]
    if backend == "llamacpp":
        return str(Path.home() / ".nmesh" / "models" / f"{model.id}-{quant}.gguf")
    return model.sources["ollama"]


def _has_source(backend: str, model: ModelSpec) -> bool:
    return {
        "vllm": "hf",
        "mlx": "hf",
        "llamacpp": "hf_gguf",
        "ollama": "ollama",
    }.get(backend, "") in model.sources


def _launch(backend: str, model: ModelSpec, quant: str, context: int, port: int,
            layers: int, tensor_parallel: int) -> LaunchSpec:
    ref = _source_for(backend, model, quant)
    if backend == "ollama":
        return LaunchSpec(["ollama", "serve"], {}, "http://127.0.0.1:11434/api/tags", True)
    if backend == "vllm":
        argv = ["vllm", "serve", ref, "--host", "127.0.0.1", "--port", str(port),
                "--max-model-len", str(context)]
        if tensor_parallel > 1:
            argv += ["--tensor-parallel-size", str(tensor_parallel)]
    elif backend == "mlx":
        argv = ["python", "-m", "mlx_lm.server", "--model", ref, "--port", str(port)]
    else:
        argv = ["llama-server", "-m", ref, "-c", str(context), "--port", str(port), "-ngl", str(layers)]
        if tensor_parallel > 1:
            argv += ["--tensor-split", ",".join(["1"] * tensor_parallel)]
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
            on_gpu = layers > 0
            gpu_bytes = base.per_layer_bytes * layers + (
                base.kv_cache_bytes + base.compute_overhead if on_gpu else 0.0
            )
            cpu_bytes = base.per_layer_bytes * (model.n_layers - layers) + (
                0.0 if on_gpu else base.kv_cache_bytes + base.compute_overhead
            )
            if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                continue
            backend, installed = _backend(profile, model, layers)
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
            candidates.append(_Candidate(model, quant, context, memory, layers, tps,
                                         backend, installed, score, bench is None))
            break
    return sorted(candidates, key=lambda item: item.score, reverse=True)


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
                 budget_source: str) -> None:
    if not candidate.installed:
        hints.append(INSTALL_HINTS[candidate.backend])
    indices = [gpu.index for gpu in profile.gpus]
    tensor_parallel = 1
    if indices:
        fit = [
            gpu for gpu in profile.gpus
            if candidate.memory.gpu_bytes <= _gpu_budget(gpu, budget_source) + 1
        ]
        if fit:
            indices = [fit[0].index]
        else:
            tensor_parallel = len(indices)
    name = group[0]
    port = 18010 + len(services)
    launch = _launch(candidate.backend, candidate.model, candidate.quant, candidate.context,
                     port, candidate.n_gpu_layers, tensor_parallel)
    if candidate.backend == "llamacpp" and "hf_gguf" in candidate.model.sources:
        launch = replace(launch, env={"NMESH_HF_REPO": candidate.model.sources["hf_gguf"]})
    service = PlannedService(
        name, group, candidate.model.id, _source_for(candidate.backend, candidate.model, candidate.quant),
        candidate.model.sources.get("hf_gguf") or candidate.model.sources.get("hf"),
        candidate.quant, candidate.backend, candidate.context,
        11434 if candidate.backend == "ollama" else port, indices,
        None if candidate.backend in {"vllm", "mlx"} else candidate.n_gpu_layers,
        profile.tier not in {Tier.T0_CPU, Tier.T1_LOW} or not services,
        candidate.memory, candidate.decode_tps, candidate.estimated, launch,
    )
    services.append(service)
    for role in group:
        role_to_service[role] = name
    if not service.resident:
        swap_group.append(name)


def _replace_gpu(service: PlannedService, indices: list[int]) -> PlannedService:
    return PlannedService(service.name, service.roles, service.model_id, service.model_ref,
                          service.download_repo,
                          service.quant, service.backend, service.context, service.port, indices,
                          service.n_gpu_layers, service.resident, service.memory, service.decode_tps,
                          service.estimated, service.launch)


def build_plan(profile: HardwareProfile, catalog: Sequence[ModelSpec],
               policy: Policy | None = None,
               bench_cache: Mapping[object, float] | None = None) -> Plan:
    selected = policy or Policy()
    roles = list(dict.fromkeys(selected.roles))
    warnings = list(profile.warnings)
    hints: list[str] = []
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
                        role_to_service, hints, selected.budget_source,
                    )
                    total_download += int(role_candidate[0].memory.disk_needed)
                else:
                    warnings.append(f"役割 {role} を満たす構成が見つかりません")
            continue
        if candidate is None:
            warnings.append(f"役割 {group[0]} を満たす構成が見つかりません")
            continue
        _add_service(
            group, candidate, profile, services, swap_group, role_to_service, hints,
            selected.budget_source,
        )
        total_download += int(candidate.memory.disk_needed)
    if len(profile.gpus) > 1:
        for index, service in enumerate(services):
            if service.memory.gpu_bytes <= _gpu_budget(
                profile.gpus[0], selected.budget_source
            ) + 1:
                services[index] = _replace_gpu(service, [profile.gpus[index % len(profile.gpus)].index])
    if total_download > selected.allow_download_gb * GIB:
        warnings.append("Planned downloads exceed policy.allow_download_gb.")
    covered = set(role_to_service)
    runnable = bool(services) and covered >= set(roles) and not hints
    return Plan(
        datetime.now(timezone.utc).isoformat(), __version__, profile, profile.tier, selected,
        services, swap_group,
        RoutingRules("rules", role_to_service, {"nmesh-auto": role_to_service.get("chat", "")}),
        warnings, list(dict.fromkeys(hints)), total_download, runnable,
    )


def _gpu_from_dict(data: object) -> GPUInfo:
    if not isinstance(data, dict):
        raise TypeError("Invalid GPU data")
    cap = data.get("compute_capability")
    return GPUInfo(int(data["index"]), str(data["name"]), str(data["vendor"]),
                   int(data["total_vram_bytes"]), int(data["free_vram_bytes"]),
                   tuple(cap) if isinstance(cap, list) else cap, bool(data["driving_display"]))


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
    )
    pol = data["policy"]
    if not isinstance(pol, dict):
        raise TypeError("Invalid policy")
    policy = Policy([str(x) for x in pol["roles"]], float(pol["min_decode_tps"]),
                    int(pol["max_context"]) if pol["max_context"] is not None else None,
                    str(pol["prefer"]), float(pol["allow_download_gb"]),
                    str(pol.get("kv_quant", "f16")), str(pol.get("budget_source", "total")))
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
        temporary.write_text(json.dumps(asdict(plan), indent=2, default=str), encoding="utf-8")
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
