from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from nmesh import __version__
from nmesh.artifact import service_fingerprint
from nmesh.artifacts import artifact_key
from nmesh.bench.cache import BENCH_HARNESS_VERSION, BenchRecord, benchmark_key
from nmesh.catalog import ModelSpec
from nmesh.eval.cache import EvalSummary
from nmesh.eval.generated import EXTENDED_TASKS
from nmesh.eval.stats import (
    fisher_two_sided,
    mcnemar_two_sided,
    min_discordant_for_significance,
    min_discordant_imbalance,
    min_resolvable_difference,
)
from nmesh.i18n import t
from nmesh.orchestrate.measure import RoleIdentity
from nmesh.paths import nmesh_home
from nmesh.probe import GPUInfo, HardwareProfile, Tier
from nmesh.spec import (
    KINDS,
    SpecConfig,
    best_for,
    decide,
    engine_identity,
    load_cache,
)

BPW = {
    "f16": 16.0, "q8_0": 8.5, "q6_k": 6.6, "q5_k_m": 5.7,
    "q4_k_m": 4.85, "q4_0": 4.55, "q3_k_m": 3.9, "q2_k": 3.35,
}
EMB_BPW_FLOOR = 5.5
QUANT_PENALTY = {
    "f16": 0.0, "q8_0": 0.5, "q6_k": 1.0, "q5_k_m": 2.0,
    "q4_k_m": 3.5, "q4_0": 5.0, "q3_k_m": 9.0, "q2_k": 16.0,
}
SPEED_REFERENCE_TPS = 30.0
GIB = 1024**3
PLAN_PATH = nmesh_home() / "plan.json"
INSTALL_HINTS = {
    "ollama": "install.ollama",
    "llamacpp": "install.llamacpp",
    "vllm": "install.vllm",
    "mlx": "install.mlx",
}


def _effective_prior(model: ModelSpec, quant: str) -> float | None:
    return None if model.quality is None else model.quality - QUANT_PENALTY[quant]


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
    model_ids: tuple[str, ...] = ()
    eval_evidence: bool = True
    spec: str = "none"
    spec_draft: str = ""
    spec_n_max: int = 3
    ignore_spec_evidence: bool = False

    def __post_init__(self) -> None:
        if self.spec not in KINDS:
            raise ValueError(f"Unknown speculation kind: {self.spec}")
        if self.spec == "draft" and not self.spec_draft.strip():
            raise ValueError("spec_draft is required for draft speculation")
        if self.spec_n_max < 1:
            raise ValueError("spec_n_max must be positive")


@dataclass(frozen=True)
class EvidenceTest:
    p_value: float
    compared: int
    paired: bool
    better_only: int
    worse_only: int


def _evidence_p_value(
    better: EvalSummary, worse: EvalSummary,
) -> EvidenceTest:
    shared = set(better.task_results) & set(worse.task_results)
    if better.task_results and worse.task_results and shared:
        b = sum(
            better.task_results[task_id] and not worse.task_results[task_id]
            for task_id in shared
        )
        c = sum(
            worse.task_results[task_id] and not better.task_results[task_id]
            for task_id in shared
        )
        return EvidenceTest(
            mcnemar_two_sided(b, c), len(shared), True, b, c,
        )
    return EvidenceTest(
        fisher_two_sided(
            better.passed,
            better.n_tasks - better.passed,
            worse.passed,
            worse.n_tasks - worse.passed,
        ),
        min(better.n_tasks, worse.n_tasks),
        False,
        0,
        0,
    )


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
    kv_quant: str = "f16"
    spec: str = "none"
    spec_draft: str = ""


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
    missing_backends: list[str] = field(default_factory=list)


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
    budget_source: str = "total", weight_bytes: float | None = None,
) -> MemoryEstimate:
    if quant not in BPW:
        raise ValueError(f"Unsupported quantization: {quant}")
    if kv_quant not in {"f16", "q8_0"}:
        raise ValueError(f"Unsupported KV quantization: {kv_quant}")
    if weight_bytes is None:
        weight_bytes = structural_weight_bytes(model, quant)
    per_layer_bytes = weight_bytes / model.n_layers
    kv_elem_bytes = {"f16": 2, "q8_0": 1}[kv_quant]
    kv_bytes_per_tok = 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_elem_bytes
    kv_cache_bytes = kv_bytes_per_tok * context * parallel_slots
    compute_overhead = 0.06 * weight_bytes + 320 * 1024**2
    vram_budget, ram_budget = _profile_budgets(profile, budget_source)
    return MemoryEstimate(
        weight_bytes, per_layer_bytes, kv_bytes_per_tok, kv_cache_bytes, compute_overhead,
        weight_bytes + kv_cache_bytes + compute_overhead, vram_budget, ram_budget,
        weight_bytes * 1.05, parallel_slots=parallel_slots,
    )


def structural_weight_bytes(model: ModelSpec, quant: str) -> float:
    bpw = BPW[quant]
    if model.vocab_size <= 0:
        return model.params * bpw / 8
    emb_one = model.vocab_size * model.hidden_size
    n_emb = 1 if model.head_layout == "shared" else 2
    elems = model.params + (
        emb_one if model.head_layout == "duplicate" else 0
    )
    emb_elems = emb_one * n_emb
    body = max(elems - emb_elems, 0.0)
    return body * bpw / 8 + emb_elems * max(EMB_BPW_FLOOR, bpw) / 8


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
                profile: HardwareProfile, weight_bytes: float | None = None) -> float:
    """Estimate decode throughput from effective memory bandwidth.

    On this machine, q4_k_m predicted 32.1 versus a 30.1 tok/s median
    (about 7% high), q4_0 predicted 34.3 versus 35.0 (about 2% low), and
    f16 predicted 30.4 versus 40.2 (about 24% pessimistic because f16 needs
    no dequantization). The 40 GB/s CPU value is a memory-bandwidth stand-in
    validated to about 7% for quantized CPU inference here. GPU bandwidth
    table values are unvalidated; these measurements do not promise
    generalization. Two same-model artifacts differing by 272,268,832 bytes
    (21%) decoded at 59.48 versus 59.43 tok/s median-of-5 on this machine,
    within 0.1%; the larger file's separate output.weight duplicates the tied
    embedding and adds no per-token work, so bytes on disk (and weight_bytes
    derived from a params-by-bpw label) can overstate bytes actually read per
    token for an artifact with an untied duplicate head.
    """
    # Throughput uses bytes read per token, not artifact bytes.
    gpu_frac = layers / model.n_layers
    if gpu_frac == 0 or not profile.gpus:
        effective = 40.0
    else:
        gpu_bw = sum(_gpu_bandwidth(gpu) for gpu in profile.gpus) / len(profile.gpus)
        effective = 1.0 / (gpu_frac / gpu_bw + (1.0 - gpu_frac) / 40.0)
    basis = memory.weight_bytes if weight_bytes is None else weight_bytes
    return 0.75 * effective * 1e9 / basis


def _backend(profile: HardwareProfile, model: ModelSpec, layers: int) -> tuple[str, bool]:
    nvidia = any(gpu.vendor == "nvidia" for gpu in profile.gpus)
    full_gpu = layers >= model.n_layers
    candidates: list[str] = []
    if profile.os == "linux" and nvidia and full_gpu and "hf" in model.sources:
        candidates.append("vllm")
    if profile.unified_memory and "hf" in model.sources:
        candidates.append("mlx")
    if "hf_gguf" in model.sources:
        candidates.append("llamacpp")
    if "ollama" in model.sources:
        candidates.append("ollama")
    if profile.os == "linux" and "hf" in model.sources:
        candidates.append("vllm")
    if profile.unified_memory and "hf" in model.sources:
        candidates.append("mlx")
    for name in candidates:
        if profile.available_backends.get(name):
            return name, True
    return (candidates[0], False) if candidates else ("", False)


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
    kv_quant: str = "f16",
    warnings: list[str] | None = None,
    gpu_devices: tuple[str, ...] | None = None,
    language: str = "en",
    roles: Sequence[str] = (),
    spec: str = "none",
    spec_draft: str = "",
    spec_n_max: int = 3,
    *,
    binary: str | None = None,
) -> LaunchSpec:
    embed_only = list(roles) == ["embed"]
    ref = _source_for(backend, model, quant)
    if backend == "ollama":
        return LaunchSpec(
            [binary or "ollama", "serve"],
            {},
            "http://127.0.0.1:11434/api/tags",
            True,
        )
    if backend == "vllm":
        argv = [
            binary or "vllm", "serve", ref, "--host", "127.0.0.1",
            "--port", str(port), "--max-model-len", str(context),
            "--max-num-seqs", str(slots),
        ]
        if tensor_parallel > 1:
            argv += ["--tensor-parallel-size", str(tensor_parallel)]
        if gpu_fraction is not None:
            argv += ["--gpu-memory-utilization", f"{gpu_fraction:.3f}"]
        if embed_only and warnings is not None:
            warnings.append(
                t("warn.embeddings_backend_unverified", language, model=model.id)
            )
    elif backend == "mlx":
        argv = ["python", "-m", "mlx_lm.server", "--model", ref, "--port", str(port)]
        if embed_only and warnings is not None:
            warnings.append(
                t("warn.embeddings_backend_unsupported", language, model=model.id)
            )
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
            binary or "llama-server", "-m", ref, "-c",
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
        if (
            backend == "llamacpp"
            and kv_quant != "f16"
            and _honors_kv_quant(backend, backend_flags)
        ):
            argv += ["--cache-type-k", kv_quant, "--cache-type-v", kv_quant]
        if backend == "llamacpp" and spec != "none" and _honors_spec(
            backend, backend_flags, spec
        ):
            if spec == "ngram":
                argv += ["--spec-type", "ngram-simple"]
            elif spec == "draft":
                argv += [
                    "--spec-type", "draft-simple",
                    "--spec-draft-model", spec_draft,
                    "--spec-draft-n-max", str(spec_n_max),
                ]
        elif spec != "none" and warnings is not None:
            warnings.append(
                t("warn.spec_unsupported", language, service=model.id)
            )
        if backend == "llamacpp" and embed_only:
            if not known or "--embeddings" in backend_flags:
                embedding_flag = "--embeddings"
            elif "--embedding" in backend_flags:
                embedding_flag = "--embedding"
            else:
                embedding_flag = None
            if embedding_flag is not None:
                argv.append(embedding_flag)
            elif warnings is not None:
                warnings.append(
                    t("warn.embeddings_unsupported", language, model=model.id)
                )
            pooling_supported = not known or "--pooling" in backend_flags
            if model.pooling and pooling_supported:
                argv.extend(["--pooling", model.pooling])
            elif warnings is not None:
                warnings.append(
                    t("warn.embeddings_pooling_unknown", language, model=model.id)
                )
            if context > 512:
                if not known:
                    logical_batch_flag = "-b"
                    physical_batch_flag = "-ub"
                else:
                    logical_batch_flag = (
                        "-b" if "-b" in backend_flags else "--batch-size"
                    )
                    physical_batch_flag = (
                        "-ub" if "-ub" in backend_flags else "--ubatch-size"
                    )
                batch_supported = not known or (
                    any(
                        flag in backend_flags
                        for flag in ("-b", "--batch-size")
                    )
                    and any(
                        flag in backend_flags
                        for flag in ("-ub", "--ubatch-size")
                    )
                )
                if batch_supported:
                    argv.extend(
                        [
                            logical_batch_flag,
                            str(context),
                            physical_batch_flag,
                            str(context),
                        ]
                    )
                elif warnings is not None:
                    warnings.append(
                        t(
                            "warn.embeddings_batch_limit",
                            language,
                            model=model.id,
                            context=context,
                        )
                    )
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
    kv_quant: str
    requested_kv_quant: str
    score: float
    estimated: bool


def _bench_value(cache: Mapping[object, float] | None, model: ModelSpec, quant: str,
                 backend: str, gpu_name: str, layers: int,
                 kv_quant: str = "f16", spec: str = "none") -> float | None:
    if cache is None:
        return None
    keys: list[object] = [
        benchmark_key(model.id, quant, backend, gpu_name, layers, kv_quant, spec)
    ]
    if kv_quant == "f16" and spec == "none":
        keys.extend((
            (model.id, quant, backend, gpu_name, layers),
            f"{model.id}:{quant}:{backend}:{gpu_name}:{layers}",
        ))
    for key in keys:
        if key in cache:
            return float(cache[key])
    return None


def _candidate_for(
    model: ModelSpec, profile: HardwareProfile, policy: Policy,
    cache: Mapping[object, float] | None,
    artifact_cache: Mapping[str, int] | None = None,
    reserved_vram_bytes: float = 0.0,
    reserved_ram_bytes: float = 0.0,
    excluded: list[dict[str, str]] | None = None,
    *,
    allow_unmeasured: bool = False,
    bench_records: Mapping[str, BenchRecord] | None = None,
    unconfirmed: list[dict[str, str]] | None = None,
) -> list[_Candidate]:
    """Build candidates using the intentionally unchanged score.

    On the bundled simulated profiles, the speed term is saturated for 82 of
    83 candidates on t3-rtx4090-24gb, 67 of 72 on t2-rtx3060-12gb, and 68 of
    89 on t4-rtx6000ada-48gb. On t3 with ``prefer="speed"``, qwen2.5-32b-
    instruct q4_k_m at 38.49 tok/s scored 140.25 above phi-4-14b q5_k_m at
    73.68 tok/s and score 138.00. With ``prefer="quality"`` and speed
    references 30/60/120/240, the choices were 32B q4_k_m at 38.5, phi-4-14b
    q6_k at 63.6, 32B q4_k_m at 38.5, and 32B q5_k_m at 9.0 tok/s,
    respectively. These are the planner's own estimates on bundled simulated
    profiles, not wall-clock measurements. Removing the clip would select the
    0.5B q2_k candidate at 3654.6 tok/s, so a scale-free speed term requires a
    quality floor; the catalog quality prior is unvalidated.
    """
    if model.quality is None and not allow_unmeasured:
        return []
    initial = min(model.max_context, policy.max_context or 8192)
    contexts = list(dict.fromkeys(context for context in (initial, 4096, 2048) if context <= initial))
    candidates: list[_Candidate] = []
    for quant, bpw in BPW.items():
        for context in contexts:
            def estimate_candidate(
                kv_quant: str, *, _quant: str = quant, _context: int = context,
            ) -> MemoryEstimate:
                measured_bytes = None
                if artifact_cache is not None:
                    repo_id = model.sources.get("hf_gguf", "")
                    measured_bytes = artifact_cache.get(
                        artifact_key(repo_id, _quant)
                    )
                estimate = estimate_memory(
                    model, _quant, _context, profile=profile, kv_quant=kv_quant,
                    budget_source=policy.budget_source,
                    weight_bytes=(
                        float(measured_bytes)
                        if measured_bytes is not None else None
                    ),
                )
                if model.roles == ["embed"]:
                    activation = min(
                        0.02 * estimate.weight_bytes * math.ceil(_context / 512),
                        512 * 1024**2,
                    )
                    overhead = estimate.compute_overhead + activation
                    estimate = replace(
                        estimate,
                        kv_bytes_per_tok=0.0,
                        kv_cache_bytes=0.0,
                        compute_overhead=overhead,
                        total_bytes=estimate.weight_bytes + overhead,
                    )
                return replace(
                    estimate,
                    vram_budget=max(estimate.vram_budget - reserved_vram_bytes, 0.0),
                    ram_budget=max(estimate.ram_budget - reserved_ram_bytes, 0.0),
                )

            accounted_kv_quant = policy.kv_quant
            base = estimate_candidate(accounted_kv_quant)
            layers = solve_gpu_layers(base, model.n_layers) if profile.gpus else 0
            if profile.tier == Tier.T0_CPU:
                layers = 0
            gpu_bytes, cpu_bytes = _split_memory(base, model.n_layers, layers)
            if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                continue
            backend, installed = _backend(profile, model, layers)
            backend_flags = profile.backend_flags.get(backend)
            if (
                policy.kv_quant != "f16"
                and not _honors_kv_quant(backend, backend_flags)
            ):
                accounted_kv_quant = "f16"
                base = estimate_candidate(accounted_kv_quant)
                layers = solve_gpu_layers(base, model.n_layers) if profile.gpus else 0
                if profile.tier == Tier.T0_CPU:
                    layers = 0
                gpu_bytes, cpu_bytes = _split_memory(base, model.n_layers, layers)
                if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                    continue
            if backend == "llamacpp" and profile.backend_gpu_devices.get("llamacpp") == ():
                layers = 0
                gpu_bytes, cpu_bytes = _split_memory(base, model.n_layers, layers)
                if gpu_bytes > base.vram_budget + 1 or cpu_bytes > base.ram_budget + 1:
                    continue
            if not _has_source(backend, model):
                continue
            gpu_name = profile.gpus[0].name if profile.gpus else "cpu"
            bench = _bench_value(
                cache, model, quant, backend, gpu_name, layers, accounted_kv_quant,
                policy.spec,
            )
            bench_record = (
                bench_records.get(
                    benchmark_key(
                        model.id, quant, backend, gpu_name, layers,
                        accounted_kv_quant, policy.spec,
                    )
                )
                if bench_records is not None else None
            )
            confirmed = (
                bench_records is None
                or (
                    bench_record is not None
                    and bench_record.harness == BENCH_HARNESS_VERSION
                    and bench_record.confirmations >= 2
                )
            )
            memory = MemoryEstimate(**{**asdict(base), "cpu_bytes": cpu_bytes,
                                       "gpu_bytes": gpu_bytes, "n_gpu_layers": layers})
            tps = bench if bench is not None else _throughput(
                model,
                memory,
                layers,
                profile,
                model.params * bpw / 8,
            )
            if tps < policy.min_decode_tps and not confirmed:
                if bench is not None and unconfirmed is not None:
                    unconfirmed.append({
                        "model": model.id,
                        "quant": quant,
                        "tps": f"{bench:.2f}",
                        "threshold": f"{policy.min_decode_tps:.2f}",
                    })
            elif tps < policy.min_decode_tps:
                if bench is not None and excluded is not None:
                    estimate = _throughput(
                        model,
                        memory,
                        layers,
                        profile,
                        model.params * bpw / 8,
                    )
                    if estimate >= policy.min_decode_tps:
                        excluded.append({
                            "model": model.id,
                            "quant": quant,
                            "tps": f"{bench:.2f}",
                            "threshold": f"{policy.min_decode_tps:.2f}",
                        })
                continue
            wq, ws = {"quality": (1.0, 0.1), "speed": (0.5, 1.0),
                      "balanced": (1.0, 0.25)}.get(policy.prefer, (1.0, 0.25))
            speed_score = min(tps, SPEED_REFERENCE_TPS) / SPEED_REFERENCE_TPS * 100 * ws
            prior = _effective_prior(model, quant)
            score = (
                speed_score
                if prior is None
                else prior * wq + speed_score
            )
            if policy.languages:
                covers = set(policy.languages).issubset(model.languages)
                score *= 1.0 if covers else 0.7
            candidates.append(_Candidate(
                model, quant, context, memory, layers, tps, backend, installed,
                accounted_kv_quant, policy.kv_quant, score, bench is None,
            ))
            break
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def _catalog_role(role: str) -> str:
    return "chat" if role == "worker" else role


def split_memory(memory: MemoryEstimate, model_layers: int, layers: int) -> tuple[float, float]:
    on_gpu = layers > 0
    gpu_bytes = memory.per_layer_bytes * layers + (
        memory.kv_cache_bytes + memory.compute_overhead if on_gpu else 0.0
    )
    cpu_bytes = memory.per_layer_bytes * (model_layers - layers) + (
        0.0 if on_gpu else memory.kv_cache_bytes + memory.compute_overhead
    )
    return gpu_bytes, cpu_bytes


_split_memory = split_memory


GPU_LAYER_FLAGS = ("-ngl", "--gpu-layers", "--n-gpu-layers")
KV_CACHE_TYPE_FLAGS = ("--cache-type-k", "--cache-type-v")
SPEC_TYPE_FLAGS = ("--spec-type",)
SPEC_DRAFT_FLAGS = ("--spec-draft-model", "--spec-draft-n-max")


def _supports_gpu_layers(
    flags: frozenset[str] | tuple[str, ...] | None,
) -> bool:
    return flags is None or any(flag in flags for flag in GPU_LAYER_FLAGS)


def _honors_kv_quant(
    backend: str,
    backend_flags: frozenset[str] | tuple[str, ...] | None,
) -> bool:
    if backend != "llamacpp":
        return False
    return backend_flags is None or all(flag in backend_flags for flag in KV_CACHE_TYPE_FLAGS)


def _honors_spec(
    backend: str,
    backend_flags: frozenset[str] | tuple[str, ...] | None,
    kind: str,
) -> bool:
    if backend != "llamacpp":
        return False
    required = SPEC_TYPE_FLAGS + (
        SPEC_DRAFT_FLAGS if kind == "draft" else ()
    )
    return backend_flags is None or all(flag in backend_flags for flag in required)


def _plan_group(group: list[str], pools: dict[str, list[_Candidate]]) -> _Candidate | None:
    if not group:
        return None
    ids = {candidate.model.id for candidate in pools.get(group[0], [])}
    for role in group[1:]:
        ids &= {candidate.model.id for candidate in pools.get(role, [])}
    return next((candidate for candidate in pools[group[0]] if candidate.model.id in ids), None)


def _speed_saturation_warning(
    role: str,
    chosen: _Candidate,
    pool: list[_Candidate],
    policy: Policy,
) -> str | None:
    if policy.prefer != "speed" or chosen.decode_tps < SPEED_REFERENCE_TPS:
        return None
    faster = [
        candidate for candidate in pool
        if candidate.decode_tps > chosen.decode_tps
        and candidate.decode_tps >= SPEED_REFERENCE_TPS
    ]
    if not faster:
        return None
    other = max(faster, key=lambda candidate: candidate.decode_tps)
    return t(
        "warn.speed_saturated",
        policy.lang,
        role=role,
        chosen=chosen.model.id,
        chosen_quant=chosen.quant,
        chosen_tps=f"{chosen.decode_tps:.1f}",
        other=other.model.id,
        other_quant=other.quant,
        other_tps=f"{other.decode_tps:.1f}",
        reference=f"{SPEED_REFERENCE_TPS:.1f}",
    )


def _reserved_memory(
    services: Sequence[PlannedService], swap_group: Sequence[str],
) -> tuple[float, float]:
    swap_names = set(swap_group)
    resident = [service for service in services if service.name not in swap_names]
    swapped = [service for service in services if service.name in swap_names]
    return (
        sum(service.memory.gpu_bytes for service in resident)
        + max((service.memory.gpu_bytes for service in swapped), default=0.0),
        sum(service.memory.cpu_bytes for service in resident)
        + max((service.memory.cpu_bytes for service in swapped), default=0.0),
    )


def _spec_draft_path(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    models = nmesh_home() / "models"
    if not models.is_dir():
        return None
    wanted = value.casefold()
    return next(
        (
            item.resolve()
            for item in models.glob("*.gguf")
            if item.stem.casefold() == wanted
        ),
        None,
    )


def _spec_identity(path: Path) -> RoleIdentity:
    return RoleIdentity(
        model_id=path.stem,
        quant="",
        backend="llamacpp",
        artifact=service_fingerprint("llamacpp", str(path)) or "",
    )


def _spec_for_service(
    candidate: _Candidate,
    profile: HardwareProfile,
    policy: Policy,
    memory: MemoryEstimate,
    warnings: list[str],
    language: str,
) -> tuple[str, str, MemoryEstimate]:
    if policy.spec == "none":
        return "none", "", memory
    if candidate.backend != "llamacpp":
        warnings.append(t("warn.spec_unsupported", language, service=candidate.model.id))
        return "none", "", memory
    backend_flags = profile.backend_flags.get(candidate.backend)
    if not _honors_spec(candidate.backend, backend_flags, policy.spec):
        warnings.append(t("warn.spec_unsupported", language, service=candidate.model.id))
        return "none", "", memory
    draft_path: Path | None = None
    if policy.spec == "draft":
        draft_path = _spec_draft_path(policy.spec_draft)
        if draft_path is None:
            warnings.append(
                t("warn.spec_draft_missing", language, service=candidate.model.id,
                  draft=policy.spec_draft)
            )
            return "none", "", memory
        if candidate.n_gpu_layers > 0:
            warnings.append(
                t("warn.spec_draft_gpu_unmodeled", language,
                  service=candidate.model.id)
            )
            return "none", "", memory
        draft_bytes = draft_path.stat().st_size
        if memory.cpu_bytes + draft_bytes > memory.ram_budget + 1:
            warnings.append(
                t(
                    "warn.spec_draft_no_fit",
                    language,
                    service=candidate.model.id,
                    bytes=draft_bytes,
                )
            )
            return "none", "", memory
    target = RoleIdentity(
        model_id=candidate.model.id,
        quant=candidate.quant,
        backend=candidate.backend,
        artifact=service_fingerprint(
            candidate.backend,
            _source_for(candidate.backend, candidate.model, candidate.quant),
        ) or "",
    )
    spec_config = SpecConfig(
        kind=policy.spec,
        draft=_spec_identity(draft_path) if draft_path is not None else None,
        n_max=policy.spec_n_max,
    )
    try:
        cache = load_cache()
    except (OSError, TypeError, ValueError):
        cache = {}
    record = best_for(
        cache,
        target,
        spec_config,
        engine_identity(profile),
    )
    decision, reason = decide(record)
    if not policy.ignore_spec_evidence and decision != "allow":
        speeds = "-"
        if record is not None:
            speeds = ", ".join(
                f"{item.name}={item.speedup:.2f}x" for item in record.classes
            ) or "-"
        warnings.append(
            t(
                "warn.spec_refused",
                language,
                service=candidate.model.id,
                reason=reason,
                speeds=speeds,
            )
        )
        return "none", "", memory
    if policy.ignore_spec_evidence:
        warnings.append(t("warn.spec_override", language, service=candidate.model.id))
    if draft_path is not None:
        draft_bytes = draft_path.stat().st_size
        memory = replace(
            memory,
            weight_bytes=memory.weight_bytes + draft_bytes,
            total_bytes=memory.total_bytes + draft_bytes,
            cpu_bytes=memory.cpu_bytes + draft_bytes,
        )
    return policy.spec, str(draft_path or ""), memory


def _add_service(group: list[str], candidate: _Candidate, profile: HardwareProfile,
                 services: list[PlannedService], swap_group: list[str],
                 role_to_service: dict[str, str], hints: list[str],
                 missing_backends: list[str],
                 budget_source: str, warnings: list[str],
                 language: str = "en",
                 resident_override: bool | None = None,
                 spec_policy: Policy | None = None) -> None:
    if not candidate.installed:
        hints.append(t(INSTALL_HINTS[candidate.backend], language))
        if candidate.backend not in missing_backends:
            missing_backends.append(candidate.backend)
    indices: list[int] = []
    tensor_parallel = 1
    name = group[0]
    port = _service_port_base() + len(services)
    layers = candidate.n_gpu_layers
    memory = candidate.memory
    spec_kind, spec_draft, memory = _spec_for_service(
        candidate,
        profile,
        spec_policy or Policy(),
        memory,
        warnings,
        language,
    )
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
        kv_quant=candidate.kv_quant,
        warnings=warnings,
        gpu_devices=gpu_devices,
        language=language,
        roles=group,
        spec=spec_kind,
        spec_draft=spec_draft,
        spec_n_max=(spec_policy.spec_n_max if spec_policy is not None else 3),
        binary=profile.backend_paths.get(candidate.backend),
    )
    if candidate.backend == "llamacpp" and "hf_gguf" in candidate.model.sources:
        launch = replace(launch, env={"NMESH_HF_REPO": candidate.model.sources["hf_gguf"]})
    service = PlannedService(
        name, group, candidate.model.id, _source_for(candidate.backend, candidate.model, candidate.quant),
        candidate.model.sources.get("hf_gguf") or candidate.model.sources.get("hf"),
        candidate.quant, candidate.backend, candidate.context,
        11434 if candidate.backend == "ollama" else port, indices,
        None if candidate.backend in {"vllm", "mlx", "ollama"} else layers,
        (
            resident_override
            if resident_override is not None
            else profile.tier not in {Tier.T0_CPU, Tier.T1_LOW} or not services
        ),
        memory, candidate.decode_tps, candidate.estimated, launch,
        candidate.model.languages,
        kv_quant=candidate.kv_quant,
        spec=spec_kind,
        spec_draft=spec_draft,
    )
    if candidate.requested_kv_quant != candidate.kv_quant:
        warnings.append(
            t(
                "warn.kv_quant_unsupported",
                language,
                service=name,
                backend=candidate.backend,
                requested=candidate.requested_kv_quant,
            )
        )
    if candidate.kv_quant != "f16":
        warnings.append(
            t(
                "warn.kv_quant_speed_unmodeled",
                language,
                service=name,
                kv_quant=candidate.kv_quant,
            )
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
    services = [
        _cpu_llamacpp_service(
            service, profile.backend_flags.get(service.backend), warnings,
            profile.backend_gpu_devices.get(service.backend), policy.lang,
        )
        for service in services
    ]
    for service in services:
        if service.backend == "ollama":
            warnings.append(
                t(
                    "warn.ollama_quant_estimate",
                    policy.lang,
                    service=service.name,
                )
            )
    if not profile.gpus:
        return services
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
    cpu_fallback_warned: set[str] = set()

    def warn_cpu_fallback(service: PlannedService) -> None:
        if (
            profile.gpus
            and service.n_gpu_layers is not None
            and service.backend not in {"ollama", "vllm", "mlx"}
            and (service.n_gpu_layers == 0 or not service.gpu_indices)
            and service.name not in cpu_fallback_warned
        ):
            warnings.append(
                t(
                    "warn.gpu_layers_cpu_fallback",
                    policy.lang,
                    service=service.name,
                )
            )
            cpu_fallback_warned.add(service.name)

    for service in ordered:
        original_layers = service.n_gpu_layers
        if service.memory.gpu_bytes <= 0:
            warn_cpu_fallback(service)
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
        elif service.backend in {"llamacpp", "vllm"} and len(indices) > 1:
            assigned = indices
            tensor_parallel = len(indices)
            placement_budget = min(remaining[index] for index in indices) * tensor_parallel
            reserve_split = True
            if (
                service.backend == "llamacpp"
                and service.n_gpu_layers is not None
                and service.memory.gpu_bytes > placement_budget + 1
            ):
                model_layers = max(
                    1, round(
                        service.memory.weight_bytes / service.memory.per_layer_bytes
                    ),
                )
                adjusted = replace(service.memory, vram_budget=placement_budget)
                layers = solve_gpu_layers(adjusted, model_layers)
                gpu_bytes, cpu_bytes = _split_memory(
                    adjusted, model_layers, layers
                )
                if cpu_bytes <= adjusted.ram_budget + 1:
                    service = replace(
                        service,
                        n_gpu_layers=layers,
                        memory=replace(
                            adjusted,
                            gpu_bytes=gpu_bytes,
                            cpu_bytes=cpu_bytes,
                        ),
                    )
                    if layers == 0:
                        assigned = []
                        tensor_parallel = 1
                        reserve_split = False
                    else:
                        reserve_split = True
            if reserve_split:
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
                    if layers == 0:
                        assigned = []
                        current = replace(current, gpu_indices=[])
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
        if (
            tensor_parallel != 1
            or current.n_gpu_layers != original_layers
            or (tensor_parallel == 1 and "--tensor-split" in current.launch.argv)
        ):
            current = replace(
                current,
                launch=_rebuild_launch(
                    current, tensor_parallel, current.n_gpu_layers,
                    profile.backend_flags.get(current.backend), warnings,
                    gpu_devices=profile.backend_gpu_devices.get(current.backend),
                    language=policy.lang,
                ),
            )
        warn_cpu_fallback(current)
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
    """Assign memory-fitting parallel slots.

    One CPU machine measured 46.2 tok/s alone, 38.69 per request at two
    slots, 36.84 at four, and 23.30 at eight, with aggregate throughput of
    71.04, 125.22, and 147.50 tok/s respectively. This is one model on one
    CPU machine with llama.cpp and does not generalize. Slot counts are still
    derived from free memory; that is not evidence of a throughput benefit.
    """
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
        supports_slots = (
            cap > 1
        )
        if service.backend == "llamacpp":
            flags = profile.backend_flags.get("llamacpp")
            if flags is not None and not any(
                flag in flags for flag in ("-np", "--parallel")
            ):
                supports_slots = False
        requested = policy.parallel_slots
        if requested is not None:
            requested = max(1, requested)
        explicit = requested is not None and requested > 1
        if explicit and not supports_slots and warnings is not None:
            warnings.append(
                t("warn.slots_unsupported", policy.lang, service=service.name,
                  backend=service.backend, requested=requested)
            )
        eligible = (
            supports_slots
            and (full_gpu or explicit)
            and service.memory.kv_bytes_per_tok > 0
        )
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
        if explicit and supports_slots and slots != requested and warnings is not None:
            warnings.append(
                t("warn.slots_clamped", policy.lang, service=service.name,
                  requested=requested, slots=slots)
            )
        if slots > 1 and warnings is not None:
            warnings.append(
                t("warn.slots_tradeoff", policy.lang, service=service.name,
                  slots=slots, tps=f"{service.decode_tps:.1f}")
            )
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
               bench_cache: Mapping[object, float] | None = None,
               eval_cache: Mapping[
                   tuple[str, str, str], float | EvalSummary
               ] | None = None,
               artifact_cache: Mapping[str, int] | None = None,
               bench_records: Mapping[str, BenchRecord] | None = None,
               *,
               eval_depth_coverage: Mapping[tuple[str, str, str], int] | None = None,
               eval_depth_lost: Mapping[tuple[str, str, str], int] | None = None,
               ) -> Plan:
    selected = policy or Policy()
    roles = list(dict.fromkeys(selected.roles))
    requested_model_ids = {
        model_id.casefold() for model_id in selected.model_ids if model_id.strip()
    }
    catalog_models = [
        model for model in catalog
        if not requested_model_ids or model.id.casefold() in requested_model_ids
    ]
    measured = {
        tuple(item.casefold() for item in key): value
        for key, value in (eval_cache or {}).items()
        if isinstance(key, tuple)
        and len(key) == 3
        and all(isinstance(item, str) for item in key)
        and (
            isinstance(value, EvalSummary)
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and 0.0 <= value <= 1.0
            )
        )
    }
    warning_params = profile.warning_params
    warnings = [
        t(
            warning,
            selected.lang,
            **(warning_params[index] if index < len(warning_params) else {}),
        )
        for index, warning in enumerate(profile.warnings)
    ]
    warnings.append(t("warn.quality_prior", selected.lang))
    stale_bench_records = sum(
        record.harness != BENCH_HARNESS_VERSION
        for record in (bench_records or {}).values()
    )
    if stale_bench_records:
        warnings.append(t(
            "warn.bench_harness_mismatch",
            selected.lang,
            count=stale_bench_records,
        ))
    known_model_ids = {model.id.casefold() for model in catalog}
    for model_id in selected.model_ids:
        if model_id.casefold() not in known_model_ids:
            warnings.append(t("warn.model_unknown", selected.lang, model=model_id))
    unmeasured = sorted(
        model.id
        for model in catalog
        if model.quality is None
        and any(_catalog_role(role) in model.roles for role in roles)
        and model.id.casefold() not in requested_model_ids
    )
    if unmeasured:
        shown = ", ".join(unmeasured[:5])
        remaining = (
            f"; {len(unmeasured) - 5} more"
            if len(unmeasured) > 5
            else ""
        )
        warnings.append(
            t(
                "warn.quality_unmeasured",
                selected.lang,
                models=shown,
                remaining=remaining,
            )
        )
    if profile.backend_gpu_devices.get("llamacpp") == () and profile.gpus:
        warnings.append(
            t("warn.backend_no_gpu", selected.lang)
        )
    hints: list[str] = []
    missing_backends: list[str] = []
    for model in catalog_models:
        if (
            any(_catalog_role(role) in model.roles for role in roles)
            and not any(source in model.sources for source in ("hf", "hf_gguf", "ollama"))
        ):
            warnings.append(
                t("warn.no_source", selected.lang, model=model.id)
            )
    if profile.tier in {Tier.T0_CPU, Tier.T1_LOW, Tier.T2_MID, Tier.T3_HIGH}:
        groups = [[role for role in roles if role in {"chat", "code"}]]
        groups += [[role] for role in roles if role == "embed"]
    else:
        groups = [[role] for role in roles if role != "worker"]
    if "worker" in roles:
        groups.append(["worker"])
    services: list[PlannedService] = []
    swap_group: list[str] = []
    role_to_service: dict[str, str] = {}
    bench_excluded: list[dict[str, str]] = []
    bench_unconfirmed: list[dict[str, str]] = []
    total_download = 0
    for group in [item for item in groups if item]:
        reserved_vram, reserved_ram = _reserved_memory(services, swap_group)
        pools = {role: sorted(
            (candidate for model in catalog_models
             if _catalog_role(role) in model.roles
             for candidate in _candidate_for(
                 model, profile, selected, bench_cache, artifact_cache,
                 reserved_vram_bytes=reserved_vram,
                 reserved_ram_bytes=reserved_ram,
                 excluded=bench_excluded,
                 allow_unmeasured=model.id.casefold() in requested_model_ids,
                 bench_records=bench_records,
                 unconfirmed=bench_unconfirmed,
             )),
            key=lambda item: item.score, reverse=True,
        ) for role in group}
        empty_pools = None
        if reserved_vram or reserved_ram:
            empty_pools = {role: sorted(
                (candidate for model in catalog_models
                 if _catalog_role(role) in model.roles
                 for candidate in _candidate_for(
                     model,
                     profile,
                     selected,
                     bench_cache,
                     artifact_cache,
                     allow_unmeasured=model.id.casefold() in requested_model_ids,
                     bench_records=bench_records,
                     unconfirmed=bench_unconfirmed,
                 )),
                key=lambda item: item.score, reverse=True,
            ) for role in group}

        def apply_eval_override(
            current_group: list[str], current: _Candidate,
            source_pools: dict[str, list[_Candidate]] = pools,
        ) -> _Candidate:
            if not selected.eval_evidence:
                return current
            selected_evidence = measured.get(
                (
                    current.model.id.casefold(),
                    current.quant.casefold(),
                    current.backend.casefold(),
                )
            )
            if not isinstance(selected_evidence, EvalSummary):
                return current
            alternatives: list[
                tuple[_Candidate, EvalSummary, EvidenceTest]
            ] = []
            alternative_pairs = sorted({
                (candidate.model.id, candidate.quant)
                for pool in source_pools.values()
                for candidate in pool
                if (
                    candidate.model.id,
                    candidate.quant,
                ) != (current.model.id, current.quant)
            })
            for model_id, quant in alternative_pairs:
                if model_id != current.model.id:
                    model_candidates = {
                        role: [
                            candidate for candidate in pool
                            if candidate.model.id == model_id
                        ]
                        for role, pool in source_pools.items()
                    }
                    planned = _plan_group(current_group, model_candidates)
                    if planned is None or planned.quant != quant:
                        continue
                restricted = {
                    role: [
                        candidate for candidate in pool
                        if (
                            candidate.model.id == model_id
                            and candidate.quant == quant
                        )
                    ]
                    for role, pool in source_pools.items()
                }
                alternative = _plan_group(current_group, restricted)
                if alternative is None:
                    continue
                alternative_evidence = measured.get(
                    (
                        alternative.model.id.casefold(),
                        alternative.quant.casefold(),
                        alternative.backend.casefold(),
                    )
                )
                if not isinstance(alternative_evidence, EvalSummary):
                    continue
                if alternative_evidence.pass_rate <= selected_evidence.pass_rate:
                    continue
                evidence_test = _evidence_p_value(
                    alternative_evidence, selected_evidence,
                )
                if evidence_test.p_value < 0.05:
                    alternatives.append((
                        alternative, alternative_evidence, evidence_test,
                    ))
            if not alternatives:
                return current
            alternative, evidence, evidence_test = alternatives[0]
            for option in alternatives[1:]:
                if (
                    option[1].pass_rate,
                    option[0].score,
                    option[0].model.id,
                ) > (
                    evidence.pass_rate,
                    alternative.score,
                    alternative.model.id,
                ):
                    alternative, evidence, evidence_test = option
            if alternative.model.id == current.model.id:
                warnings.append(t(
                    "warn.eval_evidence_override_quant",
                    selected.lang,
                    role=current_group[0],
                    model=current.model.id,
                    other_quant=alternative.quant,
                    other_rate=evidence.pass_rate,
                    selected_quant=current.quant,
                    selected_rate=selected_evidence.pass_rate,
                    other_penalty=QUANT_PENALTY[alternative.quant],
                    selected_penalty=QUANT_PENALTY[current.quant],
                    p_value=evidence_test.p_value,
                    compared=evidence_test.compared,
                ))
            else:
                warnings.append(t(
                    "warn.eval_evidence_override",
                    selected.lang,
                    role=current_group[0],
                    other=alternative.model.id,
                    other_rate=evidence.pass_rate,
                    selected=current.model.id,
                    selected_rate=selected_evidence.pass_rate,
                    other_prior=alternative.model.quality,
                    selected_prior=current.model.quality,
                    p_value=evidence_test.p_value,
                    compared=evidence_test.compared,
                ))
            return alternative

        def warn_capacity_tradeoff(
            role: str, chosen: _Candidate, empty: _Candidate | None,
            committed_vram: float = reserved_vram,
            committed_ram: float = reserved_ram,
        ) -> None:
            if (
                empty is not None
                and (
                    chosen.model.id != empty.model.id
                    or chosen.quant != empty.quant
                )
            ):
                warnings.append(
                    t(
                        "warn.selection_capacity_tradeoff",
                        selected.lang,
                        role=role,
                        chosen_model=chosen.model.id,
                        chosen_quant=chosen.quant,
                        empty_model=empty.model.id,
                        empty_quant=empty.quant,
                        reserved_vram_gib=committed_vram / GIB,
                        reserved_ram_gib=committed_ram / GIB,
                    )
                )

        if group == ["worker"]:
            lead_name = role_to_service.get("chat", "")
            lead = next(
                (service for service in services if service.name == lead_name),
                None,
            )
            lead_model = next(
                (
                    model for model in catalog
                    if model.id.casefold() == lead.model_id.casefold()
                ),
                None,
            ) if lead is not None else None
            worker_candidates = (
                [
                    item for item in pools["worker"]
                    if lead is not None and lead_model is not None
                    and item.model.id != lead.model_id
                    and item.model.params < lead_model.params
                    and item.memory.weight_bytes <= lead.memory.weight_bytes
                    and item.decode_tps >= lead.decode_tps
                ]
                if lead is not None else []
            )
            if worker_candidates:
                candidate = min(
                    worker_candidates,
                    key=lambda item: (-item.score, item.model.id, item.quant),
                )
                _add_service(
                    group, candidate, profile, services, swap_group,
                    role_to_service, hints, missing_backends,
                    selected.budget_source, warnings, selected.lang,
                    resident_override=True,
                    spec_policy=selected,
                )
                worker = services[-1]
                resident_vram, resident_ram = _reserved_memory(
                    services, swap_group
                )
                vram_budget, ram_budget = _profile_budgets(
                    profile, selected.budget_source
                )
                if (
                    not worker.resident
                    or worker.name in swap_group
                    or resident_vram > vram_budget + 1
                    or resident_ram > ram_budget + 1
                ):
                    services.pop()
                    role_to_service.pop("worker", None)
                    if worker.name in swap_group:
                        swap_group.remove(worker.name)
                    warnings.append(
                        t("warn.worker_not_coresident", selected.lang)
                    )
                else:
                    total_download += int(candidate.memory.disk_needed)
            else:
                warnings.append(t("warn.worker_not_coresident", selected.lang))
            continue

        candidate = _plan_group(group, pools)
        if candidate is None and len(group) > 1:
            for role in group:
                role_candidate = pools.get(role, [])
                if role_candidate:
                    role_candidate = role_candidate[0]
                    role_candidate = apply_eval_override([role], role_candidate)
                    saturation_warning = _speed_saturation_warning(
                        role, role_candidate, pools[role], selected,
                    )
                    if saturation_warning is not None:
                        warnings.append(saturation_warning)
                    empty_candidate = (
                        empty_pools.get(role, [None])[0] if empty_pools else None
                    )
                    warn_capacity_tradeoff(role, role_candidate, empty_candidate)
                    _add_service(
                        [role], role_candidate, profile, services, swap_group,
                        role_to_service, hints, missing_backends,
                        selected.budget_source, warnings,
                        selected.lang, spec_policy=selected,
                    )
                    total_download += int(role_candidate.memory.disk_needed)
                else:
                    warnings.append(t("warn.no_candidate", selected.lang, role=role))
            continue
        if candidate is None:
            warnings.append(t("warn.no_candidate", selected.lang, role=group[0]))
            continue
        candidate = apply_eval_override(group, candidate)
        saturation_warning = _speed_saturation_warning(
            group[0], candidate, pools[group[0]], selected,
        )
        if saturation_warning is not None:
            warnings.append(saturation_warning)
        empty_candidate = (
            _plan_group(group, empty_pools) if empty_pools is not None else None
        )
        warn_capacity_tradeoff(group[0], candidate, empty_candidate)
        _add_service(
            group, candidate, profile, services, swap_group, role_to_service, hints,
            missing_backends, selected.budget_source, warnings, selected.lang,
            spec_policy=selected,
        )
        total_download += int(candidate.memory.disk_needed)
    seen_unconfirmed: set[str] = set()
    for measurement in bench_unconfirmed:
        identity = f"{measurement['model']}|{measurement['quant']}"
        if identity in seen_unconfirmed:
            continue
        seen_unconfirmed.add(identity)
        warnings.append(t(
            "warn.bench_unconfirmed",
            selected.lang,
            model=measurement["model"],
            quant=measurement["quant"],
            tps=measurement["tps"],
            threshold=measurement["threshold"],
        ))
    seen_excluded: set[str] = set()
    for exclusion in bench_excluded:
        model_id = exclusion["model"]
        if model_id in seen_excluded:
            continue
        seen_excluded.add(model_id)
        warnings.append(t(
            "warn.bench_excluded",
            selected.lang,
            model=model_id,
            quant=exclusion["quant"],
            tps=exclusion["tps"],
            threshold=exclusion["threshold"],
        ))
    services = _place_services(services, profile, selected, swap_group, warnings)
    services = _assign_slots(services, profile, selected, swap_group, warnings)
    selected_unmeasured: set[str] = set()
    for service in services:
        if service.model_id.casefold() in requested_model_ids:
            model = next(
                (
                    item for item in catalog_models
                    if item.id.casefold() == service.model_id.casefold()
                ),
                None,
            )
            if model is not None and model.quality is None:
                selected_unmeasured.add(model.id)
    if eval_depth_coverage is not None or eval_depth_lost is not None:
        for service in services:
            key = (
                service.model_id.casefold(),
                service.quant.casefold(),
                service.backend.casefold(),
            )
            lost_depth = (
                eval_depth_lost.get(key)
                if eval_depth_lost is not None
                else None
            )
            if lost_depth is not None and service.context >= lost_depth:
                warnings.append(t(
                    "warn.context_depth_broken",
                    selected.lang,
                    model=service.model_id,
                    quant=service.quant,
                    backend=service.backend,
                    context=service.context,
                    depth=lost_depth,
                ))
                continue
            measured_depth = (
                eval_depth_coverage.get(key)
                if eval_depth_coverage is not None
                else None
            )
            if measured_depth is not None and service.context > measured_depth:
                warnings.append(t(
                    "warn.context_unmeasured",
                    selected.lang,
                    model=service.model_id,
                    quant=service.quant,
                    backend=service.backend,
                    context=service.context,
                    depth=measured_depth,
                ))
    for model_id in sorted(selected_unmeasured):
        warnings.append(
            t("warn.quality_unmeasured_selected", selected.lang, model=model_id)
        )
    if eval_cache is not None:
        catalog_by_id = {model.id: model for model in catalog}
        contradiction_pairs: set[tuple[str, str, str, str]] = set()
        indistinguishable_pairs: set[tuple[str, str, str, str]] = set()
        eval_mismatch_models: set[str] = set()
        underpowered_emitted = False
        for service in services:
            selected_model = catalog_by_id.get(service.model_id)
            if selected_model is None:
                continue
            selected_evidence = measured.get(
                (
                    service.model_id.casefold(),
                    service.quant.casefold(),
                    service.backend.casefold(),
                )
            )
            if selected_evidence is None:
                if (
                    service.model_id not in eval_mismatch_models
                    and any(
                        key[0] == service.model_id.casefold()
                        for key in measured
                    )
                ):
                    eval_mismatch_models.add(service.model_id)
                    warnings.append(t(
                        "warn.eval_config_mismatch",
                        selected.lang,
                        model=service.model_id,
                        quant=service.quant,
                        backend=service.backend,
                    ))
                continue
            selected_rate = (
                selected_evidence.pass_rate
                if isinstance(selected_evidence, EvalSummary)
                else selected_evidence
            )
            for role in service.roles:
                for other in catalog:
                    if _catalog_role(role) not in other.roles:
                        continue
                    same_model = other.id == service.model_id
                    candidates = _candidate_for(
                        other, profile, selected, bench_cache, artifact_cache,
                        bench_records=bench_records,
                        unconfirmed=bench_unconfirmed,
                    )
                    if same_model:
                        candidates = [
                            candidate for candidate in candidates
                            if candidate.quant.casefold() != service.quant.casefold()
                            and isinstance(
                                measured.get(
                                    (
                                        other.id.casefold(),
                                        candidate.quant.casefold(),
                                        candidate.backend.casefold(),
                                    )
                                ),
                                EvalSummary,
                            )
                        ]
                    else:
                        if not any(
                            key[0] == other.id.casefold() for key in measured
                        ):
                            continue
                        if candidates:
                            candidates = [max(
                                candidates, key=lambda candidate: candidate.score,
                            )]
                    for other_candidate in candidates:
                        other_evidence = measured.get(
                            (
                                other.id.casefold(),
                                other_candidate.quant.casefold(),
                                other_candidate.backend.casefold(),
                            )
                        )
                        if not isinstance(selected_evidence, EvalSummary) or not isinstance(
                            other_evidence, EvalSummary
                        ):
                            continue
                        other_rate = other_evidence.pass_rate
                        other_prior = _effective_prior(
                            other, other_candidate.quant,
                        )
                        selected_prior = _effective_prior(
                            selected_model, service.quant,
                        )
                        evidence_test = _evidence_p_value(
                            other_evidence, selected_evidence,
                        )
                        suite_size = evidence_test.compared
                        same_backend = (
                            other_candidate.backend.casefold()
                            == service.backend.casefold()
                        )
                        other_penalty = QUANT_PENALTY[other_candidate.quant]
                        selected_penalty = QUANT_PENALTY[service.quant]
                        if (
                            same_model
                            and same_backend
                            and other_penalty > selected_penalty
                            and other_rate >= selected_rate
                            and evidence_test.p_value >= 0.05
                        ):
                            pair = (
                                other.id,
                                other_candidate.quant,
                                service.model_id,
                                service.quant,
                            )
                            if pair not in indistinguishable_pairs:
                                indistinguishable_pairs.add(pair)
                                warnings.append(t(
                                    "note.quant_indistinguishable",
                                    selected.lang,
                                    role=role,
                                    model=service.model_id,
                                    other_quant=other_candidate.quant,
                                    other_rate=other_rate,
                                    selected_quant=service.quant,
                                    selected_rate=selected_rate,
                                    compared=suite_size,
                                    p_value=evidence_test.p_value,
                                    penalty_gap=other_penalty - selected_penalty,
                                    selected_penalty=selected_penalty,
                                    other_penalty=other_penalty,
                                    minimum=min_resolvable_difference(suite_size),
                                ))
                            continue
                        if (
                            other_rate <= selected_rate
                            or other_prior is None
                            or selected_prior is None
                            or other_prior >= selected_prior
                        ):
                            continue
                        if evidence_test.p_value < 0.05:
                            pair = (
                                other.id,
                                other_candidate.quant,
                                service.model_id,
                                service.quant,
                            )
                            if pair in contradiction_pairs:
                                continue
                            contradiction_pairs.add(pair)
                            if same_model:
                                warnings.append(t(
                                    "warn.quant_penalty_contradiction",
                                    selected.lang,
                                    role=role,
                                    model=service.model_id,
                                    other_quant=other_candidate.quant,
                                    other_rate=other_rate,
                                    selected_quant=service.quant,
                                    selected_rate=selected_rate,
                                    other_penalty=QUANT_PENALTY[
                                        other_candidate.quant
                                    ],
                                    selected_penalty=QUANT_PENALTY[service.quant],
                                    p_value=evidence_test.p_value,
                                    compared=evidence_test.compared,
                                ))
                            else:
                                warnings.append(t(
                                    "warn.quality_contradiction",
                                    selected.lang,
                                    role=role,
                                    other=other.id,
                                    other_rate=other_rate,
                                    selected=service.model_id,
                                    selected_rate=selected_rate,
                                    other_prior=other.quality,
                                    selected_prior=selected_model.quality,
                                    p_value=evidence_test.p_value,
                                    compared=evidence_test.compared,
                                ))
                        elif not underpowered_emitted:
                            underpowered_emitted = True
                            if evidence_test.paired:
                                note_key = "note.eval_underpowered_paired"
                                discordant = (
                                    evidence_test.better_only
                                    + evidence_test.worse_only
                                )
                                imbalance = min_discordant_imbalance(discordant)
                                note_params = {
                                    "tasks": suite_size,
                                    "discordant": discordant,
                                    "better_only": evidence_test.better_only,
                                    "worse_only": evidence_test.worse_only,
                                    "p_value": evidence_test.p_value,
                                    "required": min_discordant_for_significance(),
                                    "imbalance": imbalance if imbalance is not None else "-",
                                }
                            else:
                                note_key = (
                                    "note.eval_underpowered"
                                    if suite_size < len(EXTENDED_TASKS)
                                    else "note.eval_underpowered_full"
                                )
                                note_params = {
                                    "tasks": suite_size,
                                    "other_rate": other_rate,
                                    "selected_rate": selected_rate,
                                    "p_value": evidence_test.p_value,
                                    "minimum": min_resolvable_difference(suite_size),
                                }
                                if suite_size < len(EXTENDED_TASKS):
                                    note_params.update({
                                        "upgrade_tasks": len(EXTENDED_TASKS),
                                        "upgrade_minimum": min_resolvable_difference(
                                            len(EXTENDED_TASKS),
                                        ),
                                    })
                            warnings.append(t(
                                note_key,
                                selected.lang,
                                **note_params,
                            ))
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
        total_download, runnable, list(dict.fromkeys(missing_backends)),
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
                    tuple(str(x) for x in pol.get("model_ids", [])),
                    bool(pol.get("eval_evidence", True)),
                    str(pol.get("spec", "none")),
                    str(pol.get("spec_draft", "")),
                    int(pol.get("spec_n_max", 3)),
                    bool(pol.get("ignore_spec_evidence", False)),
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
            kv_quant=str(sd.get("kv_quant", "f16")),
            spec=str(sd.get("spec", "none")),
            spec_draft=str(sd.get("spec_draft", "")),
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
        [str(x) for x in data.get("missing_backends", [])],
    )


def save_plan(plan: Plan, path: Path | None = None) -> Path:
    target = path or (nmesh_home() / "plan.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        payload = asdict(plan)
        policy_payload = payload.get("policy")
        if isinstance(policy_payload, dict):
            for key, default in (
                ("spec", "none"),
                ("spec_draft", ""),
                ("spec_n_max", 3),
                ("ignore_spec_evidence", False),
            ):
                if policy_payload.get(key) == default:
                    policy_payload.pop(key, None)
        for service_payload in payload.get("services", []):
            if not isinstance(service_payload, dict):
                continue
            if service_payload.get("spec") == "none":
                service_payload.pop("spec", None)
            if service_payload.get("spec_draft") == "":
                service_payload.pop("spec_draft", None)
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
    target = path or (nmesh_home() / "plan.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return _plan_from_dict(payload) if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


plan = build_plan
