from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import nmesh.planner.core as planner_core
from nmesh import i18n
from nmesh.bench import benchmark_key
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.catalog.loader import _model_from_mapping
from nmesh.planner import (
    BPW,
    Policy,
    build_plan,
    estimate_memory,
    free_budgets,
    load_plan,
    save_plan,
    structural_weight_bytes,
)
from nmesh.planner.core import _gpu_budget
from nmesh.probe import GPUInfo, HardwareProfile, Tier, classify_tier, profile_from_dict

GIB = 1024**3


def assert_memory_fit(result) -> None:
    for service in result.services:
        assert service.memory.gpu_bytes <= service.memory.vram_budget + 1
        assert service.memory.cpu_bytes <= service.memory.ram_budget + 1


def test_worker_role_plans_distinct_coresident_service(
    catalog: list[ModelSpec],
) -> None:
    small = next(item for item in catalog if item.id == "qwen2.5-0.5b-instruct")
    lead_model = next(item for item in catalog if item.id == "qwen2.5-1.5b-instruct")
    worker_catalog = [
        replace(lead_model, quality=100.0),
        replace(small, quality=50.0),
    ]
    result = build_plan(
        profile(16, (24,)),
        worker_catalog,
        Policy(roles=["chat", "worker"], min_decode_tps=0),
    )
    lead = next(item for item in result.services if "chat" in item.roles)
    worker = next(item for item in result.services if "worker" in item.roles)
    assert worker.model_id != lead.model_id
    assert worker.resident
    assert worker.name not in result.swap_group
    assert worker.memory.weight_bytes <= lead.memory.weight_bytes
    assert result.routing.role_to_service["worker"] == worker.name


def test_worker_role_warns_when_no_coresident_candidate(
    catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(1),
        catalog,
        Policy(roles=["chat", "worker"], min_decode_tps=0),
    )
    assert "worker" not in result.routing.role_to_service
    assert not any("worker" in service.roles for service in result.services)
    assert any(
        "worker role was requested" in warning.lower()
        for warning in result.warnings
    )


def test_worker_role_omits_larger_model_than_small_lead(
    catalog: list[ModelSpec],
) -> None:
    small = next(item for item in catalog if item.id == "qwen2.5-0.5b-instruct")
    larger = next(item for item in catalog if item.id == "qwen2.5-1.5b-instruct")
    result = build_plan(
        profile(16, (24,)),
        [
            replace(small, quality=100.0),
            replace(larger, quality=50.0),
        ],
        Policy(roles=["chat", "worker"], min_decode_tps=0),
    )
    lead = next(item for item in result.services if "chat" in item.roles)
    assert lead.model_id == "qwen2.5-0.5b-instruct"
    assert "worker" not in result.routing.role_to_service
    assert not any("worker" in service.roles for service in result.services)
    assert any("strictly smaller" in warning for warning in result.warnings)


def test_worker_role_warns_with_planned_lead_but_no_worker(
    catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(2),
        catalog,
        Policy(roles=["chat", "worker"], min_decode_tps=0),
    )
    assert any("chat" in service.roles for service in result.services)
    assert not any("worker" in service.roles for service in result.services)
    assert "worker" not in result.routing.role_to_service
    assert any("strictly smaller" in warning for warning in result.warnings)


def test_launch_uses_resolved_backend_binary_when_present(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    resolved = planner_core._launch(
        "llamacpp", model, "q4_k_m", 4096, 18010, 0, 1,
        binary="C:/llamacpp/llama-server.exe",
    )
    default = planner_core._launch(
        "llamacpp", model, "q4_k_m", 4096, 18010, 0, 1,
    )

    assert resolved.argv[0] == "C:/llamacpp/llama-server.exe"
    assert default.argv[0] == "llama-server"


def profile(
    ram_gib: int,
    gpu_gib: tuple[int, ...] = (),
    os_name: str = "windows",
    unified: bool = False,
    backends: dict[str, str | None] | None = None,
) -> HardwareProfile:
    gpus = [
        GPUInfo(index, f"Test GPU {size}GB", "nvidia", size * GIB, size * GIB, (8, 0), False)
        for index, size in enumerate(gpu_gib)
    ]
    return HardwareProfile(
        os_name, "Test CPU", 8, 16, ram_gib * GIB, ram_gib * GIB, 100 * GIB, unified, gpus,
        backends or {"ollama": "test", "llamacpp": "test", "vllm": "test", "mlx": None},
        classify_tier(gpus, unified, ram_gib * GIB), [],
    )


@pytest.fixture
def catalog() -> list[ModelSpec]:
    return load_catalog()


def placement_catalog() -> list[ModelSpec]:
    return [
        ModelSpec(
            "large", "large", 10_000_000_000, 40, 32, 8, 128, 4096, 8192,
            ["chat"], 90.0, "apache", {"hf_gguf": "large.gguf"},
        ),
        ModelSpec(
            "small", "small", 1_000_000_000, 24, 16, 4, 128, 2048, 8192,
            ["code"], 80.0, "apache", {"hf_gguf": "small.gguf"},
        ),
    ]


def symmetric_catalog() -> list[ModelSpec]:
    return [
        ModelSpec(
            "symmetric", "symmetric", 14_000_000_000, 40, 32, 8, 128, 4096,
            8192, ["chat", "code"], 90.0, "apache", {"hf_gguf": "symmetric.gguf"},
        ),
    ]


def test_memory_regression(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    estimate = estimate_memory(model, "q4_k_m", 2048)
    assert estimate.weight_bytes == pytest.approx(
        structural_weight_bytes(model, "q4_k_m")
    )


MEASURED_ARTIFACTS = {
    ("qwen2.5-0.5b-instruct", "q4_k_m"): 491400032,
    ("qwen2.5-0.5b-instruct", "q2_k"): 415182688,
    ("qwen2.5-0.5b-instruct", "f16"): 1266425696,
    ("qwen2.5-0.5b-instruct", "q8_0"): 675710816,
    ("qwen2.5-1.5b-instruct", "q4_k_m"): 1117320736,
    ("qwen2.5-1.5b-instruct", "q2_k"): 752880160,
    ("qwen2.5-7b-instruct", "q4_k_m"): 4683073632,
    ("qwen2.5-7b-instruct", "q2_k"): 3015940000,
    ("gemma2-9b", "q4_k_m"): 5761057728,
    ("gemma2-9b", "q2_k"): 3805397952,
    ("llama3.1-8b-instruct", "q4_k_m"): 4920739232,
    ("mistral-7b-instruct", "q4_k_m"): 4372812000,
    ("phi-4-14b", "q4_k_m"): 9053114816,
    ("bge-m3", "q4_k_m"): 437778496,
    ("bge-m3", "q2_k"): 366114880,
    ("nomic-embed-text-v1.5", "q4_k_m"): 84106624,
    ("nomic-embed-text-v1.5", "q2_k"): 49361088,
}


def test_structural_estimates_match_measured_envelope(
    catalog: list[ModelSpec],
) -> None:
    models = {model.id: model for model in catalog}
    ratios = []
    for (model_id, quant), measured in MEASURED_ARTIFACTS.items():
        ratio = structural_weight_bytes(models[model_id], quant) / measured
        ratios.append(ratio)
        assert 0.79 <= ratio <= 1.30
    assert ratios


@pytest.mark.parametrize(
    "model_id,quant",
    [
        ("qwen2.5-0.5b-instruct", "q4_k_m"),
        ("qwen2.5-0.5b-instruct", "q2_k"),
        ("qwen2.5-0.5b-instruct", "f16"),
        ("qwen2.5-0.5b-instruct", "q8_0"),
        ("bge-m3", "q2_k"),
    ],
)
def test_structural_estimate_fixes_old_underestimate(
    catalog: list[ModelSpec], model_id: str, quant: str
) -> None:
    model = next(item for item in catalog if item.id == model_id)
    measured = MEASURED_ARTIFACTS[(model_id, quant)]
    old_ratio = model.params * BPW[quant] / 8 / measured
    structural_ratio = structural_weight_bytes(model, quant) / measured
    assert old_ratio < 0.79
    assert structural_ratio >= 0.79


def test_structural_estimate_never_regresses_below_old_formula(
    catalog: list[ModelSpec],
) -> None:
    models = {model.id: model for model in catalog}
    for model_id in {model_id for model_id, _ in MEASURED_ARTIFACTS}:
        model = models[model_id]
        for quant in BPW:
            assert structural_weight_bytes(model, quant) + 1e-6 >= (
                model.params * BPW[quant] / 8
            )


def test_metadata_free_models_keep_old_weight_formula() -> None:
    model = ModelSpec(
        "legacy", "test", 123_456_789, 12, 8, 2, 64, 512, 4096,
        ["chat"], 80.0, "apache", {"hf_gguf": "org/legacy"},
    )
    assert structural_weight_bytes(model, "q4_k_m") == (
        model.params * BPW["q4_k_m"] / 8
    )


def test_artifact_cache_overrides_structural_estimate(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-0.5b-instruct")
    repo_id = model.sources["hf_gguf"]
    candidates = planner_core._candidate_for(
        model,
        profile(64),
        Policy(roles=["chat"], min_decode_tps=0),
        None,
        {f"{repo_id}|q4_k_m": 999_000_000},
        allow_unmeasured=True,
    )
    candidate = next(item for item in candidates if item.quant == "q4_k_m")
    assert candidate.memory.weight_bytes == 999_000_000


def test_measured_artifact_can_change_candidate_selection() -> None:
    model = ModelSpec(
        "tight", "test", 1_000_000_000, 24, 16, 4, 128, 2048, 4096,
        ["chat"], 80.0, "apache", {"hf_gguf": "org/tight"},
    )
    policy = Policy(roles=["chat"], min_decode_tps=0)
    structural = build_plan(profile(4), [model], policy)
    measured = build_plan(
        profile(4),
        [model],
        policy,
        artifact_cache={"org/tight|q6_k": 6_000_000_000},
    )
    assert structural.services[0].quant == "q6_k"
    assert measured.services[0].quant != "q6_k"


def test_catalog_quality_null_is_explicit_and_invalid_quality_is_rejected() -> None:
    base = {
        "id": "candidate",
        "family": "Candidate",
        "params": 1,
        "n_layers": 1,
        "n_heads": 1,
        "n_kv_heads": 1,
        "head_dim": 1,
        "hidden_size": 1,
        "max_context": 1,
        "roles": ["chat"],
        "quality": None,
        "license": "apache",
        "sources": {"hf": "org/candidate"},
    }
    model = _model_from_mapping(base)
    assert model is not None
    assert model.quality is None
    assert _model_from_mapping({**base, "quality": "unknown"}) is None
    missing = dict(base)
    del missing["quality"]
    assert _model_from_mapping(missing) is None


def test_unmeasured_models_require_explicit_selection() -> None:
    model = ModelSpec(
        "candidate", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], None, "apache", {"hf_gguf": "org/candidate"},
    )
    hardware = profile(64)
    policy = Policy(roles=["chat"], min_decode_tps=0)
    assert planner_core._candidate_for(model, hardware, policy, None) == []
    candidates = planner_core._candidate_for(
        model, hardware, policy, None, allow_unmeasured=True,
    )
    assert candidates
    candidate = candidates[0]
    speed_weight = {"quality": 0.1, "speed": 1.0, "balanced": 0.25}[policy.prefer]
    expected = min(candidate.decode_tps, planner_core.SPEED_REFERENCE_TPS) / (
        planner_core.SPEED_REFERENCE_TPS
    ) * 100 * speed_weight
    assert candidate.score == pytest.approx(expected)


def test_explicit_models_restrict_roles_and_warn_for_unknown() -> None:
    selected = ModelSpec(
        "selected", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat", "code"], 80.0, "apache", {"hf_gguf": "org/selected"},
    )
    other = ModelSpec(
        "other", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat", "code"], 99.0, "apache", {"hf_gguf": "org/other"},
    )
    result = build_plan(
        profile(64),
        [selected, other],
        Policy(roles=["chat", "code"], model_ids=("SELECTED", "missing")),
    )
    assert result.services
    assert {service.model_id for service in result.services} == {"selected"}
    assert any("missing" in warning for warning in result.warnings)


def test_unmeasured_warnings_distinguish_automatic_and_selected() -> None:
    automatic = ModelSpec(
        "automatic", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], None, "apache", {"hf_gguf": "org/automatic"},
    )
    selected = ModelSpec(
        "selected", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], None, "apache", {"hf_gguf": "org/selected"},
    )
    excluded = build_plan(profile(64), [automatic], Policy(roles=["chat"]))
    assert not excluded.services
    assert any("automatic" in warning and "nmesh eval" in warning
               for warning in excluded.warnings)
    planned = build_plan(
        profile(64),
        [automatic, selected],
        Policy(roles=["chat"], model_ids=("selected",), min_decode_tps=0),
    )
    assert planned.services[0].model_id == "selected"
    assert any("selected" in warning and "speed term only" in warning
               for warning in planned.warnings)


def test_free_budget_source_limits_gpu_layers_and_old_plans_load(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    base = profile(32, (24,))
    limited = replace(base, gpus=[replace(base.gpus[0], free_vram_bytes=4 * GIB)])
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    total = build_plan(base, [model], Policy(roles=["chat"]))
    free = build_plan(limited, [model], Policy(roles=["chat"], budget_source="free"))
    total_service = total.services[0]
    free_service = free.services[0]
    assert (
        (free_service.n_gpu_layers or 0) < (total_service.n_gpu_layers or 0)
        or free_service.quant != total_service.quant
    )
    assert free_service.memory.gpu_bytes <= free_budgets(limited)[0] + 1
    estimate = estimate_memory(
        model, "q4_k_m", 2048, profile=limited, budget_source="free"
    )
    assert estimate.vram_budget == pytest.approx(free_budgets(limited)[0])

    path = tmp_path / "old-plan.json"
    save_plan(total, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["policy"]["budget_source"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.policy.budget_source == "total"


def test_plan_round_trip_preserves_explicit_model_ids(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    result = build_plan(
        profile(32),
        catalog,
        Policy(
            roles=["chat"],
            model_ids=("qwen2.5-7b-instruct",),
            eval_evidence=False,
        ),
    )
    path = tmp_path / "plan.json"
    save_plan(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["policy"]["eval_evidence"] is False
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.policy.model_ids == ("qwen2.5-7b-instruct",)
    assert loaded.policy.eval_evidence is False


def test_missing_backend_ids_are_structural_and_ordered(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    unavailable = profile(
        32,
        backends={"ollama": None, "llamacpp": None, "vllm": None, "mlx": None},
    )
    missing = build_plan(unavailable, catalog, Policy(roles=["chat"]))
    assert not missing.runnable
    assert missing.missing_backends == ["llamacpp"]

    runnable = build_plan(profile(32), catalog, Policy(roles=["chat"]))
    assert runnable.runnable
    assert runnable.missing_backends == []

    path = tmp_path / "old-plan.json"
    save_plan(runnable, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["missing_backends"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.missing_backends == []


def test_cpu_case(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(8), catalog)
    assert_memory_fit(result)
    assert result.tier == Tier.T0_CPU
    assert result.services
    assert result.runnable
    assert len(result.swap_group) == 1
    assert result.services[0].n_gpu_layers == 0
    assert result.services[0].backend in {"ollama", "llamacpp"}
    assert result.services[0].gpu_indices == []
    assert result.services[0].memory.gpu_bytes == 0
    assert result.services[0].memory.cpu_bytes <= result.services[0].memory.ram_budget


def test_low_gpu_partial_offload(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(16, (4,)), catalog, Policy(roles=["chat"]))
    assert_memory_fit(result)
    service = result.services[0]
    assert result.tier == Tier.T1_LOW
    assert service.n_gpu_layers is not None
    assert 0 < service.n_gpu_layers < 100


def test_mid_gpu_full_model(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(32, (12,)), catalog, Policy(roles=["chat"]))
    assert_memory_fit(result)
    service = result.services[0]
    assert result.tier in {Tier.T2_MID, Tier.T3_HIGH}
    if service.backend == "ollama":
        assert service.n_gpu_layers is None
    else:
        assert service.n_gpu_layers == next(
            item for item in catalog if item.id == service.model_id
        ).n_layers


def test_workstation_resident_roles(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(64, (24,)), catalog)
    assert_memory_fit(result)
    assert result.tier == Tier.T4_WORKSTATION
    assert all(service.resident for service in result.services)
    assert set(result.routing.role_to_service) == {"chat", "code", "embed"}


def test_server_vllm_single_gpu(catalog: list[ModelSpec]) -> None:
    result = build_plan(
        profile(128, (80, 80), os_name="linux"),
        catalog,
        Policy(roles=["chat"]),
    )
    assert_memory_fit(result)
    assert result.tier == Tier.T5_SERVER
    assert result.services[0].backend == "vllm"
    assert len(result.services[0].gpu_indices) == 1


def test_apple_mlx(catalog: list[ModelSpec]) -> None:
    result = build_plan(
        profile(
            64,
            (44,),
            os_name="macos",
            unified=True,
            backends={"ollama": None, "llamacpp": None, "vllm": None, "mlx": "installed"},
        ),
        catalog,
        Policy(roles=["chat"]),
    )
    assert_memory_fit(result)
    assert result.services[0].backend == "mlx"


def test_no_backend_still_plans(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(8, backends={"ollama": None, "llamacpp": None, "vllm": None, "mlx": None}),
                        catalog, Policy(roles=["chat"]))
    assert result.services
    assert result.install_hints
    assert not result.runnable


def test_ollama_only_model_never_uses_llamacpp(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        planner_core.Path,
        "home",
        classmethod(lambda _cls: tmp_path),
    )
    model_dir = tmp_path / ".nmesh" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "ollama-only-q4_k_m.gguf").write_bytes(b"stray local file")
    model = ModelSpec(
        "ollama-only", "test", 500_000_000, 24, 14, 2, 64, 896, 4096,
        ["chat"], 90.0, "apache", {"ollama": "test:model"},
    )
    result = build_plan(profile(8), [model], Policy(roles=["chat"]))
    assert result.services[0].backend == "ollama"


def test_installed_lower_preference_backend_wins() -> None:
    model = ModelSpec(
        "both-sources", "test", 500_000_000, 24, 14, 2, 64, 896, 4096,
        ["chat"], 90.0, "apache",
        {"hf_gguf": "repo", "ollama": "test:model"},
    )
    available = profile(
        8,
        backends={"ollama": "installed", "llamacpp": None, "vllm": None, "mlx": None},
    )

    assert planner_core._backend(available, model, 14) == ("ollama", True)


def test_hf_only_model_does_not_warn_about_missing_gguf_source() -> None:
    model = ModelSpec(
        "hf-only", "test", 500_000_000, 24, 14, 2, 64, 896, 4096,
        ["chat"], 90.0, "apache", {"hf": "test/model"},
    )
    result = build_plan(profile(8), [model], Policy(roles=["chat"]))
    assert not any("hf-only" in warning for warning in result.warnings)


def test_kv_quantization_is_independent(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    f16 = estimate_memory(model, "q4_k_m", 2048, kv_quant="f16")
    q8 = estimate_memory(model, "q4_k_m", 2048, kv_quant="q8_0")
    assert q8.kv_cache_bytes == pytest.approx(f16.kv_cache_bytes / 2)
    assert q8.weight_bytes == f16.weight_bytes


def test_bench_lookup_isolated_by_kv_precision() -> None:
    model = ModelSpec(
        "bench-kv", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    f16_key = benchmark_key(model.id, "q4_k_m", "llamacpp", "cpu", 0, "f16")
    q8_key = benchmark_key(model.id, "q4_k_m", "llamacpp", "cpu", 0, "q8_0")
    cache = {f16_key: 16.0, q8_key: 8.0}
    assert planner_core._bench_value(
        cache, model, "q4_k_m", "llamacpp", "cpu", 0, "f16"
    ) == 16.0
    assert planner_core._bench_value(
        cache, model, "q4_k_m", "llamacpp", "cpu", 0, "q8_0"
    ) == 8.0
    assert planner_core._bench_value(
        {f16_key: 16.0}, model, "q4_k_m", "llamacpp", "cpu", 0, "q8_0"
    ) is None


def test_supported_llamacpp_kv_quantization_is_launched(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={
            "llamacpp": (
                "--parallel", "-ngl", "--tensor-split",
                "--cache-type-k", "--cache-type-v",
            ),
        },
    )
    result = build_plan(machine, [model], Policy(roles=["chat"], kv_quant="q8_0"))
    service = result.services[0]
    f16 = estimate_memory(
        model, service.quant, service.context,
        parallel_slots=service.memory.parallel_slots, kv_quant="f16",
    )
    assert service.kv_quant == "q8_0"
    assert service.memory.kv_cache_bytes == pytest.approx(f16.kv_cache_bytes / 2)
    assert service.launch.argv[service.launch.argv.index("--cache-type-k") + 1] == "q8_0"
    assert service.launch.argv[service.launch.argv.index("--cache-type-v") + 1] == "q8_0"
    speed_warnings = [
        warning for warning in result.warnings
        if "planned throughput does not model KV cache type" in warning
    ]
    assert len(speed_warnings) == 1


def test_unsupported_llamacpp_kv_quantization_downgrades_accounting(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl", "--tensor-split")},
    )
    result = build_plan(machine, [model], Policy(roles=["chat"], kv_quant="q8_0"))
    service = result.services[0]
    f16 = estimate_memory(
        model, service.quant, service.context,
        parallel_slots=service.memory.parallel_slots, kv_quant="f16",
    )
    assert service.kv_quant == "f16"
    assert service.memory.kv_cache_bytes == pytest.approx(f16.kv_cache_bytes)
    assert not any("--cache-type" in flag for flag in service.launch.argv)
    assert any("llamacpp" in warning and "accounted at f16" in warning
               for warning in result.warnings)


def test_f16_kv_quantization_has_no_speed_warning(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(
        profile(64, (24,)), [model], Policy(roles=["chat"], kv_quant="f16")
    )
    assert not any(
        "planned throughput does not model KV cache type" in warning
        for warning in result.warnings
    )


@pytest.mark.parametrize(
    ("backend", "os_name", "gpu_gib"),
    [("ollama", "windows", ()), ("vllm", "linux", (24,))],
)
def test_non_llamacpp_kv_quantization_downgrades_accounting(
    catalog: list[ModelSpec], backend: str, os_name: str, gpu_gib: tuple[int, ...],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = profile(
        64,
        gpu_gib,
        os_name=os_name,
        backends={
            "ollama": "test" if backend == "ollama" else None,
            "llamacpp": None,
            "vllm": "test" if backend == "vllm" else None,
            "mlx": None,
        },
    )
    result = build_plan(machine, [model], Policy(roles=["chat"], kv_quant="q8_0"))
    service = result.services[0]
    f16 = estimate_memory(
        model, service.quant, service.context,
        parallel_slots=service.memory.parallel_slots, kv_quant="f16",
    )
    assert service.backend == backend
    assert service.kv_quant == "f16"
    assert service.memory.kv_cache_bytes == pytest.approx(f16.kv_cache_bytes)
    assert not any("--cache-type" in flag for flag in service.launch.argv)
    assert any(backend in warning and "accounted at f16" in warning
               for warning in result.warnings)


def test_unsupported_kv_quantization_cannot_false_fit() -> None:
    model = ModelSpec(
        "fit", "fit", 500_000_000, 24, 16, 16, 128, 2048, 8192,
        ["chat"], 90.0, "test", {"ollama": "test"},
    )
    base = profile(
        1,
        backends={"ollama": "test", "llamacpp": None, "vllm": None, "mlx": None},
    )
    machine = replace(
        base,
        total_ram_bytes=int(1.2 * GIB),
        available_ram_bytes=int(1.2 * GIB),
    )
    result = build_plan(
        machine, [model],
        Policy(roles=["chat"], max_context=2048, kv_quant="q8_0", min_decode_tps=0),
    )
    assert result.services == []
    assert any("No runnable model" in warning for warning in result.warnings)


def test_embedding_has_activation_memory_not_kv(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "bge-m3")
    result = build_plan(profile(32), catalog, Policy(roles=["embed"]))
    service = result.services[0]
    assert service.model_id == model.id
    assert service.context == min(model.max_context, 8192)
    assert service.memory.kv_cache_bytes == 0
    assert service.memory.compute_overhead > 0.06 * service.memory.weight_bytes + 320 * 1024**2


def test_cpu_quality_preference_avoids_tiny_model(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(32), catalog, Policy(roles=["chat"]))
    assert result.services[0].model_id != "qwen2.5-0.5b-instruct"


def test_sequential_selection_reserves_prior_service_capacity(tmp_path) -> None:
    profile_path = Path(__file__).parents[1] / "profiles" / "t3-rtx4090-24gb.json"
    hardware = profile_from_dict(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    catalog = load_catalog(user_path=tmp_path / "models.yaml")
    result = build_plan(
        hardware, catalog, Policy(roles=["chat", "embed"], min_decode_tps=0)
    )
    chat = next(service for service in result.services if service.name == "chat")
    embed = next(service for service in result.services if service.name == "embed")

    assert embed.memory.gpu_bytes <= max(
        _gpu_budget(hardware.gpus[0]) - chat.memory.gpu_bytes, 0.0
    ) + 1
    assert embed.quant != "f16" or embed.n_gpu_layers == 0
    assert any("capacity-forced tradeoff" in warning for warning in result.warnings)


def test_zero_selection_reservation_is_a_noop(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    hardware = profile(32, (12,))
    policy = Policy(roles=["chat"], min_decode_tps=0)
    first = build_plan(hardware, [model], policy)
    second = build_plan(hardware, [model], policy)
    first_data = asdict(first)
    second_data = asdict(second)
    first_data.pop("created_at")
    second_data.pop("created_at")
    assert first_data == second_data

    default = planner_core._candidate_for(model, hardware, policy, None)
    explicit_zero = planner_core._candidate_for(
        model, hardware, policy, None,
        reserved_vram_bytes=0.0, reserved_ram_bytes=0.0,
    )
    assert default == explicit_zero


def test_bench_exclusion_warning_reports_estimate_admission() -> None:
    model = ModelSpec(
        "bench-excluded", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    policy = Policy(roles=["chat"])
    hardware = profile(64)
    assert planner_core._candidate_for(model, hardware, policy, None)
    cache = {
        benchmark_key(model.id, quant, "llamacpp", "cpu", 0): 1.0
        for quant in planner_core.BPW
    }
    result = build_plan(hardware, [model], policy, cache)
    excluded = [warning for warning in result.warnings if "bench-excluded" in warning]
    assert len(excluded) == 1
    assert "excluded" in excluded[0]
    assert not result.services


def test_bench_exclusion_warning_requires_passing_estimate() -> None:
    model = ModelSpec(
        "bench-not-admitted", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    policy = Policy(roles=["chat"], min_decode_tps=1000.0)
    cache = {
        benchmark_key(model.id, quant, "llamacpp", "cpu", 0): 1.0
        for quant in planner_core.BPW
    }
    result = build_plan(profile(64), [model], policy, cache)
    assert not any("bench-not-admitted" in warning for warning in result.warnings)


def test_oversized_model_uses_tensor_parallel() -> None:
    model = ModelSpec("oversized", "test", 150_000_000_000, 100, 100, 100, 128,
                      12800, 4096, ["chat"], 99.0, "test", {"hf": "test/model"})
    result = build_plan(profile(128, (80, 80), os_name="linux"),
                        [model], Policy(roles=["chat"], min_decode_tps=0))
    service = result.services[0]
    assert service.backend == "vllm"
    assert service.gpu_indices == [0, 1]
    assert "--tensor-parallel-size" in service.launch.argv


def test_oversized_llamacpp_model_uses_tensor_split() -> None:
    model = ModelSpec(
        "oversized-llamacpp", "test", 40_000_000_000, 80, 32, 8, 128,
        4096, 128, ["chat"], 99.0, "test", {"hf_gguf": "test.gguf"},
    )
    result = build_plan(
        profile(128, (24, 8)),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.backend == "llamacpp"
    assert service.gpu_indices == [0, 1]
    assert "--tensor-split" in service.launch.argv


def test_oversized_llamacpp_cpu_fallback_clears_split_and_warns() -> None:
    model = ModelSpec(
        "cpu-fallback-llamacpp", "test", 60_000_000_000, 80, 80, 100, 128,
        12800, 4096, ["chat"], 99.0, "test", {"hf_gguf": "test.gguf"},
    )
    result = build_plan(
        profile(128, (24, 8)),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.backend == "llamacpp"
    assert service.n_gpu_layers == 0
    assert service.gpu_indices == []
    assert service.launch.argv[service.launch.argv.index("-ngl") + 1] == "0"
    assert "--tensor-split" not in service.launch.argv
    warning = i18n.t("warn.gpu_layers_cpu_fallback", "en", service=service.name)
    assert result.warnings.count(warning) == 1


def test_ollama_quantization_warning_is_emitted_for_placed_service() -> None:
    model = ModelSpec(
        "placed-ollama", "test", 500_000_000, 24, 14, 2, 64, 896,
        4096, ["chat"], 90.0, "test", {"ollama": "test:model"},
    )
    result = build_plan(
        profile(32, (24,)),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.backend == "ollama"
    assert service.gpu_indices == [0]
    assert sum("Ollama tag's own quantization" in warning for warning in result.warnings) == 1


def test_single_gpu_cpu_fallback_warns() -> None:
    models = [
        ModelSpec(
            "first", "first", 20_000_000_000, 80, 80, 100, 128,
            12800, 4096, ["chat"], 99.0, "test", {"hf_gguf": "first.gguf"},
        ),
        ModelSpec(
            "second", "second", 20_000_000_000, 80, 80, 100, 128,
            12800, 4096, ["code"], 99.0, "test", {"hf_gguf": "second.gguf"},
        ),
    ]
    result = build_plan(
        profile(128, (24,)),
        models,
        Policy(roles=["chat", "code"], min_decode_tps=0),
    )
    service = next(item for item in result.services if item.name == "code")
    assert service.n_gpu_layers == 0
    assert service.gpu_indices == []
    warning = i18n.t("warn.gpu_layers_cpu_fallback", "en", service=service.name)
    assert result.warnings.count(warning) == 1


def test_save_plan_replaces_atomically(tmp_path, catalog: list[ModelSpec], monkeypatch) -> None:
    first = build_plan(profile(8), catalog)
    path = tmp_path / "plan.json"
    save_plan(first, path)
    assert list(tmp_path.iterdir()) == [path]
    original = path.read_bytes()

    def fail(self, *args, **kwargs) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(planner_core.Path, "write_text", fail)
    with pytest.raises(OSError):
        save_plan(build_plan(profile(24, (24,)), catalog), path)
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == original
    assert load_plan(path) == first


def test_plan_save_load(tmp_path, catalog: list[ModelSpec]) -> None:
    from nmesh.planner import load_plan, save_plan

    machine = replace(
        profile(8),
        backend_flags={"llamacpp": ("--cache-type-k", "--cache-type-v")},
    )
    result = build_plan(machine, catalog, Policy(kv_quant="q8_0"))
    path = tmp_path / "plan.json"
    save_plan(result, path)
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded == result
    assert loaded.policy.kv_quant == "q8_0"
    assert loaded.services[0].kv_quant == result.services[0].kv_quant
    assert isinstance(loaded.services[0].memory.n_gpu_layers, int)


def test_old_plan_without_service_kv_quant_defaults_to_f16(
    tmp_path, catalog: list[ModelSpec],
) -> None:
    result = build_plan(profile(8), catalog)
    path = tmp_path / "plan.json"
    save_plan(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["services"][0]["kv_quant"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.services[0].kv_quant == "f16"


def test_plan_save_load_serializes_backend_flags_as_strings(
    tmp_path, catalog: list[ModelSpec]
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl")},
    )
    result = build_plan(machine, [model], Policy(roles=["chat"]))
    path = tmp_path / "plan.json"
    save_plan(result, path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    flags = payload["profile"]["backend_flags"]["llamacpp"]
    assert isinstance(flags, list)
    assert all(isinstance(flag, str) for flag in flags)
    assert "frozenset(" not in text
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.profile.backend_flags["llamacpp"] == ("--parallel", "-ngl")


def test_full_gpu_slots_expand_llamacpp_context_and_kv(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = result.services[0]
    assert service.backend == "llamacpp"
    assert service.memory.parallel_slots > 1
    argv = service.launch.argv
    slots = service.memory.parallel_slots
    assert argv[argv.index("--parallel") + 1] == str(slots)
    assert argv[argv.index("-c") + 1] == str(service.context * slots)
    assert service.memory.kv_cache_bytes == pytest.approx(
        service.memory.kv_bytes_per_tok * service.context * slots
    )
    gpu = next(item for item in result.profile.gpus if item.index == service.gpu_indices[0])
    assert service.memory.gpu_bytes <= planner_core._gpu_budget(gpu) + 1


def test_cpu_partial_offload_and_embedding_keep_one_slot(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    cpu = build_plan(profile(8), [model])
    assert all(item.memory.parallel_slots == 1 for item in cpu.services)
    llama = next(item for item in cpu.services if item.backend == "llamacpp")
    assert llama.launch.argv[llama.launch.argv.index("--parallel") + 1] == "1"
    assert llama.launch.argv[llama.launch.argv.index("-c") + 1] == str(llama.context)

    partial = build_plan(profile(16, (4,)), [model], Policy(roles=["chat"]))
    assert partial.services[0].memory.parallel_slots == 1

    embed = build_plan(profile(32), catalog, Policy(roles=["embed"]))
    assert embed.services[0].memory.parallel_slots == 1
    assert embed.services[0].memory.kv_bytes_per_tok == 0


def test_explicit_cpu_slots_scale_kv_cache_and_warn_tradeoff(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(
        profile(32), [model], Policy(roles=["chat"], parallel_slots=4)
    )
    service = result.services[0]
    assert service.memory.parallel_slots > 1
    assert service.memory.kv_cache_bytes == pytest.approx(
        service.memory.kv_bytes_per_tok * service.context * service.memory.parallel_slots
    )
    assert any("parallel slots" in warning for warning in result.warnings)


def test_automatic_cpu_slots_remain_one(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(profile(32), [model], Policy(roles=["chat"]))
    assert result.services[0].memory.parallel_slots == 1


def test_explicit_ollama_slots_warn_when_unsupported(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(
        profile(32, backends={"ollama": "test"}),
        [model],
        Policy(roles=["chat"], parallel_slots=4),
    )
    service = result.services[0]
    assert service.backend == "ollama"
    assert service.memory.parallel_slots == 1
    assert any("cannot serve 4 parallel slots" in warning for warning in result.warnings)


def test_explicit_cpu_slots_clamp_when_ram_is_limited(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(
        profile(8), [model], Policy(roles=["chat"], parallel_slots=8)
    )
    service = result.services[0]
    assert service.memory.parallel_slots < 8
    assert any(
        "parallel_slots 8" in warning
        and str(service.memory.parallel_slots) in warning
        for warning in result.warnings
    )


def test_llamacpp_caps_without_parallel_clamp_slots_and_context(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    no_parallel = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--tensor-split", "-ngl")},
    )
    result = build_plan(no_parallel, [model], Policy(roles=["chat"]))
    service = result.services[0]
    assert service.memory.parallel_slots == 1
    assert service.memory.kv_cache_bytes == (
        service.memory.kv_bytes_per_tok * service.context
    )
    assert "--parallel" not in service.launch.argv
    assert service.launch.argv[service.launch.argv.index("-c") + 1] == str(service.context)
    assert any("--parallel" in warning for warning in result.warnings)


def test_llamacpp_caps_without_gpu_layers_force_cpu_placement(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    no_gpu_layers = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "--tensor-split")},
    )
    result = build_plan(no_gpu_layers, [model], Policy(roles=["chat"]))
    service = result.services[0]
    assert service.n_gpu_layers == 0
    assert service.memory.gpu_bytes == 0
    assert service.memory.cpu_bytes >= service.memory.weight_bytes
    assert not any(flag in service.launch.argv for flag in ("-ngl", "--gpu-layers", "--n-gpu-layers"))
    assert any("GPU-layer flags unsupported" in warning for warning in result.warnings)


def test_unknown_llamacpp_caps_preserve_launch_behavior(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = result.services[0]
    slots = service.memory.parallel_slots
    assert "--parallel" in service.launch.argv
    assert service.launch.argv[service.launch.argv.index("--parallel") + 1] == str(slots)
    assert service.launch.argv[service.launch.argv.index("-c") + 1] == str(
        service.context * slots
    )


def test_vllm_slots_and_total_vram_fraction() -> None:
    model = ModelSpec(
        "oversized", "test", 150_000_000_000, 100, 100, 100, 128,
        12800, 4096, ["chat"], 99.0, "test", {"hf": "test/model"},
    )
    result = build_plan(
        profile(128, (80, 80), os_name="linux"),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.backend == "vllm"
    argv = service.launch.argv
    assert "--max-num-seqs" in argv
    fraction = float(argv[argv.index("--gpu-memory-utilization") + 1])
    total_vram = sum(
        gpu.total_vram_bytes for gpu in result.profile.gpus
        if gpu.index in service.gpu_indices
    )
    assert 0.10 < fraction <= 0.95
    assert fraction == round(
        min(0.95, max(0.10, service.memory.gpu_bytes / total_vram)), 3
    )


def test_forced_slots_clamp_and_one_is_silent(catalog: list[ModelSpec]) -> None:
    small = build_plan(
        profile(32, (12,)), [
            next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
        ],
        Policy(roles=["chat"], parallel_slots=32),
    )
    service = small.services[0]
    assert service.memory.parallel_slots < 32
    assert any(
        "32" in warning and str(service.memory.parallel_slots) in warning
        and service.name in warning
        for warning in small.warnings
    )

    large = build_plan(
        profile(64, (24,)), [next(
            item for item in catalog if item.id == "qwen2.5-7b-instruct"
        )],
        Policy(roles=["chat"], parallel_slots=1),
    )
    assert large.services[0].memory.parallel_slots == 1
    assert not any("parallel_slots" in warning for warning in large.warnings)


def test_parallel_slot_round_trip_and_old_memory_default(
    tmp_path, catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(64, (24,)), [next(
            item for item in catalog if item.id == "qwen2.5-7b-instruct"
        )],
        Policy(roles=["chat"], parallel_slots=2),
    )
    path = tmp_path / "slots.json"
    save_plan(result, path)
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.policy.parallel_slots == 2
    assert loaded.services[0].memory.parallel_slots == 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["services"][0]["memory"].pop("parallel_slots")
    path.write_text(json.dumps(payload), encoding="utf-8")
    old = load_plan(path)
    assert old is not None
    assert old.services[0].memory.parallel_slots == 1


def test_multi_gpu_services_use_different_cards() -> None:
    result = build_plan(
        profile(64, (24, 24)),
        symmetric_catalog(),
        Policy(roles=["chat", "code"]),
    )
    services = result.services
    assert len(services) == 2
    assert {service.gpu_indices[0] for service in services} == {0, 1}


def test_asymmetric_gpu_placement_uses_fitting_cards() -> None:
    result = build_plan(
        profile(64, (24, 8)),
        placement_catalog(),
        Policy(roles=["chat", "code"]),
    )
    assert len(result.services) == 2
    for service in result.services:
        assert len(service.gpu_indices) == 1
        gpu = next(
            gpu for gpu in result.profile.gpus if gpu.index == service.gpu_indices[0]
        )
        assert service.memory.gpu_bytes <= planner_core._gpu_budget(gpu) + 1
    assert next(item for item in result.services if item.model_id == "large").gpu_indices == [0]
    assert next(item for item in result.services if item.model_id == "small").gpu_indices == [1]


def test_swap_group_reserves_only_largest_member() -> None:
    result = build_plan(
        profile(16, (12,)),
        placement_catalog(),
        Policy(roles=["chat", "code"]),
    )
    assert len(result.services) == 2
    services = [
        replace(service, resident=False) for service in result.services
    ]
    services = planner_core._place_services(
        services, result.profile, result.policy,
        [service.name for service in services], [],
    )
    assert all(service.gpu_indices == [0] for service in services)
    reserved_vram, reserved_ram = planner_core._reserved_memory(
        services, [service.name for service in services]
    )
    assert reserved_vram == max(service.memory.gpu_bytes for service in services)
    assert reserved_ram == max(service.memory.cpu_bytes for service in services)
    assert reserved_vram < sum(service.memory.gpu_bytes for service in services)


def test_llamacpp_layers_are_resolved_against_assigned_card() -> None:
    result = build_plan(
        profile(64, (24, 8)),
        placement_catalog(),
        Policy(roles=["chat", "code"]),
    )
    large = next(service for service in result.services if service.model_id == "large")
    assert large.backend == "llamacpp"
    assert large.n_gpu_layers == 40
    oversized = replace(
        large,
        gpu_indices=[],
        memory=replace(large.memory, gpu_bytes=1 * GIB),
    )
    resolved = planner_core._place_services(
        [oversized], result.profile, result.policy, [], [],
    )[0]
    assert resolved.gpu_indices == [1]
    assert resolved.n_gpu_layers < large.n_gpu_layers
    assert resolved.memory.gpu_bytes <= planner_core._gpu_budget(result.profile.gpus[1]) + 1
    assert resolved.launch.argv[resolved.launch.argv.index("-ngl") + 1] == str(
        resolved.n_gpu_layers
    )


def test_embedding_launch_flags_are_role_aware(catalog: list[ModelSpec]) -> None:
    assert next(item for item in catalog if item.id == "bge-m3").pooling == "cls"
    assert next(item for item in catalog if item.id == "nomic-embed-text-v1.5").pooling == "mean"
    result = build_plan(
        profile(64, (24,)),
        catalog,
        Policy(roles=["chat", "embed"], min_decode_tps=0),
    )
    embed = next(service for service in result.services if service.roles == ["embed"])
    chat = next(service for service in result.services if "chat" in service.roles)
    assert "--embeddings" in embed.launch.argv
    assert embed.launch.argv[embed.launch.argv.index("--pooling") + 1] == "cls"
    assert embed.launch.argv[embed.launch.argv.index("-b") + 1] == str(embed.context)
    assert embed.launch.argv[embed.launch.argv.index("-ub") + 1] == str(embed.context)
    assert not any(
        flag in chat.launch.argv
        for flag in ("--embeddings", "--embedding", "--pooling", "-b", "-ub")
    )


def test_embedding_capability_warnings_and_flags() -> None:
    model = ModelSpec(
        "embed-test", "embed-test", 137_000_000, 12, 12, 12, 64, 768,
        8192, ["embed"], 80.0, "apache", {"hf_gguf": "embed-test.gguf"},
        pooling="mean",
    )
    no_embedding = replace(
        profile(32, (12,)),
        backend_flags={
            "llamacpp": ("--parallel", "-ngl", "--pooling", "-b", "-ub"),
        },
    )
    unsupported = build_plan(
        no_embedding, [model], Policy(roles=["embed"], min_decode_tps=0)
    )
    unsupported_service = unsupported.services[0]
    assert "--embeddings" not in unsupported_service.launch.argv
    assert any("embedding flags are unsupported" in warning for warning in unsupported.warnings)

    no_batch = replace(
        profile(32, (12,)),
        backend_flags={
            "llamacpp": ("--parallel", "-ngl", "--embeddings", "--pooling"),
        },
    )
    limited = build_plan(
        no_batch, [model], Policy(roles=["embed"], min_decode_tps=0)
    )
    limited_service = limited.services[0]
    assert "-b" not in limited_service.launch.argv
    assert "-ub" not in limited_service.launch.argv
    assert any("above 512 tokens may be rejected" in warning for warning in limited.warnings)


def test_embedding_uses_supported_flag_aliases_without_warnings() -> None:
    model = ModelSpec(
        "embed-alias", "embed-test", 137_000_000, 12, 12, 12, 64, 768,
        8192, ["embed"], 80.0, "apache", {"hf_gguf": "embed-alias.gguf"},
        pooling="mean",
    )
    aliases = replace(
        profile(32, (12,)),
        backend_flags={
            "llamacpp": (
                "--embedding",
                "--batch-size",
                "--ubatch-size",
                "--pooling",
                "--parallel",
                "-ngl",
            ),
        },
    )
    result = build_plan(
        aliases, [model], Policy(roles=["embed"], min_decode_tps=0)
    )
    argv = result.services[0].launch.argv
    assert "--embedding" in argv
    assert "--embeddings" not in argv
    assert argv[argv.index("--batch-size") + 1] == "8192"
    assert argv[argv.index("--ubatch-size") + 1] == "8192"
    assert result.warnings == [
        "Model ranking uses unverified catalog quality claims; nmesh eval measures them."
    ]


def test_embedding_warns_when_pooling_flag_is_unsupported() -> None:
    model = ModelSpec(
        "embed-no-pooling-flag", "embed-test", 137_000_000, 12, 12, 12, 64, 768,
        8192, ["embed"], 80.0, "apache", {"hf_gguf": "embed-test.gguf"},
        pooling="mean",
    )
    no_pooling = replace(
        profile(32, (12,)),
        backend_flags={
            "llamacpp": ("--embeddings", "-b", "-ub", "--parallel", "-ngl"),
        },
    )
    result = build_plan(
        no_pooling, [model], Policy(roles=["embed"], min_decode_tps=0)
    )
    assert "--pooling" not in result.services[0].launch.argv
    assert any("pooling metadata is unknown" in warning for warning in result.warnings)


def test_embedding_unknown_pooling_warns() -> None:
    model = ModelSpec(
        "embed-no-pooling", "embed-test", 137_000_000, 12, 12, 12, 64, 768,
        8192, ["embed"], 80.0, "apache", {"hf_gguf": "embed-test.gguf"},
    )
    result = build_plan(
        profile(32, (12,)), [model], Policy(roles=["embed"], min_decode_tps=0)
    )
    assert any("pooling metadata is unknown" in warning for warning in result.warnings)


def test_embedding_batch_flags_remain_at_context_after_slot_assignment(
    catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(64, (24,)),
        catalog,
        Policy(roles=["embed"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.launch.argv[service.launch.argv.index("-b") + 1] == str(service.context)
    assert service.launch.argv[service.launch.argv.index("-ub") + 1] == str(service.context)


def test_embedding_backend_warnings_are_honest() -> None:
    model = ModelSpec(
        "embed-hf", "embed-test", 137_000_000, 12, 12, 12, 64, 768,
        8192, ["embed"], 80.0, "apache", {"hf": "test/embed"},
        pooling="mean",
    )
    vllm = build_plan(
        profile(128, (80, 80), os_name="linux"),
        [model],
        Policy(roles=["embed"], min_decode_tps=0),
    )
    assert any("not verified by nmesh" in warning for warning in vllm.warnings)

    mlx = build_plan(
        profile(
            64,
            (44,),
            os_name="macos",
            unified=True,
            backends={"ollama": None, "llamacpp": None, "vllm": None, "mlx": "installed"},
        ),
        [model],
        Policy(roles=["embed"], min_decode_tps=0),
    )
    assert any("does not provide an embedding endpoint" in warning for warning in mlx.warnings)


def test_context_depth_coverage_warns_without_changing_candidate() -> None:
    model = ModelSpec(
        "depth-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 90.0, "apache", {"hf_gguf": "depth-model"},
    )
    policy = Policy(roles=["chat"], min_decode_tps=0)
    ordinary = build_plan(profile(8), [model], policy)
    service_key = (
        ordinary.services[0].model_id,
        ordinary.services[0].quant,
        ordinary.services[0].backend,
    )
    warned = build_plan(
        profile(8),
        [model],
        policy,
        eval_depth_coverage={service_key: 1024},
    )
    covered = build_plan(
        profile(8),
        [model],
        policy,
        eval_depth_coverage={service_key: 16384},
    )
    missing = build_plan(
        profile(8),
        [model],
        policy,
        eval_depth_coverage={},
    )
    ordinary_service = ordinary.services[0]
    warned_service = warned.services[0]
    assert warned_service == ordinary_service
    assert sum("quality evidence only reaches" in item for item in warned.warnings) == 1
    assert not any("quality evidence only reaches" in item for item in covered.warnings)
    assert not any("quality evidence only reaches" in item for item in missing.warnings)
