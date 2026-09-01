from __future__ import annotations

import json
from dataclasses import replace

import pytest

import nmesh.planner.core as planner_core
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.planner import Policy, build_plan, estimate_memory, free_budgets, load_plan, save_plan
from nmesh.probe import GPUInfo, HardwareProfile, Tier, classify_tier

GIB = 1024**3


def assert_memory_fit(result) -> None:
    for service in result.services:
        assert service.memory.gpu_bytes <= service.memory.vram_budget + 1
        assert service.memory.cpu_bytes <= service.memory.ram_budget + 1


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
    assert estimate.weight_bytes == pytest.approx(4.62e9, rel=0.01)


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


def test_oversized_model_uses_tensor_parallel() -> None:
    model = ModelSpec("oversized", "test", 150_000_000_000, 100, 100, 100, 128,
                      12800, 4096, ["chat"], 99.0, "test", {"hf": "test/model"})
    result = build_plan(profile(128, (80, 80), os_name="linux"),
                        [model], Policy(roles=["chat"], min_decode_tps=0))
    service = result.services[0]
    assert service.backend == "vllm"
    assert service.gpu_indices == [0, 1]
    assert "--tensor-parallel-size" in service.launch.argv


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

    result = build_plan(profile(8), catalog)
    path = tmp_path / "plan.json"
    save_plan(result, path)
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded == result
    assert isinstance(loaded.services[0].memory.n_gpu_layers, int)


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


def test_llamacpp_caps_without_parallel_clamp_slots_and_context(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    no_parallel = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": frozenset({"-ngl", "--tensor-split"})},
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
    largest = max(service.memory.gpu_bytes for service in services)
    assert largest == max(service.memory.gpu_bytes for service in services)


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
