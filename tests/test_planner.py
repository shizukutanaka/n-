from __future__ import annotations

import pytest

from nmesh.catalog import ModelSpec, load_catalog
from nmesh.planner import Policy, build_plan, estimate_memory
from nmesh.probe import GPUInfo, HardwareProfile, Tier, classify_tier

GIB = 1024**3


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


def test_memory_regression(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    estimate = estimate_memory(model, "q4_k_m", 2048)
    assert estimate.weight_bytes == pytest.approx(4.62e9, rel=0.01)


def test_cpu_case(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(8), catalog)
    assert result.tier == Tier.T0_CPU
    assert result.services
    assert len(result.swap_group) == 1
    assert result.services[0].n_gpu_layers == 0


def test_low_gpu_partial_offload(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(16, (4,)), catalog, Policy(roles=["chat"]))
    service = result.services[0]
    assert result.tier == Tier.T1_LOW
    assert service.n_gpu_layers is not None
    assert 0 < service.n_gpu_layers < 100


def test_mid_gpu_full_model(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(32, (12,)), catalog, Policy(roles=["chat"]))
    service = result.services[0]
    assert result.tier in {Tier.T2_MID, Tier.T3_HIGH}
    assert service.n_gpu_layers == next(
        item for item in catalog if item.id == service.model_id
    ).n_layers


def test_workstation_resident_roles(catalog: list[ModelSpec]) -> None:
    result = build_plan(profile(64, (24,)), catalog)
    assert result.tier == Tier.T4_WORKSTATION
    assert all(service.resident for service in result.services)
    assert set(result.routing.role_to_service) == {"chat", "code", "embed"}


def test_server_vllm_tensor_parallel(catalog: list[ModelSpec]) -> None:
    result = build_plan(
        profile(128, (80, 80), os_name="linux"),
        catalog,
        Policy(roles=["chat"]),
    )
    assert result.tier == Tier.T5_SERVER
    assert result.services[0].backend == "vllm"
    assert len(result.services[0].gpu_indices) == 2


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
    assert result.services[0].backend == "mlx"


def test_plan_save_load(tmp_path, catalog: list[ModelSpec]) -> None:
    from nmesh.planner import load_plan, save_plan

    result = build_plan(profile(8), catalog)
    path = tmp_path / "plan.json"
    save_plan(result, path)
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.services[0].model_id == result.services[0].model_id
