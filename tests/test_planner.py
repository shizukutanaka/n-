from __future__ import annotations

import json
import os
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


def test_default_roles_degrade_when_model_cannot_cover_embed(
    catalog: list[ModelSpec],
) -> None:
    model = next(
        item for item in catalog if item.id == "qwen2.5-1.5b-instruct"
    )

    result = build_plan(
        profile(8),
        [model],
        Policy(model_ids=(model.id,), min_decode_tps=0),
    )

    assert result.runnable
    assert any("chat" in service.roles for service in result.services)
    assert not any("embed" in service.roles for service in result.services)
    assert any("role embed" in warning for warning in result.warnings)


def test_explicit_roles_remain_strict_when_embed_is_missing(
    catalog: list[ModelSpec],
) -> None:
    model = next(
        item for item in catalog if item.id == "qwen2.5-1.5b-instruct"
    )

    result = build_plan(
        profile(8),
        [model],
        Policy(
            roles=["chat", "embed"],
            roles_explicit=True,
            model_ids=(model.id,),
            min_decode_tps=0,
        ),
    )

    assert result.services
    assert not result.runnable
    assert any("role embed" in warning for warning in result.warnings)


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


def test_estimate_below_floor_is_excluded_even_with_bench_records(
    catalog: list[ModelSpec],
) -> None:
    large = next(item for item in catalog if item.id == "qwen2.5-32b-instruct")
    small = next(item for item in catalog if item.id == "qwen2.5-1.5b-instruct")
    models = [large, small]
    policy = Policy(roles=["chat"], min_decode_tps=8.0)

    without_records = build_plan(
        profile(32), models, policy, bench_records=None,
    )
    with_records = build_plan(
        profile(32), models, policy, bench_records={},
    )

    assert [item.model_id for item in without_records.services] == [
        item.model_id for item in with_records.services
    ]
    assert with_records.services[0].model_id == small.id
    assert large.id not in {item.model_id for item in with_records.services}


def test_measured_unconfirmed_below_floor_is_kept(
    monkeypatch, catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-1.5b-instruct")
    monkeypatch.setattr(planner_core, "_bench_value", lambda *_args: 4.0)

    result = build_plan(
        profile(32),
        [model],
        Policy(roles=["chat"], min_decode_tps=8.0),
        bench_records={},
    )

    assert result.services[0].model_id == model.id
    assert any("needs a second agreeing measurement" in warning for warning in result.warnings)


def test_download_repo_picked_per_backend() -> None:
    model = ModelSpec(
        "m", "test", 1_000_000_000, 24, 16, 4, 128, 2048, 4096,
        ["chat"], 80.0, "apache",
        {"hf": "org/fp16", "hf_gguf": "org/gguf", "hf_mlx": "org/mlx-4bit",
         "ollama": "org/ollama"},
    )
    assert planner_core._download_repo_for("mlx", model) == "org/mlx-4bit"
    assert planner_core._download_repo_for("vllm", model) == "org/fp16"
    assert planner_core._download_repo_for("llamacpp", model) == "org/gguf"
    assert planner_core._download_repo_for("ollama", model) == "org/ollama"

    no_mlx = ModelSpec(
        "m2", "test", 1_000_000_000, 24, 16, 4, 128, 2048, 4096,
        ["chat"], 80.0, "apache", {"hf": "org/fp16"},
    )
    assert planner_core._download_repo_for("mlx", no_mlx) == "org/fp16"


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


def test_embedding_candidates_ignore_decode_threshold_and_speed(monkeypatch) -> None:
    model = ModelSpec(
        "embed-only", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed-only"},
    )
    monkeypatch.setattr(planner_core, "_throughput", lambda *_args: 0.1)
    monkeypatch.setattr(
        planner_core, "_bench_value",
        lambda *_args, **_kwargs: pytest.fail("embed candidates must not read benchmarks"),
    )
    excluded: list[dict[str, str]] = []
    unconfirmed: list[dict[str, str]] = []
    candidates = planner_core._candidate_for(
        model,
        profile(64),
        Policy(roles=["embed"], min_decode_tps=8.0, prefer="speed"),
        {},
        excluded=excluded,
        unconfirmed=unconfirmed,
    )
    assert candidates
    candidate = candidates[0]
    assert candidate.decode_tps == 0.0
    assert candidate.decode_applicable is False
    assert candidate.score == pytest.approx(
        (80.0 - planner_core.QUANT_PENALTY[candidate.quant]) * 0.5
    )
    assert excluded == []
    assert unconfirmed == []


def test_embedding_plan_decode_tps_round_trip(tmp_path) -> None:
    model = ModelSpec(
        "embed-only", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed-only"},
    )
    plan = build_plan(profile(64), [model], Policy(roles=["embed"]))
    assert plan.services[0].decode_tps is None
    path = tmp_path / "embed-plan.json"
    save_plan(plan, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["services"][0]["decode_tps"] is None
    loaded = load_plan(path)
    assert loaded is not None
    assert loaded.services[0].decode_tps is None


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
    # model_ids pins the pool: bigger MoE catalog entries legitimately
    # crowd out the embed role when they consume most of the RAM pool.
    result = build_plan(
        profile(64, (24,)), catalog,
        Policy(model_ids=("phi-4-14b", "qwen3-embedding-8b")),
    )
    assert_memory_fit(result)
    assert result.tier == Tier.T4_WORKSTATION
    assert all(service.resident for service in result.services)
    assert set(result.routing.role_to_service) == {"chat", "code", "embed"}


def test_server_vllm_single_gpu(catalog: list[ModelSpec]) -> None:
    # model_ids pins the model: a 120B MoE legitimately spans both GPUs.
    result = build_plan(
        profile(128, (80, 80), os_name="linux"),
        catalog,
        Policy(roles=["chat"], model_ids=("phi-4-14b",)),
    )
    assert_memory_fit(result)
    assert result.tier == Tier.T5_SERVER
    assert result.services[0].backend == "vllm"
    assert len(result.services[0].gpu_indices) == 1


def test_apple_mlx(catalog: list[ModelSpec]) -> None:
    # model_ids pins the model: on unified memory a heavier MoE entry can
    # legitimately fill the shared RAM pool beyond the GPU-side budget.
    result = build_plan(
        profile(
            64,
            (44,),
            os_name="macos",
            unified=True,
            backends={"ollama": None, "llamacpp": None, "vllm": None, "mlx": "installed"},
        ),
        catalog,
        Policy(roles=["chat"], model_ids=("phi-4-14b",)),
    )
    assert_memory_fit(result)
    assert result.services[0].backend == "mlx"


def _resolution_env_names() -> frozenset[str]:
    return frozenset({
        "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE",
        "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT",
        "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
    })


def test_resolution_env_leak_warns(
    catalog: list[ModelSpec], monkeypatch,
) -> None:
    for name in _resolution_env_names():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/tmp/stale")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/libs")
    result = build_plan(profile(8), [_ollama_only_model()], Policy(roles=["chat"]))
    warns = [w for w in result.warnings if "PYTHONPATH" in w]
    assert len(warns) == 1
    assert "LD_LIBRARY_PATH" in warns[0]


def test_resolution_env_warn_absent_when_clean(
    catalog: list[ModelSpec], monkeypatch,
) -> None:
    for name in list(os.environ):
        if name in _resolution_env_names():
            monkeypatch.delenv(name)
    result = build_plan(profile(8), [_ollama_only_model()], Policy(roles=["chat"]))
    assert not any(
        "module/library resolution" in w for w in result.warnings
    )


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


def _ollama_only_model() -> ModelSpec:
    return ModelSpec(
        "ollama-only", "test", 500_000_000, 24, 14, 2, 64, 896, 4096,
        ["chat"], 90.0, "apache", {"ollama": "test:model"},
    )


def test_ollama_daemon_env_pins_single_parallel_slot(monkeypatch) -> None:
    # Ollama auto-selects up to 4 parallel slots per model; the plan
    # accounts KV memory for one slot, so the daemon env pins it.
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
    result = build_plan(profile(8), [_ollama_only_model()], Policy(roles=["chat"]))
    assert result.services[0].backend == "ollama"
    assert result.services[0].launch.env["OLLAMA_NUM_PARALLEL"] == "1"


def test_ollama_daemon_env_respects_user_parallel(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
    result = build_plan(profile(8), [_ollama_only_model()], Policy(roles=["chat"]))
    assert "OLLAMA_NUM_PARALLEL" not in result.services[0].launch.env
    assert any(
        "OLLAMA_NUM_PARALLEL" in warning for warning in result.warnings
    )


def test_ollama_daemon_env_no_warn_when_user_parallel_matches(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "1")
    result = build_plan(profile(8), [_ollama_only_model()], Policy(roles=["chat"]))
    assert "OLLAMA_NUM_PARALLEL" not in result.services[0].launch.env
    assert not any(
        "OLLAMA_NUM_PARALLEL" in warning for warning in result.warnings
    )


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


def test_kv_layers_scales_kv_cache_for_hybrid_models() -> None:
    hybrid = ModelSpec(
        "hybrid", "nemotron-h", 30_000_000_000, 52, 32, 2, 128, 2688,
        262144, ["chat"], 76.0, "nvidia-open-model-license",
        {"hf_gguf": "test/repo"}, kv_layers=6,
    )
    dense = ModelSpec(
        "dense", "nemotron-h", 30_000_000_000, 52, 32, 2, 128, 2688,
        262144, ["chat"], 76.0, "nvidia-open-model-license",
        {"hf_gguf": "test/repo"},
    )
    est_h = estimate_memory(hybrid, "q4_k_m", 8192)
    est_d = estimate_memory(dense, "q4_k_m", 8192)
    assert est_h.kv_bytes_per_tok == pytest.approx(est_d.kv_bytes_per_tok * 6 / 52)
    assert est_h.weight_bytes == est_d.weight_bytes


def test_sliding_window_bounds_swa_layer_cache() -> None:
    model = ModelSpec(
        "swa", "test", 9_000_000_000, 42, 16, 8, 256, 3584,
        8192, ["chat"], 90.0, "apache", {"hf_gguf": "test/repo"},
        sliding_window=4096, sliding_window_pattern=2,
    )
    dense = ModelSpec(
        "dense", "test", 9_000_000_000, 42, 16, 8, 256, 3584,
        8192, ["chat"], 90.0, "apache", {"hf_gguf": "test/repo"},
    )
    estimate = estimate_memory(model, "q4_k_m", 8192)
    baseline = estimate_memory(dense, "q4_k_m", 8192)
    # 21 full-attention layers at 8192 cells + 21 sliding layers capped at
    # min(8192, 4096 + ubatch 512) cells, matching llama.cpp's iSWA sizing.
    expected = 2 * 8 * 256 * 2 * (21 * 8192 + 21 * (4096 + 512))
    assert estimate.kv_cache_bytes == pytest.approx(expected)
    assert estimate.kv_cache_bytes < baseline.kv_cache_bytes


def test_sliding_window_within_window_keeps_full_estimate() -> None:
    model = ModelSpec(
        "swa", "test", 9_000_000_000, 42, 16, 8, 256, 3584,
        8192, ["chat"], 90.0, "apache", {"hf_gguf": "test/repo"},
        sliding_window=4096, sliding_window_pattern=2,
    )
    estimate = estimate_memory(model, "q4_k_m", 2048)
    expected = 2 * 42 * 8 * 256 * 2 * 2048
    assert estimate.kv_cache_bytes == pytest.approx(expected)


def test_sliding_window_without_pattern_slides_every_layer() -> None:
    model = ModelSpec(
        "swa", "test", 7_000_000_000, 32, 32, 8, 128, 4096,
        32768, ["chat"], 90.0, "apache", {"hf_gguf": "test/repo"},
        sliding_window=2048,
    )
    estimate = estimate_memory(model, "q4_k_m", 8192, parallel_slots=2)
    expected = 2 * 32 * 8 * 128 * 2 * (2048 + 512) * 2
    assert estimate.kv_cache_bytes == pytest.approx(expected)


def test_catalog_sliding_window_models(catalog: list[ModelSpec]) -> None:
    fields = {
        model.id: (model.sliding_window, model.sliding_window_pattern)
        for model in catalog
        if model.sliding_window
    }
    assert fields == {
        "gemma2-2b": (4096, 2),
        "gemma2-9b": (4096, 2),
        "gpt-oss-20b": (128, 2),
        "gpt-oss-120b": (128, 2),
    }


def _draft_gguf(
    *,
    layers: int = 24,
    kv_heads: int = 8,
    key_length: int = 128,
    sliding_window: int = 0,
    pattern: tuple[bool, ...] = (),
) -> bytes:
    import struct

    from tests.test_artifact import _kv_bool_array, _kv_string, _kv_u32

    entries = [
        _kv_string("general.architecture", "draftarch"),
        _kv_u32("draftarch.block_count", layers),
        _kv_u32("draftarch.attention.head_count_kv", kv_heads),
        _kv_u32("draftarch.attention.key_length", key_length),
    ]
    if sliding_window:
        entries.append(
            _kv_u32("draftarch.attention.sliding_window", sliding_window)
        )
    if pattern:
        entries.append(
            _kv_bool_array("draftarch.attention.sliding_window_pattern", pattern)
        )
    return b"".join((
        b"GGUF",
        struct.pack("<IQQ", 3, 0, len(entries)),
        *entries,
    ))


def test_draft_kv_per_tok_reads_gguf_attention_layout(tmp_path: Path) -> None:
    path = tmp_path / "draft.gguf"
    path.write_bytes(_draft_gguf())
    rate = planner_core._draft_kv_per_tok(path, "f16", 8192)
    assert rate == pytest.approx(2 * 8 * 128 * 2 * 24)


def test_draft_kv_per_tok_applies_gguf_sliding_window(tmp_path: Path) -> None:
    path = tmp_path / "draft.gguf"
    path.write_bytes(_draft_gguf(
        sliding_window=128,
        pattern=tuple(index % 2 == 0 for index in range(24)),
    ))
    rate = planner_core._draft_kv_per_tok(path, "f16", 8192)
    # 12 sliding layers hold window + ubatch cells; 12 full layers hold ctx.
    expected = 2 * 8 * 128 * 2 * (12 + 12 * ((128 + 512) / 8192))
    assert rate == pytest.approx(expected)


def test_draft_kv_per_tok_requires_layout_metadata(tmp_path: Path) -> None:
    import struct

    path = tmp_path / "draft.gguf"
    path.write_bytes(b"not a gguf")
    assert planner_core._draft_kv_per_tok(path, "f16", 8192) is None
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    assert planner_core._draft_kv_per_tok(path, "f16", 8192) is None


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


def test_sleep_idle_seconds_is_launched_when_supported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl", "--sleep-idle-seconds")},
    )
    result = build_plan(
        machine, [model],
        Policy(roles=["chat"], sleep_idle_seconds=300),
    )
    argv = result.services[0].launch.argv
    assert argv[argv.index("--sleep-idle-seconds") + 1] == "300"


def test_sleep_idle_seconds_warns_when_unsupported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl")},
    )
    result = build_plan(
        machine, [model],
        Policy(roles=["chat"], sleep_idle_seconds=300),
    )
    assert not any(
        "--sleep-idle-seconds" in flag for flag in result.services[0].launch.argv
    )
    assert any("sleep-idle" in warning for warning in result.warnings)


def test_sleep_idle_seconds_off_by_default(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    result = build_plan(
        profile(64, (24,)), [model], Policy(roles=["chat"]),
    )
    assert not any(
        "--sleep-idle-seconds" in flag for flag in result.services[0].launch.argv
    )


def test_cache_reuse_is_launched_when_supported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl", "--cache-reuse")},
    )
    result = build_plan(
        machine, [model], Policy(roles=["chat"], cache_reuse=256),
    )
    argv = result.services[0].launch.argv
    assert argv[argv.index("--cache-reuse") + 1] == "256"


def test_cache_reuse_warns_when_unsupported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl")},
    )
    result = build_plan(
        machine, [model], Policy(roles=["chat"], cache_reuse=256),
    )
    assert not any(
        "--cache-reuse" in flag for flag in result.services[0].launch.argv
    )
    assert any("cache-reuse" in warning for warning in result.warnings)


def test_context_shift_is_launched_and_warns_of_dropped_tokens(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl", "--context-shift")},
    )
    result = build_plan(
        machine, [model], Policy(roles=["chat"], context_shift=True),
    )
    assert "--context-shift" in result.services[0].launch.argv
    assert any("silently dropped" in warning for warning in result.warnings)


def test_context_shift_warns_when_unsupported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "qwen2.5-7b-instruct")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl")},
    )
    result = build_plan(
        machine, [model], Policy(roles=["chat"], context_shift=True),
    )
    assert "--context-shift" not in result.services[0].launch.argv
    assert any("context-shift" in warning for warning in result.warnings)


def test_rerank_gets_dedicated_service_with_reranking_flag(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "bge-m3")
    result = build_plan(
        profile(64, (24,)), [model], Policy(roles=["embed", "rerank"]),
    )
    rerank = next(
        service for service in result.services if service.name == "rerank"
    )
    embed = next(
        service for service in result.services if service.name == "embed"
    )
    # llama.cpp serves rerank OR embeddings per instance (single pooling
    # mode); --reranking on the embed service zeroed embeddings (b11037).
    assert "--reranking" in rerank.launch.argv
    assert "--embeddings" not in rerank.launch.argv
    assert "--reranking" not in embed.launch.argv
    assert "--embeddings" in embed.launch.argv
    assert result.routing.role_to_service["rerank"] == "rerank"


def test_rerank_prefers_dedicated_reranker_model(catalog: list[ModelSpec]) -> None:
    # A roles=["rerank"] model should beat dual-use embed models for the
    # rerank role; dual-use models stay available as fallback.
    dedicated = next(
        item for item in catalog if item.id == "qwen3-reranker-0.6b"
    )
    result = build_plan(
        profile(64, (24,)), [dedicated], Policy(roles=["rerank"]),
    )
    rerank = next(
        service for service in result.services if service.name == "rerank"
    )
    assert rerank.model_id == "qwen3-reranker-0.6b"
    assert "--reranking" in rerank.launch.argv


def test_rerank_service_omitted_when_flag_unsupported(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "bge-m3")
    machine = replace(
        profile(64, (24,)),
        backend_flags={"llamacpp": ("--parallel", "-ngl")},
    )
    result = build_plan(
        machine, [model], Policy(roles=["embed", "rerank"]),
    )
    rerank = next(
        service for service in result.services if service.name == "rerank"
    )
    assert "--reranking" not in rerank.launch.argv
    assert any("rerank" in w for w in result.warnings)


def test_embed_and_rerank_share_model_download_once(
    catalog: list[ModelSpec],
) -> None:
    model = next(item for item in catalog if item.id == "bge-m3")
    result = build_plan(
        profile(64, (24,)), [model], Policy(roles=["embed", "rerank"]),
    )
    embed = next(s for s in result.services if s.name == "embed")
    assert result.total_download_bytes <= int(embed.memory.disk_needed) + 1


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
    result = build_plan(profile(32), [model], Policy(roles=["embed"]))
    service = result.services[0]
    assert service.model_id == model.id
    assert service.context == min(model.max_context, 8192)
    assert service.memory.kv_cache_bytes == 0
    assert service.memory.compute_overhead > 0.06 * service.memory.weight_bytes + 320 * 1024**2


def test_embedding_cap_rechecks_backend_after_layer_change(monkeypatch) -> None:
    model = ModelSpec(
        "embed-recheck", "test", 500_000_000, 8, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache",
        {"hf_gguf": "org/embed-recheck", "ollama": "org/embed-recheck"},
    )
    machine = profile(32, (24,))
    solve_calls: list[int] = []

    def solve(memory, n_layers):
        solve_calls.append(n_layers if len(solve_calls) == 0 else 0)
        return n_layers if len(solve_calls) == 1 else 0

    monkeypatch.setattr(planner_core, "solve_gpu_layers", solve)
    monkeypatch.setattr(
        planner_core,
        "_backend",
        lambda _profile, _model, layers: (
            ("llamacpp", True) if layers else ("ollama", True)
        ),
    )
    caps = {
        ("embed-recheck", quant, "llamacpp"): 2048
        for quant in BPW
    }
    candidates = planner_core._candidate_for(
        model, machine, Policy(roles=["embed"]), None,
        embed_input_caps=caps,
    )
    assert solve_calls[:2] == [model.n_layers, 0]
    assert candidates[0].backend == "ollama"


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
    assert not any("capacity-forced tradeoff" in warning for warning in result.warnings)


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
    # Tiny GPUs: the KV pool alone exceeds the combined VRAM budgets,
    # forcing the pure CPU fallback this test exercises.
    model = ModelSpec(
        "cpu-fallback-llamacpp", "test", 60_000_000_000, 80, 80, 100, 128,
        12800, 4096, ["chat"], 99.0, "test", {"hf_gguf": "test.gguf"},
    )
    result = build_plan(
        profile(128, (4, 4)),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.backend == "llamacpp"
    assert service.n_gpu_layers == 0
    assert service.gpu_indices == []
    assert service.tensor_split == ()
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
    models = [
        item for item in catalog if item.id in {"bge-m3", "qwen3-1.7b"}
    ]
    result = build_plan(
        profile(64, (24,)),
        models,
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


def test_measured_selection_replaces_the_unverified_prior_warning() -> None:
    from nmesh.eval.cache import EvalSummary

    model = ModelSpec(
        "measured-chat", "test", 500_000_000, 24, 14, 2, 64, 896,
        4096, ["chat"], 90.0, "test", {"hf_gguf": "test.gguf"},
    )
    unmeasured = build_plan(
        profile(32), [model], Policy(roles=["chat"], min_decode_tps=0),
    )
    prior_warning = i18n.t("warn.quality_prior", "en")
    assert prior_warning in unmeasured.warnings

    service = unmeasured.services[0]
    key = (
        service.model_id.casefold(),
        service.quant.casefold(),
        service.backend.casefold(),
    )
    measured = build_plan(
        profile(32), [model], Policy(roles=["chat"], min_decode_tps=0),
        eval_cache={key: EvalSummary(0.875, 14, 16, {}, "core", "v2:test", 0, False)},
    )
    assert measured.services[0].model_id == service.model_id
    assert prior_warning not in measured.warnings
    assert any(
        "measured at 14/16 on the core suite" in warning
        for warning in measured.warnings
    )


def test_partly_measured_services_keep_the_prior_warning() -> None:
    from nmesh.eval.cache import EvalSummary

    models = [
        ModelSpec(
            "measured-chat", "test", 500_000_000, 24, 14, 2, 64, 896,
            4096, ["chat"], 90.0, "test", {"hf_gguf": "chat.gguf"},
        ),
        ModelSpec(
            "unmeasured-code", "test", 500_000_000, 24, 14, 2, 64, 896,
            4096, ["code"], 90.0, "test", {"hf_gguf": "code.gguf"},
        ),
    ]
    baseline = build_plan(
        profile(32), models, Policy(roles=["chat", "code"], min_decode_tps=0),
    )
    chat = next(item for item in baseline.services if "chat" in item.roles)
    plan = build_plan(
        profile(32), models, Policy(roles=["chat", "code"], min_decode_tps=0),
        eval_cache={
            (
                chat.model_id.casefold(),
                chat.quant.casefold(),
                chat.backend.casefold(),
            ): EvalSummary(0.875, 14, 16, {}, "core", "v2:test", 0, False),
        },
    )
    assert i18n.t("warn.quality_prior", "en") in plan.warnings


def test_embedding_uses_supported_flag_aliases_without_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _resolution_env_names():
        monkeypatch.delenv(name, raising=False)
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
    broken = build_plan(
        profile(8),
        [model],
        policy,
        eval_depth_coverage={service_key: 1024},
        eval_depth_lost={service_key: 1024},
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
    assert broken.services[0] == ordinary_service
    assert any("measured as broken" in item for item in broken.warnings)
    assert not any("quality evidence only reaches" in item for item in broken.warnings)


def test_multi_gpu_services_pin_visibility_env() -> None:
    result = build_plan(
        profile(64, (24, 24)),
        symmetric_catalog(),
        Policy(roles=["chat", "code"]),
    )
    pinned = {
        service.name: service.launch.env.get("CUDA_VISIBLE_DEVICES")
        for service in result.services
    }
    assert sorted(pinned.values()) == ["0", "1"]
    for service in result.services:
        expected = ",".join(str(index) for index in service.gpu_indices)
        assert service.launch.env["CUDA_VISIBLE_DEVICES"] == expected
    assert any("pinned to GPU(s)" in warning for warning in result.warnings)


def test_single_gpu_does_not_pin() -> None:
    result = build_plan(
        profile(64, (24,)),
        placement_catalog()[:1],
        Policy(roles=["chat"]),
    )
    assert "CUDA_VISIBLE_DEVICES" not in result.services[0].launch.env


def test_full_gpu_spread_does_not_pin() -> None:
    big = ModelSpec(
        "huge", "huge", 60_000_000_000, 80, 40, 10, 128, 4096, 8192,
        ["chat"], 90.0, "apache", {"hf_gguf": "huge.gguf"},
    )
    result = build_plan(
        profile(8, (24, 24)),
        [big],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert sorted(service.gpu_indices) == [0, 1]
    assert "CUDA_VISIBLE_DEVICES" not in service.launch.env


def test_download_budget_warning_reports_actual_totals(
    catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(64, (96,)),
        catalog,
        Policy(roles=["chat"], min_decode_tps=0, allow_download_gb=0.001),
    )
    warnings = [
        warning for warning in result.warnings if "download budget" in warning
    ]
    assert warnings, result.warnings
    assert "GiB" in warnings[0] and "0.0" in warnings[0]


def test_download_budget_warning_silent_within_limit(
    catalog: list[ModelSpec],
) -> None:
    result = build_plan(
        profile(64, (96,)),
        catalog,
        Policy(roles=["chat"], min_decode_tps=0, allow_download_gb=4096.0),
    )
    assert not any("download budget" in w for w in result.warnings)


def moe_model() -> ModelSpec:
    # gpt-oss-20b-shaped MoE: 24 layers, experts ~19.1B of 20.9B params.
    return ModelSpec(
        "moe-mix", "moe-mix", 20_900_000_000, 24, 32, 8, 128, 2880, 131072,
        ["chat"], 90.0, "apache",
        {"hf_gguf": "moe-mix.gguf", "hf": "moe-mix"},
        vocab_size=201048,
        active_params=3_600_000_000,
        moe_expert_params=19_110_297_600,
        n_moe_layers=24,
    )


def test_estimate_memory_carries_moe_expert_sizing() -> None:
    estimate = estimate_memory(moe_model(), "q4_k_m", 8192)
    assert estimate.moe_layers == 24
    assert estimate.moe_expert_bytes_per_layer == pytest.approx(
        19_110_297_600 / 24 * estimate.weight_bytes / 20_900_000_000
    )
    assert estimate.n_cpu_moe == 0
    dense = estimate_memory(
        ModelSpec(
            "dense", "dense", 8_000_000_000, 36, 32, 8, 128, 4096, 8192,
            ["chat"], 80.0, "apache", {"hf_gguf": "dense.gguf"},
        ),
        "q4_k_m", 8192,
    )
    assert dense.moe_layers == 0
    assert dense.moe_expert_bytes_per_layer == 0.0


def test_solve_moe_cpu_layers_bounds() -> None:
    estimate = estimate_memory(moe_model(), "q4_k_m", 8192)
    assert planner_core.solve_moe_cpu_layers(
        replace(estimate, vram_budget=estimate.weight_bytes * 2)
    ) == 0
    assert planner_core.solve_moe_cpu_layers(estimate) is None
    estimate = replace(estimate, vram_budget=estimate.weight_bytes * 0.55)
    solved = planner_core.solve_moe_cpu_layers(estimate)
    assert solved is not None and 0 < solved < estimate.moe_layers
    adjusted = replace(estimate, n_cpu_moe=solved)
    gpu_bytes, cpu_bytes = planner_core.split_memory(
        adjusted, 24, 24
    )
    assert gpu_bytes <= estimate.vram_budget + 1
    assert cpu_bytes == pytest.approx(
        estimate.moe_expert_bytes_per_layer * solved
    )
    estimate = replace(estimate, vram_budget=1024)
    assert planner_core.solve_moe_cpu_layers(estimate) is None


def test_moe_offload_candidate_keeps_all_layers_on_gpu() -> None:
    result = build_plan(
        profile(64, (8,)),
        [moe_model()],
        Policy(roles=["chat"], min_decode_tps=0, model_ids=("moe-mix",)),
    )
    service = result.services[0]
    assert service.backend == "llamacpp"
    assert service.n_cpu_moe > 0
    assert service.n_gpu_layers == service.memory.n_gpu_layers == 24
    argv = service.launch.argv
    assert "--n-cpu-moe" in argv
    assert argv[argv.index("--n-cpu-moe") + 1] == str(service.n_cpu_moe)
    assert service.memory.gpu_bytes <= service.memory.vram_budget + 1
    assert service.memory.cpu_bytes <= service.memory.ram_budget + 1
    assert any("--n-cpu-moe" in w for w in result.warnings)
    normal = estimate_memory(moe_model(), service.quant, service.context)
    partial_layers = planner_core.solve_gpu_layers(normal, 24)
    assert partial_layers < 24


def test_moe_offload_variant_skipped_without_flag_support() -> None:
    base = profile(64, (8,))
    probed = replace(
        base, backend_flags={"llamacpp": ("--ctx-size", "--parallel")}
    )
    result = build_plan(
        probed,
        [moe_model()],
        Policy(roles=["chat"], min_decode_tps=0, model_ids=("moe-mix",)),
    )
    service = result.services[0]
    assert service.n_cpu_moe == 0
    assert "--n-cpu-moe" not in service.launch.argv


def test_moe_offload_variant_skipped_without_gpu() -> None:
    result = build_plan(
        profile(64),
        [moe_model()],
        Policy(roles=["chat"], min_decode_tps=0, model_ids=("moe-mix",)),
    )
    assert result.services[0].n_cpu_moe == 0


def test_moe_offload_variant_unused_when_model_fits() -> None:
    result = build_plan(
        profile(64, (32,)),
        [moe_model()],
        Policy(roles=["chat"], min_decode_tps=0, model_ids=("moe-mix",)),
    )
    service = result.services[0]
    assert service.n_cpu_moe == 0
    assert "--n-cpu-moe" not in service.launch.argv


def test_moe_launch_emits_flag_or_warns() -> None:
    model = moe_model()
    launched = planner_core._launch(
        "llamacpp", model, "q4_k_m", 8192, 18010, 24, 1, n_cpu_moe=4,
    )
    assert "--n-cpu-moe" in launched.argv
    assert launched.argv[launched.argv.index("--n-cpu-moe") + 1] == "4"
    warnings: list[str] = []
    launched = planner_core._launch(
        "llamacpp", model, "q4_k_m", 8192, 18010, 24, 1,
        backend_flags=("--ctx-size",), warnings=warnings, n_cpu_moe=4,
    )
    assert "--n-cpu-moe" not in launched.argv
    assert any("--n-cpu-moe" in warning for warning in warnings)


def test_benchmark_key_distinguishes_moe_offload() -> None:
    plain = benchmark_key("moe-mix", "q4_k_m", "llamacpp", "gpu", 24)
    assert "|moe" not in plain
    offloaded = benchmark_key(
        "moe-mix", "q4_k_m", "llamacpp", "gpu", 24, n_cpu_moe=10
    )
    assert offloaded == f"{plain}|moe10"


def test_moe_throughput_scales_with_offload() -> None:
    memory = estimate_memory(moe_model(), "q4_k_m", 8192)
    hw = profile(64, (8,))
    full = planner_core._throughput(moe_model(), memory, 24, hw)
    moe_memory = replace(
        memory, n_cpu_moe=12,
        gpu_bytes=memory.vram_budget * 0.9,
        cpu_bytes=memory.moe_expert_bytes_per_layer * 12,
    )
    moe = planner_core._throughput(moe_model(), moe_memory, 24, hw)
    assert moe < full
    partial = replace(memory, n_cpu_moe=0)
    layers = planner_core.solve_gpu_layers(
        replace(partial, vram_budget=hw.gpus[0].total_vram_bytes * 0.92), 24
    )
    assert 0 < layers < 24
    assert planner_core._throughput(
        moe_model(), partial, layers, hw
    ) < moe


def test_moe_plan_round_trip_preserves_offload(tmp_path) -> None:
    result = build_plan(
        profile(64, (8,)),
        [moe_model()],
        Policy(roles=["chat"], min_decode_tps=0, model_ids=("moe-mix",)),
    )
    assert result.services[0].n_cpu_moe > 0
    path = tmp_path / "moe-plan.json"
    save_plan(result, path)
    loaded = load_plan(path)
    service = loaded.services[0]
    original = result.services[0]
    assert service.n_cpu_moe == original.n_cpu_moe
    assert service.memory.n_cpu_moe == original.n_cpu_moe
    assert service.memory.moe_layers == 24
    assert service.launch.argv == original.launch.argv


def test_split_ratios_helper() -> None:
    assert planner_core._split_ratios((24.0, 12.0)) == (2, 1)
    assert planner_core._split_ratios((24.0, 24.0)) == (1, 1)
    assert planner_core._split_ratios((23.9, 16.1)) == (3, 2)
    assert planner_core._split_ratios((24.0, 0.0)) == (1, 1)
    assert planner_core._split_ratios((30.0, 20.0, 10.0)) == (3, 2, 1)


def test_tensor_split_proportional_to_gpu_budgets() -> None:
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
    assert service.tensor_split == (3, 1)
    index = service.launch.argv.index("--tensor-split")
    assert service.launch.argv[index + 1] == "3,1"
    warning = i18n.t(
        "warn.tensor_split_proportional", "en",
        service=service.name, split="3,1",
    )
    assert warning in result.warnings


def test_tensor_split_uniform_when_budgets_equal(tmp_path) -> None:
    model = ModelSpec(
        "oversized-llamacpp", "test", 40_000_000_000, 80, 32, 8, 128,
        4096, 128, ["chat"], 99.0, "test", {"hf_gguf": "test.gguf"},
    )
    result = build_plan(
        profile(128, (16, 16)),
        [model],
        Policy(roles=["chat"], min_decode_tps=0),
    )
    service = result.services[0]
    assert service.tensor_split == (1, 1)
    index = service.launch.argv.index("--tensor-split")
    assert service.launch.argv[index + 1] == "1,1"
    assert not any("usable VRAM" in warning for warning in result.warnings)
    path = tmp_path / "plan.json"
    save_plan(result, path)
    loaded = load_plan(path)
    assert loaded.services[0].tensor_split == (1, 1)
    assert loaded.services[0].launch.argv == service.launch.argv


def test_benchmark_key_distinguishes_tensor_split() -> None:
    base = benchmark_key(
        "m", "q4_k_m", "llamacpp", "gpu0", 40, tensor_split=(1, 1)
    )
    split = benchmark_key(
        "m", "q4_k_m", "llamacpp", "gpu0", 40, tensor_split=(2, 1)
    )
    assert split.endswith("|ts2-1")
    assert base != split


def test_vllm_sleep_mode_requires_swap_membership_and_0_9(
    catalog: list[ModelSpec],
) -> None:
    plan = build_plan(profile(64, (24,)), catalog, Policy(roles=["chat"]))
    service = replace(plan.services[0], backend="vllm")
    vllm_profile = replace(
        plan.profile,
        available_backends={
            **plan.profile.available_backends, "vllm": "vllm version 0.9.2"
        },
    )
    warnings: list[str] = []
    (member,) = planner_core._enable_vllm_sleep_mode(
        [service], vllm_profile, [service.name], warnings, "en"
    )
    assert member.sleep_mode
    assert member.launch.argv[-1] == "--enable-sleep-mode"
    assert member.launch.env["VLLM_SERVER_DEV_MODE"] == "1"
    assert not warnings
    # Resident members keep the dev-mode endpoints off their sockets.
    (resident,) = planner_core._enable_vllm_sleep_mode(
        [service], vllm_profile, [], [], "en"
    )
    assert not resident.sleep_mode
    assert "--enable-sleep-mode" not in resident.launch.argv
    # vLLM < 0.9 keeps the kill-and-restart swap path and warns.
    old = replace(
        vllm_profile,
        available_backends={
            **vllm_profile.available_backends, "vllm": "0.8.5"
        },
    )
    old_warnings: list[str] = []
    (unsupported,) = planner_core._enable_vllm_sleep_mode(
        [service], old, [service.name], old_warnings, "en"
    )
    assert not unsupported.sleep_mode
    assert "--enable-sleep-mode" not in unsupported.launch.argv
    assert old_warnings and "0.8.5" in old_warnings[0]
    # An unparseable version string is also treated as unsupported.
    unknown = replace(
        vllm_profile,
        available_backends={
            **vllm_profile.available_backends, "vllm": "test"
        },
    )
    unknown_warnings: list[str] = []
    (guarded,) = planner_core._enable_vllm_sleep_mode(
        [service], unknown, [service.name], unknown_warnings, "en"
    )
    assert not guarded.sleep_mode
    assert unknown_warnings


def test_sleep_mode_round_trips_plan(tmp_path, catalog: list[ModelSpec]) -> None:
    plan = build_plan(profile(64, (24,)), catalog, Policy(roles=["chat"]))
    path = tmp_path / "plan.json"
    parked = replace(plan, services=[replace(plan.services[0], sleep_mode=True)])
    save_plan(parked, path)
    loaded = load_plan(path)
    assert loaded is not None and loaded.services[0].sleep_mode
    # A False sleep_mode is omitted from saved plans, same as spec defaults.
    plain = replace(
        plan, services=[replace(plan.services[0], sleep_mode=False)]
    )
    save_plan(plain, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "sleep_mode" not in payload["services"][0]
    loaded = load_plan(path)
    assert loaded is not None and loaded.services[0].sleep_mode is False


def test_lfm2_24b_a2b_catalog_moe_anatomy(catalog: list[ModelSpec]) -> None:
    model = next(item for item in catalog if item.id == "lfm2-24b-a2b")
    assert model.active_params == 2_300_000_000
    # 10 of 40 layers are full_attention (conv backbone otherwise), and
    # MoE runs on the last 38 (num_dense_layers=2).
    assert (model.kv_layers, model.n_moe_layers) == (10, 38)
    estimate = estimate_memory(model, "q4_k_m", 8192)
    # 64 routed experts x 3 tensors x 2048x1536 params on 38 MoE layers
    # (LiquidAI/LFM2-24B-A2B config.json).
    assert estimate.moe_expert_bytes_per_layer == pytest.approx(
        22_951_231_488 / 38 * estimate.weight_bytes / 24_000_000_000
    )
