import json
from dataclasses import asdict
from pathlib import Path

from nmesh import cli
from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan
from nmesh.planner.core import _candidate_for, _gpu_budget, _speed_saturation_warning
from nmesh.probe import profile_from_dict

PROFILE_DIR = Path(__file__).parents[1] / "profiles"
PROFILE_FILES = sorted(PROFILE_DIR.glob("*.json"))


def _profile(path: Path):
    return profile_from_dict(json.loads(path.read_text(encoding="utf-8")))


def test_bundled_profiles_round_trip_and_gpu_budgets() -> None:
    catalog = load_catalog()
    layers = {model.id: model.n_layers for model in catalog}
    for path in PROFILE_FILES:
        profile = _profile(path)
        assert profile_from_dict(asdict(profile)) == profile
        plan = build_plan(profile, catalog, Policy(roles=["chat", "code", "embed"]))
        usage = {gpu.index: 0.0 for gpu in profile.gpus}
        swap_usage = {gpu.index: 0.0 for gpu in profile.gpus}
        swap_group = set(plan.swap_group)
        for service in plan.services:
            known_devices = profile.backend_gpu_devices.get(service.backend)
            if (
                known_devices == ()
                and service.backend not in {"ollama", "vllm", "mlx"}
            ):
                assert service.n_gpu_layers == 0
            if service.backend in {"ollama", "vllm", "mlx"}:
                assert service.n_gpu_layers is None
            if (
                known_devices
                and service.gpu_indices
                and service.memory.gpu_bytes > 0
                and any(device.startswith("CUDA") for device in known_devices)
            ):
                assert service.n_gpu_layers is not None
                assert 0 < service.n_gpu_layers <= layers[service.model_id]
            if not service.gpu_indices or service.memory.gpu_bytes <= 0:
                continue
            # llamacpp services may carry a proportional --tensor-split;
            # anything else divides its GPU bytes evenly across its cards.
            ratios = (
                list(service.tensor_split)
                if len(service.tensor_split) == len(service.gpu_indices)
                else [1] * len(service.gpu_indices)
            )
            ratio_total = sum(ratios)
            shares = [
                service.memory.gpu_bytes * ratio / ratio_total
                for ratio in ratios
            ]
            target = swap_usage if service.name in swap_group else usage
            for share, index in zip(shares, service.gpu_indices):
                target[index] = max(target[index], share) if (
                    service.name in swap_group
                ) else target[index] + share
        for gpu in profile.gpus:
            assert usage[gpu.index] + swap_usage[gpu.index] <= _gpu_budget(gpu) + 1


def test_asymmetric_profile_places_largest_service_on_large_card() -> None:
    profile = _profile(PROFILE_DIR / "t5-dual-asymmetric.json")
    plan = build_plan(profile, load_catalog(), Policy(roles=["chat", "code", "embed"]))
    largest = max(plan.services, key=lambda service: service.memory.gpu_bytes)
    assert 0 in largest.gpu_indices
    assert 1 not in largest.gpu_indices or len(largest.gpu_indices) > 1


def test_4090_plans_larger_model_than_gtx1650() -> None:
    catalog = load_catalog()
    policy = Policy(roles=["chat"])
    weak = build_plan(_profile(PROFILE_DIR / "t1-gtx1650-4gb.json"), catalog, policy).services[0]
    strong = build_plan(_profile(PROFILE_DIR / "t3-rtx4090-24gb.json"), catalog, policy).services[0]
    assert strong.model_id != weak.model_id
    assert strong.memory.weight_bytes > weak.memory.weight_bytes


def test_profile_plan_is_simulated_and_does_not_save(tmp_path, monkeypatch, capsys) -> None:
    from nmesh.planner import core as planner_core

    plan_path = tmp_path / "plan.json"
    monkeypatch.setattr(planner_core, "PLAN_PATH", plan_path)
    profile_path = PROFILE_DIR / "t2-rtx3060-12gb.json"
    assert cli.main(["plan", "--profile", str(profile_path), "--roles", "chat", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["simulated"] is True
    assert not plan_path.exists()


def test_simulated_plan_ignores_this_machine_measurements(monkeypatch, capsys) -> None:
    """Bench keys omit the CPU, so a CPU profile would inherit local numbers."""
    from nmesh.bench import benchmark_key

    def fake_cache() -> dict[str, float]:
        return {
            benchmark_key(model_id, quant, "llamacpp", "cpu", 0): 999.0
            for model_id in ("qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct")
            for quant in ("q4_k_m", "q5_k_m", "q6_k", "q8_0")
        }

    monkeypatch.setattr(cli, "load_cache", fake_cache)
    profile_path = PROFILE_DIR / "t0-cpu-32gb.json"
    assert cli.main(["plan", "--profile", str(profile_path), "--roles", "chat", "--json"]) == 0
    service = json.loads(capsys.readouterr().out)["services"][0]

    assert service["decode_tps"] != 999.0
    assert service["estimated"] is True


def test_speed_preference_warns_when_speed_term_saturates(tmp_path) -> None:
    catalog = load_catalog(user_path=tmp_path / "models.yaml")
    profile = _profile(PROFILE_DIR / "t3-rtx4090-24gb.json")
    plan = build_plan(profile, catalog, Policy(roles=["chat"], prefer="speed"))
    service = plan.services[0]

    assert service.model_id == "devstral-small-2507"
    warning = next(item for item in plan.warnings if "could not discriminate" in item)
    assert "devstral-small-2507" in warning
    assert "qwen2.5-0.5b-instruct" in warning
    assert "38.8 tok/s" in warning
    assert "3654.6 tok/s" in warning


def test_speed_saturation_warning_is_speed_preference_only(tmp_path) -> None:
    catalog = load_catalog(user_path=tmp_path / "models.yaml")
    profile = _profile(PROFILE_DIR / "t3-rtx4090-24gb.json")

    for prefer in ("quality", "balanced"):
        plan = build_plan(profile, catalog, Policy(roles=["chat"], prefer=prefer))
        assert not any("could not discriminate" in item for item in plan.warnings)


def test_speed_selection_regression_for_gpu_and_cpu_profiles(tmp_path) -> None:
    catalog = load_catalog(user_path=tmp_path / "models.yaml")

    gpu = build_plan(
        _profile(PROFILE_DIR / "t3-rtx4090-24gb.json"),
        catalog,
        Policy(roles=["chat"], prefer="speed"),
    )
    cpu = build_plan(
        _profile(PROFILE_DIR / "t0-cpu-32gb.json"),
        catalog,
        Policy(roles=["chat"], prefer="speed"),
    )

    assert gpu.services[0].model_id == "devstral-small-2507"
    # 24B total but only 2.3B active params: the fastest CPU decode the
    # catalog offers once the model's quality cleared the prior floor.
    assert cpu.services[0].model_id == "lfm2-24b-a2b"


def test_speed_saturation_warning_requires_a_faster_candidate(tmp_path) -> None:
    catalog = load_catalog(user_path=tmp_path / "models.yaml")
    profile = _profile(PROFILE_DIR / "t3-rtx4090-24gb.json")
    model = next(item for item in catalog if item.id == "qwen2.5-32b-instruct")
    pool = _candidate_for(
        model, profile, Policy(roles=["chat"], prefer="speed"), None,
    )
    chosen = pool[0]

    assert _speed_saturation_warning(
        "chat", chosen, [chosen], Policy(roles=["chat"], prefer="speed"),
    ) is None
