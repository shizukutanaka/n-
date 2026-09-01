import json
from dataclasses import asdict
from pathlib import Path

from nmesh import cli
from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan
from nmesh.planner.core import _gpu_budget
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
            if known_devices == ():
                assert service.n_gpu_layers == 0
            if service.backend in {"ollama", "vllm", "mlx"}:
                assert service.n_gpu_layers is None
            if known_devices and any(device.startswith("CUDA") for device in known_devices):
                assert service.n_gpu_layers is not None
                assert 0 < service.n_gpu_layers <= layers[service.model_id]
            if not service.gpu_indices or service.memory.gpu_bytes <= 0:
                continue
            per_gpu = service.memory.gpu_bytes / len(service.gpu_indices)
            target = swap_usage if service.name in swap_group else usage
            for index in service.gpu_indices:
                target[index] = max(target[index], per_gpu) if (
                    service.name in swap_group
                ) else target[index] + per_gpu
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
