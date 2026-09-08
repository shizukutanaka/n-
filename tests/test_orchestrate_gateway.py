from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

import nmesh.gateway as gateway_module
from nmesh.gateway import create_app
from nmesh.orchestrate import (
    PROTOCOL_VERSION,
    Delegation,
    DelegationRecord,
    RoleIdentity,
)
from nmesh.orchestrate.protocol import Call

from .test_api_surface import _completion_plan


def _delegation_plan() -> object:
    plan = _completion_plan(1)
    lead = replace(plan.services[0], name="chat", roles=["chat"], model_id="lead")
    worker = replace(
        plan.services[0],
        name="worker",
        roles=["worker"],
        model_id="worker",
        port=2,
        memory=replace(plan.services[0].memory, weight_bytes=1.0),
    )
    routing = replace(
        plan.routing,
        role_to_service={"chat": "chat", "worker": "worker"},
    )
    return replace(plan, services=[lead, worker], routing=routing)


def _record(plan: object) -> DelegationRecord:
    services = plan.services
    return DelegationRecord(
        lead=RoleIdentity(services[0].model_id, services[0].quant, services[0].backend),
        worker=RoleIdentity(services[1].model_id, services[1].quant, services[1].backend),
        suite="hard",
        digest="digest",
        n_tasks=10,
        worker_passed=5,
        lead_passed=5,
        delegated_passed=6,
        ceiling_passed=6,
        delegated_gained=1,
        delegated_lost=0,
        delegated_p=0.01,
        ceiling_gained=1,
        ceiling_lost=0,
        ceiling_p=0.01,
        verifier_accuracy=1.0,
        accepted_but_wrong=0,
        rejected_but_right=0,
        lead_tokens_solo=10,
        lead_tokens_delegated=20,
        seconds_solo=1.0,
        seconds_delegated=2.0,
        unscorable=0,
        reasoning_allowance=0,
        protocol=PROTOCOL_VERSION,
        at=1.0,
        repeats=2,
    )


def test_delegate_model_is_hidden_without_superior_evidence(monkeypatch) -> None:
    plan = _delegation_plan()
    monkeypatch.setattr(gateway_module, "load_cache", dict)
    with TestClient(create_app(plan)) as client:
        models = client.get("/v1/models").json()["data"]
        assert "nmesh-delegate" not in {item["id"] for item in models}
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nmesh-delegate", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == 409


def test_delegate_model_sums_usage_when_allowed(monkeypatch) -> None:
    plan = _delegation_plan()
    record = _record(plan)
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": record})

    def fake_delegate(client, prompt, max_tokens, *, lead, worker, ledger):
        del client, prompt, max_tokens, lead, worker
        ledger.worker.add(Call("worker", 3, 4, False, 0.0))
        ledger.verify.add(Call("YES", 5, 1, False, 0.0))
        return Delegation(
            "worker",
            True,
            False,
            False,
            Call("worker", 3, 4, False, 0.0),
            Call("YES", 5, 1, False, 0.0),
        )

    monkeypatch.setattr(gateway_module, "delegate", fake_delegate)
    with TestClient(create_app(plan)) as client:
        models = client.get("/v1/models").json()["data"]
        assert "nmesh-delegate" in {item["id"] for item in models}
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nmesh-delegate", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
    }
    assert response.json()["nmesh_role"] == "worker"


def test_delegate_model_refuses_unconfirmed_evidence(monkeypatch) -> None:
    plan = _delegation_plan()
    record = replace(_record(plan), repeats=1)
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": record})
    with TestClient(create_app(plan)) as client:
        models = client.get("/v1/models").json()["data"]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nmesh-delegate",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert "nmesh-delegate" not in {item["id"] for item in models}
    assert response.status_code == 409
    assert "unconfirmed" in response.json()["error"]["message"]


def test_delegate_model_reports_non_superior_evidence(monkeypatch) -> None:
    plan = _delegation_plan()
    record = replace(_record(plan), delegated_passed=4, delegated_p=0.2)
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": record})
    with TestClient(create_app(plan)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nmesh-delegate",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 409
    detail = response.json()["error"]["message"]
    assert "delegated=4" in detail
    assert "lead=5" in detail
    assert "p=0.2000" in detail


def test_delegate_model_rejects_streaming(monkeypatch) -> None:
    plan = _delegation_plan()
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": _record(plan)})
    with TestClient(create_app(plan)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nmesh-delegate",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 400


def test_embed_only_service_is_not_an_eligible_worker(monkeypatch) -> None:
    plan = _delegation_plan()
    embed = replace(plan.services[1], roles=["embed"])
    plan = replace(plan, services=[plan.services[0], embed])
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": _record(plan)})
    with TestClient(create_app(plan)) as client:
        models = client.get("/v1/models").json()["data"]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nmesh-delegate",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert "nmesh-delegate" not in {item["id"] for item in models}
    assert response.status_code == 409
    assert "two distinct" in response.json()["error"]["message"]


def test_swap_exclusive_pair_is_not_eligible(monkeypatch) -> None:
    plan = replace(_delegation_plan(), swap_group=["chat", "worker"])
    monkeypatch.setattr(gateway_module, "load_cache", lambda: {"record": _record(plan)})
    with TestClient(create_app(plan)) as client:
        models = client.get("/v1/models").json()["data"]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nmesh-delegate",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert "nmesh-delegate" not in {item["id"] for item in models}
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    assert "chat" in message and "worker" in message
    assert "mutually exclusive" in message
