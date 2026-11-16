from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from nmesh.catalog import ModelSpec
from nmesh.gateway import create_app
from nmesh.gateway.ui import build_overview
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _plan(port: int = 1):
    model = ModelSpec(
        "ui-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    service = replace(plan.services[0], port=port)
    return replace(plan, services=[service])


def test_build_overview_merges_plan_runtime_and_telemetry() -> None:
    overview = build_overview(
        planned=[{
            "name": "chat", "model": "ui-model", "backend": "llamacpp",
            "port": 8080, "roles": ["chat"],
        }],
        runtime={
            "running": True,
            "warnings": ["watch out"],
            "services": [
                {"service": "chat", "pid": 42, "running": True, "restarts": 2,
                 "model_ref": "test/repo", "backend": "llamacpp", "port": 8080},
                {"service": "adopted", "pid": 7, "running": True, "shared": True},
            ],
        },
        running=[{
            "service": "chat", "running": True, "idle": True,
            "idle_seconds": 12.5, "in_flight": 0,
        }],
        telemetry={"chat": {"samples": 3.0, "decode_tps_median": 42.0}},
        keep_alive=30.0,
    )
    assert overview["gateway_running"] is True
    assert overview["warnings"] == ["watch out"]
    assert overview["keep_alive"] == 30.0
    services = overview["services"]
    assert isinstance(services, list)
    assert [item["name"] for item in services] == ["chat", "adopted"]
    chat = services[0]
    assert chat["model"] == "ui-model"
    assert chat["backend"] == "llamacpp"
    assert chat["port"] == 8080
    assert chat["planned"] is True
    assert chat["running"] is True
    assert chat["idle"] is True
    assert chat["idle_seconds"] == 12.5
    assert chat["pid"] == 42
    assert chat["restarts"] == 2
    assert chat["telemetry"] == {"samples": 3.0, "decode_tps_median": 42.0}
    adopted = services[1]
    assert adopted["planned"] is False
    assert adopted["running"] is True
    assert adopted["shared"] is True
    assert adopted["telemetry"] is None


def test_build_overview_tolerates_empty_runtime() -> None:
    overview = build_overview(
        planned=[], runtime={}, running=[], telemetry={}, keep_alive=0.0
    )
    assert overview["gateway_running"] is False
    assert overview["warnings"] == []
    assert overview["services"] == []


def test_ui_page_is_self_contained_and_hardened() -> None:
    with TestClient(create_app(_plan())) as client:
        response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    body = response.text
    assert "/ui/api/overview" in body
    # The page must not pull any external asset: no src=/href= resources.
    assert "src=" not in body
    assert "href=" not in body


def test_ui_overview_reports_plan_services() -> None:
    plan = _plan()
    name = plan.services[0].name
    with TestClient(create_app(plan)) as client:
        response = client.get("/ui/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert [item["name"] for item in data["services"]] == [name]
    service = data["services"][0]
    assert service["model"] == "ui-model"
    assert service["planned"] is True
    assert service["running"] is False
    assert "warnings" in data


def test_ui_overview_requires_api_key_but_page_shell_does_not(monkeypatch) -> None:
    secret = "ui-test" + "-key"
    monkeypatch.setenv("NMESH_API_KEY", secret)
    with TestClient(create_app(_plan())) as client:
        assert client.get("/ui").status_code == 200
        assert client.get("/ui/api/overview").status_code == 401
        allowed = client.get(
            "/ui/api/overview",
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert allowed.status_code == 200


def test_ui_rejects_foreign_origin() -> None:
    with TestClient(create_app(_plan())) as client:
        response = client.get(
            "/ui/api/overview", headers={"Origin": "https://evil.example"}
        )
    assert response.status_code == 403


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
