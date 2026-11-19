from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nmesh import cli
from nmesh.gateway import create_app

from .test_gateway_ui import _plan


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NMESH_LANG", "en")
    monkeypatch.delenv("NMESH_API_KEY", raising=False)
    monkeypatch.setattr(cli, "gateway_health", lambda port: True)


def test_connect_without_plan_or_gateway_warns_on_stderr_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "gateway_health", lambda port: False)
    monkeypatch.setattr(cli, "load_plan", lambda: None)
    assert cli.main(["connect", "--port", "18123"]) == 0
    captured = capsys.readouterr()
    assert "http://127.0.0.1:18123/v1" in captured.out
    assert "model IDs: nmesh-auto\n" in captured.out
    assert "not required" in captured.out
    assert "pasture, rekai, sdk" in captured.out
    assert "nmesh up --detach" in captured.err
    assert "no saved plan" in captured.err
    assert "nmesh up --detach" not in captured.out


def test_connect_lists_planned_service_model_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan()
    monkeypatch.setattr(cli, "load_plan", lambda: plan)
    assert cli.main(["connect"]) == 0
    captured = capsys.readouterr()
    expected = ", ".join(
        ["nmesh-auto"] + [f"nmesh-{service.name}" for service in plan.services]
    )
    assert f"model IDs: {expected}" in captured.out
    assert captured.err == ""


def test_connect_pasture_uses_its_openai_compatible_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_plan", _plan)
    assert cli.main(["connect", "pasture", "--port", "18200"]) == 0
    captured = capsys.readouterr()
    assert (
        "PASTURE_LOCAL_BACKEND=lmstudio "
        "PASTURE_LOCAL_OPENAI_URL=http://127.0.0.1:18200/v1 "
        "PASTURE_LOCAL_MODEL=nmesh-auto pasture serve"
    ) in captured.out
    assert "Authorization" not in captured.err


def test_connect_pasture_warns_when_gateway_requires_a_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_plan", _plan)
    monkeypatch.setenv("NMESH_API_KEY", "secret-value")
    assert cli.main(["connect", "pasture"]) == 0
    captured = capsys.readouterr()
    assert "API key: required" in captured.out
    assert "401" in captured.err
    assert "secret-value" not in captured.out + captured.err


def test_connect_rekai_explains_file_origin_and_image_limits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_plan", _plan)
    assert cli.main(["connect", "rekai"]) == 0
    out = capsys.readouterr().out
    assert "endpoint URL: http://127.0.0.1:18000/v1" in out
    assert "model: nmesh-auto" in out
    assert "NMESH_GATEWAY_ORIGINS=null" in out
    assert "mmproj" in out


def test_connect_sdk_names_the_openai_sdk_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_plan", _plan)
    assert cli.main(["connect", "sdk"]) == 0
    out = capsys.readouterr().out
    assert "export OPENAI_BASE_URL=http://127.0.0.1:18000/v1" in out
    assert "OPENAI_API_KEY=" in out


def test_connect_rejects_unknown_app(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["connect", "cursor"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_connect_japanese_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NMESH_LANG", "ja")
    monkeypatch.setattr(cli, "load_plan", _plan)
    assert cli.main(["connect", "rekai"]) == 0
    out = capsys.readouterr().out
    assert "OpenAI 互換のベース URL" in out
    assert "接続先 URL: http://127.0.0.1:18000/v1" in out


def test_gateway_origin_null_matches_the_rekai_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The printed RekAI advice relies on these two gateway behaviours: a
    file:// page's `Origin: null` is refused by default and admitted once
    NMESH_GATEWAY_ORIGINS=null is set."""
    monkeypatch.delenv("NMESH_GATEWAY_ORIGINS", raising=False)
    client = TestClient(create_app(_plan(), watchdog=False))
    assert client.get("/v1/models", headers={"Origin": "null"}).status_code == 403
    monkeypatch.setenv("NMESH_GATEWAY_ORIGINS", "null")
    response = client.get("/v1/models", headers={"Origin": "null"})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "null"
    assert "nmesh-auto" in [item["id"] for item in response.json()["data"]]
