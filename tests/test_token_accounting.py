from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from nmesh import gateway
from nmesh.catalog import ModelSpec
from nmesh.gateway.tokens import (
    Calibration,
    Sums,
    calibration_for,
    estimate_tokens,
    exact_tokens,
    fit,
    record,
    split_chars,
)
from nmesh.planner import Policy, build_plan
from nmesh.runtime import RuntimeStatus

from .test_planner import profile
from .test_telemetry import _gateway_plan, _TelemetryHandler


def test_split_and_default_estimate_preserve_heuristic() -> None:
    assert split_chars("日本語abc") == (3, 3)
    assert estimate_tokens("日本語") == 3
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("") == 0


def test_fit_recovers_known_coefficients() -> None:
    rows = [(1, 3), (2, 1), (4, 2), (3, 5)] * 5
    sums = Sums(
        n=len(rows),
        s_cc=sum(c * c for c, _ in rows),
        s_co=sum(c * o for c, o in rows),
        s_oo=sum(o * o for _, o in rows),
        s_ct=sum(c * (1.25 * c + 0.5 * o) for c, o in rows),
        s_ot=sum(o * (1.25 * c + 0.5 * o) for c, o in rows),
    )
    result = fit(sums)
    assert result == Calibration(1.25, 0.5, 20, True)


def test_fit_defaults_for_under_sampled_singular_and_clamped() -> None:
    assert fit(Sums(n=19)) == Calibration(1.0, 0.25, 19, False)
    singular = Sums(n=20, s_cc=20, s_ct=20)
    assert fit(singular) == Calibration(1.0, 0.25, 20, False)
    clamped = Sums(n=20, s_cc=20, s_oo=20, s_ct=100, s_ot=1)
    assert fit(clamped) == Calibration(1.0, 0.25, 20, False)


def test_record_persists_sums_and_calibration(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    for cjk, other, tokens in ((1, 3, 2.75), (2, 1, 3.0)) * 10:
        text = "日" * cjk + "a" * other
        record("chat", text, int(tokens))
    payload = json.loads((tmp_path / "tokens.json").read_text(encoding="utf-8"))
    assert payload["services"]["chat"]["n"] == 20
    assert calibration_for("chat").samples == 20


def test_exact_tokens_timeout_falls_back_without_raising() -> None:
    class Client:
        async def post(self, *args, **kwargs):
            raise TimeoutError("timeout")

    assert asyncio.run(exact_tokens("http://127.0.0.1:1", "hello", Client())) is None


def _routing_plan(port: int = 1):
    model = ModelSpec(
        "routing-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 80.0, "test", {"hf_gguf": "test/repo"},
    )
    plan = build_plan(profile(64, (24,)), [model], Policy(roles=["chat"]))
    small = replace(plan.services[0], name="small", context=100, port=port)
    large = replace(plan.services[0], name="large", context=1000, port=port + 1)
    return replace(
        plan,
        services=[small, large],
        routing=replace(plan.routing, role_to_service={"chat": "small"}),
    )


def test_routing_band_uses_exact_count_only_when_service_is_running(monkeypatch) -> None:
    plan = _routing_plan()
    monkeypatch.setattr(
        gateway, "runtime_status",
        lambda: RuntimeStatus(True, [{"service": "small", "running": True, "backend": "llamacpp"}]),
    )
    calls: list[str] = []

    async def exact(base_url, text, client):
        calls.append(text)
        return 100

    monkeypatch.setattr(gateway, "exact_tokens", exact)
    request = {"messages": [{"content": "x" * 200}]}
    selected = asyncio.run(gateway._routing_token_hint(request, plan))
    assert selected == 100
    assert gateway.route(request, plan, token_hint=selected) == "large"
    assert calls == ["x" * 200]

    monkeypatch.setattr(gateway, "runtime_status", lambda: RuntimeStatus(False, []))
    monkeypatch.setattr(
        gateway, "exact_tokens",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected tokenize")),
    )
    selected = asyncio.run(gateway._routing_token_hint(request, plan))
    assert selected == 50
    assert gateway.route(request, plan, token_hint=selected) == "small"


def test_metrics_expose_default_token_calibration() -> None:
    plan = _routing_plan()
    with TestClient(gateway.create_app(plan)) as client:
        metrics = client.get("/metrics").json()
        calibration = metrics["token_calibration"]["small"]
        assert calibration == {
            "cjk_per_char": 1.0,
            "other_per_char": 0.25,
            "samples": 0,
            "measured": False,
        }
        text = client.get("/metrics/prometheus").text
    assert 'nmesh_token_calibration_cjk_per_char{measured="false", service="small"}' in text
    assert 'nmesh_token_calibration_samples{measured="false", service="small"} 0' in text


def test_missing_prompt_usage_does_not_update_calibration(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    _TelemetryHandler.stream = False
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _TelemetryHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        plan = _gateway_plan(upstream.server_address[1])
        with TestClient(gateway.create_app(plan)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        assert response.status_code == 200
        assert not (tmp_path / "tokens.json").exists()
    finally:
        _TelemetryHandler.stream = True
        upstream.shutdown()
        upstream.server_close()
