from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nmesh import gateway
from nmesh.catalog import ModelSpec
from nmesh.gateway.tokens import (
    Calibration,
    Sums,
    all_sums,
    calibration_for,
    calibration_key,
    estimate_tokens,
    exact_tokens,
    fit,
    load_sums,
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
    overhead = 7.0
    sums = Sums(
        n=len(rows),
        s_cc=sum(c * c for c, _ in rows),
        s_co=sum(c * o for c, o in rows),
        s_oo=sum(o * o for _, o in rows),
        s_ct=sum(c * (1.25 * c + 0.5 * o + overhead) for c, o in rows),
        s_ot=sum(o * (1.25 * c + 0.5 * o + overhead) for c, o in rows),
        s_c=sum(c for c, _ in rows),
        s_o=sum(o for _, o in rows),
        s_t=sum(1.25 * c + 0.5 * o + overhead for c, o in rows),
    )
    result = fit(sums)
    assert result.cjk_per_char == pytest.approx(1.25)
    assert result.other_per_char == pytest.approx(0.5)
    assert result.overhead == pytest.approx(overhead)
    assert result.samples == 20
    assert result.measured is True


def test_fit_uses_intercept_for_short_prompt_overhead() -> None:
    rows = [(0, 7), (0, 8), (1, 6), (2, 6), (0, 9)] * 4
    sums = Sums(
        n=len(rows),
        s_cc=sum(c * c for c, _ in rows),
        s_co=sum(c * o for c, o in rows),
        s_oo=sum(o * o for _, o in rows),
        s_ct=sum(c * (c + 0.25 * o + 26) for c, o in rows),
        s_ot=sum(o * (c + 0.25 * o + 26) for c, o in rows),
        s_c=sum(c for c, _ in rows),
        s_o=sum(o for _, o in rows),
        s_t=sum(c + 0.25 * o + 26 for c, o in rows),
    )
    result = fit(sums)
    assert result.other_per_char == pytest.approx(0.25)
    assert result.overhead == pytest.approx(26)
    assert result.other_per_char < 2.0


def test_fit_defaults_for_under_sampled_singular_and_clamped() -> None:
    assert fit(Sums(n=19)) == Calibration(1.0, 0.25, 19, False)
    singular = Sums(n=20, s_cc=20, s_ct=20)
    assert fit(singular) == Calibration(1.0, 0.25, 20, False)
    clamped = Sums(n=20, s_cc=20, s_oo=20, s_ct=100, s_ot=1)
    assert fit(clamped) == Calibration(1.0, 0.25, 20, False)


def test_record_persists_independent_chat_and_text_sums(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    chat_rows = [(1, 6, 32), (1, 7, 33), (2, 6, 33), (2, 7, 34)] * 5
    text_rows = [(2, 15, 5), (1, 15, 4), (2, 20, 6), (1, 20, 5)] * 5
    for cjk, other, tokens in chat_rows:
        record(
            calibration_key("routing-model", True),
            "日" * cjk + "a" * other,
            tokens,
        )
    for cjk, other, tokens in text_rows:
        record(
            calibration_key("routing-model", False),
            "日" * cjk + "a" * other,
            tokens,
        )
    payload = json.loads((tmp_path / "tokens.json").read_text(encoding="utf-8"))
    assert payload["models"]["routing-model|chat"]["n"] == 20
    assert payload["models"]["routing-model|text"]["n"] == 20
    chat = calibration_for(calibration_key("routing-model", True))
    text = calibration_for(calibration_key("routing-model", False))
    assert chat.samples == 20
    assert text.samples == 20
    assert chat.overhead > 10
    assert text.overhead < 5
    assert calibration_for("routing-model").samples == 0
    assert all_sums()["routing-model|chat"].n == 20


def _sums_for_rows(rows: list[tuple[int, int, int]]) -> Sums:
    return Sums(
        n=len(rows),
        s_cc=sum(c * c for c, _, _ in rows),
        s_co=sum(c * o for c, o, _ in rows),
        s_oo=sum(o * o for _, o, _ in rows),
        s_ct=sum(c * t for c, _, t in rows),
        s_ot=sum(o * t for _, o, t in rows),
        s_c=sum(c for c, _, _ in rows),
        s_o=sum(o for _, o, _ in rows),
        s_t=sum(t for _, _, t in rows),
    )


def test_endpoint_scoped_fit_matches_chat_and_text_prompt_overhead() -> None:
    chat_rows = [(1, 6, 32), (1, 7, 33), (2, 6, 33), (2, 7, 34)] * 5
    text_rows = [(2, 15, 5), (1, 15, 4), (2, 20, 6), (1, 20, 5)] * 5
    chat = fit(_sums_for_rows(chat_rows))
    text = fit(_sums_for_rows(text_rows))
    assert chat.overhead > 10
    assert text.overhead < 5


def test_old_sums_are_discarded_when_loaded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    (tmp_path / "tokens.json").write_text(json.dumps({
        "models": {
            "old-model": {
                "n": 20,
                "s_cc": 1,
                "s_co": 2,
                "s_oo": 3,
                "s_ct": 4,
                "s_ot": 5,
            },
        },
    }), encoding="utf-8")
    assert load_sums("old-model") == Sums()
    assert all_sums()["old-model"] == Sums()


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
    # The routing path memoizes runtime status for ~1s; simulate a fresh app.
    monkeypatch.setattr(gateway, "_runtime_status_cache", None)
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
        calibration = metrics["token_calibration"]["small|chat"]
        assert calibration == {
            "model": "routing-model",
            "kind": "chat",
            "cjk_per_char": 1.0,
            "other_per_char": 0.25,
            "samples": 0,
            "measured": False,
        }
        text = client.get("/metrics/prometheus").text
    labels = 'kind="chat", measured="false", model="routing-model", service="small"'
    assert "nmesh_token_calibration_cjk_per_char{" + labels + "}" in text
    assert "nmesh_token_calibration_samples{" + labels + "} 0" in text


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


def test_content_normalizes_none_and_multimodal_messages() -> None:
    assert gateway._content({
        "messages": [{"role": "assistant", "content": None}],
    }) == ""
    assert gateway._content({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "secret"}},
                {"type": "text", "text": "world"},
            ],
        }],
    }) == "hello world"
    assert gateway._content({
        "messages": [{"role": "user", "content": "plain"}],
    }) == "plain"


def test_tool_requests_skip_token_calibration_recording(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        gateway,
        "record_token_calibration",
        lambda *args: calls.append(args),
    )
    service = _routing_plan().services[0]
    for key in ("tools", "functions"):
        asyncio.run(gateway._record_prompt_calibration(
            service,
            {"messages": [{"content": "hello"}], key: [{"type": "function"}]},
            {"prompt_tokens": 5},
        ))
    asyncio.run(gateway._record_prompt_calibration(
        service,
        {"messages": [{"content": "hello"}]},
        {"prompt_tokens": 5},
    ))
    assert len(calls) == 1
    assert calls[0][1:] == ("hello", 5)
    assert calls[0][0] == calibration_key(service.model_id, True)


def test_prompt_calibration_uses_text_key_for_prompt_requests(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        gateway,
        "record_token_calibration",
        lambda *args: calls.append(args),
    )
    service = _routing_plan().services[0]
    asyncio.run(gateway._record_prompt_calibration(
        service,
        {"prompt": "hello"},
        {"prompt_tokens": 5},
    ))
    assert calls == [(calibration_key(service.model_id, False), "hello", 5)]
