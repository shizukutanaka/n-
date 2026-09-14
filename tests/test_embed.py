from __future__ import annotations

import json
from dataclasses import asdict

import httpx

from nmesh.bench.embed import (
    EMBED_HARNESS_VERSION,
    EmbedRecord,
    load_embed_cache,
    measure_embedding,
    save_embed,
)
from nmesh.catalog import ModelSpec
from nmesh.evidence_inventory import collect_evidence
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _record(**updates: object) -> EmbedRecord:
    values: dict[str, object] = {
        "model_id": "embed",
        "quant": "q8_0",
        "backend": "ollama",
        "gpu_name": "cpu",
        "n_gpu_layers": 0,
        "requested_context": 4096,
        "probe_tokens_small": 5120,
        "served_small": 2048,
        "probe_tokens_large": 8192,
        "served_large": 2048,
        "encode_tps": 500.0,
        "encode_tps_min": 400.0,
        "encode_tps_max": 600.0,
        "encode_input_tokens": 512,
        "runs": 3,
        "harness": EMBED_HARNESS_VERSION,
        "at": 1.0,
    }
    values.update(updates)
    return EmbedRecord(**values)


def test_embed_record_round_trip_and_cap_agreement(tmp_path) -> None:
    record = _record()
    path = tmp_path / "embed.json"
    save_embed(record, path)
    loaded = load_embed_cache(path)
    assert loaded[next(iter(loaded))] == record
    assert record.cap == 2048
    assert _record(
        probe_tokens_small=12288,
        served_small=2048,
        probe_tokens_large=24576,
        served_large=2047,
    ).cap == 2047
    assert _record(
        probe_tokens_small=12288,
        served_small=12288,
        probe_tokens_large=24576,
        served_large=24576,
    ).cap is None
    assert _record(
        probe_tokens_small=12288,
        served_small=2047,
        probe_tokens_large=24576,
        served_large=3000,
    ).cap is None


def test_embed_cache_rejects_malformed_records(tmp_path) -> None:
    valid = _record()
    payload = {
        "results": {
            "valid": asdict(valid),
            "negative": {**asdict(valid), "served_small": -1},
            "boolean": {**asdict(valid), "runs": True},
            "infinite": {**asdict(valid), "encode_tps": float("inf")},
            "wrong": {**asdict(valid), "backend": 1},
        }
    }
    path = tmp_path / "embed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_embed_cache(path)
    assert list(loaded) == ["valid"]


def test_measure_embedding_uses_usage_tokens_and_unique_inputs() -> None:
    values = iter((100, 2048, 2048, 100, 110, 120))
    inputs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        inputs.append(payload["input"])
        return httpx.Response(
            200,
            json={"usage": {"prompt_tokens": next(values)}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = measure_embedding(
            client,
            "http://test",
            "embed",
            requested_context=4096,
            runs=3,
        )
    finally:
        client.close()
    assert result.served_small == result.served_large == 2048
    assert result.probe_tokens_small == 6144
    assert result.probe_tokens_large == 12288
    assert result.encode_input_tokens == 110
    assert len(inputs) == len({item.split(" ")[0] for item in inputs})


def test_planner_clamps_embed_context_and_warns() -> None:
    model = ModelSpec(
        "embed", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed"},
    )
    baseline = build_plan(profile(64), [model], Policy(roles=["embed"]))
    capped = build_plan(
        profile(64),
        [model],
        Policy(roles=["embed"]),
        embed_input_caps={(
            baseline.services[0].model_id.casefold(),
            baseline.services[0].quant.casefold(),
            baseline.services[0].backend.casefold(),
        ): 2048},
    )
    assert capped.services[0].context == 2048
    assert capped.services[0].memory.total_bytes < baseline.services[0].memory.total_bytes
    argv = capped.services[0].launch.argv
    assert argv[argv.index("-c") + 1] == "2048"
    assert any("2048" in warning for warning in capped.warnings)


def test_embed_record_refused_probe_is_not_a_cap() -> None:
    record = _record(
        probe_tokens_small=6144,
        served_small=6144,
        probe_tokens_large=12288,
        served_large=0,
        refused_large=True,
    )
    assert record.cap is None
    # The identical served counts without the flag would claim cap=2048.
    assert _record(served_large=2048).cap == 2048


def test_measure_embedding_treats_http_error_probe_as_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        words = len(payload["input"].split(" "))
        if words > 2000:
            return httpx.Response(
                413,
                json={"error": {"message": "input too large"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"usage": {"prompt_tokens": words}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = measure_embedding(
            client,
            "http://test",
            "embed",
            requested_context=4096,
            runs=2,
        )
    finally:
        client.close()
    # 4096 * 3 = 12288 probe words refused, small probe (6144) also >2000 so refused.
    assert result.refused_small and result.refused_large
    assert result.served_small == 0 and result.served_large == 0


def test_measure_embedding_partial_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        words = len(payload["input"].split(" "))
        if words > 7000:
            return httpx.Response(
                413,
                json={"error": {"message": "input too large"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={"usage": {"prompt_tokens": words}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        result = measure_embedding(
            client,
            "http://test",
            "embed",
            requested_context=4096,
            runs=2,
        )
    finally:
        client.close()
    assert not result.refused_small
    assert result.served_small > 0
    assert result.refused_large
    assert result.served_large == 0


def test_measure_embedding_calibration_error_still_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "boom"}},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        try:
            measure_embedding(
                client,
                "http://test",
                "embed",
                requested_context=4096,
            )
        except httpx.HTTPStatusError:
            pass
        else:
            raise AssertionError("calibration failure must raise")
    finally:
        client.close()


def test_planner_notes_measured_untruncated_instead_of_unverified() -> None:
    model = ModelSpec(
        "embed", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed"},
    )
    baseline = build_plan(profile(64), [model], Policy(roles=["embed"]))
    key = (
        baseline.services[0].model_id.casefold(),
        baseline.services[0].quant.casefold(),
        baseline.services[0].backend.casefold(),
    )
    plan = build_plan(
        profile(64),
        [model],
        Policy(roles=["embed"]),
        embed_input_caps={},
        embed_measured={key},
    )
    assert any("no silent truncation" in warning for warning in plan.warnings)
    assert not any(
        "nmesh bench --service embed" in warning for warning in plan.warnings
    )


def test_embed_inventory_reports_refusal_not_unproven(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    save_embed(
        _record(
            probe_tokens_small=6144,
            served_small=6144,
            probe_tokens_large=12288,
            served_large=0,
            refused_large=True,
        ),
        tmp_path / "embed.json",
    )
    payload = collect_evidence()
    row = next(row for row in payload["records"] if row["kind"] == "embed")
    assert row["reasons"] == ["input_refused"]
    assert "refused at 12288 tokens" in row["value"]
    assert row["remeasure"] == ""


def test_planner_warns_when_embed_cap_is_unverified() -> None:
    model = ModelSpec(
        "embed", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed"},
    )
    plan = build_plan(
        profile(64),
        [model],
        Policy(roles=["embed"]),
        embed_input_caps={},
    )
    assert any("nmesh bench --service embed" in warning for warning in plan.warnings)


def test_embed_inventory_reports_cap_and_speed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    save_embed(_record(), tmp_path / "embed.json")
    payload = collect_evidence()
    row = next(row for row in payload["records"] if row["kind"] == "embed")
    assert row["usable"] is True
    assert row["served_cap"] == 2048
    assert row["encode_tps"] == 500.0
    assert payload["counts"]["embed"] == 1
