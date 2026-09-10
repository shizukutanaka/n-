from __future__ import annotations

import json
from dataclasses import replace

import httpx

from nmesh import cli
from nmesh.bench.retrieval import (
    RETRIEVAL_HARNESS_VERSION,
    RetrievalChunkArm,
    RetrievalLimit,
    RetrievalRecord,
    RetrievalRung,
    load_retrieval_cache,
    measure_retrieval,
    measure_retrieval_chunk_arm,
    retrieval_digest,
    save_retrieval,
)
from nmesh.catalog import ModelSpec
from nmesh.evidence_inventory import collect_evidence
from nmesh.planner import Policy, build_plan

from .test_planner import profile


def _record(rungs: tuple[RetrievalRung, ...]) -> RetrievalRecord:
    return RetrievalRecord(
        model_id="embed",
        quant="q8_0",
        backend="llamacpp",
        gpu_name="cpu",
        n_gpu_layers=0,
        rungs=rungs,
        digest=retrieval_digest(),
        harness=RETRIEVAL_HARNESS_VERSION,
        at=1.0,
    )


def _ladder(
    hits: tuple[int, ...],
    served: tuple[int, ...] = (166, 320, 603, 1177, 2346, 2921, 3512, 4370),
    saturated: tuple[bool, ...] | None = None,
) -> tuple[RetrievalRung, ...]:
    flags = saturated or (False,) * len(hits)
    return tuple(
        RetrievalRung(index, tokens, count, 8, flag)
        for index, (tokens, count, flag) in enumerate(zip(served, hits, flags), 1)
    )


def test_retrieval_properties_follow_conservative_ladder() -> None:
    record = _record(_ladder((8, 8, 8, 8, 7, 5, 2, 1)))
    assert record.control_passed is True
    assert record.usable_tokens == 2346
    assert record.degraded_tokens == 3512


def test_measure_retrieval_uses_one_request_per_document_and_query() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        requests.append(payload)
        input_value = payload["input"]
        text = input_value if isinstance(input_value, str) else ""
        vector = [1.0, 0.0] if "Meridian" in text else [0.0, 1.0]
        tokens = len(text.split())
        return httpx.Response(
            200,
            json={
                "data": [{"embedding": vector}],
                "usage": {"prompt_tokens": tokens},
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        rungs = measure_retrieval(
            client,
            "http://test",
            "embed",
            seeds=(11, 23),
            rung_words=(10, 20),
        )
    finally:
        client.close()
    assert len(requests) == 2 * 2 * (8 + 1)
    assert all(rung.hits == 2 for rung in rungs)
    assert rungs[0].served_tokens < rungs[1].served_tokens


def test_retrieval_control_failure_proves_nothing() -> None:
    record = _record(_ladder((7, 8, 8, 8, 8, 8, 2, 1)))
    assert record.control_passed is False
    assert record.usable_tokens is None
    assert record.degraded_tokens is None


def test_saturated_rungs_do_not_establish_degradation() -> None:
    record = _record(_ladder(
        (8, 8, 8, 8, 7, 2),
        served=(166, 320, 603, 1177, 2047, 2047),
        saturated=(False, False, False, False, True, True),
    ))
    assert record.usable_tokens == 1177
    assert record.degraded_tokens is None


def test_indeterminate_band_does_not_establish_degradation() -> None:
    record = _record(_ladder((8, 8, 8, 8, 7, 6, 8)))
    assert record.usable_tokens == 2346
    assert record.degraded_tokens is None


def test_chunk_arm_round_trip_and_recovery_status(tmp_path) -> None:
    record = replace(
        _record(_ladder((8, 8, 8, 8, 7, 5, 2))),
        chunk=RetrievalChunkArm(2400, 800, 1177, 8, 8),
    )
    path = tmp_path / "retrieval.json"
    save_retrieval(record, path)
    assert next(iter(load_retrieval_cache(path).values())) == record
    assert record.chunk_recovers is True
    assert replace(record, chunk=replace(record.chunk, hits=6)).chunk_recovers is False
    assert replace(record, chunk=None).chunk_recovers is None
    assert _record(_ladder((8, 8, 8))).chunk_recovers is None


def test_malformed_chunk_records_are_dropped(tmp_path) -> None:
    record = replace(
        _record(_ladder((8, 8, 8))),
        chunk=RetrievalChunkArm(2400, 800, 1177, 8, 8),
    )
    path = tmp_path / "retrieval.json"
    save_retrieval(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = next(iter(payload["results"].values()))
    payload["results"].update({
        "missing": {**valid, "chunk": {"doc_words": 2400}},
        "boolean": {
            **valid,
            "chunk": {**valid["chunk"], "chunk_words": True},
        },
        "bad_hits": {
            **valid,
            "chunk": {**valid["chunk"], "hits": 9},
        },
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(load_retrieval_cache(path)) == 1


def test_chunk_arm_uses_batch_inputs_and_max_chunk_similarity() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        requests.append(payload)
        input_value = payload["input"]
        if isinstance(input_value, list):
            vectors = [
                {"embedding": [1.0, 0.0] if "Meridian" in value else [0.0, 1.0]}
                for value in input_value
            ]
        else:
            vectors = [{"embedding": [1.0, 0.0]}]
        return httpx.Response(
            200,
            json={
                "data": vectors,
                "usage": {"prompt_tokens": 10},
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        arm = measure_retrieval_chunk_arm(
            client, "http://test", "embed",
            doc_words=10, chunk_words=4, chunk_tokens=12, seeds=(11, 23),
        )
    finally:
        client.close()
    assert arm == RetrievalChunkArm(10, 4, 12, 2, 2)
    assert len(requests) == 2 * (8 + 1)
    assert all(isinstance(request["input"], list) for request in requests[:8])


def test_retrieval_cache_round_trip_and_malformed_records(tmp_path) -> None:
    record = _record(_ladder((8, 8, 8)))
    path = tmp_path / "retrieval.json"
    save_retrieval(record, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["results"]
    valid = next(iter(values.values()))
    values.update({
        "negative": {**valid, "rungs": [{**valid["rungs"][0], "hits": -1}]},
        "boolean": {**valid, "n_gpu_layers": True},
        "bad_hits": {**valid, "rungs": [{**valid["rungs"][0], "hits": 9}]},
        "missing": {key: value for key, value in valid.items() if key != "digest"},
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_retrieval_cache(path)
    assert len(loaded) == 1
    assert next(iter(loaded.values())) == record


def test_stale_retrieval_digest_is_ignored_by_planner_and_inventory(
    tmp_path, monkeypatch,
) -> None:
    record = replace(_record(_ladder((8, 8, 8, 2))), digest="0" * 16)
    path = tmp_path / "retrieval.json"
    save_retrieval(record, path)
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    assert cli._embed_retrieval_limits() == {}
    row = next(
        row for row in collect_evidence()["records"] if row["kind"] == "retrieval"
    )
    assert row["usable"] is False
    assert "stale_digest" in row["reasons"]


def test_nondefault_retrieval_digest_is_ignored_by_planner(tmp_path, monkeypatch) -> None:
    record = replace(
        _record(_ladder((8, 8, 8, 2))),
        digest=retrieval_digest(seeds=(1, 2), rung_words=(10, 20, 30, 40)),
    )
    assert record.digest != retrieval_digest()
    path = tmp_path / "retrieval.json"
    save_retrieval(record, path)
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    assert cli._embed_retrieval_limits() == {}


def test_old_retrieval_harness_is_ignored_but_inventory_reports_it(
    tmp_path, monkeypatch,
) -> None:
    record = replace(_record(_ladder((8, 8, 8, 2))), harness="retrieval-v1")
    save_retrieval(record, tmp_path / "retrieval.json")
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    assert cli._embed_retrieval_limits() == {}
    row = next(
        row for row in collect_evidence()["records"] if row["kind"] == "retrieval"
    )
    assert row["usable"] is False
    assert "harness_mismatch" in row["reasons"]


def test_planner_warns_without_changing_candidate() -> None:
    model = ModelSpec(
        "embed", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed"},
    )
    policy = Policy(roles=["embed"])
    ordinary = build_plan(profile(64), [model], policy)
    key = (
        ordinary.services[0].model_id.casefold(),
        ordinary.services[0].quant.casefold(),
        ordinary.services[0].backend.casefold(),
    )
    limits = (
        RetrievalLimit(ordinary.services[0].context // 2, 512, True, 8, 8),
        RetrievalLimit(ordinary.services[0].context // 2, 512, False),
        RetrievalLimit(ordinary.services[0].context // 2, None, None),
    )
    for limit in limits:
        warned = build_plan(
            profile(64),
            [model],
            policy,
            embed_retrieval_limits={key: limit},
        )
        assert len(warned.services) == len(ordinary.services)
        assert warned.services[0].context == ordinary.services[0].context
        assert warned.services[0].memory == ordinary.services[0].memory
        assert warned.services[0].decode_tps == ordinary.services[0].decode_tps
        assert warned.services[0].model_id == ordinary.services[0].model_id
        assert warned.services[0].backend == ordinary.services[0].backend
        assert warned.services[0].quant == ordinary.services[0].quant
        assert any("single-vector" in warning for warning in warned.warnings)


def test_planner_selects_chunk_retrieval_warning_messages() -> None:
    model = ModelSpec(
        "embed", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["embed"], 80.0, "apache", {"hf_gguf": "org/embed"},
    )
    policy = Policy(roles=["embed"])
    ordinary = build_plan(profile(64), [model], policy)
    key = (
        ordinary.services[0].model_id.casefold(),
        ordinary.services[0].quant.casefold(),
        ordinary.services[0].backend.casefold(),
    )
    recovered = build_plan(
        profile(64), [model], policy,
        embed_retrieval_limits={
            key: RetrievalLimit(ordinary.services[0].context // 2, 512, True, 8, 8),
        },
    )
    failed = build_plan(
        profile(64), [model], policy,
        embed_retrieval_limits={
            key: RetrievalLimit(ordinary.services[0].context // 2, 512, False),
        },
    )
    unmeasured = build_plan(
        profile(64), [model], policy,
        embed_retrieval_limits={
            key: RetrievalLimit(ordinary.services[0].context // 2, None, None),
        },
    )
    assert any("recovers 8/8" in warning for warning in recovered.warnings)
    assert any("not a verified remedy" in warning for warning in failed.warnings)
    assert any("Chunk long inputs" in warning for warning in unmeasured.warnings)
