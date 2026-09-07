from dataclasses import asdict

from nmesh.bench import benchmark_key
from nmesh.orchestrate.measure import RoleIdentity
from nmesh.spec import (
    ALLOW,
    KIND_NGRAM,
    ClassEvidence,
    SpecConfig,
    SpecRecord,
    decide,
    load_cache,
    save,
)


def _record(*, identical: bool = True, speedup: float = 1.2) -> SpecRecord:
    target = RoleIdentity("target", "q4_k_m", "llamacpp", "artifact")
    return SpecRecord(
        target=target,
        spec=SpecConfig(kind=KIND_NGRAM, n_max=3),
        engine="llama.cpp",
        harness="spec-v1",
        repeats=2,
        classes=(
            ClassEvidence(
                name="copy",
                speedup=speedup,
                identical=identical,
                reference_tps=10.0,
                candidate_tps=speedup * 10.0,
                acceptance=0.8,
            ),
        ),
        at=1.0,
    )


def test_benchmark_key_preserves_none_and_separates_spec() -> None:
    base = benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0)
    assert base == "m|q4_k_m|llamacpp|cpu|0"
    assert benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0, spec="ngram") != base
    assert benchmark_key("m", "q4_k_m", "llamacpp", "cpu", 0, spec="draft") != base


def test_spec_record_save_load_round_trip(tmp_path) -> None:
    record = _record()
    path = tmp_path / "spec.json"
    save(record, path)
    loaded = load_cache(path)
    assert len(loaded) == 1
    restored = next(iter(loaded.values()))
    assert asdict(restored) == asdict(record)
    assert decide(restored) == (ALLOW, ALLOW)
