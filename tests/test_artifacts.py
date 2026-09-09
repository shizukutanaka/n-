from __future__ import annotations

from nmesh.artifacts import artifact_key, load_cache, record, save_cache


def test_artifact_cache_round_trip_and_merge(tmp_path) -> None:
    path = tmp_path / "artifacts.json"
    save_cache({"org/model|q4_k_m": 123}, path)
    record("org/model", "q2_k", 456, path)

    assert artifact_key("org/model", "q4_k_m") == "org/model|q4_k_m"
    assert load_cache(path) == {
        "org/model|q4_k_m": 123,
        "org/model|q2_k": 456,
    }


def test_artifact_cache_tolerates_missing_and_corrupt_files(tmp_path) -> None:
    path = tmp_path / "artifacts.json"
    assert load_cache(path) == {}
    path.write_text("{", encoding="utf-8")
    assert load_cache(path) == {}
