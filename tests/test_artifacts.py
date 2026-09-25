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


def test_default_artifact_cache_path_stays_in_session_home(tmp_path) -> None:
    """CACHE_PATH is bound at import time; the shared conftest must rebind it
    to the per-test NMESH_HOME or default-path calls touch the real home."""
    record("org/model", "q4_k_m", 789)
    assert (tmp_path / "artifacts.json").is_file()
    assert load_cache() == {"org/model|q4_k_m": 789}
