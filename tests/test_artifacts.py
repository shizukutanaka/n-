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


def test_artifact_cache_default_path_follows_nmesh_home(monkeypatch, tmp_path) -> None:
    home = tmp_path / "custom-home"
    home.mkdir()
    (home / "artifacts.json").write_text(
        '{"org/model|q4_k_m": 42}', encoding="utf-8"
    )
    monkeypatch.setenv("NMESH_HOME", str(home))
    # The module-level CACHE_PATH was bound at import time; load/save must
    # resolve nmesh_home() lazily rather than touching the real ~/.nmesh.
    assert load_cache() == {"org/model|q4_k_m": 42}
    assert save_cache({"k": 1}).parent == home
