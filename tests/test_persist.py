"""Regression tests for nmesh.persist — unreadable state files warn once."""

from __future__ import annotations

import sys
from pathlib import Path

from nmesh import persist
from nmesh.artifacts import load_cache as load_artifact_cache


def test_unreadable_file_warns_once(tmp_path: Path, capsys) -> None:
    persist._warned.clear()
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")

    assert persist.read_json_file(bad) is None
    assert "broken.json" in capsys.readouterr().err
    persist.read_json_file(bad)
    assert capsys.readouterr().err == ""


def test_missing_file_is_silent(tmp_path: Path, capsys) -> None:
    persist._warned.clear()
    assert persist.read_json_file(tmp_path / "absent.json") is None
    assert capsys.readouterr().err == ""


def test_loaders_surface_the_warning(tmp_path: Path, capsys) -> None:
    # A corrupt artifacts.json used to be silently discarded — every recorded
    # GGUF then failed the recorded-size integrity check and was re-downloaded.
    persist._warned.clear()
    bad = tmp_path / "artifacts.json"
    bad.write_text("not json at all", encoding="utf-8")
    assert load_artifact_cache(bad) == {}
    assert "artifacts.json" in capsys.readouterr().err


def test_unreadable_directory_path_warns(tmp_path: Path, capsys) -> None:
    persist._warned.clear()
    assert persist.read_json_file(tmp_path) is None
    assert capsys.readouterr().err != ""


def test_deeply_nested_file_warns_instead_of_recursing(tmp_path: Path, capsys) -> None:
    # json.loads raises RecursionError (a RuntimeError, not ValueError) on
    # nesting past the interpreter limit — it used to escape the catch and
    # crash the caller with a traceback instead of warning once.
    persist._warned.clear()
    bad = tmp_path / "deep.json"
    depth = sys.getrecursionlimit() * 100
    bad.write_text("[" * depth + "]" * depth, encoding="utf-8")
    assert persist.read_json_file(bad) is None
    assert "deep.json" in capsys.readouterr().err
