from __future__ import annotations

import os
import subprocess

from nmesh.probe.caps import llamacpp_caps, parse_help

HELP_EXCERPT = """\
-ngl,  --gpu-layers, --n-gpu-layers N    GPU layers
--spec-type none,draft-simple,ngram-simple
                                        prose mentioning --parallel must be ignored
--pooling [on|off|auto]                  pooling
"""


def test_parse_help_keeps_aliases_and_drops_values_and_continuations() -> None:
    flags, version = parse_help(HELP_EXCERPT)
    assert flags == frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers",
                               "--spec-type", "--pooling"})
    assert version is None


def test_caps_cache_hits_and_invalidates_on_size_and_mtime(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"one")
    cache = tmp_path / "caps.json"
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if "--list-devices" in command:
            return subprocess.CompletedProcess(command, 0, "Available devices:\n  (none)\n", "")
        return subprocess.CompletedProcess(command, 0, "-np, --parallel N  slots\nversion 1\n", "")

    monkeypatch.setattr("nmesh.probe.caps.subprocess.run", run)
    first = llamacpp_caps(str(binary), cache)
    second = llamacpp_caps(str(binary), cache)
    assert first == second
    assert len(calls) == 3
    assert first.gpu_devices == ()
    binary.write_bytes(b"two-two")
    llamacpp_caps(str(binary), cache)
    assert len(calls) == 6
    os.utime(binary, ns=(binary.stat().st_atime_ns, binary.stat().st_mtime_ns + 10_000_000))
    llamacpp_caps(str(binary), cache)
    assert len(calls) == 9
    assert cache.exists()


def test_caps_missing_binary_is_unknown(tmp_path) -> None:
    assert llamacpp_caps(str(tmp_path / "missing"), tmp_path / "caps.json") is None
