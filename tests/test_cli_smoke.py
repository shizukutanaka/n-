from __future__ import annotations

import argparse
import types

import pytest

from nmesh import cli


def _top_level_commands(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    choices: list[dict[str, argparse.ArgumentParser]] = []
    original = cli.argparse.ArgumentParser

    def recording_parser(*args, **kwargs):
        parser = original(*args, **kwargs)
        original_add_subparsers = parser.add_subparsers

        def record_subparsers(*subparser_args, **subparser_kwargs):
            action = original_add_subparsers(*subparser_args, **subparser_kwargs)
            if not choices:
                choices.append(action.choices)
            return action

        parser.add_subparsers = record_subparsers
        return parser

    isolated_argparse = types.ModuleType("argparse_proxy")
    isolated_argparse.__dict__.update(vars(cli.argparse))
    isolated_argparse.ArgumentParser = recording_parser
    monkeypatch.setattr(cli, "argparse", isolated_argparse)
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert choices
    return tuple(choices[0])


def test_every_top_level_command_help_exits_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for command in _top_level_commands(monkeypatch):
        with pytest.raises(SystemExit) as error:
            cli.main([command, "--help"])
        assert error.value.code == 0


def test_numeric_flags_reject_negative_and_zero_values() -> None:
    for argv in (
        ["plan", "--context", "-1"],
        ["plan", "--parallel-slots", "0"],
        ["plan", "--allow-download-gb", "-0.5"],
        ["up", "--spec-n-max", "0"],
        ["up", "--sleep-idle-seconds", "-1"],
        ["up", "--cache-reuse", "-1"],
        ["up", "--min-decode-tps", "-1"],
        ["serve", "--port", "0"],
        ["run", "--port", "-1", "prompt"],
        ["bench", "--tokens", "0"],
        ["jobs", "--port", "0"],
        ["orchestrate", "measure", "--reasoning-allowance", "-1"],
        ["orchestrate", "measure", "--repeats", "0"],
        ["spec", "measure", "--kind", "ngram", "--repeats", "-1"],
        ["spec", "measure", "--kind", "ngram", "--n-max", "0"],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(list(argv))
        assert error.value.code == 2


def test_version_reports_package_and_evidence_versions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "nmesh 0.1.0" in output
    assert f"bench={cli.BENCH_HARNESS_VERSION}" in output
    assert "probe_rules=" in output
