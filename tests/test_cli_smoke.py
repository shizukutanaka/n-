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


def test_plan_rejects_negative_min_decode_tps_and_download_budget() -> None:
    for flag in ("--min-decode-tps", "--allow-download-gb"):
        for command in ("plan", "up"):
            with pytest.raises(SystemExit) as error:
                cli.main([command, flag, "-1"])
            assert error.value.code == 2
        with pytest.raises(SystemExit) as error:
            cli.main(["plan", flag, "nan"])
        assert error.value.code == 2


def test_plan_accepts_zero_decode_floor_and_download_budget() -> None:
    # Zero is a meaningful bound (gate off / warn on any download); only
    # negatives are meaningless.
    assert cli._non_negative_float("0") == 0.0
    assert cli._non_negative_float("0.5") == 0.5
    with pytest.raises(argparse.ArgumentTypeError):
        cli._non_negative_float("-0.1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli._non_negative_float("nan")
