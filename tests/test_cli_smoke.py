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


def test_port_flag_rejects_out_of_range_values() -> None:
    """Port 0 silently binds an ephemeral port; out-of-range values fail
    inside uvicorn instead of at parse time."""
    for port in ("0", "-1", "65536"):
        with pytest.raises(SystemExit) as error:
            cli.main(["status", "--port", port])
        assert error.value.code == 2
