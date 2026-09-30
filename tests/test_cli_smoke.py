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


def test_orchestrate_measure_rejects_negative_reasoning_allowance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["orchestrate", "measure", "--reasoning-allowance", "-1"])
    assert error.value.code == 2
    assert "reasoning-allowance" in capsys.readouterr().err


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


@pytest.mark.parametrize("port", ["0", "-1", "65536", "99999"])
def test_port_flag_rejects_out_of_range(
    port: str, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in (
        ["jobs", "--port", port],
        ["run", "hi", "--port", port],
        ["status", "--port", port],
        ["up", "--port", port, "--dry-run"],
        ["serve", "--port", port],
        ["reload", "--port", port],
        ["unload", "--port", port],
        ["down", "--port", port],
        ["autostart", "--port", port],
    ):
        with pytest.raises(SystemExit) as error:
            cli.main(command)
        assert error.value.code == 2
        assert "between 1 and 65535" in capsys.readouterr().err


def test_port_flag_accepts_valid_port(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["jobs", "--port", "18000"]) == 1
    assert "reachable" in capsys.readouterr().err


def test_watch_rejects_unknown_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli.httpx,
        "Client",
        lambda *_args, **_kwargs: pytest.fail("unknown source must fail before any fetch"),
    )
    assert cli.main(["watch", "--sources", "zenn,bogus"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "bogus" in captured.err


def test_console_renders_bracket_text_verbatim(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._console().print("warn[zzz]here and [not-a-style]tag")
    output = capsys.readouterr().out
    assert "warn[zzz]here" in output
    assert "[not-a-style]tag" in output


def test_console_style_kwarg_does_not_leak_markup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._console().print("- path/to/x[0].log", style="yellow")
    output = capsys.readouterr().out
    assert "- path/to/x[0].log" in output
    assert "[yellow]" not in output
