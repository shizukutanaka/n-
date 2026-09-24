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


def test_models_local_table_headers_localized(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "fake.gguf").write_bytes(b"junk")
    monkeypatch.setattr(cli.i18n, "lang", lambda: "ja")
    monkeypatch.setattr(cli, "load_plan", lambda: None)
    args = argparse.Namespace(models_command="local", json=False)
    assert cli._models(args) == 0
    out = capsys.readouterr().out
    assert "パス" in out and "計画済み" in out
