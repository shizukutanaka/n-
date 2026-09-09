from types import SimpleNamespace

import pytest

from nmesh import cli


def _failed_plan() -> SimpleNamespace:
    return SimpleNamespace(
        services=[object()],
        runnable=False,
        warnings=["No runnable model found for role embed"],
        install_hints=[],
    )


def test_plan_failure_prints_warnings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _failed_plan())

    assert cli.main(["plan"]) == 1

    captured = capsys.readouterr()
    assert "plan produced no runnable services" in captured.err
    assert "role embed" in captured.err


def test_plan_failure_json_includes_plan(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_make_plan", lambda _args: _failed_plan())
    monkeypatch.setattr(
        cli,
        "_plan_json_data",
        lambda _plan: {
            "services": [{"name": "chat"}],
            "runnable": False,
            "warnings": ["No runnable model found for role embed"],
        },
    )

    assert cli.main(["plan", "--json"]) == 1

    captured = capsys.readouterr()
    assert '"services"' in captured.out
    assert '"runnable": false' in captured.out
    assert "role embed" in captured.out


@pytest.mark.parametrize("command", ["up", "serve"])
def test_runtime_commands_accept_roles(command, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "--roles", "chat", "--help"])

    assert raised.value.code == 0
    assert "--roles" in capsys.readouterr().out
