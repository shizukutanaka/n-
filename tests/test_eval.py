from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from nmesh import cli
from nmesh.catalog import ModelSpec
from nmesh.eval import CATEGORIES, TASKS, EvalRun, Task, TaskOutcome, normalize
from nmesh.eval.cache import EvalRecord, load_eval_cache, save_eval
from nmesh.eval.runner import run
from nmesh.planner import Policy, build_plan
from nmesh.runtime import RuntimeStatus

from .test_planner import profile


def test_normalize_and_representative_verifiers() -> None:
    assert normalize('```text\n"Acknowledged."\n```') == "Acknowledged"
    checks = {
        "instruction.echo": ("Acknowledged", "Nope"),
        "format.json_city": ('{"city":"Tokyo","country":"Japan"}', '{"city":"Osaka"}'),
        "arithmetic.add": ("42", "41"),
        "multilingual.ja_translate": ("猫は眠ります。", "The cat sleeps."),
    }
    for task_id, (correct, wrong) in checks.items():
        task = next(task for task in TASKS if task.id == task_id)
        assert task.check(correct)
        assert not task.check(wrong)
    arithmetic = next(task for task in TASKS if task.id == "arithmetic.add")
    assert arithmetic.check("17 + 25 = 42")
    assert not arithmetic.check("17 + 25 = 41")


class _EvalHandler(BaseHTTPRequestHandler):
    responses: ClassVar[dict[str, tuple[int, str]]] = {}
    default_status: ClassVar[int] = 200
    bodies: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.bodies.append(body)
        prompt = body["messages"][0]["content"]
        status, text = self.__class__.responses.get(prompt, (self.__class__.default_status, "ok"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        payload = json.dumps({
            "choices": [{"message": {"content": text}}],
        }).encode()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def _serve() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EvalHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_runner_all_pass_and_category_rates() -> None:
    tasks = (
        Task("one", "instruction", "one", 8, lambda text: text == "yes"),
        Task("two", "format", "two", 8, lambda text: text == "yes"),
    )
    _EvalHandler.responses = {"one": (200, "yes"), "two": (200, "yes")}
    _EvalHandler.bodies = []
    server = _serve()
    try:
        result = run(tasks, f"http://127.0.0.1:{server.server_address[1]}", "model")
    finally:
        server.shutdown()
        server.server_close()
    assert result.pass_rate == 1.0
    assert result.by_category == {"instruction": 1.0, "format": 1.0}
    assert all(outcome.passed for outcome in result.outcomes)
    assert _EvalHandler.bodies[0]["model"] == "model"
    assert _EvalHandler.bodies[0]["temperature"] == 0
    assert _EvalHandler.bodies[0]["stream"] is False
    assert _EvalHandler.bodies[0]["max_tokens"] == 8
    assert _EvalHandler.bodies[0]["messages"] == [{"role": "user", "content": "one"}]


def test_runner_mixed_and_transport_failure_continue() -> None:
    tasks = (
        Task("one", "instruction", "one", 8, lambda text: text == "yes"),
        Task("two", "instruction", "two", 8, lambda text: text == "yes"),
        Task("three", "format", "three", 8, lambda text: text == "yes"),
    )
    _EvalHandler.responses = {
        "one": (200, "yes"),
        "two": (200, "no"),
        "three": (500, "failure"),
    }
    server = _serve()
    try:
        result = run(tasks, f"http://127.0.0.1:{server.server_address[1]}", "model")
    finally:
        server.shutdown()
        server.server_close()
    assert result.passed == 1
    assert result.by_category == {"instruction": 0.5, "format": 0.0}
    assert result.outcomes[-1].passed is False
    assert result.outcomes[-1].output


def test_runner_all_transport_failures_raise() -> None:
    tasks = (Task("one", "instruction", "one", 8, lambda text: True),)
    _EvalHandler.responses = {}
    _EvalHandler.default_status = 500
    server = _serve()
    try:
        with pytest.raises(RuntimeError):
            run(tasks, f"http://127.0.0.1:{server.server_address[1]}", "model")
    finally:
        _EvalHandler.default_status = 200
        server.shutdown()
        server.server_close()


def test_eval_cache_round_trip_and_corrupt_file(tmp_path) -> None:
    path = tmp_path / "eval.json"
    result = EvalRun("model", "q4_k_m", "llamacpp", 2, 1, 0.5, {"chat": 0.5}, [], 3.0)
    save_eval(result, path)
    loaded = load_eval_cache(path)
    assert loaded["model|q4_k_m|llamacpp"].pass_rate == 0.5
    assert "outcomes" not in json.loads(path.read_text(encoding="utf-8"))["results"][
        "model|q4_k_m|llamacpp"
    ]
    path.write_text("{broken", encoding="utf-8")
    assert load_eval_cache(path) == {}
    assert load_eval_cache(tmp_path / "missing.json") == {}


def _quality_models() -> list[ModelSpec]:
    return [
        ModelSpec("prior-high", "test", 500_000_000, 24, 16, 2, 64, 1024,
                  4096, ["chat"], 90.0, "test", {"hf_gguf": "prior-high"}),
        ModelSpec("measured-high", "test", 500_000_000, 24, 16, 2, 64, 1024,
                  4096, ["chat"], 70.0, "test", {"hf_gguf": "measured-high"}),
    ]


def test_planner_quality_warning_and_unchanged_selection() -> None:
    models = _quality_models()
    policy = Policy(roles=["chat"])
    ordinary = build_plan(profile(8), models, policy)
    contradictory = build_plan(
        profile(8), models, policy,
        eval_cache={
            ("prior-high", "f16", "llamacpp"): 0.4,
            ("measured-high", "f16", "llamacpp"): 0.8,
        },
    )
    consistent = build_plan(
        profile(8), models, policy,
        eval_cache={
            ("prior-high", "f16", "llamacpp"): 0.8,
            ("measured-high", "f16", "llamacpp"): 0.7,
        },
    )
    assert contradictory.services[0].model_id == ordinary.services[0].model_id
    assert consistent.services[0].model_id == ordinary.services[0].model_id
    assert any("measured-high" in warning and "prior-high" in warning
               for warning in contradictory.warnings)
    assert not any("pass rate ranks" in warning for warning in consistent.warnings)
    assert any("unverified" in warning for warning in ordinary.warnings)


def test_eval_cli_json_includes_note(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    result = EvalRun(
        "prior-high", "q4_k_m", "llamacpp", 1, 1, 1.0,
        {category: 0.75 for category in CATEGORIES},
        [TaskOutcome("failed", "instruction", False, "bad")],
        3.0,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: RuntimeStatus(
        True, [{"service": "chat", "running": True}],
    ))
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref: result)
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["key"] == "prior-high|f16|llamacpp"
    assert output["note"].startswith("1-task")
    assert output["config_note"].startswith("This pass rate applies")
    assert output["failed"] == [{"id": "failed", "output": "bad"}]


def test_planner_deduplicates_multi_role_contradiction_warning() -> None:
    models = [replace(model, roles=["chat", "code"]) for model in _quality_models()]
    plan = build_plan(
        profile(8), models, Policy(roles=["chat", "code"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): 0.4,
            ("measured-high", "f16", "llamacpp"): 0.8,
        },
    )
    contradictions = [warning for warning in plan.warnings if "pass rate ranks" in warning]
    assert len(contradictions) == 1


def test_planner_warns_when_eval_configuration_does_not_match() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "q4_k_m", "llamacpp"): 0.4,
            ("measured-high", "f16", "llamacpp"): 0.8,
        },
    )
    assert any("prior-high" in warning and "configuration" in warning
               for warning in plan.warnings)
    assert not any("pass rate ranks" in warning for warning in plan.warnings)


def test_eval_rates_are_keyed_by_configuration(monkeypatch) -> None:
    records = {
        "old": EvalRecord("model", "f16", "llamacpp", 16, 8, 0.5, {}, 1.0),
        "new": EvalRecord("model", "f16", "llamacpp", 16, 12, 0.75, {}, 2.0),
        "other": EvalRecord("model", "q4_k_m", "ollama", 16, 13, 0.8125, {}, 1.5),
    }
    monkeypatch.setattr(cli, "load_eval_cache", lambda: records)
    assert cli._eval_rates() == {
        ("model", "f16", "llamacpp"): 0.75,
        ("model", "q4_k_m", "ollama"): 0.8125,
    }


def test_eval_cli_category_filter(monkeypatch) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    captured: list[Task] = []
    result = EvalRun("prior-high", "f16", "llamacpp", 1, 1, 1.0, {"format": 1.0}, [], 3.0)
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref: (
        captured.extend(tasks) or result
    ))
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "_print_json", lambda value: None)
    assert cli.main(["eval", "--json", "--categories", "format"]) == 0
    assert captured
    assert {task.category for task in captured} == {"format"}
