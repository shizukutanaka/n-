from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from nmesh import cli
from nmesh.catalog import ModelSpec
from nmesh.eval import (
    CATEGORIES,
    EXTENDED_CATEGORIES,
    EXTENDED_TASKS,
    GENERATED_TASKS,
    SUITES,
    TASKS,
    EvalRun,
    EvalSummary,
    Task,
    TaskOutcome,
    normalize,
    suite_digest,
)
from nmesh.eval.cache import EvalRecord, load_eval_cache, save_eval
from nmesh.eval.generated import (
    _ADDITIONS,
    _CHAR_WORDS,
    _DATES,
    _ECHO,
    _EMAILS,
    _ITEM_COUNTS,
    _JAPANESE,
    _LETTER_WORDS,
    _LOWERCASE,
    _MAXIMA,
    _PARITY,
    _PEOPLE,
    _PRODUCTS,
    _SUBSTRINGS,
    _SUBTRACTIONS,
    _UPPERCASE,
)
from nmesh.eval.runner import run
from nmesh.eval.stats import (
    fisher_two_sided,
    mcnemar_two_sided,
    min_resolvable_difference,
    wilson_interval,
)
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
    finish_reasons: ClassVar[dict[str, str]] = {}

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        self.__class__.bodies.append(body)
        prompt = body["messages"][0]["content"]
        status, text = self.__class__.responses.get(prompt, (self.__class__.default_status, "ok"))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        payload = json.dumps({
            "choices": [{
                "message": {"content": text},
                "finish_reason": self.__class__.finish_reasons.get(prompt, "stop"),
            }],
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


def test_exact_eval_statistics() -> None:
    assert fisher_two_sided(12, 4, 9, 7) == pytest.approx(0.4578, abs=0.0001)
    assert fisher_two_sided(13, 3, 12, 4) == pytest.approx(1.0)
    assert mcnemar_two_sided(1, 1) == pytest.approx(1.0)
    assert mcnemar_two_sided(6, 0) == pytest.approx(0.03125)
    assert mcnemar_two_sided(0, 0) == pytest.approx(1.0)
    assert min_resolvable_difference(16) == 5 / 16
    assert wilson_interval(12, 16) == pytest.approx((0.505, 0.898), abs=0.0005)


def test_generated_tasks_accept_canonical_and_reject_wrong_answers() -> None:
    expected: dict[str, str] = {}
    for left, right, total in _ADDITIONS:
        expected[f"arithmetic.add.{left}_{right}"] = str(total)
    for left, right, total in _SUBTRACTIONS:
        expected[f"arithmetic.subtract.{left}_{right}"] = str(total)
    for left, right, total in _PRODUCTS:
        expected[f"arithmetic.multiply.{left}_{right}"] = str(total)
    for word in _LETTER_WORDS:
        expected[f"arithmetic.count.{word}"] = str(len(word))
    for word in _UPPERCASE:
        expected[f"instruction.upper.{word}"] = word.upper()
    for word in _LOWERCASE:
        expected[f"instruction.lower.{word}"] = word.lower()
    for word in _ECHO:
        expected[f"instruction.echo.{word}"] = word
    for count in _ITEM_COUNTS:
        expected[f"instruction.items.{count}"] = ", ".join(["cat"] * count)
    for name, age in _PEOPLE:
        expected[f"format.json_person.{name.lower()}"] = json.dumps(
            {"name": name, "age": int(age)}
        )
    for word in _CHAR_WORDS:
        expected[f"format.json_count.{word}"] = json.dumps({"count": len(word)})
    for number, even in _PARITY:
        expected[f"format.json_even.{number}"] = json.dumps({"even": even})
    for index, (_, address, _target) in enumerate(_EMAILS):
        expected[f"extraction.email.{index}"] = address
    for index, (_, iso, _target) in enumerate(_DATES):
        expected[f"extraction.date.{index}"] = iso
    for index, (_, largest) in enumerate(_MAXIMA):
        expected[f"extraction.max.{index}"] = str(largest)
    for index, (_, span, _rivals) in enumerate(_SUBSTRINGS):
        expected[f"extraction.span.{index}"] = span
    for index, (_, address, _target) in enumerate(_EMAILS[:3]):
        expected[f"compliance.email.{index}"] = address
    for index, (_, iso, _target) in enumerate(_DATES[:3]):
        expected[f"compliance.date.{index}"] = iso
    for index, (_, span, _rivals) in enumerate(_SUBSTRINGS[:2]):
        expected[f"compliance.span.{index}"] = span
    for index, (_, required) in enumerate(_JAPANESE):
        expected[f"multilingual.ja.{index}"] = f"\u3053\u308c\u306f{required}\u3067\u3059"

    wrong = {
        "arithmetic": "0",
        "instruction": "Sure! Here is the answer: nope.",
        "format": '{"unexpected": 1}',
        "extraction": "I cannot find it.",
        "compliance": "Sure, here is the value.",
        "multilingual": "It is a cat.",
    }
    assert len(GENERATED_TASKS) == 88
    assert len(TASKS) == 16
    assert len(EXTENDED_TASKS) == 104
    assert SUITES == {"core": TASKS, "extended": EXTENDED_TASKS}
    assert len({task.id for task in EXTENDED_TASKS}) == 104
    assert not {task.id for task in TASKS} & {task.id for task in GENERATED_TASKS}
    assert "compliance" in EXTENDED_CATEGORIES
    assert "compliance" not in CATEGORIES
    assert min_resolvable_difference(len(EXTENDED_TASKS)) == min_resolvable_difference(104)

    for task in GENERATED_TASKS:
        answer = expected[task.id]
        assert task.check(answer), task.id
        assert task.check(f"```\n{answer}\n```"), task.id
        assert not task.check(wrong[task.category]), task.id


def test_value_extraction_and_compliance_grading() -> None:
    email = next(task for task in TASKS if task.id == "extraction.email")
    date = next(task for task in TASKS if task.id == "extraction.date")
    number = next(task for task in TASKS if task.id == "extraction.number")
    assert email.check("The address is nmesh-ops@example.com.")
    assert date.check("The date in YYYY-MM-DD form is: 2024-03-03")
    assert not email.check("Contact wrong@example.com instead.")
    assert not date.check("The date is 2024-03-04.")
    assert number.check("The answer is 236.")
    for index, (sentence, address, _target) in enumerate(_EMAILS):
        task = next(
            task for task in GENERATED_TASKS if task.id == f"extraction.email.{index}"
        )
        assert not task.check(sentence)
        assert task.check(address)
        assert task.check(f"The selected address is {address}.")
        decoy = next(
            match for match in re.findall(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", sentence
            )
            if match != address
        )
        assert not task.check(decoy)
    date_decoys = (
        "2021-06-02",
        "2025-05-06",
        "2020-08-01",
        "2023-10-03",
        "2022-03-18",
    )
    for index, (sentence, iso, _target) in enumerate(_DATES):
        task = next(
            task for task in GENERATED_TASKS if task.id == f"extraction.date.{index}"
        )
        assert not task.check(sentence)
        assert task.check(iso)
        assert task.check(f"The selected date is {iso}.")
        assert not task.check(date_decoys[index])
    for task in GENERATED_TASKS:
        if task.category == "extraction" and task.id.startswith("extraction.span."):
            source = next(
                prompt for prompt, expected, _rivals in _SUBSTRINGS
                if expected in prompt
            )
            assert not task.check(source)
    compliance = next(
        task for task in GENERATED_TASKS if task.id == "compliance.email.0"
    )
    assert compliance.check(_EMAILS[0][1])
    assert not compliance.check(f"The address is {_EMAILS[0][1]}.")


def test_suite_digest_identity() -> None:
    assert suite_digest(TASKS) == suite_digest(TASKS)
    assert suite_digest(TASKS) != suite_digest(EXTENDED_TASKS)
    from nmesh.eval import suite
    original = suite.GRADER_VERSION
    try:
        suite.GRADER_VERSION = original + 1
        assert suite_digest(TASKS) != f"v{original}:" + suite_digest(TASKS).split(":", 1)[1]
    finally:
        suite.GRADER_VERSION = original


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


def test_runner_marks_answerless_truncation_unscorable() -> None:
    tasks = (
        Task("empty", "instruction", "empty", 8, lambda text: text == "yes"),
        Task("cut", "instruction", "cut", 8, lambda text: text == "yes"),
        Task("ok", "format", "ok", 8, lambda text: text == "yes"),
    )
    _EvalHandler.responses = {
        "empty": (200, ""),
        "cut": (200, "ye"),
        "ok": (200, "yes"),
    }
    _EvalHandler.finish_reasons = {"empty": "length", "cut": "length"}
    server = _serve()
    try:
        result = run(tasks, f"http://127.0.0.1:{server.server_address[1]}", "model")
    finally:
        _EvalHandler.finish_reasons = {}
        server.shutdown()
        server.server_close()
    unscorable = {outcome.id: outcome.unscorable for outcome in result.outcomes}
    assert unscorable == {"empty": True, "cut": False, "ok": False}
    assert result.unscorable == 1
    assert result.passed == 1


def test_runner_reasoning_allowance_raises_every_budget() -> None:
    tasks = (Task("one", "instruction", "one", 8, lambda text: text == "yes"),)
    _EvalHandler.responses = {"one": (200, "yes")}
    _EvalHandler.bodies = []
    server = _serve()
    try:
        result = run(
            tasks, f"http://127.0.0.1:{server.server_address[1]}", "model",
            reasoning_allowance=504,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert _EvalHandler.bodies[0]["max_tokens"] == 512
    assert result.reasoning_allowance == 504
    assert result.unscorable == 0


def test_eval_cache_separates_reasoning_allowances(tmp_path) -> None:
    path = tmp_path / "eval.json"
    digest = suite_digest(TASKS)
    strict = EvalRun("model", "f16", "llamacpp", 2, 0, 0.0, {}, [], 1.0,
                     "", "core", digest, 2, 0)
    generous = EvalRun("model", "f16", "llamacpp", 2, 2, 1.0, {}, [], 2.0,
                       "", "core", digest, 0, 504)
    save_eval(strict, path)
    save_eval(generous, path)
    records = load_eval_cache(path)
    assert set(records) == {
        f"model|f16|llamacpp|core|{digest}",
        f"model|f16|llamacpp|core|{digest}|a504",
    }
    assert records[f"model|f16|llamacpp|core|{digest}"].unscorable == 2
    rates = cli._eval_rates(records)
    assert rates[("model", "f16", "llamacpp")].pass_rate == 1.0


def test_eval_rates_drop_runs_with_unscorable_tasks() -> None:
    digest = suite_digest(TASKS)
    records = {
        "only": EvalRecord(
            "model", "f16", "llamacpp", 104, 0, 0.0, {}, 1.0, {}, "", "core", digest, 104, 0,
        ),
    }
    assert cli._eval_rates(records) == {}


def test_eval_cli_reports_unscorable_run(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    result = EvalRun(
        "prior-high", "q4_k_m", "llamacpp", 1, 0, 0.0, {"instruction": 0.0},
        [TaskOutcome("cut", "instruction", False, "", True)],
        3.0,
        unscorable=1,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: result)
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json", "--reasoning-allowance", "504"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["unscorable"] == 1
    assert output["reasoning_allowance"] == 504
    assert "not used as planning evidence" in output["unscorable_note"]
    assert output["failed"] == [{"id": "cut", "output": "", "unscorable": True}]


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
    result = EvalRun(
        "model", "q4_k_m", "llamacpp", 2, 1, 0.5, {"chat": 0.5},
        [TaskOutcome("one", "chat", True, "yes"), TaskOutcome("two", "chat", False, "no")],
        3.0,
        "gguf:2:123:abcdef",
        "core",
        suite_digest(TASKS),
    )
    save_eval(result, path)
    loaded = load_eval_cache(path)
    key = f"model|q4_k_m|llamacpp|core|{suite_digest(TASKS)}"
    assert loaded[key].pass_rate == 0.5
    assert loaded[key].task_results == {
        "one": True, "two": False,
    }
    assert loaded[key].artifact == "gguf:2:123:abcdef"
    assert "outcomes" not in json.loads(path.read_text(encoding="utf-8"))["results"][
        key
    ]
    legacy = {
        "results": {
            "legacy": {
                "model_id": "model",
                "quant": "f16",
                "backend": "llamacpp",
                "n_tasks": 1,
                "passed": 1,
                "pass_rate": 1.0,
                "by_category": {},
                "at": 4.0,
            },
        },
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path)["legacy"].task_results == {}
    assert load_eval_cache(path)["legacy"].artifact == ""
    legacy["results"]["legacy"]["task_results"] = {"one": "yes"}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    legacy["results"]["legacy"]["task_results"] = {}
    legacy["results"]["legacy"]["artifact"] = 1
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    path.write_text("{broken", encoding="utf-8")
    assert load_eval_cache(path) == {}
    assert load_eval_cache(tmp_path / "missing.json") == {}


def test_eval_cache_keeps_suites_separate(tmp_path) -> None:
    path = tmp_path / "eval.json"
    core = EvalRun("model", "f16", "llamacpp", 16, 8, 0.5, {}, [], 1.0,
                   "", "core", suite_digest(TASKS))
    extended = EvalRun("model", "f16", "llamacpp", 104, 60, 60 / 104, {}, [], 2.0,
                       "", "extended", suite_digest(EXTENDED_TASKS))
    save_eval(core, path)
    save_eval(extended, path)
    records = load_eval_cache(path)
    assert len(records) == 2
    assert cli._eval_rates(records)[("model", "f16", "llamacpp")].n_tasks == 104


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
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.25, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
        },
    )
    consistent = build_plan(
        profile(8), models, policy,
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.8, 13, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(0.7, 11, 16, {}),
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
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: result)
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["key"].startswith("prior-high|f16|llamacpp|core|")
    assert output["suite"] == "core"
    assert output["digest"].startswith("v2:")
    assert output["note"].startswith("1-task")
    assert output["pass_rate_ci"] == pytest.approx((0.2065, 1.0), abs=0.0001)
    assert output["min_resolvable_difference"] == 1.0
    assert output["uncertainty_note"].startswith("95% Wilson interval")
    assert output["suite_upgrade_note"].startswith("The 104-task extended suite")
    assert output["config_note"].startswith("This pass rate applies")
    assert output["failed"] == [
        {"id": "failed", "output": "bad", "unscorable": False},
    ]


def test_planner_deduplicates_multi_role_contradiction_warning() -> None:
    models = [replace(model, roles=["chat", "code"]) for model in _quality_models()]
    plan = build_plan(
        profile(8), models, Policy(roles=["chat", "code"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.25, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
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


def test_planner_reports_underpowered_eval_evidence() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(12 / 16, 12, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(13 / 16, 13, 16, {}),
        },
    )
    assert not any("pass rate ranks" in warning for warning in plan.warnings)
    assert any(
        "neither confirmed nor contradicted" in note
        and "104-task extended suite" in note
        for note in plan.warnings
    )


def test_planner_warns_for_significant_eval_gap() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(4 / 16, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
        },
    )
    contradiction = next(item for item in plan.warnings if "pass rate ranks" in item)
    assert "p=" in contradiction
    assert "16 tasks" in contradiction


def test_planner_uses_paired_eval_evidence() -> None:
    models = _quality_models()
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    other = {f"task-{index}": True for index in range(16)}
    significant = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
            ("measured-high", "f16", "llamacpp"): EvalSummary(
                1.0, 16, 16, other,
            ),
        },
    )
    assert any("pass rate ranks" in warning for warning in significant.warnings)

    selected_balanced = {f"task-{index}": index % 2 == 0 for index in range(16)}
    other_balanced = {f"task-{index}": index % 2 == 1 for index in range(16)}
    balanced = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(
                0.5, 8, 16, selected_balanced,
            ),
            ("measured-high", "f16", "llamacpp"): EvalSummary(
                0.5, 8, 16, other_balanced,
            ),
        },
    )
    assert not any("pass rate ranks" in warning for warning in balanced.warnings)
    assert not any("neither confirmed nor contradicted" in note for note in balanced.warnings)


def test_planner_ignores_bare_float_eval_evidence() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): 0.25,
            ("measured-high", "f16", "llamacpp"): 0.875,
        },
    )
    assert not any("pass rate ranks" in warning for warning in plan.warnings)
    assert not any("neither confirmed nor contradicted" in note for note in plan.warnings)


def test_eval_rates_are_keyed_by_configuration(monkeypatch) -> None:
    core_digest = suite_digest(TASKS)
    records = {
        "old": EvalRecord("model", "f16", "llamacpp", 16, 8, 0.5, {}, 1.0,
                          {}, "", "core", core_digest),
        "new": EvalRecord("model", "f16", "llamacpp", 16, 12, 0.75, {}, 2.0,
                          {}, "", "core", core_digest),
        "other": EvalRecord("model", "q4_k_m", "ollama", 16, 13, 0.8125, {}, 1.5,
                            {}, "", "core", core_digest),
    }
    monkeypatch.setattr(cli, "load_eval_cache", lambda: records)
    assert cli._eval_rates() == {
        ("model", "f16", "llamacpp"): EvalSummary(0.75, 12, 16, {}),
        ("model", "q4_k_m", "ollama"): EvalSummary(0.8125, 13, 16, {}),
    }


def test_eval_rates_drops_stale_grader_records() -> None:
    stale = EvalRecord(
        "model", "f16", "llamacpp", 16, 16, 1.0, {}, 3.0,
        {}, "", "core", "v1:stale",
    )
    assert cli._eval_rates({"stale": stale}) == {}


def test_eval_cli_category_filter(monkeypatch) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    captured: list[Task] = []
    result = EvalRun("prior-high", "f16", "llamacpp", 1, 1, 1.0, {"format": 1.0}, [], 3.0)
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: (
        captured.extend(tasks) or result
    ))
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "_print_json", lambda value: None)
    assert cli.main(["eval", "--json", "--categories", "format"]) == 0
    assert captured
    assert {task.category for task in captured} == {"format"}


def test_eval_cli_default_and_compliance_categories(monkeypatch) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    captured: list[Task] = []
    result = EvalRun("prior-high", "f16", "llamacpp", 1, 1, 1.0, {}, [], 3.0)
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: (
        captured.extend(tasks) or result
    ))
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "_print_json", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    assert len(captured) == 16
    captured.clear()
    assert cli.main(["eval", "--json", "--suite", "extended",
                     "--categories", "compliance"]) == 0
    assert len(captured) == 8
    assert {task.category for task in captured} == {"compliance"}


def test_eval_cli_extended_suite(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    captured: list[Task] = []
    result = EvalRun(
        "prior-high", "f16", "llamacpp", 104, 104, 1.0,
        {category: 1.0 for category in EXTENDED_CATEGORIES}, [], 3.0,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: (
        captured.extend(tasks) or result
    ))
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json", "--suite", "extended"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(captured) == 104
    assert output["suite"] == "extended"
    assert output["n_tasks"] == 104
    assert output["suite_upgrade_note"] is None


def test_eval_divergence_reports_disagreeing_tasks() -> None:
    result = EvalRun(
        "model", "f16", "llamacpp", 2, 1, 0.5, {},
        [
            TaskOutcome("arithmetic.subtract", "arithmetic", True, "747"),
            TaskOutcome("multilingual.ja_translate", "multilingual", False, "sleeping"),
        ],
        2.0,
    )
    records = {
        "other": EvalRecord(
            "model", "f16", "ollama", 2, 1, 0.5, {}, 1.0,
            {
                "arithmetic.subtract": False,
                "multilingual.ja_translate": True,
            },
        ),
    }
    assert cli._eval_divergence(result, records) == [{
        "config": "f16|ollama",
        "artifact": None,
        "pass_rate": 0.5,
        "compared": 2,
        "disagreeing": [
            "arithmetic.subtract",
            "multilingual.ja_translate",
        ],
    }]


def test_eval_cli_warns_when_artifact_changes(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    result = EvalRun(
        "prior-high", "q4_k_m", "llamacpp", 1, 1, 1.0,
        {"instruction": 1.0},
        [TaskOutcome("one", "instruction", True, "yes")],
        3.0,
    )
    previous = EvalRecord(
        "prior-high", service_plan.services[0].quant, service_plan.services[0].backend,
        1, 1, 1.0, {}, 2.0, {}, "old-artifact", "core", suite_digest(TASKS),
    )
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "runtime_status", lambda: RuntimeStatus(
        True, [{"service": "chat", "running": True}],
    ))
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: result)
    monkeypatch.setattr(cli, "service_fingerprint", lambda backend, model_ref: "new-artifact")
    monkeypatch.setattr(
        cli,
        "load_eval_cache",
        lambda: {
            f"prior-high|{service_plan.services[0].quant}|llamacpp|core|"
            f"{suite_digest(TASKS)}": previous
        },
    )
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["artifact"] == "new-artifact"
    assert output["artifact_warning"].count("old-artifact") == 1
    assert output["artifact_warning"].count("new-artifact") == 1
