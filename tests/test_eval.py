from __future__ import annotations

import inspect
import json
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import httpx
import pytest

from nmesh import cli
from nmesh.bench.runner import measure
from nmesh.catalog import ModelSpec
from nmesh.eval import (
    CATEGORIES,
    EXTENDED_CATEGORIES,
    EXTENDED_TASKS,
    GENERATED_TASKS,
    HARD_SUITE_TASKS,
    SUITES,
    TASKS,
    EvalRun,
    EvalSummary,
    Task,
    TaskOutcome,
    needle_tasks,
    normalize,
    padded_prompt,
    suite_digest,
)
from nmesh.eval.cache import EvalRecord, eval_key, load_eval_cache, save_eval
from nmesh.eval.context import (
    ContextRecord,
    FamilyResult,
    load_context_cache,
    save_context,
)
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
from nmesh.eval.hard import HARD_TASKS
from nmesh.eval.runner import run
from nmesh.eval.stats import (
    fisher_two_sided,
    mcnemar_two_sided,
    min_discordant_for_significance,
    min_discordant_imbalance,
    min_resolvable_difference,
    wilson_interval,
)
from nmesh.eval.suite import _bool_value, _only_email
from nmesh.planner import Policy, build_plan
from nmesh.runtime import RuntimeStatus
from nmesh.telemetry import COMPARABLE_PROMPT_TOKENS

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
    assert "cache_prompt" not in _EvalHandler.bodies[0]
    assert result.cache_prompt is None


def test_runner_sends_cache_prompt_when_configured() -> None:
    tasks = (Task("one", "instruction", "one", 8, lambda text: text == "yes"),)
    _EvalHandler.responses = {"one": (200, "yes")}
    _EvalHandler.bodies = []
    server = _serve()
    try:
        result = run(
            tasks,
            f"http://127.0.0.1:{server.server_address[1]}",
            "model",
            cache_prompt=False,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert _EvalHandler.bodies[0]["cache_prompt"] is False
    assert result.cache_prompt is False


def test_runner_reports_each_outcome_via_callback() -> None:
    tasks = (
        Task("one", "instruction", "one", 8, lambda text: text == "yes"),
        Task("two", "format", "two", 8, lambda text: text == "yes"),
    )
    _EvalHandler.responses = {"one": (200, "yes"), "two": (200, "yes")}
    _EvalHandler.bodies = []
    seen: list[str] = []
    server = _serve()
    try:
        result = run(
            tasks,
            f"http://127.0.0.1:{server.server_address[1]}",
            "model",
            on_outcome=lambda outcome: seen.append(outcome.id),
        )
    finally:
        server.shutdown()
        server.server_close()
    assert seen == ["one", "two"]
    assert len(result.outcomes) == 2


def test_runner_marks_transport_failures_and_derives_timeout(monkeypatch) -> None:
    calls: list[float] = []
    value_checks: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {
                "choices": [{
                    "message": {"content": "yes"},
                    "finish_reason": "stop",
                }],
            }

    def post(self, url, *, json, timeout):
        calls.append(timeout)
        if json["messages"][0]["content"] == "transport":
            raise httpx.ReadTimeout("timed out")
        return Response()

    monkeypatch.setattr(httpx.Client, "post", post)
    tasks = (
        Task(
            "transport", "instruction", "transport", 48, lambda text: True,
            value_check=lambda text: value_checks.append(text) or True,
        ),
        Task(
            "healthy", "instruction", "healthy", 48, lambda text: text == "yes",
            value_check=lambda text: value_checks.append(text) or True,
        ),
    )
    result = run(
        tasks, "http://example.test", "model",
        reasoning_allowance=512,
    )
    outcome = result.outcomes[0]
    assert calls == [310.0, 310.0]
    assert outcome.failure_kind == "transport"
    assert outcome.value_passed is None
    assert not outcome.passed
    assert not outcome.unscorable
    assert result.transport_errors == 1
    assert value_checks == ["yes"]


def test_runner_depth_timeout_and_served_prompt_tokens(monkeypatch) -> None:
    calls: list[float] = []

    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {
                "choices": [{
                    "message": {"content": "yes"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 15000},
            }

    def post(self, url, *, json, timeout):
        calls.append(timeout)
        assert len(json["messages"][0]["content"]) > len("question")
        return Response()

    monkeypatch.setattr(httpx.Client, "post", post)
    task = Task("depth", "instruction", "question", 8, lambda text: text == "yes")
    result = run((task,), "http://example.test", "model", depth=15000)
    assert calls == [30.0 + 8 / 2.0 + 750.0]
    assert result.depth == 15000
    assert result.prompt_tokens_max == 15000


def test_padded_prompt_is_deterministic_and_grows() -> None:
    short = padded_prompt("Answer this.", 64, "same")
    repeat = padded_prompt("Answer this.", 64, "same")
    other = padded_prompt("Answer this.", 64, "other")
    longer = padded_prompt("Answer this.", 256, "same")
    assert short == repeat
    assert short != other
    assert short.endswith("Answer this.")
    assert len(longer) > len(short)


def test_bench_nominal_reference_is_below_real_depth_ceiling() -> None:
    nominal = inspect.signature(measure).parameters["prefill_tokens"].default
    assert isinstance(nominal, int)
    # Bench's nominal 512 token parameter tokenises to roughly 336 real tokens.
    assert nominal < COMPARABLE_PROMPT_TOKENS


def test_needle_tasks_have_both_positions_and_validated_checkers() -> None:
    tasks = needle_tasks(4096, "core")
    assert len(tasks) == 24
    assert {task.category for task in tasks} == {
        "context.literal", "context.latent", "context.multi", "context.update",
    }
    assert {
        task.id.split(".")[2]
        for task in tasks
        if task.category not in {"context.multi", "context.update"}
    } == {
        "p10", "p90",
    }
    for task in tasks:
        if task.category == "context.literal":
            answer = re.search(r"is ([0-9a-f]{6})\.", task.prompt)
            assert answer is not None
            assert task.check(answer.group(1))
            case_id = re.search(r"case ([0-9a-f]{8})", task.prompt).group(1)
            assert task.check(
                f"The access code for case {case_id} is {answer.group(1)}.",
            )
            assert not task.check("000000")
            assert not task.check(case_id[:6])
        elif task.category == "context.latent":
            answer = re.search(
                r"([A-Z][a-z]+) spent the whole quarter working out of "
                r"([A-Z][a-z]+)\.",
                task.prompt,
            )
            assert answer is not None
            city = answer.group(2)
            country = next(country_name for _city, country_name in (
                ("Osaka", "Japan"), ("Lyon", "France"), ("Bergen", "Norway"),
                ("Cusco", "Peru"), ("Perth", "Australia"), ("Split", "Croatia"),
            ) if _city == city)
            assert task.check(answer.group(1) + "!")
            assert task.check(f"The person who spent the quarter in {country} is {answer.group(1)}.")
            assert not task.check("Wrong")
            other = next(item for item in ("Marta", "Devrim", "Ines", "Kwame")
                         if item != answer.group(1))
            assert not task.check(
                f"The person who spent the quarter in {country} is "
                f"{answer.group(1)} and {other}.",
            )
            question = task.prompt.rsplit("Answer the question below.\n\n", 1)[-1]
            assert city in task.prompt
            assert country in question
            assert city not in question
        elif task.category == "context.multi":
            codes = re.findall(r"is ([0-9a-f]{6})\.", task.prompt)
            assert len(codes) == 4
            assert task.check(",".join(codes))
            assert task.check(", ".join(codes))
            case_ids = re.findall(r"case ([0-9a-f]{8})", task.prompt)[:4]
            assert task.check(" ".join(
                f"The access code for case {case_id} is {code}."
                for case_id, code in zip(case_ids, codes, strict=True)
            ))
            assert not task.check(",".join(reversed(codes)))
            assert not task.check(",".join(codes[:3]))
        elif task.category == "context.update":
            codes = re.findall(r"(?:is|to) ([0-9a-f]{6})\.", task.prompt)
            assert len(codes) == 2
            assert task.check(codes[1])
            case_id = re.search(r"case ([0-9a-f]{8})", task.prompt).group(1)
            assert task.check(
                f"The current access code for case {case_id} is {codes[1]}.",
            )
            assert not task.check(codes[0])
            assert not task.check(case_id[:6])
        assert not task.check("")


def test_needle_tasks_have_paired_control_content() -> None:
    control = needle_tasks(0, "core")
    deep = needle_tasks(8192, "core")
    repeat = needle_tasks(0, "core")
    assert [(task.id, task.prompt) for task in control] == [
        (task.id, task.prompt) for task in repeat
    ]
    assert [task.id for task in control] == [task.id for task in deep]
    for control_task, deep_task in zip(control, deep):
        assert control_task.category == deep_task.category
        if control_task.category == "context.multi":
            pattern = r"is ([0-9a-f]{6})\."
            control_answers = re.findall(pattern, control_task.prompt)
            deep_answers = re.findall(pattern, deep_task.prompt)
            assert control_answers == deep_answers
        elif control_task.category == "context.literal":
            pattern = r"is ([0-9a-f]{6})\."
            assert re.search(pattern, control_task.prompt).group(1) == re.search(
                pattern, deep_task.prompt,
            ).group(1)
        elif control_task.category == "context.latent":
            pattern = r"([A-Z][a-z]+) spent the whole quarter"
            assert re.search(pattern, control_task.prompt).group(1) == re.search(
                pattern, deep_task.prompt,
            ).group(1)
        else:
            pattern = r"(?:is|to) ([0-9a-f]{6})\."
            control_codes = re.findall(pattern, control_task.prompt)
            deep_codes = re.findall(pattern, deep_task.prompt)
            assert control_codes == deep_codes
            assert "Reference material:" not in control_task.prompt
            assert deep_task.prompt.index(control_codes[0]) < deep_task.prompt.index(
                "Correction:",
            )


def test_context_probe_grader_rules_change_digest() -> None:
    tasks = needle_tasks(128, "core")
    legacy = tuple(replace(task, rule="") for task in tasks)
    assert suite_digest(tasks) != suite_digest(legacy)


def test_runner_explicit_timeout_is_used_for_every_request(monkeypatch) -> None:
    calls: list[float] = []

    class Response:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {"choices": [{
                "message": {"content": "yes"},
                "finish_reason": "stop",
            }]}

    def post(self, url, *, json, timeout):
        calls.append(timeout)
        return Response()

    monkeypatch.setattr(httpx.Client, "post", post)
    tasks = (
        Task("one", "instruction", "one", 48, lambda text: True),
        Task("two", "instruction", "two", 8, lambda text: True),
    )
    run(tasks, "http://example.test", "model", timeout=5.0)
    assert calls == [5.0, 5.0]


def test_exact_eval_statistics() -> None:
    assert fisher_two_sided(12, 4, 9, 7) == pytest.approx(0.4578, abs=0.0001)
    assert fisher_two_sided(13, 3, 12, 4) == pytest.approx(1.0)
    assert mcnemar_two_sided(1, 1) == pytest.approx(1.0)
    assert mcnemar_two_sided(6, 0) == pytest.approx(0.03125)
    assert mcnemar_two_sided(0, 0) == pytest.approx(1.0)
    assert min_resolvable_difference(16) == 5 / 16
    assert min_discordant_for_significance() == 6
    assert min_discordant_imbalance(5) is None
    assert min_discordant_imbalance(6) == 6
    assert wilson_interval(12, 16) == pytest.approx((0.505, 0.898), abs=0.0005)


def test_paired_power_depends_only_on_discordant_tasks() -> None:
    outcomes_a = {}
    outcomes_b = {}
    for index in range(24):
        outcomes_a[f"task-{index}"] = True
        outcomes_b[f"task-{index}"] = False
    for index in range(24, 29):
        outcomes_a[f"task-{index}"] = False
        outcomes_b[f"task-{index}"] = True
    for index in range(29, 94):
        outcomes_a[f"task-{index}"] = True
        outcomes_b[f"task-{index}"] = True
    for index in range(94, 104):
        outcomes_a[f"task-{index}"] = False
        outcomes_b[f"task-{index}"] = False

    def discordance(left: dict[str, bool], right: dict[str, bool]) -> tuple[int, int]:
        shared = set(left) & set(right)
        return (
            sum(left[task_id] and not right[task_id] for task_id in shared),
            sum(right[task_id] and not left[task_id] for task_id in shared),
        )

    full_b, full_c = discordance(outcomes_a, outcomes_b)
    reduced_a = {
        task_id: passed for task_id, passed in outcomes_a.items()
        if not (passed and outcomes_b[task_id])
    }
    reduced_b = {
        task_id: passed for task_id, passed in outcomes_b.items()
        if not (outcomes_a[task_id] and passed)
    }
    reduced_b_count, reduced_c_count = discordance(reduced_a, reduced_b)

    assert len(outcomes_a) == len(outcomes_b) == 104
    assert (full_b, full_c) == (24, 5)
    assert (reduced_b_count, reduced_c_count) == (24, 5)
    assert mcnemar_two_sided(full_b, full_c) == pytest.approx(
        0.000546, abs=0.0000005,
    )
    assert mcnemar_two_sided(reduced_b_count, reduced_c_count) == pytest.approx(
        mcnemar_two_sided(full_b, full_c), abs=0.0000001,
    )
    assert fisher_two_sided(89, 15, 70, 34) == pytest.approx(
        0.003025, abs=0.0000005,
    )
    assert fisher_two_sided(24, 15, 5, 34) == pytest.approx(
        0.000015, abs=0.0000005,
    )
    assert fisher_two_sided(89, 15, 70, 34) != fisher_two_sided(
        24, 15, 5, 34,
    )


def test_hard_suite_and_graders() -> None:
    assert len(HARD_TASKS) == 26
    assert len(HARD_SUITE_TASKS) == 130
    assert len(SUITES["hard"]) == 130
    assert len({task.id for task in SUITES["hard"]}) == 130
    assert not {task.id for task in HARD_TASKS} & {task.id for task in EXTENDED_TASKS}
    assert suite_digest(SUITES["extended"]) == "v2:86e41db848586057"
    assert suite_digest(HARD_SUITE_TASKS) == "v2:22237baff8c47e35"
    assert suite_digest(EXTENDED_TASKS) == "v2:86e41db848586057"
    assert all(
        task.grades in {"value", "form", "value+form"}
        for task in HARD_SUITE_TASKS
    )
    assert all(
        (task.value_check is not None) == (task.grades == "value+form")
        for task in HARD_SUITE_TASKS
    )

    checks = {
        "instruction.initials.quick_amber_fox": ("QAF", "Q A F"),
        "instruction.devowel.gateway": ("gtwy", "gtrwrd"),
        "instruction.words.5": (
            "Compiler translates source into machine",
            "A compiler translates source code into machine code.",
        ),
        "multilingual.katakana.computer": ("コンピュータ", "計算機"),
        "multilingual.en_from_ja.1": (
            "I drink water every morning.",
            (
                "Here's the translation of "
                '"私は毎朝水を飲みます。" into English:\n\n'
                "I drink water every morning."
            ),
        ),
        "instruction.sort_desc.3_1_2": ("3, 2, 1", "3, 1, 2"),
    }
    for task_id, (accepted, rejected) in checks.items():
        task = next(task for task in HARD_TASKS if task.id == task_id)
        assert task.check(accepted), task_id
        assert not task.check(rejected), task_id

    kanji_17 = next(task for task in HARD_TASKS if task.id == "multilingual.kanji_number.17")
    assert kanji_17.rule == "kanji_number:v3"
    for answer in ("十七", "一十七", "壹拾柒", "壱拾七"):
        assert kanji_17.check(answer)
    assert not kanji_17.check("十八")
    assert not kanji_17.check('17 を漢数字で書くと "壹柒" です。')
    assert kanji_17.value_check is not None
    assert kanji_17.value_check("十七です")
    assert not kanji_17.value_check("17")
    assert not kanji_17.value_check("三十一")

    kanji_30 = next(task for task in HARD_TASKS if task.id == "multilingual.kanji_number.30")
    assert kanji_30.rule == "kanji_number:v3"
    for answer in ("三十", "参拾", "參拾"):
        assert kanji_30.check(answer)
    assert not kanji_30.check("三十一")
    assert kanji_30.value_check is not None
    assert not kanji_30.value_check("三十一")
    assert not kanji_30.value_check("30")
    assert kanji_30.value_check("三十")
    assert kanji_30.value_check("参拾")

    seven = next(task for task in HARD_TASKS if task.id == "multilingual.lang_lock.seven")
    assert seven.prompt == (
        "Answer only in Japanese with no Latin letters and no digits: how many "
        "days are in one week? Output the kanji numeral alone."
    )
    assert seven.check("七")
    assert seven.check("七日")
    assert not seven.check("7")
    assert suite_digest(SUITES["hard"]) == "v2:22237baff8c47e35"


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
    for index, (_, required, _value_check, _rule) in enumerate(_JAPANESE):
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
    assert SUITES == {
        "core": TASKS,
        "extended": EXTENDED_TASKS,
        "hard": HARD_SUITE_TASKS,
    }
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


def test_shipped_graders_reject_prompt_copies_and_refusals() -> None:
    """Depth probes are out of scope because their prompts legitimately contain the needle."""
    for suite_name, tasks in SUITES.items():
        for task in tasks:
            candidates = (
                ("prompt", task.prompt),
                ("refusal", "I cannot answer that."),
                ("prompt-plus-wrong", f"{task.prompt}\nzzz-not-the-answer-4242"),
                ("wrong-plus-prompt", f"zzz-not-the-answer-4242\n{task.prompt}"),
            )
            graders = [("check", task.check)]
            if task.value_check is not None:
                graders.append(("value_check", task.value_check))
            for grader_name, grader in graders:
                for candidate_kind, candidate in candidates:
                    assert not grader(candidate), (
                        f"{suite_name} {task.id} {grader_name} accepted "
                        f"{candidate_kind}"
                    )


def test_repaired_suite_value_graders_reject_measured_false_positives() -> None:
    email = next(task for task in TASKS if task.id == "extraction.email")
    assert not _only_email("nmesh-ops@example.com")("ops@nmesh-ops@example.com")
    assert email.check("The email address to contact is nmesh-ops@example.com.")

    katakana = next(
        task for task in HARD_TASKS if task.id == "multilingual.katakana.model"
    )
    assert katakana.value_check is not None
    assert not katakana.value_check("「model」はカタカナでは「モード」になります。")

    kanji = next(
        task for task in HARD_TASKS if task.id == "multilingual.kanji_number.17"
    )
    assert kanji.value_check is not None
    assert not kanji.value_check("17")
    assert not kanji.value_check("三十一")
    assert kanji.value_check("十七です")

    assert not _bool_value(True)("true or false")


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
    plain = Task("rule", "instruction", "prompt", 8, lambda text: True)
    empty_rule = Task("rule", "instruction", "prompt", 8, lambda text: True, "")
    changed_rule = Task(
        "rule", "instruction", "prompt", 8, lambda text: True, "rule:v2",
    )
    assert suite_digest((plain,)) == suite_digest((empty_rule,))
    assert suite_digest((plain,)) != suite_digest((changed_rule,))


def test_runner_reports_value_passed_separately_from_discipline() -> None:
    tasks = tuple(
        next(task for task in HARD_SUITE_TASKS if task.id == task_id)
        for task_id in (
            "instruction.initials.quick_amber_fox",
            "multilingual.lang_lock.paris",
            "multilingual.lang_lock.seven",
            "format.json_even.7",
            "arithmetic.add.58_64",
            "instruction.items.4",
            "instruction.words.5",
            "compliance.email.0",
        )
    ) + (
        Task(
            "quick-pass", "instruction", "quick-pass", 8,
            next(task for task in HARD_SUITE_TASKS
                 if task.id == "instruction.initials.quick_amber_fox").check,
            value_check=next(
                task for task in HARD_SUITE_TASKS
                if task.id == "instruction.initials.quick_amber_fox"
            ).value_check,
        ),
        Task(
            "quick-wrong", "instruction", "quick-wrong", 8,
            next(task for task in HARD_SUITE_TASKS
                 if task.id == "instruction.initials.quick_amber_fox").check,
            value_check=next(
                task for task in HARD_SUITE_TASKS
                if task.id == "instruction.initials.quick_amber_fox"
            ).value_check,
        ),
    )
    _EvalHandler.responses = {
        tasks[0].prompt: (200, "Q A F"),
        tasks[1].prompt: (200, "Paris"),
        tasks[2].prompt: (200, "7"),
        tasks[3].prompt: (200, '{"even": true}'),
        tasks[4].prompt: (200, "58 + 64 = 112"),
        tasks[5].prompt: (200, "1. Tigers\n2. Whales\n3. Elephants\n4. Pandas"),
        tasks[6].prompt: (200, "A compiler translates source code into machine code."),
        tasks[7].prompt: (200, "Ping build-team@example.org"),
        "quick-pass": (200, "QAF"),
        "quick-wrong": (200, "QWEN"),
    }
    server = _serve()
    try:
        result = run(tasks, f"http://127.0.0.1:{server.server_address[1]}", "model")
    finally:
        server.shutdown()
        server.server_close()
    outcomes = {outcome.id: outcome for outcome in result.outcomes}
    assert (outcomes[tasks[0].id].passed, outcomes[tasks[0].id].value_passed) == (
        False, True,
    )
    assert (outcomes[tasks[1].id].passed, outcomes[tasks[1].id].value_passed) == (
        False, True,
    )
    assert (outcomes[tasks[2].id].passed, outcomes[tasks[2].id].value_passed) == (
        False, True,
    )
    assert (outcomes["quick-pass"].passed, outcomes["quick-pass"].value_passed) == (
        True, True,
    )
    assert (outcomes["quick-wrong"].passed, outcomes["quick-wrong"].value_passed) == (
        False, False,
    )
    assert outcomes["format.json_even.7"].failure_kind == "value"
    assert outcomes["arithmetic.add.58_64"].failure_kind == "value"
    assert outcomes["instruction.items.4"].failure_kind == "form"
    assert outcomes["instruction.words.5"].failure_kind == "form"
    assert outcomes["compliance.email.0"].failure_kind == "form"
    assert outcomes["quick-pass"].failure_kind == ""


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


def test_eval_rates_drop_runs_with_transport_errors() -> None:
    digest = suite_digest(TASKS)
    records = {
        "transport": EvalRecord(
            "model", "f16", "llamacpp", 104, 100, 100 / 104, {}, 1.0,
            {}, "", "core", digest, 0, 0, 1,
        ),
        "valid": EvalRecord(
            "other", "f16", "llamacpp", 104, 90, 90 / 104, {}, 2.0,
            {}, "", "core", digest,
        ),
    }
    assert cli._eval_rates(records) == {
        ("other", "f16", "llamacpp"): EvalSummary(
            90 / 104, 90, 104, {}, "core", digest,
        ),
    }


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
    assert output["value_only_failures"] == 0
    assert output["value_checked_failures"] == 0
    assert output["value_note"] is None
    assert output["failed"] == [{
        "id": "cut",
        "output": "",
        "unscorable": True,
        "value_passed": None,
        "failure_kind": "",
    }]


def test_eval_cli_reports_transport_failures(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    result = EvalRun(
        "prior-high", "q4_k_m", "llamacpp", 1, 0, 0.0, {"instruction": 0.0},
        [TaskOutcome(
            "transport", "instruction", False, "ReadTimeout: timed out",
            failure_kind="transport",
        )],
        3.0,
        transport_errors=1,
    )
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(cli, "eval_run", lambda tasks, base_url, model_ref, **kwargs: result)
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["transport_errors"] == 1
    assert "host failures" in output["transport_note"]
    assert output["unscorable"] == 0
    assert output["failed"][0]["failure_kind"] == "transport"


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
        transport_errors=1,
        cache_prompt=False,
    )
    save_eval(result, path)
    loaded = load_eval_cache(path)
    key = f"model|q4_k_m|llamacpp|core|{suite_digest(TASKS)}|c0"
    assert loaded[key].pass_rate == 0.5
    assert loaded[key].task_results == {
        "one": True, "two": False,
    }
    assert loaded[key].artifact == "gguf:2:123:abcdef"
    assert loaded[key].transport_errors == 1
    assert loaded[key].cache_prompt is False
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
    assert load_eval_cache(path)["legacy"].transport_errors == 0
    assert load_eval_cache(path)["legacy"].cache_prompt is None
    assert load_eval_cache(path)["legacy"].depth == 0
    assert load_eval_cache(path)["legacy"].prompt_tokens_max == 0
    legacy["results"]["legacy"]["task_results"] = {"one": "yes"}
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    legacy["results"]["legacy"]["task_results"] = {}
    legacy["results"]["legacy"]["artifact"] = ""
    legacy["results"]["legacy"]["transport_errors"] = True
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    legacy["results"]["legacy"]["transport_errors"] = 2
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    legacy["results"]["legacy"]["task_results"] = {}
    legacy["results"]["legacy"]["artifact"] = 1
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    legacy["results"]["legacy"]["artifact"] = ""
    legacy["results"]["legacy"]["cache_prompt"] = "false"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_eval_cache(path) == {}
    for invalid in (-1, True):
        legacy["results"]["legacy"]["cache_prompt"] = None
        legacy["results"]["legacy"]["depth"] = invalid
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


def test_eval_key_separates_cache_conditions_and_allowance() -> None:
    base = ("model", "f16", "llamacpp", "core", "digest")
    assert eval_key(*base) == "model|f16|llamacpp|core|digest"
    assert eval_key(*base, cache_prompt=False) == (
        "model|f16|llamacpp|core|digest|c0"
    )
    assert eval_key(*base, cache_prompt=True) == (
        "model|f16|llamacpp|core|digest|c1"
    )
    assert eval_key(*base, allowance=504, cache_prompt=False) == (
        "model|f16|llamacpp|core|digest|a504|c0"
    )
    assert eval_key(*base, depth=0) == "model|f16|llamacpp|core|digest"
    assert eval_key(*base, depth=4096) == (
        "model|f16|llamacpp|core|digest|d4096"
    )


def test_eval_rates_ignore_deep_record_and_suite_depth_is_not_evidence() -> None:
    digest = suite_digest(TASKS)
    eval_records = {
        "shallow": EvalRecord(
            "model", "f16", "llamacpp", 16, 8, 0.5, {}, 1.0,
            {}, "", "core", digest,
        ),
        "deep": EvalRecord(
            "model", "f16", "llamacpp", 16, 16, 1.0, {}, 2.0,
            {}, "", "core", digest, depth=4096, prompt_tokens_max=3072,
        ),
    }
    assert cli._eval_rates(eval_records)[("model", "f16", "llamacpp")].pass_rate == 0.5
    assert cli._context_evidence({}) == {}
    probe_digest = suite_digest(needle_tasks(8, "core"))
    context_records = {
        "probe": ContextRecord(
            "model", "f16", "llamacpp", "core", 8, 7, probe_digest,
            (FamilyResult("context.literal", 8, 8, 8, 8),), 2.0,
        ),
    }
    assert cli._context_evidence(context_records) == {
        ("model", "f16", "llamacpp"): cli.DepthEvidence(7, 0),
    }


def test_context_cache_round_trip_and_malformed_entries_are_skipped(tmp_path) -> None:
    path = tmp_path / "context.json"
    record = ContextRecord(
        "model", "f16", "llamacpp", "core", 8, 7,
        suite_digest(needle_tasks(8, "core")),
        (FamilyResult("context.literal", 8, 8, 8, 8),), 2.0,
        "artifact", False,
    )
    save_context(record, path)
    loaded = load_context_cache(path)
    assert loaded
    assert next(iter(loaded.values())) == record
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = payload["results"].popitem()[1]
    payload["results"] = {
        "valid": valid,
        "bool-depth": {**valid, "requested_depth": True},
        "passed-too-high": {
            **valid,
            "families": [{**valid["families"][0], "passed": 9}],
        },
        "empty-families": {**valid, "families": []},
        "non-dict-family": {**valid, "families": "invalid"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_context_cache(path)
    assert list(loaded) == ["valid"]


def test_context_evidence_filters_stale_controls_and_keeps_latest() -> None:
    def record(
        depth: int,
        at: float,
        families: tuple[FamilyResult, ...],
        served: int = 0,
        digest: str | None = None,
    ) -> ContextRecord:
        return ContextRecord(
            "MODEL", "F16", "LlamaCpp", "core", depth, served,
            digest or suite_digest(needle_tasks(depth, "core")),
            families, at,
        )

    passed = (FamilyResult("context.literal", 4, 4, 4, 4),)
    lost = (FamilyResult("context.literal", 2, 4, 4, 4),)
    uncontrolled = (FamilyResult("context.literal", 0, 0, 2, 4),)
    records = {
        "old": record(4, 1.0, lost, 4),
        "latest": record(4, 2.0, passed, 3),
        "deep-lost": record(6, 3.0, lost, 6),
        "deeper-lost": record(8, 4.0, lost, 8),
        "uncontrolled": record(2, 5.0, uncontrolled, 2),
        "stale": record(1024, 6.0, passed, 1000, "stale"),
        "control": record(0, 7.0, passed, 0),
    }
    assert cli._context_evidence(records) == {
        ("model", "f16", "llamacpp"): cli.DepthEvidence(3, 6),
    }


def test_context_evidence_uses_task_paired_controls() -> None:
    digest = suite_digest(needle_tasks(4, "core"))
    paired = ContextRecord(
        "model", "f16", "llamacpp", "core", 4, 4, digest,
        (FamilyResult("context.update", 2, 4, 4, 8),), 1.0,
    )
    no_paired_tasks = ContextRecord(
        "model", "f16", "llamacpp", "core", 4, 4, digest,
        (FamilyResult("context.update", 0, 0, 4, 8),), 2.0,
    )
    assert cli._context_evidence({"no-paired": no_paired_tasks}) == {}
    assert cli._context_evidence({"paired": paired}) == {
        ("model", "f16", "llamacpp"): cli.DepthEvidence(0, 4),
    }


def _quality_models() -> list[ModelSpec]:
    return [
        ModelSpec("prior-high", "test", 500_000_000, 24, 16, 2, 64, 1024,
                  4096, ["chat"], 90.0, "test", {"hf_gguf": "prior-high"}),
        ModelSpec("measured-high", "test", 500_000_000, 24, 16, 2, 64, 1024,
                  4096, ["chat"], 70.0, "test", {"hf_gguf": "measured-high"}),
    ]


def _quant_model() -> ModelSpec:
    return ModelSpec(
        "quant-model", "test", 500_000_000, 24, 16, 2, 64, 1024,
        4096, ["chat"], 90.0, "test", {"hf_gguf": "quant-model"},
    )


def _quant_eval_cache(
    selected: dict[str, bool],
    better: dict[str, bool],
) -> dict[tuple[str, str, str], EvalSummary]:
    return {
        ("quant-model", "f16", "llamacpp"): EvalSummary(
            sum(selected.values()) / len(selected), sum(selected.values()),
            len(selected), selected,
        ),
        ("quant-model", "q4_0", "llamacpp"): EvalSummary(
            sum(better.values()) / len(better), sum(better.values()),
            len(better), better,
        ),
    }


def test_planner_overrides_same_model_quant_with_significant_evidence() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    better = {f"task-{index}": True for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache=_quant_eval_cache(selected, better),
    )
    assert plan.services[0].quant == "q4_0"
    assert any("measurement outranks the quantization penalty" in warning
               for warning in plan.warnings)


def test_planner_does_not_override_same_model_quant_when_evidence_disabled() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    better = {f"task-{index}": True for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()],
        Policy(roles=["chat"], eval_evidence=False),
        eval_cache=_quant_eval_cache(selected, better),
    )
    assert plan.services[0].quant == "f16"
    assert not any("measurement outranks the quantization penalty" in warning
                   for warning in plan.warnings)


def test_planner_does_not_override_same_model_quant_without_significance() -> None:
    selected = {f"task-{index}": index in (0, 1, 2, 3, 4) for index in range(16)}
    better = {f"task-{index}": index in (0, 1, 2, 5) for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache=_quant_eval_cache(selected, better),
    )
    assert plan.services[0].quant == "f16"
    assert not any("measurement outranks the quantization penalty" in warning
                   for warning in plan.warnings)


def test_planner_warns_when_better_quant_is_not_selected() -> None:
    model = ModelSpec(
        "tight-quant", "test", 3_000_000_000, 40, 32, 8, 128, 512, 512,
        ["chat"], 90.0, "test", {"hf_gguf": "tight-quant"},
    )
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    better = {f"task-{index}": True for index in range(16)}
    plan = build_plan(
        profile(5), [model],
        Policy(
            roles=["chat"], max_context=512, min_decode_tps=0,
            eval_evidence=False,
        ),
        eval_cache={
            ("tight-quant", "q4_k_m", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
            ("tight-quant", "q4_0", "llamacpp"): EvalSummary(
                1.0, 16, 16, better,
            ),
        },
    )
    assert plan.services[0].quant == "q4_k_m"
    assert any("QUANT_PENALTY does the opposite" in warning
               for warning in plan.warnings)


def test_planner_notes_indistinguishable_same_model_quants() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache={
            ("quant-model", "f16", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
            ("quant-model", "q4_0", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
        },
    )
    assert plan.services[0].quant == "f16"
    notes = [
        warning for warning in plan.warnings
        if "this suite cannot distinguish them" in warning
    ]
    assert len(notes) == 1
    assert "5.0 points (0.0 vs 5.0)" in notes[0]


def test_planner_skips_incomparable_eval_conditions() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache={
            ("quant-model", "f16", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
                "core", "core-digest", 0, False,
            ),
            ("quant-model", "q4_0", "llamacpp"): EvalSummary(
                14 / 16, 14, 16, selected,
                "hard", "shared-digest", 0, False,
            ),
        },
    )
    assert plan.services[0].quant == "f16"
    assert not any("this suite cannot distinguish them" in warning
                   for warning in plan.warnings)
    assert not any("pass rate ranks" in warning for warning in plan.warnings)
    incomparable = [
        warning for warning in plan.warnings
        if "different conditions" in warning
    ]
    assert len(incomparable) == 1


def test_planner_requires_matching_cache_condition_for_override() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    better = {f"task-{index}": True for index in range(16)}
    incomparable = _quant_eval_cache(selected, better)
    incomparable[("quant-model", "f16", "llamacpp")] = replace(
        incomparable[("quant-model", "f16", "llamacpp")],
        cache_prompt=True,
    )
    incomparable[("quant-model", "q4_0", "llamacpp")] = replace(
        incomparable[("quant-model", "q4_0", "llamacpp")],
        cache_prompt=False,
    )
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache=incomparable,
    )
    assert plan.services[0].quant == "f16"
    assert any("different conditions" in warning for warning in plan.warnings)
    assert not any("measurement outranks" in warning for warning in plan.warnings)

    comparable = _quant_eval_cache(selected, better)
    comparable[("quant-model", "f16", "llamacpp")] = replace(
        comparable[("quant-model", "f16", "llamacpp")],
        cache_prompt=False,
    )
    comparable[("quant-model", "q4_0", "llamacpp")] = replace(
        comparable[("quant-model", "q4_0", "llamacpp")],
        cache_prompt=False,
    )
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache=comparable,
    )
    assert plan.services[0].quant == "q4_0"


def test_planner_does_not_note_more_expensive_quant() -> None:
    blocker = ModelSpec(
        "blocker", "test", 1_000_000_000, 24, 16, 2, 64, 512, 512,
        ["chat"], 90.0, "test", {"hf_gguf": "blocker"},
    )
    model = ModelSpec(
        "tight-quant", "test", 3_000_000_000, 40, 32, 8, 128, 512, 512,
        ["embed"], 90.0, "test", {"hf_gguf": "tight-quant"},
    )
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    plan = build_plan(
        profile(4.55), [blocker, model],
        Policy(roles=["chat", "embed"], max_context=512, min_decode_tps=0),
        eval_cache={
            ("tight-quant", "q4_0", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
            ("tight-quant", "q4_k_m", "llamacpp"): EvalSummary(
                10 / 16, 10, 16, selected,
            ),
        },
    )
    assert next(item for item in plan.services
                if item.model_id == "tight-quant").quant == "q4_0"
    assert not any("this suite cannot distinguish them" in warning
                   for warning in plan.warnings)


def test_planner_does_not_note_significant_same_model_quants() -> None:
    selected = {f"task-{index}": index >= 6 for index in range(16)}
    better = {f"task-{index}": True for index in range(16)}
    plan = build_plan(
        profile(8), [_quant_model()], Policy(roles=["chat"]),
        eval_cache=_quant_eval_cache(selected, better),
    )
    assert plan.services[0].quant == "q4_0"
    assert not any("this suite cannot distinguish them" in warning
                   for warning in plan.warnings)


def test_planner_eval_evidence_override_and_prior_fallback() -> None:
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
    assert contradictory.services[0].model_id == "measured-high"
    assert consistent.services[0].model_id == ordinary.services[0].model_id
    assert any(
        "measurement outranks" in warning
        and "measured-high" in warning
        and "prior-high" in warning
        and "p=0.00" in warning
        for warning in contradictory.warnings
    )
    assert not any("pass rate ranks" in warning for warning in contradictory.warnings)
    assert not any("pass rate ranks" in warning for warning in consistent.warnings)
    assert any("unverified" in warning for warning in ordinary.warnings)
    prior_only = build_plan(
        profile(8), models, replace(policy, eval_evidence=False),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.25, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
        },
    )
    assert prior_only.services[0].model_id == "prior-high"
    assert any("pass rate ranks" in warning for warning in prior_only.warnings)


def test_eval_cli_json_includes_note(monkeypatch, capsys) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    result = EvalRun(
        "prior-high", "q4_k_m", "llamacpp", 1, 1, 1.0,
        {category: 0.75 for category in CATEGORIES},
        [TaskOutcome(
            "failed", "instruction", False, "bad",
            value_passed=True, failure_kind="form",
        )],
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
    assert output["value_only_failures"] == 1
    assert output["value_checked_failures"] == 1
    assert "output form" in output["value_note"]
    assert output["failures_by_kind"] == {"value": 0, "form": 1}
    assert "1 failures: 0 wrong answers, 1 correct answers" in output[
        "failure_kinds_note"
    ]
    assert output["failed"] == [
        {
            "id": "failed",
            "output": "bad",
            "unscorable": False,
            "value_passed": True,
            "failure_kind": "form",
        },
    ]


def test_planner_deduplicates_multi_role_eval_override_warning() -> None:
    models = [replace(model, roles=["chat", "code"]) for model in _quality_models()]
    plan = build_plan(
        profile(8), models, Policy(roles=["chat", "code"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.25, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
        },
    )
    overrides = [warning for warning in plan.warnings if "measurement outranks" in warning]
    assert len(overrides) == 1
    assert not any("pass rate ranks" in warning for warning in plan.warnings)


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


def test_planner_reports_paired_underpowered_eval_evidence() -> None:
    models = _quality_models()
    selected = {f"task-{index}": index == 0 for index in range(5)}
    better = {f"task-{index}": index in (0, 1) for index in range(5)}
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(
                1 / 5, 1, 5, selected,
            ),
            ("measured-high", "f16", "llamacpp"): EvalSummary(
                2 / 5, 2, 5, better,
            ),
        },
    )
    assert any(
        "paired tasks" in note
        and "only 1 disagreed" in note
        and "1 one way" in note
        and "0 the other" in note
        and "fewer than 6" in note
        for note in plan.warnings
    )


def test_planner_overrides_for_significant_eval_gap() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(4 / 16, 4, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(14 / 16, 14, 16, {}),
        },
    )
    override = next(item for item in plan.warnings if "measurement outranks" in item)
    assert "p=" in override
    assert "16 tasks" in override


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
    assert significant.services[0].model_id == "measured-high"
    assert any(
        "measurement outranks" in warning and "16 tasks" in warning
        for warning in significant.warnings
    )

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
    assert plan.services[0].model_id == "prior-high"
    assert not any("measurement outranks" in warning for warning in plan.warnings)
    assert not any("pass rate ranks" in warning for warning in plan.warnings)
    assert not any("neither confirmed nor contradicted" in note for note in plan.warnings)


def test_planner_does_not_override_without_significance() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.5, 8, 16, {}),
            ("measured-high", "f16", "llamacpp"): EvalSummary(11 / 16, 11, 16, {}),
        },
    )
    assert plan.services[0].model_id == "prior-high"
    assert not any("measurement outranks" in warning for warning in plan.warnings)


def test_planner_requires_the_planned_eval_configuration() -> None:
    models = _quality_models()
    plan = build_plan(
        profile(8), models, Policy(roles=["chat"]),
        eval_cache={
            ("prior-high", "f16", "llamacpp"): EvalSummary(0.25, 4, 16, {}),
            ("measured-high", "q4_k_m", "llamacpp"): EvalSummary(
                14 / 16, 14, 16, {},
            ),
            ("measured-high", "f16", "ollama"): EvalSummary(
                14 / 16, 14, 16, {},
            ),
        },
    )
    assert plan.services[0].model_id == "prior-high"
    assert not any("measurement outranks" in warning for warning in plan.warnings)


def test_planner_does_not_override_an_unmeasured_choice() -> None:
    models = [
        replace(_quality_models()[0], quality=None),
        replace(_quality_models()[1], quality=0.0),
    ]
    plan = build_plan(
        profile(8), models,
        Policy(
            roles=["chat"],
            model_ids=("prior-high", "measured-high"),
            min_decode_tps=0,
        ),
        eval_cache={
            ("measured-high", "f16", "llamacpp"): EvalSummary(1.0, 16, 16, {}),
        },
    )
    assert plan.services[0].model_id == "prior-high"
    assert not any("measurement outranks" in warning for warning in plan.warnings)


def test_cli_plan_ignore_eval_evidence_sets_policy(monkeypatch, capsys) -> None:
    models = _quality_models()
    captured = []
    monkeypatch.setattr(cli, "detect_hardware", lambda: profile(8))
    monkeypatch.setattr(cli, "load_catalog", lambda: models)
    monkeypatch.setattr(cli, "load_cache", dict)
    monkeypatch.setattr(cli, "bench_overlay", dict)
    monkeypatch.setattr(cli, "_eval_rates", dict)
    monkeypatch.setattr(cli, "save_plan", captured.append)
    assert cli.main(["plan", "--roles", "chat", "--ignore-eval-evidence"]) == 0
    capsys.readouterr()
    assert captured
    assert captured[0].policy.eval_evidence is False


def test_cli_up_ignore_eval_evidence_parses(monkeypatch) -> None:
    parsed = {}
    monkeypatch.setattr(
        cli, "_runtime", lambda args: parsed.update(vars(args)) or 0,
    )
    assert cli.main(["up", "--ignore-eval-evidence"]) == 0
    assert parsed["ignore_eval_evidence"] is True


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
        ("model", "f16", "llamacpp"): EvalSummary(
            0.75, 12, 16, {}, "core", core_digest,
        ),
        ("model", "q4_k_m", "ollama"): EvalSummary(
            0.8125, 13, 16, {}, "core", core_digest,
        ),
    }


def test_eval_rates_propagate_evidence_identity() -> None:
    core_digest = suite_digest(TASKS)
    extended_digest = suite_digest(EXTENDED_TASKS)
    records = {
        "cache-on": EvalRecord(
            "model", "f16", "llamacpp", 16, 12, 0.75, {}, 1.0,
            {}, "", "extended", extended_digest, 0, 504, 0, True,
        ),
        "cache-off": EvalRecord(
            "other", "q4_k_m", "ollama", 16, 13, 0.8125, {}, 2.0,
            {}, "", "core", core_digest, 0, 0, 0, False,
        ),
    }
    assert cli._eval_rates(records) == {
        ("model", "f16", "llamacpp"): EvalSummary(
            0.75, 12, 16, {}, "extended", extended_digest, 504, True,
        ),
        ("other", "q4_k_m", "ollama"): EvalSummary(
            0.8125, 13, 16, {}, "core", core_digest, 0, False,
        ),
    }


def test_eval_records_keep_cache_conditions_separate() -> None:
    digest = suite_digest(TASKS)
    reuse = EvalRecord(
        "model", "f16", "llamacpp", 16, 8, 0.5, {}, 1.0,
        digest=digest,
        cache_prompt=None,
    )
    clean = replace(reuse, passed=12, pass_rate=0.75, at=2.0, cache_prompt=False)
    valid, stale = cli._eval_records({"reuse": reuse, "clean": clean})
    assert stale == []
    assert len(valid) == 2
    assert {record.cache_prompt for record in valid.values()} == {None, False}


def test_mixed_case_eval_quant_matches_planned_configuration() -> None:
    digest = suite_digest(TASKS)
    records = {
        "prior": EvalRecord(
            "prior-high", "f16", "llamacpp", 16, 4, 0.25, {}, 1.0,
            {}, "", "core", digest,
        ),
        "measured": EvalRecord(
            "measured-high", "Q4_K_M", "llamacpp", 16, 14, 14 / 16, {}, 2.0,
            {}, "", "core", digest,
        ),
    }
    cache = cli._eval_rates(records)
    bench_cache = {
        (model_id, quant, "llamacpp", "cpu", 0): 0.0
        for model_id in ("prior-high", "measured-high")
        for quant in ("f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m", "q2_k")
    }
    bench_cache[("measured-high", "q4_k_m", "llamacpp", "cpu", 0)] = 200.0
    plan = build_plan(
        profile(8), _quality_models(),
        Policy(roles=["chat"], min_decode_tps=0),
        bench_cache,
        cache,
    )
    assert plan.services[0].model_id == "measured-high"
    assert not any("configuration" in warning for warning in plan.warnings)


def test_mismatched_eval_backend_still_warns_without_override() -> None:
    digest = suite_digest(TASKS)
    cache = cli._eval_rates({
        "prior": EvalRecord(
            "prior-high", "f16", "ollama", 16, 4, 0.25, {}, 1.0,
            {}, "", "core", digest,
        ),
        "measured": EvalRecord(
            "measured-high", "f16", "ollama", 16, 14, 14 / 16, {}, 2.0,
            {}, "", "core", digest,
        ),
    })
    plan = build_plan(
        profile(8), _quality_models(), Policy(roles=["chat"]), None, cache,
    )
    assert plan.services[0].model_id == "prior-high"
    assert any(
        "prior-high" in warning and "configuration" in warning
        for warning in plan.warnings
    )
    assert not any("measurement outranks" in warning for warning in plan.warnings)


def test_eval_rates_casefolded_quant_keeps_newest_record() -> None:
    digest = suite_digest(TASKS)
    records = {
        "old": EvalRecord(
            "model", "Q4_K_M", "llamacpp", 16, 4, 0.25, {}, 1.0,
            {}, "", "core", digest,
        ),
        "new": EvalRecord(
            "model", "q4_k_m", "llamacpp", 16, 14, 14 / 16, {}, 2.0,
            {}, "", "core", digest,
        ),
    }
    assert cli._eval_rates(records) == {
        ("model", "q4_k_m", "llamacpp"): EvalSummary(
            14 / 16, 14, 16, {}, "core", digest,
        ),
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
    kwargs_seen = {}
    monkeypatch.setattr(
        cli,
        "eval_run",
        lambda tasks, base_url, model_ref, **kwargs: (
            captured.extend(tasks) or kwargs_seen.update(kwargs) or result
        ),
    )
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "_print_json", lambda value: None)
    assert cli.main(["eval", "--json", "--categories", "format"]) == 0
    assert captured
    assert {task.category for task in captured} == {"format"}
    assert kwargs_seen["cache_prompt"] is False


def test_eval_cli_omits_cache_prompt_for_non_llamacpp(monkeypatch) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    service = replace(service_plan.services[0], backend="ollama")
    service_plan = replace(service_plan, services=[service])
    result = EvalRun("prior-high", "f16", "ollama", 1, 1, 1.0, {}, [], 3.0)
    kwargs_seen = {}
    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)
    monkeypatch.setattr(
        cli,
        "eval_run",
        lambda tasks, base_url, model_ref, **kwargs: (
            kwargs_seen.update(kwargs) or result
        ),
    )
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "_print_json", lambda value: None)
    assert cli.main(["eval", "--json"]) == 0
    assert kwargs_seen["cache_prompt"] is None


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


def test_eval_cli_depth_runs_suite_and_context_probe_separately(
    monkeypatch, capsys,
) -> None:
    service_plan = build_plan(profile(8), _quality_models()[:1], Policy(roles=["chat"]))
    calls: list[tuple[tuple[Task, ...], dict[str, object]]] = []
    result = EvalRun("prior-high", "f16", "llamacpp", 16, 16, 1.0, {}, [], 3.0)

    def outcomes(statuses: dict[str, list[bool]]) -> list[TaskOutcome]:
        return [
            TaskOutcome(f"{category}.{index}", category, passed, "")
            for category, values in statuses.items()
            for index, passed in enumerate(values)
        ]

    monkeypatch.setattr(cli, "load_plan", lambda: service_plan)
    monkeypatch.setattr(cli, "_service_running", lambda service, runtime: True)

    def evaluate(tasks, base_url, model_ref, **kwargs):
        calls.append((tuple(tasks), kwargs))
        if kwargs["depth"] == 0 and len(calls) in {2, 4}:
            return EvalRun(
                "prior-high", "f16", "llamacpp", 24, 16, 2 / 3, {}, outcomes({
                        "context.literal": (
                            [False] + [True] * 7 if len(calls) == 2 else [True] * 8
                        ),
                        "context.latent": [False] * 8,
                        "context.multi": [True] * 4,
                        "context.update": [True] * 4,
                }), 3.0,
            )
        if kwargs["depth"] == 4096 and any(
            task.category.startswith("context.") for task in tasks
        ):
            return EvalRun(
                "prior-high", "f16", "llamacpp", 24, 13, 13 / 24, {}, outcomes({
                    "context.literal": [False] + [True] * 7,
                    "context.latent": [False] * 8,
                    "context.multi": [True, True, True, False],
                    "context.update": [True] * 4,
                }), 3.0, prompt_tokens_max=8192,
            )
        return result

    monkeypatch.setattr(cli, "eval_run", evaluate)
    monkeypatch.setattr(cli, "save_eval", lambda value: None)
    monkeypatch.setattr(cli, "save_context", lambda value: "context.json")
    assert cli.main(["eval", "--json", "--depth", "4096"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(calls) == 4
    assert [call[1]["depth"] for call in calls] == [4096, 0, 4096, 0]
    assert not any(task.category.startswith("context.") for task in calls[0][0])
    assert all(task.category.startswith("context.") for task in calls[1][0])
    assert all(task.category.startswith("context.") for task in calls[2][0])
    assert all(task.category.startswith("context.") for task in calls[3][0])
    assert output["requested_depth"] == 4096
    assert output["served_depth"] is None
    assert output["served_depth_known"] is False
    assert output["context_probe"]["families"] == {
        "literal": {
            "passed": 7, "of": 7, "control_passed": 15, "control_of": 16,
            "attributable": True,
        },
        "latent": {
            "passed": 0, "of": 0, "control_passed": 0, "control_of": 16,
            "attributable": False,
        },
        "multi": {
            "passed": 3, "of": 4, "control_passed": 8, "control_of": 8,
            "attributable": True,
        },
        "update": {
            "passed": 4, "of": 4, "control_passed": 8, "control_of": 8,
            "attributable": True,
        },
    }
    assert output["context_probe"]["control_passed"] == 31
    assert output["context_probe"]["control_of"] == 48
    assert output["context_probe"]["attributable"] is False
    assert len(output["context_probe"]["depth_warnings"]) == 1
    assert "multi" in output["context_probe"]["depth_warnings"][0]
    assert output["context_probe"]["uncontrolled_families"] == ["latent"]
    assert "latent" in output["context_probe"]["uncontrolled_note"]


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
    assert cli._eval_divergence(result, records) == [cli._Divergence(
        config="f16|ollama",
        artifact=None,
        pass_rate=0.5,
        compared=2,
        disagreeing=[
            "arithmetic.subtract",
            "multilingual.ja_translate",
        ],
        discordant_here=1,
        discordant_there=1,
        zero_power_families=[],
    )]
    assert cli._eval_divergence(replace(result, cache_prompt=False), records) == []


def test_eval_divergence_ignores_transport_contaminated_runs() -> None:
    result = EvalRun(
        "model", "f16", "llamacpp", 1, 0, 0.0, {},
        [TaskOutcome("task", "instruction", False, "", failure_kind="transport")],
        2.0,
        transport_errors=1,
    )
    record = EvalRecord(
        "model", "q4_k_m", "ollama", 1, 1, 1.0, {}, 1.0,
        {"task": True},
    )
    assert cli._eval_divergence(result, {"record": record}) == []

    clean_result = replace(result, transport_errors=0)
    contaminated_record = replace(record, transport_errors=1)
    assert cli._eval_divergence(
        clean_result, {"record": contaminated_record},
    ) == []


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
