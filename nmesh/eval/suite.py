"""Deterministic micro-eval used to falsify the catalog's quality priors.

Every task is graded by a self-contained verifier: no LLM judge, no reference
model, no network. Numeric tasks use ``_number`` to grade the value while
tolerating surrounding prose. Value-extraction tasks grade the extracted value;
output discipline is measured by the ``instruction``, ``format`` and
``compliance`` families instead of being conflated with extraction. The suite
is small and mechanical on purpose: it measures instruction following, output
format discipline, extraction and translation direction, not knowledge. A pass
rate here is not MMLU and must never be presented as one.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_FENCE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$")
_ASCII_LETTER = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    """Strip markdown fences, surrounding quotes and trailing punctuation."""
    value = _FENCE.sub("", text.strip()).strip()
    value = value.strip("`").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value.rstrip(".!。").strip()


def _json_object(text: str) -> dict[str, object] | None:
    value = _FENCE.sub("", text.strip()).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(value[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass(frozen=True)
class Task:
    """A deterministic evaluation task and its grader identities.

    ``rule`` identifies the per-task grader semantics. Bump it whenever a
    verifier's accept/reject behavior changes, so only suites containing that
    task lose comparability. ``GRADER_VERSION`` remains for suite-wide scoring
    rule changes.

    ``grades`` declares which failure dimension the task measures:
    ``"value"`` means the primary checker tolerates surrounding form,
    ``"form"`` means there is no separable value dimension, and
    ``"value+form"`` means ``value_check`` must distinguish the two.
    """

    id: str
    category: str
    prompt: str
    max_tokens: int
    check: Callable[[str], bool]
    rule: str = ""
    value_check: Callable[[str], bool] | None = None
    grades: str = "value+form"


def _exact(expected: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return normalize(text).lower() == expected.lower()
    return check


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _only_email(expected: str) -> Callable[[str], bool]:
    """Grade the extracted address itself, tolerating prose around it."""

    def check(text: str) -> bool:
        return {found.lower() for found in _EMAIL_RE.findall(text)} == {expected.lower()}

    return check


def _only_date(expected: str) -> Callable[[str], bool]:
    """Grade the extracted ISO date itself, tolerating prose around it."""

    def check(text: str) -> bool:
        return set(_ISO_DATE_RE.findall(text)) == {expected}

    return check


def _only_span(expected: str, rivals: Sequence[str]) -> Callable[[str], bool]:
    """Grade a copied span by value: it must appear and no rival span may."""

    def check(text: str) -> bool:
        lowered = text.lower()
        if expected.lower() not in lowered:
            return False
        return not any(rival.lower() in lowered for rival in rivals)

    return check


def _number(expected: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        digits = re.findall(r"-?\d+", text.replace(",", ""))
        return len(digits) >= 1 and int(digits[-1]) == expected
    return check


def _contains_ci(expected: str) -> Callable[[str], bool]:
    """Value-level: the answer appears somewhere, ignoring case and prose."""
    return lambda text: expected.casefold() in text.casefold()


def _contains_cased(expected: str) -> Callable[[str], bool]:
    """Value-level for case tasks: the exact casing must appear."""
    return lambda text: expected in text


def _bool_value(expected: bool) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        found = re.findall(r"true|false", text.lower())
        return bool(found) and (found[-1] == "true") is expected
    return check


def _yes_no_value(expected: bool) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        found = re.findall(r"\byes\b|\bno\b", text.lower())
        return bool(found) and (found[-1] == "yes") is expected
    return check


def _items_value(count: int) -> Callable[[str], bool]:
    """Value-level for list tasks: the right number of items, any layout."""
    def check(text: str) -> bool:
        parts: list[str] = []
        for line in text.splitlines():
            stripped = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line)
            parts.extend(part.strip() for part in stripped.split(","))
        entries = [part for part in parts if part and part.replace(" ", "").isalpha()]
        return len(entries) == count
    return check


def _json_keys(expected: dict[str, str]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        parsed = _json_object(text)
        if parsed is None or set(parsed.keys()) != set(expected):
            return False
        return all(
            isinstance(parsed[key], str)
            and normalize(str(parsed[key])).lower() == value.lower()
            for key, value in expected.items()
        )
    return check


def _three_items(text: str) -> bool:
    items = [item.strip() for item in normalize(text).split(",")]
    return len(items) == 3 and all(item and item.replace(" ", "").isalpha() for item in items)


def _japanese_only(required: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        value = normalize(text)
        return required in value and not _ASCII_LETTER.search(value)
    return check


TASKS: tuple[Task, ...] = (
    Task(
        "instruction.echo",
        "instruction",
        "Reply with exactly the word Acknowledged and nothing else.",
        16,
        _exact("Acknowledged"),
        value_check=_contains_ci("Acknowledged"),
    ),
    Task(
        "instruction.single_word",
        "instruction",
        "Answer with one word only: what colour is a ripe banana?",
        16,
        _exact("yellow"),
        value_check=_contains_ci("yellow"),
    ),
    Task(
        "instruction.no_prose",
        "instruction",
        "Output only the uppercase form of the word 'mesh'. No explanation.",
        16,
        _exact("MESH"),
        value_check=_contains_cased("MESH"),
    ),
    Task(
        "instruction.three_items",
        "instruction",
        "List exactly three fruits, comma-separated, with no other text.",
        32,
        _three_items,
        value_check=_items_value(3),
    ),
    Task(
        "format.json_city",
        "format",
        'Return only JSON with exactly the keys "city" and "country" for the '
        "capital of Japan. No markdown, no commentary.",
        48,
        _json_keys({"city": "Tokyo", "country": "Japan"}),
        value_check=lambda text: (
            "tokyo" in text.casefold() and "japan" in text.casefold()
        ),
    ),
    Task(
        "format.json_bool",
        "format",
        'Return only JSON of the form {"even": <true or false>} stating whether '
        "12 is even. No other text.",
        32,
        lambda text: (_json_object(text) or {}).get("even") is True,
        value_check=_bool_value(True),
    ),
    Task(
        "format.json_number",
        "format",
        'Return only JSON of the form {"count": <integer>} with the number of '
        'characters in the word "orchestrator". No other text.',
        32,
        lambda text: (_json_object(text) or {}).get("count") == 12,
        value_check=_number(12),
    ),
    Task(
        "arithmetic.add",
        "arithmetic",
        "What is 17 + 25? Answer with the number only.",
        16,
        _number(42),
        grades="value",
    ),
    Task(
        "arithmetic.multiply",
        "arithmetic",
        "What is 12 * 12? Answer with the number only.",
        16,
        _number(144),
        grades="value",
    ),
    Task(
        "arithmetic.subtract",
        "arithmetic",
        "What is 1000 - 253? Answer with the number only.",
        16,
        _number(747),
        grades="value",
    ),
    Task(
        "arithmetic.count",
        "arithmetic",
        "How many letters are in the word 'benchmark'? Answer with the number only.",
        16,
        _number(9),
        grades="value",
    ),
    Task(
        "extraction.email",
        "extraction",
        "Extract the email address and output it alone: "
        "'Contact ops at nmesh-ops@example.com before Friday.'",
        32,
        _only_email("nmesh-ops@example.com"),
        grades="value",
    ),
    Task(
        "extraction.date",
        "extraction",
        "Extract the date in YYYY-MM-DD form and output it alone: "
        "'The release shipped on March 3, 2024 in Tokyo.'",
        32,
        _only_date("2024-03-03"),
        grades="value",
    ),
    Task(
        "extraction.number",
        "extraction",
        "Output only the largest number in this list: 18, 4, 236, 97.",
        16,
        _number(236),
        grades="value",
    ),
    Task(
        "multilingual.ja_translate",
        "multilingual",
        "次の英文を日本語に訳し、訳文だけを出力してください: 'The cat sleeps.'",
        48,
        _japanese_only("猫"),
        value_check=_contains_ci("猫"),
    ),
    Task(
        "multilingual.ja_answer",
        "multilingual",
        "日本の首都はどこですか。地名だけを日本語で出力してください。",
        32,
        _japanese_only("東京"),
        value_check=_contains_ci("東京"),
    ),
)


CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(task.category for task in TASKS))
GRADER_VERSION = 2


def suite_digest(tasks: Sequence[Task]) -> str:
    """Identify what was graded and how, so records from different rules never compare."""
    payload = "\n".join(
        f"{task.id}\x00{task.prompt}\x00{task.max_tokens}"
        + (f"\x00{task.rule}" if task.rule else "")
        for task in tasks
    )
    return (
        f"v{GRADER_VERSION}:"
        f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
    )


__all__ = [
    "CATEGORIES",
    "GRADER_VERSION",
    "TASKS",
    "Task",
    "normalize",
    "suite_digest",
]
