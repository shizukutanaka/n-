"""Deterministic micro-eval used to falsify the catalog's quality priors.

Every task is graded by a self-contained verifier: no LLM judge, no reference
model, no network. Numeric tasks use ``_number`` to grade the value while
tolerating surrounding prose. String-extraction tasks and instruction/format
tasks are strict because output discipline is what they measure. The suite is
small and mechanical on purpose: it measures instruction following, output
format discipline, extraction and translation direction, not knowledge. A pass
rate here is not MMLU and must never be presented as one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
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
    id: str
    category: str
    prompt: str
    max_tokens: int
    check: Callable[[str], bool]


def _exact(expected: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        return normalize(text).lower() == expected.lower()
    return check


def _number(expected: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        digits = re.findall(r"-?\d+", text.replace(",", ""))
        return len(digits) >= 1 and int(digits[-1]) == expected
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
    ),
    Task(
        "instruction.single_word",
        "instruction",
        "Answer with one word only: what colour is a ripe banana?",
        16,
        _exact("yellow"),
    ),
    Task(
        "instruction.no_prose",
        "instruction",
        "Output only the uppercase form of the word 'mesh'. No explanation.",
        16,
        _exact("MESH"),
    ),
    Task(
        "instruction.three_items",
        "instruction",
        "List exactly three fruits, comma-separated, with no other text.",
        32,
        _three_items,
    ),
    Task(
        "format.json_city",
        "format",
        'Return only JSON with exactly the keys "city" and "country" for the '
        "capital of Japan. No markdown, no commentary.",
        48,
        _json_keys({"city": "Tokyo", "country": "Japan"}),
    ),
    Task(
        "format.json_bool",
        "format",
        'Return only JSON of the form {"even": <true or false>} stating whether '
        "12 is even. No other text.",
        32,
        lambda text: (_json_object(text) or {}).get("even") is True,
    ),
    Task(
        "format.json_number",
        "format",
        'Return only JSON of the form {"count": <integer>} with the number of '
        'characters in the word "orchestrator". No other text.',
        32,
        lambda text: (_json_object(text) or {}).get("count") == 12,
    ),
    Task(
        "arithmetic.add",
        "arithmetic",
        "What is 17 + 25? Answer with the number only.",
        16,
        _number(42),
    ),
    Task(
        "arithmetic.multiply",
        "arithmetic",
        "What is 12 * 12? Answer with the number only.",
        16,
        _number(144),
    ),
    Task(
        "arithmetic.subtract",
        "arithmetic",
        "What is 1000 - 253? Answer with the number only.",
        16,
        _number(747),
    ),
    Task(
        "arithmetic.count",
        "arithmetic",
        "How many letters are in the word 'benchmark'? Answer with the number only.",
        16,
        _number(9),
    ),
    Task(
        "extraction.email",
        "extraction",
        "Extract the email address and output it alone: "
        "'Contact ops at nmesh-ops@example.com before Friday.'",
        32,
        _exact("nmesh-ops@example.com"),
    ),
    Task(
        "extraction.date",
        "extraction",
        "Extract the date in YYYY-MM-DD form and output it alone: "
        "'The release shipped on March 3, 2024 in Tokyo.'",
        32,
        _exact("2024-03-03"),
    ),
    Task(
        "extraction.number",
        "extraction",
        "Output only the largest number in this list: 18, 4, 236, 97.",
        16,
        _number(236),
    ),
    Task(
        "multilingual.ja_translate",
        "multilingual",
        "次の英文を日本語に訳し、訳文だけを出力してください: 'The cat sleeps.'",
        48,
        _japanese_only("猫"),
    ),
    Task(
        "multilingual.ja_answer",
        "multilingual",
        "日本の首都はどこですか。地名だけを日本語で出力してください。",
        32,
        _japanese_only("東京"),
    ),
)


CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(task.category for task in TASKS))


__all__ = ["CATEGORIES", "TASKS", "Task", "normalize"]
