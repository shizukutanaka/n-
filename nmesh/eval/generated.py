"""Deterministic extension of the micro-eval suite.

The 16-task core suite cannot resolve pass-rate differences below 0.3125 at
alpha=0.05 (exact Fisher), so it can only falsify very large quality-prior
errors. Enumerating 80 further tasks plus 8 compliance tasks with the same
code-only verifiers brings the generated suite to 88 tasks and the extended
suite to 104 tasks.

Every task here is enumerated, not sampled: no RNG, no LLM judge, no network,
so task ids and expected answers are stable across runs and machines. Tasks
within a family (for example ``arithmetic.add.*``) are not independent
samples: item outcomes inside a family correlate, so the effective sample size
is smaller than the task count and the exact tests above are optimistic to
that extent. Nothing here measures knowledge; it measures instruction
following, output format discipline, extraction and translation direction.
"""

# ruff: noqa: ISC004

from __future__ import annotations

from .hard import HARD_TASKS
from .suite import (
    TASKS,
    Task,
    _bool_value,
    _contains_cased,
    _contains_ci,
    _contains_standalone,
    _exact,
    _items_value,
    _japanese_only,
    _json_object,
    _number,
    _only_date,
    _only_email,
    _only_span,
    normalize,
)

_ADDITIONS = ((48, 27, 75), (139, 84, 223), (256, 178, 434), (67, 89, 156),
              (903, 47, 950), (1234, 876, 2110), (58, 64, 122), (415, 285, 700))
_SUBTRACTIONS = ((81, 24, 57), (200, 143, 57), (1000, 47, 953), (512, 289, 223),
                 (640, 355, 285), (100, 68, 32), (3001, 1999, 1002), (75, 38, 37))
_PRODUCTS = ((13, 7, 91), (24, 6, 144), (15, 15, 225), (32, 11, 352),
             (9, 47, 423), (120, 5, 600))
_LETTER_WORDS = ("planner", "quantization", "gateway", "throughput")
_UPPERCASE = ("plan", "slot", "quant")
_LOWERCASE = ("BENCH", "EVAL", "PROBE")
_ECHO = ("Confirmed", "Ready", "Received", "Understood")
_ITEM_COUNTS = (2, 4, 5, 6)
_PEOPLE = (("Yuki", "31"), ("Ada", "27"), ("Omar", "45"), ("Lena", "52"),
           ("Ravi", "38"), ("Mina", "24"), ("Tomas", "60"), ("Iris", "19"))
_CHAR_WORDS = ("scheduler", "slot", "backend", "context")
_PARITY = ((7, False), (18, True), (91, False), (250, True))
_EMAILS = (
    ("Ping build-team@example.org once the snapshot lands; "
     "noreply@example.org is unmonitored.",
     "build-team@example.org", "address to ping"),
    ("Escalations go to sre.oncall@example.net after 18:00; "
     "daytime mail goes to help@example.net.",
     "sre.oncall@example.net", "address for escalations after 18:00"),
    ("Invoices: billing+eu@example.com (no attachments). "
     "Refunds: refunds@example.com.",
     "billing+eu@example.com", "address for invoices"),
    ("Ask maya_ito@example.co.jp for the key; "
     "kenji_sato@example.co.jp only handles hardware.",
     "maya_ito@example.co.jp", "address to ask for the key"),
    ("Reports are sent by nightly-report@example.io daily; "
     "alerts come from pager@example.io.",
     "nightly-report@example.io", "address that sends the reports"),
)
_DATES = (
    ("The audit closed on July 9, 2021 in Osaka after opening on "
     "June 2, 2021.", "2021-07-09", "date the audit closed"),
    ("Support ends on December 31, 2025 worldwide; the deprecation notice "
     "went out on May 6, 2025.", "2025-12-31", "date support ends"),
    ("She joined on February 14, 2019 as an intern and was promoted on "
     "August 1, 2020.", "2019-02-14", "date she joined"),
    ("The outage began on October 1, 2023 at noon and was resolved on "
     "October 3, 2023.", "2023-10-01", "date the outage began"),
    ("Shipping resumes on April 5, 2022 in Berlin; it was paused on "
     "March 18, 2022.", "2022-04-05", "date shipping resumes"),
)
_MAXIMA = (
    ("512, 78, 4096, 33", 4096),
    ("7, 19, 3, 11", 19),
    ("1024, 999, 1200, 88", 1200),
    ("45, 45, 46, 12", 46),
    ("3, 300, 3000, 30", 3000),
)
_SUBSTRINGS = (
    ("Extract the model name and output it alone: "
     "'We deployed qwen2.5-7b-instruct on the spare node.'",
     "qwen2.5-7b-instruct", ("spare node", "deployed")),
    ("Extract the port and output it alone: "
     "'The gateway listens on 18000 by default.'",
     "18000", ("listens", "by default")),
    ("Extract the file name and output it alone: "
     "'Copy plan.json into the state directory.'",
     "plan.json", ("state directory", "Copy")),
    ("Extract the flag and output it alone: "
     "'Pass --parallel-slots to raise the slot count.'",
     "--parallel-slots", ("slot count", "raise")),
    ("Extract the quantization label and output it alone: "
     "'The blob was tagged Q4_K_M by the exporter.'",
     "Q4_K_M", ("exporter", "blob was")),
)
_JAPANESE = (
    (
        "次の英文を日本語に訳し、訳文だけを出力してください: 'The dog runs.'",
        "犬", _contains_ci("犬"), "",
    ),
    (
        "次の英文を日本語に訳し、訳文だけを出力してください: 'I drink water.'",
        "水", _contains_ci("水"), "",
    ),
    (
        "次の英文を日本語に訳し、訳文だけを出力してください: 'The book is new.'",
        "本", _contains_standalone("本", "日"), "ja_token:v2",
    ),
    (
        "日本で最も高い山の名前だけを日本語で出力してください。",
        "富士", _contains_ci("富士"), "",
    ),
)


def _items(count: int):
    def check(text: str) -> bool:
        items = [item.strip() for item in normalize(text).split(",")]
        return (
            len(items) == count
            and all(item and item.replace(" ", "").isalpha() for item in items)
        )
    return check


def _json_person(name: str, age: str):
    """Grade a two-key JSON object, accepting the age as a string or a number."""
    def check(text: str) -> bool:
        parsed = _json_object(text)
        if parsed is None or set(parsed) != {"name", "age"}:
            return False
        value = parsed["age"]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return False
        try:
            numeric = int(float(str(value).strip()))
        except ValueError:
            return False
        return (
            isinstance(parsed["name"], str)
            and normalize(parsed["name"]).lower() == name.lower()
            and numeric == int(age)
        )
    return check


def _json_value(key: str, expected: bool | int):
    """Grade a single-key JSON object, keeping booleans and integers distinct."""
    def check(text: str) -> bool:
        parsed = _json_object(text)
        if parsed is None or set(parsed) != {key}:
            return False
        value = parsed[key]
        if isinstance(expected, bool):
            return value is expected
        return isinstance(value, int) and not isinstance(value, bool) and value == expected
    return check


def _build() -> tuple[Task, ...]:
    tasks: list[Task] = []
    for left, right, total in _ADDITIONS:
        tasks.append(Task(
            f"arithmetic.add.{left}_{right}", "arithmetic",
            f"What is {left} + {right}? Answer with the number only.",
            16, _number(total), grades="value",
        ))
    for left, right, total in _SUBTRACTIONS:
        tasks.append(Task(
            f"arithmetic.subtract.{left}_{right}", "arithmetic",
            f"What is {left} - {right}? Answer with the number only.",
            16, _number(total), grades="value",
        ))
    for left, right, total in _PRODUCTS:
        tasks.append(Task(
            f"arithmetic.multiply.{left}_{right}", "arithmetic",
            f"What is {left} * {right}? Answer with the number only.",
            16, _number(total), grades="value",
        ))
    for word in _LETTER_WORDS:
        tasks.append(Task(
            f"arithmetic.count.{word}", "arithmetic",
            f"How many letters are in the word '{word}'? Answer with the number only.",
            16, _number(len(word)), grades="value",
        ))
    for word in _UPPERCASE:
        tasks.append(Task(
            f"instruction.upper.{word}", "instruction",
            f"Output only the uppercase form of the word '{word}'. No explanation.",
            16, _exact(word.upper()),
            value_check=_contains_cased(word.upper()),
        ))
    for word in _LOWERCASE:
        tasks.append(Task(
            f"instruction.lower.{word}", "instruction",
            f"Output only the lowercase form of the word '{word}'. No explanation.",
            16, _exact(word.lower()),
            value_check=_contains_cased(word.lower()),
        ))
    for word in _ECHO:
        tasks.append(Task(
            f"instruction.echo.{word}", "instruction",
            f"Reply with exactly the word {word} and nothing else.",
            16, _exact(word),
            grades="form",
        ))
    for count in _ITEM_COUNTS:
        tasks.append(Task(
            f"instruction.items.{count}", "instruction",
            f"List exactly {count} animals, comma-separated, with no other text.",
            48, _items(count),
            value_check=_items_value(count),
        ))
    for name, age in _PEOPLE:
        tasks.append(Task(
            f"format.json_person.{name.lower()}", "format",
            'Return only JSON with exactly the keys "name" and "age" for this '
            f"sentence: '{name} is {age} years old.' No markdown, no commentary.",
            48, _json_person(name, age),
            grades="form",
        ))
    for word in _CHAR_WORDS:
        tasks.append(Task(
            f"format.json_count.{word}", "format",
            'Return only JSON of the form {"count": <integer>} with the number of '
            f'characters in the word "{word}". No other text.',
            32, _json_value("count", len(word)),
            value_check=_number(len(word)),
        ))
    for number, even in _PARITY:
        tasks.append(Task(
            f"format.json_even.{number}", "format",
            'Return only JSON of the form {"even": <true or false>} stating whether '
            f"{number} is even. No other text.",
            32, _json_value("even", even),
            rule="bool:v2",
            value_check=_bool_value(even),
        ))
    for index, (sentence, address, target) in enumerate(_EMAILS):
        tasks.append(Task(
            f"extraction.email.{index}", "extraction",
            f"Give the {target} from this text: '{sentence}'",
            32, _only_email(address), rule="email:v2", grades="value",
        ))
    for index, (sentence, iso, target) in enumerate(_DATES):
        tasks.append(Task(
            f"extraction.date.{index}", "extraction",
            f"Give the {target} in YYYY-MM-DD form from this text: '{sentence}'",
            32, _only_date(iso), grades="value",
        ))
    for index, (listing, largest) in enumerate(_MAXIMA):
        tasks.append(Task(
            f"extraction.max.{index}", "extraction",
            f"Output only the largest number in this list: {listing}.",
            16, _number(largest), rule="max:v2", grades="value",
        ))
    for index, (prompt, expected, rivals) in enumerate(_SUBSTRINGS):
        tasks.append(Task(
            f"extraction.span.{index}", "extraction", prompt, 32,
            _only_span(expected, rivals), grades="value",
        ))
    for index, (sentence, address, target) in enumerate(_EMAILS[:3]):
        tasks.append(Task(
            f"compliance.email.{index}", "compliance",
            f"Output only the {target}, no other words: '{sentence}'",
            32, _exact(address), rule="email:v2",
            value_check=_only_email(address),
        ))
    for index, (sentence, iso, target) in enumerate(_DATES[:3]):
        tasks.append(Task(
            f"compliance.date.{index}", "compliance",
            f"Output only the {target} in YYYY-MM-DD form, no other words: "
            f"'{sentence}'",
            32, _exact(iso),
            value_check=_only_date(iso),
        ))
    for index, (prompt, expected, _rivals) in enumerate(_SUBSTRINGS[:2]):
        tasks.append(Task(
            f"compliance.span.{index}", "compliance",
            f"{prompt} Output only that value, no other words.",
            32, _exact(expected),
            value_check=_only_span(expected, _rivals),
        ))
    for index, (prompt, required, value_check, rule) in enumerate(_JAPANESE):
        tasks.append(Task(
            f"multilingual.ja.{index}", "multilingual", prompt, 48,
            _japanese_only(required),
            rule=rule,
            value_check=value_check,
        ))
    return tuple(tasks)


GENERATED_TASKS: tuple[Task, ...] = _build()


EXTENDED_TASKS: tuple[Task, ...] = TASKS + GENERATED_TASKS
HARD_SUITE_TASKS: tuple[Task, ...] = EXTENDED_TASKS + HARD_TASKS
SUITES: dict[str, tuple[Task, ...]] = {
    "core": TASKS,
    "extended": EXTENDED_TASKS,
    "hard": HARD_SUITE_TASKS,
}
EXTENDED_CATEGORIES: tuple[str, ...] = tuple(
    dict.fromkeys(task.category for task in EXTENDED_TASKS)
)


__all__ = [
    "EXTENDED_CATEGORIES",
    "EXTENDED_TASKS",
    "GENERATED_TASKS",
    "HARD_SUITE_TASKS",
    "SUITES",
]
