"""Harder paired-power tasks for the measured saturated families.

The ``instruction`` family was measured at 18/18 versus 18/18 and
``multilingual`` at 6/6 versus 6/6. Removing concordant tasks cannot raise
paired power, so the suite grows instead. All 26 tasks were authored before
measurement and all are shipped; there was no selection on measured
discordance. On this pair, 4 were discordant and 16 failed on both. Inspecting
real outputs found two defective items: one accepted-answer set was too narrow
and one prompt constraint was weaker than its grader. Both were repaired rather
than deleted; the repair changes the ``hard`` digest, superseding the two
stored 130-task records.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .suite import (
    Task,
    _contains_ci,
    _exact,
    _japanese_only,
    _only_date,
    _yes_no_value,
    normalize,
)

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_KATAKANA_ONLY = re.compile(r"^[\u30a0-\u30ff\u30fc\s]+$")
_PUNCT = re.compile(r"[.,!?;:\"'`。、！？]")
_KANJI_DIGITS = re.compile(
    r"[〇零一二三四五六七八九十百千壱壹弐貳参參肆伍陸柒捌拾]+"
)


def _word_count(count: int) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        value = normalize(text)
        if _PUNCT.search(value):
            return False
        words = value.split()
        return len(words) == count and all(word.isalpha() for word in words)
    return check


def _digits(expected: tuple[int, ...]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        found = tuple(int(item) for item in re.findall(r"-?\d+", text.replace(",", "")))
        return found == expected
    return check


def _english_only(required: tuple[str, ...]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        value = normalize(text)
        if _CJK.search(value):
            return False
        lowered = value.lower()
        return all(any(form in lowered for form in group) for group in required)
    return check


def _katakana_only(required: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        value = normalize(text)
        return bool(value) and bool(_KATAKANA_ONLY.match(value)) and required in value
    return check


def _kanji_number(forms: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda text: normalize(text) in forms


def _letters_value(expected: str) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        letters = "".join(char for char in normalize(text) if char.isalpha())
        return letters.upper() == expected
    return check


def _numbers_value(expected: tuple[int, ...]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        found = tuple(int(item) for item in re.findall(r"-?\d+", normalize(text)))
        return found == expected
    return check


def _word_occurrences(word: str, count: int) -> Callable[[str], bool]:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return lambda text: len(pattern.findall(normalize(text))) == count


def _contains_any(required: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda text: any(
        item.casefold() in normalize(text).casefold() for item in required
    )


def _numeral_value(forms: tuple[str, ...], arabic: str) -> Callable[[str], bool]:
    """True when the text names the target number, in kanji or in digits."""

    def check(text: str) -> bool:
        value = normalize(text)
        runs = set(_KANJI_DIGITS.findall(value)) | set(re.findall(r"\d+", value))
        return bool(runs & (set(forms) | {arabic}))

    return check


def _english_value(required: tuple[tuple[str, ...], ...]) -> Callable[[str], bool]:
    def check(text: str) -> bool:
        lowered = normalize(text).lower()
        return all(any(form in lowered for form in group) for group in required)
    return check


HARD_TASKS: tuple[Task, ...] = (
    Task(
        "instruction.words.5", "instruction",
        "Answer in exactly 5 words, with no punctuation at all: what does a "
        "compiler do?", 48, _word_count(5), grades="form",
    ),
    Task(
        "instruction.words.3", "instruction",
        "Answer in exactly 3 words, with no punctuation at all: describe the "
        "ocean.", 48, _word_count(3), grades="form",
    ),
    Task(
        "instruction.words.7", "instruction",
        "Answer in exactly 7 words, with no punctuation at all: why do people "
        "read books?", 48, _word_count(7), grades="form",
    ),
    Task(
        "instruction.devowel.orchestrator", "instruction",
        "Output only the word 'orchestrator' with every vowel removed. No "
        "explanation.", 24, _exact("rchstrtr"),
        value_check=_contains_ci("rchstrtr"),
    ),
    Task(
        "instruction.devowel.gateway", "instruction",
        "Output only the word 'gateway' with every vowel removed. No explanation.",
        24, _exact("gtwy"),
        value_check=_contains_ci("gtwy"),
    ),
    Task(
        "instruction.reverse.planner", "instruction",
        "Output only the word 'planner' spelled backwards, in lowercase. No "
        "explanation.", 24, _exact("rennalp"),
        value_check=_contains_ci("rennalp"),
    ),
    Task(
        "instruction.reverse.slot", "instruction",
        "Output only the word 'slot' spelled backwards, in lowercase. No "
        "explanation.", 24, _exact("tols"),
        value_check=_contains_ci("tols"),
    ),
    Task(
        "instruction.initials.quick_amber_fox", "instruction",
        "Output only the first letter of each word in 'quick amber fox', in "
        "uppercase, with no separators and no other text.", 24, _exact("QAF"),
        value_check=_letters_value("QAF"),
    ),
    Task(
        "instruction.initials.local_model_mesh", "instruction",
        "Output only the first letter of each word in 'local model mesh', in "
        "uppercase, with no separators and no other text.", 24, _exact("LMM"),
        value_check=_letters_value("LMM"),
    ),
    Task(
        "instruction.repeat.slot.4", "instruction",
        "Repeat the word slot exactly 4 times, separated by single spaces, and "
        "output nothing else.", 32, _exact("slot slot slot slot"),
        value_check=_word_occurrences("slot", 4),
    ),
    Task(
        "instruction.repeat.plan.3", "instruction",
        "Repeat the word plan exactly 3 times, separated by single spaces, and "
        "output nothing else.", 32, _exact("plan plan plan"),
        value_check=_word_occurrences("plan", 3),
    ),
    Task(
        "instruction.nth_word.3", "instruction",
        "Output only the third word of this sentence, nothing else: 'The planner "
        "selects a model for the machine.'", 24, _exact("selects"),
        value_check=_contains_ci("selects"),
    ),
    Task(
        "instruction.nth_word.5", "instruction",
        "Output only the fifth word of this sentence, nothing else: 'The gateway "
        "routes every incoming request quickly.'", 24, _exact("incoming"),
        value_check=_contains_ci("incoming"),
    ),
    Task(
        "arithmetic.prime.91", "arithmetic",
        "Answer with only the word yes or no, nothing else: is 91 a prime number?",
        16, _exact("no"),
        value_check=_yes_no_value(False),
    ),
    Task(
        "arithmetic.prime.97", "arithmetic",
        "Answer with only the word yes or no, nothing else: is 97 a prime number?",
        16, _exact("yes"),
        value_check=_yes_no_value(True),
    ),
    Task(
        "instruction.sort_desc.3_1_2", "instruction",
        "Output the numbers 3, 1, 2 in descending order, comma-separated digits "
        "only, with no other text.", 24, _digits((3, 2, 1)),
        value_check=_numbers_value((3, 2, 1)),
    ),
    Task(
        "instruction.sort_asc.40_7_19", "instruction",
        "Output the numbers 40, 7, 19 in ascending order, comma-separated digits "
        "only, with no other text.", 24, _digits((7, 19, 40)),
        value_check=_numbers_value((7, 19, 40)),
    ),
    Task(
        "multilingual.en_from_ja.0", "multilingual",
        "次の日本語を英語に訳し、英語の訳文だけを出力してください: '犬が走る。'",
        48, _english_only((("dog",), ("run", "running"))),
        value_check=_english_value((("dog",), ("run", "running"))),
    ),
    Task(
        "multilingual.en_from_ja.1", "multilingual",
        "次の日本語を英語に訳し、英語の訳文だけを出力してください: "
        "'私は毎朝水を飲みます。'",
        48, _english_only((("water",), ("drink", "drinking"))),
        value_check=_english_value((("water",), ("drink", "drinking"))),
    ),
    Task(
        "multilingual.katakana.computer", "multilingual",
        "「computer」をカタカナで書いてください。カタカナ以外は出力しないでください。",
        32, _katakana_only("コンピ"),
        value_check=_contains_any(("コンピ", "computer", "計算機")),
    ),
    Task(
        "multilingual.katakana.model", "multilingual",
        "「model」をカタカナで書いてください。カタカナ以外は出力しないでください。",
        32, _katakana_only("モデル"),
        value_check=_contains_any(("モデル", "model", "模型")),
    ),
    Task(
        "multilingual.kanji_number.17", "multilingual",
        "17 を漢数字で書いてください。漢数字だけを出力してください。",
        24,
        _kanji_number(("十七", "一十七", "壹拾柒", "壱拾七")),
        rule="kanji_number:v2",
        value_check=_numeral_value(("十七", "一十七", "壹拾柒", "壱拾七"), "17"),
    ),
    Task(
        "multilingual.kanji_number.30", "multilingual",
        "30 を漢数字で書いてください。漢数字だけを出力してください。",
        24,
        _kanji_number(("三十", "参拾", "參拾")),
        rule="kanji_number:v2",
        value_check=_numeral_value(("三十", "参拾", "參拾"), "30"),
    ),
    Task(
        "multilingual.lang_lock.paris", "multilingual",
        "Answer only in Japanese, using no Latin letters at all: what is the "
        "capital of France?", 32, _japanese_only("パリ"),
        value_check=_contains_any(("パリ", "paris")),
    ),
    Task(
        "multilingual.lang_lock.seven", "multilingual",
        "Answer only in Japanese with no Latin letters and no digits: how many "
        "days are in one week? Output the kanji numeral alone.",
        32, _japanese_only("七"),
        value_check=_contains_any(("七", "7", "seven")),
    ),
    Task(
        "multilingual.ja_extract.date", "multilingual",
        "次の文から日付だけを YYYY-MM-DD 形式で出力してください: "
        "'監査は2021年7月9日に大阪で終わった。'", 32,
        _only_date("2021-07-09"), grades="value",
    ),
)


__all__ = ["HARD_TASKS"]
