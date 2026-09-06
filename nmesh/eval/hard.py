"""Harder paired-power tasks for the measured saturated families.

The ``instruction`` family was measured at 18/18 versus 18/18 and
``multilingual`` at 6/6 versus 6/6. Removing concordant tasks cannot raise
paired power, so the suite grows instead. All 26 tasks were authored before
measurement and all are shipped; there was no selection on measured
discordance. On this pair, 4 were discordant and 16 failed on both.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .suite import Task, _exact, _japanese_only, _only_date, normalize

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_KATAKANA_ONLY = re.compile(r"^[\u30a0-\u30ff\u30fc\s]+$")
_PUNCT = re.compile(r"[.,!?;:\"'`。、！？]")


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

HARD_TASKS: tuple[Task, ...] = (
    Task(
        "instruction.words.5", "instruction",
        "Answer in exactly 5 words, with no punctuation at all: what does a "
        "compiler do?", 48, _word_count(5),
    ),
    Task(
        "instruction.words.3", "instruction",
        "Answer in exactly 3 words, with no punctuation at all: describe the "
        "ocean.", 48, _word_count(3),
    ),
    Task(
        "instruction.words.7", "instruction",
        "Answer in exactly 7 words, with no punctuation at all: why do people "
        "read books?", 48, _word_count(7),
    ),
    Task(
        "instruction.devowel.orchestrator", "instruction",
        "Output only the word 'orchestrator' with every vowel removed. No "
        "explanation.", 24, _exact("rchstrtr"),
    ),
    Task(
        "instruction.devowel.gateway", "instruction",
        "Output only the word 'gateway' with every vowel removed. No explanation.",
        24, _exact("gtwy"),
    ),
    Task(
        "instruction.reverse.planner", "instruction",
        "Output only the word 'planner' spelled backwards, in lowercase. No "
        "explanation.", 24, _exact("rennalp"),
    ),
    Task(
        "instruction.reverse.slot", "instruction",
        "Output only the word 'slot' spelled backwards, in lowercase. No "
        "explanation.", 24, _exact("tols"),
    ),
    Task(
        "instruction.initials.quick_amber_fox", "instruction",
        "Output only the first letter of each word in 'quick amber fox', in "
        "uppercase, with no separators and no other text.", 24, _exact("QAF"),
    ),
    Task(
        "instruction.initials.local_model_mesh", "instruction",
        "Output only the first letter of each word in 'local model mesh', in "
        "uppercase, with no separators and no other text.", 24, _exact("LMM"),
    ),
    Task(
        "instruction.repeat.slot.4", "instruction",
        "Repeat the word slot exactly 4 times, separated by single spaces, and "
        "output nothing else.", 32, _exact("slot slot slot slot"),
    ),
    Task(
        "instruction.repeat.plan.3", "instruction",
        "Repeat the word plan exactly 3 times, separated by single spaces, and "
        "output nothing else.", 32, _exact("plan plan plan"),
    ),
    Task(
        "instruction.nth_word.3", "instruction",
        "Output only the third word of this sentence, nothing else: 'The planner "
        "selects a model for the machine.'", 24, _exact("selects"),
    ),
    Task(
        "instruction.nth_word.5", "instruction",
        "Output only the fifth word of this sentence, nothing else: 'The gateway "
        "routes every incoming request quickly.'", 24, _exact("incoming"),
    ),
    Task(
        "arithmetic.prime.91", "arithmetic",
        "Answer with only the word yes or no, nothing else: is 91 a prime number?",
        16, _exact("no"),
    ),
    Task(
        "arithmetic.prime.97", "arithmetic",
        "Answer with only the word yes or no, nothing else: is 97 a prime number?",
        16, _exact("yes"),
    ),
    Task(
        "instruction.sort_desc.3_1_2", "instruction",
        "Output the numbers 3, 1, 2 in descending order, comma-separated digits "
        "only, with no other text.", 24, _digits((3, 2, 1)),
    ),
    Task(
        "instruction.sort_asc.40_7_19", "instruction",
        "Output the numbers 40, 7, 19 in ascending order, comma-separated digits "
        "only, with no other text.", 24, _digits((7, 19, 40)),
    ),
    Task(
        "multilingual.en_from_ja.0", "multilingual",
        "次の日本語を英語に訳し、英語の訳文だけを出力してください: '犬が走る。'",
        48, _english_only((("dog",), ("run", "running"))),
    ),
    Task(
        "multilingual.en_from_ja.1", "multilingual",
        "次の日本語を英語に訳し、英語の訳文だけを出力してください: "
        "'私は毎朝水を飲みます。'",
        48, _english_only((("water",), ("drink", "drinking"))),
    ),
    Task(
        "multilingual.katakana.computer", "multilingual",
        "「computer」をカタカナで書いてください。カタカナ以外は出力しないでください。",
        32, _katakana_only("コンピ"),
    ),
    Task(
        "multilingual.katakana.model", "multilingual",
        "「model」をカタカナで書いてください。カタカナ以外は出力しないでください。",
        32, _katakana_only("モデル"),
    ),
    Task(
        "multilingual.kanji_number.17", "multilingual",
        "17 を漢数字で書いてください。漢数字だけを出力してください。",
        24, _exact("十七"),
    ),
    Task(
        "multilingual.kanji_number.30", "multilingual",
        "30 を漢数字で書いてください。漢数字だけを出力してください。",
        24, _exact("三十"),
    ),
    Task(
        "multilingual.lang_lock.paris", "multilingual",
        "Answer only in Japanese, using no Latin letters at all: what is the "
        "capital of France?", 32, _japanese_only("パリ"),
    ),
    Task(
        "multilingual.lang_lock.seven", "multilingual",
        "Answer only in Japanese, using no Latin letters at all: how many days "
        "are in one week?", 32, _japanese_only("七"),
    ),
    Task(
        "multilingual.ja_extract.date", "multilingual",
        "次の文から日付だけを YYYY-MM-DD 形式で出力してください: "
        "'監査は2021年7月9日に大阪で終わった。'", 32, _only_date("2021-07-09"),
    ),
)


__all__ = ["HARD_TASKS"]
