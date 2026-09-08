"""Deterministic model evaluation."""

from .cache import EvalRecord, EvalSummary, load_eval_cache, save_eval
from .depth import needle_tasks, padded_prompt
from .generated import (
    EXTENDED_CATEGORIES,
    EXTENDED_TASKS,
    GENERATED_TASKS,
    HARD_SUITE_TASKS,
    SUITES,
)
from .runner import EvalRun, TaskOutcome, run
from .suite import CATEGORIES, GRADER_VERSION, TASKS, Task, normalize, suite_digest

__all__ = [
    "CATEGORIES",
    "EXTENDED_CATEGORIES",
    "EXTENDED_TASKS",
    "GENERATED_TASKS",
    "GRADER_VERSION",
    "HARD_SUITE_TASKS",
    "SUITES",
    "TASKS",
    "EvalRecord",
    "EvalRun",
    "EvalSummary",
    "Task",
    "TaskOutcome",
    "load_eval_cache",
    "needle_tasks",
    "normalize",
    "padded_prompt",
    "run",
    "save_eval",
    "suite_digest",
]
