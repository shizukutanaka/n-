"""Deterministic model evaluation."""

from .cache import EvalRecord, EvalSummary, load_eval_cache, save_eval
from .context import (
    ContextRecord,
    FamilyResult,
    context_key,
    load_context_cache,
    save_context,
)
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
    "ContextRecord",
    "EvalRecord",
    "EvalRun",
    "EvalSummary",
    "FamilyResult",
    "Task",
    "TaskOutcome",
    "context_key",
    "load_context_cache",
    "load_eval_cache",
    "needle_tasks",
    "normalize",
    "padded_prompt",
    "run",
    "save_context",
    "save_eval",
    "suite_digest",
]
