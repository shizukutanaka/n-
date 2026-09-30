"""Lead/worker delegation over the OpenAI-compatible surface.

A lead model plans nothing it cannot check: the protocol is one bounded round
of *delegate, verify, escalate*. The worker answers, the lead judges the
answer, and on anything other than an explicit acceptance the lead answers the
task itself. There is no re-delegation loop, so the number of upstream calls
per task is bounded by three regardless of how badly the worker answers.

Both models are addressed through ``/v1/chat/completions``, so the protocol
works against any backend nmesh can plan (llama.cpp, Ollama, vLLM) and against
the nmesh gateway itself.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

#: Bumped whenever the verifier prompt or the escalation rule changes. A
#: measurement made under a different protocol is not comparable, so this
#: string is part of the record identity.
PROTOCOL_VERSION = "delegate-v2"

VERIFY_PROMPT = (
    "You are a strict checker. Decide whether the ANSWER satisfies the TASK "
    "exactly, including every format requirement stated in the task.\n\n"
    "TASK:\n{prompt}\n\nANSWER:\n{answer}\n\n"
    "Reply with exactly one word: YES if the answer is correct and correctly "
    "formatted, NO otherwise."
)

#: The verifier only has to emit one word, but a model that ignores that is
#: escalated rather than parsed loosely.
VERIFY_MAX_TOKENS = 8


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible chat endpoint and the model reference to send."""

    base_url: str
    model_ref: str
    cache_prompt: bool | None = None


@dataclass(frozen=True)
class Call:
    """One upstream completion, with token counts taken from ``usage``."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    unscorable: bool
    seconds: float


@dataclass
class Cost:
    """Accumulated upstream cost for one role in one arm."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def add(self, call: Call) -> None:
        self.calls += 1
        self.prompt_tokens += call.prompt_tokens
        self.completion_tokens += call.completion_tokens
        self.seconds += call.seconds


@dataclass(frozen=True)
class Delegation:
    """The outcome of one delegated task."""

    answer: str
    accepted: bool
    escalated: bool
    unparsed_verdict: bool
    worker: Call
    verify: Call
    rescue: Call | None = None

    @property
    def unscorable(self) -> bool:
        final = self.rescue if self.rescue is not None else self.worker
        return final.unscorable


@dataclass
class Ledger:
    """Per-role cost of running a delegated workload."""

    worker: Cost = field(default_factory=Cost)
    verify: Cost = field(default_factory=Cost)
    rescue: Cost = field(default_factory=Cost)

    @property
    def lead_prompt_tokens(self) -> int:
        return self.verify.prompt_tokens + self.rescue.prompt_tokens

    @property
    def lead_completion_tokens(self) -> int:
        return self.verify.completion_tokens + self.rescue.completion_tokens

    @property
    def seconds(self) -> float:
        return self.worker.seconds + self.verify.seconds + self.rescue.seconds


def complete(
    client: httpx.Client,
    endpoint: Endpoint,
    prompt: str,
    max_tokens: int,
) -> Call:
    """Send one deterministic single-turn completion and measure it."""
    started = time.monotonic()
    payload: dict[str, object] = {
        "model": endpoint.model_ref,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if endpoint.cache_prompt is not None:
        payload["cache_prompt"] = endpoint.cache_prompt
    response = client.post(
        f"{endpoint.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
    )
    response.raise_for_status()
    body = response.json()
    return Call(
        text=_content(body),
        prompt_tokens=_usage(body, "prompt_tokens"),
        completion_tokens=_usage(body, "completion_tokens"),
        unscorable=_truncated_empty(body),
        seconds=time.monotonic() - started,
    )


def read_verdict(text: str) -> bool | None:
    """Return the verifier's decision, or ``None`` when it did not answer.

    An unreadable verdict is not an acceptance: callers escalate instead, so a
    verifier that emits prose can never silently pass a wrong answer.
    """
    upper = text.strip().upper()
    if upper.startswith("YES"):
        return True
    if upper.startswith("NO"):
        return False
    yes = upper.count("YES")
    no = upper.count("NO")
    if yes == 1 and no == 0:
        return True
    if no == 1 and yes == 0:
        return False
    return None


def delegate(
    client: httpx.Client,
    prompt: str,
    max_tokens: int,
    *,
    lead: Endpoint,
    worker: Endpoint,
    ledger: Ledger | None = None,
    worker_client: httpx.Client | None = None,
) -> Delegation:
    """Run one bounded delegate/verify/escalate round for ``prompt``.

    ``client`` reaches the lead; ``worker_client`` may differ when the two
    endpoints need different transports (e.g. only one sits behind an env
    proxy).
    """
    book = ledger if ledger is not None else Ledger()
    answer = complete(
        client if worker_client is None else worker_client,
        worker,
        prompt,
        max_tokens,
    )
    book.worker.add(answer)
    verify = complete(
        client,
        lead,
        VERIFY_PROMPT.format(prompt=prompt, answer=answer.text),
        VERIFY_MAX_TOKENS,
    )
    book.verify.add(verify)
    decision = read_verdict(verify.text)
    if decision is True:
        return Delegation(
            answer=answer.text,
            accepted=True,
            escalated=False,
            unparsed_verdict=False,
            worker=answer,
            verify=verify,
        )
    rescue = complete(client, lead, prompt, max_tokens)
    book.rescue.add(rescue)
    return Delegation(
        answer=rescue.text,
        accepted=False,
        escalated=True,
        unparsed_verdict=decision is None,
        worker=answer,
        verify=verify,
        rescue=rescue,
    )


def _content(payload: object) -> str:
    choice = _first_choice(payload)
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    return content if isinstance(content, str) else ""


def _truncated_empty(payload: object) -> bool:
    choice = _first_choice(payload)
    finish = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    return finish == "length" and not _content(payload).strip()


def _first_choice(payload: object) -> Mapping[str, object] | None:
    choices = payload.get("choices") if isinstance(payload, Mapping) else None
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, Mapping) else None


def _usage(payload: object, key: str) -> int:
    usage = payload.get("usage") if isinstance(payload, Mapping) else None
    value = usage.get(key) if isinstance(usage, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value
