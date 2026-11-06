"""A scripted LLM so every exercise runs offline and deterministically.

Same surface the curriculum uses on a real chat model: ``ainvoke``, ``invoke`` and
``with_structured_output``. It records every prompt, so tests can assert what a worker
actually saw and how many times the planner was called.

Text replies deliberately use the shape Claude returns when it thinks: content is a list of blocks,
not a string. Code that reads ``reply.content`` breaks here just as it would live, so use
``planagents.llm.reply_text``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from langchain_core.messages import AIMessage, BaseMessage

from planagents.schemas import ExecutionPlan

TEXT = "text"


class ScriptExhausted(AssertionError):
    """The code made an LLM call the test did not script."""


@dataclass
class Call:
    key: str  # Schema class name for structured calls, otherwise "text".
    prompt: str


def _render(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Iterable):
        parts = []
        for message in messages:
            parts.append(str(message.content) if isinstance(message, BaseMessage) else str(message))
        return "\n".join(parts)
    return str(messages)


class ScriptedLLM:
    """Replies come from per-key queues, or from a handler ``fn(key, prompt) -> reply``.

    ScriptedLLM({"PlanDraft": [plan], "text": ["first", "second"]})

    Keying by schema keeps tests deterministic when parallel workers interleave calls. A PlanDraft
    reply may be an ExecutionPlan, for convenience. Its runtime fields don't survive the conversion,
    just as they wouldn't if an LLM wrote them.
    """

    def __init__(self, script: dict[str, list[Any]] | Callable[[str, str], Any] | None = None):
        self._handler = script if callable(script) else None
        queues = script if isinstance(script, dict) else {}
        self._queues = defaultdict(deque, {key: deque(values) for key, values in queues.items()})
        self.calls: list[Call] = []

    def calls_for(self, key: str) -> list[Call]:
        return [call for call in self.calls if call.key == key]

    def prompts(self) -> str:
        return "\n".join(call.prompt for call in self.calls)

    def _next(self, key: str, messages: Any) -> Any:
        prompt = _render(messages)
        self.calls.append(Call(key=key, prompt=prompt))
        if self._handler is not None:
            return self._handler(key, prompt)
        if not self._queues[key]:
            raise ScriptExhausted(f"Unscripted {key!r} call. Prompt was:\n{prompt[:500]}")
        reply = self._queues[key].popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def invoke(self, messages: Any, config: Any = None, **_: Any) -> AIMessage:
        text = str(self._next(TEXT, messages))
        thinking = {"type": "thinking", "thinking": "", "signature": "scripted"}
        return AIMessage(content=[thinking, {"type": "text", "text": text}])

    async def ainvoke(self, messages: Any, config: Any = None, **_: Any) -> AIMessage:
        return self.invoke(messages)

    def with_structured_output(self, schema: type, **_: Any) -> "_Structured":
        return _Structured(self, schema)


class _Structured:
    def __init__(self, llm: ScriptedLLM, schema: type):
        self._llm, self._schema = llm, schema

    def invoke(self, messages: Any, config: Any = None, **_: Any) -> Any:
        reply = self._llm._next(self._schema.__name__, messages)
        if isinstance(reply, ExecutionPlan):
            reply = reply.to_draft()
        return reply if isinstance(reply, self._schema) else self._schema.model_validate(reply)

    async def ainvoke(self, messages: Any, config: Any = None, **_: Any) -> Any:
        return self.invoke(messages)
