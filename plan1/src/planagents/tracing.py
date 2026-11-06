"""Timing and superstep helpers used to prove what actually ran in parallel."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    start: float
    end: float = 0.0


@dataclass
class Timeline:
    """Records when each task ran. Lives outside graph state on purpose."""

    spans: dict[str, Span] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        span = Span(start=time.perf_counter())
        self.spans[name] = span
        try:
            yield
        finally:
            span.end = time.perf_counter()

    def overlap(self, *names: str) -> bool:
        """True if there is a moment when all the named tasks were running at once."""
        spans = [self.spans[name] for name in names]
        return max(s.start for s in spans) < min(s.end for s in spans)

    def starts_after(self, name: str, *upstream: str) -> bool:
        return all(self.spans[name].start >= self.spans[up].end for up in upstream)


async def trace_supersteps(graph: Any, graph_input: Any, barrier: str = "scheduler", config: Any = None) -> list[list[str]]:
    """Stream ``updates`` and group worker task IDs into the batches between barrier hits."""
    batches: list[list[str]] = [[]]
    async for chunk in graph.astream(graph_input, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == barrier:
                if batches[-1]:
                    batches.append([])
            elif node.startswith("worker") and update:
                batches[-1].extend(update.get("tasks", {}).keys())
    return [batch for batch in batches if batch]
