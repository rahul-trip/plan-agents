"""Stage 2 stretch: dispatch as soon as a dependency finishes instead of at a barrier.

A superstep (and asyncio.gather level by level) waits for the slowest task in the batch, even if
an unrelated task is already unblocked. The test runs this and your ex1 engine on the same plan and
compares them, so finish ex1 first. Then consider what the barrier buys: one atomic, checkpointable
state at each step.
"""

from __future__ import annotations

import asyncio
from graphlib import TopologicalSorter
from typing import Awaitable, Callable

from planagents.schemas import ExecutionPlan, TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


async def run_plan_eager(plan: ExecutionPlan, worker: Worker) -> dict[str, TaskNode]:
    """Keep a set of running asyncio tasks. Use asyncio.wait(return_when=FIRST_COMPLETED), call
    sorter.done() for each finished id, then start whatever get_ready() newly returns.

    Workers in this exercise are assumed to succeed; let any exception propagate.
    """
    raise NotImplementedError("loop while sorter.is_active(): start get_ready() tasks, wait FIRST_COMPLETED, record results, done()")
