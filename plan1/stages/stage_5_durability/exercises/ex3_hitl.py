"""Stage 5, ex3: human-in-the-loop with interrupt().

Two gates: `plan_review` right after planning (inspect, edit, prune), and `destructive_gate` before any
task marked destructive (approve, reject, or change its arguments).

interrupt() re-runs its node from the top on resume, so:
- Keep planning in a separate node, or every resume calls the LLM again.
- Do the side effect AFTER the interrupt, never before it.
"""

from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from planagents.dag import (
    PlanValidationError,
    Workers,
    cancel_with_dependents,
    check_deadlock,
    make_plan,
    merge_tasks,
    ready_frontier,
    validate_plan,
)
from planagents.schemas import TaskNode

Worker = Callable[[TaskNode, dict[str, str]], Awaitable[str]]


def merge_errors(existing: dict[str, str | None] | None, updates: dict[str, str | None]) -> dict[str, str | None]:
    return {**(existing or {}), **updates}


class HitlState(TypedDict, total=False):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    review_error: str | None
    gate_errors: Annotated[dict[str, str | None], merge_errors]
    final_output: str


EDIT_FIELD_TYPES = {"instruction": str, "args": dict, "destructive": bool}


def apply_review(tasks: dict[str, TaskNode], decision: Any) -> dict[str, TaskNode]:
    """Apply a human decision and return the delta for the tasks channel.

    decision looks like {'action': 'approve'|'edit'|'reject', 'edits': {id: {field: value}},
    'prune': [ids]}. Allowed edit fields are 'instruction' (a str), 'args' (a dict) and 'destructive'
    (a bool), so a reviewer can put a gate on a task the registry missed.

    - approve: no changes.
    - reject: cancel every task.
    - edit: apply edits, then prune with cancel_with_dependents (a merge reducer can't delete, and a
      dependent left pending would deadlock).
    A decision that isn't a dict, an unknown action, id or field, or a value of the wrong type raises
    PlanValidationError: a typo must not silently do nothing.
    """
    raise NotImplementedError("validate the decision, ids, fields and value types, then build the delta for each action")


def parse_gate_decision(decision: Any) -> dict[str, Any]:
    """Check a destructive-gate decision: a dict with 'approved' (a bool) and optionally 'args' (a dict).

    Anything else raises PlanValidationError. Treating a malformed answer as "rejected" would look
    safe, but it would silently cancel work a person meant to approve.
    """
    raise NotImplementedError("check the types above, raise PlanValidationError on anything else, return the decision")


def build_graph(llm, worker: Worker, checkpointer, workers: Workers | None = None):
    """planner -> plan_review -> scheduler -> (worker | destructive_gate) -> scheduler -> synthesize.

    planner: make_plan(llm, prompt, workers). `destructive` comes from the workers registry, in code.
    A gate the LLM could skip by not flagging a task would be no gate at all.

    plan_review: interrupt({'type': 'plan_review', 'tasks': [JSON dumps], 'error': state's
    review_error}), then apply_review. THE TRAP: LangGraph saves a resume value, so if this node raises
    after the interrupt, that value replays on every later resume and the thread is stuck for good. So
    catch PlanValidationError from apply_review (not around interrupt(), which pauses by raising) and
    ask again: Command(update={'review_error': str(exc)}, goto='plan_review'). On success, clear
    review_error and goto='scheduler'. Don't add a static edge out of plan_review.

    scheduler: cancels pending tasks whose parent was cancelled, as a delta. Router: destructive
    frontier tasks go to destructive_gate, others to worker; when nothing is pending, synthesize.
    Send each gate its state's gate_errors entry as payload['error'].

    destructive_gate: interrupt({'type': 'destructive_action', 'task_id': ..., 'task': dump, 'error':
    payload's error}), then parse_gate_decision. An invalid decision must not raise either: return
    {'gate_errors': {id: str(exc)}} and leave the task pending, so the router sends it here again.
    Otherwise clear its gate_errors entry. Approved: run the worker with any new args, marking
    completed or failed. Rejected: status='cancelled', and the worker never runs.
    """
    raise NotImplementedError("write the nodes; side effects only after interrupt returns; compile with the checkpointer")
