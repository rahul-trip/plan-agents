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
    # >>> TODO: validate the decision, ids, fields and value types, then build the delta for each action
    if not isinstance(decision, dict):
        raise PlanValidationError(f"Decision must be an object, got: {decision!r}")
    action = decision.get("action")
    if action == "approve":
        return {}
    if action == "reject":
        return cancel_with_dependents(tasks, list(tasks))
    if action != "edit":
        raise PlanValidationError(f"Unknown action: {action}")

    edits, prune = decision.get("edits", {}), decision.get("prune", [])
    unknown = (set(edits) | set(prune)) - set(tasks)
    if unknown:
        raise PlanValidationError(f"Unknown task IDs: {sorted(unknown)}")
    delta: dict[str, TaskNode] = {}
    for task_id, changes in edits.items():
        bad_fields = set(changes) - EDIT_FIELD_TYPES.keys()
        if bad_fields:
            raise PlanValidationError(f"Fields can't be edited: {sorted(bad_fields)}")
        for name, value in changes.items():
            if not isinstance(value, EDIT_FIELD_TYPES[name]):
                raise PlanValidationError(f"{task_id}.{name} has the wrong type: {value!r}")
        delta[task_id] = tasks[task_id].model_copy(update=changes)
    return {**delta, **cancel_with_dependents(tasks, prune)}
    # <<<


def parse_gate_decision(decision: Any) -> dict[str, Any]:
    """Check a destructive-gate decision: a dict with 'approved' (a bool) and optionally 'args' (a dict).

    Anything else raises PlanValidationError. Treating a malformed answer as "rejected" would look
    safe, but it would silently cancel work a person meant to approve.
    """
    # >>> TODO: check the types above, raise PlanValidationError on anything else, return the decision
    if not isinstance(decision, dict) or not isinstance(decision.get("approved"), bool):
        raise PlanValidationError(f"Expected {{'approved': true or false}}, got: {decision!r}")
    if "args" in decision and not isinstance(decision["args"], dict):
        raise PlanValidationError(f"args must be an object, got: {decision['args']!r}")
    return decision
    # <<<


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
    # >>> TODO: write the nodes; side effects only after interrupt returns; compile with the checkpointer
    async def planner(state: HitlState) -> dict:
        plan = await make_plan(llm, f"Plan a task DAG for: {state['objective']}", workers)
        return {"tasks": validate_plan(plan)}

    async def plan_review(state: HitlState) -> Command[Literal["plan_review", "scheduler"]]:
        tasks = state["tasks"]
        decision = interrupt(
            {
                "type": "plan_review",
                "tasks": [t.model_dump() for t in tasks.values()],
                "error": state.get("review_error"),
            }
        )
        try:
            delta = apply_review(tasks, decision)
        except PlanValidationError as exc:
            return Command(update={"review_error": str(exc)}, goto="plan_review")
        return Command(update={"tasks": delta, "review_error": None}, goto="scheduler")

    async def scheduler(state: HitlState) -> dict:
        tasks = state["tasks"]
        cancelled = [t.id for t in tasks.values() if t.status == "cancelled"]
        delta = {
            task_id: task
            for task_id, task in cancel_with_dependents(tasks, cancelled).items()
            if tasks[task_id].status == "pending"
        }
        return {"tasks": delta} if delta else {}

    def route(state: HitlState) -> list[Send] | Literal["synthesize"]:
        tasks = state["tasks"]
        if not any(t.status == "pending" for t in tasks.values()):
            return "synthesize"
        if any(t.status == "failed" for t in tasks.values()):
            return "synthesize"
        check_deadlock(tasks)
        sends = []
        for task in ready_frontier(tasks):
            upstream = {dep: tasks[dep].result or "" for dep in task.depends_on}
            if task.destructive:
                error = state.get("gate_errors", {}).get(task.id)
                sends.append(Send("destructive_gate", {"task": task, "upstream": upstream, "error": error}))
            else:
                sends.append(Send("worker", {"task": task, "upstream": upstream}))
        return sends

    async def execute(task: TaskNode, upstream: dict[str, str]) -> dict:
        try:
            update = {"status": "completed", "result": await worker(task, upstream)}
        except Exception as exc:
            update = {"status": "failed", "result": f"{type(exc).__name__}: {exc}"}
        return {"tasks": {task.id: task.model_copy(update=update)}}

    async def worker_node(payload: dict) -> dict:
        return await execute(payload["task"], payload["upstream"])

    async def destructive_gate(payload: dict) -> dict:
        task = payload["task"]
        answer = interrupt(
            {
                "type": "destructive_action",
                "task_id": task.id,
                "task": task.model_dump(),
                "error": payload.get("error"),
            }
        )
        try:
            decision = parse_gate_decision(answer)
        except PlanValidationError as exc:
            return {"gate_errors": {task.id: str(exc)}}
        cleared = {"gate_errors": {task.id: None}}
        if not decision["approved"]:
            return {**cleared, "tasks": {task.id: task.model_copy(update={"status": "cancelled"})}}
        if "args" in decision:
            task = task.model_copy(update={"args": decision["args"]})
        return {**cleared, **await execute(task, payload["upstream"])}

    async def synthesize(state: HitlState) -> dict:
        lines = [f"{t.id} [{t.status}]: {t.result}" for t in state["tasks"].values()]
        return {"final_output": "\n".join(lines)}

    builder = StateGraph(HitlState)
    builder.add_node("planner", planner)
    builder.add_node("plan_review", plan_review)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker", worker_node)
    builder.add_node("destructive_gate", destructive_gate)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "plan_review")
    builder.add_conditional_edges("scheduler", route, ["worker", "destructive_gate", "synthesize"])
    builder.add_edge("worker", "scheduler")
    builder.add_edge("destructive_gate", "scheduler")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)
    # <<<
