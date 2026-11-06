"""Stage 3 lab and capstone: SQL + Python + Search workers inside the DAG, with zero leakage.

Each domain worker sits behind a mapping node: parent state goes in as a minimal input, and only a
verified artifact comes back. Internal message histories stay inside the subgraphs.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from planagents.dag import check_deadlock, merge_tasks, ready_frontier, validate_plan
from planagents.llm import reply_text
from planagents.loader import sibling
from planagents.schemas import ExecutionPlan, TaskNode


class WorkerInputState(TypedDict):
    task: TaskNode
    inputs: dict[str, str]


class ParentState(TypedDict, total=False):
    objective: str
    tasks: Annotated[dict[str, TaskNode], merge_tasks]
    final_output: str


def search_corpus(corpus: dict[str, str], query: str, top_k: int = 2) -> list[str]:
    """Rank documents by how many distinct query words they contain, and return the best titles.

    Documents with no matches are excluded; ties keep corpus order. A local stand-in for a search
    API, so this works with no key. Swap in a real one behind the same signature.
    """
    # >>> TODO: tokenize lowercased words with re.findall(r"\w+", ...), score by set intersection, sort by score (stable)
    words = set(re.findall(r"\w+", query.lower()))
    scored = [(len(words & set(re.findall(r"\w+", text.lower()))), title) for title, text in corpus.items()]
    ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: -item[0])
    return [title for _, title in ranked[:top_k]]
    # <<<


def build_search_subgraph(llm, corpus: dict[str, str]):
    """Search -> Summarize, with the same shape as the others: {'question'} in,
    {'summary', 'verified'} out. verified is False when nothing matched (and no LLM call)."""

    class SearchInput(TypedDict):
        question: str

    class SearchOutput(TypedDict):
        summary: str
        verified: bool

    class SearchState(SearchInput, SearchOutput, total=False):
        hits: list[str]

    # >>> TODO: search node calls search_corpus; summarize node calls llm with the hit documents, or returns verified=False with no hits
    async def search(state: SearchState) -> dict:
        return {"hits": search_corpus(corpus, state["question"])}

    async def summarize(state: SearchState) -> dict:
        if not state["hits"]:
            return {"summary": "No relevant documents found.", "verified": False}
        documents = "\n".join(f"[{title}] {corpus[title]}" for title in state["hits"])
        reply = await llm.ainvoke(f"Answer '{state['question']}' using only:\n{documents}")
        return {"summary": reply_text(reply), "verified": True}

    builder = StateGraph(SearchState, input_schema=SearchInput, output_schema=SearchOutput)
    builder.add_node("search", search)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()
    # <<<


def build_parent_graph(llm, plan: ExecutionPlan, db_path: str, corpus: dict[str, str], checkpointer=None):
    """Parent DAG with worker kinds 'sql', 'python' and 'search'.

    Every subgraph is invoked inside the worker node, which is the state-mapping boundary:
    - sql: {'question': instruction + inputs, 'db_path': db_path}
    - python: {'instruction': instruction + inputs, 'test_code': task.args['test_code']}
    - search: {'question': instruction + inputs}

    The worker node returns ONLY {'tasks': {id: updated}}. result holds the summary or artifact.
    verified=False maps to status='failed'; otherwise 'completed'. Any exception (an unknown worker,
    a missing database, an LLM error) also becomes status='failed', with the error as result.

    The route function goes to 'synthesize' on any failed task or when nothing is pending, and
    otherwise sends WorkerInputState for the frontier. synthesize writes final_output, a line per task
    naming its status and result.
    """
    subgraphs = sibling(__file__, "ex4_subgraphs")
    code_graph = subgraphs.build_code_generator_subgraph(llm)
    analyst_graph = subgraphs.build_data_analyst_subgraph(llm)
    search_graph = build_search_subgraph(llm, corpus)
    # >>> TODO: write scheduler, route, worker (mapping to each subgraph), synthesize; compile with the checkpointer
    tasks = validate_plan(plan)

    async def load_plan(state: ParentState) -> dict:
        return {"tasks": tasks}

    async def scheduler(state: ParentState) -> dict:
        return {}

    def route(state: ParentState) -> list[Send] | Literal["synthesize"]:
        current = state["tasks"]
        if any(t.status == "failed" for t in current.values()):
            return "synthesize"
        if not any(t.status == "pending" for t in current.values()):
            return "synthesize"
        check_deadlock(current)
        return [
            Send(
                "worker",
                WorkerInputState(
                    task=task,
                    inputs={dep: current[dep].result or "" for dep in task.depends_on},
                ),
            )
            for task in ready_frontier(current)
        ]

    async def worker(worker_input: WorkerInputState) -> dict:
        task = worker_input["task"]
        context = "".join(f"\n[{key}] {value}" for key, value in worker_input["inputs"].items())
        prompt = task.instruction + context
        try:
            if task.worker == "sql":
                output = await analyst_graph.ainvoke({"question": prompt, "db_path": db_path})
                result = output["summary"]
            elif task.worker == "python":
                output = await code_graph.ainvoke({"instruction": prompt, "test_code": task.args["test_code"]})
                result = output["artifact"]
            elif task.worker == "search":
                output = await search_graph.ainvoke({"question": prompt})
                result = output["summary"]
            else:
                raise ValueError(f"Unknown worker: {task.worker}")
            status = "completed" if output["verified"] else "failed"
        except Exception as exc:
            status, result = "failed", f"{type(exc).__name__}: {exc}"
        return {"tasks": {task.id: task.model_copy(update={"status": status, "result": result})}}

    async def synthesize(state: ParentState) -> dict:
        lines = [f"{t.id} ({t.worker}, {t.status}): {t.result}" for t in state["tasks"].values()]
        return {"final_output": "\n".join(lines)}

    builder = StateGraph(ParentState)
    builder.add_node("load_plan", load_plan)
    builder.add_node("scheduler", scheduler)
    builder.add_node("worker", worker)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "load_plan")
    builder.add_edge("load_plan", "scheduler")
    builder.add_conditional_edges("scheduler", route, ["worker", "synthesize"])
    builder.add_edge("worker", "scheduler")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer)
    # <<<
