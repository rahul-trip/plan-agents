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
    raise NotImplementedError("tokenize lowercased words with re.findall(r\"\\w+\", ...), score by set intersection, sort by score (stable)")


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

    raise NotImplementedError("search node calls search_corpus; summarize node calls llm with the hit documents, or returns verified=False with no hits")


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
    raise NotImplementedError("write scheduler, route, worker (mapping to each subgraph), synthesize; compile with the checkpointer")
