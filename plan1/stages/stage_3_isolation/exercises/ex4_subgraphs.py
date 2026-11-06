"""Stage 3, ex4: hierarchical planning. Complex workers become compiled subgraphs.

Each subgraph runs its own loop with a private `messages` history. Its output schema exposes only
the final verified artifact, so the parent can't see the rest even by accident.

SAFETY: run_pytest executes generated code in a temp directory with a timeout. That is NOT a
sandbox, because the code keeps your user's file and network access. With a real LLM, only run
this on prompts you trust, or put it in a container or VM first.
"""

from __future__ import annotations

import asyncio
import operator
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from planagents.llm import reply_text, structured


class CodeInput(TypedDict):
    instruction: str
    test_code: str


class CodeOutput(TypedDict):
    artifact: str
    verified: bool


class CodeState(CodeInput, CodeOutput, total=False):
    code: str
    test_output: str
    passed: bool
    attempts: int
    messages: Annotated[list[str], operator.add]  # Private: never crosses the output schema.


def run_pytest(code: str, test_code: str, timeout: float = 30) -> tuple[bool, str]:
    """Write solution.py and test_solution.py to a temp dir, run pytest, return (passed, output)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "solution.py").write_text(code)
        (path / "test_solution.py").write_text(test_code)
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "."],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {timeout}s"
    return run.returncode == 0, run.stdout + run.stderr


def extract_code(reply: str) -> str:
    """Pull code out of a Markdown fence if the LLM used one."""
    match = re.search(r"```(?:python)?\n(.*?)```", reply, flags=re.S)
    return match.group(1) if match else reply


def build_code_generator_subgraph(llm, max_attempts: int = 3, runner=run_pytest):
    """Write -> Pytest -> Fix loop. input_schema=CodeInput, output_schema=CodeOutput.

    - write: llm.ainvoke with a prompt containing the instruction; store
      extract_code(reply_text(reply)); attempts + 1; add the reply text to messages.
    - pytest: run `runner(code, test_code)` in a thread; store passed and test_output.
    - After pytest: if passed or attempts reached max_attempts go to 'finish', otherwise 'fix'.
    - fix: like write, but the prompt includes the previous code and test_output.
    - finish: artifact=code, verified=passed.
    """
    raise NotImplementedError("build the StateGraph(CodeState, input_schema=CodeInput, output_schema=CodeOutput) described above")


class SqlQuery(BaseModel):
    sql: str


class AnalystInput(TypedDict):
    question: str
    db_path: str


class AnalystOutput(TypedDict):
    summary: str
    verified: bool


class AnalystState(AnalystInput, AnalystOutput, total=False):
    db_schema: str
    sql: str
    rows: str
    error: str
    attempts: int
    messages: Annotated[list[str], operator.add]


def run_readonly_query(db_path: str, sql: str) -> list[tuple]:
    """Run SQL on a read-only connection, so a generated DROP or UPDATE fails at the database.

    Opening with mode=ro is the real control. Checking the SQL text alone is easy to fool. Build the
    URI with Path(db_path).resolve().as_uri(): in a hand-built f"file:{db_path}?mode=ro", a # or ? in
    the path silently drops mode=ro, and writes succeed.
    """
    raise NotImplementedError("connect with sqlite3 using Path(...).as_uri() + \"?mode=ro\" and uri=True; execute; fetchall; close the connection")


def build_data_analyst_subgraph(llm, max_attempts: int = 3):
    """Schema Inspect -> SQL Query -> Summarize. input_schema=AnalystInput, output_schema=AnalystOutput.

    - inspect: read CREATE statements from sqlite_master (read-only) into db_schema.
    - query: structured(llm, SqlQuery) with question, schema and any previous error; run it with
      run_readonly_query. A sqlite3.Error is stored in `error` and retried, up to max_attempts.
    - summarize: llm.ainvoke over the rows gives summary (reply_text), verified=True. If attempts ran
      out: summary describes the failure, verified=False, and no LLM call.

    verified=True only means the SQL ran. It can't tell you the LLM summarised the rows correctly.
    """
    raise NotImplementedError("build the graph described above; loop query->query on error while attempts remain")
