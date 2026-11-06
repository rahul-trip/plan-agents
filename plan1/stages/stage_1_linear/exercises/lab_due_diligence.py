"""Stage 1 lab and capstone: a technical due-diligence agent.

Given a GitHub URL, plan 4 sequential audits (license, dependency vulnerabilities, architecture
summary, test coverage), run them one at a time, and add a step if a critical vulnerability
turns up. Workers talk to a RepoClient, so tests use fixtures and live runs use real APIs.

Live run (needs LLM_MODEL in .env; GITHUB_TOKEN is optional):
    uv run python stages/stage_1_linear/solutions/lab_due_diligence.py https://github.com/org/repo
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Annotated, Literal, Protocol, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from planagents.llm import reply_text, structured
from planagents.schemas import Plan, PlannedStep, Step

REMEDIATION_ID = "remediation"
CRITICAL_MARKER = "CRITICAL:"


class AuditStep(PlannedStep):
    """A Literal, not a free string: given a plain str, real LLMs invent workers that don't exist
    (one produced 'Security Specialist' here). The schema makes that impossible to return."""

    assigned_worker: Literal["license", "vulnerabilities", "architecture", "tests"]


class AuditPlan(Plan):
    steps: list[AuditStep]


class Vulnerability(BaseModel):
    id: str
    package: str
    severity: str


class RepoClient(Protocol):
    async def license(self, repo: str) -> str: ...
    async def dependencies(self, repo: str) -> list[str]: ...
    async def vulnerabilities(self, dependencies: list[str]) -> list[Vulnerability]: ...
    async def file_tree(self, repo: str) -> list[str]: ...


class FixtureRepoClient:
    """Offline client for tests and demos."""

    def __init__(self, license: str, dependencies: list[str], vulns: list[Vulnerability], tree: list[str]):
        self._license, self._dependencies, self._vulns, self._tree = license, dependencies, vulns, tree

    async def license(self, repo: str) -> str:
        return self._license

    async def dependencies(self, repo: str) -> list[str]:
        return self._dependencies

    async def vulnerabilities(self, dependencies: list[str]) -> list[Vulnerability]:
        return self._vulns

    async def file_tree(self, repo: str) -> list[str]:
        return self._tree


class GitHubOSVClient:
    """Live client: GitHub REST for repo data, OSV for vulnerabilities. Python requirements only."""

    def __init__(self, token: str | None = None, max_dependencies: int = 30):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._github = httpx.AsyncClient(base_url="https://api.github.com", headers=headers, timeout=20)
        self._osv = httpx.AsyncClient(base_url="https://api.osv.dev/v1", timeout=20)
        self._max_dependencies = max_dependencies

    async def license(self, repo: str) -> str:
        response = await self._github.get(f"/repos/{repo}/license")
        if response.status_code == 404:
            return "NOASSERTION"
        response.raise_for_status()
        return response.json()["license"]["spdx_id"]

    async def dependencies(self, repo: str) -> list[str]:
        response = await self._github.get(
            f"/repos/{repo}/contents/requirements.txt",
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        pinned = re.findall(r"^([A-Za-z0-9_.\-]+==[^\s;#]+)", response.text, flags=re.M)
        return pinned[: self._max_dependencies]

    async def vulnerabilities(self, dependencies: list[str]) -> list[Vulnerability]:
        found = []
        for dependency in dependencies:
            name, version = dependency.split("==", 1)
            response = await self._osv.post(
                "/query", json={"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
            )
            response.raise_for_status()
            for vuln in response.json().get("vulns", []):
                severity = vuln.get("database_specific", {}).get("severity", "UNKNOWN")
                found.append(Vulnerability(id=vuln["id"], package=dependency, severity=severity))
        return found

    async def file_tree(self, repo: str) -> list[str]:
        response = await self._github.get(f"/repos/{repo}/git/trees/HEAD", params={"recursive": 1})
        response.raise_for_status()
        return [item["path"] for item in response.json()["tree"] if item["type"] == "blob"]


def parse_repo(url: str) -> str:
    """'https://github.com/org/repo(.git)' -> 'org/repo'. Anything else raises ValueError."""
    raise NotImplementedError("extract org/repo with a regex, strip a trailing .git, raise ValueError on non-GitHub URLs")


def count_test_files(tree: list[str]) -> tuple[int, int]:
    """(test files, source files), counting Python files only, so certs and fixtures don't inflate it.

    A test file is under a test/ or tests/ directory, or is named test_*.py or *_test.py.
    """
    raise NotImplementedError("filter to .py files, split them with a regex on the rules above, return the two counts")


def make_workers(client: RepoClient, llm) -> dict:
    """Return workers keyed 'license', 'vulnerabilities', 'architecture', 'tests', 'remediation'.

    Each takes (step, repo) and returns a string. The architecture and remediation workers call the
    LLM and return reply_text(reply); the others are deterministic.

    - license: the result names the license.
    - vulnerabilities: names each vulnerability's id. Prefix the result with CRITICAL_MARKER when any
      has severity CRITICAL. With no dependencies to check, the result must say 'not checked': in an
      audit, "we didn't look" must never read as "none found".
    - tests: the only digits in the result are the two counts, test files first.
    """
    raise NotImplementedError("implement the five workers; audits use client, architecture/remediation summarise with llm.ainvoke and reply_text")


class DiligenceState(TypedDict, total=False):
    repo_url: str
    repo: str
    steps: Annotated[list[Step], lambda old, new: list({s.id: s for s in [*(old or []), *new]}.values())]


class CriticalFollowUp(BaseModel):
    """The replanner's description of the remediation step to add."""

    description: str = Field(description="What the remediation step must cover")


def needs_remediation(steps: list[Step]) -> bool:
    """True if a completed step reported a critical vulnerability and no remediation step exists."""
    raise NotImplementedError("check completed results for CRITICAL_MARKER and that REMEDIATION_ID is absent")


def build_graph(llm, client: RepoClient):
    """planner (structured(llm, AuditPlan)) -> executor -> replanner, looping until none is pending.

    The replanner adds a Step(id=REMEDIATION_ID, assigned_worker='remediation') when
    needs_remediation() is true, asking structured(llm, CriticalFollowUp) for its description.
    The trigger is deterministic on purpose: the LLM words the step but can't skip it.
    """
    workers = make_workers(client, llm)

    raise NotImplementedError("planner (parse_repo + structured AuditPlan), executor (first pending step), replanner (needs_remediation), loop, compile")


async def main(url: str) -> None:
    from planagents.llm import get_llm

    graph = build_graph(get_llm(), GitHubOSVClient(os.environ.get("GITHUB_TOKEN")))
    result = await graph.ainvoke({"repo_url": url})
    for step in result["steps"]:
        print(f"[{step.id} | {step.assigned_worker}] {step.result}\n")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main(sys.argv[1]))
