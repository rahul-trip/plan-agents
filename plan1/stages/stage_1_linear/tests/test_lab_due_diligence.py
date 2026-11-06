import re

import pytest
from pydantic import ValidationError

from planagents.fakes import ScriptedLLM
from planagents.schemas import Step

URL = "https://github.com/acme/widget"
TREE = ["src/app.py", "src/db.py", "tests/test_app.py", "README.md"]


def audit_plan() -> dict:
    workers = ["license", "vulnerabilities", "architecture", "tests"]
    return {
        "steps": [
            {"id": f"step_{i}", "description": f"audit {w}", "assigned_worker": w}
            for i, w in enumerate(workers, start=1)
        ]
    }


@pytest.fixture
def lab(load):
    return load("lab_due_diligence")


def client(lab, severity="LOW"):
    vuln = lab.Vulnerability(id="GHSA-1", package="flask==1.0", severity=severity)
    return lab.FixtureRepoClient("MIT", ["flask==1.0"], [vuln], TREE)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/acme/widget", "acme/widget"),
        ("https://github.com/acme/widget.git", "acme/widget"),
        ("https://github.com/acme/widget/", "acme/widget"),
    ],
)
def test_parse_repo(lab, url, expected):
    assert lab.parse_repo(url) == expected


@pytest.mark.parametrize("url", ["https://gitlab.com/a/b", "https://github.com/onlyorg", "nonsense"])
def test_parse_repo_rejects_other_urls(lab, url):
    with pytest.raises(ValueError):
        lab.parse_repo(url)


async def test_clean_repo_runs_exactly_the_four_audits_in_order(lab):
    llm = ScriptedLLM({"AuditPlan": [audit_plan()], "text": ["A small Python service."]})
    result = await lab.build_graph(llm, client(lab)).ainvoke({"repo_url": URL})
    steps = result["steps"]
    assert [s.assigned_worker for s in steps] == ["license", "vulnerabilities", "architecture", "tests"]
    assert all(s.status == "completed" for s in steps)
    assert "MIT" in steps[0].result
    assert "GHSA-1" in steps[1].result
    assert steps[2].result == "A small Python service."
    assert isinstance(steps[2].result, str), "use reply_text: Claude's reply.content can be a list"
    assert re.findall(r"\d+", steps[3].result) == ["1", "2"]


def test_count_test_files_ignores_non_python_and_recognises_naming(lab):
    tree = [
        "src/app.py",
        "src/app_test.py",
        "test_top.py",
        "tests/test_a.py",
        "tests/helpers.py",
        "tests/certs/server.pem",
        "tests/data.json",
        "README.md",
    ]
    assert lab.count_test_files(tree) == (4, 1)


async def test_no_dependencies_reads_as_not_checked_rather_than_clean(lab):
    empty = lab.FixtureRepoClient("MIT", [], [], TREE)
    step = Step(id="s", description="d", assigned_worker="vulnerabilities")
    result = await lab.make_workers(empty, ScriptedLLM({}))["vulnerabilities"](step, "acme/widget")
    assert "not checked" in result.lower()
    assert "none found" not in result.lower()


async def test_critical_vulnerability_adds_and_runs_a_remediation_step(lab):
    llm = ScriptedLLM(
        {
            "AuditPlan": [audit_plan()],
            "CriticalFollowUp": [{"description": "Upgrade flask"}],
            "text": ["architecture summary", "remediation plan"],
        }
    )
    result = await lab.build_graph(llm, client(lab, "CRITICAL")).ainvoke({"repo_url": URL})
    steps = {s.id: s for s in result["steps"]}
    assert len(steps) == 5, "exactly one step must be added, even though the trigger stays true"
    assert steps[lab.REMEDIATION_ID].status == "completed"
    assert steps[lab.REMEDIATION_ID].description == "Upgrade flask"
    assert steps[lab.REMEDIATION_ID].result == "remediation plan"
    assert steps["step_2"].result.startswith(lab.CRITICAL_MARKER)


def test_plan_schema_rejects_an_invented_worker(lab):
    """Regression: a real LLM returned assigned_worker='Security Specialist' here."""
    plan = audit_plan()
    plan["steps"][0]["assigned_worker"] = "Security Specialist"
    with pytest.raises(ValidationError):
        lab.AuditPlan.model_validate(plan)


def test_needs_remediation(lab):
    pending = Step(id="a", description="d", assigned_worker="w", result="CRITICAL: x")
    assert not lab.needs_remediation([pending]), "only completed steps count"
    done = pending.model_copy(update={"status": "completed"})
    assert lab.needs_remediation([done])
    fix = Step(id=lab.REMEDIATION_ID, description="d", assigned_worker="remediation")
    assert not lab.needs_remediation([done, fix])


@pytest.mark.live
async def test_live_client_against_public_apis(lab):
    live = lab.GitHubOSVClient()
    assert await live.license("psf/requests") == "Apache-2.0"
    assert any(path.startswith("src/") for path in await live.file_tree("psf/requests"))
    vulns = await live.vulnerabilities(["jinja2==2.4.1"])
    assert vulns, "OSV should report known issues for this old release"
