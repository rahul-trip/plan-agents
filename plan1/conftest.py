"""Shared test wiring: run one test suite against exercises/ or solutions/."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from planagents.loader import import_stage_module


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--target",
        choices=["exercises", "solutions"],
        default="exercises",
        help="Which implementation the stage tests load (default: exercises).",
    )


@pytest.fixture
def load(request: pytest.FixtureRequest):
    """load("ex1_custom_loop") imports that module from the selected target."""
    stage_dir = Path(request.path).parents[1]
    target = request.config.getoption("--target")
    return lambda name: import_stage_module(stage_dir, target, name)


@pytest.fixture
def postgres_url() -> str:
    """Connection string for the compose Postgres; skips clearly if unreachable."""
    import psycopg

    url = os.environ.get(
        "POSTGRES_URL",
        "postgresql://planagents:planagents@localhost:55432/planagents",
    )
    try:
        psycopg.connect(url, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable. Run: docker compose up -d --wait")
    return url
