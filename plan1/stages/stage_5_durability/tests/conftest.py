import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from planagents.dag import checkpoint_serde


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.postgres)])
async def saver(request):
    """Runs a test twice: quickly in memory, and against the real checkpointer."""
    if request.param == "memory":
        yield InMemorySaver(serde=checkpoint_serde())
        return
    url = request.getfixturevalue("postgres_url")
    async with AsyncPostgresSaver.from_conn_string(url, serde=checkpoint_serde()) as postgres:
        await postgres.setup()
        yield postgres
