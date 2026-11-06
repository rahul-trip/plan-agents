"""Stage 1, ex2: a reducer that updates steps in place by id.

LangGraph's plain `operator.add` would blindly append, so a completed step would appear twice:
once pending and once completed. This reducer treats the list as keyed by step id.
"""

from __future__ import annotations

from planagents.schemas import Step


def merge_steps(existing: list[Step] | None, updates: list[Step]) -> list[Step]:
    """Merge by id. Known ids are replaced where they stand; new ids go on the end, in order.

    Must not mutate either input. Applying the same update twice must change nothing.
    """
    raise NotImplementedError("build an id-keyed dict from existing, overlay updates (dicts keep first-insert order), return its values")
