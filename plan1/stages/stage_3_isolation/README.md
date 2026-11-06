# Stage 3: Context Isolation and Hierarchical Subgraphs

A common failure mode is passing the whole global state, with every worker's raw tool output and stack traces, to every later worker. It degrades reasoning and inflates token costs.

## Mechanics

- **Blackboard pattern.** Workers never read graph state. The dispatcher resolves input references, injecting only `task_1`'s summary into `task_3`'s prompt, and passes a minimal `WorkerInputState`.
- **Hierarchical subgraphs.** A complex worker ("write and debug a migration script") becomes a compiled subgraph with its own loop. The parent sees only the final verified artifact.
- **State mapping.** Invoke the subgraph *inside* a node when schemas differ. That node is the isolation boundary, because it decides what crosses in each direction. `input_schema` and `output_schema` enforce the same boundary from the subgraph's side.

## Gotchas

- `frozen=True` stops field reassignment, not mutation of a nested dict. Wrap it in `MappingProxyType` over a copy.
- Don't add a subgraph as a node here. A subgraph added directly shares channels with its parent, which is what you're trying to avoid.
- Isolation claims need evidence. Plant canary strings where they shouldn't travel, then scan **every** parent update and checkpoint for them, as the capstone test does.
- A subgraph's checkpoints live in their own namespace. "Zero leakage" here is about parent state and worker prompts. It doesn't mean subgraph data is absent from your database, which matters if that data is sensitive.
- `verified=True` is only as strong as its check. Pytest passing is real evidence for code. For the analyst it only means the SQL ran: in a live run, a local model summarised two orders as "only one order".
- Read replies with `reply_text`, not `reply.content`. The fake LLM returns Claude's block format on purpose, so `.content` fails here as it would live.

## Safety

`run_pytest` executes generated code in a temp directory with a timeout. **This is not a sandbox**: the code keeps your user's file and network access. Tests use scripted code, so they're safe. With a real LLM, only run it on prompts you trust, or inside a container or VM. SQL uses a read-only SQLite connection, which is a real control, unlike checking the SQL text.

## Read first

- [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs): state mapping and persistence modes
- [How Anthropic built its multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

## Go deeper

- [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents): the counterpoint on shared context, worth arguing with
- [Blackboard systems](https://en.wikipedia.org/wiki/Blackboard_system)
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/), [`MappingProxyType`](https://docs.python.org/3/library/types.html)

## Exercises

| File | You build | Track |
|---|---|---|
| `ex1_scoped_workers` | Workers receiving immutable state slices | Custom |
| `ex2_blackboard` | Dispatcher resolving references into `WorkerInputState` | LangGraph |
| `ex3_budget_guard` | Payload guard behind the `budget_guard` fixture in `tests/conftest.py` (study workflow #2) | Both |
| `ex4_subgraphs` | `CodeGeneratorSubgraph` and `DataAnalystSubgraph` | LangGraph |
| `lab_multi_domain` | **Capstone:** SQL + Python + Search in the parent DAG, zero leakage | LangGraph |

Until ex3 is solved, its tests also report a teardown error, "budget_guard was never exercised". That's the fixture doing its job and goes away once `check` works.

**Done when** all tests pass and `test_zero_context_leakage_in_every_parent_update_and_checkpoint` is green. Then break it on purpose: put `str(output)` in a task's `result` and watch it fail. Returning a stray `messages` key won't do it, because LangGraph silently drops keys outside the state schema.
