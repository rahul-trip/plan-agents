"""Regression tests for what scripted LLMs can't show. They run the REAL Claude and OpenAI clients
against a local stub server, so they need no key, spend nothing, and test the client side only: the
requests we send and how we read replies. They don't prove a provider's API accepts the request.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from planagents.llm import get_llm, reply_text, structured
from planagents.schemas import Plan, PlanDraft, PlannedTask, PlanValidationError, ReplanDecision

PLAN_JSON = json.dumps({"tasks": [{"id": "t1", "worker": "w", "instruction": "x", "depends_on": [], "args_json": "{}"}]})


def claude_reply(text):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def openai_reply(text):
    message = {"role": "assistant", "content": text, "refusal": None}
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-x",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def stub():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            wants_plan = "output_config" in body or "response_format" in body
            text = PLAN_JSON if wants_plan else "hello"
            reply = claude_reply(text) if "/messages" in self.path else openai_reply(text)
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def walk(node):
    """Every dict nested anywhere in a JSON schema."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


async def test_reply_text_handles_a_claude_thinking_block(stub):
    url, _ = stub
    reply = await get_llm("anthropic:claude-opus-5-5", base_url=url, api_key="x").ainvoke("hi")
    assert isinstance(reply.content, list), "if this fails, the gotcha reply_text exists for has gone"
    assert type(reply_text(reply)) is str and reply_text(reply) == "hello"


async def test_claude_structured_output_uses_json_schema_not_forced_tools(stub):
    url, requests = stub
    llm = get_llm("anthropic:claude-opus-5-5", base_url=url, api_key="x")
    draft = await structured(llm, PlanDraft).ainvoke("plan it")
    assert draft.tasks[0].id == "t1"
    assert "output_config" in requests[-1] and "tool_choice" not in requests[-1]
    assert requests[-1]["fallbacks"] == "default"


async def test_plan_schema_reaches_claude_with_a_usable_args_field(stub):
    url, requests = stub
    await structured(get_llm("anthropic:claude-opus-5-5", base_url=url, api_key="x"), PlanDraft).ainvoke("plan it")
    schema = requests[-1]["output_config"]["format"]["schema"]
    task = schema["$defs"]["PlannedTask"]["properties"]
    assert task["args_json"]["type"] == "string", "Claude's rewrite empties a free-form dict field"


async def test_openai_request_meets_strict_mode_rules(stub):
    url, requests = stub
    draft = await structured(get_llm("openai:gpt-x", base_url=url, api_key="x"), PlanDraft).ainvoke("plan it")
    assert draft.tasks[0].id == "t1"
    schema = requests[-1]["response_format"]["json_schema"]["schema"]
    for node in walk(schema):
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


async def test_openai_compatible_prefix_and_base_url(stub):
    url, requests = stub
    llm = get_llm("openai-compatible:meta-llama/Llama-3-8b", base_url=url)
    reply = await llm.ainvoke("hi")
    assert reply_text(reply) == "hello"
    assert requests[-1]["model"] == "meta-llama/Llama-3-8b"


async def test_openai_compatible_env_configuration(stub, monkeypatch):
    url, requests = stub
    monkeypatch.setenv("LLM_MODEL", "openai-compatible:local-model")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    llm = get_llm()
    draft = await structured(llm, PlanDraft).ainvoke("plan it")
    assert draft.tasks[0].id == "t1"
    assert requests[-1]["model"] == "local-model"
    assert "response_format" in requests[-1]


async def test_openai_compatible_base_url_and_api_key_aliases(stub, monkeypatch):
    url, requests = stub
    monkeypatch.setenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    monkeypatch.setenv("BASE_URL", url)
    monkeypatch.setenv("API_KEY", "custom-token-123")

    llm = get_llm()
    reply = await llm.ainvoke("hi")
    assert reply_text(reply) == "hello"
    assert requests[-1]["model"] == "mistralai/Mistral-7B-Instruct-v0.3"


@pytest.mark.parametrize(
    "text, expected",
    [('{"n": 1}', {"n": 1}), ("{'n': 1}", {"n": 1}), ("", {}), ("  ", {}), ("{}", {})],
)
def test_parsed_args_accepts_what_models_really_write(text, expected):
    task = PlannedTask(id="t", worker="w", instruction="x", depends_on=[], args_json=text)
    assert task.parsed_args() == expected


@pytest.mark.parametrize("text", ["not args", "[1, 2]", "42", "__import__('os')"])
def test_parsed_args_rejects_anything_else(text):
    task = PlannedTask(id="t", worker="w", instruction="x", depends_on=[], args_json=text)
    with pytest.raises(PlanValidationError):
        task.parsed_args()


@pytest.mark.parametrize("schema", [Plan, ReplanDecision, PlanDraft])
def test_llm_facing_schemas_have_no_runtime_fields_defaults_or_free_form_dicts(schema):
    raw = schema.model_json_schema()
    for node in walk(raw):
        assert node.get("additionalProperties") is not True
        properties = node.get("properties", {})
        assert not {"status", "result", "destructive", "args"} & set(properties)
        for field in properties.values():
            assert "default" not in field
