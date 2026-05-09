"""DAG agent tests using a scripted fake Anthropic client."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agentforge.dag import DAGAgent
from agentforge.tools.base import ToolRegistry
from agentforge.tools.examples import calculate, get_time


@dataclass
class _Block:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None


def _resp(content: list[_Block], in_tokens: int = 5, out_tokens: int = 5):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


class _FakeMessages:
    def __init__(self, scripted: list):
        self.scripted = scripted
        self.calls = 0

    async def create(self, **_kwargs):
        resp = self.scripted[self.calls]
        self.calls += 1
        return resp


class _FakeClient:
    def __init__(self, scripted):
        self.messages = _FakeMessages(scripted)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(calculate)
    reg.register(get_time)
    return reg


async def test_dag_agent_handles_text_only_response(registry):
    fake = _FakeClient(scripted=[_resp([_Block(type="text", text="hi back")])])
    agent = DAGAgent(client=fake, registry=registry, max_turns=3)
    result = await agent.run("hello")
    assert result["final_text"] == "hi back"
    assert result["turns"] == 1


async def test_dag_agent_runs_parallel_tool_calls_in_one_turn(registry):
    """The DAG executes both tools in parallel inside the execute node, then loops back."""
    fake = _FakeClient(
        scripted=[
            _resp(
                [
                    _Block(type="tool_use", name="calculate", input={"expression": "2+2"}, id="t1"),
                    _Block(type="tool_use", name="get_time", input={}, id="t2"),
                ]
            ),
            _resp([_Block(type="text", text="2+2 is 4 and the time was returned")]),
        ]
    )
    agent = DAGAgent(client=fake, registry=registry, max_turns=5)
    result = await agent.run("calculate 2+2 and tell me the time")
    assert "4" in result["final_text"] or "time" in result["final_text"]
    # Two LLM calls: one to plan tool use, one to summarize after parallel execution.
    assert fake.messages.calls == 2


async def test_dag_agent_respects_max_turns(registry):
    fake = _FakeClient(
        scripted=[
            _resp(
                [_Block(type="tool_use", name="calculate", input={"expression": "1+1"}, id=f"t{i}")]
            )
            for i in range(10)
        ]
    )
    agent = DAGAgent(client=fake, registry=registry, max_turns=2)
    result = await agent.run("loop")
    assert result["turns"] == 2
