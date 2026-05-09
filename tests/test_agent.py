"""Agent tests using a fake Anthropic client — no API key required."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agentforge.agent import Agent
from agentforge.tools.base import ToolRegistry
from agentforge.tools.examples import calculate


@dataclass
class _Block:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None


def _resp(stop_reason: str, content: list[_Block], usage_in: int = 5, usage_out: int = 5):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
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
    return reg


async def test_agent_returns_immediately_on_end_turn(registry):
    fake = _FakeClient(scripted=[_resp("end_turn", [_Block(type="text", text="hello")])])
    agent = Agent(client=fake, registry=registry)
    result = await agent.run("hi")
    assert result.final_text == "hello"
    assert result.turns == 1


async def test_agent_invokes_tool_then_returns_final_text(registry):
    fake = _FakeClient(
        scripted=[
            _resp(
                "tool_use",
                [_Block(type="tool_use", name="calculate", input={"expression": "2+2"}, id="t1")],
            ),
            _resp("end_turn", [_Block(type="text", text="2+2 is 4")]),
        ]
    )
    agent = Agent(client=fake, registry=registry)
    result = await agent.run("what is 2+2")
    assert result.final_text == "2+2 is 4"
    assert result.turns == 2
    assert any(t.tool_calls for t in result.trace)


async def test_agent_respects_max_turns(registry):
    fake = _FakeClient(
        scripted=[
            _resp(
                "tool_use",
                [_Block(type="tool_use", name="calculate", input={"expression": "1+1"}, id="t1")],
            )
        ]
        * 5
    )
    agent = Agent(client=fake, registry=registry, max_turns=3)
    result = await agent.run("loop")
    assert result.turns == 3
    assert result.final_text == "(max turns reached)"
