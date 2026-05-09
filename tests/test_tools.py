import pytest
from pydantic import BaseModel

from agentforge.tools.base import Tool, ToolRegistry
from agentforge.tools.examples import calculate, get_time, rotate_secret


class _PingInput(BaseModel):
    msg: str


def _ping_fn(inp: _PingInput) -> dict:
    return {"echo": inp.msg}


def test_registry_register_and_get():
    reg = ToolRegistry()
    t = Tool("ping", "Echo a message", _PingInput, _ping_fn)
    reg.register(t)
    assert reg.get("ping") is t


def test_registry_rejects_duplicates():
    reg = ToolRegistry()
    reg.register(Tool("ping", "x", _PingInput, _ping_fn))
    with pytest.raises(ValueError):
        reg.register(Tool("ping", "x", _PingInput, _ping_fn))


def test_anthropic_schema_includes_input_schema():
    t = Tool("ping", "Echo a message", _PingInput, _ping_fn)
    schema = t.to_anthropic_schema()
    assert schema["name"] == "ping"
    assert schema["description"] == "Echo a message"
    assert "msg" in schema["input_schema"]["properties"]


async def test_calculate_tool_evaluates_safely():
    result = await calculate.invoke({"expression": "2 + 3 * 4"})
    assert result["result"] == 14


async def test_calculate_tool_rejects_unsafe_input():
    result = await calculate.invoke({"expression": "__import__('os').system('rm -rf /')"})
    assert "error" in result


async def test_get_time_returns_iso_string():
    result = await get_time.invoke({})
    assert "iso" in result
    assert "T" in result["iso"]


async def test_rotate_secret_mock_returns_status_ok():
    result = await rotate_secret.invoke({"path": "secret/data/app", "key": "api_key"})
    assert result["status"] == "ok"
    assert result["path"] == "secret/data/app"
