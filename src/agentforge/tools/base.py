"""Typed tool contracts for agents.

Each Tool wraps a callable with a Pydantic input schema. The agent only
ever invokes tools through validated Pydantic models — never free-form code.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")

ToolFn = Callable[[InputT], Awaitable[OutputT]] | Callable[[InputT], OutputT]


class Tool(Generic[InputT, OutputT]):
    """A typed tool the agent can invoke.

    The Pydantic input schema doubles as the JSON Schema sent to the LLM,
    so the model sees exact types, descriptions, and constraints.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[InputT],
        fn: ToolFn,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._fn = fn

    async def invoke(self, raw_input: dict[str, Any]) -> OutputT:
        validated = self.input_schema.model_validate(raw_input)
        result = self._fn(validated)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Format compatible with Anthropic's tool-use API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }


class ToolRegistry:
    """Holds all tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool_obj: Tool) -> None:
        if tool_obj.name in self._tools:
            raise ValueError(f"tool {tool_obj.name!r} already registered")
        self._tools[tool_obj.name] = tool_obj

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"tool {name!r} not registered")
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def to_anthropic_schemas(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]


def tool(
    name: str,
    description: str,
    input_schema: type[BaseModel],
) -> Callable[[ToolFn], Tool]:
    """Decorator that turns a function into a Tool.

    Example:
        class GetWeatherInput(BaseModel):
            city: str

        @tool("get_weather", "Get current weather", GetWeatherInput)
        def get_weather(inp: GetWeatherInput) -> dict:
            return {"city": inp.city, "temp_f": 72}
    """

    def decorator(fn: ToolFn) -> Tool:
        return Tool(name=name, description=description, input_schema=input_schema, fn=fn)

    return decorator
