"""Agent loop using Anthropic's native tool-use API.

Flow:
  1. Send user message + tool schemas to Claude
  2. If Claude returns tool_use blocks, invoke each tool
  3. Send tool_result blocks back
  4. Repeat until Claude returns plain text (end_turn)

All tool inputs are validated through Pydantic before execution. The agent
never sees free-form code.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from agentforge.tools.base import ToolRegistry


class AnthropicClient(Protocol):
    """Subset of anthropic.AsyncAnthropic we depend on — keeps tests simple."""

    async def messages_create(self, **kwargs: Any) -> Any: ...


@dataclass
class AgentTrace:
    """One row per turn — useful for debugging and audit logs."""

    role: str
    content: Any
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class AgentResult:
    final_text: str
    turns: int
    trace: list[AgentTrace]
    input_tokens: int = 0
    output_tokens: int = 0


class Agent:
    """A tool-using agent backed by Anthropic Claude."""

    def __init__(
        self,
        client: Any,
        registry: ToolRegistry,
        model: str = "claude-sonnet-4-6",
        max_turns: int = 10,
        system: str | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.max_turns = max_turns
        self.system = system or "You are a helpful assistant. Use the provided tools when useful."

    async def run(self, prompt: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        trace: list[AgentTrace] = [AgentTrace(role="user", content=prompt)]

        input_tokens = 0
        output_tokens = 0

        for turn in range(1, self.max_turns + 1):
            response = await self._create_message(messages)

            input_tokens += getattr(response.usage, "input_tokens", 0)
            output_tokens += getattr(response.usage, "output_tokens", 0)

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            text_blocks = [block for block in response.content if block.type == "text"]
            assistant_text = "".join(b.text for b in text_blocks)

            trace.append(
                AgentTrace(
                    role="assistant",
                    content=assistant_text,
                    tool_calls=[{"name": tu.name, "input": tu.input} for tu in tool_uses],
                )
            )

            if response.stop_reason == "end_turn" or not tool_uses:
                return AgentResult(
                    final_text=assistant_text,
                    turns=turn,
                    trace=trace,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tu in tool_uses:
                tool_obj = self.registry.get(tu.name)
                output = await tool_obj.invoke(tu.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": str(output),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            trace.append(AgentTrace(role="tool", content=tool_results))

        return AgentResult(
            final_text="(max turns reached)",
            turns=self.max_turns,
            trace=trace,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _create_message(self, messages: list[dict]) -> Any:
        return await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            tools=self.registry.to_anthropic_schemas(),
            messages=messages,
        )
