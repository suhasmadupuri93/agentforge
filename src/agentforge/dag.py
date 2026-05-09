"""LangGraph-based DAG agent.

Adds a graph-shaped agent runtime on top of the linear loop in `agent.py`:

  plan ──► execute (fan-out tool calls in parallel) ──► review
                                                          │
                                                          ├─ continue ──► plan
                                                          └─ done     ──► END

The key win over the linear agent is **parallel tool execution** — when the
LLM emits multiple tool_use blocks in one turn, every Pydantic-validated tool
runs concurrently with `asyncio.gather`, which collapses sequential round-trips
into one. This is the same pattern production agent platforms use to keep
multi-tool turns under a second.

State is a TypedDict; LangGraph reduces it across nodes via the `messages`
appender and lets you attach checkpointing later for durable resume.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from agentforge.tools.base import ToolRegistry


class AgentState(TypedDict):
    """Shared state passed between graph nodes.

    `operator.add` on `messages` tells LangGraph to concatenate lists across
    nodes, which is what we want — Anthropic content blocks aren't LangChain
    messages, so we skip the `add_messages` coercion.
    """

    messages: Annotated[list[dict], operator.add]
    pending_tool_calls: list[dict]
    turns: int
    max_turns: int
    input_tokens: int
    output_tokens: int


class DAGAgent:
    """Plan → execute (parallel) → review agent built on LangGraph."""

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
        self.system = system or (
            "You are an agent that uses tools to answer questions. "
            "When multiple independent facts are needed, request them all in a single turn."
        )
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("review", self._review_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "review")
        graph.add_conditional_edges(
            "review", self._should_continue, {"continue": "plan", "done": END}
        )

        return graph.compile()

    # ── Nodes ───────────────────────────────────────────────────────────

    async def _plan_node(self, state: AgentState) -> dict:
        """Ask the LLM what to do next, given the conversation so far."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            tools=self.registry.to_anthropic_schemas(),
            messages=state["messages"],
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text")

        new_messages = []
        if response.content:
            new_messages.append({"role": "assistant", "content": response.content})

        return {
            "messages": new_messages,
            "pending_tool_calls": [
                {"id": tu.id, "name": tu.name, "input": tu.input} for tu in tool_uses
            ],
            "turns": state["turns"] + 1,
            "input_tokens": state["input_tokens"] + getattr(response.usage, "input_tokens", 0),
            "output_tokens": state["output_tokens"] + getattr(response.usage, "output_tokens", 0),
            "_assistant_text": text,
        }

    async def _execute_node(self, state: AgentState) -> dict:
        """Run every pending tool call **in parallel**."""
        if not state["pending_tool_calls"]:
            return {}

        async def run_one(call: dict) -> dict:
            tool_obj = self.registry.get(call["name"])
            output = await tool_obj.invoke(call["input"])
            return {"type": "tool_result", "tool_use_id": call["id"], "content": str(output)}

        results = await asyncio.gather(*(run_one(c) for c in state["pending_tool_calls"]))

        return {
            "messages": [{"role": "user", "content": results}],
            "pending_tool_calls": [],
        }

    async def _review_node(self, state: AgentState) -> dict:
        """Decide whether to loop back to plan or terminate."""
        return {}

    def _should_continue(self, state: AgentState) -> str:
        if state["turns"] >= state["max_turns"]:
            return "done"
        # If the last assistant message has no tool calls, we're done.
        for msg in reversed(state["messages"]):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    has_tools = any(getattr(b, "type", None) == "tool_use" for b in content)
                    if not has_tools:
                        return "done"
                break
        return "continue"

    # ── Public API ──────────────────────────────────────────────────────

    async def run(self, prompt: str) -> dict:
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": prompt}],
            "pending_tool_calls": [],
            "turns": 0,
            "max_turns": self.max_turns,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        final = await self._graph.ainvoke(initial_state)

        # Pull the final assistant text from the last assistant message.
        final_text = ""
        for msg in reversed(final["messages"]):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    final_text = "".join(
                        getattr(b, "text", "")
                        for b in content
                        if getattr(b, "type", None) == "text"
                    )
                elif isinstance(content, str):
                    final_text = content
                break

        return {
            "final_text": final_text,
            "turns": final["turns"],
            "input_tokens": final["input_tokens"],
            "output_tokens": final["output_tokens"],
        }
