"""FastAPI server exposing the agent as an HTTP endpoint."""

import os

from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentforge.agent import Agent
from agentforge.tools.base import ToolRegistry
from agentforge.tools.examples import default_tools

app = FastAPI(title="AgentForge", version="0.1.0")

_registry = ToolRegistry()
for t in default_tools():
    _registry.register(t)


def _client() -> AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    return AsyncAnthropic(api_key=api_key)


class RunRequest(BaseModel):
    prompt: str
    model: str = "claude-sonnet-4-6"
    max_turns: int = 10


class RunResponse(BaseModel):
    final_text: str
    turns: int
    input_tokens: int
    output_tokens: int


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/tools")
async def list_tools() -> dict:
    return {"tools": _registry.to_anthropic_schemas()}


@app.post("/v1/run", response_model=RunResponse)
async def run_agent(req: RunRequest) -> RunResponse:
    agent = Agent(client=_client(), registry=_registry, model=req.model, max_turns=req.max_turns)
    result = await agent.run(req.prompt)
    return RunResponse(
        final_text=result.final_text,
        turns=result.turns,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "agentforge.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8090")),
        reload=False,
    )


if __name__ == "__main__":
    run()
