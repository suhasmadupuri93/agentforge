# AgentForge

A typed agent framework for production AI agents — Pydantic schemas for tool contracts, Anthropic Claude for reasoning, FastAPI for serving.

Agents in AgentForge can only invoke **registered, schema-validated tools**. No free-form code execution, no `eval`, no surprises. The same Pydantic schema doubles as the JSON Schema sent to Claude, so the model sees exact types, descriptions, and constraints.

## Why

Most "agent" tutorials use LangChain to wire prompts together. That's fine for demos, but production agent infra needs:

- **Audit trails** — every tool call captured with inputs/outputs
- **Typed contracts** — Pydantic validates input before the tool sees it
- **No code execution** — agents run pre-defined operations, not arbitrary code
- **Provider-controlled scope** — you decide what an agent can do, not the LLM

This framework was built around those constraints, drawn from production agent platforms.

## Features

- **Tool registry** with `@tool` decorator
- **Pydantic-validated inputs** — invalid tool calls fail before execution
- **Multi-turn agent loop** with configurable max turns
- **Anthropic native tool-use API** (no LangChain dependency)
- **FastAPI server** with `/healthz`, `/tools`, `/v1/run`
- **Trace capture** — every turn (user, assistant, tool) recorded for audit
- **Token accounting** — input/output tokens reported per run

## Quick Start

```bash
# Install (creates .venv, installs deps)
make install

# Run tests (no API key needed — uses fakes)
make test

# Run the server
export ANTHROPIC_API_KEY=sk-ant-...
make run
```

In another terminal:
```bash
curl -s http://localhost:8090/tools | jq .

curl -s -X POST http://localhost:8090/v1/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What time is it, and what is 17 * 23?"}'
```

## Define a tool

```python
from pydantic import BaseModel, Field
from agentforge.tools.base import tool

class FetchKafkaOffsetInput(BaseModel):
    topic: str = Field(..., description="Kafka topic name")
    partition: int = Field(0, description="Partition to query")

@tool("fetch_kafka_offset", "Get the latest offset for a Kafka partition", FetchKafkaOffsetInput)
def fetch_kafka_offset(inp: FetchKafkaOffsetInput) -> dict:
    # your real Kafka client here
    return {"topic": inp.topic, "partition": inp.partition, "offset": 123456}
```

Register it:
```python
from agentforge.tools.base import ToolRegistry

registry = ToolRegistry()
registry.register(fetch_kafka_offset)
```

The agent now sees the schema (with descriptions) and can call this tool with validated input.

## Architecture

```
prompt ──► Agent ──► Claude (with tool schemas)
                          │
                          ├─ end_turn  ─► return final_text
                          │
                          └─ tool_use ─► validate input via Pydantic
                                          │
                                          ├─ invoke registered tool
                                          ▼
                                          tool_result ─► loop back to Claude
```

## Project Structure

```
src/agentforge/
├── tools/
│   ├── base.py        → Tool, ToolRegistry, @tool decorator
│   └── examples.py    → calculate, get_time, rotate_secret
├── agent.py           → Multi-turn loop with Anthropic tool-use
└── server.py          → FastAPI HTTP layer
tests/
├── test_tools.py
└── test_agent.py
```

## Roadmap

- [ ] LangGraph-based DAG agents (parallel tool calls, branches)
- [ ] AWS Lambda deployment template
- [ ] OpenTelemetry tracing per tool call
- [ ] PII redaction (regex + spaCy NER) on inputs/outputs
- [ ] Streaming responses (SSE)
- [ ] Per-tenant rate limits and budgets

## License

MIT
